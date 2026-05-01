"""
Alpaca stock exchange adapter for multi-asset trading.

Connects to Alpaca for US stock market data and execution, proving the
event bus architecture works across asset classes (crypto + stocks).

@decision DEC-ALPACA-001
@title Alpaca adapter for multi-asset proof-of-concept
@status accepted
@rationale Alpaca provides free stock trading API with WebSocket streaming.
Using the same ExchangeAdapter interface as Kraken demonstrates the event bus
architecture supports multiple asset classes without core changes. alpaca-py
is an optional dependency — the system works without it.

@decision DEC-ALPACA-002
@title Gate stock polling on US market hours (09:30-16:00 ET Mon-Fri)
@status accepted
@rationale Session 48 (2026-04-29, started 22:16 ET) exposed a two-layer bug:
Alpaca's StockLatestQuoteRequest returns a frozen stale quote during off-hours
(AAPL stuck at $127.885 for the first ~10.9h while the real price was $267+).
These stale ticks fed the outlier filter's cold-start window, locking the median
at $127.885 and causing 192,861 outlier_tick_rejected events for AAPL over 44h.
The orb_stocks strategy took zero fills as a result.
Fix: before each API call in _poll_market_data(), check _is_market_open() and
sleep for `poll_interval_seconds` without issuing the request when the market is
closed. This eliminates the stale-quote feed entirely during off-hours and saves
Alpaca API quota. Stock quotes have no value when exchanges are closed.
v1 limitation: no NYSE holiday handling — weekday+time window only. Tracked as
a known follow-up. US market times: 09:30-16:00 ET, Mon(0)-Fri(4).
Investigation report: tmp/session-48/INVESTIGATION_aapl_outlier.md
"""

import asyncio
import datetime
import os
from decimal import Decimal
from time import time
from typing import Any

import pytz

import structlog

from cerebrum.adapters.base import ExchangeAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import (
    Amount,
    EventType,
    OrderStatus,
    Price,
    Side,
    Symbol,
)

logger = structlog.get_logger()


class AlpacaAdapter(ExchangeAdapter):
    """
    Alpaca stock exchange adapter.

    Features:
    - Stock market data via Alpaca quote polling
    - Market order execution via Alpaca Trading API
    - Portfolio and position tracking
    - Free tier supports paper and live trading
    """

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        """
        Initialize Alpaca adapter.

        Args:
            bus: Event bus for publishing/subscribing
            config: Configuration with api_key, secret_key, paper (bool)
        """
        super().__init__(bus, config)
        self._trading_client = None
        self._data_client = None
        self._streaming_tasks: list[asyncio.Task[None]] = []
        self._connected = False
        self._current_prices: dict[Symbol, Price] = {}
        self._log = logger.bind(adapter="alpaca")

    async def connect(self) -> None:
        """Connect to Alpaca API."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            api_key = self.config.get("api_key", "")
            secret_key = self.config.get("secret_key", "")
            paper = self.config.get("paper", True)

            self._trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )

            self._data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )

            # Verify connection by fetching account
            account = self._trading_client.get_account()
            self._connected = True

            self._log.info(
                "alpaca_connected",
                paper=paper,
                equity=str(account.equity),
            )

        except ImportError:
            self._log.error(
                "alpaca_not_installed",
                message="Install alpaca-py: pip install 'cerebrumcoin[stocks]'",
            )
            raise
        except Exception as e:
            self._log.error("alpaca_connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca and clean up."""
        self._connected = False

        for task in self._streaming_tasks:
            task.cancel()

        if self._streaming_tasks:
            await asyncio.gather(*self._streaming_tasks, return_exceptions=True)

        self._log.info("alpaca_disconnected")

    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Subscribe to stock market data via polling.

        Args:
            symbols: Stock symbols to subscribe to (e.g., "AAPL", "MSFT")
        """
        if not self._connected or not self._data_client:
            raise RuntimeError("Not connected to Alpaca")

        for symbol in symbols:
            task = asyncio.create_task(
                self._poll_market_data(symbol),
                name=f"alpaca_poll_{symbol}",
            )
            self._streaming_tasks.append(task)

        self._log.info("market_data_subscribed", symbols=symbols)

    _ET = pytz.timezone("America/New_York")
    _MARKET_OPEN = datetime.time(9, 30)
    _MARKET_CLOSE = datetime.time(16, 0)

    def _is_market_open(
        self, now: datetime.datetime | None = None
    ) -> bool:
        """
        Return True if US equity markets are currently open.

        Regular session: Mon-Fri 09:30-16:00 US Eastern Time.
        v1 limitation: NYSE holidays are NOT handled — weekday + time window
        only.  This is a known follow-up (see DEC-ALPACA-002).

        Args:
            now: Timezone-aware datetime to check.  Defaults to current wall
                 clock time in ET.  Accepts any tz-aware datetime — it is
                 converted to ET internally.  Passing a value is primarily for
                 testing.
        """
        if now is None:
            now = datetime.datetime.now(tz=self._ET)
        else:
            now = now.astimezone(self._ET)

        # Weekday check: Mon=0 … Fri=4; Sat=5, Sun=6
        if now.weekday() > 4:
            return False

        current_time = now.time()
        return self._MARKET_OPEN <= current_time < self._MARKET_CLOSE

    async def _poll_market_data(self, symbol: Symbol) -> None:
        """Poll market data for a symbol.

        DEC-ALPACA-002: skips the Alpaca API call when US equity markets are
        closed (_is_market_open() returns False).  During off-hours Alpaca
        returns frozen stale quotes that corrupt the outlier filter's cold-start
        window (Session 48 root cause — see INVESTIGATION_aapl_outlier.md).
        The poll loop continues running and re-checks every poll_interval_seconds
        so it picks up trading naturally when the market opens.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        log = self._log.bind(symbol=symbol)
        log.info("market_data_polling_started")

        try:
            while self._connected and self._data_client:
                # DEC-ALPACA-002: gate on market hours — stale off-hours quotes
                # corrupt the outlier filter's cold-start window.
                if not self._is_market_open():
                    log.debug("market_closed_skipping_poll", symbol=symbol)
                    await asyncio.sleep(self.config.get("poll_interval_seconds", 5))
                    continue

                try:
                    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    quotes = self._data_client.get_stock_latest_quote(request)

                    if symbol in quotes:
                        quote = quotes[symbol]
                        mid_price = (
                            Decimal(str(quote.ask_price)) + Decimal(str(quote.bid_price))
                        ) / Decimal("2")

                        self._current_prices[symbol] = mid_price

                        event = MarketDataEvent(
                            event_type=EventType.MARKET_DATA,
                            timestamp=time(),
                            symbol=symbol,
                            price=mid_price,
                            volume=Decimal(str(quote.ask_size + quote.bid_size)),
                            bid=Decimal(str(quote.bid_price)),
                            ask=Decimal(str(quote.ask_price)),
                            spread=Decimal(str(quote.ask_price - quote.bid_price)),
                        )
                        await self.bus.publish(event)

                except Exception as e:
                    log.warning("poll_error", error=str(e))

                await asyncio.sleep(self.config.get("poll_interval_seconds", 5))

        except asyncio.CancelledError:
            log.info("market_data_polling_cancelled")

    async def execute_order(self, order: OrderEvent) -> None:
        """
        Execute order on Alpaca.

        Args:
            order: Order to execute
        """
        if not self._trading_client:
            self._log.error("not_connected", order_id=order.order_id)
            return

        # @decision DEC-LIVE-002
        # @title Dual-env live gate for Alpaca, conditional on paper=False
        # @status accepted
        # @rationale Extends DEC-LIVE-001 (Kraken, cerebrum/adapters/kraken.py:188)
        # to Alpaca: real-money orders require BOTH TRADING_MODE=live AND
        # ALPACA_LIVE_ENABLED=true. Two independent toggles force deliberate intent
        # and prevent a misconfigured paper session from accidentally placing live
        # orders.
        # Deliberate divergence from Kraken: this gate is only enforced when
        # config["paper"] is False, whereas Kraken's gate is unconditional. The
        # Alpaca SDK uses a distinct paper endpoint (paper-api.alpaca.markets), so
        # paper-mode adapters cannot reach the live brokerage even if env vars
        # leak — making the env check redundant in paper mode. Keeping paper-mode
        # frictionless (no env vars required) is the explicit goal here. Do NOT
        # "consistency-fix" this by removing the `if not paper` guard.
        paper = self.config.get("paper", True)
        if not paper:
            trading_mode = os.getenv("TRADING_MODE", "paper")
            live_enabled = os.getenv("ALPACA_LIVE_ENABLED", "false").lower() == "true"
            if trading_mode != "live" or not live_enabled:
                self._log.warning(
                    "alpaca_live_order_not_enabled",
                    order_id=order.order_id,
                    trading_mode=trading_mode,
                    live_enabled=live_enabled,
                    message=(
                        "Dual safety check failed. Set TRADING_MODE=live "
                        "AND ALPACA_LIVE_ENABLED=true"
                    ),
                )
                return

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=float(order.amount),
                side=side,
                time_in_force=TimeInForce.DAY,
            )

            alpaca_order = self._trading_client.submit_order(request)

            self._log.info(
                "order_submitted",
                order_id=order.order_id,
                alpaca_order_id=str(alpaca_order.id),
                status=str(alpaca_order.status),
            )

            # Poll for fill
            max_polls = 20
            for _ in range(max_polls):
                await asyncio.sleep(0.5)
                filled = self._trading_client.get_order_by_id(alpaca_order.id)

                if str(filled.status) == "filled":
                    # @decision DEC-ALPACA-FIX-001
                    # @title Propagate strategy_id from OrderEvent to FillEvent
                    # @status accepted
                    # @rationale Multi-strategy routing (Conductor) requires FillEvent.strategy_id
                    # to match the originating OrderEvent.strategy_id so fills are attributed to
                    # the correct strategy. Without this, all fills appear unrouted (strategy_id=None),
                    # breaking the Conductor's per-strategy P&L tracking. Matches PaperTradingAdapter
                    # pattern (cerebrum/adapters/paper.py line 251).
                    fill_event = FillEvent(
                        event_type=EventType.FILL,
                        timestamp=time(),
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        filled_amount=Decimal(str(filled.filled_qty)),
                        fill_price=Decimal(str(filled.filled_avg_price)),
                        commission=Decimal("0"),  # Alpaca is commission-free
                        commission_asset="USD",
                        exchange_order_id=str(filled.id),
                        strategy_id=order.strategy_id,
                    )
                    await self.bus.publish(fill_event)

                    self._log.info(
                        "order_filled",
                        order_id=order.order_id,
                        price=str(filled.filled_avg_price),
                        qty=str(filled.filled_qty),
                    )
                    return

            self._log.error(
                "order_fill_timeout",
                order_id=order.order_id,
                alpaca_order_id=str(alpaca_order.id),
            )

        except Exception as e:
            self._log.error(
                "order_execution_failed",
                order_id=order.order_id,
                error=str(e),
            )
            raise

    async def get_balance(self, asset: str) -> Decimal:
        """Get current buying power (USD)."""
        if not self._trading_client:
            raise RuntimeError("Not connected to Alpaca")

        account = self._trading_client.get_account()
        if asset.upper() == "USD":
            return Decimal(str(account.buying_power))
        return Decimal("0")

    async def get_current_price(self, symbol: Symbol) -> Price:
        """Get current market price from tracked data."""
        if symbol in self._current_prices:
            return self._current_prices[symbol]
        raise ValueError(f"No price data for {symbol}")

    async def get_position(self, symbol: Symbol) -> Amount:
        """Get current position size for a symbol."""
        if not self._trading_client:
            return Decimal("0")

        try:
            position = self._trading_client.get_open_position(symbol)
            return Decimal(str(position.qty))
        except Exception:
            return Decimal("0")
