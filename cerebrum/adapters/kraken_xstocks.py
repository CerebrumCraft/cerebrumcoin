"""
Kraken xStocks adapter for 24/7 tokenized US equity trading.

Streams AAPLx/USD, MSFTx/USD, etc. via the Kraken WebSocket API v2
(python-kraken-sdk) and publishes MarketDataEvent to the shared event bus.

Tokenized equities on Kraken trade around the clock, including weekends,
making them suitable for a crypto-native bot without market-hours gating.

@decision DEC-XSTOCKS-001
@title KrakenXStocksAdapter — 24/7 tokenized equities via python-kraken-sdk
@status accepted
@rationale Kraken lists tokenized US equities (xStocks) as first-class trading
pairs (e.g., AAPLx/USD) on the same WS infrastructure as crypto. Using
python-kraken-sdk's SpotWSClient with its proven v2 WebSocket protocol gives
us real-time ticker feeds without a separate data vendor. The x-suffix format
(AAPLx/USD) is Kraken's canonical naming — we preserve it without mapping.
Credentials are read from EXCHANGE_API_KEY / EXCHANGE_API_SECRET to share
the same env var namespace as the existing KrakenAdapter.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from time import time
from typing import Any

import structlog

from cerebrum.adapters.base import ExchangeAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import (
    Amount,
    EventType,
    Price,
    Side,
    Symbol,
)

logger = structlog.get_logger()


class _XStocksWSClient:
    """
    Thin wrapper around SpotWSClient that routes ticker messages to the adapter.

    Subclassing SpotWSClient is the recommended usage pattern from the
    python-kraken-sdk docs. We override on_message to parse ticker channel
    updates and call the adapter's _handle_ticker_update helper.

    This class is instantiated lazily inside connect() so that the adapter
    can be constructed without hitting the network.
    """

    def __init__(self, adapter: KrakenXStocksAdapter, key: str, secret: str) -> None:
        # Deferred import: python-kraken-sdk is an optional dependency.
        from kraken.spot import SpotWSClient  # type: ignore[import]

        self._adapter = adapter
        self._log = adapter._log.bind(component="ws_client")

        # Build a subclass at runtime so we can override on_message without
        # a module-level class definition that would force the import at load time.
        adapter_ref = adapter
        log_ref = self._log

        class _Inner(SpotWSClient):
            async def on_message(self, message: dict | list) -> None:  # type: ignore[override]
                await _XStocksWSClient._dispatch(adapter_ref, log_ref, message)

        self._client = _Inner(key=key, secret=secret)

    async def start(self) -> None:
        await self._client.start()

    async def close(self) -> None:
        await self._client.close()

    async def subscribe(self, params: dict) -> None:
        await self._client.subscribe(params=params)

    @property
    def exception_occur(self) -> bool:
        return self._client.exception_occur

    @staticmethod
    async def _dispatch(
        adapter: KrakenXStocksAdapter,
        log: Any,
        message: dict | list,
    ) -> None:
        """Parse a raw WS message and call _handle_ticker_update for ticker data."""
        if not isinstance(message, dict):
            return

        channel = message.get("channel")
        msg_type = message.get("type")

        # Skip heartbeat, status, subscription acks — only process data updates
        if channel != "ticker":
            return
        if msg_type not in ("update", "snapshot"):
            return

        data = message.get("data")
        if not data or not isinstance(data, list):
            return

        for tick in data:
            if not isinstance(tick, dict):
                continue

            symbol = tick.get("symbol")
            if not symbol:
                continue

            try:
                last = Decimal(str(tick.get("last", 0)))
                bid = Decimal(str(tick.get("bid", 0)))
                ask = Decimal(str(tick.get("ask", 0)))
                volume = Decimal(str(tick.get("volume", 0)))
            except Exception as exc:
                log.warning("xstocks_ticker_parse_error", symbol=symbol, error=str(exc))
                continue

            await adapter._handle_ticker_update(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
            )


class KrakenXStocksAdapter(ExchangeAdapter):
    """
    Kraken xStocks exchange adapter for 24/7 tokenized equity trading.

    Features:
    - WebSocket ticker feed via python-kraken-sdk SpotWSClient v2
    - Publishes MarketDataEvent for each ticker update
    - Preserves Kraken's canonical AAPLx/USD symbol format
    - Credentials from EXCHANGE_API_KEY / EXCHANGE_API_SECRET env vars

    Usage::

        adapter = KrakenXStocksAdapter(bus, {"symbols": ["AAPLx/USD", "MSFTx/USD"]})
        await adapter.connect()
        await adapter.subscribe_market_data(["AAPLx/USD", "MSFTx/USD"])
        # ... MarketDataEvents flow through the bus ...
        await adapter.disconnect()
    """

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        """
        Initialize adapter.

        Args:
            bus: Event bus for publishing MarketDataEvent
            config: Must include ``symbols`` list (e.g. ["AAPLx/USD"]).
                    Optionally ``poll_interval_seconds`` for reconnect waits.

        Raises:
            RuntimeError: If EXCHANGE_API_KEY or EXCHANGE_API_SECRET are absent.
        """
        super().__init__(bus, config)
        self._log = logger.bind(adapter="kraken_xstocks")

        api_key = os.environ.get("EXCHANGE_API_KEY", "")
        api_secret = os.environ.get("EXCHANGE_API_SECRET", "")

        if not api_key or not api_secret:
            self._log.error("kraken_xstocks_credentials_missing")
            raise RuntimeError("kraken_xstocks_credentials_missing")

        self._api_key: str = api_key
        self._api_secret: str = api_secret
        self._symbols: list[Symbol] = list(config.get("symbols", []))
        self._ws: _XStocksWSClient | None = None
        self._connected: bool = False
        self._current_prices: dict[Symbol, Price] = {}

    # ------------------------------------------------------------------
    # ExchangeAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Kraken WebSocket API and prepare for subscriptions."""
        try:
            self._ws = _XStocksWSClient(
                adapter=self,
                key=self._api_key,
                secret=self._api_secret,
            )
            await self._ws.start()
            self._connected = True
            self._log.info("kraken_xstocks_connected")
        except Exception as exc:
            self._log.error("kraken_xstocks_connection_failed", error=str(exc))
            raise ConnectionError(f"Failed to connect to Kraken xStocks WS: {exc}") from exc

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as exc:
                self._log.warning("kraken_xstocks_close_error", error=str(exc))
        self._log.info("kraken_xstocks_disconnected")

    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Subscribe to ticker feed for the given symbols.

        Args:
            symbols: xStock pairs to subscribe, e.g. ["AAPLx/USD", "MSFTx/USD"]

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._ws is None:
            raise RuntimeError("Not connected to Kraken xStocks WS")

        await self._ws.subscribe({"channel": "ticker", "symbol": symbols})
        self._log.info("kraken_xstocks_subscribed", symbols=symbols)

    async def execute_order(self, order: OrderEvent) -> None:
        """
        Order execution placeholder.

        xStocks order execution is out of scope for Phase 14 (data feed only).
        Raises NotImplementedError to fail loudly if accidentally called.
        """
        raise NotImplementedError(
            "KrakenXStocksAdapter: order execution not yet implemented. "
            "Use PaperTradingAdapter for xStocks paper trading."
        )

    async def get_balance(self, asset: str) -> Decimal:
        """Balance queries not yet implemented for xStocks adapter."""
        raise NotImplementedError(
            "KrakenXStocksAdapter: get_balance not implemented. "
            "Query balance via REST KrakenAdapter or PaperTradingAdapter."
        )

    async def get_current_price(self, symbol: Symbol) -> Price:
        """Return latest tracked price for a symbol."""
        if symbol in self._current_prices:
            return self._current_prices[symbol]
        raise ValueError(f"No price data for {symbol}")

    async def get_position(self, symbol: Symbol) -> Amount:
        """Position tracking deferred to PaperTradingAdapter."""
        return Decimal("0")

    # ------------------------------------------------------------------
    # Internal helpers (callable from tests without live WS)
    # ------------------------------------------------------------------

    async def _handle_ticker_update(
        self,
        symbol: Symbol,
        bid: Decimal,
        ask: Decimal,
        last: Decimal,
        volume: Decimal,
    ) -> None:
        """
        Build and publish a MarketDataEvent for a single ticker tick.

        This method is intentionally separated from the WS callback so unit
        tests can call it directly with deterministic values, keeping tests
        fast and connection-free.

        Args:
            symbol: Trading pair (e.g. "AAPLx/USD")
            bid:    Best bid price
            ask:    Best ask price
            last:   Last traded price
            volume: 24-hour volume
        """
        spread = ask - bid if ask and bid else Decimal("0")
        mid = last  # Use last trade as canonical price, consistent with KrakenAdapter

        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol=symbol,
            price=mid,
            volume=volume,
            bid=bid,
            ask=ask,
            spread=spread if spread > 0 else None,
        )

        self._current_prices[symbol] = mid
        await self.bus.publish(event)

        self._log.debug(
            "kraken_xstocks_tick",
            symbol=symbol,
            price=float(mid),
            bid=float(bid),
            ask=float(ask),
        )
