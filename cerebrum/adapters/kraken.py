"""
Kraken exchange adapter with WebSocket market data streaming.

Connects to Kraken via ccxt.pro for real-time price feeds.

@decision DEC-KRAKEN-001
@title ccxt.pro async WebSocket for real-time data
@status accepted
@rationale ccxt.pro provides unified WebSocket interface across exchanges with automatic
reconnection. Async watchTicker streams price updates without polling. Kraken-specific
symbol mapping handled internally (BTC/USD -> XBT/USD).
"""

import asyncio
import os
from datetime import datetime, timezone
from decimal import Decimal
from time import time
from typing import Any

import ccxt.pro as ccxtpro
import structlog

from cerebrum.adapters.base import ExchangeAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import (
    Amount,
    EventType,
    OrderStatus,
    OrderType,
    Price,
    Side,
    Symbol,
)

logger = structlog.get_logger()


class KrakenAdapter(ExchangeAdapter):
    """
    Kraken exchange adapter with real-time WebSocket streaming.

    Features:
    - WebSocket market data via ccxt.pro
    - Automatic reconnection on disconnects
    - Rate limiting
    - Live order execution (Phase 6)

    Note: ccxt.pro handles Kraken's BTC/XBT mapping internally,
    so we pass standard symbols (BTC/USD) directly to the exchange.
    """

    def __init__(self, bus: EventBus, config: dict[str, Any]) -> None:
        """
        Initialize Kraken adapter.

        Args:
            bus: Event bus for publishing market data
            config: Configuration with api_key, api_secret, etc.
        """
        super().__init__(bus, config)
        self._exchange: ccxtpro.kraken | None = None
        self._streaming_tasks: list[asyncio.Task[None]] = []
        self._connected = False
        self._log = logger.bind(adapter="kraken")

    async def connect(self) -> None:
        """Connect to Kraken exchange."""
        try:
            self._exchange = ccxtpro.kraken({
                'apiKey': self.config.get('api_key', ''),
                'secret': self.config.get('api_secret', ''),
                'enableRateLimit': True,
                'rateLimit': 60000 / self.config.get('rate_limit_per_minute', 60),
            })

            # Test connection by fetching markets
            await self._exchange.load_markets()
            self._connected = True
            self._log.info("kraken_connected")

        except Exception as e:
            self._log.error("kraken_connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to Kraken: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Kraken and clean up."""
        self._connected = False

        # Cancel streaming tasks
        for task in self._streaming_tasks:
            task.cancel()

        if self._streaming_tasks:
            await asyncio.gather(*self._streaming_tasks, return_exceptions=True)

        # Close exchange connection
        if self._exchange:
            await self._exchange.close()

        self._log.info("kraken_disconnected")

    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Subscribe to real-time market data via WebSocket.

        Args:
            symbols: Trading pairs to subscribe to (standard format, e.g., BTC/USD)
        """
        if not self._connected or not self._exchange:
            raise RuntimeError("Not connected to Kraken")

        # ccxt.pro handles symbol mapping internally (BTC/USD -> XBT/USD)
        for symbol in symbols:
            task = asyncio.create_task(
                self._stream_ticker(symbol),
                name=f"kraken_stream_{symbol}"
            )
            self._streaming_tasks.append(task)

        self._log.info("market_data_subscribed", symbols=symbols)

    async def _stream_ticker(self, symbol: Symbol) -> None:
        """
        Stream ticker data for a symbol via WebSocket.

        Publishes MarketDataEvent to the bus on each update.
        """
        log = self._log.bind(symbol=symbol)
        log.info("ticker_stream_started")

        try:
            while self._connected and self._exchange:
                try:
                    # Watch ticker updates (WebSocket)
                    ticker = await self._exchange.watch_ticker(symbol)

                    # Parse ticker data
                    event = MarketDataEvent(
                        event_type=EventType.MARKET_DATA,
                        timestamp=time(),
                        symbol=symbol,  # Use symbol as-is (ccxt.pro handles mapping)
                        price=Decimal(str(ticker['last'])),
                        volume=Decimal(str(ticker.get('baseVolume', 0))),
                        bid=Decimal(str(ticker['bid'])) if ticker.get('bid') else None,
                        ask=Decimal(str(ticker['ask'])) if ticker.get('ask') else None,
                        spread=Decimal(str(ticker['ask'] - ticker['bid']))
                        if ticker.get('ask') and ticker.get('bid') else None,
                    )

                    # Publish to event bus
                    await self.bus.publish(event)
                    log.info("market_data_published", price=float(event.price),
                            bid=float(event.bid) if event.bid else None,
                            ask=float(event.ask) if event.ask else None)

                except ccxtpro.NetworkError as e:
                    log.warning("network_error", error=str(e))
                    await asyncio.sleep(5)  # Backoff before retry

                except Exception as e:
                    log.error("stream_error", error=str(e))
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            log.info("ticker_stream_cancelled")
        finally:
            log.info("ticker_stream_stopped")

    async def execute_order(self, order: OrderEvent) -> None:
        """
        Execute order on Kraken.

        @decision DEC-LIVE-001
        Safety: Requires both TRADING_MODE=live AND KRAKEN_LIVE_ENABLED=true env vars.
        Uses ccxt create_market_order for execution, polls fetch_order for fill status.
        
        Args:
            order: Order to execute
            
        Raises:
            RuntimeError: If live trading not enabled
            ccxt.InvalidOrder: If order is invalid
            ccxt.InsufficientFunds: If insufficient balance
        """
        # Safety check: dual-gate for live trading
        trading_mode = os.getenv("TRADING_MODE", "paper")
        live_enabled = os.getenv("KRAKEN_LIVE_ENABLED", "false").lower() == "true"
        
        if trading_mode != "live" or not live_enabled:
            self._log.warning(
                "live_order_not_enabled",
                order_id=order.order_id,
                trading_mode=trading_mode,
                live_enabled=live_enabled,
                message="Dual safety check failed. Set TRADING_MODE=live AND KRAKEN_LIVE_ENABLED=true"
            )
            return
        
        self._log.warning(
            "executing_live_order",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            amount=str(order.amount),
            message="REAL MONEY ORDER - This will execute on live exchange"
        )
        
        try:
            # Convert to ccxt format
            side_str = "buy" if order.side == Side.BUY else "sell"
            amount_float = float(order.amount)
            
            # Execute market order via ccxt
            ccxt_order = await self._exchange.create_market_order(
                symbol=order.symbol,
                side=side_str,
                amount=amount_float,
            )
            
            self._log.info(
                "order_submitted",
                order_id=order.order_id,
                ccxt_order_id=ccxt_order["id"],
                status=ccxt_order["status"],
            )
            
            # Poll for fill (Kraken usually fills market orders immediately)
            max_polls = 10
            for i in range(max_polls):
                await asyncio.sleep(0.5)  # Wait 500ms between polls
                
                filled_order = await self._exchange.fetch_order(
                    ccxt_order["id"],
                    symbol=order.symbol,
                )
                
                if filled_order["status"] == "closed":
                    # Order filled - publish FillEvent
                    fill_price = Decimal(str(filled_order["average"]))
                    fill_amount = Decimal(str(filled_order["filled"]))
                    
                    fill_event = FillEvent(
                        event_type=EventType.FILL,
                        timestamp=time(),
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        fill_price=fill_price,
                        filled_amount=fill_amount,
                        commission=Decimal(str(filled_order.get("fee", {}).get("cost", 0))),
                        commission_asset="USD",
                        exchange_order_id=str(filled_order["id"]),
                    )
                    
                    await self.bus.publish(fill_event)
                    
                    self._log.info(
                        "order_filled",
                        order_id=order.order_id,
                        price=str(fill_price),
                        amount=str(fill_amount),
                    )
                    return
                    
            # Timeout - order didn't fill in time
            self._log.error(
                "order_fill_timeout",
                order_id=order.order_id,
                ccxt_order_id=ccxt_order["id"],
                status=filled_order["status"],
            )
            
        except Exception as e:
            self._log.error(
                "order_execution_failed",
                order_id=order.order_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    async def get_balance(self, asset: str) -> Decimal:
        """Get current balance for an asset."""
        if not self._exchange:
            raise RuntimeError("Not connected to Kraken")

        balance = await self._exchange.fetch_balance()
        return Decimal(str(balance.get(asset, {}).get('free', 0)))

    async def get_current_price(self, symbol: Symbol) -> Price:
        """Get current market price (ccxt.pro handles symbol mapping)."""
        if not self._exchange:
            raise RuntimeError("Not connected to Kraken")

        ticker = await self._exchange.fetch_ticker(symbol)
        return Decimal(str(ticker['last']))
