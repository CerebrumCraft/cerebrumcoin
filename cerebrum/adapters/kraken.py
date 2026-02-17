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

        Note: For Phase 1, this is a stub. Real execution comes in Phase 6.
        Paper trading handles order execution.
        """
        self._log.warning(
            "live_order_not_implemented",
            order_id=order.order_id,
            message="Use paper trading adapter for Phase 1"
        )

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
