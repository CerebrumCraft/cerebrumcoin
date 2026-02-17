"""
Base exchange adapter interface.

All exchange adapters (Kraken, Binance, paper trading, etc.) implement this interface.

@decision DEC-ADAPTER-001
@title Abstract adapter interface for exchange independence
@status accepted
@rationale Allows swapping between exchanges (Kraken, Binance, paper) without changing
core logic. All adapters publish MarketDataEvent and subscribe to OrderEvent. This enables
multi-exchange trading and backtesting with the same codebase.
"""

from abc import ABC, abstractmethod
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent
from cerebrum.core.types import Amount, Price, Side, Symbol

logger = structlog.get_logger()


class ExchangeAdapter(ABC):
    """
    Abstract base class for all exchange adapters.

    Adapters handle:
    - Market data streaming (publishes MarketDataEvent)
    - Order execution (subscribes to OrderEvent, publishes FillEvent)
    - Exchange-specific API nuances
    """

    def __init__(self, bus: EventBus, config: dict[str, any]) -> None:
        """
        Initialize adapter.

        Args:
            bus: Event bus for publishing/subscribing
            config: Adapter-specific configuration
        """
        self.bus = bus
        self.config = config
        self._log = logger.bind(adapter=self.__class__.__name__)

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to exchange and start data streaming.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange and clean up resources."""
        pass

    @abstractmethod
    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Subscribe to real-time market data for symbols.

        Publishes MarketDataEvent to the bus.

        Args:
            symbols: Trading pairs to subscribe to
        """
        pass

    @abstractmethod
    async def execute_order(self, order: OrderEvent) -> None:
        """
        Execute an order on the exchange.

        Publishes FillEvent when order fills.

        Args:
            order: Order to execute
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Decimal:
        """
        Get current balance for an asset.

        Args:
            asset: Asset symbol (e.g., "USD", "BTC")

        Returns:
            Current balance
        """
        pass

    @abstractmethod
    async def get_current_price(self, symbol: Symbol) -> Price:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Current price
        """
        pass

    async def get_position(self, symbol: Symbol) -> Amount:
        """
        Get current position size for a symbol.

        Default implementation returns 0. Override if exchange has position API.

        Args:
            symbol: Trading pair

        Returns:
            Position size (positive for long, negative for short)
        """
        return Decimal("0")
