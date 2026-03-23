"""
Paper trading execution engine.

Simulates order execution with realistic slippage and commissions.
Maintains portfolio state in memory with file persistence.

@decision DEC-PAPER-001
@title File-based state persistence for paper trading
@status accepted
@rationale Simple JSON file persists balances and positions across restarts. No database
needed for Phase 1. State includes: balances (USD, BTC, etc.), open positions, trade history.
Atomic writes prevent corruption. Scales to thousands of trades before needing optimization.
"""

import asyncio
import json
import uuid
from decimal import Decimal
from pathlib import Path
from time import time
from typing import Any

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


class PaperTradingAdapter(ExchangeAdapter):
    """
    Paper trading simulator with realistic execution.

    Features:
    - Simulated slippage and commissions
    - Instant fills at market price + slippage
    - Portfolio state persistence
    - No real money at risk
    """

    def __init__(
        self,
        bus: EventBus,
        config: dict[str, Any],
        initial_balance: Decimal,
        commission_percent: Decimal,
        slippage_percent: Decimal,
        state_file: Path,
    ) -> None:
        """
        Initialize paper trading adapter.

        Args:
            bus: Event bus
            config: Adapter configuration
            initial_balance: Starting USD balance
            commission_percent: Commission as % of trade value
            slippage_percent: Slippage as % of price
            state_file: Path to state persistence file
        """
        super().__init__(bus, config)
        self._initial_balance = initial_balance
        self._commission_percent = commission_percent
        self._slippage_percent = slippage_percent
        self._state_file = state_file

        # Portfolio state
        self._balances: dict[str, Decimal] = {}
        self._positions: dict[Symbol, Decimal] = {}
        self._current_prices: dict[Symbol, Price] = {}
        self._trade_history: list[dict[str, Any]] = []

        self._log = logger.bind(adapter="paper_trading")

    async def connect(self) -> None:
        """Initialize paper trading state."""
        # Load state from file if exists
        if self._state_file.exists():
            self._load_state()
            self._log.info("paper_state_loaded", state_file=str(self._state_file))
        else:
            # Initialize fresh state
            self._balances = {"USD": self._initial_balance}
            self._positions = {}
            self._trade_history = []
            self._save_state()
            self._log.info(
                "paper_state_initialized",
                initial_balance=str(self._initial_balance)
            )

        # Subscribe to market data to track current prices
        self.bus.subscribe(
            EventType.MARKET_DATA,
            self._handle_market_data,
            "paper_trading_price_tracker"
        )

        # Subscribe to order events
        self.bus.subscribe(
            EventType.ORDER,
            self.execute_order,
            "paper_trading_executor"
        )

    async def disconnect(self) -> None:
        """Save state and disconnect."""
        self._save_state()
        self._log.info("paper_trading_disconnected")

    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Paper trading doesn't stream data—it relies on real adapter.

        Args:
            symbols: Ignored (market data comes from Kraken adapter)
        """
        self._log.info("paper_trading_uses_external_market_data")

    async def _handle_market_data(self, event: MarketDataEvent) -> None:
        """Track current prices from market data events."""
        self._current_prices[event.symbol] = event.price

    async def execute_order(self, order: OrderEvent) -> None:
        """
        Simulate order execution.

        Args:
            order: Order to execute
        """
        log = self._log.bind(order_id=order.order_id)

        try:
            # Get current price
            if order.symbol not in self._current_prices:
                log.warning("no_price_data", symbol=order.symbol)
                return

            current_price = self._current_prices[order.symbol]

            # Calculate execution price with slippage
            if order.side == Side.BUY:
                slippage_factor = Decimal("1") + (self._slippage_percent / Decimal("100"))
                fill_price = current_price * slippage_factor
            else:  # SELL
                slippage_factor = Decimal("1") - (self._slippage_percent / Decimal("100"))
                fill_price = current_price * slippage_factor

            # Calculate commission
            trade_value = fill_price * order.amount
            commission = trade_value * (self._commission_percent / Decimal("100"))

            # Check if we have sufficient balance
            base_asset, quote_asset = order.symbol.split("/")

            if order.side == Side.BUY:
                required_usd = trade_value + commission
                if self._balances.get(quote_asset, Decimal("0")) < required_usd:
                    log.warning(
                        "insufficient_balance",
                        required=str(required_usd),
                        available=str(self._balances.get(quote_asset, Decimal("0")))
                    )
                    return

                # Execute buy
                self._balances[quote_asset] = self._balances.get(
                    quote_asset, Decimal("0")
                ) - required_usd
                self._positions[order.symbol] = self._positions.get(
                    order.symbol, Decimal("0")
                ) + order.amount

            else:  # SELL
                if self._positions.get(order.symbol, Decimal("0")) < order.amount:
                    log.warning(
                        "insufficient_position",
                        required=str(order.amount),
                        available=str(self._positions.get(order.symbol, Decimal("0")))
                    )
                    return

                # Execute sell
                self._positions[order.symbol] = self._positions.get(
                    order.symbol, Decimal("0")
                ) - order.amount
                self._balances[quote_asset] = self._balances.get(
                    quote_asset, Decimal("0")
                ) + trade_value - commission

            # Record trade
            trade_record = {
                "order_id": order.order_id,
                "timestamp": time(),
                "symbol": order.symbol,
                "side": order.side.value,
                "amount": str(order.amount),
                "fill_price": str(fill_price),
                "commission": str(commission),
            }
            self._trade_history.append(trade_record)

            # Publish fill event — propagate strategy_id from order for multi-strategy routing
            fill_event = FillEvent(
                event_type=EventType.FILL,
                timestamp=time(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                filled_amount=order.amount,
                fill_price=fill_price,
                commission=commission,
                commission_asset=quote_asset,
                exchange_order_id=f"paper_{uuid.uuid4().hex[:8]}",
                strategy_id=order.strategy_id,
            )
            await self.bus.publish(fill_event)

            # Save state
            self._save_state()

            log.info(
                "order_filled",
                symbol=order.symbol,
                side=order.side.value,
                amount=str(order.amount),
                fill_price=str(fill_price),
                commission=str(commission),
            )

        except Exception as e:
            log.error("order_execution_error", error=str(e))

    async def get_balance(self, asset: str) -> Decimal:
        """Get current balance for an asset."""
        return self._balances.get(asset, Decimal("0"))

    async def get_current_price(self, symbol: Symbol) -> Price:
        """Get current market price from tracked data."""
        if symbol not in self._current_prices:
            raise ValueError(f"No price data for {symbol}")
        return self._current_prices[symbol]

    async def get_position(self, symbol: Symbol) -> Amount:
        """Get current position size."""
        return self._positions.get(symbol, Decimal("0"))

    def _save_state(self) -> None:
        """Persist state to file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "balances": {k: str(v) for k, v in self._balances.items()},
            "positions": {k: str(v) for k, v in self._positions.items()},
            "current_prices": {k: str(v) for k, v in self._current_prices.items()},
            "trade_history": self._trade_history,
        }

        # Atomic write
        temp_file = self._state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self._state_file)

    def _load_state(self) -> None:
        """Load state from file."""
        try:
            with open(self._state_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    # Empty file, initialize fresh state
                    self._balances = {"USD": self._initial_balance}
                    self._positions = {}
                    self._current_prices = {}
                    self._trade_history = []
                    return

                state = json.loads(content)

            self._balances = {k: Decimal(v) for k, v in state.get("balances", {}).items()}
            self._positions = {k: Decimal(v) for k, v in state.get("positions", {}).items()}
            self._current_prices = {k: Decimal(v) for k, v in state.get("current_prices", {}).items()}
            self._trade_history = state.get("trade_history", [])
        except (json.JSONDecodeError, ValueError):
            # Corrupt file, reinitialize
            self._balances = {"USD": self._initial_balance}
            self._positions = {}
            self._current_prices = {}
            self._trade_history = []

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get current portfolio state for monitoring."""
        total_value = self._balances.get("USD", Decimal("0"))

        # Add value of positions
        for symbol, amount in self._positions.items():
            if amount > 0 and symbol in self._current_prices:
                total_value += amount * self._current_prices[symbol]

        return {
            "balances": {k: str(v) for k, v in self._balances.items()},
            "positions": {k: str(v) for k, v in self._positions.items()},
            "total_value_usd": str(total_value),
            "trade_count": len(self._trade_history),
            "pnl_usd": str(total_value - self._initial_balance),
        }
