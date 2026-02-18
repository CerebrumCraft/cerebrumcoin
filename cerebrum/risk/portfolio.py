"""
Portfolio state tracking for risk calculations.

Maintains current positions, balances, and P&L for risk management decisions.

@decision DEC-RISK-002
@title Portfolio state tracking for exposure calculations
@status accepted
@rationale Centralized position tracking enables position sizing, exposure limits,
drawdown detection, and P&L calculations. Subscribes to FillEvents to maintain
accurate state. Provides query interface for risk rules.
"""

from dataclasses import dataclass
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, MarketDataEvent
from cerebrum.core.types import Amount, EventType, Price, Side, Symbol

logger = structlog.get_logger()


@dataclass
class Position:
    """Represents an open position."""
    symbol: Symbol
    amount: Amount  # Positive for long, negative for short
    average_entry_price: Price
    current_price: Price
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    
    def update_price(self, price: Price) -> None:
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = self.amount * (price - self.average_entry_price)


class PortfolioTracker:
    """
    Tracks portfolio state for risk management.
    
    Features:
    - Position tracking per symbol
    - Cash balance management
    - Realized and unrealized P&L
    - Total exposure calculation
    - Peak equity tracking for drawdown
    """
    
    def __init__(
        self,
        bus: EventBus,
        initial_balance: Decimal,
    ) -> None:
        """
        Initialize portfolio tracker.
        
        Args:
            bus: Event bus
            initial_balance: Starting cash balance
        """
        self._bus = bus
        self._cash_balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: dict[Symbol, Position] = {}
        self._peak_equity = initial_balance
        self._total_realized_pnl = Decimal("0.0")

        # Track latest prices for all symbols (needed for market order sizing)
        self._latest_prices: dict[Symbol, Price] = {}

        self._log = logger.bind(component="portfolio_tracker")
        
        # Subscribe to fills and market data
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="portfolio_tracker",
        )
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="portfolio_tracker_prices",
        )
        
        self._log.info("portfolio_tracker_initialized", initial_balance=str(initial_balance))
    
    async def _on_fill(self, event: Event) -> None:
        """Handle fill events to update positions."""
        if not isinstance(event, FillEvent):
            return
        
        symbol = event.symbol
        filled_amount = event.filled_amount
        fill_price = event.fill_price
        commission = event.commission
        
        # Adjust for side
        if event.side == Side.SELL:
            filled_amount = -filled_amount
        
        # Update or create position
        if symbol in self._positions:
            pos = self._positions[symbol]
            
            # Check if closing/reducing position
            if (pos.amount > 0 and filled_amount < 0) or (pos.amount < 0 and filled_amount > 0):
                # Closing trade: realize P&L
                close_amount = min(abs(filled_amount), abs(pos.amount))
                pnl_per_unit = fill_price - pos.average_entry_price
                realized_pnl = close_amount * pnl_per_unit * (1 if pos.amount > 0 else -1)
                
                pos.realized_pnl += realized_pnl
                self._total_realized_pnl += realized_pnl
                
                # Update position
                new_amount = pos.amount + filled_amount
                if abs(new_amount) < Decimal("0.0001"):  # Position closed
                    del self._positions[symbol]
                    self._log.info(
                        "position_closed",
                        symbol=symbol,
                        realized_pnl=str(realized_pnl),
                    )
                else:
                    pos.amount = new_amount
            else:
                # Adding to position: update average entry
                total_cost = (pos.amount * pos.average_entry_price) + (filled_amount * fill_price)
                new_amount = pos.amount + filled_amount
                pos.average_entry_price = total_cost / new_amount
                pos.amount = new_amount
        else:
            # New position
            self._positions[symbol] = Position(
                symbol=symbol,
                amount=filled_amount,
                average_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
            )
            self._log.info(
                "position_opened",
                symbol=symbol,
                amount=str(filled_amount),
                price=str(fill_price),
            )
        
        # Update cash balance
        cost = filled_amount * fill_price
        self._cash_balance -= cost + commission
        
        # Update peak equity
        equity = self.get_total_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity
    
    async def _on_market_data(self, event: Event) -> None:
        """Update position prices with market data."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        # Cache latest price for all symbols (used for market order sizing)
        self._latest_prices[symbol] = event.price

        # Update position price if we have one
        if symbol in self._positions:
            self._positions[symbol].update_price(event.price)
    
    def get_position(self, symbol: Symbol) -> Position | None:
        """Get current position for a symbol."""
        return self._positions.get(symbol)
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash_balance
    
    def get_total_equity(self) -> Decimal:
        """Get total equity (cash + position market values)."""
        position_value = sum(
            abs(pos.amount) * pos.current_price
            for pos in self._positions.values()
        )
        return self._cash_balance + position_value
    
    def get_total_exposure(self) -> Decimal:
        """Get total position exposure (sum of position values)."""
        return sum(
            abs(pos.amount * pos.current_price)
            for pos in self._positions.values()
        )
    
    def get_drawdown_percent(self) -> Decimal:
        """Get current drawdown from peak equity."""
        equity = self.get_total_equity()
        if self._peak_equity == 0:
            return Decimal("0.0")
        drawdown = (self._peak_equity - equity) / self._peak_equity * 100
        return max(Decimal("0.0"), drawdown)
    
    def get_pnl(self) -> tuple[Decimal, Decimal]:
        """Get (realized_pnl, unrealized_pnl)."""
        unrealized = sum(pos.unrealized_pnl for pos in self._positions.values())
        return self._total_realized_pnl, unrealized

    def get_latest_price(self, symbol: Symbol) -> Price | None:
        """Get the latest known price for a symbol."""
        return self._latest_prices.get(symbol)
