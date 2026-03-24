"""
Portfolio state tracking for risk calculations.

Maintains current positions, balances, and P&L for risk management decisions.

@decision DEC-RISK-002
@title Portfolio state tracking for exposure calculations
@status accepted
@rationale Centralized position tracking enables position sizing, exposure limits,
drawdown detection, and P&L calculations. Subscribes to FillEvents to maintain
accurate state. Provides query interface for risk rules.

@decision DEC-RISK-004
@title strategy_id filtering in PortfolioTracker prevents double-counting in multi-strategy mode
@status accepted
@rationale In multi-strategy mode, each strategy has its own PortfolioTracker. All
FillEvents share one event bus, so without filtering every PortfolioTracker would
process every fill — causing triple-counting of positions and incorrect P&L. When
strategy_id is provided at construction, _on_fill skips events whose strategy_id
does not match. None means "accept all" (single-strategy backward compatibility).
"""

import time as time_module
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, MarketDataEvent, PositionUpdateEvent
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
    entry_time: float = field(default_factory=time_module.time)  # Unix epoch seconds

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
        strategy_id: str | None = None,
    ) -> None:
        """
        Initialize portfolio tracker.

        Args:
            bus: Event bus
            initial_balance: Starting cash balance
            strategy_id: When provided, _on_fill ignores FillEvents whose
                strategy_id does not match. None accepts all fills (single-
                strategy backward-compatible mode). See DEC-RISK-004.
        """
        self._bus = bus
        self._cash_balance = initial_balance
        self._initial_balance = initial_balance
        self._strategy_id = strategy_id
        self._positions: dict[Symbol, Position] = {}
        self._peak_equity = initial_balance
        self._total_realized_pnl = Decimal("0.0")

        # Track latest prices for all symbols (needed for market order sizing)
        self._latest_prices: dict[Symbol, Price] = {}

        self._log = logger.bind(
            component="portfolio_tracker",
            strategy_id=strategy_id or "global",
        )

        # Use a unique subscriber name so multiple PortfolioTrackers on the same
        # bus do not collide. When strategy_id is None the legacy name is preserved
        # for backward compatibility with tests that assert subscriber counts.
        fill_sub_name = (
            f"portfolio_tracker_{strategy_id}" if strategy_id else "portfolio_tracker"
        )
        price_sub_name = (
            f"portfolio_tracker_prices_{strategy_id}"
            if strategy_id
            else "portfolio_tracker_prices"
        )

        # Subscribe to fills and market data
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name=fill_sub_name,
        )
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name=price_sub_name,
        )

        self._log.info("portfolio_tracker_initialized", initial_balance=str(initial_balance))

    async def _on_fill(self, event: Event) -> None:
        """Handle fill events to update positions."""
        if not isinstance(event, FillEvent):
            return

        # DEC-RISK-004: In multi-strategy mode each PortfolioTracker must only
        # process fills that belong to its own strategy. strategy_id=None means
        # accept all (single-strategy backward-compat path).
        if self._strategy_id is not None:
            event_sid = getattr(event, "strategy_id", None)
            if event_sid != self._strategy_id:
                return

        symbol = event.symbol
        filled_amount = event.filled_amount
        fill_price = event.fill_price
        commission = event.commission

        # Adjust for side
        if event.side == Side.SELL:
            filled_amount = -filled_amount

        # @decision DEC-DASH-001: Publish PositionUpdateEvent on fills only (not price ticks)
        # to keep dashboard updated without flooding the event bus.
        #
        # Update or create position, then publish PositionUpdateEvent so subscribers
        # (e.g. the monitoring dashboard) stay in sync with current position state.
        # On position closure we publish amount=0 so subscribers can remove the entry.
        position_event: PositionUpdateEvent | None = None

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
                    # Capture state before deletion so we can publish event
                    closed_entry_price = pos.average_entry_price
                    closed_realized_pnl = pos.realized_pnl
                    del self._positions[symbol]
                    self._log.info(
                        "position_closed",
                        symbol=symbol,
                        realized_pnl=str(realized_pnl),
                    )
                    # amount=0 signals closure to subscribers
                    position_event = PositionUpdateEvent(
                        event_type=EventType.POSITION_UPDATE,
                        timestamp=event.timestamp,
                        symbol=symbol,
                        amount=Decimal("0"),
                        average_entry_price=closed_entry_price,
                        current_price=fill_price,
                        unrealized_pnl=Decimal("0.0"),
                        realized_pnl=closed_realized_pnl,
                    )
                else:
                    pos.amount = new_amount
                    position_event = PositionUpdateEvent(
                        event_type=EventType.POSITION_UPDATE,
                        timestamp=event.timestamp,
                        symbol=symbol,
                        amount=pos.amount,
                        average_entry_price=pos.average_entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=pos.realized_pnl,
                    )
            else:
                # Adding to position: update average entry (preserve original entry_time)
                total_cost = (pos.amount * pos.average_entry_price) + (filled_amount * fill_price)
                new_amount = pos.amount + filled_amount
                pos.average_entry_price = total_cost / new_amount
                pos.amount = new_amount
                position_event = PositionUpdateEvent(
                    event_type=EventType.POSITION_UPDATE,
                    timestamp=event.timestamp,
                    symbol=symbol,
                    amount=pos.amount,
                    average_entry_price=pos.average_entry_price,
                    current_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                )
        else:
            # New position — record entry time at fill event timestamp
            self._positions[symbol] = Position(
                symbol=symbol,
                amount=filled_amount,
                average_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
                entry_time=event.timestamp,
            )
            self._log.info(
                "position_opened",
                symbol=symbol,
                amount=str(filled_amount),
                price=str(fill_price),
            )
            position_event = PositionUpdateEvent(
                event_type=EventType.POSITION_UPDATE,
                timestamp=event.timestamp,
                symbol=symbol,
                amount=filled_amount,
                average_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
            )

        # Update cash balance
        cost = filled_amount * fill_price
        self._cash_balance -= cost + commission

        # Update peak equity
        equity = self.get_total_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Publish position state so dashboard and other subscribers stay current.
        # Published after cash/equity updates so subscribers see consistent state.
        if position_event is not None:
            await self._bus.publish(position_event)

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

    def get_all_positions(self) -> dict[Symbol, "Position"]:
        """Get all open positions (snapshot copy)."""
        return dict(self._positions)

    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash_balance

    def get_total_equity(self) -> Decimal:
        """Get total equity (cash + signed position market values).

        Short positions (negative amount) correctly reduce equity since the
        trader owes the asset. Long positions (positive amount) increase equity.
        Do NOT use abs() here — that was the double-counting bug (DEC-RISK-003).
        """
        position_value = sum(
            pos.amount * pos.current_price  # signed: long adds, short subtracts
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

    def adjust_balance(self, delta: Decimal) -> None:
        """
        Adjust cash balance by delta (positive = add, negative = remove).

        Used by the Conductor to redistribute capital between strategy
        portfolios when Darwinian allocation changes. Also updates peak equity
        so drawdown calculations remain meaningful after capital injection.

        Args:
            delta: Amount to add (positive) or remove (negative) from cash.
        """
        self._cash_balance += delta
        new_equity = self.get_total_equity()
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity
        self._log.debug(
            "balance_adjusted",
            delta=str(delta),
            new_cash_balance=str(self._cash_balance),
        )
