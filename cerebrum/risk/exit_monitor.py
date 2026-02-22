"""
Exit monitor for automatic position management.

Proactively watches open positions and emits exit orders when stop-loss,
take-profit, or time-based criteria are met.

@decision DEC-EXIT-001
@title ExitMonitor as separate component from RiskManager
@status accepted
@rationale RiskRules evaluate proposed orders (reactive). ExitMonitor watches
open positions and proactively generates exit orders when exit criteria are
triggered. Separation of concerns: rules say yes/no to inbound signals, the
monitor independently protects capital by closing stale or losing positions.
This component subscribes to MARKET_DATA (not SIGNAL) so it fires on every
price tick rather than waiting for the signal pipeline to produce a SELL.
"""

import time
from decimal import Decimal
from uuid import uuid4

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, MarketDataEvent, OrderEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side, Symbol
from cerebrum.risk.portfolio import PortfolioTracker

logger = structlog.get_logger()


class ExitMonitor:
    """
    Monitors open positions and emits exit orders on breach of exit criteria.

    Exit criteria (checked on every MARKET_DATA tick for the position's symbol):
    - Stop-loss: unrealized loss exceeds stop_loss_percent of position value
    - Take-profit: unrealized gain exceeds take_profit_percent of position value
    - Time-based: position age exceeds max_position_age_minutes

    When a criterion is breached, a SELL OrderEvent (MARKET type) is published
    directly to the bus. The risk manager and paper/live adapter will process it
    through the normal order flow.

    Only long positions (amount > 0) are handled — the system currently only
    takes long positions via BUY signals.
    """

    def __init__(
        self,
        bus: EventBus,
        portfolio: PortfolioTracker,
        stop_loss_percent: Decimal = Decimal("2.0"),
        take_profit_percent: Decimal = Decimal("3.0"),
        max_position_age_minutes: int = 120,
    ) -> None:
        """
        Initialize exit monitor.

        Args:
            bus: Event bus for subscribing to market data and publishing orders
            portfolio: Portfolio tracker providing current position state
            stop_loss_percent: Close position if loss exceeds this % of entry value
            take_profit_percent: Close position if gain exceeds this % of entry value
            max_position_age_minutes: Close position if open longer than this many minutes
        """
        self._bus = bus
        self._portfolio = portfolio
        self._stop_loss_pct = stop_loss_percent
        self._take_profit_pct = take_profit_percent
        self._max_age_seconds = max_position_age_minutes * 60

        # Track symbols we've already triggered an exit for to avoid duplicate orders
        self._pending_exits: set[Symbol] = set()

        self._log = logger.bind(component="exit_monitor")

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="exit_monitor",
        )

        # Also listen for fills to clear pending_exits when position is closed
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="exit_monitor_fills",
        )

        self._log.info(
            "exit_monitor_initialized",
            stop_loss_pct=str(stop_loss_percent),
            take_profit_pct=str(take_profit_percent),
            max_age_minutes=max_position_age_minutes,
        )

    async def _on_fill(self, event: Event) -> None:
        """Clear pending exit flag when a fill confirms the position is closing."""
        from cerebrum.core.events import FillEvent
        if not isinstance(event, FillEvent):
            return
        # If we get a SELL fill, the position is closing/closed — remove from pending
        if event.side == Side.SELL and event.symbol in self._pending_exits:
            self._pending_exits.discard(event.symbol)
            self._log.debug("exit_pending_cleared", symbol=event.symbol)

    async def _on_market_data(self, event: Event) -> None:
        """Check exit criteria for the position in the updated symbol."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        current_price = event.price
        current_time = event.timestamp

        # Skip if we already have a pending exit order for this symbol
        if symbol in self._pending_exits:
            return

        pos = self._portfolio.get_position(symbol)
        if pos is None:
            return

        # Only handle long positions (amount > 0)
        if pos.amount <= Decimal("0"):
            return

        exit_reason: str | None = None
        order_type = OrderType.MARKET

        # --- Stop-loss check ---
        # Loss% = (entry - current) / entry * 100  (positive = loss for long)
        if pos.average_entry_price > Decimal("0"):
            loss_pct = (
                (pos.average_entry_price - current_price)
                / pos.average_entry_price
                * Decimal("100")
            )
            if loss_pct >= self._stop_loss_pct:
                exit_reason = (
                    f"stop_loss: loss {loss_pct:.2f}% >= threshold {self._stop_loss_pct}%"
                )
                order_type = OrderType.STOP_LOSS

        # --- Take-profit check ---
        if exit_reason is None and pos.average_entry_price > Decimal("0"):
            gain_pct = (
                (current_price - pos.average_entry_price)
                / pos.average_entry_price
                * Decimal("100")
            )
            if gain_pct >= self._take_profit_pct:
                exit_reason = (
                    f"take_profit: gain {gain_pct:.2f}% >= threshold {self._take_profit_pct}%"
                )
                order_type = OrderType.TAKE_PROFIT

        # --- Time-based check ---
        if exit_reason is None and self._max_age_seconds > 0:
            age_seconds = current_time - pos.entry_time
            if age_seconds >= self._max_age_seconds:
                age_minutes = age_seconds / 60
                exit_reason = (
                    f"time_exit: age {age_minutes:.1f}min >= max {self._max_age_seconds / 60:.0f}min"
                )

        if exit_reason is None:
            return

        # Emit a SELL market order to close the position
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=current_time,
            order_id=str(uuid4()),
            symbol=symbol,
            side=Side.SELL,
            order_type=order_type,
            amount=abs(pos.amount),
            status=OrderStatus.PENDING,
            metadata={"exit_reason": exit_reason, "source": "exit_monitor"},
        )

        self._pending_exits.add(symbol)
        self._log.info(
            "exit_order_emitted",
            symbol=symbol,
            reason=exit_reason,
            amount=str(pos.amount),
            price=str(current_price),
            entry_price=str(pos.average_entry_price),
        )

        await self._bus.publish(order)
