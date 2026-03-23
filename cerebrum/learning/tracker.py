"""
Trade outcome tracker for learning system.

Listens to FillEvents and tracks trade lifecycle (entry → exit).
Publishes TradeClosedEvent when positions are closed.

@decision DEC-LEARN-001
@title Trade outcome tracking with signal snapshots
@status accepted
@rationale Capture signal state at trade entry to attribute P&L to specific signals.
FIFO matching for trade lifecycle (OPEN→CLOSED). Signal snapshot enables
post-trade analysis: which signals triggered the trade, and how did they perform.
"""

import time
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent, TradeClosedEvent, TradeOpenedEvent
from cerebrum.core.state import StateManager, TradeRecord
from cerebrum.core.types import Side

logger = structlog.get_logger()


class TradeTracker:
    """
    Tracks trade lifecycle and publishes TradeClosedEvent.

    Subscribes to: OrderEvent (to capture signal snapshot), FillEvent (to track entries/exits)
    Publishes: TradeOpenedEvent, TradeClosedEvent
    """

    def __init__(self, bus: EventBus, state: StateManager, current_regime: str) -> None:
        """
        Initialize trade tracker.

        Args:
            bus: Event bus for pub/sub
            state: State manager for persistence
            current_regime: Current market regime (updated externally)
        """
        self._bus = bus
        self._state = state
        self._current_regime = current_regime
        self._pending_signals: dict[str, dict] = {}  # order_id -> signal_snapshot
        self._log = logger.bind(component="trade_tracker")

    async def start(self) -> None:
        """Start listening to events."""
        from cerebrum.core.types import EventType
        self._bus.subscribe(EventType.ORDER, self._on_order, "trade_tracker_orders")
        self._bus.subscribe(EventType.FILL, self._on_fill, "trade_tracker_fills")
        self._log.info("trade_tracker_started")

    async def _on_order(self, event: OrderEvent) -> None:
        """Capture signal snapshot when order is created."""
        if event.metadata and "signals" in event.metadata:
            self._pending_signals[event.order_id] = event.metadata["signals"]

    async def _on_fill(self, event: FillEvent) -> None:
        """Track trade entry/exit on fill.

        The system is long-only: BUY fills open trades, SELL fills close them.
        FIFO matching by symbol: the oldest open BUY trade is closed by the next
        SELL fill for that symbol.

        Bug fix (Session 6): the original code called _open_trade for a SELL fill
        with no matching open BUY trade, creating a phantom "short" position in the
        DB. On the next BUY fill, get_open_trades() returned that phantom SELL trade
        and "closed" it instead of opening a real long — leaving 18/19 actual long
        trades permanently stuck as OPEN. Fix: skip silently (with a warning) when
        a SELL fill has no matching open BUY trade to close.
        """
        # Get signal snapshot from pending orders
        signal_snapshot = self._pending_signals.pop(event.order_id, {})

        # Check for open trades to close (FIFO matching)
        open_trades = await self._state.get_open_trades(event.symbol)

        if event.side == Side.BUY:
            if open_trades and open_trades[0].side == Side.SELL:
                # Closing a short position (future multi-strategy support)
                await self._close_trade(open_trades[0], event)
            else:
                # Opening a long position (normal case for long-only system)
                await self._open_trade(event, signal_snapshot)
        else:  # SELL
            if open_trades and open_trades[0].side == Side.BUY:
                # Closing a long position (normal exit path)
                await self._close_trade(open_trades[0], event)
            else:
                # No matching open BUY trade — this is an unmatched sell fill.
                # Do NOT open a phantom short: the system is long-only, and
                # creating a phantom SELL open trade corrupts future BUY matching.
                self._log.warning(
                    "unmatched_sell_fill_skipped",
                    symbol=event.symbol,
                    order_id=event.order_id,
                    open_trade_count=len(open_trades),
                    reason="no_open_buy_trade_to_close",
                )

    async def _open_trade(self, fill: FillEvent, signal_snapshot: dict) -> None:
        """Record trade entry."""
        trade = TradeRecord(
            id=None,
            symbol=fill.symbol,
            side=fill.side,
            entry_time=fill.timestamp,
            entry_price=fill.fill_price,
            exit_time=None,
            exit_price=None,
            quantity=fill.filled_amount,
            pnl=None,
            signal_snapshot=signal_snapshot,
            regime=self._current_regime,
            status="OPEN",
        )

        trade_id = await self._state.save_trade(trade)

        # Publish TradeOpenedEvent
        await self._bus.publish(TradeOpenedEvent(
            event_type=None,  # Will be set by __post_init__
            timestamp=time.time(),
            trade_id=trade_id,
            symbol=fill.symbol,
            side=fill.side,
            entry_price=fill.fill_price,
            quantity=fill.filled_amount,
            signal_snapshot=signal_snapshot,
            regime=self._current_regime,
        ))

        self._log.info("trade_opened", trade_id=trade_id, symbol=fill.symbol, side=fill.side.value)

    async def _close_trade(self, open_trade: TradeRecord, fill: FillEvent) -> None:
        """Record trade exit and calculate P&L."""
        assert open_trade.id is not None

        # Calculate P&L
        if open_trade.side == Side.BUY:
            pnl = (fill.fill_price - open_trade.entry_price) * open_trade.quantity
        else:  # SHORT
            pnl = (open_trade.entry_price - fill.fill_price) * open_trade.quantity

        # Subtract commissions
        pnl -= fill.commission

        # Update trade record
        await self._state.update_trade(
            open_trade.id,
            exit_time=fill.timestamp,
            exit_price=fill.fill_price,
            pnl=pnl,
            status="CLOSED",
        )

        # Publish TradeClosedEvent
        await self._bus.publish(TradeClosedEvent(
            event_type=None,  # Will be set by __post_init__
            timestamp=time.time(),
            trade_id=open_trade.id,
            symbol=open_trade.symbol,
            side=open_trade.side,
            entry_price=open_trade.entry_price,
            exit_price=fill.fill_price,
            quantity=open_trade.quantity,
            pnl=pnl,
            signal_snapshot=open_trade.signal_snapshot,
            regime=open_trade.regime,
            entry_time=open_trade.entry_time,
            exit_time=fill.timestamp,
        ))

        self._log.info(
            "trade_closed",
            trade_id=open_trade.id,
            symbol=open_trade.symbol,
            pnl=float(pnl),
        )

    def update_regime(self, regime: str) -> None:
        """Update current regime for new trades."""
        self._current_regime = regime
