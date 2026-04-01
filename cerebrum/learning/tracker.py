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
from cerebrum.core.events import FillEvent, OrderEvent, RegimeChangeEvent, TradeClosedEvent, TradeOpenedEvent
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
        # Bug fix (Session 7): subscribe to regime changes so _current_regime stays
        # current. Without this subscription the tracker initialised with "UNKNOWN"
        # and never updated it — every trade record was stamped with "UNKNOWN" even
        # though the regime detector was working correctly.
        self._bus.subscribe(EventType.REGIME_CHANGE, self._on_regime_change, "trade_tracker_regime")
        self._log.info("trade_tracker_started")

    async def _on_regime_change(self, event: RegimeChangeEvent) -> None:
        """Update current regime when market regime changes.

        Bug fix (Session 7): without this handler the tracker never updated
        _current_regime from its initial "UNKNOWN" value, so every trade record
        was stored with regime="UNKNOWN" regardless of actual market conditions.
        """
        self.update_regime(event.to_regime)
        self._log.info("regime_updated", regime=event.to_regime)

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

        Debug logging (Session 7): log open trade count and matched trade ID on
        every fill to make FIFO matching decisions observable in session logs.
        """
        # Get signal snapshot from pending orders
        signal_snapshot = self._pending_signals.pop(event.order_id, {})

        # Check for open trades to close (FIFO matching, strategy-scoped).
        # Bug fix (Session 20): pass strategy_id so get_open_trades filters to
        # only this strategy's open positions. Prevents cross-strategy FIFO
        # collisions where strategy-A's SELL fill closes strategy-B's open BUY
        # trade (DEC-TRACK-001). getattr guards callers that pre-date the field.
        open_trades = await self._state.get_open_trades(
            event.symbol,
            strategy_id=getattr(event, 'strategy_id', None),
        )

        # Session 7 debug: log FIFO matching state for every fill
        self._log.debug(
            "fill_received",
            symbol=event.symbol,
            side=event.side.value,
            order_id=event.order_id,
            open_trade_count=len(open_trades),
            oldest_open_trade_id=open_trades[0].id if open_trades else None,
        )

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
                self._log.debug(
                    "fifo_match_closing",
                    symbol=event.symbol,
                    open_trade_id=open_trades[0].id,
                    entry_price=str(open_trades[0].entry_price),
                    fill_price=str(event.fill_price),
                )
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
            # Bug fix (Session 7): capture strategy_id from fill so the trade
            # record can be attributed to the originating strategy for multi-strategy
            # P&L analysis. getattr guards against callers that pre-date the field.
            strategy_id=getattr(fill, 'strategy_id', None),
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

    async def close_orphan_trades(self, active_strategies: list[str]) -> int:
        """Close OPEN trades from inactive or unknown strategies.

        Called once at startup (after strategy registry init) to neutralise
        stale OPEN rows left by previous sessions that ran different strategies.
        Two categories of orphans are closed:

        - strategy_id IS NULL — trades opened before strategy_id was added to
          FillEvent (pre-Session-7 sessions) or opened by the single-strategy
          legacy path which stamps no strategy_id.
        - strategy_id NOT IN active_strategies — trades from a strategy that
          is registered in this session's config (e.g. momentum) but was later
          disabled (DEC-TUNE-008). These will never receive a matching SELL fill,
          so they must be closed at startup to avoid corrupting FIFO matching for
          the active strategies.

        Each orphan is closed with pnl=0 and a log line so the closure is
        visible in session logs without distorting P&L statistics. Returns the
        number of rows closed.

        @decision DEC-TRACK-002
        @title Orphan trade cleanup at startup
        @status accepted
        @rationale Disabled strategies accumulate OPEN trades across sessions.
        Without cleanup, a future session that re-enables a strategy would pick
        up stale OPEN trades and immediately close them on the first SELL fill,
        producing phantom P&L. Cleaning at startup (before the event loop begins
        accepting live fills) is safe and race-free. pnl=0 is used instead of
        marking as CANCELLED because the trades table has no CANCELLED status and
        pnl=0 correctly signals "no realized gain/loss from this row".
        """
        import time as _time_mod

        assert self._state._db is not None

        # Fetch orphan trade IDs before the bulk UPDATE so we can log them
        # individually. Two queries (SELECT then UPDATE) is acceptable here
        # because close_orphan_trades runs once at startup before any live
        # fills arrive — there is no concurrent writer to race with.
        if active_strategies:
            placeholders = ",".join("?" * len(active_strategies))
            select_sql = (
                f"SELECT id, symbol, strategy_id FROM trades "
                f"WHERE status = 'OPEN' AND "
                f"(strategy_id IS NULL OR strategy_id NOT IN ({placeholders}))"
            )
            select_params: list = list(active_strategies)
        else:
            # No active strategies — every OPEN trade is an orphan
            select_sql = (
                "SELECT id, symbol, strategy_id FROM trades WHERE status = 'OPEN'"
            )
            select_params = []

        async with self._state._db.execute(select_sql, select_params) as cursor:
            orphan_rows = await cursor.fetchall()

        count = len(orphan_rows)
        if count == 0:
            self._log.info("orphan_scan_no_orphans", active_strategies=active_strategies)
            return 0

        now = _time_mod.time()
        if active_strategies:
            update_sql = (
                f"UPDATE trades SET status = 'CLOSED', pnl = '0', exit_time = ? "
                f"WHERE status = 'OPEN' AND "
                f"(strategy_id IS NULL OR strategy_id NOT IN ({placeholders}))"
            )
            update_params: list = [now] + list(active_strategies)
        else:
            update_sql = (
                "UPDATE trades SET status = 'CLOSED', pnl = '0', exit_time = ? "
                "WHERE status = 'OPEN'"
            )
            update_params = [now]

        await self._state._db.execute(update_sql, update_params)
        await self._state._db.commit()

        for row in orphan_rows:
            self._log.warning(
                "orphan_trade_closed",
                trade_id=row["id"] if hasattr(row, "__getitem__") else row[0],
                symbol=row["symbol"] if hasattr(row, "__getitem__") else row[1],
                strategy_id=row["strategy_id"] if hasattr(row, "__getitem__") else row[2],
                reason="strategy_not_active_in_this_session",
            )

        self._log.info(
            "orphan_scan_complete",
            closed_count=count,
            active_strategies=active_strategies,
        )
        return count

    def update_regime(self, regime: str) -> None:
        """Update current regime for new trades."""
        self._current_regime = regime
