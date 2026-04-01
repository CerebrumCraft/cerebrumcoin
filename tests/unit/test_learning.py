"""
Tests for learning components.

@decision DEC-TEST-LEARN-001
@title Test trade tracker lifecycle and sell-fill correlation fix
@status accepted
@rationale Covers the Session 6 bug (sell fills creating phantom shorts instead
of closing matching open BUY trades). Tests prove: (1) BUY fill opens a trade,
(2) subsequent SELL fill closes it via FIFO matching, (3) unmatched SELL fill is
silently skipped and does NOT open a phantom SELL record that would corrupt the
next BUY fill's matching. Real EventBus and in-memory SQLite — no mocks.
"""

import asyncio
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent, TradeClosedEvent
from cerebrum.core.state import StateManager
from cerebrum.core.types import EventType, OrderType, Side, SignalType
from cerebrum.learning.adapter import WeightAdapter
from cerebrum.learning.scorer import SignalScorer
from cerebrum.learning.tracker import TradeTracker


@pytest.fixture
async def event_bus():
    """Create event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def state_manager():
    """Create temporary state manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()
        yield state
        await state.close()


@pytest.mark.asyncio
async def test_trade_tracker_lifecycle(event_bus, state_manager):
    """Test tracking trade lifecycle from open to close."""
    # Capture published events — register BEFORE tracker starts
    published_events = []

    async def capture_event(event):
        published_events.append(event)

    event_bus.subscribe(EventType.TRADE_OPENED, capture_event, "test_trade_opened")
    event_bus.subscribe(EventType.TRADE_CLOSED, capture_event, "test_trade_closed")

    tracker = TradeTracker(event_bus, state_manager, "BULL")
    await tracker.start()
    await asyncio.sleep(0.05)  # Let consumer tasks start

    # Create order with signal snapshot
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=1000.0,
        order_id="order_1",
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
        metadata={"signals": {"TECHNICAL": {"strength": 0.8}}},
    )
    await event_bus.publish(order)

    # Simulate buy fill (opening position)
    fill_open = FillEvent(
        event_type=EventType.FILL,
        timestamp=1001.0,
        order_id="order_1",
        symbol="BTC/USDT",
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5"),
        commission_asset="USDT",
    )
    await event_bus.publish(fill_open)

    # Wait for event chain: fill → tracker processes → TradeOpenedEvent → capture
    # The chain involves async DB writes, so allow sufficient time
    for _ in range(20):
        await asyncio.sleep(0.1)
        if len(published_events) >= 1:
            break

    # Verify TradeOpenedEvent was published
    assert len(published_events) == 1
    assert isinstance(published_events[0], TradeClosedEvent) is False

    # Verify trade is in database
    open_trades = await state_manager.get_open_trades("BTC/USDT")
    assert len(open_trades) == 1
    assert open_trades[0].entry_price == Decimal("50000")

    # Simulate sell fill (closing position)
    fill_close = FillEvent(
        event_type=EventType.FILL,
        timestamp=1100.0,
        order_id="order_2",
        symbol="BTC/USDT",
        side=Side.SELL,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("51000"),
        commission=Decimal("5"),
        commission_asset="USDT",
    )
    await event_bus.publish(fill_close)

    # Wait for event chain: fill → tracker processes → TradeClosedEvent → capture
    for _ in range(20):
        await asyncio.sleep(0.1)
        if len(published_events) >= 2:
            break

    # Verify TradeClosedEvent was published
    assert len(published_events) == 2
    assert isinstance(published_events[1], TradeClosedEvent)
    assert published_events[1].pnl > 0  # Profitable trade

    # Verify trade is closed in database
    open_trades = await state_manager.get_open_trades("BTC/USDT")
    assert len(open_trades) == 0

    closed_trades = await state_manager.get_closed_trades()
    assert len(closed_trades) == 1
    assert closed_trades[0].exit_price == Decimal("51000")


@pytest.mark.asyncio
async def test_signal_scorer_calculates_metrics(event_bus, state_manager):
    """Test signal scorer calculates win rate and profit factor."""
    scorer = SignalScorer(event_bus, state_manager, min_sample_size=3)
    await scorer.start()

    # Create closed trades with signal snapshots
    trades_data = [
        # Winner
        {
            "symbol": "BTC/USDT",
            "side": Side.BUY,
            "entry_price": Decimal("50000"),
            "exit_price": Decimal("51000"),
            "pnl": Decimal("95"),  # 100 - 5 commission
            "signal": SignalType.TECHNICAL,
        },
        # Winner
        {
            "symbol": "BTC/USDT",
            "side": Side.BUY,
            "entry_price": Decimal("52000"),
            "exit_price": Decimal("53000"),
            "pnl": Decimal("95"),
            "signal": SignalType.TECHNICAL,
        },
        # Loser
        {
            "symbol": "BTC/USDT",
            "side": Side.BUY,
            "entry_price": Decimal("54000"),
            "exit_price": Decimal("53500"),
            "pnl": Decimal("-55"),  # -50 - 5 commission
            "signal": SignalType.TECHNICAL,
        },
    ]

    # Manually insert trades
    for trade_data in trades_data:
        from cerebrum.core.state import TradeRecord

        trade = TradeRecord(
            id=None,
            symbol=trade_data["symbol"],
            side=trade_data["side"],
            entry_time=1000.0,
            entry_price=trade_data["entry_price"],
            exit_time=1100.0,
            exit_price=trade_data["exit_price"],
            quantity=Decimal("0.1"),
            pnl=trade_data["pnl"],
            signal_snapshot={
                trade_data["signal"].value: {"strength": 0.8, "confidence": 0.9}
            },
            regime="BULL",
            status="CLOSED",
        )
        await state_manager.save_trade(trade)

    # Trigger scoring with TradeClosedEvent
    event = TradeClosedEvent(
        event_type=EventType.TRADE_CLOSED,
        timestamp=1100.0,
        trade_id=1,
        symbol="BTC/USDT",
        side=Side.BUY,
        entry_price=Decimal("50000"),
        exit_price=Decimal("51000"),
        quantity=Decimal("0.1"),
        pnl=Decimal("95"),
        signal_snapshot={"TECHNICAL": {"strength": 0.8}},
        regime="BULL",
        entry_time=1000.0,
        exit_time=1100.0,
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)  # Allow event processing

    # Verify score was calculated and saved
    score = await state_manager.get_signal_score(SignalType.TECHNICAL, "BULL")
    assert score is not None
    assert score.win_rate == Decimal("2") / Decimal("3")  # 2 winners out of 3
    assert score.sample_size == 3


@pytest.mark.asyncio
async def test_weight_adapter_adjusts_weights(event_bus, state_manager):
    """Test weight adapter adjusts signal weights based on scores."""
    # Track weight updates
    updated_weights = {}

    def weight_callback(signal_type, regime, weight):
        updated_weights[(signal_type, regime)] = weight

    adapter = WeightAdapter(event_bus, state_manager, weight_callback)
    await adapter.start()

    # Publish score update
    from cerebrum.core.events import ScoreUpdateEvent

    scores = {
        SignalType.TECHNICAL: {
            "win_rate": Decimal("0.7"),
            "profit_factor": Decimal("2.0"),
            "sharpe_ratio": Decimal("1.5"),
        }
    }

    event = ScoreUpdateEvent(
        event_type=EventType.SCORE_UPDATE,
        timestamp=1000.0,
        regime="BULL",
        scores=scores,
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)  # Allow event processing

    # Verify weight was updated
    assert (SignalType.TECHNICAL, "BULL") in updated_weights
    new_weight = updated_weights[(SignalType.TECHNICAL, "BULL")]

    # Weight should be adjusted based on profit factor and win rate
    # Expected: (2.0 * 0.7) + (0.7 * 2 * 0.3) = 1.4 + 0.42 = 1.82
    # EMA smoothing: 0.1 * 1.82 + 0.9 * 1.0 = 0.182 + 0.9 = 1.082
    assert new_weight > Decimal("1.0")  # Should increase from base
    assert new_weight < Decimal("2.0")  # Should be within bounds


@pytest.mark.asyncio
async def test_weight_adapter_respects_bounds(event_bus, state_manager):
    """Test weight adapter enforces floor and ceiling."""

    updated_weights = {}

    def weight_callback(signal_type, regime, weight):
        updated_weights[(signal_type, regime)] = weight

    adapter = WeightAdapter(
        event_bus,
        state_manager,
        weight_callback,
        floor=Decimal("0.5"),
        ceiling=Decimal("1.5"),
    )
    await adapter.start()

    # Publish score that would produce very high weight
    from cerebrum.core.events import ScoreUpdateEvent

    scores = {
        SignalType.TECHNICAL: {
            "win_rate": Decimal("0.95"),
            "profit_factor": Decimal("10.0"),  # Extremely high
            "sharpe_ratio": Decimal("3.0"),
        }
    }

    event = ScoreUpdateEvent(
        event_type=EventType.SCORE_UPDATE,
        timestamp=1000.0,
        regime="BULL",
        scores=scores,
    )
    await event_bus.publish(event)
    await asyncio.sleep(0.1)  # Allow event processing

    # Weight should be clamped to ceiling
    new_weight = updated_weights[(SignalType.TECHNICAL, "BULL")]
    assert new_weight <= Decimal("1.5")


# ---------------------------------------------------------------------------
# Tests: sell fill correlation fix (Session 6 bug)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracker_sell_fill_closes_matching_buy_trade(event_bus, state_manager):
    """A SELL fill should close the matching open BUY trade (FIFO).

    This is the core behavior restored by the Session 6 fix: before the fix,
    an unmatched SELL fill opened a phantom short, which then consumed the
    next BUY fill and left the real long trade permanently OPEN in the DB.
    """
    closed_events = []

    async def capture_closed(event):
        closed_events.append(event)

    event_bus.subscribe(EventType.TRADE_CLOSED, capture_closed, "test_closed_capture")

    tracker = TradeTracker(event_bus, state_manager, "SIDEWAYS")
    await tracker.start()
    await asyncio.sleep(0.05)

    # BUY fill — opens a long trade
    buy_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=2000.0,
        order_id="buy_order_1",
        symbol="BTC/USDT",
        side=Side.BUY,
        filled_amount=Decimal("0.2"),
        fill_price=Decimal("60000"),
        commission=Decimal("12"),
        commission_asset="USDT",
    )
    await event_bus.publish(buy_fill)

    # Wait for open trade to land in DB
    for _ in range(20):
        await asyncio.sleep(0.1)
        open_trades = await state_manager.get_open_trades("BTC/USDT")
        if open_trades:
            break

    assert len(open_trades) == 1, "BUY fill must open a trade"
    assert open_trades[0].entry_price == Decimal("60000")

    # SELL fill — must close the BUY trade above
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=2100.0,
        order_id="sell_order_1",
        symbol="BTC/USDT",
        side=Side.SELL,
        filled_amount=Decimal("0.2"),
        fill_price=Decimal("61000"),
        commission=Decimal("12"),
        commission_asset="USDT",
    )
    await event_bus.publish(sell_fill)

    # Wait for TradeClosedEvent
    for _ in range(20):
        await asyncio.sleep(0.1)
        if closed_events:
            break

    assert len(closed_events) == 1, "SELL fill must produce one TradeClosedEvent"
    assert isinstance(closed_events[0], TradeClosedEvent)

    # Trade should be CLOSED in the DB — no open trades remain
    open_trades_after = await state_manager.get_open_trades("BTC/USDT")
    assert len(open_trades_after) == 0, "After SELL fill, no open trades should remain"

    closed_trades = await state_manager.get_closed_trades()
    assert len(closed_trades) == 1
    assert closed_trades[0].exit_price == Decimal("61000")


@pytest.mark.asyncio
async def test_tracker_unmatched_sell_fill_is_skipped(event_bus, state_manager):
    """An unmatched SELL fill (no open BUY trade) must be skipped, not create a phantom short.

    Before the Session 6 fix, a SELL fill with no matching open trade would call
    _open_trade() and create a phantom SELL record. The next BUY fill would then
    'close' that phantom instead of opening a real long, leaving actual buys stuck
    as OPEN forever. The fix: emit a warning and return without touching the DB.
    """
    tracker = TradeTracker(event_bus, state_manager, "BULL")
    await tracker.start()
    await asyncio.sleep(0.05)

    # SELL fill with no preceding BUY — should be silently skipped
    unmatched_sell = FillEvent(
        event_type=EventType.FILL,
        timestamp=3000.0,
        order_id="orphan_sell_1",
        symbol="ETH/USDT",
        side=Side.SELL,
        filled_amount=Decimal("1.0"),
        fill_price=Decimal("3500"),
        commission=Decimal("3.5"),
        commission_asset="USDT",
    )
    await event_bus.publish(unmatched_sell)
    await asyncio.sleep(0.3)

    # No trades should exist in the DB (no phantom short opened)
    open_trades = await state_manager.get_open_trades("ETH/USDT")
    assert len(open_trades) == 0, "Unmatched SELL fill must not create a phantom short trade"

    closed_trades = await state_manager.get_closed_trades()
    assert len(closed_trades) == 0, "Unmatched SELL fill must not create any closed trade"

    # Now submit a real BUY — it must open a fresh long, not 'close' the phantom
    real_buy = FillEvent(
        event_type=EventType.FILL,
        timestamp=3100.0,
        order_id="real_buy_1",
        symbol="ETH/USDT",
        side=Side.BUY,
        filled_amount=Decimal("1.0"),
        fill_price=Decimal("3600"),
        commission=Decimal("3.6"),
        commission_asset="USDT",
    )
    await event_bus.publish(real_buy)

    for _ in range(20):
        await asyncio.sleep(0.1)
        open_trades = await state_manager.get_open_trades("ETH/USDT")
        if open_trades:
            break

    assert len(open_trades) == 1, "BUY after unmatched SELL must open a new long trade"
    assert open_trades[0].side == Side.BUY
    assert open_trades[0].entry_price == Decimal("3600")


# ---------------------------------------------------------------------------
# Tests: strategy-aware FIFO matching (Session 20 fix — DEC-TRACK-001)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_aware_fifo_closes_only_matching_strategy(event_bus, state_manager):
    """SELL fill with strategy_id=A must close only strategy-A's open trade.

    Regression test for DEC-TRACK-001: before the fix, get_open_trades() was
    called without strategy_id, so the oldest OPEN trade across ALL strategies
    for a symbol was returned. If strategy-B had an earlier BUY than strategy-A,
    a SELL fill from strategy-A would incorrectly close strategy-B's trade.

    Setup: two open BUY trades for BTC/USDT — one from "mean_reversion" (older)
    and one from "range_trading" (newer). A SELL fill arrives tagged with
    strategy_id="range_trading". After the fix, only the range_trading trade
    is closed; mean_reversion's trade remains open.
    """
    from cerebrum.core.state import TradeRecord

    # Insert two OPEN BUY trades directly into the DB (bypassing the event bus)
    # so we control entry_time precisely.
    older_trade = TradeRecord(
        id=None,
        symbol="BTC/USDT",
        side=Side.BUY,
        entry_time=5000.0,   # older
        entry_price=Decimal("62000"),
        exit_time=None,
        exit_price=None,
        quantity=Decimal("0.1"),
        pnl=None,
        signal_snapshot={},
        regime="BULL",
        status="OPEN",
        strategy_id="mean_reversion",
    )
    newer_trade = TradeRecord(
        id=None,
        symbol="BTC/USDT",
        side=Side.BUY,
        entry_time=5100.0,   # newer
        entry_price=Decimal("62500"),
        exit_time=None,
        exit_price=None,
        quantity=Decimal("0.1"),
        pnl=None,
        signal_snapshot={},
        regime="BULL",
        status="OPEN",
        strategy_id="range_trading",
    )
    older_id = await state_manager.save_trade(older_trade)
    newer_id = await state_manager.save_trade(newer_trade)

    # Verify both trades are open before the SELL fill
    all_open = await state_manager.get_open_trades("BTC/USDT")
    assert len(all_open) == 2, "precondition: two open trades"

    closed_events = []

    async def capture_closed(event):
        closed_events.append(event)

    event_bus.subscribe(EventType.TRADE_CLOSED, capture_closed, "test_strategy_fifo")

    tracker = TradeTracker(event_bus, state_manager, "BULL")
    await tracker.start()
    await asyncio.sleep(0.05)

    # SELL fill tagged with range_trading — must close newer_trade only
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=5200.0,
        order_id="sell_range_1",
        symbol="BTC/USDT",
        side=Side.SELL,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("63000"),
        commission=Decimal("6"),
        commission_asset="USDT",
        strategy_id="range_trading",
    )
    await event_bus.publish(sell_fill)

    # Wait for TradeClosedEvent
    for _ in range(20):
        await asyncio.sleep(0.1)
        if closed_events:
            break

    assert len(closed_events) == 1, "exactly one trade must be closed"
    closed_event = closed_events[0]
    assert isinstance(closed_event, TradeClosedEvent)

    # The closed trade must be the range_trading one (newer_id), not mean_reversion
    closed_db = await state_manager.get_trade(newer_id)
    assert closed_db is not None
    assert closed_db.status == "CLOSED", "range_trading trade must be CLOSED"
    assert closed_db.exit_price == Decimal("63000")

    # mean_reversion trade must remain open — strategy isolation preserved
    open_mr = await state_manager.get_trade(older_id)
    assert open_mr is not None
    assert open_mr.status == "OPEN", "mean_reversion trade must remain OPEN"


# ---------------------------------------------------------------------------
# Tests: orphan trade scanner (DEC-TRACK-002)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_orphan_trades_cleans_inactive_and_null_strategies(event_bus, state_manager):
    """close_orphan_trades() closes NULL-strategy and inactive-strategy OPEN trades.

    Three OPEN trades:
      1. strategy_id=None (pre-strategy-id session)
      2. strategy_id="momentum" (disabled strategy)
      3. strategy_id="mean_reversion" (active strategy — must NOT be closed)

    After calling close_orphan_trades(["mean_reversion"]):
      - Trades 1 and 2 must be CLOSED with pnl=0
      - Trade 3 must remain OPEN
      - Return value must be 2
    """
    from cerebrum.core.state import TradeRecord

    def make_trade(strategy_id):
        return TradeRecord(
            id=None,
            symbol="ETH/USDT",
            side=Side.BUY,
            entry_time=6000.0,
            entry_price=Decimal("3000"),
            exit_time=None,
            exit_price=None,
            quantity=Decimal("0.5"),
            pnl=None,
            signal_snapshot={},
            regime="SIDEWAYS",
            status="OPEN",
            strategy_id=strategy_id,
        )

    null_id = await state_manager.save_trade(make_trade(None))
    momentum_id = await state_manager.save_trade(make_trade("momentum"))
    active_id = await state_manager.save_trade(make_trade("mean_reversion"))

    tracker = TradeTracker(event_bus, state_manager, "SIDEWAYS")
    await tracker.start()
    await asyncio.sleep(0.05)

    closed_count = await tracker.close_orphan_trades(["mean_reversion"])

    assert closed_count == 2, f"expected 2 orphans closed, got {closed_count}"

    # NULL-strategy trade must be CLOSED
    null_trade = await state_manager.get_trade(null_id)
    assert null_trade is not None
    assert null_trade.status == "CLOSED", "NULL-strategy trade must be closed"
    assert null_trade.pnl == Decimal("0"), "orphan pnl must be 0"

    # momentum trade must be CLOSED
    momentum_trade = await state_manager.get_trade(momentum_id)
    assert momentum_trade is not None
    assert momentum_trade.status == "CLOSED", "momentum (inactive) trade must be closed"
    assert momentum_trade.pnl == Decimal("0"), "orphan pnl must be 0"

    # mean_reversion trade must remain OPEN — it is active
    active_trade = await state_manager.get_trade(active_id)
    assert active_trade is not None
    assert active_trade.status == "OPEN", "active strategy trade must NOT be closed"


@pytest.mark.asyncio
async def test_close_orphan_trades_returns_zero_when_no_orphans(event_bus, state_manager):
    """close_orphan_trades() returns 0 when all OPEN trades belong to active strategies."""
    from cerebrum.core.state import TradeRecord

    trade = TradeRecord(
        id=None,
        symbol="BTC/USDT",
        side=Side.BUY,
        entry_time=7000.0,
        entry_price=Decimal("65000"),
        exit_time=None,
        exit_price=None,
        quantity=Decimal("0.05"),
        pnl=None,
        signal_snapshot={},
        regime="BULL",
        status="OPEN",
        strategy_id="range_trading",
    )
    await state_manager.save_trade(trade)

    tracker = TradeTracker(event_bus, state_manager, "BULL")
    await tracker.start()
    await asyncio.sleep(0.05)

    count = await tracker.close_orphan_trades(["range_trading"])
    assert count == 0, "no orphans when all trades belong to active strategies"

    # Trade must still be open
    open_trades = await state_manager.get_open_trades("BTC/USDT")
    assert len(open_trades) == 1
