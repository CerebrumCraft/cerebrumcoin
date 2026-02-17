"""Tests for learning components."""

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
