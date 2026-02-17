"""Tests for state manager."""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.state import SignalScore, StateManager, TradeRecord
from cerebrum.core.types import Side, SignalType


@pytest.mark.asyncio
async def test_state_manager_initialization():
    """Test state manager initializes database schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Verify database exists
        assert db_path.exists()

        await state.close()


@pytest.mark.asyncio
async def test_save_and_retrieve_trade():
    """Test saving and retrieving trade records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Create trade
        trade = TradeRecord(
            id=None,
            symbol="BTC/USDT",
            side=Side.BUY,
            entry_time=1234567890.0,
            entry_price=Decimal("50000"),
            exit_time=None,
            exit_price=None,
            quantity=Decimal("0.1"),
            pnl=None,
            signal_snapshot={"TECHNICAL": {"strength": 0.8}},
            regime="BULL",
            status="OPEN",
        )

        # Save trade
        trade_id = await state.save_trade(trade)
        assert trade_id > 0

        # Retrieve trade
        retrieved = await state.get_trade(trade_id)
        assert retrieved is not None
        assert retrieved.id == trade_id
        assert retrieved.symbol == "BTC/USDT"
        assert retrieved.side == Side.BUY
        assert retrieved.entry_price == Decimal("50000")
        assert retrieved.status == "OPEN"

        await state.close()


@pytest.mark.asyncio
async def test_update_trade():
    """Test updating trade records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Create and save trade
        trade = TradeRecord(
            id=None,
            symbol="BTC/USDT",
            side=Side.BUY,
            entry_time=1234567890.0,
            entry_price=Decimal("50000"),
            exit_time=None,
            exit_price=None,
            quantity=Decimal("0.1"),
            pnl=None,
            signal_snapshot={},
            regime="BULL",
            status="OPEN",
        )
        trade_id = await state.save_trade(trade)

        # Update trade
        await state.update_trade(
            trade_id,
            exit_time=1234567900.0,
            exit_price=Decimal("51000"),
            pnl=Decimal("100"),
            status="CLOSED",
        )

        # Verify update
        updated = await state.get_trade(trade_id)
        assert updated is not None
        assert updated.exit_price == Decimal("51000")
        assert updated.pnl == Decimal("100")
        assert updated.status == "CLOSED"

        await state.close()


@pytest.mark.asyncio
async def test_get_open_and_closed_trades():
    """Test querying open and closed trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Create open trade
        open_trade = TradeRecord(
            id=None,
            symbol="BTC/USDT",
            side=Side.BUY,
            entry_time=1234567890.0,
            entry_price=Decimal("50000"),
            exit_time=None,
            exit_price=None,
            quantity=Decimal("0.1"),
            pnl=None,
            signal_snapshot={},
            regime="BULL",
            status="OPEN",
        )
        await state.save_trade(open_trade)

        # Create closed trade
        closed_trade = TradeRecord(
            id=None,
            symbol="ETH/USDT",
            side=Side.SELL,
            entry_time=1234567800.0,
            entry_price=Decimal("3000"),
            exit_time=1234567850.0,
            exit_price=Decimal("2900"),
            quantity=Decimal("1.0"),
            pnl=Decimal("100"),
            signal_snapshot={},
            regime="BEAR",
            status="CLOSED",
        )
        await state.save_trade(closed_trade)

        # Query open trades
        open_trades = await state.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].symbol == "BTC/USDT"
        assert open_trades[0].status == "OPEN"

        # Query closed trades
        closed_trades = await state.get_closed_trades()
        assert len(closed_trades) == 1
        assert closed_trades[0].symbol == "ETH/USDT"
        assert closed_trades[0].status == "CLOSED"

        await state.close()


@pytest.mark.asyncio
async def test_signal_scores():
    """Test saving and retrieving signal scores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Create score
        score = SignalScore(
            signal_type=SignalType.TECHNICAL,
            regime="BULL",
            win_rate=Decimal("0.65"),
            profit_factor=Decimal("1.8"),
            sharpe_ratio=Decimal("1.2"),
            sample_size=50,
            updated_at=datetime.now(),
        )

        # Save score
        await state.save_signal_score(score)

        # Retrieve score
        retrieved = await state.get_signal_score(SignalType.TECHNICAL, "BULL")
        assert retrieved is not None
        assert retrieved.signal_type == SignalType.TECHNICAL
        assert retrieved.regime == "BULL"
        assert retrieved.win_rate == Decimal("0.65")
        assert retrieved.profit_factor == Decimal("1.8")
        assert retrieved.sample_size == 50

        await state.close()


@pytest.mark.asyncio
async def test_weight_history():
    """Test saving and retrieving weight history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Save weights
        await state.save_weight(SignalType.TECHNICAL, "BULL", Decimal("1.2"), 1000.0)
        await state.save_weight(SignalType.TECHNICAL, "BULL", Decimal("1.3"), 2000.0)
        await state.save_weight(SignalType.TECHNICAL, "BULL", Decimal("1.1"), 3000.0)

        # Retrieve history
        history = await state.get_weight_history(SignalType.TECHNICAL, "BULL")
        assert len(history) == 3
        # Should be in descending timestamp order
        assert history[0][0] == 3000.0
        assert history[0][1] == Decimal("1.1")
        assert history[1][0] == 2000.0
        assert history[2][0] == 1000.0

        await state.close()


@pytest.mark.asyncio
async def test_generic_state_storage():
    """Test key-value state storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()

        # Set state
        await state.set_state("last_regime", "BULL")
        await state.set_state("trade_count", "42")

        # Get state
        regime = await state.get_state("last_regime")
        count = await state.get_state("trade_count")

        assert regime == "BULL"
        assert count == "42"

        # Non-existent key
        missing = await state.get_state("nonexistent")
        assert missing is None

        await state.close()
