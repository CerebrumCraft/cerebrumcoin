"""Tests for session reporter."""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.state import StateManager, TradeRecord
from cerebrum.core.types import Side
from cerebrum.monitoring.reporter import SessionReporter


@pytest.fixture
async def state_manager_with_trades():
    """Create state manager with sample trades."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()
        
        # Add sample trades
        for i in range(5):
            trade = TradeRecord(
                id=None,
                symbol="BTC/USD",
                side=Side.BUY,
                entry_time=1000.0 + i * 100,
                entry_price=Decimal("50000"),
                exit_time=1100.0 + i * 100,
                exit_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("100"),
                signal_snapshot={},
                regime="BULL",
                status="CLOSED",
            )
            await state.save_trade(trade)
        
        yield state
        await state.close()


@pytest.mark.asyncio
async def test_reporter_generates_report(state_manager_with_trades):
    """Test reporter generates comprehensive report."""
    reporter = SessionReporter(state_manager_with_trades)
    
    report = await reporter.generate_report(
        initial_balance=Decimal("10000.0"),
        output_file=None,
    )
    
    assert "CerebrumCoin Session Report" in report
    assert "Total Trades: 5" in report
    assert "Win Rate: 100.00%" in report
    assert "BULL" in report


@pytest.mark.asyncio
async def test_reporter_saves_to_file(state_manager_with_trades):
    """Test reporter saves report to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "report.txt"
        
        reporter = SessionReporter(state_manager_with_trades)
        await reporter.generate_report(
            initial_balance=Decimal("10000.0"),
            output_file=output_file,
        )
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "CerebrumCoin Session Report" in content
