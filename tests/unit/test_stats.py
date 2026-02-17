"""
Tests for performance statistics calculator.

@decision DEC-TEST-006
@title Deterministic stats tests with trade fixtures
@status accepted
@rationale Performance metrics must be mathematically correct. Use deterministic
trade fixtures (known PnL, known equity curves) to verify Sharpe, Sortino, drawdown,
profit factor, win rate calculations match expected values.
"""

import pytest
from decimal import Decimal

from cerebrum.monitoring.stats import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_profit_factor,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_avg_profit,
    calculate_avg_loss,
    calculate_consecutive_wins,
    calculate_consecutive_losses,
    calculate_performance_metrics,
)
from cerebrum.core.state import TradeRecord
from cerebrum.core.types import Side


@pytest.fixture
def winning_trades():
    """Sample winning trades."""
    return [
        TradeRecord(
            id=1, symbol="BTC/USD", side=Side.BUY,
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=1100.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=2, symbol="BTC/USD", side=Side.BUY,
            entry_time=1200.0, entry_price=Decimal("51000"),
            exit_time=1300.0, exit_price=Decimal("52000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=3, symbol="BTC/USD", side=Side.BUY,
            entry_time=1400.0, entry_price=Decimal("52000"),
            exit_time=1500.0, exit_price=Decimal("53000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
    ]


@pytest.fixture
def mixed_trades():
    """Mixed winning and losing trades."""
    return [
        TradeRecord(
            id=1, symbol="BTC/USD", side=Side.BUY,
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=1100.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=2, symbol="BTC/USD", side=Side.BUY,
            entry_time=1200.0, entry_price=Decimal("51000"),
            exit_time=1300.0, exit_price=Decimal("50500"),
            quantity=Decimal("0.1"), pnl=Decimal("-50"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=3, symbol="BTC/USD", side=Side.BUY,
            entry_time=1400.0, entry_price=Decimal("50500"),
            exit_time=1500.0, exit_price=Decimal("52000"),
            quantity=Decimal("0.1"), pnl=Decimal("150"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=4, symbol="BTC/USD", side=Side.BUY,
            entry_time=1600.0, entry_price=Decimal("52000"),
            exit_time=1700.0, exit_price=Decimal("51500"),
            quantity=Decimal("0.1"), pnl=Decimal("-50"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
    ]


def test_win_rate_all_winners(winning_trades):
    """Test win rate with all winning trades."""
    win_rate = calculate_win_rate(winning_trades)
    assert win_rate == Decimal("100.0")


def test_win_rate_mixed(mixed_trades):
    """Test win rate with mixed results."""
    win_rate = calculate_win_rate(mixed_trades)
    assert win_rate == Decimal("50.0")


def test_win_rate_empty():
    """Test win rate with no trades."""
    assert calculate_win_rate([]) == Decimal("0.0")


def test_avg_profit(mixed_trades):
    """Test average profit calculation."""
    avg_profit = calculate_avg_profit(mixed_trades)
    assert avg_profit == Decimal("125.0")


def test_avg_loss(mixed_trades):
    """Test average loss calculation."""
    avg_loss = calculate_avg_loss(mixed_trades)
    assert avg_loss == Decimal("-50.0")


def test_profit_factor(mixed_trades):
    """Test profit factor calculation."""
    pf = calculate_profit_factor(mixed_trades)
    assert pf == Decimal("2.5")


def test_profit_factor_no_losses(winning_trades):
    """Test profit factor with no losses."""
    pf = calculate_profit_factor(winning_trades)
    assert pf > Decimal("999999")


def test_profit_factor_only_losses():
    """Test profit factor with only losses."""
    losing_trades = [
        TradeRecord(
            id=1, symbol="BTC/USD", side=Side.BUY,
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=1100.0, exit_price=Decimal("49000"),
            quantity=Decimal("0.1"), pnl=Decimal("-100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
    ]
    pf = calculate_profit_factor(losing_trades)
    assert pf == Decimal("0.0")


def test_sharpe_ratio(mixed_trades):
    """Test Sharpe ratio calculation."""
    sharpe = calculate_sharpe_ratio(mixed_trades, risk_free_rate=Decimal("0.0"))
    assert Decimal("0.3") < sharpe < Decimal("0.5")


def test_sharpe_ratio_zero_variance(winning_trades):
    """Test Sharpe ratio with zero variance."""
    sharpe = calculate_sharpe_ratio(winning_trades, risk_free_rate=Decimal("0.0"))
    assert sharpe > Decimal("10")


def test_sortino_ratio(mixed_trades):
    """Test Sortino ratio calculation."""
    sortino = calculate_sortino_ratio(mixed_trades, target_return=Decimal("0.0"))
    sharpe = calculate_sharpe_ratio(mixed_trades, risk_free_rate=Decimal("0.0"))
    assert sortino > sharpe


def test_max_drawdown():
    """Test maximum drawdown calculation."""
    trades = [
        TradeRecord(
            id=1, symbol="BTC/USD", side=Side.BUY,
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=1100.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=2, symbol="BTC/USD", side=Side.BUY,
            entry_time=1200.0, entry_price=Decimal("51000"),
            exit_time=1300.0, exit_price=Decimal("50500"),
            quantity=Decimal("0.1"), pnl=Decimal("-50"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=3, symbol="BTC/USD", side=Side.BUY,
            entry_time=1400.0, entry_price=Decimal("50500"),
            exit_time=1500.0, exit_price=Decimal("52000"),
            quantity=Decimal("0.1"), pnl=Decimal("150"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
    ]
    max_dd, max_dd_pct = calculate_max_drawdown(trades, initial_balance=Decimal("10000"))
    assert max_dd == Decimal("50")
    assert Decimal("0.4") < max_dd_pct < Decimal("0.5")


def test_consecutive_wins(mixed_trades):
    """Test consecutive wins calculation."""
    max_consecutive = calculate_consecutive_wins(mixed_trades)
    assert max_consecutive == 1


def test_consecutive_wins_streak():
    """Test consecutive wins with streak."""
    trades = [
        TradeRecord(
            id=1, symbol="BTC/USD", side=Side.BUY,
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=1100.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=2, symbol="BTC/USD", side=Side.BUY,
            entry_time=1200.0, entry_price=Decimal("51000"),
            exit_time=1300.0, exit_price=Decimal("52000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=3, symbol="BTC/USD", side=Side.BUY,
            entry_time=1400.0, entry_price=Decimal("52000"),
            exit_time=1500.0, exit_price=Decimal("53000"),
            quantity=Decimal("0.1"), pnl=Decimal("100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
        TradeRecord(
            id=4, symbol="BTC/USD", side=Side.BUY,
            entry_time=1600.0, entry_price=Decimal("53000"),
            exit_time=1700.0, exit_price=Decimal("52000"),
            quantity=Decimal("0.1"), pnl=Decimal("-100"),
            signal_snapshot={}, regime="BULL", status="CLOSED"
        ),
    ]
    max_consecutive = calculate_consecutive_wins(trades)
    assert max_consecutive == 3


def test_consecutive_losses(mixed_trades):
    """Test consecutive losses calculation."""
    max_consecutive = calculate_consecutive_losses(mixed_trades)
    assert max_consecutive == 1


def test_performance_metrics_comprehensive(mixed_trades):
    """Test comprehensive metrics calculation."""
    metrics = calculate_performance_metrics(
        mixed_trades,
        initial_balance=Decimal("10000"),
        risk_free_rate=Decimal("0.02")
    )

    assert "total_trades" in metrics
    assert "win_rate" in metrics
    assert "profit_factor" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "max_drawdown_pct" in metrics
    assert "total_pnl" in metrics
    assert "avg_profit" in metrics
    assert "avg_loss" in metrics
    assert "max_consecutive_wins" in metrics
    assert "max_consecutive_losses" in metrics

    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == Decimal("50.0")
    assert metrics["total_pnl"] == Decimal("150")


def test_performance_metrics_empty():
    """Test metrics with no trades."""
    metrics = calculate_performance_metrics([], initial_balance=Decimal("10000"))

    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == Decimal("0.0")
    assert metrics["total_pnl"] == Decimal("0.0")
