"""
Performance statistics calculator for trading system.

Pure functions for calculating Sharpe ratio, Sortino ratio, profit factor,
drawdown, win rate, and other performance metrics from trade history.

@decision DEC-MONITOR-001
@title Pure function stats calculator
@status accepted
@rationale Performance metrics must be deterministic and testable. Pure functions
with no side effects enable easy unit testing. All metrics calculated from TradeRecord
list and initial balance. No database access or event bus dependencies.
"""

from decimal import Decimal
from typing import Any

import structlog

from cerebrum.core.state import TradeRecord

logger = structlog.get_logger()


def calculate_win_rate(trades: list[TradeRecord]) -> Decimal:
    """Calculate win rate percentage."""
    if not trades:
        return Decimal("0.0")
    
    winners = sum(1 for t in trades if t.pnl and t.pnl > 0)
    return Decimal(str(winners)) / Decimal(str(len(trades))) * Decimal("100.0")


def calculate_avg_profit(trades: list[TradeRecord]) -> Decimal:
    """Calculate average profit of winning trades."""
    winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
    if not winning_trades:
        return Decimal("0.0")
    
    total_profit = sum(t.pnl for t in winning_trades if t.pnl)
    return total_profit / Decimal(str(len(winning_trades)))


def calculate_avg_loss(trades: list[TradeRecord]) -> Decimal:
    """Calculate average loss of losing trades."""
    losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
    if not losing_trades:
        return Decimal("0.0")
    
    total_loss = sum(t.pnl for t in losing_trades if t.pnl)
    return total_loss / Decimal(str(len(losing_trades)))


def calculate_profit_factor(trades: list[TradeRecord]) -> Decimal:
    """
    Calculate profit factor (total profit / total loss).
    
    Returns very large number if no losses.
    """
    total_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
    
    if total_loss == 0:
        return Decimal("1000000") if total_profit > 0 else Decimal("0.0")
    
    return total_profit / total_loss


def calculate_sharpe_ratio(
    trades: list[TradeRecord],
    risk_free_rate: Decimal = Decimal("0.0")
) -> Decimal:
    """
    Calculate Sharpe ratio (annualized excess return / volatility).
    
    Uses trade returns. If variance is zero, returns very large number.
    """
    if not trades:
        return Decimal("0.0")
    
    returns = [t.pnl for t in trades if t.pnl]
    if not returns:
        return Decimal("0.0")
    
    mean_return = sum(returns) / Decimal(str(len(returns)))
    
    # Calculate variance
    variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
    
    if variance == 0:
        # No variance means consistent returns
        return Decimal("1000000") if mean_return > risk_free_rate else Decimal("0.0")
    
    std_dev = variance.sqrt()
    excess_return = mean_return - risk_free_rate
    
    return excess_return / std_dev if std_dev > 0 else Decimal("0.0")


def calculate_sortino_ratio(
    trades: list[TradeRecord],
    target_return: Decimal = Decimal("0.0")
) -> Decimal:
    """
    Calculate Sortino ratio (excess return / downside deviation).
    
    Only considers downside volatility (negative returns).
    """
    if not trades:
        return Decimal("0.0")
    
    returns = [t.pnl for t in trades if t.pnl]
    if not returns:
        return Decimal("0.0")
    
    mean_return = sum(returns) / Decimal(str(len(returns)))
    
    # Calculate downside deviation (only negative returns)
    downside_returns = [r for r in returns if r < target_return]
    if not downside_returns:
        return Decimal("1000000") if mean_return > target_return else Decimal("0.0")
    
    downside_variance = sum((r - target_return) ** 2 for r in downside_returns) / Decimal(str(len(downside_returns)))
    downside_deviation = downside_variance.sqrt()
    
    excess_return = mean_return - target_return
    
    return excess_return / downside_deviation if downside_deviation > 0 else Decimal("0.0")


def calculate_max_drawdown(
    trades: list[TradeRecord],
    initial_balance: Decimal
) -> tuple[Decimal, Decimal]:
    """
    Calculate maximum drawdown in absolute and percentage terms.
    
    Returns (max_drawdown_usd, max_drawdown_pct).
    """
    if not trades:
        return Decimal("0.0"), Decimal("0.0")
    
    # Build equity curve
    equity = initial_balance
    peak = initial_balance
    max_dd = Decimal("0.0")
    
    for trade in trades:
        if trade.pnl:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
    
    max_dd_pct = (max_dd / peak * Decimal("100.0")) if peak > 0 else Decimal("0.0")
    
    return max_dd, max_dd_pct


def calculate_consecutive_wins(trades: list[TradeRecord]) -> int:
    """Calculate maximum consecutive winning trades."""
    if not trades:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for trade in trades:
        if trade.pnl and trade.pnl > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def calculate_consecutive_losses(trades: list[TradeRecord]) -> int:
    """Calculate maximum consecutive losing trades."""
    if not trades:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for trade in trades:
        if trade.pnl and trade.pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def calculate_performance_metrics(
    trades: list[TradeRecord],
    initial_balance: Decimal,
    risk_free_rate: Decimal = Decimal("0.02")
) -> dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Returns dictionary with all key metrics:
    - total_trades
    - win_rate
    - profit_factor
    - sharpe_ratio
    - sortino_ratio
    - max_drawdown (USD)
    - max_drawdown_pct
    - total_pnl
    - avg_profit
    - avg_loss
    - max_consecutive_wins
    - max_consecutive_losses
    """
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    max_dd, max_dd_pct = calculate_max_drawdown(trades, initial_balance)
    
    return {
        "total_trades": len(trades),
        "win_rate": calculate_win_rate(trades),
        "profit_factor": calculate_profit_factor(trades),
        "sharpe_ratio": calculate_sharpe_ratio(trades, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(trades, target_return=Decimal("0.0")),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_pnl": total_pnl,
        "avg_profit": calculate_avg_profit(trades),
        "avg_loss": calculate_avg_loss(trades),
        "max_consecutive_wins": calculate_consecutive_wins(trades),
        "max_consecutive_losses": calculate_consecutive_losses(trades),
    }
