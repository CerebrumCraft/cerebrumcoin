#!/usr/bin/env python3
"""
Display trading statistics from database.

@decision DEC-MONITOR-004
@title CLI stats viewer with filtering
@status accepted
@rationale Quick stats viewing without running full system. Reads directly from
cerebrum.db, supports filtering by regime/date/symbol. Uses stats.py for calculations.
"""

import argparse
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from cerebrum.core.state import StateManager
from cerebrum.monitoring.stats import calculate_performance_metrics


async def show_stats(
    db_path: Path,
    regime: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    initial_balance: Decimal = Decimal("10000.0"),
) -> None:
    """Display statistics from database."""
    # Initialize state manager
    state_manager = StateManager(db_path)
    await state_manager.initialize()
    
    try:
        # Get trades
        trades = await state_manager.get_closed_trades(regime=regime, limit=limit)
        
        # Filter by symbol if requested
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Calculate metrics
        metrics = calculate_performance_metrics(trades, initial_balance)
        
        # Display
        print("\n" + "=" * 80)
        print(" CerebrumCoin Statistics ".center(80, "="))
        print("=" * 80)
        
        if regime:
            print(f"\nRegime: {regime}")
        if symbol:
            print(f"Symbol: {symbol}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Avg Profit: ${metrics['avg_profit']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"  Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        print("\n" + "=" * 80 + "\n")
    
    finally:
        await state_manager.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Display trading statistics")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to database file",
    )
    parser.add_argument(
        "--regime",
        type=str,
        help="Filter by regime (BULL, BEAR, SIDEWAYS)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Filter by symbol (e.g., BTC/USD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of trades to analyze",
    )
    parser.add_argument(
        "--balance",
        type=Decimal,
        default=Decimal("10000.0"),
        help="Initial balance for metrics calculation",
    )
    
    args = parser.parse_args()
    
    asyncio.run(show_stats(
        args.db,
        regime=args.regime,
        symbol=args.symbol,
        limit=args.limit,
        initial_balance=args.balance,
    ))


if __name__ == "__main__":
    main()
