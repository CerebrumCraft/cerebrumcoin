"""
Session report generator.

Generates summary reports of trading sessions.

@decision DEC-MONITOR-003
@title Session reporter with file and console output
@status accepted
@rationale Post-session analysis requires comprehensive reports. Reporter uses stats.py
for calculations and outputs to both console and file. Report includes all metrics,
trade list, and regime breakdown.
"""

from decimal import Decimal
from pathlib import Path

import structlog

from cerebrum.core.state import StateManager, TradeRecord
from cerebrum.monitoring.stats import calculate_performance_metrics

logger = structlog.get_logger()


class SessionReporter:
    """Generates trading session reports."""
    
    def __init__(self, state_manager: StateManager) -> None:
        """Initialize reporter."""
        self._state_manager = state_manager
        self._log = logger.bind(component="reporter")
    
    async def generate_report(
        self,
        initial_balance: Decimal,
        output_file: Path | None = None,
    ) -> str:
        """
        Generate comprehensive session report.
        
        Args:
            initial_balance: Starting balance
            output_file: Optional file path to save report
        
        Returns:
            Report text
        """
        # Get all closed trades
        trades = await self._state_manager.get_closed_trades()
        
        # Calculate metrics
        metrics = calculate_performance_metrics(trades, initial_balance)
        
        # Build report
        report_lines = [
            "=" * 80,
            " CerebrumCoin Session Report ".center(80, "="),
            "=" * 80,
            "",
            "Performance Metrics:",
            f"  Total Trades: {metrics['total_trades']}",
            f"  Win Rate: {metrics['win_rate']:.2f}%",
            f"  Profit Factor: {metrics['profit_factor']:.2f}",
            f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}",
            f"  Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)",
            f"  Total PnL: ${metrics['total_pnl']:.2f}",
            f"  Avg Profit: ${metrics['avg_profit']:.2f}",
            f"  Avg Loss: ${metrics['avg_loss']:.2f}",
            f"  Max Consecutive Wins: {metrics['max_consecutive_wins']}",
            f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}",
            "",
        ]
        
        # Regime breakdown
        regime_stats = await self._get_regime_breakdown(trades)
        if regime_stats:
            report_lines.append("Regime Breakdown:")
            for regime, stats in regime_stats.items():
                report_lines.append(f"  {regime}: {stats['count']} trades, PnL ${stats['pnl']:.2f}")
            report_lines.append("")
        
        # Trade list
        if trades:
            report_lines.append(f"Trade History ({len(trades)} trades):")
            for trade in trades[-20:]:  # Last 20 trades
                report_lines.append(
                    f"  {trade.symbol} {trade.side.value}: "
                    f"Entry ${trade.entry_price} → Exit ${trade.exit_price} "
                    f"| PnL ${trade.pnl:.2f} | Regime {trade.regime}"
                )
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # Output to console
        print(report)
        
        # Output to file
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report)
            self._log.info("report_saved", file=str(output_file))
        
        return report
    
    async def _get_regime_breakdown(
        self,
        trades: list[TradeRecord]
    ) -> dict[str, dict]:
        """Calculate statistics per regime."""
        breakdown = {}
        for trade in trades:
            if trade.regime not in breakdown:
                breakdown[trade.regime] = {"count": 0, "pnl": Decimal("0.0")}
            breakdown[trade.regime]["count"] += 1
            if trade.pnl:
                breakdown[trade.regime]["pnl"] += trade.pnl
        return breakdown
