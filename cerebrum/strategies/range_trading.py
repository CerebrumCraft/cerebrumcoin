"""
Range trading strategy configuration for SIDEWAYS markets.

Buys near support and sells near resistance within confirmed price ranges.
Uses S/R signals exclusively (filtered via signal_source_filter). Exempt from
SidewaysSuppressionRule because SIDEWAYS is its target regime.

@decision DEC-RANGE-006
@title Dedicated range strategy with S/R-only signal filtering
@status accepted
@rationale Mean reversion uses same RSI/MACD signals with different weights —
it's "momentum-lite" not true range trading. A dedicated strategy filters to
S/R signals only and uses structural exits at resistance/support levels.
Capital is 1/4 of $10k ($2,500) for equal-split 4-strategy mode. The
SidewaysSuppressionRule exemption is required because that rule blocks BUY
entries in low-volatility SIDEWAYS markets — which is exactly where range
trading operates. Without the exemption, every range-trading signal would be
denied by the guard that exists to protect the other strategies from the same
market conditions range_trading is designed to exploit.
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig


def _create_range_exit_monitor(bus, portfolio, config, app_config):
    """
    Factory that constructs a RangeDetector + RangeExitMonitor pair.

    Called by StrategyRegistry._build_pipeline() when cfg.exit_monitor_factory
    is set. Returns a fully wired RangeExitMonitor whose internal RangeDetector
    is also stored as monitor._range_detector for external access (e.g. testing,
    conductor introspection).

    Args:
        bus: EventBus shared by all pipeline components.
        portfolio: This strategy's PortfolioTracker instance.
        config: The StrategyConfig for the range_trading strategy.
        app_config: The global Config (paper.toml values).

    Returns:
        RangeExitMonitor wired to bus and portfolio.
    """
    from cerebrum.strategies.range_detector import RangeDetector
    from cerebrum.risk.range_exit_monitor import RangeExitMonitor

    detector = RangeDetector(bus=bus, min_bounces=3)
    monitor = RangeExitMonitor(
        bus=bus,
        portfolio=portfolio,
        range_detector=detector,
        # DEC-RANGE-007: tag emitted orders so per-strategy portfolio routes fills
        strategy_id=config.name,
        # DEC-EXIT-006: read min_hold_minutes from exit_config (default 0)
        min_hold_minutes=int(config.exit_config.get("min_hold_minutes", 0)),
    )
    # Attach detector to monitor for external access (registry / conductor).
    monitor._range_detector = detector
    return monitor


RANGE_TRADING_CONFIG = StrategyConfig(
    name="range_trading",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.0"),  # S/R is the only source after filter
        SignalType.SENTIMENT: Decimal("0.0"),
        SignalType.NEWS: Decimal("0.0"),
        SignalType.REGIME: Decimal("0.0"),
    },
    aggregator_threshold=Decimal("0.2"),
    signal_source_filter="SupportResistance",
    risk_overrides={
        "min_signal_strength": "0.5",  # DEC-TUNE-012: raised from 0.3 — weaker S/R signals were generating commission-losing trades
        "position_size_percent": "5.0",  # DEC-SIZING-002: raised from 2% — at 2% × $5k = $100, any signal_strength < 1.0 caused denial. At 5% × 0.6 floor × $5k = $150, comfortably above $100 min.
        "post_fill_cooldown_seconds": 1800,  # DEC-TUNE-010: cooldown 900→1800s (was DEC-TUNE-009: 300→900s) to further reduce trade frequency and commission drag
    },
    exit_config={
        "stop_loss_percent": "0.5",
        "take_profit_percent": "1.0",
        "max_position_age_minutes": 60,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "0.2",
        "min_hold_minutes": 15,  # DEC-EXIT-006: skip SL/TP for first 15 min to reduce premature exits
    },
    initial_balance=Decimal("5000.00"),  # DEC-TUNE-008: 2-strategy split — $5,000 each (was $1,667 across 6)
    symbols=["BTC/USD", "ETH/USD"],  # DEC-TUNE-013: DOGE removed — only 2 trades in session 28 (insufficient data); BTC+ETH only
    exit_monitor_factory=_create_range_exit_monitor,
)
