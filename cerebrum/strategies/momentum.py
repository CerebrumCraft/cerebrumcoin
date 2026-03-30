"""
MomentumStrategy configuration — the default single-strategy config.

This config exactly mirrors the current paper.toml tuned values so that
running with a single MomentumStrategy produces identical behavior to the
pre-refactor main.py wiring. It is the backward-compatibility anchor.

Weights match the SignalAggregator defaults that have been in production
since Phase 2. The threshold, risk_overrides, and exit_config reflect the
paper.toml values tuned through Sessions 2–6.

@decision DEC-STRAT-006
@title Backward-compatible single-strategy mode in main.py
@status accepted
@rationale If no strategy config is present in TOML, main.py creates a single
MomentumStrategy pipeline using this config — producing behaviour identical to
the pre-refactor wiring. MOMENTUM_CONFIG.initial_balance is set to 1/4 of $10k
to match the 6-strategy equal-split (momentum + mean_reversion + breakout +
range_trading + swing_trading + news_driven). The risk_overrides and exit_config mirror paper.toml exactly so
Session 2–6 tuning is preserved. stop_loss_percent tightened from 1.5% to 1.0%
per Session 11 sensitivity analysis: SL=1.0% yielded AdjPnL=$183 vs $65 at 1.5%
(+$118 improvement). See DEC-TUNE-004.
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig

# MomentumStrategy: the current production strategy.
#
# aggregator_weights: match SignalAggregator defaults (technical-first)
# aggregator_threshold: 0.4 — matches paper.toml aggregation_threshold
# risk_overrides: paper.toml [risk] section values
# exit_config: paper.toml exit parameters
# initial_balance: 1/6 of $10k — equal split across 6 strategies
# symbols: current paper trading symbols
MOMENTUM_CONFIG = StrategyConfig(
    name="momentum",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.0"),
        SignalType.SENTIMENT: Decimal("0.5"),
        SignalType.NEWS: Decimal("0.3"),
        SignalType.REGIME: Decimal("0.7"),
    },
    aggregator_threshold=Decimal("0.4"),
    risk_overrides={
        "min_signal_strength": "0.6",
        "position_size_percent": "5.0",
        "stop_loss_percent": "1.0",
        "take_profit_percent": "3.0",
        "post_fill_cooldown_seconds": 900,
        "volatility_gate_min_range_pct": "0.5",
        "volatility_gate_window_size": 300,
        "sideways_suppression_min_range_pct": "1.0",
        "sideways_suppression_window_size": 18000,
        "macro_volatility_min_range_pct": "0.8",
        "macro_volatility_window_size": 18000,
    },
    exit_config={
        "stop_loss_percent": "1.0",
        "take_profit_percent": "3.0",
        "max_position_age_minutes": 120,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "0.3",
    },
    initial_balance=Decimal("1666.67"),  # 1/6 of $10k for 6-strategy split
    # @decision DEC-TUNE-006
    # @title Remove BTC/USD from momentum strategy
    # @status accepted
    # @rationale Session 18: momentum bought BTC at $67,653 and sold at $67,342 (-$311 move).
    #   Short-timeframe momentum signals aren't catching BTC trends. BTC exposure remains via
    #   mean_reversion (top performer, +$877) and news_driven (+$650). Per-pair thresholds
    #   tabled for future implementation.
    symbols=["ETH/USD", "SOL/USD", "DOGE/USD"],
)
