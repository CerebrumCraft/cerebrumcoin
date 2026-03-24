"""
Strategy abstraction layer for CerebrumCoin.

Provides the StrategyConfig data model, built-in strategy configs,
StrategyRegistry for lifecycle management, and GlobalPortfolio for
cross-strategy aggregation.

Phase 11A: Multi-strategy foundation. Each strategy gets isolated pipelines
(SignalAggregator, PortfolioTracker, ExitMonitor, RiskManager) wired through
a shared EventBus.
"""

from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.registry import StrategyRegistry
from cerebrum.strategies.global_portfolio import GlobalPortfolio
from cerebrum.strategies.mean_reversion import MeanReversionStrategy, MEAN_REVERSION_CONFIG
from cerebrum.strategies.breakout import BreakoutStrategy, BREAKOUT_CONFIG
from cerebrum.strategies.momentum import MOMENTUM_CONFIG

__all__ = [
    "StrategyConfig",
    "StrategyRegistry",
    "GlobalPortfolio",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "MOMENTUM_CONFIG",
    "MEAN_REVERSION_CONFIG",
    "BREAKOUT_CONFIG",
]
