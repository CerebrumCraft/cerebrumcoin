"""
Mean reversion strategy configuration.

Optimized for SIDEWAYS and range-bound markets where price oscillates around
a mean. Favors Bollinger Bands and RSI (oversold/overbought) over trend-following
indicators like MACD. Tighter take-profit targets since price moves are smaller.

@decision DEC-STRAT-001
@title Mean reversion strategy preset with BB/RSI emphasis
@status accepted
@rationale Session 4 data showed 73% WR in non-BEAR regimes but 0% in SIDEWAYS
(session 5). Mean reversion explicitly configures the system for range-bound
conditions: boost BB/RSI weights, suppress MACD, use tighter TP, and enable
support/resistance signals for bounce detection.

@decision DEC-STRAT-008
@title MEAN_REVERSION_CONFIG as StrategyConfig for StrategyRegistry wiring
@status accepted
@rationale The StrategyRegistry expects StrategyConfig instances (pure data
objects). MeanReversionStrategy was implemented as a custom dataclass — useful
for documentation but not directly consumable by the registry. MEAN_REVERSION_CONFIG
bridges the two: it extracts the tuned parameters from MeanReversionStrategy and
packages them as a StrategyConfig. Capital allocation is 1/3 of $10k ($3,333) for
equal-split 3-strategy mode.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from cerebrum.core.types import SignalType


@dataclass(frozen=True)
class MeanReversionStrategy:
    """
    Mean reversion strategy preset.

    Designed for range-bound markets where price reverts to a mean value.
    Emphasizes oscillators (RSI, Bollinger Bands) and support/resistance
    levels over trend-following indicators.

    Attributes:
        name: Strategy identifier.
        signal_weights: Per-signal-type weight overrides for the aggregator.
            Boosts TECHNICAL (BB/RSI) and keeps SENTIMENT low since sentiment
            is noise in ranging markets (DEC-SENT-001).
        aggregation_threshold: Lower than default (0.3 vs 0.4) because mean
            reversion signals are inherently weaker — price near a band edge
            may only produce 0.3-0.5 strength.
        take_profit_percent: Tighter target (1.5%) since range-bound moves
            are small. Session 5 showed 3% TP never triggered in <0.5% range.
        stop_loss_percent: Tight stop (1.0%) to cut losses quickly when the
            range breaks and mean reversion fails.
        position_size_percent: Smaller positions (3%) since mean reversion
            has lower conviction than trend-following.
        preferred_regimes: Regimes where this strategy should be active.
            SIDEWAYS is the primary regime; UNKNOWN is included as fallback
            since ambiguous conditions often turn out to be ranging.
        sr_weight: Weight for support/resistance signals within the TECHNICAL
            signal type. Higher than other technicals because S/R bounces are
            the core mean reversion signal.
    """

    name: str = "mean_reversion"

    signal_weights: dict[SignalType, Decimal] = field(default_factory=lambda: {
        SignalType.TECHNICAL: Decimal("1.2"),
        SignalType.SENTIMENT: Decimal("0.3"),
        SignalType.NEWS: Decimal("0.2"),
        SignalType.REGIME: Decimal("0.5"),
    })

    aggregation_threshold: Decimal = Decimal("0.3")
    take_profit_percent: Decimal = Decimal("1.5")
    stop_loss_percent: Decimal = Decimal("1.0")
    position_size_percent: Decimal = Decimal("3.0")

    preferred_regimes: tuple[str, ...] = ("SIDEWAYS", "UNKNOWN")

    sr_weight: Decimal = Decimal("1.5")

    def is_active_for_regime(self, regime: str) -> bool:
        """
        Check whether this strategy should be active in the given regime.

        Args:
            regime: Current market regime string (BULL, BEAR, SIDEWAYS, etc.)

        Returns:
            True if the regime is in the preferred list.
        """
        return regime in self.preferred_regimes

    def get_effective_weights(self, regime: str) -> dict[SignalType, Decimal]:
        """
        Return signal weights adjusted for the current regime.

        In SIDEWAYS markets, technical weight is boosted further and sentiment
        is suppressed. In other preferred regimes, base weights are returned.

        Args:
            regime: Current market regime string.

        Returns:
            Dict mapping SignalType to weight Decimal.
        """
        weights = dict(self.signal_weights)

        if regime == "SIDEWAYS":
            # Extra boost to technicals in confirmed sideways
            weights[SignalType.TECHNICAL] = weights[SignalType.TECHNICAL] * Decimal("1.1")
            weights[SignalType.SENTIMENT] = weights[SignalType.SENTIMENT] * Decimal("0.5")

        return weights


# StrategyConfig instance for use with StrategyRegistry (DEC-STRAT-008).
# Parameters derived from MeanReversionStrategy defaults tuned through Session 6.
# Capital is 1/4 of $10k — the 4-strategy equal-split starting point.
from cerebrum.strategies.base import StrategyConfig  # noqa: E402 (below class def intentional)

MEAN_REVERSION_CONFIG = StrategyConfig(
    name="mean_reversion",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.2"),
        SignalType.SENTIMENT: Decimal("0.3"),
        SignalType.NEWS: Decimal("0.2"),
        SignalType.REGIME: Decimal("0.5"),
    },
    aggregator_threshold=Decimal("0.3"),
    risk_overrides={
        "min_signal_strength": "0.5",
        "position_size_percent": "3.0",
        "post_fill_cooldown_seconds": 600,
    },
    exit_config={
        "stop_loss_percent": "1.0",
        "take_profit_percent": "1.5",
        "max_position_age_minutes": 90,
        "adaptive_tp": True,
        "tp_multiplier": "1.2",
        "min_tp_percent": "0.2",
    },
    initial_balance=Decimal("2000.00"),  # 1/5 of $10k for 5-strategy split
    symbols=["BTC/USD", "ETH/USD"],
)
