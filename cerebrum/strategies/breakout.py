"""
Breakout strategy configuration.

Optimized for trending markets (BULL/BEAR) where price breaks through
support/resistance levels with momentum. Favors MACD and VWAP over
oscillators. Wider take-profit targets to capture trend continuation.

@decision DEC-STRAT-002
@title Breakout strategy preset with MACD/VWAP emphasis
@status accepted
@rationale Session 4 showed 73% WR in non-BEAR trending conditions. Breakout
strategy capitalizes on these by boosting trend-following indicators (MACD, VWAP),
using wider TP to ride moves, and activating only when regime confirms a trend.
Support/resistance signals are used inversely: a break *through* S/R triggers
entry rather than a bounce off it.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from cerebrum.core.types import SignalType


@dataclass(frozen=True)
class BreakoutStrategy:
    """
    Breakout strategy preset.

    Designed for trending markets where price breaks key levels with momentum.
    Emphasizes trend-following indicators (MACD, VWAP) over oscillators and
    uses support/resistance as breakout confirmation rather than bounce signals.

    Attributes:
        name: Strategy identifier.
        signal_weights: Per-signal-type weight overrides for the aggregator.
            Boosts TECHNICAL (MACD/VWAP) and REGIME (trend confirmation).
            Sentiment gets moderate weight since sentiment extremes can
            confirm trend continuation.
        aggregation_threshold: Higher than default (0.5 vs 0.4) to require
            stronger conviction before entering breakout trades. False
            breakouts are costly — the wider TP means a failed trade sits
            open longer before stop-loss triggers.
        take_profit_percent: Wider target (4.0%) to capture trend moves.
            Breakouts that succeed often run 2-5% before retracing.
        stop_loss_percent: Standard stop (2.0%) — wider than mean reversion
            because breakout entries need room to breathe through initial
            volatility after the breakout.
        position_size_percent: Larger positions (5%) since trend-following
            has higher conviction when regime confirms the direction.
        preferred_regimes: Regimes where this strategy should be active.
            BULL and VOLATILE are the primary regimes. BEAR is excluded
            because DEC-REGIME-004 halts trading in BEAR anyway.
        sr_weight: Weight for S/R signals. Lower than mean reversion (0.8 vs
            1.5) because breakout uses S/R as confirmation, not primary signal.
    """

    name: str = "breakout"

    signal_weights: dict[SignalType, Decimal] = field(default_factory=lambda: {
        SignalType.TECHNICAL: Decimal("1.3"),
        SignalType.SENTIMENT: Decimal("0.6"),
        SignalType.NEWS: Decimal("0.5"),
        SignalType.REGIME: Decimal("0.9"),
    })

    aggregation_threshold: Decimal = Decimal("0.5")
    take_profit_percent: Decimal = Decimal("4.0")
    stop_loss_percent: Decimal = Decimal("2.0")
    position_size_percent: Decimal = Decimal("5.0")

    preferred_regimes: tuple[str, ...] = ("BULL", "VOLATILE")

    sr_weight: Decimal = Decimal("0.8")

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

        In BULL markets, trend-following signals get an additional boost.
        In VOLATILE markets, all weights are slightly reduced for caution
        but the TECHNICAL boost is preserved.

        Args:
            regime: Current market regime string.

        Returns:
            Dict mapping SignalType to weight Decimal.
        """
        weights = dict(self.signal_weights)

        if regime == "BULL":
            # Boost trend-following in confirmed uptrend
            weights[SignalType.TECHNICAL] = weights[SignalType.TECHNICAL] * Decimal("1.2")
            weights[SignalType.REGIME] = weights[SignalType.REGIME] * Decimal("1.1")

        elif regime == "VOLATILE":
            # Slightly reduce all weights in volatile conditions
            for sig_type in weights:
                weights[sig_type] = weights[sig_type] * Decimal("0.9")

        return weights
