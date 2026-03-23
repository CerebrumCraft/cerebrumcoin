"""
Unit tests for strategy configuration presets.

Tests MeanReversionStrategy and BreakoutStrategy: regime affinity,
weight calculations, and parameter validation.

@decision DEC-TEST-STRAT-001
@title Strategy config tests covering regime affinity and weight math
@status accepted
@rationale Strategy presets drive aggregator weights and risk params. Tests verify
that mean reversion favors SIDEWAYS with tight TP, breakout favors BULL/VOLATILE
with wide TP, and the two strategies cover complementary regimes without overlap.
"""

from decimal import Decimal

import pytest

from cerebrum.core.types import SignalType
from cerebrum.strategies.mean_reversion import MeanReversionStrategy
from cerebrum.strategies.breakout import BreakoutStrategy


# ---------------------------------------------------------------------------
# MeanReversionStrategy tests
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    """Tests for mean reversion strategy preset."""

    def test_default_name(self):
        """Strategy has correct default name."""
        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_reversion"

    def test_preferred_regimes(self):
        """Mean reversion is active in SIDEWAYS and UNKNOWN."""
        strategy = MeanReversionStrategy()
        assert strategy.is_active_for_regime("SIDEWAYS")
        assert strategy.is_active_for_regime("UNKNOWN")
        assert not strategy.is_active_for_regime("BULL")
        assert not strategy.is_active_for_regime("BEAR")
        assert not strategy.is_active_for_regime("VOLATILE")

    def test_signal_weights_favor_technicals(self):
        """Technical signals should have the highest base weight."""
        strategy = MeanReversionStrategy()
        weights = strategy.signal_weights
        assert weights[SignalType.TECHNICAL] > weights[SignalType.SENTIMENT]
        assert weights[SignalType.TECHNICAL] > weights[SignalType.NEWS]
        assert weights[SignalType.TECHNICAL] > weights[SignalType.REGIME]

    def test_sentiment_suppressed(self):
        """Sentiment should be low since it's noise in ranging markets."""
        strategy = MeanReversionStrategy()
        assert strategy.signal_weights[SignalType.SENTIMENT] <= Decimal("0.4")

    def test_tighter_take_profit(self):
        """Mean reversion should use tighter TP than default 3%."""
        strategy = MeanReversionStrategy()
        assert strategy.take_profit_percent < Decimal("3.0")

    def test_tighter_stop_loss(self):
        """Mean reversion should use tighter stop loss."""
        strategy = MeanReversionStrategy()
        assert strategy.stop_loss_percent <= Decimal("1.5")

    def test_smaller_position_size(self):
        """Mean reversion uses smaller positions (lower conviction)."""
        strategy = MeanReversionStrategy()
        assert strategy.position_size_percent <= Decimal("5.0")

    def test_higher_sr_weight(self):
        """S/R weight should be higher than base technical weight."""
        strategy = MeanReversionStrategy()
        assert strategy.sr_weight > Decimal("1.0")

    def test_effective_weights_sideways(self):
        """SIDEWAYS regime should boost technicals further."""
        strategy = MeanReversionStrategy()
        base_weights = strategy.signal_weights
        effective = strategy.get_effective_weights("SIDEWAYS")

        # Technical should be boosted in SIDEWAYS
        assert effective[SignalType.TECHNICAL] > base_weights[SignalType.TECHNICAL]
        # Sentiment should be further suppressed in SIDEWAYS
        assert effective[SignalType.SENTIMENT] < base_weights[SignalType.SENTIMENT]

    def test_effective_weights_unknown(self):
        """UNKNOWN regime should return base weights (no modification)."""
        strategy = MeanReversionStrategy()
        base_weights = strategy.signal_weights
        effective = strategy.get_effective_weights("UNKNOWN")

        # Should be same as base in UNKNOWN (no regime-specific adjustment)
        assert effective[SignalType.TECHNICAL] == base_weights[SignalType.TECHNICAL]
        assert effective[SignalType.SENTIMENT] == base_weights[SignalType.SENTIMENT]

    def test_lower_aggregation_threshold(self):
        """Mean reversion should use a lower threshold (signals are weaker)."""
        strategy = MeanReversionStrategy()
        assert strategy.aggregation_threshold <= Decimal("0.4")

    def test_frozen_dataclass(self):
        """Strategy config should be immutable."""
        strategy = MeanReversionStrategy()
        with pytest.raises(AttributeError):
            strategy.name = "modified"


# ---------------------------------------------------------------------------
# BreakoutStrategy tests
# ---------------------------------------------------------------------------


class TestBreakoutStrategy:
    """Tests for breakout strategy preset."""

    def test_default_name(self):
        """Strategy has correct default name."""
        strategy = BreakoutStrategy()
        assert strategy.name == "breakout"

    def test_preferred_regimes(self):
        """Breakout is active in BULL and VOLATILE."""
        strategy = BreakoutStrategy()
        assert strategy.is_active_for_regime("BULL")
        assert strategy.is_active_for_regime("VOLATILE")
        assert not strategy.is_active_for_regime("BEAR")
        assert not strategy.is_active_for_regime("SIDEWAYS")
        assert not strategy.is_active_for_regime("UNKNOWN")

    def test_signal_weights_favor_technicals(self):
        """Technical signals should have the highest weight."""
        strategy = BreakoutStrategy()
        weights = strategy.signal_weights
        assert weights[SignalType.TECHNICAL] >= weights[SignalType.SENTIMENT]
        assert weights[SignalType.TECHNICAL] >= weights[SignalType.NEWS]
        assert weights[SignalType.TECHNICAL] >= weights[SignalType.REGIME]

    def test_higher_regime_weight(self):
        """Breakout should weight regime higher than mean reversion."""
        breakout = BreakoutStrategy()
        mean_rev = MeanReversionStrategy()
        assert breakout.signal_weights[SignalType.REGIME] > mean_rev.signal_weights[SignalType.REGIME]

    def test_wider_take_profit(self):
        """Breakout should use wider TP to capture trend moves."""
        strategy = BreakoutStrategy()
        assert strategy.take_profit_percent >= Decimal("3.0")

    def test_wider_stop_loss(self):
        """Breakout should use wider stop to ride through volatility."""
        strategy = BreakoutStrategy()
        assert strategy.stop_loss_percent >= Decimal("1.5")

    def test_larger_position_size(self):
        """Breakout uses larger positions (higher conviction)."""
        strategy = BreakoutStrategy()
        mean_rev = MeanReversionStrategy()
        assert strategy.position_size_percent >= mean_rev.position_size_percent

    def test_lower_sr_weight(self):
        """Breakout uses S/R as confirmation, not primary signal."""
        strategy = BreakoutStrategy()
        mean_rev = MeanReversionStrategy()
        assert strategy.sr_weight < mean_rev.sr_weight

    def test_higher_aggregation_threshold(self):
        """Breakout requires stronger conviction to enter."""
        strategy = BreakoutStrategy()
        mean_rev = MeanReversionStrategy()
        assert strategy.aggregation_threshold > mean_rev.aggregation_threshold

    def test_effective_weights_bull(self):
        """BULL regime should boost trend-following signals."""
        strategy = BreakoutStrategy()
        base_weights = strategy.signal_weights
        effective = strategy.get_effective_weights("BULL")

        # Technical and regime should be boosted
        assert effective[SignalType.TECHNICAL] > base_weights[SignalType.TECHNICAL]
        assert effective[SignalType.REGIME] > base_weights[SignalType.REGIME]

    def test_effective_weights_volatile(self):
        """VOLATILE regime should slightly reduce all weights."""
        strategy = BreakoutStrategy()
        base_weights = strategy.signal_weights
        effective = strategy.get_effective_weights("VOLATILE")

        for sig_type in base_weights:
            assert effective[sig_type] < base_weights[sig_type]

    def test_effective_weights_nonpreferred_regime(self):
        """Non-preferred regime returns base weights."""
        strategy = BreakoutStrategy()
        base_weights = strategy.signal_weights
        effective = strategy.get_effective_weights("BEAR")

        # No adjustments for non-preferred regimes
        assert effective[SignalType.TECHNICAL] == base_weights[SignalType.TECHNICAL]

    def test_frozen_dataclass(self):
        """Strategy config should be immutable."""
        strategy = BreakoutStrategy()
        with pytest.raises(AttributeError):
            strategy.name = "modified"


# ---------------------------------------------------------------------------
# Cross-strategy comparison tests
# ---------------------------------------------------------------------------


class TestStrategyComparison:
    """Tests that verify the strategies are meaningfully different."""

    def test_different_preferred_regimes(self):
        """Mean reversion and breakout should not overlap regimes."""
        mr = MeanReversionStrategy()
        bo = BreakoutStrategy()
        mr_regimes = set(mr.preferred_regimes)
        bo_regimes = set(bo.preferred_regimes)
        assert mr_regimes.isdisjoint(bo_regimes), "Strategies should cover different regimes"

    def test_take_profit_ordering(self):
        """Breakout TP should be wider than mean reversion TP."""
        mr = MeanReversionStrategy()
        bo = BreakoutStrategy()
        assert bo.take_profit_percent > mr.take_profit_percent

    def test_complementary_coverage(self):
        """Together, both strategies should cover at least 3 regimes."""
        mr = MeanReversionStrategy()
        bo = BreakoutStrategy()
        all_regimes = set(mr.preferred_regimes) | set(bo.preferred_regimes)
        assert len(all_regimes) >= 3

    def test_both_have_sr_weight(self):
        """Both strategies should define an S/R weight."""
        mr = MeanReversionStrategy()
        bo = BreakoutStrategy()
        assert mr.sr_weight > Decimal("0")
        assert bo.sr_weight > Decimal("0")
