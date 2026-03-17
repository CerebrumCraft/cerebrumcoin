"""
Market regime detection for CerebrumCoin.

Detects market regimes (BULL, BEAR, SIDEWAYS, VOLATILE) using HMM or rule-based fallback.

@decision DEC-INT-003
@title Optional HMM with rule-based fallback for regime detection
@status accepted
@rationale HMM (via hmmlearn) provides probabilistic regime detection but requires
optional dependency. Rule-based fallback (volatility + trend) works without ML libs.
Both approaches detect 4 regimes: BULL, BEAR, SIDEWAYS, VOLATILE.

@decision DEC-INT-005
@title Regime-based signal weight adjustment
@status accepted
@rationale Different regimes favor different strategies. BULL: boost trend-following.
BEAR: boost risk-off. VOLATILE: reduce confidence. SIDEWAYS: favor mean-reversion.

@decision DEC-REGIME-001
@title Cumulative return + MA slope for slow-trend detection
@status accepted
@rationale The original detector used only np.mean(returns) with a 0.2% threshold.
This missed slow bleeds (e.g. -0.01%/step * 100 steps = -1% cumulative) because each
individual return was below the threshold. The fix adds two additional signals:
  1. cumulative_return: (last_price - first_price) / first_price — captures total drift
  2. ma_slope: slope of short-term SMA, normalized by price — captures directional momentum
A regime is classified as BEAR/BULL if EITHER (a) mean_return exceeds the threshold
(existing logic, unchanged) OR (b) cumulative return AND MA slope both agree on direction.
Confidence is derived from how many of the 3 metrics (mean_return, cumulative, ma_slope)
agree with the classified direction.

@decision DEC-REGIME-003
@title Dual-window regime detection for ultra-slow drift
@status accepted
@rationale Single window (5 min) cannot detect drifts slower than ~0.04%/min.
Adding a 50-min long window catches cumulative drifts as small as 0.1%,
preventing the bot from buying into slow bleeds that SIDEWAYS classification misses.
Session 3 evidence: 0/28 win rate, -$128 PnL with 1% drift over 6.5 hours undetected.
The long window only overrides a SIDEWAYS classification — if the short window already
detects BULL or BEAR, the long window does not interfere.
"""

import asyncio
from collections import deque
from decimal import Decimal
from typing import Deque

import numpy as np
import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, MarketDataEvent, RegimeChangeEvent
from cerebrum.core.types import EventType, Symbol

logger = structlog.get_logger()


class RegimeDetector:
    """
    Market regime detection using HMM or rule-based approach.

    Regimes:
    - BULL: Strong uptrend, low volatility
    - BEAR: Strong downtrend, low volatility
    - SIDEWAYS: No clear trend, low volatility
    - VOLATILE: High volatility regardless of trend

    The rule-based detector uses three complementary metrics to catch slow trends:
    - mean_return: per-step average return (catches strong trends quickly)
    - cumulative_return: total drift over the window (catches slow bleeds)
    - ma_slope: moving-average slope (confirms directional momentum)

    A second, independent long-window price history (default ~50 min) provides a
    fallback when the short window classifies SIDEWAYS: if the long window detects
    a cumulative drift beyond long_cumulative_threshold, it overrides to BULL/BEAR.

    All thresholds are configurable. Confidence (0.0-1.0) is reported with each
    RegimeChangeEvent based on how many metrics agree with the classification.
    """

    def __init__(
        self,
        bus: EventBus,
        window_size: int = 100,
        update_interval: int = 20,
        use_hmm: bool = False,
        cumulative_trend_threshold: float = 0.005,
        ma_slope_threshold: float = 0.00005,
        mean_return_threshold: float = 0.002,
        volatility_threshold: float = 0.03,
        ma_period: int = 10,
        long_window_size: int = 3000,
        long_cumulative_threshold: float = 0.001,
    ) -> None:
        """
        Initialize regime detector.

        Args:
            bus: Event bus
            window_size: Number of price points for short-window regime calculation
            update_interval: Update regime every N market data events
            use_hmm: Use HMM if hmmlearn available (else rule-based)
            cumulative_trend_threshold: Cumulative return threshold for slow-trend detection
            ma_slope_threshold: MA slope threshold for directional momentum
            mean_return_threshold: Per-step mean return threshold (original logic)
            volatility_threshold: Volatility threshold for VOLATILE regime
            ma_period: Moving-average period for slope calculation
            long_window_size: Number of price points for long-window drift detection (~50 min)
            long_cumulative_threshold: Cumulative return threshold for long-window override (0.1%)
        """
        self._bus = bus
        self._window_size = window_size
        self._update_interval = update_interval
        self._use_hmm = use_hmm

        # Configurable thresholds
        self._cumulative_trend_threshold = cumulative_trend_threshold
        self._ma_slope_threshold = ma_slope_threshold
        self._mean_return_threshold = mean_return_threshold
        self._volatility_threshold = volatility_threshold
        self._ma_period = ma_period

        # Long-window parameters (DEC-REGIME-003)
        self._long_window_size = long_window_size
        self._long_cumulative_threshold = long_cumulative_threshold

        # Per-symbol price history — short window and long window are independent deques
        self._price_history: dict[Symbol, Deque[Decimal]] = {}
        self._long_price_history: dict[Symbol, Deque[Decimal]] = {}
        self._event_counts: dict[Symbol, int] = {}

        # Current regime per symbol
        self._current_regime: dict[Symbol, str] = {}

        # Last computed metrics (set by _detect_regime_rules, read by _update_regime)
        self._last_metrics: dict[str, float] = {}

        self._log = logger.bind(component="regime_detector")

        # Check if HMM is available
        self._hmm_available = False
        if use_hmm:
            try:
                from hmmlearn.hmm import GaussianHMM  # noqa: F401
                self._hmm_available = True
                self._log.info("hmm_enabled")
            except ImportError:
                self._log.warning("hmm_unavailable",
                                  message="hmmlearn not installed. Using rule-based fallback.")

        # Subscribe to market data
        bus.subscribe(EventType.MARKET_DATA, self._on_market_data, subscriber_name="regime_detector")

    async def _on_market_data(self, event: Event) -> None:
        """Handle market data and update regime."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.price

        # Initialize tracking for new symbol
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._window_size)
            self._long_price_history[symbol] = deque(maxlen=self._long_window_size)
            self._event_counts[symbol] = 0
            self._current_regime[symbol] = "UNKNOWN"

        # Add price to both short and long history deques
        self._price_history[symbol].append(price)
        self._long_price_history[symbol].append(price)
        self._event_counts[symbol] += 1

        # Check if it's time to update regime
        if len(self._price_history[symbol]) < 30:  # Need minimum data
            return

        if self._event_counts[symbol] % self._update_interval == 0:
            await self._update_regime(symbol)

    async def _update_regime(self, symbol: Symbol) -> None:
        """Calculate and update regime for a symbol."""
        prices = list(self._price_history[symbol])
        long_prices = list(self._long_price_history[symbol])

        if len(prices) < 30:
            return

        # Detect regime — both methods now return (regime, confidence)
        if self._hmm_available and self._use_hmm:
            new_regime, confidence = self._detect_regime_hmm(prices)
        else:
            new_regime, confidence = self._detect_regime_rules(prices, long_prices=long_prices)

        # Check for regime change
        old_regime = self._current_regime[symbol]
        if new_regime != old_regime:
            self._current_regime[symbol] = new_regime

            event = RegimeChangeEvent(
                event_type=EventType.REGIME_CHANGE,
                timestamp=asyncio.get_event_loop().time(),
                from_regime=old_regime,
                to_regime=new_regime,
                confidence=Decimal(str(confidence)),
                indicators={
                    "symbol": symbol,
                    "price_points": len(prices),
                    **{k: round(v, 8) for k, v in self._last_metrics.items()},
                },
            )

            await self._bus.publish(event)

            self._log.info(
                "regime_change",
                symbol=symbol,
                from_regime=old_regime,
                to_regime=new_regime,
                confidence=round(confidence, 2),
            )

    def _detect_regime_rules(
        self,
        prices: list[Decimal],
        long_prices: list[Decimal] | None = None,
    ) -> tuple[str, float]:
        """Rule-based regime detection with cumulative return and MA slope.

        Returns (regime, confidence) where confidence is 0.0-1.0.

        Three complementary metrics are combined for the short window:
          1. mean_return: average per-step return — catches strong trends quickly
          2. cumulative_return: total drift from first to last price — catches slow bleeds
          3. ma_slope: normalized slope of the short-term SMA — confirms momentum

        Classification logic:
          - VOLATILE if volatility > volatility_threshold (takes priority)
          - BULL/BEAR if mean_return alone exceeds mean_return_threshold
          - BULL/BEAR if BOTH cumulative_return AND ma_slope exceed their thresholds
            (handles the slow-bleed case: each step is tiny but net drift is large)
          - SIDEWAYS otherwise

        If regime == SIDEWAYS and long_prices is provided with >= 100 points,
        the long window is checked for ultra-slow drift (DEC-REGIME-003):
          - long_cumulative + long_ma_slope both bearish -> override to BEAR (confidence 0.7)
          - long_cumulative + long_ma_slope both bullish -> override to BULL (confidence 0.7)

        Args:
            prices: Short-window price list (used for primary classification)
            long_prices: Optional long-window price list (used only if short window -> SIDEWAYS)
        """
        prices_float = [float(p) for p in prices]
        returns = np.diff(prices_float) / prices_float[:-1]

        # Core metrics
        volatility = float(np.std(returns))
        mean_return = float(np.mean(returns))

        # Cumulative return: captures total drift over window
        cumulative = (prices_float[-1] - prices_float[0]) / prices_float[0]

        # MA slope: slope of short-term SMA, normalized by price
        ma_period = self._ma_period
        if len(prices_float) >= ma_period:
            sma = np.convolve(prices_float, np.ones(ma_period) / ma_period, mode='valid')
            if len(sma) >= 2:
                ma_slope = (sma[-1] - sma[0]) / (len(sma) * prices_float[0])
            else:
                ma_slope = 0.0
        else:
            ma_slope = 0.0

        # Store metrics for indicator reporting (read by _update_regime)
        self._last_metrics = {
            "volatility": volatility,
            "mean_return": mean_return,
            "cumulative_return": cumulative,
            "ma_slope": ma_slope,
        }

        # Classify regime using short window
        if volatility > self._volatility_threshold:
            regime = "VOLATILE"
        elif mean_return > self._mean_return_threshold:
            regime = "BULL"
        elif mean_return < -self._mean_return_threshold:
            regime = "BEAR"
        elif cumulative > self._cumulative_trend_threshold and ma_slope > self._ma_slope_threshold:
            regime = "BULL"
        elif cumulative < -self._cumulative_trend_threshold and ma_slope < -self._ma_slope_threshold:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        # Long-window override (DEC-REGIME-003): only when short window says SIDEWAYS
        # and we have enough long-window data to be meaningful.
        if regime == "SIDEWAYS" and long_prices is not None and len(long_prices) >= 100:
            long_prices_float = [float(p) for p in long_prices]
            long_cumulative = (
                (long_prices_float[-1] - long_prices_float[0]) / long_prices_float[0]
            )

            # Compute MA slope on long window using the same approach
            if len(long_prices_float) >= ma_period:
                long_sma = np.convolve(
                    long_prices_float, np.ones(ma_period) / ma_period, mode='valid'
                )
                if len(long_sma) >= 2:
                    long_ma_slope = (
                        (long_sma[-1] - long_sma[0]) / (len(long_sma) * long_prices_float[0])
                    )
                else:
                    long_ma_slope = 0.0
            else:
                long_ma_slope = 0.0

            # Override SIDEWAYS only when both long metrics agree on direction
            if long_cumulative < -self._long_cumulative_threshold and long_ma_slope < 0:
                regime = "BEAR"
                self._last_metrics["long_cumulative_return"] = long_cumulative
                self._last_metrics["long_ma_slope"] = long_ma_slope
                return regime, 0.7

            if long_cumulative > self._long_cumulative_threshold and long_ma_slope > 0:
                regime = "BULL"
                self._last_metrics["long_cumulative_return"] = long_cumulative
                self._last_metrics["long_ma_slope"] = long_ma_slope
                return regime, 0.7

        # Calculate confidence based on metric agreement
        if regime == "VOLATILE":
            confidence = 0.6
        elif regime == "SIDEWAYS":
            # Variable SIDEWAYS confidence based on actual volatility level.
            # Dead-flat (volatility near 0) -> high confidence (0.9): we are
            # very sure it's sideways, so SidewaysSuppressionRule can act.
            # Near-volatile threshold -> low confidence (0.3): borderline case.
            # Formula: confidence = min(0.9, max(0.3, 1.0 - volatility/volatility_threshold))
            # This is a refinement of DEC-INT-003 — the existing decision captures
            # the regime detection approach; this refines the confidence derivation.
            sideways_conf = 1.0 - (volatility / self._volatility_threshold)
            confidence = min(0.9, max(0.3, sideways_conf))
        else:
            direction = 1 if regime == "BULL" else -1
            agreements = 0
            if (mean_return * direction) > 0:
                agreements += 1
            if (cumulative * direction) > 0:
                agreements += 1
            if (ma_slope * direction) > 0:
                agreements += 1

            if agreements >= 3:
                confidence = 0.9
            elif agreements >= 2:
                confidence = 0.7
            else:
                confidence = 0.5

        return regime, confidence

    def _detect_regime_hmm(self, prices: list[Decimal]) -> tuple[str, float]:
        """HMM-based regime detection.

        Returns (regime, confidence). HMM does not compute per-classification
        confidence yet, so 0.7 is returned as a fixed placeholder.
        Falls back to _detect_regime_rules on any error.
        """
        try:
            from hmmlearn.hmm import GaussianHMM

            prices_float = np.array([float(p) for p in prices])
            returns = np.diff(prices_float) / prices_float[:-1]

            # Features: returns and volatility
            features = np.column_stack([
                returns,
                np.abs(returns),  # Volatility proxy
            ])

            # Train HMM with 4 states
            model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
            model.fit(features)

            # Predict current state
            states = model.predict(features)
            current_state = states[-1]

            # Map state to regime based on characteristics
            state_means = model.means_[current_state]
            mean_return = state_means[0]
            mean_vol = state_means[1]

            # Classify using same thresholds as rule-based for consistency
            if mean_vol > self._volatility_threshold:
                regime = "VOLATILE"
            elif mean_return > self._mean_return_threshold:
                regime = "BULL"
            elif mean_return < -self._mean_return_threshold:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"

            # Populate _last_metrics for indicator reporting
            self._last_metrics = {
                "volatility": float(mean_vol),
                "mean_return": float(mean_return),
                "cumulative_return": 0.0,  # Not computed by HMM path
                "ma_slope": 0.0,
            }

            return regime, 0.7  # HMM doesn't compute confidence yet

        except Exception as e:
            self._log.error("hmm_detection_failed", error=str(e))
            return self._detect_regime_rules(prices)
