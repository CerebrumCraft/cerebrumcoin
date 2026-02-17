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
    """

    def __init__(
        self,
        bus: EventBus,
        window_size: int = 100,
        update_interval: int = 20,  # Check regime every N market data events
        use_hmm: bool = False,
    ) -> None:
        """
        Initialize regime detector.

        Args:
            bus: Event bus
            window_size: Number of price points for regime calculation
            update_interval: Update regime every N market data events
            use_hmm: Use HMM if hmmlearn available (else rule-based)
        """
        self._bus = bus
        self._window_size = window_size
        self._update_interval = update_interval
        self._use_hmm = use_hmm

        # Per-symbol price history
        self._price_history: dict[Symbol, Deque[Decimal]] = {}
        self._event_counts: dict[Symbol, int] = {}

        # Current regime per symbol
        self._current_regime: dict[Symbol, str] = {}

        self._log = logger.bind(component="regime_detector")

        # Check if HMM is available
        self._hmm_available = False
        if use_hmm:
            try:
                from hmmlearn.hmm import GaussianHMM
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
            self._event_counts[symbol] = 0
            self._current_regime[symbol] = "UNKNOWN"

        # Add price to history
        self._price_history[symbol].append(price)
        self._event_counts[symbol] += 1

        # Check if it's time to update regime
        if len(self._price_history[symbol]) < 30:  # Need minimum data
            return

        if self._event_counts[symbol] % self._update_interval == 0:
            await self._update_regime(symbol)

    async def _update_regime(self, symbol: Symbol) -> None:
        """Calculate and update regime for a symbol."""
        prices = list(self._price_history[symbol])
        
        if len(prices) < 30:
            return

        # Detect regime
        if self._hmm_available and self._use_hmm:
            new_regime = self._detect_regime_hmm(prices)
        else:
            new_regime = self._detect_regime_rules(prices)

        # Check for regime change
        old_regime = self._current_regime[symbol]
        if new_regime != old_regime:
            self._current_regime[symbol] = new_regime

            # Emit regime change event
            event = RegimeChangeEvent(
                event_type=EventType.REGIME_CHANGE,
                timestamp=asyncio.get_event_loop().time(),
                from_regime=old_regime,
                to_regime=new_regime,
                confidence=Decimal("0.7"),  # TODO: Calculate actual confidence
                indicators={
                    "symbol": symbol,
                    "price_points": len(prices),
                },
            )

            await self._bus.publish(event)

            self._log.info(
                "regime_change",
                symbol=symbol,
                from_regime=old_regime,
                to_regime=new_regime,
            )

    def _detect_regime_rules(self, prices: list[Decimal]) -> str:
        """Rule-based regime detection (fallback)."""
        prices_float = [float(p) for p in prices]
        returns = np.diff(prices_float) / prices_float[:-1]

        # Calculate metrics
        volatility = float(np.std(returns))
        trend = float(np.mean(returns))

        # Thresholds
        HIGH_VOL = 0.03  # 3% daily volatility
        STRONG_TREND = 0.002  # 0.2% daily trend

        # Classify
        if volatility > HIGH_VOL:
            return "VOLATILE"
        elif trend > STRONG_TREND:
            return "BULL"
        elif trend < -STRONG_TREND:
            return "BEAR"
        else:
            return "SIDEWAYS"

    def _detect_regime_hmm(self, prices: list[Decimal]) -> str:
        """HMM-based regime detection."""
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

            # Classify
            HIGH_VOL = 0.03
            STRONG_TREND = 0.002

            if mean_vol > HIGH_VOL:
                return "VOLATILE"
            elif mean_return > STRONG_TREND:
                return "BULL"
            elif mean_return < -STRONG_TREND:
                return "BEAR"
            else:
                return "SIDEWAYS"

        except Exception as e:
            self._log.error("hmm_detection_failed", error=str(e))
            return self._detect_regime_rules(prices)
