"""
Signal aggregator for weighted combination of multiple signal sources.

Combines technical, sentiment, and other signals into a unified trading decision
using configurable weights and a threshold-based debounce mechanism.

@decision DEC-AGG-001
@title Signal aggregator with weighted combination and debounce
@status accepted
@rationale Multiple signals produce conflicting recommendations. Weighted voting with
confidence-adjusted strengths produces a unified decision. Debounce threshold (e.g., 0.3)
prevents signal flapping on weak/noisy indicators. Only emits when aggregate exceeds threshold.

@decision DEC-INT-005
@title Regime-aware signal weight adjustment
@status accepted
@rationale Different market regimes favor different strategies. BULL: boost trend-following
(technical). BEAR: boost risk-off (reduce all weights). VOLATILE: reduce confidence and
increase threshold. SIDEWAYS: favor mean-reversion. Regime changes trigger dynamic weight
adjustment for context-aware trading.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
import time as _time_module
from typing import Callable, Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType, Symbol

logger = structlog.get_logger()


@dataclass
class SignalWeight:
    """Weight configuration for a signal type."""
    signal_type: SignalType
    weight: Decimal
    enabled: bool = True


class SignalAggregator:
    """
    Aggregates multiple signals into a unified trading decision.
    
    Features:
    - Weighted combination of signals by type
    - Confidence-adjusted signal strength
    - Threshold-based emission (prevents weak signals)
    - Time-based signal window for aggregation
    - Per-symbol signal tracking
    """
    
    def __init__(
        self,
        bus: EventBus,
        weights: dict[SignalType, Decimal] | None = None,
        threshold: Decimal = Decimal("0.3"),
        window_seconds: int = 5,
        buy_suppression_factor: str = "0.2",
        buy_suppression_min_confidence: str = "0.8",
        strategy_id: str | None = None,
        signal_source_filter: str | None = None,
        signal_timeframe_filter: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize signal aggregator.

        Args:
            bus: Event bus
            weights: Signal type weights (default: equal weight)
            threshold: Minimum aggregate strength to emit signal
            window_seconds: Time window for signal aggregation
            buy_suppression_factor: Multiplier applied to buy score in high-confidence BEAR regime
            buy_suppression_min_confidence: Minimum regime confidence to trigger buy suppression
            strategy_id: Optional strategy identifier. When set, emitted COMBINED
                         SignalEvents are tagged with this strategy_id so the matching
                         RiskManager can filter signals to only its own pipeline
                         (DEC-STRAT-005). None preserves backward-compatible behaviour.
            signal_source_filter: When set, only signals whose metadata["source"]
                         matches this string are admitted to the buffer. All other
                         signals are silently dropped before aggregation. None
                         disables filtering (default, backward-compatible).
            signal_timeframe_filter: When set, only signals whose metadata["timeframe"]
                         matches this string are admitted to the buffer. Signals
                         from generators with a different timeframe are silently
                         dropped. None disables filtering (default, backward-compatible).
            clock: Callable returning current time as float (Unix epoch seconds).
                   Default: time.time (wall-clock). Inject a BacktestClock instance
                   in backtest mode so signal expiry uses simulated historical time
                   instead of wall-clock time. This prevents all historical signals
                   from being instantly expired when backtesting against past data.
                   (DEC-BACKTEST-004)
        """
        self._bus = bus
        self._threshold = threshold
        self._window_seconds = window_seconds
        # Injectable clock: default to wall-clock time.time for live trading.
        # Backtest injects a BacktestClock that tracks the latest candle timestamp.
        # @decision DEC-BACKTEST-004: virtual clock injection for backtest mode.
        self._clock: Callable[[], float] = clock if clock is not None else _time_module.time
        self._regime_confidence: Decimal = Decimal("0.0")
        self._buy_suppression_factor = Decimal(buy_suppression_factor)
        self._buy_suppression_min_confidence = Decimal(buy_suppression_min_confidence)
        # strategy_id is stored and stamped onto every COMBINED signal emitted.
        # DEC-STRAT-005: None means no filtering — backward compatible.
        self._strategy_id = strategy_id

        # Source filter: when set, only signals from the named generator are accepted.
        # DEC-SIGNAL-002: metadata["source"] is injected by SignalGenerator._create_signal().
        self._signal_source_filter = signal_source_filter

        # Timeframe filter: when set, only signals with matching metadata["timeframe"] accepted.
        # DEC-SIGNAL-003: metadata["timeframe"] is injected by SignalGenerator._create_signal().
        self._signal_timeframe_filter = signal_timeframe_filter
        
        # Default weights: technical signals weighted higher
        self._weights: dict[SignalType, Decimal] = weights or {
            SignalType.TECHNICAL: Decimal("1.0"),
            SignalType.SENTIMENT: Decimal("0.5"),
            SignalType.NEWS: Decimal("0.3"),
            SignalType.REGIME: Decimal("0.7"),
        }
        
        # Per-symbol signal buffer: tracks recent signals in time window
        self._signal_buffer: dict[Symbol, Deque[SignalEvent]] = defaultdict(
            lambda: deque(maxlen=50)
        )
        
        # Track last emission time per symbol (for debounce)
        self._last_emission: dict[Symbol, float] = {}

        # Current market regime
        self._current_regime: str = "UNKNOWN"

        # Base weights (stored for regime adjustments)
        self._base_weights = self._weights.copy()

        # Per-regime learned weights (updated by WeightAdapter)
        self._regime_weights: dict[str, dict[SignalType, Decimal]] = {}

        # Bind log with strategy context if present
        self._log = logger.bind(
            component="signal_aggregator",
            **({"strategy_id": strategy_id} if strategy_id else {}),
        )

        # Use strategy-scoped subscriber names so multiple aggregators on the
        # same bus don't collide. Backward compat: legacy names when no strategy_id.
        signal_sub_name = (
            f"signal_aggregator_{strategy_id}" if strategy_id else "signal_aggregator"
        )
        regime_sub_name = (
            f"signal_aggregator_regime_{strategy_id}"
            if strategy_id
            else "signal_aggregator_regime"
        )

        # Subscribe to all signal types
        bus.subscribe(
            EventType.SIGNAL,
            self._on_signal,
            subscriber_name=signal_sub_name,
        )

        # Subscribe to regime changes
        bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            subscriber_name=regime_sub_name,
        )

        self._log.info(
            "signal_aggregator_initialized",
            weights={k.value: str(v) for k, v in self._weights.items()},
            threshold=str(threshold),
            window_seconds=window_seconds,
            signal_source_filter=signal_source_filter,
            signal_timeframe_filter=signal_timeframe_filter,
        )
    
    async def _on_signal(self, event: Event) -> None:
        """Handle incoming signals and aggregate."""
        if not isinstance(event, SignalEvent):
            return

        # CRITICAL: Ignore our own combined signals to prevent feedback loop
        if event.signal_type == SignalType.COMBINED:
            return

        # Filter by signal source if configured (e.g., range_trading only wants S/R signals)
        if self._signal_source_filter:
            source = event.metadata.get("source") if event.metadata else None
            if source != self._signal_source_filter:
                return

        # Filter by timeframe if configured (e.g., swing strategy only wants 1h signals)
        if self._signal_timeframe_filter:
            tf = event.metadata.get("timeframe") if event.metadata else None
            if tf != self._signal_timeframe_filter:
                return

        symbol = event.symbol
        current_time = self._clock()

        # Add signal to buffer
        self._signal_buffer[symbol].append(event)
        
        # Clean old signals outside time window
        self._clean_old_signals(symbol, current_time)
        
        # Aggregate signals
        aggregate = self._aggregate_signals(symbol, current_time)
        
        if aggregate is None:
            return
        
        # Check if aggregate exceeds threshold
        if aggregate.strength >= self._threshold:
            # Debounce: don't emit if we recently emitted for this symbol
            last_emit = self._last_emission.get(symbol, 0)
            if current_time - last_emit < self._window_seconds:
                return

            # Emit combined signal
            await self._bus.publish(aggregate)
            self._last_emission[symbol] = current_time

            self._log.info(
                "combined_signal_emitted",
                symbol=symbol,
                action=aggregate.action.value,
                strength=str(aggregate.strength),
                confidence=str(aggregate.confidence),
                contributing_signals=len(self._signal_buffer[symbol]),
            )
    
    def _clean_old_signals(self, symbol: Symbol, current_time: float) -> None:
        """Remove signals outside the aggregation window."""
        buffer = self._signal_buffer[symbol]
        cutoff_time = current_time - self._window_seconds
        
        # Remove signals older than window
        while buffer and buffer[0].timestamp < cutoff_time:
            buffer.popleft()
    
    def _aggregate_signals(
        self,
        symbol: Symbol,
        current_time: float,
    ) -> SignalEvent | None:
        """
        Aggregate signals within the time window.
        
        Uses weighted voting: each signal contributes its strength * weight * confidence.
        """
        signals = self._signal_buffer[symbol]
        
        if not signals:
            return None
        
        # Separate by action
        buy_score = Decimal("0.0")
        sell_score = Decimal("0.0")
        buy_weight_sum = Decimal("0.0")
        sell_weight_sum = Decimal("0.0")
        confidence_sum = Decimal("0.0")

        for signal in signals:
            # Get weight for this signal type
            weight = self._weights.get(signal.signal_type, Decimal("0.5"))

            # Weighted strength contribution (no confidence dilution)
            contribution = signal.strength * weight

            if signal.action == SignalAction.BUY:
                buy_score += contribution
                buy_weight_sum += weight
            elif signal.action == SignalAction.SELL:
                sell_score += contribution
                sell_weight_sum += weight
            # HOLD and CLOSE don't contribute to directional score

            confidence_sum += signal.confidence

        if buy_weight_sum == 0 and sell_weight_sum == 0:
            return None

        # Normalize by sum of weights for that action (weighted average)
        # Multiple agreeing signals with same weight reinforce to the average strength
        buy_score_norm = buy_score / buy_weight_sum if buy_weight_sum > 0 else Decimal("0.0")
        sell_score_norm = sell_score / sell_weight_sum if sell_weight_sum > 0 else Decimal("0.0")

        # Consensus multiplier: reward agreement across signal generators.
        # When all weight is on one side, multiplier = sqrt(1.0) = 1.0 (no change).
        # When weight is split 50/50, each side gets sqrt(0.5) ≈ 0.71, making it
        # harder for either direction to cross the threshold.
        # sqrt gives diminishing returns so even a 75% majority still gets 0.87x.
        #
        # @decision DEC-AGG-002
        # @title Consensus multiplier via sqrt(buy_weight_fraction)
        # @status accepted
        # @rationale 4 weak signals all agreeing at 0.3 should outscore a 50/50 split.
        # Old normalization (divide by directional weight sum) made 4x0.3 BUY = 0.3
        # regardless of opposing signals. Now a split suppresses both directions.
        import math as _math
        total_weight = buy_weight_sum + sell_weight_sum
        if total_weight > Decimal("0"):
            buy_consensus = buy_weight_sum / total_weight
            sell_consensus = sell_weight_sum / total_weight
            buy_score_norm = buy_score_norm * Decimal(str(_math.sqrt(float(buy_consensus))))
            sell_score_norm = sell_score_norm * Decimal(str(_math.sqrt(float(sell_consensus))))
        
        # @decision DEC-REGIME-002: Suppress buy signals in high-confidence BEAR regime.
        # When regime detector is confident we're in a downtrend (confidence >= 0.8),
        # multiply buy score by 0.2 to prevent buying into falling markets.
        # This addresses the paper-trading session where 0/20 trades won because
        # the BEAR regime was misclassified as SIDEWAYS and buy signals fired freely.
        if self._current_regime == "BEAR" and self._regime_confidence >= self._buy_suppression_min_confidence:
            buy_score_norm *= self._buy_suppression_factor

        # Determine aggregate action and strength
        if buy_score_norm > sell_score_norm:
            action = SignalAction.BUY
            strength = buy_score_norm
        elif sell_score_norm > buy_score_norm:
            action = SignalAction.SELL
            strength = sell_score_norm
        else:
            action = SignalAction.HOLD
            strength = Decimal("0.0")
        
        # Average confidence
        avg_confidence = confidence_sum / len(signals) if signals else Decimal("0.5")
        
        # Clamp values
        strength = max(Decimal("0.0"), min(Decimal("1.0"), strength))
        avg_confidence = max(Decimal("0.0"), min(Decimal("1.0"), avg_confidence))
        
        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=current_time,
            signal_type=SignalType.COMBINED,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=avg_confidence,
            reason=f"Aggregated {len(signals)} signals: buy={buy_score_norm:.2f}, sell={sell_score_norm:.2f}",
            strategy_id=self._strategy_id,  # DEC-STRAT-005: tag for RiskManager routing
        )
    
    def get_signal_count(self, symbol: Symbol) -> int:
        """Get number of signals in buffer for a symbol."""
        return len(self._signal_buffer[symbol])
    
    def set_weight(self, signal_type: SignalType, weight: Decimal) -> None:
        """Update weight for a signal type."""
        self._weights[signal_type] = weight
        self._log.info("weight_updated", signal_type=signal_type.value, weight=str(weight))

    def set_regime_weight(self, signal_type: SignalType, regime: str, weight: Decimal) -> None:
        """Update learned weight for a signal type in a specific regime."""
        if regime not in self._regime_weights:
            self._regime_weights[regime] = {}
        self._regime_weights[regime][signal_type] = weight

        # Apply immediately if this is the current regime
        if regime == self._current_regime:
            self._weights[signal_type] = weight
            self._log.info(
                "regime_weight_updated",
                signal_type=signal_type.value,
                regime=regime,
                weight=str(weight),
            )

    async def _on_regime_change(self, event: Event) -> None:
        """Handle regime changes and adjust weights."""
        if not isinstance(event, RegimeChangeEvent):
            return

        self._current_regime = event.to_regime
        self._regime_confidence = event.confidence

        # Use learned weights if available for this regime
        if event.to_regime in self._regime_weights:
            for signal_type, weight in self._regime_weights[event.to_regime].items():
                self._weights[signal_type] = weight
            self._log.info("applied_learned_weights", regime=event.to_regime)
            return

        # Otherwise, apply rule-based regime adjustments
        if event.to_regime == "BULL":
            # Boost trend-following (technical signals)
            self._weights[SignalType.TECHNICAL] = self._base_weights[SignalType.TECHNICAL] * Decimal("1.2")
            self._weights[SignalType.SENTIMENT] = self._base_weights[SignalType.SENTIMENT] * Decimal("0.8")
            self._weights[SignalType.NEWS] = self._base_weights[SignalType.NEWS] * Decimal("0.9")

        elif event.to_regime == "BEAR":
            # Boost risk-off, reduce all signals
            self._weights[SignalType.TECHNICAL] = self._base_weights[SignalType.TECHNICAL] * Decimal("0.8")
            self._weights[SignalType.SENTIMENT] = self._base_weights[SignalType.SENTIMENT] * Decimal("1.2")
            self._weights[SignalType.NEWS] = self._base_weights[SignalType.NEWS] * Decimal("1.1")

        elif event.to_regime == "VOLATILE":
            # Reduce all weights, increase threshold (done via confidence reduction in aggregate)
            self._weights[SignalType.TECHNICAL] = self._base_weights[SignalType.TECHNICAL] * Decimal("0.7")
            self._weights[SignalType.SENTIMENT] = self._base_weights[SignalType.SENTIMENT] * Decimal("0.6")
            self._weights[SignalType.NEWS] = self._base_weights[SignalType.NEWS] * Decimal("0.5")

        elif event.to_regime == "SIDEWAYS":
            # Favor mean-reversion (reduce trend-following)
            self._weights[SignalType.TECHNICAL] = self._base_weights[SignalType.TECHNICAL] * Decimal("0.9")
            # @decision DEC-SENT-001: Reduce sentiment weight in non-trending regimes.
            # SIDEWAYS: 0.4x multiplier (effective 0.2) — sentiment is noise in ranging markets.
            # UNKNOWN: 0.6x multiplier (effective 0.3) — partial dampening when regime unclear.
            # Prevents Fear&Greed index from dominating signal aggregation.
            self._weights[SignalType.SENTIMENT] = self._base_weights[SignalType.SENTIMENT] * Decimal("0.4")
            self._weights[SignalType.NEWS] = self._base_weights[SignalType.NEWS] * Decimal("1.0")

        else:
            # Unknown regime: use base weights with dampened sentiment
            # @decision DEC-SENT-001 (continued): 0.6x multiplier in UNKNOWN regime prevents
            # sentiment dominance when the regime classifier lacks confidence.
            self._weights = self._base_weights.copy()
            self._weights[SignalType.SENTIMENT] = self._base_weights[SignalType.SENTIMENT] * Decimal("0.6")

        self._log.info(
            "regime_weights_adjusted",
            regime=event.to_regime,
            weights={k.value: str(v) for k, v in self._weights.items()},
        )
