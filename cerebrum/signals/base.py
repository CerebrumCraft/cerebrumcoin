"""
Base signal generator interface for CerebrumCoin.

All signal generators inherit from SignalGenerator and implement generate_signal().
The base class handles event bus subscription and data accumulation.

@decision DEC-SIGNAL-001
@title Abstract signal generator with automatic data accumulation
@status accepted
@rationale All technical signals need historical price data (e.g., RSI needs 14+ periods).
Base class accumulates MarketDataEvents per symbol, maintaining a sliding window.
Subclasses implement generate_signal() which receives the accumulated data.
This enables clean separation: data management in base, indicator logic in subclasses.

@decision DEC-SIGNAL-002
@title Source metadata injected by _create_signal
@status accepted
@rationale Range trading and other strategy aggregators need to filter incoming
signals by originating generator (e.g., accept only SupportResistance signals).
Injecting metadata["source"] = self._name in _create_signal() provides this at
the single common creation point, so every subclass gets it for free.

@decision DEC-SIGNAL-003
@title Timeframe metadata injected by _create_signal
@status accepted
@rationale Multi-timeframe swing trading strategies need to distinguish signals
produced from different bar sizes (e.g., 1-minute scalp signals vs 1-hour swing
signals). Injecting metadata["timeframe"] = self._timeframe in _create_signal()
at the base level means every subclass carries timeframe context for free.
The default "1m" preserves backward compatibility with all existing generators
that do not specify a timeframe.

@decision DEC-DIAG-003
@title Outlier-rejection filter in SignalGenerator for bimodal price feeds
@status accepted
@rationale Session 43 confirmed a bimodal NVDA price distribution (two clusters:
~$103 and ~$210, 2:1 ratio, 14,543 ticks). The anomaly was actively producing buy
signals on spurious $210 prices. The filter uses a rolling median (robust to the
outliers themselves) over the last `outlier_window` ticks. Any tick deviating more
than `outlier_deviation_threshold` (default 50%) from the median is rejected before
entering _data. Cold-start guard (outlier_min_ticks = window//2) prevents false
rejects during accumulation. Both window and threshold are configurable kwargs so
individual generators can tune the filter for their symbol's typical volatility.
All rejections are logged at WARNING level and counted in outlier_rejected_counts.
"""

import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from decimal import Decimal
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, MarketDataEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType, Symbol

logger = structlog.get_logger()


class SignalGenerator(ABC):
    """
    Abstract base class for all signal generators.

    Features:
    - Automatic MarketDataEvent subscription
    - Per-symbol data accumulation with configurable window size
    - Abstract generate_signal() for subclass implementation
    - Automatic SignalEvent emission
    """

    def __init__(
        self,
        bus: EventBus,
        signal_type: SignalType,
        window_size: int = 100,
        name: str | None = None,
        timeframe: str = "1m",
        outlier_window: int = 20,
        outlier_deviation_threshold: float = 0.5,
        outlier_min_ticks: int | None = None,
    ) -> None:
        """
        Initialize signal generator.

        Args:
            bus: Event bus for publishing/subscribing
            signal_type: Type of signal this generator produces
            window_size: Number of market data points to retain per symbol
            name: Human-readable name for logging
            timeframe: Bar timeframe this generator operates on (e.g. "1m", "1h",
                       "4h", "1d"). Stamped into every signal's metadata so
                       downstream aggregators can filter by timeframe.
                       Default "1m" preserves backward compatibility.
            outlier_window: Number of recent ticks used to compute the rolling
                median for outlier detection (DEC-DIAG-003). Default 20.
            outlier_deviation_threshold: Fractional deviation from rolling median
                that triggers rejection. Default 0.5 (50%). A tick with price P
                is rejected when abs(P - median) / median > threshold.
            outlier_min_ticks: Minimum accumulated ticks per symbol before
                outlier detection activates. During cold-start, all ticks are
                accepted regardless of deviation. Default: outlier_window // 2.
        """
        self._bus = bus
        self._signal_type = signal_type
        self._window_size = window_size
        self._name = name or self.__class__.__name__
        self._timeframe = timeframe

        # Per-symbol data accumulation: symbol -> deque of MarketDataEvents
        self._data: dict[Symbol, Deque[MarketDataEvent]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # DEC-DIAG-003: outlier-rejection filter state
        # _outlier_price_window: rolling deque of recent accepted prices per symbol
        # used to compute the median reference for deviation checks.
        self._outlier_window: int = outlier_window
        self._outlier_deviation_threshold: float = outlier_deviation_threshold
        self._outlier_min_ticks: int = (
            outlier_min_ticks
            if outlier_min_ticks is not None
            else max(1, outlier_window // 2)
        )
        self._outlier_price_window: dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=outlier_window)
        )
        # Per-symbol rejection counters — incremented each time a tick is dropped.
        self._outlier_rejected_counts: dict[Symbol, int] = {}

        self._log = logger.bind(component=f"signal_{self._name}")

        # Subscribe to market data
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name=self._name,
        )

        self._log.info(
            "signal_generator_initialized",
            window_size=window_size,
            outlier_window=outlier_window,
            outlier_deviation_threshold=outlier_deviation_threshold,
            outlier_min_ticks=self._outlier_min_ticks,
        )

    def _is_outlier_tick(self, symbol: Symbol, price: float) -> bool:
        """Return True if this tick's price is an outlier vs the rolling median.

        Uses the rolling median of the last outlier_window accepted prices.
        During cold-start (fewer than outlier_min_ticks accumulated for this
        symbol), always returns False so all ticks are accepted.

        DEC-DIAG-003: median is used (not mean) because the mean is polluted by
        the very outliers we want to detect. For the NVDA bimodal case, median
        of 20 ~$103 ticks = ~$103; a $210 tick deviates 103% > 50% threshold.
        """
        price_window = self._outlier_price_window[symbol]
        if len(price_window) < self._outlier_min_ticks:
            return False  # cold-start: accept all ticks

        median = statistics.median(price_window)
        if median == 0.0:
            return False  # avoid division by zero for zero-price edge case

        deviation = abs(price - median) / abs(median)
        return deviation > self._outlier_deviation_threshold

    async def _on_market_data(self, event: Event) -> None:
        """
        Handle incoming market data events.

        Accumulates data and triggers signal generation when ready.
        Outlier ticks (DEC-DIAG-003) are rejected before entering _data so
        that RSI, Bollinger Band, and other indicator calculations are not
        polluted by bimodal price feeds (e.g., NVDA split-adjusted anomaly
        in Session 43: two clusters at ~$103 and ~$210).
        """
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = float(event.price)

        # DEC-DIAG-003: outlier filter — reject before accumulation
        if self._is_outlier_tick(symbol, price):
            self._outlier_rejected_counts[symbol] = (
                self._outlier_rejected_counts.get(symbol, 0) + 1
            )
            self._log.warning(
                "outlier_tick_rejected",
                symbol=symbol,
                price=price,
                rejection_count=self._outlier_rejected_counts[symbol],
                outlier_window=self._outlier_window,
                threshold=self._outlier_deviation_threshold,
            )
            return  # do not add to _data or update price window

        # Accepted tick: update the rolling price window used for median
        self._outlier_price_window[symbol].append(price)

        self._data[symbol].append(event)

        # Only generate signal if we have enough data
        if len(self._data[symbol]) >= self._get_min_periods():
            signal = self._generate_signal(symbol, list(self._data[symbol]))

            if signal is not None:
                await self._bus.publish(signal)

                self._log.debug(
                    "signal_generated",
                    symbol=symbol,
                    action=signal.action.value,
                    strength=str(signal.strength),
                    confidence=str(signal.confidence),
                )

    @property
    def outlier_rejected_counts(self) -> dict[str, int]:
        """Return a copy of per-symbol outlier rejection counters.

        Keys are symbol strings (e.g. "NVDA"). Values are the number of ticks
        rejected as outliers since this generator was created (DEC-DIAG-003).
        Returns a copy so callers cannot mutate the live dict.
        """
        return dict(self._outlier_rejected_counts)

    @abstractmethod
    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """
        Generate a signal from accumulated market data.

        Subclasses implement their indicator logic here.

        Args:
            symbol: Trading symbol
            data: List of market data events (oldest to newest)

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        pass

    @abstractmethod
    def _get_min_periods(self) -> int:
        """
        Get minimum number of periods required for this signal.

        Returns:
            Minimum data points needed before generating signals
        """
        pass

    def _create_signal(
        self,
        symbol: Symbol,
        action: SignalAction,
        strength: Decimal,
        confidence: Decimal,
        timestamp: float,
        reason: str | None = None,
    ) -> SignalEvent:
        """
        Create a SignalEvent with this generator's type.

        Args:
            symbol: Trading symbol
            action: Signal action (BUY/SELL/HOLD)
            strength: Signal strength (0.0 to 1.0)
            confidence: Confidence level (0.0 to 1.0)
            timestamp: Event timestamp
            reason: Optional explanation

        Returns:
            SignalEvent ready for publication
        """
        # Clamp strength and confidence to [0.0, 1.0]
        strength = max(Decimal("0.0"), min(Decimal("1.0"), strength))
        confidence = max(Decimal("0.0"), min(Decimal("1.0"), confidence))

        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=timestamp,
            signal_type=self._signal_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            reason=reason,
            metadata={"source": self._name, "timeframe": self._timeframe},
        )

    def get_data_count(self, symbol: Symbol) -> int:
        """Get number of data points accumulated for a symbol."""
        return len(self._data.get(symbol, []))
