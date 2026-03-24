"""
Support and resistance level detection signal generator.

Identifies key price levels where price has historically bounced (support) or
been rejected (resistance), then generates signals when price approaches these
levels. Used by mean reversion strategy for bounce trades and by breakout
strategy for breakout confirmation.

@decision DEC-SIGNAL-006
@title Pivot-based S/R detection with proximity signals
@status accepted
@rationale Simple pivot point detection (local highs/lows over N candles) is
robust and interpretable. More complex methods (order flow, volume profile)
require data we don't have from Kraken WebSocket. Pivot-based S/R detects
levels that matter to other traders using the same technique, creating
self-fulfilling prophecy effect. Proximity threshold (0.3%) filters noise
while catching actionable bounces.
"""

from collections import defaultdict
from decimal import Decimal
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import SignalAction, SignalType, Symbol

from .base import SignalGenerator
from .candles import Candle, CandleAggregator

logger = structlog.get_logger()


class SupportResistanceSignal(SignalGenerator):
    """
    Support/resistance level detection and proximity signal generator.

    Detects S/R levels using pivot points (local highs/lows) from candle data,
    then generates BUY signals near support and SELL signals near resistance.

    Algorithm:
    1. Scan completed candles for pivot highs/lows using a lookback window.
    2. Cluster nearby pivots into S/R levels (within cluster_threshold_pct).
    3. Score levels by touch count (more touches = stronger level).
    4. When current price is within proximity_pct of a level, emit a signal
       with strength proportional to the level's touch count.

    Attributes:
        _candle_agg: Candle aggregator for OHLCV data.
        _pivot_lookback: Number of candles on each side to confirm a pivot.
        _min_touches: Minimum touches for a level to be considered valid.
        _proximity_pct: How close price must be to S/R level to trigger signal.
        _cluster_threshold_pct: Distance within which pivots are merged into
            a single S/R level.
        _max_levels: Maximum S/R levels tracked per symbol.
        _levels: Cached S/R levels per symbol: list of (price, touch_count).
    """

    def __init__(
        self,
        bus: EventBus,
        candle_agg: CandleAggregator,
        pivot_lookback: int = 5,
        min_touches: int = 2,
        proximity_pct: float = 0.3,
        cluster_threshold_pct: float = 0.2,
        max_levels: int = 10,
    ) -> None:
        """
        Initialize support/resistance signal generator.

        Args:
            bus: Event bus for publishing/subscribing.
            candle_agg: Candle aggregator for OHLCV data.
            pivot_lookback: Candles on each side to confirm a pivot point.
            min_touches: Minimum touch count for a valid S/R level.
            proximity_pct: Price distance (%) to S/R level that triggers signal.
            cluster_threshold_pct: Distance (%) within which pivots merge.
            max_levels: Maximum number of S/R levels to track per symbol.
        """
        # Need enough candles for pivot detection + clustering
        window_size = (pivot_lookback * 2 + 1) * 3 + 50
        super().__init__(
            bus,
            SignalType.TECHNICAL,
            window_size=window_size,
            name="SupportResistance",
        )
        self._candle_agg = candle_agg
        self._pivot_lookback = pivot_lookback
        self._min_touches = min_touches
        self._proximity_pct = proximity_pct
        self._cluster_threshold_pct = cluster_threshold_pct
        self._max_levels = max_levels
        self._levels: dict[Symbol, list[tuple[Decimal, int]]] = defaultdict(list)
        self._log = logger.bind(component="signal_sr")

    def _get_min_periods(self) -> int:
        """Need at least enough data for pivot detection."""
        return self._pivot_lookback * 2 + 1

    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """
        Generate signal based on proximity to support/resistance levels.

        Recalculates S/R levels from candle data, then checks if the current
        price is near any level. Near support = BUY, near resistance = SELL.
        """
        candles = self._candle_agg.get_candles(
            symbol, count=self._pivot_lookback * 10 + 20
        )

        if len(candles) < self._pivot_lookback * 2 + 1:
            return None

        # Detect pivot points
        pivots = self._detect_pivots(candles)

        if not pivots:
            return None

        # Cluster pivots into S/R levels
        levels = self._cluster_pivots(pivots)

        # Filter by minimum touches
        levels = [(price, count) for price, count in levels if count >= self._min_touches]

        if not levels:
            return None

        # Sort by touch count (strongest first) and limit
        levels.sort(key=lambda x: x[1], reverse=True)
        levels = levels[: self._max_levels]

        # Cache for external inspection
        self._levels[symbol] = levels

        # Get current price
        current_price = data[-1].price

        # Find nearest S/R level and determine signal
        return self._check_proximity(symbol, current_price, levels, data[-1].timestamp)

    def _detect_pivots(self, candles: list[Candle]) -> list[tuple[Decimal, str]]:
        """
        Detect pivot highs and lows from candle data.

        A pivot high is a candle whose high is higher than the highs of
        `pivot_lookback` candles on each side. Similarly for pivot lows.

        Args:
            candles: List of candles (oldest to newest).

        Returns:
            List of (price, type) tuples where type is "high" or "low".
        """
        pivots: list[tuple[Decimal, str]] = []
        lookback = self._pivot_lookback

        for i in range(lookback, len(candles) - lookback):
            candle = candles[i]

            # Check for pivot high
            is_pivot_high = all(
                candle.high >= candles[j].high
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )

            # Check for pivot low
            is_pivot_low = all(
                candle.low <= candles[j].low
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )

            if is_pivot_high:
                pivots.append((candle.high, "high"))
            if is_pivot_low:
                pivots.append((candle.low, "low"))

        return pivots

    def _cluster_pivots(
        self, pivots: list[tuple[Decimal, str]]
    ) -> list[tuple[Decimal, int]]:
        """
        Cluster nearby pivot points into S/R levels.

        Pivots within cluster_threshold_pct of each other are merged into a
        single level. The level price is the average of the cluster, and the
        touch count is the number of pivots in the cluster.

        Args:
            pivots: List of (price, type) from pivot detection.

        Returns:
            List of (average_price, touch_count) for each S/R level.
        """
        if not pivots:
            return []

        # Sort pivots by price
        sorted_prices = sorted([p[0] for p in pivots])

        clusters: list[list[Decimal]] = []
        current_cluster: list[Decimal] = [sorted_prices[0]]

        for price in sorted_prices[1:]:
            cluster_avg = sum(current_cluster) / len(current_cluster)

            # Check if this price is close enough to the current cluster
            if cluster_avg > 0:
                distance_pct = abs(float(price - cluster_avg) / float(cluster_avg)) * 100

                if distance_pct <= self._cluster_threshold_pct:
                    current_cluster.append(price)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [price]
            else:
                current_cluster.append(price)

        # Don't forget the last cluster
        clusters.append(current_cluster)

        # Convert clusters to (average_price, touch_count)
        levels: list[tuple[Decimal, int]] = []
        for cluster in clusters:
            avg_price = sum(cluster) / len(cluster)
            # Round to 2 decimal places for cleaner levels
            avg_price = Decimal(str(round(float(avg_price), 2)))
            levels.append((avg_price, len(cluster)))

        return levels

    def _check_proximity(
        self,
        symbol: Symbol,
        current_price: Decimal,
        levels: list[tuple[Decimal, int]],
        timestamp: float,
    ) -> SignalEvent | None:
        """
        Check if current price is near any S/R level and generate signal.

        Near support (price slightly above level) = BUY signal.
        Near resistance (price slightly below level) = SELL signal.
        Strength scales with the level's touch count.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            levels: List of (price, touch_count) S/R levels.
            timestamp: Event timestamp.

        Returns:
            SignalEvent if near an S/R level, None otherwise.
        """
        best_signal: SignalEvent | None = None
        best_strength = Decimal("0")

        for level_price, touch_count in levels:
            if level_price == 0:
                continue

            distance_pct = float(
                abs(current_price - level_price) / level_price
            ) * 100

            if distance_pct > self._proximity_pct:
                continue

            # Price is near this level
            # Strength: base 0.4 + 0.1 per touch (capped at 1.0)
            # More touches = stronger level = higher confidence signal
            raw_strength = 0.4 + (touch_count - self._min_touches) * 0.1
            strength = Decimal(str(min(raw_strength, 1.0)))

            # Proximity bonus: closer to level = stronger signal
            # At level: 1.0x, at proximity edge: 0.6x
            proximity_factor = Decimal(str(1.0 - (distance_pct / self._proximity_pct) * 0.4))
            strength = strength * proximity_factor

            if strength <= best_strength:
                continue

            # Determine direction based on price relative to level
            if current_price <= level_price:
                # Price at or below level = potential support bounce = BUY
                action = SignalAction.BUY
                reason = f"Near support {level_price} ({touch_count} touches, {distance_pct:.2f}% away)"
            else:
                # Price at or above level = potential resistance rejection = SELL
                action = SignalAction.SELL
                reason = f"Near resistance {level_price} ({touch_count} touches, {distance_pct:.2f}% away)"

            # Confidence based on touch count (more touches = more reliable)
            confidence = Decimal(str(min(0.5 + touch_count * 0.05, 0.85)))

            best_signal = self._create_signal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                timestamp=timestamp,
                reason=reason,
            )
            best_strength = strength

        return best_signal

    def get_levels(self, symbol: Symbol) -> list[tuple[Decimal, int]]:
        """
        Get cached S/R levels for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            List of (price, touch_count) tuples, strongest first.
        """
        return list(self._levels.get(symbol, []))
