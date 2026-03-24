"""
RangeDetector: bounce counting and range validation for SIDEWAYS markets.

Monitors SupportResistance signals and regime state to identify tradeable
price ranges. Counts distinct bounces off support and resistance levels to
confirm a range is "real" (price has repeatedly respected the levels).

A range is considered confirmed when:
- At least `min_bounces` total bounces have been observed across support
  and resistance levels
- The range width (resistance - support) / support * 100 >= min_range_width_pct
- The regime is currently SIDEWAYS (or was when the range was built)
- The range data is not stale (last_updated within staleness window)

@decision DEC-RANGE-001
@title RangeDetector as a queryable state object, not an event emitter
@status accepted
@rationale Range detection requires accumulating bounce evidence over time.
Making it a queryable state object (get_range() returns RangeState | None)
allows the range trading strategy to poll at trade-decision time, rather
than reacting to ephemeral events. This is simpler to test and avoids
publishing half-confirmed ranges that a strategy might act on prematurely.

@decision DEC-RANGE-002
@title Bounce deduplication via proximity zone tracking
@status accepted
@rationale Without deduplication, a burst of S/R signals during a single
price visit to a zone would inflate the bounce count. We track whether price
is currently inside each proximity zone. A bounce is only counted when price
re-enters the zone from outside (transitions from out→in). The 0.5% zone-exit
threshold matches the S/R generator's proximity_pct (0.3%) with slack so
normal market microstructure doesn't cause spurious zone exits/entries.

@decision DEC-RANGE-003
@title Regex-based level extraction from signal.reason string
@status accepted
@rationale The SupportResistanceSignal encodes the actual level price in its
reason string (e.g. "Near support 69500.00 (3 touches, 0.15% away)"). This
is a stable format (DEC-SIGNAL-006). Parsing it avoids coupling RangeDetector
to SupportResistanceSignal internals or requiring a shared data structure.
If the format changes, the regex will fail loudly at parse time, making
the coupling visible.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, Symbol

logger = structlog.get_logger()

# Regex to extract level type ("support" or "resistance") and price from reason string.
# Matches: "Near support 69500.00 (3 touches, 0.15% away)"
#       or "Near resistance 70200.00 (2 touches, 0.10% away)"
_LEVEL_RE = re.compile(
    r"Near (support|resistance)\s+([\d]+(?:\.[\d]+)?)",
    re.IGNORECASE,
)

# Distance threshold (%) at which we consider price to have "left" a proximity zone.
# Must be larger than the S/R generator's proximity_pct (0.3%) to tolerate microstructure.
_ZONE_EXIT_THRESHOLD_PCT = Decimal("0.5")


@dataclass(frozen=True)
class RangeState:
    """
    Immutable snapshot of a detected trading range.

    Attributes:
        support_level: Price floor where bounces have been counted.
        resistance_level: Price ceiling where bounces have been counted.
        bounce_count: Total confirmed bounces (support + resistance combined).
        range_confirmed: True when bounce_count >= min_bounces AND
            range_width_pct >= min_range_width_pct.
        range_width_pct: (resistance - support) / support * 100.
        last_updated: Unix timestamp of most recent bounce observation.
    """

    support_level: Decimal
    resistance_level: Decimal
    bounce_count: int
    range_confirmed: bool
    range_width_pct: Decimal
    last_updated: float


@dataclass
class _SymbolRange:
    """
    Mutable, per-symbol range accumulation state.

    Not exposed externally — `get_range()` converts this to a frozen RangeState.
    """

    support_level: Decimal | None = None
    resistance_level: Decimal | None = None
    support_bounces: int = 0
    resistance_bounces: int = 0
    in_support_proximity: bool = False
    in_resistance_proximity: bool = False
    last_updated: float = 0.0
    last_price: Decimal | None = None


class RangeDetector:
    """
    Detects tradeable price ranges from S/R signal bounce patterns.

    Subscribes to:
    - EventType.SIGNAL: Filters for source=SupportResistance signals to count
      bounces. Only processes in SIDEWAYS regime.
    - EventType.REGIME_CHANGE: Invalidates ranges when leaving SIDEWAYS.
    - EventType.MARKET_DATA: Tracks price to handle zone exits and detect
      breakouts (price beyond support/resistance by breakout_margin_pct).

    Call `get_range(symbol)` to query the current range state for a symbol.
    Returns None if no range data is available, stale, or if a breakout was
    detected.
    """

    def __init__(
        self,
        bus: EventBus,
        min_bounces: int = 3,
        min_range_width_pct: Decimal = Decimal("0.6"),
        breakout_margin_pct: Decimal = Decimal("0.5"),
        level_staleness_minutes: int = 120,
    ) -> None:
        """
        Initialize RangeDetector.

        Args:
            bus: Event bus for subscriptions.
            min_bounces: Total bounces (support + resistance) required to
                confirm a range.
            min_range_width_pct: Minimum width (%) between support and
                resistance for a range to be confirmable.
            breakout_margin_pct: Distance (%) beyond a level that triggers
                a breakout and invalidates the range.
            level_staleness_minutes: Minutes after last bounce update before
                `get_range()` returns None (stale range).
        """
        self._bus = bus
        self._min_bounces = min_bounces
        self._min_range_width_pct = min_range_width_pct
        self._breakout_margin_pct = breakout_margin_pct
        self._staleness_seconds = level_staleness_minutes * 60.0

        self._current_regime: str = "UNKNOWN"
        self._ranges: dict[Symbol, _SymbolRange] = defaultdict(_SymbolRange)
        self._log = logger.bind(component="range_detector")

    async def start(self) -> None:
        """
        Subscribe to relevant event types.

        Must be called after the EventBus has been started.
        """
        self._bus.subscribe(EventType.SIGNAL, self._on_signal, "range_detector_signal")
        self._bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            "range_detector_regime",
        )
        self._bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            "range_detector_market_data",
        )
        self._log.info("range_detector_started")

    async def _on_signal(self, event: SignalEvent) -> None:  # type: ignore[override]
        """
        Process S/R signals to count bounces.

        Only processes SignalEvents with metadata.source == "SupportResistance"
        when the current regime is SIDEWAYS.
        """
        if not isinstance(event, SignalEvent):
            return

        # Gate: only process SupportResistance signals
        if not event.metadata or event.metadata.get("source") != "SupportResistance":
            return

        # Gate: only count bounces in SIDEWAYS regime
        if self._current_regime != "SIDEWAYS":
            return

        # Extract level price from reason string
        if not event.reason:
            return

        match = _LEVEL_RE.search(event.reason)
        if not match:
            self._log.warning(
                "range_detector_parse_fail",
                reason=event.reason,
                symbol=event.symbol,
            )
            return

        level_type = match.group(1).lower()  # "support" or "resistance"
        try:
            level_price = Decimal(match.group(2))
        except InvalidOperation:
            self._log.warning(
                "range_detector_price_parse_fail",
                raw_price=match.group(2),
                symbol=event.symbol,
            )
            return

        state = self._ranges[event.symbol]

        if level_type == "support" and event.action == SignalAction.BUY:
            self._handle_support_signal(event.symbol, state, level_price, event.timestamp)
        elif level_type == "resistance" and event.action == SignalAction.SELL:
            self._handle_resistance_signal(event.symbol, state, level_price, event.timestamp)

    def _handle_support_signal(
        self,
        symbol: Symbol,
        state: _SymbolRange,
        level_price: Decimal,
        timestamp: float,
    ) -> None:
        """
        Process a support proximity signal.

        Updates the tracked support level (or resets if significantly
        different) and counts a bounce only on zone re-entry.
        """
        # If we see a support level significantly different from the tracked one,
        # update the level (the S/R generator may refine levels over time).
        if state.support_level is None:
            state.support_level = level_price
        else:
            # Allow the support level to drift slightly (use the new reading)
            state.support_level = level_price

        if not state.in_support_proximity:
            # Price just entered support zone — count as a bounce
            state.in_support_proximity = True
            state.support_bounces += 1
            state.last_updated = timestamp
            self._log.debug(
                "support_bounce_counted",
                symbol=symbol,
                bounce_count=state.support_bounces,
                level=str(level_price),
            )

    def _handle_resistance_signal(
        self,
        symbol: Symbol,
        state: _SymbolRange,
        level_price: Decimal,
        timestamp: float,
    ) -> None:
        """
        Process a resistance proximity signal.

        Mirrors _handle_support_signal for resistance levels.
        """
        if state.resistance_level is None:
            state.resistance_level = level_price
        else:
            state.resistance_level = level_price

        if not state.in_resistance_proximity:
            # Price just entered resistance zone — count as a bounce
            state.in_resistance_proximity = True
            state.resistance_bounces += 1
            state.last_updated = timestamp
            self._log.debug(
                "resistance_bounce_counted",
                symbol=symbol,
                bounce_count=state.resistance_bounces,
                level=str(level_price),
            )

    async def _on_regime_change(self, event: RegimeChangeEvent) -> None:  # type: ignore[override]
        """
        Track regime transitions and invalidate ranges when leaving SIDEWAYS.
        """
        if not isinstance(event, RegimeChangeEvent):
            return

        old_regime = self._current_regime
        self._current_regime = event.to_regime

        if old_regime == "SIDEWAYS" and event.to_regime != "SIDEWAYS":
            # Leaving SIDEWAYS — all accumulated ranges are invalidated
            self._ranges.clear()
            self._log.info(
                "ranges_invalidated_on_regime_change",
                from_regime=old_regime,
                to_regime=event.to_regime,
            )

    async def _on_market_data(self, event: MarketDataEvent) -> None:  # type: ignore[override]
        """
        Track price movements for zone-exit detection and breakout detection.

        - Zone exits: when price moves sufficiently far from a level, the
          proximity flag is cleared so the next entry is counted as a bounce.
        - Breakout: when price moves beyond a level by breakout_margin_pct,
          the range for that symbol is cleared entirely.
        """
        if not isinstance(event, MarketDataEvent):
            return

        state = self._ranges.get(event.symbol)
        if state is None:
            return

        price = event.price
        state.last_price = price

        # --- Support zone tracking ---
        if state.support_level is not None:
            support = state.support_level
            if support > Decimal("0"):
                dist_pct = abs(price - support) / support * Decimal("100")

                # Zone exit: price moved far enough away from support
                if state.in_support_proximity and dist_pct > _ZONE_EXIT_THRESHOLD_PCT:
                    state.in_support_proximity = False

                # Breakout below support
                below_support = support > price  # price is below support level
                if below_support and dist_pct > self._breakout_margin_pct:
                    self._log.info(
                        "breakout_below_support",
                        symbol=event.symbol,
                        price=str(price),
                        support=str(support),
                        dist_pct=str(dist_pct),
                    )
                    del self._ranges[event.symbol]
                    return

        # --- Resistance zone tracking ---
        if state.resistance_level is not None:
            resistance = state.resistance_level
            if resistance > Decimal("0"):
                dist_pct = abs(price - resistance) / resistance * Decimal("100")

                # Zone exit: price moved far enough away from resistance
                if state.in_resistance_proximity and dist_pct > _ZONE_EXIT_THRESHOLD_PCT:
                    state.in_resistance_proximity = False

                # Breakout above resistance
                above_resistance = price > resistance
                if above_resistance and dist_pct > self._breakout_margin_pct:
                    self._log.info(
                        "breakout_above_resistance",
                        symbol=event.symbol,
                        price=str(price),
                        resistance=str(resistance),
                        dist_pct=str(dist_pct),
                    )
                    del self._ranges[event.symbol]
                    return

    def get_range(self, symbol: Symbol) -> RangeState | None:
        """
        Return the current range state for a symbol.

        Returns None if:
        - No bounce data has been collected for the symbol
        - Either support or resistance level has not yet been observed
        - The range data is stale (last_updated older than staleness window)

        The returned RangeState.range_confirmed is True only when both:
        - bounce_count >= min_bounces
        - range_width_pct >= min_range_width_pct

        Args:
            symbol: Trading symbol (e.g. "BTC/USD").

        Returns:
            RangeState snapshot or None.
        """
        state = self._ranges.get(symbol)
        if state is None:
            return None

        if state.support_level is None or state.resistance_level is None:
            return None

        # Staleness check
        if state.last_updated == 0.0:
            return None

        age_seconds = time.time() - state.last_updated
        if age_seconds > self._staleness_seconds:
            return None

        support = state.support_level
        resistance = state.resistance_level

        # Compute range width (always use absolute difference to handle
        # cases where labels might be inverted)
        if support > Decimal("0"):
            range_width_pct = abs(resistance - support) / support * Decimal("100")
        else:
            return None

        total_bounces = state.support_bounces + state.resistance_bounces

        range_confirmed = (
            total_bounces >= self._min_bounces
            and range_width_pct >= self._min_range_width_pct
        )

        return RangeState(
            support_level=support,
            resistance_level=resistance,
            bounce_count=total_bounces,
            range_confirmed=range_confirmed,
            range_width_pct=range_width_pct,
            last_updated=state.last_updated,
        )
