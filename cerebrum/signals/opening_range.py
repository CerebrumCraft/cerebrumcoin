"""
Opening Range Breakout (ORB) signal generator for US equities.

@decision DEC-STOCKS-004
@title ORB signal with source-based isolation
@status accepted
@rationale Dedicated equity-native signal source. Uses signal_source_filter
on the consuming strategy to prevent contamination with crypto strategies.
Builds per-symbol high/low over the first N minutes after 09:30 ET, then
emits a buy signal on break above (with configurable buffer) or a sell
signal on break below. Each symbol's ORB resets daily.

Implementation notes:
- Inherits from SignalGenerator to get EventBus subscription and data
  accumulation for free. _get_min_periods() returns 1 so the base class
  never gates us out; all real gating is time-based inside _generate_signal().
- timestamp on MarketDataEvent is a Unix epoch float. We convert to ET
  datetime here for all time comparisons.
- _create_signal() auto-injects metadata["source"] and metadata["timeframe"].
  ORB-specific metadata (orb_high, orb_low, price) is passed via `reason`
  as a human-readable string since _create_signal() has no metadata kwarg.
- SignalType.TECHNICAL is used (no dedicated ORB type exists in the enum).
- One buy signal and one sell signal emitted per symbol per day at most
  (_signaled_directions tracks which directions have fired today).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import SignalAction, SignalType, Symbol
from cerebrum.signals.base import SignalGenerator
from cerebrum.utils.trading_session import ET, RTH_OPEN, is_rth_now

logger = structlog.get_logger()


@dataclass
class _ORBState:
    """Per-symbol opening-range accumulator. Reset each calendar day."""
    high: Decimal | None = None
    low: Decimal | None = None
    tick_count: int = 0
    frozen: bool = False   # True once range window has elapsed
    valid: bool = True     # False if range was rejected (too few ticks, degenerate width)
    date: date | None = None
    signaled: set[str] = field(default_factory=set)  # "buy" / "sell" fired today


class OpeningRangeSignal(SignalGenerator):
    """
    Opening Range Breakout signal generator for US equities.

    Builds a per-symbol high/low during the first `range_minutes` of the
    regular trading session (09:30–09:30+N ET).  After the window closes,
    emits:
      - BUY  when price breaks above range_high * (1 + buffer_bps/10000)
      - SELL when price breaks below range_low  * (1 - buffer_bps/10000)

    Each direction fires at most once per symbol per calendar day.
    """

    NAME = "OpeningRange"

    def __init__(self, event_bus: EventBus, config: dict[str, Any]) -> None:
        """
        Args:
            event_bus: EventBus instance (passed as `bus` to base).
            config: Dict with keys:
                symbols            – list of ticker strings to track
                range_minutes      – minutes after 09:30 to build range (default 15)
                breakout_buffer_bps – buffer above/below range before signal fires (default 5)
                min_range_bps      – minimum range width to be considered valid (default 20)
                max_range_bps      – maximum range width; wider ranges are rejected (default 500)
        """
        symbols: list[str] = config.get("symbols", [])
        super().__init__(
            bus=event_bus,
            signal_type=SignalType.TECHNICAL,
            window_size=1,   # ORB is time-gated, not window-gated; 1 keeps base happy
            name=self.NAME,
            timeframe="1m",
        )
        self._symbols: set[str] = set(symbols)
        self._range_minutes: int = int(config.get("range_minutes", 15))
        self._breakout_buffer_bps: Decimal = Decimal(str(config.get("breakout_buffer_bps", 5)))
        self._min_range_bps: Decimal = Decimal(str(config.get("min_range_bps", 20)))
        self._max_range_bps: Decimal = Decimal(str(config.get("max_range_bps", 500)))
        self._states: dict[str, _ORBState] = {s: _ORBState() for s in self._symbols}
        self._log = logger.bind(component="signal_opening_range")

    # ------------------------------------------------------------------
    # Base-class contract
    # ------------------------------------------------------------------

    def _get_min_periods(self) -> int:
        """Always 1 — ORB gates on wall-clock time, not data-window depth."""
        return 1

    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """
        Core ORB logic.  Called by the base class whenever a new tick arrives
        for `symbol` and len(data) >= _get_min_periods() (i.e., always).
        """
        if symbol not in self._symbols:
            return None

        latest = data[-1]
        price: Decimal = latest.price
        ts_epoch: float = latest.timestamp

        # Convert Unix epoch → ET datetime
        et_time = datetime.fromtimestamp(ts_epoch, tz=ET)
        today = et_time.date()

        state = self._states[symbol]

        # Daily reset
        if state.date != today:
            self._states[symbol] = _ORBState(date=today)
            state = self._states[symbol]

        # Gate on regular trading hours
        if not is_rth_now(datetime.fromtimestamp(ts_epoch, tz=ET)):
            return None

        # Time elapsed since 09:30 ET today
        open_dt = datetime.combine(today, RTH_OPEN, tzinfo=ET)
        minutes_since_open = (et_time - open_dt).total_seconds() / 60.0

        # --- Phase 1: accumulate the opening range ---
        if minutes_since_open < self._range_minutes:
            if state.high is None or price > state.high:
                state.high = price
            if state.low is None or price < state.low:
                state.low = price
            state.tick_count += 1
            return None

        # --- Phase 2: freeze the range (runs exactly once per day) ---
        if not state.frozen:
            state.frozen = True
            state.valid = self._validate_range(symbol, state)

        if not state.valid:
            return None

        # --- Phase 3: check for breakout ---
        buffer = self._breakout_buffer_bps / Decimal("10000")

        if "buy" not in state.signaled:
            threshold_up = state.high * (Decimal("1") + buffer)
            if price > threshold_up:
                state.signaled.add("buy")
                return self._emit(symbol, SignalAction.BUY, price, state, latest.timestamp)

        if "sell" not in state.signaled:
            threshold_dn = state.low * (Decimal("1") - buffer)
            if price < threshold_dn:
                state.signaled.add("sell")
                return self._emit(symbol, SignalAction.SELL, price, state, latest.timestamp)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_range(self, symbol: str, state: _ORBState) -> bool:
        """
        Validate the frozen opening range.

        Rejects if:
        - Fewer than 10 ticks were received during the window (data gap)
        - High == Low (degenerate / single-price market)
        - Range width (in bps) is outside [min_range_bps, max_range_bps]

        Returns True if the range is usable, False otherwise.
        """
        if state.tick_count < 10:
            self._log.warning(
                "orb_range_rejected_insufficient_ticks",
                symbol=symbol,
                tick_count=state.tick_count,
            )
            return False

        if state.high is None or state.low is None or state.high == state.low:
            self._log.warning("orb_range_rejected_degenerate", symbol=symbol)
            return False

        mid = (state.high + state.low) / Decimal("2")
        range_bps = (state.high - state.low) / mid * Decimal("10000")

        if range_bps < self._min_range_bps:
            self._log.warning(
                "orb_range_rejected_too_narrow",
                symbol=symbol,
                range_bps=str(range_bps),
                min_bps=str(self._min_range_bps),
            )
            return False

        if range_bps > self._max_range_bps:
            self._log.warning(
                "orb_range_rejected_too_wide",
                symbol=symbol,
                range_bps=str(range_bps),
                max_bps=str(self._max_range_bps),
            )
            return False

        self._log.info(
            "orb_range_frozen",
            symbol=symbol,
            high=str(state.high),
            low=str(state.low),
            range_bps=str(range_bps),
            tick_count=state.tick_count,
        )
        return True

    def _emit(
        self,
        symbol: str,
        action: SignalAction,
        price: Decimal,
        state: _ORBState,
        timestamp: float,
    ) -> SignalEvent:
        """Build and return a SignalEvent for an ORB breakout."""
        mid = (state.high + state.low) / Decimal("2")
        range_bps = (state.high - state.low) / mid * Decimal("10000")
        # Strength proportional to range width relative to max (wider range → stronger conviction)
        strength = min(Decimal("1"), range_bps / self._max_range_bps)

        reason = (
            f"ORB {action.value.upper()}: price={price} "
            f"range=[{state.high},{state.low}] "
            f"range_bps={range_bps:.1f}"
        )

        self._log.info(
            "orb_signal_emitted",
            symbol=symbol,
            action=action.value,
            price=str(price),
            orb_high=str(state.high),
            orb_low=str(state.low),
            range_bps=str(range_bps),
            strength=str(strength),
        )

        return self._create_signal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=Decimal("0.75"),
            timestamp=timestamp,
            reason=reason,
        )
