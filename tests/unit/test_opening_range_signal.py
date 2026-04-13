"""
Unit tests for OpeningRangeSignal.

Key facts discovered from reading opening_range.py and base.py:
- Constructor: OpeningRangeSignal(event_bus, config) — passes event_bus as bus= to base
- _generate_signal(symbol, data) receives list[MarketDataEvent], uses data[-1].price and
  data[-1].timestamp (Unix epoch float).
- Signals route via: base._on_market_data → _generate_signal → if not None → bus.publish(signal)
- bus.publish is an async method; tests mock the bus and assert on publish calls.
- SignalAction.BUY = "buy", SignalAction.SELL = "sell"
- _create_signal() returns a SignalEvent with metadata["source"] = "OpeningRange"
- RTH is 09:30–16:00 ET; is_rth_now checks >= 09:30 and < close
- Tests use a regular trading day: 2026-06-15 (Monday, not holiday)
  09:30 ET = 13:30 UTC (EDT is UTC-4)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.signals.opening_range import OpeningRangeSignal, _ORBState

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# A known regular trading day: 2026-06-15 (Monday, not holiday, not early-close)
TRADE_DATE_ET = datetime(2026, 6, 15, 9, 30, tzinfo=ET)   # 09:30 ET (RTH open)
PREMARKET_ET  = datetime(2026, 6, 15, 8, 0, tzinfo=ET)    # 08:00 ET (pre-market)
# Next trading day: 2026-06-16 (Tuesday)
NEXT_DAY_ET   = datetime(2026, 6, 16, 9, 30, tzinfo=ET)


def _epoch(dt: datetime) -> float:
    """Convert tz-aware datetime to Unix epoch float."""
    return dt.timestamp()


def _make_bus() -> MagicMock:
    """
    Create a mock EventBus.

    The base class calls bus.subscribe(...) in __init__ and later
    awaits bus.publish(signal). We need subscribe to work synchronously
    and publish to be awaitable.
    """
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.publish = AsyncMock()
    return bus


def _make_gen(
    symbols=("AAPL",),
    range_minutes=15,
    breakout_buffer_bps=5,
    min_range_bps=20,
    max_range_bps=500,
) -> tuple[OpeningRangeSignal, MagicMock]:
    """Return (generator, mock_bus)."""
    bus = _make_bus()
    config = {
        "symbols": list(symbols),
        "range_minutes": range_minutes,
        "breakout_buffer_bps": breakout_buffer_bps,
        "min_range_bps": min_range_bps,
        "max_range_bps": max_range_bps,
    }
    gen = OpeningRangeSignal(event_bus=bus, config=config)
    return gen, bus


def _make_event(symbol: str, price: float, dt: datetime) -> MarketDataEvent:
    """Create a MarketDataEvent for a given symbol/price/datetime."""
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=_epoch(dt),
        symbol=symbol,
        price=Decimal(str(price)),
        volume=Decimal("100"),
    )


def _call_generate(gen: OpeningRangeSignal, symbol: str, events: list[MarketDataEvent]):
    """
    Directly call _generate_signal bypassing the async bus machinery.
    Returns the SignalEvent or None.
    """
    return gen._generate_signal(symbol, events)


# ---------------------------------------------------------------------------
# Test 1 — ignores ticks for non-configured symbols
# ---------------------------------------------------------------------------

def test_ignores_non_configured_symbol():
    """Ticks for TSLA (not in symbols list) produce no signal."""
    gen, bus = _make_gen(symbols=("AAPL",))

    # Send a tick well after range window would close
    dt = TRADE_DATE_ET + timedelta(minutes=30)
    event = _make_event("TSLA", 200.0, dt)
    result = _call_generate(gen, "TSLA", [event])

    assert result is None
    bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — no signal during range-building window
# ---------------------------------------------------------------------------

def test_no_signal_during_range_building():
    """Ticks within first 15 min accumulate state but emit no signal."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15)

    # 12 ticks at 09:30 through 09:41 ET
    events = []
    for i in range(12):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        events.append(_make_event("AAPL", 150.0 + i * 0.1, dt))

    # Each call receives all events so far (cumulative)
    result = None
    for idx in range(len(events)):
        result = _call_generate(gen, "AAPL", events[: idx + 1])

    assert result is None
    # State should have accumulated ticks
    assert gen._states["AAPL"].tick_count == 12
    assert gen._states["AAPL"].frozen is False


# ---------------------------------------------------------------------------
# Test 3 — buy emits on breakout above ORB high + buffer
# ---------------------------------------------------------------------------

def test_buy_signal_on_breakout_above_range():
    """After range is built and frozen, price above high+buffer triggers BUY."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15, breakout_buffer_bps=5)

    # Build range: 10 ticks, low=150.0, high=150.75
    # range=0.75, mid=150.375 → bps≈49.9 (within 20–500)
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        price = 150.0 if i % 2 == 0 else 150.75  # alternating low/high
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    # Force a freeze tick at 09:45 ET (after 15 min window)
    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    freeze_event = _make_event("AAPL", 150.375, freeze_dt)
    all_events = range_events + [freeze_event]
    _call_generate(gen, "AAPL", all_events)
    assert gen._states["AAPL"].frozen is True
    assert gen._states["AAPL"].valid is True

    # Breakout: price > high * (1 + 5/10000) = 150.75 * 1.0005 = 150.8254
    breakout_dt = TRADE_DATE_ET + timedelta(minutes=20)
    breakout_event = _make_event("AAPL", 151.0, breakout_dt)
    all_events_2 = all_events + [breakout_event]
    signal = _call_generate(gen, "AAPL", all_events_2)

    assert signal is not None
    assert isinstance(signal, SignalEvent)
    assert signal.action == SignalAction.BUY
    assert signal.symbol == "AAPL"
    assert signal.metadata.get("source") == "OpeningRange"


# ---------------------------------------------------------------------------
# Test 4 — sell emits on breakdown below ORB low - buffer
# ---------------------------------------------------------------------------

def test_sell_signal_on_breakdown_below_range():
    """After range is frozen, price below low-buffer triggers SELL."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15, breakout_buffer_bps=5)

    # Build range: 10 ticks, low=150.0, high=150.75
    # range=0.75, mid=150.375 → ~49.9 bps (within 20–500)
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        price = 150.0 if i % 2 == 0 else 150.75
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    # Freeze at 09:45
    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 150.375, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    # Breakdown: price < low * (1 - 5/10000) = 150.0 * 0.9995 = 149.925
    breakdown_dt = TRADE_DATE_ET + timedelta(minutes=20)
    breakdown_event = _make_event("AAPL", 149.8, breakdown_dt)
    signal = _call_generate(gen, "AAPL", all_events + [breakdown_event])

    assert signal is not None
    assert signal.action == SignalAction.SELL
    assert signal.symbol == "AAPL"
    assert signal.metadata.get("source") == "OpeningRange"


# ---------------------------------------------------------------------------
# Test 5 — range rejected: too tight
# ---------------------------------------------------------------------------

def test_range_rejected_too_tight():
    """Range < min_range_bps (20 bps) causes state.valid=False; no breakout signals."""
    # Use tighter min_range_bps=100 so a 10-cent range on a $150 stock (~7 bps) is rejected
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15, min_range_bps=100)

    # Build 10 ticks with nearly identical prices (range ~6.7 bps = 0.1/149.95 * 10000)
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        # low=149.9, high=150.0 → range=0.1, mid=149.95 → bps≈6.7
        price = 149.9 if i % 2 == 0 else 150.0
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    # Freeze at 09:45
    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 149.95, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    assert gen._states["AAPL"].frozen is True
    assert gen._states["AAPL"].valid is False

    # A clear breakout attempt still returns None
    breakout_dt = TRADE_DATE_ET + timedelta(minutes=20)
    signal = _call_generate(gen, "AAPL", all_events + [_make_event("AAPL", 200.0, breakout_dt)])
    assert signal is None


# ---------------------------------------------------------------------------
# Test 6 — range rejected: too wide
# ---------------------------------------------------------------------------

def test_range_rejected_too_wide():
    """Range > max_range_bps causes state.valid=False; no breakout signals."""
    # max_range_bps=100 but our range will be ~667 bps
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15, max_range_bps=100)

    # low=100, high=110 → range=10, mid=105 → bps≈952
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        price = 100.0 if i % 2 == 0 else 110.0
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 105.0, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    assert gen._states["AAPL"].frozen is True
    assert gen._states["AAPL"].valid is False

    breakout_dt = TRADE_DATE_ET + timedelta(minutes=20)
    signal = _call_generate(gen, "AAPL", all_events + [_make_event("AAPL", 200.0, breakout_dt)])
    assert signal is None


# ---------------------------------------------------------------------------
# Test 7 — range rejected: insufficient ticks (< 10)
# ---------------------------------------------------------------------------

def test_range_rejected_insufficient_ticks():
    """Fewer than 10 ticks in the range window → range invalid → no signals."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15)

    # Only 5 ticks in the window
    range_events = []
    for i in range(5):
        dt = TRADE_DATE_ET + timedelta(minutes=i * 2)   # spread them out
        price = 100.0 if i % 2 == 0 else 110.0
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    # Freeze at 09:45
    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 105.0, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    assert gen._states["AAPL"].frozen is True
    assert gen._states["AAPL"].valid is False
    assert gen._states["AAPL"].tick_count == 5

    # Breakout attempt → no signal
    breakout_dt = TRADE_DATE_ET + timedelta(minutes=20)
    signal = _call_generate(gen, "AAPL", all_events + [_make_event("AAPL", 200.0, breakout_dt)])
    assert signal is None


# ---------------------------------------------------------------------------
# Test 8 — daily reset
# ---------------------------------------------------------------------------

def test_daily_reset_on_next_day_tick():
    """A tick from the next calendar day resets the ORB state for the symbol."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15)

    # Build + freeze a valid range on day 1: low=150.0, high=150.75 (~49.9 bps)
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        price = 150.0 if i % 2 == 0 else 150.75
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 150.375, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    day1_date = gen._states["AAPL"].date
    assert gen._states["AAPL"].frozen is True
    assert day1_date is not None

    # Send a tick at 09:30 on the next trading day (still in range window)
    next_day_event = _make_event("AAPL", 120.0, NEXT_DAY_ET)
    _call_generate(gen, "AAPL", [next_day_event])

    # State should have been reset for the new day
    new_state = gen._states["AAPL"]
    assert new_state.date != day1_date
    assert new_state.frozen is False
    assert new_state.tick_count == 1   # this one in-window tick was accumulated


# ---------------------------------------------------------------------------
# Test 9 — only one buy per day (no whipsaw)
# ---------------------------------------------------------------------------

def test_only_one_buy_signal_per_day():
    """After a BUY fires, a subsequent breakout tick does not fire another BUY."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15, breakout_buffer_bps=5)

    # Build 10-tick range: low=150.0, high=150.75
    # range=0.75, mid=150.375 → ~49.9 bps (within 20–500)
    range_events = []
    for i in range(10):
        dt = TRADE_DATE_ET + timedelta(minutes=i)
        price = 150.0 if i % 2 == 0 else 150.75
        range_events.append(_make_event("AAPL", price, dt))

    for idx in range(len(range_events)):
        _call_generate(gen, "AAPL", range_events[: idx + 1])

    # Freeze
    freeze_dt = TRADE_DATE_ET + timedelta(minutes=15)
    all_events = range_events + [_make_event("AAPL", 150.375, freeze_dt)]
    _call_generate(gen, "AAPL", all_events)

    # First breakout: price > 150.75 * 1.0005 = 150.8254
    t1 = TRADE_DATE_ET + timedelta(minutes=20)
    sig1 = _call_generate(gen, "AAPL", all_events + [_make_event("AAPL", 151.0, t1)])
    assert sig1 is not None
    assert sig1.action == SignalAction.BUY
    assert "buy" in gen._states["AAPL"].signaled

    # Second breakout tick → no second BUY
    t2 = TRADE_DATE_ET + timedelta(minutes=21)
    all_events_after = all_events + [
        _make_event("AAPL", 151.0, t1),
        _make_event("AAPL", 152.0, t2),
    ]
    sig2 = _call_generate(gen, "AAPL", all_events_after)
    assert sig2 is None


# ---------------------------------------------------------------------------
# Test 10 — pre-market (out-of-RTH) ticks ignored
# ---------------------------------------------------------------------------

def test_premarket_ticks_do_not_populate_range():
    """Ticks before 09:30 ET are ignored — they don't accumulate into the range."""
    gen, bus = _make_gen(symbols=("AAPL",), range_minutes=15)

    # Send 15 pre-market ticks (08:00–08:14 ET)
    pre_events = []
    for i in range(15):
        dt = PREMARKET_ET + timedelta(minutes=i)
        pre_events.append(_make_event("AAPL", 150.0 + i, dt))

    # Process all pre-market ticks
    for idx in range(len(pre_events)):
        result = _call_generate(gen, "AAPL", pre_events[: idx + 1])
        assert result is None

    # State should not have accumulated any ticks (pre-market gate returns early)
    # The state may have been reset to today's date but tick_count stays 0
    assert gen._states["AAPL"].tick_count == 0
    assert gen._states["AAPL"].frozen is False
