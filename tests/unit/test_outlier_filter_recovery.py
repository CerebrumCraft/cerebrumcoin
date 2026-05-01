"""
Regression tests for the outlier-filter self-healing (Part C) and the Alpaca
market-hours gate (Part A).

Part C — outlier filter recovery (DEC-DIAG-004):
  Session 48 caused 192,861 `outlier_tick_rejected` events for AAPL over
  44h26m.  The root cause was a two-stage failure:
    1. Alpaca off-hours polling delivered $127.885 stale quotes for ~10.9h
       while the real price was $267+.
    2. The outlier filter cold-started on those stale ticks, locked its rolling
       median at $127.885, and rejected every real price forever because
       rejected ticks never enter the window so the median can never adapt.

  The fix adds a per-symbol consecutive-rejection counter.  After
  `outlier_consecutive_reject_reset` consecutive rejections (default 60), the
  window is cleared and the cold-start guard re-engages, allowing the real
  price cluster to establish a new median.

  A short burst of bad ticks (< threshold) must still be rejected without
  triggering a reset — that is the NVDA DEC-DIAG-003 use-case.

Part A — Alpaca market-hours gate:
  The `AlpacaAdapter._is_market_open()` helper must return False outside the
  09:30-16:00 ET / Mon-Fri window (no holiday handling for v1) and True inside
  it.  The polling task itself must skip the API call when the helper returns
  False.

@decision DEC-DIAG-004
@title Regression tests for Session 48 AAPL lock-in fix (A+C)
@status accepted
@rationale Part A + Part C collectively prevent the brittle-median-lock-in
pattern from trapping a symbol forever.  These tests serve as the regression
guard: if either fix regresses, the test suite catches it before shipping.
Test methodology: ConcreteSignalGenerator (same as test_nvda_outlier_rejection.py)
for filter tests; AlpacaAdapter._is_market_open() unit tests for Part A.
No live API keys are needed for any test in this file.
"""

import asyncio
from decimal import Decimal
from time import time
from unittest.mock import MagicMock, patch

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent
from cerebrum.core.types import EventType, SignalType


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_nvda_outlier_rejection.py patterns)
# ---------------------------------------------------------------------------

def _make_market_event(symbol: str, price: float, ts: float | None = None) -> MarketDataEvent:
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=ts or time(),
        symbol=symbol,
        price=Decimal(str(price)),
        volume=Decimal("100"),
    )


def _make_test_generator(
    bus: EventBus,
    outlier_window: int = 20,
    outlier_deviation_threshold: float = 0.5,
    outlier_min_ticks: int | None = None,
    outlier_consecutive_reject_reset: int = 60,
):
    """Build a minimal concrete SignalGenerator with the self-healing filter."""
    from cerebrum.signals.base import SignalGenerator

    class _NoopGenerator(SignalGenerator):
        def __init__(self, b, **kwargs):
            super().__init__(
                bus=b,
                signal_type=SignalType.TECHNICAL,
                window_size=50,
                name="recovery_test_gen",
                **kwargs,
            )

        def _get_min_periods(self) -> int:
            return 1

        def _generate_signal(self, symbol, data):
            return None

    kwargs = dict(
        outlier_window=outlier_window,
        outlier_deviation_threshold=outlier_deviation_threshold,
        outlier_consecutive_reject_reset=outlier_consecutive_reject_reset,
    )
    if outlier_min_ticks is not None:
        kwargs["outlier_min_ticks"] = outlier_min_ticks
    return _NoopGenerator(bus, **kwargs)


@pytest.fixture
async def bus():
    b = EventBus(queue_size=500)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Part C — self-healing outlier filter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_outlier_filter_recovers_from_bad_cold_start(bus):
    """
    Cold-start absorbs a stale-price cluster, then the feed switches to the
    real cluster.  The filter must adapt rather than reject every legitimate
    tick forever (Session 48 AAPL lock-in regression).

    Sequence:
      • outlier_window=20, outlier_min_ticks=10, reset_threshold=30
      • 10 stale ticks @ $127.885 → cold-start fills window, median=$127.885
      • 60 real ticks @ $267.565 → first 30 get rejected (no reset yet)
                                 → tick #30+1 triggers reset
                                 → next 10 ticks (cold-start) are accepted
                                 → window refills at real price

    After all 60 real ticks, the _data window must contain entries for AAPL,
    proving the filter broke out of the lock-in.
    """
    sg = _make_test_generator(
        bus,
        outlier_window=20,
        outlier_min_ticks=10,
        outlier_consecutive_reject_reset=30,
    )

    # Cold-start: 10 stale ticks @ $127.885 (mirrors Session 48 Alpaca off-hours quote)
    for _ in range(10):
        await sg._on_market_data(_make_market_event("AAPL", 127.885))

    # Confirm median is now locked at the stale price
    assert len(sg._outlier_price_window["AAPL"]) == 10

    # Feed switches to real prices — reset_threshold=30, so after 30 consecutive
    # rejections the window should clear and cold-start re-engages.
    for _ in range(60):
        await sg._on_market_data(_make_market_event("AAPL", 267.565))

    # After enough sustained real ticks the filter must have adapted.
    assert len(sg._data["AAPL"]) >= 20, (
        "filter locked to bad cold-start cluster — never adapted to real prices"
    )


@pytest.mark.asyncio
async def test_reset_counter_increments_on_rejection(bus):
    """Consecutive-rejection counter must increment on each rejected tick."""
    sg = _make_test_generator(
        bus,
        outlier_window=10,
        outlier_min_ticks=5,
        outlier_consecutive_reject_reset=100,  # high threshold — won't fire
    )

    # Build a stable baseline
    for p in [100.0] * 10:
        await sg._on_market_data(_make_market_event("BASE/X", p))

    initial_rejected = sg.outlier_rejected_counts.get("BASE/X", 0)

    # Send 5 consecutive outliers
    for _ in range(5):
        await sg._on_market_data(_make_market_event("BASE/X", 999.0))

    total_rejected = sg.outlier_rejected_counts.get("BASE/X", 0)
    assert total_rejected == initial_rejected + 5
    # Consecutive counter should be at 5 (haven't reset yet)
    assert sg._outlier_consecutive_rejections.get("BASE/X", 0) == 5


@pytest.mark.asyncio
async def test_accepted_tick_resets_consecutive_counter(bus):
    """An accepted tick must reset the consecutive-rejection counter to 0."""
    sg = _make_test_generator(
        bus,
        outlier_window=10,
        outlier_min_ticks=5,
        outlier_consecutive_reject_reset=100,  # won't fire
    )

    # Baseline
    for p in [100.0] * 10:
        await sg._on_market_data(_make_market_event("CONSEC/X", p))

    # 3 consecutive outliers
    for _ in range(3):
        await sg._on_market_data(_make_market_event("CONSEC/X", 999.0))

    assert sg._outlier_consecutive_rejections.get("CONSEC/X", 0) == 3

    # One accepted tick — counter should go back to 0
    await sg._on_market_data(_make_market_event("CONSEC/X", 101.0))
    assert sg._outlier_consecutive_rejections.get("CONSEC/X", 0) == 0


@pytest.mark.asyncio
async def test_reset_clears_price_window(bus):
    """
    When the reset fires (N consecutive rejections), the price window must be
    cleared to empty so cold-start re-engages.
    """
    sg = _make_test_generator(
        bus,
        outlier_window=20,
        outlier_min_ticks=10,
        outlier_consecutive_reject_reset=5,  # tiny threshold for quick test
    )

    # Build a baseline that will later look stale
    for p in [100.0] * 12:
        await sg._on_market_data(_make_market_event("RESET/X", p))

    assert len(sg._outlier_price_window["RESET/X"]) == 12

    # Send 5 outliers — should trigger reset
    for _ in range(5):
        await sg._on_market_data(_make_market_event("RESET/X", 999.0))

    # Price window must be empty after reset
    assert len(sg._outlier_price_window["RESET/X"]) == 0, (
        "price window should be cleared after consecutive-reject reset"
    )
    # Consecutive counter must also be cleared
    assert sg._outlier_consecutive_rejections.get("RESET/X", 0) == 0


@pytest.mark.asyncio
async def test_nvda_short_burst_does_not_trigger_reset(bus):
    """
    DEC-DIAG-003 preservation: a short burst of bad ticks (< reset threshold)
    must be rejected without triggering a window reset.

    This is the NVDA bimodal $103/$210 use-case: occasional spurious ticks at
    the wrong price level should not cause the filter to forget the correct median.
    """
    sg = _make_test_generator(
        bus,
        outlier_window=20,
        outlier_min_ticks=10,
        outlier_consecutive_reject_reset=30,  # threshold well above short burst
    )

    # NVDA baseline at ~$103
    nvda_baseline = [103.0 + (i % 3) * 0.5 for i in range(20)]
    for p in nvda_baseline:
        await sg._on_market_data(_make_market_event("NVDA", p))

    price_window_size_before = len(sg._outlier_price_window["NVDA"])

    # Short burst of 5 bad ticks at $210 (the NVDA bimodal anomaly)
    bad_prices = [210.0, 211.5, 209.8, 210.3, 212.0]
    for p in bad_prices:
        await sg._on_market_data(_make_market_event("NVDA", p))

    # All 5 should be rejected
    assert sg.outlier_rejected_counts.get("NVDA", 0) == 5

    # Price window should NOT have been reset — still has the baseline size
    # (capped at outlier_window=20, but definitely non-zero)
    assert len(sg._outlier_price_window["NVDA"]) >= 18, (
        "short burst triggered an unexpected window reset — DEC-DIAG-003 regression"
    )

    # Consecutive rejection counter must be back in the short-burst range (not reset)
    assert sg._outlier_consecutive_rejections.get("NVDA", 0) == 5

    # Good NVDA ticks still accepted after the burst
    for p in [103.0, 104.0, 103.5]:
        await sg._on_market_data(_make_market_event("NVDA", p))

    assert sg._outlier_consecutive_rejections.get("NVDA", 0) == 0


@pytest.mark.asyncio
async def test_reset_triggers_warning_log(bus):
    """
    The reset event fires when consecutive rejections reach the threshold.
    We verify this indirectly: the price window is cleared (reset happened)
    and the consecutive counter is zeroed.  The actual log is emitted to
    structlog (verified visually in captured stdout during test runs) but
    structlog does not route through Python's standard logging module so
    caplog cannot capture it reliably — the observable side-effect is the
    window clear.
    """
    sg = _make_test_generator(
        bus,
        outlier_window=10,
        outlier_min_ticks=5,
        outlier_consecutive_reject_reset=5,
    )

    # Baseline
    for p in [200.0] * 8:
        await sg._on_market_data(_make_market_event("LOG/X", p))

    # Price window populated
    assert len(sg._outlier_price_window["LOG/X"]) == 8

    # Trigger reset
    for _ in range(5):
        await sg._on_market_data(_make_market_event("LOG/X", 999.0))

    # Reset fired: window cleared and counter zeroed
    assert len(sg._outlier_price_window["LOG/X"]) == 0, (
        "price window should be empty after reset — log was not triggered"
    )
    assert sg._outlier_consecutive_rejections.get("LOG/X", 0) == 0


@pytest.mark.asyncio
async def test_filter_recovery_after_two_sequential_bad_clusters(bus):
    """
    Verify the filter can recover from a second bad cluster even after it
    previously recovered from the first one.  Two sequential resets must each
    produce a valid adaption.
    """
    sg = _make_test_generator(
        bus,
        outlier_window=20,
        outlier_min_ticks=10,
        outlier_consecutive_reject_reset=15,
    )

    # Cluster 1 (bad cold-start at $50)
    for _ in range(10):
        await sg._on_market_data(_make_market_event("TWO/X", 50.0))

    # Feed switches to $200 — 15 rejections triggers first reset
    for _ in range(20):
        await sg._on_market_data(_make_market_event("TWO/X", 200.0))

    data_after_first = len(sg._data["TWO/X"])
    assert data_after_first >= 5, "filter did not recover after first bad cluster"

    # Simulate a second bad cluster at $50 — enough to trigger a second reset
    for _ in range(20):
        await sg._on_market_data(_make_market_event("TWO/X", 50.0))

    # Feed switches back to $200
    for _ in range(25):
        await sg._on_market_data(_make_market_event("TWO/X", 200.0))

    data_after_second = len(sg._data["TWO/X"])
    assert data_after_second >= 10, "filter did not recover after second bad cluster"


# ---------------------------------------------------------------------------
# Part A — Alpaca market-hours gate
# ---------------------------------------------------------------------------

alpaca_module = pytest.importorskip("alpaca")


def _make_alpaca_adapter():
    from cerebrum.adapters.alpaca import AlpacaAdapter
    bus = MagicMock()
    return AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test", "paper": True})


class TestAlpacaMarketHoursGate:
    """
    Unit-test _is_market_open() without hitting the Alpaca API.

    All times are tested against the US Eastern timezone window:
    Mon-Fri 09:30-16:00 ET.  The v1 implementation does NOT handle NYSE
    holidays — that is a known follow-up per the brief.
    """

    def _market_open_at(self, weekday: int, hour: int, minute: int):
        """
        Return True/False for _is_market_open() at the given ET datetime.

        weekday: 0=Mon … 4=Fri, 5=Sat, 6=Sun
        hour/minute: wall-clock ET (no DST needed — pytz handles it)
        """
        import datetime
        import pytz
        from cerebrum.adapters.alpaca import AlpacaAdapter

        bus = MagicMock()
        adapter = AlpacaAdapter(bus, {"api_key": "t", "secret_key": "t"})

        # Build a naive datetime, localize it to ET, then pass to _is_market_open.
        et = pytz.timezone("America/New_York")
        # Use a Monday in 2026 as our anchor (2026-05-04 = Monday)
        base_date = datetime.date(2026, 5, 4)
        delta = datetime.timedelta(days=weekday)  # 0=Mon → date stays Monday
        target_date = base_date + delta
        naive_dt = datetime.datetime.combine(
            target_date,
            datetime.time(hour, minute, 0),
        )
        aware_dt = et.localize(naive_dt)

        return adapter._is_market_open(now=aware_dt)

    def test_monday_930_is_open(self):
        assert self._market_open_at(0, 9, 30) is True

    def test_monday_1200_is_open(self):
        assert self._market_open_at(0, 12, 0) is True

    def test_monday_1559_is_open(self):
        assert self._market_open_at(0, 15, 59) is True

    def test_monday_1600_is_closed(self):
        """16:00 ET is the market close; polling should stop."""
        assert self._market_open_at(0, 16, 0) is False

    def test_monday_0929_is_closed(self):
        """One minute before open — must be closed."""
        assert self._market_open_at(0, 9, 29) is False

    def test_monday_2200_is_closed(self):
        """Late night — clearly closed."""
        assert self._market_open_at(0, 22, 0) is False

    def test_saturday_1200_is_closed(self):
        assert self._market_open_at(5, 12, 0) is False

    def test_sunday_1200_is_closed(self):
        assert self._market_open_at(6, 12, 0) is False

    def test_friday_1559_is_open(self):
        assert self._market_open_at(4, 15, 59) is True

    def test_friday_1600_is_closed(self):
        assert self._market_open_at(4, 16, 0) is False


# @mock-exempt: AlpacaAdapter._data_client is an external Alpaca API boundary —
# mocking at the client level to avoid real network calls in unit tests.
@pytest.mark.asyncio
async def test_poll_skips_api_call_outside_market_hours():
    """
    _poll_market_data must skip the Alpaca API call when _is_market_open()
    returns False.  The poll loop should still sleep and re-check, not error.
    """
    alpaca_skip = pytest.importorskip("alpaca")

    from cerebrum.adapters.alpaca import AlpacaAdapter
    from cerebrum.core.bus import EventBus

    bus = EventBus(queue_size=50)
    await bus.start()

    mock_data_client = MagicMock()
    mock_data_client.get_stock_latest_quote = MagicMock(return_value={})

    adapter = AlpacaAdapter(bus, {"api_key": "t", "secret_key": "t",
                                   "poll_interval_seconds": 0.01})
    adapter._data_client = mock_data_client
    adapter._connected = True

    # Patch _is_market_open to always return False
    adapter._is_market_open = MagicMock(return_value=False)

    # Run the polling task briefly
    task = asyncio.create_task(adapter._poll_market_data("AAPL"))
    await asyncio.sleep(0.05)  # let it iterate a few times
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # API must NOT have been called
    mock_data_client.get_stock_latest_quote.assert_not_called()

    await bus.stop()


# @mock-exempt: AlpacaAdapter._data_client and mock_quote are external Alpaca API
# objects — mocking at the API boundary to avoid real network calls.
@pytest.mark.asyncio
async def test_poll_calls_api_during_market_hours():
    """
    When _is_market_open() returns True, the adapter must call the Alpaca API
    and publish a MarketDataEvent.
    """
    alpaca_skip = pytest.importorskip("alpaca")

    from cerebrum.adapters.alpaca import AlpacaAdapter
    from cerebrum.core.bus import EventBus

    bus = EventBus(queue_size=50)
    await bus.start()

    mock_data_client = MagicMock()

    # Alpaca quote mock (external API boundary)
    mock_quote = MagicMock()
    mock_quote.ask_price = 267.60
    mock_quote.bid_price = 267.50
    mock_quote.ask_size = 100
    mock_quote.bid_size = 100
    mock_data_client.get_stock_latest_quote = MagicMock(
        return_value={"AAPL": mock_quote}
    )

    adapter = AlpacaAdapter(bus, {
        "api_key": "t", "secret_key": "t",
        "poll_interval_seconds": 0.01,
    })
    adapter._data_client = mock_data_client
    adapter._connected = True

    # Patch _is_market_open to always return True
    adapter._is_market_open = MagicMock(return_value=True)

    task = asyncio.create_task(adapter._poll_market_data("AAPL"))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # API must have been called at least once
    assert mock_data_client.get_stock_latest_quote.call_count >= 1

    await bus.stop()
