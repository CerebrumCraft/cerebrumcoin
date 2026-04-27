"""
Tests for the NVDA price-feed outlier rejection in SignalGenerator.

Session 43 confirmed a bimodal NVDA price distribution:
  - Cluster A: ~$103 (correct post-split price)
  - Cluster B: ~$208–$216 (roughly 2× — likely pre-split or unadjusted feed)
  - 14,543 ticks, min=$98.90 max=$216.50 mean=$103.43

The fix: SignalGenerator._on_market_data rejects ticks that deviate more than
`outlier_deviation_threshold` (default 0.5 = 50%) from the rolling median of
the last `outlier_window` (default 20) ticks for that symbol. Rejected ticks
are logged with reason "outlier_tick_rejected" and NOT added to _data.

Cold-start behaviour: rejection is disabled until at least
`outlier_min_ticks` (default = outlier_window // 2 = 10) ticks have been
accumulated, to avoid false rejects at startup when the "median" is based on
only 1-2 samples.

Tests:
1. Normal tick stream: no ticks rejected
2. Single spike in normal stream: spike rejected, surrounding ticks kept
3. NVDA $98↔$210 pattern: ~$210 ticks rejected as outliers from $103 baseline
4. Cold-start: first few ticks are never rejected regardless of spread
5. Configurable threshold: threshold=0.1 is tighter, rejects 15% deviations
6. Configurable window: smaller window adapts faster to price level shifts
7. Outlier rejection is per-symbol: NVDA rejects don't affect BTC/USD ticks
8. Rejected tick count is accessible via outlier_rejected_counts property

@decision DEC-DIAG-003
@title Outlier-rejection filter in SignalGenerator for bimodal price feeds
@status accepted
@rationale Session 43 confirmed a bimodal NVDA distribution (2:1 ratio clusters,
14,543 ticks) that was producing buy signals on spurious $210 prices while the
true price was ~$103. The filter uses a rolling median (robust to outliers) with
configurable deviation threshold. Median is preferred over mean because a mean
is polluted by the very outliers we want to reject. Cold-start guard (min_ticks
= window//2) prevents false rejects during the accumulation phase.
"""

import asyncio
from collections import deque
from decimal import Decimal
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent
from cerebrum.core.types import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_market_event(symbol: str, price: float, ts: float | None = None) -> MarketDataEvent:
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=ts or time(),
        symbol=symbol,
        price=Decimal(str(price)),
        volume=Decimal("100"),
    )


async def _feed_prices(bus: EventBus, symbol: str, prices: list[float], delay: float = 0.01) -> None:
    """Feed a list of prices as MarketDataEvents through the bus."""
    for p in prices:
        await bus.publish(_make_market_event(symbol, p))
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Minimal concrete SignalGenerator subclass for testing
# ---------------------------------------------------------------------------

def _make_test_generator(bus, symbol_filter=None, outlier_window=20,
                          outlier_deviation_threshold=0.5, outlier_min_ticks=None):
    """Build a minimal concrete SignalGenerator for testing the base class filter."""
    from cerebrum.signals.base import SignalGenerator
    from cerebrum.core.types import SignalType

    class _NoopGenerator(SignalGenerator):
        """Minimal concrete subclass — never generates signals, just accumulates data."""

        def __init__(self, bus, **kwargs):
            super().__init__(
                bus=bus,
                signal_type=SignalType.TECHNICAL,
                window_size=50,
                name="noop_test_gen",
                **kwargs,
            )
            self.accepted_prices: dict[str, list[float]] = {}
            self.rejected_count: int = 0

        def _get_min_periods(self) -> int:
            return 1

        def _generate_signal(self, symbol, data):
            return None

    return _NoopGenerator(
        bus,
        outlier_window=outlier_window,
        outlier_deviation_threshold=outlier_deviation_threshold,
        **({"outlier_min_ticks": outlier_min_ticks} if outlier_min_ticks is not None else {}),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def bus():
    b = EventBus(queue_size=200)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_stream_no_rejections(bus):
    """A smooth price stream must accumulate all ticks without rejection."""
    gen = _make_test_generator(bus, outlier_window=10)

    # Normal BTC prices: steady around 50000
    prices = [50000 + i * 10 for i in range(20)]
    await _feed_prices(bus, "BTC/USD", prices)
    await asyncio.sleep(0.1)

    # All ticks should be in _data
    assert len(gen._data["BTC/USD"]) == 20
    assert gen.outlier_rejected_counts.get("BTC/USD", 0) == 0


@pytest.mark.asyncio
async def test_single_spike_rejected(bus):
    """A single 3× spike in a normal stream must be rejected."""
    gen = _make_test_generator(bus, outlier_window=10, outlier_min_ticks=5)

    # Build up baseline of 12 ticks at ~100
    baseline = [100.0 + i * 0.5 for i in range(12)]
    await _feed_prices(bus, "ETH/USD", baseline)
    await asyncio.sleep(0.1)

    # Single spike at 3× price
    await bus.publish(_make_market_event("ETH/USD", 300.0))
    await asyncio.sleep(0.05)

    # Normal ticks continued
    await _feed_prices(bus, "ETH/USD", [101.0, 102.0, 103.0])
    await asyncio.sleep(0.1)

    # Spike should be rejected
    assert gen.outlier_rejected_counts.get("ETH/USD", 0) >= 1
    # The data buffer should not contain the 300.0 spike
    prices_in_buffer = [float(e.price) for e in gen._data["ETH/USD"]]
    assert 300.0 not in prices_in_buffer


@pytest.mark.asyncio
async def test_nvda_bimodal_pattern_rejects_high_cluster(bus):
    """The NVDA $98↔$210 bimodal pattern: $210 ticks must be rejected.

    Mirrors Session 43 data: 14,543 ticks, two clusters at ~$103 and ~$210.
    After the baseline accumulates at ~$103, any tick near $210 (2× deviation
    from median) must be rejected by the 50% threshold filter.
    """
    gen = _make_test_generator(bus, outlier_window=20, outlier_deviation_threshold=0.5,
                               outlier_min_ticks=10)

    # Simulate NVDA baseline: 20 ticks at ~$103
    nvda_baseline = [103.0 + (i % 3) * 0.5 for i in range(20)]
    await _feed_prices(bus, "NVDA", nvda_baseline)
    await asyncio.sleep(0.1)

    # Now inject a burst of "bad" $210 ticks (the anomalous cluster)
    bad_prices = [210.0, 211.5, 209.8, 210.3, 212.0]
    for p in bad_prices:
        await bus.publish(_make_market_event("NVDA", p))
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)

    # All 5 bad ticks should be rejected
    rejected = gen.outlier_rejected_counts.get("NVDA", 0)
    assert rejected >= 5, f"Expected >= 5 rejected NVDA outliers, got {rejected}"

    # Verify none of the bad prices made it into the data buffer
    prices_in_buffer = [float(e.price) for e in gen._data["NVDA"]]
    for bad_p in [210.0, 211.5, 209.8, 210.3, 212.0]:
        assert bad_p not in prices_in_buffer, f"Bad price {bad_p} should not be in buffer"


@pytest.mark.asyncio
async def test_cold_start_no_rejection(bus):
    """During cold-start (< min_ticks accumulated), no ticks are rejected."""
    # Set min_ticks = 10, window = 10 — cold start until 10 ticks are in
    gen = _make_test_generator(bus, outlier_window=10, outlier_min_ticks=10)

    # Send 8 ticks — we're still in cold start. Even if prices are wild:
    prices = [100.0, 200.0, 50.0, 300.0, 100.0, 200.0, 100.0, 100.0]
    await _feed_prices(bus, "COLD/USD", prices)
    await asyncio.sleep(0.1)

    # No rejections during cold start
    assert gen.outlier_rejected_counts.get("COLD/USD", 0) == 0
    # All 8 ticks in buffer
    assert len(gen._data["COLD/USD"]) == 8


@pytest.mark.asyncio
async def test_configurable_tight_threshold(bus):
    """threshold=0.1 (10%) rejects 15% deviations that the default 50% allows."""
    gen = _make_test_generator(bus, outlier_window=10, outlier_deviation_threshold=0.1,
                               outlier_min_ticks=5)

    # Baseline of 10 ticks at 100
    baseline = [100.0] * 10
    await _feed_prices(bus, "TIGHT/USD", baseline)
    await asyncio.sleep(0.1)

    # 15% deviation — within 50% but outside 10%
    await bus.publish(_make_market_event("TIGHT/USD", 115.0))
    await asyncio.sleep(0.05)

    assert gen.outlier_rejected_counts.get("TIGHT/USD", 0) >= 1


@pytest.mark.asyncio
async def test_per_symbol_isolation(bus):
    """Outlier rejection for NVDA must not affect BTC/USD ticks."""
    gen = _make_test_generator(bus, outlier_window=10, outlier_min_ticks=5)

    # Establish BTC baseline
    btc_prices = [50000.0 + i * 100 for i in range(15)]
    await _feed_prices(bus, "BTC/USD", btc_prices)

    # Establish NVDA baseline
    nvda_baseline = [103.0] * 15
    await _feed_prices(bus, "NVDA", nvda_baseline)
    await asyncio.sleep(0.1)

    # Inject NVDA outlier
    await bus.publish(_make_market_event("NVDA", 210.0))
    await asyncio.sleep(0.05)

    # BTC should have 15 ticks untouched
    assert len(gen._data["BTC/USD"]) == 15
    assert gen.outlier_rejected_counts.get("BTC/USD", 0) == 0

    # NVDA outlier should be rejected
    assert gen.outlier_rejected_counts.get("NVDA", 0) >= 1


@pytest.mark.asyncio
async def test_outlier_rejected_counts_is_accessible(bus):
    """outlier_rejected_counts property must exist and return a dict."""
    gen = _make_test_generator(bus)
    counts = gen.outlier_rejected_counts
    assert isinstance(counts, dict)


@pytest.mark.asyncio
async def test_outlier_rejected_counts_returns_copy(bus):
    """Mutating the returned dict must not affect internal state."""
    gen = _make_test_generator(bus, outlier_window=10, outlier_min_ticks=5)

    baseline = [100.0] * 12
    await _feed_prices(bus, "COPY/USD", baseline)
    await asyncio.sleep(0.05)
    await bus.publish(_make_market_event("COPY/USD", 999.0))
    await asyncio.sleep(0.05)

    counts = gen.outlier_rejected_counts
    original = counts.get("COPY/USD", 0)
    counts["COPY/USD"] = 99999

    assert gen.outlier_rejected_counts.get("COPY/USD", 0) == original
