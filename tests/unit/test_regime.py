"""
Unit tests for regime detection.

Tests the enhanced regime detector with cumulative return, MA slope,
and confidence calculation.

@decision DEC-REGIME-001
@title Test coverage for slow-trend detection (Issue #1 fix)
@status accepted
@rationale The paper trading session exposed that np.mean(returns) misses slow bleeds.
Tests verify: (1) slow downtrend → BEAR, (2) flat oscillation → SIDEWAYS,
(3) confidence derivation, (4) indicator propagation to RegimeChangeEvent.
"""

import asyncio
from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, RegimeChangeEvent
from cerebrum.core.types import EventType
from cerebrum.signals.regime import RegimeDetector


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


def _make_detector(bus, **kwargs):
    """Helper to create detector with test-friendly defaults."""
    defaults = dict(window_size=100, update_interval=10, use_hmm=False)
    defaults.update(kwargs)
    return RegimeDetector(bus, **defaults)


async def _feed_prices(bus, prices, symbol="BTC/USD"):
    """Feed a list of prices through the event bus."""
    for i, price in enumerate(prices):
        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=float(i),
            symbol=symbol,
            price=Decimal(str(price)),
            volume=Decimal("1.0"),
        )
        await bus.publish(event)
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_slow_downtrend_detected_as_bear(event_bus):
    """Slow downtrend (0.01%/step, ~1% cumulative over 100 points) -> BEAR.

    This is THE FIX for Issue #1: the old detector classified this as SIDEWAYS
    because np.mean(returns) averaged out the small individual moves.
    """
    detector = _make_detector(event_bus)

    # 100 prices declining by $5/step from $50000
    # Each step: -5/50000 = -0.0001 (0.01%) -- below old 0.002 threshold
    # Cumulative: -500/50000 = -0.01 (-1%) -- above 0.005 threshold
    prices = [50000 - i * 5 for i in range(100)]
    await _feed_prices(event_bus, prices)

    assert detector._current_regime["BTC/USD"] == "BEAR"


@pytest.mark.asyncio
async def test_strong_uptrend_detected_as_bull(event_bus):
    """Strong uptrend with large per-step moves -> BULL."""
    detector = _make_detector(event_bus)

    # 100 prices rising by $200/step -- clearly above mean_return threshold
    prices = [50000 + i * 200 for i in range(100)]
    await _feed_prices(event_bus, prices)

    assert detector._current_regime["BTC/USD"] == "BULL"


@pytest.mark.asyncio
async def test_flat_market_detected_as_sideways(event_bus):
    """Oscillating prices with no net drift -> SIDEWAYS."""
    detector = _make_detector(event_bus)

    # Oscillate +/- $50 around $50000
    prices = [50000 + (50 if i % 2 == 0 else -50) for i in range(100)]
    await _feed_prices(event_bus, prices)

    assert detector._current_regime["BTC/USD"] == "SIDEWAYS"


@pytest.mark.asyncio
async def test_slow_uptrend_detected_as_bull(event_bus):
    """Slow uptrend (small per-step, large cumulative) -> BULL."""
    detector = _make_detector(event_bus)

    # 100 prices rising by $5/step
    prices = [50000 + i * 5 for i in range(100)]
    await _feed_prices(event_bus, prices)

    assert detector._current_regime["BTC/USD"] == "BULL"


@pytest.mark.asyncio
async def test_confidence_three_metrics_agree(event_bus):
    """Strong downtrend with all 3 metrics agreeing -> confidence >= 0.85."""
    detector = _make_detector(event_bus)

    # Strong downtrend: large per-step, large cumulative, negative slope
    prices = [50000 - i * 200 for i in range(100)]
    await _feed_prices(event_bus, prices)

    # Directly call _detect_regime_rules to check confidence value
    regime, confidence = detector._detect_regime_rules(
        [Decimal(str(p)) for p in prices]
    )
    assert regime == "BEAR"
    assert confidence >= 0.85


@pytest.mark.asyncio
async def test_confidence_slow_trend_high_confidence(event_bus):
    """Slow downtrend detected via cumulative+slope gets high confidence.

    With step=$5 over 100 points, all 3 metrics (mean_return, cumulative, ma_slope)
    agree on the downward direction, so confidence=0.9 even though each individual
    step return is tiny (0.01%). This confirms slow-trend detection is robust and
    produces actionable confidence scores.
    """
    detector = _make_detector(event_bus)

    prices = [50000 - i * 5 for i in range(100)]

    regime, confidence = detector._detect_regime_rules(
        [Decimal(str(p)) for p in prices]
    )
    assert regime == "BEAR"
    # All 3 metrics agree on direction -> confidence = 0.9
    assert confidence >= 0.85


@pytest.mark.asyncio
async def test_regime_event_carries_indicators(event_bus):
    """RegimeChangeEvent should include cumulative_return, ma_slope, etc."""
    detector = _make_detector(event_bus)

    regime_changes = []

    async def collect(event):
        if isinstance(event, RegimeChangeEvent):
            regime_changes.append(event)

    event_bus.subscribe(EventType.REGIME_CHANGE, collect, subscriber_name="test")

    # Feed strong trend to trigger regime change from UNKNOWN
    prices = [50000 + i * 200 for i in range(100)]
    await _feed_prices(event_bus, prices)

    assert len(regime_changes) >= 1
    indicators = regime_changes[-1].indicators
    assert "cumulative_return" in indicators
    assert "ma_slope" in indicators
    assert "volatility" in indicators
    assert "mean_return" in indicators


# --- DEC-REGIME-003: Dual-window tests ---


@pytest.mark.asyncio
async def test_slow_drift_detected_by_long_window():
    """Short window says SIDEWAYS, but long window catches the slow bleed.

    Short window (100 steps at -0.00003/step): cumulative ~-0.3% -- below
    the 0.5% short threshold so short window classifies SIDEWAYS.
    Long window (500 steps): cumulative ~-1.5% -- well above 0.1% long threshold.
    """
    bus = EventBus()
    await bus.start()
    detector = RegimeDetector(
        bus,
        window_size=100,
        long_window_size=500,
        long_cumulative_threshold=0.001,
    )

    base = 70000.0
    prices = [Decimal(str(base * (1 - 0.00003 * i))) for i in range(500)]

    regime, confidence = detector._detect_regime_rules(
        prices[-100:],   # short window — last 100 points (~0.3% drift)
        long_prices=prices,  # long window — all 500 points (~1.5% drift)
    )
    assert regime == "BEAR", f"Expected BEAR, got {regime}"
    assert confidence >= 0.6

    await bus.stop()


@pytest.mark.asyncio
async def test_long_window_no_override_when_short_window_has_trend():
    """When short window already detects BULL/BEAR, long window does not interfere."""
    bus = EventBus()
    await bus.start()
    detector = RegimeDetector(bus, window_size=100, long_window_size=500)

    # Strong uptrend in short window — well above mean_return_threshold
    prices_short = [Decimal(str(70000 + i * 200)) for i in range(100)]
    prices_long = [Decimal(str(70000 + i * 50)) for i in range(500)]

    regime, confidence = detector._detect_regime_rules(
        prices_short, long_prices=prices_long
    )
    assert regime == "BULL"

    await bus.stop()


@pytest.mark.asyncio
async def test_long_window_insufficient_data_no_effect():
    """Long window with fewer than 100 points does not affect classification."""
    bus = EventBus()
    await bus.start()
    detector = RegimeDetector(
        bus,
        window_size=100,
        long_window_size=500,
        long_cumulative_threshold=0.001,
    )

    # Flat short window -> SIDEWAYS, tiny long window (50 pts, below minimum of 100)
    prices_short = [Decimal("70000")] * 100
    prices_long = [Decimal("70000")] * 50  # Below the 100-point minimum

    regime, _ = detector._detect_regime_rules(
        prices_short, long_prices=prices_long
    )
    assert regime == "SIDEWAYS"

    await bus.stop()
