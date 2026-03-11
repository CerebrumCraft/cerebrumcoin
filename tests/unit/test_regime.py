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
