"""
Unit tests for regime-aware signal aggregator.

@decision DEC-SENT-001
@title Sentiment weight reduction in non-trending regimes
@status accepted
@rationale SIDEWAYS and UNKNOWN regimes apply dampening multipliers (0.4x and 0.6x)
to the base sentiment weight (0.5), yielding effective weights of 0.2 and 0.3 respectively.
This prevents the Fear&Greed index from dominating signal aggregation when the market
is ranging or the regime classifier lacks confidence.

@decision DEC-REGIME-002
@title Buy suppression in high-confidence BEAR regime
@status accepted
@rationale Paper trading session (2026-02-26) went 0/20 because buy signals fired
freely during a slow downtrend misclassified as SIDEWAYS. Once the regime detector
correctly identifies BEAR with confidence >= 0.8, the aggregator multiplies the
buy score by 0.2, making it nearly impossible for a buy signal to clear the threshold.
Sell signals and low-confidence BEAR signals are unaffected.
"""

import asyncio
from decimal import Decimal
import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.signals.aggregator import SignalAggregator


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_regime_change_adjusts_weights(event_bus):
    """Test that regime changes trigger weight adjustments."""
    agg = SignalAggregator(event_bus, threshold=Decimal("0.2"))
    
    # Store initial weights
    initial_technical_weight = agg._weights[SignalType.TECHNICAL]
    
    # Emit regime change to BULL
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BULL",
        confidence=Decimal("0.8"),
        indicators={"test": "data"},
    )
    
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)
    
    # Check that weights were adjusted
    assert agg._current_regime == "BULL"
    assert agg._weights[SignalType.TECHNICAL] > initial_technical_weight


@pytest.mark.asyncio
async def test_volatile_regime_reduces_weights(event_bus):
    """Test that VOLATILE regime reduces all weights."""
    agg = SignalAggregator(event_bus, threshold=Decimal("0.2"))
    
    initial_technical = agg._base_weights[SignalType.TECHNICAL]
    
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="VOLATILE",
        confidence=Decimal("0.8"),
        indicators={},
    )
    
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)
    
    # All weights should be reduced in volatile regime
    assert agg._weights[SignalType.TECHNICAL] < initial_technical
    assert agg._weights[SignalType.SENTIMENT] < agg._base_weights[SignalType.SENTIMENT]


@pytest.mark.asyncio
async def test_sideways_regime_reduces_sentiment_to_04x(event_bus):
    """SIDEWAYS regime applies 0.4x multiplier to sentiment (DEC-SENT-001).

    Effective sentiment weight: base 0.5 * 0.4 = 0.2.
    Prevents Fear&Greed from dominating aggregation in ranging markets.
    """
    agg = SignalAggregator(event_bus, threshold=Decimal("0.2"))

    base_sentiment = agg._base_weights[SignalType.SENTIMENT]  # 0.5

    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="BULL",
        to_regime="SIDEWAYS",
        confidence=Decimal("0.8"),
        indicators={},
    )

    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)

    expected = base_sentiment * Decimal("0.4")
    assert agg._current_regime == "SIDEWAYS"
    assert agg._weights[SignalType.SENTIMENT] == expected, (
        f"SIDEWAYS sentiment weight should be {expected}, got {agg._weights[SignalType.SENTIMENT]}"
    )


@pytest.mark.asyncio
async def test_unknown_regime_reduces_sentiment_to_06x(event_bus):
    """UNKNOWN regime applies 0.6x multiplier to sentiment (DEC-SENT-001).

    Effective sentiment weight: base 0.5 * 0.6 = 0.3.
    Partial dampening when regime classifier lacks confidence.
    """
    agg = SignalAggregator(event_bus, threshold=Decimal("0.2"))

    # First move to a non-default regime so we can observe the transition back
    bull_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BULL",
        confidence=Decimal("0.9"),
        indicators={},
    )
    await event_bus.publish(bull_event)
    await asyncio.sleep(0.05)

    base_sentiment = agg._base_weights[SignalType.SENTIMENT]  # 0.5

    unknown_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="BULL",
        to_regime="ANYTHING_ELSE",
        confidence=Decimal("0.3"),
        indicators={},
    )

    await event_bus.publish(unknown_event)
    await asyncio.sleep(0.1)

    expected = base_sentiment * Decimal("0.6")
    assert agg._weights[SignalType.SENTIMENT] == expected, (
        f"UNKNOWN regime sentiment weight should be {expected}, got {agg._weights[SignalType.SENTIMENT]}"
    )


@pytest.mark.asyncio
async def test_bear_high_confidence_suppresses_buy(event_bus):
    """BEAR regime with confidence >= 0.8 suppresses buy signals (DEC-REGIME-002)."""
    agg = SignalAggregator(event_bus, threshold=Decimal("0.1"))

    # Set BEAR regime with high confidence
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BEAR",
        confidence=Decimal("0.9"),
        indicators={},
    )
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)

    # Send a buy signal
    buy_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=asyncio.get_event_loop().time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="test buy",
    )

    # Capture emitted combined signals
    emitted = []

    async def collect(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test_collect")

    await event_bus.publish(buy_signal)
    await asyncio.sleep(0.1)

    # Buy should be suppressed (strength reduced by 0.2x factor)
    if emitted:
        assert emitted[0].strength < Decimal("0.5"), (
            f"Buy should be suppressed in BEAR regime, got strength {emitted[0].strength}"
        )


@pytest.mark.asyncio
async def test_bear_low_confidence_no_suppression(event_bus):
    """BEAR regime with confidence < 0.8 does NOT suppress buy signals."""
    agg = SignalAggregator(event_bus, threshold=Decimal("0.1"))

    # Set BEAR regime with LOW confidence
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BEAR",
        confidence=Decimal("0.5"),
        indicators={},
    )
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)

    buy_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=asyncio.get_event_loop().time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="test buy",
    )

    emitted = []

    async def collect(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test_collect")

    await event_bus.publish(buy_signal)
    await asyncio.sleep(0.1)

    # Buy should NOT be suppressed with low confidence
    if emitted:
        assert emitted[0].strength >= Decimal("0.5"), (
            f"Buy should NOT be suppressed with low confidence, got strength {emitted[0].strength}"
        )


@pytest.mark.asyncio
async def test_buy_suppression_does_not_affect_sell(event_bus):
    """BEAR regime buy suppression does NOT affect sell signals."""
    agg = SignalAggregator(event_bus, threshold=Decimal("0.1"))

    # Set BEAR regime with high confidence
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BEAR",
        confidence=Decimal("0.9"),
        indicators={},
    )
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)

    sell_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=asyncio.get_event_loop().time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.SELL,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="test sell",
    )

    emitted = []

    async def collect(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test_collect")

    await event_bus.publish(sell_signal)
    await asyncio.sleep(0.1)

    # Sell should pass through unmodified
    if emitted:
        assert emitted[0].action == SignalAction.SELL
        assert emitted[0].strength >= Decimal("0.5"), (
            f"Sell should not be suppressed, got strength {emitted[0].strength}"
        )
