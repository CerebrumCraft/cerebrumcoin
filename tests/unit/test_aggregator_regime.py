"""
Unit tests for regime-aware signal aggregator.

@decision DEC-SENT-001
@title Sentiment weight reduction in non-trending regimes
@status accepted
@rationale SIDEWAYS and UNKNOWN regimes apply dampening multipliers (0.4x and 0.6x)
to the base sentiment weight (0.5), yielding effective weights of 0.2 and 0.3 respectively.
This prevents the Fear&Greed index from dominating signal aggregation when the market
is ranging or the regime classifier lacks confidence.
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
