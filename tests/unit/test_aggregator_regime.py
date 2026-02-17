"""
Unit tests for regime-aware signal aggregator.
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
