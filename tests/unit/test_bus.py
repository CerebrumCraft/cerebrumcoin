"""
Unit tests for the event bus.

@decision DEC-TEST-002
@title Async test fixtures for event bus validation
@status accepted
@rationale Event bus is async—tests must be async to verify queue behavior, subscriber
isolation, and graceful shutdown. Uses pytest-asyncio for async test support. No mocks—
tests validate real async queue behavior and concurrent event delivery.
"""

import asyncio
from decimal import Decimal
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType


@pytest.fixture
async def bus():
    """Create and start an event bus for testing."""
    bus = EventBus(queue_size=10)
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_bus_start_stop():
    """Test event bus lifecycle."""
    bus = EventBus()
    assert not bus._running

    await bus.start()
    assert bus._running

    await bus.stop()
    assert not bus._running


@pytest.mark.asyncio
async def test_publish_with_no_subscribers(bus):
    """Test publishing to an event type with no subscribers."""
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    # Should not raise
    await bus.publish(event)


@pytest.mark.asyncio
async def test_single_subscriber(bus):
    """Test single subscriber receives events."""
    received_events = []

    async def handler(event):
        received_events.append(event)

    bus.subscribe(EventType.MARKET_DATA, handler, "test_subscriber")

    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    await bus.publish(event)

    # Give handler time to process
    await asyncio.sleep(0.1)

    assert len(received_events) == 1
    assert received_events[0].symbol == "BTC/USD"
    assert received_events[0].price == Decimal("50000.0")


@pytest.mark.asyncio
async def test_multiple_subscribers(bus):
    """Test multiple subscribers receive the same event."""
    received_1 = []
    received_2 = []

    async def handler_1(event):
        received_1.append(event)

    async def handler_2(event):
        received_2.append(event)

    bus.subscribe(EventType.MARKET_DATA, handler_1, "subscriber_1")
    bus.subscribe(EventType.MARKET_DATA, handler_2, "subscriber_2")

    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(received_1) == 1
    assert len(received_2) == 1
    assert received_1[0].symbol == received_2[0].symbol


@pytest.mark.asyncio
async def test_subscriber_isolation(bus):
    """Test that subscriber failures don't affect others."""
    received_good = []

    async def failing_handler(event):
        raise ValueError("Handler error")

    async def good_handler(event):
        received_good.append(event)

    bus.subscribe(EventType.MARKET_DATA, failing_handler, "failing_subscriber")
    bus.subscribe(EventType.MARKET_DATA, good_handler, "good_subscriber")

    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    # Good handler should still receive event despite failing handler
    assert len(received_good) == 1


@pytest.mark.asyncio
async def test_type_filtering(bus):
    """Test that subscribers only receive events of subscribed type."""
    market_data_received = []
    signal_received = []

    async def market_handler(event):
        market_data_received.append(event)

    async def signal_handler(event):
        signal_received.append(event)

    bus.subscribe(EventType.MARKET_DATA, market_handler, "market_subscriber")
    bus.subscribe(EventType.SIGNAL, signal_handler, "signal_subscriber")

    # Publish market data
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)

    # Publish signal
    signal_event = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )
    await bus.publish(signal_event)

    await asyncio.sleep(0.1)

    # Each subscriber should only receive events of their type
    assert len(market_data_received) == 1
    assert len(signal_received) == 1
    assert isinstance(market_data_received[0], MarketDataEvent)
    assert isinstance(signal_received[0], SignalEvent)


@pytest.mark.asyncio
async def test_get_subscriber_count(bus):
    """Test subscriber count tracking."""
    async def dummy_handler(event):
        pass

    assert bus.get_subscriber_count(EventType.MARKET_DATA) == 0

    bus.subscribe(EventType.MARKET_DATA, dummy_handler, "sub1")
    assert bus.get_subscriber_count(EventType.MARKET_DATA) == 1

    bus.subscribe(EventType.MARKET_DATA, dummy_handler, "sub2")
    assert bus.get_subscriber_count(EventType.MARKET_DATA) == 2

    assert bus.get_subscriber_count(EventType.SIGNAL) == 0


@pytest.mark.asyncio
async def test_graceful_shutdown(bus):
    """Test that shutdown waits for tasks to complete."""
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(EventType.MARKET_DATA, handler, "subscriber")

    # Give subscriber task time to start
    await asyncio.sleep(0.05)

    # Publish events
    for i in range(3):
        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol="BTC/USD",
            price=Decimal(str(50000 + i)),
            volume=Decimal("100.0"),
        )
        await bus.publish(event)

    # Give time for processing
    await asyncio.sleep(0.2)

    # Stop bus
    await bus.stop()

    # Events should be processed
    assert len(received) == 3
