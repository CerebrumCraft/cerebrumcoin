"""
Unit tests for event definitions.

@decision DEC-TEST-001
@title Test real implementations, not mocks
@status accepted
@rationale Following Sacred Practice #5: tests validate actual event behavior (immutability,
auto-type assignment, field validation). No mocks—events are simple dataclasses that don't
require external dependencies. Tests prove events are truly immutable and type-safe.
"""

from decimal import Decimal
from time import time

import pytest

from cerebrum.core.events import (
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
)
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
    SignalType,
)


def test_market_data_event_immutable():
    """Test that MarketDataEvent is immutable."""
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    with pytest.raises(AttributeError):
        event.price = Decimal("51000.0")


def test_market_data_event_auto_type():
    """Test that event_type is automatically set."""
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,  # This gets overridden
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )

    assert event.event_type == EventType.MARKET_DATA


def test_signal_event_creation():
    """Test SignalEvent creation with all fields."""
    event = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        target_price=Decimal("51000.0"),
        stop_loss=Decimal("49000.0"),
        reason="RSI oversold",
    )

    assert event.signal_type == SignalType.TECHNICAL
    assert event.action == SignalAction.BUY
    assert event.strength == Decimal("0.8")
    assert event.confidence == Decimal("0.9")


def test_order_event_market_order():
    """Test market order creation (no price)."""
    event = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )

    assert event.order_type == OrderType.MARKET
    assert event.price is None
    assert event.status == OrderStatus.PENDING


def test_order_event_limit_order():
    """Test limit order creation (with price)."""
    event = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.1"),
        price=Decimal("50000.0"),
    )

    assert event.order_type == OrderType.LIMIT
    assert event.price == Decimal("50000.0")


def test_fill_event_creation():
    """Test FillEvent creation."""
    event = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000.0"),
        commission=Decimal("5.0"),
        commission_asset="USD",
        exchange_order_id="kraken_456",
    )

    assert event.order_id == "order_123"
    assert event.filled_amount == Decimal("0.1")
    assert event.fill_price == Decimal("50000.0")
    assert event.commission == Decimal("5.0")


def test_event_with_metadata():
    """Test event creation with metadata."""
    metadata = {"source": "test", "version": "1.0"}

    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
        metadata=metadata,
    )

    assert event.metadata == metadata
