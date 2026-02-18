"""
Tests for Kraken live order execution.

Mocks ccxt API calls to test live order logic without hitting real exchange.

@decision DEC-TEST-007
@title Mock external APIs in plugin and adapter tests
@status accepted
@rationale Tests mock ccxt at the API boundary (create_market_order, fetch_order) to verify
our safety checks, order submission logic, fill polling, and FillEvent publication without
executing real trades. Environment variables control safety gates.
"""

import asyncio
import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cerebrum.adapters.kraken import KrakenAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent
from cerebrum.core.types import EventType, OrderType, Side


@pytest.mark.asyncio
async def test_live_order_safety_gate_disabled():
    """Test that live orders are blocked when safety gates not enabled."""
    bus = EventBus()
    await bus.start()

    adapter = KrakenAdapter(bus, {"api_key": "test", "api_secret": "test"})
    adapter._exchange = AsyncMock()
    adapter._connected = True

    # Create order
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_order_1",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
    )

    # Execute with safety gates OFF (default)
    with patch.dict(os.environ, {}, clear=True):
        await adapter.execute_order(order)

    # Should NOT have called ccxt
    adapter._exchange.create_market_order.assert_not_called()

    await bus.stop()


@pytest.mark.asyncio
async def test_live_order_safety_gate_partial():
    """Test that both gates must be enabled."""
    bus = EventBus()
    await bus.start()

    adapter = KrakenAdapter(bus, {"api_key": "test", "api_secret": "test"})
    adapter._exchange = AsyncMock()
    adapter._connected = True

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_order_2",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
    )

    # Only TRADING_MODE set
    with patch.dict(os.environ, {"TRADING_MODE": "live"}, clear=True):
        await adapter.execute_order(order)
    adapter._exchange.create_market_order.assert_not_called()

    # Only KRAKEN_LIVE_ENABLED set
    with patch.dict(os.environ, {"KRAKEN_LIVE_ENABLED": "true"}, clear=True):
        await adapter.execute_order(order)
    adapter._exchange.create_market_order.assert_not_called()

    await bus.stop()


@pytest.mark.asyncio
async def test_live_order_execution_success():
    """Test successful live order execution and fill."""
    bus = EventBus()
    await bus.start()

    # Collect published events
    fills = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(EventType.FILL, capture_fill, "test_capture")

    adapter = KrakenAdapter(bus, {"api_key": "test", "api_secret": "test"})
    adapter._connected = True

    # Mock ccxt exchange
    mock_exchange = AsyncMock()
    mock_exchange.create_market_order = AsyncMock(return_value={
        "id": "kraken_order_123",
        "status": "open",
    })
    mock_exchange.fetch_order = AsyncMock(return_value={
        "id": "kraken_order_123",
        "status": "closed",
        "average": 50000.0,
        "filled": 0.01,
        "fee": {"cost": 5.0},
    })
    adapter._exchange = mock_exchange

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_order_3",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
    )

    # Execute with both gates ON
    with patch.dict(os.environ, {"TRADING_MODE": "live", "KRAKEN_LIVE_ENABLED": "true"}, clear=True):
        await adapter.execute_order(order)

    # Let async subscriber process the event
    await asyncio.sleep(0.1)

    # Verify ccxt calls
    mock_exchange.create_market_order.assert_called_once_with(
        symbol="BTC/USD",
        side="buy",
        amount=0.01,
    )
    mock_exchange.fetch_order.assert_called()

    # Verify FillEvent published
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == "test_order_3"
    assert fill.symbol == "BTC/USD"
    assert fill.side == Side.BUY
    assert fill.fill_price == Decimal("50000.0")
    assert fill.filled_amount == Decimal("0.01")
    assert fill.commission == Decimal("5.0")

    await bus.stop()


@pytest.mark.asyncio
async def test_live_order_execution_timeout():
    """Test order fill timeout handling."""
    bus = EventBus()
    await bus.start()

    adapter = KrakenAdapter(bus, {"api_key": "test", "api_secret": "test"})
    adapter._connected = True

    # Mock ccxt - order never fills
    mock_exchange = AsyncMock()
    mock_exchange.create_market_order = AsyncMock(return_value={
        "id": "kraken_order_timeout",
        "status": "open",
    })
    mock_exchange.fetch_order = AsyncMock(return_value={
        "id": "kraken_order_timeout",
        "status": "open",  # Still open after all polls
        "average": None,
        "filled": 0,
    })
    adapter._exchange = mock_exchange

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_order_timeout",
        symbol="BTC/USD",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
    )

    with patch.dict(os.environ, {"TRADING_MODE": "live", "KRAKEN_LIVE_ENABLED": "true"}, clear=True):
        await adapter.execute_order(order)

    # Should have polled multiple times
    assert mock_exchange.fetch_order.call_count == 10

    await bus.stop()


@pytest.mark.asyncio
async def test_live_order_execution_error():
    """Test error handling in live order execution."""
    bus = EventBus()
    await bus.start()

    adapter = KrakenAdapter(bus, {"api_key": "test", "api_secret": "test"})
    adapter._connected = True

    # Mock ccxt - order submission fails
    mock_exchange = AsyncMock()
    mock_exchange.create_market_order = AsyncMock(
        side_effect=Exception("Insufficient funds")
    )
    adapter._exchange = mock_exchange

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_order_error",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("100.0"),  # Too large
    )

    with patch.dict(os.environ, {"TRADING_MODE": "live", "KRAKEN_LIVE_ENABLED": "true"}, clear=True):
        with pytest.raises(Exception, match="Insufficient funds"):
            await adapter.execute_order(order)

    await bus.stop()
