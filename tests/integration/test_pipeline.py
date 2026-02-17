"""
Integration tests for the full event pipeline.

@decision DEC-TEST-005
@title End-to-end pipeline test with mock Kraken data
@status accepted
@rationale Integration test verifies the complete flow: MarketDataEvent → OrderEvent →
FillEvent through the real event bus. Mocks only the external Kraken WebSocket (acceptable
per Sacred Practice #5). Tests prove components integrate correctly via events.
"""

import asyncio
from decimal import Decimal
from pathlib import Path
import tempfile
from time import time

import pytest

from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import EventType, OrderType, Side


@pytest.fixture
async def bus():
    """Create and start an event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def temp_state_file():
    """Create a temporary state file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.mark.asyncio
async def test_full_pipeline_market_data_to_fill(bus, temp_state_file):
    """
    Test complete pipeline: market data → order → fill.

    Simulates:
    1. Market data arrives from exchange
    2. Strategy generates order
    3. Paper executor fills order
    4. Fill event published
    """
    fills = []

    async def fill_collector(event: FillEvent):
        fills.append(event)

    # Subscribe to fill events
    bus.subscribe(EventType.FILL, fill_collector, "fill_collector")

    # Initialize paper trading
    paper = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.16"),
        slippage_percent=Decimal("0.1"),
        state_file=temp_state_file,
    )
    await paper.connect()

    # Step 1: Publish market data (simulates Kraken adapter)
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
        bid=Decimal("49995.0"),
        ask=Decimal("50005.0"),
        spread=Decimal("10.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    # Step 2: Publish order (simulates strategy decision)
    order_event = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="test_order_001",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.05"),
    )
    await bus.publish(order_event)
    await asyncio.sleep(0.2)

    # Step 3: Verify fill was published
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == "test_order_001"
    assert fill.symbol == "BTC/USD"
    assert fill.side == Side.BUY
    assert fill.filled_amount == Decimal("0.05")

    # Verify slippage applied
    expected_fill_price = Decimal("50000.0") * Decimal("1.001")  # 0.1% slippage
    assert fill.fill_price == expected_fill_price

    # Verify position
    position = await paper.get_position("BTC/USD")
    assert position == Decimal("0.05")

    # Verify balance updated
    trade_value = fill.fill_price * fill.filled_amount
    expected_balance = Decimal("10000.0") - trade_value - fill.commission
    actual_balance = await paper.get_balance("USD")
    assert actual_balance == expected_balance

    await paper.disconnect()


@pytest.mark.asyncio
async def test_pipeline_multiple_trades(bus, temp_state_file):
    """Test pipeline with multiple sequential trades."""
    fills = []

    async def fill_collector(event: FillEvent):
        fills.append(event)

    bus.subscribe(EventType.FILL, fill_collector, "fill_collector")

    paper = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )
    await paper.connect()

    # Trade 1: Buy BTC
    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    ))
    await asyncio.sleep(0.1)

    await bus.publish(OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_1",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    ))
    await asyncio.sleep(0.2)

    # Trade 2: Buy ETH
    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="ETH/USD",
        price=Decimal("3000.0"),
        volume=Decimal("500.0"),
    ))
    await asyncio.sleep(0.1)

    await bus.publish(OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_2",
        symbol="ETH/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1.0"),
    ))
    await asyncio.sleep(0.2)

    # Trade 3: Sell some BTC
    await bus.publish(OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_3",
        symbol="BTC/USD",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("0.05"),
    ))
    await asyncio.sleep(0.2)

    # Verify all fills
    assert len(fills) == 3

    # Verify positions
    btc_position = await paper.get_position("BTC/USD")
    assert btc_position == Decimal("0.05")  # 0.1 bought - 0.05 sold

    eth_position = await paper.get_position("ETH/USD")
    assert eth_position == Decimal("1.0")

    # Verify portfolio summary
    summary = paper.get_portfolio_summary()
    assert summary["trade_count"] == 3
    assert "BTC/USD" in summary["positions"]
    assert "ETH/USD" in summary["positions"]

    await paper.disconnect()


@pytest.mark.asyncio
async def test_pipeline_concurrent_events(bus, temp_state_file):
    """Test that pipeline handles concurrent events correctly."""
    fills = []

    async def fill_collector(event: FillEvent):
        fills.append(event)

    bus.subscribe(EventType.FILL, fill_collector, "fill_collector")

    paper = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )
    await paper.connect()

    # Publish market data for multiple symbols concurrently
    market_events = [
        MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol="BTC/USD",
            price=Decimal("50000.0"),
            volume=Decimal("100.0"),
        ),
        MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol="ETH/USD",
            price=Decimal("3000.0"),
            volume=Decimal("500.0"),
        ),
    ]

    for event in market_events:
        await bus.publish(event)

    await asyncio.sleep(0.1)

    # Both prices should be tracked
    btc_price = await paper.get_current_price("BTC/USD")
    eth_price = await paper.get_current_price("ETH/USD")

    assert btc_price == Decimal("50000.0")
    assert eth_price == Decimal("3000.0")

    await paper.disconnect()


@pytest.mark.asyncio
async def test_pipeline_event_isolation(bus, temp_state_file):
    """Test that one subscriber's failure doesn't affect others."""
    fills = []
    errors_caught = []

    async def fill_collector(event: FillEvent):
        fills.append(event)

    async def failing_subscriber(event: FillEvent):
        errors_caught.append("entered")
        raise ValueError("Simulated error")

    # Subscribe both handlers
    bus.subscribe(EventType.FILL, fill_collector, "good_subscriber")
    bus.subscribe(EventType.FILL, failing_subscriber, "failing_subscriber")

    paper = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )
    await paper.connect()

    # Execute trade
    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    ))
    await asyncio.sleep(0.1)

    await bus.publish(OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_1",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    ))
    await asyncio.sleep(0.2)

    # Good subscriber should still receive fill despite failing subscriber
    assert len(fills) == 1
    assert len(errors_caught) == 1  # Failing subscriber was called

    await paper.disconnect()
