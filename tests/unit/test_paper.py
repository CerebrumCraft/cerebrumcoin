"""
Unit tests for paper trading adapter.

@decision DEC-TEST-004
@title Test paper trading with real event bus integration
@status accepted
@rationale Paper adapter tests verify order execution, balance tracking, and commission
calculation against a real event bus. No mocks—tests use actual EventBus and verify
events flow correctly (OrderEvent → FillEvent). State persistence tested with temp files.
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
async def test_paper_adapter_initialization(bus, temp_state_file):
    """Test paper adapter initializes with correct balance."""
    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter.connect()

    balance = await adapter.get_balance("USD")
    assert balance == Decimal("10000.0")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_adapter_tracks_prices(bus, temp_state_file):
    """Test that adapter tracks market data prices."""
    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter.connect()

    # Publish market data
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    # Check price is tracked
    price = await adapter.get_current_price("BTC/USD")
    assert price == Decimal("50000.0")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_adapter_buy_order(bus, temp_state_file):
    """Test buy order execution with slippage and commission."""
    fills = []

    async def fill_handler(event: FillEvent):
        fills.append(event)

    bus.subscribe(EventType.FILL, fill_handler, "test_fill_subscriber")

    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),  # 0.1%
        slippage_percent=Decimal("0.05"),   # 0.05%
        state_file=temp_state_file,
    )

    await adapter.connect()

    # Publish market price
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    # Submit buy order
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )
    await bus.publish(order)
    await asyncio.sleep(0.2)

    # Verify fill event
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == "order_123"
    assert fill.filled_amount == Decimal("0.1")

    # Verify fill price has slippage (50000 * 1.0005 = 50025)
    expected_price = Decimal("50000.0") * Decimal("1.0005")
    assert fill.fill_price == expected_price

    # Verify commission
    trade_value = expected_price * Decimal("0.1")
    expected_commission = trade_value * Decimal("0.001")
    assert fill.commission == expected_commission

    # Verify balances
    usd_balance = await adapter.get_balance("USD")
    expected_usd = Decimal("10000.0") - trade_value - expected_commission
    assert usd_balance == expected_usd

    # Verify position
    position = await adapter.get_position("BTC/USD")
    assert position == Decimal("0.1")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_adapter_sell_order(bus, temp_state_file):
    """Test sell order execution."""
    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter.connect()

    # Set price
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    # Buy first to have position
    buy_order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="buy_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )
    await bus.publish(buy_order)
    await asyncio.sleep(0.2)

    # Now sell
    sell_order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="sell_123",
        symbol="BTC/USD",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("0.05"),
    )
    await bus.publish(sell_order)
    await asyncio.sleep(0.2)

    # Verify position reduced
    position = await adapter.get_position("BTC/USD")
    assert position == Decimal("0.05")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_adapter_insufficient_balance(bus, temp_state_file):
    """Test that orders fail with insufficient balance."""
    fills = []

    async def fill_handler(event: FillEvent):
        fills.append(event)

    bus.subscribe(EventType.FILL, fill_handler, "test_fill_subscriber")

    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("100.0"),  # Small balance
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter.connect()

    # Set price
    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    # Try to buy too much
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1.0"),  # Requires ~50,000 USD
    )
    await bus.publish(order)
    await asyncio.sleep(0.2)

    # Should not fill
    assert len(fills) == 0

    # Balance unchanged
    balance = await adapter.get_balance("USD")
    assert balance == Decimal("100.0")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_adapter_state_persistence(bus, temp_state_file):
    """Test that state persists across restarts."""
    # First session: make a trade
    adapter1 = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter1.connect()

    market_event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000.0"),
        volume=Decimal("100.0"),
    )
    await bus.publish(market_event)
    await asyncio.sleep(0.1)

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="order_123",
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )
    await bus.publish(order)
    await asyncio.sleep(0.2)

    balance1 = await adapter1.get_balance("USD")
    position1 = await adapter1.get_position("BTC/USD")

    await adapter1.disconnect()

    # Second session: load state
    adapter2 = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000.0"),  # This should be overridden by saved state
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=temp_state_file,
    )

    await adapter2.connect()

    balance2 = await adapter2.get_balance("USD")
    position2 = await adapter2.get_position("BTC/USD")

    # State should match
    assert balance2 == balance1
    assert position2 == position1

    await adapter2.disconnect()
