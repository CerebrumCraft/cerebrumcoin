# @mock-exempt: Alpaca API is an external service boundary — mocking TradingClient
# and StockHistoricalDataClient at the API boundary, not internal code.
"""
Tests for Alpaca stock exchange adapter.

Mocks alpaca-py clients to test adapter logic without hitting real exchange.

@decision DEC-TEST-008
@title Mock Alpaca API at client boundary
@status accepted
@rationale Tests mock TradingClient and StockHistoricalDataClient to verify
our adapter logic (connect, market data, order execution, balance queries)
without requiring Alpaca API keys or market hours.
"""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

alpaca = pytest.importorskip("alpaca")

from cerebrum.adapters.alpaca import AlpacaAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent
from cerebrum.core.types import EventType, OrderType, Side


@pytest.fixture
def mock_account():
    """Create a mock Alpaca account (external API object)."""
    account = MagicMock()
    account.equity = "50000.00"
    account.buying_power = "25000.00"
    return account


@pytest.fixture
def mock_trading_client(mock_account):
    """Create a mock Alpaca TradingClient (external API boundary)."""
    client = MagicMock()
    client.get_account.return_value = mock_account
    return client


@pytest.fixture
def mock_data_client():
    """Create a mock Alpaca StockHistoricalDataClient (external API boundary)."""
    return MagicMock()


@pytest.mark.asyncio
async def test_alpaca_connect(mock_trading_client, mock_data_client):
    """Test Alpaca adapter connection."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {
        "api_key": "test_key",
        "secret_key": "test_secret",
        "paper": True,
    })

    # Simulate successful connection
    adapter._trading_client = mock_trading_client
    adapter._data_client = mock_data_client
    adapter._connected = True

    assert adapter._connected
    assert adapter._trading_client is not None

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_get_balance(mock_trading_client):
    """Test getting USD balance from Alpaca."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    balance = await adapter.get_balance("USD")
    assert balance == Decimal("25000.00")

    # Non-USD returns 0
    balance_btc = await adapter.get_balance("BTC")
    assert balance_btc == Decimal("0")

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_get_position(mock_trading_client):
    """Test getting position size from Alpaca."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    # Alpaca position object (external API response)
    mock_position = MagicMock()
    mock_position.qty = "10"
    mock_trading_client.get_open_position.return_value = mock_position

    position = await adapter.get_position("AAPL")
    assert position == Decimal("10")

    # No position returns 0
    mock_trading_client.get_open_position.side_effect = Exception("No position")
    position = await adapter.get_position("MSFT")
    assert position == Decimal("0")

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_get_current_price():
    """Test getting current price from tracked data."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._current_prices["AAPL"] = Decimal("175.50")

    price = await adapter.get_current_price("AAPL")
    assert price == Decimal("175.50")

    # Unknown symbol raises ValueError
    with pytest.raises(ValueError, match="No price data"):
        await adapter.get_current_price("UNKNOWN")

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_execute_order(mock_trading_client):
    """Test order execution via Alpaca Trading API (external boundary)."""
    bus = EventBus()
    await bus.start()

    fills = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(EventType.FILL, capture_fill, "test_capture")

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    # Alpaca order response (external API object)
    mock_order = MagicMock()
    mock_order.id = "alpaca_order_123"
    mock_order.status = "accepted"
    mock_trading_client.submit_order.return_value = mock_order

    # Alpaca filled order response (external API object)
    mock_filled = MagicMock()
    mock_filled.id = "alpaca_order_123"
    mock_filled.status = "filled"
    mock_filled.filled_qty = "10"
    mock_filled.filled_avg_price = "175.25"
    mock_trading_client.get_order_by_id.return_value = mock_filled

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="test_stock_order",
        symbol="AAPL",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("10"),
    )

    await adapter.execute_order(order)
    await asyncio.sleep(0.1)  # Let async subscriber process

    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == "test_stock_order"
    assert fill.symbol == "AAPL"
    assert fill.side == Side.BUY
    assert fill.filled_amount == Decimal("10")
    assert fill.fill_price == Decimal("175.25")
    assert fill.commission == Decimal("0")  # Alpaca is commission-free

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_live_order_requires_dual_env_gate(mock_trading_client, monkeypatch):
    """Live Alpaca orders are blocked unless both live safety env vars are set."""
    bus = EventBus()
    await bus.start()

    monkeypatch.delenv("TRADING_MODE", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_ENABLED", raising=False)

    adapter = AlpacaAdapter(
        bus,
        {"api_key": "test", "secret_key": "test", "paper": False},
    )
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="blocked_live_stock_order",
        symbol="AAPL",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("10"),
    )

    await adapter.execute_order(order)

    mock_trading_client.submit_order.assert_not_called()

    await bus.stop()


# @mock-exempt: Alpaca TradingClient is an external service boundary — mocking at the API layer.
@pytest.mark.asyncio
async def test_alpaca_execute_order_propagates_strategy_id(mock_trading_client):
    """
    Test that strategy_id from OrderEvent is propagated to FillEvent (DEC-ALPACA-FIX-001).

    Multi-strategy routing requires FillEvent.strategy_id to match the originating
    OrderEvent.strategy_id so the Conductor can attribute fills to the correct strategy.
    """
    bus = EventBus()
    await bus.start()

    fills = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(EventType.FILL, capture_fill, "test_capture_strategy")

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    mock_order = MagicMock()
    mock_order.id = "alpaca_order_456"
    mock_order.status = "accepted"
    mock_trading_client.submit_order.return_value = mock_order

    mock_filled = MagicMock()
    mock_filled.id = "alpaca_order_456"
    mock_filled.status = "filled"
    mock_filled.filled_qty = "5"
    mock_filled.filled_avg_price = "200.00"
    mock_trading_client.get_order_by_id.return_value = mock_filled

    # Order with a specific strategy_id
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="order_with_strategy",
        symbol="MSFT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("5"),
        strategy_id="momentum",
    )

    await adapter.execute_order(order)
    await asyncio.sleep(0.1)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.strategy_id == "momentum", (
        "FillEvent.strategy_id must match OrderEvent.strategy_id (DEC-ALPACA-FIX-001)"
    )

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_execute_order_no_strategy_id(mock_trading_client):
    """Test that strategy_id=None is propagated when OrderEvent has no strategy_id."""
    bus = EventBus()
    await bus.start()

    fills = []

    async def capture_fill(event: FillEvent) -> None:
        fills.append(event)

    bus.subscribe(EventType.FILL, capture_fill, "test_capture_none_strategy")

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._trading_client = mock_trading_client
    adapter._connected = True

    mock_order_obj = MagicMock()
    mock_order_obj.id = "alpaca_order_789"
    mock_order_obj.status = "accepted"
    mock_trading_client.submit_order.return_value = mock_order_obj

    mock_filled_obj = MagicMock()
    mock_filled_obj.id = "alpaca_order_789"
    mock_filled_obj.status = "filled"
    mock_filled_obj.filled_qty = "3"
    mock_filled_obj.filled_avg_price = "150.00"
    mock_trading_client.get_order_by_id.return_value = mock_filled_obj

    # Order without strategy_id (default None)
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=0.0,
        order_id="order_no_strategy",
        symbol="AAPL",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("3"),
    )

    await adapter.execute_order(order)
    await asyncio.sleep(0.1)

    assert len(fills) == 1
    assert fills[0].strategy_id is None

    await bus.stop()


@pytest.mark.asyncio
async def test_alpaca_disconnect():
    """Test Alpaca adapter disconnection."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._connected = True

    await adapter.disconnect()
    assert not adapter._connected

    await bus.stop()
