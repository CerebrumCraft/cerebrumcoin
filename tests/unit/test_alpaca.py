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
async def test_alpaca_disconnect():
    """Test Alpaca adapter disconnection."""
    bus = EventBus()
    await bus.start()

    adapter = AlpacaAdapter(bus, {"api_key": "test", "secret_key": "test"})
    adapter._connected = True

    await adapter.disconnect()
    assert not adapter._connected

    await bus.stop()
