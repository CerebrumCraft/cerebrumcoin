"""
Unit tests for KrakenXStocksAdapter.

Tests verify constructor, credential handling, ticker publishing,
and symbol format — without a live WebSocket connection.

@decision DEC-TEST-009
@title Test KrakenXStocksAdapter at internal boundary
@status accepted
@rationale Tests call _handle_ticker_update directly to verify MarketDataEvent
publishing without requiring a live Kraken WS connection or valid API keys.
All Kraken SDK construction is patched at the class boundary.
"""

import asyncio
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from cerebrum.adapters.kraken_xstocks import KrakenXStocksAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent
from cerebrum.core.types import EventType


# ---------------------------------------------------------------------------
# Test 1: Constructor stores symbols from config
# ---------------------------------------------------------------------------

def test_constructor_stores_symbols():
    """Constructor reads symbols from config and stores them."""
    bus = EventBus()
    config = {
        "symbols": ["AAPLx/USD", "MSFTx/USD"],
    }

    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "key123", "EXCHANGE_API_SECRET": "secret123"}):
        adapter = KrakenXStocksAdapter(bus, config)

    assert adapter._symbols == ["AAPLx/USD", "MSFTx/USD"]


# ---------------------------------------------------------------------------
# Test 2: Constructor reads credentials from environment
# ---------------------------------------------------------------------------

def test_constructor_reads_credentials_from_env():
    """Constructor pulls API key/secret from os.environ."""
    bus = EventBus()
    config = {"symbols": ["AAPLx/USD"]}

    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "mykey", "EXCHANGE_API_SECRET": "mysecret"}):
        adapter = KrakenXStocksAdapter(bus, config)

    assert adapter._api_key == "mykey"
    assert adapter._api_secret == "mysecret"


# ---------------------------------------------------------------------------
# Test 3: Constructor raises without credentials
# ---------------------------------------------------------------------------

def test_constructor_raises_without_credentials():
    """Constructor raises RuntimeError('kraken_xstocks_credentials_missing') when env vars absent."""
    bus = EventBus()
    config = {"symbols": ["AAPLx/USD"]}

    env_without_creds = {
        k: v for k, v in os.environ.items()
        if k not in ("EXCHANGE_API_KEY", "EXCHANGE_API_SECRET")
    }

    with patch.dict(os.environ, env_without_creds, clear=True):
        with pytest.raises(RuntimeError, match="kraken_xstocks_credentials_missing"):
            KrakenXStocksAdapter(bus, config)


# ---------------------------------------------------------------------------
# Test 4: _handle_ticker_update publishes MarketDataEvent to bus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_ticker_update_publishes_event():
    """_handle_ticker_update builds and publishes a MarketDataEvent."""
    bus = EventBus()
    await bus.start()

    received: list[MarketDataEvent] = []

    async def capture(event: MarketDataEvent) -> None:
        received.append(event)

    bus.subscribe(EventType.MARKET_DATA, capture, "test_capture")

    config = {"symbols": ["AAPLx/USD"]}

    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "k", "EXCHANGE_API_SECRET": "s"}):
        adapter = KrakenXStocksAdapter(bus, config)

    await adapter._handle_ticker_update(
        symbol="AAPLx/USD",
        bid=Decimal("174.50"),
        ask=Decimal("174.55"),
        last=Decimal("174.52"),
        volume=Decimal("123456.0"),
    )

    # Give the event loop a tick to dispatch
    await asyncio.sleep(0)

    assert len(received) == 1
    evt = received[0]
    assert evt.event_type == EventType.MARKET_DATA
    assert evt.symbol == "AAPLx/USD"
    assert evt.bid == Decimal("174.50")
    assert evt.ask == Decimal("174.55")
    assert evt.price == Decimal("174.52")
    assert evt.volume == Decimal("123456.0")

    await bus.stop()


# ---------------------------------------------------------------------------
# Test 5: Symbol normalization preserves AAPLx/USD format
# ---------------------------------------------------------------------------

def test_symbol_format_preserved():
    """xStock symbols like AAPLx/USD are stored without transformation."""
    bus = EventBus()
    symbols = ["AAPLx/USD", "GOOGLx/USD", "TSLAx/USD"]
    config = {"symbols": symbols}

    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "k", "EXCHANGE_API_SECRET": "s"}):
        adapter = KrakenXStocksAdapter(bus, config)

    # All symbols preserved as-is
    assert adapter._symbols == symbols
    for sym in symbols:
        assert sym in adapter._symbols


# ---------------------------------------------------------------------------
# Test 6: Adapter instantiates without error when all deps/creds present
# ---------------------------------------------------------------------------

def test_instantiation_success():
    """Adapter instantiates cleanly given valid env vars and config."""
    bus = EventBus()
    config = {
        "symbols": ["AAPLx/USD"],
    }

    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "valid_key", "EXCHANGE_API_SECRET": "valid_secret"}):
        adapter = KrakenXStocksAdapter(bus, config)

    assert adapter is not None
    assert isinstance(adapter, KrakenXStocksAdapter)
    assert adapter._connected is False
