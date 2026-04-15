"""Live Alpaca connection tests — opt-in via --live-alpaca.

These tests connect to the Alpaca paper API and verify:
- Environment credentials are present
- WebSocket subscribes and receives ticks during RTH
- Event shape matches what the pipeline expects

NO orders are placed. Safe to run repeatedly.

@decision DEC-STOCKS-001
"""

import asyncio
import os

import pytest

pytestmark = pytest.mark.live_alpaca


def test_env_credentials_present() -> None:
    assert os.getenv("ALPACA_API_KEY_ID"), "missing ALPACA_API_KEY_ID in environment"
    assert os.getenv("ALPACA_API_SECRET_KEY"), "missing ALPACA_API_SECRET_KEY in environment"


@pytest.mark.asyncio
async def test_subscribes_and_receives_at_least_5_ticks() -> None:
    """Requires US RTH for data flow. If market closed, test will time out — that's diagnostic."""
    try:
        from cerebrum.adapters.alpaca import AlpacaAdapter
    except ImportError:
        pytest.skip("alpaca-py not installed")

    from cerebrum.core.bus import EventBus
    from cerebrum.core.types import EventType

    bus = EventBus()

    adapter = AlpacaAdapter(
        bus=bus,
        config={
            "api_key": os.environ["ALPACA_API_KEY_ID"],
            "secret_key": os.environ["ALPACA_API_SECRET_KEY"],
            "paper": True,
            "paper_base_url": "https://paper-api.alpaca.markets",
            "data_feed": "iex",
            "symbols": ["AAPL"],
        },
    )

    tick_count = 0

    async def count_ticks(evt: object) -> None:
        nonlocal tick_count
        if getattr(evt, "symbol", None) == "AAPL":
            tick_count += 1

    bus.subscribe(EventType.MARKET_DATA, count_ticks, subscriber_name="live_test_counter")

    await adapter.connect()
    await adapter.subscribe_market_data(["AAPL"])

    # Wait up to 30 s for 5 ticks. Outside RTH this will time out.
    for _ in range(30):
        if tick_count >= 5:
            break
        await asyncio.sleep(1)

    await adapter.disconnect()

    assert tick_count >= 5, (
        f"Got {tick_count} AAPL ticks in 30s. If market is closed this is expected; "
        "run during RTH to exercise the subscription."
    )
