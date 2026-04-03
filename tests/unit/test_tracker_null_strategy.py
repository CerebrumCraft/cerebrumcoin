"""
Test that TradeTracker skips fills with no strategy_id.

@decision DEC-TEST-015
@title Null strategy_id guard in TradeTracker
@status accepted
@rationale 45 orphan trades with strategy_id=NULL caused -$166 in losses.
"""

import time
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_null_strategy_fill_skipped(bus):
    """BUY fill with no strategy_id should not open a trade."""
    from cerebrum.learning.tracker import TradeTracker
    from cerebrum.core.state import StateManager

    # @mock-exempt: StateManager is a SQLite database boundary — in-memory fixture not available for tracker unit tests
    state = AsyncMock(spec=StateManager)
    state.get_open_trades = AsyncMock(return_value=[])
    state.save_trade = AsyncMock(return_value=1)

    tracker = TradeTracker(bus=bus, state=state, current_regime="SIDEWAYS")

    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        filled_amount=Decimal("0.01"),
        fill_price=Decimal("50000"),
        commission=Decimal("0.08"),
        commission_asset="USD",
        strategy_id=None,
    )

    await tracker._on_fill(fill)
    state.save_trade.assert_not_called()
