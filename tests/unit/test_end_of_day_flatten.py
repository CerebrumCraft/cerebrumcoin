"""Unit tests for EndOfDayFlatten (DEC-STOCKS-003)."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, AsyncMock

import pytest

from cerebrum.risk.end_of_day_flatten import EndOfDayFlatten

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def _epoch(dt):
    return dt.timestamp()


class _Position:
    def __init__(self, symbol, amount):
        self.symbol = symbol
        self.amount = Decimal(str(amount))


def _make_flatten(open_positions: list[_Position]):
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    portfolio = MagicMock()
    # get_all_positions() returns dict[symbol, position]
    portfolio.get_all_positions = MagicMock(
        return_value={p.symbol: p for p in open_positions}
    )
    flat = EndOfDayFlatten(
        bus=bus,
        portfolio=portfolio,
        stock_symbols=["AAPL", "MSFT", "NVDA"],
        flatten_offset_minutes=5,
        strategy_id="orb_stocks",
    )
    return flat, bus


def _market_data_at(epoch: float):
    event = MagicMock()
    event.timestamp = epoch
    event.symbol = "AAPL"  # irrelevant — flatten iterates portfolio
    return event


@pytest.mark.asyncio
async def test_fires_close_order_at_1555_with_open_position():
    """Close order emitted at exactly 15:55 ET (close - 5 min) on a normal day."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    # 15:55 ET on 2026-06-15 (EDT = UTC-4) → 19:55 UTC
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 1
    order = bus.publish.await_args[0][0]
    assert order.symbol == "AAPL"
    assert order.metadata.get("source") == "end_of_day_flatten"
    assert order.metadata.get("exit_reason") == "end_of_day_flatten"
    assert order.strategy_id == "orb_stocks"


@pytest.mark.asyncio
async def test_no_fire_at_1554():
    """No close order before the flatten window opens."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    # 15:54 ET on 2026-06-15 → 19:54 UTC
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 54))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_no_fire_with_no_open_positions():
    """No order emitted when portfolio has no open positions."""
    f, bus = _make_flatten([])
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_multiple_positions_each_get_close_order():
    """One close order per open position."""
    f, bus = _make_flatten([_Position("AAPL", "10"), _Position("MSFT", "5")])
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 2
    symbols = {call[0][0].symbol for call in bus.publish.await_args_list}
    assert symbols == {"AAPL", "MSFT"}


@pytest.mark.asyncio
async def test_early_close_day_fires_at_1255():
    """On Black Friday 2026-11-27 (13:00 ET close), flatten at 12:55 ET."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    # 12:55 ET on 2026-11-27. EST = UTC-5. 12:55 ET = 17:55 UTC.
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 11, 27, 17, 55))))
    assert bus.publish.await_count == 1


@pytest.mark.asyncio
async def test_early_close_day_no_fire_at_1254():
    """On early-close day, no fire one minute before the window."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    # 12:54 ET on 2026-11-27 = 17:54 UTC
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 11, 27, 17, 54))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_idempotent_does_not_double_fire():
    """Second tick in the flatten window does not emit a second close order."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    ts = _epoch(_utc(2026, 6, 15, 19, 55))
    await f._on_market_data(_market_data_at(ts))
    await f._on_market_data(_market_data_at(ts + 1))
    assert bus.publish.await_count == 1


@pytest.mark.asyncio
async def test_no_fire_on_holiday():
    """No close order on a NYSE holiday (market is closed)."""
    f, bus = _make_flatten([_Position("AAPL", "10")])
    # 2026-11-26 is Thanksgiving (NYSE holiday). 19:55 UTC would be 14:55 ET.
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 11, 26, 19, 55))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_zero_amount_position_not_closed():
    """A position with amount=0 does not generate a close order."""
    f, bus = _make_flatten([_Position("AAPL", "0")])
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_symbol_not_in_stock_symbols_ignored():
    """Crypto/non-stock positions are ignored even if in portfolio."""
    f, bus = _make_flatten([_Position("BTC/USD", "0.1")])
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 0


@pytest.mark.asyncio
async def test_short_position_emits_buy_to_cover():
    """A short position (negative amount) emits a BUY close order."""
    from cerebrum.core.types import Side
    f, bus = _make_flatten([_Position("AAPL", "-5")])
    await f._on_market_data(_market_data_at(_epoch(_utc(2026, 6, 15, 19, 55))))
    assert bus.publish.await_count == 1
    order = bus.publish.await_args[0][0]
    assert order.side == Side.BUY
    assert order.amount == Decimal("5")
