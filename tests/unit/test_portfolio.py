"""
Unit tests for PortfolioTracker closed-trade deque (DEC-CONDUCTOR-008).

Tests verify:
1. _on_fill populates _closed_trades on position close with correct fields.
2. get_closed_trades(window) returns only trades within the window.
3. The deque evicts stale entries on each append.
4. save_snapshot / restore_snapshot round-trips closed_trades correctly.
5. Trades older than the window are not restored (stale-drop on restore).

Real EventBus + real PortfolioTracker — no mocks of internal modules.
FakeClock pattern mirrors test_allocator.py for time-controlled assertions.
"""

import asyncio
import time
from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side
from cerebrum.risk.portfolio import PortfolioTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill(
    symbol: str,
    side: Side,
    amount: Decimal,
    price: Decimal,
    ts: float,
    strategy_id: str | None = "test_strat",
) -> FillEvent:
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=ts,
        order_id=f"ord_{ts}",
        symbol=symbol,
        side=side,
        filled_amount=amount,
        fill_price=price,
        commission=Decimal("0"),
        commission_asset="USD",
        exchange_order_id=f"ex_{ts}",
        strategy_id=strategy_id,
    )


async def _open_then_close(
    bus: EventBus,
    tracker: PortfolioTracker,
    symbol: str = "BTC/USD",
    qty: Decimal = Decimal("1"),
    entry_price: Decimal = Decimal("50000"),
    exit_price: Decimal = Decimal("51000"),
    entry_ts: float = 1_000_000.0,
    exit_ts: float = 1_003_600.0,
    strategy_id: str | None = "test_strat",
) -> None:
    """Publish an open fill then a closing fill; wait for async delivery."""
    buy = _make_fill(symbol, Side.BUY, qty, entry_price, entry_ts, strategy_id)
    await bus.publish(buy)
    await asyncio.sleep(0.05)  # let event deliver

    sell = _make_fill(symbol, Side.SELL, qty, exit_price, exit_ts, strategy_id)
    await bus.publish(sell)
    await asyncio.sleep(0.05)


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def tracker(bus):
    return PortfolioTracker(
        bus=bus,
        initial_balance=Decimal("10000"),
        strategy_id="test_strat",
        sharpe_window_hours=1.0,  # 3600s window for easy math
    )


# ---------------------------------------------------------------------------
# Test 1: _on_fill records a TradeRecord on position close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_records_closed_trade_on_fill(bus, tracker):
    """
    After an open fill + closing fill, exactly one TradeRecord appears in
    _closed_trades with correct symbol, pnl, entry_price, exit_price,
    and strategy_id.
    """
    entry_price = Decimal("50000")
    exit_price = Decimal("51000")
    qty = Decimal("1")
    expected_pnl = qty * (exit_price - entry_price)  # 1000

    await _open_then_close(
        bus, tracker,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        entry_ts=1_000_000.0,
        exit_ts=1_003_600.0,
    )

    trades = tracker.get_closed_trades()
    assert len(trades) == 1, f"Expected 1 closed trade, got {len(trades)}"

    t = trades[0]
    assert t.symbol == "BTC/USD"
    assert t.pnl == expected_pnl, f"Expected pnl={expected_pnl}, got {t.pnl}"
    assert t.entry_price == entry_price
    assert t.exit_price == exit_price
    assert t.strategy_id == "test_strat"
    assert t.status == "closed"


# ---------------------------------------------------------------------------
# Test 2: get_closed_trades(window) filters by entry_time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_closed_trades_window_filters(bus, tracker):
    """
    get_closed_trades(window_seconds) returns only trades whose entry_time
    falls within [now - window_seconds, now].

    We seed two trades: one old (entry_time far in the past relative to
    the window we query) and one recent. The old one should be excluded.
    """
    # Two trades with very different entry timestamps — we control them
    # by directly inserting into the deque via _append_closed_trade to
    # decouple this test from event-bus timing.
    now = time.time()

    # Old trade: entry_time 7200s ago (outside 3600s query window)
    tracker._append_closed_trade(
        symbol="BTC/USD",
        side=Side.BUY,
        entry_time=now - 7200,
        entry_price=Decimal("49000"),
        exit_time=now - 7100,
        exit_price=Decimal("49100"),
        quantity=Decimal("1"),
        pnl=Decimal("100"),
    )

    # Recent trade: entry_time 60s ago (inside 3600s query window)
    tracker._append_closed_trade(
        symbol="BTC/USD",
        side=Side.BUY,
        entry_time=now - 60,
        entry_price=Decimal("50000"),
        exit_time=now - 30,
        exit_price=Decimal("51000"),
        quantity=Decimal("1"),
        pnl=Decimal("1000"),
    )

    # Query with 3600s window — should get only the recent trade
    recent = tracker.get_closed_trades(window_seconds=3600)
    assert len(recent) == 1, f"Expected 1 trade in window, got {len(recent)}"
    assert recent[0].pnl == Decimal("1000")

    # Query with None — returns all (deque bounded by 1h window at construction)
    # The old trade at -7200s is OUTSIDE the 1h deque retention too, so it
    # was evicted when the recent trade was appended.
    all_trades = tracker.get_closed_trades()
    # The old trade (entry_time = now-7200) was beyond the 1h deque window and
    # was evicted when the second trade (now-60) was appended. Only 1 survives.
    assert len(all_trades) == 1


# ---------------------------------------------------------------------------
# Test 3: deque evicts stale entries on append
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_closed_trades_deque_evicts_stale(bus, tracker):
    """
    When a new trade is appended, entries older than _sharpe_window_seconds
    are evicted. The deque stays bounded.
    """
    now = time.time()

    # Fill the deque with 5 old trades (all outside the 1h window)
    for i in range(5):
        tracker._append_closed_trade(
            symbol="BTC/USD",
            side=Side.BUY,
            entry_time=now - 7200 - i,
            entry_price=Decimal("50000"),
            exit_time=now - 7100 - i,
            exit_price=Decimal("50100"),
            quantity=Decimal("1"),
            pnl=Decimal("100"),
        )

    # All 5 are in the deque (eviction only happens on the NEXT append)
    assert len(tracker._closed_trades) == 5

    # Append a fresh trade — should evict all 5 stale ones
    tracker._append_closed_trade(
        symbol="ETH/USD",
        side=Side.BUY,
        entry_time=now - 100,
        entry_price=Decimal("2000"),
        exit_time=now - 50,
        exit_price=Decimal("2100"),
        quantity=Decimal("1"),
        pnl=Decimal("100"),
    )

    assert len(tracker._closed_trades) == 1, (
        f"Expected 1 trade after eviction, got {len(tracker._closed_trades)}"
    )
    assert tracker._closed_trades[0].symbol == "ETH/USD"


# ---------------------------------------------------------------------------
# Test 4: save_snapshot / restore_snapshot round-trips closed_trades
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_snapshot_roundtrip_closed_trades(bus, tracker):
    """
    save_snapshot() includes closed_trades; restore_snapshot() repopulates
    the deque. The round-tripped trade has correct pnl, symbol, and side.
    """
    now = time.time()

    tracker._append_closed_trade(
        symbol="SOL/USD",
        side=Side.BUY,
        entry_time=now - 300,
        entry_price=Decimal("150"),
        exit_time=now - 100,
        exit_price=Decimal("160"),
        quantity=Decimal("10"),
        pnl=Decimal("100"),
    )

    snapshot = tracker.save_snapshot()
    assert "closed_trades" in snapshot
    assert len(snapshot["closed_trades"]) == 1
    assert snapshot["closed_trades"][0]["symbol"] == "SOL/USD"
    assert snapshot["closed_trades"][0]["pnl"] == "100"

    # Create a fresh tracker and restore
    tracker2 = PortfolioTracker(
        bus=bus,
        initial_balance=Decimal("10000"),
        strategy_id="test_strat",
        sharpe_window_hours=1.0,
    )
    tracker2.restore_snapshot(snapshot)

    trades2 = tracker2.get_closed_trades()
    assert len(trades2) == 1
    assert trades2[0].symbol == "SOL/USD"
    assert trades2[0].pnl == Decimal("100")
    assert trades2[0].side == Side.BUY


# ---------------------------------------------------------------------------
# Test 5: restore_snapshot drops trades outside current window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_restore_drops_stale_trades(bus, tracker):
    """
    Trades persisted in a snapshot whose entry_time is older than
    _sharpe_window_seconds are silently dropped on restore, so the deque
    stays bounded even after loading a stale file.
    """
    now = time.time()

    # Manually craft a snapshot with one stale trade (8h old) and one fresh
    stale_ts = now - 8 * 3600  # outside 1h window
    fresh_ts = now - 60

    snapshot = {
        "cash_balance": "10000",
        "peak_equity": "10000",
        "total_realized_pnl": "0",
        "positions": {},
        "closed_trades": [
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "entry_time": stale_ts,
                "entry_price": "49000",
                "exit_time": stale_ts + 100,
                "exit_price": "49100",
                "quantity": "1",
                "pnl": "100",
                "strategy_id": "test_strat",
            },
            {
                "symbol": "ETH/USD",
                "side": "buy",
                "entry_time": fresh_ts,
                "entry_price": "2000",
                "exit_time": fresh_ts + 60,
                "exit_price": "2100",
                "quantity": "1",
                "pnl": "100",
                "strategy_id": "test_strat",
            },
        ],
    }

    tracker2 = PortfolioTracker(
        bus=bus,
        initial_balance=Decimal("10000"),
        strategy_id="test_strat",
        sharpe_window_hours=1.0,
    )
    tracker2.restore_snapshot(snapshot)

    trades = tracker2.get_closed_trades()
    assert len(trades) == 1, f"Expected 1 (stale dropped), got {len(trades)}"
    assert trades[0].symbol == "ETH/USD"
