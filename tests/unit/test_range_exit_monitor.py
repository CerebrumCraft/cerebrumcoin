"""
Unit tests for RangeExitMonitor.

Tests cover structural S/R exits, regime invalidation exits, time-based exits,
the mid-range no-exit guard, and fallback percentage exits when no confirmed
range exists. All tests use real EventBus, PortfolioTracker, and RangeDetector
instances — no internal module mocking.

@decision DEC-TEST-012
@title Test RangeExitMonitor with real bus, portfolio, and range_detector
@status accepted
@rationale RangeExitMonitor behaviour depends on the interplay of three
components (EventBus, PortfolioTracker, RangeDetector). Using real
implementations validates the full event-driven flow and ensures the
components compose correctly. RangeDetector internal state is seeded
directly (its _ranges dict) to avoid re-implementing signal accumulation
in test fixtures.
"""

import asyncio
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent, RegimeChangeEvent
from cerebrum.core.types import EventType, Side
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.range_exit_monitor import RangeExitMonitor
from cerebrum.strategies.range_detector import RangeDetector, RangeState, _SymbolRange


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start a real EventBus."""
    b = EventBus(queue_size=200)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def portfolio(bus):
    """PortfolioTracker with $10 000 initial balance."""
    return PortfolioTracker(bus, initial_balance=Decimal("10000.0"))


@pytest.fixture
async def range_detector(bus):
    """RangeDetector with subscriptions started."""
    rd = RangeDetector(bus)
    await rd.start()
    return rd


@pytest.fixture
async def monitor(bus, portfolio, range_detector):
    """
    RangeExitMonitor wired to bus, portfolio, and range_detector.

    breakdown_margin_pct=0.3 is intentionally less than the RangeDetector's
    default breakout_margin_pct=0.5. This ensures the monitor triggers a
    support_breakdown exit before the RangeDetector clears the range on a
    price breakout. If both thresholds were equal, the RangeDetector's
    MARKET_DATA handler (which runs before the monitor's) would delete the
    range, leaving the monitor with no structural reference and causing it to
    fall back to percentage-based exits instead.
    """
    return RangeExitMonitor(
        bus=bus,
        portfolio=portfolio,
        range_detector=range_detector,
        resistance_proximity_pct=Decimal("0.3"),
        breakdown_margin_pct=Decimal("0.3"),
        max_hold_minutes=60,
        fallback_tp_pct=Decimal("1.0"),
        fallback_sl_pct=Decimal("0.8"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _open_position(
    bus: EventBus,
    symbol: str,
    amount: Decimal,
    price: Decimal,
    entry_time: float | None = None,
) -> None:
    """Send a BUY FillEvent to open a position in PortfolioTracker."""
    ts = entry_time if entry_time is not None else time.time()
    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=ts,
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        filled_amount=amount,
        fill_price=price,
        commission=Decimal("0.0"),
        commission_asset="USD",
    )
    await bus.publish(fill)
    await asyncio.sleep(0.1)


async def _send_price(bus: EventBus, symbol: str, price: Decimal, ts: float | None = None) -> None:
    """Publish a MarketDataEvent for the given symbol/price."""
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=ts if ts is not None else time.time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )
    await bus.publish(event)
    await asyncio.sleep(0.1)


def _seed_confirmed_range(
    range_detector: RangeDetector,
    symbol: str,
    support: Decimal,
    resistance: Decimal,
) -> None:
    """
    Directly seed RangeDetector's internal state so get_range() returns a
    confirmed range without replaying signal events.
    """
    state = _SymbolRange(
        support_level=support,
        resistance_level=resistance,
        support_bounces=2,
        resistance_bounces=2,
        in_support_proximity=False,
        in_resistance_proximity=False,
        last_updated=time.time(),
        last_price=None,
    )
    range_detector._ranges[symbol] = state
    # Also update the current regime so staleness check passes
    range_detector._current_regime = "SIDEWAYS"


def _capture_orders(bus: EventBus) -> list[OrderEvent]:
    """
    Subscribe to ORDER events and return a list that accumulates published
    OrderEvents during the test.
    """
    captured: list[OrderEvent] = []

    async def _handler(event: object) -> None:
        if isinstance(event, OrderEvent):
            captured.append(event)

    bus.subscribe(EventType.ORDER, _handler, subscriber_name="test_order_capture")
    return captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resistance_exit(bus, portfolio, range_detector, monitor):
    """
    When price reaches within proximity_pct of resistance, a SELL is emitted.
    Resistance = 70 000, proximity = 0.3% → threshold = 70 000 * 0.997 = 69 790.
    Any price >= 69 790 should trigger.
    """
    orders = _capture_orders(bus)
    symbol = "BTC/USD"
    support = Decimal("68000")
    resistance = Decimal("70000")
    _seed_confirmed_range(range_detector, symbol, support, resistance)

    await _open_position(bus, symbol, Decimal("0.1"), Decimal("69000"))
    # Price at 99.8% of resistance — above the 99.7% threshold
    await _send_price(bus, symbol, Decimal("69850"))

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    order = orders[0]
    assert order.side == Side.SELL
    assert order.symbol == symbol
    assert "resistance_exit" in order.metadata["exit_reason"]


@pytest.mark.asyncio
async def test_support_breakdown_exit(bus, portfolio, range_detector, monitor):
    """
    When price falls below support by breakdown_margin_pct, a SELL is emitted.
    Support = 68 000, margin = 0.5% → threshold = 68 000 * 0.995 = 67 660.
    Any price <= 67 660 should trigger.
    """
    orders = _capture_orders(bus)
    symbol = "ETH/USD"
    support = Decimal("3000")
    resistance = Decimal("3200")
    _seed_confirmed_range(range_detector, symbol, support, resistance)

    await _open_position(bus, symbol, Decimal("1.0"), Decimal("3100"))
    # Breakdown price: 0.4% below support (3000 * 0.996 = 2988).
    # This crosses the monitor's breakdown_margin_pct=0.3% threshold (2991)
    # but stays above the RangeDetector's breakout_margin_pct=0.5% threshold
    # (2985), so the range is still present when the monitor checks it.
    breakdown_price = Decimal("2988")
    await _send_price(bus, symbol, breakdown_price)

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    order = orders[0]
    assert order.side == Side.SELL
    assert order.symbol == symbol
    assert "support_breakdown" in order.metadata["exit_reason"]


@pytest.mark.asyncio
async def test_regime_change_exit(bus, portfolio, range_detector, monitor):
    """
    When a REGIME_CHANGE event fires from SIDEWAYS to BEAR, SELL orders are
    emitted for all open positions.
    """
    orders = _capture_orders(bus)
    _seed_confirmed_range(range_detector, "BTC/USD", Decimal("68000"), Decimal("70000"))
    _seed_confirmed_range(range_detector, "ETH/USD", Decimal("3000"), Decimal("3200"))

    await _open_position(bus, "BTC/USD", Decimal("0.1"), Decimal("69000"))
    await _open_position(bus, "ETH/USD", Decimal("1.0"), Decimal("3100"))

    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=time.time(),
        from_regime="SIDEWAYS",
        to_regime="BEAR",
        confidence=Decimal("0.85"),
        indicators={"signal": "bear_detected"},
    )
    await bus.publish(regime_event)
    await asyncio.sleep(0.15)

    # Should have two SELL orders (one per symbol)
    sell_orders = [o for o in orders if o.side == Side.SELL]
    symbols_sold = {o.symbol for o in sell_orders}
    assert "BTC/USD" in symbols_sold, "Expected SELL for BTC/USD"
    assert "ETH/USD" in symbols_sold, "Expected SELL for ETH/USD"
    for o in sell_orders:
        assert "regime_invalidation" in o.metadata["exit_reason"]


@pytest.mark.asyncio
async def test_time_based_exit(bus, portfolio, range_detector, monitor):
    """
    When a position is older than max_hold_minutes, a SELL is emitted on the
    next price tick regardless of price level.
    """
    orders = _capture_orders(bus)
    symbol = "BTC/USD"
    _seed_confirmed_range(range_detector, symbol, Decimal("68000"), Decimal("70000"))

    # Open position with an entry_time that is 61 minutes in the past
    old_entry_time = time.time() - (61 * 60)
    await _open_position(bus, symbol, Decimal("0.1"), Decimal("69000"), entry_time=old_entry_time)

    # Send a mid-range price tick that would not otherwise trigger any exit
    await _send_price(bus, symbol, Decimal("69000"))

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    assert "time_exit" in orders[0].metadata["exit_reason"]


@pytest.mark.asyncio
async def test_no_exit_mid_range(bus, portfolio, range_detector, monitor):
    """
    When price is in the middle of the confirmed range, no SELL is emitted.
    """
    orders = _capture_orders(bus)
    symbol = "BTC/USD"
    support = Decimal("68000")
    resistance = Decimal("70000")
    _seed_confirmed_range(range_detector, symbol, support, resistance)

    await _open_position(bus, symbol, Decimal("0.1"), Decimal("69000"))
    # Mid-range price — not near resistance, not below support
    await _send_price(bus, symbol, Decimal("69100"))

    assert len(orders) == 0, f"Expected 0 SELL orders mid-range, got {len(orders)}"


@pytest.mark.asyncio
async def test_fallback_tp_when_no_range(bus, portfolio, range_detector, monitor):
    """
    When no confirmed range exists for the symbol, a SELL is emitted when
    gain exceeds fallback_tp_pct (1.0%).
    """
    orders = _capture_orders(bus)
    symbol = "BTC/USD"
    # No range seeded — get_range() returns None for this symbol

    entry_price = Decimal("69000")
    await _open_position(bus, symbol, Decimal("0.1"), entry_price)

    # Price up 1.1% — above fallback_tp_pct of 1.0%
    tp_price = entry_price * Decimal("1.011")
    await _send_price(bus, symbol, tp_price)

    assert len(orders) == 1, f"Expected 1 SELL order (fallback TP), got {len(orders)}"
    assert "fallback_take_profit" in orders[0].metadata["exit_reason"]


@pytest.mark.asyncio
async def test_fallback_sl_when_no_range(bus, portfolio, range_detector, monitor):
    """
    When no confirmed range exists, a SELL is emitted when loss exceeds
    fallback_sl_pct (0.8%).
    """
    orders = _capture_orders(bus)
    symbol = "ETH/USD"

    entry_price = Decimal("3100")
    await _open_position(bus, symbol, Decimal("1.0"), entry_price)

    # Price down 0.9% — beyond fallback_sl_pct of 0.8%
    sl_price = entry_price * Decimal("0.991")
    await _send_price(bus, symbol, sl_price)

    assert len(orders) == 1, f"Expected 1 SELL order (fallback SL), got {len(orders)}"
    assert "fallback_stop_loss" in orders[0].metadata["exit_reason"]


@pytest.mark.asyncio
async def test_no_duplicate_exits(bus, portfolio, range_detector, monitor):
    """
    Once a SELL order is pending, further price ticks should not emit
    additional SELL orders (pending_exits dedup guard).
    """
    orders = _capture_orders(bus)
    symbol = "BTC/USD"
    _seed_confirmed_range(range_detector, symbol, Decimal("68000"), Decimal("70000"))

    await _open_position(bus, symbol, Decimal("0.1"), Decimal("69000"))

    # Three ticks all near resistance
    for _ in range(3):
        await _send_price(bus, symbol, Decimal("69850"))

    assert len(orders) == 1, f"Expected exactly 1 SELL (dedup), got {len(orders)}"


@pytest.mark.asyncio
async def test_pending_exit_cleared_on_fill(bus, portfolio, range_detector, monitor):
    """
    After a SELL FillEvent for the symbol, the pending_exits flag is cleared
    so future exits can be triggered if the position reopens.
    """
    symbol = "BTC/USD"
    orders = _capture_orders(bus)
    _seed_confirmed_range(range_detector, symbol, Decimal("68000"), Decimal("70000"))

    await _open_position(bus, symbol, Decimal("0.1"), Decimal("69000"))
    # Trigger resistance exit
    await _send_price(bus, symbol, Decimal("69850"))
    assert len(orders) == 1

    # Simulate the fill that closes the position
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.SELL,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("69850"),
        commission=Decimal("0.0"),
        commission_asset="USD",
    )
    await bus.publish(sell_fill)
    await asyncio.sleep(0.1)

    # pending_exits should be cleared
    assert symbol not in monitor._pending_exits


@pytest.mark.asyncio
async def test_regime_change_non_sideways_origin_ignored(bus, portfolio, range_detector, monitor):
    """
    A REGIME_CHANGE that does not originate from SIDEWAYS should NOT trigger
    forced exits (e.g. BEAR→SIDEWAYS is a recovery, not invalidation).
    """
    orders = _capture_orders(bus)
    _seed_confirmed_range(range_detector, "BTC/USD", Decimal("68000"), Decimal("70000"))
    await _open_position(bus, "BTC/USD", Decimal("0.1"), Decimal("69000"))

    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=time.time(),
        from_regime="BEAR",
        to_regime="SIDEWAYS",
        confidence=Decimal("0.75"),
        indicators={},
    )
    await bus.publish(regime_event)
    await asyncio.sleep(0.15)

    assert len(orders) == 0, "BEAR→SIDEWAYS should not trigger forced exits"
