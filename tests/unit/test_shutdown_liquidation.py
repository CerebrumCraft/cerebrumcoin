"""
Tests for graceful position liquidation on shutdown (DEC-SHUTDOWN-001).

Verifies that _close_all_positions() publishes the correct OrderEvents
for long positions (SELL), short positions (BUY to cover), and skips
zero-amount positions.
"""

import asyncio
from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import OrderEvent
from cerebrum.core.types import EventType, Side
from cerebrum.main import CerebrumCoin
from cerebrum.risk.portfolio import PortfolioTracker


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    return b


def _make_app(bus: EventBus) -> CerebrumCoin:
    """Create a minimal CerebrumCoin instance for testing _close_all_positions."""
    app = CerebrumCoin.__new__(CerebrumCoin)
    app.bus = bus
    app.portfolio = None
    app.strategy_registry = None
    app._log = __import__("structlog").get_logger().bind(component="test")
    return app


def _add_position(portfolio: PortfolioTracker, symbol: str, amount: Decimal, price: Decimal) -> None:
    """Manually inject a position into a portfolio tracker."""
    from cerebrum.risk.portfolio import Position
    portfolio._positions[symbol] = Position(
        symbol=symbol,
        amount=amount,
        average_entry_price=price,
        current_price=price,
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )
    portfolio._positions[symbol].update_price(price)


@pytest.mark.asyncio
async def test_long_position_emits_sell(bus):
    """Long positions should produce SELL orders."""
    app = _make_app(bus)
    app.portfolio = PortfolioTracker(bus, Decimal("10000"))
    _add_position(app.portfolio, "BTC/USD", Decimal("0.5"), Decimal("60000"))

    received: list[OrderEvent] = []
    bus.subscribe(EventType.ORDER, lambda e: received.append(e), subscriber_name="test")

    await app._close_all_positions()
    await asyncio.sleep(0)

    assert len(received) == 1
    assert received[0].side == Side.SELL
    assert received[0].amount == Decimal("0.5")
    assert received[0].symbol == "BTC/USD"
    assert received[0].metadata["exit_reason"] == "shutdown_liquidation"


@pytest.mark.asyncio
async def test_short_position_emits_buy(bus):
    """Short positions should produce BUY orders to cover."""
    app = _make_app(bus)
    app.portfolio = PortfolioTracker(bus, Decimal("10000"))
    _add_position(app.portfolio, "ETH/USD", Decimal("-0.5"), Decimal("2000"))

    received: list[OrderEvent] = []
    bus.subscribe(EventType.ORDER, lambda e: received.append(e), subscriber_name="test")

    await app._close_all_positions()
    await asyncio.sleep(0)

    assert len(received) == 1
    assert received[0].side == Side.BUY
    assert received[0].amount == Decimal("0.5")


@pytest.mark.asyncio
async def test_zero_position_skipped(bus):
    """Zero-amount positions should not generate orders."""
    app = _make_app(bus)
    app.portfolio = PortfolioTracker(bus, Decimal("10000"))
    _add_position(app.portfolio, "BTC/USD", Decimal("0"), Decimal("60000"))

    received: list[OrderEvent] = []
    bus.subscribe(EventType.ORDER, lambda e: received.append(e), subscriber_name="test")

    await app._close_all_positions()
    await asyncio.sleep(0)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_no_positions_no_orders(bus):
    """Empty portfolio should emit no orders."""
    app = _make_app(bus)
    app.portfolio = PortfolioTracker(bus, Decimal("10000"))

    received: list[OrderEvent] = []
    bus.subscribe(EventType.ORDER, lambda e: received.append(e), subscriber_name="test")

    await app._close_all_positions()
    await asyncio.sleep(0)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_positions_all_closed(bus):
    """Multiple positions across symbols should all get liquidation orders."""
    app = _make_app(bus)
    app.portfolio = PortfolioTracker(bus, Decimal("10000"))
    _add_position(app.portfolio, "BTC/USD", Decimal("0.1"), Decimal("60000"))
    _add_position(app.portfolio, "ETH/USD", Decimal("-2.0"), Decimal("2000"))

    received: list[OrderEvent] = []
    bus.subscribe(EventType.ORDER, lambda e: received.append(e), subscriber_name="test")

    await app._close_all_positions()
    await asyncio.sleep(0)

    assert len(received) == 2
    sides = {r.symbol: r.side for r in received}
    assert sides["BTC/USD"] == Side.SELL
    assert sides["ETH/USD"] == Side.BUY
