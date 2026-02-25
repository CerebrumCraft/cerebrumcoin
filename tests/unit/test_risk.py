"""
Unit tests for risk management system.

Tests portfolio tracking, risk rules, and risk manager.

@decision DEC-RISK-TEST-001
@title Unit tests verify fill-driven PositionUpdateEvent publishing
@status accepted
@rationale Portfolio tracker must publish PositionUpdateEvent on every fill so the
dashboard displays accurate open positions without requiring direct access to the
portfolio's internal state. Tests here confirm event content and the closure signal
(amount=0) that allows subscribers to remove closed positions from their tracking dicts.
"""

import asyncio
from decimal import Decimal
from time import time
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent, PositionUpdateEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    MaxDrawdownRule,
    MaxPositionSizeRule,
    MinSignalStrengthRule,
    PositionSizingRule,
    RuleDecision,
)


@pytest.fixture
async def bus():
    """Create and start event bus."""
    bus = EventBus(queue_size=50)
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def portfolio(bus):
    """Create portfolio tracker."""
    return PortfolioTracker(bus, initial_balance=Decimal("10000.0"))


@pytest.mark.asyncio
async def test_portfolio_fill_tracking(bus, portfolio):
    """Test portfolio tracks fills correctly."""
    symbol = "BTC/USD"
    
    # Simulate a buy fill
    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="test-1",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    
    await bus.publish(fill)
    await asyncio.sleep(0.1)
    
    # Check position created
    pos = portfolio.get_position(symbol)
    assert pos is not None
    assert pos.amount == Decimal("0.1")
    assert pos.average_entry_price == Decimal("50000")
    
    # Check cash balance reduced
    expected_cash = Decimal("10000") - Decimal("5000") - Decimal("5.0")
    assert portfolio.get_cash_balance() == expected_cash


@pytest.mark.asyncio
async def test_portfolio_position_closing(bus, portfolio):
    """Test portfolio handles position closing."""
    symbol = "BTC/USD"
    
    # Buy
    buy_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="buy-1",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(buy_fill)
    await asyncio.sleep(0.1)
    
    # Sell (close position at profit)
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="sell-1",
        symbol=symbol,
        side=Side.SELL,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("55000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(sell_fill)
    await asyncio.sleep(0.1)
    
    # Position should be closed
    pos = portfolio.get_position(symbol)
    assert pos is None
    
    # Check realized P&L
    realized, _ = portfolio.get_pnl()
    expected_pnl = Decimal("0.1") * (Decimal("55000") - Decimal("50000"))
    assert realized == expected_pnl


@pytest.mark.asyncio
async def test_portfolio_price_updates(bus, portfolio):
    """Test portfolio updates position prices from market data."""
    symbol = "BTC/USD"
    
    # Create position
    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="test-1",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(fill)
    await asyncio.sleep(0.1)
    
    # Send market data with new price
    market_data = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol=symbol,
        price=Decimal("52000"),
        volume=Decimal("1.0"),
    )
    await bus.publish(market_data)
    await asyncio.sleep(0.1)
    
    # Check position updated
    pos = portfolio.get_position(symbol)
    assert pos.current_price == Decimal("52000")
    
    # Check unrealized P&L
    expected_upnl = Decimal("0.1") * (Decimal("52000") - Decimal("50000"))
    assert pos.unrealized_pnl == expected_upnl


def test_max_position_size_rule():
    """Test max position size rule."""
    rule = MaxPositionSizeRule(max_position_usd=Decimal("1000.0"))
    
    # Mock portfolio with no position
    class MockPortfolio:
        def get_position(self, symbol):
            return None
    
    portfolio = MockPortfolio()
    
    # Create signal and order
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )
    
    # Small order (within limit)
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.01"),
        price=Decimal("50000"),
    )
    
    result = rule.evaluate(signal, order, portfolio)
    assert result.decision == RuleDecision.APPROVE
    
    # Large order (exceeds limit)
    large_order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("0.5"),
        price=Decimal("50000"),  # 0.5 * 50000 = 25000 > 1000
    )
    
    result = rule.evaluate(signal, large_order, portfolio)
    assert result.decision == RuleDecision.MODIFY
    assert result.modified_amount is not None


def test_min_signal_strength_rule():
    """Test minimum signal strength rule."""
    rule = MinSignalStrengthRule(min_strength=Decimal("0.5"))
    
    class MockPortfolio:
        pass
    
    portfolio = MockPortfolio()
    
    # Weak signal
    weak_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.3"),
        confidence=Decimal("0.8"),
    )
    
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )
    
    result = rule.evaluate(weak_signal, order, portfolio)
    assert result.decision == RuleDecision.DENY
    
    # Strong signal
    strong_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.7"),
        confidence=Decimal("0.8"),
    )
    
    result = rule.evaluate(strong_signal, order, portfolio)
    assert result.decision == RuleDecision.APPROVE


def test_position_sizing_rule():
    """Test position sizing rule."""
    rule = PositionSizingRule(position_size_percent=Decimal("2.0"))
    
    class MockPortfolio:
        def get_total_equity(self):
            return Decimal("10000.0")
    
    portfolio = MockPortfolio()
    
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.5"),
        confidence=Decimal("0.8"),
    )
    
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("1.0"),
        price=Decimal("50000"),
    )
    
    result = rule.evaluate(signal, order, portfolio)
    assert result.decision == RuleDecision.MODIFY
    
    # Check sizing: 2% of 10000 = 200 USD
    # At price 50000, amount = 200/50000 = 0.004
    # Adjusted by strength 0.5 = 0.002
    expected_amount = Decimal("10000") * Decimal("0.02") / Decimal("50000") * Decimal("0.5")
    assert result.modified_amount == expected_amount


def test_position_sizing_rule_denies_when_no_price():
    """PositionSizingRule must DENY when no price is available.

    Regression test: previously returned APPROVE, leaving the 1.0 BTC
    placeholder amount unmodified, which caused exchange rejection as
    insufficient_balance (~$68k required, $10k available).
    """
    rule = PositionSizingRule(position_size_percent=Decimal("2.0"))

    class MockPortfolio:
        def get_total_equity(self):
            return Decimal("10000.0")

        def get_latest_price(self, symbol):
            # Simulate startup condition: no market data has arrived yet
            return None

    portfolio = MockPortfolio()

    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )

    # Market order with no price (the Fear & Greed startup scenario)
    order = OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1.0"),  # placeholder amount — must NOT pass through
    )

    result = rule.evaluate(signal, order, portfolio)
    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY when no price data available, got {result.decision}: {result.reason}"
    )
    assert "no price data" in result.reason.lower()


@pytest.mark.asyncio
async def test_risk_manager_integration(bus, portfolio):
    """Test risk manager applies rules correctly.

    Market data must be seeded before the signal fires so PositionSizingRule
    can resolve a price for the market order. Without price data the rule now
    correctly DENYs the order (regression fix for the passthrough bug).
    """
    # Create rules
    rules = [
        MinSignalStrengthRule(min_strength=Decimal("0.4")),
        PositionSizingRule(position_size_percent=Decimal("2.0")),
    ]

    risk_manager = RiskManager(bus, portfolio, rules=rules)

    # Collect orders
    orders = []

    async def order_collector(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, order_collector, "order_collector")

    # Seed market data so PositionSizingRule can resolve a price
    market_data = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    )
    await bus.publish(market_data)
    await asyncio.sleep(0.1)

    # Send strong combined signal
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )

    await bus.publish(signal)
    await asyncio.sleep(0.3)

    # Order should be generated (strong signal + price data available)
    assert len(orders) >= 1

    # Order amount should be sized appropriately
    order = orders[0]
    assert order.symbol == "BTC/USD"
    assert order.side == Side.BUY
    # Amount should be positive (sized by PositionSizingRule)
    assert order.amount > Decimal("0")


@pytest.mark.asyncio
async def test_risk_manager_rejects_weak_signals(bus, portfolio):
    """Test risk manager rejects weak signals."""
    rules = [MinSignalStrengthRule(min_strength=Decimal("0.5"))]
    risk_manager = RiskManager(bus, portfolio, rules=rules)
    
    orders = []
    
    async def order_collector(event):
        if isinstance(event, OrderEvent):
            orders.append(event)
    
    bus.subscribe(EventType.ORDER, order_collector, "order_collector")
    
    # Send weak signal
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.3"),
        confidence=Decimal("0.7"),
    )
    
    await bus.publish(signal)
    await asyncio.sleep(0.3)

    # No order should be generated
    assert len(orders) == 0


@pytest.mark.asyncio
async def test_fill_publishes_position_update_event(bus, portfolio):
    """Fill must publish PositionUpdateEvent with correct position data.

    Verifies DEC-DASH-001: the portfolio tracker notifies subscribers on every
    fill so the dashboard can display accurate open positions without directly
    accessing the portfolio's internal state.
    """
    symbol = "BTC/USD"
    position_events: list[PositionUpdateEvent] = []

    async def position_collector(event):
        if isinstance(event, PositionUpdateEvent):
            position_events.append(event)

    bus.subscribe(EventType.POSITION_UPDATE, position_collector, "test_position_collector")

    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="test-pu-1",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(fill)
    await asyncio.sleep(0.1)

    assert len(position_events) == 1, f"Expected 1 PositionUpdateEvent, got {len(position_events)}"
    evt = position_events[0]
    assert evt.symbol == symbol
    assert evt.amount == Decimal("0.1")
    assert evt.average_entry_price == Decimal("50000")
    assert evt.current_price == Decimal("50000")
    assert evt.unrealized_pnl == Decimal("0.0")


@pytest.mark.asyncio
async def test_position_close_publishes_zero_amount_event(bus, portfolio):
    """Position closure must publish PositionUpdateEvent with amount=0.

    Verifies DEC-DASH-001: subscribers (e.g. the dashboard) use amount=0 as
    a sentinel to remove the position entry from their tracking dicts.
    """
    symbol = "BTC/USD"
    position_events: list[PositionUpdateEvent] = []

    async def position_collector(event):
        if isinstance(event, PositionUpdateEvent):
            position_events.append(event)

    bus.subscribe(EventType.POSITION_UPDATE, position_collector, "test_close_collector")

    # Open a position
    buy_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="open-pu-1",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(buy_fill)
    await asyncio.sleep(0.1)

    # Close the position
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="close-pu-1",
        symbol=symbol,
        side=Side.SELL,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("55000"),
        commission=Decimal("5.0"),
        commission_asset="USD",
    )
    await bus.publish(sell_fill)
    await asyncio.sleep(0.1)

    # Two events: one for open, one for close
    assert len(position_events) == 2, f"Expected 2 PositionUpdateEvents, got {len(position_events)}"

    close_evt = position_events[1]
    assert close_evt.symbol == symbol
    assert close_evt.amount == Decimal("0"), (
        f"Expected amount=0 for closed position, got {close_evt.amount}"
    )
    assert close_evt.current_price == Decimal("55000")
