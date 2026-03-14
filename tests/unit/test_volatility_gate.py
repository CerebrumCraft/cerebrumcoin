"""
Unit tests for VolatilityGateRule.

Tests cover:
1. DENY when volatility below threshold (flat prices)
2. APPROVE when volatility above threshold
3. Boundary test at exact threshold (equal = APPROVE, just below = DENY)
4. Per-symbol independence — BTC window does not affect ETH
5. Bus subscription wiring — MARKET_DATA events update the price window
6. Insufficient data → APPROVE (cold start, fewer ticks than window_size)
7. Reason string includes volatility value and threshold

VolatilityGateRule subscribes to MARKET_DATA events via the bus (same pattern
as PostFillCooldownRule). The rule uses a real EventBus to validate end-to-end
subscription wiring.

@decision DEC-TEST-012
@title Test VolatilityGateRule with real EventBus and injected prices
@status accepted
@rationale VolatilityGateRule self-subscribes to MARKET_DATA events. Testing
with a real EventBus validates subscription wiring and per-symbol deque logic.
Unlike PostFillCooldownRule, there is no wall-clock dependency — tests inject
prices directly via bus.publish(MarketDataEvent) and verify evaluate() outcomes.
Cold-start behavior (insufficient data) is tested by publishing fewer ticks
than window_size and confirming APPROVE is returned.
"""

import asyncio
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, OrderEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import RuleDecision, VolatilityGateRule


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start event bus."""
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


def _make_rule(
    bus: EventBus,
    min_range_pct: Decimal = Decimal("0.5"),
    window_size: int = 10,
) -> VolatilityGateRule:
    """
    Create a VolatilityGateRule subscribed to the given bus.

    Uses a small window_size (default 10) so tests don't need to publish
    hundreds of events to fill the window.
    """
    return VolatilityGateRule(
        min_range_pct=min_range_pct,
        window_size=window_size,
        bus=bus,
    )


def _make_market_data(symbol: str, price: Decimal) -> MarketDataEvent:
    """Create a MarketDataEvent for the given symbol and price."""
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time.time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )


def _make_order(symbol: str) -> OrderEvent:
    """Create a minimal BUY OrderEvent for the given symbol."""
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
        status=OrderStatus.PENDING,
    )


def _make_signal(symbol: str) -> SignalEvent:
    """Create a minimal SignalEvent."""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.7"),
    )


async def _publish_prices(
    bus: EventBus,
    symbol: str,
    prices: list,
    settle_delay: float = 0.15,
) -> None:
    """
    Publish a sequence of MarketDataEvents to the bus and wait for delivery.

    Args:
        bus: Event bus to publish on.
        symbol: Symbol for all events.
        prices: Sequence of prices to publish (in order).
        settle_delay: Seconds to wait after last publish for async delivery.
    """
    for price in prices:
        await bus.publish(_make_market_data(symbol, price))
    await asyncio.sleep(settle_delay)


# ---------------------------------------------------------------------------
# Test 1: DENY when volatility below threshold (flat prices)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_when_below_threshold(bus):
    """
    With a flat price window (zero range), volatility is well below the
    0.5% threshold — the rule should deny the order.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=10)

    # Publish 10 identical prices — range = 0%
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY for flat prices, got {result.decision}: {result.reason}"
    )
    assert result.risk_level == RiskLevel.LOW
    # Reason should mention threshold value
    assert "0.5" in result.reason or "threshold" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 2: APPROVE when volatility above threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_when_above_threshold(bus):
    """
    With a wide price range (2% spread), volatility exceeds the 0.5%
    threshold — the rule should approve the order.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=10)

    # Price swings from 50000 to 51000 — range = 1000/50000 = 2.0%
    prices = [Decimal("50000")] * 5 + [Decimal("51000")] * 5
    await _publish_prices(bus, "BTC/USD", prices)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE for 2% range, got {result.decision}: {result.reason}"
    )
    assert result.risk_level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# Test 3: Boundary at exact threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_at_exact_threshold(bus):
    """
    At exactly the threshold value, the rule should APPROVE (>= threshold).
    Just below the threshold, the rule should DENY.

    Threshold = 0.5%. We use a window of 10 prices.

    Exact: min=50000, max=50250 -> range = 250/50000 = 0.5% exactly -> APPROVE.
    Below: min=50000, max=50249 -> range = 249/50000 = 0.498% -> DENY.
    """
    # Test exact threshold — should APPROVE
    rule_exact = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=10)
    prices_exact = [Decimal("50000")] * 5 + [Decimal("50250")] * 5
    await _publish_prices(bus, "BTC/USD", prices_exact)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result_exact = rule_exact.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result_exact.decision == RuleDecision.APPROVE, (
        f"At exact threshold (0.5%), expected APPROVE, got {result_exact.decision}: "
        f"{result_exact.reason}"
    )

    # Test just below threshold — should DENY
    # Use a second bus instance to get a clean rule with an isolated price window
    bus2 = EventBus(queue_size=1000)
    await bus2.start()
    try:
        rule_below = _make_rule(bus2, min_range_pct=Decimal("0.5"), window_size=10)
        # 249/50000 = 0.498% < 0.5%
        prices_below = [Decimal("50000")] * 5 + [Decimal("50249")] * 5
        await _publish_prices(bus2, "BTC/USD", prices_below)

        result_below = rule_below.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_below.decision == RuleDecision.DENY, (
            f"Just below threshold (0.498%), expected DENY, got {result_below.decision}: "
            f"{result_below.reason}"
        )
    finally:
        await bus2.stop()


# ---------------------------------------------------------------------------
# Test 4: Per-symbol independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    """
    BTC/USD prices should not affect the ETH/USD volatility window.
    BTC/USD is flat (DENY), ETH/USD has sufficient range (APPROVE).
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=10)

    # BTC/USD: flat prices -> below threshold
    btc_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", btc_prices)

    # ETH/USD: wide range -> above threshold (1% range)
    eth_prices = [Decimal("3000")] * 5 + [Decimal("3030")] * 5  # 30/3000 = 1%
    await _publish_prices(bus, "ETH/USD", eth_prices)

    btc_signal = _make_signal("BTC/USD")
    btc_order = _make_order("BTC/USD")
    btc_result = rule.evaluate(btc_signal, btc_order, portfolio=None)  # type: ignore[arg-type]

    eth_signal = _make_signal("ETH/USD")
    eth_order = _make_order("ETH/USD")
    eth_result = rule.evaluate(eth_signal, eth_order, portfolio=None)  # type: ignore[arg-type]

    assert btc_result.decision == RuleDecision.DENY, (
        f"BTC/USD flat should DENY, got {btc_result.decision}: {btc_result.reason}"
    )
    assert eth_result.decision == RuleDecision.APPROVE, (
        f"ETH/USD wide range should APPROVE, got {eth_result.decision}: {eth_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 5: Bus subscription wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_subscription_wiring(bus):
    """
    MarketDataEvents published to the bus should automatically update the rule's
    per-symbol price window via the subscription set up in __init__.
    Verifies end-to-end bus subscription wiring.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=5)

    # Before any market data: insufficient data -> APPROVE (cold start)
    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result_before = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
    assert result_before.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE before any market data (cold start), got {result_before.decision}"
    )

    # Publish flat prices — window fills up, volatility is 0% -> DENY
    flat_prices = [Decimal("50000")] * 5
    await _publish_prices(bus, "BTC/USD", flat_prices)

    # Now the window has data and volatility is below threshold
    result_after = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
    assert result_after.decision == RuleDecision.DENY, (
        f"Expected DENY after flat price data fills window, got {result_after.decision}"
    )


# ---------------------------------------------------------------------------
# Test 6: Insufficient data -> APPROVE (cold start)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_approve(bus):
    """
    When fewer ticks than window_size have been received for a symbol,
    the rule should APPROVE to avoid blocking early trades.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=100)

    # Publish only 5 prices — window_size is 100, so this is insufficient data
    partial_prices = [Decimal("50000")] * 5
    await _publish_prices(bus, "BTC/USD", partial_prices)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE during cold start (5 of 100 ticks), "
        f"got {result.decision}: {result.reason}"
    )
    assert "insufficient" in result.reason.lower() or "warming" in result.reason.lower(), (
        f"Expected cold-start language in reason, got: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 7: Reason string content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_reason_includes_values(bus):
    """
    When denying, the reason string should include both the current volatility
    value and the threshold so operators can diagnose the condition.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("0.5"), window_size=10)

    # Publish flat prices — range = 0%
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY
    # Reason must include the threshold so operators can see what was needed
    assert "0.5" in result.reason, (
        f"Expected threshold '0.5' in deny reason, got: {result.reason}"
    )
    # Reason must mention the symbol
    assert "BTC/USD" in result.reason, (
        f"Expected symbol in deny reason, got: {result.reason}"
    )
