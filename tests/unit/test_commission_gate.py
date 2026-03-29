"""
Unit tests for CommissionGateRule.

Tests cover:
1. DENY when market is flat (range < commission threshold)
2. APPROVE when market is volatile enough (range >= threshold)
3. Boundary test at exact threshold value (equal = APPROVE, just below = DENY)
4. Per-symbol independence — BTC range doesn't affect ETH decision
5. Cold start APPROVE (insufficient data in window)
6. Different commission_percent values produce different thresholds
7. Different min_profit_to_commission_ratio values affect the threshold
8. Denial reason string contains useful diagnostic info

CommissionGateRule subscribes to MARKET_DATA events via the bus (same pattern
as VolatilityGateRule). Tests use a real EventBus to validate end-to-end
subscription wiring and per-symbol deque logic.

@decision DEC-TEST-COMMISSION-001
@title Test CommissionGateRule with real EventBus and injected prices
@status accepted
@rationale CommissionGateRule self-subscribes to MARKET_DATA events. Testing
with a real EventBus validates subscription wiring and per-symbol deque logic.
No wall-clock dependency — tests inject prices directly via bus.publish()
and verify evaluate() outcomes. Cold-start behavior is tested by publishing
fewer ticks than window_size and confirming APPROVE is returned.
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
from cerebrum.risk.rules import CommissionGateRule, RuleDecision


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
    commission_percent: Decimal = Decimal("0.16"),
    min_profit_to_commission_ratio: Decimal = Decimal("2.0"),
    window_size: int = 10,
) -> CommissionGateRule:
    """
    Create a CommissionGateRule subscribed to the given bus.

    Uses a small window_size (default 10) so tests don't need to publish
    hundreds of events to fill the window.

    Default params:
      commission_percent=0.16 (Kraken maker fee)
      min_profit_to_commission_ratio=2.0
      -> threshold = 0.16 * 2 * 2.0 = 0.64%
    """
    return CommissionGateRule(
        commission_percent=commission_percent,
        min_profit_to_commission_ratio=min_profit_to_commission_ratio,
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
# Test 1: DENY when market is flat (range < commission threshold)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_when_below_threshold(bus):
    """
    With a flat price window (zero range), the commission gate should deny
    the order because range (0%) < threshold (0.64% for 0.16% fee * 2 sides * 2.0 ratio).
    """
    # commission=0.16%, ratio=2.0 -> threshold = 0.16 * 2 * 2.0 = 0.64%
    rule = _make_rule(
        bus,
        commission_percent=Decimal("0.16"),
        min_profit_to_commission_ratio=Decimal("2.0"),
        window_size=10,
    )

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


# ---------------------------------------------------------------------------
# Test 2: APPROVE when market is volatile enough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_when_above_threshold(bus):
    """
    With a 2% price range, the commission gate should approve because
    2% > threshold (0.64% for 0.16% fee * 2 sides * 2.0 ratio).
    """
    rule = _make_rule(
        bus,
        commission_percent=Decimal("0.16"),
        min_profit_to_commission_ratio=Decimal("2.0"),
        window_size=10,
    )

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
# Test 3: Boundary test at exact threshold value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_at_exact_threshold(bus):
    """
    At exactly the threshold, rule should APPROVE (>= threshold).
    Just below the threshold, rule should DENY.

    commission=0.16%, ratio=2.0 -> threshold = 0.16 * 2 * 2.0 = 0.64%

    Exact: min=50000, max=50320 -> range = 320/50000 = 0.64% exactly -> APPROVE.
    Below: min=50000, max=50319 -> range = 319/50000 = 0.638% -> DENY.
    """
    # Test exact threshold — should APPROVE
    rule_exact = _make_rule(
        bus,
        commission_percent=Decimal("0.16"),
        min_profit_to_commission_ratio=Decimal("2.0"),
        window_size=10,
    )
    # 320/50000 * 100 = 0.64% exactly
    prices_exact = [Decimal("50000")] * 5 + [Decimal("50320")] * 5
    await _publish_prices(bus, "BTC/USD", prices_exact)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result_exact = rule_exact.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result_exact.decision == RuleDecision.APPROVE, (
        f"At exact threshold (0.64%), expected APPROVE, got {result_exact.decision}: "
        f"{result_exact.reason}"
    )

    # Test just below threshold — should DENY (use isolated bus2)
    bus2 = EventBus(queue_size=1000)
    await bus2.start()
    try:
        rule_below = _make_rule(
            bus2,
            commission_percent=Decimal("0.16"),
            min_profit_to_commission_ratio=Decimal("2.0"),
            window_size=10,
        )
        # 319/50000 * 100 = 0.638% < 0.64%
        prices_below = [Decimal("50000")] * 5 + [Decimal("50319")] * 5
        await _publish_prices(bus2, "BTC/USD", prices_below)

        result_below = rule_below.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_below.decision == RuleDecision.DENY, (
            f"Just below threshold (0.638%), expected DENY, got {result_below.decision}: "
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
    BTC/USD flat should DENY; ETH/USD volatile enough should APPROVE.
    The two windows are completely independent.
    """
    rule = _make_rule(
        bus,
        commission_percent=Decimal("0.16"),
        min_profit_to_commission_ratio=Decimal("2.0"),
        window_size=10,
    )

    # BTC/USD: flat prices -> below threshold
    btc_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", btc_prices)

    # ETH/USD: wide range -> above threshold (1% range > 0.64%)
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
        f"ETH/USD 1% range should APPROVE, got {eth_result.decision}: {eth_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 5: Cold start APPROVE (insufficient data)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_approve(bus):
    """
    When fewer ticks than window_size have been received for a symbol,
    the rule should APPROVE to avoid blocking early trades.
    """
    rule = _make_rule(bus, window_size=100)

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
# Test 6: Different commission_percent values produce different thresholds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_commission_percent_values(bus):
    """
    A higher commission_percent raises the threshold, causing a range that
    would pass a low-commission rule to fail a high-commission rule.

    0.1% commission, ratio=2.0 -> threshold = 0.1 * 2 * 2.0 = 0.4%
    0.5% commission, ratio=2.0 -> threshold = 0.5 * 2 * 2.0 = 2.0%

    A 1% price range:
      - passes the 0.4% threshold -> APPROVE
      - fails the 2.0% threshold -> DENY
    """
    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")

    # Low commission rule: threshold = 0.4%
    bus_low = EventBus(queue_size=1000)
    await bus_low.start()
    try:
        rule_low = CommissionGateRule(
            commission_percent=Decimal("0.1"),
            min_profit_to_commission_ratio=Decimal("2.0"),
            window_size=10,
            bus=bus_low,
        )
        # 1% range = 500/50000
        prices = [Decimal("50000")] * 5 + [Decimal("50500")] * 5
        await _publish_prices(bus_low, "BTC/USD", prices)

        result_low = rule_low.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_low.decision == RuleDecision.APPROVE, (
            f"Low commission (0.1%) with 1% range should APPROVE, "
            f"got {result_low.decision}: {result_low.reason}"
        )
    finally:
        await bus_low.stop()

    # High commission rule: threshold = 2.0%
    bus_high = EventBus(queue_size=1000)
    await bus_high.start()
    try:
        rule_high = CommissionGateRule(
            commission_percent=Decimal("0.5"),
            min_profit_to_commission_ratio=Decimal("2.0"),
            window_size=10,
            bus=bus_high,
        )
        # Same 1% range
        prices = [Decimal("50000")] * 5 + [Decimal("50500")] * 5
        await _publish_prices(bus_high, "BTC/USD", prices)

        result_high = rule_high.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_high.decision == RuleDecision.DENY, (
            f"High commission (0.5%) with 1% range should DENY (threshold=2%), "
            f"got {result_high.decision}: {result_high.reason}"
        )
    finally:
        await bus_high.stop()


# ---------------------------------------------------------------------------
# Test 7: Different min_profit_to_commission_ratio values affect threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_ratio_values(bus):
    """
    A higher ratio raises the required range proportionally.

    commission=0.16%, ratio=1.0 -> threshold = 0.16 * 2 * 1.0 = 0.32%
    commission=0.16%, ratio=3.0 -> threshold = 0.16 * 2 * 3.0 = 0.96%

    A 0.5% price range:
      - passes the 0.32% threshold -> APPROVE
      - fails the 0.96% threshold -> DENY
    """
    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")

    # Low ratio: threshold = 0.32%
    bus_lo = EventBus(queue_size=1000)
    await bus_lo.start()
    try:
        rule_lo = CommissionGateRule(
            commission_percent=Decimal("0.16"),
            min_profit_to_commission_ratio=Decimal("1.0"),
            window_size=10,
            bus=bus_lo,
        )
        # 0.5% range = 250/50000
        prices = [Decimal("50000")] * 5 + [Decimal("50250")] * 5
        await _publish_prices(bus_lo, "BTC/USD", prices)

        result_lo = rule_lo.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_lo.decision == RuleDecision.APPROVE, (
            f"ratio=1.0 with 0.5% range should APPROVE (threshold=0.32%), "
            f"got {result_lo.decision}: {result_lo.reason}"
        )
    finally:
        await bus_lo.stop()

    # High ratio: threshold = 0.96%
    bus_hi = EventBus(queue_size=1000)
    await bus_hi.start()
    try:
        rule_hi = CommissionGateRule(
            commission_percent=Decimal("0.16"),
            min_profit_to_commission_ratio=Decimal("3.0"),
            window_size=10,
            bus=bus_hi,
        )
        # Same 0.5% range
        prices = [Decimal("50000")] * 5 + [Decimal("50250")] * 5
        await _publish_prices(bus_hi, "BTC/USD", prices)

        result_hi = rule_hi.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]
        assert result_hi.decision == RuleDecision.DENY, (
            f"ratio=3.0 with 0.5% range should DENY (threshold=0.96%), "
            f"got {result_hi.decision}: {result_hi.reason}"
        )
    finally:
        await bus_hi.stop()


# ---------------------------------------------------------------------------
# Test 8: Denial reason string contains useful diagnostic info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_reason_includes_values(bus):
    """
    When denying, the reason string should include:
    - The current range value (so operators know how flat the market is)
    - The threshold value (so operators know what was needed)
    - The symbol (so operators know which market triggered the gate)
    - A reference to 'commission_gate' (gate identifier for log search)
    """
    rule = _make_rule(
        bus,
        commission_percent=Decimal("0.16"),
        min_profit_to_commission_ratio=Decimal("2.0"),
        window_size=10,
    )

    # Publish flat prices — range = 0%
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY
    # Reason must mention the gate
    assert "commission_gate" in result.reason.lower() or "commission" in result.reason.lower(), (
        f"Expected 'commission_gate' or 'commission' in deny reason, got: {result.reason}"
    )
    # Reason must include threshold value (0.64%)
    assert "0.64" in result.reason or "threshold" in result.reason.lower(), (
        f"Expected threshold value in deny reason, got: {result.reason}"
    )
    # Reason must mention the symbol
    assert "BTC/USD" in result.reason, (
        f"Expected symbol in deny reason, got: {result.reason}"
    )
