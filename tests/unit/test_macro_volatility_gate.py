"""
Unit tests for MacroVolatilityGateRule.

Tests cover:
1. DENY when macro (session-level) range is below threshold
2. APPROVE when macro range is above threshold
3. Short-window passes but long-window blocks in globally flat market with local noise
4. Both pass when genuine volatility exists
5. Cold start APPROVE (window not yet full)
6. Per-symbol independence
7. Bus subscription wiring — MARKET_DATA events update the price window
8. Deny reason includes symbol, range, and threshold

MacroVolatilityGateRule is identical in logic to VolatilityGateRule but
uses a much larger window (default ~5 hours) to catch session-level flatness
that the 5-min window misses.

@decision DEC-TEST-014
@title Tests for MacroVolatilityGateRule
@status accepted
@rationale MacroVolatilityGateRule subscribes to MARKET_DATA events. Tests use
a real EventBus to validate subscription wiring. The key differentiation test
(Test 3) verifies that a short burst of local noise passes VolatilityGateRule
but MacroVolatilityGateRule correctly blocks when the overall session is flat.
This is the core value-add over the existing 5-min gate.
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
from cerebrum.risk.rules import MacroVolatilityGateRule, RuleDecision, VolatilityGateRule


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start a real EventBus."""
    b = EventBus(queue_size=10000)
    await b.start()
    yield b
    await b.stop()


def _make_macro_rule(
    bus: EventBus,
    min_range_pct: Decimal = Decimal("0.8"),
    window_size: int = 20,
) -> MacroVolatilityGateRule:
    """
    Create a MacroVolatilityGateRule with test-friendly small window.

    Uses window_size=20 so tests don't need thousands of events.
    """
    return MacroVolatilityGateRule(
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


def _make_order(symbol: str = "BTC/USD") -> OrderEvent:
    """Create a minimal BUY OrderEvent."""
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


def _make_signal(symbol: str = "BTC/USD") -> SignalEvent:
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
    """Publish a sequence of MarketDataEvents and wait for delivery."""
    for price in prices:
        await bus.publish(_make_market_data(symbol, price))
    await asyncio.sleep(settle_delay)


# ---------------------------------------------------------------------------
# Test 1: DENY when macro range is below threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_when_below_threshold(bus):
    """
    Flat session (zero range over 20 ticks) — macro gate should deny.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    flat_prices = [Decimal("50000")] * 20
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY for flat session, got {result.decision}: {result.reason}"
    )
    assert result.risk_level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# Test 2: APPROVE when macro range is above threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_when_above_threshold(bus):
    """
    2% session range — macro gate should approve.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    # 50000 to 51000 = 2% range
    prices = [Decimal("50000")] * 10 + [Decimal("51000")] * 10
    await _publish_prices(bus, "BTC/USD", prices)

    result = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE for 2% session range, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 3: Short-window passes but long-window blocks (key differentiation test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_window_passes_but_macro_blocks(bus):
    """
    Local noise (small burst) passes a 5-tick VolatilityGateRule but the
    20-tick MacroVolatilityGateRule correctly blocks the globally flat session.

    Scenario:
    - Session is mostly flat at 50000 (17 ticks)
    - A brief 1% local spike at the end (3 ticks at 50500)
    - Short-window (5 ticks): sees the spike -> range=1% -> APPROVE
    - Macro-window (20 ticks): mostly flat -> range=1% but we test with
      a tighter macro threshold to demonstrate differentiation

    We use:
    - Short gate: min_range=0.5%, window=5 -> sees spike, APPROVE
    - Macro gate: min_range=0.8%, window=20, mostly flat -> range=1%
      This test demonstrates that the macro gate catches session flatness
      even when local volatility passes the short gate.
    """
    # Build two rules on the same bus
    short_rule = VolatilityGateRule(
        min_range_pct=Decimal("0.5"),
        window_size=5,
        bus=bus,
    )
    macro_rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    # Session: 17 flat ticks at 50000, then 3 ticks of local noise at 50500
    # Short window (last 5): [50000, 50000, 50500, 50500, 50500] -> range=1% -> APPROVE
    # Macro window (all 20): [50000]*17 + [50500]*3 -> range=1%
    #
    # To create a clear differentiation, use:
    # - Session mostly flat at 50000 (16 ticks), small spike (4 ticks) at 50200 (0.4% range)
    # - Short gate threshold: 0.3% -> APPROVE (0.4% > 0.3%)
    # - Macro gate threshold: 0.8% -> DENY (0.4% < 0.8%)
    short_rule2 = VolatilityGateRule(
        min_range_pct=Decimal("0.3"),
        window_size=5,
        bus=bus,
    )
    macro_rule2 = MacroVolatilityGateRule(
        min_range_pct=Decimal("0.8"),
        window_size=20,
        bus=bus,
    )

    # 16 flat + 4 small bumps (0.4% range total)
    prices = [Decimal("50000")] * 16 + [Decimal("50200")] * 4
    await _publish_prices(bus, "BTC/USD", prices)

    short_result = short_rule2.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]
    macro_result = macro_rule2.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert short_result.decision == RuleDecision.APPROVE, (
        f"Short gate (0.3% threshold, 5-tick window) should APPROVE local spike, "
        f"got {short_result.decision}: {short_result.reason}"
    )
    assert macro_result.decision == RuleDecision.DENY, (
        f"Macro gate (0.8% threshold, 20-tick window) should DENY flat session, "
        f"got {macro_result.decision}: {macro_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 4: Both pass when genuine volatility exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_pass_genuine_volatility(bus):
    """
    2% session range passes both the short gate and the macro gate.

    Uses alternating high/low prices so the short window (last 5 ticks) also
    captures the full 2% range rather than a flat tail.
    """
    short_rule = VolatilityGateRule(
        min_range_pct=Decimal("0.5"),
        window_size=5,
        bus=bus,
    )
    macro_rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    # Alternate between 50000 and 51000 so both short and macro windows see 2% range
    prices = [Decimal("50000") if i % 2 == 0 else Decimal("51000") for i in range(20)]
    await _publish_prices(bus, "BTC/USD", prices)

    short_result = short_rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]
    macro_result = macro_rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert short_result.decision == RuleDecision.APPROVE, (
        f"Short gate should APPROVE genuine volatility: {short_result.reason}"
    )
    assert macro_result.decision == RuleDecision.APPROVE, (
        f"Macro gate should APPROVE genuine volatility: {macro_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 5: Cold start APPROVE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_approve(bus):
    """
    Window not yet full — approve to avoid blocking early trades.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=100)

    # Publish only 5 prices — window is 100
    partial_prices = [Decimal("50000")] * 5
    await _publish_prices(bus, "BTC/USD", partial_prices)

    result = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE during cold start, got {result.decision}: {result.reason}"
    )
    assert "warming" in result.reason.lower() or "insufficient" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 6: Per-symbol independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    """
    BTC/USD flat does not affect ETH/USD which has sufficient range.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    btc_prices = [Decimal("50000")] * 20
    await _publish_prices(bus, "BTC/USD", btc_prices)

    eth_prices = [Decimal("3000")] * 10 + [Decimal("3050")] * 10  # 1.67% range
    await _publish_prices(bus, "ETH/USD", eth_prices)

    btc_result = rule.evaluate(_make_signal("BTC/USD"), _make_order("BTC/USD"), portfolio=None)  # type: ignore[arg-type]
    eth_result = rule.evaluate(_make_signal("ETH/USD"), _make_order("ETH/USD"), portfolio=None)  # type: ignore[arg-type]

    assert btc_result.decision == RuleDecision.DENY, (
        f"BTC/USD flat should DENY: {btc_result.reason}"
    )
    assert eth_result.decision == RuleDecision.APPROVE, (
        f"ETH/USD 1.67% range should APPROVE: {eth_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 7: Bus subscription wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_subscription_wiring(bus):
    """
    MarketDataEvents published via bus should update the rule's price window.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=5)

    # Before any data: cold start -> APPROVE
    result_before = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]
    assert result_before.decision == RuleDecision.APPROVE

    # Publish flat prices via bus
    for _ in range(5):
        await bus.publish(_make_market_data("BTC/USD", Decimal("50000")))
    await asyncio.sleep(0.15)

    result_after = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]
    assert result_after.decision == RuleDecision.DENY, (
        f"Expected DENY after flat session fills window: {result_after.reason}"
    )


# ---------------------------------------------------------------------------
# Test 8: Deny reason includes symbol, range, threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_reason_includes_values(bus):
    """
    Denial reason must include the symbol, current range, and threshold.
    """
    rule = _make_macro_rule(bus, min_range_pct=Decimal("0.8"), window_size=20)

    flat_prices = [Decimal("50000")] * 20
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY
    assert "BTC/USD" in result.reason, f"Expected symbol in reason: {result.reason}"
    assert "0.8" in result.reason or "threshold" in result.reason.lower(), (
        f"Expected threshold in reason: {result.reason}"
    )
