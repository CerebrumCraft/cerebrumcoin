"""
Unit tests for PostFillCooldownRule.

Tests cover:
1. DENY within cooldown period — place fill, immediately try order, should be denied
2. APPROVE after cooldown expires — advance clock past cooldown, order approved
3. Per-symbol independence — BTC fill doesn't cooldown ETH orders
4. FILL event auto-records via bus subscription — publish FillEvent, verify cooldown active
5. No cooldown on first order (no prior fill)

PostFillCooldownRule accepts an injectable _clock callable so tests control
the clock without mocking internal modules or sleeping real seconds. The bus
subscription is exercised with a real EventBus to validate end-to-end wiring.

@decision DEC-TEST-010
@title Test PostFillCooldownRule with injectable clock and real EventBus
@status accepted
@rationale PostFillCooldownRule self-subscribes to FillEvents via the bus.
Testing with a real EventBus validates the subscription wiring, async event
delivery, and the per-symbol cooldown logic together. Time-passage tests use
an injectable _clock callable (not mocks) — the clock is an external dependency
(wall time) that the rule explicitly supports injecting for testability.
"""

import asyncio
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import PostFillCooldownRule, RuleDecision


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start event bus."""
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


def _make_rule(
    bus: EventBus,
    cooldown_seconds: int = 300,
    clock=None,
) -> PostFillCooldownRule:
    """
    Create a PostFillCooldownRule subscribed to the given bus.

    Args:
        clock: Optional callable returning current time (float).
               Pass a FakeClock instance for time-controlled tests.
    """
    return PostFillCooldownRule(
        cooldown_seconds=cooldown_seconds,
        bus=bus,
        _clock=clock,
    )


class FakeClock:
    """
    Controllable wall-clock substitute.

    Starts at a fixed epoch and advances only when explicitly told to.
    This lets tests simulate the passage of minutes without sleeping.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        self._now += seconds


def _make_fill(symbol: str, ts: float | None = None) -> FillEvent:
    """Create a FillEvent for the given symbol."""
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=ts if ts is not None else time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("0.0"),
        commission_asset="USD",
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


# ---------------------------------------------------------------------------
# Test 1: DENY within cooldown period
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_within_cooldown(bus):
    """
    After a fill, a new order for the same symbol is immediately denied
    because the cooldown window has not expired.

    Uses FakeClock fixed at a single point in time so elapsed=0 < cooldown.
    """
    clock = FakeClock(start=1_000_000.0)
    rule = _make_rule(bus, cooldown_seconds=300, clock=clock)

    # Publish a fill event — FillEvent.timestamp is the fill time recorded
    # by the rule's _on_fill handler.
    fill = _make_fill("BTC/USD", ts=clock())
    await bus.publish(fill)
    await asyncio.sleep(0.15)  # allow bus dispatch to rule's handler

    # Clock has NOT advanced — elapsed = 0 < 300s cooldown
    signal = _make_signal("BTC/USD")
    order = _make_order("BTC/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY within cooldown, got {result.decision}: {result.reason}"
    )
    assert "cooldown" in result.reason.lower()
    assert "BTC/USD" in result.reason


# ---------------------------------------------------------------------------
# Test 2: APPROVE after cooldown expires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_after_cooldown(bus):
    """
    After the cooldown window expires, the same symbol's orders are approved.

    FakeClock is advanced past the cooldown threshold between the fill and
    the evaluate() call — no mocking, no sleeping.
    """
    clock = FakeClock(start=1_000_000.0)
    rule = _make_rule(bus, cooldown_seconds=60, clock=clock)

    fill = _make_fill("ETH/USD", ts=clock())
    await bus.publish(fill)
    await asyncio.sleep(0.15)

    # Advance clock 61 seconds — cooldown has expired
    clock.advance(61.0)

    signal = _make_signal("ETH/USD")
    order = _make_order("ETH/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE after cooldown, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 3: Per-symbol independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    """
    A fill for BTC/USD should not impose a cooldown on ETH/USD orders.
    """
    clock = FakeClock(start=1_000_000.0)
    rule = _make_rule(bus, cooldown_seconds=300, clock=clock)

    # Publish a fill for BTC/USD only
    fill = _make_fill("BTC/USD", ts=clock())
    await bus.publish(fill)
    await asyncio.sleep(0.15)

    # ETH/USD order should NOT be denied — no fill recorded for ETH
    eth_signal = _make_signal("ETH/USD")
    eth_order = _make_order("ETH/USD")
    eth_result = rule.evaluate(eth_signal, eth_order, portfolio=None)  # type: ignore[arg-type]

    assert eth_result.decision == RuleDecision.APPROVE, (
        f"ETH/USD order should be independent of BTC/USD fill. "
        f"Got {eth_result.decision}: {eth_result.reason}"
    )

    # BTC/USD should still be denied (clock not advanced)
    btc_signal = _make_signal("BTC/USD")
    btc_order = _make_order("BTC/USD")
    btc_result = rule.evaluate(btc_signal, btc_order, portfolio=None)  # type: ignore[arg-type]

    assert btc_result.decision == RuleDecision.DENY, (
        f"BTC/USD order should still be denied. Got {btc_result.decision}: {btc_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 4: FILL event auto-records via bus subscription
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_event_auto_records(bus):
    """
    Publishing a FillEvent to the bus should automatically update the rule's
    internal _last_fill_time via the subscription set up in __init__.
    Verifies the bus subscription wiring works end-to-end.
    """
    clock = FakeClock(start=1_000_000.0)
    rule = _make_rule(bus, cooldown_seconds=300, clock=clock)

    # Before any fill: no cooldown — order should be approved
    signal = _make_signal("SOL/USD")
    order = _make_order("SOL/USD")
    result_before = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result_before.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE before any fill, got {result_before.decision}"
    )
    assert "SOL/USD" not in rule._last_fill_time, (
        "No fill should be recorded before publishing a FillEvent"
    )

    # Publish fill via bus — the rule's _on_fill handler receives it asynchronously
    fill = _make_fill("SOL/USD", ts=clock())
    await bus.publish(fill)
    await asyncio.sleep(0.15)  # allow async delivery to the rule's queue

    # _last_fill_time should now have an entry for SOL/USD
    assert "SOL/USD" in rule._last_fill_time, (
        "FillEvent should have been auto-recorded via bus subscription"
    )

    # Cooldown is now active — order denied
    result_after = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result_after.decision == RuleDecision.DENY, (
        f"Expected DENY after fill event auto-recorded, got {result_after.decision}"
    )
    assert "SOL/USD" in result_after.reason


# ---------------------------------------------------------------------------
# Test 5: No cooldown on first order (no prior fill)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cooldown_without_fill(bus):
    """
    With no prior fill for a symbol, orders should always be approved.
    """
    clock = FakeClock(start=1_000_000.0)
    rule = _make_rule(bus, cooldown_seconds=300, clock=clock)

    signal = _make_signal("ADA/USD")
    order = _make_order("ADA/USD")
    result = rule.evaluate(signal, order, portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE with no prior fill, got {result.decision}: {result.reason}"
    )
    assert result.risk_level == RiskLevel.LOW
