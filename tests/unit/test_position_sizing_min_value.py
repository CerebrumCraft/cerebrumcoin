"""
Unit tests for PositionSizingRule min_trade_value_usd parameter.

Tests the minimum trade value floor that prevents commission-dominated micro-trades.

@decision DEC-TEST-SIZING-001
@title Tests for PositionSizingRule min_trade_value_usd
@status accepted
@rationale min_trade_value_usd prevents structurally unprofitable trades where
commission (0.32% round-trip on Kraken) would exceed profit potential. Tests
verify: above-minimum passes, below-minimum denies, None default preserves
backward compatibility, and the strength-adjusted value (not raw target) is
compared against the floor.
"""

from decimal import Decimal
from time import time
from uuid import uuid4

from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderType,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import PositionSizingRule, RuleDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockPortfolio:
    """Minimal portfolio mock for PositionSizingRule tests."""

    def __init__(self, equity: Decimal, price: Decimal) -> None:
        self._equity = equity
        self._price = price

    def get_total_equity(self) -> Decimal:
        return self._equity

    def get_latest_price(self, symbol: str) -> Decimal:
        return self._price


def make_signal(strength: Decimal, symbol: str = "ETH/USD") -> SignalEvent:
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.COMBINED,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=strength,
        confidence=Decimal("0.8"),
    )


def make_order(price: Decimal, symbol: str = "ETH/USD") -> OrderEvent:
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        amount=Decimal("1.0"),
        price=price,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_above_minimum_returns_modify():
    """Trade above minimum -> MODIFY (normal sizing proceeds).

    Scenario: 5% of $5000 equity = $250 target. Strength=1.0 -> $250 value.
    Min is $100. $250 > $100 -> MODIFY.
    """
    rule = PositionSizingRule(
        position_size_percent=Decimal("5.0"),
        min_trade_value_usd=Decimal("100"),
    )
    # Price $1000, equity $5000 -> target $250 -> at strength 1.0 -> $250 value
    portfolio = MockPortfolio(equity=Decimal("5000"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("1.0"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.MODIFY, (
        f"Expected MODIFY for $250 trade (min $100), got {result.decision}: {result.reason}"
    )
    # Sanity: amount should be sized correctly
    # 5% of 5000 = 250, at price 1000 = 0.25 units, strength 1.0 -> 0.25
    assert result.modified_amount == Decimal("0.25")


def test_below_minimum_returns_deny():
    """Trade below minimum -> DENY with 'below minimum' in reason.

    Scenario: 2% of $1650 equity = $33 target. Strength=1.0 -> $33 value.
    Min is $100. $33 < $100 -> DENY.
    """
    rule = PositionSizingRule(
        position_size_percent=Decimal("2.0"),
        min_trade_value_usd=Decimal("100"),
    )
    # 2% of $1650 = $33 at strength 1.0
    portfolio = MockPortfolio(equity=Decimal("1650"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("1.0"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY for $33 trade (min $100), got {result.decision}: {result.reason}"
    )
    assert "below minimum" in result.reason.lower(), (
        f"Expected 'below minimum' in reason, got: {result.reason}"
    )


def test_no_minimum_allows_any_size():
    """No minimum set (None) -> MODIFY regardless of small size (backward compat).

    Scenario: tiny equity, no min_trade_value_usd -> rule passes through without check.
    """
    rule = PositionSizingRule(position_size_percent=Decimal("2.0"))  # min=None default

    # Very small trade: 2% of $100 = $2
    portfolio = MockPortfolio(equity=Decimal("100"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("1.0"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.MODIFY, (
        f"Expected MODIFY when no minimum is set, got {result.decision}: {result.reason}"
    )


def test_signal_multiplier_floored_at_06():
    """DEC-SIZING-002: signal strength below 0.6 is floored to 0.6.

    Scenario: 5% of $5000 = $250 target. Strength=0.3 -> floored to 0.6 -> $150 value.
    Without floor: 0.3 -> $75 -> DENY. With floor: 0.6 -> $150 -> MODIFY.
    """
    rule = PositionSizingRule(
        position_size_percent=Decimal("5.0"),
        min_trade_value_usd=Decimal("100"),
    )
    portfolio = MockPortfolio(equity=Decimal("5000"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("0.3"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.MODIFY, (
        f"Expected MODIFY (0.3 floored to 0.6 -> $150 > $100), got {result.decision}: {result.reason}"
    )
    # 5% of 5000 = 250, at price 1000 = 0.25 units, floored strength 0.6 -> 0.15
    assert result.modified_amount == Decimal("0.15"), (
        f"Expected 0.15 (0.25 * 0.6 floor), got {result.modified_amount}"
    )


def test_signal_multiplier_above_floor_unchanged():
    """DEC-SIZING-002: signal strength above 0.6 is used as-is.

    Scenario: 5% of $5000 = $250 target. Strength=0.8 -> $200 value (no floor).
    """
    rule = PositionSizingRule(
        position_size_percent=Decimal("5.0"),
        min_trade_value_usd=Decimal("100"),
    )
    portfolio = MockPortfolio(equity=Decimal("5000"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("0.8"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.MODIFY
    # 5% of 5000 = 250, at price 1000 = 0.25 units, strength 0.8 -> 0.20
    assert result.modified_amount == Decimal("0.200"), (
        f"Expected 0.200 (0.25 * 0.8), got {result.modified_amount}"
    )


def test_signal_multiplier_in_log_message():
    """DEC-SIZING-002: log message includes the effective multiplier value."""
    rule = PositionSizingRule(
        position_size_percent=Decimal("5.0"),
        min_trade_value_usd=Decimal("100"),
    )
    portfolio = MockPortfolio(equity=Decimal("5000"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("0.4"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert "multiplier=0.60" in result.reason, (
        f"Expected 'multiplier=0.60' (floored from 0.4) in reason, got: {result.reason}"
    )


def test_strength_adjusted_value_checked_not_raw():
    """Strength-adjusted value (not raw target) is compared against floor.

    Scenario: 5% of $2000 equity = $100 raw target. Strength=0.5 -> $50 adjusted value.
    Min is $100. $50 < $100 -> DENY (raw $100 would pass, adjusted $50 must not).
    """
    rule = PositionSizingRule(
        position_size_percent=Decimal("5.0"),
        min_trade_value_usd=Decimal("100"),
    )
    # 5% of $2000 = $100 raw; strength 0.5 -> $50 effective
    portfolio = MockPortfolio(equity=Decimal("2000"), price=Decimal("1000"))
    signal = make_signal(strength=Decimal("0.5"))
    order = make_order(price=Decimal("1000"))

    result = rule.evaluate(signal, order, portfolio)

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY for $50 adjusted value (min $100), got {result.decision}: {result.reason}"
    )
    assert "below minimum" in result.reason.lower(), (
        f"Expected 'below minimum' in reason, got: {result.reason}"
    )
