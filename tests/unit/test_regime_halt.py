"""
Tests for RegimeTradeHaltRule.

@decision DEC-TEST-011
@title Regime trade halt rule tests
@status accepted
@rationale Validates that trading halts during high-confidence BEAR and permits
during non-BEAR or low-confidence BEAR. Uses a real EventBus and real
RegimeChangeEvent (same pattern as test_cooldown_rule.py) to validate the bus
subscription wiring end-to-end. The evaluate() signature requires signal, order,
and portfolio arguments — portfolio is passed as None (unused by this rule).
"""

import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import OrderEvent, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import RegimeTradeHaltRule, RuleDecision


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start a real EventBus."""
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def halt_rule(bus):
    """RegimeTradeHaltRule with 0.7 confidence threshold, subscribed to bus."""
    return RegimeTradeHaltRule(min_confidence=Decimal("0.7"), bus=bus)


def _make_regime_event(
    symbol: str,
    to_regime: str,
    confidence: str,
    from_regime: str = "SIDEWAYS",
    ts: float = 1.0,
) -> RegimeChangeEvent:
    """Construct a RegimeChangeEvent with the given symbol embedded in indicators."""
    return RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=ts,
        from_regime=from_regime,
        to_regime=to_regime,
        confidence=Decimal(confidence),
        indicators={"symbol": symbol},
    )


def _make_order(symbol: str = "BTC/USD", side: str = "buy") -> OrderEvent:
    """Create a minimal OrderEvent."""
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY if side == "buy" else Side.SELL,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bear_high_confidence_halts_buy(halt_rule):
    """BEAR with confidence >= 0.7 should deny a buy order."""
    event = _make_regime_event("BTC/USD", "BEAR", "0.8")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY
    assert "BEAR" in result.reason
    assert result.risk_level == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_bear_high_confidence_halts_sell(halt_rule):
    """BEAR with confidence >= 0.7 should also deny a sell order."""
    event = _make_regime_event("BTC/USD", "BEAR", "0.9")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD", "sell"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY
    assert "BEAR" in result.reason


@pytest.mark.asyncio
async def test_bear_low_confidence_allows_trading(halt_rule):
    """BEAR with confidence < 0.7 should allow orders through."""
    event = _make_regime_event("BTC/USD", "BEAR", "0.5")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_bear_at_threshold_halts_trading(halt_rule):
    """BEAR at exactly the threshold (0.7) should trigger the halt."""
    event = _make_regime_event("BTC/USD", "BEAR", "0.7")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY


@pytest.mark.asyncio
async def test_sideways_allows_trading(halt_rule):
    """SIDEWAYS regime (any confidence) should always allow orders."""
    event = _make_regime_event("ETH/USD", "SIDEWAYS", "0.9")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal("ETH/USD"), _make_order("ETH/USD"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_bull_allows_trading(halt_rule):
    """BULL regime should allow orders regardless of confidence."""
    event = _make_regime_event("BTC/USD", "BULL", "0.9", from_regime="BEAR")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_unknown_symbol_allows_trading(halt_rule):
    """Symbol with no regime data recorded should allow orders."""
    result = halt_rule.evaluate(_make_signal("XRP/USD"), _make_order("XRP/USD"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_halt_is_per_symbol(halt_rule):
    """BEAR halt on BTC/USD should not affect ETH/USD."""
    btc_event = _make_regime_event("BTC/USD", "BEAR", "0.9")
    eth_event = _make_regime_event("ETH/USD", "BULL", "0.8", from_regime="UNKNOWN")

    await halt_rule._on_regime_change(btc_event)
    await halt_rule._on_regime_change(eth_event)

    btc_result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert btc_result.decision == RuleDecision.DENY  # BTC halted

    eth_result = halt_rule.evaluate(_make_signal("ETH/USD"), _make_order("ETH/USD"), None)  # type: ignore[arg-type]
    assert eth_result.decision == RuleDecision.APPROVE  # ETH still trading


@pytest.mark.asyncio
async def test_regime_change_from_bear_resumes_trading(halt_rule):
    """When regime changes from BEAR to non-BEAR, trading should resume."""
    bear_event = _make_regime_event("BTC/USD", "BEAR", "0.9", ts=1.0)
    await halt_rule._on_regime_change(bear_event)

    halted = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert halted.decision == RuleDecision.DENY

    # Regime transitions to SIDEWAYS
    sideways_event = _make_regime_event("BTC/USD", "SIDEWAYS", "0.6", from_regime="BEAR", ts=2.0)
    await halt_rule._on_regime_change(sideways_event)

    resumed = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert resumed.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_bus_subscription_wiring(bus):
    """Publishing a RegimeChangeEvent via bus should update the rule's state."""
    import asyncio

    rule = RegimeTradeHaltRule(min_confidence=Decimal("0.7"), bus=bus)

    # Before any event: no regime data — should approve
    result_before = rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert result_before.decision == RuleDecision.APPROVE
    assert "BTC/USD" not in rule._regimes

    # Publish via bus — rule's _on_regime_change receives it asynchronously
    event = _make_regime_event("BTC/USD", "BEAR", "0.85")
    await bus.publish(event)
    await asyncio.sleep(0.15)  # allow async dispatch to rule's handler

    # _regimes should now record BTC/USD
    assert "BTC/USD" in rule._regimes
    regime, confidence = rule._regimes["BTC/USD"]
    assert regime == "BEAR"
    assert confidence == Decimal("0.85")

    # Now the rule should deny orders for BTC/USD
    result_after = rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert result_after.decision == RuleDecision.DENY


@pytest.mark.asyncio
async def test_reason_includes_confidence_and_threshold(halt_rule):
    """Denial reason should include numeric confidence and threshold for observability."""
    event = _make_regime_event("BTC/USD", "BEAR", "0.85")
    await halt_rule._on_regime_change(event)

    result = halt_rule.evaluate(_make_signal(), _make_order("BTC/USD"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY
    assert "0.85" in result.reason
    assert "0.70" in result.reason
    assert "BTC/USD" in result.reason


# ---------------------------------------------------------------------------
# DEC-REGIME-006: UNKNOWN regime halt tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_regime_halts_trading(bus):
    """UNKNOWN regime should be halted when included in halt_regimes."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR", "UNKNOWN"},
    )
    event = _make_regime_event("BTC/USD", "UNKNOWN", "0.5")
    await rule._on_regime_change(event)

    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY
    assert "UNKNOWN" in result.reason


@pytest.mark.asyncio
async def test_unknown_regime_allowed_when_not_in_halt_regimes(bus):
    """UNKNOWN should be allowed when halt_regimes only contains BEAR."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR"},
    )
    event = _make_regime_event("BTC/USD", "UNKNOWN", "0.5")
    await rule._on_regime_change(event)

    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_no_regime_data_halts_when_unknown_in_halt_regimes(bus):
    """When no regime event received yet and UNKNOWN is in halt_regimes, should deny."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR", "UNKNOWN"},
    )
    # No regime event published — _regimes is empty for this symbol
    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)  # type: ignore[arg-type]
    assert result.decision == RuleDecision.DENY
