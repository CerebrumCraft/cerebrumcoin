"""
Tests for MaxOpenPositionsRule.

@decision DEC-TEST-014
@title MaxOpenPositionsRule tests
@status accepted
@rationale Validates position count cap per (strategy, symbol). Uses real EventBus
and FillEvent to test the fill-tracking subscription.
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
from cerebrum.risk.rules import MaxOpenPositionsRule, RuleDecision


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


def _make_fill(symbol="BTC/USD", side="buy", strategy_id="mean_reversion"):
    return FillEvent(
        event_type=EventType.FILL, timestamp=time.time(), order_id=str(uuid4()),
        symbol=symbol, side=Side.BUY if side == "buy" else Side.SELL,
        filled_amount=Decimal("0.01"), fill_price=Decimal("50000"),
        commission=Decimal("0.08"), commission_asset="USD", strategy_id=strategy_id,
    )


def _make_order(symbol="BTC/USD", side="buy", strategy_id="mean_reversion"):
    return OrderEvent(
        event_type=EventType.ORDER, timestamp=time.time(), order_id=str(uuid4()),
        symbol=symbol, side=Side.BUY if side == "buy" else Side.SELL,
        order_type=OrderType.MARKET, amount=Decimal("0.01"),
        status=OrderStatus.PENDING, strategy_id=strategy_id,
    )


def _make_signal(symbol="BTC/USD"):
    return SignalEvent(
        event_type=EventType.SIGNAL, timestamp=time.time(),
        signal_type=SignalType.TECHNICAL, symbol=symbol,
        action=SignalAction.BUY, strength=Decimal("0.8"), confidence=Decimal("0.7"),
    )


@pytest.mark.asyncio
async def test_approve_below_limit(bus):
    rule = MaxOpenPositionsRule(max_positions=2, bus=bus)
    result = rule.evaluate(_make_signal(), _make_order(), None)
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_deny_at_limit(bus):
    rule = MaxOpenPositionsRule(max_positions=2, bus=bus)
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "mean_reversion"), None)
    assert result.decision == RuleDecision.DENY


@pytest.mark.asyncio
async def test_always_approve_sells(bus):
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "sell", "mean_reversion"), None)
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_sell_decrements_counter(bus):
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    assert rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "mean_reversion"), None).decision == RuleDecision.DENY
    await rule._on_fill(_make_fill("BTC/USD", "sell", "mean_reversion"))
    assert rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "mean_reversion"), None).decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    assert rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "mean_reversion"), None).decision == RuleDecision.DENY
    assert rule.evaluate(_make_signal("ETH/USD"), _make_order("ETH/USD", "buy", "mean_reversion"), None).decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_per_strategy_independence(bus):
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    assert rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "mean_reversion"), None).decision == RuleDecision.DENY
    assert rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy", "range_trading"), None).decision == RuleDecision.APPROVE
