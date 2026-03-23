"""
Tests for GlobalTradeRateLimitRule.

Follows the PostFillCooldownRule test pattern: real EventBus, injectable
FakeClock for deterministic time control, no mocks of internal modules.
Uses asyncio_mode=auto (pytest.ini) so all async tests run without explicit
@pytest.mark.asyncio decoration.

@decision DEC-STRAT-007
@title GlobalTradeRateLimitRule for cross-strategy commission control
@status accepted
@rationale Session 4 showed 64% commission drag from rapid-fire fills.
Per-symbol cooldown (DEC-COOL-001) controls intra-strategy rate; global
rate limit controls total system throughput across all strategies.
"""

from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent, SignalEvent
from cerebrum.core.types import EventType, OrderType, RiskLevel, Side, SignalAction, SignalType
from cerebrum.risk.global_trade_rate import GlobalTradeRateLimitRule
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import RuleDecision


class FakeClock:
    """Injectable clock for deterministic time control."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


def make_fill(
    symbol: str = "BTC/USD",
    ts: float = 1000.0,
    strategy_id: str | None = None,
) -> FillEvent:
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=ts,
        order_id=f"o_{ts}_{strategy_id}",
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000.0"),
        commission=Decimal("8.0"),
        commission_asset="USD",
        strategy_id=strategy_id,
    )


def make_signal(symbol: str = "BTC/USD") -> SignalEvent:
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=1000.0,
        signal_type=SignalType.COMBINED,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
    )


def make_order(symbol: str = "BTC/USD") -> OrderEvent:
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=1000.0,
        order_id="ord1",
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.1"),
    )


class TestGlobalTradeRateLimitUnderLimit:
    """APPROVE when fill count is below the limit."""

    async def test_approve_when_no_fills(self, bus):
        """Fresh rule with no fills approves immediately."""
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=10, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.APPROVE

    async def test_approve_when_below_limit(self, bus):
        """After 3 fills with limit=5, next order is approved."""
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=5, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        import asyncio
        for i in range(3):
            await bus.publish(make_fill(ts=1000.0 + i))
        await asyncio.sleep(0.05)

        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.APPROVE
        assert "3/5" in result.reason


class TestGlobalTradeRateLimitAtLimit:
    """DENY when fill count reaches the limit."""

    async def test_deny_at_limit(self, bus):
        """Exactly max_trades fills → DENY."""
        import asyncio
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=3, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        for i in range(3):
            await bus.publish(make_fill(ts=1000.0 + i))
        await asyncio.sleep(0.05)

        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.DENY
        assert result.risk_level == RiskLevel.MEDIUM
        assert "3/3" in result.reason

    async def test_deny_above_limit(self, bus):
        """More fills than limit → still DENY."""
        import asyncio
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=2, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        for i in range(5):
            await bus.publish(make_fill(ts=1000.0 + i))
        await asyncio.sleep(0.05)

        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.DENY


class TestGlobalTradeRateLimitExpiry:
    """Old fills expire from the rolling window after 1 hour."""

    async def test_fills_expire_after_one_hour(self, bus):
        """Fills older than 3600s are pruned; previously denied becomes APPROVE."""
        import asyncio
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=2, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        # Record 2 fills at t=1000
        for i in range(2):
            await bus.publish(make_fill(ts=1000.0 + i))
        await asyncio.sleep(0.05)

        # At t=1000 → DENY (limit reached)
        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.DENY

        # Advance clock past 1 hour from the first fill (t=1000 + 3601 = 4601)
        clock.advance(3601.0)

        # Now both fills are older than 1h → pruned → APPROVE
        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.APPROVE

    async def test_partial_expiry(self, bus):
        """Only fills older than 1h expire; recent fills still count."""
        import asyncio
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=3, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        # Old fill at t=1000
        await bus.publish(make_fill(ts=1000.0))
        await asyncio.sleep(0.05)

        # Advance 2000s and add two more fills
        clock.advance(2000.0)
        for i in range(2):
            await bus.publish(make_fill(ts=3000.0 + i))
        await asyncio.sleep(0.05)

        # At t=3000: all 3 fills within 1h window → DENY (limit=3)
        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.DENY

        # Advance past 1h from the first fill: clock goes to 4602
        # (3000 + 1602 > 1000 + 3600 = 4600)
        clock.advance(1602.0)

        # Old fill at t=1000 has expired; 2 recent fills remain → APPROVE
        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.APPROVE
        assert "2/3" in result.reason


class TestGlobalTradeRateLimitCrossStrategy:
    """Fills from multiple strategies are all counted."""

    async def test_fills_from_different_strategies_counted_together(self, bus):
        """Global rate counts fills regardless of strategy_id."""
        import asyncio
        clock = FakeClock(1000.0)
        rule = GlobalTradeRateLimitRule(max_trades_per_hour=3, bus=bus, _clock=clock)
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        await bus.publish(make_fill(strategy_id="alpha", ts=1000.0))
        await bus.publish(make_fill(strategy_id="beta", ts=1001.0))
        await bus.publish(make_fill(strategy_id="gamma", ts=1002.0))
        await asyncio.sleep(0.05)

        # 3 fills from 3 strategies → limit reached → DENY
        result = rule.evaluate(make_signal(), make_order(), portfolio)
        assert result.decision == RuleDecision.DENY
