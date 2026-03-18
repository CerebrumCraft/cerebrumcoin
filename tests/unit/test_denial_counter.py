"""
Unit tests for RiskManager denial counters and Dashboard Guard Denials display.

Tests:
1. denial_counts starts empty
2. Counter increments on DENY
3. Counter does NOT increment on APPROVE or MODIFY
4. Multiple denials from same rule accumulate correctly
5. Multiple rules tracked independently
6. denial_counts returns a copy (mutations don't affect live dict)
7. Dashboard Guard Denials section renders when risk_manager provided
8. Dashboard Guard Denials section absent when risk_manager is None

@decision DEC-TEST-017
@title Tests for RiskManager denial counters
@status accepted
@rationale Verifies counters increment on DENY, remain at zero for
APPROVE/MODIFY, denial_counts returns a copy not the live dict,
and Dashboard correctly renders the Guard Denials section.
"""

import asyncio
import tempfile
from decimal import Decimal
from io import StringIO
from pathlib import Path
from time import time
from unittest.mock import patch  # @mock-exempt: patching builtins.print to capture stdout — console I/O boundary, not internal code
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import EventType, RiskLevel, Side, SignalAction, SignalType
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    MinSignalStrengthRule,
    PositionSizingRule,
    RuleDecision,
    RuleResult,
    RiskRule,
)
from cerebrum.core.state import StateManager
from cerebrum.monitoring.dashboard import Dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class AlwaysDenyRule(RiskRule):
    """Test rule that always returns DENY."""

    def __init__(self) -> None:
        super().__init__("always_deny")

    def evaluate(self, signal, order, portfolio) -> RuleResult:
        return RuleResult(
            decision=RuleDecision.DENY,
            reason="test deny",
            risk_level=RiskLevel.LOW,
        )


class AlwaysApproveRule(RiskRule):
    """Test rule that always returns APPROVE."""

    def __init__(self) -> None:
        super().__init__("always_approve")

    def evaluate(self, signal, order, portfolio) -> RuleResult:
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason="test approve",
            risk_level=RiskLevel.LOW,
        )


class AlwaysModifyRule(RiskRule):
    """Test rule that always returns MODIFY with a small amount."""

    def __init__(self) -> None:
        super().__init__("always_modify")

    def evaluate(self, signal, order, portfolio) -> RuleResult:
        return RuleResult(
            decision=RuleDecision.MODIFY,
            reason="test modify",
            risk_level=RiskLevel.LOW,
            modified_amount=Decimal("0.001"),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    b = EventBus(queue_size=50)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def portfolio(bus):
    return PortfolioTracker(bus, initial_balance=Decimal("10000.0"))


@pytest.fixture
async def state_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()
        yield state
        await state.close()


# ---------------------------------------------------------------------------
# Tests: denial_counts initial state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denial_counts_starts_empty(bus, portfolio):
    """denial_counts must be an empty dict on initialization."""
    risk_manager = RiskManager(bus, portfolio, rules=[])
    assert risk_manager.denial_counts == {}


# ---------------------------------------------------------------------------
# Tests: counter increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denial_counts_increments_on_deny(bus, portfolio):
    """Sending a signal that triggers a DENY must increment the rule's counter."""
    rule = AlwaysDenyRule()
    risk_manager = RiskManager(bus, portfolio, rules=[rule])

    # Seed price so PositionSizingRule (if present) has data
    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

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
    await asyncio.sleep(0.2)

    counts = risk_manager.denial_counts
    assert counts.get("always_deny", 0) == 1


@pytest.mark.asyncio
async def test_denial_counts_accumulates(bus, portfolio):
    """Multiple signals that hit the same DENY rule must accumulate the count."""
    rule = AlwaysDenyRule()
    risk_manager = RiskManager(bus, portfolio, rules=[rule])

    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

    for _ in range(3):
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
        await asyncio.sleep(0.1)

    assert risk_manager.denial_counts.get("always_deny", 0) == 3


# ---------------------------------------------------------------------------
# Tests: counter does NOT increment for APPROVE or MODIFY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denial_counts_no_increment_on_approve(bus, portfolio):
    """APPROVE rules must not appear in denial_counts."""
    # Use MinSignalStrength with low threshold (will always APPROVE for strong signal)
    # followed by PositionSizing (will MODIFY)
    rules = [
        AlwaysApproveRule(),
        AlwaysModifyRule(),
    ]
    risk_manager = RiskManager(bus, portfolio, rules=rules)

    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

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
    await asyncio.sleep(0.2)

    counts = risk_manager.denial_counts
    assert "always_approve" not in counts
    assert "always_modify" not in counts


# ---------------------------------------------------------------------------
# Tests: multiple rules tracked independently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denial_counts_multiple_rules_independent(bus, portfolio):
    """Each rule's counter is tracked separately; only the first DENY fires."""
    deny1 = AlwaysDenyRule()
    deny1._RiskRule__dict = {}  # ensure distinct instance
    deny1.name = "deny_rule_1"

    deny2 = AlwaysDenyRule()
    deny2.name = "deny_rule_2"

    # deny_rule_1 fires first and blocks — deny_rule_2 never evaluated
    risk_manager = RiskManager(bus, portfolio, rules=[deny1, deny2])

    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

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
    await asyncio.sleep(0.2)

    counts = risk_manager.denial_counts
    # Only the first deny rule fires; second rule never reached
    assert counts.get("deny_rule_1", 0) == 1
    assert counts.get("deny_rule_2", 0) == 0


# ---------------------------------------------------------------------------
# Tests: denial_counts returns a copy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denial_counts_returns_copy(bus, portfolio):
    """Mutating the returned dict must not affect internal state."""
    rule = AlwaysDenyRule()
    risk_manager = RiskManager(bus, portfolio, rules=[rule])

    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

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
    await asyncio.sleep(0.2)

    counts = risk_manager.denial_counts
    counts["always_deny"] = 9999  # mutate the copy

    # Internal state must be unchanged
    assert risk_manager.denial_counts.get("always_deny", 0) == 1


# ---------------------------------------------------------------------------
# Tests: Dashboard Guard Denials display
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_shows_guard_denials_when_risk_manager_provided(
    bus, state_manager
):
    """Guard Denials section must appear when risk_manager is provided."""
    # Create a RiskManager with a known denial
    portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
    deny_rule = AlwaysDenyRule()
    risk_manager = RiskManager(bus, portfolio, rules=[deny_rule])

    # Trigger one denial so counters are non-zero
    await bus.publish(MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol="BTC/USD",
        price=Decimal("50000"),
        volume=Decimal("1.0"),
    ))
    await asyncio.sleep(0.05)

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
    await asyncio.sleep(0.2)

    dashboard = Dashboard(
        bus,
        state_manager,
        update_interval_seconds=3600,
        risk_manager=risk_manager,
    )
    await dashboard.start()

    captured = StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(" ".join(str(x) for x in a) + "\n")):
        await dashboard._display_stats()

    await dashboard.stop()

    output = captured.getvalue()
    assert "Guard Denials" in output
    assert "always_deny" in output
    assert "1 denials" in output


@pytest.mark.asyncio
async def test_dashboard_no_guard_denials_when_no_risk_manager(bus, state_manager):
    """Guard Denials section must be absent when risk_manager=None (default)."""
    dashboard = Dashboard(
        bus,
        state_manager,
        update_interval_seconds=3600,
        # risk_manager not provided — defaults to None
    )
    await dashboard.start()

    captured = StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(" ".join(str(x) for x in a) + "\n")):
        await dashboard._display_stats()

    await dashboard.stop()

    output = captured.getvalue()
    assert "Guard Denials" not in output


@pytest.mark.asyncio
async def test_dashboard_guard_denials_no_denials_yet(bus, state_manager):
    """Guard Denials section with zero denials must show '(no denials yet)'."""
    portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
    risk_manager = RiskManager(bus, portfolio, rules=[AlwaysApproveRule()])

    dashboard = Dashboard(
        bus,
        state_manager,
        update_interval_seconds=3600,
        risk_manager=risk_manager,
    )
    await dashboard.start()

    captured = StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: captured.write(" ".join(str(x) for x in a) + "\n")):
        await dashboard._display_stats()

    await dashboard.stop()

    output = captured.getvalue()
    assert "Guard Denials" in output
    assert "no denials yet" in output
