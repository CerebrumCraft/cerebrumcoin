"""
Unit tests for StalenessGateRule.

All time comparisons use an injected _today_fn so tests are not coupled
to wall-clock time. Today is pinned to 2026-04-19.

Coverage:
  1. Reject a filing 46 days old (exceeds 45-day ceiling)
  2. Accept a filing 44 days old (within ceiling)
  3. Edge day — exactly 45 days old is within ceiling (≤ vs >)
  4. Non-congressional signal (no filing_date in metadata) → approve unconditionally
  5. Unparseable filing_date → deny

@decision DEC-TEST-STALENESS-001
@title Inject fixed today_fn to make staleness tests time-independent
@status accepted
@rationale StalenessGateRule compares today - filing_date > ceiling. Tests that
depend on datetime.now() would drift as real calendar advances. Injectable
_today_fn fixes the comparison date to 2026-04-19, making tests deterministic
for all future runs without needing freezegun or unittest.mock.datetime.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from time import time
from uuid import uuid4

import pytest

from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import EventType, OrderType, RiskLevel, Side, SignalAction, SignalType
from cerebrum.risk.rules import RuleDecision
from cerebrum.risk.staleness_gate import StalenessGateRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date(2026, 4, 19)


def _today() -> date:
    return _TODAY


def _make_rule(ceiling: int = 45) -> StalenessGateRule:
    return StalenessGateRule(staleness_ceiling_days=ceiling, _today_fn=_today)


def _make_signal(filing_date: str | None) -> SignalEvent:
    """Build a minimal SignalEvent with optional filing_date in metadata."""
    metadata = {"source": "Congressional", "filing_date": filing_date} if filing_date is not None else None
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.NEWS,
        symbol="NVDA",
        action=SignalAction.BUY,
        strength=Decimal("0.75"),
        confidence=Decimal("0.65"),
        metadata=metadata,
    )


def _make_order() -> OrderEvent:
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol="NVDA",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1"),
    )


class _FakePortfolio:
    """Minimal portfolio stub — StalenessGateRule never reads portfolio."""
    pass


# ---------------------------------------------------------------------------
# 1. Reject a 46-day-old filing
# ---------------------------------------------------------------------------


def test_rejects_46_day_old_filing() -> None:
    """Filing 46 days before today (2026-03-04) must be denied."""
    # 2026-04-19 - 46 days = 2026-03-04
    signal = _make_signal("2026-03-04")
    order = _make_order()
    rule = _make_rule(ceiling=45)

    result = rule.evaluate(signal, order, _FakePortfolio())

    assert result.decision == RuleDecision.DENY
    assert "46" in result.reason   # age should appear in the reason
    assert "45" in result.reason   # ceiling should appear in the reason


# ---------------------------------------------------------------------------
# 2. Accept a 44-day-old filing
# ---------------------------------------------------------------------------


def test_accepts_44_day_old_filing() -> None:
    """Filing 44 days before today (2026-03-06) must be approved."""
    # 2026-04-19 - 44 days = 2026-03-06
    signal = _make_signal("2026-03-06")
    order = _make_order()
    rule = _make_rule(ceiling=45)

    result = rule.evaluate(signal, order, _FakePortfolio())

    assert result.decision == RuleDecision.APPROVE


# ---------------------------------------------------------------------------
# 3. Edge case — exactly 45 days old is accepted (> not >=)
# ---------------------------------------------------------------------------


def test_exactly_45_days_accepted() -> None:
    """Filing exactly 45 days old (2026-03-05) must be approved — boundary is exclusive."""
    # 2026-04-19 - 45 days = 2026-03-05
    signal = _make_signal("2026-03-05")
    order = _make_order()
    rule = _make_rule(ceiling=45)

    result = rule.evaluate(signal, order, _FakePortfolio())

    assert result.decision == RuleDecision.APPROVE


# ---------------------------------------------------------------------------
# 4. Non-congressional signal (no filing_date) → approve unconditionally
# ---------------------------------------------------------------------------


def test_no_filing_date_approves() -> None:
    """A SignalEvent without metadata (e.g. technical signal) is approved."""
    # Build signal with no metadata at all
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="NVDA",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.7"),
        metadata=None,
    )
    order = _make_order()
    rule = _make_rule()

    result = rule.evaluate(signal, order, _FakePortfolio())

    assert result.decision == RuleDecision.APPROVE


# ---------------------------------------------------------------------------
# 5. Unparseable filing_date → deny
# ---------------------------------------------------------------------------


def test_unparseable_filing_date_denied() -> None:
    """A filing_date that cannot be parsed as ISO-8601 must be denied."""
    signal = _make_signal("not-a-date")
    order = _make_order()
    rule = _make_rule()

    result = rule.evaluate(signal, order, _FakePortfolio())

    assert result.decision == RuleDecision.DENY
    assert "not-a-date" in result.reason
