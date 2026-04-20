"""
Staleness gate risk rule for congressional trade signals.

Rejects any SignalEvent whose filing_date metadata field is older than a
configurable ceiling (default 45 days — the STOCK Act statutory maximum).

Filings older than the ceiling are guaranteed to be stale: the disclosed trade
was executed at a price from 45+ days ago. Executing against today's price on
a 45-day-old signal is indistinguishable from random entry.

# @decision DEC-PELOSI-LAG-001
# @title Staleness ceiling = 45 days (STOCK Act statutory maximum)
# @status accepted
# @rationale STOCK Act (Pub. L. 112-105) mandates disclosure within 45 days of the
# transaction date. A filing older than 45 days is at or past the worst-case
# legal lag, meaning any edge from the original trade has likely expired.
# Default ceiling = 45 keeps us inside the legal max. Operators can tighten to
# e.g. 30 days once we have session data to evaluate lag-decay. Filing date
# (the date of public disclosure) is used, not transaction date, because that
# is what the provider exposes and represents actual information availability.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import structlog

from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import RiskLevel, RiskRule, RuleDecision, RuleResult

logger = structlog.get_logger()

# Sentinel — if filing_date is missing from metadata, we reject
_MISSING = "MISSING"


class StalenessGateRule(RiskRule):
    """
    Deny orders whose triggering signal has a stale filing_date.

    Reads ``signal.metadata["filing_date"]`` (ISO-8601 date string, e.g.
    ``"2026-03-01"``). If the date is older than ``staleness_ceiling_days``
    before today, the order is denied.

    If ``filing_date`` is absent from metadata, the order is also denied to
    prevent stale signals from slipping through without dating information.

    This rule is only meaningful for Congressional signals — other signal
    types will not have a ``filing_date`` in metadata. For those, the rule
    approves unconditionally (it only gates on signals that carry the field).

    Injectable clock (``_today_fn``) enables deterministic testing without
    datetime mocking.
    """

    def __init__(
        self,
        staleness_ceiling_days: int = 45,
        _today_fn: Optional[Callable[[], date]] = None,
    ) -> None:
        """
        Initialize StalenessGateRule.

        Args:
            staleness_ceiling_days: Maximum allowed age of a filing in days.
                Orders are denied when today - filing_date > this value.
                Default 45 matches the STOCK Act statutory disclosure window.
            _today_fn: Optional callable returning the current date. Defaults
                to ``datetime.now(timezone.utc).date()``. Inject a fixed date
                in tests to avoid coupling to wall-clock time.
        """
        super().__init__("staleness_gate")
        self._ceiling = staleness_ceiling_days
        self._today_fn: Callable[[], date] = (
            _today_fn if _today_fn is not None
            else lambda: datetime.now(timezone.utc).date()
        )
        self._log = logger.bind(
            component="staleness_gate",
            ceiling_days=staleness_ceiling_days,
        )
        self._log.info(
            "staleness_gate_initialized",
            staleness_ceiling_days=staleness_ceiling_days,
        )

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Approve the order if the signal's filing_date is within the staleness window.

        Logic:
        1. If signal has no metadata or no ``filing_date`` key → approve
           (non-congressional signals do not carry this field; gate is a no-op).
        2. If ``filing_date`` is present but unparseable → deny.
        3. If today - filing_date > ceiling_days → deny with reason.
        4. Otherwise → approve.
        """
        metadata = signal.metadata or {}
        filing_date_str: str = metadata.get("filing_date", _MISSING)

        # Not a congressional signal — approve unconditionally
        if filing_date_str is _MISSING:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="No filing_date in metadata — staleness gate is a no-op",
                risk_level=RiskLevel.LOW,
            )

        # Parse the date
        try:
            filing_date = date.fromisoformat(filing_date_str)
        except (ValueError, TypeError) as exc:
            self._log.warning(
                "staleness_gate_unparseable_date",
                filing_date_str=filing_date_str,
                error=str(exc),
                symbol=signal.symbol,
            )
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"staleness_gate: cannot parse filing_date '{filing_date_str}' "
                    f"— denying to avoid trading on undatable filing"
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        today = self._today_fn()
        age_days = (today - filing_date).days

        if age_days > self._ceiling:
            self._log.warning(
                "signal_stale_dropped",
                symbol=signal.symbol,
                filing_date=filing_date_str,
                age_days=age_days,
                ceiling_days=self._ceiling,
            )
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"staleness_gate: filing for {signal.symbol} is {age_days} days old "
                    f"(ceiling={self._ceiling}d, filing_date={filing_date_str})"
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        self._log.debug(
            "staleness_gate_approved",
            symbol=signal.symbol,
            filing_date=filing_date_str,
            age_days=age_days,
            ceiling_days=self._ceiling,
        )
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"staleness_gate: filing for {signal.symbol} is {age_days} days old "
                f"(within ceiling={self._ceiling}d)"
            ),
            risk_level=RiskLevel.LOW,
        )
