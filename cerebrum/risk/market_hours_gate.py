"""MarketHoursGateRule — denies stock orders outside RTH.

@decision DEC-STOCKS-003
@title RTH-only enforcement with entry cutoff before close
@status accepted
@rationale Prevents the risk manager from approving stock orders during
non-trading hours, weekends, NYSE holidays, or the last N minutes before
close (entry cutoff). Crypto symbols pass through unchanged based on the
`stock_symbols` membership check.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from cerebrum.risk.rules import RiskRule, RuleDecision, RuleResult, RiskLevel
from cerebrum.utils.trading_session import (
    is_rth_now,
    minutes_until_close,
)


class MarketHoursGateRule(RiskRule):
    """Deny stock orders outside regular trading hours or within entry cutoff."""

    RULE_NAME = "market_hours_gate"

    def __init__(
        self,
        stock_symbols: list[str],
        entry_cutoff_minutes_before_close: int = 15,
        now_utc_provider: Callable[[], datetime] | None = None,
    ):
        super().__init__(self.RULE_NAME)
        self._stock_symbols: set[str] = set(stock_symbols)
        self._entry_cutoff: int = int(entry_cutoff_minutes_before_close)
        self._now_utc_provider = now_utc_provider
        self._now_utc_override: datetime | None = None  # test hook

    def _current_time(self) -> datetime | None:
        if self._now_utc_override is not None:
            return self._now_utc_override
        if self._now_utc_provider is not None:
            return self._now_utc_provider()
        return None  # None → trading_session uses real clock

    def evaluate(self, signal: Any, order: Any, portfolio: Any) -> RuleResult:
        symbol = order.symbol
        if symbol not in self._stock_symbols:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="market_hours_gate: non-stock symbol passes through",
                risk_level=RiskLevel.LOW,
            )

        now = self._current_time()

        if not is_rth_now(now):
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"market_hours_gate: {symbol} outside RTH (weekend/holiday/before-open/after-close)",
                risk_level=RiskLevel.LOW,
            )

        mins = minutes_until_close(now)
        if mins is None:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"market_hours_gate: {symbol} minutes_until_close unavailable (market closed)",
                risk_level=RiskLevel.LOW,
            )

        if mins < self._entry_cutoff:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"market_hours_gate: {symbol} within {self._entry_cutoff}-min entry cutoff "
                    f"before close ({mins} min remaining)"
                ),
                risk_level=RiskLevel.LOW,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=f"market_hours_gate: {symbol} inside RTH with {mins} min to close",
            risk_level=RiskLevel.LOW,
        )
