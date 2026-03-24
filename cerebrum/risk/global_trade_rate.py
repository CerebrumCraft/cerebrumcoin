"""
GlobalTradeRateLimitRule: cross-strategy fill rate limiter.

Limits the total number of fills (executions) across ALL strategies per
rolling hour window. This is a system-level guard, distinct from the
per-symbol PostFillCooldownRule which operates within a single strategy.

Use case: with 3 strategies each capable of trading independently, the
combined fill rate could be 3x that of a single strategy. Without a global
cap, commission drag (the #1 enemy per session logs) triples. This rule
ensures the whole system stays within a sensible throughput budget.

@decision DEC-STRAT-007
@title GlobalTradeRateLimitRule for cross-strategy commission control
@status accepted
@rationale Session 4 showed 64% commission drag ($115 on $179 gross) from
rapid-fire fills. The per-symbol PostFillCooldownRule (DEC-COOL-001) limits
fills per symbol per strategy, but with multiple strategies each having their
own cooldown timer, the aggregate fill rate can still blow out. A global
rolling-window counter across all FillEvents bounds total throughput
regardless of strategy count. The 1-hour window matches the natural trading
rhythm observed across sessions: a burst of trades then a quiet period.
Design mirrors PostFillCooldownRule exactly: self-subscribes to FILL events,
injectable clock for testing, deque-based rolling window.
"""

import time as time_module
from collections import deque
from typing import TYPE_CHECKING

import structlog

from cerebrum.core.events import Event, FillEvent, OrderEvent, SignalEvent
from cerebrum.core.types import EventType, RiskLevel
from cerebrum.risk.rules import RiskRule, RuleResult, RuleDecision
from cerebrum.risk.portfolio import PortfolioTracker

if TYPE_CHECKING:
    from cerebrum.core.bus import EventBus

logger = structlog.get_logger()

_SECONDS_PER_HOUR = 3600


class GlobalTradeRateLimitRule(RiskRule):
    """
    Deny new orders when total fills across all strategies in the last hour
    exceeds the configured maximum.

    Self-subscribes to ALL FillEvents on the bus (not filtered by strategy_id)
    so it sees fills from every strategy. The rolling window is a deque of
    fill timestamps; entries older than 1 hour are pruned on every evaluate()
    call.

    Cold start: the deque starts empty, so the first max_trades_per_hour
    fills are always allowed. This is intentional — the system should trade
    freely at startup and only throttle when the rate is genuinely excessive.
    """

    def __init__(
        self,
        max_trades_per_hour: int,
        bus: "EventBus",
        _clock=None,
    ) -> None:
        """
        Initialize global trade rate limit rule.

        Args:
            max_trades_per_hour: Maximum fills (any strategy, any symbol)
                                 allowed in a rolling 1-hour window. Orders
                                 that would push the count above this are
                                 denied until old fills age out.
            bus: Event bus to subscribe to FillEvents.
            _clock: Callable returning current time as float (default: time.time).
                    Injectable for deterministic testing without sleeps.
        """
        super().__init__("global_trade_rate_limit")
        self._max_trades = max_trades_per_hour
        self._clock = _clock if _clock is not None else time_module.time
        # Deque of fill timestamps (unix epoch float). No maxlen — we prune by age.
        self._fill_times: deque[float] = deque()

        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="global_trade_rate_rule",
        )

        self._log.info(
            "global_trade_rate_limit_initialized",
            max_trades_per_hour=max_trades_per_hour,
        )

    async def _on_fill(self, event: Event) -> None:
        """Record fill timestamp when any FillEvent is received."""
        if not isinstance(event, FillEvent):
            return
        self._fill_times.append(event.timestamp)
        self._log.debug(
            "global_fill_recorded",
            timestamp=event.timestamp,
            symbol=event.symbol,
            strategy_id=event.strategy_id,
            total_in_window=len(self._fill_times),
        )

    def _prune_old_fills(self, current_time: float) -> None:
        """Remove fill timestamps older than 1 hour from the left of the deque."""
        cutoff = current_time - _SECONDS_PER_HOUR
        while self._fill_times and self._fill_times[0] < cutoff:
            self._fill_times.popleft()

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Deny the order if the rolling hourly fill count is at or above the limit.

        Prunes expired fills before counting, so the count always reflects
        only fills within the last hour.
        """
        current_time = self._clock()
        self._prune_old_fills(current_time)

        fills_in_window = len(self._fill_times)

        if fills_in_window >= self._max_trades:
            # Find when the oldest fill will age out to give a useful message
            oldest = self._fill_times[0] if self._fill_times else current_time
            expires_in = int(_SECONDS_PER_HOUR - (current_time - oldest))
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Global trade rate limit: {fills_in_window}/{self._max_trades} "
                    f"fills in the last hour. Next slot in ~{expires_in}s."
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"Global trade rate OK: {fills_in_window}/{self._max_trades} "
                f"fills in the last hour"
            ),
            risk_level=RiskLevel.LOW,
        )
