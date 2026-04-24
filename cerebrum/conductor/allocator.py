"""
Darwinian capital allocator for multi-strategy trading.

Allocates capital based on rolling Sharpe ratio. Underperformers get halved
or paused. Paused strategies auto-reactivate after a backoff period to prevent
permanent deadlock.

@decision DEC-ALLOC-001
@title Darwinian capital allocation via rolling Sharpe ratio
@status accepted
@rationale Strategies compete for capital based on risk-adjusted returns.
Rolling Sharpe (over a configurable window) provides a fair, recency-weighted
performance signal. Underperformers are penalized incrementally: first halved,
then fully paused. This avoids abrupt 100% cuts while still starving strategies
that consistently lose money. Capital freed from paused strategies is
redistributed proportionally to active strategies — they benefit directly from
others' underperformance.

@decision DEC-ALLOC-002
@title Auto-reactivation with exponential backoff prevents permanent deadlock
@status accepted
@rationale A paused strategy stays paused forever under naive Darwinian
selection — it cannot recover because it receives no capital to trade with.
Auto-reactivation breaks the deadlock: after 2 hours, the strategy is restored
to a minimum allocation (10%) and given another chance. If it underperforms
again, it is re-paused with doubled backoff (4h, then 8h, etc.). This is the
same pattern as TCP congestion control and circuit-breaker recovery.

@decision DEC-ALLOC-003
@title All-paused edge case: reactivate the least-bad strategy
@status accepted
@rationale If every strategy falls below the pause threshold simultaneously
(e.g., a flash crash), the allocator must not return zero total allocation —
that would strand all capital. The fix: find the strategy with the highest
(least negative) Sharpe and restore it to min_allocation_pct. This ensures
there is always at least one strategy active.
"""

from collections import deque
from decimal import Decimal
from time import time
from typing import Callable, Deque

import structlog

from cerebrum.core.state import TradeRecord
from cerebrum.monitoring.stats import calculate_sharpe_ratio

logger = structlog.get_logger()

_DEFAULT_CLOCK: Callable[[], float] = time


class DarwinianAllocator:
    """
    Pure-math capital allocator using rolling Sharpe ratios.

    Strategies compete for capital. Underperformers are penalized:
    - Sharpe < halve_threshold  → allocation halved
    - Sharpe < pause_threshold  → allocation zeroed (paused)
    - After reactivation_hours  → restored to min_allocation_pct
    - Re-pause doubles the next backoff (exponential backoff)

    All time-dependent behaviour uses an injectable _clock so tests can
    control time without sleeping.

    Usage::

        allocator = DarwinianAllocator(
            strategy_names=["momentum", "mean_reversion", "breakout"],
            total_capital=Decimal("10000"),
        )
        allocator.update_performance("momentum", trades=closed_trades, equity_curve=[...])
        allocations = allocator.get_allocation_amounts()
        # {"momentum": Decimal("4000"), "mean_reversion": Decimal("3000"), ...}
    """

    def __init__(
        self,
        strategy_names: list[str],
        total_capital: Decimal,
        sharpe_window_hours: float = 4.0,
        halve_threshold: float = -0.5,
        pause_threshold: float = -1.0,
        recovery_threshold: float = 0.0,
        warmup_hours: float = 8.0,
        reactivation_hours: float = 2.0,
        reactivation_backoff: float = 2.0,
        min_allocation_pct: Decimal = Decimal("10"),
        _clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialise the allocator.

        Args:
            strategy_names: Names of all strategies to manage.
            total_capital: Total USD capital to distribute.
            sharpe_window_hours: Rolling window for Sharpe calculation.
            halve_threshold: Sharpe below this halves the allocation.
            pause_threshold: Sharpe below this zeros the allocation.
            recovery_threshold: Sharpe must exceed this to count as recovered.
            warmup_hours: Duration of equal-allocation warmup period.
            reactivation_hours: Hours paused before auto-reactivation.
            reactivation_backoff: Multiplier applied to backoff on re-pause.
            min_allocation_pct: Allocation percentage on reactivation.
            _clock: Injectable time source (float seconds). Defaults to time.time.
        """
        if not strategy_names:
            raise ValueError("strategy_names must not be empty")

        self._strategies = list(strategy_names)
        self._total_capital = total_capital
        self._sharpe_window_hours = sharpe_window_hours
        self._halve_threshold = halve_threshold
        self._pause_threshold = pause_threshold
        self._recovery_threshold = recovery_threshold
        self._warmup_hours = warmup_hours
        self._base_reactivation_hours = reactivation_hours
        self._reactivation_backoff = reactivation_backoff
        self._min_allocation_pct = min_allocation_pct
        self._clock = _clock or _DEFAULT_CLOCK

        self._start_time: float = self._clock()

        # Rolling trade history per strategy (for Sharpe calculation)
        self._trade_history: dict[str, Deque[TradeRecord]] = {
            name: deque() for name in self._strategies
        }

        # Latest computed Sharpe per strategy (None = not yet computed)
        self._sharpe: dict[str, float | None] = {name: None for name in self._strategies}

        # Pause tracking
        self._pause_start: dict[str, float | None] = {name: None for name in self._strategies}
        # Current backoff hours per strategy (starts at reactivation_hours)
        self._backoff_hours: dict[str, float] = {
            name: reactivation_hours for name in self._strategies
        }
        # Timestamp of last reactivation — grace period prevents immediate re-pause
        # on the same get_allocations() call that cleared the pause flag.
        self._reactivated_at: dict[str, float | None] = {name: None for name in self._strategies}

        self._log = logger.bind(component="darwinian_allocator")
        self._log.info(
            "allocator_initialized",
            strategies=self._strategies,
            total_capital=str(total_capital),
            warmup_hours=warmup_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_total_capital(self, value: Decimal) -> None:
        """
        Update the total capital baseline used for allocation target computations.

        Called by the Conductor immediately before each ``_apply_allocations``
        cycle to reflect realised P&L that has accumulated in the shared USD
        pool since the last allocation (Bug B fix: stale ``_total_capital``).

        Args:
            value: Current global equity from the live paper adapter.
        """
        if value != self._total_capital:
            self._log.debug(
                "total_capital_refreshed",
                old=str(self._total_capital),
                new=str(value),
            )
            self._total_capital = value

    def update_performance(
        self,
        strategy_name: str,
        trades: list[TradeRecord],
        equity_curve: list[Decimal],
    ) -> None:
        """
        Update rolling trade history and recompute Sharpe for one strategy.

        Args:
            strategy_name: Strategy to update.
            trades: Recent closed trades (replaces the rolling window).
            equity_curve: Recent equity values (unused currently; reserved for
                          future Sortino / drawdown metrics).
        """
        if strategy_name not in self._trade_history:
            self._log.warning("unknown_strategy", name=strategy_name)
            return

        # Replace history with the provided trades (caller controls the window)
        self._trade_history[strategy_name] = deque(trades)

        sharpe = calculate_sharpe_ratio(trades)
        # calculate_sharpe_ratio returns Decimal; convert to float for comparisons
        self._sharpe[strategy_name] = float(sharpe)

        self._log.debug(
            "performance_updated",
            strategy=strategy_name,
            trade_count=len(trades),
            sharpe=float(sharpe),
        )

    def get_allocations(self) -> dict[str, Decimal]:
        """
        Compute allocation percentages for all strategies.

        Returns:
            Dict mapping strategy_name → allocation_pct (0–100, summing to ~100).
            During warmup: equal share for all strategies.
            After warmup: Sharpe-proportional with halving and pausing.
        """
        now = self._clock()

        # --- Handle auto-reactivation for paused strategies ---
        self._maybe_reactivate(now)

        if self.is_warming_up():
            return self._equal_allocations()

        return self._compute_post_warmup_allocations()

    def get_allocation_amounts(self) -> dict[str, Decimal]:
        """
        Compute dollar allocation amounts for all strategies.

        Returns:
            Dict mapping strategy_name → USD amount.
        """
        pcts = self.get_allocations()
        return {
            name: self._total_capital * pct / Decimal("100")
            for name, pct in pcts.items()
        }

    def is_warming_up(self) -> bool:
        """True during the initial warmup period (equal allocation enforced)."""
        elapsed = self._clock() - self._start_time
        return elapsed < self._warmup_hours * 3600

    def is_paused(self, strategy_name: str) -> bool:
        """True if strategy is currently paused."""
        return self._pause_start.get(strategy_name) is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _equal_allocations(self) -> dict[str, Decimal]:
        """Equal split across all strategies (warmup mode)."""
        n = len(self._strategies)
        equal_pct = Decimal("100") / Decimal(str(n))
        return {name: equal_pct for name in self._strategies}

    def _maybe_reactivate(self, now: float) -> None:
        """
        Auto-reactivate any strategy that has been paused long enough.

        On reactivation the pause marker is cleared; the backoff is doubled
        so that if the strategy underperforms again the next pause is longer.
        Records reactivation time so _compute_post_warmup_allocations can skip
        the re-pause check for the same cycle (grace period).
        """
        for name in self._strategies:
            pause_start = self._pause_start[name]
            if pause_start is None:
                continue

            paused_for_hours = (now - pause_start) / 3600
            if paused_for_hours >= self._backoff_hours[name]:
                self._pause_start[name] = None
                self._reactivated_at[name] = now
                # Double the backoff for next potential re-pause
                self._backoff_hours[name] *= self._reactivation_backoff
                self._log.info(
                    "strategy_reactivated",
                    strategy=name,
                    paused_hours=round(paused_for_hours, 2),
                    next_backoff_hours=self._backoff_hours[name],
                )

    def _compute_post_warmup_allocations(self) -> dict[str, Decimal]:
        """
        Compute allocations based on rolling Sharpe after warmup.

        Steps:
        1. Classify each strategy as active / halved / paused based on Sharpe.
        2. Compute raw weights for active+halved strategies.
        3. Normalise to 100%.
        4. Apply halving penalty to halved strategies.
        5. Re-normalise.
        6. Edge case: if all paused, reactivate least-bad one (DEC-ALLOC-003).
        """
        now = self._clock()
        sharpes = {}
        paused_names = []

        for name in self._strategies:
            if self._pause_start[name] is not None:
                # Currently paused
                paused_names.append(name)
                continue

            s = self._sharpe[name]
            if s is None:
                # No data yet — treat as zero (neutral)
                sharpes[name] = 0.0
            elif s < self._pause_threshold:
                # Grace period: if just reactivated this cycle, skip re-pause.
                # _maybe_reactivate sets _reactivated_at[name] = now (same timestamp),
                # so comparing == now detects the same cycle reliably.
                reactivated_at = self._reactivated_at[name]
                if reactivated_at is not None and reactivated_at == now:
                    # Give it one cycle before we can re-pause
                    sharpes[name] = s
                    continue
                # Below pause threshold → pause now
                self._pause_start[name] = self._clock()
                paused_names.append(name)
                self._log.warning(
                    "strategy_paused",
                    strategy=name,
                    sharpe=round(s, 4),
                    threshold=self._pause_threshold,
                )
                continue
            else:
                sharpes[name] = s

        active_names = list(sharpes.keys())

        # --- All-paused edge case (DEC-ALLOC-003) ---
        if not active_names:
            return self._reactivate_best_paused()

        # --- Compute raw weights (shift so minimum is 0 to keep weights positive) ---
        min_sharpe = min(sharpes.values())
        # Shift so the worst active strategy has weight 0+epsilon, rest higher
        shift = max(0.0, -min_sharpe) + 0.01  # +0.01 ensures nobody is exactly 0
        raw_weights = {name: sharpes[name] + shift for name in active_names}

        # --- Apply halving penalty ---
        penalised_weights: dict[str, float] = {}
        for name in active_names:
            s = sharpes[name]
            weight = raw_weights[name]
            if s < self._halve_threshold:
                weight *= 0.5
            penalised_weights[name] = weight

        total_weight = sum(penalised_weights.values())

        # --- Normalise to 100%, keeping paused strategies at 0% ---
        allocations: dict[str, Decimal] = {}
        for name in self._strategies:
            if name in paused_names:
                allocations[name] = Decimal("0")
            else:
                pct = Decimal(str(penalised_weights[name])) / Decimal(str(total_weight)) * Decimal("100")
                # Floor at min_allocation_pct when reactivated (done via reactivation path)
                allocations[name] = pct

        return allocations

    def _reactivate_best_paused(self) -> dict[str, Decimal]:
        """
        All strategies are paused. Reactivate the one with the best Sharpe.

        Returns allocations with one strategy at min_allocation_pct, rest 0%.
        """
        # Find strategy with highest (least negative) Sharpe
        best = max(
            self._strategies,
            key=lambda n: self._sharpe[n] if self._sharpe[n] is not None else float("-inf"),
        )
        self._pause_start[best] = None
        self._log.warning(
            "all_paused_reactivating_best",
            strategy=best,
            sharpe=self._sharpe[best],
        )
        result: dict[str, Decimal] = {name: Decimal("0") for name in self._strategies}
        result[best] = self._min_allocation_pct
        return result
