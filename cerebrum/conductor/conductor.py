"""
LLM Conductor: event-driven capital allocation with Claude reasoning.

Combines Darwinian (math-only) allocation with optional LLM overrides.
Triggers on REGIME_CHANGE and NEWS events; also polls every 15 minutes.
Daily Opus review provides deeper strategic analysis.

@decision DEC-CONDUCTOR-001
@title Event-driven + polling hybrid LLM conductor
@status accepted
@rationale Pure polling misses immediate regime changes. Pure event-driven
misses slow degradation between events. Hybrid: fast path (REGIME_CHANGE →
Haiku in seconds), slow path (15-min poll → Haiku if conditions warrant),
deep path (midnight Opus review for full-day retrospective). Haiku for speed
and cost; Opus once daily for depth.

@decision DEC-CONDUCTOR-002
@title Freeze allocations on API failure, never reset
@status accepted
@rationale A trading system must degrade gracefully. If the Claude API is
unavailable (timeout, 5xx, rate limit), the correct action is to keep the
last known-good allocations unchanged. Resetting to equal allocations on
failure would discard learned performance data and could reallocate capital
into a paused underperformer. Logging at WARNING level gives operators
visibility without crashing the bot.

@decision DEC-CONDUCTOR-003
@title Math-only mode when no API key provided
@status accepted
@rationale DarwinianAllocator alone is genuinely useful — it adjusts capital
based on rolling Sharpe without any LLM cost. Operators who don't want LLM
overhead (or haven't configured an API key) get full Darwinian allocation.
The Conductor detects missing key at startup and skips all LLM code paths.
"""

import asyncio
import json
from collections import deque
from decimal import Decimal
from time import time
from typing import Callable, Deque

import structlog

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, RegimeChangeEvent
from cerebrum.core.types import EventType
from cerebrum.strategies.registry import StrategyRegistry

logger = structlog.get_logger()

_DEFAULT_CLOCK: Callable[[], float] = time


class Conductor:
    """
    Orchestrates capital allocation across strategies using LLM reasoning.

    In math-only mode (no API key): runs DarwinianAllocator on a poll loop
    and applies allocations every poll_interval_seconds.

    In LLM mode: additionally calls Claude Haiku on REGIME_CHANGE, high-
    relevance NEWS events, and periodically during the poll loop. Claude Opus
    runs once per day for a deeper review.

    LLM responses are treated as advisory overrides: if Claude returns valid
    JSON with allocation percentages, those override the Darwinian math. If
    the call fails or returns invalid JSON, the math-only allocations are used.

    Usage::

        conductor = Conductor(bus, registry, allocator, api_key)
        await conductor.start()
        # ... runs until stop() is called ...
        await conductor.stop()
    """

    def __init__(
        self,
        bus: EventBus,
        registry: StrategyRegistry,
        allocator: DarwinianAllocator,
        anthropic_api_key: str | None = None,
        haiku_model: str = "claude-haiku-4-5-20251001",
        opus_model: str = "claude-opus-4-6-20250610",
        poll_interval_seconds: int = 900,
        daily_review_hour: int = 0,
        max_haiku_calls_per_hour: int = 20,
        max_opus_calls_per_day: int = 2,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialise the Conductor.

        Args:
            bus: Shared event bus.
            registry: StrategyRegistry for accessing per-strategy portfolios.
            allocator: DarwinianAllocator to drive capital decisions.
            anthropic_api_key: Anthropic API key. None → math-only mode.
            haiku_model: Model ID for fast allocation calls.
            opus_model: Model ID for daily deep-review calls.
            poll_interval_seconds: Seconds between poll-loop iterations.
            daily_review_hour: UTC hour (0–23) to run Opus daily review.
            max_haiku_calls_per_hour: Rate cap for Haiku calls.
            max_opus_calls_per_day: Rate cap for Opus calls.
            _clock: Injectable time source for testing (defaults to time.time).
        """
        self._bus = bus
        self._registry = registry
        self._allocator = allocator
        self._api_key = anthropic_api_key
        self._haiku_model = haiku_model
        self._opus_model = opus_model
        self._poll_interval = poll_interval_seconds
        self._daily_review_hour = daily_review_hour
        self._max_haiku_per_hour = max_haiku_calls_per_hour
        self._max_opus_per_day = max_opus_calls_per_day
        self._clock = _clock or _DEFAULT_CLOCK

        # Rate limiting — deque of call timestamps
        self._haiku_call_times: Deque[float] = deque(maxlen=max_haiku_calls_per_hour)
        self._opus_call_times: Deque[float] = deque(maxlen=max_opus_calls_per_day)

        # Last known allocations (for fallback on API failure)
        self._last_allocations: dict[str, Decimal] = {}

        # Latest regime seen (for Haiku context)
        self._latest_regime: str = "UNKNOWN"
        self._latest_regime_confidence: Decimal = Decimal("0")

        # Track last Opus review date (UTC date string "YYYY-MM-DD")
        self._last_opus_date: str = ""

        # Copilot mode: when True, allocation changes are queued as pending
        # rather than applied immediately. The dashboard exposes approve/reject
        # endpoints. A newer proposal overwrites an unresolved one (DEC-DASH-003).
        self.copilot_mode: bool = False
        self._pending_allocation: dict[str, Decimal] | None = None
        self._pending_reasoning: str | None = None

        self._running = False
        self._poll_task: asyncio.Task | None = None

        self._llm_enabled = bool(anthropic_api_key)

        self._log = logger.bind(component="conductor")

        if not self._llm_enabled:
            self._log.warning(
                "conductor_math_only_mode",
                message="No Anthropic API key — running Darwinian allocation only.",
            )
        else:
            self._log.info(
                "conductor_initialized",
                haiku_model=haiku_model,
                opus_model=opus_model,
                poll_interval_seconds=poll_interval_seconds,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to events and start the polling loop."""
        if self._running:
            return

        self._running = True

        self._bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            subscriber_name="conductor_regime",
        )
        self._bus.subscribe(
            EventType.NEWS,
            self._on_news,
            subscriber_name="conductor_news",
        )

        self._poll_task = asyncio.create_task(self._poll_loop())
        self._log.info("conductor_started")

    async def stop(self) -> None:
        """Cancel the polling loop and unsubscribe from events."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
            self._poll_task = None

        self._log.info("conductor_stopped")

    # ------------------------------------------------------------------
    # Copilot mode — human-in-the-loop approval
    # ------------------------------------------------------------------

    async def approve_pending(self) -> None:
        """
        Apply the pending allocation that was queued in copilot mode.

        No-op if there is no pending allocation.
        """
        if self._pending_allocation is not None:
            self._log.info(
                "copilot_approved",
                allocations={k: str(v) for k, v in self._pending_allocation.items()},
            )
            await self._apply_allocations(self._pending_allocation, _bypass_copilot=True)
            self._pending_allocation = None
            self._pending_reasoning = None

    async def reject_pending(self) -> None:
        """
        Discard the pending allocation. The last applied allocation remains in force.
        """
        if self._pending_allocation is not None:
            self._log.info(
                "copilot_rejected",
                allocations={k: str(v) for k, v in self._pending_allocation.items()},
            )
            self._pending_allocation = None
            self._pending_reasoning = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_regime_change(self, event: Event) -> None:
        """REGIME_CHANGE triggers a Haiku re-evaluation."""
        if not isinstance(event, RegimeChangeEvent):
            return

        self._latest_regime = event.to_regime
        self._latest_regime_confidence = event.confidence

        self._log.info(
            "regime_change_received",
            from_regime=event.from_regime,
            to_regime=event.to_regime,
            confidence=str(event.confidence),
        )

        if not self._llm_enabled:
            # Math-only: apply Darwinian allocations directly
            allocations = self._allocator.get_allocations()
            await self._apply_allocations(allocations)
            return

        context = self._build_haiku_context(trigger="regime_change")
        overrides = await self._call_haiku(context)
        allocations = overrides if overrides else self._allocator.get_allocations()
        await self._apply_allocations(allocations)

    async def _on_news(self, event: Event) -> None:
        """High-relevance news triggers a Haiku assessment."""
        if not self._llm_enabled:
            return

        context = self._build_haiku_context(trigger="news_event")
        overrides = await self._call_haiku(context)
        if overrides:
            await self._apply_allocations(overrides)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Poll every poll_interval_seconds.

        On each tick:
        1. Compute Darwinian allocations (math).
        2. Optionally call Haiku for LLM override.
        3. Check if it's time for daily Opus review.
        4. Apply final allocations.
        """
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                # Base allocations from math
                allocations = self._allocator.get_allocations()

                if self._llm_enabled:
                    # Haiku override on every poll tick (within rate limit)
                    context = self._build_haiku_context(trigger="poll")
                    overrides = await self._call_haiku(context)
                    if overrides:
                        allocations = overrides

                    # Daily Opus review
                    opus_result = await self._maybe_run_opus_review()
                    if opus_result:
                        allocations = opus_result

                await self._apply_allocations(allocations)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("poll_loop_error", error=str(exc))

    # ------------------------------------------------------------------
    # Allocation application
    # ------------------------------------------------------------------

    async def _apply_allocations(
        self,
        allocations: dict[str, Decimal],
        _bypass_copilot: bool = False,
    ) -> None:
        """
        Apply allocation percentages to strategy portfolios.

        When copilot_mode is True (and _bypass_copilot is False), the allocation
        is queued as pending instead of applied immediately. The human must call
        approve_pending() via the dashboard to commit the change (DEC-DASH-003).

        Calls registry.get_portfolio(name).adjust_balance() to redistribute
        capital. Saves allocations as fallback for API failure recovery.

        Args:
            allocations: Dict of strategy_name → allocation_pct (0–100).
            _bypass_copilot: Internal flag — set True by approve_pending() so
                             the approval path actually applies the allocation.
        """
        if self.copilot_mode and not _bypass_copilot:
            # Queue for human approval — overwrite any previous pending proposal
            self._pending_allocation = dict(allocations)
            self._pending_reasoning = (
                f"Triggered at regime={self._latest_regime} "
                f"confidence={self._latest_regime_confidence}"
            )
            self._log.info(
                "copilot_allocation_queued",
                allocations={k: str(v) for k, v in allocations.items()},
            )
            return

        self._last_allocations = dict(allocations)
        total_capital = self._allocator._total_capital

        for name, pct in allocations.items():
            portfolio = self._registry.get_portfolio(name)
            if portfolio is None:
                self._log.warning("portfolio_not_found", strategy=name)
                continue

            target_balance = total_capital * pct / Decimal("100")
            current_balance = portfolio.get_cash_balance()
            delta = target_balance - current_balance

            if abs(delta) > Decimal("0.01"):  # ignore sub-cent adjustments
                portfolio.adjust_balance(delta)
                self._log.info(
                    "allocation_applied",
                    strategy=name,
                    pct=str(round(pct, 2)),
                    target_balance=str(round(target_balance, 2)),
                    delta=str(round(delta, 2)),
                )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _build_haiku_context(self, trigger: str) -> str:
        """Build the prompt context string for a Haiku allocation call."""
        strategy_lines = []
        for name in self._allocator._strategies:
            sharpe = self._allocator._sharpe.get(name)
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "N/A (no data)"
            paused = self._allocator.is_paused(name)
            portfolio = self._registry.get_portfolio(name)
            balance = str(round(portfolio.get_cash_balance(), 2)) if portfolio else "N/A"
            current_pct = str(round(self._last_allocations.get(name, Decimal("0")), 1))
            strategy_lines.append(
                f"  - {name}: sharpe={sharpe_str}, paused={paused}, "
                f"balance=${balance}, allocation={current_pct}%"
            )

        strategies_block = "\n".join(strategy_lines)
        regime_block = (
            f"Current regime: {self._latest_regime} "
            f"(confidence: {self._latest_regime_confidence})"
        )

        return (
            f"Trigger: {trigger}\n"
            f"{regime_block}\n\n"
            f"Strategy performance:\n{strategies_block}\n\n"
            f"Total capital: ${self._allocator._total_capital}"
        )

    async def _call_haiku(self, context: str) -> dict[str, Decimal] | None:
        """
        Call Claude Haiku for a fast allocation decision.

        Returns allocation override dict (strategy → pct) or None if:
        - Rate limit exceeded
        - API call fails (DEC-CONDUCTOR-002)
        - Response is not valid JSON
        - Response is null (model indicates no change needed)

        Args:
            context: Prompt context describing current state.

        Returns:
            Dict of allocation overrides or None.
        """
        if not self._check_haiku_rate_limit():
            self._log.debug("haiku_rate_limit_exceeded")
            return None

        strategy_names = self._allocator._strategies
        prompt = (
            f"{context}\n\n"
            "Given the current market conditions and strategy performance, "
            "should we adjust strategy capital allocations?\n\n"
            f"Strategies: {strategy_names}\n\n"
            "Return JSON: {\"strategy_name\": allocation_pct, ...} where percentages sum to 100. "
            "If no changes are needed, return null.\n\n"
            "Important: Only return the JSON object or null — no explanation."
        )

        try:
            from anthropic import AsyncAnthropic  # noqa: PLC0415

            client = AsyncAnthropic(api_key=self._api_key, timeout=30)
            response = await client.messages.create(
                model=self._haiku_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            self._haiku_call_times.append(self._clock())

            content = response.content[0].text.strip()

            # Strip markdown code fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            if content.lower() in ("null", "none", ""):
                self._log.debug("haiku_no_change_needed")
                return None

            raw = json.loads(content)
            if not isinstance(raw, dict):
                self._log.warning("haiku_invalid_response_type", type=type(raw).__name__)
                return None

            # Validate and convert — only accept known strategy names
            overrides: dict[str, Decimal] = {}
            for name, pct in raw.items():
                if name not in strategy_names:
                    self._log.warning("haiku_unknown_strategy", name=name)
                    continue
                overrides[name] = Decimal(str(pct))

            if not overrides:
                return None

            self._log.info(
                "haiku_override_applied",
                allocations={k: str(v) for k, v in overrides.items()},
            )
            return overrides

        except ImportError:
            self._log.error("anthropic_sdk_not_installed")
            self._llm_enabled = False
            return None
        except Exception as exc:
            # DEC-CONDUCTOR-002: freeze on API failure
            self._log.warning("haiku_call_failed", error=str(exc))
            return None

    async def _call_opus_daily_review(self) -> dict[str, Decimal] | None:
        """
        Call Claude Opus for deep daily strategy review.

        Provides full per-strategy performance summary and asks for
        allocation recommendations and strategic observations.

        Returns allocation override dict or None on failure.
        """
        if not self._check_opus_rate_limit():
            self._log.debug("opus_rate_limit_exceeded")
            return None

        strategy_lines = []
        for name in self._allocator._strategies:
            sharpe = self._allocator._sharpe.get(name)
            sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
            paused = self._allocator.is_paused(name)
            portfolio = self._registry.get_portfolio(name)
            balance = str(round(portfolio.get_cash_balance(), 2)) if portfolio else "N/A"
            trade_count = len(self._allocator._trade_history.get(name, []))
            strategy_lines.append(
                f"  Strategy: {name}\n"
                f"    Rolling Sharpe: {sharpe_str}\n"
                f"    Currently paused: {paused}\n"
                f"    Cash balance: ${balance}\n"
                f"    Trades in window: {trade_count}"
            )

        strategies_block = "\n".join(strategy_lines)
        prompt = (
            f"Daily strategy review for CerebrumCoin autonomous trading bot.\n\n"
            f"Current market regime: {self._latest_regime} "
            f"(confidence: {self._latest_regime_confidence})\n\n"
            f"Strategy performance over last 24h:\n{strategies_block}\n\n"
            f"Total capital: ${self._allocator._total_capital}\n\n"
            "Please provide:\n"
            "1. A brief assessment of each strategy's performance\n"
            "2. Capital allocation recommendations as JSON: "
            "{\"strategy_name\": allocation_pct, ...} summing to 100\n"
            "3. Any strategic observations for tomorrow\n\n"
            "Return your allocation recommendation as a JSON block: "
            "```json\n{...}\n```"
        )

        try:
            from anthropic import AsyncAnthropic  # noqa: PLC0415

            client = AsyncAnthropic(api_key=self._api_key, timeout=60)
            response = await client.messages.create(
                model=self._opus_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            self._opus_call_times.append(self._clock())

            content = response.content[0].text

            # Extract JSON block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                # Try to find bare JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                if start == -1 or end == 0:
                    self._log.warning("opus_no_json_in_response")
                    return None
                json_str = content[start:end]

            raw = json.loads(json_str)
            if not isinstance(raw, dict):
                return None

            strategy_names = self._allocator._strategies
            overrides: dict[str, Decimal] = {}
            for name, pct in raw.items():
                if name not in strategy_names:
                    continue
                overrides[name] = Decimal(str(pct))

            self._log.info(
                "opus_daily_review_applied",
                allocations={k: str(v) for k, v in overrides.items()},
            )
            return overrides or None

        except ImportError:
            self._log.error("anthropic_sdk_not_installed")
            self._llm_enabled = False
            return None
        except Exception as exc:
            self._log.warning("opus_call_failed", error=str(exc))
            return None

    async def _maybe_run_opus_review(self) -> dict[str, Decimal] | None:
        """
        Run Opus daily review if it's the right hour and hasn't run today.

        Returns allocation overrides or None.
        """
        import datetime  # noqa: PLC0415

        now_utc = datetime.datetime.fromtimestamp(self._clock(), tz=datetime.timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")

        if (
            now_utc.hour == self._daily_review_hour
            and self._last_opus_date != today_str
        ):
            self._last_opus_date = today_str
            return await self._call_opus_daily_review()

        return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_haiku_rate_limit(self) -> bool:
        """Return True if a Haiku call is within rate limit."""
        return self._check_rate_limit(self._haiku_call_times, 3600, self._max_haiku_per_hour)

    def _check_opus_rate_limit(self) -> bool:
        """Return True if an Opus call is within rate limit."""
        return self._check_rate_limit(self._opus_call_times, 86400, self._max_opus_per_day)

    def _check_rate_limit(
        self,
        call_times: Deque[float],
        window_seconds: float,
        max_calls: int,
    ) -> bool:
        """
        Generic rate-limit check using deque of call timestamps.

        Prunes timestamps outside the window before checking count.
        Mirrors the pattern in cerebrum/intelligence/llm.py.
        """
        now = self._clock()
        cutoff = now - window_seconds
        while call_times and call_times[0] < cutoff:
            call_times.popleft()
        return len(call_times) < max_calls
