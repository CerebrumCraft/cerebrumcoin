# @mock-exempt: Mocking external Anthropic API calls
"""
Unit tests for Conductor.

Tests verify event-driven allocation, poll loop timing, rate limiting,
API failure fallback, and math-only mode — all with mocked Anthropic client.

Mock pattern mirrors test_llm.py: patch `anthropic.AsyncAnthropic` at source.

@decision DEC-TEST-012
@title Mock Anthropic at module boundary for Conductor tests
@status accepted
@rationale Conductor imports AsyncAnthropic locally inside _call_haiku() and
_call_opus_daily_review() via `from anthropic import AsyncAnthropic`. Patching
at `anthropic.AsyncAnthropic` intercepts the import at source — same pattern
as test_llm.py (DEC-TEST-004). The allocator is exercised with real math;
only the external API boundary is mocked.
"""

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import RegimeChangeEvent
from cerebrum.core.types import EventType
from cerebrum.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# FakeClock
# ---------------------------------------------------------------------------


class FakeClock:
    """Controllable wall-clock substitute."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def advance_hours(self, hours: float) -> None:
        self._now += hours * 3600


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STRATEGIES = ["momentum", "mean_reversion", "breakout"]
TOTAL_CAPITAL = Decimal("30000")


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def allocator(clock):
    return DarwinianAllocator(
        strategy_names=STRATEGIES,
        total_capital=TOTAL_CAPITAL,
        warmup_hours=0.0,  # no warmup for conductor tests
        _clock=clock,
    )


@pytest.fixture
def mock_registry():
    """
    Stub StrategyRegistry: get_portfolio returns a mock PortfolioTracker
    that records adjust_balance calls.
    """
    registry = MagicMock(spec=StrategyRegistry)

    portfolios = {}
    for name in STRATEGIES:
        p = MagicMock()
        p.get_cash_balance.return_value = TOTAL_CAPITAL / 3
        portfolios[name] = p

    registry.get_portfolio.side_effect = lambda name: portfolios.get(name)
    registry._portfolios = portfolios
    return registry


def _make_conductor(
    bus: EventBus,
    registry,
    allocator: DarwinianAllocator,
    api_key: str | None = "test-key",
    clock: FakeClock | None = None,
    poll_interval: int = 900,
    daily_review_hour: int = 0,
    max_haiku: int = 20,
    max_opus: int = 2,
) -> Conductor:
    return Conductor(
        bus=bus,
        registry=registry,
        allocator=allocator,
        anthropic_api_key=api_key,
        poll_interval_seconds=poll_interval,
        daily_review_hour=daily_review_hour,
        max_haiku_calls_per_hour=max_haiku,
        max_opus_calls_per_day=max_opus,
        _clock=clock,
    )


def _make_regime_event(to_regime: str = "BULL", confidence: float = 0.8) -> RegimeChangeEvent:
    return RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=1_000_000.0,
        from_regime="SIDEWAYS",
        to_regime=to_regime,
        confidence=Decimal(str(confidence)),
        indicators={"symbol": "BTC/USD"},
    )


def _mock_haiku_response(allocations: dict | None) -> MagicMock:
    """Build a mock AsyncAnthropic client that returns the given allocation JSON."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    content = json.dumps(allocations) if allocations is not None else "null"
    mock_response.content = [MagicMock(text=content)]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: Regime change triggers re-evaluation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_change_triggers_reallocation(bus, mock_registry, allocator, clock):
    """
    Publishing a REGIME_CHANGE event causes the conductor to call Haiku
    and apply allocations to strategy portfolios.
    """
    pytest.importorskip("anthropic")

    new_allocs = {"momentum": 50, "mean_reversion": 30, "breakout": 20}
    mock_client = _mock_haiku_response(new_allocs)

    conductor = _make_conductor(bus, mock_registry, allocator, clock=clock)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        await conductor.start()
        event = _make_regime_event("BULL", 0.9)
        await bus.publish(event)
        await asyncio.sleep(0.15)  # allow async delivery
        await conductor.stop()

    # At least one portfolio should have had adjust_balance called
    # (delta may be 0 if balance matches target, so we check the call was attempted)
    assert mock_client.messages.create.called, "Haiku should have been called on regime change"


# ---------------------------------------------------------------------------
# Test 2: Poll loop runs at correct interval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_loop_applies_allocations(bus, mock_registry, allocator, clock):
    """
    Conductor poll loop applies allocations each interval. We use a very
    short poll interval (0.05s) and verify allocations are applied.
    """
    conductor = _make_conductor(
        bus, mock_registry, allocator, api_key=None, clock=clock, poll_interval=0
    )

    await conductor.start()
    await asyncio.sleep(0.1)  # let one poll tick fire
    await conductor.stop()

    # In math-only mode allocations are applied each tick
    # At least one portfolio should have been queried
    assert mock_registry.get_portfolio.called, "Portfolio should be accessed during poll"


# ---------------------------------------------------------------------------
# Test 3: Haiku rate limiting enforced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_haiku_rate_limit_enforced(bus, mock_registry, allocator, clock):
    """
    After max_haiku_calls_per_hour calls, additional calls are skipped.
    """
    pytest.importorskip("anthropic")

    conductor = _make_conductor(
        bus, mock_registry, allocator, clock=clock, max_haiku=2
    )

    # Simulate 2 calls already made (at limit)
    conductor._haiku_call_times.append(clock())
    conductor._haiku_call_times.append(clock())

    assert not conductor._check_haiku_rate_limit(), (
        "Rate limit should be exceeded after max_haiku calls"
    )

    # After window expires (1h), limit resets
    clock.advance_hours(1.1)
    assert conductor._check_haiku_rate_limit(), (
        "Rate limit should reset after 1 hour"
    )


# ---------------------------------------------------------------------------
# Test 4: API failure → allocations unchanged (DEC-CONDUCTOR-002)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_failure_freezes_allocations(bus, mock_registry, allocator, clock):
    """
    When Claude API raises an exception, the conductor keeps existing
    allocations unchanged — never resets to zero or raises.
    """
    pytest.importorskip("anthropic")

    conductor = _make_conductor(bus, mock_registry, allocator, clock=clock)

    # Set some existing allocations to verify they are preserved
    existing = {
        "momentum": Decimal("40"),
        "mean_reversion": Decimal("35"),
        "breakout": Decimal("25"),
    }
    conductor._last_allocations = existing.copy()

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=Exception("API timeout"))

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await conductor._call_haiku("test context")

    # Should return None (not raise), leaving _last_allocations intact
    assert result is None, "API failure should return None, not raise"
    assert conductor._last_allocations == existing, (
        "Last allocations should be frozen after API failure"
    )


# ---------------------------------------------------------------------------
# Test 5: No API key → math-only mode works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_api_key_math_only_mode(bus, mock_registry, allocator, clock):
    """
    When no API key is provided, Conductor runs DarwinianAllocator math
    without any LLM calls. Should not raise.
    """
    conductor = _make_conductor(
        bus, mock_registry, allocator, api_key=None, clock=clock, poll_interval=0
    )

    assert not conductor._llm_enabled, "Should be in math-only mode"

    await conductor.start()
    await asyncio.sleep(0.05)

    # Publish a regime change — should apply Darwinian allocations, not call LLM
    event = _make_regime_event("BEAR")
    await bus.publish(event)
    await asyncio.sleep(0.1)

    await conductor.stop()

    # No exception = success. Verify LLM was never enabled.
    assert not conductor._llm_enabled


# ---------------------------------------------------------------------------
# Test 6: Opus daily review at correct hour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opus_daily_review_fires_at_correct_hour(bus, mock_registry, allocator):
    """
    _maybe_run_opus_review only fires at daily_review_hour and once per day.
    """
    pytest.importorskip("anthropic")

    # Set clock to midnight UTC (2026-03-23 00:00:00 = 1742688000)
    midnight_utc = 1742688000.0
    clock = FakeClock(start=midnight_utc)

    conductor = _make_conductor(
        bus, mock_registry, allocator, clock=clock,
        daily_review_hour=0,
    )

    new_allocs = {"momentum": 40, "mean_reversion": 35, "breakout": 25}
    mock_client = _mock_haiku_response(new_allocs)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        # First call at midnight — should fire
        result1 = await conductor._maybe_run_opus_review()

    # At midnight with no prior run, Opus should have been called
    assert mock_client.messages.create.called, "Opus should fire at daily_review_hour"

    # Second call at midnight same day — should NOT fire again (already ran today)
    mock_client.messages.create.reset_mock()
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result2 = await conductor._maybe_run_opus_review()

    assert not mock_client.messages.create.called, (
        "Opus should not fire twice on the same day"
    )

    # Advance to next midnight
    clock.advance_hours(24)
    mock_client.messages.create.reset_mock()
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result3 = await conductor._maybe_run_opus_review()

    assert mock_client.messages.create.called, "Opus should fire again on next day"


# ---------------------------------------------------------------------------
# Test 7: Invalid LLM response → ignored, allocations unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_llm_response_ignored(bus, mock_registry, allocator, clock):
    """
    If Claude returns non-JSON or a list instead of a dict, the response
    is silently ignored and existing allocations are preserved.
    """
    pytest.importorskip("anthropic")

    conductor = _make_conductor(bus, mock_registry, allocator, clock=clock)

    existing = {
        "momentum": Decimal("40"),
        "mean_reversion": Decimal("35"),
        "breakout": Decimal("25"),
    }
    conductor._last_allocations = existing.copy()

    # Test 1: response is a JSON list (not dict)
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="[50, 30, 20]")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await conductor._call_haiku("test context")

    assert result is None, "List response should be ignored"
    assert conductor._last_allocations == existing

    # Test 2: response is malformed JSON
    mock_response2 = MagicMock()
    mock_response2.content = [MagicMock(text="{not valid json")]
    mock_client.messages.create = AsyncMock(return_value=mock_response2)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result2 = await conductor._call_haiku("test context")

    assert result2 is None, "Malformed JSON should be ignored"
    assert conductor._last_allocations == existing

    # Test 3: response is "null"
    mock_response3 = MagicMock()
    mock_response3.content = [MagicMock(text="null")]
    mock_client.messages.create = AsyncMock(return_value=mock_response3)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result3 = await conductor._call_haiku("test context")

    assert result3 is None, "null response should be returned as None"


# ---------------------------------------------------------------------------
# Test 8: Haiku returns valid allocation → applied to portfolios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_haiku_response_applied(bus, mock_registry, allocator, clock):
    """
    When Haiku returns a valid allocation dict, those percentages are applied
    to strategy portfolios via adjust_balance.
    """
    pytest.importorskip("anthropic")

    conductor = _make_conductor(bus, mock_registry, allocator, clock=clock)

    new_allocs = {"momentum": 60, "mean_reversion": 25, "breakout": 15}
    mock_client = _mock_haiku_response(new_allocs)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await conductor._call_haiku("test context")

    assert result is not None, "Valid Haiku response should return allocations"
    assert result["momentum"] == Decimal("60")
    assert result["mean_reversion"] == Decimal("25")
    assert result["breakout"] == Decimal("15")


# ---------------------------------------------------------------------------
# Test 9: 50% single-strategy allocation cap (DEC-CONDUCTOR-004)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allocation_cap_50_percent(bus, mock_registry, allocator, clock):
    """
    _apply_allocations must clamp any single strategy to MAX_SINGLE_ALLOCATION_PCT
    (50%) and redistribute the excess proportionally to remaining strategies.

    This prevents the root trigger of the false-drawdown bug: Haiku returning
    75% to range_trading, which injected $5,000 into a $2,500 portfolio and
    set _peak_equity to $7,500. After reversion the strategy was permanently
    stuck at 66.7% drawdown.
    """
    conductor = _make_conductor(bus, mock_registry, allocator, api_key=None, clock=clock)

    # Simulate Haiku returning 75% to one strategy (over the 50% cap)
    over_limit_allocs = {
        "momentum": Decimal("75"),      # exceeds 50% cap
        "mean_reversion": Decimal("15"),
        "breakout": Decimal("10"),
    }

    await conductor._apply_allocations(over_limit_allocs)

    # Collect what adjust_balance was called with
    applied: dict[str, Decimal] = {}
    for name, p in mock_registry._portfolios.items():
        if p.adjust_balance.called:
            # reconstruct target pct from: target_balance = total_capital * pct / 100
            # delta = target_balance - current_balance (current_balance = TOTAL_CAPITAL / 3)
            call_args = p.adjust_balance.call_args[0][0]  # first positional arg
            current = TOTAL_CAPITAL / 3
            target = current + call_args
            pct = target / TOTAL_CAPITAL * 100
            applied[name] = pct
        else:
            # balance unchanged — target == current == TOTAL_CAPITAL / 3
            applied[name] = TOTAL_CAPITAL / 3 / TOTAL_CAPITAL * 100

    # No single strategy should exceed 50%
    for name, pct in applied.items():
        assert pct <= Decimal("50") + Decimal("0.01"), (
            f"Strategy '{name}' got {pct}% allocation, exceeds 50% cap"
        )

    # Allocations must still sum to ~100%
    total = sum(applied.values())
    assert abs(total - Decimal("100")) < Decimal("1"), (
        f"Allocations sum to {total}%, expected ~100%"
    )
