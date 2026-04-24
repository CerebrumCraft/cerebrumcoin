"""
Tests for ProfileManager — Phase 14A hot-swappable risk profiles.

Tests use real component instances (PositionSizingRule, MinSignalStrengthRule,
PostFillCooldownRule, ExitMonitor, SignalAggregator) wired through a minimal
stub StrategyRegistry. No internal mocks — Sacred Practice #5.

The stub registry exposes the same accessor interface as the real StrategyRegistry
so ProfileManager.apply_profile() can walk pipelines without needing a full
event bus or live trading session.

Tests that instantiate real bus-subscribing components (PortfolioTracker,
ExitMonitor, SignalAggregator, PostFillCooldownRule, RiskManager) are marked
@pytest.mark.asyncio because bus.subscribe() calls asyncio.create_task(), which
requires a running event loop. ProfileManager.apply_profile() itself is synchronous,
but the test setup must run inside an event loop. This matches the pattern used
in test_exit_monitor.py, test_strategy_registry.py, and other suites.

@decision DEC-TEST-PROFILE-001
@title ProfileManager tests use real pipeline components, stub registry
@status accepted
@rationale ProfileManager mutates private attributes on real component instances.
Testing with real components (PositionSizingRule, ExitMonitor, SignalAggregator)
directly verifies the attribute names documented in DEC-PROFILE-002 are correct.
A stub StrategyRegistry is used to avoid the async start_all() pipeline lifecycle
while still exercising real component wiring. This is not a mock of internal logic —
it is a minimal factory that returns real component instances under a known name.
Tests are async (not because apply_profile is async, but because the real bus
components require an event loop during construction — see DEC-TEST-002).
"""

from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import ProfileConfig
from cerebrum.profiles.manager import ProfileManager
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    MinSignalStrengthRule,
    PositionSizingRule,
    PostFillCooldownRule,
)
from cerebrum.signals.aggregator import SignalAggregator


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SAMPLE_TOML = {
    "profiles": {
        "conservative": {
            "position_size_percent": "3.0",
            "stop_loss_percent": "0.8",
            "take_profit_percent": "2.0",
            "max_position_age_minutes": 60,
            "post_fill_cooldown_seconds": 3600,
            "min_signal_strength": "0.75",
            "aggregation_threshold": "0.5",
        },
        "moderate": {
            "position_size_percent": "7.0",
            "stop_loss_percent": "1.0",
            "take_profit_percent": "3.0",
            "max_position_age_minutes": 60,
            "post_fill_cooldown_seconds": 1800,
            "min_signal_strength": "0.65",
            "aggregation_threshold": "0.4",
        },
        "aggressive": {
            "position_size_percent": "7.0",
            "stop_loss_percent": "1.5",
            "take_profit_percent": "4.0",
            "max_position_age_minutes": 180,
            "post_fill_cooldown_seconds": 900,
            "min_signal_strength": "0.55",
            "aggregation_threshold": "0.35",
        },
    }
}


# ---------------------------------------------------------------------------
# Stub registry
# ---------------------------------------------------------------------------

class _StubRegistry:
    """
    Minimal stub exposing the StrategyRegistry accessor interface.

    Holds one pre-built pipeline under a fixed strategy name so
    ProfileManager.apply_profile() can walk it without a live trading session.
    All components are real instances — only the registry wrapper is a stub.
    """

    def __init__(
        self,
        strategy_name: str,
        risk_manager: RiskManager | None,
        exit_monitor: ExitMonitor | None,
        aggregator: SignalAggregator | None,
        portfolio: PortfolioTracker | None,
    ) -> None:
        self._name = strategy_name
        self._risk_manager = risk_manager
        self._exit_monitor = exit_monitor
        self._aggregator = aggregator
        self._portfolio = portfolio

    def active_strategy_names(self) -> list[str]:
        return [self._name]

    def get_risk_manager(self, name: str) -> RiskManager | None:
        return self._risk_manager if name == self._name else None

    def get_exit_monitor(self, name: str) -> ExitMonitor | None:
        return self._exit_monitor if name == self._name else None

    def get_aggregator(self, name: str) -> SignalAggregator | None:
        return self._aggregator if name == self._name else None

    def get_portfolio(self, name: str) -> PortfolioTracker | None:
        return self._portfolio if name == self._name else None


class _EmptyRegistry:
    """Registry with no active strategies — for testing the no-pipeline path."""

    def active_strategy_names(self) -> list[str]:
        return []

    def get_risk_manager(self, name: str):
        return None

    def get_exit_monitor(self, name: str):
        return None

    def get_aggregator(self, name: str):
        return None

    def get_portfolio(self, name: str):
        return None


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def _make_bus() -> EventBus:
    return EventBus()


def _make_portfolio(bus: EventBus) -> PortfolioTracker:
    return PortfolioTracker(bus=bus, initial_balance=Decimal("5000"))


def _make_exit_monitor(
    bus: EventBus,
    portfolio: PortfolioTracker,
    stop_loss: str = "1.0",
    take_profit: str = "3.0",
    max_age_minutes: int = 120,
) -> ExitMonitor:
    return ExitMonitor(
        bus=bus,
        portfolio=portfolio,
        stop_loss_percent=Decimal(stop_loss),
        take_profit_percent=Decimal(take_profit),
        max_position_age_minutes=max_age_minutes,
        adaptive_tp=False,
        tp_multiplier=Decimal("1.5"),
        min_tp_percent=Decimal("0.3"),
        strategy_id="test_strategy",
    )


def _make_aggregator(bus: EventBus, threshold: str = "0.4") -> SignalAggregator:
    return SignalAggregator(
        bus=bus,
        threshold=Decimal(threshold),
        window_seconds=5,
        strategy_id="test_strategy",
    )


def _make_risk_manager(
    bus: EventBus,
    portfolio: PortfolioTracker,
    rules: list,
) -> RiskManager:
    return RiskManager(
        bus=bus,
        portfolio=portfolio,
        rules=rules,
        strategy_id="test_strategy",
    )


def _make_full_registry(strategy_name: str = "test_strategy") -> tuple[_StubRegistry, dict]:
    """Build a stub registry backed by real component instances."""
    bus = _make_bus()
    portfolio = _make_portfolio(bus)
    sizing = PositionSizingRule(position_size_percent=Decimal("5.0"))
    strength = MinSignalStrengthRule(min_strength=Decimal("0.65"))
    cooldown = PostFillCooldownRule(cooldown_seconds=1800, bus=bus)
    exit_mon = _make_exit_monitor(bus, portfolio)
    aggregator = _make_aggregator(bus, "0.4")
    risk_mgr = _make_risk_manager(bus, portfolio, [sizing, strength, cooldown])

    registry = _StubRegistry(
        strategy_name=strategy_name,
        risk_manager=risk_mgr,
        exit_monitor=exit_mon,
        aggregator=aggregator,
        portfolio=portfolio,
    )
    components = {
        "sizing": sizing,
        "strength": strength,
        "cooldown": cooldown,
        "exit_monitor": exit_mon,
        "aggregator": aggregator,
    }
    return registry, components


# ---------------------------------------------------------------------------
# Tests: parse_profiles_from_toml
# ---------------------------------------------------------------------------

class TestParseProfilesFromToml:
    """ProfileManager correctly parses [profiles.*] sections from raw TOML."""

    def test_all_three_profiles_loaded(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        assert set(mgr.list_profiles()) == {"conservative", "moderate", "aggressive"}

    def test_profile_values_parsed_as_decimal(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        cfg = mgr.get_profile_config("conservative")
        assert cfg.position_size_percent == Decimal("3.0")
        assert cfg.stop_loss_percent == Decimal("0.8")
        assert cfg.min_signal_strength == Decimal("0.75")

    def test_empty_profiles_section_loads_cleanly(self):
        mgr = ProfileManager(_EmptyRegistry(), {})
        assert mgr.list_profiles() == []

    def test_partial_profile_fields_allowed(self):
        """Profiles with only some fields set are valid — all others stay None."""
        toml = {"profiles": {"slim": {"stop_loss_percent": "0.5"}}}
        mgr = ProfileManager(_EmptyRegistry(), toml)
        cfg = mgr.get_profile_config("slim")
        assert cfg.stop_loss_percent == Decimal("0.5")
        assert cfg.position_size_percent is None
        assert cfg.aggregation_threshold is None


# ---------------------------------------------------------------------------
# Tests: list_profiles
# ---------------------------------------------------------------------------

class TestListProfiles:
    """list_profiles() returns names of all parsed profiles."""

    def test_returns_all_profile_names(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        assert sorted(mgr.list_profiles()) == ["aggressive", "conservative", "moderate"]

    def test_returns_empty_list_when_no_profiles(self):
        mgr = ProfileManager(_EmptyRegistry(), {})
        assert mgr.list_profiles() == []


# ---------------------------------------------------------------------------
# Tests: get_active_profile
# ---------------------------------------------------------------------------

class TestGetActiveProfile:
    """get_active_profile() tracks which profile is current."""

    def test_default_is_empty_string_when_not_set(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        assert mgr.get_active_profile() == ""

    def test_default_profile_reflects_constructor_arg(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML, default_profile="moderate")
        assert mgr.get_active_profile() == "moderate"

    @pytest.mark.asyncio
    async def test_active_profile_updates_after_apply(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("conservative")
        assert mgr.get_active_profile() == "conservative"

    @pytest.mark.asyncio
    async def test_active_profile_updates_on_second_apply(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("conservative")
        mgr.apply_profile("aggressive")
        assert mgr.get_active_profile() == "aggressive"


# ---------------------------------------------------------------------------
# Tests: apply_unknown_profile_raises
# ---------------------------------------------------------------------------

class TestApplyUnknownProfileRaises:
    """apply_profile() raises ValueError for nonexistent profile names."""

    @pytest.mark.asyncio
    async def test_raises_for_unknown_profile(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        with pytest.raises(ValueError, match="Unknown profile 'nonexistent'"):
            mgr.apply_profile("nonexistent")

    def test_get_profile_config_raises_for_unknown(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        with pytest.raises(ValueError, match="Unknown profile 'missing'"):
            mgr.get_profile_config("missing")

    def test_error_message_lists_available_profiles(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        with pytest.raises(ValueError) as exc_info:
            mgr.get_profile_config("typo")
        msg = str(exc_info.value)
        # At least one known profile name appears in error
        assert any(p in msg for p in ["conservative", "moderate", "aggressive"])


# ---------------------------------------------------------------------------
# Tests: apply_profile updates risk params
# ---------------------------------------------------------------------------

class TestApplyProfileUpdatesRiskParams:
    """apply_profile() mutates PositionSizingRule, MinSignalStrengthRule, PostFillCooldownRule."""

    @pytest.mark.asyncio
    async def test_conservative_tightens_position_size(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["sizing"]._size_percent == Decimal("5.0")
        mgr.apply_profile("conservative")
        assert components["sizing"]._size_percent == Decimal("3.0")

    @pytest.mark.asyncio
    async def test_aggressive_increases_position_size(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("aggressive")
        assert components["sizing"]._size_percent == Decimal("7.0")

    @pytest.mark.asyncio
    async def test_conservative_raises_min_signal_strength(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["strength"]._min_strength == Decimal("0.65")
        mgr.apply_profile("conservative")
        assert components["strength"]._min_strength == Decimal("0.75")

    @pytest.mark.asyncio
    async def test_aggressive_lowers_min_signal_strength(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("aggressive")
        assert components["strength"]._min_strength == Decimal("0.55")

    @pytest.mark.asyncio
    async def test_conservative_increases_cooldown(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["cooldown"]._cooldown_seconds == 1800
        mgr.apply_profile("conservative")
        assert components["cooldown"]._cooldown_seconds == 3600

    @pytest.mark.asyncio
    async def test_aggressive_decreases_cooldown(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("aggressive")
        assert components["cooldown"]._cooldown_seconds == 900

    @pytest.mark.asyncio
    async def test_returns_changes_dict_with_risk_keys(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        changes = mgr.apply_profile("moderate")
        assert "test_strategy.position_sizing._size_percent" in changes
        assert "test_strategy.min_signal_strength._min_strength" in changes
        assert "test_strategy.post_fill_cooldown._cooldown_seconds" in changes


# ---------------------------------------------------------------------------
# Tests: apply_profile updates exit params
# ---------------------------------------------------------------------------

class TestApplyProfileUpdatesExitParams:
    """apply_profile() mutates ExitMonitor thresholds."""

    @pytest.mark.asyncio
    async def test_conservative_tightens_stop_loss(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["exit_monitor"]._stop_loss_pct == Decimal("1.0")
        mgr.apply_profile("conservative")
        assert components["exit_monitor"]._stop_loss_pct == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_aggressive_widens_stop_loss(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("aggressive")
        assert components["exit_monitor"]._stop_loss_pct == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_conservative_lowers_take_profit(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["exit_monitor"]._take_profit_pct == Decimal("3.0")
        mgr.apply_profile("conservative")
        assert components["exit_monitor"]._take_profit_pct == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_max_position_age_converted_to_seconds(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        # moderate: max_position_age_minutes=60 → _max_age_seconds=3600 (Phase A sync, DEC-PROFILE-MODERATE-SYNC-001)
        mgr.apply_profile("moderate")
        assert components["exit_monitor"]._max_age_seconds == 60 * 60

    @pytest.mark.asyncio
    async def test_conservative_reduces_max_age_seconds(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("conservative")
        assert components["exit_monitor"]._max_age_seconds == 60 * 60

    @pytest.mark.asyncio
    async def test_exit_monitor_changes_in_returned_dict(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        changes = mgr.apply_profile("conservative")
        assert "test_strategy.exit_monitor._stop_loss_pct" in changes
        assert "test_strategy.exit_monitor._take_profit_pct" in changes
        assert "test_strategy.exit_monitor._max_age_seconds" in changes


# ---------------------------------------------------------------------------
# Tests: apply_profile updates aggregator
# ---------------------------------------------------------------------------

class TestApplyProfileUpdatesAggregator:
    """apply_profile() mutates SignalAggregator threshold."""

    @pytest.mark.asyncio
    async def test_conservative_raises_threshold(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        assert components["aggregator"]._threshold == Decimal("0.4")
        mgr.apply_profile("conservative")
        assert components["aggregator"]._threshold == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_aggressive_lowers_threshold(self):
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        mgr.apply_profile("aggressive")
        assert components["aggregator"]._threshold == Decimal("0.35")

    @pytest.mark.asyncio
    async def test_aggregator_change_in_returned_dict(self):
        registry, _ = _make_full_registry()
        mgr = ProfileManager(registry, SAMPLE_TOML)
        changes = mgr.apply_profile("aggressive")
        assert "test_strategy.aggregator._threshold" in changes


# ---------------------------------------------------------------------------
# Tests: no active pipelines
# ---------------------------------------------------------------------------

class TestApplyProfileNoActivePipelines:
    """apply_profile() handles an empty registry gracefully."""

    def test_returns_empty_dict_when_no_pipelines(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        changes = mgr.apply_profile("conservative")
        assert changes == {}

    def test_still_updates_active_profile_when_no_pipelines(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        mgr.apply_profile("conservative")
        assert mgr.get_active_profile() == "conservative"

    def test_raises_before_empty_check_for_unknown_profile(self):
        mgr = ProfileManager(_EmptyRegistry(), SAMPLE_TOML)
        with pytest.raises(ValueError):
            mgr.apply_profile("does_not_exist")


# ---------------------------------------------------------------------------
# Tests: partial profile application
# ---------------------------------------------------------------------------

class TestPartialProfileApplication:
    """Profiles with only some fields set only override those specific fields."""

    @pytest.mark.asyncio
    async def test_partial_profile_only_touches_specified_fields(self):
        """A profile with only stop_loss_percent should not touch position size."""
        toml = {"profiles": {"sl_only": {"stop_loss_percent": "0.5"}}}
        registry, components = _make_full_registry()
        mgr = ProfileManager(registry, toml)

        original_size = components["sizing"]._size_percent
        original_strength = components["strength"]._min_strength
        original_threshold = components["aggregator"]._threshold

        mgr.apply_profile("sl_only")

        assert components["exit_monitor"]._stop_loss_pct == Decimal("0.5")
        assert components["sizing"]._size_percent == original_size
        assert components["strength"]._min_strength == original_strength
        assert components["aggregator"]._threshold == original_threshold
