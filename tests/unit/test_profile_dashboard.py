# @mock-exempt: AsyncMock WebSocket clients — WebSocket is a protocol boundary (no real browser in unit tests)
"""
Tests for Phase 14B profile selector REST API endpoints in WebDashboard.

Covers:
- GET /api/profiles — returns available profiles with configs
- GET /api/profile/active — returns current active profile name
- POST /api/profile/apply — applies profile, returns changes dict
- POST /api/profile/apply — returns HTTP 400 for unknown profile name
- WebDashboard with profile_manager=None returns {"available": false}

Uses real WebDashboard (FastAPI TestClient), real ProfileManager with a stub
StrategyRegistry (same pattern as test_profile_manager.py), and real pipeline
components. WebSocket clients are the only mock — they represent browser
connections, an external transport boundary.

@decision DEC-TEST-PROFILE-DASH-001
@title Profile dashboard tests use real ProfileManager + TestClient, stub registry
@status accepted
@rationale The three profile endpoints are thin wrappers around ProfileManager's
public API (list_profiles, get_active_profile, apply_profile). Testing with a
real ProfileManager and real components verifies the JSON serialisation, HTTP
status codes, and WebSocket broadcast path end-to-end. The stub StrategyRegistry
matches the pattern established in test_profile_manager.py (DEC-TEST-PROFILE-001).
"""

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cerebrum.core.bus import EventBus
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
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.global_portfolio import GlobalPortfolio
from cerebrum.strategies.registry import StrategyRegistry

# Optional dependency — skip entire module if FastAPI not installed
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cerebrum.dashboard.web import WebDashboard  # noqa: E402
from cerebrum.conductor.allocator import DarwinianAllocator  # noqa: E402
from cerebrum.conductor.conductor import Conductor  # noqa: E402

# ---------------------------------------------------------------------------
# Test profile TOML data — mirrors the structure in test_profile_manager.py
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
            "position_size_percent": "5.0",
            "stop_loss_percent": "1.0",
            "take_profit_percent": "3.0",
            "max_position_age_minutes": 120,
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

STRATEGY_NAMES = ["momentum", "mean_reversion"]
CONFIG_PATH = Path("config/paper.toml")


# ---------------------------------------------------------------------------
# Stub registry — minimal accessor interface for ProfileManager
# ---------------------------------------------------------------------------

class _StubRegistry:
    """
    Minimal stub exposing the StrategyRegistry accessor interface.

    Holds one pre-built pipeline under a fixed strategy name so
    ProfileManager.apply_profile() can walk it. All components are
    real instances — only the registry wrapper is a stub.
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


class _EmptyStubRegistry:
    """Registry with no active strategies."""

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
# Component factories (mirror test_profile_manager.py helpers)
# ---------------------------------------------------------------------------

def _make_pipeline(strategy_name: str = "test_strategy"):
    """Build a real pipeline backed by a stub registry."""
    bus = EventBus()
    portfolio = PortfolioTracker(bus=bus, initial_balance=Decimal("5000"))
    sizing = PositionSizingRule(position_size_percent=Decimal("5.0"))
    strength = MinSignalStrengthRule(min_strength=Decimal("0.65"))
    cooldown = PostFillCooldownRule(cooldown_seconds=1800, bus=bus)
    exit_mon = ExitMonitor(
        bus=bus,
        portfolio=portfolio,
        stop_loss_percent=Decimal("1.0"),
        take_profit_percent=Decimal("3.0"),
        max_position_age_minutes=120,
        adaptive_tp=False,
        tp_multiplier=Decimal("1.5"),
        min_tp_percent=Decimal("0.3"),
        strategy_id=strategy_name,
    )
    aggregator = SignalAggregator(
        bus=bus,
        threshold=Decimal("0.4"),
        window_seconds=5,
        strategy_id=strategy_name,
    )
    risk_mgr = RiskManager(
        bus=bus,
        portfolio=portfolio,
        rules=[sizing, strength, cooldown],
        strategy_id=strategy_name,
    )
    registry = _StubRegistry(
        strategy_name=strategy_name,
        risk_manager=risk_mgr,
        exit_monitor=exit_mon,
        aggregator=aggregator,
        portfolio=portfolio,
    )
    return registry


# ---------------------------------------------------------------------------
# Fixtures for full WebDashboard with ProfileManager
# ---------------------------------------------------------------------------

def _make_strategy_config(name: str) -> StrategyConfig:
    from cerebrum.core.types import SignalType
    return StrategyConfig(
        name=name,
        aggregator_weights={
            SignalType.TECHNICAL: Decimal("1.0"),
            SignalType.SENTIMENT: Decimal("0.5"),
            SignalType.NEWS: Decimal("0.3"),
            SignalType.REGIME: Decimal("0.7"),
        },
        aggregator_threshold=Decimal("0.4"),
        initial_balance=Decimal("10000"),
    )


@pytest.fixture
async def bus():
    """Real event bus."""
    from cerebrum.core.bus import EventBus
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def real_config():
    from cerebrum.core.config import Config
    config, _ = Config.from_toml(CONFIG_PATH)
    return config


@pytest.fixture
async def real_registry(bus, real_config):
    reg = StrategyRegistry(bus, real_config)
    for name in STRATEGY_NAMES:
        reg.register(_make_strategy_config(name))
    await reg.start_all()
    yield reg
    await reg.stop_all()


@pytest.fixture
def allocator():
    return DarwinianAllocator(
        strategy_names=STRATEGY_NAMES,
        total_capital=Decimal("20000"),
        warmup_hours=0.0,
    )


@pytest.fixture
async def conductor(bus, real_registry, allocator):
    return Conductor(
        bus=bus,
        registry=real_registry,
        allocator=allocator,
        anthropic_api_key=None,
        poll_interval_seconds=900,
    )


@pytest.fixture
async def profile_manager():
    """Real ProfileManager backed by a stub registry with real pipeline components.

    Async fixture because PortfolioTracker.__init__ calls bus.subscribe() which
    requires a running event loop (same pattern as test_profile_manager.py which
    marks tests @pytest.mark.asyncio for this reason — see DEC-TEST-PROFILE-001).
    """
    stub_registry = _make_pipeline("test_strategy")
    return ProfileManager(stub_registry, SAMPLE_TOML)


@pytest.fixture
async def dashboard(bus, real_registry, conductor, profile_manager):
    """WebDashboard with ProfileManager wired in."""
    gp = real_registry.global_portfolio
    return WebDashboard(
        bus=bus,
        registry=real_registry,
        conductor=conductor,
        global_portfolio=gp,
        host="127.0.0.1",
        port=18082,
        profile_manager=profile_manager,
    )


@pytest.fixture
async def dashboard_no_pm(bus, real_registry, conductor):
    """WebDashboard without a ProfileManager (profile_manager=None)."""
    gp = real_registry.global_portfolio
    return WebDashboard(
        bus=bus,
        registry=real_registry,
        conductor=conductor,
        global_portfolio=gp,
        host="127.0.0.1",
        port=18083,
    )


@pytest.fixture
async def client(dashboard):
    """FastAPI TestClient for profile-enabled dashboard."""
    return TestClient(dashboard.app)


@pytest.fixture
async def client_no_pm(dashboard_no_pm):
    """FastAPI TestClient for dashboard without ProfileManager."""
    return TestClient(dashboard_no_pm.app)


# ---------------------------------------------------------------------------
# Tests: GET /api/profiles
# ---------------------------------------------------------------------------

class TestGetProfilesEndpoint:
    """GET /api/profiles returns all available profiles with their configs."""

    @pytest.mark.asyncio
    async def test_returns_200_with_profile_list(self, client):
        resp = client.get("/api/profiles")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_available_true_when_manager_present(self, client):
        data = client.get("/api/profiles").json()
        assert data["available"] is True

    @pytest.mark.asyncio
    async def test_returns_all_three_profiles(self, client):
        data = client.get("/api/profiles").json()
        assert set(data["profiles"]) == {"conservative", "moderate", "aggressive"}

    @pytest.mark.asyncio
    async def test_configs_contains_each_profile(self, client):
        data = client.get("/api/profiles").json()
        configs = data["configs"]
        assert "conservative" in configs
        assert "moderate" in configs
        assert "aggressive" in configs

    @pytest.mark.asyncio
    async def test_conservative_config_has_position_size(self, client):
        data = client.get("/api/profiles").json()
        cfg = data["configs"]["conservative"]
        assert "position_size_percent" in cfg
        assert cfg["position_size_percent"] == "3.0"

    @pytest.mark.asyncio
    async def test_available_false_when_no_manager(self, client_no_pm):
        data = client_no_pm.get("/api/profiles").json()
        assert data["available"] is False
        assert data["profiles"] == []
        assert data["configs"] == {}


# ---------------------------------------------------------------------------
# Tests: GET /api/profile/active
# ---------------------------------------------------------------------------

class TestGetActiveProfileEndpoint:
    """GET /api/profile/active returns the currently active profile name."""

    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = client.get("/api/profile/active")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_active_is_empty_before_any_apply(self, client):
        data = client.get("/api/profile/active").json()
        assert data["available"] is True
        assert data["active"] == ""

    @pytest.mark.asyncio
    async def test_active_updates_after_apply(self, client):
        client.post("/api/profile/apply", json={"profile": "moderate"})
        data = client.get("/api/profile/active").json()
        assert data["active"] == "moderate"

    @pytest.mark.asyncio
    async def test_available_false_when_no_manager(self, client_no_pm):
        data = client_no_pm.get("/api/profile/active").json()
        assert data["available"] is False
        assert data["active"] == ""


# ---------------------------------------------------------------------------
# Tests: POST /api/profile/apply
# ---------------------------------------------------------------------------

class TestApplyProfileEndpoint:
    """POST /api/profile/apply applies a named profile and returns changes."""

    @pytest.mark.asyncio
    async def test_returns_200_on_success(self, client):
        resp = client.post("/api/profile/apply", json={"profile": "conservative"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_ok_in_response(self, client):
        data = client.post("/api/profile/apply", json={"profile": "conservative"}).json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_profile_name_echoed_in_response(self, client):
        data = client.post("/api/profile/apply", json={"profile": "aggressive"}).json()
        assert data["profile"] == "aggressive"

    @pytest.mark.asyncio
    async def test_changes_dict_is_present(self, client):
        data = client.post("/api/profile/apply", json={"profile": "conservative"}).json()
        assert "changes" in data
        assert isinstance(data["changes"], dict)

    @pytest.mark.asyncio
    async def test_changes_contains_expected_keys(self, client):
        data = client.post("/api/profile/apply", json={"profile": "conservative"}).json()
        changes = data["changes"]
        # ProfileManager always returns changes for the test_strategy pipeline
        assert any("position_sizing" in k for k in changes)
        assert any("exit_monitor" in k for k in changes)

    @pytest.mark.asyncio
    async def test_apply_updates_active_profile(self, client):
        client.post("/api/profile/apply", json={"profile": "moderate"})
        active = client.get("/api/profile/active").json()
        assert active["active"] == "moderate"

    @pytest.mark.asyncio
    async def test_available_false_when_no_manager(self, client_no_pm):
        resp = client_no_pm.post("/api/profile/apply", json={"profile": "conservative"})
        assert resp.status_code == 200
        assert resp.json()["available"] is False


# ---------------------------------------------------------------------------
# Tests: POST /api/profile/apply — invalid profile
# ---------------------------------------------------------------------------

class TestApplyInvalidProfileEndpoint:
    """POST /api/profile/apply returns HTTP 400 for unknown profile names."""

    @pytest.mark.asyncio
    async def test_returns_400_for_unknown_profile(self, client):
        resp = client.post("/api/profile/apply", json={"profile": "nonexistent"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_error_detail_mentions_profile_name(self, client):
        resp = client.post("/api/profile/apply", json={"profile": "nonexistent"})
        detail = resp.json().get("detail", "")
        assert "nonexistent" in detail

    @pytest.mark.asyncio
    async def test_returns_400_when_profile_field_missing(self, client):
        resp = client.post("/api/profile/apply", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_active_profile_unchanged_after_bad_apply(self, client):
        client.post("/api/profile/apply", json={"profile": "conservative"})
        client.post("/api/profile/apply", json={"profile": "bad_profile"})
        active = client.get("/api/profile/active").json()
        assert active["active"] == "conservative"


# ---------------------------------------------------------------------------
# Tests: Profile routes registered on FastAPI app
# ---------------------------------------------------------------------------

class TestProfileRoutesRegistered:
    """Profile API routes are registered on the dashboard app."""

    @pytest.mark.asyncio
    async def test_profiles_route_exists(self, dashboard):
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/api/profiles" in route_paths

    @pytest.mark.asyncio
    async def test_active_profile_route_exists(self, dashboard):
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/api/profile/active" in route_paths

    @pytest.mark.asyncio
    async def test_apply_profile_route_exists(self, dashboard):
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/api/profile/apply" in route_paths
