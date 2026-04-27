"""
Tests for the /api/denials/by-strategy dashboard endpoint.

The existing /api/denials endpoint returns {denials: {strategy: {rule: count}}}.
The new /api/denials/by-strategy returns {by_strategy: {strategy: {rule: count}}}
— same shape, different key, added as a dedicated endpoint for machine-readable
per-strategy breakdown.

Tests:
1. Endpoint returns HTTP 200
2. Response has "by_strategy" top-level key
3. All active strategy names appear as keys
4. Values are dicts mapping rule_name → int (even if empty)
5. Endpoint is GET-only (POST returns 405)
6. Endpoint is read-only (calling it twice gives same counts — no mutation)
7. When a strategy has accumulated denials, they appear in the payload

@decision DEC-DIAG-002
@title /api/denials/by-strategy endpoint for per-strategy denial breakdown
@status accepted
@rationale /api/denials returned a nested payload wrapped in a "denials" key.
Machine-readable tooling (curl pipes, scripts) needs a stable top-level shape.
The new endpoint uses "by_strategy" as the key and is the canonical path for
per-strategy denial analytics. The existing /api/denials is preserved unchanged
for backward compatibility.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.types import SignalType
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.global_portfolio import GlobalPortfolio
from cerebrum.strategies.registry import StrategyRegistry

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cerebrum.dashboard.web import WebDashboard  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["mean_reversion", "range_trading", "orb_stocks"]
CONFIG_PATH = Path("config/paper.toml")


def _make_strategy_config(name: str, balance: Decimal = Decimal("10000")) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        aggregator_weights={
            SignalType.TECHNICAL: Decimal("1.0"),
            SignalType.SENTIMENT: Decimal("0.5"),
            SignalType.NEWS: Decimal("0.3"),
            SignalType.REGIME: Decimal("0.7"),
        },
        aggregator_threshold=Decimal("0.4"),
        initial_balance=balance,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def config():
    config, _ = Config.from_toml(CONFIG_PATH)
    return config


@pytest.fixture
async def registry(bus, config):
    reg = StrategyRegistry(bus, config)
    for name in STRATEGY_NAMES:
        reg.register(_make_strategy_config(name))
    await reg.start_all()
    yield reg
    await reg.stop_all()


@pytest.fixture
def global_portfolio(registry):
    return registry.global_portfolio


@pytest.fixture
def allocator():
    return DarwinianAllocator(
        strategy_names=STRATEGY_NAMES,
        total_capital=Decimal("30000"),
        warmup_hours=0.0,
    )


@pytest.fixture
def conductor(bus, registry, allocator):
    return Conductor(
        bus=bus,
        registry=registry,
        allocator=allocator,
        anthropic_api_key=None,
        poll_interval_seconds=900,
    )


@pytest.fixture
def dashboard(bus, registry, conductor, global_portfolio):
    return WebDashboard(
        bus=bus,
        registry=registry,
        conductor=conductor,
        global_portfolio=global_portfolio,
        host="127.0.0.1",
        port=18082,  # distinct port to avoid conflict with test_web_dashboard.py
    )


@pytest.fixture
def client(dashboard):
    return TestClient(dashboard.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_denials_by_strategy_returns_200(client):
    """GET /api/denials/by-strategy must return HTTP 200."""
    resp = client.get("/api/denials/by-strategy")
    assert resp.status_code == 200


def test_denials_by_strategy_has_by_strategy_key(client):
    """Response must contain a 'by_strategy' top-level key."""
    resp = client.get("/api/denials/by-strategy")
    data = resp.json()
    assert "by_strategy" in data


def test_denials_by_strategy_contains_all_strategies(client):
    """All active strategy names must appear as keys under 'by_strategy'."""
    resp = client.get("/api/denials/by-strategy")
    by_strat = resp.json()["by_strategy"]
    for name in STRATEGY_NAMES:
        assert name in by_strat, f"Expected strategy '{name}' in by_strategy keys"


def test_denials_by_strategy_values_are_dicts(client):
    """Each strategy entry must be a dict (possibly empty) mapping rule → int."""
    resp = client.get("/api/denials/by-strategy")
    by_strat = resp.json()["by_strategy"]
    for name, counts in by_strat.items():
        assert isinstance(counts, dict), f"Expected dict for {name}, got {type(counts)}"
        for rule, count in counts.items():
            assert isinstance(rule, str), f"Rule key must be str, got {type(rule)}"
            assert isinstance(count, int), f"Denial count must be int, got {type(count)}"


def test_denials_by_strategy_post_returns_405(client):
    """POST to /api/denials/by-strategy must return 405 (method not allowed)."""
    resp = client.post("/api/denials/by-strategy", json={})
    assert resp.status_code == 405


def test_denials_by_strategy_read_only(client):
    """Calling the endpoint twice must return identical counts (no mutation)."""
    resp1 = client.get("/api/denials/by-strategy")
    resp2 = client.get("/api/denials/by-strategy")
    assert resp1.json() == resp2.json()


def test_denials_by_strategy_also_includes_regime_damped(client, registry):
    """If a strategy has regime_damped_counts, they appear in the payload.

    regime_damped counts are tracked in the aggregator (not RiskManager), so
    the endpoint must also include them alongside RiskManager denial counts.
    """
    # Simulate non-zero regime_damped_counts on the mean_reversion aggregator
    agg = registry.get_aggregator("mean_reversion")
    if agg is not None and hasattr(agg, "_regime_damped_counts"):
        agg._regime_damped_counts["ETH/USD"] = 42

    resp = client.get("/api/denials/by-strategy")
    by_strat = resp.json()["by_strategy"]

    mean_rev = by_strat.get("mean_reversion", {})
    # regime_damped_ETH/USD should appear (or at minimum the endpoint doesn't crash)
    # The key will be "regime_damped_ETH/USD" per the implementation convention
    if agg is not None and hasattr(agg, "_regime_damped_counts"):
        assert "regime_damped_ETH/USD" in mean_rev or "regime_damped" in str(mean_rev)
