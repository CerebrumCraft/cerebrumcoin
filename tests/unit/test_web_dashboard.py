# @mock-exempt: AsyncMock WebSocket clients — WebSocket is a protocol boundary (no real browser in unit tests)
"""
Tests for the Phase 11D web dashboard (cerebrum.dashboard.web).

Covers:
- WebDashboard initialization and route registration
- REST API endpoints (/api/strategies, /api/regime, /api/denials, /api/conductor)
- Copilot mode endpoints (toggle, approve, reject, status)
- WebSocket connection and init payload
- Event handler broadcasting (_on_fill, _on_regime_change)
- Conductor copilot mode (queue, approve, reject pending allocations)
- Equity history tracking
- Fallback HTML when template is missing

Uses real EventBus, real Conductor, real StrategyRegistry with real pipelines,
real PortfolioTrackers, real RiskManagers, and real GlobalPortfolio. Only
WebSocket clients are mocked (they represent browser connections — an external
protocol boundary that cannot be instantiated without a real HTTP transport).

@decision DEC-TEST-013
@title Real-object dashboard tests with TestClient
@status accepted
@rationale WebDashboard orchestrates reads from Conductor, StrategyRegistry,
and GlobalPortfolio. Using real instances of all three verifies the dashboard
integrates correctly with the actual data shapes (Decimal allocations, denial
counter dicts, portfolio equity values). WebSocket clients are the only mock
— they represent browser connections (external transport boundary).
"""

import asyncio
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import FillEvent, RegimeChangeEvent
from cerebrum.core.types import EventType, Side, SignalType
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.global_portfolio import GlobalPortfolio
from cerebrum.strategies.registry import StrategyRegistry

# Optional dependency — skip if FastAPI not installed
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cerebrum.dashboard.web import WebDashboard  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["momentum", "mean_reversion", "breakout"]
CONFIG_PATH = Path("config/paper.toml")


def _make_strategy_config(name: str, balance: Decimal = Decimal("10000")) -> StrategyConfig:
    """Build a minimal StrategyConfig for testing."""
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
# Fixtures — all real objects
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Real event bus."""
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def config():
    """Real Config with defaults."""
    config, _raw_toml = Config.from_toml(CONFIG_PATH)
    return config


@pytest.fixture
async def registry(bus, config):
    """Real StrategyRegistry with three strategies started."""
    reg = StrategyRegistry(bus, config)
    for name in STRATEGY_NAMES:
        reg.register(_make_strategy_config(name))
    await reg.start_all()
    yield reg
    await reg.stop_all()


@pytest.fixture
def global_portfolio(registry):
    """Real GlobalPortfolio from the registry."""
    return registry.global_portfolio


@pytest.fixture
def allocator():
    """Real DarwinianAllocator."""
    return DarwinianAllocator(
        strategy_names=STRATEGY_NAMES,
        total_capital=Decimal("30000"),
        warmup_hours=0.0,
    )


@pytest.fixture
def conductor(bus, registry, allocator):
    """Real Conductor in math-only mode (no API key)."""
    return Conductor(
        bus=bus,
        registry=registry,
        allocator=allocator,
        anthropic_api_key=None,
        poll_interval_seconds=900,
    )


@pytest.fixture
def dashboard(bus, registry, conductor, global_portfolio):
    """Real WebDashboard with real collaborators."""
    return WebDashboard(
        bus=bus,
        registry=registry,
        conductor=conductor,
        global_portfolio=global_portfolio,
        host="127.0.0.1",
        port=18080,
    )


@pytest.fixture
def client(dashboard):
    """FastAPI TestClient for synchronous HTTP testing."""
    return TestClient(dashboard.app)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestDashboardInit:
    """Dashboard creates FastAPI app with all expected routes."""

    @pytest.mark.asyncio
    async def test_creates_fastapi_app(self, dashboard):
        assert dashboard.app is not None
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/" in route_paths
        assert "/api/strategies" in route_paths
        assert "/api/regime" in route_paths
        assert "/api/denials" in route_paths
        assert "/api/conductor" in route_paths
        assert "/ws" in route_paths

    @pytest.mark.asyncio
    async def test_copilot_routes_registered(self, dashboard):
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/api/copilot/approve" in route_paths
        assert "/api/copilot/reject" in route_paths
        assert "/api/copilot/status" in route_paths
        assert "/api/copilot/toggle" in route_paths


# ---------------------------------------------------------------------------
# Test: GET /api/strategies
# ---------------------------------------------------------------------------


class TestStrategiesEndpoint:
    """GET /api/strategies returns per-strategy stats from real registry."""

    @pytest.mark.asyncio
    async def test_returns_all_strategies(self, client):
        resp = client.get("/api/strategies")
        assert resp.status_code == 200
        data = resp.json()

        assert "strategies" in data
        assert "global_equity" in data
        assert "global_drawdown_pct" in data

        strats = data["strategies"]
        for name in STRATEGY_NAMES:
            assert name in strats
            s = strats[name]
            assert "allocation_pct" in s
            assert "equity" in s
            assert "cash" in s
            assert "pnl" in s
            assert "denials" in s

    @pytest.mark.asyncio
    async def test_equity_matches_initial_balance(self, client):
        """Each strategy starts with $10,000 initial balance."""
        resp = client.get("/api/strategies")
        data = resp.json()

        for name in STRATEGY_NAMES:
            s = data["strategies"][name]
            assert s["equity"] == 10000.0
            assert s["cash"] == 10000.0

    @pytest.mark.asyncio
    async def test_global_equity_is_sum(self, client):
        """Global equity should be sum of all strategy equities."""
        resp = client.get("/api/strategies")
        data = resp.json()
        assert data["global_equity"] == 30000.0

    @pytest.mark.asyncio
    async def test_global_drawdown_starts_at_zero(self, client):
        resp = client.get("/api/strategies")
        data = resp.json()
        assert data["global_drawdown_pct"] == 0.0


# ---------------------------------------------------------------------------
# Test: GET /api/regime
# ---------------------------------------------------------------------------


class TestRegimeEndpoint:
    """GET /api/regime returns Conductor's regime state."""

    @pytest.mark.asyncio
    async def test_initial_regime_is_unknown(self, client):
        resp = client.get("/api/regime")
        assert resp.status_code == 200
        data = resp.json()
        assert data["regime"] == "UNKNOWN"
        assert data["confidence"] == "0"


# ---------------------------------------------------------------------------
# Test: GET /api/denials
# ---------------------------------------------------------------------------


class TestDenialsEndpoint:
    """GET /api/denials returns per-strategy denial counts."""

    @pytest.mark.asyncio
    async def test_initial_denials_empty(self, client):
        resp = client.get("/api/denials")
        assert resp.status_code == 200
        data = resp.json()
        assert "denials" in data
        for name in STRATEGY_NAMES:
            assert name in data["denials"]
            assert data["denials"][name] == {}


# ---------------------------------------------------------------------------
# Test: GET /api/conductor
# ---------------------------------------------------------------------------


class TestConductorEndpoint:
    """GET /api/conductor returns Conductor internal state."""

    @pytest.mark.asyncio
    async def test_conductor_state(self, client):
        resp = client.get("/api/conductor")
        assert resp.status_code == 200
        data = resp.json()

        assert "last_allocations" in data
        assert data["latest_regime"] == "UNKNOWN"
        assert data["regime_confidence"] == "0"
        assert data["llm_enabled"] is False  # no API key
        assert data["copilot_mode"] is False
        assert data["has_pending"] is False


# ---------------------------------------------------------------------------
# Test: GET /api/equity_history
# ---------------------------------------------------------------------------


class TestEquityHistoryEndpoint:
    """GET /api/equity_history returns equity curve data."""

    @pytest.mark.asyncio
    async def test_empty_initially(self, client):
        resp = client.get("/api/equity_history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["history"] == []


# ---------------------------------------------------------------------------
# Test: Copilot mode endpoints
# ---------------------------------------------------------------------------


class TestCopilotEndpoints:
    """Copilot mode toggle, status, approve, reject via real Conductor."""

    @pytest.mark.asyncio
    async def test_toggle_on(self, client, conductor):
        assert conductor.copilot_mode is False
        resp = client.post("/api/copilot/toggle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["copilot_mode"] is True
        assert conductor.copilot_mode is True

    @pytest.mark.asyncio
    async def test_toggle_off_again(self, client, conductor):
        conductor.copilot_mode = True
        resp = client.post("/api/copilot/toggle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["copilot_mode"] is False

    @pytest.mark.asyncio
    async def test_status_no_pending(self, client, conductor):
        conductor.copilot_mode = True
        resp = client.get("/api/copilot/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["copilot_mode"] is True
        assert data["has_pending"] is False
        assert data["pending_allocation"] is None

    @pytest.mark.asyncio
    async def test_status_with_pending(self, client, conductor):
        conductor.copilot_mode = True
        conductor._pending_allocation = {
            "momentum": Decimal("60"),
            "mean_reversion": Decimal("25"),
            "breakout": Decimal("15"),
        }
        conductor._pending_reasoning = "Triggered at regime=BULL confidence=0.9"

        resp = client.get("/api/copilot/status")
        data = resp.json()
        assert data["has_pending"] is True
        assert data["pending_allocation"]["momentum"] == "60"
        assert "BULL" in data["pending_reasoning"]

    @pytest.mark.asyncio
    async def test_approve_calls_conductor(self, client, conductor):
        conductor.copilot_mode = True
        conductor._pending_allocation = {
            "momentum": Decimal("50"),
            "mean_reversion": Decimal("30"),
            "breakout": Decimal("20"),
        }
        conductor._pending_reasoning = "test"

        resp = client.post("/api/copilot/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # Pending should be cleared after approval
        assert conductor._pending_allocation is None

    @pytest.mark.asyncio
    async def test_reject_clears_pending(self, client, conductor):
        conductor.copilot_mode = True
        conductor._pending_allocation = {
            "momentum": Decimal("50"),
            "mean_reversion": Decimal("30"),
            "breakout": Decimal("20"),
        }

        resp = client.post("/api/copilot/reject")
        assert resp.status_code == 200
        assert conductor._pending_allocation is None

    @pytest.mark.asyncio
    async def test_approve_errors_when_copilot_off(self, client, conductor):
        conductor.copilot_mode = False
        resp = client.post("/api/copilot/approve")
        data = resp.json()
        assert data["status"] == "error"
        assert "not enabled" in data["message"]

    @pytest.mark.asyncio
    async def test_reject_errors_when_copilot_off(self, client, conductor):
        conductor.copilot_mode = False
        resp = client.post("/api/copilot/reject")
        data = resp.json()
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# Test: Index page
# ---------------------------------------------------------------------------


class TestIndexPage:
    """GET / serves the dashboard HTML."""

    @pytest.mark.asyncio
    async def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "CerebrumCoin" in resp.text

    @pytest.mark.asyncio
    async def test_fallback_html(self, dashboard):
        html = dashboard._fallback_html()
        assert "CerebrumCoin Dashboard" in html
        assert "Template not found" in html


# ---------------------------------------------------------------------------
# Test: Event handlers — WebSocket broadcast
# (AsyncMock for WebSocket clients: external transport boundary)
# ---------------------------------------------------------------------------


class TestEventBroadcast:
    """Event handlers broadcast fill and regime data to WS clients."""

    @pytest.mark.asyncio
    async def test_on_fill_broadcasts(self, dashboard):
        ws_mock = AsyncMock()
        dashboard._ws_clients.add(ws_mock)

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="test-order-1",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )

        await dashboard._on_fill(fill)

        # Two messages: fill notification + strategy_update
        assert ws_mock.send_text.call_count == 2

        import json
        first_msg = json.loads(ws_mock.send_text.call_args_list[0][0][0])
        assert first_msg["type"] == "fill"
        assert first_msg["data"]["symbol"] == "BTC/USD"
        assert first_msg["data"]["strategy"] == "momentum"

        second_msg = json.loads(ws_mock.send_text.call_args_list[1][0][0])
        assert second_msg["type"] == "strategy_update"
        assert "strategies" in second_msg["data"]

    @pytest.mark.asyncio
    async def test_on_fill_tracks_equity_history(self, dashboard):
        assert len(dashboard._equity_history) == 0

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="test-order-2",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
        )

        await dashboard._on_fill(fill)
        assert len(dashboard._equity_history) == 1
        assert dashboard._equity_history[0]["equity"] == 30000.0

    @pytest.mark.asyncio
    async def test_equity_history_capped_at_500(self, dashboard):
        dashboard._equity_history = [{"ts": i, "equity": 10000.0} for i in range(500)]

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="test-cap",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
        )

        await dashboard._on_fill(fill)
        assert len(dashboard._equity_history) == 500

    @pytest.mark.asyncio
    async def test_on_regime_change_broadcasts(self, dashboard):
        ws_mock = AsyncMock()
        dashboard._ws_clients.add(ws_mock)

        event = RegimeChangeEvent(
            event_type=EventType.REGIME_CHANGE,
            timestamp=1_000_000.0,
            from_regime="SIDEWAYS",
            to_regime="BULL",
            confidence=Decimal("0.9"),
            indicators={"symbol": "BTC/USD"},
        )

        await dashboard._on_regime_change(event)
        assert ws_mock.send_text.call_count == 1

        import json
        msg = json.loads(ws_mock.send_text.call_args[0][0])
        assert msg["type"] == "regime_change"
        assert msg["data"]["symbol"] == "BTC/USD"
        assert msg["data"]["from"] == "SIDEWAYS"
        assert msg["data"]["to"] == "BULL"
        assert msg["data"]["confidence"] == "0.9"

    @pytest.mark.asyncio
    async def test_on_fill_ignores_non_fill_events(self, dashboard):
        ws_mock = AsyncMock()
        dashboard._ws_clients.add(ws_mock)

        event = RegimeChangeEvent(
            event_type=EventType.REGIME_CHANGE,
            timestamp=1_000_000.0,
            from_regime="SIDEWAYS",
            to_regime="BULL",
            confidence=Decimal("0.9"),
            indicators={},
        )

        await dashboard._on_fill(event)
        ws_mock.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_regime_ignores_non_regime_events(self, dashboard):
        ws_mock = AsyncMock()
        dashboard._ws_clients.add(ws_mock)

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="test",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
        )

        await dashboard._on_regime_change(fill)
        ws_mock.send_text.assert_not_called()


# ---------------------------------------------------------------------------
# Test: WebSocket broadcast resilience
# ---------------------------------------------------------------------------


class TestBroadcastResilience:
    """Broadcast gracefully handles dead clients and empty client sets."""

    @pytest.mark.asyncio
    async def test_removes_dead_clients(self, dashboard):
        alive = AsyncMock()
        dead = AsyncMock()
        dead.send_text.side_effect = ConnectionError("disconnected")

        dashboard._ws_clients = {alive, dead}
        await dashboard._broadcast({"type": "test", "data": {}})

        assert alive in dashboard._ws_clients
        assert dead not in dashboard._ws_clients

    @pytest.mark.asyncio
    async def test_no_clients_is_noop(self, dashboard):
        dashboard._ws_clients.clear()
        await dashboard._broadcast({"type": "test", "data": {}})


# ---------------------------------------------------------------------------
# Test: Conductor copilot mode logic (real Conductor)
# ---------------------------------------------------------------------------


class TestConductorCopilotMode:
    """Conductor copilot mode queues, approves, rejects allocations."""

    @pytest.mark.asyncio
    async def test_copilot_queues_allocation(self, conductor):
        conductor.copilot_mode = True

        allocations = {
            "momentum": Decimal("50"),
            "mean_reversion": Decimal("30"),
            "breakout": Decimal("20"),
        }

        await conductor._apply_allocations(allocations)

        assert conductor._pending_allocation is not None
        assert conductor._pending_allocation["momentum"] == Decimal("50")
        assert conductor._pending_reasoning is not None
        # _last_allocations should NOT have the new values
        assert conductor._last_allocations.get("momentum") != Decimal("50")

    @pytest.mark.asyncio
    async def test_approve_applies_allocation(self, conductor):
        conductor.copilot_mode = True

        # Use allocations within the 50% single-strategy cap (DEC-CONDUCTOR-004)
        # so the values pass through unmodified and the assertion is exact.
        allocations = {
            "momentum": Decimal("50"),
            "mean_reversion": Decimal("30"),
            "breakout": Decimal("20"),
        }
        await conductor._apply_allocations(allocations)
        assert conductor._pending_allocation is not None

        await conductor.approve_pending()

        assert conductor._pending_allocation is None
        assert conductor._pending_reasoning is None
        assert conductor._last_allocations["momentum"] == Decimal("50")

    @pytest.mark.asyncio
    async def test_reject_clears_without_applying(self, conductor):
        conductor.copilot_mode = True

        conductor._last_allocations = {
            "momentum": Decimal("40"),
            "mean_reversion": Decimal("35"),
            "breakout": Decimal("25"),
        }
        original = dict(conductor._last_allocations)

        await conductor._apply_allocations({
            "momentum": Decimal("70"),
            "mean_reversion": Decimal("20"),
            "breakout": Decimal("10"),
        })

        await conductor.reject_pending()

        assert conductor._pending_allocation is None
        assert conductor._last_allocations == original

    @pytest.mark.asyncio
    async def test_approve_noop_when_nothing_pending(self, conductor):
        conductor.copilot_mode = True
        assert conductor._pending_allocation is None
        await conductor.approve_pending()
        assert conductor._pending_allocation is None

    @pytest.mark.asyncio
    async def test_newer_proposal_overwrites_pending(self, conductor):
        """DEC-DASH-003: newer proposal overwrites unresolved pending."""
        conductor.copilot_mode = True

        await conductor._apply_allocations({
            "momentum": Decimal("50"),
            "mean_reversion": Decimal("30"),
            "breakout": Decimal("20"),
        })
        assert conductor._pending_allocation["momentum"] == Decimal("50")

        await conductor._apply_allocations({
            "momentum": Decimal("70"),
            "mean_reversion": Decimal("20"),
            "breakout": Decimal("10"),
        })
        assert conductor._pending_allocation["momentum"] == Decimal("70")

    @pytest.mark.asyncio
    async def test_bypass_copilot_applies_directly(self, conductor):
        """_bypass_copilot=True skips the queue (used by approve_pending)."""
        conductor.copilot_mode = True

        allocations = {
            "momentum": Decimal("45"),
            "mean_reversion": Decimal("35"),
            "breakout": Decimal("20"),
        }
        await conductor._apply_allocations(allocations, _bypass_copilot=True)

        # Should have applied directly, not queued
        assert conductor._pending_allocation is None
        assert conductor._last_allocations["momentum"] == Decimal("45")


# ---------------------------------------------------------------------------
# Test: WebSocket init payload
# ---------------------------------------------------------------------------


class TestWebSocketInit:
    """WebSocket connection receives init payload with strategy data."""

    @pytest.mark.asyncio
    async def test_ws_sends_init_on_connect(self, dashboard):
        test_client = TestClient(dashboard.app)
        with test_client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "init"
            assert "strategies" in data["data"]
            assert "global_equity" in data["data"]
            for name in STRATEGY_NAMES:
                assert name in data["data"]["strategies"]


# ---------------------------------------------------------------------------
# Test: Lifecycle (start/stop)
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Dashboard start/stop lifecycle does not raise."""

    @pytest.mark.asyncio
    async def test_start_stop(self, dashboard, bus):
        await dashboard.start()
        await asyncio.sleep(0.1)
        await dashboard.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_ws_clients(self, dashboard):
        ws_mock = AsyncMock()
        dashboard._ws_clients.add(ws_mock)

        await dashboard.stop()
        assert len(dashboard._ws_clients) == 0


# ---------------------------------------------------------------------------
# Test: Phase 12F — GET /api/strategy_equity_history
# ---------------------------------------------------------------------------


class TestStrategyEquityHistoryEndpoint:
    """GET /api/strategy_equity_history returns per-strategy equity curves."""

    @pytest.mark.asyncio
    async def test_empty_initially(self, client):
        resp = client.get("/api/strategy_equity_history")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        assert data["strategies"] == {}

    @pytest.mark.asyncio
    async def test_returns_per_strategy_structure(self, dashboard):
        """After a fill, each strategy gets an equity snapshot entry."""
        # Pre-seed one history entry per strategy to simulate a fill snapshot
        for name in STRATEGY_NAMES:
            dashboard._strategy_equity_history[name] = [
                {"ts": 1_000_000, "equity": 10000.0}
            ]

        client = TestClient(dashboard.app)
        resp = client.get("/api/strategy_equity_history")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        for name in STRATEGY_NAMES:
            assert name in data["strategies"]
            points = data["strategies"][name]
            assert len(points) == 1
            assert points[0]["equity"] == 10000.0
            assert "ts" in points[0]

    @pytest.mark.asyncio
    async def test_fill_triggers_strategy_snapshot(self, dashboard):
        """_on_fill snapshots all strategy equities after each fill."""
        assert dashboard._strategy_equity_history == {}

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="snap-test",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill)

        # All registered strategies should have a snapshot
        for name in STRATEGY_NAMES:
            assert name in dashboard._strategy_equity_history
            assert len(dashboard._strategy_equity_history[name]) == 1
            assert dashboard._strategy_equity_history[name][0]["equity"] == 10000.0

    @pytest.mark.asyncio
    async def test_strategy_equity_history_capped_at_500(self, dashboard):
        """Each strategy history is capped at 500 points."""
        for name in STRATEGY_NAMES:
            dashboard._strategy_equity_history[name] = [
                {"ts": i, "equity": 10000.0} for i in range(500)
            ]

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_500.0,
            order_id="cap-test",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill)

        for name in STRATEGY_NAMES:
            assert len(dashboard._strategy_equity_history[name]) == 500


# ---------------------------------------------------------------------------
# Test: Phase 12F — GET /api/scorecard
# ---------------------------------------------------------------------------


class TestScorecardEndpoint:
    """GET /api/scorecard returns go-live criteria with pass/fail."""

    @pytest.mark.asyncio
    async def test_returns_required_fields(self, client):
        resp = client.get("/api/scorecard")
        assert resp.status_code == 200
        data = resp.json()
        assert "criteria" in data
        assert "verdict" in data
        assert "kill_alerts" in data

    @pytest.mark.asyncio
    async def test_initial_verdict_is_insufficient(self, client):
        """With no fills, scorecard should show INSUFFICIENT."""
        resp = client.get("/api/scorecard")
        data = resp.json()
        assert data["verdict"] == "INSUFFICIENT"

    @pytest.mark.asyncio
    async def test_criteria_have_required_structure(self, client):
        resp = client.get("/api/scorecard")
        data = resp.json()
        for c in data["criteria"]:
            assert "name" in c
            assert "target" in c
            assert "current" in c
            assert "pass" in c

    @pytest.mark.asyncio
    async def test_sharpe_criterion_is_na(self, client):
        """Sharpe ratio criterion is always N/A (requires analyze.py)."""
        resp = client.get("/api/scorecard")
        data = resp.json()
        sharpe = next(c for c in data["criteria"] if "Sharpe" in c["name"])
        assert sharpe["pass"] is None
        assert "analyze.py" in sharpe["current"]

    @pytest.mark.asyncio
    async def test_kill_alerts_empty_at_start(self, client):
        """No kill alerts at startup (zero drawdown, zero P&L loss)."""
        resp = client.get("/api/scorecard")
        data = resp.json()
        assert data["kill_alerts"] == []

    @pytest.mark.asyncio
    async def test_verdict_nogo_after_fill(self, dashboard):
        """After first fill, verdict becomes NO-GO (criteria not yet met)."""
        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="sc-test",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill)
        assert dashboard._first_fill_time is not None

        client = TestClient(dashboard.app)
        resp = client.get("/api/scorecard")
        data = resp.json()
        # Has data now, but days_trading < 30, fill_count < 50 → NO-GO
        assert data["verdict"] == "NO-GO"

    def test_days_trading_seeded_from_db(self, bus, registry, conductor, global_portfolio, tmp_path):
        """DEC-DASH-006: days_trading > 0 after restart when trades DB has prior fills.

        Creates a real SQLite DB with a trade entry_time 10 days in the past,
        passes db_path to WebDashboard, and verifies /api/scorecard reports
        days_trading >= 10 without any FillEvent being fired in this session.
        """
        import sqlite3
        import time

        # Build a minimal trades table with one old entry
        db = tmp_path / "cerebrum.db"
        ten_days_ago = time.time() - (10 * 86400)
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE trades (id INTEGER PRIMARY KEY, entry_time REAL)"
        )
        conn.execute("INSERT INTO trades (entry_time) VALUES (?)", (ten_days_ago,))
        conn.commit()
        conn.close()

        dash = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18099,
            db_path=db,
        )

        # _first_fill_time must be seeded immediately — no fill events needed
        assert dash._first_fill_time is not None
        assert abs(dash._first_fill_time - ten_days_ago) < 1.0

        # Scorecard must reflect the seeded time
        client = TestClient(dash.app)
        resp = client.get("/api/scorecard")
        assert resp.status_code == 200
        data = resp.json()
        days_criterion = next(
            c for c in data["criteria"] if "Days" in c["name"]
        )
        # "current" is formatted as a string (e.g. "10.0") — parse before comparing
        assert float(days_criterion["current"]) >= 10.0

    def test_days_trading_defaults_zero_when_no_db(self, bus, registry, conductor, global_portfolio):
        """Without db_path, _first_fill_time stays None and days_trading is 0."""
        dash = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18098,
        )
        assert dash._first_fill_time is None

        client = TestClient(dash.app)
        resp = client.get("/api/scorecard")
        data = resp.json()
        days_criterion = next(
            c for c in data["criteria"] if "Days" in c["name"]
        )
        # "current" is formatted as a string (e.g. "0.0") — parse before comparing
        assert float(days_criterion["current"]) == 0.0

    def test_days_trading_graceful_on_missing_db(self, bus, registry, conductor, global_portfolio, tmp_path):
        """_seed_first_fill_time swallows errors for missing/empty DB paths."""
        nonexistent = tmp_path / "does_not_exist.db"
        # Should not raise — falls back to in-memory (None)
        dash = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18097,
            db_path=nonexistent,
        )
        # sqlite3.connect creates an empty file with no tables — MIN query fails
        # gracefully, _first_fill_time stays None
        assert dash._first_fill_time is None


# ---------------------------------------------------------------------------
# Test: Phase 12F — GET /api/commission
# ---------------------------------------------------------------------------


class TestCommissionEndpoint:
    """GET /api/commission returns per-strategy commission drag data."""

    @pytest.mark.asyncio
    async def test_returns_required_fields(self, client):
        resp = client.get("/api/commission")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_initial_commission_zero(self, client):
        """Before any fills, all commission values are zero."""
        resp = client.get("/api/commission")
        data = resp.json()
        total = data["total"]
        assert total["commission"] == 0.0
        assert total["net_pnl"] == 0.0
        assert total["gross_pnl"] == 0.0
        assert total["drag_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_per_strategy_structure(self, client):
        """Each strategy entry has the four required fields."""
        resp = client.get("/api/commission")
        data = resp.json()
        for name in STRATEGY_NAMES:
            assert name in data["strategies"]
            s = data["strategies"][name]
            assert "gross_pnl" in s
            assert "commission" in s
            assert "net_pnl" in s
            assert "drag_pct" in s

    @pytest.mark.asyncio
    async def test_commission_tracked_after_fill(self, dashboard):
        """Fill events accumulate commission per strategy."""
        assert dashboard._commission_totals == {}

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="comm-test",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.135"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill)

        assert "momentum" in dashboard._commission_totals
        assert abs(dashboard._commission_totals["momentum"] - 0.135) < 1e-6

    @pytest.mark.asyncio
    async def test_commission_accumulates_across_fills(self, dashboard):
        """Multiple fills from the same strategy accumulate commission."""
        for i in range(3):
            fill = FillEvent(
                event_type=EventType.FILL,
                timestamp=1_000_000.0 + i,
                order_id=f"comm-acc-{i}",
                symbol="BTC/USD",
                side=Side.BUY,
                filled_amount=Decimal("0.001"),
                fill_price=Decimal("84000"),
                commission=Decimal("0.10"),
                commission_asset="USD",
                strategy_id="breakout",
            )
            await dashboard._on_fill(fill)

        assert abs(dashboard._commission_totals["breakout"] - 0.30) < 1e-6

    @pytest.mark.asyncio
    async def test_fill_count_tracked_per_strategy(self, dashboard):
        """Fill counts increment independently per strategy."""
        for strat, count in [("momentum", 2), ("mean_reversion", 1)]:
            for i in range(count):
                fill = FillEvent(
                    event_type=EventType.FILL,
                    timestamp=1_000_000.0 + i,
                    order_id=f"fc-{strat}-{i}",
                    symbol="BTC/USD",
                    side=Side.BUY,
                    filled_amount=Decimal("0.001"),
                    fill_price=Decimal("84000"),
                    commission=Decimal("0.10"),
                    commission_asset="USD",
                    strategy_id=strat,
                )
                await dashboard._on_fill(fill)

        assert dashboard._fill_counts["momentum"] == 2
        assert dashboard._fill_counts["mean_reversion"] == 1
        assert dashboard._fill_counts.get("breakout", 0) == 0

    @pytest.mark.asyncio
    async def test_first_fill_time_recorded(self, dashboard):
        """First fill records _first_fill_time; subsequent fills do not overwrite."""
        assert dashboard._first_fill_time is None

        fill1 = FillEvent(
            event_type=EventType.FILL,
            timestamp=1_000_000.0,
            order_id="fft-1",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill1)
        assert dashboard._first_fill_time == 1_000_000.0

        fill2 = FillEvent(
            event_type=EventType.FILL,
            timestamp=2_000_000.0,
            order_id="fft-2",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.001"),
            fill_price=Decimal("84000"),
            commission=Decimal("0.10"),
            commission_asset="USD",
            strategy_id="momentum",
        )
        await dashboard._on_fill(fill2)
        # Should still be the first fill's timestamp
        assert dashboard._first_fill_time == 1_000_000.0


# ---------------------------------------------------------------------------
# Test: Phase 12F — New routes registered
# ---------------------------------------------------------------------------


class TestPhase12FRoutesRegistered:
    """Phase 12F endpoints are present in the FastAPI route list."""

    @pytest.mark.asyncio
    async def test_new_routes_registered(self, dashboard):
        route_paths = [r.path for r in dashboard.app.routes]
        assert "/api/strategy_equity_history" in route_paths
        assert "/api/scorecard" in route_paths
        assert "/api/commission" in route_paths


# ---------------------------------------------------------------------------
# Test: _get_global_equity — paper adapter vs GlobalPortfolio fallback
# @mock-exempt: PaperTradingAdapter is an external exchange adapter boundary
#               (simulates a live exchange connection). Mocking it here is
#               appropriate — we need to control its return value without
#               starting a real paper trading session.
# ---------------------------------------------------------------------------


class TestGetGlobalEquityPaperAdapterGroundTruth:
    """With paper_adapter set, _get_global_equity uses adapter as ground truth.

    Root cause fixed: GlobalPortfolio.get_total_equity() sums all 6 strategy
    PortfolioTracker equities, double-counting capital because each tracker
    holds its own cash allocation + position marks independently of the real
    exchange state (~$11,200 displayed vs ~$9,994 actual). Using
    PaperTradingAdapter.get_portfolio_summary() as the single source of truth
    corrects this display bug.
    """

    @pytest.mark.asyncio
    async def test_with_paper_adapter_returns_adapter_value(
        self, bus, registry, conductor, global_portfolio
    ):
        """_get_global_equity() returns adapter total_value_usd, not GlobalPortfolio sum."""
        from unittest.mock import MagicMock
        paper_adapter = MagicMock()
        paper_adapter.get_portfolio_summary.return_value = {
            "total_value_usd": "9994.00",
            "balances": {"USD": "9994.00"},
            "positions": {},
            "trade_count": 42,
            "pnl_usd": "-6.00",
        }

        d = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18090,
            paper_adapter=paper_adapter,
        )

        result = d._get_global_equity()
        assert result == Decimal("9994.00")

    @pytest.mark.asyncio
    async def test_build_strategies_payload_uses_adapter_equity(
        self, bus, registry, conductor, global_portfolio
    ):
        """_build_strategies_payload()['global_equity'] reflects adapter value, not inflated sum."""
        from unittest.mock import MagicMock
        paper_adapter = MagicMock()
        paper_adapter.get_portfolio_summary.return_value = {
            "total_value_usd": "9994.00",
            "balances": {"USD": "9994.00"},
            "positions": {},
            "trade_count": 5,
            "pnl_usd": "-6.00",
        }

        d = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18091,
            paper_adapter=paper_adapter,
        )

        payload = d._build_strategies_payload()
        # GlobalPortfolio for 3 strategies at $10k each = $30k (inflated).
        # The adapter returns $9,994 — dashboard must use the adapter value.
        assert payload["global_equity"] == pytest.approx(9994.0)

    @pytest.mark.asyncio
    async def test_without_paper_adapter_falls_back_to_global_portfolio(
        self, bus, registry, conductor, global_portfolio
    ):
        """Without paper_adapter (live mode), falls back to GlobalPortfolio."""
        d = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18092,
            paper_adapter=None,
        )

        result = d._get_global_equity()
        # Three strategies at $10,000 each = $30,000
        assert result == Decimal("30000.00")

    @pytest.mark.asyncio
    async def test_drawdown_still_uses_global_portfolio(
        self, bus, registry, conductor, global_portfolio
    ):
        """get_total_drawdown() is still sourced from GlobalPortfolio regardless of adapter."""
        from unittest.mock import MagicMock
        paper_adapter = MagicMock()
        paper_adapter.get_portfolio_summary.return_value = {
            "total_value_usd": "9994.00",
        }

        d = WebDashboard(
            bus=bus,
            registry=registry,
            conductor=conductor,
            global_portfolio=global_portfolio,
            host="127.0.0.1",
            port=18093,
            paper_adapter=paper_adapter,
        )

        payload = d._build_strategies_payload()
        # Drawdown still comes from GlobalPortfolio — starts at 0.0
        assert payload["global_drawdown_pct"] == pytest.approx(0.0)
