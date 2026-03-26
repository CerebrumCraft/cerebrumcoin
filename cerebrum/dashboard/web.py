"""
FastAPI + WebSocket web dashboard for CerebrumCoin multi-strategy visualization.

Provides a live trading terminal UI with:
- Per-strategy capital allocation bars (animated, color-coded)
- Real-time regime state badge
- P&L panel with per-strategy breakdowns
- Guard denial counters
- Conductor reasoning log
- Copilot mode: human-in-the-loop approval for allocation changes
- Per-strategy equity curves (Phase 12F)
- Go-live scorecard panel (Phase 12F)
- Commission drag visualization (Phase 12F)
- Guard denial heatmap with color-coded counts (Phase 12F)

@decision DEC-DASH-001
@title FastAPI + uvicorn run in background asyncio task, not a subprocess
@status accepted
@rationale Running uvicorn.Server inside the existing asyncio event loop (via
asyncio.create_task) avoids subprocess management, shared-memory IPC, and
second event loop complexity. The server's lifespan is tied to the main
application: when CerebrumCoin.stop() is called, the server task is cancelled.
This is the recommended pattern in uvicorn's own documentation for embedding.

@decision DEC-DASH-002
@title htmx + FastAPI web dashboard for multi-strategy visualization
@status accepted
@rationale Pure server-side rendering with htmx for partial updates eliminates
the React/npm build step entirely. The dashboard is a dev/ops tool for a local
Python project — Python-native tooling matches the project's learning goals
(per eng review decision). htmx's hx-trigger="every 5s" gives near-real-time
updates without a persistent WebSocket requirement for the full page. WebSocket
is used only for push events (fills, regime changes, conductor decisions) that
need sub-second latency.

@decision DEC-DASH-003
@title Copilot mode queues pending allocations rather than blocking the Conductor
@status accepted
@rationale When copilot_mode=True the Conductor produces an allocation proposal
but defers _apply_allocations() until the human approves via the dashboard.
Rejected proposals are silently discarded — the last applied allocation stays
in force. This is non-blocking: the event loop is never suspended waiting for
human input. A pending proposal is overwritten if a newer one arrives before
approval, so the human always sees the freshest proposal.

@decision DEC-DASH-004
@title Phase 12F state tracked from FillEvents in-memory — no DB queries
@status accepted
@rationale The dashboard runs embedded in the trading process and must stay
lightweight. Per-strategy equity history, fill counts, commission totals, and
realized P&L are accumulated incrementally from FillEvent callbacks rather
than querying the SQLite trade database. This is sufficient for real-time
visualization. For authoritative scorecard analysis (Sharpe, full attribution),
scripts/analyze.py queries the trade DB directly and is noted in the scorecard
as a supplement for criteria that cannot be computed inline.
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

# FastAPI is an optional dependency (pip install cerebrumcoin[dashboard])
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, RegimeChangeEvent
from cerebrum.core.types import EventType

if TYPE_CHECKING:
    from cerebrum.conductor.conductor import Conductor
    from cerebrum.strategies.global_portfolio import GlobalPortfolio
    from cerebrum.strategies.registry import StrategyRegistry

logger = structlog.get_logger()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

# Go-live scorecard criteria definitions.
# Each tuple: (key, label, target_desc, target_value, comparison)
# comparison "gte": current >= target_value passes
# comparison "lte": current <= target_value passes
_SCORECARD_CRITERIA = [
    ("days_trading",    "Days trading",              ">= 30 days",  30.0,  "gte"),
    ("net_pnl",         "Net P&L",                   ">= $0",        0.0,  "gte"),
    ("max_drawdown",    "Max drawdown",               "<= 15%",      15.0,  "lte"),
    ("commission_drag", "Commission drag",            "<= 40%",      40.0,  "lte"),
    ("fill_count",      "Total fills",                ">= 50",       50.0,  "gte"),
    ("strategy_count",  "Active strategies",          ">= 2",         2.0,  "gte"),
    ("concentration",   "Single strategy P&L share", "<= 80%",      80.0,  "lte"),
]

# Kill-alert thresholds — levels that indicate immediate risk
_KILL_DRAWDOWN_PCT = 20.0    # global drawdown exceeding this triggers alert
_KILL_NET_PNL = -500.0       # net P&L below this triggers alert


class WebDashboard:
    """
    FastAPI + WebSocket web dashboard for multi-strategy visualization.

    Lifecycle::

        dashboard = WebDashboard(bus, registry, conductor, global_portfolio)
        await dashboard.start()   # non-blocking: server runs as background task
        # ... trading runs ...
        await dashboard.stop()    # graceful shutdown

    WebSocket push events:
        strategy_update     — per-strategy P&L / allocation (on FILL)
        regime_change       — regime transition with confidence
        guard_denial        — denial count delta per strategy/rule
        conductor_reasoning — LLM allocation decision with reasoning
        fill                — individual order fill notification

    Phase 12F additions (DEC-DASH-004):
        /api/strategy_equity_history — per-strategy equity curves
        /api/scorecard               — go-live criteria evaluation
        /api/commission              — per-strategy commission/drag data
        Denial heatmap data included in /api/denials (count magnitudes)
    """

    def __init__(
        self,
        bus: EventBus,
        registry: "StrategyRegistry",
        conductor: "Conductor",
        global_portfolio: "GlobalPortfolio",
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """
        Initialise the web dashboard.

        Args:
            bus: Shared event bus — dashboard subscribes to FILL, REGIME_CHANGE,
                 POSITION_UPDATE, RISK_ALERT events.
            registry: StrategyRegistry for per-strategy component access.
            conductor: Conductor for allocation state and copilot mode.
            global_portfolio: GlobalPortfolio for aggregate equity view.
            host: Bind address (default 127.0.0.1 — localhost only).
            port: HTTP port (default 8080).
        """
        if not _FASTAPI_AVAILABLE:
            raise ImportError(
                "WebDashboard requires FastAPI: pip install cerebrumcoin[dashboard]"
            )

        self._bus = bus
        self._registry = registry
        self._conductor = conductor
        self._global_portfolio = global_portfolio
        self._host = host
        self._port = port

        self.app = FastAPI(title="CerebrumCoin Dashboard")
        self._ws_clients: set[WebSocket] = set()
        self._server_task: asyncio.Task | None = None
        self._server: uvicorn.Server | None = None

        # Local cache of denial counts for delta broadcasting
        self._last_denial_counts: dict[str, dict[str, int]] = {}

        # Global equity history for chart (capped at 500 points)
        self._equity_history: list[dict[str, Any]] = []

        # Phase 12F: per-strategy equity snapshots, capped at 500 points each
        self._strategy_equity_history: dict[str, list[dict[str, Any]]] = {}

        # Phase 12F: fill counts per strategy
        self._fill_counts: dict[str, int] = {}

        # Phase 12F: cumulative commission in USD per strategy
        self._commission_totals: dict[str, float] = {}

        # Phase 12F: timestamp of first fill (for "days trading" criterion)
        self._first_fill_time: float | None = None

        self._log = logger.bind(component="web_dashboard")
        self._setup_routes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start FastAPI server as a background asyncio task."""
        # Subscribe to live events
        self._bus.subscribe(EventType.FILL, self._on_fill, "dashboard_fill")
        self._bus.subscribe(
            EventType.REGIME_CHANGE, self._on_regime_change, "dashboard_regime"
        )

        # Mount static files if directory exists
        if _STATIC_DIR.exists():
            self.app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._port,
            log_level="warning",  # suppress uvicorn's own INFO spam
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        self._server_task = asyncio.create_task(
            self._server.serve(), name="web_dashboard_server"
        )
        self._log.info(
            "web_dashboard_started",
            url=f"http://{self._host}:{self._port}",
        )

    async def stop(self) -> None:
        """Shut down the server and close all WebSocket connections."""
        # Close WebSocket clients gracefully
        for ws in list(self._ws_clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._ws_clients.clear()

        if self._server:
            self._server.should_exit = True

        if self._server_task:
            self._server_task.cancel()
            await asyncio.gather(self._server_task, return_exceptions=True)
            self._server_task = None

        self._log.info("web_dashboard_stopped")

    # ------------------------------------------------------------------
    # Route setup
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:
        """Register all HTTP and WebSocket routes."""
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Serve the main dashboard page."""
            template_path = _TEMPLATES_DIR / "index.html"
            if template_path.exists():
                return HTMLResponse(content=template_path.read_text())
            return HTMLResponse(content=self._fallback_html())

        @app.get("/api/strategies")
        async def get_strategies() -> dict:
            """Per-strategy stats: allocation, P&L, positions, denial counts."""
            return self._build_strategies_payload()

        @app.get("/api/regime")
        async def get_regime() -> dict:
            """Current regime state from the Conductor's last observed regime."""
            return {
                "regime": self._conductor._latest_regime,
                "confidence": str(self._conductor._latest_regime_confidence),
            }

        @app.get("/api/denials")
        async def get_denials() -> dict:
            """Guard denial counts per strategy and per rule."""
            return self._build_denials_payload()

        @app.get("/api/conductor")
        async def get_conductor() -> dict:
            """Conductor state: last allocations, mode, copilot pending."""
            return self._build_conductor_payload()

        @app.get("/api/equity_history")
        async def get_equity_history() -> dict:
            """Equity curve history for the Chart.js line chart."""
            return {"history": self._equity_history}

        # --- Phase 12F endpoints ---

        @app.get("/api/strategy_equity_history")
        async def get_strategy_equity_history() -> dict:
            """Per-strategy equity curves for multi-line Chart.js chart."""
            return {"strategies": self._strategy_equity_history}

        @app.get("/api/scorecard")
        async def get_scorecard() -> dict:
            """Go-live scorecard with criteria pass/fail evaluation."""
            return self._build_scorecard_payload()

        @app.get("/api/commission")
        async def get_commission() -> dict:
            """Per-strategy commission drag data."""
            return self._build_commission_payload()

        # --- Copilot routes ---

        @app.post("/api/copilot/approve")
        async def copilot_approve() -> dict:
            """Apply the pending Conductor allocation."""
            if not self._conductor.copilot_mode:
                return {"status": "error", "message": "Copilot mode is not enabled"}
            await self._conductor.approve_pending()
            await self._broadcast(
                {
                    "type": "conductor_reasoning",
                    "data": {
                        "action": "approved",
                        "reasoning": "Human approved pending allocation",
                        "allocations": {
                            k: str(v)
                            for k, v in self._conductor._last_allocations.items()
                        },
                    },
                }
            )
            return {"status": "ok", "message": "Pending allocation approved"}

        @app.post("/api/copilot/reject")
        async def copilot_reject() -> dict:
            """Discard the pending Conductor allocation."""
            if not self._conductor.copilot_mode:
                return {"status": "error", "message": "Copilot mode is not enabled"}
            await self._conductor.reject_pending()
            return {"status": "ok", "message": "Pending allocation rejected"}

        @app.get("/api/copilot/status")
        async def copilot_status() -> dict:
            """Current copilot state."""
            pending = self._conductor._pending_allocation
            return {
                "copilot_mode": self._conductor.copilot_mode,
                "has_pending": pending is not None,
                "pending_allocation": (
                    {k: str(v) for k, v in pending.items()} if pending else None
                ),
                "pending_reasoning": self._conductor._pending_reasoning,
            }

        @app.post("/api/copilot/toggle")
        async def copilot_toggle() -> dict:
            """Toggle copilot mode on/off."""
            self._conductor.copilot_mode = not self._conductor.copilot_mode
            return {"copilot_mode": self._conductor.copilot_mode}

        # --- WebSocket ---

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._ws_clients.add(websocket)
            self._log.info(
                "ws_client_connected", total_clients=len(self._ws_clients)
            )
            try:
                # Send current state on connect
                await websocket.send_json(
                    {"type": "init", "data": self._build_strategies_payload()}
                )
                # Keep alive — wait for disconnect
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._ws_clients.discard(websocket)
                self._log.info(
                    "ws_client_disconnected", total_clients=len(self._ws_clients)
                )

    # ------------------------------------------------------------------
    # Event handlers (subscribed to event bus)
    # ------------------------------------------------------------------

    async def _on_fill(self, event: Event) -> None:
        """Broadcast fill event and updated strategy stats to WS clients.

        Phase 12F: also snapshots per-strategy equity on every fill, and
        accumulates fill counts and commission totals for scorecard/commission
        panels (DEC-DASH-004).
        """
        if not isinstance(event, FillEvent):
            return

        ts = int(event.timestamp) if event.timestamp else int(time.time())
        strategy = event.strategy_id or "unknown"

        # Phase 12F: record first fill time for "days trading" criterion
        if self._first_fill_time is None:
            self._first_fill_time = float(event.timestamp) if event.timestamp else time.time()

        # Phase 12F: increment fill count per strategy
        self._fill_counts[strategy] = self._fill_counts.get(strategy, 0) + 1

        # Phase 12F: accumulate commission per strategy
        commission_usd = float(event.commission)
        self._commission_totals[strategy] = (
            self._commission_totals.get(strategy, 0.0) + commission_usd
        )

        # Phase 12F: snapshot per-strategy equity for equity curves
        for name in self._registry.active_strategy_names():
            portfolio = self._registry.get_portfolio(name)
            if portfolio:
                history = self._strategy_equity_history.setdefault(name, [])
                history.append({"ts": ts, "equity": float(portfolio.get_total_equity())})
                if len(history) > 500:
                    self._strategy_equity_history[name] = history[-500:]

        # Push the fill notification to WebSocket clients
        fill_data: dict[str, Any] = {
            "strategy": strategy,
            "symbol": event.symbol,
            "side": str(event.side),
            "amount": str(event.filled_amount),
            "price": str(event.fill_price),
        }
        await self._broadcast({"type": "fill", "data": fill_data})

        # Push updated strategy stats
        strategies = self._build_strategies_payload()
        await self._broadcast({"type": "strategy_update", "data": strategies})

        # Snapshot global equity for the global equity chart
        total_equity = float(self._global_portfolio.get_total_equity())
        self._equity_history.append({"ts": ts, "equity": total_equity})
        if len(self._equity_history) > 500:
            self._equity_history = self._equity_history[-500:]

    async def _on_regime_change(self, event: Event) -> None:
        """Broadcast regime change to WS clients."""
        if not isinstance(event, RegimeChangeEvent):
            return

        await self._broadcast(
            {
                "type": "regime_change",
                "data": {
                    "symbol": (event.indicators or {}).get("symbol", ""),
                    "from": event.from_regime,
                    "to": event.to_regime,
                    "confidence": str(event.confidence),
                },
            }
        )

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_strategies_payload(self) -> dict:
        """Build per-strategy stats dict for the /api/strategies endpoint."""
        strategies: dict[str, Any] = {}
        total_alloc = sum(
            v for v in self._conductor._last_allocations.values()
        ) or Decimal("100")

        for name in self._registry.active_strategy_names():
            portfolio = self._registry.get_portfolio(name)
            risk_mgr = self._registry.get_risk_manager(name)

            equity = float(portfolio.get_total_equity()) if portfolio else 0.0
            cash = float(portfolio.get_cash_balance()) if portfolio else 0.0

            alloc_pct = float(
                self._conductor._last_allocations.get(name, Decimal("0"))
            )

            # Denial counts from RiskManager
            denials: dict[str, int] = {}
            if risk_mgr and hasattr(risk_mgr, "denial_counts"):
                denials = {k: v for k, v in risk_mgr.denial_counts.items()}

            strategies[name] = {
                "allocation_pct": alloc_pct,
                "equity": equity,
                "cash": cash,
                "pnl": equity - cash,  # unrealised + realised simplified view
                "denials": denials,
            }

        global_equity = float(self._global_portfolio.get_total_equity())
        global_drawdown = float(self._global_portfolio.get_total_drawdown())

        return {
            "strategies": strategies,
            "global_equity": global_equity,
            "global_drawdown_pct": global_drawdown,
        }

    def _build_denials_payload(self) -> dict:
        """Build denial counts per strategy per rule."""
        result: dict[str, Any] = {}
        for name in self._registry.active_strategy_names():
            risk_mgr = self._registry.get_risk_manager(name)
            if risk_mgr and hasattr(risk_mgr, "denial_counts"):
                result[name] = dict(risk_mgr.denial_counts)
            else:
                result[name] = {}
        return {"denials": result}

    def _build_conductor_payload(self) -> dict:
        """Build Conductor state payload."""
        return {
            "last_allocations": {
                k: str(v) for k, v in self._conductor._last_allocations.items()
            },
            "latest_regime": self._conductor._latest_regime,
            "regime_confidence": str(self._conductor._latest_regime_confidence),
            "llm_enabled": self._conductor._llm_enabled,
            "copilot_mode": self._conductor.copilot_mode,
            "has_pending": self._conductor._pending_allocation is not None,
            "pending_reasoning": self._conductor._pending_reasoning,
        }

    def _build_scorecard_payload(self) -> dict:
        """
        Build go-live scorecard payload (DEC-DASH-004).

        Evaluates computable criteria inline from FillEvent-tracked state.
        Criteria requiring full trade history (Sharpe) are noted as requiring
        scripts/analyze.py for authoritative evaluation.

        Returns:
            dict with:
                criteria   — list of {name, target, current, pass} dicts
                verdict    — "GO" / "NO-GO" / "INSUFFICIENT"
                kill_alerts — list of breach messages (empty if none)
        """
        now = time.time()

        # Days trading since first observed fill
        days_trading = (
            (now - self._first_fill_time) / 86400.0
            if self._first_fill_time is not None
            else 0.0
        )

        # Net P&L: sum realized P&L across all strategies from PortfolioTracker
        net_pnl = 0.0
        per_strategy_realized: dict[str, float] = {}
        for name in self._registry.active_strategy_names():
            portfolio = self._registry.get_portfolio(name)
            if portfolio and hasattr(portfolio, "_total_realized_pnl"):
                rpnl = float(portfolio._total_realized_pnl)
                per_strategy_realized[name] = rpnl
                net_pnl += rpnl

        # Max drawdown from global portfolio
        max_drawdown = float(self._global_portfolio.get_total_drawdown())

        # Commission drag: commission / (commission + net_pnl) * 100
        total_commission = sum(self._commission_totals.values())
        gross_pnl_est = net_pnl + total_commission
        if gross_pnl_est > 0.0:
            commission_drag = (total_commission / gross_pnl_est) * 100.0
        elif total_commission > 0.0:
            commission_drag = 100.0  # commissions exceed gains
        else:
            commission_drag = 0.0

        # Total fills and active strategy count
        total_fills = sum(self._fill_counts.values())
        strategy_count = float(max(
            len(self._registry.active_strategy_names()),
            sum(1 for n in self._registry.active_strategy_names()
                if self._fill_counts.get(n, 0) > 0),
        ))

        # Single-strategy P&L concentration (only meaningful when profitable)
        if per_strategy_realized and net_pnl > 0.0:
            max_single = max(per_strategy_realized.values())
            concentration = (max_single / net_pnl) * 100.0
        else:
            concentration = 0.0

        current_values: dict[str, float] = {
            "days_trading":    days_trading,
            "net_pnl":         net_pnl,
            "max_drawdown":    max_drawdown,
            "commission_drag": commission_drag,
            "fill_count":      float(total_fills),
            "strategy_count":  strategy_count,
            "concentration":   concentration,
        }

        # Evaluate each criterion
        criteria = []
        all_pass = True
        any_data = total_fills > 0 or days_trading > 0.0

        for key, label, target_desc, target_val, comparison in _SCORECARD_CRITERIA:
            current = current_values[key]
            passed = current >= target_val if comparison == "gte" else current <= target_val
            if not passed:
                all_pass = False
            criteria.append({
                "name": label,
                "target": target_desc,
                "current": f"{current:.1f}",
                "pass": passed,
            })

        # Sharpe cannot be computed inline — requires trade DB via analyze.py
        criteria.append({
            "name": "Sharpe ratio (per strategy)",
            "target": ">= 0.5",
            "current": "N/A — run analyze.py",
            "pass": None,
        })

        if not any_data:
            verdict = "INSUFFICIENT"
        elif all_pass:
            verdict = "GO"
        else:
            verdict = "NO-GO"

        # Kill alerts for breach of hard limits
        kill_alerts: list[str] = []
        if max_drawdown > _KILL_DRAWDOWN_PCT:
            kill_alerts.append(
                f"DRAWDOWN: {max_drawdown:.1f}% exceeds {_KILL_DRAWDOWN_PCT}% kill threshold"
            )
        if net_pnl < _KILL_NET_PNL:
            kill_alerts.append(
                f"NET P&L: ${net_pnl:.2f} below ${_KILL_NET_PNL:.0f} kill threshold"
            )

        return {
            "criteria": criteria,
            "verdict": verdict,
            "kill_alerts": kill_alerts,
        }

    def _build_commission_payload(self) -> dict:
        """
        Build per-strategy commission drag data (DEC-DASH-004).

        Gross P&L is estimated as net_realized_pnl + commission (i.e. what
        P&L would be without any commission charges).

        Returns:
            dict with:
                strategies — per-strategy {gross_pnl, commission, net_pnl, drag_pct}
                total      — aggregate totals across all strategies
        """
        strategies: dict[str, Any] = {}
        total_gross = 0.0
        total_commission = 0.0

        for name in self._registry.active_strategy_names():
            portfolio = self._registry.get_portfolio(name)
            commission = self._commission_totals.get(name, 0.0)

            net_pnl = 0.0
            if portfolio and hasattr(portfolio, "_total_realized_pnl"):
                net_pnl = float(portfolio._total_realized_pnl)

            gross_pnl = net_pnl + commission
            if gross_pnl > 0.0:
                drag_pct = (commission / gross_pnl) * 100.0
            elif commission > 0.0:
                drag_pct = 100.0
            else:
                drag_pct = 0.0

            strategies[name] = {
                "gross_pnl":  round(gross_pnl, 4),
                "commission": round(commission, 4),
                "net_pnl":    round(net_pnl, 4),
                "drag_pct":   round(drag_pct, 2),
            }

            total_gross += gross_pnl
            total_commission += commission

        total_net = total_gross - total_commission
        if total_gross > 0.0:
            total_drag = (total_commission / total_gross) * 100.0
        elif total_commission > 0.0:
            total_drag = 100.0
        else:
            total_drag = 0.0

        return {
            "strategies": strategies,
            "total": {
                "gross_pnl":  round(total_gross, 4),
                "commission": round(total_commission, 4),
                "net_pnl":    round(total_net, 4),
                "drag_pct":   round(total_drag, 2),
            },
        }

    # ------------------------------------------------------------------
    # WebSocket broadcast
    # ------------------------------------------------------------------

    async def _broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected WebSocket clients."""
        if not self._ws_clients:
            return
        dead: set[WebSocket] = set()
        payload = json.dumps(message)
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    # ------------------------------------------------------------------
    # Fallback HTML (when templates dir missing)
    # ------------------------------------------------------------------

    def _fallback_html(self) -> str:
        """Minimal fallback page shown when templates/index.html is missing."""
        return (
            "<html><body style='background:#111;color:#0f0;font-family:monospace'>"
            "<h1>CerebrumCoin Dashboard</h1>"
            "<p>Template not found. Place templates/index.html in the dashboard package.</p>"
            "</body></html>"
        )
