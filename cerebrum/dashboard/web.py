"""
FastAPI + WebSocket web dashboard for CerebrumCoin multi-strategy visualization.

Provides a live trading terminal UI with:
- Per-strategy capital allocation bars (animated, color-coded)
- Real-time regime state badge
- P&L panel with per-strategy breakdowns
- Guard denial counters
- Conductor reasoning log
- Copilot mode: human-in-the-loop approval for allocation changes

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

        # Equity history for chart (capped at 500 points)
        self._equity_history: list[dict[str, Any]] = []

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
        """Broadcast fill event and updated strategy stats to WS clients."""
        if not isinstance(event, FillEvent):
            return

        # Push the fill notification
        fill_data: dict[str, Any] = {
            "strategy": event.strategy_id or "unknown",
            "symbol": event.symbol,
            "side": str(event.side),
            "amount": str(event.filled_amount),
            "price": str(event.fill_price),
        }
        await self._broadcast({"type": "fill", "data": fill_data})

        # Push updated strategy stats
        strategies = self._build_strategies_payload()
        await self._broadcast({"type": "strategy_update", "data": strategies})

        # Snapshot equity for chart
        total_equity = float(self._global_portfolio.get_total_equity())
        self._equity_history.append(
            {"ts": int(time.time()), "equity": total_equity}
        )
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
