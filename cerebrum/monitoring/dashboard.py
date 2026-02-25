"""
Real-time monitoring dashboard for trading system.

Subscribes to events and displays live statistics every N seconds.

@decision DEC-MONITOR-002
@title Event-driven dashboard with periodic updates
@status accepted
@rationale Dashboard subscribes to FillEvent, PositionUpdateEvent, and TradeClosedEvent
to maintain real-time state. Uses asyncio.sleep for periodic console updates (default 30s).
Non-blocking design ensures dashboard doesn't slow down trading decisions.
"""

import asyncio
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, PositionUpdateEvent, TradeClosedEvent
from cerebrum.core.state import StateManager
from cerebrum.core.types import EventType
from cerebrum.monitoring.stats import calculate_performance_metrics

logger = structlog.get_logger()


class Dashboard:
    """
    Real-time monitoring dashboard.
    
    Features:
    - Portfolio equity and P&L
    - Open positions
    - Recent trades
    - Performance metrics (updated periodically)
    """
    
    def __init__(
        self,
        bus: EventBus,
        state_manager: StateManager,
        update_interval_seconds: int = 30,
        initial_balance: Decimal = Decimal("10000.0"),
    ) -> None:
        """
        Initialize dashboard.
        
        Args:
            bus: Event bus
            state_manager: State manager for trade history
            update_interval_seconds: How often to update display
            initial_balance: Starting balance for metrics calculation
        """
        self._bus = bus
        self._state_manager = state_manager
        self._update_interval = update_interval_seconds
        self._initial_balance = initial_balance
        self._running = False
        self._update_task: asyncio.Task | None = None
        
        # Current state (updated from events)
        self._current_equity = initial_balance
        self._total_pnl = Decimal("0.0")
        self._open_positions: dict = {}
        self._recent_trades: list = []
        
        self._log = logger.bind(component="dashboard")
    
    async def start(self) -> None:
        """Start the dashboard."""
        self._running = True
        
        # Subscribe to relevant events
        self._bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="dashboard_fill",
        )
        self._bus.subscribe(
            EventType.POSITION_UPDATE,
            self._on_position_update,
            subscriber_name="dashboard_position",
        )
        self._bus.subscribe(
            EventType.TRADE_CLOSED,
            self._on_trade_closed,
            subscriber_name="dashboard_trade",
        )
        
        # Start periodic update task
        self._update_task = asyncio.create_task(self._update_loop())
        
        self._log.info("dashboard_started", update_interval=self._update_interval)
    
    async def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        self._log.info("dashboard_stopped")
    
    async def _on_fill(self, event: Event) -> None:
        """Handle fill events."""
        if isinstance(event, FillEvent):
            pass  # Portfolio tracker handles equity updates
    
    async def _on_position_update(self, event: Event) -> None:
        """Handle position update events.

        amount=0 signals that the position was closed — remove it from the
        open-positions dict so the dashboard shows accurate state.
        """
        if isinstance(event, PositionUpdateEvent):
            if event.amount == 0:
                # Position closed: remove from tracking dict
                self._open_positions.pop(event.symbol, None)
            else:
                self._open_positions[event.symbol] = {
                    "amount": event.amount,
                    "entry_price": event.average_entry_price,
                    "current_price": event.current_price,
                    "unrealized_pnl": event.unrealized_pnl,
                }
    
    async def _on_trade_closed(self, event: Event) -> None:
        """Handle trade closed events."""
        if isinstance(event, TradeClosedEvent):
            self._recent_trades.insert(0, {
                "symbol": event.symbol,
                "side": event.side,
                "pnl": event.pnl,
                "exit_time": event.exit_time,
            })
            # Keep only last 10 trades
            self._recent_trades = self._recent_trades[:10]
    
    async def _update_loop(self) -> None:
        """Periodic update loop."""
        while self._running:
            try:
                await asyncio.sleep(self._update_interval)
                await self._display_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error("dashboard_update_error", error=str(e))
    
    async def _display_stats(self) -> None:
        """Display current statistics."""
        # Get closed trades for metrics
        closed_trades = await self._state_manager.get_closed_trades(limit=100)
        
        # Calculate metrics
        metrics = calculate_performance_metrics(
            closed_trades,
            initial_balance=self._initial_balance,
        )
        
        # Display header
        print("\n" + "=" * 80)
        print(" CerebrumCoin Dashboard ".center(80, "="))
        print("=" * 80)
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        
        # Open positions
        if self._open_positions:
            print(f"\nOpen Positions ({len(self._open_positions)}):")
            for symbol, pos in self._open_positions.items():
                print(f"  {symbol}: {pos['amount']} @ ${pos['entry_price']} "
                      f"(Unrealized PnL: ${pos['unrealized_pnl']:.2f})")
        else:
            print("\nOpen Positions: None")
        
        # Recent trades
        if self._recent_trades:
            print(f"\nRecent Trades ({len(self._recent_trades)}):")
            for trade in self._recent_trades[:5]:
                print(f"  {trade['symbol']} {trade['side'].value}: PnL ${trade['pnl']:.2f}")
        
        print("=" * 80 + "\n")
