"""
Exit monitor for automatic position management.

Proactively watches open positions and emits exit orders when stop-loss,
take-profit, or time-based criteria are met.

@decision DEC-EXIT-001
@title ExitMonitor as separate component from RiskManager
@status accepted
@rationale RiskRules evaluate proposed orders (reactive). ExitMonitor watches
open positions and proactively generates exit orders when exit criteria are
triggered. Separation of concerns: rules say yes/no to inbound signals, the
monitor independently protects capital by closing stale or losing positions.
This component subscribes to MARKET_DATA (not SIGNAL) so it fires on every
price tick rather than waiting for the signal pipeline to produce a SELL.

@decision DEC-EXIT-002
@title Adaptive take-profit based on recent price range
@status accepted
@rationale Session 5 showed 0/17 wins because the fixed 3% take-profit is
unreachable when intraday range is <0.5%. All positions hit 120-min max-age
timeout at a guaranteed loss. Adaptive TP scales the take-profit target to
actual market conditions: effective_tp = max(min_tp, range_pct * tp_multiplier).
In a 0.4% range market, effective_tp = max(0.3%, 0.4% * 1.5) = 0.6% — still
reachable while covering commission. In a 2% range market, effective_tp = 3.0%
(the fixed default). The min_tp floor ensures we never set TP below commission
cost. Backward-compatible: adaptive_tp=False keeps the fixed behaviour.

@decision DEC-EXIT-003
@title ExitMonitor carries strategy_id and tags emitted OrderEvents
@status accepted
@rationale In multi-strategy mode, PortfolioTracker filters fills by
strategy_id. If ExitMonitor emits a SELL OrderEvent without strategy_id, the
paper adapter propagates a FillEvent with strategy_id=None, which bypasses the
per-strategy portfolio. The position in the strategy's portfolio never
decreases, causing the exit monitor to re-fire on every subsequent tick
(infinite exit loop). Passing strategy_id=None (default) preserves backward
compatibility with the single-strategy path in main.py, where every fill is
accepted regardless of strategy tag.

@decision DEC-EXIT-004
@title _on_fill clears pending_exits only when position is actually gone
@status accepted
@rationale The original implementation cleared pending_exits on any SELL fill,
even partial fills or fills for a different strategy's order on the same symbol.
The correct invariant is: the pending flag should stay set until the portfolio
confirms the position has been fully closed (amount < 0.0001). This prevents a
second exit order being emitted between the fill arriving and the portfolio
processing it, while still allowing the exit monitor to re-arm once the
position genuinely reaches zero.
"""

import time
from collections import deque
from decimal import Decimal
from uuid import uuid4

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, MarketDataEvent, OrderEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side, Symbol
from cerebrum.risk.portfolio import PortfolioTracker

logger = structlog.get_logger()


class ExitMonitor:
    """
    Monitors open positions and emits exit orders on breach of exit criteria.

    Exit criteria (checked on every MARKET_DATA tick for the position's symbol):
    - Stop-loss: unrealized loss exceeds stop_loss_percent of position value
    - Take-profit: unrealized gain exceeds take_profit_percent (or adaptive TP) of position value
    - Time-based: position age exceeds max_position_age_minutes

    When a criterion is breached, a SELL OrderEvent (MARKET type) is published
    directly to the bus. The risk manager and paper/live adapter will process it
    through the normal order flow.

    Only long positions (amount > 0) are handled — the system currently only
    takes long positions via BUY signals.

    Adaptive take-profit (DEC-EXIT-002): when adaptive_tp=True, the effective
    take-profit is computed as max(min_tp_percent, range_pct * tp_multiplier)
    where range_pct is the price range over the last tp_window_size ticks.
    This prevents the 3% TP from being unreachable in low-vol sessions.
    """

    def __init__(
        self,
        bus: EventBus,
        portfolio: PortfolioTracker,
        stop_loss_percent: Decimal = Decimal("2.0"),
        take_profit_percent: Decimal = Decimal("3.0"),
        max_position_age_minutes: int = 120,
        adaptive_tp: bool = False,
        tp_multiplier: Decimal = Decimal("1.5"),
        min_tp_percent: Decimal = Decimal("0.3"),
        tp_window_size: int = 18000,
        strategy_id: str | None = None,
    ) -> None:
        """
        Initialize exit monitor.

        Args:
            bus: Event bus for subscribing to market data and publishing orders
            portfolio: Portfolio tracker providing current position state
            stop_loss_percent: Close position if loss exceeds this % of entry value
            take_profit_percent: Close position if gain exceeds this % of entry value
                                 (used as fixed TP when adaptive_tp=False, and as
                                 the upper bound when adaptive_tp=True)
            max_position_age_minutes: Close position if open longer than this many minutes
            adaptive_tp: When True, compute effective TP from recent price range.
                         When False (default), use fixed take_profit_percent.
            tp_multiplier: Scale factor applied to range_pct for adaptive TP.
                           effective_tp = max(min_tp_percent, range_pct * tp_multiplier)
            min_tp_percent: Floor for adaptive TP — never target less than this %.
                            Should exceed round-trip commission cost (~0.32%).
            tp_window_size: Number of recent price ticks to use for adaptive TP
                            range calculation. Default ~18000 = 5 hours at 1 tick/sec.
            strategy_id: Optional strategy identifier. When set, all emitted
                         OrderEvents are tagged with this strategy_id so that
                         per-strategy PortfolioTrackers correctly attribute fills
                         (DEC-EXIT-003). None (default) = accept all fills,
                         backward-compatible with single-strategy mode.
        """
        self._bus = bus
        self._portfolio = portfolio
        self._stop_loss_pct = stop_loss_percent
        self._take_profit_pct = take_profit_percent
        self._max_age_seconds = max_position_age_minutes * 60
        # DEC-EXIT-003: tag emitted orders so per-strategy portfolios route fills correctly
        self._strategy_id = strategy_id

        # Adaptive TP parameters (DEC-EXIT-002)
        self._adaptive_tp = adaptive_tp
        self._tp_multiplier = tp_multiplier
        self._min_tp_percent = min_tp_percent
        self._tp_window_size = tp_window_size
        # Per-symbol rolling price windows for adaptive TP range calculation
        self._tp_price_windows: dict[Symbol, deque] = {}

        # Track symbols we've already triggered an exit for to avoid duplicate orders
        self._pending_exits: set[Symbol] = set()

        self._log = logger.bind(component="exit_monitor")

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="exit_monitor",
        )

        # Also listen for fills to clear pending_exits when position is closed
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="exit_monitor_fills",
        )

        self._log.info(
            "exit_monitor_initialized",
            stop_loss_pct=str(stop_loss_percent),
            take_profit_pct=str(take_profit_percent),
            max_age_minutes=max_position_age_minutes,
            adaptive_tp=adaptive_tp,
            tp_multiplier=str(tp_multiplier) if adaptive_tp else "n/a",
            min_tp_percent=str(min_tp_percent) if adaptive_tp else "n/a",
            tp_window_size=tp_window_size if adaptive_tp else "n/a",
        )

    def _compute_effective_tp(self, symbol: Symbol) -> Decimal:
        """
        Compute the effective take-profit percentage for the given symbol.

        When adaptive_tp=False: returns the fixed take_profit_percent.
        When adaptive_tp=True: returns max(min_tp_percent, range_pct * tp_multiplier)
        where range_pct is computed from the rolling price window. Falls back to
        take_profit_percent if the window is not yet full (cold start).

        Args:
            symbol: Trading symbol to compute TP for.

        Returns:
            Effective take-profit percentage as a Decimal.
        """
        if not self._adaptive_tp:
            return self._take_profit_pct

        window = self._tp_price_windows.get(symbol)
        if window is None or len(window) < self._tp_window_size:
            # Cold start — use the configured fixed TP as fallback
            return self._take_profit_pct

        price_min = min(window)
        price_max = max(window)
        if price_min == Decimal("0"):
            return self._take_profit_pct

        range_pct = (price_max - price_min) / price_min * Decimal("100")
        adaptive = range_pct * self._tp_multiplier
        effective = max(self._min_tp_percent, adaptive)

        self._log.debug(
            "adaptive_tp_computed",
            symbol=symbol,
            range_pct=float(range_pct),
            adaptive=float(adaptive),
            effective=float(effective),
        )
        return effective

    async def _on_fill(self, event: Event) -> None:
        """Clear pending exit flag only when the position is fully closed.

        DEC-EXIT-004: Clearing on any SELL fill was too eager — partial fills
        and fills routed to other strategies (strategy_id mismatch) would
        prematurely re-arm the exit monitor, causing a second exit order to be
        emitted before the portfolio had processed the fill. We now only clear
        pending_exits when the portfolio confirms the position amount is gone.
        """
        from cerebrum.core.events import FillEvent
        if not isinstance(event, FillEvent):
            return
        if event.side != Side.SELL:
            return
        symbol = event.symbol
        if symbol not in self._pending_exits:
            return
        # Only clear the pending flag once the position is actually gone
        pos = self._portfolio.get_position(symbol)
        if pos is None or abs(pos.amount) < Decimal("0.0001"):
            self._pending_exits.discard(symbol)
            self._log.debug("exit_pending_cleared", symbol=symbol)

    async def _on_market_data(self, event: Event) -> None:
        """Check exit criteria for the position in the updated symbol."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        current_price = event.price
        current_time = event.timestamp

        # Update the adaptive TP price window if adaptive mode is enabled
        if self._adaptive_tp:
            if symbol not in self._tp_price_windows:
                self._tp_price_windows[symbol] = deque(maxlen=self._tp_window_size)
            self._tp_price_windows[symbol].append(current_price)

        # Skip if we already have a pending exit order for this symbol
        if symbol in self._pending_exits:
            return

        pos = self._portfolio.get_position(symbol)
        if pos is None:
            return

        # Only handle long positions (amount > 0)
        if pos.amount <= Decimal("0"):
            return

        exit_reason: str | None = None
        order_type = OrderType.MARKET

        # --- Stop-loss check ---
        # Loss% = (entry - current) / entry * 100  (positive = loss for long)
        if pos.average_entry_price > Decimal("0"):
            loss_pct = (
                (pos.average_entry_price - current_price)
                / pos.average_entry_price
                * Decimal("100")
            )
            if loss_pct >= self._stop_loss_pct:
                exit_reason = (
                    f"stop_loss: loss {loss_pct:.2f}% >= threshold {self._stop_loss_pct}%"
                )
                order_type = OrderType.STOP_LOSS

        # --- Take-profit check ---
        if exit_reason is None and pos.average_entry_price > Decimal("0"):
            gain_pct = (
                (current_price - pos.average_entry_price)
                / pos.average_entry_price
                * Decimal("100")
            )
            effective_tp = self._compute_effective_tp(symbol)
            if gain_pct >= effective_tp:
                exit_reason = (
                    f"take_profit: gain {gain_pct:.2f}% >= threshold {effective_tp}%"
                )
                order_type = OrderType.TAKE_PROFIT

        # --- Time-based check ---
        if exit_reason is None and self._max_age_seconds > 0:
            age_seconds = current_time - pos.entry_time
            if age_seconds >= self._max_age_seconds:
                age_minutes = age_seconds / 60
                exit_reason = (
                    f"time_exit: age {age_minutes:.1f}min >= max {self._max_age_seconds / 60:.0f}min"
                )

        if exit_reason is None:
            return

        # Emit a SELL market order to close the position
        # DEC-EXIT-003: tag with strategy_id so per-strategy portfolios route the fill
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=current_time,
            order_id=str(uuid4()),
            symbol=symbol,
            side=Side.SELL,
            order_type=order_type,
            amount=abs(pos.amount),
            status=OrderStatus.PENDING,
            metadata={"exit_reason": exit_reason, "source": "exit_monitor"},
            strategy_id=self._strategy_id,
        )

        self._pending_exits.add(symbol)
        self._log.info(
            "exit_order_emitted",
            symbol=symbol,
            reason=exit_reason,
            amount=str(pos.amount),
            price=str(current_price),
            entry_price=str(pos.average_entry_price),
        )

        await self._bus.publish(order)
