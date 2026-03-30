"""
Range exit monitor for structural S/R-based exits.

Replaces percentage-based take-profit/stop-loss with structural exits:
1. Resistance exit: sell when price reaches resistance level
2. Support breakdown: sell when price breaks below support
3. Regime invalidation: sell all when regime leaves SIDEWAYS
4. Time-based safety: sell after max_hold_minutes

@decision DEC-RANGE-004
@title Structural exits over percentage-based for range trading
@status accepted
@rationale Fixed % TP is unreachable in tight ranges (Session 5: 0/17).
Structural exits at S/R levels match actual market structure — sell at the
top of the range, stop at the bottom. Time-based safety catches positions
stuck mid-range.

@decision DEC-RANGE-005
@title Fallback to percentage-based exits when no confirmed range exists
@status accepted
@rationale After regime change or when S/R levels haven't accumulated enough
bounces, there is no structural reference frame. Rather than holding positions
with no exit criteria, we fall back to fixed-percentage TP/SL (defaults 1.0%
TP / 0.8% SL) which are narrow enough to be reachable in the low-volatility
markets where range trading operates.

@decision DEC-RANGE-007
@title RangeExitMonitor carries strategy_id (same rationale as DEC-EXIT-003)
@status accepted
@rationale Mirrors the fix in ExitMonitor: without strategy_id on emitted
OrderEvents, fills bypass the per-strategy PortfolioTracker routing, leaving
positions open and causing an infinite re-fire loop on every market tick.
_on_fill also uses the position-amount guard (DEC-EXIT-004) to prevent
premature pending_exits clearance on partial fills.
"""

import time
from decimal import Decimal
from uuid import uuid4

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, MarketDataEvent, OrderEvent, RegimeChangeEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side, Symbol
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.range_detector import RangeDetector

logger = structlog.get_logger()


class RangeExitMonitor:
    """
    Monitors open positions and emits structural S/R-based exit orders.

    Exit criteria checked on every MARKET_DATA tick for each open position:

    With confirmed range (range_detector.get_range() returns a confirmed RangeState):
    - Resistance exit: sell when price >= resistance * (1 - proximity_pct/100)
      Captures profit at the top of the range before price turns.
    - Support breakdown: sell when price <= support * (1 - breakdown_margin_pct/100)
      Cuts loss quickly when price breaks through the floor.

    Without confirmed range (None or range_confirmed=False):
    - Fallback TP: sell when gain_pct >= fallback_tp_pct
    - Fallback SL: sell when loss_pct >= fallback_sl_pct

    Always active:
    - Time-based: sell when position age > max_hold_minutes * 60
    - Regime invalidation: sell ALL positions when REGIME_CHANGE leaves SIDEWAYS

    A pending exit set prevents duplicate SELL orders per symbol. The flag is
    cleared when a SELL FillEvent arrives, confirming the position is closing.
    """

    def __init__(
        self,
        bus: EventBus,
        portfolio: PortfolioTracker,
        range_detector: RangeDetector,
        resistance_proximity_pct: Decimal = Decimal("0.3"),
        breakdown_margin_pct: Decimal = Decimal("0.5"),
        max_hold_minutes: int = 60,
        fallback_tp_pct: Decimal = Decimal("1.0"),
        fallback_sl_pct: Decimal = Decimal("0.8"),
        strategy_id: str | None = None,
    ) -> None:
        """
        Initialize RangeExitMonitor.

        Args:
            bus: Event bus for subscribing to events and publishing orders.
            portfolio: Portfolio tracker providing current position state.
            range_detector: RangeDetector to query for current S/R levels.
            resistance_proximity_pct: Distance (%) below resistance that triggers
                a resistance exit. At 0.3%, price at 99.7% of resistance level
                triggers the sell — captures profit before the exact top.
            breakdown_margin_pct: Distance (%) below support that triggers a
                support breakdown exit. At 0.5%, price must fall 0.5% below
                support to confirm breakdown (avoids false triggers).
            max_hold_minutes: Maximum minutes to hold any position before
                forced time-based exit.
            fallback_tp_pct: Take-profit percentage when no confirmed range
                exists. Should be reachable in low-vol markets (< 1%).
            fallback_sl_pct: Stop-loss percentage when no confirmed range
                exists. Tight to limit losses on failed range entries.
            strategy_id: Optional strategy identifier. When set, all emitted
                         OrderEvents are tagged so per-strategy PortfolioTrackers
                         route fills correctly (DEC-RANGE-007 / DEC-EXIT-003).
                         None (default) = backward-compatible single-strategy mode.
        """
        self._bus = bus
        self._portfolio = portfolio
        self._range_detector = range_detector
        self._resistance_proximity_pct = resistance_proximity_pct
        self._breakdown_margin_pct = breakdown_margin_pct
        self._max_hold_seconds = max_hold_minutes * 60
        self._fallback_tp_pct = fallback_tp_pct
        self._fallback_sl_pct = fallback_sl_pct
        # DEC-RANGE-007: tag emitted orders so per-strategy portfolios route fills correctly
        self._strategy_id = strategy_id

        # Symbols for which a SELL order is already in-flight (dedup guard)
        self._pending_exits: set[Symbol] = set()

        self._log = logger.bind(component="range_exit_monitor")

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="range_exit_monitor_market_data",
        )
        bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            subscriber_name="range_exit_monitor_regime",
        )
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="range_exit_monitor_fills",
        )

        self._log.info(
            "range_exit_monitor_initialized",
            resistance_proximity_pct=str(resistance_proximity_pct),
            breakdown_margin_pct=str(breakdown_margin_pct),
            max_hold_minutes=max_hold_minutes,
            fallback_tp_pct=str(fallback_tp_pct),
            fallback_sl_pct=str(fallback_sl_pct),
        )

    async def _on_fill(self, event: Event) -> None:
        """Clear pending exit flag only when the position is fully closed.

        DEC-EXIT-004 / DEC-RANGE-007: Mirror the ExitMonitor fix. Clearing on
        any SELL fill was too eager — we check the portfolio position amount
        before clearing so that partial fills or cross-strategy fills don't
        prematurely re-arm the exit monitor.
        """
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
            self._log.debug("range_exit_pending_cleared", symbol=symbol)

    async def _on_regime_change(self, event: Event) -> None:
        """Emit SELL for all open positions when regime leaves SIDEWAYS."""
        if not isinstance(event, RegimeChangeEvent):
            return

        if event.from_regime != "SIDEWAYS":
            return

        positions = self._portfolio.get_all_positions()
        if not positions:
            return

        self._log.info(
            "range_exit_regime_invalidation",
            from_regime=event.from_regime,
            to_regime=event.to_regime,
            open_positions=list(positions.keys()),
        )

        for symbol, pos in positions.items():
            if symbol in self._pending_exits:
                continue
            if pos.amount <= Decimal("0"):
                continue
            await self._emit_sell(
                symbol=symbol,
                amount=pos.amount,
                current_price=pos.current_price,
                timestamp=event.timestamp,
                reason=f"regime_invalidation: {event.from_regime}→{event.to_regime}",
            )

    async def _on_market_data(self, event: Event) -> None:
        """Check exit criteria for the position in the updated symbol."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        price = event.price
        ts = event.timestamp

        # Skip if we already have a pending exit for this symbol
        if symbol in self._pending_exits:
            return

        pos = self._portfolio.get_position(symbol)
        if pos is None:
            return

        # Only handle long positions
        if pos.amount <= Decimal("0"):
            return

        exit_reason: str | None = None

        # --- Structural S/R exits (when a confirmed range exists) ---
        range_state = self._range_detector.get_range(symbol)
        if range_state is not None and range_state.range_confirmed:
            resistance = range_state.resistance_level
            support = range_state.support_level

            # Resistance exit: price near or above resistance ceiling
            # Trigger when price >= resistance * (1 - proximity_pct / 100)
            if resistance > Decimal("0"):
                resistance_threshold = resistance * (
                    Decimal("1") - self._resistance_proximity_pct / Decimal("100")
                )
                if price >= resistance_threshold:
                    exit_reason = (
                        f"resistance_exit: price {price} >= threshold {resistance_threshold:.4f} "
                        f"(resistance {resistance}, proximity {self._resistance_proximity_pct}%)"
                    )

            # Support breakdown: price has fallen below support floor
            # Trigger when price <= support * (1 - breakdown_margin_pct / 100)
            if exit_reason is None and support > Decimal("0"):
                breakdown_threshold = support * (
                    Decimal("1") - self._breakdown_margin_pct / Decimal("100")
                )
                if price <= breakdown_threshold:
                    exit_reason = (
                        f"support_breakdown: price {price} <= threshold {breakdown_threshold:.4f} "
                        f"(support {support}, margin {self._breakdown_margin_pct}%)"
                    )

        else:
            # --- Fallback percentage-based exits ---
            if pos.average_entry_price > Decimal("0"):
                gain_pct = (
                    (price - pos.average_entry_price)
                    / pos.average_entry_price
                    * Decimal("100")
                )
                loss_pct = (
                    (pos.average_entry_price - price)
                    / pos.average_entry_price
                    * Decimal("100")
                )

                if gain_pct >= self._fallback_tp_pct:
                    exit_reason = (
                        f"fallback_take_profit: gain {gain_pct:.2f}% >= {self._fallback_tp_pct}%"
                    )
                elif loss_pct >= self._fallback_sl_pct:
                    exit_reason = (
                        f"fallback_stop_loss: loss {loss_pct:.2f}% >= {self._fallback_sl_pct}%"
                    )

        # --- Time-based exit (always applies regardless of range state) ---
        if exit_reason is None and self._max_hold_seconds > 0:
            age_seconds = ts - pos.entry_time
            if age_seconds >= self._max_hold_seconds:
                age_minutes = age_seconds / 60
                exit_reason = (
                    f"time_exit: age {age_minutes:.1f}min >= max {self._max_hold_seconds / 60:.0f}min"
                )

        if exit_reason is None:
            return

        await self._emit_sell(
            symbol=symbol,
            amount=pos.amount,
            current_price=price,
            timestamp=ts,
            reason=exit_reason,
        )

    async def _emit_sell(
        self,
        symbol: Symbol,
        amount: Decimal,
        current_price: Decimal,
        timestamp: float,
        reason: str,
    ) -> None:
        """
        Publish a SELL MARKET OrderEvent and mark the symbol as pending exit.

        Args:
            symbol: Symbol to sell.
            amount: Absolute position size to sell.
            current_price: Current market price (for logging).
            timestamp: Event timestamp.
            reason: Human-readable exit reason stored in order metadata.
        """
        # DEC-RANGE-007: tag with strategy_id so per-strategy portfolios route the fill
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=timestamp,
            order_id=str(uuid4()),
            symbol=symbol,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            amount=abs(amount),
            price=None,
            status=OrderStatus.PENDING,
            metadata={"exit_reason": reason, "source": "range_exit_monitor"},
            strategy_id=self._strategy_id,
        )

        self._pending_exits.add(symbol)
        self._log.info(
            "range_exit_order_emitted",
            symbol=symbol,
            reason=reason,
            amount=str(amount),
            price=str(current_price),
        )

        await self._bus.publish(order)
