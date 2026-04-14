"""
EndOfDayFlatten — auto-close stock positions near RTH close.

Subscribes to MARKET_DATA, checks elapsed time against the close - flatten_offset
window, and emits MARKET OrderEvents for every open stock position that hasn't
been flattened today.  Idempotent: tracks (date, symbol) pairs already fired to
prevent duplicate close orders across ticks.

@decision DEC-STOCKS-003
@title End-of-day forced flatten for stock positions
@status accepted
@rationale Zero-overnight-stock-exposure mandate. At close - flatten_offset_minutes,
this module emits market-order closes for any open positions it manages,
tagged with strategy_id="orb_stocks" for correct routing. Subscribes to
MARKET_DATA so the time check runs on each tick (no separate timer).
Idempotent: (date, symbol) pairs already flattened for the day are
tracked to prevent duplicate close orders. Mirrors ExitMonitor structure.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from cerebrum.core.bus import EventBus
from cerebrum.core.events import OrderEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side
from cerebrum.utils.trading_session import ET, rth_close_for

logger = logging.getLogger(__name__)


class EndOfDayFlatten:
    """Emits close orders for open stock positions before RTH close.

    Architecture mirrors ExitMonitor: subscribes to MARKET_DATA on the bus,
    iterates open portfolio positions on each tick, and emits MARKET OrderEvents
    for qualifying symbols once the flatten window is reached.

    The flatten window opens at (close - flatten_offset_minutes) on every
    trading day, including early-close days (Black Friday, Christmas Eve, etc.).
    On weekends and NYSE holidays rth_close_for() returns None, so no orders
    are ever emitted on non-trading days.
    """

    def __init__(
        self,
        bus: EventBus,
        portfolio: Any,
        stock_symbols: list[str],
        flatten_offset_minutes: int = 5,
        strategy_id: str = "orb_stocks",
    ) -> None:
        """
        Args:
            bus: Event bus — used for subscribe + publish.
            portfolio: PortfolioTracker (or compatible mock). Must expose
                       get_all_positions() -> dict[symbol, position].
            stock_symbols: Symbols managed by this flattener.  Positions in
                           other symbols (crypto, etc.) are ignored.
            flatten_offset_minutes: How many minutes before RTH close to start
                                    flattening.  Default 5 → flatten at 15:55 ET.
            strategy_id: Passed through to emitted OrderEvents so the
                         per-strategy PortfolioTracker routes fills correctly
                         (DEC-EXIT-003 pattern).
        """
        self._bus = bus
        self._portfolio = portfolio
        self._stock_symbols: frozenset[str] = frozenset(stock_symbols)
        self._flatten_offset_minutes: int = int(flatten_offset_minutes)
        self._strategy_id: str = strategy_id

        # (date, symbol) pairs already flattened — idempotency guard
        self._fired_today: set[tuple[date, str]] = set()

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="end_of_day_flatten",
        )

        logger.info(
            "end_of_day_flatten_initialized",
            extra={
                "stock_symbols": sorted(stock_symbols),
                "flatten_offset_minutes": flatten_offset_minutes,
                "strategy_id": strategy_id,
            },
        )

    async def _on_market_data(self, event: Any) -> None:
        """Check whether the flatten window has opened and emit close orders.

        Called on every MARKET_DATA tick. The timestamp on the event is a Unix
        epoch float (Timestamp = float, per cerebrum/core/types.py).
        """
        ts_epoch: float = event.timestamp
        now_et: datetime = datetime.fromtimestamp(ts_epoch, tz=ET)
        today: date = now_et.date()

        close = rth_close_for(today)
        if close is None:
            # Weekend or NYSE holiday — market closed, nothing to do.
            return

        # Flatten window: current minute >= close - offset
        close_minute = close.hour * 60 + close.minute
        current_minute = now_et.hour * 60 + now_et.minute
        if current_minute < close_minute - self._flatten_offset_minutes:
            return

        # Iterate all open positions and close qualifying ones
        positions: dict[str, Any] = self._portfolio.get_all_positions()
        for symbol, pos in positions.items():
            if symbol not in self._stock_symbols:
                continue
            if pos.amount == Decimal("0"):
                continue
            key = (today, symbol)
            if key in self._fired_today:
                continue
            self._fired_today.add(key)
            await self._publish_close(pos, ts_epoch)

    async def _publish_close(self, pos: Any, ts_epoch: float) -> None:
        """Emit a MARKET close order for the given position."""
        side = Side.SELL if pos.amount > Decimal("0") else Side.BUY
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=ts_epoch,
            order_id=str(uuid4()),
            symbol=pos.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=abs(pos.amount),
            status=OrderStatus.PENDING,
            metadata={
                "source": "end_of_day_flatten",
                "exit_reason": "end_of_day_flatten",
            },
            strategy_id=self._strategy_id,
        )
        logger.info(
            "end_of_day_flatten_order_emitted",
            extra={
                "symbol": pos.symbol,
                "side": side.value,
                "amount": str(pos.amount),
                "strategy_id": self._strategy_id,
            },
        )
        await self._bus.publish(order)
