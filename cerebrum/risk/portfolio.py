"""
Portfolio state tracking for risk calculations.

Maintains current positions, balances, and P&L for risk management decisions.

@decision DEC-RISK-002
@title Portfolio state tracking for exposure calculations
@status accepted
@rationale Centralized position tracking enables position sizing, exposure limits,
drawdown detection, and P&L calculations. Subscribes to FillEvents to maintain
accurate state. Provides query interface for risk rules.

@decision DEC-RISK-004
@title strategy_id filtering in PortfolioTracker prevents double-counting in multi-strategy mode
@status accepted
@rationale In multi-strategy mode, each strategy has its own PortfolioTracker. All
FillEvents share one event bus, so without filtering every PortfolioTracker would
process every fill — causing triple-counting of positions and incorrect P&L. When
strategy_id is provided at construction, _on_fill skips events whose strategy_id
does not match. None means "accept all" (single-strategy backward compatibility).

@decision DEC-PERSIST-001
@title Per-strategy PortfolioTracker snapshots in paper_state.json
@status accepted
@rationale Each strategy has an isolated PortfolioTracker (cash, positions, peak_equity,
realized_pnl). Without per-strategy snapshots, all per-strategy equity is lost on restart
and the dashboard shows stale global aggregates. save_snapshot() serialises all mutable
state to a plain dict using str(Decimal) for lossless round-trip. restore_snapshot()
deserialises it. initial_balance is NOT restored — it is fixed at construction time and
must not drift. unrealized_pnl is intentionally set to Decimal("0") on restore; it will
be recalculated on the next MARKET_DATA tick.

@decision DEC-CONDUCTOR-008
@title Rolling closed-trades deque in PortfolioTracker for Darwinian Sharpe feed
@status accepted
@rationale DarwinianAllocator.update_performance() needs a list of closed TradeRecords
per strategy per cycle. The existing StateManager trade table is SQLite + async — a sync
read path on a 15-minute allocation cycle would either block or require awaiting inside
_apply_allocations. An in-memory deque bounded by sharpe_window_hours (default 4h) avoids
the async boundary, has O(1) append and O(1) eviction, and matches the dashboard pattern
(DEC-DASH-007 also uses in-memory snapshots). Strategy-scoped tracker means no
cross-strategy filtering. Trades older than the window are evicted on each append.
"""

import time as time_module
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, FillEvent, MarketDataEvent, PositionUpdateEvent
from cerebrum.core.state import TradeRecord
from cerebrum.core.types import Amount, EventType, Price, Side, Symbol

logger = structlog.get_logger()


@dataclass
class Position:
    """Represents an open position."""
    symbol: Symbol
    amount: Amount  # Positive for long, negative for short
    average_entry_price: Price
    current_price: Price
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    entry_time: float = field(default_factory=time_module.time)  # Unix epoch seconds

    def update_price(self, price: Price) -> None:
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = self.amount * (price - self.average_entry_price)


class PortfolioTracker:
    """
    Tracks portfolio state for risk management.

    Features:
    - Position tracking per symbol
    - Cash balance management
    - Realized and unrealized P&L
    - Total exposure calculation
    - Peak equity tracking for drawdown
    """

    # Default retention window for closed trades fed to DarwinianAllocator (4h).
    # Matches DarwinianAllocator.sharpe_window_hours default so the Conductor
    # always passes the correct window without configuration duplication.
    # Override by passing sharpe_window_hours at construction.
    _DEFAULT_SHARPE_WINDOW_HOURS: float = 4.0

    def __init__(
        self,
        bus: EventBus,
        initial_balance: Decimal,
        strategy_id: str | None = None,
        sharpe_window_hours: float = _DEFAULT_SHARPE_WINDOW_HOURS,
    ) -> None:
        """
        Initialize portfolio tracker.

        Args:
            bus: Event bus
            initial_balance: Starting cash balance
            strategy_id: When provided, _on_fill ignores FillEvents whose
                strategy_id does not match. None accepts all fills (single-
                strategy backward-compatible mode). See DEC-RISK-004.
            sharpe_window_hours: Retention window for closed trades deque
                (DEC-CONDUCTOR-008). Trades older than this are evicted on
                each append. Defaults to 4h to match DarwinianAllocator default.
        """
        self._bus = bus
        self._cash_balance = initial_balance
        self._initial_balance = initial_balance
        self._strategy_id = strategy_id
        self._positions: dict[Symbol, Position] = {}
        self._peak_equity = initial_balance
        self._total_realized_pnl = Decimal("0.0")
        self._sharpe_window_seconds: float = sharpe_window_hours * 3600

        # Rolling closed-trades deque for DarwinianAllocator Sharpe feed.
        # Bounded by wall-clock retention (_sharpe_window_seconds); stale entries
        # are evicted on each append in _on_fill. See DEC-CONDUCTOR-008.
        self._closed_trades: Deque[TradeRecord] = deque()

        # Track latest prices for all symbols (needed for market order sizing)
        self._latest_prices: dict[Symbol, Price] = {}

        self._log = logger.bind(
            component="portfolio_tracker",
            strategy_id=strategy_id or "global",
        )

        # Use a unique subscriber name so multiple PortfolioTrackers on the same
        # bus do not collide. When strategy_id is None the legacy name is preserved
        # for backward compatibility with tests that assert subscriber counts.
        fill_sub_name = (
            f"portfolio_tracker_{strategy_id}" if strategy_id else "portfolio_tracker"
        )
        price_sub_name = (
            f"portfolio_tracker_prices_{strategy_id}"
            if strategy_id
            else "portfolio_tracker_prices"
        )

        # Subscribe to fills and market data
        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name=fill_sub_name,
        )
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name=price_sub_name,
        )

        self._log.info("portfolio_tracker_initialized", initial_balance=str(initial_balance))

    async def _on_fill(self, event: Event) -> None:
        """Handle fill events to update positions."""
        if not isinstance(event, FillEvent):
            return

        # DEC-RISK-004: In multi-strategy mode each PortfolioTracker must only
        # process fills that belong to its own strategy. strategy_id=None means
        # accept all (single-strategy backward-compat path).
        if self._strategy_id is not None:
            event_sid = getattr(event, "strategy_id", None)
            if event_sid != self._strategy_id:
                return

        symbol = event.symbol
        filled_amount = event.filled_amount
        fill_price = event.fill_price
        commission = event.commission

        # Adjust for side
        if event.side == Side.SELL:
            filled_amount = -filled_amount

        # @decision DEC-DASH-001: Publish PositionUpdateEvent on fills only (not price ticks)
        # to keep dashboard updated without flooding the event bus.
        #
        # Update or create position, then publish PositionUpdateEvent so subscribers
        # (e.g. the monitoring dashboard) stay in sync with current position state.
        # On position closure we publish amount=0 so subscribers can remove the entry.
        position_event: PositionUpdateEvent | None = None

        if symbol in self._positions:
            pos = self._positions[symbol]

            # Check if closing/reducing position
            if (pos.amount > 0 and filled_amount < 0) or (pos.amount < 0 and filled_amount > 0):
                # Closing trade: realize P&L
                close_amount = min(abs(filled_amount), abs(pos.amount))
                pnl_per_unit = fill_price - pos.average_entry_price
                realized_pnl = close_amount * pnl_per_unit * (1 if pos.amount > 0 else -1)

                pos.realized_pnl += realized_pnl
                self._total_realized_pnl += realized_pnl

                # Update position
                new_amount = pos.amount + filled_amount
                if abs(new_amount) < Decimal("0.0001"):  # Position closed
                    # Capture state before deletion so we can publish event
                    closed_entry_price = pos.average_entry_price
                    closed_entry_time = pos.entry_time
                    closed_realized_pnl = pos.realized_pnl
                    del self._positions[symbol]
                    self._log.info(
                        "position_closed",
                        symbol=symbol,
                        realized_pnl=str(realized_pnl),
                    )
                    # Record closed trade for DarwinianAllocator Sharpe feed
                    # (DEC-CONDUCTOR-008). Side is the original open side: if
                    # filled_amount is negative (sell-to-close) the position
                    # was a long, so opening side was BUY.
                    open_side = Side.SELL if filled_amount > 0 else Side.BUY
                    self._append_closed_trade(
                        symbol=symbol,
                        side=open_side,
                        entry_time=closed_entry_time,
                        entry_price=closed_entry_price,
                        exit_time=event.timestamp,
                        exit_price=fill_price,
                        quantity=close_amount,
                        pnl=realized_pnl,
                    )
                    # amount=0 signals closure to subscribers
                    position_event = PositionUpdateEvent(
                        event_type=EventType.POSITION_UPDATE,
                        timestamp=event.timestamp,
                        symbol=symbol,
                        amount=Decimal("0"),
                        average_entry_price=closed_entry_price,
                        current_price=fill_price,
                        unrealized_pnl=Decimal("0.0"),
                        realized_pnl=closed_realized_pnl,
                    )
                else:
                    pos.amount = new_amount
                    position_event = PositionUpdateEvent(
                        event_type=EventType.POSITION_UPDATE,
                        timestamp=event.timestamp,
                        symbol=symbol,
                        amount=pos.amount,
                        average_entry_price=pos.average_entry_price,
                        current_price=pos.current_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=pos.realized_pnl,
                    )
            else:
                # Adding to position: update average entry (preserve original entry_time)
                total_cost = (pos.amount * pos.average_entry_price) + (filled_amount * fill_price)
                new_amount = pos.amount + filled_amount
                pos.average_entry_price = total_cost / new_amount
                pos.amount = new_amount
                position_event = PositionUpdateEvent(
                    event_type=EventType.POSITION_UPDATE,
                    timestamp=event.timestamp,
                    symbol=symbol,
                    amount=pos.amount,
                    average_entry_price=pos.average_entry_price,
                    current_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                )
        else:
            # New position — record entry time at fill event timestamp
            self._positions[symbol] = Position(
                symbol=symbol,
                amount=filled_amount,
                average_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
                entry_time=event.timestamp,
            )
            self._log.info(
                "position_opened",
                symbol=symbol,
                amount=str(filled_amount),
                price=str(fill_price),
            )
            position_event = PositionUpdateEvent(
                event_type=EventType.POSITION_UPDATE,
                timestamp=event.timestamp,
                symbol=symbol,
                amount=filled_amount,
                average_entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=Decimal("0.0"),
                realized_pnl=Decimal("0.0"),
            )

        # Update cash balance
        cost = filled_amount * fill_price
        self._cash_balance -= cost + commission

        # Update peak equity
        equity = self.get_total_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Publish position state so dashboard and other subscribers stay current.
        # Published after cash/equity updates so subscribers see consistent state.
        if position_event is not None:
            await self._bus.publish(position_event)

    async def _on_market_data(self, event: Event) -> None:
        """Update position prices with market data."""
        if not isinstance(event, MarketDataEvent):
            return

        symbol = event.symbol
        # Cache latest price for all symbols (used for market order sizing)
        self._latest_prices[symbol] = event.price

        # Update position price if we have one
        if symbol in self._positions:
            self._positions[symbol].update_price(event.price)

    # ------------------------------------------------------------------
    # Closed-trade deque (DEC-CONDUCTOR-008)
    # ------------------------------------------------------------------

    def _append_closed_trade(
        self,
        symbol: Symbol,
        side: Side,
        entry_time: float,
        entry_price: Decimal,
        exit_time: float,
        exit_price: Decimal,
        quantity: Decimal,
        pnl: Decimal,
    ) -> None:
        """
        Append a completed round-trip to the rolling closed-trades deque.

        Evicts entries older than _sharpe_window_seconds before appending so
        the deque stays bounded without a fixed max-length. Called internally
        from _on_fill at the position_closed branch (DEC-CONDUCTOR-008).

        Args:
            symbol: Traded symbol.
            side: Opening side (BUY for long trades, SELL for shorts).
            entry_time: Unix epoch of position open.
            entry_price: Average entry price.
            exit_time: Unix epoch of position close (fill timestamp).
            exit_price: Fill price on the closing leg.
            quantity: Size of the closed portion.
            pnl: Realized P&L for this close (commission not deducted — the
                 Sharpe calc only needs relative return sign and magnitude).
        """
        # Evict stale records before appending
        cutoff = exit_time - self._sharpe_window_seconds
        while self._closed_trades and self._closed_trades[0].entry_time < cutoff:
            self._closed_trades.popleft()

        record = TradeRecord(
            id=None,
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            signal_snapshot={},
            regime="UNKNOWN",  # regime not tracked at this layer
            status="closed",
            strategy_id=self._strategy_id,
        )
        self._closed_trades.append(record)

    def get_closed_trades(self, window_seconds: float | None = None) -> list[TradeRecord]:
        """
        Return closed trades within the given window.

        Args:
            window_seconds: If provided, only trades whose entry_time falls
                within the last window_seconds are returned. If None, returns
                all trades currently in the deque (already bounded by
                _sharpe_window_seconds from construction).

        Returns:
            List of TradeRecord (oldest first). Safe to iterate while new
            trades are being appended — returns a snapshot copy.
        """
        if window_seconds is None:
            return list(self._closed_trades)

        now = time_module.time()
        cutoff = now - window_seconds
        return [t for t in self._closed_trades if t.entry_time >= cutoff]

    def get_position(self, symbol: Symbol) -> Position | None:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[Symbol, "Position"]:
        """Get all open positions (snapshot copy)."""
        return dict(self._positions)

    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash_balance

    def get_total_equity(self) -> Decimal:
        """Get total equity (cash + signed position market values).

        Short positions (negative amount) correctly reduce equity since the
        trader owes the asset. Long positions (positive amount) increase equity.
        Do NOT use abs() here — that was the double-counting bug (DEC-RISK-003).
        """
        position_value = sum(
            pos.amount * pos.current_price  # signed: long adds, short subtracts
            for pos in self._positions.values()
        )
        return self._cash_balance + position_value

    def get_total_exposure(self) -> Decimal:
        """Get total position exposure (sum of position values)."""
        return sum(
            abs(pos.amount * pos.current_price)
            for pos in self._positions.values()
        )

    def get_drawdown_percent(self) -> Decimal:
        """Get current drawdown from peak equity."""
        equity = self.get_total_equity()
        if self._peak_equity == 0:
            return Decimal("0.0")
        drawdown = (self._peak_equity - equity) / self._peak_equity * 100
        return max(Decimal("0.0"), drawdown)

    def get_pnl(self) -> tuple[Decimal, Decimal]:
        """Get (realized_pnl, unrealized_pnl)."""
        unrealized = sum(pos.unrealized_pnl for pos in self._positions.values())
        return self._total_realized_pnl, unrealized

    def get_latest_price(self, symbol: Symbol) -> Price | None:
        """Get the latest known price for a symbol."""
        return self._latest_prices.get(symbol)

    def adjust_balance(self, delta: Decimal) -> None:
        """
        Adjust cash balance by delta (positive = add, negative = remove).

        Used by the Conductor to redistribute capital between strategy
        portfolios when Darwinian allocation changes. Updates peak equity
        so drawdown calculations remain meaningful after capital flows.

        Peak equity treatment by direction:

        - Injection (delta > 0): peak rises normally — we now have more capital
          at risk and the high-water mark should reflect that.
        - Withdrawal (delta < 0): peak is *also lowered* by the withdrawal
          amount (floored at new equity). This prevents a transient Conductor
          spike from leaving a permanently elevated peak that triggers a false
          max-drawdown circuit-breaker.

          Example (Session 9 root cause):
            T+90s  inject +$5,000  -> cash=$7,500, peak=$7,500
            T+3m   withdraw -$5,000 -> cash=$2,500
            Without this fix: peak stays at $7,500, drawdown = 66.7% -> circuit
            breaker fires, strategy dead for entire session.
            With this fix: peak lowers to max($2,500, $7,500-$5,000) = $2,500
            -> drawdown = 0%, trading resumes normally.

        Real trading losses (which also reduce equity) are still captured: the
        max(new_equity, ...) floor ensures peak never drops below current
        equity, so a genuine loss component within the same withdrawal is
        preserved.

        @decision DEC-RISK-005
        @title Peak equity lowered on capital withdrawal to prevent false drawdown
        @status accepted
        @rationale Conductor reallocation injects then later withdraws capital.
        Without peak-lowering on withdrawal, _peak_equity holds a transient high
        and the drawdown calculation permanently exceeds the circuit-breaker
        threshold, blocking all trading. The fix: on negative delta, lower peak
        by the same amount (floor at new equity so real losses are preserved).

        Args:
            delta: Amount to add (positive) or remove (negative) from cash.
        """
        self._cash_balance += delta
        new_equity = self.get_total_equity()
        if delta < Decimal("0"):
            # Capital withdrawal: lower peak proportionally so drawdown stays
            # relative to actual allocated capital, not a transient spike.
            self._peak_equity = max(new_equity, self._peak_equity + delta)
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity
        self._log.debug(
            "balance_adjusted",
            delta=str(delta),
            new_cash_balance=str(self._cash_balance),
            new_peak_equity=str(self._peak_equity),
        )

    def save_snapshot(self) -> dict:
        """
        Serialise portfolio state for persistence.

        Returns a plain dict of strings (all Decimals as str for lossless
        round-trip through JSON). Intended for PaperTradingAdapter._save_state()
        to embed under strategy_snapshots[strategy_id].

        Note: initial_balance is included for informational purposes only — it
        is NOT used by restore_snapshot() because the balance is fixed at
        construction time.

        closed_trades is serialised so Sharpe history survives restarts
        (DEC-CONDUCTOR-012). Each trade is a minimal flat dict — only the
        fields needed by calculate_sharpe_ratio (pnl) and _append_closed_trade
        (all others) are stored.

        Returns:
            Dict suitable for json.dumps() containing cash_balance,
            initial_balance, peak_equity, total_realized_pnl, positions, and
            closed_trades.
        """
        from cerebrum.core.types import Side as _Side  # local import avoids circular

        return {
            "cash_balance": str(self._cash_balance),
            "initial_balance": str(self._initial_balance),
            "peak_equity": str(self._peak_equity),
            "total_realized_pnl": str(self._total_realized_pnl),
            "positions": {
                symbol: {
                    "amount": str(pos.amount),
                    "average_entry_price": str(pos.average_entry_price),
                    "current_price": str(pos.current_price),
                    "realized_pnl": str(pos.realized_pnl),
                    "entry_time": pos.entry_time,
                }
                for symbol, pos in self._positions.items()
            },
            # DEC-CONDUCTOR-012: persist closed trades so Sharpe survives restart
            "closed_trades": [
                {
                    "symbol": str(t.symbol),
                    "side": t.side.value,
                    "entry_time": t.entry_time,
                    "entry_price": str(t.entry_price),
                    "exit_time": t.exit_time,
                    "exit_price": str(t.exit_price) if t.exit_price is not None else None,
                    "quantity": str(t.quantity),
                    "pnl": str(t.pnl) if t.pnl is not None else None,
                    "strategy_id": t.strategy_id,
                }
                for t in self._closed_trades
            ],
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        """
        Restore portfolio state from a saved snapshot.

        Overwrites cash_balance, peak_equity, total_realized_pnl, positions,
        and closed_trades from the snapshot dict. initial_balance is
        intentionally NOT restored — it is fixed at construction time.

        unrealized_pnl on restored positions is set to Decimal("0"); it will
        be recalculated automatically on the next MARKET_DATA tick via
        _on_market_data → Position.update_price().

        closed_trades: older entries are silently dropped if they fall outside
        the current _sharpe_window_seconds. v3 snapshots without the
        closed_trades key load cleanly (deque stays empty). See DEC-CONDUCTOR-012.

        Args:
            snapshot: Dict previously produced by save_snapshot().
        """
        from cerebrum.core.types import Side as _Side  # local import avoids circular

        self._cash_balance = Decimal(snapshot["cash_balance"])
        self._peak_equity = Decimal(snapshot["peak_equity"])
        self._total_realized_pnl = Decimal(snapshot["total_realized_pnl"])
        self._positions = {}
        for symbol, pos_data in snapshot.get("positions", {}).items():
            self._positions[symbol] = Position(
                symbol=symbol,
                amount=Decimal(pos_data["amount"]),
                average_entry_price=Decimal(pos_data["average_entry_price"]),
                current_price=Decimal(pos_data["current_price"]),
                unrealized_pnl=Decimal("0"),  # recalculated on next price tick
                realized_pnl=Decimal(pos_data["realized_pnl"]),
                entry_time=pos_data.get("entry_time", 0.0),
            )

        # DEC-CONDUCTOR-012: restore closed trades for Sharpe continuity across restart.
        # Stale entries (outside current window) are dropped via the cutoff filter
        # so the deque remains bounded after restore. Absent key → empty deque (v3 compat).
        self._closed_trades = deque()
        now = time_module.time()
        cutoff = now - self._sharpe_window_seconds
        for td in snapshot.get("closed_trades", []):
            entry_time = float(td.get("entry_time", 0.0))
            if entry_time < cutoff:
                continue  # older than window — skip
            exit_price_raw = td.get("exit_price")
            pnl_raw = td.get("pnl")
            self._closed_trades.append(
                TradeRecord(
                    id=None,
                    symbol=td["symbol"],
                    side=_Side(td["side"]),
                    entry_time=entry_time,
                    entry_price=Decimal(td["entry_price"]),
                    exit_time=float(td.get("exit_time") or 0.0),
                    exit_price=Decimal(exit_price_raw) if exit_price_raw is not None else None,
                    quantity=Decimal(td["quantity"]),
                    pnl=Decimal(pnl_raw) if pnl_raw is not None else None,
                    signal_snapshot={},
                    regime="UNKNOWN",
                    status="closed",
                    strategy_id=td.get("strategy_id"),
                )
            )

        self._log.info(
            "portfolio_snapshot_restored",
            cash_balance=str(self._cash_balance),
            peak_equity=str(self._peak_equity),
            positions=list(self._positions.keys()),
            closed_trades_restored=len(self._closed_trades),
        )
