"""
Paper trading execution engine.

Simulates order execution with realistic slippage and commissions.
Maintains portfolio state in memory with file persistence.

@decision DEC-PAPER-001
@title File-based state persistence for paper trading
@status accepted
@rationale Simple JSON file persists balances and positions across restarts. No database
needed for Phase 1. State includes: balances (USD, BTC, etc.), open positions, trade history.
Atomic writes prevent corruption. Scales to thousands of trades before needing optimization.

@decision DEC-PERSIST-001
@title Per-strategy PortfolioTracker snapshots in paper_state.json
@status accepted
@rationale v2 state format adds "version": 2 and "strategy_snapshots": {name: snapshot}
alongside the existing v1 fields. v1 files (no version key) load cleanly: _state_version
defaults to 1 and _strategy_snapshots to {}. set_strategy_portfolios() registers the live
PortfolioTracker instances; _save_state() calls save_snapshot() on each and embeds the
results. get_strategy_snapshot() provides read access for main.py restore wiring.
"""

import asyncio
import json
import shutil
import uuid
from decimal import Decimal
from pathlib import Path
from time import time
from typing import Any, Callable

import structlog

from cerebrum.adapters.base import ExchangeAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import (
    Amount,
    EventType,
    OrderStatus,
    OrderType,
    Price,
    Side,
    Symbol,
)

logger = structlog.get_logger()


class PaperTradingAdapter(ExchangeAdapter):
    """
    Paper trading simulator with realistic execution.

    Features:
    - Simulated slippage and commissions
    - Instant fills at market price + slippage
    - Portfolio state persistence
    - No real money at risk
    """

    def __init__(
        self,
        bus: EventBus,
        config: dict[str, Any],
        initial_balance: Decimal,
        commission_percent: Decimal,
        slippage_percent: Decimal,
        state_file: Path,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize paper trading adapter.

        Args:
            bus: Event bus
            config: Adapter configuration
            initial_balance: Starting USD balance
            commission_percent: Commission as % of trade value
            slippage_percent: Slippage as % of price
            state_file: Path to state persistence file
            clock: Callable returning current time as a float (seconds since
                epoch). Defaults to ``time.time`` for live trading. Pass a
                BacktestClock instance in backtest mode so fill timestamps use
                the same virtual time base as signal aggregation and cooldown
                rules — avoiding permanent cooldown blocks caused by the gap
                between wall-clock time and historical candle timestamps.
                (DEC-BACKTEST-004)
        """
        super().__init__(bus, config)
        self._initial_balance = initial_balance
        self._commission_percent = commission_percent
        self._slippage_percent = slippage_percent
        self._state_file = state_file
        self._clock: Callable[[], float] = clock if clock is not None else time

        # Per-symbol commission override map (optional).
        # Enables future per-asset commission modeling — e.g., Alpaca paper trading
        # charges 0% for US equities while Kraken charges ~0.26% for crypto.
        # Set via config key "commission_by_symbol": {"AAPL/USD": 0.0, "MSFT/USD": 0.0}.
        # Symbols absent from this map fall back to commission_percent.
        self._commission_by_symbol: dict[str, Decimal] = {
            str(k): Decimal(str(v))
            for k, v in config.get("commission_by_symbol", {}).items()
        }

        # Portfolio state
        self._balances: dict[str, Decimal] = {}
        self._positions: dict[Symbol, Decimal] = {}
        self._current_prices: dict[Symbol, Price] = {}
        self._trade_history: list[dict[str, Any]] = []

        # Per-strategy persistence (v2 state format — DEC-PERSIST-001)
        # Populated via set_strategy_portfolios(); empty until called.
        self._strategy_portfolios: dict[str, Any] = {}
        # Loaded from file by _load_state(); empty for v1 files.
        self._strategy_snapshots: dict[str, dict] = {}
        self._state_version: int = 1

        self._log = logger.bind(adapter="paper_trading")

    async def connect(self) -> None:
        """Initialize paper trading state."""
        # Load state from file if exists
        if self._state_file.exists():
            # Migrate v2 → v3 before parsing (idempotent on v3 input).
            # Always runs regardless of whether orb_stocks is enabled — the
            # orb_stocks snapshot sits dormant until the strategy is activated.
            migrate_state_v2_to_v3(self._state_file, initial_balance_orb=5000.0)
            self._load_state()
            self._log.info("paper_state_loaded", state_file=str(self._state_file))
        else:
            # Initialize fresh state
            self._balances = {"USD": self._initial_balance}
            self._positions = {}
            self._trade_history = []
            self._save_state()
            self._log.info(
                "paper_state_initialized",
                initial_balance=str(self._initial_balance)
            )

        # Subscribe to market data to track current prices
        self.bus.subscribe(
            EventType.MARKET_DATA,
            self._handle_market_data,
            "paper_trading_price_tracker"
        )

        # Subscribe to order events
        self.bus.subscribe(
            EventType.ORDER,
            self.execute_order,
            "paper_trading_executor"
        )

    async def disconnect(self) -> None:
        """Save state and disconnect."""
        self._save_state()
        self._log.info("paper_trading_disconnected")

    async def subscribe_market_data(self, symbols: list[Symbol]) -> None:
        """
        Paper trading doesn't stream data—it relies on real adapter.

        Args:
            symbols: Ignored (market data comes from Kraken adapter)
        """
        self._log.info("paper_trading_uses_external_market_data")

    async def _handle_market_data(self, event: MarketDataEvent) -> None:
        """Track current prices from market data events."""
        self._current_prices[event.symbol] = event.price

    async def execute_order(self, order: OrderEvent) -> None:
        """
        Simulate order execution.

        Args:
            order: Order to execute
        """
        log = self._log.bind(order_id=order.order_id)

        try:
            # Get current price
            if order.symbol not in self._current_prices:
                log.warning("no_price_data", symbol=order.symbol)
                return

            current_price = self._current_prices[order.symbol]

            # Calculate execution price with slippage
            if order.side == Side.BUY:
                slippage_factor = Decimal("1") + (self._slippage_percent / Decimal("100"))
                fill_price = current_price * slippage_factor
            else:  # SELL
                slippage_factor = Decimal("1") - (self._slippage_percent / Decimal("100"))
                fill_price = current_price * slippage_factor

            # Calculate commission — per-symbol override takes priority over the global rate.
            # This supports zero-commission brokers (e.g. Alpaca paper for US equities)
            # coexisting with fee-bearing venues (e.g. Kraken crypto at ~0.26%).
            trade_value = fill_price * order.amount
            per_symbol_pct = self._commission_by_symbol.get(str(order.symbol))
            if per_symbol_pct is not None:
                commission = trade_value * (per_symbol_pct / Decimal("100"))
            else:
                commission = trade_value * (self._commission_percent / Decimal("100"))

            # Check if we have sufficient balance
            base_asset, quote_asset = order.symbol.split("/")

            if order.side == Side.BUY:
                required_usd = trade_value + commission
                if self._balances.get(quote_asset, Decimal("0")) < required_usd:
                    log.warning(
                        "insufficient_balance",
                        required=str(required_usd),
                        available=str(self._balances.get(quote_asset, Decimal("0")))
                    )
                    return

                # Execute buy
                self._balances[quote_asset] = self._balances.get(
                    quote_asset, Decimal("0")
                ) - required_usd
                self._positions[order.symbol] = self._positions.get(
                    order.symbol, Decimal("0")
                ) + order.amount

            else:  # SELL
                if self._positions.get(order.symbol, Decimal("0")) < order.amount:
                    log.warning(
                        "insufficient_position",
                        required=str(order.amount),
                        available=str(self._positions.get(order.symbol, Decimal("0")))
                    )
                    return

                # Execute sell
                self._positions[order.symbol] = self._positions.get(
                    order.symbol, Decimal("0")
                ) - order.amount
                self._balances[quote_asset] = self._balances.get(
                    quote_asset, Decimal("0")
                ) + trade_value - commission

            # Record trade
            trade_record = {
                "order_id": order.order_id,
                "timestamp": self._clock(),
                "symbol": order.symbol,
                "side": order.side.value,
                "amount": str(order.amount),
                "fill_price": str(fill_price),
                "commission": str(commission),
            }
            self._trade_history.append(trade_record)

            # Publish fill event — propagate strategy_id from order for multi-strategy routing
            fill_event = FillEvent(
                event_type=EventType.FILL,
                timestamp=self._clock(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                filled_amount=order.amount,
                fill_price=fill_price,
                commission=commission,
                commission_asset=quote_asset,
                exchange_order_id=f"paper_{uuid.uuid4().hex[:8]}",
                strategy_id=order.strategy_id,
            )
            await self.bus.publish(fill_event)

            # Save state
            self._save_state()

            log.info(
                "order_filled",
                symbol=order.symbol,
                side=order.side.value,
                amount=str(order.amount),
                fill_price=str(fill_price),
                commission=str(commission),
            )

        except Exception as e:
            log.error("order_execution_error", error=str(e))

    async def get_balance(self, asset: str) -> Decimal:
        """Get current balance for an asset."""
        return self._balances.get(asset, Decimal("0"))

    async def get_current_price(self, symbol: Symbol) -> Price:
        """Get current market price from tracked data."""
        if symbol not in self._current_prices:
            raise ValueError(f"No price data for {symbol}")
        return self._current_prices[symbol]

    async def get_position(self, symbol: Symbol) -> Amount:
        """Get current position size."""
        return self._positions.get(symbol, Decimal("0"))

    def set_strategy_portfolios(self, portfolios: dict[str, Any]) -> None:
        """
        Register per-strategy PortfolioTracker instances for state persistence.

        Must be called before _save_state() for v2 snapshots to be written.
        Typically called by main.py after strategy_registry.start_all() and
        snapshot restore have both completed.

        Args:
            portfolios: Mapping of strategy name → PortfolioTracker instance.
        """
        self._strategy_portfolios = portfolios
        self._log.info(
            "strategy_portfolios_registered",
            strategies=list(portfolios.keys()),
        )

    def get_strategy_snapshot(self, name: str) -> dict | None:
        """
        Return the saved snapshot for a strategy, or None if not present.

        Used by main.py to check whether a snapshot exists before calling
        portfolio.restore_snapshot(). Returns None for:
        - v1 state files (no strategy_snapshots key)
        - strategies added after the last save (new strategies)
        - strategies removed between saves (callers should ignore these)

        Args:
            name: Strategy name (must match the key used in set_strategy_portfolios).

        Returns:
            Snapshot dict from save_snapshot(), or None.
        """
        return self._strategy_snapshots.get(name)

    def _save_state(self) -> None:
        """
        Persist state to file.

        Writes v2 format when strategy portfolios are registered (includes
        version=2 and strategy_snapshots). Writes v1 format otherwise for
        backward compatibility with existing tooling that reads the file.
        """
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "balances": {k: str(v) for k, v in self._balances.items()},
            "positions": {k: str(v) for k, v in self._positions.items()},
            "current_prices": {k: str(v) for k, v in self._current_prices.items()},
            "trade_history": self._trade_history,
        }

        # v2: embed per-strategy snapshots when portfolios are registered
        if self._strategy_portfolios:
            state["version"] = 2
            state["strategy_snapshots"] = {
                name: portfolio.save_snapshot()
                for name, portfolio in self._strategy_portfolios.items()
            }

        # Atomic write
        temp_file = self._state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self._state_file)

    def _load_state(self) -> None:
        """
        Load state from file.

        Handles both v1 (no version key) and v2 (version=2, strategy_snapshots)
        formats. v1 files load without error; _strategy_snapshots is left empty
        and _state_version is set to 1. Corrupt files reinitialise fresh state.
        """
        try:
            with open(self._state_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    # Empty file — initialise fresh state
                    self._balances = {"USD": self._initial_balance}
                    self._positions = {}
                    self._current_prices = {}
                    self._trade_history = []
                    return

                state = json.loads(content)

            self._balances = {k: Decimal(v) for k, v in state.get("balances", {}).items()}
            self._positions = {k: Decimal(v) for k, v in state.get("positions", {}).items()}
            self._current_prices = {k: Decimal(v) for k, v in state.get("current_prices", {}).items()}
            self._trade_history = state.get("trade_history", [])

            # v2 fields — default to v1 values when absent (backward compat)
            self._state_version = state.get("version", 1)
            self._strategy_snapshots = state.get("strategy_snapshots", {})

        except (json.JSONDecodeError, ValueError):
            # Corrupt file — reinitialise fresh state
            self._balances = {"USD": self._initial_balance}
            self._positions = {}
            self._current_prices = {}
            self._trade_history = []
            self._state_version = 1
            self._strategy_snapshots = {}

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get current portfolio state for monitoring."""
        total_value = self._balances.get("USD", Decimal("0"))

        # Add value of positions
        for symbol, amount in self._positions.items():
            if amount > 0 and symbol in self._current_prices:
                total_value += amount * self._current_prices[symbol]

        return {
            "balances": {k: str(v) for k, v in self._balances.items()},
            "positions": {k: str(v) for k, v in self._positions.items()},
            "total_value_usd": str(total_value),
            "trade_count": len(self._trade_history),
            "pnl_usd": str(total_value - self._initial_balance),
        }


def migrate_state_v2_to_v3(path: Path | str, *, initial_balance_orb: float) -> dict:
    """Migrate paper_state.json from v2 to v3 by adding an orb_stocks snapshot.

    @decision DEC-STOCKS-006
    @title Atomic v2→v3 state migration with .v2.bak backup
    @status accepted
    @rationale When stocks are enabled for the first time, the state file
    must gain an `orb_stocks` strategy snapshot without disturbing the
    existing crypto snapshots or open positions. Atomic write (tmp + rename)
    ensures partial migrations can't corrupt the file; a .v2.bak backup
    preserves the v2 state for rollback.

    Preserves all existing v2 fields verbatim. Adds `orb_stocks` with
    initial cash. Idempotent on v3 input (no backup created, no file write).
    """
    path = Path(path)
    content = path.read_text().strip()
    if not content:
        # Empty file — nothing to migrate; _load_state() will initialise fresh state
        return {}

    data = json.loads(content)

    if data.get("version", 2) >= 3:
        return data  # already migrated — no-op

    # Backup before mutation
    backup = path.parent / f"{path.stem}.v2.bak{path.suffix}"
    shutil.copy(path, backup)

    # Migrate in-memory
    data["version"] = 3
    snapshots = data.setdefault("strategy_snapshots", {})
    if "orb_stocks" not in snapshots:
        snapshots["orb_stocks"] = {
            "cash_balance": str(initial_balance_orb),
            "initial_balance": str(initial_balance_orb),
            "peak_equity": str(initial_balance_orb),
            "total_realized_pnl": "0",
            "positions": {},
        }
    if "xstocks_reversion" not in snapshots:
        snapshots["xstocks_reversion"] = {
            "cash_balance": str(initial_balance_orb),
            "initial_balance": str(initial_balance_orb),
            "peak_equity": str(initial_balance_orb),
            "total_realized_pnl": "0",
            "positions": {},
        }
    if "pelosi_follow" not in snapshots:
        snapshots["pelosi_follow"] = {
            "cash_balance": str(initial_balance_orb),
            "initial_balance": str(initial_balance_orb),
            "peak_equity": str(initial_balance_orb),
            "total_realized_pnl": "0",
            "positions": {},
        }

    # Atomic write
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

    return data
