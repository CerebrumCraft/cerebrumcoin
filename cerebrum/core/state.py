"""
State persistence manager using SQLite for learning system.

Manages trade history, signal scores, weight history, and system state.
Provides async interface for concurrent access from event handlers.

@decision DEC-STATE-001
@title SQLite with aiosqlite for async state persistence
@status accepted
@rationale Learning system needs durable storage for trade outcomes, signal performance,
and weight evolution. SQLite provides zero-ops persistence with ACID guarantees.
aiosqlite enables async access without blocking the event loop. Schema supports
per-regime tracking for adaptive weight profiles.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from cerebrum.core.types import Side, SignalType, Symbol

logger = structlog.get_logger()


@dataclass
class TradeRecord:
    """Record of a completed or open trade."""
    id: int | None
    symbol: Symbol
    side: Side
    entry_time: float
    entry_price: Decimal
    exit_time: float | None
    exit_price: Decimal | None
    quantity: Decimal
    pnl: Decimal | None
    signal_snapshot: dict[str, Any]
    regime: str
    status: str
    # Bug fix (Session 7): strategy_id added for multi-strategy P&L attribution.
    # Optional with None default so existing call-sites and test fixtures that
    # construct TradeRecord without this field continue to work unchanged.
    strategy_id: str | None = None


@dataclass
class SignalScore:
    """Performance metrics for a signal type in a specific regime."""
    signal_type: SignalType
    regime: str
    win_rate: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    sample_size: int
    updated_at: datetime


class StateManager:
    """
    Manages persistent state for the learning system.

    Features:
    - Trade history tracking (open and closed trades)
    - Signal performance scoring per regime
    - Weight adjustment history
    - Generic key-value state storage
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize state manager."""
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._log = logger.bind(component="state_manager", db_path=str(db_path))

    async def initialize(self) -> None:
        """Initialize database connection and create schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._create_schema()
        self._log.info("state_manager_initialized")

    async def _create_schema(self) -> None:
        """Create database schema if not exists."""
        assert self._db is not None
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time REAL NOT NULL,
                entry_price TEXT NOT NULL,
                exit_time REAL,
                exit_price TEXT,
                quantity TEXT NOT NULL,
                pnl TEXT,
                signal_snapshot TEXT NOT NULL,
                regime TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS signal_scores (
                signal_type TEXT NOT NULL,
                regime TEXT NOT NULL,
                win_rate TEXT NOT NULL,
                profit_factor TEXT NOT NULL,
                sharpe_ratio TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (signal_type, regime)
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                regime TEXT NOT NULL,
                weight TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_weight_history_signal ON weight_history(signal_type, regime)")
        # Bug fix (Session 7): add strategy_id column for multi-strategy attribution.
        # ALTER TABLE is used (not CREATE TABLE) so that existing production databases
        # with live trade history gain the column without losing data. The try/except
        # swallows the "duplicate column" error when running against a DB that was
        # already migrated (e.g. a second process startup or a test re-run).
        try:
            await self._db.execute("ALTER TABLE trades ADD COLUMN strategy_id TEXT")
        except Exception:
            pass  # Column already exists — safe to ignore
        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._log.info("state_manager_closed")

    async def save_trade(self, trade: TradeRecord) -> int:
        """Save a trade record. Returns trade ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            """INSERT INTO trades (symbol, side, entry_time, entry_price, exit_time, exit_price,
            quantity, pnl, signal_snapshot, regime, status, strategy_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade.symbol, trade.side.value, trade.entry_time, str(trade.entry_price),
             trade.exit_time, str(trade.exit_price) if trade.exit_price else None,
             str(trade.quantity), str(trade.pnl) if trade.pnl else None,
             json.dumps(trade.signal_snapshot), trade.regime, trade.status,
             trade.strategy_id)
        )
        await self._db.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    async def update_trade(self, trade_id: int, **updates: Any) -> None:
        """Update a trade record."""
        assert self._db is not None
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            if isinstance(value, Decimal):
                values.append(str(value))
            elif isinstance(value, Side):
                values.append(value.value)
            else:
                values.append(value)
        values.append(trade_id)
        await self._db.execute(f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?", values)
        await self._db.commit()

    async def get_trade(self, trade_id: int) -> TradeRecord | None:
        """Get a trade record by ID."""
        assert self._db is not None
        async with self._db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)) as cursor:
            row = await cursor.fetchone()
            return self._row_to_trade_record(row) if row else None

    async def get_open_trades(
        self,
        symbol: Symbol | None = None,
        strategy_id: str | None = None,
    ) -> list[TradeRecord]:
        """Get all open trades, optionally filtered by symbol and/or strategy_id.

        Results are always ordered by entry_time ASC so that FIFO matching
        in TradeTracker._on_fill() reliably picks the oldest open trade
        via open_trades[0].

        Bug fix (Session 7): the previous queries had no ORDER BY clause.
        SQLite returns rows in an unspecified order without it, which meant
        FIFO matching was non-deterministic and could close the wrong trade
        when multiple open positions existed for the same symbol.

        Bug fix (Session 20): added strategy_id parameter. When provided,
        adds ``AND strategy_id = ?`` so FIFO matching stays within the
        originating strategy's trades and never closes an orphan trade from
        a previous session or a different strategy. The symbol-only path is
        preserved for backward compatibility with callers that don't pass
        strategy_id (DEC-TRACK-001).
        """
        assert self._db is not None
        conditions = ["status = 'OPEN'"]
        params: list[str] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if strategy_id is not None:
            conditions.append("strategy_id = ?")
            params.append(strategy_id)

        where = " AND ".join(conditions)
        query = f"SELECT * FROM trades WHERE {where} ORDER BY entry_time ASC"
        async with self._db.execute(query, params) as cursor:
            return [self._row_to_trade_record(row) for row in await cursor.fetchall()]

    async def get_closed_trades(self, regime: str | None = None, limit: int | None = None) -> list[TradeRecord]:
        """Get closed trades, optionally filtered by regime."""
        assert self._db is not None
        if regime:
            query, params = "SELECT * FROM trades WHERE status = 'CLOSED' AND regime = ? ORDER BY exit_time DESC", (regime,)
        else:
            query, params = "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_time DESC", ()
        if limit:
            query += f" LIMIT {limit}"
        async with self._db.execute(query, params) as cursor:
            return [self._row_to_trade_record(row) for row in await cursor.fetchall()]

    def _row_to_trade_record(self, row: aiosqlite.Row) -> TradeRecord:
        """Convert database row to TradeRecord.

        strategy_id is read via dict(row) so that rows from older databases
        that pre-date the ALTER TABLE migration return None rather than raising
        an IndexError.
        """
        row_dict = dict(row)
        return TradeRecord(
            id=row_dict["id"], symbol=row_dict["symbol"], side=Side(row_dict["side"]),
            entry_time=row_dict["entry_time"], entry_price=Decimal(row_dict["entry_price"]),
            exit_time=row_dict["exit_time"],
            exit_price=Decimal(row_dict["exit_price"]) if row_dict["exit_price"] else None,
            quantity=Decimal(row_dict["quantity"]),
            pnl=Decimal(row_dict["pnl"]) if row_dict["pnl"] else None,
            signal_snapshot=json.loads(row_dict["signal_snapshot"]),
            regime=row_dict["regime"],
            status=row_dict["status"],
            strategy_id=row_dict.get("strategy_id"),
        )

    async def save_signal_score(self, score: SignalScore) -> None:
        """Save or update a signal score."""
        assert self._db is not None
        await self._db.execute(
            """INSERT OR REPLACE INTO signal_scores (signal_type, regime, win_rate, profit_factor,
            sharpe_ratio, sample_size, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (score.signal_type.value, score.regime, str(score.win_rate), str(score.profit_factor),
             str(score.sharpe_ratio), score.sample_size, score.updated_at.isoformat())
        )
        await self._db.commit()

    async def get_signal_score(self, signal_type: SignalType, regime: str) -> SignalScore | None:
        """Get signal score for a specific signal type and regime."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM signal_scores WHERE signal_type = ? AND regime = ?",
            (signal_type.value, regime)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return SignalScore(
                signal_type=SignalType(row["signal_type"]), regime=row["regime"],
                win_rate=Decimal(row["win_rate"]), profit_factor=Decimal(row["profit_factor"]),
                sharpe_ratio=Decimal(row["sharpe_ratio"]), sample_size=row["sample_size"],
                updated_at=datetime.fromisoformat(row["updated_at"])
            )

    async def get_all_signal_scores(self, regime: str) -> list[SignalScore]:
        """Get all signal scores for a regime."""
        assert self._db is not None
        async with self._db.execute("SELECT * FROM signal_scores WHERE regime = ?", (regime,)) as cursor:
            return [
                SignalScore(
                    signal_type=SignalType(row["signal_type"]), regime=row["regime"],
                    win_rate=Decimal(row["win_rate"]), profit_factor=Decimal(row["profit_factor"]),
                    sharpe_ratio=Decimal(row["sharpe_ratio"]), sample_size=row["sample_size"],
                    updated_at=datetime.fromisoformat(row["updated_at"])
                )
                for row in await cursor.fetchall()
            ]

    async def save_weight(self, signal_type: SignalType, regime: str, weight: Decimal, timestamp: float) -> None:
        """Save a weight adjustment to history."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO weight_history (signal_type, regime, weight, timestamp) VALUES (?, ?, ?, ?)",
            (signal_type.value, regime, str(weight), timestamp)
        )
        await self._db.commit()

    async def get_weight_history(self, signal_type: SignalType, regime: str, limit: int = 100) -> list[tuple[float, Decimal]]:
        """Get weight history for a signal type and regime."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT timestamp, weight FROM weight_history WHERE signal_type = ? AND regime = ? ORDER BY timestamp DESC LIMIT ?",
            (signal_type.value, regime, limit)
        ) as cursor:
            return [(row["timestamp"], Decimal(row["weight"])) for row in await cursor.fetchall()]

    async def set_state(self, key: str, value: str) -> None:
        """Set a state value."""
        assert self._db is not None
        await self._db.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)", (key, value))
        await self._db.commit()

    async def get_state(self, key: str) -> str | None:
        """Get a state value."""
        assert self._db is not None
        async with self._db.execute("SELECT value FROM system_state WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            return row["value"] if row else None
