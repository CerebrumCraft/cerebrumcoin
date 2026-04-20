"""
SQLite-backed deduplication ledger for congressional trade filings.

Persists every filing we have seen to data/cerebrum.db (shared DB, new table).
The ledger is the single source of truth for "have we already emitted a signal
for this filing?" — prevents double-emission across restarts.

# @decision DEC-PELOSI-DATA-001
# @title Finnhub free-tier congressional-trading endpoint
# @status accepted
# @rationale Zero cost, stable JSON contract, filing_id exposed. Non-commercial ToS
# acceptable for paper-only v1. Revisit Quiver if paper validates and we want live.
# Finnhub ToS: https://finnhub.io/terms (non-commercial use permitted on free tier).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

# Default shared DB — same file used by the web dashboard and trade tracker.
# Using a single DB avoids multi-file complexity and lets the data folder stay tidy.
_DEFAULT_DB_PATH = Path("data/cerebrum.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS congressional_filings (
    filing_id   TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    action      TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


class CongressionalLedger:
    """
    SQLite-backed dedup ledger for congressional trade filings.

    Thread-safety: This class is NOT thread-safe. It is intended for use
    inside a single asyncio event loop; sqlite3 operations are synchronous
    but complete in microseconds (tiny DB, primary-key lookup). Wrapping in
    asyncio.to_thread() is intentionally skipped for v1 simplicity — add if
    profiling shows it necessary.

    In-memory mode: When db_path is Path(":memory:"), a single persistent
    connection is kept open for the lifetime of the object. sqlite3's
    :memory: databases are per-connection — re-connecting creates a fresh
    empty DB, discarding the schema. This is the correct pattern for
    in-memory usage (tests, ephemeral environments).

    Schema: congressional_filings
        filing_id   TEXT PRIMARY KEY   — Finnhub's id field
        symbol      TEXT               — e.g. "NVDA"
        filing_date TEXT               — ISO-8601 date string from provider
        action      TEXT               — "stock_buy", "call_buy", "stock_sell", etc.
        recorded_at TEXT               — ISO-8601 UTC datetime when we first saw it
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        """
        Initialize ledger, creating the table if it doesn't exist.

        Args:
            db_path: Path to the SQLite database file.
                     Parent directory must already exist.
                     Defaults to data/cerebrum.db (shared with other components).
                     Pass Path(":memory:") for an in-memory database (tests).
        """
        self._db_path = db_path
        self._log = logger.bind(component="congressional_ledger", db_path=str(db_path))

        # For :memory: keep a single connection open — re-connecting creates a
        # fresh empty database each time, losing the schema and all records.
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if str(db_path) == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")

        self._init_db()

    def _init_db(self) -> None:
        """Create the congressional_filings table if it does not exist."""
        try:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE)
                conn.commit()
                self._log.info("congressional_ledger_initialized")
            finally:
                if self._persistent_conn is None:
                    conn.close()
        except sqlite3.Error as exc:
            self._log.error("congressional_ledger_init_failed", error=str(exc))
            raise

    def _connect(self) -> sqlite3.Connection:
        """Return a sqlite3 connection (persistent for :memory:, new for file paths)."""
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(str(self._db_path))

    def record(
        self,
        filing_id: str,
        symbol: str,
        filing_date: str,
        action: str,
    ) -> bool:
        """
        Persist a new filing to the ledger.

        If the filing_id already exists, this is a no-op (returns False) so
        the caller knows not to re-emit a signal.

        Args:
            filing_id:   Unique identifier from the data provider (Finnhub id).
            symbol:      Ticker symbol (e.g. "NVDA").
            filing_date: ISO-8601 date string from the provider (e.g. "2026-03-01").
            action:      Normalized action string ("stock_buy", "call_buy", etc.).

        Returns:
            True  — filing is new; was successfully inserted.
            False — filing_id already in the ledger (duplicate); nothing written.
        """
        recorded_at = datetime.now(timezone.utc).isoformat()
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO congressional_filings
                        (filing_id, symbol, filing_date, action, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (filing_id, symbol, filing_date, action, recorded_at),
                )
                conn.commit()
                rows_inserted = conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]
            finally:
                if self._persistent_conn is None:
                    conn.close()
        except sqlite3.Error as exc:
            self._log.error(
                "congressional_ledger_record_failed",
                filing_id=filing_id,
                symbol=symbol,
                error=str(exc),
            )
            return False

        if rows_inserted == 0:
            self._log.debug(
                "congressional_filing_duplicate",
                filing_id=filing_id,
                symbol=symbol,
            )
            return False

        self._log.info(
            "congressional_filing_recorded",
            filing_id=filing_id,
            symbol=symbol,
            filing_date=filing_date,
            action=action,
        )
        return True

    def has_seen(self, filing_id: str) -> bool:
        """
        Return True if this filing_id has been recorded previously.

        Args:
            filing_id: The unique identifier to look up.

        Returns:
            True if the filing_id exists in the ledger, False otherwise.
        """
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT 1 FROM congressional_filings WHERE filing_id = ? LIMIT 1",
                    (filing_id,),
                ).fetchone()
            finally:
                if self._persistent_conn is None:
                    conn.close()
            return row is not None
        except sqlite3.Error as exc:
            self._log.error(
                "congressional_ledger_lookup_failed",
                filing_id=filing_id,
                error=str(exc),
            )
            # Fail open — treat as unseen so we don't silently drop signals
            return False
