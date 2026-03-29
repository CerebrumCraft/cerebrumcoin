"""
Unit tests for scripts/fix_orphaned_trades.py.

Uses in-memory SQLite databases seeded with known data to verify:
- OPEN trades are marked CLOSED with pnl=0 and a current exit_time
- CLOSED trades are untouched
- dry_run=True produces no DB mutations
- Running twice is idempotent (second run finds 0 OPEN trades)
- NULL signal_snapshot is handled gracefully
- Malformed JSON signal_snapshot is wrapped rather than lost
- cleanup marker is appended to signal_snapshot on every fixed row

@decision DEC-TEST-CLEANUP-001
@title Tests for fix_orphaned_trades using in-memory SQLite
@status accepted
@rationale fix_orphaned_trades uses raw sqlite3. Tests use an in-memory SQLite
DB with seeded trade data to verify mutation correctness without touching the
production database. Follows DEC-TEST-016 / DEC-ANALYZE-002 pattern. All
assertions read back from the DB after the function returns — no inspection of
internal state.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.fix_orphaned_trades import _CLEANUP_MARKER, fix_orphaned_trades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    exit_time TEXT,
    exit_price TEXT,
    quantity TEXT NOT NULL,
    pnl TEXT,
    signal_snapshot TEXT,
    regime TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    created_at TEXT NOT NULL,
    strategy_id TEXT
)
"""

_INSERT_TRADE = """
INSERT INTO trades
    (symbol, side, entry_time, entry_price, quantity,
     pnl, signal_snapshot, regime, status, created_at, strategy_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _make_db(tmp_path: Path, open_count: int = 5, closed_count: int = 3) -> Path:
    """
    Create a temp SQLite DB with the full trades schema.

    Seeds `open_count` OPEN trades and `closed_count` CLOSED trades with
    distinct symbols so tests can identify rows after the fix.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        open_count: Number of OPEN trades to seed.
        closed_count: Number of CLOSED trades to seed.

    Returns:
        Path to the created DB file.
    """
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_CREATE_TRADES)

    for i in range(open_count):
        conn.execute(
            _INSERT_TRADE,
            (
                f"BTC/USD",
                "buy",
                f"2026-03-01T{i:02d}:00:00+00:00",  # entry_time
                "50000.00",
                "0.002",
                None,    # pnl — NULL for open trades
                json.dumps({"combined": {"strength": 0.7, "action": "buy"}}),
                "SIDEWAYS",
                "OPEN",
                "2026-03-01T00:00:00+00:00",
                f"momentum_{i}",
            ),
        )

    for i in range(closed_count):
        conn.execute(
            _INSERT_TRADE,
            (
                "ETH/USD",
                "buy",
                f"2026-03-02T{i:02d}:00:00+00:00",
                "2000.00",
                "0.05",
                "3.50",  # pnl — already set
                json.dumps({"combined": {"strength": 0.8, "action": "buy"}}),
                "BULL",
                "CLOSED",
                "2026-03-02T00:00:00+00:00",
                "mean_reversion",
            ),
        )

    conn.commit()
    conn.close()
    return db


def _fetch_all(db: Path, status: str) -> list[dict]:
    """Return all trade rows matching `status` as list of dicts."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades WHERE status = ? ORDER BY id ASC", (status,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Test: normal execute path — 5 OPEN + 3 CLOSED
# ---------------------------------------------------------------------------


def test_fix_open_trades_are_closed(tmp_path):
    """5 OPEN trades must become CLOSED after fix with dry_run=False."""
    db = _make_db(tmp_path, open_count=5, closed_count=3)

    result = fix_orphaned_trades(db, dry_run=False)

    assert result["fixed"] == 5
    assert result["skipped"] == 0

    closed = _fetch_all(db, "CLOSED")
    # Originally 3 CLOSED + 5 newly closed = 8 total
    assert len(closed) == 8


def test_fix_sets_pnl_to_zero(tmp_path):
    """Every previously-OPEN row must have pnl='0' after fix."""
    db = _make_db(tmp_path, open_count=5, closed_count=0)

    fix_orphaned_trades(db, dry_run=False)

    closed = _fetch_all(db, "CLOSED")
    for row in closed:
        assert row["pnl"] == "0", f"row {row['id']} has pnl={row['pnl']!r}, expected '0'"


def test_fix_sets_exit_time(tmp_path):
    """Every previously-OPEN row must have a non-null exit_time after fix."""
    db = _make_db(tmp_path, open_count=3, closed_count=0)

    fix_orphaned_trades(db, dry_run=False)

    closed = _fetch_all(db, "CLOSED")
    for row in closed:
        assert row["exit_time"] is not None
        # Must be an ISO timestamp (contains 'T')
        assert "T" in row["exit_time"], f"row {row['id']} exit_time={row['exit_time']!r} not ISO"


def test_fix_appends_cleanup_marker_to_snapshot(tmp_path):
    """signal_snapshot must contain the cleanup marker keys after fix."""
    db = _make_db(tmp_path, open_count=2, closed_count=0)

    fix_orphaned_trades(db, dry_run=False)

    closed = _fetch_all(db, "CLOSED")
    for row in closed:
        snap = json.loads(row["signal_snapshot"])
        for key, val in _CLEANUP_MARKER.items():
            assert snap.get(key) == val, (
                f"row {row['id']} snapshot missing marker key {key!r}"
            )


def test_fix_preserves_existing_snapshot_data(tmp_path):
    """Original signal_snapshot keys must survive alongside the cleanup marker."""
    db = _make_db(tmp_path, open_count=1, closed_count=0)

    fix_orphaned_trades(db, dry_run=False)

    closed = _fetch_all(db, "CLOSED")
    snap = json.loads(closed[0]["signal_snapshot"])
    # Original key from _make_db seed
    assert "combined" in snap


def test_closed_trades_untouched(tmp_path):
    """Pre-existing CLOSED trades must not be modified by the fix."""
    db = _make_db(tmp_path, open_count=5, closed_count=3)

    # Capture CLOSED rows before fix
    before = _fetch_all(db, "CLOSED")
    assert len(before) == 3

    fix_orphaned_trades(db, dry_run=False)

    # The 3 originally-CLOSED rows must be byte-for-byte identical
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # They are the first 5 inserts = OPEN, next 3 = CLOSED (ids 6,7,8)
    after_ids = {r["id"] for r in _fetch_all(db, "CLOSED")}
    before_ids = {r["id"] for r in before}
    # The original 3 closed IDs must still be present
    assert before_ids.issubset(after_ids)

    # Verify their pnl was NOT touched (original value was '3.50')
    conn2 = sqlite3.connect(str(db))
    conn2.row_factory = sqlite3.Row
    for row in before:
        orig = dict(
            conn2.execute("SELECT pnl FROM trades WHERE id = ?", (row["id"],)).fetchone()
        )
        assert orig["pnl"] == "3.50", f"row {row['id']} pnl changed unexpectedly"
    conn2.close()
    conn.close()


# ---------------------------------------------------------------------------
# Test: dry_run=True does not mutate
# ---------------------------------------------------------------------------


def test_dry_run_does_not_modify(tmp_path):
    """dry_run=True must leave all OPEN trades untouched in the DB."""
    db = _make_db(tmp_path, open_count=5, closed_count=3)

    result = fix_orphaned_trades(db, dry_run=True)

    # Return value still reports what would have been fixed
    assert result["fixed"] == 5

    # DB must be unchanged — still 5 OPEN
    open_rows = _fetch_all(db, "OPEN")
    assert len(open_rows) == 5

    # And still 3 CLOSED
    closed_rows = _fetch_all(db, "CLOSED")
    assert len(closed_rows) == 3


def test_dry_run_open_pnl_unchanged(tmp_path):
    """dry_run=True must not set pnl on OPEN rows."""
    db = _make_db(tmp_path, open_count=3, closed_count=0)

    fix_orphaned_trades(db, dry_run=True)

    open_rows = _fetch_all(db, "OPEN")
    for row in open_rows:
        assert row["pnl"] is None, f"row {row['id']} pnl changed during dry-run"


# ---------------------------------------------------------------------------
# Test: idempotency
# ---------------------------------------------------------------------------


def test_idempotency(tmp_path):
    """Running fix twice: second run must return fixed=0, skipped=0."""
    db = _make_db(tmp_path, open_count=5, closed_count=0)

    first = fix_orphaned_trades(db, dry_run=False)
    assert first["fixed"] == 5

    second = fix_orphaned_trades(db, dry_run=False)
    assert second == {"fixed": 0, "skipped": 0}


def test_idempotency_dry_run(tmp_path):
    """After execute run, a subsequent dry-run also returns fixed=0."""
    db = _make_db(tmp_path, open_count=3, closed_count=0)

    fix_orphaned_trades(db, dry_run=False)
    result = fix_orphaned_trades(db, dry_run=True)

    assert result == {"fixed": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# Test: edge cases — NULL and malformed signal_snapshot
# ---------------------------------------------------------------------------


def test_null_signal_snapshot(tmp_path):
    """A NULL signal_snapshot must be treated as {} and the marker appended."""
    db = tmp_path / "null_snap.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_CREATE_TRADES)
    conn.execute(
        _INSERT_TRADE,
        (
            "BTC/USD", "buy", "2026-03-01T00:00:00+00:00",
            "50000.00", "0.001", None,
            None,  # NULL snapshot
            "SIDEWAYS", "OPEN", "2026-03-01T00:00:00+00:00", None,
        ),
    )
    conn.commit()
    conn.close()

    result = fix_orphaned_trades(db, dry_run=False)
    assert result["fixed"] == 1

    closed = _fetch_all(db, "CLOSED")
    snap = json.loads(closed[0]["signal_snapshot"])
    assert snap["cleanup"] == _CLEANUP_MARKER["cleanup"]


def test_malformed_json_snapshot(tmp_path):
    """A malformed JSON snapshot must be wrapped in _original key, not dropped."""
    db = tmp_path / "bad_snap.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_CREATE_TRADES)
    conn.execute(
        _INSERT_TRADE,
        (
            "BTC/USD", "buy", "2026-03-01T00:00:00+00:00",
            "50000.00", "0.001", None,
            "NOT_VALID_JSON",  # malformed
            "SIDEWAYS", "OPEN", "2026-03-01T00:00:00+00:00", None,
        ),
    )
    conn.commit()
    conn.close()

    result = fix_orphaned_trades(db, dry_run=False)
    assert result["fixed"] == 1

    closed = _fetch_all(db, "CLOSED")
    snap = json.loads(closed[0]["signal_snapshot"])
    # Original value preserved under _original key
    assert snap.get("_original") == "NOT_VALID_JSON"
    # Marker still appended
    assert snap["cleanup"] == _CLEANUP_MARKER["cleanup"]


# ---------------------------------------------------------------------------
# Test: empty database (no rows at all)
# ---------------------------------------------------------------------------


def test_empty_db(tmp_path):
    """A DB with no trades must return fixed=0 without error."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_CREATE_TRADES)
    conn.commit()
    conn.close()

    result = fix_orphaned_trades(db, dry_run=False)
    assert result == {"fixed": 0, "skipped": 0}


def test_only_closed_trades(tmp_path):
    """A DB with only CLOSED trades must return fixed=0 without touching them."""
    db = _make_db(tmp_path, open_count=0, closed_count=4)

    result = fix_orphaned_trades(db, dry_run=False)
    assert result == {"fixed": 0, "skipped": 0}

    closed = _fetch_all(db, "CLOSED")
    assert len(closed) == 4
    for row in closed:
        assert row["pnl"] == "3.50"
