"""
Unit tests for export scripts: export_finetune, export_trades_csv, export_weights.

Uses in-memory SQLite databases seeded with known data to verify:
- JSONL output is valid JSON per line
- JSONL records have correct structure (system/user/assistant messages)
- CSV output has correct headers
- CSV rows have correct column counts and values
- session-start filtering works (only trades >= timestamp included)
- hold_duration_min is computed correctly
- ISO timestamp conversion is correct
- weight_history export produces correct headers and row count

@decision DEC-TEST-016
@title Tests for export scripts using in-memory SQLite
@status accepted
@rationale Export scripts use raw sqlite3. Tests use in-memory SQLite DB with
seeded trade data to verify JSONL validity, CSV headers, and record counts
without touching the production database.
"""

import csv
import io
import json
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

# Import the testable functions directly (not via CLI)
from scripts.export_finetune import build_fine_tune_record, export_finetune, format_duration
from scripts.export_trades_csv import CSV_COLUMNS, export_trades_csv, row_to_csv_dict, unix_to_iso
from scripts.export_weights import export_weights


# ---------------------------------------------------------------------------
# Fixtures — in-memory (for unit tests) and temp-file (for CLI functions)
# ---------------------------------------------------------------------------


def make_trades_db(path: Path) -> None:
    """Create a minimal trades table and seed test data."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE trades (
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
        """
    )
    # Three closed trades, one open
    conn.executemany(
        """
        INSERT INTO trades
            (symbol, side, entry_time, entry_price, exit_time, exit_price,
             quantity, pnl, signal_snapshot, regime, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            # Trade 1: BTC winner, entry_time=1000
            (
                "BTC/USD", "buy", 1000.0, "50000.00",
                1004000.0, "51500.00",
                "0.004", "6.00",
                '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
                "BULL", "CLOSED",
            ),
            # Trade 2: ETH loser, entry_time=2000
            (
                "ETH/USD", "buy", 2000.0, "2000.00",
                2003600.0, "1980.00",
                "0.05", "-1.00",
                '{"combined": {"strength": 0.6, "confidence": 0.65, "action": "buy"}}',
                "SIDEWAYS", "CLOSED",
            ),
            # Trade 3: BTC trade, entry_time=3000 (used for session-start filtering)
            (
                "BTC/USD", "buy", 3000.0, "52000.00",
                3007200.0, "53000.00",
                "0.003", "3.00",
                '{"combined": {"strength": 0.7, "confidence": 0.70, "action": "buy"}}',
                "BULL", "CLOSED",
            ),
            # Trade 4: OPEN — must be excluded from exports
            (
                "BTC/USD", "buy", 9000.0, "55000.00",
                None, None,
                "0.002", None,
                '{"combined": {"strength": 0.5, "confidence": 0.60, "action": "buy"}}',
                "UNKNOWN", "OPEN",
            ),
        ],
    )
    conn.commit()
    conn.close()


def make_weights_db(path: Path) -> None:
    """Create a minimal weight_history table and seed test data."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT NOT NULL,
            regime TEXT NOT NULL,
            weight TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO weight_history (signal_type, regime, weight, timestamp) VALUES (?, ?, ?, ?)",
        [
            ("rsi", "BULL", "0.8", 1000.0),
            ("macd", "BULL", "0.6", 2000.0),
            ("rsi", "SIDEWAYS", "0.4", 3000.0),
        ],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests: format_duration
# ---------------------------------------------------------------------------


def test_format_duration_hours():
    assert "hours" in format_duration(7200.0)
    assert "2.0" in format_duration(7200.0)


def test_format_duration_minutes():
    assert "minutes" in format_duration(600.0)
    assert "10.0" in format_duration(600.0)


def test_format_duration_seconds():
    result = format_duration(45.0)
    assert "seconds" in result
    assert "45" in result


# ---------------------------------------------------------------------------
# Tests: build_fine_tune_record
# ---------------------------------------------------------------------------


def test_build_fine_tune_record_structure():
    """Record must have exactly 3 messages: system, user, assistant."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "BULL",
    )
    record = build_fine_tune_record(row)
    assert "messages" in record
    assert len(record["messages"]) == 3
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_build_fine_tune_record_win_label():
    """Positive PnL must produce WIN label in assistant content."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "BULL",
    )
    record = build_fine_tune_record(row)
    assert "WIN" in record["messages"][2]["content"]


def test_build_fine_tune_record_loss_label():
    """Negative PnL must produce LOSS label in assistant content."""
    row = (
        2, "ETH/USD", "buy", 2000.0, 5600.0,
        "2000.00", "1980.00", "0.05", "-1.00",
        '{"combined": {"strength": 0.6, "confidence": 0.65, "action": "buy"}}',
        "SIDEWAYS",
    )
    record = build_fine_tune_record(row)
    assert "LOSS" in record["messages"][2]["content"]


def test_build_fine_tune_record_regime_in_user():
    """Regime must appear in user message content."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "SIDEWAYS",
    )
    record = build_fine_tune_record(row)
    assert "SIDEWAYS" in record["messages"][1]["content"]


def test_build_fine_tune_record_malformed_snapshot():
    """Malformed signal_snapshot JSON must not raise — fall back to defaults."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        "NOT_VALID_JSON",
        "BULL",
    )
    record = build_fine_tune_record(row)
    assert "messages" in record
    assert len(record["messages"]) == 3


def test_build_fine_tune_record_none_exit():
    """None exit_time/exit_price/pnl (open trade) must produce OPEN outcome."""
    row = (
        4, "BTC/USD", "buy", 9000.0, None,
        "55000.00", None, "0.002", None,
        '{"combined": {"strength": 0.5, "confidence": 0.60, "action": "buy"}}',
        "UNKNOWN",
    )
    record = build_fine_tune_record(row)
    assert "OPEN" in record["messages"][2]["content"]


# ---------------------------------------------------------------------------
# Tests: export_finetune (file-based)
# ---------------------------------------------------------------------------


def test_export_finetune_count(tmp_path):
    """export_finetune returns correct record count (only CLOSED trades)."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "out.jsonl"

    count = export_finetune(db, session_start=0.0, output_path=output)
    assert count == 3  # 3 CLOSED trades, 1 OPEN excluded


def test_export_finetune_valid_jsonl(tmp_path):
    """Every line in the output must be valid JSON with a 'messages' key."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "out.jsonl"

    export_finetune(db, session_start=0.0, output_path=output)

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)  # raises if invalid JSON
        assert "messages" in record
        assert len(record["messages"]) == 3


def test_export_finetune_session_start_filter(tmp_path):
    """session_start filter must exclude trades with entry_time < threshold."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "out.jsonl"

    # Only trade 3 has entry_time >= 2500
    count = export_finetune(db, session_start=2500.0, output_path=output)
    assert count == 1

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert "BTC/USD" in record["messages"][1]["content"]


def test_export_finetune_creates_parent_dirs(tmp_path):
    """Output file parent directories must be created if they don't exist."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "deep" / "nested" / "out.jsonl"

    export_finetune(db, session_start=0.0, output_path=output)
    assert output.exists()


# ---------------------------------------------------------------------------
# Tests: unix_to_iso
# ---------------------------------------------------------------------------


def test_unix_to_iso_basic():
    """unix_to_iso must produce a valid ISO string for a known timestamp."""
    result = unix_to_iso(0.0)
    assert result == "1970-01-01T00:00:00+00:00"


def test_unix_to_iso_none():
    """unix_to_iso must return empty string for None input."""
    assert unix_to_iso(None) == ""


# ---------------------------------------------------------------------------
# Tests: row_to_csv_dict
# ---------------------------------------------------------------------------


def test_row_to_csv_dict_columns():
    """row_to_csv_dict must return all expected CSV columns."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "BULL", "CLOSED",
    )
    result = row_to_csv_dict(row)
    for col in CSV_COLUMNS:
        assert col in result, f"Missing column: {col}"


def test_row_to_csv_dict_hold_duration():
    """hold_duration_min must be (exit_time - entry_time) / 60."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 1000.0 + 7200.0,  # 2 hours = 120 min
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "BULL", "CLOSED",
    )
    result = row_to_csv_dict(row)
    assert result["hold_duration_min"] == "120.00"


def test_row_to_csv_dict_signal_columns():
    """Signal columns must be extracted from signal_snapshot JSON."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.71, "confidence": 0.65, "action": "buy"}}',
        "BULL", "CLOSED",
    )
    result = row_to_csv_dict(row)
    assert result["signal_strength"] == 0.71
    assert result["signal_confidence"] == 0.65
    assert result["signal_action"] == "buy"


def test_row_to_csv_dict_exit_reason_empty():
    """exit_reason must always be empty string (not in DB schema)."""
    row = (
        1, "BTC/USD", "buy", 1000.0, 4600.0,
        "50000.00", "51500.00", "0.004", "6.00",
        '{"combined": {"strength": 0.8, "confidence": 0.75, "action": "buy"}}',
        "BULL", "CLOSED",
    )
    result = row_to_csv_dict(row)
    assert result["exit_reason"] == ""


# ---------------------------------------------------------------------------
# Tests: export_trades_csv (file-based)
# ---------------------------------------------------------------------------


def test_export_trades_csv_headers(tmp_path):
    """CSV output must have exactly the expected column headers."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "trades.csv"

    export_trades_csv(db, output)

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS


def test_export_trades_csv_row_count(tmp_path):
    """export_trades_csv returns count of CLOSED rows only."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "trades.csv"

    count = export_trades_csv(db, output)
    assert count == 3


def test_export_trades_csv_session_start_filter(tmp_path):
    """session_start filter must exclude trades before the threshold."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "trades.csv"

    count = export_trades_csv(db, output, session_start=2500.0)
    assert count == 1

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTC/USD"


def test_export_trades_csv_iso_timestamps(tmp_path):
    """entry_time and exit_time must be ISO strings, not raw floats."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "trades.csv"

    export_trades_csv(db, output)

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    # ISO strings contain 'T' separator and '+' or 'Z' timezone
    assert "T" in first_row["entry_time"]
    assert "T" in first_row["exit_time"]


def test_export_trades_csv_no_session_start_exports_all(tmp_path):
    """Omitting session_start exports all CLOSED trades."""
    db = tmp_path / "test.db"
    make_trades_db(db)
    output = tmp_path / "trades.csv"

    count = export_trades_csv(db, output, session_start=None)
    assert count == 3


# ---------------------------------------------------------------------------
# Tests: export_weights (file-based)
# ---------------------------------------------------------------------------


def test_export_weights_headers(tmp_path):
    """Weights CSV must have the expected column headers."""
    db = tmp_path / "test.db"
    make_weights_db(db)
    output = tmp_path / "weights.csv"

    export_weights(db, output)

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["id", "signal_type", "regime", "weight", "timestamp"]


def test_export_weights_row_count(tmp_path):
    """export_weights returns correct row count."""
    db = tmp_path / "test.db"
    make_weights_db(db)
    output = tmp_path / "weights.csv"

    count = export_weights(db, output)
    assert count == 3


def test_export_weights_iso_timestamps(tmp_path):
    """timestamp column must be ISO strings."""
    db = tmp_path / "test.db"
    make_weights_db(db)
    output = tmp_path / "weights.csv"

    export_weights(db, output)

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    assert "T" in first_row["timestamp"]


def test_export_weights_values(tmp_path):
    """Values in exported CSV must match seeded data."""
    db = tmp_path / "test.db"
    make_weights_db(db)
    output = tmp_path / "weights.csv"

    export_weights(db, output)

    with output.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # First row: rsi, BULL, 0.8 (sorted by timestamp ASC)
    assert rows[0]["signal_type"] == "rsi"
    assert rows[0]["regime"] == "BULL"
    assert rows[0]["weight"] == "0.8"
