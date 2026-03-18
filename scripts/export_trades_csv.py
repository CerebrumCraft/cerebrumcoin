#!/usr/bin/env python3
"""
Export closed trades to CSV for parameter tuning analysis.

Reads the trades table from cerebrum.db and writes a flat CSV with one row
per closed trade. The JSON signal_snapshot column is destructured into
individual columns for easy analysis in pandas, Excel, or R.

@decision DEC-EXPORT-002
@title Trades CSV exporter with flattened signal_snapshot columns
@status accepted
@rationale Parameter tuning requires flat tabular data. The JSON signal_snapshot
is destructured into signal_strength, signal_confidence, signal_action columns.
ISO timestamps replace unix floats for human readability. hold_duration_min is
derived from entry/exit timestamps. exit_reason column is present in headers
but empty (not stored in the trades table schema).

Usage:
    python scripts/export_trades_csv.py \\
        --db data/cerebrum.db \\
        --output tmp/trades.csv

    # Filter to a specific session:
    python scripts/export_trades_csv.py \\
        --db data/cerebrum.db \\
        --session-start 1773461248 \\
        --output tmp/session6_trades.csv
"""

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


# CSV column order matches the spec exactly
CSV_COLUMNS = [
    "id",
    "symbol",
    "side",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "pnl",
    "regime",
    "hold_duration_min",
    "exit_reason",
    "signal_strength",
    "signal_confidence",
    "signal_action",
    "status",
]


def unix_to_iso(ts: float | None) -> str:
    """
    Convert a unix timestamp float to an ISO 8601 string (UTC).

    Args:
        ts: Unix timestamp, or None for open/missing exit times.

    Returns:
        ISO string like "2026-03-14T04:07:28+00:00", or "" if ts is None.
    """
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def parse_session_start(value: str) -> float:
    """
    Parse --session-start as a unix timestamp float or ISO datetime string.

    Args:
        value: Unix timestamp or ISO datetime string.

    Returns:
        Unix timestamp as float.
    """
    try:
        return float(value)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    raise argparse.ArgumentTypeError(
        f"Cannot parse session-start '{value}'. "
        "Use a unix timestamp or ISO datetime (e.g. 2026-03-14T04:00:00)."
    )


def row_to_csv_dict(row: tuple) -> dict:
    """
    Convert a single trades row to a flat dict for CSV output.

    Destructures signal_snapshot JSON, converts timestamps to ISO strings,
    and computes hold_duration_min.

    Args:
        row: Tuple of (id, symbol, side, entry_time, exit_time, entry_price,
             exit_price, quantity, pnl, signal_snapshot, regime, status).

    Returns:
        Flat dict with keys matching CSV_COLUMNS.
    """
    (
        trade_id, symbol, side, entry_time, exit_time,
        entry_price, exit_price, quantity, pnl,
        signal_snapshot_raw, regime, status,
    ) = row

    # Parse signal snapshot
    try:
        snapshot = json.loads(signal_snapshot_raw)
    except (json.JSONDecodeError, TypeError):
        snapshot = {}

    combined = snapshot.get("combined", {})
    signal_strength = combined.get("strength", "")
    signal_confidence = combined.get("confidence", "")
    signal_action = combined.get("action", "")

    # Hold duration in minutes (None if exit not recorded)
    hold_duration_min = ""
    if entry_time is not None and exit_time is not None:
        try:
            hold_duration_min = f"{(float(exit_time) - float(entry_time)) / 60:.2f}"
        except (ValueError, TypeError):
            hold_duration_min = ""

    return {
        "id": trade_id,
        "symbol": symbol,
        "side": side,
        "entry_time": unix_to_iso(entry_time),
        "exit_time": unix_to_iso(exit_time),
        "entry_price": entry_price,
        "exit_price": exit_price if exit_price is not None else "",
        "quantity": quantity,
        "pnl": pnl if pnl is not None else "",
        "regime": regime,
        "hold_duration_min": hold_duration_min,
        "exit_reason": "",  # not stored in trades schema
        "signal_strength": signal_strength,
        "signal_confidence": signal_confidence,
        "signal_action": signal_action,
        "status": status,
    }


def export_trades_csv(
    db_path: Path,
    output_path: Path,
    session_start: float | None = None,
) -> int:
    """
    Export closed trades to CSV.

    Args:
        db_path: Path to the SQLite database.
        output_path: Path to write the CSV output file.
        session_start: Optional unix timestamp filter — only trades with
                       entry_time >= session_start are included. If None,
                       all closed trades are exported.

    Returns:
        Number of rows written.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        if session_start is not None:
            cursor = conn.execute(
                """
                SELECT id, symbol, side, entry_time, exit_time,
                       entry_price, exit_price, quantity, pnl,
                       signal_snapshot, regime, status
                FROM trades
                WHERE status = 'CLOSED'
                  AND entry_time >= ?
                ORDER BY entry_time ASC
                """,
                (session_start,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, symbol, side, entry_time, exit_time,
                       entry_price, exit_price, quantity, pnl,
                       signal_snapshot, regime, status
                FROM trades
                WHERE status = 'CLOSED'
                ORDER BY entry_time ASC
                """
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in cursor:
                writer.writerow(row_to_csv_dict(row))
                count += 1

    finally:
        conn.close()

    return count


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export closed trades to CSV for parameter tuning."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to SQLite database (default: data/cerebrum.db)",
    )
    parser.add_argument(
        "--session-start",
        type=parse_session_start,
        default=None,
        help=(
            "Optional unix timestamp or ISO datetime filter. "
            "Only trades with entry_time >= this value are included. "
            "Omit to export all closed trades."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the CSV file.",
    )

    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")

    count = export_trades_csv(args.db, args.output, session_start=args.session_start)
    print(f"Exported {count} rows to {args.output}")


if __name__ == "__main__":
    main()
