#!/usr/bin/env python3
"""
Export signal weight history to CSV.

Reads the weight_history table from cerebrum.db and writes a flat CSV with
one row per weight record. Useful for visualising how the adaptive learner
adjusts signal weights over time across different market regimes.

@decision DEC-EXPORT-003
@title Weight history CSV exporter with ISO timestamps
@status accepted
@rationale Signal weight evolution is key data for understanding adaptive
learning behaviour. The weight_history table stores unix timestamps; converting
to ISO strings makes the CSV directly usable in analysis tools. Pure stdlib
sqlite3 keeps the script dependency-free.

Usage:
    python scripts/export_weights.py \\
        --db data/cerebrum.db \\
        --output tmp/weights.csv
"""

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = ["id", "signal_type", "regime", "weight", "timestamp"]


def unix_to_iso(ts: float | None) -> str:
    """
    Convert a unix timestamp float to an ISO 8601 string (UTC).

    Args:
        ts: Unix timestamp float.

    Returns:
        ISO string like "2026-03-14T04:07:28+00:00", or "" if ts is None.
    """
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def export_weights(db_path: Path, output_path: Path) -> int:
    """
    Export weight_history table to CSV.

    Args:
        db_path: Path to the SQLite database.
        output_path: Path to write the CSV output file.

    Returns:
        Number of rows written.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, signal_type, regime, weight, timestamp
            FROM weight_history
            ORDER BY timestamp ASC
            """
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in cursor:
                row_id, signal_type, regime, weight, ts = row
                writer.writerow({
                    "id": row_id,
                    "signal_type": signal_type,
                    "regime": regime,
                    "weight": weight,
                    "timestamp": unix_to_iso(ts),
                })
                count += 1

    finally:
        conn.close()

    return count


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export signal weight history to CSV."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to SQLite database (default: data/cerebrum.db)",
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

    count = export_weights(args.db, args.output)
    print(f"Exported {count} rows to {args.output}")


if __name__ == "__main__":
    main()
