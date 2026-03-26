#!/usr/bin/env python3
"""
Automated weekly report writer for CerebrumCoin strategy attribution.

Auto-detects week number from the earliest trade timestamp and writes a
markdown report to data/reports/week-{N}.md. Creates the reports directory
if needed.

@decision DEC-ANALYZE-003
@title weekly_report.py delegates to generate_report() from analyze.py
@status accepted
@rationale Code reuse without circular imports: weekly_report imports the
pure generate_report() function from analyze.py rather than shelling out.
Week number is auto-detected from earliest trade timestamp to avoid manual
tracking. Follows DEC-EXPORT-001 pattern: raw sqlite3, no cerebrum imports.

Usage:
    python3 scripts/weekly_report.py --db data/cerebrum.db
    python3 scripts/weekly_report.py --db data/cerebrum.db --balance 10000
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze import fetch_all_closed_trades, generate_report


def _detect_week_number(db_path: Path) -> int:
    """
    Detect the week number from the earliest trade timestamp.

    Week 1 starts at the earliest trade. Each subsequent 7-day period
    is a new week. Returns the current week number based on how many
    full 7-day periods have elapsed since the first trade.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Current week number (1-indexed). Returns 1 if no trades found.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT MIN(entry_time), MAX(entry_time) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()
    finally:
        conn.close()

    if not row or row[0] is None:
        return 1

    earliest = float(row[0])
    latest = float(row[1]) if row[1] is not None else earliest

    # Now is approximated from the latest trade to stay deterministic in tests
    elapsed_seconds = latest - earliest
    week_number = int(elapsed_seconds / (7 * 86400)) + 1
    return max(1, week_number)


def write_weekly_report(
    db_path: Path,
    output_dir: Path,
    initial_balance: Decimal = Decimal("10000"),
) -> Path:
    """
    Generate and write a weekly markdown report.

    Auto-detects the week number from trade history, then calls generate_report()
    from analyze.py. Writes to output_dir/week-{N}.md.

    Args:
        db_path: Path to the SQLite database.
        output_dir: Directory to write the report file.
        initial_balance: Starting balance for drawdown calculation.

    Returns:
        Path to the written report file.
    """
    week_number = _detect_week_number(db_path)
    output_path = output_dir / f"week-{week_number}.md"

    generate_report(
        db_path,
        strategy_only=False,
        as_json=False,
        output_path=output_path,
        initial_balance=initial_balance,
    )

    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Write an automated weekly strategy attribution report."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to SQLite database (default: data/cerebrum.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports"),
        help="Directory to write weekly reports (default: data/reports)",
    )
    parser.add_argument(
        "--balance",
        type=Decimal,
        default=Decimal("10000"),
        help="Initial balance for drawdown calculation (default: 10000).",
    )

    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")

    output_path = write_weekly_report(args.db, args.output_dir, args.balance)
    print(f"Weekly report written to: {output_path}")


if __name__ == "__main__":
    main()
