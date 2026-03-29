#!/usr/bin/env python3
"""
One-time cleanup script: mark orphaned OPEN trades as CLOSED.

Before commit 4802ee3 the paper adapter correlated sell fills to open positions
in memory, but never called trade_tracker.close_trade(). This left all sell-side
fills unrecorded in SQLite, stranding 39 trades in status='OPEN' indefinitely.

This script is idempotent: running it a second time finds zero OPEN trades and
does nothing. It is safe to run against production data — use --dry-run first
to preview changes, then --execute to commit them.

@decision DEC-CLEANUP-001
@title One-time orphaned-trade fix using pure stdlib sqlite3
@status accepted
@rationale The cleanup must not import any cerebrum module because StateManager
is async and requires an event loop. A self-contained stdlib script can be run
outside the venv by any operator without spinning up the full system. The
signal_snapshot JSON is patched in Python (not SQL) so that NULL and malformed
values are both handled gracefully. Dry-run mode is the default to prevent
accidental data mutation.

Usage:
    # Preview what would be changed (safe, default):
    python3 scripts/fix_orphaned_trades.py --db data/cerebrum.db --dry-run

    # Apply the fix:
    python3 scripts/fix_orphaned_trades.py --db data/cerebrum.db --execute
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


# Marker appended to signal_snapshot so the fix is auditable in the DB.
_CLEANUP_MARKER = {
    "cleanup": "orphaned_trade_fix",
    "reason": "pre_sell_correlation_fix",
}


def fix_orphaned_trades(db_path: Path, dry_run: bool = True) -> dict[str, int]:
    """
    Fix orphaned OPEN trades by marking them CLOSED with pnl=0.

    Trades are considered orphaned when status='OPEN' and the corresponding
    sell fill was never recorded (bug fixed in commit 4802ee3). Each affected
    row receives:
      - status    = 'CLOSED'
      - pnl       = '0'
      - exit_time = current UTC ISO timestamp
      - signal_snapshot: the existing JSON (or {}) with _CLEANUP_MARKER merged in

    Args:
        db_path: Path to the SQLite database file.
        dry_run: When True (default), print the preview table but do NOT commit
                 any changes. When False, commit the UPDATE.

    Returns:
        dict with keys:
          "fixed"   -- number of rows updated (or that would be updated in dry-run)
          "skipped" -- always 0 from this query; included for call-site symmetry
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT id, symbol, side, entry_time, entry_price,
                   signal_snapshot, regime, strategy_id
            FROM trades
            WHERE status = 'OPEN'
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()

        if not rows:
            _print_summary([], dry_run)
            return {"fixed": 0, "skipped": 0}

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        updates: list[tuple] = []

        for row in rows:
            # Patch signal_snapshot: preserve existing data, append marker.
            raw = row["signal_snapshot"]
            try:
                snapshot: dict = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                # Wrap unreadable value so it is not silently discarded.
                snapshot = {"_original": str(raw)}
            snapshot.update(_CLEANUP_MARKER)
            new_snapshot = json.dumps(snapshot)

            updates.append((
                "CLOSED",    # status
                "0",         # pnl
                now_iso,     # exit_time
                new_snapshot,
                row["id"],
            ))

        _print_summary(rows, dry_run)

        if not dry_run:
            conn.executemany(
                """
                UPDATE trades
                SET status = ?,
                    pnl    = ?,
                    exit_time = ?,
                    signal_snapshot = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()
            print(f"\nCommitted: {len(updates)} rows updated.")
        else:
            print("\nDry-run complete. No changes written. Use --execute to apply.")

    finally:
        conn.close()

    return {"fixed": len(updates), "skipped": 0}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_summary(rows: list, dry_run: bool) -> None:
    """Print a human-readable summary table of affected trades."""
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"\n{'=' * 60}")
    print(f"  fix_orphaned_trades  [{mode}]")
    print(f"{'=' * 60}")

    if not rows:
        print("  No OPEN trades found -- nothing to fix.")
        print(f"{'=' * 60}\n")
        return

    print(f"  {'ID':>4}  {'Symbol':<10}  {'Side':<5}  {'Entry Time':<30}  {'Strategy'}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*5}  {'-'*30}  {'-'*20}")
    for row in rows:
        strategy = row["strategy_id"] or "(none)"
        entry = row["entry_time"] or ""
        print(f"  {row['id']:>4}  {row['symbol']:<10}  {row['side']:<5}  {str(entry):<30}  {strategy}")

    print(f"\n  Total OPEN trades found: {len(rows)}")
    if dry_run:
        print("  Action: would set status=CLOSED, pnl=0, exit_time=<now>")
    else:
        print("  Action: setting status=CLOSED, pnl=0, exit_time=<now>")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Mark orphaned OPEN trades as CLOSED (one-time data cleanup). "
            "Default is --dry-run; use --execute to commit changes."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to SQLite database (default: data/cerebrum.db)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Preview changes without writing anything (default).",
    )
    mode_group.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Commit the changes to the database.",
    )

    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")

    result = fix_orphaned_trades(args.db, dry_run=args.dry_run)
    print(f"\nResult: fixed={result['fixed']}, skipped={result['skipped']}")


if __name__ == "__main__":
    main()
