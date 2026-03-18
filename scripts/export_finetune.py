#!/usr/bin/env python3
"""
Export closed trades as JSONL for LLM fine-tuning.

Reads the trades table from cerebrum.db and writes one JSON record per closed
trade in OpenAI fine-tuning format (system/user/assistant message triplets).

Each record teaches the model to map market context (regime, signal strength,
signal confidence, entry price) to a trading decision and its outcome, enabling
supervised fine-tuning on real trade history.

@decision DEC-EXPORT-001
@title LLM fine-tuning JSONL exporter using raw sqlite3 (no ORM)
@status accepted
@rationale Export scripts are standalone tools that run outside the full
cerebrum system. Using raw sqlite3 avoids importing asyncio-based StateManager
and all its transitive dependencies. Pure stdlib keeps the scripts deployable
anywhere Python 3.10+ is available without a venv.

Usage:
    python scripts/export_finetune.py \\
        --db data/cerebrum.db \\
        --session-start 0 \\
        --output tmp/finetune.jsonl
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def parse_session_start(value: str) -> float:
    """
    Parse --session-start as either a unix timestamp float or ISO datetime string.

    Args:
        value: Unix timestamp (e.g. "1773461248") or ISO datetime
               (e.g. "2026-03-14T04:00:00" or "2026-03-14 04:00:00").

    Returns:
        Unix timestamp as float.
    """
    try:
        return float(value)
    except ValueError:
        pass

    # Try ISO datetime formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    raise argparse.ArgumentTypeError(
        f"Cannot parse session-start '{value}'. "
        "Use a unix timestamp (e.g. 1773461248) or ISO datetime (e.g. 2026-03-14T04:00:00)."
    )


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string like "10.6 hours" or "45.2 minutes".
    """
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    elif seconds >= 60:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds:.0f} seconds"


def build_fine_tune_record(row: tuple) -> dict:
    """
    Convert a single trades row into an OpenAI fine-tuning JSONL record.

    Args:
        row: Tuple of (id, symbol, side, entry_time, exit_time, entry_price,
             exit_price, quantity, pnl, signal_snapshot, regime).

    Returns:
        Dict with "messages" key containing system/user/assistant message list.
    """
    (
        trade_id, symbol, side, entry_time, exit_time,
        entry_price, exit_price, quantity, pnl, signal_snapshot_raw, regime,
    ) = row

    # Parse signal snapshot — gracefully handle malformed JSON
    try:
        snapshot = json.loads(signal_snapshot_raw)
    except (json.JSONDecodeError, TypeError):
        snapshot = {}

    combined = snapshot.get("combined", {})
    strength = combined.get("strength", 0.0)
    confidence = combined.get("confidence", 0.0)
    action = combined.get("action", side)

    # Format prices as floats for readability
    try:
        entry_price_f = float(entry_price)
        exit_price_f = float(exit_price) if exit_price is not None else None
        quantity_f = float(quantity)
        pnl_f = float(pnl) if pnl is not None else None
    except (ValueError, TypeError):
        entry_price_f = 0.0
        exit_price_f = None
        quantity_f = 0.0
        pnl_f = None

    # Calculate hold duration
    if exit_time is not None and entry_time is not None:
        hold_seconds = float(exit_time) - float(entry_time)
        hold_str = format_duration(hold_seconds)
    else:
        hold_str = "unknown"

    # User content: market context at time of trade
    user_content = (
        f"Market context for {symbol}:\n"
        f"- Regime: {regime}\n"
        f"- Signal strength: {strength:.2f}\n"
        f"- Signal confidence: {confidence:.2f}\n"
        f"- Signal action: {action}\n"
        f"- Entry price: ${entry_price_f:,.2f}"
    )

    # Assistant content: decision and outcome
    if pnl_f is not None and exit_price_f is not None:
        outcome_label = "WIN" if pnl_f >= 0 else "LOSS"
        assistant_content = (
            f"Decision: {action.upper()} @ ${entry_price_f:,.2f}, "
            f"size {quantity_f:.5f} {symbol.split('/')[0]}\n\n"
            f"Outcome: {outcome_label} {'+' if pnl_f >= 0 else ''}{pnl_f:.2f} "
            f"(hold time: {hold_str}, exit: ${exit_price_f:,.2f})"
        )
    else:
        assistant_content = (
            f"Decision: {action.upper()} @ ${entry_price_f:,.2f}, "
            f"size {quantity_f:.5f} {symbol.split('/')[0]}\n\n"
            f"Outcome: OPEN (hold time: {hold_str})"
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a crypto trading assistant. Given market conditions, "
                    "decide whether to BUY, SELL, or HOLD. Explain your reasoning."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]
    }


def export_finetune(db_path: Path, session_start: float, output_path: Path) -> int:
    """
    Export closed trades to JSONL fine-tuning format.

    Args:
        db_path: Path to the SQLite database.
        session_start: Unix timestamp — only trades with entry_time >= this are included.
        output_path: Path to write the JSONL output file.

    Returns:
        Number of records exported.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, symbol, side, entry_time, exit_time,
                   entry_price, exit_price, quantity, pnl,
                   signal_snapshot, regime
            FROM trades
            WHERE status = 'CLOSED'
              AND entry_time >= ?
            ORDER BY entry_time ASC
            """,
            (session_start,),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with output_path.open("w", encoding="utf-8") as f:
            for row in cursor:
                record = build_fine_tune_record(row)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    finally:
        conn.close()

    return count


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export closed trades as JSONL for LLM fine-tuning."
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
        default="0",
        help=(
            "Unix timestamp or ISO datetime. Only trades with entry_time >= "
            "this value are included. Default: 0 (all trades)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the JSONL file.",
    )

    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")

    count = export_finetune(args.db, args.session_start, args.output)
    print(f"Exported {count} records to {args.output}")


if __name__ == "__main__":
    main()
