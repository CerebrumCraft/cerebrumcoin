#!/usr/bin/env python3
"""Record Alpaca tick stream to JSONL. Bootstrap helper for ORB integration test fixtures.

Usage:
  python scripts/record_alpaca_ticks.py --symbols AAPL,MSFT,NVDA \\
      --start 09:30 --end 16:00 --date 2026-03-10 \\
      --out tests/fixtures/alpaca_mixed_stocks_2026-03-10.jsonl

Requires ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY in environment (load .env first).

@decision DEC-STOCKS-004
@title Fixture-recording helper for ORB integration tests
@status accepted
@rationale Integration tests need realistic AAPL tick streams for ORB
replay coverage. This helper records live IEX data during RTH to JSONL,
one quote per line. Tests load the JSONL via `pipeline.publish_market_data()`
and assert on resulting strategy snapshots. Refresh fixtures annually.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


async def _record(symbols: list[str], out_path: str, start_et: str, end_et: str, date_str: str) -> int:
    try:
        from alpaca.data.live.stock import StockDataStream
    except ImportError:
        print("alpaca-py not installed. Run: pip install alpaca-py", file=sys.stderr)
        return 2

    api_key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not secret:
        print("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in environment", file=sys.stderr)
        return 2

    stream = StockDataStream(api_key, secret, feed="iex")

    start_dt = datetime.fromisoformat(f"{date_str}T{start_et}:00").replace(tzinfo=ET)
    end_dt = datetime.fromisoformat(f"{date_str}T{end_et}:00").replace(tzinfo=ET)

    tick_count = 0
    out_file = open(out_path, "w")

    async def on_quote(data):
        nonlocal tick_count
        ts = getattr(data, "timestamp", None) or datetime.now(tz=ET)
        ts_et = ts.astimezone(ET)
        if not (start_dt <= ts_et <= end_dt):
            return
        record = {
            "symbol": getattr(data, "symbol", None),
            "bid": str(getattr(data, "bid_price", "")),
            "ask": str(getattr(data, "ask_price", "")),
            "bid_size": str(getattr(data, "bid_size", "")),
            "ask_size": str(getattr(data, "ask_size", "")),
            "timestamp": ts.isoformat(),
        }
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        tick_count += 1

    stream.subscribe_quotes(on_quote, *symbols)

    print(f"Recording {symbols} from {start_et} to {end_et} ET on {date_str} → {out_path}",
          file=sys.stderr)
    try:
        # Alpaca SDK provides .run() or ._run_forever() — use whichever is current
        if hasattr(stream, "run"):
            await stream.run()
        else:
            await stream._run_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. Recorded {tick_count} ticks to {out_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        out_file.close()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Record Alpaca ticks to JSONL.")
    ap.add_argument("--symbols", required=True, help="comma-separated list, e.g. AAPL,MSFT,NVDA")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (ET)")
    ap.add_argument("--start", default="09:30", help="HH:MM ET (default 09:30)")
    ap.add_argument("--end", default="16:00", help="HH:MM ET (default 16:00)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided.", file=sys.stderr)
        return 2

    return asyncio.run(_record(symbols, args.out, args.start, args.end, args.date))


if __name__ == "__main__":
    sys.exit(main())
