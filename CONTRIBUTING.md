## Running Live Alpaca Tests

By default, Alpaca-dependent tests (`tests/live/`) are skipped. Opt-in with
the `--live-alpaca` flag:

```bash
pytest --live-alpaca
```

Requires:
- `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` in `.env` (from a paper
  Alpaca account at https://alpaca.markets — free tier).
- US market hours (09:30–16:00 ET, Mon–Fri, non-holiday) for the tests
  that stream real-time data.

These tests connect to Alpaca's live paper-trading endpoint. They DO NOT
place orders — only subscribe to market data and verify event shapes.

Live ORB smoke test (`tests/live/test_live_orb_smoke.py`) is manual-only:
run it during an RTH window when you want to verify full pipeline behavior
against real Alpaca data. Not in CI.
