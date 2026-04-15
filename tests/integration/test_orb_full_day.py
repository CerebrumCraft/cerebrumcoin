"""Replay integration test — full AAPL trading day through the ORB pipeline.

Task 14. Requires a live-recorded fixture:
  tests/fixtures/alpaca_aapl_2026-03-10.jsonl

To produce the fixture:
  python scripts/record_alpaca_ticks.py --symbols AAPL --date YYYY-MM-DD \\
      --start 09:30 --end 16:00 --out tests/fixtures/alpaca_aapl_YYYY-MM-DD.jsonl
"""
import pytest


@pytest.mark.skip(reason="needs live-recorded fixture — see scripts/record_alpaca_ticks.py")
def test_orb_full_day_aapl():
    pass
