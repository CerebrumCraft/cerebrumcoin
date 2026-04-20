"""
Unit tests for CongressionalLedger.

Uses an in-memory SQLite database (:memory:) to avoid touching
data/cerebrum.db and to keep tests fast and isolated.

Coverage:
  1. record() returns True for new filing_id
  2. record() returns False for duplicate filing_id (no re-insert)
  3. has_seen() returns True after record(), False before

@decision DEC-TEST-LEDGER-001
@title Use :memory: SQLite for CongressionalLedger tests
@status accepted
@rationale Keeps tests hermetic — no dependency on data/cerebrum.db path.
:memory: is automatically discarded after each test. CongressionalLedger
accepts any Path; passing Path(":memory:") lets sqlite3 use the in-memory
backend, which is the correct isolation boundary for unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cerebrum.data.congressional_ledger import CongressionalLedger


@pytest.fixture
def ledger() -> CongressionalLedger:
    """Return a fresh in-memory ledger for each test."""
    return CongressionalLedger(db_path=Path(":memory:"))


# ---------------------------------------------------------------------------
# 1. record() → True for new filing
# ---------------------------------------------------------------------------


def test_record_new_filing_returns_true(ledger: CongressionalLedger) -> None:
    """First record() call for a filing_id must return True."""
    result = ledger.record(
        filing_id="filing-001",
        symbol="NVDA",
        filing_date="2026-04-01",
        action="stock_buy",
    )
    assert result is True


# ---------------------------------------------------------------------------
# 2. record() → False for duplicate filing_id
# ---------------------------------------------------------------------------


def test_record_duplicate_returns_false(ledger: CongressionalLedger) -> None:
    """Second record() call with the same filing_id must return False."""
    ledger.record(
        filing_id="filing-001",
        symbol="NVDA",
        filing_date="2026-04-01",
        action="stock_buy",
    )
    result = ledger.record(
        filing_id="filing-001",
        symbol="NVDA",
        filing_date="2026-04-01",
        action="stock_buy",
    )
    assert result is False


# ---------------------------------------------------------------------------
# 3. has_seen() — False before, True after
# ---------------------------------------------------------------------------


def test_has_seen_false_before_record(ledger: CongressionalLedger) -> None:
    """has_seen() must return False for a filing_id not yet recorded."""
    assert ledger.has_seen("filing-unknown") is False


def test_has_seen_true_after_record(ledger: CongressionalLedger) -> None:
    """has_seen() must return True after record() is called."""
    ledger.record(
        filing_id="filing-002",
        symbol="AVGO",
        filing_date="2026-04-05",
        action="call_buy",
    )
    assert ledger.has_seen("filing-002") is True
