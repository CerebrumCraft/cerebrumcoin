"""
Test state file migration v2 → v3 (DEC-STOCKS-006) and v3 → v4 (DEC-CONDUCTOR-012).

@decision DEC-CONDUCTOR-012
@title Regression tests: closed_trades persist across _save_state / _load_state cycle
@status accepted
@rationale _save_state() previously hardcoded "version": 2 when writing strategy
snapshots, causing connect() on the next restart to re-run the v2→v3 migration which
replaces snapshots from scratch — wiping closed_trades and resetting Sharpe history.
These tests enforce: (a) _save_state writes CURRENT_STATE_VERSION, (b) closed_trades
survive a full save → reload cycle, (c) migration functions are no-ops on v4 input.
"""
import json
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock  # @mock-exempt: EventBus requires asyncio loop; pure state-layer tests don't need a live bus

import pytest

from cerebrum.adapters.paper import (
    CURRENT_STATE_VERSION,
    PaperTradingAdapter,
    migrate_state_v2_to_v3,
    migrate_state_v3_to_v4,
)
from cerebrum.core.types import Side
from cerebrum.risk.portfolio import PortfolioTracker


def _v2_state():
    return {
        "version": 2,
        "balances": {"USD": "9805.02"},
        "positions": {"SOL/USD": "2.107"},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "mean_reversion": {
                "cash_balance": "5000",
                "initial_balance": "5000",
                "peak_equity": "5100",
                "total_realized_pnl": "1.45",
                "positions": {},
            },
            "range_trading": {
                "cash_balance": "5000",
                "initial_balance": "5000",
                "peak_equity": "5000",
                "total_realized_pnl": "-1.88",
                "positions": {},
            },
        },
    }


def test_migration_adds_orb_stocks_snapshot(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrated = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    assert migrated["version"] == 3
    assert "orb_stocks" in migrated["strategy_snapshots"]
    assert migrated["strategy_snapshots"]["orb_stocks"]["cash_balance"] == "5000.0"
    assert migrated["strategy_snapshots"]["orb_stocks"]["initial_balance"] == "5000.0"
    assert migrated["strategy_snapshots"]["orb_stocks"]["positions"] == {}
    assert migrated["strategy_snapshots"]["orb_stocks"]["total_realized_pnl"] == "0"


def test_migration_preserves_existing_crypto_snapshots(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrated = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    assert migrated["strategy_snapshots"]["mean_reversion"]["cash_balance"] == "5000"
    assert migrated["strategy_snapshots"]["mean_reversion"]["total_realized_pnl"] == "1.45"
    assert migrated["strategy_snapshots"]["range_trading"]["total_realized_pnl"] == "-1.88"
    # Global state preserved too
    assert migrated["positions"]["SOL/USD"] == "2.107"
    assert migrated["balances"]["USD"] == "9805.02"


def test_migration_writes_backup(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    backup = tmp_path / "paper_state.v2.bak.json"
    assert backup.exists()
    assert json.loads(backup.read_text())["version"] == 2


def test_migration_is_idempotent_on_v3_input(tmp_path):
    already_v3 = {**_v2_state(), "version": 3}
    already_v3["strategy_snapshots"]["orb_stocks"] = {
        "cash_balance": "5000",
        "initial_balance": "5000",
        "peak_equity": "5000",
        "total_realized_pnl": "0",
        "positions": {},
    }
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(already_v3))

    result = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)

    # No duplicate orb_stocks, no backup file (already v3)
    assert result["version"] == 3
    backup = tmp_path / "paper_state.v2.bak.json"
    assert not backup.exists()
    # orb_stocks snapshot unchanged
    assert result["strategy_snapshots"]["orb_stocks"]["cash_balance"] == "5000"


# ---------------------------------------------------------------------------
# v3 → v4 migration tests (DEC-CONDUCTOR-012)
# ---------------------------------------------------------------------------


def _v3_state():
    """A v3 state file with two strategy snapshots (no closed_trades key)."""
    return {
        "version": 3,
        "balances": {"USD": "9805.02"},
        "positions": {},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "mean_reversion": {
                "cash_balance": "5000",
                "initial_balance": "5000",
                "peak_equity": "5100",
                "total_realized_pnl": "1.45",
                "positions": {},
            },
            "range_trading": {
                "cash_balance": "4900",
                "initial_balance": "5000",
                "peak_equity": "5000",
                "total_realized_pnl": "-1.88",
                "positions": {},
            },
        },
    }


def test_v3_to_v4_adds_closed_trades_key(tmp_path):
    """migrate_state_v3_to_v4 adds closed_trades: [] to every snapshot."""
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v3_state()))

    result = migrate_state_v3_to_v4(p)

    assert result["version"] == 4
    for name in ("mean_reversion", "range_trading"):
        assert "closed_trades" in result["strategy_snapshots"][name], (
            f"closed_trades missing from {name} snapshot"
        )
        assert result["strategy_snapshots"][name]["closed_trades"] == []


def test_v3_to_v4_preserves_existing_fields(tmp_path):
    """migrate_state_v3_to_v4 preserves all non-closed_trades snapshot fields verbatim."""
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v3_state()))

    result = migrate_state_v3_to_v4(p)

    # Snapshot values intact
    mr = result["strategy_snapshots"]["mean_reversion"]
    assert mr["cash_balance"] == "5000"
    assert mr["total_realized_pnl"] == "1.45"
    # Global state intact
    assert result["balances"]["USD"] == "9805.02"
    assert result["positions"] == {}


def test_v3_to_v4_writes_backup(tmp_path):
    """migrate_state_v3_to_v4 creates a .v3.bak backup before writing."""
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v3_state()))

    migrate_state_v3_to_v4(p)

    backup = tmp_path / "paper_state.v3.bak.json"
    assert backup.exists(), "Expected .v3.bak.json backup to be created"
    original = json.loads(backup.read_text())
    assert original["version"] == 3
    assert "closed_trades" not in original["strategy_snapshots"]["mean_reversion"]


def test_v3_to_v4_is_idempotent_on_v4_input(tmp_path):
    """migrate_state_v3_to_v4 is a no-op when file is already v4 (no backup written)."""
    already_v4 = _v3_state()
    already_v4["version"] = 4
    for snap in already_v4["strategy_snapshots"].values():
        snap["closed_trades"] = []

    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(already_v4))
    mtime_before = p.stat().st_mtime

    result = migrate_state_v3_to_v4(p)

    assert result["version"] == 4
    backup = tmp_path / "paper_state.v3.bak.json"
    assert not backup.exists(), "No backup should be written for already-v4 input"


def test_v3_to_v4_roundtrip_closed_trades_survive(tmp_path):
    """
    Full round-trip: write a v3 state, migrate to v4, populate closed_trades
    manually, save again, reload — closed_trades survive.

    This validates the end-to-end path that PortfolioTracker.save_snapshot()
    and restore_snapshot() would exercise in production after the migration.
    """
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v3_state()))

    # Step 1: migrate to v4
    data = migrate_state_v3_to_v4(p)
    assert data["version"] == 4

    # Step 2: simulate a session writing closed trades into the snapshot
    trade_entry = {
        "symbol": "BTC/USD",
        "side": "buy",
        "entry_time": 1_000_000.0,
        "entry_price": "50000",
        "exit_time": 1_003_600.0,
        "exit_price": "51000",
        "quantity": "0.1",
        "pnl": "100",
        "strategy_id": "mean_reversion",
    }
    data["strategy_snapshots"]["mean_reversion"]["closed_trades"].append(trade_entry)
    p.write_text(json.dumps(data, indent=2))

    # Step 3: reload and verify
    reloaded = json.loads(p.read_text())
    mr_trades = reloaded["strategy_snapshots"]["mean_reversion"]["closed_trades"]
    assert len(mr_trades) == 1
    assert mr_trades[0]["symbol"] == "BTC/USD"
    assert mr_trades[0]["pnl"] == "100"
    # Other snapshot untouched
    assert reloaded["strategy_snapshots"]["range_trading"]["closed_trades"] == []


# ---------------------------------------------------------------------------
# Restart-cycle regression tests (DEC-CONDUCTOR-012 companion fix)
# ---------------------------------------------------------------------------

_MOCK_BUS = MagicMock()  # @mock-exempt: pure state-layer tests; no event dispatch needed


def _make_adapter(state_file: Path) -> PaperTradingAdapter:
    """Construct a PaperTradingAdapter pointing at *state_file* (no asyncio required)."""
    return PaperTradingAdapter(
        bus=_MOCK_BUS,
        config={},
        initial_balance=Decimal("10000"),
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=state_file,
    )


def test_save_state_writes_current_version(tmp_path):
    """_save_state() must write CURRENT_STATE_VERSION, not a hardcoded literal.

    Regression for the defect where version=2 was hardcoded, causing migrations
    to re-run on every restart and wipe closed_trades.
    """
    state_file = tmp_path / "paper_state.json"
    adapter = _make_adapter(state_file)

    # Seed minimal state so there is something to save
    adapter._balances = {"USD": Decimal("10000")}
    adapter._positions = {}
    adapter._current_prices = {}
    adapter._trade_history = []

    # Register a portfolio so the version branch is taken
    tracker = PortfolioTracker(_MOCK_BUS, initial_balance=Decimal("5000"), strategy_id="s1")
    adapter.set_strategy_portfolios({"s1": tracker})

    adapter._save_state()

    on_disk = json.loads(state_file.read_text())
    assert on_disk["version"] == CURRENT_STATE_VERSION, (
        f"_save_state wrote version={on_disk['version']}, expected {CURRENT_STATE_VERSION}. "
        "Hardcoded version literals in _save_state cause migrations to re-run on restart."
    )


def test_closed_trades_survive_restart_cycle(tmp_path):
    """closed_trades written by save_snapshot() must still be present after reload.

    Full cycle: populate tracker with closed trades → _save_state → fresh adapter
    → _load_state → get_strategy_snapshot → restore_snapshot → check deque.

    This is the end-to-end regression for DEC-CONDUCTOR-012: if _save_state writes
    the wrong version, migrations wipe closed_trades on reload.
    """
    state_file = tmp_path / "paper_state.json"

    # --- Session A: build state and save ---
    adapter_a = _make_adapter(state_file)
    adapter_a._balances = {"USD": Decimal("9800")}
    adapter_a._positions = {}
    adapter_a._current_prices = {}
    adapter_a._trade_history = []

    tracker_a = PortfolioTracker(_MOCK_BUS, initial_balance=Decimal("5000"), strategy_id="mr")
    now = time.time()
    tracker_a._append_closed_trade(
        symbol="BTC/USD",
        side=Side.BUY,
        entry_time=now - 3600,
        entry_price=Decimal("60000"),
        exit_time=now - 3500,
        exit_price=Decimal("61000"),
        quantity=Decimal("0.01"),
        pnl=Decimal("10"),
    )
    adapter_a.set_strategy_portfolios({"mr": tracker_a})
    adapter_a._save_state()

    # Confirm the file is v4 (not v2)
    saved = json.loads(state_file.read_text())
    assert saved["version"] == CURRENT_STATE_VERSION
    assert "closed_trades" in saved["strategy_snapshots"]["mr"]
    assert len(saved["strategy_snapshots"]["mr"]["closed_trades"]) == 1

    # --- Session B: simulate restart — fresh adapter loads the file ---
    adapter_b = _make_adapter(state_file)
    adapter_b._load_state()

    snap = adapter_b.get_strategy_snapshot("mr")
    assert snap is not None, "Snapshot missing after reload"
    assert "closed_trades" in snap, "closed_trades key missing from reloaded snapshot"
    assert len(snap["closed_trades"]) == 1, (
        f"Expected 1 closed trade after restart, got {len(snap['closed_trades'])}"
    )

    # Restore into a fresh tracker and check the deque
    tracker_b = PortfolioTracker(_MOCK_BUS, initial_balance=Decimal("5000"), strategy_id="mr")
    tracker_b.restore_snapshot(snap)
    assert len(tracker_b._closed_trades) == 1
    assert tracker_b._closed_trades[0].pnl == Decimal("10")


def test_migrations_skip_on_v4_file(tmp_path):
    """Both migration functions must be no-ops when the file is already v4.

    Idempotency ensures a restart after the fix doesn't double-migrate or
    corrupt a file that was correctly saved as v4.
    """
    v4_state = {
        "version": 4,
        "balances": {"USD": "9800"},
        "positions": {},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "mr": {
                "cash_balance": "5000",
                "initial_balance": "5000",
                "peak_equity": "5100",
                "total_realized_pnl": "10",
                "positions": {},
                "closed_trades": [{"symbol": "BTC/USD", "pnl": "10"}],
            }
        },
    }
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(v4_state))

    # v2→v3 migration: file is v4, must return unchanged (no backup, no write)
    result_v3 = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    assert result_v3["version"] == 4
    assert not (tmp_path / "paper_state.v2.bak.json").exists()

    # v3→v4 migration: already v4, must return unchanged (no backup, no write)
    result_v4 = migrate_state_v3_to_v4(p)
    assert result_v4["version"] == 4
    assert not (tmp_path / "paper_state.v3.bak.json").exists()

    # closed_trades untouched
    reloaded = json.loads(p.read_text())
    assert reloaded["strategy_snapshots"]["mr"]["closed_trades"] == [{"symbol": "BTC/USD", "pnl": "10"}]
