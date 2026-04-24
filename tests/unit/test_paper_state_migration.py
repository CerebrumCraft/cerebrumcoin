"""Test state file migration v2 → v3 (DEC-STOCKS-006) and v3 → v4 (DEC-CONDUCTOR-012)."""
import json
from pathlib import Path

import pytest

from cerebrum.adapters.paper import migrate_state_v2_to_v3, migrate_state_v3_to_v4


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
