"""Test state file migration v2 → v3 (DEC-STOCKS-006)."""
import json
from pathlib import Path

import pytest

from cerebrum.adapters.paper import migrate_state_v2_to_v3


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
