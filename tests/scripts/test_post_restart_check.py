"""
Regression tests for scripts/post_restart_check.sh — Check 3 (Global equity vs
per-strategy sum) with v4 state schema snapshot positions.

@decision DEC-CHECK-002
@title Test coverage for post_restart_check.sh Check 3 with v4 snapshot positions
@status accepted
@rationale Check 3 iterated snapshot positions with float(qty) but v4 schema stores
positions as dicts {amount, average_entry_price, ...} not raw strings. This raised
TypeError whenever any strategy held an open position, silently aborting the check.
Tests use a synthetic fake-worktree (temp dir + .git pointer to real repo) to invoke
the script via subprocess without requiring a live session. Three cases: open position
passes, no positions passes, equity mismatch fails.

In the v4 schema (CURRENT_STATE_VERSION=4), strategy snapshot positions are stored
as dicts:
    {symbol: {amount, average_entry_price, current_price, realized_pnl, entry_time}}

Prior to the fix, Check 3 called float(qty) where qty was a dict, raising TypeError
whenever any strategy held an open position.

These tests:
1. Verify Check 3 passes for a v4 state with an open position (was TypeError before fix).
2. Verify Check 3 passes for a v4 state with no open positions (regression guard).
3. Verify Check 3 fails (exit non-zero) when equity is genuinely mismatched.

The script is invoked via subprocess with a synthetic fake-worktree directory
containing a minimal .git file (so git ancestry checks work) and a
data/paper_state.json fixture.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "post_restart_check.sh"
GIT_COMMON_DIR = REPO_ROOT / ".git"

# Commit that Check 1 requires to be an ancestor of HEAD.
# This is the DEC-EQUITY-002 commit baked into the real repo history.
EQUITY_FIX_COMMIT = "c26da8a"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_worktree(tmp_path: Path, state: dict) -> Path:
    """
    Build a minimal directory that post_restart_check.sh treats as a valid worktree.

    - Creates a .git file pointing at the real repo common dir so that
      'git -C <dir> rev-parse HEAD' returns the real HEAD and Check 1 passes.
    - Writes state as data/paper_state.json.
    - Omits logs/ and cerebrum.db so Checks 4 and 5 are SKIP (not FAIL).
    """
    wt = tmp_path / "fake_worktree"
    wt.mkdir()

    # Point git at the real repo so ancestry checks work.
    # Format: "gitdir: <absolute-path-to-worktrees/NAME/.git>" or the main .git dir.
    # Simplest: point directly at the live worktree's git metadata.
    live_wt_git = REPO_ROOT / ".worktrees" / "fix-snapshot-positions" / ".git"
    if live_wt_git.exists():
        git_target = live_wt_git
    else:
        git_target = GIT_COMMON_DIR
    (wt / ".git").write_text(f"gitdir: {git_target}\n")

    data_dir = wt / "data"
    data_dir.mkdir()
    (data_dir / "paper_state.json").write_text(json.dumps(state, indent=2))

    return wt


def _run_check(worktree: Path) -> subprocess.CompletedProcess:
    """Run post_restart_check.sh against the given worktree directory."""
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), str(worktree)],
        capture_output=True,
        text=True,
    )


def _make_v4_state_with_open_position() -> dict:
    """
    Synthetic v4 state where mean_reversion holds 0.1 ETH/USD @ $2000.

    Equity math (Check 3):
      - Adapter: cash=9800 + 0.1 ETH @ $2000 = 9800 + 200 = 10000
      - mean_reversion snapshot: cash=4800 + 0.1 ETH @ $2000 = 4800 + 200 = 5000
      - range_trading snapshot: cash=5000 (no positions)
      - strategy_total = 5000 + 5000 = 10000
      - diff = 0.0 <= 1.0 tolerance => OK
    """
    return {
        "version": 4,
        "balances": {"USD": "9800.0"},
        # Top-level positions: flat {sym: qty_string} (adapter ledger, unchanged in v4)
        "positions": {
            "ETH/USD": "0.1"
        },
        "current_prices": {
            "ETH/USD": "2000.0"
        },
        "strategy_snapshots": {
            "mean_reversion": {
                "cash_balance": "4800.0",
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                # v4 schema: dict with amount + metadata
                "positions": {
                    "ETH/USD": {
                        "amount": "0.1",
                        "average_entry_price": "2000.0",
                        "current_price": "2000.0",
                        "realized_pnl": "0.0",
                        "entry_time": 1777396701.687334,
                    }
                },
                "closed_trades": [],
            },
            "range_trading": {
                "cash_balance": "5000.0",
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                "positions": {},
                "closed_trades": [],
            },
        },
    }


def _make_v4_state_no_open_positions() -> dict:
    """
    Synthetic v4 state where no strategy holds any open position.

    Equity math (Check 3):
      - Adapter: cash=10000 (no open positions)
      - mean_reversion: cash=5000
      - range_trading: cash=5000
      - strategy_total = 10000 = adapter_total => OK
    """
    return {
        "version": 4,
        "balances": {"USD": "10000.0"},
        "positions": {},
        "current_prices": {"ETH/USD": "2000.0"},
        "strategy_snapshots": {
            "mean_reversion": {
                "cash_balance": "5000.0",
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                "positions": {},
                "closed_trades": [],
            },
            "range_trading": {
                "cash_balance": "5000.0",
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                "positions": {},
                "closed_trades": [],
            },
        },
    }


def _make_v4_state_equity_mismatch() -> dict:
    """
    Synthetic v4 state where strategy equity > adapter equity by $500.
    Check 3 should FAIL (diff > 1.0 tolerance).
    """
    return {
        "version": 4,
        "balances": {"USD": "9500.0"},   # adapter has only $9500
        "positions": {},
        "current_prices": {},
        "strategy_snapshots": {
            "mean_reversion": {
                "cash_balance": "5000.0",  # strategies together claim $10000
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                "positions": {},
                "closed_trades": [],
            },
            "range_trading": {
                "cash_balance": "5000.0",
                "initial_balance": "5000.0",
                "peak_equity": "5000.0",
                "total_realized_pnl": "0.0",
                "positions": {},
                "closed_trades": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheck3SnapshotPositions:
    """Check 3: Global equity vs per-strategy sum, with v4 snapshot positions."""

    def test_open_position_as_dict_passes_check3(self, tmp_path):
        """
        Primary regression: v4 state with an open position stored as a dict
        must NOT raise TypeError and must produce [OK] for Check 3.

        Before the fix: float(qty) with qty=dict => TypeError => [FAIL] / crash.
        After the fix: qty["amount"] is read correctly => [OK].
        """
        state = _make_v4_state_with_open_position()
        wt = _make_fake_worktree(tmp_path, state)
        result = _run_check(wt)

        combined_output = result.stdout + result.stderr
        assert "TypeError" not in combined_output, (
            f"TypeError raised — fix not applied.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Check 3" in result.stdout, (
            f"Check 3 section missing from output:\n{result.stdout}"
        )
        # The check 3 section should contain [OK]
        check3_section = _extract_check3_output(result.stdout)
        assert "[OK]" in check3_section, (
            f"Check 3 did not pass.\nCheck 3 output:\n{check3_section}\n"
            f"Full stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "[FAIL]" not in check3_section, (
            f"Check 3 unexpectedly failed.\nCheck 3 output:\n{check3_section}"
        )

    def test_no_open_positions_passes_check3(self, tmp_path):
        """
        Regression guard: the no-position path (empty snapshot positions dict)
        must still pass Check 3 after the fix.
        """
        state = _make_v4_state_no_open_positions()
        wt = _make_fake_worktree(tmp_path, state)
        result = _run_check(wt)

        combined_output = result.stdout + result.stderr
        assert "TypeError" not in combined_output, (
            f"TypeError raised:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        check3_section = _extract_check3_output(result.stdout)
        assert "[OK]" in check3_section, (
            f"Check 3 failed for no-positions case.\nCheck 3 output:\n{check3_section}\n"
            f"Full stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_equity_mismatch_fails_check3(self, tmp_path):
        """
        Sanity check: a genuine equity mismatch (diff > $1) must still produce
        [FAIL] and exit non-zero, confirming the check is not vacuously passing.
        """
        state = _make_v4_state_equity_mismatch()
        wt = _make_fake_worktree(tmp_path, state)
        result = _run_check(wt)

        check3_section = _extract_check3_output(result.stdout)
        assert "[FAIL]" in check3_section, (
            f"Expected [FAIL] for equity mismatch but got:\n{check3_section}\n"
            f"Full stdout:\n{result.stdout}"
        )
        # Overall script should exit non-zero when any check fails
        assert result.returncode != 0, (
            f"Expected non-zero exit for equity mismatch but got returncode={result.returncode}"
        )


# ---------------------------------------------------------------------------
# Helper: extract the Check 3 output block from full script stdout
# ---------------------------------------------------------------------------

def _extract_check3_output(stdout: str) -> str:
    """
    Return the portion of stdout between the Check 3 header and the next
    '--- Check' header (or end of output).
    """
    lines = stdout.splitlines()
    in_check3 = False
    section = []
    for line in lines:
        if "Check 3" in line and "---" in line:
            in_check3 = True
            section.append(line)
            continue
        if in_check3:
            if line.startswith("---") and "Check 3" not in line:
                break
            section.append(line)
    return "\n".join(section)
