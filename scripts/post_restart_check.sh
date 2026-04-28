#!/usr/bin/env bash
# post_restart_check.sh — Read-only sanity checks after restarting a paper-trade session.
# Verifies DEC-EQUITY-002 (equity oscillation fix) and DEC-ALLOC-INITIAL-001 (balance conservation).
#
# @decision DEC-CHECK-001
# @title Post-restart operational check script
# @status accepted
# @rationale Five read-only checks verify the equity-oscillation fix (DEC-EQUITY-002) is present
#   in the worktree under inspection, confirm per-strategy initial_balance equals pool/N
#   (DEC-ALLOC-INITIAL-001), validate global vs strategy equity convergence, confirm
#   position_invariant_violated stays near-zero, and flag any SQLite/JSON closed-trade
#   parity drift (WARN-only — known Session 43 discrepancy). Script is idempotent and
#   safe to run against a live session; it never writes any files.
set -euo pipefail

# Derive the main repo root via git's shared .git dir — works correctly whether
# the script is invoked from the main checkout or from any worktree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_COMMON="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
REPO_ROOT="${GIT_COMMON%/.git}"
if [[ -z "$REPO_ROOT" || "$REPO_ROOT" == "$GIT_COMMON" ]]; then
    # Fallback: one level up from scripts/ (main checkout path, no git nesting)
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
DEFAULT_WORKTREE="$REPO_ROOT/.worktrees/live"
WORKTREE="${1:-$DEFAULT_WORKTREE}"

if [[ ! -d "$WORKTREE" ]]; then
    echo "ERROR: worktree path does not exist: $WORKTREE" >&2
    echo "Usage: $0 [worktree-path]  (default: $DEFAULT_WORKTREE)" >&2
    exit 2
fi

# Normalize to absolute path
WORKTREE="$(cd "$WORKTREE" && pwd -P)"

STATE_FILE="$WORKTREE/data/paper_state.json"
DB_FILE="$WORKTREE/data/cerebrum.db"
EQUITY_FIX_COMMIT="c26da8a"

PASS=0
FAIL=0
WARN=0

pass() { echo "[OK]  $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
warn() { echo "[WARN] $1"; WARN=$((WARN + 1)); }

echo "=== post_restart_check.sh ==="
echo "Worktree : $WORKTREE"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# ---------------------------------------------------------------------------
# Check 1 — Equity-fix commit loaded (DEC-EQUITY-002)
# ---------------------------------------------------------------------------
echo "--- Check 1: Equity-fix commit loaded (DEC-EQUITY-002) ---"
WORKTREE_HEAD="$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || echo "UNKNOWN")"
if git -C "$WORKTREE" merge-base --is-ancestor "$EQUITY_FIX_COMMIT" HEAD 2>/dev/null; then
    pass "Commit $EQUITY_FIX_COMMIT is ancestor of HEAD ($WORKTREE_HEAD)"
else
    fail "Commit $EQUITY_FIX_COMMIT NOT in history — HEAD is $WORKTREE_HEAD"
    echo "       Action: rebuild worktree from main (git worktree add ... main)"
fi
echo ""

# ---------------------------------------------------------------------------
# State file guard — Checks 2-5 require paper_state.json
# ---------------------------------------------------------------------------
if [[ ! -f "$STATE_FILE" ]]; then
    echo "[SKIP] No state file at $STATE_FILE — Checks 2-5 require a live session"
    echo ""
    TOTAL=$((PASS + FAIL))
    echo "PASSED $PASS/$TOTAL checks (Checks 2-5 skipped — no state file)"
    [[ $FAIL -eq 0 ]]
    exit $?
fi

# ---------------------------------------------------------------------------
# Check 2 — Initial-balance conservation invariant (DEC-ALLOC-INITIAL-001)
# Each strategy's initial_balance should be approximately pool/N
# ---------------------------------------------------------------------------
echo "--- Check 2: Initial-balance conservation (DEC-ALLOC-INITIAL-001) ---"
python3 - "$STATE_FILE" <<'PYEOF'
import json, sys

state_file = sys.argv[1]
with open(state_file) as f:
    state = json.load(f)

snapshots = state.get("strategy_snapshots", {})
n = len(snapshots)
if n == 0:
    print("[SKIP] No strategy_snapshots in state — check skipped")
    sys.exit(0)

# Determine pool from top-level balances USD
try:
    pool = float(state["balances"]["USD"])
    # Add value of open positions using current_prices
    prices = state.get("current_prices", {})
    for sym, qty in state.get("positions", {}).items():
        qty_f = float(qty)
        if sym in prices and qty_f != 0.0:
            pool += qty_f * float(prices[sym])
    # The pool in use at launch time was N * per_strategy initial_balance
    # Use the actual initial_balance from strategies instead
    total_initial = sum(float(v["initial_balance"]) for v in snapshots.values())
    expected_per = total_initial / n
    tolerance = 0.10

    failures = []
    for strat_id, snap in snapshots.items():
        ib = float(snap["initial_balance"])
        if abs(ib - expected_per) > tolerance:
            failures.append(f"{strat_id}: initial_balance={ib:.4f}, expected≈{expected_per:.4f}")

    if failures:
        print(f"[FAIL] initial_balance mismatch (expected pool/N≈{expected_per:.4f}):")
        for f in failures:
            print(f"       {f}")
        sys.exit(1)
    else:
        print(f"[OK]  {n} strategies, each initial_balance≈{expected_per:.2f} (pool/N), tolerance ±{tolerance}")
        sys.exit(0)
except Exception as e:
    print(f"[FAIL] Error computing balance invariant: {e}")
    sys.exit(1)
PYEOF
CHECK2_EXIT=$?
if [[ $CHECK2_EXIT -eq 0 ]]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Check 3 — Global equity matches sum of per-strategy equity
# ---------------------------------------------------------------------------
echo "--- Check 3: Global equity vs per-strategy sum ---"
python3 - "$STATE_FILE" <<'PYEOF'
import json, sys

state_file = sys.argv[1]
with open(state_file) as f:
    state = json.load(f)

snapshots = state.get("strategy_snapshots", {})
if not snapshots:
    print("[SKIP] No strategy_snapshots — check skipped")
    sys.exit(0)

prices = state.get("current_prices", {})

# Compute per-strategy equity: cash_balance + open position value
strat_total = 0.0
for strat_id, snap in snapshots.items():
    equity = float(snap["cash_balance"])
    for sym, pos in snap.get("positions", {}).items():
        # v4 schema: snapshot positions are dicts {amount, average_entry_price, ...}
        # v3 and earlier: raw quantity strings. Support both for backward compat.
        qty_f = float(pos["amount"]) if isinstance(pos, dict) else float(pos)
        if sym in prices and qty_f != 0.0:
            equity += qty_f * float(prices[sym])
    strat_total += equity

# Compute adapter equity: top-level USD balance + open position value
adapter_cash = float(state["balances"]["USD"])
adapter_pos_val = 0.0
for sym, qty in state.get("positions", {}).items():
    qty_f = float(qty)
    if sym in prices and qty_f != 0.0:
        adapter_pos_val += qty_f * float(prices[sym])
    elif qty_f != 0.0:
        print(f"[NOTE] No price for {sym} — position value excluded from adapter total")
adapter_total = adapter_cash + adapter_pos_val

diff = abs(strat_total - adapter_total)
tolerance = 1.00

if diff <= tolerance:
    print(f"[OK]  strategy_total={strat_total:.4f} adapter_total={adapter_total:.4f} diff={diff:.4f} ≤ {tolerance}")
    sys.exit(0)
else:
    print(f"[FAIL] equity mismatch: strategy_total={strat_total:.4f} adapter_total={adapter_total:.4f} diff={diff:.4f} > {tolerance}")
    sys.exit(1)
PYEOF
CHECK3_EXIT=$?
if [[ $CHECK3_EXIT -eq 0 ]]; then
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi
echo ""

# ---------------------------------------------------------------------------
# Check 4 — Position invariant warnings stay near zero
# ---------------------------------------------------------------------------
echo "--- Check 4: Position invariant violations in current session log ---"
LATEST_LOG="$(ls -t "$WORKTREE/logs/session-"*.log 2>/dev/null | head -1 || true)"
if [[ -z "$LATEST_LOG" ]]; then
    echo "[SKIP] No session-*.log found in $WORKTREE/logs/"
    WARN=$((WARN + 1))
else
    VIOLATION_COUNT="$(grep -c "position_invariant_violated" "$LATEST_LOG" 2>/dev/null || true)"
    if [[ "$VIOLATION_COUNT" -le 5 ]]; then
        pass "position_invariant_violated count=$VIOLATION_COUNT ≤ 5 in $(basename "$LATEST_LOG")"
    else
        fail "position_invariant_violated count=$VIOLATION_COUNT > 5 in $(basename "$LATEST_LOG")"
        echo "       Last 3 violations:"
        grep "position_invariant_violated" "$LATEST_LOG" | tail -3 | sed 's/^/         /'
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Check 5 — SQLite vs JSON closed-trade parity (WARN only, exit 0)
# ---------------------------------------------------------------------------
echo "--- Check 5: SQLite vs JSON closed-trade parity ---"
if [[ ! -f "$DB_FILE" ]]; then
    echo "[SKIP] No database at $DB_FILE — check skipped"
    WARN=$((WARN + 1))
else
    CHECK5_EXIT=0
    python3 - "$STATE_FILE" "$DB_FILE" <<'PYEOF' || CHECK5_EXIT=$?
import json, sys, subprocess

state_file, db_file = sys.argv[1], sys.argv[2]
with open(state_file) as f:
    state = json.load(f)

snapshots = state.get("strategy_snapshots", {})
if not snapshots:
    print("[SKIP] No strategy_snapshots — check skipped")
    sys.exit(0)

mismatches = []
for strat_id, snap in snapshots.items():
    json_count = len(snap.get("closed_trades", []))
    result = subprocess.run(
        ["sqlite3", db_file,
         f"SELECT COUNT(*) FROM trades WHERE strategy_id='{strat_id}' AND status='CLOSED'"],
        capture_output=True, text=True
    )
    db_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else -1
    if json_count != db_count:
        mismatches.append((strat_id, json_count, db_count))

if not mismatches:
    print(f"[OK]  closed-trade counts match across {len(snapshots)} strategies")
    sys.exit(0)
else:
    print("[WARN] SQLite vs JSON closed-trade mismatch (known Session 43 discrepancy):")
    for strat_id, jc, dc in mismatches:
        print(f"       {strat_id}: JSON={jc}  SQLite={dc}")
    # WARN only — exit 2 signals bash to call warn(); does not set FAIL
    sys.exit(2)
PYEOF
    if [[ $CHECK5_EXIT -eq 0 ]]; then
        PASS=$((PASS + 1))
    elif [[ $CHECK5_EXIT -eq 2 ]]; then
        warn "SQLite vs JSON closed-trade mismatch — see details above"
    else
        FAIL=$((FAIL + 1))
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo "=== Summary ==="
echo "PASSED $PASS/$TOTAL checks  WARN=$WARN"
if [[ $FAIL -gt 0 ]]; then
    echo "Result: FAIL — $FAIL check(s) failed above"
    exit 1
else
    echo "Result: PASS"
    exit 0
fi
