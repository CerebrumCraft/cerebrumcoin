#!/usr/bin/env bash
# @decision DEC-LAUNCH-001: sources $REPO/.env + $REPO/.venv to avoid per-session key re-exports
# Launch a paper-trade session from the CWD (repo root or worktree).
set -euo pipefail

REPO=/home/j/CerebrumCraft/CerebrumCoin
ENV_FILE="${REPO}/.env"
PY="${REPO}/.venv/bin/python3"

CWD="$(pwd -P)"
CONFIG="${CWD}/config/paper.toml"
[[ -f "$CONFIG"  ]] || { echo "no config/paper.toml at $CWD — run from repo root or a worktree" >&2; exit 2; }
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 2; }
[[ -x "$PY"       ]] || { echo "missing venv at $PY" >&2; exit 2; }

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${EXCHANGE_API_KEY:=${KRAKEN_API_KEY:-}}"
: "${EXCHANGE_API_SECRET:=${KRAKEN_API_SECRET:-}}"
export EXCHANGE_API_KEY EXCHANGE_API_SECRET

for var in ALPACA_API_KEY_ID ALPACA_API_SECRET_KEY KRAKEN_API_KEY KRAKEN_API_SECRET FINNHUB_API_KEY; do
  [[ -n "${!var:-}" ]] || { echo "env var $var is empty — check $ENV_FILE" >&2; exit 3; }
done

mkdir -p logs
BASENAME="${1:-session-auto-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="logs/${BASENAME}.log"
ERR="logs/${BASENAME}.stderr"
PID_FILE="logs/${BASENAME}.pid"

[[ -e "$LOG" || -e "$ERR" ]] && { echo "log files exist: $LOG / $ERR — pick a different basename" >&2; exit 4; }

nohup "$PY" -m cerebrum --config "$CONFIG" >"$LOG" 2>"$ERR" </dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "$PID_FILE"

echo "launched: pid=$PID cwd=$CWD log=$LOG stderr=$ERR pid_file=$PID_FILE"
