#!/usr/bin/env python3
"""
Parameter sensitivity analysis for CerebrumCoin.

Reads closed trades from cerebrum.db and simulates alternative exit parameters
to estimate how P&L would have changed under different settings.

Important limitation: We can't replay the full pipeline (signals -> risk ->
execution) — we only have completed trades, not the raw market data stream.
What we CAN do is analyse how different exit parameters would have changed
the OUTCOME of existing trades.

Simulation approach per trade:
- SL: If the trade lost more than alt_sl%, cap the loss at alt_sl%.
- TP: If the trade gained more than alt_tp%, cap the gain at alt_tp%.
- Age: If the trade was held longer than alt_max_age, linearly interpolate
       P&L to the alt_max_age exit time (directionally correct; acknowledged
       as rough).
- Cooldown: Filter out any trade whose entry_time is within cooldown_seconds
            of the previous trade for the same symbol.

Follows DEC-EXPORT-001 pattern: raw sqlite3, no ORM, no async cerebrum imports.
Imports TradeRow and calculate_commission from analyze.py (DEC-SENSITIVITY-001).

@decision DEC-SENSITIVITY-001
@title sensitivity.py imports TradeRow/calculate_commission from analyze.py
@status accepted
@rationale Reuses the stable TradeRow dataclass and round-trip commission
formula from analyze.py rather than duplicating. Simulation functions
(simulate_sl, simulate_tp, simulate_age, simulate_cooldown) are pure functions
that accept a TradeRow and return a simulated P&L Decimal — no side effects,
trivially testable without a database.

Usage:
    python3 scripts/sensitivity.py --db data/cerebrum.db
    python3 scripts/sensitivity.py --db data/cerebrum.db \\
        --stop-loss 0.5,1.0,1.5,2.0,2.5 \\
        --take-profit 1.0,2.0,3.0,4.0,5.0 \\
        --max-age 30,60,90,120,180,240
    python3 scripts/sensitivity.py --db data/cerebrum.db --json
    python3 scripts/sensitivity.py --db data/cerebrum.db --strategy momentum
"""

import argparse
import json
import sqlite3
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from scripts.analyze import (
    COMMISSION_RATE,
    TradeRow,
    calculate_commission,
    calculate_sharpe_from_rows,
)


# ---------------------------------------------------------------------------
# Constants — default parameter grids
# ---------------------------------------------------------------------------

DEFAULT_STOP_LOSS_PCT: list[Decimal] = [
    Decimal("0.5"),
    Decimal("0.8"),
    Decimal("1.0"),
    Decimal("1.5"),
    Decimal("2.0"),
    Decimal("2.5"),
    Decimal("3.0"),
]

DEFAULT_TAKE_PROFIT_PCT: list[Decimal] = [
    Decimal("1.0"),
    Decimal("1.5"),
    Decimal("2.0"),
    Decimal("3.0"),
    Decimal("4.0"),
    Decimal("5.0"),
]

DEFAULT_MAX_AGE_MINUTES: list[int] = [30, 60, 90, 120, 180, 240, 480]

GRID_CAP = 1000

# Current (baseline) parameter values from paper.toml
CURRENT_SL_PCT = Decimal("1.5")
CURRENT_TP_PCT = Decimal("3.0")
CURRENT_MAX_AGE_MIN = 120


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------

def _row_to_trade(row: tuple) -> TradeRow:
    """Convert a sqlite3 row tuple to a TradeRow."""
    (
        trade_id, symbol, side, entry_time, entry_price,
        exit_time, exit_price, quantity, pnl, regime, strategy_id,
    ) = row
    return TradeRow(
        id=trade_id,
        symbol=symbol,
        side=side,
        entry_time=float(entry_time),
        entry_price=Decimal(str(entry_price)),
        exit_time=float(exit_time) if exit_time is not None else None,
        exit_price=Decimal(str(exit_price)) if exit_price is not None else None,
        quantity=Decimal(str(quantity)),
        pnl=Decimal(str(pnl)) if pnl is not None else None,
        regime=regime or "UNKNOWN",
        strategy_id=strategy_id,
    )


def fetch_closed_trades_for_sensitivity(
    db_path: Path,
    strategy: str | None = None,
) -> list[TradeRow]:
    """
    Fetch all CLOSED trades with non-null pnl, ordered by entry_time ASC.

    Optionally filtered to a single strategy_id.

    Args:
        db_path: Path to the SQLite database file.
        strategy: If provided, only return trades with this strategy_id.

    Returns:
        List of TradeRow objects sorted by entry_time ascending.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        if strategy is not None:
            cursor = conn.execute(
                """
                SELECT id, symbol, side, entry_time, entry_price,
                       exit_time, exit_price, quantity, pnl, regime, strategy_id
                FROM trades
                WHERE status = 'CLOSED'
                  AND pnl IS NOT NULL
                  AND strategy_id = ?
                ORDER BY entry_time ASC
                """,
                (strategy,),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, symbol, side, entry_time, entry_price,
                       exit_time, exit_price, quantity, pnl, regime, strategy_id
                FROM trades
                WHERE status = 'CLOSED'
                  AND pnl IS NOT NULL
                ORDER BY entry_time ASC
                """
            )
        return [_row_to_trade(row) for row in cursor]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Simulation functions (pure — no I/O)
# ---------------------------------------------------------------------------

def simulate_sl(trade: TradeRow, alt_sl_pct: Decimal) -> Decimal:
    """
    Simulate an alternative stop-loss percentage on a single closed trade.

    If the trade's percentage loss exceeds alt_sl_pct, the trade would have
    been stopped at that level. Returns the simulated P&L.

    For winning trades or trades whose loss is within the alt SL threshold,
    the original P&L is returned unchanged.

    Args:
        trade: Closed TradeRow with non-null pnl.
        alt_sl_pct: Alternative stop-loss percentage (e.g. Decimal("1.5")
                    for 1.5%).

    Returns:
        Simulated P&L as Decimal.
    """
    if trade.pnl is None:
        return Decimal("0")

    entry_value = trade.entry_price * trade.quantity
    if entry_value == 0:
        return trade.pnl

    pnl_pct = trade.pnl / entry_value * Decimal("100")

    # Only applies to losing trades that exceed the alt SL
    sl_threshold = -alt_sl_pct
    if pnl_pct < sl_threshold:
        # Would have been stopped at alt_sl_pct
        return -(entry_value * alt_sl_pct / Decimal("100"))

    return trade.pnl


def simulate_tp(trade: TradeRow, alt_tp_pct: Decimal) -> Decimal:
    """
    Simulate an alternative take-profit percentage on a single closed trade.

    If the trade's percentage gain exceeds alt_tp_pct, the trade would have
    taken profit at that level. Returns the simulated P&L.

    For losing trades or trades whose gain is within the alt TP threshold,
    the original P&L is returned unchanged.

    Args:
        trade: Closed TradeRow with non-null pnl.
        alt_tp_pct: Alternative take-profit percentage (e.g. Decimal("2.0")
                    for 2.0%).

    Returns:
        Simulated P&L as Decimal.
    """
    if trade.pnl is None:
        return Decimal("0")

    entry_value = trade.entry_price * trade.quantity
    if entry_value == 0:
        return trade.pnl

    pnl_pct = trade.pnl / entry_value * Decimal("100")

    # Only applies to winning trades that exceed the alt TP
    if pnl_pct > alt_tp_pct:
        return entry_value * alt_tp_pct / Decimal("100")

    return trade.pnl


def simulate_age(trade: TradeRow, alt_max_age_minutes: int) -> Decimal:
    """
    Simulate an alternative max position age on a single closed trade.

    If the trade was held longer than alt_max_age_minutes, estimate the P&L
    at the alt exit time using linear interpolation:
        simulated_pnl = actual_pnl * (alt_max_age / actual_hold_duration)

    This is a rough directional estimate — acknowledged in the spec.
    Trades already shorter than alt_max_age are returned unchanged.

    Args:
        trade: Closed TradeRow with entry_time and exit_time.
        alt_max_age_minutes: Alternative maximum hold time in minutes.

    Returns:
        Simulated P&L as Decimal.
    """
    if trade.pnl is None:
        return Decimal("0")

    if trade.exit_time is None or trade.entry_time is None:
        return trade.pnl

    actual_seconds = trade.exit_time - trade.entry_time
    alt_seconds = alt_max_age_minutes * 60.0

    if actual_seconds <= 0:
        return trade.pnl

    if actual_seconds <= alt_seconds:
        # Trade exited before or exactly at the alt age limit — unchanged
        return trade.pnl

    # Linear interpolation
    ratio = Decimal(str(alt_seconds)) / Decimal(str(actual_seconds))
    return trade.pnl * ratio


def simulate_cooldown(
    rows: list[TradeRow],
    cooldown_seconds: int,
) -> list[TradeRow]:
    """
    Simulate an alternative cooldown period by filtering trades per symbol.

    For each symbol, keep only trades that occur at least cooldown_seconds
    after the previous kept trade for that symbol. Trades are processed in
    entry_time order (assumed sorted ascending by the DB query).

    Args:
        rows: List of TradeRows sorted by entry_time ASC.
        cooldown_seconds: Minimum gap between kept trades per symbol.

    Returns:
        Filtered list of TradeRows that survive the cooldown constraint.
    """
    if not rows:
        return []

    last_entry_time: dict[str, float] = {}
    kept: list[TradeRow] = []

    for row in rows:
        symbol = row.symbol
        last_t = last_entry_time.get(symbol)

        if last_t is None or (row.entry_time - last_t) >= cooldown_seconds:
            kept.append(row)
            last_entry_time[symbol] = row.entry_time

    return kept


# ---------------------------------------------------------------------------
# SimResult — aggregate statistics for a set of (possibly modified) trades
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    Aggregated statistics for a set of simulated trade outcomes.

    All P&L values are in USD. Commission is applied round-trip
    (0.16% * 2 = 0.32% of entry_value) per DEC-ANALYZE-001.
    """

    trade_count: int
    win_rate: Decimal
    total_pnl: Decimal
    adj_pnl: Decimal      # after commission
    sharpe: Decimal

    @classmethod
    def from_rows(cls, rows: list[TradeRow]) -> "SimResult":
        """
        Build a SimResult from a list of TradeRows.

        Each row's pnl is taken as already simulated (simulate_sl/tp/age
        should have been applied before calling this). Commission is
        calculated from entry_price * quantity * COMMISSION_RATE * 2.

        Args:
            rows: List of TradeRows with pnl reflecting simulated outcomes.

        Returns:
            SimResult aggregating the set.
        """
        if not rows:
            return cls(
                trade_count=0,
                win_rate=Decimal("0"),
                total_pnl=Decimal("0"),
                adj_pnl=Decimal("0"),
                sharpe=Decimal("0"),
            )

        total_pnl = Decimal("0")
        adj_pnl = Decimal("0")
        wins = 0

        for r in rows:
            if r.pnl is None:
                continue
            commission = calculate_commission(r.entry_price, r.quantity)
            total_pnl += r.pnl
            adj_pnl += r.pnl - commission
            if r.pnl > 0:
                wins += 1

        n = len(rows)
        win_rate = Decimal(str(wins)) / Decimal(str(n)) * Decimal("100") if n > 0 else Decimal("0")
        sharpe = calculate_sharpe_from_rows(rows)

        return cls(
            trade_count=n,
            win_rate=win_rate,
            total_pnl=total_pnl,
            adj_pnl=adj_pnl,
            sharpe=sharpe,
        )


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(
    sl_values: list[Decimal] | None,
    tp_values: list[Decimal] | None,
    age_values: list[int] | None,
) -> list[tuple[Decimal | None, Decimal | None, int | None]]:
    """
    Build the parameter combination grid.

    None means "don't sweep this dimension" — the dimension contributes a
    single (None,) placeholder, keeping the grid structure consistent.

    If total combinations exceed GRID_CAP (1000), emits a UserWarning and
    truncates the grid to GRID_CAP entries.

    Args:
        sl_values: Stop-loss percentages to sweep, or None.
        tp_values: Take-profit percentages to sweep, or None.
        age_values: Max position age values (minutes) to sweep, or None.

    Returns:
        List of (sl, tp, age) tuples. None entries are placeholders.
    """
    sl_dim: list[Decimal | None] = sl_values if sl_values is not None else [None]
    tp_dim: list[Decimal | None] = tp_values if tp_values is not None else [None]
    age_dim: list[int | None] = age_values if age_values is not None else [None]

    grid: list[tuple[Decimal | None, Decimal | None, int | None]] = []
    for sl in sl_dim:
        for tp in tp_dim:
            for age in age_dim:
                grid.append((sl, tp, age))

    if len(grid) > GRID_CAP:
        warnings.warn(
            f"Grid size {len(grid)} exceeds cap {GRID_CAP}. "
            f"Truncated to {GRID_CAP} combinations. "
            f"Use individual parameter sweeps for full coverage.",
            UserWarning,
            stacklevel=2,
        )
        grid = grid[:GRID_CAP]

    return grid


# ---------------------------------------------------------------------------
# Per-parameter sweep helpers
# ---------------------------------------------------------------------------

def _sweep_sl(
    rows: list[TradeRow],
    sl_values: list[Decimal],
    baseline: SimResult,
) -> list[dict[str, Any]]:
    """Run stop-loss sweep and return table rows."""
    results = []
    for sl_pct in sl_values:
        simulated = [
            TradeRow(
                id=r.id, symbol=r.symbol, side=r.side,
                entry_time=r.entry_time, entry_price=r.entry_price,
                exit_time=r.exit_time, exit_price=r.exit_price,
                quantity=r.quantity,
                pnl=simulate_sl(r, sl_pct),
                regime=r.regime, strategy_id=r.strategy_id,
            )
            for r in rows
        ]
        result = SimResult.from_rows(simulated)
        delta = result.adj_pnl - baseline.adj_pnl
        is_baseline = sl_pct == CURRENT_SL_PCT
        results.append({
            "param": sl_pct,
            "result": result,
            "delta": delta,
            "is_baseline": is_baseline,
        })
    return results


def _sweep_tp(
    rows: list[TradeRow],
    tp_values: list[Decimal],
    baseline: SimResult,
) -> list[dict[str, Any]]:
    """Run take-profit sweep and return table rows."""
    results = []
    for tp_pct in tp_values:
        simulated = [
            TradeRow(
                id=r.id, symbol=r.symbol, side=r.side,
                entry_time=r.entry_time, entry_price=r.entry_price,
                exit_time=r.exit_time, exit_price=r.exit_price,
                quantity=r.quantity,
                pnl=simulate_tp(r, tp_pct),
                regime=r.regime, strategy_id=r.strategy_id,
            )
            for r in rows
        ]
        result = SimResult.from_rows(simulated)
        delta = result.adj_pnl - baseline.adj_pnl
        is_baseline = tp_pct == CURRENT_TP_PCT
        results.append({
            "param": tp_pct,
            "result": result,
            "delta": delta,
            "is_baseline": is_baseline,
        })
    return results


def _sweep_age(
    rows: list[TradeRow],
    age_values: list[int],
    baseline: SimResult,
) -> list[dict[str, Any]]:
    """Run max-age sweep and return table rows."""
    results = []
    for age_min in age_values:
        simulated = [
            TradeRow(
                id=r.id, symbol=r.symbol, side=r.side,
                entry_time=r.entry_time, entry_price=r.entry_price,
                exit_time=r.exit_time, exit_price=r.exit_price,
                quantity=r.quantity,
                pnl=simulate_age(r, age_min),
                regime=r.regime, strategy_id=r.strategy_id,
            )
            for r in rows
        ]
        result = SimResult.from_rows(simulated)
        delta = result.adj_pnl - baseline.adj_pnl
        is_baseline = age_min == CURRENT_MAX_AGE_MIN
        results.append({
            "param": age_min,
            "result": result,
            "delta": delta,
            "is_baseline": is_baseline,
        })
    return results


def _best_param(sweep_rows: list[dict[str, Any]]) -> Any:
    """Return the param value with the highest adj_pnl."""
    if not sweep_rows:
        return None
    return max(sweep_rows, key=lambda r: r["result"].adj_pnl)["param"]


# ---------------------------------------------------------------------------
# Top 5 grid search (top-3 from each sweep, max 27 combos)
# ---------------------------------------------------------------------------

def _top5_grid(
    rows: list[TradeRow],
    sl_candidates: list[Decimal],
    tp_candidates: list[Decimal],
    age_candidates: list[int],
) -> list[dict[str, Any]]:
    """
    Run a limited grid over the top-3 values from each sweep.

    At most 27 combinations (3*3*3). Returns sorted list (best adj_pnl first).
    """
    combos = []
    for sl in sl_candidates[:3]:
        for tp in tp_candidates[:3]:
            for age in age_candidates[:3]:
                simulated = []
                for r in rows:
                    pnl = r.pnl if r.pnl is not None else Decimal("0")
                    pnl = simulate_sl(
                        TradeRow(
                            id=r.id, symbol=r.symbol, side=r.side,
                            entry_time=r.entry_time, entry_price=r.entry_price,
                            exit_time=r.exit_time, exit_price=r.exit_price,
                            quantity=r.quantity, pnl=pnl,
                            regime=r.regime, strategy_id=r.strategy_id,
                        ),
                        sl,
                    )
                    pnl = simulate_tp(
                        TradeRow(
                            id=r.id, symbol=r.symbol, side=r.side,
                            entry_time=r.entry_time, entry_price=r.entry_price,
                            exit_time=r.exit_time, exit_price=r.exit_price,
                            quantity=r.quantity, pnl=pnl,
                            regime=r.regime, strategy_id=r.strategy_id,
                        ),
                        tp,
                    )
                    pnl = simulate_age(
                        TradeRow(
                            id=r.id, symbol=r.symbol, side=r.side,
                            entry_time=r.entry_time, entry_price=r.entry_price,
                            exit_time=r.exit_time, exit_price=r.exit_price,
                            quantity=r.quantity, pnl=pnl,
                            regime=r.regime, strategy_id=r.strategy_id,
                        ),
                        age,
                    )
                    simulated.append(
                        TradeRow(
                            id=r.id, symbol=r.symbol, side=r.side,
                            entry_time=r.entry_time, entry_price=r.entry_price,
                            exit_time=r.exit_time, exit_price=r.exit_price,
                            quantity=r.quantity, pnl=pnl,
                            regime=r.regime, strategy_id=r.strategy_id,
                        )
                    )
                result = SimResult.from_rows(simulated)
                combos.append({"sl": sl, "tp": tp, "age": age, "result": result})

    combos.sort(key=lambda c: c["result"].adj_pnl, reverse=True)
    return combos[:5]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_delta(delta: Decimal) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}${delta:.2f}"


def _fmt_table_row(
    param_str: str,
    result: SimResult,
    delta: Decimal,
    is_baseline: bool,
) -> str:
    marker = "baseline" if is_baseline else _fmt_delta(delta)
    better = " <-- better" if (delta > 0 and not is_baseline) else ""
    return (
        f"  {param_str:<8} {result.trade_count:<7} "
        f"{result.win_rate:>6.1f}%  "
        f"${result.total_pnl:>8.2f}  "
        f"${result.adj_pnl:>8.2f}  "
        f"{result.sharpe:>6.2f}  "
        f"{marker}{better}"
    )


def _format_sl_table(sl_rows: list[dict[str, Any]]) -> str:
    lines = ["=== Stop-Loss Sensitivity ==="]
    lines.append(
        f"  {'SL%':<8} {'Trades':<7} {'WinRate':>7}  "
        f"{'TotalPnL':>9}  {'AdjPnL':>9}  {'Sharpe':>6}  vs Current"
    )
    for row in sl_rows:
        param_str = f"{row['param']}%"
        lines.append(_fmt_table_row(param_str, row["result"], row["delta"], row["is_baseline"]))
    return "\n".join(lines)


def _format_tp_table(tp_rows: list[dict[str, Any]]) -> str:
    lines = ["=== Take-Profit Sensitivity ==="]
    lines.append(
        f"  {'TP%':<8} {'Trades':<7} {'WinRate':>7}  "
        f"{'TotalPnL':>9}  {'AdjPnL':>9}  {'Sharpe':>6}  vs Current"
    )
    for row in tp_rows:
        param_str = f"{row['param']}%"
        lines.append(_fmt_table_row(param_str, row["result"], row["delta"], row["is_baseline"]))
    return "\n".join(lines)


def _format_age_table(age_rows: list[dict[str, Any]]) -> str:
    lines = ["=== Max Position Age Sensitivity ==="]
    lines.append(
        f"  {'Age(min)':<8} {'Trades':<7} {'WinRate':>7}  "
        f"{'TotalPnL':>9}  {'AdjPnL':>9}  {'Sharpe':>6}  vs Current"
    )
    for row in age_rows:
        param_str = f"{row['param']}min"
        lines.append(_fmt_table_row(param_str, row["result"], row["delta"], row["is_baseline"]))
    return "\n".join(lines)


def _format_top5(top5: list[dict[str, Any]]) -> str:
    lines = ["=== Top 5 Configurations ==="]
    for i, combo in enumerate(top5, start=1):
        r = combo["result"]
        lines.append(
            f"  #{i}: SL={combo['sl']}%, TP={combo['tp']}%, Age={combo['age']}min"
            f" -> AdjPnL=${r.adj_pnl:.2f}, Sharpe={r.sharpe:.2f}"
        )
    return "\n".join(lines)


def _format_recommendations(
    sl_rows: list[dict[str, Any]],
    tp_rows: list[dict[str, Any]],
    age_rows: list[dict[str, Any]],
) -> str:
    lines = ["=== Recommendations ==="]

    def _rec(sweep: list[dict[str, Any]], current_val: Any, unit: str, name: str) -> str:
        if not sweep:
            return f"  - {name}: No data."
        best = max(sweep, key=lambda r: r["result"].adj_pnl)
        baseline_row = next((r for r in sweep if r["is_baseline"]), None)
        if best["is_baseline"]:
            return f"  - {name}: Current {current_val}{unit} is optimal in this dataset."
        delta = best["result"].adj_pnl - (baseline_row["result"].adj_pnl if baseline_row else Decimal("0"))
        return (
            f"  - {name}: Current {current_val}{unit} is suboptimal. "
            f"Consider {best['param']}{unit} ({_fmt_delta(delta)} improvement)"
        )

    lines.append(_rec(sl_rows, CURRENT_SL_PCT, "%", "Stop-loss"))
    lines.append(_rec(tp_rows, CURRENT_TP_PCT, "%", "Take-profit"))
    lines.append(_rec(age_rows, CURRENT_MAX_AGE_MIN, "min", "Max age"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def run_sensitivity_report(
    db_path: Path,
    sl_values: list[Decimal] | None = None,
    tp_values: list[Decimal] | None = None,
    age_values: list[int] | None = None,
    strategy: str | None = None,
    as_json: bool = False,
) -> str:
    """
    Generate a parameter sensitivity report for closed trades in db_path.

    Runs individual sweeps for each provided parameter list. Defaults to
    the full DEFAULT_* grids when None is passed.

    Args:
        db_path: Path to the SQLite database file.
        sl_values: Stop-loss percentages to sweep (None = use defaults).
        tp_values: Take-profit percentages to sweep (None = use defaults).
        age_values: Max-age values in minutes to sweep (None = use defaults).
        strategy: If set, filter to this strategy_id only.
        as_json: If True, return a JSON string instead of human-readable text.

    Returns:
        Formatted report string (text or JSON).
    """
    sl_values = sl_values if sl_values is not None else DEFAULT_STOP_LOSS_PCT
    tp_values = tp_values if tp_values is not None else DEFAULT_TAKE_PROFIT_PCT
    age_values = age_values if age_values is not None else DEFAULT_MAX_AGE_MINUTES

    rows = fetch_closed_trades_for_sensitivity(db_path, strategy=strategy)

    if not rows:
        if as_json:
            return json.dumps({
                "sensitivity": {},
                "message": "No closed trades found.",
                "strategy_filter": strategy,
            }, indent=2)
        return "No closed trades found in the database."

    # Baseline: actual outcomes with no modification
    baseline = SimResult.from_rows(rows)

    # Individual sweeps
    sl_rows = _sweep_sl(rows, sl_values, baseline)
    tp_rows = _sweep_tp(rows, tp_values, baseline)
    age_rows = _sweep_age(rows, age_values, baseline)

    # Top 5: best 3 from each individual sweep
    def _top3_by_adj(sweep: list[dict[str, Any]]) -> list[Any]:
        return [r["param"] for r in sorted(sweep, key=lambda x: x["result"].adj_pnl, reverse=True)[:3]]

    sl_candidates = _top3_by_adj(sl_rows) if sl_rows else [CURRENT_SL_PCT]
    tp_candidates = _top3_by_adj(tp_rows) if tp_rows else [CURRENT_TP_PCT]
    age_candidates = _top3_by_adj(age_rows) if age_rows else [CURRENT_MAX_AGE_MIN]

    top5 = _top5_grid(rows, sl_candidates, tp_candidates, age_candidates)

    if as_json:
        def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
            r = row["result"]
            return {
                "param": str(row["param"]),
                "trade_count": r.trade_count,
                "win_rate": float(r.win_rate),
                "total_pnl": float(r.total_pnl),
                "adj_pnl": float(r.adj_pnl),
                "sharpe": float(r.sharpe),
                "vs_current": float(row["delta"]),
                "is_baseline": row["is_baseline"],
            }

        top5_dicts = [
            {
                "sl": str(c["sl"]),
                "tp": str(c["tp"]),
                "age_minutes": c["age"],
                "adj_pnl": float(c["result"].adj_pnl),
                "sharpe": float(c["result"].sharpe),
            }
            for c in top5
        ]

        return json.dumps(
            {
                "sensitivity": {
                    "stop_loss": [_row_to_dict(r) for r in sl_rows],
                    "take_profit": [_row_to_dict(r) for r in tp_rows],
                    "max_age": [_row_to_dict(r) for r in age_rows],
                },
                "top5": top5_dicts,
                "baseline": {
                    "trade_count": baseline.trade_count,
                    "adj_pnl": float(baseline.adj_pnl),
                    "sharpe": float(baseline.sharpe),
                },
                "strategy_filter": strategy,
            },
            indent=2,
        )

    # Human-readable text report
    sections = []
    if sl_rows:
        sections.append(_format_sl_table(sl_rows))
    if tp_rows:
        sections.append(_format_tp_table(tp_rows))
    if age_rows:
        sections.append(_format_age_table(age_rows))
    if top5:
        sections.append(_format_top5(top5))
    sections.append(_format_recommendations(sl_rows, tp_rows, age_rows))

    header_parts = [f"Trades: {baseline.trade_count}"]
    if strategy:
        header_parts.append(f"Strategy: {strategy}")
    header_parts.append(f"Baseline AdjPnL: ${baseline.adj_pnl:.2f}")
    header = "=== Sensitivity Analysis === " + " | ".join(header_parts)

    return header + "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_decimal_list(s: str) -> list[Decimal]:
    return [Decimal(v.strip()) for v in s.split(",") if v.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(v.strip()) for v in s.split(",") if v.strip()]


def main() -> None:
    """CLI entry point for parameter sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity analysis for CerebrumCoin trades.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/sensitivity.py --db data/cerebrum.db
  python3 scripts/sensitivity.py --db data/cerebrum.db --stop-loss 0.5,1.0,1.5,2.0
  python3 scripts/sensitivity.py --db data/cerebrum.db --json
  python3 scripts/sensitivity.py --db data/cerebrum.db --strategy momentum
        """,
    )
    parser.add_argument("--db", required=True, help="Path to cerebrum.db SQLite file")
    parser.add_argument(
        "--stop-loss",
        metavar="PCT,...",
        help="Comma-separated stop-loss percentages (default: 0.5,0.8,1.0,1.5,2.0,2.5,3.0)",
    )
    parser.add_argument(
        "--take-profit",
        metavar="PCT,...",
        help="Comma-separated take-profit percentages (default: 1.0,1.5,2.0,3.0,4.0,5.0)",
    )
    parser.add_argument(
        "--max-age",
        metavar="MIN,...",
        help="Comma-separated max position age in minutes (default: 30,60,90,120,180,240,480)",
    )
    parser.add_argument(
        "--strategy",
        metavar="NAME",
        help="Filter analysis to a single strategy_id (e.g. momentum)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON instead of human-readable text",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        parser.error(f"Database file not found: {db_path}")

    sl_values = _parse_decimal_list(args.stop_loss) if args.stop_loss else None
    tp_values = _parse_decimal_list(args.take_profit) if args.take_profit else None
    age_values = _parse_int_list(args.max_age) if args.max_age else None

    report = run_sensitivity_report(
        db_path=db_path,
        sl_values=sl_values,
        tp_values=tp_values,
        age_values=age_values,
        strategy=args.strategy,
        as_json=args.as_json,
    )
    print(report)


if __name__ == "__main__":
    main()
