#!/usr/bin/env python3
"""
Strategy attribution analysis for CerebrumCoin.

Reads closed trades from cerebrum.db and produces a comprehensive report
covering aggregate performance, per-strategy attribution, regime-conditioned
performance, and a go-live scorecard.

Follows DEC-EXPORT-001 pattern: raw sqlite3, no ORM, no async imports from
cerebrum. Decimal for all P&L arithmetic. Pure stdlib except argparse.

@decision DEC-ANALYZE-001
@title analyze.py replicates stats logic inline (no cerebrum imports)
@status accepted
@rationale Standalone scripts avoid importing the asyncio-based StateManager
and all transitive dependencies. Sharpe/Sortino/drawdown logic replicated
using raw Decimal arithmetic and sqlite3. Keeps the script deployable anywhere
Python 3.10+ is available without a full venv. Follows DEC-EXPORT-001.

Usage:
    python3 scripts/analyze.py --db data/cerebrum.db
    python3 scripts/analyze.py --db data/cerebrum.db --output data/reports/analysis.md
    python3 scripts/analyze.py --db data/cerebrum.db --strategy-only
    python3 scripts/analyze.py --db data/cerebrum.db --json
"""

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kraken maker fee, applied to both entry and exit (round-trip = x2)
COMMISSION_RATE = Decimal("0.0016")

# Minimum trade counts per strategy for go-live readiness (criterion 8)
STRATEGY_TRADE_MINIMUMS: dict[str, int] = {
    "momentum": 50,
    "mean_reversion": 50,
    "breakout": 50,
    "range_trading": 30,
    "swing_trading": 15,
    "news_driven": 10,
}

# Kill thresholds
KILL_DRAWDOWN_PCT = Decimal("7")   # > 7% drawdown triggers KILL
DRAWDOWN_CRITERION_PCT = Decimal("5")  # < 5% drawdown for GO criterion

# Scorecard go criteria
MIN_PAPER_DAYS = 30
MIN_SHARPE_STRATEGIES = 2
SHARPE_THRESHOLD = Decimal("0.5")
MAX_COMMISSION_DRAG_PCT = Decimal("40")
MAX_SINGLE_STRATEGY_SHARE_PCT = Decimal("60")
MIN_REALLOCATION_EVENTS = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeRow:
    """
    Single closed trade row read from the trades table.

    All price/quantity/pnl fields are stored as TEXT in SQLite (Decimal
    strings). Parsed to Decimal on construction to avoid float rounding.
    """

    id: int
    symbol: str
    side: str
    entry_time: float
    entry_price: Decimal
    exit_time: float | None
    exit_price: Decimal | None
    quantity: Decimal
    pnl: Decimal | None
    regime: str
    strategy_id: str | None


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


def fetch_all_closed_trades(db_path: Path) -> list[TradeRow]:
    """
    Fetch all CLOSED trades from the database, ordered by entry_time ASC.

    Includes legacy trades with NULL strategy_id.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of TradeRow objects sorted by entry_time ascending.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, symbol, side, entry_time, entry_price,
                   exit_time, exit_price, quantity, pnl, regime, strategy_id
            FROM trades
            WHERE status = 'CLOSED'
            ORDER BY entry_time ASC
            """
        )
        return [_row_to_trade(row) for row in cursor]
    finally:
        conn.close()


def fetch_strategy_trades(db_path: Path) -> list[TradeRow]:
    """
    Fetch only CLOSED trades that have a non-NULL strategy_id.

    Excludes legacy trades (strategy_id IS NULL).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of TradeRow objects with strategy_id set, sorted by entry_time.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT id, symbol, side, entry_time, entry_price,
                   exit_time, exit_price, quantity, pnl, regime, strategy_id
            FROM trades
            WHERE status = 'CLOSED'
              AND strategy_id IS NOT NULL
            ORDER BY entry_time ASC
            """
        )
        return [_row_to_trade(row) for row in cursor]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_by_strategy(rows: list[TradeRow]) -> dict[str, list[TradeRow]]:
    """
    Group trade rows by strategy_id.

    Trades with strategy_id=None are placed in the 'legacy (NULL)' bucket.

    Args:
        rows: List of TradeRow objects.

    Returns:
        Dict mapping strategy name (or 'legacy (NULL)') to list of rows.
    """
    groups: dict[str, list[TradeRow]] = {}
    for row in rows:
        key = row.strategy_id if row.strategy_id is not None else "legacy (NULL)"
        groups.setdefault(key, []).append(row)
    return groups


def group_by_regime(rows: list[TradeRow]) -> dict[str, list[TradeRow]]:
    """
    Group trade rows by regime.

    Args:
        rows: List of TradeRow objects.

    Returns:
        Dict mapping regime string to list of rows.
    """
    groups: dict[str, list[TradeRow]] = {}
    for row in rows:
        groups.setdefault(row.regime, []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Statistics (replicated from cerebrum/monitoring/stats.py without imports)
# ---------------------------------------------------------------------------

def calculate_commission(entry_price: Decimal, quantity: Decimal) -> Decimal:
    """
    Calculate round-trip commission for a single trade.

    Kraken maker fee (0.16%) applied to both entry and exit sides.
    commission = entry_price * quantity * COMMISSION_RATE * 2

    Args:
        entry_price: Entry price as Decimal.
        quantity: Trade quantity as Decimal.

    Returns:
        Round-trip commission in USD.
    """
    return entry_price * quantity * COMMISSION_RATE * 2


def calculate_win_rate(rows: list[TradeRow]) -> Decimal:
    """
    Calculate win rate as a percentage (0-100).

    Args:
        rows: List of closed trade rows.

    Returns:
        Win rate percentage, or 0 if no trades.
    """
    if not rows:
        return Decimal("0")
    winners = sum(1 for r in rows if r.pnl is not None and r.pnl > 0)
    return Decimal(str(winners)) / Decimal(str(len(rows))) * Decimal("100")


def calculate_total_pnl(rows: list[TradeRow]) -> Decimal:
    """Sum raw P&L across all rows (before commission adjustment)."""
    return sum((r.pnl for r in rows if r.pnl is not None), Decimal("0"))


def calculate_commission_adjusted_pnl(rows: list[TradeRow]) -> Decimal:
    """
    Calculate total P&L after subtracting round-trip commissions.

    Args:
        rows: List of closed trade rows.

    Returns:
        Commission-adjusted net P&L.
    """
    total = Decimal("0")
    for r in rows:
        if r.pnl is None:
            continue
        commission = calculate_commission(r.entry_price, r.quantity)
        total += r.pnl - commission
    return total


def calculate_gross_pnl_for_drag(rows: list[TradeRow]) -> tuple[Decimal, Decimal]:
    """
    Return (total_commission, gross_pnl) for commission drag calculation.

    gross_pnl is the sum of all positive P&L (winning trades only).
    Used for commission drag % = total_commission / gross_pnl * 100.

    Args:
        rows: List of closed trade rows.

    Returns:
        Tuple of (total_commission, gross_pnl).
    """
    total_commission = Decimal("0")
    gross_pnl = Decimal("0")
    for r in rows:
        if r.pnl is None:
            continue
        total_commission += calculate_commission(r.entry_price, r.quantity)
        if r.pnl > 0:
            gross_pnl += r.pnl
    return total_commission, gross_pnl


def calculate_avg_hold_minutes(rows: list[TradeRow]) -> Decimal:
    """
    Calculate average hold time in minutes across closed trades.

    Args:
        rows: List of closed trade rows with entry_time and exit_time.

    Returns:
        Average hold time in minutes, or 0 if no completed trades.
    """
    durations = []
    for r in rows:
        if r.exit_time is not None and r.entry_time is not None:
            durations.append(r.exit_time - r.entry_time)
    if not durations:
        return Decimal("0")
    avg_seconds = sum(durations) / len(durations)
    return Decimal(str(avg_seconds)) / Decimal("60")


def calculate_sharpe_from_rows(
    rows: list[TradeRow],
    risk_free_rate: Decimal = Decimal("0"),
) -> Decimal:
    """
    Calculate Sharpe ratio using percentage returns (pnl / entry_value).

    Replicates DEC-MONITOR-001 logic from cerebrum/monitoring/stats.py
    without importing the cerebrum package.

    Uses percentage returns so a $5 gain on $100 (5%) is treated differently
    from $5 on $2,500 (0.2%). Trades with zero entry_value are skipped.
    Zero variance (perfectly consistent returns) returns sentinel 1000000.

    Args:
        rows: List of closed trade rows.
        risk_free_rate: Risk-free rate (default 0).

    Returns:
        Sharpe ratio as Decimal, or 0 if insufficient data.
    """
    if not rows:
        return Decimal("0")

    returns = []
    for r in rows:
        if r.pnl is None:
            continue
        entry_value = r.entry_price * r.quantity
        if not entry_value or entry_value == 0:
            continue
        returns.append(r.pnl / entry_value)

    if not returns:
        return Decimal("0")

    mean_return = sum(returns) / Decimal(str(len(returns)))
    variance = sum((x - mean_return) ** 2 for x in returns) / Decimal(str(len(returns)))

    if variance == 0:
        return Decimal("1000000") if mean_return > risk_free_rate else Decimal("0")

    std_dev = variance.sqrt()
    excess = mean_return - risk_free_rate
    return excess / std_dev if std_dev > 0 else Decimal("0")


def calculate_sortino_from_rows(
    rows: list[TradeRow],
    target_return: Decimal = Decimal("0"),
) -> Decimal:
    """
    Calculate Sortino ratio using percentage returns (pnl / entry_value).

    Only considers downside volatility (returns below target_return).
    Replicates DEC-MONITOR-001 logic without importing cerebrum.

    Args:
        rows: List of closed trade rows.
        target_return: Minimum acceptable return (default 0).

    Returns:
        Sortino ratio as Decimal, or 0 if insufficient data.
    """
    if not rows:
        return Decimal("0")

    returns = []
    for r in rows:
        if r.pnl is None:
            continue
        entry_value = r.entry_price * r.quantity
        if not entry_value or entry_value == 0:
            continue
        returns.append(r.pnl / entry_value)

    if not returns:
        return Decimal("0")

    mean_return = sum(returns) / Decimal(str(len(returns)))
    downside = [x for x in returns if x < target_return]

    if not downside:
        return Decimal("1000000") if mean_return > target_return else Decimal("0")

    downside_variance = sum((x - target_return) ** 2 for x in downside) / Decimal(str(len(downside)))
    downside_dev = downside_variance.sqrt()
    excess = mean_return - target_return
    return excess / downside_dev if downside_dev > 0 else Decimal("0")


def calculate_drawdown_from_rows(
    rows: list[TradeRow],
    initial_balance: Decimal,
) -> tuple[Decimal, Decimal]:
    """
    Calculate maximum drawdown in absolute USD and percentage terms.

    Builds an equity curve from initial_balance + cumulative P&L.
    Peak is tracked; drawdown = peak - current_equity.

    Args:
        rows: List of closed trade rows, ordered by entry_time ASC.
        initial_balance: Starting equity before any trades.

    Returns:
        Tuple of (max_drawdown_usd, max_drawdown_pct).
    """
    if not rows:
        return Decimal("0"), Decimal("0")

    equity = initial_balance
    peak = initial_balance
    max_dd = Decimal("0")

    for r in rows:
        if r.pnl is None:
            continue
        equity += r.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / peak * Decimal("100")) if peak > 0 else Decimal("0")
    return max_dd, max_dd_pct


def _days_span(rows: list[TradeRow]) -> float:
    """Return calendar days between earliest and latest trade entry_time."""
    if len(rows) < 2:
        return 0.0
    earliest = min(r.entry_time for r in rows)
    latest = max(r.entry_time for r in rows)
    return (latest - earliest) / 86400.0


# ---------------------------------------------------------------------------
# Scorecard evaluation
# ---------------------------------------------------------------------------

def evaluate_scorecard(
    rows: list[TradeRow],
    initial_balance: Decimal,
) -> dict[str, Any]:
    """
    Evaluate the 8 go-live criteria and detect kill conditions.

    Criteria:
      1. Continuous paper trading days >= 30
      2. Net P&L after commission > 0
      3. Strategies with Sharpe > 0.5 >= 2
      4. Max drawdown from peak < 5%
      5. Commission drag (% of gross) < 40%
      6. Max single strategy P&L share < 60%
      7. Darwinian reallocations >= 3 (N/A, not in DB)
      8. Strategy trade minimums met

    Kill criteria (checked separately, not blocking GO):
      - Drawdown > 7% -> KILL warning
      - (3 consecutive losing weeks requires log analysis -> skipped)

    Args:
        rows: All closed trade rows (including legacy).
        initial_balance: Starting equity for drawdown calculation.

    Returns:
        Dict with keys: criteria (list), kill_criteria (list of strings),
        verdict (str: GO / NO-GO / INSUFFICIENT DATA).
    """
    criteria: list[dict[str, Any]] = []
    kill_criteria: list[str] = []

    if not rows:
        # Build placeholder criteria with N/A
        for i, name in enumerate([
            "Continuous paper trading days",
            "Net P&L after commission",
            "Strategies with Sharpe > 0.5",
            "Max drawdown from peak",
            "Commission drag (% of gross)",
            "Max single strategy P&L share",
            "Darwinian reallocations",
            "Strategy trade minimums met",
        ], start=1):
            criteria.append({
                "id": i, "name": name, "target": "N/A",
                "current": "N/A", "pass": False,
            })
        return {"criteria": criteria, "kill_criteria": [], "verdict": "INSUFFICIENT DATA"}

    # Compute shared metrics
    dd_abs, dd_pct = calculate_drawdown_from_rows(rows, initial_balance)
    total_commission, gross_pnl = calculate_gross_pnl_for_drag(rows)
    net_pnl = calculate_commission_adjusted_pnl(rows)
    days = _days_span(rows)

    # Per-strategy P&L for criteria 3, 6, 8
    strategy_rows = [r for r in rows if r.strategy_id is not None]
    strategy_groups = group_by_strategy(strategy_rows)
    strategy_pnls = {
        name: calculate_commission_adjusted_pnl(grp)
        for name, grp in strategy_groups.items()
        if name != "legacy (NULL)"
    }

    # Criterion 1: continuous paper trading days >= 30
    c1_pass = days >= MIN_PAPER_DAYS
    criteria.append({
        "id": 1,
        "name": "Continuous paper trading days",
        "target": f">= {MIN_PAPER_DAYS}",
        "current": f"{days:.1f} days",
        "pass": c1_pass,
    })

    # Criterion 2: net P&L after commission > 0
    c2_pass = net_pnl > 0
    criteria.append({
        "id": 2,
        "name": "Net P&L after commission",
        "target": "> $0",
        "current": f"${net_pnl:.2f}",
        "pass": c2_pass,
    })

    # Criterion 3: strategies with Sharpe > 0.5 >= 2
    sharpe_above_threshold = 0
    for name, grp in strategy_groups.items():
        if name == "legacy (NULL)":
            continue
        s = calculate_sharpe_from_rows(grp)
        if s > SHARPE_THRESHOLD:
            sharpe_above_threshold += 1
    c3_pass = sharpe_above_threshold >= MIN_SHARPE_STRATEGIES
    criteria.append({
        "id": 3,
        "name": f"Strategies with Sharpe > {SHARPE_THRESHOLD}",
        "target": f">= {MIN_SHARPE_STRATEGIES}",
        "current": str(sharpe_above_threshold),
        "pass": c3_pass,
    })

    # Criterion 4: max drawdown < 5%
    c4_pass = dd_pct < DRAWDOWN_CRITERION_PCT
    criteria.append({
        "id": 4,
        "name": "Max drawdown from peak",
        "target": f"< {DRAWDOWN_CRITERION_PCT}%",
        "current": f"{dd_pct:.2f}%",
        "pass": c4_pass,
    })

    # Criterion 5: commission drag < 40%
    if gross_pnl > 0:
        drag_pct = total_commission / gross_pnl * Decimal("100")
        c5_pass = drag_pct < MAX_COMMISSION_DRAG_PCT
        drag_str = f"{drag_pct:.1f}%"
    else:
        drag_pct = Decimal("0")
        c5_pass = False
        drag_str = "N/A (no gross profit)"
    criteria.append({
        "id": 5,
        "name": "Commission drag (% of gross)",
        "target": f"< {MAX_COMMISSION_DRAG_PCT}%",
        "current": drag_str,
        "pass": c5_pass,
    })

    # Criterion 6: max single strategy P&L share < 60%
    total_strat_pnl = sum(abs(v) for v in strategy_pnls.values())
    if strategy_pnls and total_strat_pnl > 0:
        max_share_pct = max(abs(v) for v in strategy_pnls.values()) / total_strat_pnl * Decimal("100")
        c6_pass = max_share_pct < MAX_SINGLE_STRATEGY_SHARE_PCT
        share_str = f"{max_share_pct:.1f}%"
    else:
        max_share_pct = Decimal("0")
        c6_pass = False
        share_str = "N/A (no strategy trades)"
    criteria.append({
        "id": 6,
        "name": "Max single strategy P&L share",
        "target": f"< {MAX_SINGLE_STRATEGY_SHARE_PCT}%",
        "current": share_str,
        "pass": c6_pass,
    })

    # Criterion 7: Darwinian reallocations — not tracked in DB
    criteria.append({
        "id": 7,
        "name": "Darwinian reallocations (>= 10pp)",
        "target": f">= {MIN_REALLOCATION_EVENTS}",
        "current": "N/A (requires log analysis)",
        "pass": False,
    })

    # Criterion 8: strategy trade minimums met
    minimums_status: list[str] = []
    all_minimums_met = True
    for strategy, minimum in STRATEGY_TRADE_MINIMUMS.items():
        count = len(strategy_groups.get(strategy, []))
        met = count >= minimum
        if not met:
            all_minimums_met = False
        minimums_status.append(f"{strategy}:{count}/{minimum}")
    criteria.append({
        "id": 8,
        "name": "Strategy trade minimums met",
        "target": "all met",
        "current": ", ".join(minimums_status) if minimums_status else "N/A",
        "pass": all_minimums_met,
    })

    # Kill criteria
    if dd_pct > KILL_DRAWDOWN_PCT:
        kill_criteria.append(
            f"KILL: Drawdown {dd_pct:.2f}% exceeds {KILL_DRAWDOWN_PCT}% — Investigate immediately"
        )

    # Verdict
    # Criterion 7 is always False (data not available) — exclude from GO check
    go_eligible = [c for c in criteria if c["id"] != 7]
    insufficient = len(rows) < 10  # very few trades = not enough data
    if insufficient:
        verdict = "INSUFFICIENT DATA"
    elif all(c["pass"] for c in go_eligible):
        verdict = "GO"
    else:
        verdict = "NO-GO"

    return {"criteria": criteria, "kill_criteria": kill_criteria, "verdict": verdict}


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _fmt_decimal(value: Decimal, prefix: str = "$", decimals: int = 2) -> str:
    """Format a Decimal value for display."""
    sign = "-" if value < 0 else ""
    return f"{sign}{prefix}{abs(value):.{decimals}f}"


def _section_aggregate(rows: list[TradeRow], initial_balance: Decimal) -> dict[str, Any]:
    """Build aggregate overview section data."""
    if not rows:
        return {
            "total_trades": 0, "win_rate": "0.00%", "total_pnl": "$0.00",
            "adjusted_pnl": "$0.00", "sharpe": "0.00", "sortino": "0.00",
            "btc_trades": 0, "eth_trades": 0,
            "btc_pnl": "$0.00", "eth_pnl": "$0.00",
            "max_drawdown_pct": "0.00%",
        }

    btc_rows = [r for r in rows if "BTC" in r.symbol]
    eth_rows = [r for r in rows if "ETH" in r.symbol]
    _, dd_pct = calculate_drawdown_from_rows(rows, initial_balance)

    return {
        "total_trades": len(rows),
        "win_rate": f"{calculate_win_rate(rows):.2f}%",
        "total_pnl": _fmt_decimal(calculate_total_pnl(rows)),
        "adjusted_pnl": _fmt_decimal(calculate_commission_adjusted_pnl(rows)),
        "sharpe": f"{calculate_sharpe_from_rows(rows):.2f}",
        "sortino": f"{calculate_sortino_from_rows(rows):.2f}",
        "btc_trades": len(btc_rows),
        "eth_trades": len(eth_rows),
        "btc_pnl": _fmt_decimal(calculate_commission_adjusted_pnl(btc_rows)),
        "eth_pnl": _fmt_decimal(calculate_commission_adjusted_pnl(eth_rows)),
        "max_drawdown_pct": f"{dd_pct:.2f}%",
    }


def _section_per_strategy(all_rows: list[TradeRow]) -> list[dict[str, Any]]:
    """Build per-strategy attribution section data, sorted by adj P&L desc."""
    groups = group_by_strategy(all_rows)
    results = []
    for strategy, rows in groups.items():
        adj_pnl = calculate_commission_adjusted_pnl(rows)
        results.append({
            "strategy": strategy,
            "trade_count": len(rows),
            "win_rate": f"{calculate_win_rate(rows):.2f}%",
            "avg_pnl": _fmt_decimal(
                calculate_total_pnl(rows) / Decimal(str(len(rows))) if rows else Decimal("0")
            ),
            "total_pnl": _fmt_decimal(calculate_total_pnl(rows)),
            "adjusted_pnl": _fmt_decimal(adj_pnl),
            "sharpe": f"{calculate_sharpe_from_rows(rows):.2f}",
            "avg_hold_min": f"{calculate_avg_hold_minutes(rows):.1f}",
            "_sort_key": adj_pnl,
        })
    results.sort(key=lambda x: x["_sort_key"], reverse=True)
    for r in results:
        del r["_sort_key"]
    return results


def _section_regime(all_rows: list[TradeRow]) -> list[dict[str, Any]]:
    """Build regime-conditioned performance data (strategy x regime)."""
    groups = group_by_strategy(all_rows)
    rows_out = []
    for strategy, s_rows in sorted(groups.items()):
        regime_groups = group_by_regime(s_rows)
        for regime, r_rows in sorted(regime_groups.items()):
            if not r_rows:
                continue
            rows_out.append({
                "strategy": strategy,
                "regime": regime,
                "trade_count": len(r_rows),
                "win_rate": f"{calculate_win_rate(r_rows):.2f}%",
                "avg_pnl": _fmt_decimal(
                    calculate_total_pnl(r_rows) / Decimal(str(len(r_rows)))
                ),
            })
    return rows_out


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_text(
    aggregate: dict[str, Any],
    per_strategy: list[dict[str, Any]],
    regime_data: list[dict[str, Any]],
    scorecard: dict[str, Any],
    days: float,
) -> str:
    """Format the full report as human-readable text."""
    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append(" CerebrumCoin Strategy Attribution Report ".center(80, "="))
    lines.append(f" Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ".center(80, "="))
    lines.append(sep)

    # A) Aggregate Overview
    lines.append("\n## A) Aggregate Overview\n")
    a = aggregate
    lines.append(f"  Total trades:       {a['total_trades']}")
    lines.append(f"  Win rate:           {a['win_rate']}")
    lines.append(f"  Total P&L (raw):    {a['total_pnl']}")
    lines.append(f"  Adj P&L (net comm): {a['adjusted_pnl']}")
    lines.append(f"  Sharpe ratio:       {a['sharpe']}")
    lines.append(f"  Sortino ratio:      {a['sortino']}")
    lines.append(f"  Max drawdown:       {a['max_drawdown_pct']}")
    lines.append(f"  BTC trades:         {a['btc_trades']} ({a['btc_pnl']} adj)")
    lines.append(f"  ETH trades:         {a['eth_trades']} ({a['eth_pnl']} adj)")
    lines.append(f"  Data span:          {days:.1f} days")

    # B) Per-Strategy Attribution
    lines.append("\n## B) Per-Strategy Attribution\n")
    if not per_strategy:
        lines.append("  No trades found.")
    else:
        header = f"  {'Strategy':<22} {'Trades':>6} {'WinRate':>8} {'AvgPnL':>10} {'TotalPnL':>10} {'AdjPnL':>10} {'Sharpe':>7} {'AvgHold':>8}"
        lines.append(header)
        lines.append("  " + "-" * 78)
        for s in per_strategy:
            lines.append(
                f"  {s['strategy']:<22} {s['trade_count']:>6} {s['win_rate']:>8} "
                f"{s['avg_pnl']:>10} {s['total_pnl']:>10} {s['adjusted_pnl']:>10} "
                f"{s['sharpe']:>7} {s['avg_hold_min']:>7}m"
            )

    # C) Regime-Conditioned Performance
    lines.append("\n## C) Regime-Conditioned Performance\n")
    if not regime_data:
        lines.append("  No trades found.")
    else:
        header = f"  {'Strategy':<22} {'Regime':<10} {'Trades':>6} {'WinRate':>8} {'AvgPnL':>10}"
        lines.append(header)
        lines.append("  " + "-" * 60)
        for r in regime_data:
            lines.append(
                f"  {r['strategy']:<22} {r['regime']:<10} {r['trade_count']:>6} "
                f"{r['win_rate']:>8} {r['avg_pnl']:>10}"
            )

    # D) Guard Effectiveness
    lines.append("\n## D) Guard Effectiveness\n")
    lines.append("  Guard denial counts are in-memory only (not persisted to DB).")
    lines.append("  Relative trade activity by strategy (proxy for guard pass-through rate):")
    if per_strategy:
        for s in per_strategy:
            lines.append(f"    {s['strategy']:<22} {s['trade_count']:>5} trades")
    else:
        lines.append("    No strategy-tagged trades found.")

    # E) Go-Live Scorecard
    lines.append("\n## E) Go-Live Scorecard\n")
    lines.append(f"  {'#':<3} {'Criterion':<42} {'Target':<20} {'Current':<25} {'Pass?'}")
    lines.append("  " + "-" * 98)
    for c in scorecard["criteria"]:
        pass_str = "PASS" if c["pass"] else "FAIL"
        lines.append(
            f"  {c['id']:<3} {c['name']:<42} {c['target']:<20} {c['current']:<25} {pass_str}"
        )

    lines.append("")
    if scorecard["kill_criteria"]:
        lines.append("  --- KILL CONDITIONS DETECTED ---")
        for k in scorecard["kill_criteria"]:
            lines.append(f"  *** {k}")
        lines.append("")

    verdict = scorecard["verdict"]
    lines.append(f"  VERDICT: {verdict}")
    lines.append(sep)

    return "\n".join(lines)


def _format_markdown(
    aggregate: dict[str, Any],
    per_strategy: list[dict[str, Any]],
    regime_data: list[dict[str, Any]],
    scorecard: dict[str, Any],
    days: float,
) -> str:
    """Format the full report as GitHub-flavored Markdown."""
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# CerebrumCoin Strategy Attribution Report")
    lines.append(f"\n_Generated: {ts}_\n")

    # A) Aggregate Overview
    lines.append("## A) Aggregate Overview\n")
    a = aggregate
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total trades | {a['total_trades']} |")
    lines.append(f"| Win rate | {a['win_rate']} |")
    lines.append(f"| Total P&L (raw) | {a['total_pnl']} |")
    lines.append(f"| Adj P&L (net comm) | {a['adjusted_pnl']} |")
    lines.append(f"| Sharpe ratio | {a['sharpe']} |")
    lines.append(f"| Sortino ratio | {a['sortino']} |")
    lines.append(f"| Max drawdown | {a['max_drawdown_pct']} |")
    lines.append(f"| BTC trades | {a['btc_trades']} ({a['btc_pnl']} adj) |")
    lines.append(f"| ETH trades | {a['eth_trades']} ({a['eth_pnl']} adj) |")
    lines.append(f"| Data span | {days:.1f} days |")

    # B) Per-Strategy Attribution
    lines.append("\n## B) Per-Strategy Attribution\n")
    lines.append("| Strategy | Trades | WinRate | AvgPnL | TotalPnL | AdjPnL | Sharpe | AvgHold |")
    lines.append("|----------|--------|---------|--------|----------|--------|--------|---------|")
    for s in per_strategy:
        lines.append(
            f"| {s['strategy']} | {s['trade_count']} | {s['win_rate']} | "
            f"{s['avg_pnl']} | {s['total_pnl']} | {s['adjusted_pnl']} | "
            f"{s['sharpe']} | {s['avg_hold_min']}m |"
        )

    # C) Regime-Conditioned Performance
    lines.append("\n## C) Regime-Conditioned Performance\n")
    lines.append("| Strategy | Regime | Trades | WinRate | AvgPnL |")
    lines.append("|----------|--------|--------|---------|--------|")
    for r in regime_data:
        lines.append(
            f"| {r['strategy']} | {r['regime']} | {r['trade_count']} | "
            f"{r['win_rate']} | {r['avg_pnl']} |"
        )

    # D) Guard Effectiveness
    lines.append("\n## D) Guard Effectiveness\n")
    lines.append("Guard denial counts are in-memory only (not persisted to DB).")
    lines.append("Relative trade activity by strategy (proxy for guard pass-through rate):\n")
    lines.append("| Strategy | Trades |")
    lines.append("|----------|--------|")
    for s in per_strategy:
        lines.append(f"| {s['strategy']} | {s['trade_count']} |")

    # E) Go-Live Scorecard
    lines.append("\n## E) Go-Live Scorecard\n")
    lines.append("| # | Criterion | Target | Current | Pass? |")
    lines.append("|---|-----------|--------|---------|-------|")
    for c in scorecard["criteria"]:
        pass_str = "PASS" if c["pass"] else "FAIL"
        lines.append(
            f"| {c['id']} | {c['name']} | {c['target']} | {c['current']} | {pass_str} |"
        )

    if scorecard["kill_criteria"]:
        lines.append("\n### KILL CONDITIONS DETECTED\n")
        for k in scorecard["kill_criteria"]:
            lines.append(f"> **{k}**")

    verdict = scorecard["verdict"]
    lines.append(f"\n### Verdict: **{verdict}**\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generate_report entry point
# ---------------------------------------------------------------------------

def generate_report(
    db_path: Path,
    strategy_only: bool = False,
    as_json: bool = False,
    output_path: Path | None = None,
    initial_balance: Decimal = Decimal("10000"),
) -> str:
    """
    Generate the full strategy attribution report.

    Reads all closed trades from db_path, computes all sections, and returns
    the report as a string (text, markdown, or JSON depending on flags).

    Args:
        db_path: Path to the SQLite database.
        strategy_only: If True, skip legacy trades in per-strategy section.
        as_json: If True, return JSON string instead of formatted text.
        output_path: If set, write markdown to this path in addition to returning text.
        initial_balance: Starting balance for drawdown calculation.

    Returns:
        Report string (text, markdown with output_path, or JSON).
    """
    all_rows = fetch_all_closed_trades(db_path)
    display_rows = fetch_strategy_trades(db_path) if strategy_only else all_rows

    aggregate = _section_aggregate(all_rows, initial_balance)
    per_strategy = _section_per_strategy(display_rows)
    regime_data = _section_regime(display_rows)
    scorecard = evaluate_scorecard(all_rows, initial_balance)
    days = _days_span(all_rows)

    if as_json:
        data = {
            "aggregate": aggregate,
            "per_strategy": per_strategy,
            "regime": regime_data,
            "scorecard": scorecard,
            "days_span": days,
        }
        return json.dumps(data, indent=2, default=str)

    if output_path is not None:
        md = _format_markdown(aggregate, per_strategy, regime_data, scorecard, days)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        return md

    return _format_text(aggregate, per_strategy, regime_data, scorecard, days)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for strategy attribution analysis."""
    parser = argparse.ArgumentParser(
        description="Strategy attribution analysis — reads cerebrum.db and reports "
                    "which strategies are making money."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/cerebrum.db"),
        help="Path to SQLite database (default: data/cerebrum.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this path (in addition to stdout).",
    )
    parser.add_argument(
        "--strategy-only",
        action="store_true",
        default=False,
        help="Skip legacy trades (NULL strategy_id) in per-strategy analysis.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="as_json",
        help="Output report as JSON for programmatic consumption.",
    )
    parser.add_argument(
        "--balance",
        type=Decimal,
        default=Decimal("10000"),
        help="Initial balance for drawdown calculation (default: 10000).",
    )

    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"Database not found: {args.db}")

    report = generate_report(
        args.db,
        strategy_only=args.strategy_only,
        as_json=args.as_json,
        output_path=args.output,
        initial_balance=args.balance,
    )
    print(report)
    if args.output and not args.as_json:
        print(f"\nMarkdown report written to: {args.output}")


if __name__ == "__main__":
    main()
