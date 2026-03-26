"""
Unit tests for scripts/analyze.py — strategy attribution analysis.

Uses in-memory SQLite with seeded trade data to verify:
- Empty database returns zero totals, INSUFFICIENT DATA scorecard
- All NULL strategy_id treated as legacy bucket
- Mixed legacy + strategy-tagged trades split correctly
- Scorecard PASS/FAIL criteria logic
- Kill criterion detection (drawdown > 7%)
- Commission-adjusted P&L calculation accuracy
- Per-strategy Sharpe uses percentage returns (pnl / entry_value)
- JSON output mode produces valid structure
- --strategy-only flag skips legacy trades

@decision DEC-ANALYZE-002
@title Tests for analyze.py using in-memory SQLite
@status accepted
@rationale analyze.py uses raw sqlite3 (DEC-EXPORT-001 pattern). Tests use
in-memory SQLite with seeded data to verify all report sections, scorecard
logic, and output modes without touching the production database.
"""

import json
import sqlite3
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.analyze import (
    COMMISSION_RATE,
    STRATEGY_TRADE_MINIMUMS,
    TradeRow,
    calculate_commission,
    calculate_drawdown_from_rows,
    calculate_sharpe_from_rows,
    evaluate_scorecard,
    fetch_all_closed_trades,
    fetch_strategy_trades,
    generate_report,
    group_by_regime,
    group_by_strategy,
)


# ---------------------------------------------------------------------------
# Helpers — build temp-file SQLite DB
# ---------------------------------------------------------------------------


def make_db(trades: list[dict]) -> Path:
    """Create a temp SQLite DB with a trades table seeded from dicts."""
    tmp = Path(tempfile.mktemp(suffix=".db"))
    conn = sqlite3.connect(str(tmp))
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_time REAL NOT NULL,
            entry_price TEXT NOT NULL,
            exit_time REAL,
            exit_price TEXT,
            quantity TEXT NOT NULL,
            pnl TEXT,
            signal_snapshot TEXT NOT NULL DEFAULT '{}',
            regime TEXT NOT NULL DEFAULT 'UNKNOWN',
            status TEXT NOT NULL DEFAULT 'CLOSED',
            strategy_id TEXT
        )
        """
    )
    for t in trades:
        conn.execute(
            """
            INSERT INTO trades
                (symbol, side, entry_time, entry_price, exit_time, exit_price,
                 quantity, pnl, regime, status, strategy_id)
            VALUES (:symbol, :side, :entry_time, :entry_price, :exit_time,
                    :exit_price, :quantity, :pnl, :regime, :status, :strategy_id)
            """,
            {
                "symbol": t.get("symbol", "BTC/USD"),
                "side": t.get("side", "buy"),
                "entry_time": t.get("entry_time", 1000.0),
                "entry_price": str(t.get("entry_price", "50000.00")),
                "exit_time": t.get("exit_time"),
                "exit_price": str(t.get("exit_price")) if t.get("exit_price") is not None else None,
                "quantity": str(t.get("quantity", "0.001")),
                "pnl": str(t.get("pnl")) if t.get("pnl") is not None else None,
                "regime": t.get("regime", "BULL"),
                "status": t.get("status", "CLOSED"),
                "strategy_id": t.get("strategy_id"),
            },
        )
    conn.commit()
    conn.close()
    return tmp


# Standard seeded trades for mixed tests
BTC_WIN = {
    "symbol": "BTC/USD",
    "entry_time": 1000.0,
    "exit_time": 4600.0,
    "entry_price": "50000.00",
    "exit_price": "51000.00",
    "quantity": "0.004",
    "pnl": "4.00",
    "regime": "BULL",
    "strategy_id": "momentum",
}

ETH_LOSS = {
    "symbol": "ETH/USD",
    "entry_time": 2000.0,
    "exit_time": 5600.0,
    "entry_price": "2000.00",
    "exit_price": "1960.00",
    "quantity": "0.05",
    "pnl": "-2.00",
    "regime": "SIDEWAYS",
    "strategy_id": "mean_reversion",
}

BTC_LEGACY = {
    "symbol": "BTC/USD",
    "entry_time": 500.0,
    "exit_time": 3000.0,
    "entry_price": "48000.00",
    "exit_price": "48500.00",
    "quantity": "0.002",
    "pnl": "1.00",
    "regime": "BULL",
    "strategy_id": None,  # legacy
}


# ---------------------------------------------------------------------------
# Unit tests: calculate_commission
# ---------------------------------------------------------------------------


def test_commission_round_trip():
    """Round-trip commission = entry_price * qty * COMMISSION_RATE * 2."""
    result = calculate_commission(Decimal("50000"), Decimal("0.001"))
    expected = Decimal("50000") * Decimal("0.001") * COMMISSION_RATE * 2
    assert result == expected


def test_commission_is_positive():
    """Commission must always be non-negative."""
    result = calculate_commission(Decimal("1000"), Decimal("0.1"))
    assert result > 0


def test_commission_accuracy():
    """Commission on $50k * 0.001 BTC must equal exactly $0.16."""
    result = calculate_commission(Decimal("50000"), Decimal("0.001"))
    assert result == Decimal("0.16")


# ---------------------------------------------------------------------------
# Unit tests: calculate_sharpe_from_rows
# ---------------------------------------------------------------------------


def test_sharpe_empty_returns_zero():
    """Sharpe of empty list must be 0."""
    assert calculate_sharpe_from_rows([]) == Decimal("0")


def test_sharpe_uses_percentage_returns():
    """Sharpe must use pnl / (entry_price * quantity), not raw dollar P&L."""
    # Perfect consistency (all same pct return) -> sentinel value 1000000
    rows = [
        TradeRow(
            id=i,
            symbol="BTC/USD",
            side="buy",
            entry_time=float(i * 1000),
            entry_price=Decimal("100"),
            exit_time=float(i * 1000 + 3600),
            exit_price=Decimal("105"),
            quantity=Decimal("1"),
            pnl=Decimal("5"),
            regime="BULL",
            strategy_id="momentum",
        )
        for i in range(5)
    ]
    sharpe = calculate_sharpe_from_rows(rows)
    assert sharpe == Decimal("1000000")


def test_sharpe_mixed_wins_losses_is_finite():
    """Sharpe on mixed PnL must be a finite non-zero Decimal."""
    rows = [
        TradeRow(
            id=1, symbol="BTC/USD", side="buy",
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=4600.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.004"), pnl=Decimal("4"),
            regime="BULL", strategy_id="momentum",
        ),
        TradeRow(
            id=2, symbol="ETH/USD", side="buy",
            entry_time=2000.0, entry_price=Decimal("2000"),
            exit_time=5600.0, exit_price=Decimal("1980"),
            quantity=Decimal("0.05"), pnl=Decimal("-1"),
            regime="SIDEWAYS", strategy_id="momentum",
        ),
    ]
    sharpe = calculate_sharpe_from_rows(rows)
    assert isinstance(sharpe, Decimal)
    assert sharpe != Decimal("0")
    assert sharpe != Decimal("1000000")


# ---------------------------------------------------------------------------
# Unit tests: calculate_drawdown_from_rows
# ---------------------------------------------------------------------------


def test_drawdown_empty():
    """Drawdown on empty rows returns (0, 0)."""
    dd_abs, dd_pct = calculate_drawdown_from_rows([], Decimal("10000"))
    assert dd_abs == Decimal("0")
    assert dd_pct == Decimal("0")


def test_drawdown_all_wins_is_zero():
    """All-winning equity curve has zero drawdown."""
    rows = [
        TradeRow(
            id=i, symbol="BTC/USD", side="buy",
            entry_time=float(i * 1000), entry_price=Decimal("50000"),
            exit_time=float(i * 1000 + 3600), exit_price=Decimal("51000"),
            quantity=Decimal("0.001"), pnl=Decimal("1"),
            regime="BULL", strategy_id="momentum",
        )
        for i in range(3)
    ]
    dd_abs, dd_pct = calculate_drawdown_from_rows(rows, Decimal("10000"))
    assert dd_abs == Decimal("0")
    assert dd_pct == Decimal("0")


def test_drawdown_peak_then_loss():
    """After a peak, a loss must be detected as drawdown."""
    rows = [
        TradeRow(
            id=1, symbol="BTC/USD", side="buy",
            entry_time=1000.0, entry_price=Decimal("50000"),
            exit_time=4600.0, exit_price=Decimal("51000"),
            quantity=Decimal("0.001"), pnl=Decimal("100"),
            regime="BULL", strategy_id="momentum",
        ),
        TradeRow(
            id=2, symbol="BTC/USD", side="buy",
            entry_time=2000.0, entry_price=Decimal("50000"),
            exit_time=5600.0, exit_price=Decimal("49000"),
            quantity=Decimal("0.001"), pnl=Decimal("-200"),
            regime="BEAR", strategy_id="momentum",
        ),
    ]
    dd_abs, dd_pct = calculate_drawdown_from_rows(rows, Decimal("10000"))
    assert dd_abs == Decimal("200")
    expected_pct = Decimal("200") / Decimal("10100") * Decimal("100")
    assert abs(dd_pct - expected_pct) < Decimal("0.001")


# ---------------------------------------------------------------------------
# Unit tests: fetch_all_closed_trades, fetch_strategy_trades
# ---------------------------------------------------------------------------


def test_fetch_all_closed_trades_empty():
    """Empty DB returns empty list."""
    db = make_db([])
    rows = fetch_all_closed_trades(db)
    assert rows == []


def test_fetch_all_closed_trades_excludes_open():
    """OPEN trades must not appear in result."""
    db = make_db([
        {**BTC_WIN, "status": "CLOSED"},
        {**ETH_LOSS, "status": "OPEN", "pnl": None},
    ])
    rows = fetch_all_closed_trades(db)
    assert len(rows) == 1
    assert rows[0].symbol == "BTC/USD"


def test_fetch_all_closed_trades_includes_legacy():
    """Trades with NULL strategy_id must be returned by fetch_all."""
    db = make_db([BTC_WIN, BTC_LEGACY])
    rows = fetch_all_closed_trades(db)
    assert len(rows) == 2


def test_fetch_strategy_trades_excludes_legacy():
    """fetch_strategy_trades must exclude rows with NULL strategy_id."""
    db = make_db([BTC_WIN, BTC_LEGACY])
    rows = fetch_strategy_trades(db)
    assert len(rows) == 1
    assert rows[0].strategy_id == "momentum"


def test_fetch_strategy_trades_empty_when_all_legacy():
    """DB with only legacy trades returns empty list from fetch_strategy_trades."""
    db = make_db([BTC_LEGACY])
    rows = fetch_strategy_trades(db)
    assert rows == []


def test_fetch_all_ordered_by_entry_time():
    """fetch_all_closed_trades must return rows ordered by entry_time ASC."""
    db = make_db([BTC_WIN, ETH_LOSS, BTC_LEGACY])
    rows = fetch_all_closed_trades(db)
    times = [r.entry_time for r in rows]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# Unit tests: group_by_strategy
# ---------------------------------------------------------------------------


def test_group_by_strategy_creates_legacy_bucket():
    """Rows with strategy_id=None must appear in 'legacy (NULL)' bucket."""
    all_rows = fetch_all_closed_trades(make_db([BTC_WIN, BTC_LEGACY]))
    groups = group_by_strategy(all_rows)
    assert "legacy (NULL)" in groups
    assert "momentum" in groups


def test_group_by_strategy_counts():
    """Each strategy bucket must contain the correct trade count."""
    db = make_db([BTC_WIN, ETH_LOSS, BTC_LEGACY])
    all_rows = fetch_all_closed_trades(db)
    groups = group_by_strategy(all_rows)
    assert len(groups["momentum"]) == 1
    assert len(groups["mean_reversion"]) == 1
    assert len(groups["legacy (NULL)"]) == 1


def test_group_by_strategy_empty_returns_empty():
    """Empty row list produces empty groups dict."""
    assert group_by_strategy([]) == {}


# ---------------------------------------------------------------------------
# Unit tests: group_by_regime
# ---------------------------------------------------------------------------


def test_group_by_regime_splits_correctly():
    """Rows must be grouped by regime field."""
    db = make_db([BTC_WIN, ETH_LOSS])
    rows = fetch_all_closed_trades(db)
    groups = group_by_regime(rows)
    assert "BULL" in groups
    assert "SIDEWAYS" in groups
    assert len(groups["BULL"]) == 1
    assert len(groups["SIDEWAYS"]) == 1


def test_group_by_regime_empty():
    """Empty rows return empty dict."""
    assert group_by_regime([]) == {}


# ---------------------------------------------------------------------------
# Scorecard helpers
# ---------------------------------------------------------------------------


def _make_many_trades(
    strategy: str,
    count: int,
    pnl_per_trade: float,
    regime: str = "BULL",
    entry_time_start: float = 0.0,
) -> list[dict]:
    """Generate `count` seeded trades for a single strategy, spread 12h apart."""
    trades = []
    for i in range(count):
        entry_t = entry_time_start + i * 43200.0  # 12h apart -> 60 trades = 30 days
        trades.append(
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "entry_time": entry_t,
                "entry_price": "50000.00",
                "exit_time": entry_t + 3600.0,
                "exit_price": "51000.00",
                "quantity": "0.001",
                "pnl": str(pnl_per_trade),
                "regime": regime,
                "status": "CLOSED",
                "strategy_id": strategy,
            }
        )
    return trades


# ---------------------------------------------------------------------------
# Unit tests: evaluate_scorecard
# ---------------------------------------------------------------------------


def test_scorecard_insufficient_data_empty():
    """Empty DB must produce INSUFFICIENT DATA verdict."""
    db = make_db([])
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    assert scorecard["verdict"] == "INSUFFICIENT DATA"


def test_scorecard_net_pnl_passes_on_winners():
    """All winning trades must produce a passing net P&L criterion."""
    trades = _make_many_trades("momentum", 60, 5.0)
    db = make_db(trades)
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    net_crit = next(c for c in scorecard["criteria"] if c["id"] == 2)
    assert net_crit["pass"] is True


def test_scorecard_net_pnl_fails_on_losers():
    """Consistent losses must fail the net P&L criterion."""
    trades = _make_many_trades("momentum", 60, -5.0)
    db = make_db(trades)
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    net_crit = next(c for c in scorecard["criteria"] if c["id"] == 2)
    assert net_crit["pass"] is False


def test_scorecard_kill_criterion_on_large_drawdown():
    """Drawdown > 7% must trigger a KILL warning in kill_criteria."""
    trades = [
        {
            "symbol": "BTC/USD", "entry_time": 1000.0, "exit_time": 4600.0,
            "entry_price": "50000", "exit_price": "51000",
            "quantity": "0.001", "pnl": "100",
            "regime": "BULL", "status": "CLOSED", "strategy_id": "momentum",
        },
        {
            "symbol": "BTC/USD", "entry_time": 2000.0, "exit_time": 5600.0,
            "entry_price": "50000", "exit_price": "45000",
            "quantity": "0.2", "pnl": "-1000",
            "regime": "BEAR", "status": "CLOSED", "strategy_id": "momentum",
        },
    ]
    db = make_db(trades)
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    assert len(scorecard["kill_criteria"]) > 0
    kill_text = " ".join(scorecard["kill_criteria"])
    assert "KILL" in kill_text


def test_scorecard_verdict_no_go_or_insufficient_on_few_trades():
    """Two trades yield NO-GO or INSUFFICIENT DATA, not GO."""
    db = make_db([BTC_WIN, ETH_LOSS])
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    assert scorecard["verdict"] in ("NO-GO", "INSUFFICIENT DATA")


def test_scorecard_has_eight_criteria():
    """Scorecard must always have exactly 8 criteria entries."""
    db = make_db([BTC_WIN])
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    assert len(scorecard["criteria"]) == 8


def test_scorecard_criteria_have_required_keys():
    """Each criterion entry must have id, name, target, current, pass keys."""
    db = make_db([BTC_WIN])
    rows = fetch_all_closed_trades(db)
    scorecard = evaluate_scorecard(rows, Decimal("10000"))
    for criterion in scorecard["criteria"]:
        for key in ("id", "name", "target", "current", "pass"):
            assert key in criterion


# ---------------------------------------------------------------------------
# Unit tests: generate_report (integration)
# ---------------------------------------------------------------------------


def test_generate_report_empty_db_does_not_raise():
    """generate_report on empty DB must not raise and returns a string."""
    db = make_db([])
    report = generate_report(db, strategy_only=False)
    assert isinstance(report, str)
    assert len(report) > 0


def test_generate_report_contains_required_sections():
    """Report must contain all expected section headers."""
    trades = _make_many_trades("momentum", 10, 2.0)
    db = make_db(trades + [BTC_LEGACY])
    report = generate_report(db, strategy_only=False)
    assert "Aggregate Overview" in report
    assert "Per-Strategy Attribution" in report
    assert "Regime" in report
    assert "Go-Live Scorecard" in report


def test_generate_report_strategy_only_excludes_legacy():
    """With strategy_only=True, legacy (NULL) must not appear in output."""
    db = make_db([BTC_WIN, BTC_LEGACY])
    report = generate_report(db, strategy_only=True)
    assert "legacy (NULL)" not in report


def test_generate_report_json_mode_parseable():
    """JSON output must be valid JSON with expected top-level keys."""
    db = make_db([BTC_WIN, ETH_LOSS])
    report_json = generate_report(db, strategy_only=False, as_json=True)
    data = json.loads(report_json)
    assert "aggregate" in data
    assert "per_strategy" in data
    assert "scorecard" in data


def test_generate_report_markdown_writes_file(tmp_path):
    """output_path argument writes a markdown file to disk."""
    db = make_db([BTC_WIN, ETH_LOSS])
    out_path = tmp_path / "reports" / "test.md"
    generate_report(db, strategy_only=False, output_path=out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "#" in content


# ---------------------------------------------------------------------------
# Unit tests: Commission-adjusted P&L accuracy
# ---------------------------------------------------------------------------


def test_commission_adjusted_pnl_accuracy():
    """Commission-adjusted P&L must equal raw PnL minus calculated commission."""
    # entry=$50000, qty=0.001, pnl=$5.00
    # Commission = 50000 * 0.001 * 0.0016 * 2 = $0.16
    # Adjusted = 5.00 - 0.16 = 4.84
    entry_price = Decimal("50000")
    quantity = Decimal("0.001")
    raw_pnl = Decimal("5.00")
    commission = calculate_commission(entry_price, quantity)
    adjusted = raw_pnl - commission
    assert commission == Decimal("0.16")
    assert adjusted == Decimal("4.84")


def test_commission_drag_percentage():
    """Commission drag % = total_commission / gross_pnl * 100."""
    gross = Decimal("5.00")
    commission = Decimal("0.16")
    drag = commission / gross * Decimal("100")
    assert abs(drag - Decimal("3.2")) < Decimal("0.01")


# ---------------------------------------------------------------------------
# Unit tests: STRATEGY_TRADE_MINIMUMS constants
# ---------------------------------------------------------------------------


def test_strategy_trade_minimums_all_defined():
    """All six strategies must have minimums defined."""
    expected = {
        "momentum", "mean_reversion", "breakout",
        "range_trading", "swing_trading", "news_driven",
    }
    assert set(STRATEGY_TRADE_MINIMUMS.keys()) == expected


def test_strategy_trade_minimums_values():
    """Trade minimums must match spec values exactly."""
    assert STRATEGY_TRADE_MINIMUMS["momentum"] == 50
    assert STRATEGY_TRADE_MINIMUMS["mean_reversion"] == 50
    assert STRATEGY_TRADE_MINIMUMS["breakout"] == 50
    assert STRATEGY_TRADE_MINIMUMS["range_trading"] == 30
    assert STRATEGY_TRADE_MINIMUMS["swing_trading"] == 15
    assert STRATEGY_TRADE_MINIMUMS["news_driven"] == 10
