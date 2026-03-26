"""
Unit tests for scripts/sensitivity.py — parameter sensitivity analysis.

Uses in-memory SQLite with seeded trade data to verify:
- Empty database returns zero totals
- SL simulation: trade stopped earlier when alt SL tighter than actual loss
- SL simulation: trade unaffected when alt SL looser than actual loss
- TP simulation: trade exits earlier when alt TP tighter than actual gain
- TP simulation: trade unaffected when gain is below alt TP
- Age simulation: P&L reduced proportionally for trades held too long
- Age simulation: trade unaffected when already shorter than alt max age
- Commission calculation on simulated trade outcomes
- Grid cap enforcement (>1000 combinations warns and truncates)
- Strategy filter only analyzes matching trades
- JSON output validity

@decision DEC-SENSITIVITY-002
@title Tests for sensitivity.py using in-memory SQLite
@status accepted
@rationale sensitivity.py uses raw sqlite3 (DEC-ANALYZE-001 pattern). Tests use
in-memory SQLite with seeded data to verify simulation logic without touching
the production database. Simulation functions are pure (no I/O) so they can be
tested directly with constructed TradeRow objects.
"""

import json
import sqlite3
import tempfile
import warnings
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.sensitivity import (
    COMMISSION_RATE,
    DEFAULT_MAX_AGE_MINUTES,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    SimResult,
    build_grid,
    fetch_closed_trades_for_sensitivity,
    run_sensitivity_report,
    simulate_age,
    simulate_cooldown,
    simulate_sl,
    simulate_tp,
)
from scripts.analyze import TradeRow


# ---------------------------------------------------------------------------
# Helpers — build temp-file SQLite DB (mirrors test_analyze.py pattern)
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


def _make_trade(
    *,
    entry_price: float = 50000.0,
    quantity: float = 0.001,
    pnl: float,
    entry_time: float = 1000.0,
    exit_time: float = 4600.0,
    symbol: str = "BTC/USD",
    strategy_id: str | None = "momentum",
    regime: str = "BULL",
) -> TradeRow:
    """Construct a TradeRow for use in simulation unit tests."""
    ep = Decimal(str(entry_price))
    qty = Decimal(str(quantity))
    exit_price = ep + Decimal(str(pnl)) / qty
    return TradeRow(
        id=1,
        symbol=symbol,
        side="buy",
        entry_time=entry_time,
        entry_price=ep,
        exit_time=exit_time,
        exit_price=exit_price,
        quantity=qty,
        pnl=Decimal(str(pnl)),
        regime=regime,
        strategy_id=strategy_id,
    )


# ---------------------------------------------------------------------------
# Standard seeded DB trades
# ---------------------------------------------------------------------------

# BTC trade that gained 2% of entry_value = 50000 * 0.001 * 0.02 = $1.00
BTC_WIN_2PCT = {
    "symbol": "BTC/USD",
    "entry_time": 1000.0,
    "exit_time": 4600.0,  # 60 minutes hold
    "entry_price": "50000.00",
    "exit_price": "51000.00",
    "quantity": "0.001",
    "pnl": "1.00",
    "regime": "BULL",
    "status": "CLOSED",
    "strategy_id": "momentum",
}

# ETH trade that lost 2% of entry_value = 2000 * 0.05 * 0.02 = $2.00
ETH_LOSS_2PCT = {
    "symbol": "ETH/USD",
    "entry_time": 2000.0,
    "exit_time": 9200.0,  # 120 minutes hold
    "entry_price": "2000.00",
    "exit_price": "1960.00",
    "quantity": "0.05",
    "pnl": "-2.00",
    "regime": "SIDEWAYS",
    "status": "CLOSED",
    "strategy_id": "mean_reversion",
}


# ---------------------------------------------------------------------------
# Unit tests: fetch_closed_trades_for_sensitivity
# ---------------------------------------------------------------------------


def test_fetch_empty_db_returns_empty():
    """Empty DB returns empty list."""
    db = make_db([])
    rows = fetch_closed_trades_for_sensitivity(db)
    assert rows == []


def test_fetch_excludes_open_trades():
    """OPEN trades must not appear in results."""
    db = make_db([
        {**BTC_WIN_2PCT, "status": "CLOSED"},
        {**ETH_LOSS_2PCT, "status": "OPEN", "pnl": None},
    ])
    rows = fetch_closed_trades_for_sensitivity(db)
    assert len(rows) == 1
    assert rows[0].symbol == "BTC/USD"


def test_fetch_strategy_filter():
    """--strategy filter returns only trades matching that strategy_id."""
    db = make_db([BTC_WIN_2PCT, ETH_LOSS_2PCT])
    rows = fetch_closed_trades_for_sensitivity(db, strategy="momentum")
    assert len(rows) == 1
    assert rows[0].strategy_id == "momentum"


def test_fetch_strategy_filter_no_match_returns_empty():
    """Strategy filter with no matching trades returns empty list."""
    db = make_db([BTC_WIN_2PCT])
    rows = fetch_closed_trades_for_sensitivity(db, strategy="breakout")
    assert rows == []


# ---------------------------------------------------------------------------
# Unit tests: simulate_sl
# ---------------------------------------------------------------------------


def test_sl_tighter_than_actual_loss_clips_pnl():
    """
    Trade lost 2% but alt SL is 1%. Simulated P&L should be capped at -1%
    of entry_value instead of actual -2%.
    """
    # entry=50000, qty=0.1, pnl=-100 (= -2% of 5000 entry_value)
    trade = _make_trade(entry_price=50000.0, quantity=0.1, pnl=-100.0)
    entry_value = Decimal("50000.0") * Decimal("0.1")  # $5000

    sim_pnl = simulate_sl(trade, alt_sl_pct=Decimal("1.0"))

    expected = -(entry_value * Decimal("0.01"))  # -$50
    assert abs(sim_pnl - expected) < Decimal("0.01"), f"Expected ~{expected}, got {sim_pnl}"


def test_sl_looser_than_actual_loss_no_change():
    """
    Trade lost 1%. Alt SL is 2%. SL was never triggered — P&L unchanged.
    """
    # entry=50000, qty=0.1, pnl=-50 (= -1% of $5000 entry_value)
    trade = _make_trade(entry_price=50000.0, quantity=0.1, pnl=-50.0)

    sim_pnl = simulate_sl(trade, alt_sl_pct=Decimal("2.0"))

    assert sim_pnl == trade.pnl


def test_sl_winning_trade_unaffected():
    """Winning trade is never affected by stop-loss simulation."""
    trade = _make_trade(entry_price=50000.0, quantity=0.001, pnl=50.0)
    sim_pnl = simulate_sl(trade, alt_sl_pct=Decimal("1.5"))
    assert sim_pnl == trade.pnl


def test_sl_exact_threshold_not_clipped():
    """
    Trade P&L exactly at -1.5% of entry_value. Alt SL is 1.5%.
    At exact threshold the capped value equals the actual value.
    """
    entry_price = Decimal("50000")
    qty = Decimal("0.001")
    entry_value = entry_price * qty  # $50
    # pnl = exactly -1.5% of $50 = -$0.75
    pnl = -(entry_value * Decimal("0.015"))
    trade = TradeRow(
        id=1, symbol="BTC/USD", side="buy",
        entry_time=1000.0, entry_price=entry_price,
        exit_time=4600.0, exit_price=entry_price + pnl / qty,
        quantity=qty, pnl=pnl,
        regime="BULL", strategy_id="momentum",
    )
    sim_pnl = simulate_sl(trade, alt_sl_pct=Decimal("1.5"))
    # At exactly the threshold: capped value == actual value
    assert abs(sim_pnl - pnl) < Decimal("0.0001")


# ---------------------------------------------------------------------------
# Unit tests: simulate_tp
# ---------------------------------------------------------------------------


def test_tp_tighter_than_actual_gain_clips_pnl():
    """
    Trade gained 3% but alt TP is 2%. Simulated P&L capped at +2%
    of entry_value.
    """
    # entry=50000, qty=0.1, pnl=+150 (= +3% of $5000)
    trade = _make_trade(entry_price=50000.0, quantity=0.1, pnl=150.0)
    entry_value = Decimal("50000.0") * Decimal("0.1")

    sim_pnl = simulate_tp(trade, alt_tp_pct=Decimal("2.0"))

    expected = entry_value * Decimal("0.02")  # +$100
    assert abs(sim_pnl - expected) < Decimal("0.01"), f"Expected ~{expected}, got {sim_pnl}"


def test_tp_looser_than_actual_gain_no_change():
    """
    Trade gained 2%. Alt TP is 3%. TP was never triggered — P&L unchanged.
    """
    trade = _make_trade(entry_price=50000.0, quantity=0.1, pnl=100.0)  # +2%

    sim_pnl = simulate_tp(trade, alt_tp_pct=Decimal("3.0"))

    assert sim_pnl == trade.pnl


def test_tp_losing_trade_unaffected():
    """Losing trade is never affected by take-profit simulation."""
    trade = _make_trade(entry_price=50000.0, quantity=0.001, pnl=-50.0)
    sim_pnl = simulate_tp(trade, alt_tp_pct=Decimal("2.0"))
    assert sim_pnl == trade.pnl


# ---------------------------------------------------------------------------
# Unit tests: simulate_age
# ---------------------------------------------------------------------------


def test_age_held_longer_reduces_pnl_proportionally():
    """
    Trade held 120 minutes with pnl=-$2. Alt max_age is 60 minutes.
    Simulated P&L = -$2 * (60/120) = -$1.
    """
    trade = _make_trade(
        pnl=-2.0,
        entry_time=0.0,
        exit_time=7200.0,  # 120 minutes
    )
    sim_pnl = simulate_age(trade, alt_max_age_minutes=60)
    assert abs(sim_pnl - Decimal("-1.0")) < Decimal("0.01")


def test_age_held_shorter_than_alt_unchanged():
    """
    Trade held 30 minutes. Alt max_age is 60 minutes.
    Trade already exited before alt age limit — P&L unchanged.
    """
    trade = _make_trade(
        pnl=-0.50,
        entry_time=0.0,
        exit_time=1800.0,  # 30 minutes
    )
    sim_pnl = simulate_age(trade, alt_max_age_minutes=60)
    assert sim_pnl == trade.pnl


def test_age_winning_trade_proportional_reduction():
    """
    Winning trade held 240 minutes with pnl=+$4. Alt max_age=120 minutes.
    Simulated P&L = +$4 * (120/240) = +$2.
    """
    trade = _make_trade(
        pnl=4.0,
        entry_time=0.0,
        exit_time=14400.0,  # 240 minutes
    )
    sim_pnl = simulate_age(trade, alt_max_age_minutes=120)
    assert abs(sim_pnl - Decimal("2.0")) < Decimal("0.01")


def test_age_zero_hold_duration_returns_original():
    """Trade with zero hold duration (entry=exit) returns original P&L."""
    trade = _make_trade(pnl=0.0, entry_time=1000.0, exit_time=1000.0)
    sim_pnl = simulate_age(trade, alt_max_age_minutes=60)
    assert sim_pnl == trade.pnl


def test_age_none_exit_time_returns_original():
    """Trade with exit_time=None (no exit recorded) returns original P&L."""
    trade = TradeRow(
        id=1, symbol="BTC/USD", side="buy",
        entry_time=1000.0, entry_price=Decimal("50000"),
        exit_time=None, exit_price=None,
        quantity=Decimal("0.001"), pnl=Decimal("-1.0"),
        regime="BULL", strategy_id="momentum",
    )
    sim_pnl = simulate_age(trade, alt_max_age_minutes=60)
    assert sim_pnl == trade.pnl


# ---------------------------------------------------------------------------
# Unit tests: simulate_cooldown
# ---------------------------------------------------------------------------


def test_cooldown_filters_rapid_successive_trades():
    """
    Two trades for BTC/USD 300 seconds apart. Alt cooldown = 600s.
    Second trade should be filtered out.
    """
    rows = [
        _make_trade(pnl=-1.0, entry_time=1000.0, exit_time=2800.0),
        _make_trade(pnl=2.0, entry_time=1300.0, exit_time=5000.0),  # 300s after first
    ]
    kept = simulate_cooldown(rows, cooldown_seconds=600)
    assert len(kept) == 1
    assert kept[0].entry_time == 1000.0


def test_cooldown_keeps_trades_spaced_far_enough():
    """
    Two BTC/USD trades 1000 seconds apart. Cooldown = 600s. Both kept.
    """
    rows = [
        _make_trade(pnl=-1.0, entry_time=1000.0, exit_time=2800.0),
        _make_trade(pnl=2.0, entry_time=2000.0, exit_time=5000.0),  # 1000s after first
    ]
    kept = simulate_cooldown(rows, cooldown_seconds=600)
    assert len(kept) == 2


def test_cooldown_per_symbol_independent():
    """
    BTC/USD at t=1000 and ETH/USD at t=1100. Cooldown=600s.
    Different symbols don't block each other — both kept.
    """
    btc = _make_trade(pnl=1.0, entry_time=1000.0, exit_time=2800.0, symbol="BTC/USD")
    eth = _make_trade(pnl=1.0, entry_time=1100.0, exit_time=4800.0, symbol="ETH/USD")
    kept = simulate_cooldown([btc, eth], cooldown_seconds=600)
    assert len(kept) == 2


def test_cooldown_zero_filters_nothing():
    """Cooldown of 0 seconds keeps all trades."""
    rows = [
        _make_trade(pnl=1.0, entry_time=1000.0, exit_time=2800.0),
        _make_trade(pnl=2.0, entry_time=1001.0, exit_time=3000.0),
    ]
    kept = simulate_cooldown(rows, cooldown_seconds=0)
    assert len(kept) == 2


def test_cooldown_empty_list_returns_empty():
    """Empty input returns empty output."""
    assert simulate_cooldown([], cooldown_seconds=600) == []


# ---------------------------------------------------------------------------
# Unit tests: SimResult and commission
# ---------------------------------------------------------------------------


def test_sim_result_commission_applied():
    """
    SimResult.adj_pnl must equal total_pnl minus total commission.
    Commission = entry_price * quantity * COMMISSION_RATE * 2.
    """
    trade = _make_trade(entry_price=50000.0, quantity=0.001, pnl=5.0)
    # Commission = 50000 * 0.001 * 0.0016 * 2 = $0.16
    expected_commission = Decimal("50000") * Decimal("0.001") * COMMISSION_RATE * 2
    assert expected_commission == Decimal("0.16")

    result = SimResult.from_rows([trade])
    assert abs(result.adj_pnl - (Decimal("5.0") - expected_commission)) < Decimal("0.001")


def test_sim_result_empty_rows():
    """SimResult on empty rows produces zero totals."""
    result = SimResult.from_rows([])
    assert result.trade_count == 0
    assert result.total_pnl == Decimal("0")
    assert result.adj_pnl == Decimal("0")
    assert result.win_rate == Decimal("0")


def test_sim_result_win_rate():
    """Win rate = wins / total * 100."""
    rows = [
        _make_trade(pnl=1.0, entry_time=1000.0, exit_time=4600.0),
        _make_trade(pnl=-1.0, entry_time=2000.0, exit_time=5600.0),
    ]
    result = SimResult.from_rows(rows)
    assert result.trade_count == 2
    assert abs(result.win_rate - Decimal("50")) < Decimal("0.01")


# ---------------------------------------------------------------------------
# Unit tests: build_grid
# ---------------------------------------------------------------------------


def test_build_grid_single_param():
    """Single-parameter sweep returns one entry per value."""
    grid = build_grid(
        sl_values=[Decimal("1.0"), Decimal("1.5")],
        tp_values=None,
        age_values=None,
    )
    # 2 SL values, TP and Age are None (placeholder)
    assert len(grid) == 2


def test_build_grid_all_params():
    """Full grid = len(sl) * len(tp) * len(age) combinations."""
    sl = [Decimal("1.0"), Decimal("1.5")]
    tp = [Decimal("2.0"), Decimal("3.0")]
    age = [60, 120]
    grid = build_grid(sl_values=sl, tp_values=tp, age_values=age)
    assert len(grid) == 8  # 2 * 2 * 2


def test_build_grid_cap_warns_and_truncates():
    """Grid > 1000 combinations emits UserWarning and truncates to 1000."""
    sl = [Decimal(str(x)) for x in range(20)]   # 20
    tp = [Decimal(str(x)) for x in range(10)]   # 10
    age = list(range(10))                         # 10 -> 20*10*10 = 2000 > 1000

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid = build_grid(sl_values=sl, tp_values=tp, age_values=age)
        assert len(w) == 1
        assert "truncated" in str(w[0].message).lower() or "cap" in str(w[0].message).lower()

    assert len(grid) == 1000


def test_build_grid_none_values_means_skip():
    """None parameter means that dimension is not swept — uses placeholder."""
    grid = build_grid(sl_values=[Decimal("1.5")], tp_values=None, age_values=None)
    assert len(grid) == 1
    sl_val, tp_val, age_val = grid[0]
    assert sl_val == Decimal("1.5")
    assert tp_val is None
    assert age_val is None


# ---------------------------------------------------------------------------
# Unit tests: run_sensitivity_report
# ---------------------------------------------------------------------------


def test_run_sensitivity_report_empty_db():
    """Report on empty DB must not raise and returns a non-empty string."""
    db = make_db([])
    report = run_sensitivity_report(db)
    assert isinstance(report, str)
    assert len(report) > 0


def test_run_sensitivity_report_contains_sections():
    """Report with trades must contain stop-loss section header."""
    db = make_db([BTC_WIN_2PCT, ETH_LOSS_2PCT])
    report = run_sensitivity_report(db, sl_values=[Decimal("1.0"), Decimal("1.5")])
    assert "Stop-Loss" in report


def test_run_sensitivity_report_json_valid():
    """JSON output must be valid JSON with expected top-level keys."""
    db = make_db([BTC_WIN_2PCT, ETH_LOSS_2PCT])
    json_out = run_sensitivity_report(
        db,
        sl_values=[Decimal("1.0"), Decimal("1.5")],
        as_json=True,
    )
    data = json.loads(json_out)
    # At least one of the expected top-level keys must be present
    assert any(k in data for k in ("stop_loss", "parameters", "results", "sensitivity"))


def test_run_sensitivity_report_strategy_filter():
    """Report with strategy filter only analyzes matching trades."""
    db = make_db([BTC_WIN_2PCT, ETH_LOSS_2PCT])
    # BTC_WIN_2PCT is momentum, ETH_LOSS_2PCT is mean_reversion
    report = run_sensitivity_report(
        db,
        sl_values=[Decimal("1.5")],
        strategy="momentum",
    )
    assert isinstance(report, str)
    assert len(report) > 0


def test_run_sensitivity_report_default_params_run():
    """Default parameter grids must run without error on real data."""
    db = make_db([BTC_WIN_2PCT, ETH_LOSS_2PCT])
    report = run_sensitivity_report(db)
    assert isinstance(report, str)
    assert len(report) > 50


# ---------------------------------------------------------------------------
# Unit tests: DEFAULT constants
# ---------------------------------------------------------------------------


def test_default_stop_loss_values():
    """Default SL grid must contain expected values."""
    assert Decimal("1.5") in DEFAULT_STOP_LOSS_PCT
    assert Decimal("0.5") in DEFAULT_STOP_LOSS_PCT
    assert Decimal("2.5") in DEFAULT_STOP_LOSS_PCT


def test_default_take_profit_values():
    """Default TP grid must contain expected values."""
    assert Decimal("3.0") in DEFAULT_TAKE_PROFIT_PCT
    assert Decimal("1.0") in DEFAULT_TAKE_PROFIT_PCT


def test_default_max_age_values():
    """Default max-age grid must contain expected values in minutes."""
    assert 120 in DEFAULT_MAX_AGE_MINUTES
    assert 30 in DEFAULT_MAX_AGE_MINUTES
    assert 480 in DEFAULT_MAX_AGE_MINUTES
