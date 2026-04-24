"""
Unit tests for PortfolioTracker snapshot persistence and PaperTradingAdapter v2 state format.

Covers:
- save_snapshot / restore_snapshot round-trip (no positions)
- save_snapshot / restore_snapshot with open positions
- PaperTradingAdapter _save_state writes strategy_snapshots (v2 format)
- PaperTradingAdapter _load_state handles v1 files (no version key) without error
- Restore ignores snapshots for strategies removed between restarts
- New strategy (no snapshot) starts fresh

@decision DEC-PERSIST-001
@title Per-strategy PortfolioTracker snapshots in paper_state.json
@status accepted
@rationale Each strategy has an isolated PortfolioTracker (cash, positions, peak_equity,
realized_pnl). Without per-strategy snapshots, all per-strategy equity is lost on restart
and the dashboard shows stale global aggregates. v2 state format adds a "strategy_snapshots"
key alongside the existing v1 fields. v1 files load cleanly (no version key = treat as v1,
empty snapshots). The initial_balance field is stored but not restored — it is fixed at
construction time and must not drift across restarts.
"""

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.risk.portfolio import PortfolioTracker, Position
from cerebrum.adapters.paper import PaperTradingAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Start and stop a real EventBus per test."""
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def temp_state_file():
    """Yield a temp path. File is deleted so adapter treats it as a fresh start."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        p = Path(f.name)
    p.unlink()
    yield p
    if p.exists():
        p.unlink()


def _make_tracker(
    bus: EventBus,
    balance: Decimal = Decimal("10000"),
    strategy_id: str = "test",
) -> PortfolioTracker:
    return PortfolioTracker(bus=bus, initial_balance=balance, strategy_id=strategy_id)


def _make_adapter(
    bus: EventBus,
    state_file: Path,
    balance: Decimal = Decimal("10000"),
) -> PaperTradingAdapter:
    return PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=balance,
        commission_percent=Decimal("0.1"),
        slippage_percent=Decimal("0.05"),
        state_file=state_file,
    )


# ---------------------------------------------------------------------------
# PortfolioTracker snapshot tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_save_restore_roundtrip(bus):
    """save_snapshot then restore_snapshot on a fresh tracker produces identical state."""
    tracker = _make_tracker(bus, Decimal("10000"), "momentum")

    # Simulate Conductor reallocating cash to this strategy
    tracker.adjust_balance(Decimal("500"))   # cash = 10500

    snapshot = tracker.save_snapshot()

    # Restore into a new tracker
    tracker2 = _make_tracker(bus, Decimal("10000"), "momentum")
    tracker2.restore_snapshot(snapshot)

    assert tracker2.get_cash_balance() == tracker.get_cash_balance()
    realized1, unrealized1 = tracker.get_pnl()
    realized2, unrealized2 = tracker2.get_pnl()
    assert realized2 == realized1
    # No positions in either
    assert tracker2.get_all_positions() == {}


@pytest.mark.asyncio
async def test_portfolio_save_restore_with_positions(bus):
    """
    Snapshot with open positions round-trips: symbol, amount, average_entry_price,
    current_price, realized_pnl, entry_time all preserved.
    """
    tracker = _make_tracker(bus, Decimal("10000"), "breakout")

    # Inject a position directly (bypassing fill event for simplicity)
    entry_ts = 1_700_000_000.0
    tracker._positions["BTC/USD"] = Position(
        symbol="BTC/USD",
        amount=Decimal("0.05"),
        average_entry_price=Decimal("50000"),
        current_price=Decimal("51000"),
        unrealized_pnl=Decimal("50"),
        realized_pnl=Decimal("10"),
        entry_time=entry_ts,
    )
    tracker._cash_balance = Decimal("7500")
    tracker._total_realized_pnl = Decimal("10")
    tracker._peak_equity = Decimal("10200")

    snapshot = tracker.save_snapshot()

    tracker2 = _make_tracker(bus, Decimal("10000"), "breakout")
    tracker2.restore_snapshot(snapshot)

    assert tracker2.get_cash_balance() == Decimal("7500")
    realized, _ = tracker2.get_pnl()
    assert realized == Decimal("10")

    pos = tracker2.get_position("BTC/USD")
    assert pos is not None
    assert pos.symbol == "BTC/USD"
    assert pos.amount == Decimal("0.05")
    assert pos.average_entry_price == Decimal("50000")
    assert pos.current_price == Decimal("51000")
    assert pos.realized_pnl == Decimal("10")
    assert pos.entry_time == entry_ts


@pytest.mark.asyncio
async def test_portfolio_snapshot_peak_equity_preserved(bus):
    """Peak equity is stored and restored so drawdown calculations remain accurate."""
    tracker = _make_tracker(bus, Decimal("5000"), "mean_reversion")
    tracker._peak_equity = Decimal("5800")
    tracker._cash_balance = Decimal("4900")

    snapshot = tracker.save_snapshot()
    tracker2 = _make_tracker(bus, Decimal("5000"), "mean_reversion")
    tracker2.restore_snapshot(snapshot)

    assert tracker2._peak_equity == Decimal("5800")
    assert tracker2.get_cash_balance() == Decimal("4900")


# ---------------------------------------------------------------------------
# PaperTradingAdapter v2 state format tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_state_v2_includes_strategy_snapshots(bus, temp_state_file):
    """After set_strategy_portfolios(), _save_state() writes CURRENT_STATE_VERSION and strategy_snapshots.

    Previously this test asserted version==2 (the old hardcoded literal). Since the
    DEC-CONDUCTOR-012 companion fix, _save_state() writes CURRENT_STATE_VERSION (4)
    so migrations are skipped on restart and closed_trades survive. The test name is
    preserved for history; the assertion is updated to the correct value.
    """
    from cerebrum.adapters.paper import CURRENT_STATE_VERSION

    adapter = _make_adapter(bus, temp_state_file)
    await adapter.connect()

    t1 = _make_tracker(bus, Decimal("3000"), "momentum")
    t1._cash_balance = Decimal("2800")
    t2 = _make_tracker(bus, Decimal("2000"), "breakout")
    t2._cash_balance = Decimal("2100")

    adapter.set_strategy_portfolios({"momentum": t1, "breakout": t2})
    adapter._save_state()

    raw = json.loads(temp_state_file.read_text())

    assert raw.get("version") == CURRENT_STATE_VERSION
    assert "strategy_snapshots" in raw
    snaps = raw["strategy_snapshots"]
    assert "momentum" in snaps
    assert "breakout" in snaps
    assert snaps["momentum"]["cash_balance"] == "2800"
    assert snaps["breakout"]["cash_balance"] == "2100"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_state_v1_backward_compat(bus, temp_state_file):
    """
    Loading a v1 state file (no 'version' key, no 'strategy_snapshots') succeeds
    without error and returns empty snapshots.
    """
    v1_state = {
        "balances": {"USD": "9500.00"},
        "positions": {"BTC/USD": "0.1"},
        "current_prices": {"BTC/USD": "48000"},
        "trade_history": [],
        # No 'version', no 'strategy_snapshots'
    }
    temp_state_file.write_text(json.dumps(v1_state))

    adapter = _make_adapter(bus, temp_state_file)
    await adapter.connect()   # must not raise

    # v1 files are migrated to v3 then v4 on connect() (both migrations are
    # idempotent and always run). The version in-memory after load reflects
    # the fully-migrated file.
    assert adapter._state_version == 4
    assert adapter.get_strategy_snapshot("momentum") is None
    assert adapter.get_strategy_snapshot("any_strategy") is None

    # Balances loaded correctly from v1 file
    balance = await adapter.get_balance("USD")
    assert balance == Decimal("9500.00")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_restore_with_missing_strategy(bus, temp_state_file):
    """
    A strategy that existed in the saved snapshot but is no longer active is
    available via get_strategy_snapshot() but ignored when the caller skips it —
    no error, active strategies still restore correctly.
    """
    v2_state = {
        "version": 2,
        "balances": {"USD": "8000"},
        "positions": {},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "momentum": {
                "cash_balance": "4000",
                "initial_balance": "5000",
                "peak_equity": "4200",
                "total_realized_pnl": "50",
                "positions": {},
            },
            "old_strategy": {
                "cash_balance": "4000",
                "initial_balance": "5000",
                "peak_equity": "4100",
                "total_realized_pnl": "20",
                "positions": {},
            },
        },
    }
    temp_state_file.write_text(json.dumps(v2_state))

    adapter = _make_adapter(bus, temp_state_file)
    await adapter.connect()   # must not raise

    # Active strategy snapshot is accessible
    momentum_snap = adapter.get_strategy_snapshot("momentum")
    assert momentum_snap is not None
    assert momentum_snap["cash_balance"] == "4000"

    # Caller only restores active strategies — old_strategy is simply not restored
    tracker = _make_tracker(bus, Decimal("5000"), "momentum")
    tracker.restore_snapshot(momentum_snap)
    assert tracker.get_cash_balance() == Decimal("4000")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_restore_with_new_strategy(bus, temp_state_file):
    """
    A new strategy added between restarts (no snapshot in file) gets None from
    get_strategy_snapshot and starts fresh with its configured initial_balance.
    """
    v2_state = {
        "version": 2,
        "balances": {"USD": "9000"},
        "positions": {},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "momentum": {
                "cash_balance": "4500",
                "initial_balance": "5000",
                "peak_equity": "4600",
                "total_realized_pnl": "30",
                "positions": {},
            },
        },
    }
    temp_state_file.write_text(json.dumps(v2_state))

    adapter = _make_adapter(bus, temp_state_file)
    await adapter.connect()

    # New strategy has no snapshot
    new_snap = adapter.get_strategy_snapshot("new_strategy")
    assert new_snap is None

    # Tracker starts fresh at initial balance (no restore called)
    new_tracker = _make_tracker(bus, Decimal("3000"), "new_strategy")
    assert new_tracker.get_cash_balance() == Decimal("3000")

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_paper_state_v2_round_trip_persistence(bus, temp_state_file):
    """
    Full round-trip: adapter saves v2 state, new adapter loads it and strategy
    snapshot values are accessible.
    """
    adapter1 = _make_adapter(bus, temp_state_file)
    await adapter1.connect()

    t = _make_tracker(bus, Decimal("5000"), "range_trading")
    t._cash_balance = Decimal("4750")
    t._peak_equity = Decimal("5100")
    t._total_realized_pnl = Decimal("75")

    adapter1.set_strategy_portfolios({"range_trading": t})
    adapter1._save_state()
    await adapter1.disconnect()

    # Second session: load state
    adapter2 = _make_adapter(bus, temp_state_file, balance=Decimal("5000"))
    await adapter2.connect()

    # v2 state is migrated v2→v3→v4 on connect() — snapshot data preserved verbatim
    assert adapter2._state_version == 4
    snap = adapter2.get_strategy_snapshot("range_trading")
    assert snap is not None
    assert snap["cash_balance"] == "4750"
    assert snap["peak_equity"] == "5100"
    assert snap["total_realized_pnl"] == "75"

    await adapter2.disconnect()
