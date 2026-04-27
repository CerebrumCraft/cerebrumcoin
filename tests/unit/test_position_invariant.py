"""
Tests for post-fill position invariant check in PaperTradingAdapter.

Issue #32: ~2,000 insufficient_position warnings/session because per-strategy
PortfolioTracker positions drift from the PaperTradingAdapter global position
ledger during the session. DEC-RECONCILE-001 fixes this at startup but the
drift re-accumulates on every fill where two strategies compete for the same
symbol or where exit monitors fire independently of the owning strategy.

Fix validated here: after every fill in execute_order, the adapter checks that:
  sum(tracker._positions[symbol].amount for all trackers) == adapter._positions[symbol]
On violation: logs `position_invariant_violated` and runs in-place reconciliation
(same logic as the startup DEC-RECONCILE-001 pass).

@decision DEC-RECONCILE-002
@title In-session post-fill position invariant + fix-up mirrors startup reconciliation
@status accepted
@rationale The startup reconciliation (DEC-RECONCILE-001) only runs at connect()
time. Without a per-fill check, the drift re-accumulates during the session
(~2,000 insufficient_position warnings per overnight run). Mirroring the same
scale-down/zero-out logic at fill time catches and corrects drift the moment it
occurs. The invariant check is lightweight: O(S) where S = number of strategies
holding the filled symbol (typically 1–3). The structured log event
`position_invariant_violated` provides an audit trail for diagnosis.
"""

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest
import structlog
import structlog.testing

from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import OrderEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side
from cerebrum.risk.portfolio import Position, PortfolioTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def paper(bus, tmp_path):
    """PaperTradingAdapter with $10k initial balance."""
    sf = tmp_path / "state.json"
    adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=Decimal("10000"),
        commission_percent=Decimal("0.16"),
        slippage_percent=Decimal("0"),
        state_file=sf,
    )
    return adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(symbol: str, side: Side, amount: Decimal, strategy_id: str) -> OrderEvent:
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=f"ord-{strategy_id}-{side.value}",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        amount=amount,
        status=OrderStatus.PENDING,
        strategy_id=strategy_id,
    )


def _inject_position(tracker: PortfolioTracker, symbol: str, amount: Decimal, price: Decimal) -> None:
    """Directly inject a position into a tracker to simulate pre-existing drift."""
    tracker._positions[symbol] = Position(
        symbol=symbol,
        amount=amount,
        average_entry_price=price,
        current_price=price,
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )


# ---------------------------------------------------------------------------
# Test: invariant fires and reconciles when drift is injected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_invariant_detects_drift_and_reconciles(bus, paper):
    """
    Simulate drift: two PortfolioTrackers together hold MORE SOL/USD than the
    paper adapter's actual position. Execute a sell. After the fill, the
    post-fill invariant check must detect the violation, log
    `position_invariant_violated`, and reconcile tracker positions so their
    sum does not exceed the paper adapter amount.

    This is the exact Session 26 pattern: SOL/USD tracker total was 2.155
    but paper adapter held 0, causing infinite insufficient_position warnings.
    """
    await paper.connect()

    pt_strat_a = PortfolioTracker(bus, Decimal("5000"), strategy_id="strat_a")
    pt_strat_b = PortfolioTracker(bus, Decimal("5000"), strategy_id="strat_b")
    paper.set_strategy_portfolios({"strat_a": pt_strat_a, "strat_b": pt_strat_b})

    paper._current_prices["SOL/USD"] = Decimal("150")

    # Paper adapter: 1.0 SOL
    paper._positions["SOL/USD"] = Decimal("1.0")
    paper._balances["USD"] = Decimal("8500")

    # Inject drift: trackers together hold 1.4 SOL (0.8 + 0.6) > 1.0 in paper
    _inject_position(pt_strat_a, "SOL/USD", Decimal("0.8"), Decimal("150"))
    _inject_position(pt_strat_b, "SOL/USD", Decimal("0.6"), Decimal("150"))

    # Capture log output via structlog.testing
    with structlog.testing.capture_logs() as cap_logs:
        sell_order = _make_order("SOL/USD", Side.SELL, Decimal("1.0"), "strat_a")
        await paper.execute_order(sell_order)
        await asyncio.sleep(0.2)  # let all subscriber tasks process

    # Paper adapter sold 1.0 SOL → 0 remaining
    remaining = paper._positions.get("SOL/USD", Decimal("0"))
    assert remaining == Decimal("0"), (
        f"After selling all 1.0 SOL, expected 0 remaining, got {remaining}"
    )

    # After sell + tracker processing + invariant reconciliation:
    # tracker total for SOL/USD should not exceed paper amount (0)
    strat_a_pos = pt_strat_a._positions.get("SOL/USD")
    strat_b_pos = pt_strat_b._positions.get("SOL/USD")
    tracker_a_amt = strat_a_pos.amount if strat_a_pos else Decimal("0")
    tracker_b_amt = strat_b_pos.amount if strat_b_pos else Decimal("0")
    tracker_total = tracker_a_amt + tracker_b_amt
    paper_amount = paper._positions.get("SOL/USD", Decimal("0"))

    assert tracker_total <= paper_amount + Decimal("0.0001"), (
        f"After invariant reconciliation: tracker total={tracker_total} must not "
        f"exceed paper amount={paper_amount}. "
        f"Invariant check is not implemented or not reconciling."
    )

    # The invariant violation must have been logged
    log_events = [e["event"] for e in cap_logs]
    assert "position_invariant_violated" in log_events, (
        f"Expected 'position_invariant_violated' in log events. Got: {log_events}. "
        f"The in-session invariant check is not implemented or not logging violations."
    )


@pytest.mark.asyncio
async def test_position_invariant_violation_log_includes_symbol(bus, paper):
    """
    The `position_invariant_violated` log event must include the symbol,
    tracker_total, and paper_amount fields for actionable diagnostics.
    """
    await paper.connect()

    pt = PortfolioTracker(bus, Decimal("5000"), strategy_id="solo")
    paper.set_strategy_portfolios({"solo": pt})

    paper._current_prices["BTC/USD"] = Decimal("50000")
    # Paper: 0.01 BTC; tracker: 0.02 BTC (2× — drift injected)
    paper._positions["BTC/USD"] = Decimal("0.01")
    paper._balances["USD"] = Decimal("4500")
    _inject_position(pt, "BTC/USD", Decimal("0.02"), Decimal("50000"))

    with structlog.testing.capture_logs() as cap_logs:
        sell_order = _make_order("BTC/USD", Side.SELL, Decimal("0.01"), "solo")
        await paper.execute_order(sell_order)
        await asyncio.sleep(0.15)

    # Find the violation log entry
    violation_entries = [
        e for e in cap_logs if e.get("event") == "position_invariant_violated"
    ]
    assert violation_entries, (
        f"Expected 'position_invariant_violated' log. Got events: "
        f"{[e['event'] for e in cap_logs]}"
    )
    entry = violation_entries[0]
    assert "symbol" in entry, f"Violation log must include 'symbol'. Got: {entry}"
    assert entry["symbol"] == "BTC/USD", (
        f"Violation log symbol must be 'BTC/USD'. Got: {entry['symbol']}"
    )
    assert "tracker_total" in entry, f"Violation log must include 'tracker_total'. Got: {entry}"
    assert "paper_amount" in entry, f"Violation log must include 'paper_amount'. Got: {entry}"


@pytest.mark.asyncio
async def test_no_invariant_violation_when_positions_are_consistent(bus, paper):
    """
    When per-strategy tracker positions are consistent with the paper adapter,
    no invariant violation should be logged.
    """
    await paper.connect()

    pt = PortfolioTracker(bus, Decimal("5000"), strategy_id="clean")
    paper.set_strategy_portfolios({"clean": pt})

    paper._current_prices["ETH/USD"] = Decimal("3000")

    with structlog.testing.capture_logs() as cap_logs:
        # Clean BUY
        buy_order = _make_order("ETH/USD", Side.BUY, Decimal("0.1"), "clean")
        await paper.execute_order(buy_order)
        await asyncio.sleep(0.15)

        # Clean SELL
        sell_order = _make_order("ETH/USD", Side.SELL, Decimal("0.1"), "clean")
        await paper.execute_order(sell_order)
        await asyncio.sleep(0.15)

    # No violation should have been logged during a clean round-trip
    violation_events = [
        e for e in cap_logs if e.get("event") == "position_invariant_violated"
    ]
    assert not violation_events, (
        f"No invariant violation expected for clean round-trip, "
        f"but got: {violation_events}"
    )

    # Positions should be in sync
    paper_eth = paper._positions.get("ETH/USD", Decimal("0"))
    tracker_pos = pt._positions.get("ETH/USD")
    tracker_eth = tracker_pos.amount if tracker_pos else Decimal("0")

    assert abs(paper_eth - tracker_eth) < Decimal("0.0001"), (
        f"After clean round-trip: paper={paper_eth}, tracker={tracker_eth} should match"
    )


@pytest.mark.asyncio
async def test_invariant_reconcile_zero_out_when_paper_has_none(bus, paper):
    """
    When paper adapter has 0 of a symbol but trackers still hold some
    (the Session 26 SOL/USD phantom pattern), the reconciliation must
    zero out all tracker positions for that symbol.
    """
    await paper.connect()

    pt_a = PortfolioTracker(bus, Decimal("5000"), strategy_id="a")
    pt_b = PortfolioTracker(bus, Decimal("5000"), strategy_id="b")
    paper.set_strategy_portfolios({"a": pt_a, "b": pt_b})

    paper._current_prices["SOL/USD"] = Decimal("100")
    # Paper has 1.0 SOL; we'll sell it all
    paper._positions["SOL/USD"] = Decimal("1.0")
    paper._balances["USD"] = Decimal("8000")

    # Trackers both hold SOL — a:0.7, b:0.5 = 1.2 total (drifted)
    _inject_position(pt_a, "SOL/USD", Decimal("0.7"), Decimal("100"))
    _inject_position(pt_b, "SOL/USD", Decimal("0.5"), Decimal("100"))

    # Sell 1.0 SOL — paper goes to 0, but trackers still hold 0.2 total after
    # strat_a's fill is processed (0.7 - 1.0 = position closed + potentially negative)
    with structlog.testing.capture_logs() as cap_logs:
        sell_order = _make_order("SOL/USD", Side.SELL, Decimal("1.0"), "a")
        await paper.execute_order(sell_order)
        await asyncio.sleep(0.2)

    # After reconciliation: neither tracker should hold SOL/USD exceeding paper=0
    pos_a = pt_a._positions.get("SOL/USD")
    pos_b = pt_b._positions.get("SOL/USD")
    amt_a = pos_a.amount if pos_a else Decimal("0")
    amt_b = pos_b.amount if pos_b else Decimal("0")

    paper_sol = paper._positions.get("SOL/USD", Decimal("0"))
    total_tracker = amt_a + amt_b

    assert total_tracker <= paper_sol + Decimal("0.0001"), (
        f"After paper sells all SOL, tracker total ({total_tracker}) must not "
        f"exceed paper amount ({paper_sol}). Reconciliation failed."
    )
