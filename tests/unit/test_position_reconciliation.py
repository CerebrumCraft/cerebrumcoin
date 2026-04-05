"""
Unit tests for startup position reconciliation.

Verifies that when PortfolioTracker positions (per-strategy) exceed the
paper adapter's global position ledger, the strategy positions are scaled
down proportionally. This prevents exit orders from being rejected with
`insufficient_position` after a session restart where the state has drifted.

@decision DEC-RECONCILE-001
@title Startup position reconciliation between portfolio trackers and paper adapter
@status accepted
@rationale After restart, per-strategy PortfolioTracker positions restored from
snapshots may exceed the global PaperTradingAdapter._positions ledger (which
tracks actual simulated fills). This drift causes exit monitor sell orders to be
rejected with `insufficient_position`, leaving zombie OPEN trades that can never
close. Reconciliation at startup scales down portfolio positions proportionally
so their sum never exceeds the paper adapter amount.
"""

from decimal import Decimal

import pytest

from cerebrum.risk.portfolio import PortfolioTracker, Position
from cerebrum.core.bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def bus():
    """Create and start an event bus for tests."""
    b = EventBus(queue_size=50)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Helpers — build minimal PortfolioTracker with injected positions
# ---------------------------------------------------------------------------

def _make_portfolio(bus: EventBus, positions: dict[str, Decimal]) -> PortfolioTracker:
    """Create a PortfolioTracker with pre-set positions (bypasses fill events)."""
    pt = PortfolioTracker(bus=bus, initial_balance=Decimal("5000"), strategy_id="test")
    for symbol, amount in positions.items():
        pt._positions[symbol] = Position(
            symbol=symbol,
            amount=amount,
            average_entry_price=Decimal("100"),
            current_price=Decimal("100"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
    return pt


def _reconcile(portfolios: dict, paper_positions: dict[str, Decimal]) -> None:
    """
    Inline reconciliation logic — mirrors what main.py does.

    Extracted here so we can test the logic independently of CerebrumBot startup.
    """
    # Collect all symbols referenced by any portfolio
    all_symbols: set[str] = set()
    for pt in portfolios.values():
        all_symbols.update(pt._positions.keys())

    for symbol in all_symbols:
        paper_amount = paper_positions.get(symbol, Decimal("0"))

        # Sum total held across all strategy portfolios for this symbol
        portfolio_total = sum(
            pt._positions[symbol].amount
            for pt in portfolios.values()
            if symbol in pt._positions
        )

        if portfolio_total == Decimal("0"):
            continue  # Nothing to reconcile

        if paper_amount == Decimal("0"):
            # Paper adapter has none — zero out all portfolio positions
            for pt in portfolios.values():
                if symbol in pt._positions:
                    pt._positions[symbol].amount = Decimal("0")
        elif portfolio_total > paper_amount:
            # Scale each strategy's position proportionally
            scale = paper_amount / portfolio_total
            for pt in portfolios.values():
                if symbol in pt._positions:
                    pt._positions[symbol].amount = pt._positions[symbol].amount * scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPositionReconciliation:
    """Tests for startup position reconciliation logic."""

    @pytest.mark.asyncio
    async def test_scale_down_when_portfolio_exceeds_paper(self, bus):
        """
        When total portfolio positions exceed paper adapter amount,
        each strategy's position should be scaled down proportionally.
        """
        # Two strategies each hold 1.0 BTC => portfolio total = 2.0
        # Paper adapter has 1.0 BTC => each strategy should be scaled to 0.5
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"BTC/USD": Decimal("1.0")}),
            "range_trading": _make_portfolio(bus, {"BTC/USD": Decimal("1.0")}),
        }
        paper_positions = {"BTC/USD": Decimal("1.0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["BTC/USD"].amount == Decimal("0.5")
        assert portfolios["range_trading"]._positions["BTC/USD"].amount == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_no_change_when_portfolio_equals_paper(self, bus):
        """When portfolio total equals paper adapter amount, no change should occur."""
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"ETH/USD": Decimal("2.0")}),
        }
        paper_positions = {"ETH/USD": Decimal("2.0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["ETH/USD"].amount == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_no_change_when_portfolio_less_than_paper(self, bus):
        """When portfolio total is less than paper adapter amount, no change should occur."""
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"ETH/USD": Decimal("0.5")}),
        }
        paper_positions = {"ETH/USD": Decimal("2.0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["ETH/USD"].amount == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_zero_out_when_paper_has_zero(self, bus):
        """
        When paper adapter has zero for a symbol but portfolios have positions,
        all portfolio positions for that symbol must be zeroed.
        """
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"SOL/USD": Decimal("5.0")}),
            "range_trading": _make_portfolio(bus, {"SOL/USD": Decimal("3.0")}),
        }
        paper_positions = {"SOL/USD": Decimal("0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["SOL/USD"].amount == Decimal("0")
        assert portfolios["range_trading"]._positions["SOL/USD"].amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_uneven_split_preserved_proportionally(self, bus):
        """
        When strategies hold unequal amounts, scaling preserves their ratio.
        Strategy A holds 3x, strategy B holds 1x => after scaling, still 3:1 ratio.
        """
        portfolios = {
            "A": _make_portfolio(bus, {"BTC/USD": Decimal("3.0")}),
            "B": _make_portfolio(bus, {"BTC/USD": Decimal("1.0")}),
        }
        # Total = 4.0, paper = 2.0 => scale factor 0.5
        paper_positions = {"BTC/USD": Decimal("2.0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["A"]._positions["BTC/USD"].amount == Decimal("1.5")
        assert portfolios["B"]._positions["BTC/USD"].amount == Decimal("0.5")
        # Verify total now matches paper
        total = sum(
            pt._positions["BTC/USD"].amount for pt in portfolios.values()
        )
        assert total == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_reconcile_only_affects_drifted_symbol(self, bus):
        """
        Reconciliation for one symbol must not affect other symbols.
        """
        portfolios = {
            "mean_reversion": _make_portfolio(
                bus, {"BTC/USD": Decimal("2.0"), "ETH/USD": Decimal("1.0")}
            ),
        }
        # BTC drifted (portfolio 2.0 > paper 1.0), ETH is fine (1.0 == 1.0)
        paper_positions = {"BTC/USD": Decimal("1.0"), "ETH/USD": Decimal("1.0")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["BTC/USD"].amount == Decimal("1.0")
        assert portfolios["mean_reversion"]._positions["ETH/USD"].amount == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_empty_portfolios_no_error(self, bus):
        """Reconciliation with no positions in any portfolio does not raise."""
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {}),
        }
        paper_positions = {"BTC/USD": Decimal("1.0")}

        # Should not raise
        _reconcile(portfolios, paper_positions)

    @pytest.mark.asyncio
    async def test_single_strategy_scale_down(self, bus):
        """Single-strategy case: position scaled to match paper adapter exactly."""
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"BTC/USD": Decimal("5.0")}),
        }
        paper_positions = {"BTC/USD": Decimal("2.5")}

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["BTC/USD"].amount == Decimal("2.5")

    @pytest.mark.asyncio
    async def test_symbol_missing_from_paper_treated_as_zero(self, bus):
        """
        If a symbol is in a portfolio but absent from paper_positions dict,
        it should be treated as paper_amount=0 and zeroed out.
        """
        portfolios = {
            "mean_reversion": _make_portfolio(bus, {"XRP/USD": Decimal("100.0")}),
        }
        paper_positions: dict[str, Decimal] = {}  # XRP not present at all

        _reconcile(portfolios, paper_positions)

        assert portfolios["mean_reversion"]._positions["XRP/USD"].amount == Decimal("0")
