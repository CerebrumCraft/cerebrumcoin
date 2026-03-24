"""
Tests for GlobalPortfolio — cross-strategy equity aggregation.

@decision DEC-STRAT-004
@title GlobalPortfolio as read-only aggregation view
@status accepted
@rationale Each strategy has an isolated PortfolioTracker. GlobalPortfolio
provides a read-only aggregate view without event subscriptions or mutation.
Tests use real EventBus and real PortfolioTrackers — no mocks.
"""

from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.global_portfolio import GlobalPortfolio


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


class TestGlobalPortfolioEquity:
    """Total equity aggregation across strategies."""

    def test_empty_portfolio(self):
        """GlobalPortfolio with no strategies returns zero equity."""
        gp = GlobalPortfolio({})
        assert gp.get_total_equity() == Decimal("0.0")

    async def test_single_strategy_equity(self, bus):
        """Single strategy equity equals its PortfolioTracker equity."""
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("5000.0"))
        gp = GlobalPortfolio({"alpha": portfolio})
        assert gp.get_total_equity() == Decimal("5000.0")

    async def test_two_strategies_sum_correctly(self, bus):
        """Two strategies with different balances sum to total equity."""
        # Use separate buses so fills don't cross-contaminate
        bus2 = EventBus()
        await bus2.start()
        try:
            p1 = PortfolioTracker(bus, initial_balance=Decimal("3000.0"))
            p2 = PortfolioTracker(bus2, initial_balance=Decimal("7000.0"))
            gp = GlobalPortfolio({"alpha": p1, "beta": p2})
            assert gp.get_total_equity() == Decimal("10000.0")
        finally:
            await bus2.stop()

    async def test_get_strategy_equity_delegates_correctly(self, bus):
        """get_strategy_equity returns the specific strategy's equity."""
        bus2 = EventBus()
        await bus2.start()
        try:
            p1 = PortfolioTracker(bus, initial_balance=Decimal("1234.56"))
            p2 = PortfolioTracker(bus2, initial_balance=Decimal("9876.54"))
            gp = GlobalPortfolio({"alpha": p1, "beta": p2})
            assert gp.get_strategy_equity("alpha") == Decimal("1234.56")
            assert gp.get_strategy_equity("beta") == Decimal("9876.54")
        finally:
            await bus2.stop()

    async def test_get_strategy_equity_missing_returns_zero(self, bus):
        """get_strategy_equity returns Decimal('0.0') for unknown strategy."""
        p1 = PortfolioTracker(bus, initial_balance=Decimal("5000.0"))
        gp = GlobalPortfolio({"alpha": p1})
        assert gp.get_strategy_equity("nonexistent") == Decimal("0.0")


class TestGlobalPortfolioDrawdown:
    """Global drawdown computed from combined equity peak."""

    async def test_no_drawdown_at_start(self, bus):
        """Drawdown is zero with no positions and no peak yet."""
        p = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
        gp = GlobalPortfolio({"alpha": p})
        # First call sets peak == current equity → 0% drawdown
        assert gp.get_total_drawdown() == Decimal("0.0")

    async def test_drawdown_after_simulated_loss(self, bus):
        """Drawdown is > 0 after equity falls below recorded peak."""
        p = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
        gp = GlobalPortfolio({"alpha": p})

        # Record the peak at $10k
        assert gp.get_total_equity() == Decimal("10000.0")
        assert gp.get_total_drawdown() == Decimal("0.0")

        # Simulate a loss by charging a large commission via a fill
        # BUY 0.1 BTC at $50k = $5000 cost + $500 commission = $5500 cash out
        # Cash: $10000 - $5500 = $4500; position value: 0.1 * $50000 = $5000
        # Equity: $4500 + $5000 = $9500 < $10000 peak → drawdown > 0
        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=1000.0,
            order_id="o1",
            symbol="BTC/USD",
            side=Side.BUY,
            filled_amount=Decimal("0.1"),
            fill_price=Decimal("50000.0"),
            commission=Decimal("500.0"),
            commission_asset="USD",
        )
        await bus.publish(fill)

        # Give fill handler time to process
        import asyncio
        await asyncio.sleep(0.05)

        drawdown = gp.get_total_drawdown()
        assert drawdown > Decimal("0.0")

    def test_empty_portfolio_drawdown_is_zero(self):
        """Empty portfolio has zero drawdown (no division by zero)."""
        gp = GlobalPortfolio({})
        assert gp.get_total_drawdown() == Decimal("0.0")


class TestGlobalPortfolioPositions:
    """Merged position view across all strategies."""

    async def test_get_all_positions_empty(self, bus):
        """No positions across any strategy → empty dict."""
        bus2 = EventBus()
        await bus2.start()
        try:
            p1 = PortfolioTracker(bus, initial_balance=Decimal("5000.0"))
            p2 = PortfolioTracker(bus2, initial_balance=Decimal("5000.0"))
            gp = GlobalPortfolio({"alpha": p1, "beta": p2})
            assert gp.get_all_positions() == {}
        finally:
            await bus2.stop()

    async def test_get_all_positions_keyed_by_strategy_symbol(self, bus):
        """Position keys are 'strategy:symbol' to avoid collisions."""
        # Use separate buses so only alpha's portfolio sees the fill
        bus_alpha = EventBus()
        await bus_alpha.start()
        try:
            p_alpha = PortfolioTracker(bus_alpha, initial_balance=Decimal("5000.0"))
            p_beta = PortfolioTracker(bus, initial_balance=Decimal("5000.0"))
            gp = GlobalPortfolio({"alpha": p_alpha, "beta": p_beta})

            # Publish fill only to alpha's bus → only alpha gets a position
            fill = FillEvent(
                event_type=EventType.FILL,
                timestamp=1001.0,
                order_id="o1",
                symbol="BTC/USD",
                side=Side.BUY,
                filled_amount=Decimal("0.1"),
                fill_price=Decimal("50000.0"),
                commission=Decimal("8.0"),
                commission_asset="USD",
            )
            await bus_alpha.publish(fill)

            import asyncio
            await asyncio.sleep(0.05)

            positions = gp.get_all_positions()
            assert "alpha:BTC/USD" in positions
            assert "beta:BTC/USD" not in positions
        finally:
            await bus_alpha.stop()

    async def test_get_strategy_names(self, bus):
        """get_strategy_names returns all registered strategy names."""
        bus2 = EventBus()
        await bus2.start()
        try:
            p1 = PortfolioTracker(bus, initial_balance=Decimal("5000.0"))
            p2 = PortfolioTracker(bus2, initial_balance=Decimal("5000.0"))
            gp = GlobalPortfolio({"alpha": p1, "beta": p2})
            names = gp.get_strategy_names()
            assert set(names) == {"alpha", "beta"}
        finally:
            await bus2.stop()
