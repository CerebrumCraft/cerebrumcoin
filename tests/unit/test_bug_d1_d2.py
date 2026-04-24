"""
Unit tests for Bug D1 and Bug D2 fixes.

Bug D1: strategy_id threading through execution path — PortfolioTracker._on_fill
        gates correctly on matching/mismatching/missing strategy_id.
Bug D2: per-strategy initial_balance dynamic = pool / N — sum of all tracker
        balances equals pool_usd regardless of N strategies active.

@decision DEC-FILL-STRATEGY-ID-001
@title Tests guard the strategy_id gate in PortfolioTracker._on_fill
@status accepted
@rationale _on_fill must accept fills tagged with its own strategy_id and
silently reject fills tagged with a different strategy_id or with no strategy_id.
Without the strict gate, one strategy's fills would be double-counted by all
other PortfolioTrackers sharing the same event bus — breaking P&L attribution
and the DarwinianAllocator Sharpe feed. The "missing strategy_id" case is a
regression guard against Bug D1 re-introduction: if the execution path loses
the tag, every strategy's tracker silently skips the fill rather than
attributing it to all strategies (which was the Bug D1 symptom observed in
Session 40 — closed_trades stayed empty despite a confirmed round-trip).

@decision DEC-ALLOC-INITIAL-001
@title Tests verify pool / N dynamic balance allocation in StrategyRegistry
@status accepted
@rationale Tests here confirm: (a) with N strategies and pool_usd set, each
tracker starts with pool/N; (b) the sum of all balances equals pool_usd within
rounding; (c) an explicit cfg.initial_balance override is honoured when pool_usd
is None (legacy/backtest path). These tests are the acceptance criteria for the
Bug D2 fix which eliminates the $15k phantom capital that was inflating position
sizes ~50% above the $10k pool.
"""

import asyncio
from decimal import Decimal
from pathlib import Path
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side, SignalType
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config/paper.toml")


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def config():
    cfg, _raw = Config.from_toml(CONFIG_PATH)
    return cfg


def _make_fill(
    symbol: str = "ETH/USD",
    side: Side = Side.BUY,
    amount: Decimal = Decimal("0.05"),
    price: Decimal = Decimal("2300.00"),
    strategy_id: str | None = None,
) -> FillEvent:
    """Helper: build a minimal FillEvent with optional strategy_id."""
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id="test-order-1",
        symbol=symbol,
        side=side,
        filled_amount=amount,
        fill_price=price,
        commission=Decimal("0.01"),
        commission_asset="USD",
        exchange_order_id="paper_test",
        strategy_id=strategy_id,
    )


def _make_strategy(name: str, balance: Decimal = Decimal("3333.33")) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        aggregator_weights={
            SignalType.TECHNICAL: Decimal("1.0"),
            SignalType.SENTIMENT: Decimal("0.5"),
        },
        aggregator_threshold=Decimal("0.4"),
        initial_balance=balance,
    )


# ---------------------------------------------------------------------------
# Bug D1 — PortfolioTracker._on_fill strategy_id gate
# ---------------------------------------------------------------------------


class TestPortfolioTrackerFillGate:
    """
    Guard suite for the strategy_id filter in PortfolioTracker._on_fill.

    Regression coverage:
    - Matching strategy_id: fill accepted, position opened, closed_trades updated
    - Mismatching strategy_id: fill silently rejected (no position change)
    - Missing strategy_id (None): fill silently rejected (Bug D1 regression guard)
    """

    @pytest.mark.asyncio
    async def test_matching_strategy_id_opens_position(self, bus):
        """Fill tagged with matching strategy_id opens a position in the tracker."""
        tracker = PortfolioTracker(
            bus=bus,
            initial_balance=Decimal("5000.00"),
            strategy_id="mean_reversion",
        )

        fill = _make_fill(
            side=Side.BUY,
            amount=Decimal("0.05"),
            price=Decimal("2000.00"),
            strategy_id="mean_reversion",
        )
        await bus.publish(fill)
        # Yield so the subscriber task processes the queued event
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        pos = tracker.get_position("ETH/USD")
        assert pos is not None, "Position should be opened for matching strategy_id"
        assert pos.amount == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_matching_strategy_id_appends_closed_trade(self, bus):
        """
        A full round-trip (BUY then SELL) with matching strategy_id appends to
        _closed_trades. This is the primary regression guard for Bug D1:
        if strategy_id is dropped in the execution path, _closed_trades stays empty
        and DarwinianAllocator starves for Sharpe inputs.
        """
        tracker = PortfolioTracker(
            bus=bus,
            initial_balance=Decimal("5000.00"),
            strategy_id="mean_reversion",
        )

        buy_fill = _make_fill(
            side=Side.BUY,
            amount=Decimal("0.05"),
            price=Decimal("2000.00"),
            strategy_id="mean_reversion",
        )
        await bus.publish(buy_fill)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Sell to close the position
        sell_fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=time() + 60,
            order_id="test-order-2",
            symbol="ETH/USD",
            side=Side.SELL,
            filled_amount=Decimal("0.05"),
            fill_price=Decimal("2020.00"),  # small profit
            commission=Decimal("0.01"),
            commission_asset="USD",
            exchange_order_id="paper_test_2",
            strategy_id="mean_reversion",
        )
        await bus.publish(sell_fill)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        closed = tracker.get_closed_trades()
        assert len(closed) == 1, (
            f"Expected 1 closed trade, got {len(closed)}. "
            "Bug D1 regression: strategy_id may be dropped in the execution path."
        )
        assert closed[0].strategy_id == "mean_reversion"

    @pytest.mark.asyncio
    async def test_mismatching_strategy_id_is_silently_rejected(self, bus):
        """
        Fill tagged with a different strategy_id must not affect this tracker.
        Prevents cross-strategy attribution bugs when strategies share a symbol
        (e.g. mean_reversion and range_trading both trading ETH/USD).
        """
        tracker = PortfolioTracker(
            bus=bus,
            initial_balance=Decimal("5000.00"),
            strategy_id="mean_reversion",
        )

        fill = _make_fill(
            side=Side.BUY,
            amount=Decimal("0.05"),
            price=Decimal("2000.00"),
            strategy_id="range_trading",  # wrong strategy
        )
        await bus.publish(fill)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        pos = tracker.get_position("ETH/USD")
        assert pos is None, "Tracker must not accept fills from a different strategy_id"
        assert tracker.get_cash_balance() == Decimal("5000.00"), (
            "Cash balance must not change when fill belongs to another strategy"
        )

    @pytest.mark.asyncio
    async def test_missing_strategy_id_is_silently_rejected(self, bus):
        """
        Fill with strategy_id=None must be rejected by a strategy-scoped tracker.

        This is the Bug D1 regression guard: the paper adapter previously emitted
        FillEvents without strategy_id (None). If the strict gate is relaxed to
        accept None, every strategy's tracker would process every fill — causing
        triple-counting for 3-strategy setups. The gate MUST stay strict.
        """
        tracker = PortfolioTracker(
            bus=bus,
            initial_balance=Decimal("5000.00"),
            strategy_id="mean_reversion",
        )

        fill = _make_fill(
            side=Side.BUY,
            amount=Decimal("0.05"),
            price=Decimal("2000.00"),
            strategy_id=None,  # no tag — simulates the Bug D1 pre-fix state
        )
        await bus.publish(fill)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        pos = tracker.get_position("ETH/USD")
        assert pos is None, (
            "Tracker must reject fills with strategy_id=None when it has a strategy_id. "
            "Accepting None would re-introduce Bug D1 and break Darwinian Sharpe feed."
        )

    @pytest.mark.asyncio
    async def test_global_tracker_accepts_any_fill(self, bus):
        """
        A tracker constructed without strategy_id (global/legacy mode) accepts
        all fills regardless of their strategy_id tag. Backward compatibility.
        """
        tracker = PortfolioTracker(
            bus=bus,
            initial_balance=Decimal("10000.00"),
            strategy_id=None,  # global tracker — accept all
        )

        fill = _make_fill(
            side=Side.BUY,
            strategy_id="mean_reversion",
        )
        await bus.publish(fill)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        pos = tracker.get_position("ETH/USD")
        assert pos is not None, "Global tracker (strategy_id=None) must accept tagged fills"


# ---------------------------------------------------------------------------
# Bug D2 — StrategyRegistry pool / N dynamic allocation
# ---------------------------------------------------------------------------


class TestRegistryDynamicInitialBalance:
    """
    Guard suite for the pool / N dynamic balance allocation (DEC-ALLOC-INITIAL-001).

    Confirms:
    - With N=2 strategies: each tracker starts at pool/2, sum = pool
    - With N=3 strategies: each tracker starts at pool/3, sum = pool (within rounding)
    - With N=4 strategies: each tracker starts at pool/4, sum = pool (within rounding)
    - Legacy path (pool_usd=None): cfg.initial_balance is used unchanged
    """

    @pytest.mark.asyncio
    async def test_n2_balances_sum_to_pool(self, bus, config):
        """With 2 strategies and pool_usd=10000, each tracker gets $5000 and sum=$10000."""
        pool = Decimal("10000.00")
        registry = StrategyRegistry(bus, config, pool_usd=pool)
        registry.register(_make_strategy("alpha"))
        registry.register(_make_strategy("beta"))
        await registry.start_all()

        port_a = registry.get_portfolio("alpha")
        port_b = registry.get_portfolio("beta")
        assert port_a is not None
        assert port_b is not None

        balance_a = port_a.get_cash_balance()
        balance_b = port_b.get_cash_balance()

        # Each should be pool / 2 = $5000
        expected = pool / Decimal("2")
        assert balance_a == expected, f"alpha balance: expected {expected}, got {balance_a}"
        assert balance_b == expected, f"beta balance: expected {expected}, got {balance_b}"

        total = balance_a + balance_b
        assert abs(total - pool) < Decimal("0.01"), (
            f"Sum of balances {total} should equal pool {pool}"
        )

    @pytest.mark.asyncio
    async def test_n3_balances_sum_to_pool(self, bus, config):
        """With 3 strategies and pool_usd=10000, sum of balances = $10000 within rounding."""
        pool = Decimal("10000.00")
        registry = StrategyRegistry(bus, config, pool_usd=pool)
        registry.register(_make_strategy("alpha"))
        registry.register(_make_strategy("beta"))
        registry.register(_make_strategy("gamma"))
        await registry.start_all()

        balances = [
            registry.get_portfolio(name).get_cash_balance()
            for name in ("alpha", "beta", "gamma")
        ]
        total = sum(balances)

        # Each should be pool / 3 ≈ $3333.33
        expected_each = pool / Decimal("3")
        for name, bal in zip(("alpha", "beta", "gamma"), balances):
            assert abs(bal - expected_each) < Decimal("0.01"), (
                f"{name} balance {bal} deviates from expected {expected_each}"
            )

        assert abs(total - pool) < Decimal("0.10"), (
            f"Sum of balances {total} must equal pool {pool} within $0.10 rounding tolerance"
        )

    @pytest.mark.asyncio
    async def test_n4_balances_sum_to_pool(self, bus, config):
        """With 4 strategies and pool_usd=10000, sum of balances = $10000 within rounding."""
        pool = Decimal("10000.00")
        registry = StrategyRegistry(bus, config, pool_usd=pool)
        for name in ("alpha", "beta", "gamma", "delta"):
            registry.register(_make_strategy(name))
        await registry.start_all()

        balances = [
            registry.get_portfolio(name).get_cash_balance()
            for name in ("alpha", "beta", "gamma", "delta")
        ]
        total = sum(balances)

        expected_each = pool / Decimal("4")
        for name, bal in zip(("alpha", "beta", "gamma", "delta"), balances):
            assert abs(bal - expected_each) < Decimal("0.01"), (
                f"{name} balance {bal} should be {expected_each} ({pool}/4), not 5000.00"
            )

        assert abs(total - pool) < Decimal("0.10"), (
            f"Sum {total} must equal pool {pool}"
        )

    @pytest.mark.asyncio
    async def test_legacy_path_uses_cfg_initial_balance(self, bus, config):
        """
        When pool_usd is NOT set on the registry, cfg.initial_balance is used
        unchanged. This preserves the backtest and single-strategy test paths.
        """
        registry = StrategyRegistry(bus, config, pool_usd=None)
        registry.register(_make_strategy("alpha", balance=Decimal("7777.00")))
        await registry.start_all()

        port = registry.get_portfolio("alpha")
        assert port is not None
        assert port.get_cash_balance() == Decimal("7777.00"), (
            "Without pool_usd, cfg.initial_balance must be used directly"
        )

    @pytest.mark.asyncio
    async def test_pool_usd_overrides_cfg_initial_balance(self, bus, config):
        """
        When pool_usd IS set, cfg.initial_balance (e.g. the legacy 5000.0) is
        ignored in favour of pool / N. This is the core Bug D2 fix validation.
        """
        pool = Decimal("9000.00")
        # Strategy config has 5000.0 — would triple-count in the old code
        registry = StrategyRegistry(bus, config, pool_usd=pool)
        registry.register(_make_strategy("alpha", balance=Decimal("5000.00")))
        registry.register(_make_strategy("beta", balance=Decimal("5000.00")))
        registry.register(_make_strategy("gamma", balance=Decimal("5000.00")))
        await registry.start_all()

        balances = [
            registry.get_portfolio(name).get_cash_balance()
            for name in ("alpha", "beta", "gamma")
        ]
        total = sum(balances)

        # Must be pool ($9000), NOT 3 × 5000 = $15000
        assert abs(total - pool) < Decimal("0.10"), (
            f"Total {total} must equal pool {pool}, not {Decimal('15000.00')} "
            "(Bug D2: cfg.initial_balance=5000 must be overridden by pool/N)"
        )
        for name, bal in zip(("alpha", "beta", "gamma"), balances):
            assert abs(bal - Decimal("3000.00")) < Decimal("0.01"), (
                f"{name} balance {bal} should be 3000.00 (9000/3), not 5000.00"
            )
