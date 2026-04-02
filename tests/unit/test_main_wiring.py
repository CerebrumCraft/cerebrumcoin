"""
Tests for Phase 11E main.py multi-strategy wiring.

@decision DEC-TEST-MAIN-001
@title Test strategy_id filtering and config instances with real implementations
@status accepted
@rationale Sacred Practice #5: all tests use real implementations. No internal
mocks. PortfolioTracker, StrategyRegistry, and CerebrumCoin._setup_* methods
are exercised directly with in-memory EventBus instances and real Config loaded
from paper.toml. bus.subscribe() calls asyncio.create_task(), so any test that
constructs PortfolioTracker must run inside an async test (event loop present).

Covers:
1. PortfolioTracker strategy_id filtering (DEC-RISK-004) — prevents double-
   counting fills across 3 strategy portfolios on a shared bus.
2. MEAN_REVERSION_CONFIG and BREAKOUT_CONFIG are valid StrategyConfig instances
   with correct parameter values.
3. StrategyRegistry.register() / start_all() / stop_all() lifecycle with all
   three configs.
4. CerebrumCoin._setup_single_strategy() wires portfolio, risk_manager,
   exit_monitor, signal_agg and leaves multi-strategy attrs None.
5. CerebrumCoin._setup_multi_strategy() starts all three pipelines, sets
   strategy_registry, allocator, conductor.
"""

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side, SignalType
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.breakout import BREAKOUT_CONFIG
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
from cerebrum.strategies.momentum import MOMENTUM_CONFIG
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
from cerebrum.strategies.swing_trading import SWING_TRADING_CONFIG


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _load_config():
    """Load real config from paper.toml (falls back to default.toml)."""
    from cerebrum.core.config import Config
    paper = Path(__file__).parents[2] / "config" / "paper.toml"
    default = Path(__file__).parents[2] / "config" / "default.toml"
    config, _raw_toml = Config.from_toml(paper if paper.exists() else default)
    return config


def _make_fill(strategy_id: str | None, symbol: str = "BTC/USD") -> FillEvent:
    """Build a real FillEvent with the given strategy_id."""
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id=str(uuid.uuid4()),
        symbol=symbol,
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("50000"),
        commission=Decimal("5"),
        commission_asset="USD",
        strategy_id=strategy_id,
    )


# ---------------------------------------------------------------------------
# PortfolioTracker strategy_id filtering (DEC-RISK-004)
# ---------------------------------------------------------------------------

class TestPortfolioTrackerStrategyIdFilter:
    """strategy_id=None accepts all fills; strategy_id='x' accepts only x's fills."""

    @pytest.mark.asyncio
    async def test_no_strategy_id_accepts_all_fills(self):
        bus = EventBus()
        await bus.start()
        try:
            portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000"), strategy_id=None)
            await bus.publish(_make_fill("momentum"))
            await asyncio.sleep(0.05)
            pos = portfolio.get_position("BTC/USD")
            assert pos is not None, "strategy_id=None tracker must accept all fills"
            assert pos.amount == Decimal("0.1")
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_matching_strategy_id_accepts_fill(self):
        bus = EventBus()
        await bus.start()
        try:
            portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000"), strategy_id="momentum")
            await bus.publish(_make_fill("momentum"))
            await asyncio.sleep(0.05)
            pos = portfolio.get_position("BTC/USD")
            assert pos is not None
            assert pos.amount == Decimal("0.1")
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_mismatched_strategy_id_ignores_fill(self):
        bus = EventBus()
        await bus.start()
        try:
            portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000"), strategy_id="mean_reversion")
            await bus.publish(_make_fill("momentum"))  # different strategy
            await asyncio.sleep(0.05)
            pos = portfolio.get_position("BTC/USD")
            assert pos is None, "Mismatched strategy_id fill must be ignored"
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_three_portfolios_no_cross_contamination(self):
        """Each of three strategy portfolios processes only its own fills."""
        bus = EventBus()
        await bus.start()
        try:
            p_momentum = PortfolioTracker(bus, initial_balance=Decimal("3333"), strategy_id="momentum")
            p_mean_rev = PortfolioTracker(bus, initial_balance=Decimal("3333"), strategy_id="mean_reversion")
            p_breakout = PortfolioTracker(bus, initial_balance=Decimal("3333"), strategy_id="breakout")

            for name in ("momentum", "mean_reversion", "breakout"):
                await bus.publish(_make_fill(name))
            await asyncio.sleep(0.1)

            expected_cash = Decimal("3333") - Decimal("0.1") * Decimal("50000") - Decimal("5")
            for portfolio, name in (
                (p_momentum, "momentum"),
                (p_mean_rev, "mean_reversion"),
                (p_breakout, "breakout"),
            ):
                pos = portfolio.get_position("BTC/USD")
                assert pos is not None, f"{name}: position must exist"
                assert pos.amount == Decimal("0.1"), f"{name}: double-counting detected"
                assert portfolio.get_cash_balance() == expected_cash, (
                    f"{name}: cash wrong — cross-fill contamination detected"
                )
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_fill_without_strategy_id_accepted_by_unfiltered_portfolio(self):
        """Legacy fills without strategy_id are accepted by strategy_id=None portfolio."""
        bus = EventBus()
        await bus.start()
        try:
            portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000"), strategy_id=None)
            # No strategy_id keyword — defaults to None
            fill = FillEvent(
                event_type=EventType.FILL,
                timestamp=time(),
                order_id=str(uuid.uuid4()),
                symbol="ETH/USD",
                side=Side.BUY,
                filled_amount=Decimal("1.0"),
                fill_price=Decimal("3000"),
                commission=Decimal("3"),
                commission_asset="USD",
            )
            await bus.publish(fill)
            await asyncio.sleep(0.05)
            assert portfolio.get_position("ETH/USD") is not None
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_filtered_portfolio_ignores_fill_with_no_strategy_id(self):
        """A filtered portfolio ignores fills where strategy_id is absent/None."""
        bus = EventBus()
        await bus.start()
        try:
            portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000"), strategy_id="momentum")
            fill = FillEvent(
                event_type=EventType.FILL,
                timestamp=time(),
                order_id=str(uuid.uuid4()),
                symbol="ETH/USD",
                side=Side.BUY,
                filled_amount=Decimal("1.0"),
                fill_price=Decimal("3000"),
                commission=Decimal("3"),
                commission_asset="USD",
                # strategy_id defaults to None — should be ignored by "momentum" portfolio
            )
            await bus.publish(fill)
            await asyncio.sleep(0.05)
            assert portfolio.get_position("ETH/USD") is None
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# Strategy config instances
# ---------------------------------------------------------------------------

class TestStrategyConfigs:
    """MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG correctness."""

    def test_all_are_strategy_config_instances(self):
        for cfg in (MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG):
            assert isinstance(cfg, StrategyConfig), f"{cfg.name} must be a StrategyConfig"

    def test_correct_names(self):
        assert MOMENTUM_CONFIG.name == "momentum"
        assert MEAN_REVERSION_CONFIG.name == "mean_reversion"
        assert BREAKOUT_CONFIG.name == "breakout"

    def test_momentum_has_sixth_balance(self):
        """MOMENTUM_CONFIG uses 1/6 of $10k for the 6-strategy equal split."""
        assert MOMENTUM_CONFIG.initial_balance == Decimal("1666.67")

    def test_mean_reversion_has_half_balance(self):
        """DEC-TUNE-008: mean_reversion gets $5k — 2-strategy split of $10k."""
        assert MEAN_REVERSION_CONFIG.initial_balance == Decimal("5000.00")

    def test_breakout_has_sixth_balance(self):
        """Breakout is inactive (DEC-TUNE-006) but config retains its original allocation."""
        assert BREAKOUT_CONFIG.initial_balance == Decimal("1666.67")

    def test_mean_reversion_has_tighter_tp(self):
        """Mean reversion targets small range-bound moves."""
        assert MEAN_REVERSION_CONFIG.exit_config["take_profit_percent"] == "1.5"

    def test_breakout_has_wider_tp(self):
        """Breakout rides trend moves — needs wider TP."""
        assert BREAKOUT_CONFIG.exit_config["take_profit_percent"] == "4.0"

    def test_mean_reversion_threshold_lower_than_momentum(self):
        """Range-bound signals are inherently weaker; lower threshold needed."""
        assert MEAN_REVERSION_CONFIG.aggregator_threshold < MOMENTUM_CONFIG.aggregator_threshold

    def test_breakout_threshold_higher_than_momentum(self):
        """Breakout requires strong conviction to avoid false breakouts."""
        assert BREAKOUT_CONFIG.aggregator_threshold > MOMENTUM_CONFIG.aggregator_threshold

    def test_all_have_eth_symbol(self):
        """All three base configs trade ETH/USD."""
        for cfg in (MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG):
            assert "ETH/USD" in cfg.symbols

    def test_btc_only_in_mean_reversion(self):
        """BTC/USD removed from momentum (DEC-TUNE-006) and breakout (DEC-TUNE-007).
        mean_reversion retains BTC/USD as top Session 18 performer (+$877)."""
        assert "BTC/USD" not in MOMENTUM_CONFIG.symbols
        assert "BTC/USD" in MEAN_REVERSION_CONFIG.symbols
        assert "BTC/USD" not in BREAKOUT_CONFIG.symbols

    def test_all_have_technical_weight(self):
        for cfg in (MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG):
            assert SignalType.TECHNICAL in cfg.aggregator_weights

    def test_configs_are_immutable(self):
        """frozen=True prevents accidental runtime mutation via normal attribute assignment."""
        # Python's frozen dataclass raises FrozenInstanceError (subclass of AttributeError)
        # on normal attribute assignment. We test that — not object.__setattr__ which bypasses.
        with pytest.raises(AttributeError):
            MEAN_REVERSION_CONFIG.aggregator_threshold = Decimal("0.99")  # type: ignore[misc]

    def test_breakout_config_independent_of_mean_reversion(self):
        """Configs are separate objects — mutating one does not affect the other."""
        assert BREAKOUT_CONFIG.name != MEAN_REVERSION_CONFIG.name
        assert BREAKOUT_CONFIG.aggregator_threshold != MEAN_REVERSION_CONFIG.aggregator_threshold

    def test_range_trading_is_strategy_config_instance(self):
        assert isinstance(RANGE_TRADING_CONFIG, StrategyConfig)
        assert RANGE_TRADING_CONFIG.name == "range_trading"

    def test_range_trading_has_half_balance(self):
        """DEC-TUNE-008: range_trading gets $5k — 2-strategy split of $10k."""
        assert RANGE_TRADING_CONFIG.initial_balance == Decimal("5000.00")

    def test_range_trading_filters_to_support_resistance(self):
        """Only S/R signals feed range trading — other sources are excluded."""
        assert RANGE_TRADING_CONFIG.signal_source_filter == "SupportResistance"

    def test_range_trading_has_exit_monitor_factory(self):
        """Range trading uses structural exits — factory must be callable."""
        assert RANGE_TRADING_CONFIG.exit_monitor_factory is not None
        assert callable(RANGE_TRADING_CONFIG.exit_monitor_factory)

    def test_range_trading_suppresses_sentiment_and_news(self):
        """S/R-only strategy sets sentiment/news weights to 0."""
        assert RANGE_TRADING_CONFIG.aggregator_weights[SignalType.SENTIMENT] == Decimal("0.0")
        assert RANGE_TRADING_CONFIG.aggregator_weights[SignalType.NEWS] == Decimal("0.0")

    def test_range_trading_low_aggregator_threshold(self):
        """Lower threshold (0.2) because S/R signals are the only source."""
        assert RANGE_TRADING_CONFIG.aggregator_threshold == Decimal("0.2")

    def test_swing_trading_is_strategy_config_instance(self):
        assert isinstance(SWING_TRADING_CONFIG, StrategyConfig)
        assert SWING_TRADING_CONFIG.name == "swing_trading"

    def test_swing_trading_has_sixth_balance(self):
        """Swing trading uses 1/6 of $10k — equal split with other 5 strategies."""
        assert SWING_TRADING_CONFIG.initial_balance == Decimal("1666.67")

    def test_swing_trading_filters_1h_timeframe(self):
        """Only 1h signals feed swing trading — 1m scalp signals are excluded."""
        assert SWING_TRADING_CONFIG.signal_timeframe_filter == "1h"

    def test_swing_trading_wider_tp_and_longer_hold(self):
        """Wider TP (5%) and longer max age (480 min) match the 1h trading rhythm."""
        assert SWING_TRADING_CONFIG.exit_config["take_profit_percent"] == "5.0"
        assert SWING_TRADING_CONFIG.exit_config["max_position_age_minutes"] == 480

    def test_swing_trading_technical_weight_higher_than_sentiment(self):
        """Technical signals drive 1h swing decisions; sentiment is noise at this scale."""
        assert (
            SWING_TRADING_CONFIG.aggregator_weights[SignalType.TECHNICAL]
            > SWING_TRADING_CONFIG.aggregator_weights[SignalType.SENTIMENT]
        )


# ---------------------------------------------------------------------------
# StrategyRegistry lifecycle with all three configs
# ---------------------------------------------------------------------------

class TestStrategyRegistryAllConfigs:
    """Registry starts, accesses, and stops all three strategy pipelines."""

    def _three_equal_configs(self) -> list[StrategyConfig]:
        """Return three StrategyConfig instances for registry lifecycle tests.

        Uses a $3333.33 momentum copy so this fixture's totals are predictable.
        mean_reversion uses its real config ($5000 after DEC-TUNE-008 2-way split).
        breakout uses its real config ($1666.67, inactive but config unchanged).
        """
        return [
            StrategyConfig(
                name="momentum",
                aggregator_weights=MOMENTUM_CONFIG.aggregator_weights,
                aggregator_threshold=MOMENTUM_CONFIG.aggregator_threshold,
                risk_overrides=MOMENTUM_CONFIG.risk_overrides,
                exit_config=MOMENTUM_CONFIG.exit_config,
                initial_balance=Decimal("3333.33"),
                symbols=MOMENTUM_CONFIG.symbols,
            ),
            MEAN_REVERSION_CONFIG,
            BREAKOUT_CONFIG,
        ]

    @pytest.mark.asyncio
    async def test_register_all_three(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        config = _load_config()
        registry = StrategyRegistry(bus=bus, config=config)
        for cfg in self._three_equal_configs():
            registry.register(cfg)
        assert set(registry.list_strategies()) == {"momentum", "mean_reversion", "breakout"}

    @pytest.mark.asyncio
    async def test_start_all_creates_pipelines(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            registry = StrategyRegistry(bus=bus, config=config)
            for cfg in self._three_equal_configs():
                registry.register(cfg)
            await registry.start_all(shared_global_rules=[])
            assert set(registry.active_strategy_names()) == {"momentum", "mean_reversion", "breakout"}
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_each_strategy_gets_own_portfolio(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            registry = StrategyRegistry(bus=bus, config=config)
            for cfg in self._three_equal_configs():
                registry.register(cfg)
            await registry.start_all()

            portfolios = [registry.get_portfolio(n) for n in ("momentum", "mean_reversion", "breakout")]
            assert all(p is not None for p in portfolios), "All portfolios must exist"
            # Each portfolio is a distinct object
            assert portfolios[0] is not portfolios[1]
            assert portfolios[1] is not portfolios[2]
            assert portfolios[0] is not portfolios[2]
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_global_portfolio_sums_balances(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            registry = StrategyRegistry(bus=bus, config=config)
            for cfg in self._three_equal_configs():
                registry.register(cfg)
            await registry.start_all()

            gp = registry.global_portfolio
            total = gp.get_total_equity()
            # momentum=$3333.33 (test fixture) + mean_reversion=$5000.00 (DEC-TUNE-008) + breakout=$1666.67
            expected = Decimal("3333.33") + Decimal("5000.00") + Decimal("1666.67")
            assert total == expected
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_stop_all_clears_pipelines(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            registry = StrategyRegistry(bus=bus, config=config)
            for cfg in self._three_equal_configs():
                registry.register(cfg)
            await registry.start_all()
            assert len(registry.active_strategy_names()) == 3
            await registry.stop_all()
            assert len(registry.active_strategy_names()) == 0
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_duplicate_registration_raises(self):
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        config = _load_config()
        registry = StrategyRegistry(bus=bus, config=config)
        registry.register(MEAN_REVERSION_CONFIG)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MEAN_REVERSION_CONFIG)

    @pytest.mark.asyncio
    async def test_per_strategy_portfolios_filtered_by_strategy_id(self):
        """After start_all, each portfolio only sees its own fills (DEC-RISK-004)."""
        from cerebrum.strategies.registry import StrategyRegistry
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            registry = StrategyRegistry(bus=bus, config=config)
            for cfg in self._three_equal_configs():
                registry.register(cfg)
            await registry.start_all()

            # Publish one fill tagged "momentum"
            await bus.publish(_make_fill("momentum"))
            await asyncio.sleep(0.1)

            # Only momentum portfolio should have a position
            assert registry.get_portfolio("momentum").get_position("BTC/USD") is not None
            assert registry.get_portfolio("mean_reversion").get_position("BTC/USD") is None
            assert registry.get_portfolio("breakout").get_position("BTC/USD") is None
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# CerebrumCoin._setup_single_strategy
# ---------------------------------------------------------------------------

class TestSetupSingleStrategy:
    """Single-strategy path wires the legacy pipeline, leaves multi-strategy attrs None."""

    @pytest.mark.asyncio
    async def test_wires_portfolio_risk_manager_exit_monitor_signal_agg(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            app._setup_single_strategy()

            assert app.portfolio is not None
            assert app.risk_manager is not None
            assert app.exit_monitor is not None
            assert app.signal_agg is not None
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_multi_strategy_attrs_remain_none(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            app._setup_single_strategy()

            assert app.strategy_registry is None
            assert app.allocator is None
            assert app.conductor is None
            assert app.web_dashboard is None
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_portfolio_accepts_all_fills_no_strategy_id_filter(self):
        """Single-strategy portfolio uses strategy_id=None — accepts every fill."""
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            app._setup_single_strategy()

            assert app.portfolio._strategy_id is None
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_portfolio_initial_balance_matches_config(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            app._setup_single_strategy()

            expected = Decimal(str(config.paper.initial_balance_usd))
            assert app.portfolio.get_cash_balance() == expected
        finally:
            await bus.stop()


# ---------------------------------------------------------------------------
# CerebrumCoin._setup_multi_strategy
# ---------------------------------------------------------------------------

class TestSetupMultiStrategy:
    """Multi-strategy path starts all three pipelines, sets registry/allocator/conductor."""

    @pytest.mark.asyncio
    async def test_registry_allocator_conductor_created(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            await app._setup_multi_strategy()

            assert app.strategy_registry is not None
            assert app.allocator is not None
            assert app.conductor is not None
        finally:
            if app.conductor:
                await app.conductor.stop()
            if app.web_dashboard:
                await app.web_dashboard.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_two_pipelines_active(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            await app._setup_multi_strategy()

            active = set(app.strategy_registry.active_strategy_names())
            # Only mean_reversion and range_trading active (DEC-TUNE-008).
            # momentum, breakout, news_driven disabled (signal cannibalization).
            # swing_trading disabled (DEC-TUNE-005).
            assert active == {"mean_reversion", "range_trading"}
        finally:
            if app.conductor:
                await app.conductor.stop()
            if app.web_dashboard:
                await app.web_dashboard.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_single_strategy_attrs_remain_none_in_multi_mode(self):
        """Multi-strategy setup must not populate legacy single-strategy attrs."""
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            await app._setup_multi_strategy()

            assert app.portfolio is None
            assert app.risk_manager is None
            assert app.exit_monitor is None
            assert app.signal_agg is None
        finally:
            if app.conductor:
                await app.conductor.stop()
            if app.web_dashboard:
                await app.web_dashboard.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_allocator_knows_active_strategies(self):
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            await app._setup_multi_strategy()

            # Only mean_reversion and range_trading active (DEC-TUNE-008).
            # swing_trading disabled (DEC-TUNE-005).
            assert set(app.allocator._strategies) == {"mean_reversion", "range_trading"}
        finally:
            if app.conductor:
                await app.conductor.stop()
            if app.web_dashboard:
                await app.web_dashboard.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_web_dashboard_created_if_fastapi_available(self):
        """WebDashboard is created when fastapi is installed; None otherwise."""
        from cerebrum.main import CerebrumCoin
        bus = EventBus()
        await bus.start()
        try:
            config = _load_config()
            app = CerebrumCoin(config)
            app.bus = bus
            await app._setup_multi_strategy()

            try:
                from cerebrum.dashboard.web import WebDashboard
                assert app.web_dashboard is not None
                assert isinstance(app.web_dashboard, WebDashboard)
            except ImportError:
                assert app.web_dashboard is None
        finally:
            if app.web_dashboard:
                await app.web_dashboard.stop()
            if app.conductor:
                await app.conductor.stop()
            await bus.stop()
