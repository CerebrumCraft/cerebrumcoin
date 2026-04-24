"""
Tests for StrategyRegistry — pipeline lifecycle and strategy_id routing.

Covers:
1. Registering a single strategy creates all pipeline components
2. Registering multiple strategies creates independent pipelines
3. Error isolation: one strategy failing start doesn't block others
4. strategy_id filtering: strategy A's aggregator only feeds strategy A's
   RiskManager, not strategy B's

Uses real EventBus and real Config — no mocks of internal modules.

@decision DEC-STRAT-003
@title StrategyRegistry owns pipeline lifecycle with error isolation
@status accepted
@rationale One failing strategy must not crash others. Pipeline components
are created per-strategy with strategy_id tagging for signal routing isolation.
Tests verify the full wiring: aggregator emits tagged COMBINED signals →
RiskManager receives only its own strategy's signals.
"""

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Fixtures and helpers
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
    config, _raw_toml = Config.from_toml(CONFIG_PATH)
    return config


def make_strategy(
    name: str,
    balance: Decimal = Decimal("5000.0"),
) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        aggregator_weights={
            SignalType.TECHNICAL: Decimal("1.0"),
            SignalType.SENTIMENT: Decimal("0.5"),
            SignalType.NEWS: Decimal("0.3"),
            SignalType.REGIME: Decimal("0.7"),
        },
        aggregator_threshold=Decimal("0.4"),
        initial_balance=balance,
    )


# ---------------------------------------------------------------------------
# Single-strategy pipeline creation
# ---------------------------------------------------------------------------

class TestStrategyRegistryPipelineCreation:
    """Registering a strategy creates all required pipeline components."""

    async def test_single_strategy_creates_all_components(self, bus, config):
        """After start_all, all four pipeline components exist."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        await registry.start_all()

        assert registry.get_aggregator("alpha") is not None
        assert registry.get_portfolio("alpha") is not None
        assert registry.get_exit_monitor("alpha") is not None
        assert registry.get_risk_manager("alpha") is not None

        assert isinstance(registry.get_aggregator("alpha"), SignalAggregator)
        assert isinstance(registry.get_portfolio("alpha"), PortfolioTracker)
        assert isinstance(registry.get_exit_monitor("alpha"), ExitMonitor)
        assert isinstance(registry.get_risk_manager("alpha"), RiskManager)

    async def test_portfolio_initial_balance(self, bus, config):
        """Portfolio tracker is initialized with strategy's initial_balance."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha", balance=Decimal("3333.33")))
        await registry.start_all()

        portfolio = registry.get_portfolio("alpha")
        assert portfolio is not None
        assert portfolio.get_cash_balance() == Decimal("3333.33")

    async def test_unknown_strategy_returns_none(self, bus, config):
        """Accessing a non-registered strategy returns None for all components."""
        registry = StrategyRegistry(bus, config)
        assert registry.get_aggregator("ghost") is None
        assert registry.get_portfolio("ghost") is None
        assert registry.get_exit_monitor("ghost") is None
        assert registry.get_risk_manager("ghost") is None

    async def test_duplicate_registration_raises(self, bus, config):
        """Registering the same strategy name twice raises ValueError."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(make_strategy("alpha"))

    async def test_list_strategies(self, bus, config):
        """list_strategies returns all registered names."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        registry.register(make_strategy("beta"))
        assert set(registry.list_strategies()) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Multiple independent pipelines
# ---------------------------------------------------------------------------

class TestStrategyRegistryMultiplePipelines:
    """Multiple strategies get truly independent pipeline objects."""

    async def test_two_strategies_have_separate_components(self, bus, config):
        """alpha and beta get different component instances."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha", balance=Decimal("3000.0")))
        registry.register(make_strategy("beta", balance=Decimal("7000.0")))
        await registry.start_all()

        agg_a = registry.get_aggregator("alpha")
        agg_b = registry.get_aggregator("beta")
        assert agg_a is not agg_b

        port_a = registry.get_portfolio("alpha")
        port_b = registry.get_portfolio("beta")
        assert port_a is not port_b

        rm_a = registry.get_risk_manager("alpha")
        rm_b = registry.get_risk_manager("beta")
        assert rm_a is not rm_b

    async def test_portfolio_balances_are_independent(self, bus, config):
        """Each strategy has its own cash balance."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha", balance=Decimal("3000.0")))
        registry.register(make_strategy("beta", balance=Decimal("7000.0")))
        await registry.start_all()

        assert registry.get_portfolio("alpha").get_cash_balance() == Decimal("3000.0")
        assert registry.get_portfolio("beta").get_cash_balance() == Decimal("7000.0")

    async def test_active_strategy_names(self, bus, config):
        """active_strategy_names returns names of started pipelines."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        registry.register(make_strategy("beta"))
        await registry.start_all()

        active = registry.active_strategy_names()
        assert set(active) == {"alpha", "beta"}

    async def test_global_portfolio_spans_all_strategies(self, bus, config):
        """global_portfolio aggregates equity across all strategies."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha", balance=Decimal("3000.0")))
        registry.register(make_strategy("beta", balance=Decimal("7000.0")))
        await registry.start_all()

        gp = registry.global_portfolio
        assert gp.get_total_equity() == Decimal("10000.0")


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------

class TestStrategyRegistryErrorIsolation:
    """One failing strategy does not prevent others from starting."""

    async def test_bad_strategy_does_not_block_others(self, bus, config):
        """
        When one strategy's balance is invalid, others still start.

        We simulate a startup failure by registering a strategy with a
        negative initial_balance that causes PortfolioTracker to accept it
        (it doesn't validate sign), then manually testing that the registry's
        error isolation catches exceptions during pipeline construction.

        The real isolation test: we subclass StrategyRegistry to inject a
        failing _build_pipeline for one strategy.
        """
        from cerebrum.strategies.registry import StrategyRegistry, _StrategyPipeline

        class FaultyRegistry(StrategyRegistry):
            def _build_pipeline(self, cfg, shared_global_rules, effective_balance=None):
                if cfg.name == "broken":
                    raise RuntimeError("Simulated pipeline construction failure")
                return super()._build_pipeline(cfg, shared_global_rules, effective_balance)

        registry = FaultyRegistry(bus, config)
        registry.register(make_strategy("broken"))
        registry.register(make_strategy("healthy"))
        await registry.start_all()

        # broken failed → None; healthy succeeded → not None
        assert registry.get_portfolio("broken") is None
        assert registry.get_portfolio("healthy") is not None

    async def test_stop_all_clears_pipelines(self, bus, config):
        """After stop_all, active_strategy_names returns empty list."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        await registry.start_all()
        assert "alpha" in registry.active_strategy_names()

        await registry.stop_all()
        assert registry.active_strategy_names() == []


# ---------------------------------------------------------------------------
# strategy_id routing
# ---------------------------------------------------------------------------

class TestStrategyRegistrySignalRouting:
    """strategy_id filtering ensures pipeline isolation."""

    async def test_aggregator_tags_combined_signals(self, bus, config):
        """SignalAggregator emits COMBINED signals tagged with strategy_id."""
        import time as time_mod
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        await registry.start_all()

        combined_signals: list[SignalEvent] = []

        async def capture(event):
            if (
                isinstance(event, SignalEvent)
                and event.signal_type == SignalType.COMBINED
            ):
                combined_signals.append(event)

        bus.subscribe(EventType.SIGNAL, capture, subscriber_name="test_capture")

        # Use current wall-clock timestamps so signals are not evicted by the
        # aggregator's time-window cleanup (which compares against time.time()).
        now = time_mod.time()
        for i in range(3):
            raw = SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=now + i,
                signal_type=SignalType.TECHNICAL,
                symbol="BTC/USD",
                action=SignalAction.BUY,
                strength=Decimal("0.8"),
                confidence=Decimal("0.9"),
            )
            await bus.publish(raw)

        await asyncio.sleep(0.2)

        # At least one COMBINED signal should have been emitted
        assert len(combined_signals) > 0
        # All COMBINED signals from alpha's aggregator must be tagged "alpha"
        for sig in combined_signals:
            assert sig.strategy_id == "alpha", (
                f"Expected strategy_id='alpha', got '{sig.strategy_id}'"
            )

    async def test_risk_manager_ignores_other_strategy_signals(self, bus, config):
        """RiskManager for 'beta' ignores COMBINED signals tagged 'alpha'."""
        registry = StrategyRegistry(bus, config)
        registry.register(make_strategy("alpha"))
        registry.register(make_strategy("beta"))
        await registry.start_all()

        beta_orders: list[OrderEvent] = []

        async def capture_orders(event):
            if isinstance(event, OrderEvent):
                beta_orders.append(event)

        # We want to check whether beta's RM emits orders when alpha's
        # aggregator signal fires. We'll publish a COMBINED signal tagged
        # "alpha" directly and verify beta's RiskManager ignores it.
        bus.subscribe(EventType.ORDER, capture_orders, subscriber_name="order_capture")

        # Manually emit a COMBINED signal tagged "alpha" with high strength
        alpha_signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=1000.0,
            signal_type=SignalType.COMBINED,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.9"),
            confidence=Decimal("0.9"),
            strategy_id="alpha",
        )
        await bus.publish(alpha_signal)
        await asyncio.sleep(0.15)

        # beta's RiskManager should have ignored the alpha-tagged signal.
        # alpha's RiskManager will process it — it may be denied by sizing
        # rules (no price data) but it will not produce an ORDER from beta.
        # We verify by checking that no ORDER carries beta's strategy_id.
        # (Alpha's RM may or may not emit an order depending on rules.)
        for order in beta_orders:
            assert order.strategy_id != "beta", (
                "beta's RiskManager should not have processed alpha's signal"
            )
