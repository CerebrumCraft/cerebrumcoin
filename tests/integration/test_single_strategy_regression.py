"""
Regression test: single MomentumStrategy produces identical pipeline behavior
to the pre-refactor direct main.py wiring.

This is the most critical test for Phase 11A. It proves that:
1. A StrategyRegistry with a single MomentumStrategy configuration creates
   a pipeline that behaves identically to the pre-existing wiring.
2. COMBINED signals emitted by the strategy's aggregator are tagged with
   the strategy_id "momentum".
3. The strategy's RiskManager receives and processes those signals.
4. The strategy's PortfolioTracker starts at the same initial balance as
   the current paper.toml configuration.
5. The exit monitor is configured with the same thresholds.

Backward compatibility guarantee: existing paper trading sessions that run
with a single strategy MUST see zero behavioural change from this refactor.

@decision DEC-STRAT-006
@title Backward-compatible single-strategy mode in main.py
@status accepted
@rationale If no strategy config is present in TOML, main.py creates a single
MomentumStrategy pipeline identical to current main.py wiring. MOMENTUM_CONFIG
uses initial_balance=$10k (not the 1/3 default) and all risk_overrides mirror
paper.toml exactly. This test validates that equivalence.
"""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    MaxDrawdownRule,
    MaxPositionSizeRule,
    MaxTotalExposureRule,
    MinSignalStrengthRule,
    PositionSizingRule,
    PostFillCooldownRule,
)
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.strategies.momentum import MOMENTUM_CONFIG
from cerebrum.strategies.registry import StrategyRegistry

CONFIG_PATH = Path("config/paper.toml")


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def config():
    return Config.from_toml(CONFIG_PATH)


class TestMomentumConfigValues:
    """MOMENTUM_CONFIG matches paper.toml values exactly."""

    def test_initial_balance_is_sixth_paper_balance(self):
        """MOMENTUM_CONFIG.initial_balance = 1/6 of $10k for 6-strategy split."""
        assert MOMENTUM_CONFIG.initial_balance == Decimal("1666.67")

    def test_aggregator_threshold_matches_paper_toml(self):
        """Threshold 0.4 matches paper.toml aggregation_threshold."""
        assert MOMENTUM_CONFIG.aggregator_threshold == Decimal("0.4")

    def test_aggregator_weights_match_defaults(self):
        """Weights match the SignalAggregator defaults in production."""
        weights = MOMENTUM_CONFIG.aggregator_weights
        assert weights[SignalType.TECHNICAL] == Decimal("1.0")
        assert weights[SignalType.SENTIMENT] == Decimal("0.5")
        assert weights[SignalType.NEWS] == Decimal("0.3")
        assert weights[SignalType.REGIME] == Decimal("0.7")

    def test_risk_overrides_match_paper_toml(self):
        """Risk overrides reflect the paper.toml [risk] section tuning.

        stop_loss_percent is 1.0% (tightened from 1.5% per Session 11
        sensitivity analysis — DEC-TUNE-004).
        """
        overrides = MOMENTUM_CONFIG.risk_overrides
        assert overrides["min_signal_strength"] == "0.6"
        assert overrides["position_size_percent"] == "5.0"
        assert overrides["stop_loss_percent"] == "1.0"

    def test_exit_config_matches_paper_toml(self):
        """Exit config reflects stop_loss, take_profit, age.

        stop_loss_percent is 1.0% (tightened from 1.5% per Session 11
        sensitivity analysis — DEC-TUNE-004).
        """
        ec = MOMENTUM_CONFIG.exit_config
        assert ec["stop_loss_percent"] == "1.0"
        assert ec["take_profit_percent"] == "3.0"
        assert ec["max_position_age_minutes"] == 120
        assert ec["adaptive_tp"] is True

    def test_symbols_match_paper_trading_symbols(self):
        """Default symbols are BTC/USD and ETH/USD."""
        assert "BTC/USD" in MOMENTUM_CONFIG.symbols
        assert "ETH/USD" in MOMENTUM_CONFIG.symbols

    def test_strategy_name(self):
        """Strategy name is 'momentum'."""
        assert MOMENTUM_CONFIG.name == "momentum"


class TestSingleStrategyPipelineEquivalence:
    """
    StrategyRegistry with MomentumStrategy produces a pipeline equivalent
    to the pre-refactor direct wiring.
    """

    async def test_portfolio_starts_at_fifth_paper_balance(self, bus, config):
        """Portfolio balance matches MOMENTUM_CONFIG.initial_balance (1/5 of $10k)."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        portfolio = registry.get_portfolio("momentum")
        assert portfolio is not None
        # 1/6 of $10k — equal split across 6 strategies
        assert portfolio.get_cash_balance() == Decimal("1666.67")

    async def test_aggregator_uses_correct_threshold(self, bus, config):
        """Aggregator threshold matches paper.toml aggregation_threshold."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        agg = registry.get_aggregator("momentum")
        assert agg is not None
        assert agg._threshold == Decimal("0.4")

    async def test_risk_manager_has_min_signal_strength_rule(self, bus, config):
        """RiskManager includes MinSignalStrengthRule with paper.toml value."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        rm = registry.get_risk_manager("momentum")
        assert rm is not None

        min_strength_rules = [
            r for r in rm._rules if isinstance(r, MinSignalStrengthRule)
        ]
        assert len(min_strength_rules) == 1
        assert min_strength_rules[0]._min_strength == Decimal("0.6")

    async def test_risk_manager_has_position_sizing_rule(self, bus, config):
        """RiskManager includes PositionSizingRule with paper.toml 5% size."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        rm = registry.get_risk_manager("momentum")
        sizing_rules = [r for r in rm._rules if isinstance(r, PositionSizingRule)]
        assert len(sizing_rules) == 1
        assert sizing_rules[0]._size_percent == Decimal("5.0")

    async def test_exit_monitor_stop_loss_matches_paper(self, bus, config):
        """ExitMonitor stop-loss matches MOMENTUM_CONFIG: 1.0% (DEC-TUNE-004)."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        em = registry.get_exit_monitor("momentum")
        assert em is not None
        assert em._stop_loss_pct == Decimal("1.0")

    async def test_exit_monitor_take_profit_matches_paper(self, bus, config):
        """ExitMonitor take-profit matches paper.toml take_profit_percent = 3.0."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        em = registry.get_exit_monitor("momentum")
        assert em._take_profit_pct == Decimal("3.0")

    async def test_exit_monitor_adaptive_tp_enabled(self, bus, config):
        """Adaptive TP is enabled as configured in paper.toml."""
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        em = registry.get_exit_monitor("momentum")
        assert em._adaptive_tp is True


class TestSingleStrategySignalFlow:
    """
    COMBINED signals flow through the single-strategy pipeline correctly.
    """

    async def test_combined_signal_tagged_with_momentum_strategy_id(
        self, bus, config
    ):
        """Aggregator tags COMBINED signals with strategy_id='momentum'."""
        import time as time_mod
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        combined_signals: list[SignalEvent] = []

        async def capture(event):
            if (
                isinstance(event, SignalEvent)
                and event.signal_type == SignalType.COMBINED
            ):
                combined_signals.append(event)

        bus.subscribe(EventType.SIGNAL, capture, subscriber_name="regression_capture")

        # Use current wall-clock timestamps so signals survive the aggregator's
        # time-window cleanup (cutoff = time.time() - window_seconds).
        now = time_mod.time()
        for i in range(4):
            raw = SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=now + i,
                signal_type=SignalType.TECHNICAL,
                symbol="BTC/USD",
                action=SignalAction.BUY,
                strength=Decimal("0.85"),
                confidence=Decimal("0.9"),
            )
            await bus.publish(raw)

        await asyncio.sleep(0.2)

        assert len(combined_signals) > 0, (
            "Expected at least one COMBINED signal from momentum aggregator"
        )
        for sig in combined_signals:
            assert sig.strategy_id == "momentum", (
                f"Expected strategy_id='momentum', got '{sig.strategy_id}'"
            )

    async def test_risk_manager_filters_to_own_strategy_signals(
        self, bus, config
    ):
        """
        Momentum RiskManager ignores COMBINED signals tagged for other strategies.

        Publishes a COMBINED signal tagged 'other_strategy' and verifies the
        momentum RiskManager does not emit an order.
        """
        registry = StrategyRegistry(bus, config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all()

        orders_emitted: list[OrderEvent] = []

        async def capture_orders(event):
            if isinstance(event, OrderEvent):
                orders_emitted.append(event)

        bus.subscribe(EventType.ORDER, capture_orders, subscriber_name="order_spy")

        # Emit a COMBINED signal for a different strategy
        foreign_signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=1000.0,
            signal_type=SignalType.COMBINED,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.95"),
            confidence=Decimal("0.95"),
            strategy_id="other_strategy",  # Not "momentum"
        )
        await bus.publish(foreign_signal)
        await asyncio.sleep(0.15)

        # Momentum RiskManager must have ignored this signal — no orders
        # from momentum's pipeline (orders_emitted should be empty since
        # the only signal published was tagged for another strategy)
        assert orders_emitted == [], (
            f"Momentum RiskManager should have ignored foreign signal, "
            f"but emitted {len(orders_emitted)} order(s)"
        )
