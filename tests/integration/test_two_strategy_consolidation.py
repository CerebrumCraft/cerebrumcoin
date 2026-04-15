"""
Integration test for 2-strategy consolidated pipeline.

Verifies that the strategy-consolidation decisions (DEC-TUNE-008) produce
the correct runtime state:
  - Only mean_reversion and range_trading are active
  - Each strategy receives $5,000 initial capital
  - Signal source filtering isolates the two strategies: RSI (TECHNICAL)
    signals reach mean_reversion but are rejected by range_trading's
    SupportResistance-only filter

Uses real EventBus, real Config, and the production StrategyConfig objects
(MEAN_REVERSION_CONFIG, RANGE_TRADING_CONFIG) — no mocks of internal modules.

@decision DEC-TUNE-008
@title 2-strategy split: mean_reversion + range_trading at $5,000 each
@status accepted
@rationale Consolidating 6 strategies down to 2 with equal $5,000 capital
allocations (from the $10,000 portfolio) reduces commission drag and focuses
capital on the two highest-performing strategies from Session 18.
"""

import asyncio
import time as time_mod
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
from cerebrum.strategies.registry import StrategyRegistry

CONFIG_PATH = Path("config/paper.toml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
async def two_strategy_registry(bus, config):
    """StrategyRegistry with only mean_reversion and range_trading registered."""
    registry = StrategyRegistry(bus, config)
    registry.register(MEAN_REVERSION_CONFIG)
    registry.register(RANGE_TRADING_CONFIG)
    await registry.start_all()
    yield registry
    await registry.stop_all()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTwoStrategyConsolidation:
    """Integration tests for the 2-strategy consolidated pipeline (DEC-TUNE-008)."""

    async def test_only_two_strategies_active(self, two_strategy_registry):
        """After start_all, only mean_reversion and range_trading are active."""
        active = two_strategy_registry.active_strategy_names()
        assert set(active) == {"mean_reversion", "range_trading"}, (
            f"Expected exactly mean_reversion + range_trading, got: {active}"
        )

    async def test_capital_allocation(self, two_strategy_registry):
        """Each strategy's PortfolioTracker starts with $5,000 equity (DEC-TUNE-008)."""
        mr_portfolio = two_strategy_registry.get_portfolio("mean_reversion")
        rt_portfolio = two_strategy_registry.get_portfolio("range_trading")

        assert mr_portfolio is not None, "mean_reversion portfolio should exist"
        assert rt_portfolio is not None, "range_trading portfolio should exist"

        assert mr_portfolio.get_cash_balance() == Decimal("5000.00"), (
            f"mean_reversion expected $5,000 balance, got {mr_portfolio.get_cash_balance()}"
        )
        assert rt_portfolio.get_cash_balance() == Decimal("5000.00"), (
            f"range_trading expected $5,000 balance, got {rt_portfolio.get_cash_balance()}"
        )

    async def test_signal_isolation(self, bus, two_strategy_registry):
        """
        RSI TECHNICAL signals reach mean_reversion but are filtered by range_trading.

        range_trading uses signal_source_filter="SupportResistance", so any signal
        whose metadata["source"] != "SupportResistance" is silently dropped before
        it enters the aggregator's buffer. mean_reversion has no source filter and
        accepts all TECHNICAL signals.
        """
        mr_agg = two_strategy_registry.get_aggregator("mean_reversion")
        rt_agg = two_strategy_registry.get_aggregator("range_trading")

        assert mr_agg is not None, "mean_reversion aggregator should exist"
        assert rt_agg is not None, "range_trading aggregator should exist"

        # Publish an RSI TECHNICAL signal with source="RSI" — should pass
        # mean_reversion's aggregator (no filter) and be rejected by
        # range_trading's aggregator (filter="SupportResistance").
        # Use current wall-clock time so the signal is not evicted by the
        # aggregator's time-window cleanup (which compares against time.time()).
        now = time_mod.time()
        rsi_signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=now,
            signal_type=SignalType.TECHNICAL,
            symbol="ETH/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.9"),
            metadata={"source": "RSI", "timeframe": "1m"},
        )
        await bus.publish(rsi_signal)

        # Allow the async event dispatch to propagate
        await asyncio.sleep(0.1)

        # mean_reversion's buffer should contain the RSI signal
        mr_buffer = mr_agg._signal_buffer.get("ETH/USD", [])
        assert len(mr_buffer) >= 1, (
            f"mean_reversion aggregator should have buffered the RSI signal "
            f"(buffer length: {len(mr_buffer)})"
        )

        # range_trading's buffer should remain empty — source filter rejects RSI
        rt_buffer = rt_agg._signal_buffer.get("ETH/USD", [])
        assert len(rt_buffer) == 0, (
            f"range_trading aggregator should have rejected the RSI signal "
            f"(buffer length: {len(rt_buffer)}, expected 0)"
        )
