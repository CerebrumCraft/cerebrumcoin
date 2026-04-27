"""
Tests for per-strategy equity snapshot correctness in WebDashboard._on_fill.

Issue #36: /api/strategy_equity_history shows phantom sloshing because the
dashboard's _on_fill snapshots equity at the moment the FILL event arrives,
BEFORE the per-strategy PortfolioTracker has processed the same event. Both
subscribe to the same bus; the tracker's queue drains asynchronously.

Fix validated here: after the snapshot call, the equity values must reflect
the post-fill state (tracker has processed the fill before we read equity).

@decision DEC-EQUITY-001
@title Per-strategy equity snapshot must be taken after tracker processes fill
@status accepted
@rationale Dashboard _on_fill was snapshotting portfolio.get_total_equity()
immediately on FILL arrival, before PortfolioTracker._on_fill had consumed
the event from its own queue. This caused the per-strategy chart to show
pre-fill equity values, producing anti-correlated phantom swings. Fix: yield
with asyncio.sleep(0) in dashboard._on_fill before reading tracker equities.
"""

import asyncio
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side, SignalType
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.global_portfolio import GlobalPortfolio
from cerebrum.strategies.registry import StrategyRegistry

fastapi = pytest.importorskip("fastapi")
from cerebrum.dashboard.web import WebDashboard  # noqa: E402

CONFIG_PATH = Path("config/paper.toml")
POOL_USD = Decimal("10000")


def _make_strategy_config(name: str, balance: Decimal = Decimal("5000")) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        aggregator_weights={
            SignalType.TECHNICAL: Decimal("1.0"),
        },
        aggregator_threshold=Decimal("0.4"),
        initial_balance=balance,
    )


@pytest.fixture
async def bus():
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def registry(bus):
    config, _ = Config.from_toml(CONFIG_PATH)
    reg = StrategyRegistry(bus, config, pool_usd=POOL_USD)
    reg.register(_make_strategy_config("strat_alpha"))
    reg.register(_make_strategy_config("strat_beta"))
    await reg.start_all()
    yield reg
    await reg.stop_all()


@pytest.fixture
def allocator():
    return DarwinianAllocator(
        strategy_names=["strat_alpha", "strat_beta"],
        total_capital=POOL_USD,
        warmup_hours=0.0,
    )


@pytest.fixture
def conductor(bus, registry, allocator):
    return Conductor(
        bus=bus,
        registry=registry,
        allocator=allocator,
        anthropic_api_key=None,
        poll_interval_seconds=900,
    )


@pytest.fixture
def dashboard(bus, registry, conductor):
    gp = registry.global_portfolio
    return WebDashboard(
        bus=bus,
        registry=registry,
        conductor=conductor,
        global_portfolio=gp,
        host="127.0.0.1",
        port=18182,
    )


@pytest.mark.asyncio
async def test_per_strategy_equity_sum_matches_pool_after_fill(bus, registry, dashboard):
    """
    After a closed round-trip trade, the sum of per-strategy equity snapshots
    captured in _on_fill must equal pool_usd + realized_pnl (not a constant).

    This test was failing before the DEC-EQUITY-001 fix because the dashboard
    snapshotted equity before the PortfolioTracker processed the fill event,
    showing pre-fill (constant) values instead of post-fill values.
    """
    # Subscribe dashboard to FILL events (simulating WebDashboard.start())
    bus.subscribe(EventType.FILL, dashboard._on_fill, "test_dashboard_fill")

    # Get the two trackers
    pt_alpha = registry.get_portfolio("strat_alpha")
    pt_beta = registry.get_portfolio("strat_beta")
    assert pt_alpha is not None
    assert pt_beta is not None

    initial_alpha = pt_alpha.get_total_equity()
    initial_beta = pt_beta.get_total_equity()
    initial_sum = initial_alpha + initial_beta

    # BUY: strat_alpha buys 0.01 BTC/USD at 50000
    buy_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id="buy-1",
        symbol="BTC/USD",
        side=Side.BUY,
        filled_amount=Decimal("0.01"),
        fill_price=Decimal("50000"),
        commission=Decimal("0.80"),
        commission_asset="USD",
        exchange_order_id="paper_buy_1",
        strategy_id="strat_alpha",
    )
    await bus.publish(buy_fill)
    await asyncio.sleep(0.2)  # let all subscribers process

    # SELL: close the position at 50251 (+$2.51 gross, -$1.616 commission)
    sell_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id="sell-1",
        symbol="BTC/USD",
        side=Side.SELL,
        filled_amount=Decimal("0.01"),
        fill_price=Decimal("50251"),
        commission=Decimal("0.804016"),
        commission_asset="USD",
        exchange_order_id="paper_sell_1",
        strategy_id="strat_alpha",
    )
    await bus.publish(sell_fill)
    await asyncio.sleep(0.2)  # let all subscribers process

    # The most recent equity snapshot for each strategy (from dashboard._on_fill)
    alpha_history = dashboard._strategy_equity_history.get("strat_alpha", [])
    beta_history = dashboard._strategy_equity_history.get("strat_beta", [])

    assert len(alpha_history) > 0, "Dashboard should have snapshotted strat_alpha equity"
    assert len(beta_history) > 0, "Dashboard should have snapshotted strat_beta equity"

    last_alpha_equity = Decimal(str(alpha_history[-1]["equity"]))
    last_beta_equity = Decimal(str(beta_history[-1]["equity"]))
    equity_sum = last_alpha_equity + last_beta_equity

    # After a profitable round-trip: sum must be > initial_sum (not a constant $10,000)
    # strat_alpha made ~$2.51 - $1.604 = ~$0.906 net
    expected_realized_pnl = (
        Decimal("50251") - Decimal("50000")
    ) * Decimal("0.01") - Decimal("0.80") - Decimal("0.804016")

    # Allow 1-cent tolerance for floating-point conversion
    tolerance = Decimal("0.02")
    actual_pnl = equity_sum - initial_sum
    assert abs(actual_pnl - expected_realized_pnl) < tolerance, (
        f"Equity sum should reflect realized PnL={expected_realized_pnl:.4f} "
        f"but got {actual_pnl:.4f} (sum={equity_sum:.4f}, initial={initial_sum:.4f}). "
        f"This indicates the dashboard snapshotted equity before the tracker processed the fill."
    )


@pytest.mark.asyncio
async def test_per_strategy_equity_sum_not_constant_across_fills(bus, registry, dashboard):
    """
    The sum of per-strategy equities must NOT be constant across fill events.

    Before the fix, every snapshot showed sum=pool_usd (constant) because the
    tracker had not yet debited/credited cash. After the fix, the sum should
    change as PnL accumulates.
    """
    bus.subscribe(EventType.FILL, dashboard._on_fill, "test_dashboard_fill2")

    pt_alpha = registry.get_portfolio("strat_alpha")
    assert pt_alpha is not None

    # BUY with commission reduces equity immediately
    buy_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id="buy-2",
        symbol="ETH/USD",
        side=Side.BUY,
        filled_amount=Decimal("0.1"),
        fill_price=Decimal("3000"),
        commission=Decimal("0.48"),
        commission_asset="USD",
        exchange_order_id="paper_buy_2",
        strategy_id="strat_alpha",
    )
    await bus.publish(buy_fill)
    await asyncio.sleep(0.2)

    alpha_history = dashboard._strategy_equity_history.get("strat_alpha", [])
    beta_history = dashboard._strategy_equity_history.get("strat_beta", [])

    assert len(alpha_history) > 0
    assert len(beta_history) > 0

    last_alpha = Decimal(str(alpha_history[-1]["equity"]))
    last_beta = Decimal(str(beta_history[-1]["equity"]))
    equity_sum = last_alpha + last_beta

    # After a BUY with commission, strat_alpha equity should be:
    # initial_alpha - commission (commission is a real cost even if position value is same)
    # The sum should be slightly less than pool due to the commission
    pool = Decimal("10000")
    commission_paid = Decimal("0.48")

    # After buy: alpha has same position value but paid commission, so equity slightly down
    # (assuming entry price == current price, unrealized PnL = 0 immediately)
    expected_sum = pool - commission_paid
    tolerance = Decimal("0.05")
    assert abs(equity_sum - expected_sum) < tolerance, (
        f"Sum after BUY should be pool-commission={expected_sum:.4f}, "
        f"got {equity_sum:.4f}. "
        f"Dashboard may still be snapshotting pre-fill tracker state."
    )
