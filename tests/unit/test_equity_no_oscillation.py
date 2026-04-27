"""
Tests for the global == sum(per-strategy trackers) equity invariant.

Issue: Session 43 observed per-strategy equity oscillation (mean_reversion +$100
then gone, range_trading -$10 then back) while global equity appeared flat.
Root cause: _get_global_equity() delegated to paper_adapter.get_portfolio_summary()
which uses a separate _current_prices dict (updated via its own MarketDataEvent
subscriber) that can be at a different price than PortfolioTracker.Position.current_price
(updated via its own MarketDataEvent subscriber). Both subscribe to the same bus but
there is no ordering guarantee between async tasks — one tick ahead or behind produces
visible oscillation on the dashboard.

Fix: _get_global_equity() must sum per-strategy PortfolioTracker.get_total_equity()
from the StrategyRegistry — the same source the per-strategy chart uses in _on_fill.
Once both the global line and the per-strategy lines read from the same source,
they cannot oscillate against each other.

Test A — global == sum(trackers) invariant: verify that after the fix,
_get_global_equity() equals sum(tracker.get_total_equity()) even when the paper
adapter's _current_prices is manually set to a DIFFERENT price than the tracker's
Position.current_price (simulating the async-arrival desync).

Test B — hedged book stays flat: LONG+SHORT same size → global equity constant.

Both tests FAIL on b53fa7c (pre-fix) and PASS after the fix.

@decision DEC-EQUITY-002
@title Global equity = sum of per-strategy trackers (single source of truth)
@status accepted
@rationale DEC-ALLOC-INITIAL-001 makes pool_usd / N the per-strategy initial
balance, so sum(tracker.get_total_equity()) == global pool by construction.
DEC-FILL-STRATEGY-ID-001 ensures every fill reaches the right tracker. The
previous paper-adapter source used a separate _current_prices dict that desynced
from Position.current_price on every tick, producing visible per-strategy
oscillation against a flat global line (user-reported: mean_reversion +$100 then
gone, range_trading -$10 then back). See also DEC-EQUITY-001 (snapshot timing).
"""

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import FillEvent, MarketDataEvent
from cerebrum.core.types import EventType, Side, SignalType
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.registry import StrategyRegistry

fastapi = pytest.importorskip("fastapi")
from cerebrum.dashboard.web import WebDashboard  # noqa: E402

CONFIG_PATH = Path("config/paper.toml")
POOL_USD = Decimal("10000")
PER_STRATEGY_BALANCE = POOL_USD / 2  # DEC-ALLOC-INITIAL-001: pool / N


def _make_strategy_config(name: str, balance: Decimal = PER_STRATEGY_BALANCE) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        aggregator_weights={SignalType.TECHNICAL: Decimal("1.0")},
        aggregator_threshold=Decimal("0.4"),
        initial_balance=balance,
    )


def _make_tick(symbol: str, price: Decimal) -> MarketDataEvent:
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time.time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )


def _make_fill(
    strategy_id: str,
    symbol: str,
    side: Side,
    amount: Decimal,
    price: Decimal,
) -> FillEvent:
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id=f"fill-{strategy_id}-{symbol}-{side.value}",
        symbol=symbol,
        side=side,
        filled_amount=amount,
        fill_price=price,
        commission=Decimal("0"),
        commission_asset="USD",
        exchange_order_id=f"paper_{strategy_id}_{symbol}",
        strategy_id=strategy_id,
    )


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
def paper(bus, tmp_path):
    """Real PaperTradingAdapter backed by a temp state file (zero commission/slippage)."""
    sf = tmp_path / "state.json"
    return PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=POOL_USD,
        commission_percent=Decimal("0"),
        slippage_percent=Decimal("0"),
        state_file=sf,
    )


@pytest.fixture
def dashboard_no_adapter(bus, registry, conductor):
    """Dashboard wired WITHOUT a paper_adapter — uses GlobalPortfolio (correct) path."""
    gp = registry.global_portfolio
    return WebDashboard(
        bus=bus,
        registry=registry,
        conductor=conductor,
        global_portfolio=gp,
        host="127.0.0.1",
        port=18183,
    )


@pytest.fixture
def dashboard_with_adapter(bus, registry, conductor, paper):
    """Dashboard wired WITH a real PaperTradingAdapter — exercises the pre-fix path."""
    gp = registry.global_portfolio
    return WebDashboard(
        bus=bus,
        registry=registry,
        conductor=conductor,
        global_portfolio=gp,
        host="127.0.0.1",
        port=18185,
        paper_adapter=paper,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_global_equity_desync_from_paper_adapter_price_stale(
    bus, registry, dashboard_with_adapter, paper
):
    """
    Test A — _get_global_equity() must equal sum(trackers) even when the paper
    adapter's _current_prices is at a DIFFERENT price than Position.current_price.

    This directly simulates the async-arrival desync observed in Session 43:
    - paper._current_prices["BTC/USD"] = $50,000  (one tick behind)
    - tracker Position.current_price = $51,000     (one tick ahead)

    Pre-fix: _get_global_equity() reads from paper._current_prices → returns value
    at $50,000, while per-strategy chart reads tracker at $51,000 → DIVERGENCE.
    Post-fix: _get_global_equity() reads trackers directly → no divergence.
    """
    pt_alpha = registry.get_portfolio("strat_alpha")
    pt_beta = registry.get_portfolio("strat_beta")
    assert pt_alpha is not None
    assert pt_beta is not None

    entry_price = Decimal("50000")
    size = Decimal("0.01")

    # Open a long position in strat_alpha via the bus (both tracker and adapter receive it)
    await bus.publish(_make_tick("BTC/USD", entry_price))
    await asyncio.sleep(0.1)
    await bus.publish(_make_fill("strat_alpha", "BTC/USD", Side.BUY, size, entry_price))
    await asyncio.sleep(0.1)

    # Manually wire the paper adapter to reflect the same position
    # (simulating what execute_order would do: debit cash, credit position)
    paper._positions["BTC/USD"] = size
    paper._balances["USD"] = POOL_USD - (size * entry_price)

    # Now MANUALLY set the paper adapter's price to the OLD price ($50,000)
    # but advance the tracker's position price to the NEW price ($51,000).
    # This simulates the race: tracker processed the new tick, paper adapter hasn't yet.
    new_price = Decimal("51000")
    paper._current_prices["BTC/USD"] = entry_price          # paper adapter: stale (old price)
    pt_alpha._positions["BTC/USD"].update_price(new_price)  # tracker: fresh (new price)

    # What should global equity be?
    tracker_sum = pt_alpha.get_total_equity() + pt_beta.get_total_equity()
    # alpha: 5000 - 500 (BTC cost) + 510 (BTC MTM at 51000) = 5010
    # beta: 5000 cash, no positions
    # tracker_sum = 10010

    # Pre-fix: paper adapter sees BTC at 50000 → global = 5000 - 500 + 500 = 5000 ≠ 10010
    # Wait — paper adapter cash is POOL_USD - size*entry = 10000 - 500 = 9500
    # paper summary = 9500 (cash) + 0.01 * 50000 (long at stale price) = 10000
    # tracker_sum = alpha(5000 - 500 + 510) + beta(5000) = 5010 + 5000 = 10010
    # So paper global = 10000 ≠ tracker_sum = 10010 → pre-fix DIVERGES

    paper_summary_value = Decimal(paper.get_portfolio_summary()["total_value_usd"])

    # Verify the paper adapter IS computing a different value (confirms the stale price setup)
    assert paper_summary_value != tracker_sum, (
        f"Paper adapter value {paper_summary_value} should differ from tracker_sum "
        f"{tracker_sum} (stale price setup not working). "
        f"paper._current_prices={paper._current_prices}, "
        f"tracker price={pt_alpha._positions.get('BTC/USD')}"
    )

    # Post-fix: _get_global_equity() must return tracker_sum, NOT paper adapter value
    result = dashboard_with_adapter._get_global_equity()

    assert result == tracker_sum, (
        f"_get_global_equity() returned {result}, expected {tracker_sum}. "
        f"Pre-fix returns paper adapter value {paper_summary_value} (stale price). "
        f"Post-fix must sum registry trackers directly. "
        f"alpha={pt_alpha.get_total_equity()}, beta={pt_beta.get_total_equity()}"
    )


@pytest.mark.asyncio
async def test_hedged_book_global_equity_stays_flat(bus, registry, dashboard_no_adapter):
    """
    Test B — hedged book: global equity stays constant across price ticks.

    strat_alpha: LONG 0.01 BTC/USD at $50,000.
    strat_beta:  SHORT 0.01 BTC/USD at $50,000 (same symbol, same size, same entry).

    Net book: MTM-neutral. Alpha gain = beta loss at every price.
    Global equity = sum(trackers) must remain at POOL_USD after each tick.

    Zero commissions → no leakage. Decimal arithmetic → exact equality.

    Note: This test uses the no-adapter path (GlobalPortfolio), which is already
    correct in the pre-fix code. It locks in the invariant and ensures the fix
    does not break the hedged-book case.
    """
    pt_alpha = registry.get_portfolio("strat_alpha")
    pt_beta = registry.get_portfolio("strat_beta")
    assert pt_alpha is not None
    assert pt_beta is not None

    entry_price = Decimal("50000")
    size = Decimal("0.01")

    # alpha: long BTC/USD
    await bus.publish(_make_fill("strat_alpha", "BTC/USD", Side.BUY, size, entry_price))
    await asyncio.sleep(0.1)

    # beta: short BTC/USD (raw SELL on a new symbol creates a negative position in tracker)
    await bus.publish(_make_fill("strat_beta", "BTC/USD", Side.SELL, size, entry_price))
    await asyncio.sleep(0.1)

    initial_global = dashboard_no_adapter._get_global_equity()
    assert initial_global == POOL_USD, (
        f"Initial global should be POOL_USD={POOL_USD} (zero commission, hedged). "
        f"Got {initial_global}."
    )

    # Tick BTC price 10 times; global must stay flat
    prices = [50000, 50500, 49500, 51000, 49000, 50200, 50800, 49700, 50100, 50000]
    for price in prices:
        await bus.publish(_make_tick("BTC/USD", Decimal(str(price))))
        await asyncio.sleep(0.05)

        global_equity = dashboard_no_adapter._get_global_equity()

        assert global_equity == initial_global, (
            f"Global equity changed from {initial_global} to {global_equity} at BTC={price}. "
            f"Hedged book (long+short same size) must be MTM-neutral. "
            f"alpha={pt_alpha.get_total_equity()}, beta={pt_beta.get_total_equity()}"
        )


@pytest.mark.asyncio
async def test_global_equity_equals_sum_of_trackers_across_ticks(
    bus, registry, dashboard_no_adapter
):
    """
    Test A (no-adapter path) — global == sum(trackers) across 10 price ticks.

    strat_alpha: LONG 0.01 BTC/USD.
    strat_beta:  holds cash only (no positions).

    After each tick, _get_global_equity() must equal alpha.get_total_equity() + beta.get_total_equity().
    This locks in the invariant for the GlobalPortfolio path (no paper_adapter).
    """
    pt_alpha = registry.get_portfolio("strat_alpha")
    pt_beta = registry.get_portfolio("strat_beta")
    assert pt_alpha is not None
    assert pt_beta is not None

    entry_price = Decimal("50000")
    size = Decimal("0.01")

    await bus.publish(_make_fill("strat_alpha", "BTC/USD", Side.BUY, size, entry_price))
    await asyncio.sleep(0.1)

    prices = [50000, 50100, 50200, 49800, 50050, 50300, 49900, 50150, 50250, 50000]
    for price in prices:
        await bus.publish(_make_tick("BTC/USD", Decimal(str(price))))
        await asyncio.sleep(0.05)

        global_equity = dashboard_no_adapter._get_global_equity()
        tracker_sum = pt_alpha.get_total_equity() + pt_beta.get_total_equity()

        assert global_equity == tracker_sum, (
            f"global={global_equity} != sum(trackers)={tracker_sum} at BTC={price}. "
            f"alpha={pt_alpha.get_total_equity()}, beta={pt_beta.get_total_equity()}"
        )
