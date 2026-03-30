"""
Unit tests for ExitMonitor and related improvements.

Tests cover:
1. ExitMonitor stop-loss triggering
2. ExitMonitor take-profit triggering
3. ExitMonitor time-based exit
4. No duplicate exit orders (pending_exits guard)
5. Aggregator consensus multiplier rewards agreement
6. VWAP neutral zone suppresses near-VWAP signals
7. ExitMonitor emits OrderEvent with correct strategy_id (DEC-EXIT-003)
8. Regression: pending_exits stays set while position > 0 (DEC-EXIT-004)

@decision DEC-TEST-009
@title Test exit monitor with real EventBus and PortfolioTracker
@status accepted
@rationale ExitMonitor interacts with EventBus and PortfolioTracker via events.
Using real implementations validates the full event flow without mocks.
"""

import asyncio
from decimal import Decimal
from time import time
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, MarketDataEvent, OrderEvent
from cerebrum.core.types import EventType, OrderType, Side, SignalAction, SignalType
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.portfolio import PortfolioTracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def bus():
    """Create and start event bus."""
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def portfolio(bus):
    """Create portfolio tracker with $10k balance."""
    return PortfolioTracker(bus, initial_balance=Decimal("10000.0"))


def _make_exit_monitor(bus, portfolio, stop_loss=Decimal("2.0"), take_profit=Decimal("3.0"), max_age_minutes=120):
    return ExitMonitor(
        bus,
        portfolio,
        stop_loss_percent=stop_loss,
        take_profit_percent=take_profit,
        max_position_age_minutes=max_age_minutes,
    )


async def _open_position(bus, symbol: str, amount: Decimal, price: Decimal, entry_time: float | None = None) -> None:
    """Helper: send a BUY fill to open a position."""
    ts = entry_time if entry_time is not None else time()
    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=ts,
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        filled_amount=amount,
        fill_price=price,
        commission=Decimal("0.0"),
        commission_asset="USD",
    )
    await bus.publish(fill)
    await asyncio.sleep(0.15)


async def _update_price(bus, symbol: str, price: Decimal, ts: float | None = None) -> None:
    """Helper: emit a MarketDataEvent to update position price."""
    md = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=ts if ts is not None else time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )
    await bus.publish(md)
    await asyncio.sleep(0.15)


# ---------------------------------------------------------------------------
# ExitMonitor tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exit_monitor_stop_loss(bus, portfolio):
    """Stop-loss: price drop beyond threshold emits a SELL order."""
    monitor = _make_exit_monitor(bus, portfolio, stop_loss=Decimal("2.0"))

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    entry_price = Decimal("50000")
    await _open_position(bus, symbol, Decimal("0.1"), entry_price)

    # Drop price 3% below entry — should trigger 2% stop-loss
    drop_price = entry_price * Decimal("0.97")  # -3%
    await _update_price(bus, symbol, drop_price)

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    order = orders[0]
    assert order.side == Side.SELL
    assert order.symbol == symbol
    assert order.order_type == OrderType.STOP_LOSS
    assert order.amount == Decimal("0.1")
    assert order.metadata is not None
    assert "stop_loss" in order.metadata.get("exit_reason", "")


@pytest.mark.asyncio
async def test_exit_monitor_take_profit(bus, portfolio):
    """Take-profit: price rise beyond threshold emits a SELL order."""
    monitor = _make_exit_monitor(bus, portfolio, take_profit=Decimal("3.0"))

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "ETH/USD"
    entry_price = Decimal("3000")
    await _open_position(bus, symbol, Decimal("1.0"), entry_price)

    # Rise price 4% above entry — should trigger 3% take-profit
    rise_price = entry_price * Decimal("1.04")  # +4%
    await _update_price(bus, symbol, rise_price)

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    order = orders[0]
    assert order.side == Side.SELL
    assert order.symbol == symbol
    assert order.order_type == OrderType.TAKE_PROFIT
    assert "take_profit" in order.metadata.get("exit_reason", "")


@pytest.mark.asyncio
async def test_exit_monitor_time_based(bus, portfolio):
    """Time-based exit: old position triggers exit on next market data tick."""
    # Set a very short max age (1 minute) and back-date the entry
    monitor = _make_exit_monitor(bus, portfolio, max_age_minutes=1)

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    # Entry time 10 minutes in the past — will exceed 1-minute threshold
    old_entry_time = time() - 600
    await _open_position(bus, symbol, Decimal("0.1"), Decimal("50000"), entry_time=old_entry_time)

    # Send a market data tick at current time — should trigger time exit
    await _update_price(bus, symbol, Decimal("50100"))

    assert len(orders) == 1, f"Expected 1 time-exit order, got {len(orders)}"
    order = orders[0]
    assert order.side == Side.SELL
    assert "time_exit" in order.metadata.get("exit_reason", "")


@pytest.mark.asyncio
async def test_exit_monitor_no_duplicate_orders(bus, portfolio):
    """Once an exit is triggered, subsequent ticks must not emit another order."""
    monitor = _make_exit_monitor(bus, portfolio, stop_loss=Decimal("2.0"))

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    await _open_position(bus, symbol, Decimal("0.1"), Decimal("50000"))

    # First tick — triggers stop-loss
    await _update_price(bus, symbol, Decimal("48000"))  # -4%
    # Second tick — should NOT emit another order (pending_exits guard)
    await _update_price(bus, symbol, Decimal("47000"))  # even lower

    assert len(orders) == 1, f"Expected exactly 1 exit order, got {len(orders)}"


@pytest.mark.asyncio
async def test_exit_monitor_no_exit_within_thresholds(bus, portfolio):
    """Price movement within thresholds must not trigger any exit."""
    monitor = _make_exit_monitor(
        bus, portfolio,
        stop_loss=Decimal("2.0"),
        take_profit=Decimal("3.0"),
        max_age_minutes=120,
    )

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    entry_price = Decimal("50000")
    await _open_position(bus, symbol, Decimal("0.1"), entry_price)

    # 1% drop — below 2% stop-loss threshold
    await _update_price(bus, symbol, entry_price * Decimal("0.99"))
    # 2% gain — below 3% take-profit threshold
    await _update_price(bus, symbol, entry_price * Decimal("1.02"))

    assert len(orders) == 0, f"Expected no orders within thresholds, got {len(orders)}"


# ---------------------------------------------------------------------------
# strategy_id routing tests (DEC-EXIT-003 / DEC-EXIT-004)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_monitor_emits_order_with_strategy_id(bus, portfolio):
    """ExitMonitor with strategy_id tags emitted OrderEvents for per-strategy routing.

    DEC-EXIT-003: In multi-strategy mode, exit orders must carry strategy_id
    so the paper adapter propagates a tagged FillEvent that reaches the correct
    per-strategy PortfolioTracker.
    """
    monitor = ExitMonitor(
        bus,
        portfolio,
        stop_loss_percent=Decimal("2.0"),
        take_profit_percent=Decimal("3.0"),
        max_position_age_minutes=120,
        strategy_id="momentum",
    )

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    entry_price = Decimal("50000")
    await _open_position(bus, symbol, Decimal("0.1"), entry_price)

    # Drop price beyond stop-loss to trigger exit
    await _update_price(bus, symbol, entry_price * Decimal("0.97"))

    assert len(orders) == 1, f"Expected 1 SELL order, got {len(orders)}"
    assert orders[0].strategy_id == "momentum", (
        f"Expected strategy_id='momentum', got {orders[0].strategy_id!r}"
    )


@pytest.mark.asyncio
async def test_exit_monitor_strategy_id_none_by_default(bus, portfolio):
    """ExitMonitor without strategy_id emits OrderEvents with strategy_id=None.

    Backward compatibility: single-strategy path passes no strategy_id.
    The emitted order must have strategy_id=None so the paper adapter accepts
    it and all downstream components handle it as before.
    """
    monitor = _make_exit_monitor(bus, portfolio, stop_loss=Decimal("2.0"))

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    await _open_position(bus, symbol, Decimal("0.1"), Decimal("50000"))
    await _update_price(bus, symbol, Decimal("48000"))  # -4%, triggers stop-loss

    assert len(orders) == 1
    assert orders[0].strategy_id is None, (
        f"Expected strategy_id=None for single-strategy mode, got {orders[0].strategy_id!r}"
    )


@pytest.mark.asyncio
async def test_exit_monitor_pending_stays_set_while_position_open(bus):
    """Regression: pending_exits must stay set if position is still open after fill.

    DEC-EXIT-004: Simulates the infinite exit loop scenario. ExitMonitor emits
    a SELL, we send a SELL fill but the portfolio still shows an open position
    (e.g., partial fill or multi-strategy mismatch). The pending flag must stay
    set so no second exit order is fired on the next market data tick.

    Steps:
    1. Open a position (BUY fill, 0.2 BTC)
    2. Trigger stop-loss → exit order emitted, symbol added to pending_exits
    3. Send a partial SELL fill (0.1 BTC — position still has 0.1 BTC remaining)
    4. Verify pending_exits still contains the symbol
    5. Send a second market data tick → verify NO second order is emitted
    """
    portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
    monitor = ExitMonitor(
        bus,
        portfolio,
        stop_loss_percent=Decimal("2.0"),
        take_profit_percent=Decimal("3.0"),
        max_position_age_minutes=120,
        strategy_id="test_strategy",
    )

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    entry_price = Decimal("50000")
    amount = Decimal("0.2")

    # Step 1: open position with 0.2 BTC
    await _open_position(bus, symbol, amount, entry_price)

    # Step 2: trigger stop-loss
    drop_price = entry_price * Decimal("0.97")  # -3%, exceeds 2% stop-loss
    await _update_price(bus, symbol, drop_price)

    assert len(orders) == 1, "Expected exactly 1 exit order after stop-loss trigger"
    assert symbol in monitor._pending_exits, "Symbol should be in pending_exits after trigger"

    # Step 3: simulate a PARTIAL SELL fill (only 0.1 of 0.2 BTC filled)
    # Portfolio processes this fill first (subscribed before ExitMonitor),
    # leaving 0.1 BTC remaining in position.
    partial_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.SELL,
        filled_amount=Decimal("0.1"),  # partial — position not gone
        fill_price=drop_price,
        commission=Decimal("0.0"),
        commission_asset="USD",
        strategy_id="test_strategy",
    )
    await bus.publish(partial_fill)
    await asyncio.sleep(0.15)

    # Step 4: pending_exits must still be set (position is not fully closed)
    assert symbol in monitor._pending_exits, (
        "pending_exits must stay set while position amount > 0 (partial fill)"
    )

    # Step 5: another market data tick must not emit a second exit order
    await _update_price(bus, symbol, drop_price * Decimal("0.99"))

    assert len(orders) == 1, (
        f"Expected exactly 1 exit order total — got {len(orders)} "
        "(infinite exit loop regression: second order fired after partial fill)"
    )


@pytest.mark.asyncio
async def test_exit_monitor_pending_cleared_when_position_fully_closed(bus):
    """After position reaches zero, pending_exits is cleared and re-arming is possible.

    DEC-EXIT-004: Verifies the happy path — a full SELL fill closes the position,
    pending_exits is cleared, and a brand-new position can trigger a fresh exit.
    """
    portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
    monitor = ExitMonitor(
        bus,
        portfolio,
        stop_loss_percent=Decimal("2.0"),
        take_profit_percent=Decimal("3.0"),
        max_position_age_minutes=120,
        strategy_id="test_strategy",
    )

    orders: list[OrderEvent] = []

    async def collect(event):
        if isinstance(event, OrderEvent):
            orders.append(event)

    bus.subscribe(EventType.ORDER, collect, "order_collector")

    symbol = "BTC/USD"
    entry_price = Decimal("50000")
    amount = Decimal("0.1")

    # Open position and trigger stop-loss
    await _open_position(bus, symbol, amount, entry_price)
    drop_price = entry_price * Decimal("0.97")
    await _update_price(bus, symbol, drop_price)
    assert len(orders) == 1

    # Full SELL fill — closes the position entirely
    full_fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.SELL,
        filled_amount=amount,
        fill_price=drop_price,
        commission=Decimal("0.0"),
        commission_asset="USD",
        strategy_id="test_strategy",
    )
    await bus.publish(full_fill)
    await asyncio.sleep(0.15)

    # pending_exits should be cleared once position reaches zero
    assert symbol not in monitor._pending_exits, (
        "pending_exits should be cleared after full position closure"
    )


# ---------------------------------------------------------------------------
# Aggregator consensus multiplier tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aggregator_consensus_rewards_agreement():
    """4 weak agreeing BUY signals at 0.3 should outscore a split (2 BUY, 2 SELL at 0.3).

    With the consensus multiplier:
    - All-BUY consensus = sqrt(1.0) = 1.0x → score = 0.3
    - Split 50/50 consensus = sqrt(0.5) ≈ 0.707x → score ≈ 0.212 each direction

    We test _aggregate_signals directly by pre-populating the internal buffer,
    since the method reads from self._signal_buffer[symbol].
    """
    from collections import defaultdict
    from cerebrum.core.events import SignalEvent
    from cerebrum.signals.aggregator import SignalAggregator

    b = EventBus(queue_size=50)
    await b.start()

    agg = SignalAggregator(b, threshold=Decimal("0.1"), window_seconds=60)

    now = time()

    # --- Case 1: 4 agreeing BUY signals ---
    agreeing = [
        SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=now - i,
            signal_type=SignalType.TECHNICAL,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.3"),
            confidence=Decimal("0.7"),
        )
        for i in range(4)
    ]
    agg._signal_buffer["BTC/USD"] = agreeing
    result_agree = agg._aggregate_signals("BTC/USD", now)
    assert result_agree is not None, "4 agreeing BUY signals should produce a result"
    assert result_agree.action == SignalAction.BUY
    agree_strength = result_agree.strength

    # --- Case 2: 2 BUY + 2 SELL (perfect split) ---
    split = [
        SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=now - i,
            signal_type=SignalType.TECHNICAL,
            symbol="ETH/USD",
            action=SignalAction.BUY if i < 2 else SignalAction.SELL,
            strength=Decimal("0.3"),
            confidence=Decimal("0.7"),
        )
        for i in range(4)
    ]
    agg._signal_buffer["ETH/USD"] = split
    result_split = agg._aggregate_signals("ETH/USD", now)

    # Split signals: each direction suppressed by consensus multiplier sqrt(0.5)≈0.707
    # Agreeing signals: full multiplier sqrt(1.0)=1.0 → stronger result
    if result_split is not None and result_split.action in (SignalAction.BUY, SignalAction.SELL):
        assert result_split.strength < agree_strength, (
            f"Split score ({result_split.strength}) should be weaker than "
            f"all-agree score ({agree_strength})"
        )

    await b.stop()


# ---------------------------------------------------------------------------
# VWAP neutral zone tests
# ---------------------------------------------------------------------------

async def _seed_candles(bus, symbol: str, prices: list[float], interval: int = 60) -> None:
    """Publish MarketDataEvents spaced by interval seconds to build candles."""
    # Use a base time far enough in the past that all ticks fall in different candles
    base_ts = time() - len(prices) * interval * 2
    for i, price in enumerate(prices):
        # One tick at the very start of each candle interval
        md = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=base_ts + i * interval + 1,
            symbol=symbol,
            price=Decimal(str(price)),
            volume=Decimal("10.0"),
        )
        await bus.publish(md)
    # Allow the candle aggregator time to process all ticks
    await asyncio.sleep(0.3)


@pytest.mark.asyncio
async def test_vwap_neutral_zone_produces_hold():
    """Price within 0.5% of VWAP should produce HOLD, not BUY or SELL.

    We seed 25 candles all at the same price so VWAP == current price (0% distance),
    which is well inside the 0.5% neutral zone. Signals are collected via bus subscription
    since VWAPSignal is event-driven and publishes results to the bus.
    """
    from cerebrum.core.events import SignalEvent
    from cerebrum.signals.candles import CandleAggregator
    from cerebrum.signals.technical import VWAPSignal

    b = EventBus(queue_size=200)
    await b.start()
    candle_agg = CandleAggregator(b, interval_seconds=60)
    sig = VWAPSignal(b, candle_agg, period=20)

    signals: list[SignalEvent] = []

    async def collect_signal(event):
        if isinstance(event, SignalEvent):
            signals.append(event)

    b.subscribe(EventType.SIGNAL, collect_signal, "test_vwap_collector")

    symbol = "BTC/USD"
    # 25 candles all at 50000 — VWAP = 50000, current = 50000, distance = 0%
    await _seed_candles(b, symbol, [50000.0] * 25)

    # All generated signals should be HOLD (within neutral zone)
    vwap_signals = [s for s in signals if s.symbol == symbol]
    assert len(vwap_signals) > 0, "Expected at least one signal to be emitted"
    for s in vwap_signals:
        assert s.action == SignalAction.HOLD, (
            f"Expected HOLD within neutral zone, got {s.action} strength={s.strength}"
        )
        assert s.strength == Decimal("0.0")

    await b.stop()


@pytest.mark.asyncio
async def test_vwap_outside_neutral_zone_produces_signal():
    """Price >0.5% above VWAP should produce a BUY signal with positive strength.

    Strategy: seed 25 candles (24 at 3000, 1 at 3090) BEFORE subscribing VWAPSignal,
    so candle history is built up. Then send one fresh tick at 3090 AFTER VWAPSignal
    is subscribed — this tick triggers _generate_signal with 25 closed candles where
    the last is 3090, producing a BUY signal.
    """
    from cerebrum.core.events import SignalEvent
    from cerebrum.signals.candles import CandleAggregator
    from cerebrum.signals.technical import VWAPSignal

    b = EventBus(queue_size=200)
    await b.start()
    candle_agg = CandleAggregator(b, interval_seconds=60)

    symbol = "ETH/USD"
    # Seed 25 candles (24 at 3000, 1 at 3090) BEFORE creating VWAPSignal
    # so candle_agg has the history but VWAPSignal hasn't generated any signals yet
    prices = [3000.0] * 24 + [3090.0, 3090.0]  # extra tick closes the 3090 candle
    await _seed_candles(b, symbol, prices)

    # Now create VWAPSignal — it will start receiving future ticks.
    # VWAPSignal's base class requires _get_min_periods() raw ticks before firing,
    # so we must send at least 20 more ticks to fill its data buffer.
    sig = VWAPSignal(b, candle_agg, period=20)

    signals: list[SignalEvent] = []

    async def collect_signal(event):
        if isinstance(event, SignalEvent):
            signals.append(event)

    b.subscribe(EventType.SIGNAL, collect_signal, "test_vwap_collector_buy")

    # Send 22 ticks at 3090 — this fills VWAPSignal's data buffer (min_periods=20)
    # and also keeps adding 3090 candles so VWAP stays well above the neutral zone.
    # Each tick spaced in a new interval so candle_agg produces fresh closed candles.
    now = time()
    for i in range(22):
        md = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=now + i * 60 + 1,
            symbol=symbol,
            price=Decimal("3090"),
            volume=Decimal("10.0"),
        )
        await b.publish(md)
    await asyncio.sleep(0.4)

    eth_signals = [s for s in signals if s.symbol == symbol]
    assert len(eth_signals) > 0, (
        f"Expected at least one signal after filling VWAPSignal data buffer. "
        f"Candle count: {len(candle_agg.get_candles(symbol, count=50))}"
    )

    # At least one BUY signal with positive strength
    buy_signals = [s for s in eth_signals if s.action == SignalAction.BUY]
    assert len(buy_signals) > 0, (
        f"Expected BUY signal above neutral zone. Got: "
        f"{[(s.action.value, str(s.strength)) for s in eth_signals]}"
    )
    assert buy_signals[-1].strength > Decimal("0.0")

    await b.stop()
