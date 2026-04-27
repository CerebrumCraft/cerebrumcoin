"""
Tests asserting mean_reversion ETH-only symbol gate behavior (DEC-DIAG-004).

Session 43 showed mean_reversion produced fills only for ETH/USD despite
BTC/USD, SOL/USD, and DOGE/USD sending signals. This test suite makes that
behavior explicit and machine-verifiable so future engineers know exactly
which gate is responsible before making any changes.

The gate chain:
    SignalAggregator._symbol_allowed(symbol) → False for non-ETH
    → signal dropped before aggregation
    → RiskManager never sees the signal
    → no order emitted

Tests:
1. MEAN_REVERSION_CONFIG.symbols == ["ETH/USD"] (assertion on the data)
2. SignalAggregator with ETH-only filter: ETH signals aggregate, others drop
3. BTC/USD signal to mean_reversion aggregator is silently dropped
4. SOL/USD signal to mean_reversion aggregator is silently dropped
5. DOGE/USD signal to mean_reversion aggregator is silently dropped
6. Commission-floor math: $3,333 × 7% × 0.5 strength = $116.65 > $100 min
7. min_signal_strength=0.5 in risk_overrides (documents the second gate)

@decision DEC-DIAG-004
@title mean_reversion ETH-only symbol gate — behavior tests
@status accepted
@rationale These tests document the CURRENT gating behavior so the next
engineer modifying mean_reversion knows exactly what will change when they
add a new symbol. Failure = the gate was changed without updating this test,
which is intentional: the tests serve as a change-awareness tripwire.
"""

import asyncio
from decimal import Decimal
from pathlib import Path
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(symbol: str, strength: str = "0.8") -> SignalEvent:
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal(strength),
        confidence=Decimal("0.9"),
    )


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Tests: config assertions (documentation as code)
# ---------------------------------------------------------------------------

def test_mean_reversion_symbols_is_eth_only():
    """MEAN_REVERSION_CONFIG.symbols must be exactly ['ETH/USD'].

    This is the primary gate that explains why BTC/SOL/DOGE never trade
    in mean_reversion (Session 43 observation). Changing this list will
    intentionally break this test — update DEC-DIAG-004 rationale too.
    """
    assert MEAN_REVERSION_CONFIG.symbols == ["ETH/USD"], (
        "mean_reversion is intentionally ETH-only (DEC-TUNE-013). "
        "If you're adding symbols, update this test AND verify commission-floor math."
    )


def test_mean_reversion_min_signal_strength():
    """min_signal_strength=0.5 is the second gate (after symbol filter).

    Documents that even if a non-ETH symbol somehow reached the RiskManager,
    it would also need strength >= 0.5 to pass. This gate is enforced by
    the MinSignalStrengthRule configured from risk_overrides.
    """
    assert MEAN_REVERSION_CONFIG.risk_overrides["min_signal_strength"] == "0.5"


def test_mean_reversion_position_size_percent():
    """position_size_percent=7.0 must be in risk_overrides (DEC-TUNE-017).

    The commission-floor math depends on this value. If it drops back to 5.0,
    trades at 0.5 strength × $3,333 pool × 5% = $83.33 — below the $100 floor.
    """
    assert MEAN_REVERSION_CONFIG.risk_overrides["position_size_percent"] == "7.0"


def test_commission_floor_math_passes_for_eth():
    """$3,333 × 7% × 0.5 strength = $116.65 — above the $100 commission floor.

    This test encodes the math so a future change to pool_usd, N (strategy count),
    or position_size_percent can be cross-checked here.
    """
    pool_usd = Decimal("10000")
    n_strategies = 3
    per_strategy = pool_usd / n_strategies  # $3,333.33...
    position_size_pct = Decimal(MEAN_REVERSION_CONFIG.risk_overrides["position_size_percent"]) / 100
    min_signal_strength = Decimal(MEAN_REVERSION_CONFIG.risk_overrides["min_signal_strength"])
    min_trade_value_floor = Decimal("100")

    trade_value = per_strategy * position_size_pct * min_signal_strength
    assert trade_value > min_trade_value_floor, (
        f"Trade value ${trade_value:.2f} must exceed ${min_trade_value_floor} commission floor. "
        f"Adjust position_size_percent or pool allocation."
    )


# ---------------------------------------------------------------------------
# Tests: aggregator symbol filter behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_eth_signal_reaches_aggregator(bus):
    """ETH/USD signal must be admitted by the mean_reversion aggregator."""
    from cerebrum.signals.aggregator import SignalAggregator

    emitted: list[SignalEvent] = []

    async def capture(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    bus.subscribe(EventType.SIGNAL, capture, subscriber_name="test_capture_eth")

    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=120,
        strategy_id="mean_reversion",
        symbols=MEAN_REVERSION_CONFIG.symbols,  # ["ETH/USD"]
    )

    await bus.publish(_make_signal("ETH/USD", strength="0.9"))
    await asyncio.sleep(0.2)

    eth_combined = [e for e in emitted if e.symbol == "ETH/USD" and e.strategy_id == "mean_reversion"]
    assert len(eth_combined) >= 1, "ETH/USD signal must produce a COMBINED signal from mean_reversion aggregator"


@pytest.mark.asyncio
async def test_btc_signal_dropped_by_aggregator(bus):
    """BTC/USD signal must be silently dropped by mean_reversion aggregator."""
    from cerebrum.signals.aggregator import SignalAggregator

    emitted: list[SignalEvent] = []

    async def capture(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    bus.subscribe(EventType.SIGNAL, capture, subscriber_name="test_capture_btc")

    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=120,
        strategy_id="mean_reversion",
        symbols=MEAN_REVERSION_CONFIG.symbols,  # ["ETH/USD"] — BTC not in list
    )

    await bus.publish(_make_signal("BTC/USD", strength="0.99"))
    await asyncio.sleep(0.2)

    btc_combined = [e for e in emitted if e.symbol == "BTC/USD" and e.strategy_id == "mean_reversion"]
    assert len(btc_combined) == 0, (
        "BTC/USD signal must be dropped by mean_reversion aggregator (DEC-TUNE-013). "
        "To trade BTC, add 'BTC/USD' to MEAN_REVERSION_CONFIG.symbols."
    )


@pytest.mark.asyncio
async def test_sol_signal_dropped_by_aggregator(bus):
    """SOL/USD signal must be silently dropped by mean_reversion aggregator."""
    from cerebrum.signals.aggregator import SignalAggregator

    emitted: list[SignalEvent] = []

    async def capture(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    bus.subscribe(EventType.SIGNAL, capture, subscriber_name="test_capture_sol")

    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=120,
        strategy_id="mean_reversion",
        symbols=MEAN_REVERSION_CONFIG.symbols,
    )

    await bus.publish(_make_signal("SOL/USD", strength="0.99"))
    await asyncio.sleep(0.2)

    sol_combined = [e for e in emitted if e.symbol == "SOL/USD"]
    assert len(sol_combined) == 0, "SOL/USD must be dropped (removed in DEC-TUNE-013: 0% WR, -$28.90)"


@pytest.mark.asyncio
async def test_doge_signal_dropped_by_aggregator(bus):
    """DOGE/USD signal must be silently dropped by mean_reversion aggregator."""
    from cerebrum.signals.aggregator import SignalAggregator

    emitted: list[SignalEvent] = []

    async def capture(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            emitted.append(event)

    bus.subscribe(EventType.SIGNAL, capture, subscriber_name="test_capture_doge")

    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=120,
        strategy_id="mean_reversion",
        symbols=MEAN_REVERSION_CONFIG.symbols,
    )

    await bus.publish(_make_signal("DOGE/USD", strength="0.99"))
    await asyncio.sleep(0.2)

    doge_combined = [e for e in emitted if e.symbol == "DOGE/USD"]
    assert len(doge_combined) == 0, "DOGE/USD was never added to mean_reversion symbol list"
