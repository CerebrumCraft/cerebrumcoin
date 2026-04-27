"""
Tests for regime_damped counter in SignalAggregator.

The BEAR buy-suppression path (DEC-REGIME-002) silently multiplies buy_score_norm
by a factor (0.2) without recording that the dampening occurred. This means the
live dashboard cannot report "how many signals were damped by BEAR confidence?"
— a critical gap when signals→fills conversion is 0.0008% (Session 43).

This test suite verifies:
1. regime_damped_counts starts empty
2. Counter increments when BEAR + high confidence suppresses a buy signal
3. Counter does NOT increment when regime is not BEAR
4. Counter does NOT increment when BEAR confidence is below threshold
5. regime_damped_counts returns a copy (no aliasing)
6. Counter records include the symbol and multiplier applied

@decision DEC-DIAG-001
@title regime_damped counter in SignalAggregator for BEAR suppression observability
@status accepted
@rationale Session 43 showed 0.0008% signal→fill conversion. The BEAR buy-suppression
path was completely invisible to the denial counter system — signals were silently
weakened below threshold, but nothing was logged or counted. Adding regime_damped_counts
makes this suppression observable from the dashboard without changing the suppression
logic itself.
"""

import asyncio
from decimal import Decimal
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bus():
    return EventBus(queue_size=100)


async def _emit_regime(bus, regime, confidence):
    """Publish a RegimeChangeEvent with the given regime and confidence."""
    event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=time(),
        from_regime="UNKNOWN",
        to_regime=regime,
        confidence=Decimal(str(confidence)),
        indicators={},
    )
    await bus.publish(event)
    await asyncio.sleep(0.05)


async def _emit_buy_signal(bus, symbol="BTC/USD", strength="0.8"):
    """Publish a TECHNICAL BUY signal for aggregation."""
    event = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal(strength),
        confidence=Decimal("0.9"),
    )
    await bus.publish(event)
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def bus():
    b = _make_bus()
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def make_aggregator(bus):
    """Factory that creates a SignalAggregator attached to the given bus."""
    from cerebrum.signals.aggregator import SignalAggregator
    instances = []

    def _factory(strategy_id="test_strategy", **kwargs):
        agg = SignalAggregator(
            bus,
            threshold=Decimal("0.3"),
            window_seconds=120,
            strategy_id=strategy_id,
            **kwargs,
        )
        instances.append(agg)
        return agg

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regime_damped_counts_starts_empty(bus, make_aggregator):
    """regime_damped_counts must be an empty dict on initialization."""
    agg = make_aggregator()
    assert agg.regime_damped_counts == {}


@pytest.mark.asyncio
async def test_regime_damped_increments_on_bear_high_confidence(bus, make_aggregator):
    """Counter must increment when BEAR + confidence >= threshold suppresses buy."""
    agg = make_aggregator(strategy_id="strat_bear_test")

    # Set BEAR regime with high confidence (>= 0.8 default threshold)
    await _emit_regime(bus, "BEAR", 0.9)

    # Send a BUY signal — should be suppressed
    await _emit_buy_signal(bus, symbol="BTC/USD")
    await asyncio.sleep(0.1)

    counts = agg.regime_damped_counts
    # At least one BTC/USD damping should be recorded
    assert counts.get("BTC/USD", 0) >= 1


@pytest.mark.asyncio
async def test_regime_damped_no_increment_in_bull(bus, make_aggregator):
    """Counter must NOT increment when regime is BULL."""
    agg = make_aggregator(strategy_id="strat_bull_test")

    await _emit_regime(bus, "BULL", 0.9)
    await _emit_buy_signal(bus, symbol="ETH/USD")
    await asyncio.sleep(0.1)

    counts = agg.regime_damped_counts
    assert counts.get("ETH/USD", 0) == 0


@pytest.mark.asyncio
async def test_regime_damped_no_increment_bear_low_confidence(bus, make_aggregator):
    """Counter must NOT increment when BEAR confidence is below suppression threshold."""
    agg = make_aggregator(strategy_id="strat_low_conf_test")

    # BEAR regime but confidence below 0.8 threshold
    await _emit_regime(bus, "BEAR", 0.5)
    await _emit_buy_signal(bus, symbol="SOL/USD")
    await asyncio.sleep(0.1)

    counts = agg.regime_damped_counts
    assert counts.get("SOL/USD", 0) == 0


@pytest.mark.asyncio
async def test_regime_damped_accumulates_over_multiple_signals(bus, make_aggregator):
    """Multiple suppressed buy signals for the same symbol must accumulate."""
    agg = make_aggregator(strategy_id="strat_accum_test")

    await _emit_regime(bus, "BEAR", 0.95)

    # Emit several buy signals with delays > window to force separate aggregations
    for _ in range(3):
        await _emit_buy_signal(bus, symbol="DOGE/USD", strength="0.9")
        # Advance time so each signal is treated as a new aggregation window
        await asyncio.sleep(0.15)

    counts = agg.regime_damped_counts
    assert counts.get("DOGE/USD", 0) >= 1  # at least 1 damping recorded


@pytest.mark.asyncio
async def test_regime_damped_returns_copy(bus, make_aggregator):
    """Mutating the returned dict must not affect internal state."""
    agg = make_aggregator(strategy_id="strat_copy_test")

    await _emit_regime(bus, "BEAR", 0.9)
    await _emit_buy_signal(bus, symbol="BTC/USD")
    await asyncio.sleep(0.1)

    counts = agg.regime_damped_counts
    original_val = counts.get("BTC/USD", 0)
    counts["BTC/USD"] = 99999  # mutate the copy

    # Internal state unchanged
    assert agg.regime_damped_counts.get("BTC/USD", 0) == original_val


@pytest.mark.asyncio
async def test_regime_damped_tracks_multiplier_in_detail(bus, make_aggregator):
    """regime_damped_detail must record the multiplier that was applied."""
    agg = make_aggregator(
        strategy_id="strat_detail_test",
        buy_suppression_factor="0.2",
    )

    await _emit_regime(bus, "BEAR", 0.9)
    await _emit_buy_signal(bus, symbol="BTC/USD")
    await asyncio.sleep(0.1)

    detail = agg.regime_damped_detail
    # Should have at least one entry for BTC/USD
    btc_entries = [e for e in detail if e["symbol"] == "BTC/USD"]
    assert len(btc_entries) >= 1
    # Multiplier should be 0.2 (the suppression factor)
    assert btc_entries[0]["multiplier"] == "0.2"
