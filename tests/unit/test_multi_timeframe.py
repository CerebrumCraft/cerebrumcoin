"""
Unit tests for timeframe metadata in signal events and aggregator timeframe filtering.

Verifies that SignalGenerator stamps metadata["timeframe"] on every signal it
creates, that the default timeframe is "1m", and that SignalAggregator with
signal_timeframe_filter drops signals from non-matching timeframes while
accepting signals from the matching timeframe.

@decision DEC-SIGNAL-003
@title Timeframe metadata injection for multi-timeframe strategies
@status accepted
@rationale Multi-timeframe swing strategies need to run separate generators
on different bar sizes (e.g., 1m scalp vs 1h swing). Injecting
metadata["timeframe"] in _create_signal() at the base level gives every
subclass this for free. The aggregator's signal_timeframe_filter allows
a strategy to accept only the timeframe relevant to its trading logic.
"""

import asyncio
from decimal import Decimal
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.base import SignalGenerator


class TimeframeTestGenerator(SignalGenerator):
    """Minimal concrete subclass for testing timeframe metadata."""

    def __init__(
        self,
        bus: EventBus,
        name: str = "TestGen",
        timeframe: str = "1m",
    ) -> None:
        super().__init__(
            bus,
            SignalType.TECHNICAL,
            window_size=10,
            name=name,
            timeframe=timeframe,
        )

    def _get_min_periods(self) -> int:
        return 1

    def _generate_signal(self, symbol, data):
        return None  # Not used in these tests


@pytest.fixture
async def bus():
    b = EventBus(queue_size=20)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Part 1: SignalGenerator timeframe metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_generator_includes_timeframe_metadata(bus):
    """Generator with timeframe='1h' produces signals with metadata['timeframe']=='1h'."""
    gen = TimeframeTestGenerator(bus, name="SwingRSI", timeframe="1h")
    signal = gen._create_signal(
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.7"),
        confidence=Decimal("0.8"),
        timestamp=time(),
    )
    assert signal.metadata is not None
    assert signal.metadata["timeframe"] == "1h"


@pytest.mark.asyncio
async def test_signal_generator_default_timeframe_is_1m(bus):
    """When no timeframe is given, _timeframe defaults to '1m' and is stamped on signals."""
    gen = TimeframeTestGenerator(bus, name="DefaultGen")
    # No timeframe kwarg — falls back to default "1m"
    assert gen._timeframe == "1m"

    signal = gen._create_signal(
        symbol="ETH/USD",
        action=SignalAction.SELL,
        strength=Decimal("0.5"),
        confidence=Decimal("0.6"),
        timestamp=time(),
    )
    assert signal.metadata is not None
    assert signal.metadata["timeframe"] == "1m"


@pytest.mark.asyncio
async def test_signal_generator_timeframe_and_source_both_present(bus):
    """Both 'source' and 'timeframe' metadata keys must appear on every signal."""
    gen = TimeframeTestGenerator(bus, name="MACD_4h", timeframe="4h")
    signal = gen._create_signal(
        symbol="BTC/USD",
        action=SignalAction.HOLD,
        strength=Decimal("0.3"),
        confidence=Decimal("0.5"),
        timestamp=time(),
    )
    assert signal.metadata is not None
    assert "source" in signal.metadata
    assert "timeframe" in signal.metadata
    assert signal.metadata["source"] == "MACD_4h"
    assert signal.metadata["timeframe"] == "4h"


# ---------------------------------------------------------------------------
# Part 3: SignalAggregator timeframe filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_filters_by_timeframe(bus):
    """Aggregator with signal_timeframe_filter='1h' drops 1m signals, accepts 1h signals."""
    agg = SignalAggregator(
        bus,
        signal_timeframe_filter="1h",
        threshold=Decimal("0.1"),  # low threshold so any accepted signal aggregates
    )

    # --- Signal with non-matching timeframe ("1m") should be dropped ---
    signal_1m = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.9"),
        confidence=Decimal("0.9"),
        reason="1m buy signal",
        metadata={"source": "RSI", "timeframe": "1m"},
    )
    await bus.publish(signal_1m)
    await asyncio.sleep(0.1)

    assert len(agg._signal_buffer["BTC/USD"]) == 0, (
        f"1m signal should have been filtered out; buffer has "
        f"{len(agg._signal_buffer['BTC/USD'])} entries"
    )

    # --- Signal with matching timeframe ("1h") should be accepted ---
    signal_1h = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.9"),
        confidence=Decimal("0.9"),
        reason="1h buy signal",
        metadata={"source": "RSI", "timeframe": "1h"},
    )
    await bus.publish(signal_1h)
    await asyncio.sleep(0.1)

    assert len(agg._signal_buffer["BTC/USD"]) == 1, (
        f"1h signal should have been accepted; buffer has "
        f"{len(agg._signal_buffer['BTC/USD'])} entries"
    )


@pytest.mark.asyncio
async def test_aggregator_no_timeframe_filter_accepts_all(bus):
    """Aggregator without signal_timeframe_filter accepts signals of any timeframe."""
    agg = SignalAggregator(
        bus,
        signal_timeframe_filter=None,
        threshold=Decimal("0.1"),
    )

    for tf in ("1m", "1h", "4h", "1d"):
        sig = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=time(),
            signal_type=SignalType.TECHNICAL,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.5"),
            confidence=Decimal("0.7"),
            metadata={"source": "TestGen", "timeframe": tf},
        )
        await bus.publish(sig)

    await asyncio.sleep(0.1)

    # All 4 signals should be in the buffer
    assert len(agg._signal_buffer["BTC/USD"]) == 4, (
        f"Expected 4 signals (one per timeframe), got "
        f"{len(agg._signal_buffer['BTC/USD'])}"
    )


@pytest.mark.asyncio
async def test_1h_candle_aggregator_independent(bus):
    """1h and 1m CandleAggregators produce candles independently."""
    from cerebrum.signals.candles import CandleAggregator
    from cerebrum.core.events import MarketDataEvent

    agg_1m = CandleAggregator(bus, interval_seconds=60)
    agg_1h = CandleAggregator(bus, interval_seconds=3600)

    # Feed a market data event
    event = MarketDataEvent(
        event_type=None, timestamp=time(),
        symbol="BTC/USD", price=Decimal("70000"),
        bid=Decimal("69999"), ask=Decimal("70001"),
        volume=Decimal("1.0"),
    )
    await bus.publish(event)
    await asyncio.sleep(0.05)

    # Both aggregators maintain separate state (no shared candle data structures)
    assert agg_1m._interval == 60
    assert agg_1h._interval == 3600

    # They are separate objects — not sharing state
    assert agg_1m is not agg_1h
    assert agg_1m._current_candles is not agg_1h._current_candles
    assert agg_1m._completed_candles is not agg_1h._completed_candles

    # Both should have received the event and started tracking the symbol
    candles_1m = agg_1m.get_candles("BTC/USD")
    candles_1h = agg_1h.get_candles("BTC/USD")

    # Completed candles are empty (no interval boundary crossed with a single tick)
    # but current_candles should be populated
    assert len(candles_1m) == 0  # no completed candle yet (no boundary crossing)
    assert len(candles_1h) == 0  # same
    assert agg_1m._current_candles["BTC/USD"] is not None
    assert agg_1h._current_candles["BTC/USD"] is not None


@pytest.mark.asyncio
async def test_aggregator_drops_signal_with_no_metadata_when_timeframe_filter_set(bus):
    """Signal with no metadata is dropped when timeframe filter is active."""
    agg = SignalAggregator(
        bus,
        signal_timeframe_filter="1h",
        threshold=Decimal("0.1"),
    )

    # Signal with metadata=None
    sig_no_meta = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="ETH/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.8"),
        metadata=None,
    )
    await bus.publish(sig_no_meta)
    await asyncio.sleep(0.1)

    assert len(agg._signal_buffer["ETH/USD"]) == 0, (
        "Signal with metadata=None should be dropped when timeframe filter is active"
    )
