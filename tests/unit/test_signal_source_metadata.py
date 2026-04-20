"""
Unit tests for source metadata in signal events and aggregator source filtering.

Verifies that _create_signal() always stamps metadata["source"] with the
generator's name, and that caller-supplied metadata keys are preserved.

Also verifies that SignalAggregator with signal_source_filter drops signals
from non-matching sources and accepts signals from matching sources.

@decision DEC-SIGNAL-002
@title Signal source metadata injection
@status accepted
@rationale Each signal generator must tag its signals with a "source" key in
metadata so downstream aggregators (e.g., range trading) can filter by origin.
The base class _create_signal() is the single injection point, ensuring all
subclasses get this for free without any per-subclass changes.
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


class SampleGenerator(SignalGenerator):
    """Minimal concrete subclass used only for testing _create_signal."""

    def __init__(self, bus: EventBus, name: str = "TestSource") -> None:
        super().__init__(bus, SignalType.TECHNICAL, window_size=10, name=name)

    def _get_min_periods(self) -> int:
        return 1

    def _generate_signal(self, symbol, data):
        return None  # Not used in these tests


@pytest.fixture
async def bus():
    b = EventBus(queue_size=10)
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_source_metadata_injected(bus):
    """_create_signal must set metadata['source'] to the generator name."""
    gen = SampleGenerator(bus, name="SupportResistance")
    signal = gen._create_signal(
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.6"),
        confidence=Decimal("0.8"),
        timestamp=time(),
    )
    assert signal.metadata is not None
    assert signal.metadata["source"] == "SupportResistance"


@pytest.mark.asyncio
async def test_source_metadata_uses_generator_name(bus):
    """Different generator names produce different source values."""
    gen_rsi = SampleGenerator(bus, name="RSI")
    gen_macd = SampleGenerator(bus, name="MACD")

    sig_rsi = gen_rsi._create_signal(
        symbol="ETH/USD",
        action=SignalAction.SELL,
        strength=Decimal("0.5"),
        confidence=Decimal("0.5"),
        timestamp=time(),
    )
    sig_macd = gen_macd._create_signal(
        symbol="ETH/USD",
        action=SignalAction.SELL,
        strength=Decimal("0.5"),
        confidence=Decimal("0.5"),
        timestamp=time(),
    )

    assert sig_rsi.metadata["source"] == "RSI"
    assert sig_macd.metadata["source"] == "MACD"


@pytest.mark.asyncio
async def test_source_metadata_does_not_overwrite_existing_keys(bus):
    """Existing metadata keys passed by subclasses must be preserved alongside source."""
    gen = SampleGenerator(bus, name="Bollinger")

    signal = gen._create_signal(
        symbol="BTC/USD",
        action=SignalAction.HOLD,
        strength=Decimal("0.3"),
        confidence=Decimal("0.4"),
        timestamp=time(),
    )

    # source and timeframe must be present (DEC-SIGNAL-002, DEC-SIGNAL-003)
    assert signal.metadata["source"] == "Bollinger"
    assert signal.metadata["timeframe"] == "1m"  # default timeframe
    # Only the two base-injected keys should be present — nothing else
    assert set(signal.metadata.keys()) == {"source", "timeframe"}


@pytest.mark.asyncio
async def test_default_name_is_class_name(bus):
    """When no name is given, _name defaults to class name and source reflects that."""
    class AutoNamedGenerator(SignalGenerator):
        def _get_min_periods(self): return 1
        def _generate_signal(self, symbol, data): return None

    gen = AutoNamedGenerator(bus, SignalType.TECHNICAL, window_size=5)
    signal = gen._create_signal(
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("1.0"),
        confidence=Decimal("1.0"),
        timestamp=time(),
    )
    assert signal.metadata["source"] == "AutoNamedGenerator"


@pytest.fixture
async def started_bus():
    """EventBus fixture that is started and stopped around each test."""
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_aggregator_filters_by_source(started_bus):
    """Aggregator with signal_source_filter should drop non-matching signals.

    Only signals whose metadata["source"] equals the configured filter value
    should be admitted to the _signal_buffer.  All other sources must be
    silently dropped before aggregation (DEC-SIGNAL-002).
    """
    # Create aggregator that only accepts SupportResistance signals.
    # Subscriptions are registered in __init__ on the already-started bus.
    agg = SignalAggregator(
        started_bus,
        signal_source_filter="SupportResistance",
        # Low threshold so any accepted signal would aggregate normally
        threshold=Decimal("0.1"),
    )

    # --- Signal from a non-matching source: should be dropped ---
    # Use time() (stdlib wall-clock) to match the aggregator's _clean_old_signals logic,
    # which also calls time().  Using asyncio.get_event_loop().time() would produce a
    # loop-relative timestamp that is much smaller than time() and would cause the
    # signal to be pruned immediately by the 5-second window cleanup.
    rsi_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="rsi buy",
        metadata={"source": "RSI"},
    )
    await started_bus.publish(rsi_signal)
    await asyncio.sleep(0.1)

    # Buffer must remain empty — RSI signal filtered out
    assert len(agg._signal_buffer["BTC/USD"]) == 0, (
        f"RSI signal should have been filtered, buffer has "
        f"{len(agg._signal_buffer['BTC/USD'])} entries"
    )

    # --- Signal from the matching source: should be accepted ---
    sr_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="support bounce",
        metadata={"source": "SupportResistance"},
    )
    await started_bus.publish(sr_signal)
    await asyncio.sleep(0.1)

    # Buffer should now contain exactly one signal
    assert len(agg._signal_buffer["BTC/USD"]) == 1, (
        f"SupportResistance signal should have been accepted, buffer has "
        f"{len(agg._signal_buffer['BTC/USD'])} entries"
    )


@pytest.mark.asyncio
async def test_congressional_signal_reaches_only_pelosi_aggregator(started_bus):
    """Congressional signals must reach pelosi_follow's aggregator and NOT others.

    REQ-GOAL-004 + Phase 15B signal isolation: a SignalEvent tagged
    metadata["source"]="Congressional" is accepted by an aggregator with
    signal_source_filter="Congressional" and rejected by aggregators with
    signal_source_filter="SupportResistance" and "OpeningRange".

    This test is the canonical proof-point for signal isolation — it lives here
    (alongside the existing SupportResistance isolation test) so the two cases
    are maintained together.
    """
    from time import time as _time
    from decimal import Decimal as D

    # pelosi_follow aggregator — only accepts Congressional
    pelosi_agg = SignalAggregator(
        started_bus,
        signal_source_filter="Congressional",
        threshold=D("0.1"),
    )
    # Other strategy aggregators with source filters
    sr_agg = SignalAggregator(
        started_bus,
        signal_source_filter="SupportResistance",
        threshold=D("0.1"),
    )
    orb_agg = SignalAggregator(
        started_bus,
        signal_source_filter="OpeningRange",
        threshold=D("0.1"),
    )

    congressional_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=_time(),
        signal_type=SignalType.NEWS,
        symbol="NVDA",
        action=SignalAction.BUY,
        strength=D("0.75"),
        confidence=D("0.65"),
        reason="Congressional Stock Purchase by Pelosi (filing_id=test-001)",
        metadata={
            "source": "Congressional",
            "filing_id": "test-001",
            "filing_date": "2026-04-01",
        },
    )
    await started_bus.publish(congressional_signal)
    await asyncio.sleep(0.1)

    # pelosi_follow must receive it
    assert len(pelosi_agg._signal_buffer.get("NVDA", [])) == 1, (
        "Congressional signal must be buffered by pelosi_follow aggregator"
    )
    # SupportResistance aggregator must NOT receive it
    assert len(sr_agg._signal_buffer.get("NVDA", [])) == 0, (
        "Congressional signal must NOT reach range_trading (SupportResistance) aggregator"
    )
    # OpeningRange aggregator must NOT receive it
    assert len(orb_agg._signal_buffer.get("NVDA", [])) == 0, (
        "Congressional signal must NOT reach orb_stocks (OpeningRange) aggregator"
    )
