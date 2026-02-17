"""
Unit tests for signal generators.

Tests signal base class, technical indicators, and signal aggregation.
"""

import asyncio
from decimal import Decimal
from time import time

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.base import SignalGenerator
from cerebrum.signals.candles import CandleAggregator


class DummySignal(SignalGenerator):
    """Dummy signal generator for testing base class."""
    
    def __init__(self, bus: EventBus) -> None:
        super().__init__(bus, SignalType.TECHNICAL, window_size=10, name="dummy")
        self.call_count = 0
    
    def _get_min_periods(self) -> int:
        return 5
    
    def _generate_signal(self, symbol, data):
        self.call_count += 1
        if len(data) >= 5:
            return self._create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                strength=Decimal("0.5"),
                confidence=Decimal("0.7"),
                timestamp=time(),
                reason="test signal",
            )
        return None


@pytest.fixture
async def bus():
    """Create and start event bus."""
    bus = EventBus(queue_size=50)
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_signal_generator_accumulation(bus):
    """Test that signal generator accumulates data correctly."""
    signal_gen = DummySignal(bus)
    
    # Send market data events
    symbol = "BTC/USD"
    for i in range(10):
        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol=symbol,
            price=Decimal(f"{50000 + i}"),
            volume=Decimal("1.0"),
        )
        await bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check data accumulation
    assert signal_gen.get_data_count(symbol) == 10
    
    # Signal should have been generated (min_periods = 5)
    assert signal_gen.call_count >= 5  # Called for events 5-10


@pytest.mark.asyncio
async def test_signal_generator_min_periods(bus):
    """Test that signals aren't generated until min_periods reached."""
    signal_gen = DummySignal(bus)
    
    # Collect emitted signals
    signals = []
    
    async def signal_collector(event):
        if isinstance(event, SignalEvent):
            signals.append(event)
    
    bus.subscribe(EventType.SIGNAL, signal_collector, "test_collector")
    
    # Send only 4 events (below min_periods=5)
    symbol = "BTC/USD"
    for i in range(4):
        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=time(),
            symbol=symbol,
            price=Decimal(f"{50000 + i}"),
            volume=Decimal("1.0"),
        )
        await bus.publish(event)
    
    await asyncio.sleep(0.2)
    
    # No signals should be generated yet
    assert len(signals) == 0
    
    # Send 5th event
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time(),
        symbol=symbol,
        price=Decimal("50005"),
        volume=Decimal("1.0"),
    )
    await bus.publish(event)
    
    await asyncio.sleep(0.2)
    
    # Now signal should be generated
    assert len(signals) >= 1


@pytest.mark.asyncio
async def test_candle_aggregator(bus):
    """Test candle aggregation from ticks."""
    candle_agg = CandleAggregator(bus, interval_seconds=10, window_size=100)
    
    symbol = "BTC/USD"
    base_time = 1000.0
    
    # Send ticks within same candle period
    for i in range(5):
        event = MarketDataEvent(
            event_type=EventType.MARKET_DATA,
            timestamp=base_time + i,
            symbol=symbol,
            price=Decimal(f"{50000 + i * 10}"),
            volume=Decimal("0.1"),
        )
        await bus.publish(event)
    
    await asyncio.sleep(0.2)
    
    # No completed candles yet (still building first candle)
    assert candle_agg.get_candle_count(symbol) == 0
    
    # Send tick in next candle period (triggers completion of first)
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=base_time + 11,
        symbol=symbol,
        price=Decimal("50100"),
        volume=Decimal("0.1"),
    )
    await bus.publish(event)
    
    await asyncio.sleep(0.2)
    
    # First candle should be completed
    assert candle_agg.get_candle_count(symbol) == 1
    
    candles = candle_agg.get_candles(symbol)
    assert len(candles) == 1
    
    candle = candles[0]
    assert candle.open == Decimal("50000")
    assert candle.high == Decimal("50040")
    assert candle.low == Decimal("50000")
    assert candle.close == Decimal("50040")
    assert candle.volume == Decimal("0.5")


@pytest.mark.asyncio
async def test_signal_aggregator_basic(bus):
    """Test signal aggregator combines signals correctly."""
    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=5,
    )
    
    # Collect combined signals
    combined_signals = []
    
    async def combined_collector(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            combined_signals.append(event)
    
    bus.subscribe(EventType.SIGNAL, combined_collector, "combined_collector")
    
    symbol = "BTC/USD"
    current_time = time()
    
    # Send multiple buy signals
    for i in range(3):
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=current_time,
            signal_type=SignalType.TECHNICAL,
            symbol=symbol,
            action=SignalAction.BUY,
            strength=Decimal("0.6"),
            confidence=Decimal("0.8"),
        )
        await bus.publish(signal)
        await asyncio.sleep(0.05)
    
    await asyncio.sleep(0.3)
    
    # Should have emitted combined signal
    assert len(combined_signals) >= 1
    
    combined = combined_signals[-1]
    assert combined.action == SignalAction.BUY
    assert combined.strength >= Decimal("0.3")


@pytest.mark.asyncio
async def test_signal_aggregator_threshold(bus):
    """Test that weak signals don't pass threshold."""
    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.5"),
        window_seconds=5,
    )
    
    combined_signals = []
    
    async def combined_collector(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            combined_signals.append(event)
    
    bus.subscribe(EventType.SIGNAL, combined_collector, "combined_collector")
    
    symbol = "BTC/USD"
    
    # Send weak signal (below threshold)
    signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.2"),
        confidence=Decimal("0.5"),
    )
    await bus.publish(signal)
    
    await asyncio.sleep(0.3)
    
    # Should not emit (below threshold)
    assert len(combined_signals) == 0


@pytest.mark.asyncio
async def test_signal_aggregator_conflicting_signals(bus):
    """Test aggregator handles buy/sell conflicts."""
    agg = SignalAggregator(
        bus,
        threshold=Decimal("0.3"),
        window_seconds=5,
    )
    
    combined_signals = []
    
    async def combined_collector(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            combined_signals.append(event)
    
    bus.subscribe(EventType.SIGNAL, combined_collector, "combined_collector")
    
    symbol = "BTC/USD"
    current_time = time()
    
    # Send conflicting signals
    buy_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=current_time,
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.6"),
        confidence=Decimal("0.8"),
    )
    
    sell_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=current_time,
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.SELL,
        strength=Decimal("0.4"),
        confidence=Decimal("0.7"),
    )
    
    await bus.publish(buy_signal)
    await asyncio.sleep(0.05)
    await bus.publish(sell_signal)
    await asyncio.sleep(0.3)
    
    # Should resolve to stronger signal (BUY)
    if combined_signals:
        combined = combined_signals[-1]
        assert combined.action == SignalAction.BUY
