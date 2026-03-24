"""
Integration test for the full strategy loop: data → signals → risk → orders → fills.

Tests the complete Phase 2 pipeline end-to-end.
"""

import asyncio
from decimal import Decimal
from time import time

import pytest

from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, OrderEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import MinSignalStrengthRule, PositionSizingRule
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.candles import CandleAggregator


@pytest.mark.asyncio
async def test_full_strategy_loop():
    """
    Test complete flow: MarketData → Candles → Signals → Aggregator → Risk → Order → Fill.
    
    This validates Phase 2 integration.
    """
    bus = EventBus(queue_size=100)
    await bus.start()

    paper_adapter = None

    try:
        # Setup components
        candle_agg = CandleAggregator(bus, interval_seconds=5, window_size=50)

        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))

        signal_agg = SignalAggregator(
            bus,
            threshold=Decimal("0.3"),
            window_seconds=5,
        )

        risk_rules = [
            PositionSizingRule(position_size_percent=Decimal("2.0")),
            MinSignalStrengthRule(min_strength=Decimal("0.3")),
        ]
        risk_manager = RiskManager(bus, portfolio, rules=risk_rules)

        from pathlib import Path
        state_file = Path("/tmp/test_paper_state.json")
        # Remove stale state so each run starts fresh
        state_file.unlink(missing_ok=True)
        paper_adapter = PaperTradingAdapter(
            bus,
            {},
            initial_balance=Decimal("10000.0"),
            commission_percent=Decimal("0.1"),
            slippage_percent=Decimal("0.05"),
            state_file=state_file,
        )
        await paper_adapter.connect()
        
        # Event collectors
        orders = []
        fills = []
        
        async def order_collector(event):
            if isinstance(event, OrderEvent):
                orders.append(event)
        
        async def fill_collector(event):
            from cerebrum.core.events import FillEvent
            if isinstance(event, FillEvent):
                fills.append(event)
        
        bus.subscribe(EventType.ORDER, order_collector, "test_order_collector")
        bus.subscribe(EventType.FILL, fill_collector, "test_fill_collector")
        
        # Simulate market data
        symbol = "BTC/USD"
        base_time = 1000.0
        base_price = Decimal("50000")
        
        # Send ticks to build up candles
        for i in range(30):
            event = MarketDataEvent(
                event_type=EventType.MARKET_DATA,
                timestamp=base_time + i,
                symbol=symbol,
                price=base_price + Decimal(i * 10),
                volume=Decimal("0.1"),
            )
            await bus.publish(event)
            await asyncio.sleep(0.02)
        
        # Wait for candles to accumulate
        await asyncio.sleep(0.5)
        
        # Manually inject a strong buy signal (simulating what technical indicators would produce)
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=time(),
            signal_type=SignalType.TECHNICAL,
            symbol=symbol,
            action=SignalAction.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.9"),
            reason="Test signal",
        )
        await bus.publish(signal)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify pipeline executed
        # 1. Signal was published ✓
        # 2. Aggregator should have combined it
        # 3. Risk manager should have approved and sized it
        # 4. Order should have been created
        assert len(orders) >= 1, "Order should have been generated"
        
        order = orders[0]
        assert order.symbol == symbol
        
        # 5. Paper adapter should have filled it
        await asyncio.sleep(0.3)
        assert len(fills) >= 1, "Order should have been filled"
        
        fill = fills[0]
        assert fill.symbol == symbol
        assert fill.filled_amount > Decimal("0")
        
        # 6. Portfolio should track the position
        pos = portfolio.get_position(symbol)
        assert pos is not None, "Position should exist"
        assert pos.amount == fill.filled_amount
        
        # 7. Cash balance should be reduced
        initial_cash = Decimal("10000.0")
        current_cash = portfolio.get_cash_balance()
        assert current_cash < initial_cash, "Cash should be reduced after buy"
        
    finally:
        if paper_adapter:
            await paper_adapter.disconnect()
        await bus.stop()


@pytest.mark.asyncio
async def test_strategy_loop_rejects_weak_signals():
    """Test that weak signals don't result in orders."""
    bus = EventBus(queue_size=100)
    await bus.start()

    paper_adapter = None

    try:
        portfolio = PortfolioTracker(bus, initial_balance=Decimal("10000.0"))
        
        signal_agg = SignalAggregator(
            bus,
            threshold=Decimal("0.5"),  # High threshold
            window_seconds=5,
        )
        
        risk_rules = [MinSignalStrengthRule(min_strength=Decimal("0.5"))]
        risk_manager = RiskManager(bus, portfolio, rules=risk_rules)

        from pathlib import Path
        state_file = Path("/tmp/test_paper_state2.json")
        # Remove stale state so each run starts fresh
        state_file.unlink(missing_ok=True)
        paper_adapter = PaperTradingAdapter(
            bus,
            {},
            initial_balance=Decimal("10000.0"),
            commission_percent=Decimal("0.1"),
            slippage_percent=Decimal("0.05"),
            state_file=state_file,
        )
        await paper_adapter.connect()
        
        orders = []
        
        async def order_collector(event):
            if isinstance(event, OrderEvent):
                orders.append(event)
        
        bus.subscribe(EventType.ORDER, order_collector, "test_order_collector")
        
        # Send weak signal
        signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=time(),
            signal_type=SignalType.TECHNICAL,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.2"),  # Weak
            confidence=Decimal("0.5"),
            reason="Weak test signal",
        )
        await bus.publish(signal)
        
        await asyncio.sleep(0.5)
        
        # No order should be generated (below threshold and min strength)
        assert len(orders) == 0, "Weak signal should not generate order"
        
    finally:
        if paper_adapter:
            await paper_adapter.disconnect()
        await bus.stop()


@pytest.mark.asyncio
async def test_multiple_signals_aggregate():
    """Test that multiple signals combine in aggregator."""
    bus = EventBus(queue_size=100)
    await bus.start()
    
    try:
        signal_agg = SignalAggregator(
            bus,
            threshold=Decimal("0.4"),
            window_seconds=5,
        )
        
        # Collect combined signals
        combined = []
        
        async def combined_collector(event):
            if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
                combined.append(event)
        
        bus.subscribe(EventType.SIGNAL, combined_collector, "test_combined_collector")
        
        symbol = "BTC/USD"
        
        # Send multiple moderate signals
        for i in range(3):
            signal = SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=time(),
                signal_type=SignalType.TECHNICAL,
                symbol=symbol,
                action=SignalAction.BUY,
                strength=Decimal("0.4"),
                confidence=Decimal("0.7"),
                reason=f"Signal {i}",
            )
            await bus.publish(signal)
            await asyncio.sleep(0.05)
        
        await asyncio.sleep(0.5)
        
        # Should produce combined signal
        assert len(combined) >= 1, "Signals should aggregate"
        
        agg_signal = combined[-1]
        assert agg_signal.action == SignalAction.BUY
        assert agg_signal.strength >= Decimal("0.4")
        
    finally:
        await bus.stop()
