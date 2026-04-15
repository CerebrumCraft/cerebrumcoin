"""Shared fixtures + helpers for ORB integration tests.

Provides build_test_pipeline() which constructs a minimal in-memory bus
+ SignalAggregators and exposes helpers to inject SignalEvents and inspect
what symbols each aggregator actually admitted.

Design note: The EventBus captures bound-method references at subscribe()
time, so wrapping _on_signal after construction does NOT intercept bus
dispatch. Instead, MiniPipeline.aggregator_input_symbols() reads the
aggregator's internal _signal_buffer (a defaultdict keyed by symbol), which
is populated only for signals that passed the symbols filter. This is the
canonical observable of DEC-STOCKS-005.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from time import time
from typing import Any

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.signals.aggregator import SignalAggregator


@dataclass
class MiniPipeline:
    """Minimal in-memory pipeline harness for integration tests.

    Lets tests inject SignalEvents directly onto the bus and inspect what
    reached each aggregator's internal buffer. Not a substitute for full
    main.py wiring — just exercises the signal/aggregator filtering path in
    isolation.

    Deliberately injects SignalEvent directly (bypassing MarketDataEvent →
    signal-generator path) because:
    - The isolation property under test lives entirely inside SignalAggregator
      (the symbols filter in _on_signal).
    - Driving ORB through a full MarketDataEvent stream would require live-RTH
      timestamps and range accumulation, introducing unrelated complexity.
    """

    bus: EventBus
    aggregators: dict[str, SignalAggregator]

    async def publish_signal(
        self,
        symbol: str,
        action: SignalAction = SignalAction.BUY,
        signal_type: SignalType = SignalType.TECHNICAL,
        strength: Decimal = Decimal("0.7"),
        confidence: Decimal = Decimal("0.8"),
        timestamp: float | None = None,
    ) -> None:
        """Publish a raw SignalEvent through the bus.

        All subscribed aggregators will receive the event and decide
        independently whether to admit it based on their symbols filter.
        """
        ts = timestamp if timestamp is not None else time()
        event = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=ts,
            signal_type=signal_type,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            reason="synthetic tick for isolation test",
        )
        await self.bus.publish(event)
        # Let the bus drain the subscriber queues.
        # Each asyncio.sleep(0) yields once; 3 iterations is enough because the
        # bus task and both aggregator tasks are queued behind each other.
        for _ in range(5):
            await asyncio.sleep(0)
        # Small real-time yield to ensure background tasks have processed.
        await asyncio.sleep(0.05)

    def aggregator_input_symbols(self, strategy_id: str) -> set[str]:
        """Return the set of symbols that actually entered the aggregator's buffer.

        Uses _signal_buffer.keys() — populated only for signals that passed
        the symbols filter. Empty keys are excluded via the comprehension.
        """
        agg = self.aggregators[strategy_id]
        return {
            sym
            for sym, buf in agg._signal_buffer.items()
            if len(buf) > 0
        }

    async def stop(self) -> None:
        """Gracefully stop the event bus."""
        await self.bus.stop()


async def build_test_pipeline(
    *,
    crypto_strategies: list[str] | None = None,
    stock_strategies: list[str] | None = None,
    crypto_symbols: list[str] | None = None,
    stock_symbols: list[str] | None = None,
) -> MiniPipeline:
    """Build a minimal pipeline with a real EventBus + real SignalAggregators.

    Each strategy gets a SignalAggregator scoped to its symbol list.
    Returns a MiniPipeline that exposes publish_signal() and
    aggregator_input_symbols() for test assertions.

    Args:
        crypto_strategies: Strategy IDs for crypto aggregators.
        stock_strategies:  Strategy IDs for stock aggregators.
        crypto_symbols:    Symbols the crypto aggregators are allowed to trade.
        stock_symbols:     Symbols the stock aggregators are allowed to trade.

    Returns:
        MiniPipeline ready for use (bus already started).
    """
    crypto_strategies = crypto_strategies or []
    stock_strategies = stock_strategies or []
    crypto_symbols = crypto_symbols or []
    stock_symbols = stock_symbols or []

    bus = EventBus()
    await bus.start()

    aggregators: dict[str, SignalAggregator] = {}

    # Crypto strategies — scoped to crypto symbols
    for strat in crypto_strategies:
        agg = SignalAggregator(
            bus=bus,
            strategy_id=strat,
            weights={SignalType.TECHNICAL: Decimal("1.0")},
            threshold=Decimal("0.3"),
            window_seconds=60,
            symbols=crypto_symbols if crypto_symbols else None,
        )
        aggregators[strat] = agg

    # Stock strategies — scoped to stock symbols
    for strat in stock_strategies:
        agg = SignalAggregator(
            bus=bus,
            strategy_id=strat,
            weights={SignalType.TECHNICAL: Decimal("1.0")},
            threshold=Decimal("0.3"),
            window_seconds=60,
            symbols=stock_symbols if stock_symbols else None,
        )
        aggregators[strat] = agg

    return MiniPipeline(bus=bus, aggregators=aggregators)
