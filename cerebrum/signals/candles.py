"""
Candle aggregator for converting MarketDataEvent ticks into OHLCV bars.

Technical indicators require OHLCV candles, not individual ticks. This module
aggregates ticks into time-based candles.

@decision DEC-SIGNAL-002
@title Candle aggregator for OHLCV bar construction
@status accepted
@rationale Technical indicators require OHLCV bars, not ticks. Time-based aggregation
with configurable intervals (1m, 5m, 15m, 1h, etc.) enables indicator calculations.
Uses a sliding window to maintain recent candles for indicator warmup periods.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, MarketDataEvent
from cerebrum.core.types import EventType, Price, Symbol, Timestamp, Volume

logger = structlog.get_logger()


@dataclass
class Candle:
    """OHLCV candle for a time period."""
    symbol: Symbol
    timestamp: Timestamp  # Candle start time
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    
    def update(self, price: Price, volume: Volume) -> None:
        """Update candle with new tick data."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume


class CandleAggregator:
    """
    Aggregates MarketDataEvent ticks into OHLCV candles.
    
    Features:
    - Configurable time intervals (seconds)
    - Per-symbol candle tracking
    - Automatic candle rotation on time boundary
    - Maintains sliding window of recent candles
    """
    
    def __init__(
        self,
        bus: EventBus,
        interval_seconds: int = 60,
        window_size: int = 200,
    ) -> None:
        """
        Initialize candle aggregator.
        
        Args:
            bus: Event bus for subscribing to market data
            interval_seconds: Candle interval in seconds (default: 60 = 1 minute)
            window_size: Number of candles to retain per symbol
        """
        self._bus = bus
        self._interval = interval_seconds
        self._window_size = window_size
        
        # Per-symbol candle storage
        self._current_candles: dict[Symbol, Candle | None] = defaultdict(lambda: None)
        self._completed_candles: dict[Symbol, Deque[Candle]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        self._log = logger.bind(component="candle_aggregator")
        
        # Subscribe to market data
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="candle_aggregator",
        )
        
        self._log.info(
            "candle_aggregator_initialized",
            interval_seconds=interval_seconds,
            window_size=window_size,
        )
    
    async def _on_market_data(self, event: Event) -> None:
        """Handle incoming market data and update candles."""
        if not isinstance(event, MarketDataEvent):
            return
        
        symbol = event.symbol
        price = event.price
        volume = event.volume
        timestamp = event.timestamp
        
        # Calculate candle start time (floor to interval boundary)
        candle_time = self._floor_timestamp(timestamp)
        
        current_candle = self._current_candles[symbol]
        
        # Check if we need to start a new candle
        if current_candle is None or current_candle.timestamp != candle_time:
            # Complete and store the old candle
            if current_candle is not None:
                self._completed_candles[symbol].append(current_candle)
                self._log.debug(
                    "candle_completed",
                    symbol=symbol,
                    timestamp=current_candle.timestamp,
                    open=str(current_candle.open),
                    high=str(current_candle.high),
                    low=str(current_candle.low),
                    close=str(current_candle.close),
                    volume=str(current_candle.volume),
                )
            
            # Start new candle
            self._current_candles[symbol] = Candle(
                symbol=symbol,
                timestamp=candle_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
            )
        else:
            # Update current candle
            current_candle.update(price, volume)
    
    def _floor_timestamp(self, timestamp: Timestamp) -> Timestamp:
        """Floor timestamp to candle interval boundary."""
        return float(int(timestamp // self._interval) * self._interval)
    
    def get_candles(self, symbol: Symbol, count: int | None = None) -> list[Candle]:
        """
        Get completed candles for a symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of recent candles to return (None = all)
        
        Returns:
            List of candles (oldest to newest)
        """
        candles = list(self._completed_candles[symbol])
        if count is not None:
            candles = candles[-count:]
        return candles
    
    def get_candle_count(self, symbol: Symbol) -> int:
        """Get number of completed candles for a symbol."""
        return len(self._completed_candles[symbol])
