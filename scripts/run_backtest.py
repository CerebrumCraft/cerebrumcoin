#!/usr/bin/env python3
"""
Historical data backtesting script.

@decision DEC-MONITOR-005
@title Backtest runner with OHLCV replay
@status accepted
@rationale Validates strategy on historical data. Uses ccxt to fetch OHLCV data,
caches to CSV for reuse. Replays data through existing event pipeline (same code
as live trading). Configurable date ranges and speedup factors.
"""

import argparse
import asyncio
import csv
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import ccxt.async_support as ccxt

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent
from cerebrum.core.types import EventType


async def fetch_ohlcv_data(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Path,
) -> list[dict]:
    """
    Fetch OHLCV data from exchange with caching.
    
    Returns list of candles with keys: timestamp, open, high, low, close, volume
    """
    # Generate cache filename
    cache_file = cache_dir / f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_date.date()}_{end_date.date()}.csv"
    
    # Check cache first
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        candles = []
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append({
                    'timestamp': float(row['timestamp']),
                    'open': Decimal(row['open']),
                    'high': Decimal(row['high']),
                    'low': Decimal(row['low']),
                    'close': Decimal(row['close']),
                    'volume': Decimal(row['volume']),
                })
        return candles
    
    # Fetch from exchange
    print(f"Fetching {symbol} data from {exchange_name}...")
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    
    try:
        # Convert dates to milliseconds
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current = since
        
        while current < end_ms:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not ohlcv:
                break
            
            all_candles.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1000:
                break
        
        # Convert to our format
        candles = []
        for candle in all_candles:
            if candle[0] >= end_ms:
                break
            candles.append({
                'timestamp': float(candle[0] / 1000),
                'open': Decimal(str(candle[1])),
                'high': Decimal(str(candle[2])),
                'low': Decimal(str(candle[3])),
                'close': Decimal(str(candle[4])),
                'volume': Decimal(str(candle[5])),
            })
        
        # Cache results
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w', newline='') as f:
            if candles:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                writer.writeheader()
                for candle in candles:
                    writer.writerow({k: str(v) for k, v in candle.items()})
        
        print(f"Cached {len(candles)} candles to {cache_file}")
        return candles
    
    finally:
        await exchange.close()


async def replay_backtest(
    candles: list[dict],
    symbol: str,
    speedup: float = 1.0,
) -> None:
    """
    Replay candles through event bus.
    
    NOTE: This is a simplified version. Full implementation would integrate
    with the complete trading pipeline (signals, risk, execution).
    """
    bus = EventBus()
    await bus.start()
    
    try:
        print(f"\nReplaying {len(candles)} candles for {symbol}...")
        print(f"Speedup factor: {speedup}x")
        print(f"Date range: {datetime.fromtimestamp(candles[0]['timestamp'])} to {datetime.fromtimestamp(candles[-1]['timestamp'])}")
        
        for i, candle in enumerate(candles):
            # Create market data event
            event = MarketDataEvent(
                event_type=EventType.MARKET_DATA,
                timestamp=candle['timestamp'],
                symbol=symbol,
                price=candle['close'],
                volume=candle['volume'],
            )
            
            await bus.publish(event)
            
            # Simulate time passing (scaled by speedup)
            if i < len(candles) - 1:
                time_delta = candles[i + 1]['timestamp'] - candle['timestamp']
                await asyncio.sleep(time_delta / speedup)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(candles)} candles...")
        
        print("\nBacktest replay complete!")
    
    finally:
        await bus.stop()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USD",
        help="Trading pair (default: BTC/USD)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="kraken",
        help="Exchange name (default: kraken)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Candle timeframe (default: 1m)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to backtest (default: 7)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1000.0,
        help="Replay speedup factor (default: 1000x)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/backtest_cache"),
        help="Cache directory for OHLCV data",
    )
    
    args = parser.parse_args()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    async def run():
        # Fetch/load data
        candles = await fetch_ohlcv_data(
            args.exchange,
            args.symbol,
            args.timeframe,
            start_date,
            end_date,
            args.cache_dir,
        )
        
        if not candles:
            print("No data available!")
            return
        
        # Run backtest
        await replay_backtest(candles, args.symbol, args.speedup)
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
