# Multi-Timeframe Swing Trading Strategy

## Problem

Commission drag is the #1 enemy. Session 4: 64% of gross profit ($115 commission on $179 gross). The current momentum strategy trades on 1-minute candles, producing many small trades that barely cover round-trip commission (~0.32%). Longer timeframes produce fewer, higher-conviction trades that move further — reducing commission as a percentage of profit.

## Solution

A **5th strategy** (`swing_trading`) that runs the same RSI/MACD/Bollinger pipeline on **1-hour candles** instead of 1-minute candles. This requires multi-timeframe infrastructure: a second CandleAggregator instance and a second set of signal generators, with timeframe-based signal routing.

## Architecture

### New Infrastructure

**Second CandleAggregator** (`interval_seconds=3600`):
- Created in `main.py` alongside the existing 60s aggregator
- Subscribes to the same `MARKET_DATA` events — aggregates independently
- Both aggregators coexist, producing candles at different rates

**Second set of signal generators** (RSI, MACD, Bollinger, VWAP):
- Same classes as existing generators
- Wired to the 1h CandleAggregator
- Each tags signals with `metadata={"source": "RSI", "timeframe": "1h"}`
- Signal names suffixed: `"RSI_1h"`, `"MACD_1h"`, etc. to distinguish in logs

**Timeframe metadata on all signals:**
- Existing 1m generators get `metadata["timeframe"] = "1m"` (backward compat — existing strategies ignore this field)
- New 1h generators get `metadata["timeframe"] = "1h"`

### Signal Routing

Add `signal_timeframe_filter: str | None = None` to `StrategyConfig` (same pattern as `signal_source_filter`).

In `SignalAggregator._on_signal()`, after the existing source filter check:
```python
if self._signal_timeframe_filter:
    timeframe = event.metadata.get("timeframe") if event.metadata else None
    if timeframe != self._signal_timeframe_filter:
        return
```

Swing strategy sets `signal_timeframe_filter="1h"`. Existing strategies don't set it (None = accept all timeframes, which means they continue accepting 1m signals as before).

### Strategy Config

```python
SWING_TRADING_CONFIG = StrategyConfig(
    name="swing_trading",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.5"),   # High trust in 1h signals
        SignalType.SENTIMENT: Decimal("0.3"),   # Sentiment is noise on short timeframes, signal on long
        SignalType.NEWS: Decimal("0.4"),         # News matters more on longer timeframes
        SignalType.REGIME: Decimal("0.8"),       # Regime is very relevant for multi-hour holds
    },
    aggregator_threshold=Decimal("0.5"),         # Higher bar — fewer, better trades
    signal_timeframe_filter="1h",                # Only accept 1h signals
    risk_overrides={
        "min_signal_strength": "0.5",            # Only strong 1h signals
        "position_size_percent": "5.0",          # Larger positions (higher conviction)
        "post_fill_cooldown_seconds": 3600,      # 1 hour between entries
    },
    exit_config={
        "stop_loss_percent": "3.0",              # Wide stop — 1h trends are noisy
        "take_profit_percent": "5.0",            # Wide target — let winners run
        "max_position_age_minutes": 480,         # 8 hours max hold
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "1.0",                 # Min 1% TP even in tight ranges
    },
    initial_balance=Decimal("2000.00"),           # 5-strategy split
    symbols=["BTC/USD", "ETH/USD"],
)
```

### Capital Allocation (5-strategy split)

| Strategy | Balance | Rationale |
|----------|---------|-----------|
| momentum | $2,000 | Reduced from $2,500 — fast trading, high churn |
| mean_reversion | $2,000 | UNKNOWN/light ranging |
| breakout | $2,000 | BULL/VOLATILE trend captures |
| range_trading | $2,000 | SIDEWAYS S/R bounces |
| swing_trading | $2,000 | Long-timeframe high-conviction trades |

The conductor will rebalance based on market conditions and strategy performance.

### Integration Points

**main.py:**
- Create second CandleAggregator (1h)
- Create second set of signal generators (RSI_1h, MACD_1h, BB_1h, VWAP_1h)
- Register SWING_TRADING_CONFIG in StrategyRegistry

**StrategyConfig:**
- Add `signal_timeframe_filter: str | None = None`

**SignalAggregator:**
- Add timeframe filter check in `_on_signal()` (same pattern as source filter)

**Signal generators (base.py):**
- Add `timeframe` to the metadata dict in `_create_signal()`. Default to `"1m"` for existing generators. New 1h generators pass `timeframe="1h"` via constructor param.

### What Stays Unchanged

- Existing 4 strategies — no behavioral changes (they don't set timeframe filter, so they accept 1m signals as before)
- Standard ExitMonitor — swing uses the same one with wider config
- No new exit monitor, no new risk rules
- Regime detection, guards, conductor, Darwinian allocator — all automatic

### Testing Strategy

1. Unit test: Second CandleAggregator produces candles independently
2. Unit test: Timeframe filter in SignalAggregator drops non-matching signals
3. Unit test: 1h generators tag signals with correct timeframe
4. Integration: Swing strategy only receives 1h signals
5. Regression: Existing strategies unaffected
6. Paper trading: Session 10 with swing_trading enabled

### Success Criteria

- Swing strategy trades less frequently than momentum (target: 2-6 trades/day vs 20+)
- Average hold time > 1 hour
- Commission as % of gross < 30% (vs 64% in Session 4)
- P&L per trade is larger (wider moves on 1h timeframe)
