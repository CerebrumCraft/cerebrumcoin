# Multi-Timeframe Swing Trading Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5th strategy that trades on 1-hour candles for fewer, higher-conviction trades that reduce commission drag.

**Architecture:** Second CandleAggregator (1h) with its own signal generators. Timeframe metadata on all signals. `signal_timeframe_filter` on StrategyConfig/SignalAggregator for routing. SwingTradingConfig with wider exits.

**Tech Stack:** Python 3.12, asyncio, EventBus pub/sub, pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-24-multi-timeframe-swing-strategy-design.md`

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Modify | `cerebrum/signals/base.py` | Add `timeframe` param to SignalGenerator, include in metadata |
| Modify | `cerebrum/strategies/base.py` | Add `signal_timeframe_filter` to StrategyConfig |
| Modify | `cerebrum/signals/aggregator.py` | Add timeframe filter in `_on_signal()` |
| Create | `cerebrum/strategies/swing_trading.py` | SWING_TRADING_CONFIG |
| Modify | `cerebrum/main.py` | Create 1h CandleAggregator, 1h signal generators, register swing strategy |
| Modify | `cerebrum/strategies/momentum.py` | Update initial_balance to $2,000 |
| Modify | `cerebrum/strategies/mean_reversion.py` | Update initial_balance to $2,000 |
| Modify | `cerebrum/strategies/breakout.py` | Update initial_balance to $2,000 |
| Modify | `cerebrum/strategies/range_trading.py` | Update initial_balance to $2,000 |
| Create | `tests/unit/test_multi_timeframe.py` | Timeframe filtering + 1h candle tests |

---

### Task 1: Add `timeframe` metadata to signal generators

**Files:**
- Modify: `cerebrum/signals/base.py` — `__init__` and `_create_signal`
- Test: `tests/unit/test_multi_timeframe.py`

- [ ] **Step 1: Write test**

```python
# tests/unit/test_multi_timeframe.py
import pytest
from decimal import Decimal
from cerebrum.core.bus import EventBus
from cerebrum.core.types import SignalType, SignalAction

@pytest.mark.asyncio
async def test_signal_generator_includes_timeframe_metadata():
    """Signal generators tag signals with their timeframe."""
    from cerebrum.signals.base import SignalGenerator

    class TestGen(SignalGenerator):
        def _get_min_periods(self): return 1
        def _generate_signal(self, symbol, data): return None

    bus = EventBus()
    gen = TestGen(bus, SignalType.TECHNICAL, window_size=10, name="RSI", timeframe="1h")
    sig = gen._create_signal(
        symbol="BTC/USD", action=SignalAction.BUY,
        strength=Decimal("0.5"), confidence=Decimal("0.5"),
        timestamp=1.0, reason="test",
    )
    assert sig.metadata["timeframe"] == "1h"
    assert sig.metadata["source"] == "RSI"

@pytest.mark.asyncio
async def test_signal_generator_default_timeframe_is_1m():
    """When no timeframe specified, defaults to 1m."""
    from cerebrum.signals.base import SignalGenerator

    class TestGen(SignalGenerator):
        def _get_min_periods(self): return 1
        def _generate_signal(self, symbol, data): return None

    bus = EventBus()
    gen = TestGen(bus, SignalType.TECHNICAL, window_size=10, name="RSI")
    sig = gen._create_signal(
        symbol="BTC/USD", action=SignalAction.BUY,
        strength=Decimal("0.5"), confidence=Decimal("0.5"),
        timestamp=1.0, reason="test",
    )
    assert sig.metadata["timeframe"] == "1m"
```

- [ ] **Step 2: Run test — expect FAIL** (SignalGenerator doesn't accept timeframe param yet)

- [ ] **Step 3: Implement**

In `cerebrum/signals/base.py`:
- Add `timeframe: str = "1m"` param to `SignalGenerator.__init__`. Store as `self._timeframe`.
- In `_create_signal`, add `"timeframe": self._timeframe` to the metadata dict (alongside existing `"source": self._name`).

- [ ] **Step 4: Run test — expect PASS. Run full suite — no regressions.**

- [ ] **Step 5: Commit:** `feat: add timeframe metadata to signal generators (default 1m)`

---

### Task 2: Add `signal_timeframe_filter` to StrategyConfig and SignalAggregator

**Files:**
- Modify: `cerebrum/strategies/base.py` — add field
- Modify: `cerebrum/signals/aggregator.py` — add filter
- Test: `tests/unit/test_multi_timeframe.py`

- [ ] **Step 1: Write test**

```python
@pytest.mark.asyncio
async def test_aggregator_filters_by_timeframe():
    """Aggregator with signal_timeframe_filter drops non-matching timeframes."""
    from cerebrum.signals.aggregator import SignalAggregator
    from cerebrum.core.events import SignalEvent
    from cerebrum.core.types import EventType
    from time import time

    bus = EventBus()
    agg = SignalAggregator(
        bus=bus,
        weights={SignalType.TECHNICAL: Decimal("1.0")},
        threshold=Decimal("0.2"),
        window_seconds=60,
        strategy_id="swing_trading",
        signal_timeframe_filter="1h",
    )
    await agg.start()

    # 1m signal — should be ignored
    await bus.publish(SignalEvent(
        event_type=None, timestamp=time(),
        signal_type=SignalType.TECHNICAL, symbol="BTC/USD",
        action=SignalAction.BUY, strength=Decimal("0.8"),
        confidence=Decimal("0.8"),
        metadata={"source": "RSI", "timeframe": "1m"},
    ))
    assert len(agg._signal_buffer.get("BTC/USD", [])) == 0

    # 1h signal — should be accepted
    await bus.publish(SignalEvent(
        event_type=None, timestamp=time(),
        signal_type=SignalType.TECHNICAL, symbol="BTC/USD",
        action=SignalAction.BUY, strength=Decimal("0.8"),
        confidence=Decimal("0.8"),
        metadata={"source": "RSI", "timeframe": "1h"},
    ))
    assert len(agg._signal_buffer.get("BTC/USD", [])) == 1
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

In `cerebrum/strategies/base.py`, add after `signal_source_filter`:
```python
signal_timeframe_filter: str | None = None  # Only accept signals with this metadata.timeframe
```

In `cerebrum/signals/aggregator.py`:
- Add `signal_timeframe_filter: str | None = None` to `__init__`. Store as `self._signal_timeframe_filter`.
- In `_on_signal()`, after the existing source filter block, add:
```python
if self._signal_timeframe_filter:
    tf = event.metadata.get("timeframe") if event.metadata else None
    if tf != self._signal_timeframe_filter:
        return
```

- [ ] **Step 4: Run test — PASS. Full suite — no regressions.**

- [ ] **Step 5: Commit:** `feat: add signal_timeframe_filter to StrategyConfig and SignalAggregator`

---

### Task 3: Wire timeframe filter through StrategyRegistry

**Files:**
- Modify: `cerebrum/strategies/registry.py` — pass `signal_timeframe_filter` to aggregator

- [ ] **Step 1: Modify `_build_pipeline()`**

Find where SignalAggregator is constructed. It already passes `signal_source_filter=cfg.signal_source_filter`. Add:
```python
signal_timeframe_filter=cfg.signal_timeframe_filter,
```

- [ ] **Step 2: Run full suite — no regressions**

- [ ] **Step 3: Commit:** `feat: wire signal_timeframe_filter through StrategyRegistry`

---

### Task 4: Create 1h signal generators and SwingTradingConfig

**Files:**
- Create: `cerebrum/strategies/swing_trading.py`
- Modify: `cerebrum/main.py` — create 1h CandleAggregator, 1h generators, register swing strategy

- [ ] **Step 1: Create `cerebrum/strategies/swing_trading.py`**

```python
"""
Swing trading strategy configuration.

Trades on 1-hour candles for fewer, higher-conviction trades.
Directly addresses commission drag (#1 enemy from Session 4: 64% of gross).

@decision DEC-SWING-001
@title 1-hour timeframe swing strategy to reduce commission drag
@status accepted
@rationale Session 4 data showed $115 commission on $179 gross profit (64%).
Most commission-positive trades held 1-4 hours. 1h candles produce signals
that capture multi-hour trends, reducing trade frequency while increasing
per-trade profitability. Same RSI/MACD/BB pipeline, different timeframe.
"""

from decimal import Decimal
from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig

SWING_TRADING_CONFIG = StrategyConfig(
    name="swing_trading",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.5"),
        SignalType.SENTIMENT: Decimal("0.3"),
        SignalType.NEWS: Decimal("0.4"),
        SignalType.REGIME: Decimal("0.8"),
    },
    aggregator_threshold=Decimal("0.5"),
    signal_timeframe_filter="1h",
    risk_overrides={
        "min_signal_strength": "0.5",
        "position_size_percent": "5.0",
        "post_fill_cooldown_seconds": 3600,
    },
    exit_config={
        "stop_loss_percent": "3.0",
        "take_profit_percent": "5.0",
        "max_position_age_minutes": 480,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "1.0",
    },
    initial_balance=Decimal("2000.00"),
    symbols=["BTC/USD", "ETH/USD"],
)
```

- [ ] **Step 2: Modify `cerebrum/main.py`**

In `_setup_signal_generators()` or equivalent:

1. Create a second CandleAggregator:
```python
self.candle_agg_1h = CandleAggregator(self.bus, interval_seconds=3600)
```

2. Create 1h signal generators (same classes, different aggregator and timeframe):
```python
rsi_1h = RSISignal(self.bus, self.candle_agg_1h, timeframe="1h", ...)
macd_1h = MACDSignal(self.bus, self.candle_agg_1h, timeframe="1h", ...)
bb_1h = BollingerSignal(self.bus, self.candle_agg_1h, timeframe="1h", ...)
vwap_1h = VWAPSignal(self.bus, self.candle_agg_1h, timeframe="1h", ...)
```

Check each signal generator's `__init__` to see if they accept a `timeframe` param — they inherit from `SignalGenerator` which now has it (Task 1). If any generator overrides `__init__`, ensure `timeframe` is passed through via `**kwargs` or explicit param.

3. Import and register SWING_TRADING_CONFIG:
```python
from cerebrum.strategies.swing_trading import SWING_TRADING_CONFIG
self.strategy_registry.register(SWING_TRADING_CONFIG)
```

- [ ] **Step 3: Update initial_balance for 5-strategy split**

Change to $2,000 in:
- `cerebrum/strategies/momentum.py` — `Decimal("2500.00")` → `Decimal("2000.00")`
- `cerebrum/strategies/mean_reversion.py` — `Decimal("2500.00")` → `Decimal("2000.00")`
- `cerebrum/strategies/breakout.py` — `Decimal("2500.00")` → `Decimal("2000.00")`
- `cerebrum/strategies/range_trading.py` — `Decimal("2500.00")` → `Decimal("2000.00")`

Update test assertions in `tests/unit/test_main_wiring.py` accordingly.

- [ ] **Step 4: Run full suite — all tests pass**

- [ ] **Step 5: Commit:** `feat: add swing_trading strategy with 1h candles and multi-timeframe infrastructure`

---

### Task 5: Integration test

**Files:**
- Create or extend: `tests/unit/test_multi_timeframe.py`

- [ ] **Step 1: Write integration test**

```python
@pytest.mark.asyncio
async def test_swing_strategy_only_receives_1h_signals():
    """Swing aggregator accepts 1h signals, ignores 1m signals."""
    # Create bus, two aggregators (1m momentum, 1h swing)
    # Publish a 1m RSI signal — momentum buffer has it, swing buffer empty
    # Publish a 1h RSI signal — swing buffer has it, momentum ignores it (no timeframe filter)
    # Actually momentum has no timeframe filter so it accepts BOTH — that's fine

@pytest.mark.asyncio
async def test_1h_candle_aggregator_independent():
    """1h CandleAggregator produces candles independently from 1m."""
    # Create two CandleAggregators (60s and 3600s)
    # Feed market data ticks
    # Verify 1m aggregator produces candles quickly
    # Verify 1h aggregator hasn't produced a candle yet (needs 3600s of data)
```

- [ ] **Step 2: Run tests — PASS. Full suite — no regressions.**

- [ ] **Step 3: Commit:** `test: add multi-timeframe integration tests`

---

## Verification

1. `python3 -m pytest tests/ -x -q` — all 470+ tests pass
2. Check Session 10 logs for `swing_trading` strategy registration and `timeframe=1h` signals
3. Verify swing_trading trades less frequently than momentum (target: 2-6/day vs 20+)
