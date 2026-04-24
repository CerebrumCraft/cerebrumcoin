# Range Trading Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4th trading strategy that buys near support and sells near resistance in SIDEWAYS markets, using structural S/R levels for entry/exit instead of fixed percentages.

**Architecture:** New RangeDetector monitors S/R bounces and validates tradeable ranges (3+ bounces). New RangeExitMonitor exits at resistance/support-breakdown/regime-change. SidewaysSuppressionRule gets optional `exempt_strategies` param so range_trading can trade in SIDEWAYS. Strategy integrates via existing StrategyRegistry + StrategyConfig pattern.

**Tech Stack:** Python 3.12, asyncio, EventBus pub/sub, aiosqlite, pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-24-range-trading-strategy-design.md`

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Create | `cerebrum/strategies/range_detector.py` | RangeDetector + RangeState: bounce counting, range validation |
| Create | `cerebrum/strategies/range_trading.py` | RANGE_TRADING_CONFIG: StrategyConfig for range strategy |
| Create | `cerebrum/risk/range_exit_monitor.py` | Structural exits at resistance/support breakdown |
| Modify | `cerebrum/strategies/base.py` | Add `signal_source_filter` and `exit_monitor_factory` to StrategyConfig |
| Modify | `cerebrum/signals/base.py` | Add `source` to SignalEvent metadata in `_create_signal()` |
| Modify | `cerebrum/signals/aggregator.py` | Filter signals by `signal_source_filter` in `_on_signal()` |
| Modify | `cerebrum/risk/rules.py` | Add `exempt_strategies` to SidewaysSuppressionRule |
| Modify | `cerebrum/strategies/registry.py` | Use `exit_monitor_factory`, pass `exempt_strategies` to guards |
| Modify | `cerebrum/main.py` | Register range_trading strategy, pass exempt_strategies |
| Modify | `config/paper.toml` | Add `[strategy.range_trading]` config section |
| Create | `tests/unit/test_range_detector.py` | RangeDetector unit tests |
| Create | `tests/unit/test_range_exit_monitor.py` | RangeExitMonitor unit tests |
| Create | `tests/unit/test_range_trading_integration.py` | End-to-end range trading cycle |

---

### Task 1: Add `source` metadata to signal generators

**Files:**
- Modify: `cerebrum/signals/base.py:135-165` (`_create_signal` method)
- Modify: `cerebrum/signals/support_resistance.py:307-315` (`_create_signal` call)
- Test: `tests/unit/test_support_resistance.py` (existing — verify source metadata)

- [ ] **Step 1: Write test for source metadata**

```python
# tests/unit/test_signal_source_metadata.py
import pytest
from decimal import Decimal
from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalType

@pytest.mark.asyncio
async def test_sr_signal_has_source_metadata():
    """S/R signals should include source='support_resistance' in metadata."""
    bus = EventBus()
    received = []

    async def capture(event):
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, capture, "test")

    # Create S/R generator and feed it enough data to produce a signal
    # (We'll check that existing tests still pass AND metadata.source is set)
    # For now, verify the _create_signal method sets metadata.source
    from cerebrum.signals.base import SignalGenerator
    from cerebrum.core.types import SignalAction

    class TestGenerator(SignalGenerator):
        def _get_min_periods(self):
            return 1
        def _generate_signal(self, symbol, data):
            return self._create_signal(
                symbol=symbol, action=SignalAction.BUY,
                strength=Decimal("0.5"), confidence=Decimal("0.5"),
                timestamp=1.0, reason="test",
            )

    gen = TestGenerator(bus, SignalType.TECHNICAL, window_size=10, name="TestGen")
    sig = gen._create_signal(
        symbol="BTC/USD", action=SignalAction.BUY,
        strength=Decimal("0.5"), confidence=Decimal("0.5"),
        timestamp=1.0, reason="test",
    )
    assert sig.metadata is not None
    assert sig.metadata.get("source") == "TestGen"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_signal_source_metadata.py -v`
Expected: FAIL (metadata doesn't have source key yet)

- [ ] **Step 3: Modify `_create_signal` in `cerebrum/signals/base.py`**

In the `_create_signal` method (~line 135), add `metadata={"source": self._name}` to the SignalEvent constructor. The `self._name` is already set in `__init__` (e.g., "SupportResistance", "RSI", "MACD").

Check the existing `_create_signal` return statement and add the metadata field. If the method already passes metadata, merge the source into it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_signal_source_metadata.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `pytest tests/ -x -q`
Expected: All 428+ tests pass

- [ ] **Step 6: Commit**

```bash
git add cerebrum/signals/base.py tests/unit/test_signal_source_metadata.py
git commit -m "feat: add source metadata to signal events for source filtering"
```

---

### Task 2: Add `signal_source_filter` and `exit_monitor_factory` to StrategyConfig

**Files:**
- Modify: `cerebrum/strategies/base.py:39-77` (StrategyConfig dataclass)

- [ ] **Step 1: Add fields to StrategyConfig**

Add two optional fields to the `StrategyConfig` frozen dataclass:

```python
signal_source_filter: str | None = None  # Only accept signals with this metadata.source
exit_monitor_factory: Any = None  # Optional callable(bus, portfolio, cfg, app_config) -> ExitMonitor
```

Both default to `None` for backward compatibility. Existing strategies don't set them.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass (new fields have defaults, no existing code breaks)

- [ ] **Step 3: Commit**

```bash
git add cerebrum/strategies/base.py
git commit -m "feat: add signal_source_filter and exit_monitor_factory to StrategyConfig"
```

---

### Task 3: Add signal source filtering to SignalAggregator

**Files:**
- Modify: `cerebrum/signals/aggregator.py:156-175` (`_on_signal` method)
- Test: new test in `tests/unit/test_signal_source_metadata.py`

- [ ] **Step 1: Write test for signal filtering**

```python
@pytest.mark.asyncio
async def test_aggregator_filters_by_source():
    """Aggregator with signal_source_filter should drop non-matching signals."""
    from cerebrum.signals.aggregator import SignalAggregator
    bus = EventBus()
    agg = SignalAggregator(
        bus=bus,
        weights={SignalType.TECHNICAL: Decimal("1.0")},
        threshold=Decimal("0.2"),
        window_seconds=60,
        strategy_id="range_trading",
        signal_source_filter="SupportResistance",
    )

    # Publish an RSI signal (source="RSI") — should be ignored
    rsi_signal = SignalEvent(
        event_type=EventType.SIGNAL, timestamp=1.0,
        signal_type=SignalType.TECHNICAL, symbol="BTC/USD",
        action=SignalAction.BUY, strength=Decimal("0.8"),
        confidence=Decimal("0.8"), metadata={"source": "RSI"},
    )
    await bus.publish(rsi_signal)
    # aggregator should have 0 signals in buffer for BTC/USD
    assert len(agg._signal_buffer.get("BTC/USD", [])) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_signal_source_metadata.py::test_aggregator_filters_by_source -v`
Expected: FAIL (SignalAggregator doesn't accept signal_source_filter param yet)

- [ ] **Step 3: Add `signal_source_filter` param to SignalAggregator.__init__**

In `cerebrum/signals/aggregator.py`, add `signal_source_filter: str | None = None` param to `__init__`. Store as `self._signal_source_filter`.

In `_on_signal` method (~line 156), after the COMBINED check, add:

```python
# Filter by signal source if configured (range_trading only wants S/R signals)
if self._signal_source_filter and event.metadata:
    if event.metadata.get("source") != self._signal_source_filter:
        return
elif self._signal_source_filter:
    return  # No metadata at all — filter it out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_signal_source_metadata.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add cerebrum/signals/aggregator.py tests/unit/test_signal_source_metadata.py
git commit -m "feat: add signal_source_filter to SignalAggregator for source-based filtering"
```

---

### Task 4: Add `exempt_strategies` to SidewaysSuppressionRule

**Files:**
- Modify: `cerebrum/risk/rules.py` (SidewaysSuppressionRule class)
- Test: `tests/unit/test_sideways_suppression.py` (existing + new test)

- [ ] **Step 1: Write test for strategy exemption**

```python
# Add to tests/unit/test_sideways_suppression.py
@pytest.mark.asyncio
async def test_exempt_strategy_approved_in_sideways():
    """Strategy in exempt_strategies should be approved even in SIDEWAYS low-vol."""
    bus = EventBus()
    rule = SidewaysSuppressionRule(
        bus=bus,
        min_range_pct=Decimal("1.0"),
        window_size=100,
        exempt_strategies={"range_trading"},
    )
    # Set regime to SIDEWAYS
    # ... publish RegimeChangeEvent, feed low-vol prices ...
    # Create a signal with strategy_id="range_trading"
    # Evaluate — should APPROVE
    # Create a signal with strategy_id="momentum"
    # Evaluate — should DENY
```

Fill in the full test following the pattern in the existing `test_sideways_suppression.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sideways_suppression.py::test_exempt_strategy_approved_in_sideways -v`
Expected: FAIL

- [ ] **Step 3: Add `exempt_strategies` param to SidewaysSuppressionRule**

In `cerebrum/risk/rules.py`, add `exempt_strategies: set[str] | None = None` param to `SidewaysSuppressionRule.__init__`. Store as `self._exempt_strategies = exempt_strategies or set()`.

At the top of the `evaluate` method, add:

```python
if signal.strategy_id and signal.strategy_id in self._exempt_strategies:
    return RuleResult(
        decision=RuleDecision.APPROVE,
        reason=f"Strategy {signal.strategy_id} exempt from sideways suppression",
        risk_level=RiskLevel.LOW,
    )
```

- [ ] **Step 4: Run test to verify it passes + full suite**

Run: `pytest tests/unit/test_sideways_suppression.py -v && pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cerebrum/risk/rules.py tests/unit/test_sideways_suppression.py
git commit -m "feat: add exempt_strategies to SidewaysSuppressionRule for range trading"
```

---

### Task 5: Build RangeDetector

**Files:**
- Create: `cerebrum/strategies/range_detector.py`
- Create: `tests/unit/test_range_detector.py`

- [ ] **Step 1: Write RangeDetector tests**

Test cases:
1. `test_range_not_confirmed_with_fewer_bounces` — 2 bounces, range_confirmed=False
2. `test_range_confirmed_after_three_bounces` — 3 bounces, range_confirmed=True
3. `test_bounce_deduplication` — continuous ticks near support only count as 1 bounce
4. `test_range_invalidated_on_regime_change` — SIDEWAYS→BEAR clears range state
5. `test_range_invalidated_on_breakout` — price breaks beyond support/resistance
6. `test_range_rejected_when_too_narrow` — range_width_pct < min_range_width_pct
7. `test_stale_levels_expire` — levels older than staleness timeout are dropped
8. `test_get_range_returns_none_for_unknown_symbol` — no data = None

Use real EventBus. Feed S/R-style SignalEvents with `metadata={"source": "SupportResistance"}` and `action=BUY` (near support) or `action=SELL` (near resistance). Use RegimeChangeEvent for regime transitions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_range_detector.py -v`
Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement RangeDetector**

Create `cerebrum/strategies/range_detector.py`:

```python
"""
Range detector for SIDEWAYS market range identification.

Monitors S/R signals to identify tradeable price ranges. A range is confirmed
when at least min_bounces bounces have been observed across support and
resistance levels. Ranges are invalidated on regime change or breakout.

@decision DEC-RANGE-001
@title RangeDetector as queryable state object (not event emitter)
@status accepted
@rationale Simpler than adding a new EventType. The RangeExitMonitor and
signal aggregator query get_range() on demand rather than subscribing to
range update events. Reduces event bus traffic and coupling.
"""

import time
from dataclasses import dataclass
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, Symbol

logger = structlog.get_logger()


@dataclass(frozen=True)
class RangeState:
    """Immutable snapshot of a detected trading range."""
    support_level: Decimal
    resistance_level: Decimal
    bounce_count: int
    range_confirmed: bool
    range_width_pct: Decimal
    last_updated: float


class RangeDetector:
    """Monitors S/R bounces and validates tradeable ranges."""

    def __init__(
        self,
        bus: EventBus,
        min_bounces: int = 3,
        min_range_width_pct: Decimal = Decimal("0.6"),
        breakout_margin_pct: Decimal = Decimal("0.5"),
        level_staleness_minutes: int = 120,
    ) -> None:
        self._bus = bus
        self._min_bounces = min_bounces
        self._min_range_width_pct = min_range_width_pct
        self._breakout_margin_pct = breakout_margin_pct
        self._staleness_seconds = level_staleness_minutes * 60
        self._log = logger.bind(component="range_detector")

        # Per-symbol state
        self._support: dict[Symbol, Decimal] = {}
        self._resistance: dict[Symbol, Decimal] = {}
        self._support_bounces: dict[Symbol, int] = {}
        self._resistance_bounces: dict[Symbol, int] = {}
        self._last_updated: dict[Symbol, float] = {}
        self._in_support_proximity: dict[Symbol, bool] = {}
        self._in_resistance_proximity: dict[Symbol, bool] = {}
        self._current_regime: str = "UNKNOWN"

    async def start(self) -> None:
        self._bus.subscribe(EventType.SIGNAL, self._on_signal, "range_detector_signals")
        self._bus.subscribe(EventType.REGIME_CHANGE, self._on_regime_change, "range_detector_regime")
        self._bus.subscribe(EventType.MARKET_DATA, self._on_market_data, "range_detector_prices")
        self._log.info("range_detector_started")

    async def _on_regime_change(self, event: RegimeChangeEvent) -> None:
        old = self._current_regime
        self._current_regime = event.to_regime
        if old == "SIDEWAYS" and event.to_regime != "SIDEWAYS":
            self._invalidate_all_ranges("regime_change")

    async def _on_signal(self, event: Event) -> None:
        if not isinstance(event, SignalEvent):
            return
        if not event.metadata or event.metadata.get("source") != "SupportResistance":
            return
        if self._current_regime != "SIDEWAYS":
            return

        symbol = event.symbol
        now = time.time()

        if event.action == SignalAction.BUY:
            # Near support
            was_in = self._in_support_proximity.get(symbol, False)
            self._in_support_proximity[symbol] = True
            if not was_in:  # Bounce dedup: only count on re-entry
                self._support_bounces[symbol] = self._support_bounces.get(symbol, 0) + 1
                if event.target_price:
                    self._support[symbol] = event.target_price
                elif symbol not in self._support:
                    # Estimate from signal (use current knowledge)
                    pass
                self._last_updated[symbol] = now
        elif event.action == SignalAction.SELL:
            # Near resistance
            was_in = self._in_resistance_proximity.get(symbol, False)
            self._in_resistance_proximity[symbol] = True
            if not was_in:
                self._resistance_bounces[symbol] = self._resistance_bounces.get(symbol, 0) + 1
                if event.target_price:
                    self._resistance[symbol] = event.target_price
                self._last_updated[symbol] = now

    async def _on_market_data(self, event: Event) -> None:
        """Track when price leaves S/R proximity zones for bounce dedup."""
        from cerebrum.core.events import MarketDataEvent
        if not isinstance(event, MarketDataEvent):
            return
        symbol = event.symbol
        price = event.price

        # Check support proximity exit
        support = self._support.get(symbol)
        if support and support > 0:
            dist = abs(float(price - support) / float(support)) * 100
            if dist > 0.5:  # Left the zone
                self._in_support_proximity[symbol] = False

        # Check resistance proximity exit
        resistance = self._resistance.get(symbol)
        if resistance and resistance > 0:
            dist = abs(float(price - resistance) / float(resistance)) * 100
            if dist > 0.5:
                self._in_resistance_proximity[symbol] = False

        # Check breakout
        if support and price < support * (1 - self._breakout_margin_pct / 100):
            self._invalidate_range(symbol, "support_breakdown")
        if resistance and price > resistance * (1 + self._breakout_margin_pct / 100):
            self._invalidate_range(symbol, "resistance_breakout")

    def get_range(self, symbol: Symbol) -> RangeState | None:
        support = self._support.get(symbol)
        resistance = self._resistance.get(symbol)
        if not support or not resistance:
            return None

        # Staleness check
        last = self._last_updated.get(symbol, 0)
        if time.time() - last > self._staleness_seconds:
            return None

        bounce_count = (
            self._support_bounces.get(symbol, 0)
            + self._resistance_bounces.get(symbol, 0)
        )
        width = (resistance - support) / support * 100 if support > 0 else Decimal("0")
        confirmed = (
            bounce_count >= self._min_bounces
            and width >= self._min_range_width_pct
        )

        return RangeState(
            support_level=support,
            resistance_level=resistance,
            bounce_count=bounce_count,
            range_confirmed=confirmed,
            range_width_pct=Decimal(str(round(float(width), 4))),
            last_updated=last,
        )

    def _invalidate_range(self, symbol: Symbol, reason: str) -> None:
        self._support.pop(symbol, None)
        self._resistance.pop(symbol, None)
        self._support_bounces.pop(symbol, None)
        self._resistance_bounces.pop(symbol, None)
        self._last_updated.pop(symbol, None)
        self._in_support_proximity.pop(symbol, None)
        self._in_resistance_proximity.pop(symbol, None)
        self._log.info("range_invalidated", symbol=symbol, reason=reason)

    def _invalidate_all_ranges(self, reason: str) -> None:
        symbols = list(self._support.keys()) + list(self._resistance.keys())
        for sym in set(symbols):
            self._invalidate_range(sym, reason)
```

Note: The S/R signal generator doesn't set `target_price` on its signals. The RangeDetector will need to extract levels from the signal's `reason` string or directly query the S/R generator's `get_levels()` method. During implementation, check how to best get the actual support/resistance price. The simplest approach: pass the `SupportResistanceSignal` instance to RangeDetector so it can call `get_levels(symbol)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_range_detector.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add cerebrum/strategies/range_detector.py tests/unit/test_range_detector.py
git commit -m "feat: add RangeDetector for SIDEWAYS market range identification"
```

---

### Task 6: Build RangeExitMonitor

**Files:**
- Create: `cerebrum/risk/range_exit_monitor.py`
- Create: `tests/unit/test_range_exit_monitor.py`

- [ ] **Step 1: Write RangeExitMonitor tests**

Test cases:
1. `test_resistance_exit` — price reaches resistance → SELL order emitted
2. `test_support_breakdown_exit` — price falls below support by breakdown_margin → SELL
3. `test_regime_change_exit` — SIDEWAYS→BEAR → all positions exited
4. `test_time_based_exit` — position held > max_hold_minutes → SELL
5. `test_no_exit_mid_range` — price in middle of range, no exit triggered
6. `test_no_exit_without_range` — no confirmed range, falls back to standard behavior

Use real EventBus, mock PortfolioTracker positions, inject RangeDetector state.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_range_exit_monitor.py -v`
Expected: ImportError

- [ ] **Step 3: Implement RangeExitMonitor**

Create `cerebrum/risk/range_exit_monitor.py`. Pattern after `cerebrum/risk/exit_monitor.py` but replace percentage-based exits with structural exits:

- Subscribe to `MARKET_DATA` and `REGIME_CHANGE` events
- On each tick, check open positions against RangeDetector's `get_range()`
- If range exists: use resistance for TP, support-breakdown for SL
- If no range: fall back to configured percentage-based TP/SL
- On regime change from SIDEWAYS: exit all positions immediately
- Time-based: check position age against max_hold_minutes

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_range_exit_monitor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add cerebrum/risk/range_exit_monitor.py tests/unit/test_range_exit_monitor.py
git commit -m "feat: add RangeExitMonitor with structural S/R exits"
```

---

### Task 7: Create RangeTradingStrategy config

**Files:**
- Create: `cerebrum/strategies/range_trading.py`

- [ ] **Step 1: Create the config file**

Follow the pattern from `cerebrum/strategies/mean_reversion.py`. Create `RANGE_TRADING_CONFIG` as a `StrategyConfig` instance with:

```python
from cerebrum.strategies.base import StrategyConfig
from cerebrum.core.types import SignalType
from decimal import Decimal

RANGE_TRADING_CONFIG = StrategyConfig(
    name="range_trading",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.0"),
        SignalType.SENTIMENT: Decimal("0.0"),
        SignalType.NEWS: Decimal("0.0"),
        SignalType.REGIME: Decimal("0.0"),
    },
    aggregator_threshold=Decimal("0.2"),
    signal_source_filter="SupportResistance",
    risk_overrides={
        "min_signal_strength": "0.3",
        "position_size_percent": "2.0",
        "post_fill_cooldown_seconds": 300,
    },
    exit_config={
        "stop_loss_percent": "0.8",
        "take_profit_percent": "1.0",
        "max_position_age_minutes": 60,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "0.2",
    },
    initial_balance=Decimal("2500.00"),  # 25% of $10k (4-strategy split)
    symbols=["BTC/USD", "ETH/USD"],
)
```

Set `exit_monitor_factory` to a callable that creates RangeExitMonitor. This requires importing range_exit_monitor — do a deferred import inside the factory to avoid circular imports.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add cerebrum/strategies/range_trading.py
git commit -m "feat: add RANGE_TRADING_CONFIG for SIDEWAYS market range strategy"
```

---

### Task 8: Wire into StrategyRegistry

**Files:**
- Modify: `cerebrum/strategies/registry.py:254-358` (`_build_pipeline`)
- Modify: `cerebrum/signals/aggregator.py` (pass source filter from StrategyConfig)

- [ ] **Step 1: Modify `_build_pipeline` to use `exit_monitor_factory`**

In `cerebrum/strategies/registry.py:288-316`, change the ExitMonitor construction to:

```python
# --- ExitMonitor — apply exit_config overrides ---
if cfg.exit_monitor_factory:
    exit_monitor = cfg.exit_monitor_factory(
        bus=self._bus,
        portfolio=portfolio,
        config=cfg,
        app_config=self._config,
    )
else:
    # Default ExitMonitor construction (existing code)
    exit_overrides = cfg.exit_config
    exit_monitor = ExitMonitor(...)
```

- [ ] **Step 2: Pass `signal_source_filter` to SignalAggregator**

In `_build_pipeline`, when constructing SignalAggregator (~line 271), pass:

```python
signal_source_filter=cfg.signal_source_filter,
```

- [ ] **Step 3: Pass `exempt_strategies` when constructing global guards**

In `cerebrum/main.py`, where SidewaysSuppressionRule is constructed, add `exempt_strategies={"range_trading"}`.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cerebrum/strategies/registry.py cerebrum/signals/aggregator.py cerebrum/main.py
git commit -m "feat: wire range_trading through StrategyRegistry with source filtering and guard exemption"
```

---

### Task 9: Register range_trading in main.py and add config

**Files:**
- Modify: `cerebrum/main.py` (register RANGE_TRADING_CONFIG)
- Modify: `config/paper.toml` (add range_trading section)

- [ ] **Step 1: Import and register RANGE_TRADING_CONFIG**

In `cerebrum/main.py`, alongside the imports of MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG, add:

```python
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
```

In the `_setup_multi_strategy` method, add:

```python
self.strategy_registry.register(RANGE_TRADING_CONFIG)
```

- [ ] **Step 2: Add config to paper.toml**

```toml
[strategy.range_trading]
enabled = true
preferred_regimes = ["SIDEWAYS"]
position_size_percent = "2.0"
take_profit_percent = "1.0"
stop_loss_percent = "0.8"
max_position_age_minutes = 60
post_fill_cooldown_seconds = 300
aggregation_threshold = "0.2"
min_signal_strength = "0.3"

[strategy.range_trading.range]
min_bounces = 3
breakout_margin_pct = "0.5"
resistance_proximity_pct = "0.3"
support_proximity_pct = "0.3"
min_range_width_pct = "0.6"
max_hold_minutes = 60
level_staleness_minutes = 120
```

- [ ] **Step 3: Update initial_balance for 4-strategy split**

Update MOMENTUM_CONFIG, MEAN_REVERSION_CONFIG, BREAKOUT_CONFIG initial_balance from $3,333.33 to $2,500.00 (25% each for 4 strategies). The conductor will rebalance after its first call.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cerebrum/main.py config/paper.toml cerebrum/strategies/momentum.py cerebrum/strategies/mean_reversion.py cerebrum/strategies/breakout.py
git commit -m "feat: register range_trading strategy, add paper.toml config, rebalance capital allocation"
```

---

### Task 10: Integration test — full range trading cycle

**Files:**
- Create: `tests/unit/test_range_trading_integration.py`

- [ ] **Step 1: Write integration test**

End-to-end test:
1. Create EventBus, all components
2. Set regime to SIDEWAYS
3. Feed S/R BUY signals at support ($69,500) — 3 bounces with price leaving/re-entering proximity zone between each
4. Verify RangeDetector confirms range
5. Feed another BUY signal near support
6. Verify signal passes through aggregator (source filter OK), risk manager (sideways suppression exempted), and produces an OrderEvent
7. Feed a FillEvent
8. Feed price approaching resistance ($70,200)
9. Verify RangeExitMonitor emits SELL OrderEvent
10. Verify existing strategies (momentum) are still DENIED by SidewaysSuppressionRule

- [ ] **Step 2: Run test**

Run: `pytest tests/unit/test_range_trading_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All 440+ tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_range_trading_integration.py
git commit -m "test: add end-to-end range trading integration test"
```

---

## Verification

After all tasks complete:

1. `pytest tests/ -v` — all tests pass (440+)
2. Check Session 8 logs for new `range_trading` strategy registration
3. In next session (Session 9): verify range_trading trades during SIDEWAYS and sits out during BULL/BEAR
4. Verify other strategies remain unaffected
