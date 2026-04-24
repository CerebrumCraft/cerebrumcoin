# Session 23 Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix five data-identified issues causing consistent paper trading losses (-$224/14 days)

**Architecture:** Config-driven tuning + one new risk rule (MaxOpenPositionsRule) + one guard in TradeTracker. All changes follow existing patterns in rules.py.

**Tech Stack:** Python 3.12, structlog, EventBus pub/sub, pydantic BaseSettings config

---

### Task 1: Block UNKNOWN regime trading (DEC-REGIME-006)

**Files:**
- Modify: `cerebrum/risk/rules.py:453-541` (RegimeTradeHaltRule)
- Modify: `cerebrum/core/config.py:277-310` (RegimeConfig)
- Modify: `cerebrum/main.py:380-383,462-465` (RegimeTradeHaltRule wiring)
- Modify: `config/paper.toml:67-80` (regime section)
- Modify: `tests/unit/test_regime_halt.py`

- [ ] **Step 1: Write failing tests for UNKNOWN regime halt**

Add two tests to `tests/unit/test_regime_halt.py`:

```python
@pytest.mark.asyncio
async def test_unknown_regime_halts_trading(bus):
    """UNKNOWN regime should be halted when included in halt_regimes."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR", "UNKNOWN"},
    )
    event = _make_regime_event("BTC/USD", "UNKNOWN", "0.5")
    await rule._on_regime_change(event)

    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)
    assert result.decision == RuleDecision.DENY
    assert "UNKNOWN" in result.reason


@pytest.mark.asyncio
async def test_unknown_regime_allowed_when_not_in_halt_regimes(bus):
    """UNKNOWN should be allowed when halt_regimes only contains BEAR."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR"},
    )
    event = _make_regime_event("BTC/USD", "UNKNOWN", "0.5")
    await rule._on_regime_change(event)

    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_no_regime_data_halts_when_unknown_in_halt_regimes(bus):
    """When no regime event received yet and UNKNOWN is in halt_regimes, should deny."""
    rule = RegimeTradeHaltRule(
        min_confidence=Decimal("0.7"),
        bus=bus,
        halt_regimes={"BEAR", "UNKNOWN"},
    )
    # No regime event published — _regimes is empty for this symbol
    result = rule.evaluate(_make_signal(), _make_order("BTC/USD", "buy"), None)
    assert result.decision == RuleDecision.DENY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_regime_halt.py -v -k "unknown"`
Expected: FAIL — `RegimeTradeHaltRule.__init__` doesn't accept `halt_regimes`

- [ ] **Step 3: Add halt_regimes config field**

In `cerebrum/core/config.py`, add to `RegimeConfig` after `min_hold_count` (~line 310):

```python
    halt_regimes: list[str] = Field(
        default=["BEAR", "UNKNOWN"],
        description=(
            "Regimes that trigger full trade halt. UNKNOWN blocks trading "
            "before the regime detector has enough data (startup/reconnect)."
        ),
    )
```

- [ ] **Step 4: Modify RegimeTradeHaltRule to accept halt_regimes**

In `cerebrum/risk/rules.py`, update `RegimeTradeHaltRule`:

Change `__init__` signature (line 470):
```python
    def __init__(
        self,
        min_confidence: Decimal,
        bus: "EventBus",
        halt_regimes: set[str] | None = None,
    ) -> None:
```

Add after `self._min_confidence = min_confidence` (line 480):
```python
        self._halt_regimes = halt_regimes if halt_regimes is not None else {"BEAR"}
```

Update the `evaluate` method. Replace the current logic (lines 516-541):

```python
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Deny the order if the symbol is in a halted regime."""
        symbol = order.symbol
        regime_info = self._regimes.get(symbol)

        if regime_info is None:
            # No regime data yet (cold start). If UNKNOWN is in halt_regimes,
            # treat missing data as UNKNOWN and deny.
            if "UNKNOWN" in self._halt_regimes:
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason=(
                        f"Trading halted: {symbol} has no regime data yet "
                        f"(UNKNOWN in halt_regimes)"
                    ),
                    risk_level=RiskLevel.HIGH,
                )
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="No regime data — trading allowed",
                risk_level=RiskLevel.LOW,
            )

        regime, confidence = regime_info
        if regime in self._halt_regimes and confidence >= self._min_confidence:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Trading halted: {symbol} in {regime} regime "
                    f"(confidence={float(confidence):.2f}, "
                    f"threshold={float(self._min_confidence):.2f})"
                ),
                risk_level=RiskLevel.HIGH,
            )

        # Special case: UNKNOWN regime halts regardless of confidence
        # (there IS no meaningful confidence for UNKNOWN)
        if regime in self._halt_regimes and regime == "UNKNOWN":
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"Trading halted: {symbol} in {regime} regime (no confidence threshold for UNKNOWN)",
                risk_level=RiskLevel.HIGH,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=f"Regime {regime} does not trigger halt",
            risk_level=RiskLevel.LOW,
        )
```

Update the `@decision` docstring at the top of the class to include DEC-REGIME-006.

- [ ] **Step 5: Wire halt_regimes in main.py**

In `cerebrum/main.py`, update both `RegimeTradeHaltRule` instantiations (lines 380-383 and 462-465):

```python
            RegimeTradeHaltRule(
                min_confidence=Decimal(str(config.regime.bear_halt_min_confidence)),
                bus=self.bus,
                halt_regimes=set(config.regime.halt_regimes),
            ),
```

- [ ] **Step 6: Add halt_regimes to paper.toml**

In `config/paper.toml`, add after `bear_halt_min_confidence` (line 77):

```toml
halt_regimes = ["BEAR", "UNKNOWN"]  # DEC-REGIME-006: block trading before regime detector has data
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_regime_halt.py -v`
Expected: ALL PASS (existing tests still pass because `halt_regimes` defaults to `{"BEAR"}` via backward-compat)

- [ ] **Step 8: Commit**

```bash
git add cerebrum/risk/rules.py cerebrum/core/config.py cerebrum/main.py config/paper.toml tests/unit/test_regime_halt.py
git commit -m "feat: configurable halt_regimes — block UNKNOWN regime trading (DEC-REGIME-006)"
```

---

### Task 2: Remove BTC from mean_reversion (DEC-TUNE-009)

**Files:**
- Modify: `cerebrum/strategies/mean_reversion.py:145`

- [ ] **Step 1: Update mean_reversion symbols**

In `cerebrum/strategies/mean_reversion.py`, change line 145:

From:
```python
    symbols=["BTC/USD", "ETH/USD"],
```

To:
```python
    symbols=["ETH/USD", "SOL/USD", "DOGE/USD"],  # DEC-TUNE-009: BTC removed (8% WR over 14 days)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -x -q`
Expected: All pass (symbols are data, not logic — no test should break)

- [ ] **Step 3: Commit**

```bash
git add cerebrum/strategies/mean_reversion.py
git commit -m "tune: remove BTC from mean_reversion, add SOL+DOGE (DEC-TUNE-009)"
```

---

### Task 3: Reduce max position age to 60 min (DEC-EXIT-005)

**Files:**
- Modify: `config/paper.toml:42`

- [ ] **Step 1: Update max_position_age_minutes**

In `config/paper.toml`, change line 42:

From:
```toml
max_position_age_minutes = 120
```

To:
```toml
max_position_age_minutes = 60  # DEC-EXIT-005: >2hr holds -$216/14d (25% WR); 15-60min 40% WR
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -x -q`
Expected: All pass (config value, no test should hardcode 120)

- [ ] **Step 3: Commit**

```bash
git add config/paper.toml
git commit -m "tune: reduce max_position_age to 60 min (DEC-EXIT-005)"
```

---

### Task 4: Cap open positions per strategy/symbol (DEC-RISK-005)

**Files:**
- Create: `tests/unit/test_max_open_positions.py`
- Modify: `cerebrum/risk/rules.py` (append new class)
- Modify: `cerebrum/core/config.py:42-130` (RiskConfig)
- Modify: `cerebrum/main.py:459-497` (global_guards list)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_max_open_positions.py`:

```python
"""
Tests for MaxOpenPositionsRule.

@decision DEC-TEST-014
@title MaxOpenPositionsRule tests
@status accepted
@rationale Validates position count cap per (strategy, symbol). Uses real EventBus
and FillEvent to test the fill-tracking subscription. Covers: deny at limit,
approve below limit, always approve sells, per-symbol independence, counter
decrement on sell.
"""

import asyncio
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent, OrderEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import MaxOpenPositionsRule, RuleDecision


@pytest.fixture
async def bus():
    """Create and start event bus."""
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


def _make_fill(
    symbol: str = "BTC/USD",
    side: str = "buy",
    strategy_id: str = "mean_reversion",
) -> FillEvent:
    return FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY if side == "buy" else Side.SELL,
        filled_amount=Decimal("0.01"),
        fill_price=Decimal("50000"),
        commission=Decimal("0.08"),
        strategy_id=strategy_id,
    )


def _make_order(
    symbol: str = "BTC/USD",
    side: str = "buy",
    strategy_id: str = "mean_reversion",
) -> OrderEvent:
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY if side == "buy" else Side.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
        status=OrderStatus.PENDING,
        strategy_id=strategy_id,
    )


def _make_signal(symbol: str = "BTC/USD") -> SignalEvent:
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.7"),
    )


@pytest.mark.asyncio
async def test_approve_below_limit(bus):
    """Should approve buy when open count < max."""
    rule = MaxOpenPositionsRule(max_positions=2, bus=bus)
    result = rule.evaluate(_make_signal(), _make_order(), None)
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_deny_at_limit(bus):
    """Should deny buy when open count >= max."""
    rule = MaxOpenPositionsRule(max_positions=2, bus=bus)

    # Simulate 2 buy fills
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))
    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))

    result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "mean_reversion"),
        None,
    )
    assert result.decision == RuleDecision.DENY
    assert "max_open_positions" in result.reason.lower() or "2" in result.reason


@pytest.mark.asyncio
async def test_always_approve_sells(bus):
    """Sell orders should always be approved regardless of position count."""
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)

    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))

    result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "sell", "mean_reversion"),
        None,
    )
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_sell_decrements_counter(bus):
    """After a sell fill, the open count should decrease and allow a new buy."""
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)

    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))

    # At limit — deny
    result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "mean_reversion"),
        None,
    )
    assert result.decision == RuleDecision.DENY

    # Sell fill decrements
    await rule._on_fill(_make_fill("BTC/USD", "sell", "mean_reversion"))

    # Now below limit — approve
    result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "mean_reversion"),
        None,
    )
    assert result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    """BTC fills should not affect ETH position count."""
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)

    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))

    # BTC at limit
    btc_result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "mean_reversion"),
        None,
    )
    assert btc_result.decision == RuleDecision.DENY

    # ETH still open
    eth_result = rule.evaluate(
        _make_signal("ETH/USD"),
        _make_order("ETH/USD", "buy", "mean_reversion"),
        None,
    )
    assert eth_result.decision == RuleDecision.APPROVE


@pytest.mark.asyncio
async def test_per_strategy_independence(bus):
    """mean_reversion fills should not affect range_trading position count."""
    rule = MaxOpenPositionsRule(max_positions=1, bus=bus)

    await rule._on_fill(_make_fill("BTC/USD", "buy", "mean_reversion"))

    # mean_reversion at limit
    mr_result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "mean_reversion"),
        None,
    )
    assert mr_result.decision == RuleDecision.DENY

    # range_trading still open
    rt_result = rule.evaluate(
        _make_signal(),
        _make_order("BTC/USD", "buy", "range_trading"),
        None,
    )
    assert rt_result.decision == RuleDecision.APPROVE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_max_open_positions.py -v`
Expected: FAIL — `MaxOpenPositionsRule` not defined

- [ ] **Step 3: Implement MaxOpenPositionsRule**

Append to `cerebrum/risk/rules.py` after line 1111:

```python


class MaxOpenPositionsRule(RiskRule):
    """
    Cap the number of open positions per (strategy, symbol) pair.

    Subscribes to FillEvents and maintains an in-memory counter. BUY fills
    increment, SELL fills decrement. Orders are denied when the count reaches
    max_positions. SELL orders are always approved (must be able to close).

    @decision DEC-RISK-005
    @title Cap open positions per strategy/symbol
    @status accepted
    @rationale Session 23 data showed 10 DOGE positions piling up in mean_reversion.
    Without a cap, the system accumulates correlated risk and ties up capital in
    multiple entries on the same thesis. Limit of 2 allows one position + one
    averaging opportunity.
    """

    def __init__(
        self,
        max_positions: int,
        bus: "EventBus",
    ) -> None:
        """
        Initialize max open positions rule.

        Args:
            max_positions: Maximum concurrent open positions per (strategy, symbol).
            bus: Event bus to subscribe to FillEvents.
        """
        super().__init__("max_open_positions")
        self._max_positions = max_positions
        # (strategy_id, symbol) -> open position count
        self._open_counts: dict[tuple[str, str], int] = {}

        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="max_open_positions_rule",
        )

        self._log.info(
            "max_open_positions_initialized",
            max_positions=max_positions,
        )

    async def _on_fill(self, event: Event) -> None:
        """Track open position count per (strategy, symbol)."""
        if not isinstance(event, FillEvent):
            return
        strategy = getattr(event, "strategy_id", None) or "unknown"
        key = (strategy, event.symbol)
        if event.side == Side.BUY:
            self._open_counts[key] = self._open_counts.get(key, 0) + 1
        else:  # SELL
            self._open_counts[key] = max(0, self._open_counts.get(key, 0) - 1)
        self._log.debug(
            "position_count_updated",
            strategy=strategy,
            symbol=event.symbol,
            side=event.side.value,
            open_count=self._open_counts[key],
        )

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Deny BUY orders when position count at limit. Always approve SELLs."""
        if order.side == Side.SELL:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="Sell always approved (closing position)",
                risk_level=RiskLevel.LOW,
            )

        strategy = getattr(order, "strategy_id", None) or "unknown"
        key = (strategy, order.symbol)
        count = self._open_counts.get(key, 0)

        if count >= self._max_positions:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Max open positions reached for {strategy}/{order.symbol}: "
                    f"{count} >= {self._max_positions}"
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=f"Open positions {count}/{self._max_positions} for {strategy}/{order.symbol}",
            risk_level=RiskLevel.LOW,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_max_open_positions.py -v`
Expected: ALL PASS

- [ ] **Step 5: Add config field**

In `cerebrum/core/config.py`, add to `RiskConfig` after `macro_volatility_window_size`:

```python
    max_open_positions_per_symbol: int = Field(
        default=2,
        description=(
            "Maximum concurrent open positions per (strategy, symbol) pair. "
            "Prevents position pile-up. Set to 2 for one entry + one averaging opportunity."
        ),
    )
```

- [ ] **Step 6: Wire into main.py**

In `cerebrum/main.py`, add import (line 67 area):

```python
    MaxOpenPositionsRule,
```

Add to `global_guards` list in `_start_multi_strategy` (after `GlobalTradeRateLimitRule`, ~line 496):

```python
            MaxOpenPositionsRule(
                max_positions=config.risk.max_open_positions_per_symbol,
                bus=self.bus,
            ),
```

Also add to the single-strategy `risk_rules` list (after `PostFillCooldownRule`, ~line 387):

```python
            MaxOpenPositionsRule(
                max_positions=config.risk.max_open_positions_per_symbol,
                bus=self.bus,
            ),
```

- [ ] **Step 7: Add config to paper.toml**

In `config/paper.toml`, add after `min_profit_to_commission_ratio` line (~line 59):

```toml
# Max open positions per strategy/symbol pair — prevents position pile-up
max_open_positions_per_symbol = 2  # DEC-RISK-005: 10 DOGE piled up in mean_reversion
```

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add cerebrum/risk/rules.py cerebrum/core/config.py cerebrum/main.py config/paper.toml tests/unit/test_max_open_positions.py
git commit -m "feat: MaxOpenPositionsRule — cap 2 per strategy/symbol (DEC-RISK-005)"
```

---

### Task 5: Guard against null strategy_id in TradeTracker (DEC-CLEANUP-002)

**Files:**
- Modify: `cerebrum/learning/tracker.py:78-124`
- Create: `tests/unit/test_tracker_null_strategy.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_tracker_null_strategy.py`:

```python
"""
Test that TradeTracker skips fills with no strategy_id.

@decision DEC-TEST-015
@title Null strategy_id guard in TradeTracker
@status accepted
@rationale 45 orphan trades with strategy_id=NULL caused -$166 in losses.
Guard prevents new orphans from being tracked.
"""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import FillEvent
from cerebrum.core.types import EventType, Side


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_null_strategy_fill_skipped(bus):
    """BUY fill with no strategy_id should not open a trade."""
    from cerebrum.learning.tracker import TradeTracker
    from cerebrum.learning.state import StateManager

    state = AsyncMock(spec=StateManager)
    state.get_open_trades = AsyncMock(return_value=[])
    state.save_trade = AsyncMock(return_value=1)

    tracker = TradeTracker(bus=bus, state=state, current_regime="SIDEWAYS")

    fill = FillEvent(
        event_type=EventType.FILL,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol="BTC/USD",
        side=Side.BUY,
        filled_amount=Decimal("0.01"),
        fill_price=Decimal("50000"),
        commission=Decimal("0.08"),
        strategy_id=None,  # No strategy
    )

    await tracker._on_fill(fill)

    # save_trade should NOT have been called
    state.save_trade.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_tracker_null_strategy.py -v`
Expected: FAIL — `save_trade` IS called (no guard exists yet)

- [ ] **Step 3: Add null strategy_id guard**

In `cerebrum/learning/tracker.py`, add at the start of `_on_fill` method (after line 78's docstring block, before `signal_snapshot = self._pending_signals.pop(...)`):

```python
        # DEC-CLEANUP-002: skip fills with no strategy_id to prevent orphan trades.
        # Pre-Session-7 fills and legacy single-strategy code paths may emit fills
        # without strategy_id — these create unattributed trades that pile up losses.
        strategy_id = getattr(event, 'strategy_id', None)
        if strategy_id is None:
            self._log.warning(
                "null_strategy_fill_skipped",
                symbol=event.symbol,
                side=event.side.value,
                order_id=event.order_id,
                reason="no_strategy_id",
            )
            return
```

Insert this before `signal_snapshot = self._pending_signals.pop(event.order_id, {})` (currently line 96).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_tracker_null_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add cerebrum/learning/tracker.py tests/unit/test_tracker_null_strategy.py
git commit -m "fix: guard against null strategy_id fills in TradeTracker (DEC-CLEANUP-002)"
```

---

### Task 6: Run orphan cleanup script

**Files:**
- Run: `scripts/fix_orphaned_trades.py`

- [ ] **Step 1: Back up the database**

```bash
cp data/cerebrum.db data/cerebrum_pre_cleanup_backup.db
```

- [ ] **Step 2: Check current orphan count**

```bash
python3 -c "
import sqlite3
db = sqlite3.connect('data/cerebrum.db')
ct = db.execute('SELECT COUNT(*) FROM trades WHERE strategy_id IS NULL AND status=\"OPEN\"').fetchone()[0]
print(f'Open orphan trades: {ct}')
"
```

- [ ] **Step 3: Run cleanup script**

```bash
python scripts/fix_orphaned_trades.py data/cerebrum.db
```

- [ ] **Step 4: Verify orphans are closed**

```bash
python3 -c "
import sqlite3
db = sqlite3.connect('data/cerebrum.db')
ct = db.execute('SELECT COUNT(*) FROM trades WHERE strategy_id IS NULL AND status=\"OPEN\"').fetchone()[0]
print(f'Open orphan trades after cleanup: {ct}')
"
```

Expected: 0 open orphans

- [ ] **Step 5: No commit needed** — DB changes are data, not code

---

### Task 7: Full regression test + verify

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All pass (766+ tests), no regressions

- [ ] **Step 2: Verify config changes parse correctly**

```bash
python3 -c "
from cerebrum.core.config import Config
c = Config.from_toml('config/paper.toml')
print(f'halt_regimes: {c.regime.halt_regimes}')
print(f'max_position_age_minutes: {c.risk.max_position_age_minutes}')
print(f'max_open_positions_per_symbol: {c.risk.max_open_positions_per_symbol}')
print(f'mean_reversion symbols: check strategy file')
"
```

Expected:
- `halt_regimes: ['BEAR', 'UNKNOWN']`
- `max_position_age_minutes: 60`
- `max_open_positions_per_symbol: 2`
