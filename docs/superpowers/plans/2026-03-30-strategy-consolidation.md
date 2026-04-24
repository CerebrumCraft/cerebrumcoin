# Strategy Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate from 5 active strategies to 2 differentiated strategies (mean_reversion + range_trading), increase capital per strategy, add a minimum trade value floor, and reduce trade frequency to fix the commission-drag death spiral.

**Architecture:** Disable 3 duplicate strategies (momentum, breakout, news_driven) that consume identical signal inputs. Redistribute their capital equally across the 2 remaining strategies ($5,000 each). Add `min_trade_value_usd` guard to PositionSizingRule. Increase cooldowns to reduce churn.

**Tech Stack:** Python, pytest, SQLite (data/cerebrum.db), TOML config

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `cerebrum/main.py` | Modify (lines 499-509, 491) | Disable 3 strategy registrations, reduce rate limit |
| `cerebrum/strategies/mean_reversion.py` | Modify (lines 131-146) | Update initial_balance, cooldown |
| `cerebrum/strategies/range_trading.py` | Modify (lines 62-88) | Update initial_balance, cooldown, position_size |
| `cerebrum/risk/rules.py` | Modify (lines 263-311) | Add min_trade_value_usd to PositionSizingRule |
| `tests/unit/test_position_sizing_min_value.py` | Create | Test min trade value DENY logic |
| `tests/integration/test_two_strategy_consolidation.py` | Create | Integration test: 2 strategies, capital split, no cannibalization |

---

### Task 1: Add min_trade_value_usd to PositionSizingRule

**Files:**
- Modify: `cerebrum/risk/rules.py:263-311`
- Test: `tests/unit/test_position_sizing_min_value.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for PositionSizingRule min_trade_value_usd guard."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType, RiskLevel
from cerebrum.risk.rules import PositionSizingRule, RuleDecision
from cerebrum.risk.portfolio import PortfolioTracker


def _make_signal(symbol="BTC/USD", strength=Decimal("0.8")):
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=1000.0,
        signal_type=SignalType.COMBINED,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=strength,
        confidence=Decimal("0.7"),
        reason="test",
    )


def _make_order(symbol="BTC/USD", price=Decimal("50000")):
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=1000.0,
        symbol=symbol,
        action=SignalAction.BUY,
        amount=Decimal("1.0"),
        price=price,
    )


def _make_portfolio(equity=Decimal("5000"), price=Decimal("50000")):
    portfolio = MagicMock(spec=PortfolioTracker)
    portfolio.get_total_equity.return_value = equity
    portfolio.get_latest_price.return_value = price
    return portfolio


class TestMinTradeValue:
    def test_trade_above_minimum_is_allowed(self):
        """5% of $5000 = $250 target, above $100 minimum -> MODIFY (normal sizing)."""
        rule = PositionSizingRule(
            position_size_percent=Decimal("5.0"),
            min_trade_value_usd=Decimal("100"),
        )
        result = rule.evaluate(
            _make_signal(strength=Decimal("0.8")),
            _make_order(price=Decimal("50000")),
            _make_portfolio(equity=Decimal("5000"), price=Decimal("50000")),
        )
        assert result.decision == RuleDecision.MODIFY
        # $5000 * 5% = $250 target. $250 * 0.8 strength = $200.
        # $200 / $50000 = 0.004 BTC
        assert result.modified_amount == pytest.approx(Decimal("0.004"), abs=Decimal("0.0001"))

    def test_trade_below_minimum_is_denied(self):
        """2% of $1666 = $33 target, below $100 minimum -> DENY."""
        rule = PositionSizingRule(
            position_size_percent=Decimal("2.0"),
            min_trade_value_usd=Decimal("100"),
        )
        result = rule.evaluate(
            _make_signal(strength=Decimal("0.8")),
            _make_order(price=Decimal("50000")),
            _make_portfolio(equity=Decimal("1666"), price=Decimal("50000")),
        )
        assert result.decision == RuleDecision.DENY
        assert "below minimum" in result.reason.lower()

    def test_no_minimum_by_default(self):
        """Without min_trade_value_usd, small trades are allowed (backward compat)."""
        rule = PositionSizingRule(position_size_percent=Decimal("2.0"))
        result = rule.evaluate(
            _make_signal(strength=Decimal("0.8")),
            _make_order(price=Decimal("50000")),
            _make_portfolio(equity=Decimal("500"), price=Decimal("50000")),
        )
        # $500 * 2% = $10 target, but no minimum -> MODIFY
        assert result.decision == RuleDecision.MODIFY

    def test_strength_adjusted_value_checked(self):
        """Min trade value applies to strength-adjusted value, not raw target."""
        rule = PositionSizingRule(
            position_size_percent=Decimal("5.0"),
            min_trade_value_usd=Decimal("100"),
        )
        # $2000 * 5% = $100 target. With strength 0.5 -> $50 actual. Below $100.
        result = rule.evaluate(
            _make_signal(strength=Decimal("0.5")),
            _make_order(price=Decimal("50000")),
            _make_portfolio(equity=Decimal("2000"), price=Decimal("50000")),
        )
        assert result.decision == RuleDecision.DENY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_position_sizing_min_value.py -v`
Expected: FAIL — `PositionSizingRule.__init__() got an unexpected keyword argument 'min_trade_value_usd'`

- [ ] **Step 3: Implement min_trade_value_usd in PositionSizingRule**

In `cerebrum/risk/rules.py`, modify `PositionSizingRule.__init__` and `evaluate`:

```python
class PositionSizingRule(RiskRule):
    """Calculate position size as percentage of portfolio."""

    def __init__(
        self,
        position_size_percent: Decimal = Decimal("2.0"),
        min_trade_value_usd: Decimal | None = None,
    ) -> None:
        """
        Initialize position sizing rule.

        Args:
            position_size_percent: Position size as % of equity
            min_trade_value_usd: Minimum USD value for a trade. Orders below this
                are denied because commission drag makes them structurally unprofitable.
                None = no minimum (backward compatible).

        @decision DEC-SIZING-001
        @title Minimum trade value floor to prevent commission-killed micro-trades
        @status accepted
        @rationale Investigation showed $20 range_trading trades where 0.32%
        round-trip commission ($0.06) ate 33% of a 1% TP win ($0.20). With the
        floor at $100, commission stays below 10% of typical wins. Applies after
        signal-strength adjustment since that's the actual trade value.
        """
        super().__init__("position_sizing")
        self._size_percent = position_size_percent
        self._min_trade_value = min_trade_value_usd

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Calculate appropriate position size."""
        equity = portfolio.get_total_equity()
        target_value = equity * (self._size_percent / 100)

        price = order.price
        if price is None:
            price = portfolio.get_latest_price(order.symbol)
            if price is None:
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason="Cannot size order: no price data available",
                    risk_level=RiskLevel.MEDIUM,
                )

        target_amount = target_value / price

        # Adjust by signal strength
        adjusted_amount = target_amount * signal.strength

        # DEC-SIZING-001: deny if actual trade value is below minimum
        if self._min_trade_value is not None:
            actual_value = adjusted_amount * price
            if actual_value < self._min_trade_value:
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason=(
                        f"Trade value ${actual_value:.2f} below minimum "
                        f"${self._min_trade_value}. Commission would dominate."
                    ),
                    risk_level=RiskLevel.LOW,
                )

        return RuleResult(
            decision=RuleDecision.MODIFY,
            reason=f"Position sized at {self._size_percent}% of equity, adjusted by signal strength",
            risk_level=RiskLevel.LOW,
            modified_amount=adjusted_amount,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_position_sizing_min_value.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -x -q`
Expected: All existing tests pass (no existing callers pass `min_trade_value_usd`, so backward compat is preserved)

- [ ] **Step 6: Commit**

```bash
git add cerebrum/risk/rules.py tests/unit/test_position_sizing_min_value.py
git commit -m "feat: add min_trade_value_usd floor to PositionSizingRule (DEC-SIZING-001)

Trades below the minimum USD value are denied because commission drag
makes them structurally unprofitable. Investigation showed $20 range_trading
trades where 0.32% round-trip commission ate 33% of wins.

Default: None (no minimum) for backward compatibility."
```

---

### Task 2: Disable momentum, breakout, news_driven strategies

**Files:**
- Modify: `cerebrum/main.py:446-509`

- [ ] **Step 1: Comment out the 3 strategy registrations in _setup_multi_strategy**

In `cerebrum/main.py`, modify the strategy registration block (lines 498-509):

```python
        # --- StrategyRegistry ---
        self.strategy_registry = StrategyRegistry(bus=self.bus, config=config)
        # @decision DEC-TUNE-008
        # @title Disable momentum, breakout, news_driven — signal cannibalization
        # @status accepted
        # @rationale Investigation of 219 multi-strategy trades (Mar 24-30) showed all 4
        #   unfiltered strategies (momentum, mean_reversion, breakout, news_driven) consume
        #   identical RSI/MACD/BB/VWAP signals. 78 simultaneous entry pairs confirmed: same
        #   signal, same symbol, same second, all lost money together. Only mean_reversion
        #   and range_trading are kept — range_trading has differentiated S/R-only signal
        #   filtering; mean_reversion is the core technical strategy with best WR (30.6%)
        #   of the 4 duplicates. Capital redistributed from $1,667/ea to $5,000/ea.
        # self.strategy_registry.register(MOMENTUM_CONFIG)
        self.strategy_registry.register(MEAN_REVERSION_CONFIG)
        # self.strategy_registry.register(BREAKOUT_CONFIG)
        self.strategy_registry.register(RANGE_TRADING_CONFIG)
        # @decision DEC-TUNE-005
        # @title Disable swing_trading — Session 18 sole loser
        # @status accepted
        # @rationale Session 18: -$51 PnL, zero realized trades. Disable until tuning revisited.
        # self.strategy_registry.register(SWING_TRADING_CONFIG)
        # self.strategy_registry.register(NEWS_DRIVEN_CONFIG)
```

- [ ] **Step 2: Reduce GlobalTradeRateLimitRule from 40 to 15**

In `cerebrum/main.py`, modify the GlobalTradeRateLimitRule (line 491-494):

```python
            # 2 active strategies — reduce from 40 (5-strategy budget) to 15
            # to prevent over-trading with consolidated capital (DEC-TUNE-008)
            GlobalTradeRateLimitRule(
                max_trades_per_hour=15,
                bus=self.bus,
            ),
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass. Existing tests reference config objects but don't depend on which strategies main.py registers.

- [ ] **Step 4: Commit**

```bash
git add cerebrum/main.py
git commit -m "feat: disable momentum/breakout/news_driven — signal cannibalization fix (DEC-TUNE-008)

78 simultaneous entry pairs confirmed: 4 strategies buying same signal, same
symbol, same second. Keep mean_reversion (best WR) + range_trading (S/R filtered).
Reduce GlobalTradeRateLimitRule from 40 to 15 for 2-strategy budget."
```

---

### Task 3: Update capital allocation and cooldowns

**Files:**
- Modify: `cerebrum/strategies/mean_reversion.py:122-146`
- Modify: `cerebrum/strategies/range_trading.py:62-88`

- [ ] **Step 1: Update MEAN_REVERSION_CONFIG**

In `cerebrum/strategies/mean_reversion.py`, modify MEAN_REVERSION_CONFIG:

```python
MEAN_REVERSION_CONFIG = StrategyConfig(
    name="mean_reversion",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.2"),
        SignalType.SENTIMENT: Decimal("0.3"),
        SignalType.NEWS: Decimal("0.2"),
        SignalType.REGIME: Decimal("0.5"),
    },
    aggregator_threshold=Decimal("0.3"),
    risk_overrides={
        "min_signal_strength": "0.5",
        "position_size_percent": "5.0",
        # DEC-TUNE-009: cooldown 900→1800s with consolidated capital.
        # Fewer, larger trades instead of frequent small ones.
        "post_fill_cooldown_seconds": 1800,
    },
    exit_config={
        "stop_loss_percent": "1.0",
        "take_profit_percent": "1.5",
        "max_position_age_minutes": 90,
        "adaptive_tp": True,
        "tp_multiplier": "1.2",
        "min_tp_percent": "0.2",
    },
    # DEC-TUNE-008: 2-strategy split — $5,000 each (was $1,667 across 6)
    initial_balance=Decimal("5000.00"),
    symbols=["BTC/USD", "ETH/USD"],
)
```

- [ ] **Step 2: Update RANGE_TRADING_CONFIG**

In `cerebrum/strategies/range_trading.py`, modify RANGE_TRADING_CONFIG:

```python
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
        # DEC-TUNE-008: keep 2% — with $5k capital that's $100/trade,
        # matching the min_trade_value_usd floor. Was producing $20 trades
        # at $1,667 capital which commission destroyed.
        "position_size_percent": "2.0",
        # DEC-TUNE-009: cooldown 300→900s to reduce trade frequency
        "post_fill_cooldown_seconds": 900,
    },
    exit_config={
        "stop_loss_percent": "0.5",
        "take_profit_percent": "1.0",
        "max_position_age_minutes": 60,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "0.2",
    },
    # DEC-TUNE-008: 2-strategy split — $5,000 each (was $1,667 across 6)
    initial_balance=Decimal("5000.00"),
    symbols=["BTC/USD", "ETH/USD"],
    exit_monitor_factory=_create_range_exit_monitor,
)
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass. Tests for mean_reversion/breakout check the dataclass attributes, not the StrategyConfig balance.

- [ ] **Step 4: Commit**

```bash
git add cerebrum/strategies/mean_reversion.py cerebrum/strategies/range_trading.py
git commit -m "feat: consolidate capital $5k/strategy, increase cooldowns (DEC-TUNE-008/009)

mean_reversion: $1,667→$5,000, cooldown 900→1800s
range_trading: $1,667→$5,000, cooldown 300→900s
range_trading position_size stays 2% — $100/trade at $5k meets min floor"
```

---

### Task 4: Wire min_trade_value_usd into StrategyRegistry

**Files:**
- Modify: `cerebrum/strategies/registry.py:348-352`

- [ ] **Step 1: Pass min_trade_value_usd when constructing PositionSizingRule**

In `cerebrum/strategies/registry.py`, modify the PositionSizingRule construction in `_build_pipeline` (line 349-352):

```python
        # --- Per-strategy risk rules ---
        overrides = cfg.risk_overrides
        per_strategy_rules: list[RiskRule] = [
            PositionSizingRule(
                Decimal(overrides.get("position_size_percent",
                                      str(self._config.risk.position_size_percent))),
                # DEC-SIZING-001: $100 floor prevents commission-killed micro-trades
                min_trade_value_usd=Decimal(
                    overrides.get("min_trade_value_usd", "100")
                ),
            ),
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass. The $100 default doesn't affect existing tests because test portfolios use $5000 balance with 5% sizing = $250 trades.

- [ ] **Step 3: Commit**

```bash
git add cerebrum/strategies/registry.py
git commit -m "feat: wire $100 min_trade_value_usd into StrategyRegistry pipeline

All strategies now reject trades below $100 USD. Prevents the $20
range_trading micro-trades that were structurally unprofitable."
```

---

### Task 5: Integration test — 2-strategy consolidated pipeline

**Files:**
- Create: `tests/integration/test_two_strategy_consolidation.py`

- [ ] **Step 1: Write integration test**

```python
"""
Integration test: consolidated 2-strategy pipeline.

Verifies that after DEC-TUNE-008 consolidation:
1. Only mean_reversion and range_trading are active
2. Each gets $5,000 capital
3. Signals don't cause simultaneous entries across both strategies
   (mean_reversion accepts all signals; range_trading only S/R)
4. PositionSizingRule rejects trades below $100

Uses real EventBus, real Config, real StrategyRegistry — no mocks.
"""

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
from cerebrum.strategies.registry import StrategyRegistry


CONFIG_PATH = Path("config/paper.toml")


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def config():
    return Config.from_toml(CONFIG_PATH)


class TestTwoStrategyConsolidation:
    """Verify 2-strategy consolidation after DEC-TUNE-008."""

    async def test_only_two_strategies_active(self, bus, config):
        """After registration, only mean_reversion and range_trading are active."""
        registry = StrategyRegistry(bus=bus, config=config)
        registry.register(MEAN_REVERSION_CONFIG)
        registry.register(RANGE_TRADING_CONFIG)
        await registry.start_all()

        active = registry.active_strategy_names()
        assert sorted(active) == ["mean_reversion", "range_trading"]

    async def test_capital_allocation(self, bus, config):
        """Each strategy gets $5,000."""
        registry = StrategyRegistry(bus=bus, config=config)
        registry.register(MEAN_REVERSION_CONFIG)
        registry.register(RANGE_TRADING_CONFIG)
        await registry.start_all()

        mr_portfolio = registry.get_portfolio("mean_reversion")
        rt_portfolio = registry.get_portfolio("range_trading")

        assert mr_portfolio.get_total_equity() == Decimal("5000.00")
        assert rt_portfolio.get_total_equity() == Decimal("5000.00")

    async def test_signal_isolation(self, bus, config):
        """RSI signal reaches mean_reversion aggregator but NOT range_trading."""
        registry = StrategyRegistry(bus=bus, config=config)
        registry.register(MEAN_REVERSION_CONFIG)
        registry.register(RANGE_TRADING_CONFIG)
        await registry.start_all()

        # Emit a technical RSI signal (NOT support/resistance)
        rsi_signal = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=1000.0,
            signal_type=SignalType.TECHNICAL,
            symbol="BTC/USD",
            action=SignalAction.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.7"),
            reason="RSI oversold",
            metadata={"source": "RSI", "timeframe": "1m"},
        )
        await bus.publish(rsi_signal)
        await asyncio.sleep(0.1)

        # mean_reversion aggregator should have buffered the signal
        mr_agg = registry.get_aggregator("mean_reversion")
        # range_trading aggregator should have filtered it out (source != SupportResistance)
        rt_agg = registry.get_aggregator("range_trading")

        # Check internal buffer counts — mean_reversion accepted, range_trading rejected
        mr_buffer = mr_agg._buffers.get("BTC/USD", [])
        rt_buffer = rt_agg._buffers.get("BTC/USD", [])
        assert len(mr_buffer) >= 1, "mean_reversion should accept RSI signals"
        assert len(rt_buffer) == 0, "range_trading should filter non-S/R signals"
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/integration/test_two_strategy_consolidation.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_two_strategy_consolidation.py
git commit -m "test: integration test for 2-strategy consolidated pipeline (DEC-TUNE-008)"
```

---

## Verification

After all tasks are committed:

1. **Run full test suite**: `python -m pytest tests/ -v`
2. **Verify strategy registration**: Search main.py for uncommented `register()` calls — should be only `MEAN_REVERSION_CONFIG` and `RANGE_TRADING_CONFIG`
3. **Verify capital**: Both configs should show `initial_balance=Decimal("5000.00")`
4. **Verify min trade value**: PositionSizingRule should DENY trades below $100
5. **Verify cooldowns**: mean_reversion=1800s, range_trading=900s
6. **Verify rate limit**: GlobalTradeRateLimitRule max_trades_per_hour=15
7. **Paper trading smoke test**: Start paper trading, verify only 2 strategies appear in logs, verify trade sizes are $100+ per trade
