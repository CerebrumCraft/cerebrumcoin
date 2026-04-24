# Session 23 Tuning — Design Spec

## Problem

Paper trading is consistently unprofitable over the last 14 days: 272 closed trades, -$224 net P&L, only 1 profitable day out of 10. Data analysis reveals five fixable issues, ranked by impact.

## Data-Driven Changes

### 1. Block UNKNOWN regime trading (DEC-REGIME-006)

**Impact:** -$168.50 over 14 days (57 trades, 12% WR)
**Root cause:** Regime detector starts at UNKNOWN per symbol until ~30 ticks arrive. Trades enter before the detector has data.
**Fix:** Make `RegimeTradeHaltRule` accept a configurable set of halt regimes instead of hardcoding BEAR only.

**File:** `cerebrum/risk/rules.py` — `RegimeTradeHaltRule`
- Add `halt_regimes: set[str]` parameter (default `{"BEAR", "UNKNOWN"}`)
- Change condition from `regime == "BEAR"` to `regime in self._halt_regimes`
- Confidence check only applies when regime has confidence data; UNKNOWN with no confidence still halts

**Config:** `config/paper.toml`
```toml
halt_regimes = ["BEAR", "UNKNOWN"]
```

### 2. Remove BTC from mean_reversion (DEC-TUNE-009)

**Impact:** BTC 8% WR, -$87 over 14 days across strategies; mean_reversion specifically -$31.89
**Fix:** Remove `"BTC/USD"` from mean_reversion symbol list.

**File:** `cerebrum/strategies/mean_reversion.py` — symbols field
- Change from `["BTC/USD", "ETH/USD"]` to `["ETH/USD", "SOL/USD", "DOGE/USD"]`
- Rationale: BTC's low volatility relative to its price makes mean reversion signals unreliable. SOL and DOGE have better mean reversion characteristics (SOL 33% WR, best performer).

### 3. Reduce max position age to 60 min (DEC-EXIT-005)

**Impact:** >2hr holds: 148 trades, 25% WR, -$215.84 (avg -$1.46/trade). 15-60min holds: 42 trades, 40% WR, -$3.09 (avg -$0.07/trade).
**Fix:** Change `max_position_age_minutes` from 120 to 60 in paper.toml.

**File:** `config/paper.toml`
```toml
max_position_age_minutes = 60
```

Also update `RangeExitMonitor` `max_hold_minutes` if it has a separate config (currently defaults to 60 already — confirm).

### 4. Cap open positions per strategy/symbol (DEC-RISK-005)

**Impact:** 22 open trades piling up (10 DOGE in mean_reversion alone). Ties up capital and creates correlated risk.
**Fix:** New `MaxOpenPositionsRule` risk rule.

**File:** `cerebrum/risk/rules.py` — new class

Pattern: subscribe to fill events (like `PostFillCooldownRule`) and maintain an in-memory counter of open positions per `(strategy_id, symbol)`. Increment on buy fill, decrement on sell fill. `evaluate()` is sync — it reads the counter, no async DB queries needed.

```python
class MaxOpenPositionsRule(RiskRule):
    """Deny new buys when open position count per strategy+symbol exceeds limit."""
    
    def __init__(self, max_positions: int, bus: EventBus):
        self._max_positions = max_positions
        self._open_counts: dict[tuple[str, str], int] = {}  # (strategy_id, symbol) -> count
        # Subscribe to fill events to track open/close
    
    def _on_fill(self, event: FillEvent):
        key = (event.strategy_id, event.symbol)
        if event.side == "buy":
            self._open_counts[key] = self._open_counts.get(key, 0) + 1
        elif event.side == "sell":
            self._open_counts[key] = max(0, self._open_counts.get(key, 0) - 1)
    
    def evaluate(self, signal, order, portfolio) -> RuleResult:
        key = (order.strategy_id, order.symbol)
        if order.side == "buy" and self._open_counts.get(key, 0) >= self._max_positions:
            return DENY
        return APPROVE
```

**Config:** `config/paper.toml`
```toml
max_open_positions_per_symbol = 2
```

**Wiring:** Add to RiskManager in `cerebrum/main.py` alongside existing rules.

### 5. Close null-strategy orphan trades (DEC-CLEANUP-002)

**Impact:** 45 null-strategy trades in last 14 days, -$166.62. These are pre-Session-7 legacy trades with no strategy_id.
**Fix:** Two parts:
- A) Run `scripts/fix_orphaned_trades.py` to mark existing orphans as CLOSED in the DB
- B) Add a guard in TradeTracker to skip fills with no strategy_id, preventing new orphan trades from being tracked

**File:** `cerebrum/learning/tracker.py` — `_on_fill`
- If `strategy_id is None`, log a warning and return early (don't open a trade)

## Files to Modify

| File | Change |
|------|--------|
| `cerebrum/risk/rules.py` | Configurable halt_regimes + new MaxOpenPositionsRule |
| `cerebrum/strategies/mean_reversion.py` | Remove BTC/USD from symbols |
| `config/paper.toml` | max_age 60, halt_regimes, max_open_positions |
| `cerebrum/main.py` | Wire MaxOpenPositionsRule into RiskManager |
| `cerebrum/learning/tracker.py` | Guard against null strategy_id fills |
| `cerebrum/core/config.py` | Add halt_regimes and max_open_positions_per_symbol fields |

## Testing

- Unit tests for MaxOpenPositionsRule (deny at limit, approve below, always approve sells)
- Unit test for RegimeTradeHaltRule with UNKNOWN regime
- Unit test for TradeTracker null strategy_id guard
- Existing test suite must pass (766 tests)
- Run paper trading session and verify: no UNKNOWN trades, no >60min holds, no position pile-up

## Verification

1. `python -m pytest tests/ -x -q` — full suite passes
2. Start bot, watch logs for UNKNOWN regime denials in first few minutes
3. After 1 hour, check DB: no new null-strategy trades, no positions >60min old
4. After 4 hours, check position counts stay at 2 or below per strategy/symbol
