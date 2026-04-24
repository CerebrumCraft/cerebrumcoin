# Range Trading Strategy for SIDEWAYS Markets

## Problem

CerebrumCoin currently sits idle during SIDEWAYS markets. Session 5 proved that the momentum strategy loses 100% of the time in tight ranges (0/17, -$61). Guards were added (SidewaysSuppressionRule, VolatilityGateRule, MacroVolatilityGateRule) to prevent this — correctly blocking trend-following strategies from trading in SIDEWAYS.

However, SIDEWAYS markets are tradeable with the right approach: buying near support and selling near resistance within an established range. The system has S/R signal generation (Phase 11B) and a mean_reversion strategy, but mean_reversion is blocked by the same global guards and uses the same RSI/MACD signals (just with different weights) — it's not a true range trader.

## Solution

A dedicated **RangeTrading strategy** (4th strategy) that:
- Only activates in SIDEWAYS regime
- Uses S/R levels as the primary entry signal (not RSI/MACD)
- Requires a confirmed range (3+ bounces) before entering
- Exits structurally at resistance (not a fixed % take-profit)
- Is exempt from SidewaysSuppressionRule (that's its home turf)

## Architecture

### New Components

#### 1. RangeDetector (`cerebrum/strategies/range_detector.py`)

Monitors S/R signals and regime state to identify tradeable ranges.

**Inputs:**
- Subscribes to `EventType.SIGNAL` (filters for S/R signal type from `SupportResistanceSignalGenerator`)
- Subscribes to `EventType.REGIME_CHANGE`

**State (per symbol):**
- `support_level: Decimal | None` — current support price
- `resistance_level: Decimal | None` — current resistance price
- `bounce_count: int` — total confirmed bounces (support + resistance)
- `range_confirmed: bool` — True when `bounce_count >= min_bounces`
- `range_width_pct: Decimal` — `(resistance - support) / support * 100`

**Behavior:**
- On S/R BUY signal (price near support): increment support bounces, update support level
- On S/R SELL signal (price near resistance): increment resistance bounces, update resistance level
- Range confirmed when `bounce_count >= 3` (configurable via `min_bounces`)
- Range invalidated when regime changes from SIDEWAYS to BULL/BEAR/VOLATILE
- Range invalidated when price breaks below support or above resistance by > `breakout_margin_pct` (default 0.5%)
- Exposes `get_range(symbol) -> RangeState | None` query method (no custom event type needed — simpler than a new EventType)
- RangeState is a frozen dataclass: `support_level`, `resistance_level`, `bounce_count`, `range_confirmed`, `range_width_pct`, `last_updated`
- S/R levels expire after `level_staleness_minutes` (default 120 min) — prevents trading on stale levels
- **Bounce deduplication**: A bounce is only counted when price *leaves* the proximity zone and *re-enters* it. Continuous ticks near support do not increment the counter. Tracked via a `_in_proximity: dict[symbol, bool]` flag per level.

#### 2. RangeTradingStrategy config (`cerebrum/strategies/range_trading.py`)

A `StrategyConfig` dataclass (same pattern as momentum/mean_reversion/breakout).

**Signal filtering:** S/R signals and RSI/MACD signals are both `SignalType.TECHNICAL`. The range strategy cannot use the standard aggregator weight system to boost S/R over RSI/MACD. Instead, the range strategy's SignalAggregator uses a `signal_source_filter` — it only accepts signals from `SupportResistanceSignalGenerator` (identified by signal metadata `source="support_resistance"`). RSI/MACD/Bollinger signals are dropped entirely. This requires adding an optional `source` field to SignalEvent metadata (set by each signal generator) and a filter in the aggregator.

**Config values:**
```python
name = "range_trading"
preferred_regimes = ("SIDEWAYS",)
signal_source_filter = "support_resistance"  # Only accept S/R signals
signal_weights = {
    SignalType.TECHNICAL: Decimal("1.0"),   # S/R is the only technical signal (filtered)
    SignalType.SENTIMENT: Decimal("0.0"),   # Ignore sentiment entirely
    SignalType.NEWS: Decimal("0.0"),        # Ignore news entirely
    SignalType.REGIME: Decimal("0.0"),      # Regime handled by activation, not weighting
}
aggregation_threshold = Decimal("0.2")      # Low threshold — S/R alone should be sufficient
min_signal_strength = Decimal("0.3")        # Accept S/R signals with 2+ touches
position_size_percent = Decimal("2.0")      # Small positions (tight range = lower conviction)
take_profit_percent = Decimal("1.0")        # Fallback TP if structural exit fails
stop_loss_percent = Decimal("0.8")          # Tight stop — below support = range broke
max_position_age_minutes = 60               # Shorter timeout — ranges resolve faster
post_fill_cooldown_seconds = 300            # 5 min between entries (not too aggressive)
```

#### 3. RangeExitMonitor (`cerebrum/risk/range_exit_monitor.py`)

Specialized exit monitor that uses structural levels rather than fixed percentages.

**Exit triggers (evaluated on every MARKET_DATA tick for open range positions):**

1. **Resistance exit (primary):** Price within 0.3% of resistance level → SELL
   - This is the profit target — "sell at the top of the range"
   - Uses RangeDetector's current resistance level

2. **Support breakdown (stop-loss):** Price falls below support by `breakdown_margin_pct` (default 0.5%) → SELL
   - Range has broken down — exit with loss
   - More meaningful than a fixed % stop because it's tied to market structure

3. **Range invalidation:** Regime changes from SIDEWAYS → SELL all range positions
   - A breakout is happening — the range is no longer valid
   - Subscribes to REGIME_CHANGE events

4. **Time-based safety:** Position held > `max_hold_minutes` (default 60) → SELL
   - Price stuck mid-range, not reaching either side
   - Prevents capital from being locked in stale positions

**Priority:** Range invalidation > Support breakdown > Resistance exit > Time-based

#### 4. Strategy-Aware Guard Exemption

Modify global guard rules to accept an optional `exempt_strategies` set:

```python
class SidewaysSuppressionRule:
    def __init__(self, ..., exempt_strategies: set[str] | None = None):
        self._exempt_strategies = exempt_strategies or set()

    def evaluate(self, signal, order, portfolio):
        if signal.strategy_id in self._exempt_strategies:
            return RuleResult(decision=RuleDecision.APPROVE, ...)
        # ... existing logic unchanged
```

The StrategyRegistry passes `exempt_strategies={"range_trading"}` when constructing global rules.

Only SidewaysSuppressionRule is exempted. VolatilityGateRule and MacroVolatilityGateRule remain active for range_trading — they provide an independent safety check that the market has *some* movement. The RangeDetector's 3-bounce requirement provides the strategy-specific validation; the volatility gates provide a global sanity check.

### Integration Points

**StrategyRegistry:**
- Registers `range_trading` as 4th strategy
- Wires RangeDetector as a dependency
- Add `exit_monitor_factory` field to StrategyConfig (optional callable). When present, registry calls it instead of constructing the default ExitMonitor. RangeTradingStrategy sets this to construct a RangeExitMonitor. All other strategies leave it None (default ExitMonitor).
- Passes exempt_strategies to global guard construction

**Conductor:**
- Naturally allocates to range_trading when SIDEWAYS detected
- Can zero it out in BULL/BEAR (strategy won't generate signals anyway due to preferred_regimes)
- **Capital allocation**: The 4 strategies share the total paper balance. In SIDEWAYS, the conductor should allocate ~40% to range_trading, ~30% to mean_reversion, ~0% to momentum, ~30% to breakout. In non-SIDEWAYS, range_trading gets 0%. Initial default allocation (before conductor's first call): equal 25% each.

**Darwinian Allocator:**
- Tracks range_trading fitness independently
- Can reduce allocation if range trading underperforms

**Dashboard:**
- Shows range_trading as 4th strategy in the web UI
- Displays range state (support/resistance levels, bounce count, confirmed/invalidated)

### Event Flow

```
1. Market enters SIDEWAYS regime
2. S/R signals detect bounces at support ($69,500) and resistance ($70,200)
3. RangeDetector counts bounces: 1... 2... 3 → range confirmed
4. Price approaches support ($69,500)
5. S/R generator emits BUY signal (strength 0.6, near support)
6. RangeSignalAggregator: S/R weight 2.0 → combined strength 0.48 > threshold 0.2 → emit
7. RiskManager (range_trading): SidewaysSuppressionRule → EXEMPT → APPROVE
8. PaperAdapter: BUY 0.002 BTC @ $69,550
9. Price rises toward resistance ($70,200)
10. RangeExitMonitor: price within 0.3% of resistance → SELL @ $70,180
11. P&L: +$1.26 (0.9% gain minus commission)
```

### Failure Modes

| Scenario | Detection | Response |
|----------|-----------|----------|
| False range (breaks down) | Price below support - 0.5% | Stop-loss exit |
| Regime change during trade | REGIME_CHANGE event | Immediate exit |
| Range too tight to profit | range_width_pct < min_range_width (0.6%) | RangeDetector rejects range |
| Price stuck mid-range | max_hold_minutes exceeded | Time-based exit |
| S/R levels shift | New S/R signals update levels | RangeDetector adjusts, positions use original levels |
| S/R levels go stale | level_staleness_minutes (120) exceeded | Range invalidated, no new entries |
| Continuous ticks near support | Bounce deduplication flag | Only count bounce when price leaves + re-enters zone |

### Configuration (paper.toml additions)

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
sr_weight = "2.0"

[strategy.range_trading.range]
min_bounces = 3
breakout_margin_pct = "0.5"
resistance_proximity_pct = "0.3"
support_proximity_pct = "0.3"
min_range_width_pct = "0.6"
max_hold_minutes = 60
level_staleness_minutes = 120
```

### What Stays Unchanged

- Momentum, mean_reversion, breakout strategies — no changes
- SidewaysSuppressionRule logic — only adds optional exempt_strategies param
- Existing risk rules, event bus, paper adapter — unchanged
- Session 8 behavior — range_trading would be a new addition, not modifying existing behavior

### Testing Strategy

1. **Unit tests**: RangeDetector (bounce counting, range validation, invalidation)
2. **Unit tests**: RangeExitMonitor (resistance exit, support breakdown, time exit, regime exit)
3. **Unit tests**: Strategy-aware guard exemption
4. **Integration test**: Full range trading cycle (range detected → buy at support → sell at resistance)
5. **Regression test**: Existing strategies unaffected by guard exemption changes
6. **Paper trading**: Session 9 with range_trading enabled — verify it trades SIDEWAYS and sits out BULL/BEAR

### Success Criteria

- Range strategy enters trades in SIDEWAYS markets where other strategies are blocked
- Exits at resistance (structural) rather than fixed % TP
- Stops out when range breaks down (not riding a breakout into losses)
- Doesn't trade when range isn't confirmed (< 3 bounces)
- Other strategies remain unaffected by guard exemption
- Commission-positive: range width must exceed round-trip commission (~0.32%)
