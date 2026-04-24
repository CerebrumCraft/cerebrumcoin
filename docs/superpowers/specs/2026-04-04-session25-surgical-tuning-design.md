# Session 25 Surgical Tuning — "Fewer, Bigger, Longer"

## Context

Session 24 ran 163.5 hours (6.8 days) with 377 fills and 358 completed round-trips.
Net PnL: -$55.91 (-0.56%). **Gross PnL was only -$9.75** — commission ($49.27) is
505% of the gross loss. The strategies pick direction nearly at breakeven; commission
is the entire problem.

Key data points driving these changes:
- 86 sub-15min round-trips at 6% WR = pure commission burn
- 6-24h holds had 32% WR — the best bucket by far
- DOGE accumulated (45 buys / 31 sells, $1,108 exposure) under mean_reversion
- Range trading cooldown (900s) is half of mean_reversion (1800s), causing 2x churn
- Range trading `min_signal_strength` at 0.3 is barely above noise

## Changes

### 1. Add `min_hold_minutes` to ExitMonitor and RangeExitMonitor

**Decision ID:** DEC-EXIT-006

**What:** New parameter `min_hold_minutes` (default 15). Exit monitors skip SL/TP
evaluation for positions younger than this threshold. Max-age exits still work
(a position won't be held *longer* than max_age just because of the floor).

**Where:**
- `cerebrum/risk/exit_monitor.py` — `_check_exits()` method: skip SL/TP/adaptive-TP
  checks when `position_age < min_hold_minutes`. Time-exit still triggers at max_age.
- `cerebrum/risk/range_exit_monitor.py` — same pattern in its exit check method:
  skip resistance-proximity / fallback exits when position too young.
- Config: read from `exit_config` dict in StrategyConfig, falling back to paper.toml
  `[risk] min_hold_minutes`.

**Why:** 86 trades under 15 minutes had 6% WR. These are noise — the price hasn't
moved enough to distinguish signal from commission. A 15-minute floor lets the
position develop before evaluating exit conditions.

**Risk:** A flash crash during the hold period won't trigger SL for up to 15 min.
At 1.0% SL on a $250 position, max unprotected loss is $2.50. Acceptable given
the $5.70 saved by eliminating the sub-15min bucket.

### 2. Range trading `post_fill_cooldown_seconds`: 900 -> 1800

**Decision ID:** DEC-TUNE-010

**Where:** `cerebrum/strategies/range_trading.py` — `risk_overrides` dict.

**Why:** Range trading has half the cooldown of mean_reversion, producing 2x the
churn on shared symbols (ETH). Aligning to 1800s reduces trade frequency without
changing signal quality.

### 3. DOGE: mean_reversion -> range_trading

**Decision ID:** DEC-TUNE-011

**Where:**
- `cerebrum/strategies/mean_reversion.py` — remove "DOGE/USD" from `symbols` list
- `cerebrum/strategies/range_trading.py` — add "DOGE/USD" to `symbols` list

**Why:** DOGE under mean_reversion accumulated 45 buys vs 31 sells ($1,108 exposure)
because mean reversion keeps buying dips that don't revert. Range trading has
structural exits (sell at resistance, stop at support breakdown) that close
positions based on market structure. Also gets tighter params: 0.5% SL, 1.0% TP,
60 min max age.

### 4. Range trading `min_signal_strength`: 0.3 -> 0.5

**Decision ID:** DEC-TUNE-012

**Where:** `cerebrum/strategies/range_trading.py` — `risk_overrides` dict.

**Why:** S/R signals at 0.3 conviction are barely above noise. Raising to 0.5
filters weak bounces while keeping the strategy active. Still lower than
mean_reversion's 0.5 (same level), but the signal_source_filter already
constrains range_trading to S/R-only signals which are inherently sparser.

## Files to Modify

| File | Change |
|------|--------|
| `cerebrum/risk/exit_monitor.py` | Add `min_hold_minutes` param + skip logic |
| `cerebrum/risk/range_exit_monitor.py` | Add `min_hold_minutes` param + skip logic |
| `cerebrum/strategies/mean_reversion.py` | Remove DOGE/USD from symbols |
| `cerebrum/strategies/range_trading.py` | Add DOGE/USD, cooldown 900->1800, signal 0.3->0.5 |
| `config/paper.toml` | Add `min_hold_minutes = 15` under [risk] |
| Tests for exit monitors | Verify min_hold_minutes suppresses early exits |

## What We're NOT Changing

- Position sizes (5% is tuned)
- Mean_reversion signal threshold (0.5 works)
- Max position age (90/60 min)
- Aggregation thresholds or window
- Stop loss / take profit percentages
- Commission gate ratio (3.0)
- Volatility gate thresholds

## Verification

1. Run existing test suite — all 700+ tests pass
2. Add unit tests for min_hold_minutes:
   - Position at 10 min age: SL/TP not checked, no exit emitted
   - Position at 16 min age: SL/TP checked normally
   - Position at max_age: time-exit still fires regardless of min_hold
3. Verify config changes: start paper trading, confirm in logs:
   - `min_hold_minutes: 15` in exit_monitor_initialized
   - DOGE/USD appears in range_trading signals, not mean_reversion
   - Range trading cooldown shows 1800s
   - Range trading min_signal_strength shows 0.5
4. Let Session 25 run 24h, compare fill count and commission to Session 24's first 24h
