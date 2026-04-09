# Session 28 Tuning — Symbol Pruning and Config Tightening

**Date:** 2026-04-09
**Decision IDs:** DEC-TUNE-013, DEC-TUNE-014, DEC-TUNE-015
**Branch:** feature/session28-tuning

---

## Problem Statement

Session 28 paper trading data revealed three structural issues causing preventable losses:

### 1. Mean-reversion trading unprofitable symbols
- mean_reversion SOL/USD: **0% WR on 10 trades, -$28.90**
- mean_reversion BTC/USD: **0% WR on 9 trades**
- mean_reversion ETH/USD: 30% WR (the only symbol with positive signal)
- SOL mean-reversion fails because SOL exhibits stronger trend/momentum behavior rather than mean-reverting price action. BTC was already removed (DEC-TUNE-009).

### 2. range_trading DOGE insufficient data
- range_trading DOGE/USD: only **2 trades** in session 28 — not enough to assess viability
- DOGE was added in DEC-TUNE-011 as an experiment; session 28 shows it does not generate sufficient signal volume to be useful
- Keeping low-activity symbols consumes capital allocation slots without return

### 3. max_position_age timeout exits are guaranteed losers
- mean_reversion max_position_age was 90 minutes
- Timeout exits (position held past max age without hitting TP/SL) produced **-$5.14 across 10 timeout exits**
- At 90 minutes, the market has had enough time to confirm the mean-reversion trade either worked or failed — 45 minutes is sufficient to distinguish genuine mean-reversion (fast) from a stale losing position
- Halving the timeout cuts expected timeout-exit losses roughly in half and frees capital sooner

### 4. SIDEWAYS suppression threshold too permissive
- `sideways_suppression_min_range_pct = 1.0` was introduced after Session 5 (Issue #3) to block BUY entries in flat markets
- Session 28 showed **85–90% loss rate** on entries that passed the 1.0% gate
- The market was frequently trading in 1.0–1.4% ranges that technically cleared the gate but were still too narrow to reach take-profit before commissions eroded the position
- Raising to 1.5% blocks these marginal entries

---

## Changes

### DEC-TUNE-013: Symbol Pruning

**mean_reversion.py**
- `symbols=["ETH/USD", "SOL/USD"]` → `symbols=["ETH/USD"]`
- Rationale: 0% WR on SOL over 10 trades in session 28. Mean-reversion works on ETH (range-bound, liquid); SOL trends too strongly.

**range_trading.py**
- `symbols=["BTC/USD", "ETH/USD", "DOGE/USD"]` → `symbols=["BTC/USD", "ETH/USD"]`
- Rationale: DOGE produced only 2 trades in session 28 — insufficient volume to justify capital allocation. BTC+ETH have proven range-trading signal density.

### DEC-TUNE-014: Reduce mean_reversion max_position_age 90→45 minutes

**mean_reversion.py**
- `"max_position_age_minutes": 90` → `"max_position_age_minutes": 45`
- Rationale: 90-min timeout exits were guaranteed losers (-$5.14/10 trades). Mean-reversion is a fast-signal strategy — if the trade hasn't resolved in 45 minutes, it's a failed trade. Shorter timeout cuts losses sooner and recycles capital.

### DEC-TUNE-015: Raise SIDEWAYS suppression threshold 1.0→1.5%

**config/paper.toml**
- `sideways_suppression_min_range_pct = "1.0"` → `sideways_suppression_min_range_pct = "1.5"`
- Rationale: Session 28 showed 85% loss rate on entries in 1.0–1.4% sideways ranges. The 1.0% gate was too permissive — commissions (0.32%) + slippage (0.1%) + spread mean a 1.5% minimum range is required for take-profit to be reachable.

---

## Expected Impact

| Metric | Before | Expected After |
|--------|--------|----------------|
| mean_reversion active symbols | ETH, SOL | ETH only |
| range_trading active symbols | BTC, ETH, DOGE | BTC, ETH |
| mean_reversion timeout exits | ~10/session at -$5.14 total | ~5/session at -$2.50 total |
| SIDEWAYS entry denials | ~15% more entries allowed | ~25% more entries denied |
| SOL mean-reversion losses | ~-$28.90/session | $0 (symbol removed) |

Total expected PnL improvement: ~+$30–35/session from symbol pruning alone, plus reduction in SIDEWAYS churn losses.

---

## Verification Plan

1. **Unit tests:** Run `pytest tests/ -x -q` — all 782+ tests must pass (config-only changes, no logic touched)
2. **Session 29 monitoring targets:**
   - mean_reversion should show 0 SOL trades (symbol removed)
   - range_trading should show 0 DOGE trades (symbol removed)
   - mean_reversion timeout exits should drop from ~10 to ~5 per session
   - SIDEWAYS-regime denials should increase (check denial counter in dashboard)
3. **Win-rate target:** mean_reversion WR should improve from ~15% (diluted by SOL/BTC) to ~30%+ (ETH-only)
