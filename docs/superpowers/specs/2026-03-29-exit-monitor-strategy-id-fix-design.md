# Fix: Exit Monitor strategy_id Routing + Pending Guard

**Date:** 2026-03-29
**Status:** Approved
**Issue:** Exit monitors emit OrderEvents without `strategy_id`, causing fills to bypass per-strategy PortfolioTracker routing. This creates an infinite exit loop in multi-strategy mode.

## Problem

Both `ExitMonitor` and `RangeExitMonitor` were written before the multi-strategy refactoring (Phase 11). They were never updated to accept and propagate `strategy_id` like `RiskManager`, `PortfolioTracker`, and `SignalAggregator` were.

### Bug Chain

1. Exit monitor detects exit condition (stop-loss, time-exit, etc.)
2. Emits `OrderEvent` with `strategy_id=None`
3. Paper adapter fills the order, creates `FillEvent` with `strategy_id=None`
4. Strategy's `PortfolioTracker` (e.g. `strategy_id="range_trading"`) ignores the fill (`None != "range_trading"`)
5. Position amount in the strategy portfolio never decreases
6. `_on_fill` clears `pending_exits` on any sell fill (checks symbol only)
7. Next tick: position still exists at same amount → exit triggers again
8. Repeat until aggregate position is fully sold off in tiny slices

### Observed Impact (Session 18)

- 37 sells of 0.0378 SOL over 3 minutes instead of 1 sell of ~1.4 SOL
- Extra commission: ~$0.185 vs ~$0.05 for a single bulk sell
- `range_exit_monitor` fired on every market tick due to 1075-minute position age (60-min max)
- Same pattern observed on `ExitMonitor` (non-range) later in the session

## Design

### Approach: strategy_id propagation + position-check pending guard

Two changes fix both root causes:

1. **Route fills correctly** — add `strategy_id` to exit monitor constructors and set it on emitted OrderEvents
2. **Guard pending-exits robustly** — only clear `pending_exits` when the position is actually gone (amount ≈ 0), not on any sell fill

### File Changes

#### `cerebrum/risk/exit_monitor.py`

- **Constructor**: Add `strategy_id: str | None = None` parameter, store as `self._strategy_id`
- **`_on_market_data`**: Set `strategy_id=self._strategy_id` on the emitted `OrderEvent`
- **`_on_fill`**: Replace eager clear with position-amount check:
  ```python
  async def _on_fill(self, event: Event) -> None:
      if not isinstance(event, FillEvent):
          return
      if event.side == Side.SELL and event.symbol in self._pending_exits:
          pos = self._portfolio.get_position(event.symbol)
          if pos is None or abs(pos.amount) < Decimal("0.0001"):
              self._pending_exits.discard(event.symbol)
  ```

#### `cerebrum/risk/range_exit_monitor.py`

Identical pattern:
- **Constructor**: Add `strategy_id: str | None = None`, store it
- **`_emit_sell`**: Set `strategy_id=self._strategy_id` on the `OrderEvent`
- **`_on_fill`**: Same position-amount check before clearing `pending_exits`

#### `cerebrum/strategies/registry.py` (`_build_pipeline`)

- Pass `strategy_id=cfg.name` when constructing `ExitMonitor`
- Pass `strategy_id=cfg.name` to `exit_monitor_factory` so custom factories receive it

#### `cerebrum/strategies/range_trading.py` (`_create_range_exit_monitor`)

- Accept `strategy_id` parameter from the factory signature
- Pass it through to `RangeExitMonitor(..., strategy_id=strategy_id)`

#### `cerebrum/main.py` (single-strategy legacy path)

- No change needed — single-strategy mode uses `strategy_id=None` which means "accept all fills" (backward compatible)

### What This Does NOT Change

- `PaperTradingAdapter` — already propagates `strategy_id` from order to fill correctly
- `PortfolioTracker` — already filters fills by `strategy_id` correctly (DEC-RISK-004)
- `RiskManager` — already sets `strategy_id` on OrderEvents correctly
- `FillEvent` / `OrderEvent` schemas — already have `strategy_id` fields
- Single-strategy mode — backward compatible via `strategy_id=None`

## Tests

1. **ExitMonitor**: Verify `strategy_id` appears on emitted OrderEvents when set
2. **RangeExitMonitor**: Same verification
3. **Regression test**: Simulate rapid-fire exit loop — emit exit, fill it, verify `pending_exits` stays set while position > 0, clears only when position reaches 0
4. **Backward compat**: Verify single-strategy mode (strategy_id=None) still works

## Verification

1. Run full test suite: `pytest tests/`
2. Check that existing exit monitor tests pass with the new `strategy_id` parameter
3. Verify regression test demonstrates the fix: single exit order closes the position, no rapid-fire loop
