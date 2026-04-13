# Stocks Support — Opening Range Breakout (ORB) via Alpaca

## Context

CerebrumCoin has been crypto-only (Kraken adapter, 4 symbols, 2 strategies). An
Alpaca stock adapter already exists at `cerebrum/adapters/alpaca.py` (288 lines,
`@decision DEC-ALPACA-001` accepted) with tests at `tests/unit/test_alpaca.py`
(317 lines), but it has never been wired into `main.py` — the `[alpaca]`
section in `config/paper.toml` is commented out (lines 88–94) with a note:
"uncomment and fill in credentials to enable."

This spec connects that existing adapter, adds an **Opening Range Breakout (ORB)
strategy** as the required stocks-native strategy (per the project's
`project_stocks_strategy.md` memory: *"add at least one new strategy — not a
crypto strategy copy"*), and introduces the market-hours awareness the codebase
currently lacks.

**Goal:** run `python -m cerebrum --mode paper --config config/paper.toml` and
have three strategies trading — `mean_reversion` + `range_trading` on crypto
(unchanged), and `orb_stocks` on AAPL/MSFT/NVDA during US market hours — all
in one process, one event bus, one state file.

**Key decisions locked during brainstorming** (2026-04-12):

| Dimension | Choice |
|---|---|
| Fill path | Local simulation via existing `paper_trading` adapter (same as crypto) |
| Process topology | Same cerebrum process as crypto — shared bus, shared ledger |
| Trading hours | RTH only (9:30–16:00 ET), auto-flat at 15:55 ET |
| Strategy v1 | Opening Range Breakout (15-min range, breakout entry) |
| Symbols | AAPL, MSFT, NVDA |
| Capital | $5,000 dedicated — total portfolio $10k → $15k |

## Non-Goals

- No Alpaca paper-API fill routing (local sim only — deferred).
- No pre-market or after-hours trading.
- No extended-hours SIP feed (free IEX only).
- No shorting stocks (`side="sell"` on ORB is for exit only, not short entry).
- No pairs/sector/cross-asset strategies.
- No fractional-share logic beyond what paper_trading already does with Decimals.
- No touching crypto strategies' behavior.

## Architecture

**New files (4):**

| File | Purpose | ~LOC |
|---|---|---|
| `cerebrum/utils/trading_session.py` | Pure helper: `is_rth_now()`, `minutes_until_close()`, `is_market_holiday(date)`, `early_close_time_for(date)`. Uses `zoneinfo.ZoneInfo("America/New_York")`. Static NYSE holiday + early-close calendars through 2028. | 70 |
| `cerebrum/signals/opening_range.py` | `OpeningRangeSignal` generator. Per-symbol opening-range tracker (09:30–09:45 ET). Emits buy/sell on breakout with `metadata={"source": "OpeningRange", "orb_high": …, "orb_low": …}`. | 120 |
| `cerebrum/risk/market_hours_gate.py` | Risk rule: denies orders on configured stock symbols outside RTH or within `entry_cutoff_minutes_before_close`. Crypto symbols pass through unchanged (membership check). | 50 |
| `cerebrum/risk/end_of_day_flatten.py` | Exit rule: at 15:55 ET (or early-close equivalent), emits close orders for any open stock positions under `orb_stocks`. Bypasses `commission_gate`. | 60 |

**Modified files (3):**

| File | Change | ~Lines |
|---|---|---|
| `cerebrum/main.py` | Detect `[alpaca]` config, instantiate `AlpacaAdapter`, register `OpeningRangeSignal`, wire new risk/exit rules. Extend `SignalAggregator` wiring to filter by strategy `symbols` list (prevents AAPL ticks reaching mean_reversion). | ~30 |
| `config/paper.toml` | Uncomment+fill `[alpaca]`; add `[signal.opening_range]`, `[strategy.orb_stocks]`, `[risk.market_hours_gate]`, `[risk.end_of_day_flatten]`. | ~40 |
| `cerebrum/adapters/paper_trading.py` | Verify asset-agnostic fills (probably already work). Add optional `commission_by_symbol` lookup for per-asset commission modeling (stocks typically 0 at Alpaca paper; fall back to existing flat % when absent). | ~15 |

**Unchanged:** `cerebrum/adapters/alpaca.py`, `tests/unit/test_alpaca.py`, all
crypto strategy code, the event bus, dashboard, all existing risk rules,
`mean_reversion` / `range_trading` configs.

## Data Flow

A single AAPL day, end to end:

1. **09:00 ET** — cerebrum running 24/7; Kraken streaming crypto normally.
   `AlpacaAdapter.subscribe_market_data(["AAPL","MSFT","NVDA"])` subscription
   active but quiet pre-open. `OpeningRangeSignal` internal state
   `{"AAPL": None, "MSFT": None, "NVDA": None}`.
2. **09:30:01 ET** — First AAPL tick. `MarketDataEvent` published. `OpeningRangeSignal`
   starts tracking high/low; no emission yet. Crypto signals receive the tick but
   `SignalAggregator` symbols-filter prevents contamination (see invariant 1 below).
3. **09:30 → 09:45 ET** — ORB builds per symbol. `MarketHoursGate` open.
4. **09:45:00 ET** — Range frozen per symbol, e.g. `{"AAPL": {"high": 182.43,
   "low": 181.98, "valid": true}}`. If fewer than 10 ticks or `range < min_range_bps`,
   the symbol's ORB is marked `invalid` for the day (no signals emit for it).
5. **10:12 ET** — AAPL tick at 182.51, above `orb_high + breakout_buffer_bps`.
   `OpeningRangeSignal` emits buy signal with source="OpeningRange".
6. **10:12 ET** — Signal routed through `signal_aggregator_orb_stocks`
   (signal_source_filter="OpeningRange" + symbols ∈ {AAPL,MSFT,NVDA} + threshold).
7. **10:12 ET** — `risk_manager_orb_stocks` evaluates rules including
   `market_hours_gate` (pass), `position_sizing` (compute $1000 / fill_price shares),
   `commission_gate`, `volatility_gate`, `min_signal_strength`. If approved:
   `OrderEvent` → `PaperTradingAdapter.execute_order()` → `FillEvent`.
8. **10:12 → 15:55 ET** — `ExitMonitor(orb_stocks)` watches for SL (0.5%
   below fill, or — stricter — below `orb_low`, whichever is tighter),
   fixed TP at `take_profit_percent = 1.0`, or `max_position_age_minutes`.
   Adaptive range-based TP is deferred (see Out of Scope).
9. **15:55:00 ET** — `EndOfDayFlatten` fires. For any open `orb_stocks` position,
   emits close order bypassing `commission_gate`. Retries up to 3 times at
   15:56/15:57/15:58 if rejected.
10. **16:00 ET → 09:30 ET next day** — Alpaca quiet. Crypto continues. ORB state
    reset on first tick after next 09:30 ET.

### Invariants

1. **Stock ticks never reach crypto strategy aggregators.**
   Enforced by adding a `symbols` membership filter in `SignalAggregator`
   subscription (new in `main.py` wiring). Unit test asserts this.
2. **Crypto ticks never reach `orb_stocks` aggregator.**
   Doubly enforced: (a) `OpeningRangeSignal` only emits for its configured
   stock symbols; (b) `signal_aggregator_orb_stocks` has `signal_source_filter="OpeningRange"`.
3. **Zero overnight stock exposure under normal operation.**
   `EndOfDayFlatten` runs unconditionally during RTH days with open positions.
   Failure path (rejected 3× retries) logged as ERROR + `needs_manual_intervention`
   flag in state file.

## Configuration

### `config/paper.toml` additions

```toml
[alpaca]
enabled = true
api_key_env = "ALPACA_API_KEY_ID"
secret_key_env = "ALPACA_API_SECRET_KEY"
paper_base_url = "https://paper-api.alpaca.markets"
data_feed = "iex"
symbols = ["AAPL", "MSFT", "NVDA"]

[signal.opening_range]
enabled = true
range_minutes = 15
breakout_buffer_bps = 5
min_range_bps = 20
max_range_bps = 500
symbols = ["AAPL", "MSFT", "NVDA"]

[strategy.orb_stocks]
enabled = true
initial_balance = 5000.0
symbols = ["AAPL", "MSFT", "NVDA"]
signal_source_filter = "OpeningRange"
aggregation_threshold = 0.4
position_size_percent = 20.0
stop_loss_percent = 0.5
take_profit_percent = 1.0
max_position_age_minutes = 390
min_hold_minutes = 5
post_fill_cooldown_seconds = 600
min_signal_strength = 0.6

[strategy.orb_stocks.weights]
technical = 1.0
sentiment = 0.0
news = 0.0
regime = 0.0

[risk.market_hours_gate]
enabled = true
rth_start = "09:30"
rth_end   = "16:00"
tz = "America/New_York"
stock_symbols = ["AAPL", "MSFT", "NVDA"]
allow_holidays = false
entry_cutoff_minutes_before_close = 15

[risk.end_of_day_flatten]
enabled = true
flatten_at = "15:55"
stock_symbols = ["AAPL", "MSFT", "NVDA"]
bypass_commission_gate = true
```

### `.env` additions

```
ALPACA_API_KEY_ID=PK...
ALPACA_API_SECRET_KEY=...
```

### Portfolio after change

| Strategy | Balance | Asset |
|---|---|---|
| mean_reversion | $5,000 | crypto |
| range_trading | $5,000 | crypto |
| orb_stocks | $5,000 | stocks |
| **Total** | **$15,000** | |

### State file migration (v2 → v3)

- `data/paper_state.json` bumps `version` from 2 → 3.
- Migration adds empty `orb_stocks` snapshot with `cash_balance = 5000`,
  `initial_balance = 5000`, `positions = {}`.
- Existing crypto snapshots and open positions (including current SOL)
  preserved verbatim.
- Backup written to `data/paper_state.v2.bak.json` atomically.
- If migration fails mid-write, original v2 state untouched; backup ensures
  recoverability.

### Kill switch

Setting either `[alpaca].enabled = false` or
`[strategy.orb_stocks].enabled = false` fully deactivates the stock path on
next restart. Crypto unaffected.

## Error Handling

### Install / startup

| Failure | Detection | Containment | Log event |
|---|---|---|---|
| `alpaca-py` not installed | `ModuleNotFoundError` at adapter ctor | Disable alpaca for this run, continue crypto-only | `alpaca_adapter_unavailable` |
| `.env` creds missing with `[alpaca].enabled=true` | Empty env var at `connect()` | Abort startup | `alpaca_credentials_missing` |
| Invalid API key | WebSocket 401 | Disable adapter, continue crypto-only | `alpaca_auth_failed` |
| `orb_stocks` enabled but alpaca disabled | Startup wiring check | Fail fast with clear message | `orb_requires_alpaca` |

### Runtime feed

| Failure | Detection | Containment | Log event |
|---|---|---|---|
| WS disconnect | Ping timeout / close frame | Exponential backoff reconnect. 5× failures in 60s → stop subscribing | `alpaca_disconnected`, `alpaca_reconnecting`, `alpaca_reconnected` |
| Heartbeat lost > 30s during RTH | TradingSession monitor | `market_hours_gate` denies new entries until resume | `alpaca_stream_stale` |
| ORB insufficient data | At 09:45, range has < 10 ticks or `high==low==first` | Mark symbol ORB `invalid` for the day | `orb_insufficient_data` |

### Calendar / timing

| Failure | Detection | Containment | Log event |
|---|---|---|---|
| NYSE holiday | `TradingSession.is_market_holiday(today)` | `market_hours_gate` denies all stock entries; flatten no-op | `market_closed_holiday` |
| Early close (e.g. day-after-Thanksgiving 13:00) | Static `early_close_days` map | Flatten runs at `close − 5 min` | `early_close_detected` |
| Process restart mid-day | Startup time > 09:45 ET | `OpeningRangeSignal` tries historical bar backfill; if fails, marks remaining symbols invalid for the day | `orb_post_start_recovery` |

### End-of-day flatten

| Failure | Detection | Containment | Log event |
|---|---|---|---|
| Close order rejected | No `FillEvent` / `OrderRejectedEvent` | Retry 3× at 15:56/57/58. Final failure → ERROR + state flag | `end_of_day_flatten_FAILED` |
| Partial fill | `FillEvent.filled_amount < requested` | Resubmit remainder up to 3× | (same) |
| Overnight exposure on restart | Next-day startup reconciliation | Log + surface on dashboard | `overnight_stock_exposure_detected` |

### Observable contract

All failure events are emitted as structured JSON log entries with stable
event-name strings, greppable in `logs/sessionNN.log`.

## Testing

### Unit tests (~30 new, no network)

| File | Coverage |
|---|---|
| `tests/unit/test_trading_session.py` | `is_rth_now()` boundaries (09:29 deny, 09:30 pass, 16:00 deny, weekend deny). Holiday detection. Early close. DST transitions (Mar/Nov). ~12 tests. |
| `tests/unit/test_opening_range_signal.py` | ORB build during window. No emission pre-09:45. Breakout with `breakout_buffer_bps`. `min_range_bps`/`max_range_bps` rejection. Daily reset. Insufficient-data path. ~10 tests. |
| `tests/unit/test_market_hours_gate.py` | Stock outside RTH → deny. Crypto anytime → pass. Entry cutoff. Holiday → all stock denied. ~6 tests. |
| `tests/unit/test_end_of_day_flatten.py` | Fires at 15:55 with open position. No-op with no positions. Multiple positions. Commission-gate bypass. Retry on rejection. ~5 tests. |
| `tests/unit/test_paper_state_migration.py` | v2 → v3 adds `orb_stocks`, preserves crypto. Corrupt file → fallback to backup. ~4 tests. |

### Integration tests (~6 new, fixture replay)

| File | Coverage |
|---|---|
| `tests/integration/test_orb_full_day.py` | Replay recorded AAPL day, assert entry at breakout, EOD exit, expected realized PnL. |
| `tests/integration/test_orb_nyse_holiday.py` | Holiday date → zero entries, no flatten, no errors. |
| `tests/integration/test_orb_early_close.py` | Black Friday → flatten at 12:55. |
| `tests/integration/test_cross_asset_isolation.py` | Mixed BTC+AAPL tick stream. Assert no cross-aggregator contamination via event bus trace. |
| `tests/integration/test_stream_stale.py` | 60s feed gap during RTH → gate denies entries until resume. |
| `tests/integration/test_eod_flatten_with_partial_fill.py` | Partial fill → remainder resubmitted. |

### Live tests (opt-in, gated, address open todo #5)

| File | Coverage |
|---|---|
| `tests/live/test_alpaca_live_connection.py` | `@pytest.mark.live_alpaca`. Runs only with `pytest --live-alpaca`. Connects, subscribes, reads ≥5 ticks, asserts event shape. No orders placed. |
| `tests/live/test_live_orb_smoke.py` | Same gate. 20-min live RTH run of full pipeline. Manual invocation — not CI. |

CI runs unit + integration only. Humans run live before merging.

### Fixtures

- `tests/fixtures/alpaca_aapl_2026-03-10.jsonl` — ~6k tick lines, one JSON
  per line, captured from live IEX. ~400 KB. Committed.
- `tests/fixtures/alpaca_mixed_stocks_2026-03-10.jsonl` — same date,
  3 symbols, for cross-asset test.
- `scripts/record_alpaca_ticks.py` — bootstrap helper (~80 lines), kept for
  annual fixture refresh.

### Regression gates (must-pass on PR)

1. All existing tests (~782) still pass with `[alpaca].enabled = false`.
2. Cold-start time with alpaca disabled stays under 3 s.
3. `/api/strategies` returns 3 strategies when enabled, 2 when disabled.
4. `pytest tests/unit tests/integration` under 60 s.

### Coverage target

- 100% branch coverage on the 4 new files.
- Touched lines in `main.py` and `paper_trading.py` covered by at least one
  integration test.

## Decision IDs

- **DEC-STOCKS-001** — Same-process, unified-bus topology for stocks.
  *Rationale:* matches config-over-code philosophy; no duplicated plumbing;
  crypto untouched.
- **DEC-STOCKS-002** — Local paper-trading simulation for fills (not Alpaca
  paper API). *Rationale:* unified ledger with crypto; deferred realism;
  re-evaluate once ORB PnL stabilizes.
- **DEC-STOCKS-003** — RTH only with 15:55 auto-flatten; no overnight stock
  exposure. *Rationale:* zero gap risk, simplest exit model, matches user
  preference.
- **DEC-STOCKS-004** — ORB as first strategy with dedicated `OpeningRangeSignal`
  generator and `signal_source_filter="OpeningRange"`. *Rationale:* honors
  `project_stocks_strategy.md` memory (stocks-native, genuine signal
  isolation from crypto strategies).
- **DEC-STOCKS-005** — `SignalAggregator` gains a per-strategy `symbols`
  filter at the subscription layer. *Rationale:* prevents cross-asset signal
  contamination (stock ticks reaching mean_reversion aggregator, etc.)
  cleanly and generically.
- **DEC-STOCKS-006** — State file migration v2 → v3 with atomic write +
  backup. *Rationale:* safe to roll back; no loss of open SOL position or
  crypto strategy history.

Each decision is annotated in-code via `@decision` headers at the relevant
file top (follows existing Code-is-Truth practice).

## Rollout Plan

Ordered phases; each phase ships independently, gated by tests passing:

1. **Phase 1 — `trading_session.py` + tests.** Pure utility, no wiring.
   Merge on green.
2. **Phase 2 — `opening_range.py` signal + tests.** Replay fixtures.
   Emits signals in isolation but nothing consumes them yet.
3. **Phase 3 — Aggregator symbols-filter (DEC-STOCKS-005) + tests.**
   Prevents contamination before introducing the stock symbols.
4. **Phase 4 — Alpaca wiring in `main.py`.** `[alpaca].enabled=false` by
   default. Regression: crypto-only path identical.
5. **Phase 5 — `market_hours_gate.py` + `end_of_day_flatten.py` + tests.**
6. **Phase 6 — `[strategy.orb_stocks]` registration + paper_state migration.**
7. **Phase 7 — Integration tests + fixture recording.**
8. **Phase 8 — Live-tier tests** (`@pytest.mark.live_alpaca`) + manual
   20-minute live smoke run during an actual RTH window.

Each phase gets its own GitHub issue generated by the Planner from this spec.

## Out of Scope / Future

- Alpaca paper-API execution path (DEC-STOCKS-002 reversal candidate).
- Adaptive TP sized from ORB range (e.g. `TP = 1.5 × range`). Defer until
  fixed-percent TP has baseline data.
- Second stocks strategy (VWAP reversion / gap trade) once ORB baselined.
- Short entries on breakdown (`orb_stocks` sells currently only close longs).
- SIP data feed upgrade.
- Extended hours.
- Cross-asset strategies (e.g. SPY correlation with BTC).
- Per-strategy dashboard filters for stocks (dashboard shows `orb_stocks`
  like any other strategy via existing `/api/strategies`).
