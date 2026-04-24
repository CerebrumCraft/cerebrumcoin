# Pelosi-Follow Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** planned (Phase 15 candidate)
**Owner:** planner — 2026-04-19
**Supersedes:** none
**Related:** `docs/superpowers/plans/2026-04-12-stocks-orb-implementation.md` (established stocks foundation), `project_stocks_strategy.md` (requires dedicated strategy with signal isolation)

---

## Original Intent

> "Scope a new `pelosi_follow` strategy for the CerebrumCoin paper-trading bot. The user wants to mirror Nancy Pelosi's publicly disclosed STOCK Act trades."

The premise: public STOCK Act disclosures (PTRs — Periodic Transaction Reports) of congressional trades — particularly Rep. Pelosi's — are widely cited as producing above-market returns despite a legally-mandated 30–45 day filing lag. A paper-trading strategy that mirrors disclosed trades tests whether the edge survives the lag, commission drag, and options-blindness that the user's current adapter imposes.

This is a **paper-first** exercise. Real money is not on the table. The deliverable is a strategy that ingests the most recent public disclosures, translates them into equity orders the existing `AlpacaAdapter` + `PaperTradingAdapter` path can execute, and runs for at least two weeks so we can measure hit rate, drawdown, and commission drag under real filing cadence.

---

## Context

### Existing foundation we reuse

- **Equity plumbing is live.** `AlpacaAdapter` (DEC-ALPACA-001, Phase 6) streams US equity market data to the bus. `PaperTradingAdapter` executes simulated fills. `MarketHoursGate` (DEC-STOCKS-003) and `EndOfDayFlatten` (DEC-STOCKS-003) are wired and proven in `orb_stocks`. `signal_source_filter` (DEC-STOCKS-005) prevents cross-strategy signal contamination.
- **Strategy abstraction is live.** `StrategyRegistry` (DEC-STRAT-001, Phase 11A) registers N independent strategy instances with per-strategy balance, signal aggregator, risk manager, and exit monitor. `orb_stocks` is the reference pattern we mirror.
- **State persistence is live.** `migrate_state_v2_to_v3` in `cerebrum/adapters/paper.py` extends the snapshot schema for new strategies.
- **Config layering is live.** `config/paper.toml` has the `[strategy.X]` + `[signal.X]` + `[strategy.X.weights]` convention; both ORB stocks and xStocks-reversion follow it.
- **The memory directive `project_stocks_strategy.md` is binding.** Any new equity strategy must consume signals distinct from crypto strategies. `pelosi_follow` satisfies this via `signal_source_filter="Congressional"`.

### What is different about this strategy

- **Signal source is exogenous and event-driven, not technical.** Existing signals (RSI/MACD/BB/VWAP/OpeningRange/SupportResistance) derive from market data already streaming on the bus. Congressional trades are filed on an external server, are sparse (a handful per month), and arrive with 30–45 day delay.
- **Signal cardinality is low.** Pelosi discloses ~5–25 trades per year. This is two orders of magnitude fewer events than a technical signal generator produces.
- **Options dominate her volume.** Empirically ~60% of her disclosed trades are options (calls on NVDA, AVGO, TEM, VST). The paper adapter executes equities only. An options-ignore policy leaves a small, biased sample.
- **Entry timing is not actionable in the normal sense.** By the time a filing is public, the trade is 14–45 days old. This is not a news-driven signal we race; it is a follow-the-leader signal we accept as stale by design.

---

## Problem Statement

The user wants to test whether mirroring disclosed STOCK Act trades produces positive PnL in paper, net of:

1. **Filing lag** (up to 45 days — the disclosed entry price is not available to the mirror).
2. **Options blindness** (roughly 60% of her volume is options, which the adapter cannot execute).
3. **Commission drag** (Session 18 data: 64% of gross PnL was eaten by commissions on a higher-velocity strategy; a sparse strategy may be even more commission-sensitive on a per-trade basis).
4. **Sample sparsity** (2-week paper validation yields 1–5 trades at most; statistically inconclusive but still a sanity gate on the pipeline).

If the mirror fails to break even in 2 weeks of paper, we have not disproven the edge — the window is too short — but we have proven the pipeline works and can extend to a top-N congress-member universe (Phase D) for a 3-month paper run.

**Evidence basis:** Unverified retail-investor claims that Pelosi outperforms SPY by 10–30% annualized. The NANC ETF (Unusual Whales Subversive Democratic Trading ETF) provides a public proxy with a real track record but is itself subject to survivorship/selection bias. Planner has NOT verified these claims — this plan deliberately treats the edge as an open question and designs a cheap-to-cancel test.

---

## Goals

- **REQ-GOAL-001:** Deliver a working `pelosi_follow` strategy that ingests public STOCK Act disclosures, deduplicates them, and emits `Congressional` signals the existing aggregator + risk + paper adapter path can execute.
- **REQ-GOAL-002:** Run a 2-week paper validation with live polling enabled, producing a signed PnL number, a hit rate, and a commission-drag ratio the user can evaluate.
- **REQ-GOAL-003:** Honor Sacred Practice #6 (no implementation without plan) and Practice #2 (worktree isolation). All source changes ship on a branch, not main.
- **REQ-GOAL-004:** Preserve signal isolation per `project_stocks_strategy.md` — `pelosi_follow` consumes only its own `Congressional` signals and does not contaminate any other strategy.

## Non-Goals

- **REQ-NOGO-001:** Live trading. Paper only for this plan. Live graduation requires a separate plan, explicit user approval, and DEC-LIVE-001-style dual safety gate.
- **REQ-NOGO-002:** Options execution. Puts and calls are ignored in Phase A–C. A "bullishness proxy via underlying" policy is the compromise for calls; puts are dropped. Real options support is out of scope.
- **REQ-NOGO-003:** Backtesting against historical filings. The 45-day delay contaminates backtests (the backtester knows the filing date but not the trade date, making any backtest either cheating or unrealistic). Paper validation is the only honest signal.
- **REQ-NOGO-004:** Top-N congress-member universe in v1. Phase D sketches the extension but does not deliver it.
- **REQ-NOGO-005:** Paid data sources in v1. Quiver Quantitative ($50–150/mo) is tabled. v1 uses a free or free-tier provider so the cost of a failed experiment is zero dollars.
- **REQ-NOGO-006:** Political/moral stance on congressional trading. This is a market-data experiment; the plan is agnostic on the ethics and stays strictly mechanical.

---

## Requirements

### Must-Have (P0)

- **REQ-P0-001:** New signal generator `CongressionalTradeSignal` in `cerebrum/signals/congressional.py` that polls the chosen data source on a configurable interval, parses filings into `SignalEvent`s tagged `metadata["source"]="Congressional"`, and publishes to the bus.
  - **Acceptance:** Given a fixture JSON of 3 disclosed trades, when the generator processes them, then 3 `SignalEvent`s are published with correct `symbol`, `action` (BUY for purchases + call buys, SELL for sales), and `source="Congressional"`.

- **REQ-P0-002:** Dedup ledger in `data/congressional_trades.db` (SQLite) or `data/congressional_trades.jsonl` keyed by the upstream provider's filing_id (or a stable hash of `{member_id, ticker, transaction_date, amount_bucket}` if no id is exposed). A filing seen once is never re-emitted.
  - **Acceptance:** Given a fixture with 3 filings, when the generator runs twice, then only the first run emits signals; the second run emits zero.

- **REQ-P0-003:** New strategy config `[strategy.pelosi_follow]` in `config/paper.toml` + `cerebrum/strategies/pelosi_follow.py`, mirroring `orb_stocks` structure with `signal_source_filter="Congressional"`, `initial_balance=5000`, and a starting symbol universe restricted to liquid large-caps Pelosi has historically traded (e.g. NVDA, AVGO, MSFT, TEM, VST, AAPL, GOOG, PANW).
  - **Acceptance:** `pelosi_follow` registers in `StrategyRegistry` when `enabled=true`; does not register when `enabled=false`; does not receive signals from `OpeningRangeSignal` or technical generators.

- **REQ-P0-004:** Options policy — equity-only execution. Option buys on tickers in the universe are converted to underlying-share BUY signals at reduced position size (`options_bullish_proxy_fraction = 0.5`). Option sells/puts are dropped entirely and logged to a `options_skipped` metric.
  - **Acceptance:** Given a fixture with 1 stock buy, 1 call buy, 1 put buy, 1 stock sell, when the generator processes them, then 2 BUY signals (stock + call-proxy at half size) and 1 SELL signal fire; the put buy is logged as skipped.

- **REQ-P0-005:** Staleness gate — reject any signal whose disclosed `transaction_date` is older than `max_signal_staleness_days` (configurable, default 45 — the statutory ceiling). Filings older than this window are logged and dropped.
  - **Acceptance:** Given a filing with `transaction_date` 46 days before now, when the generator processes it, then no signal is emitted and a `signal_stale_dropped` log line is produced.

- **REQ-P0-006:** Feature flag. Both `[signal.congressional].enabled` and `[strategy.pelosi_follow].enabled` default to `false`. Merging Phase A/B to main does not change runtime behavior of any session.
  - **Acceptance:** With flags off, `pytest -q --ignore=tests/live` and an 8-second startup smoke test show zero references to `pelosi`/`congressional` in the log and no test count regressions.

- **REQ-P0-007:** `MarketHoursGate` + `EndOfDayFlatten` compatibility. Pelosi-follow symbols are equity symbols; they respect RTH gating and flatten at 15:55 ET (same config as `orb_stocks`).
  - **Acceptance:** A `Congressional` BUY signal emitted at 20:00 ET is denied by `MarketHoursGate`; an open position at 15:54:50 ET is flattened by `EndOfDayFlatten`.

- **REQ-P0-008:** Test fixtures. All new unit tests use fixture JSON under `tests/fixtures/congressional/` — no network calls in unit tests.
  - **Acceptance:** `pytest tests/unit/test_congressional_signal.py -q` runs to green with the network cable unplugged.

### Nice-to-Have (P1)

- **REQ-P1-001:** Position sizing proportional to the disclosed amount bucket. STOCK Act discloses ranges (`$1,001–$15,000`, `$15,001–$50,000`, `$50,001–$100,000`, etc.). Use the bucket midpoint as a size multiplier capped at `position_size_percent` of the strategy allocation.
- **REQ-P1-002:** Hold-until-sell semantics. Keep the position open until Pelosi discloses a matching sale on the same ticker; if no sale disclosed within `max_hold_days` (default 60), fall back to the strategy's `max_position_age_minutes` / `take_profit_percent` exit.
- **REQ-P1-003:** `scripts/show_congressional.py` CLI reporting ingested filings, emitted signals, skipped filings, and current open positions attributed to this strategy.

### Future Consideration (P2)

- **REQ-P2-001:** Top-N congress-member universe. Extend the generator to filter by a configurable list (`members = ["Pelosi", "Crenshaw", "Gottheimer", ...]`) with per-member signal weights. Architect the v1 ledger schema (`member_id` column) so this is purely a config change, not a migration.
- **REQ-P2-002:** Options support via Alpaca options API (when the adapter supports it) — restore the 60% of her volume we currently drop.
- **REQ-P2-003:** Backtest harness with realistic filing-lag simulation (a backtest that knows `filing_date` but only uses `transaction_date + lag` as the entry signal).

---

## Success Metrics

> Included per planner guidance — this strategy has measurable outcomes. A 2-week paper window is diagnostic-only; do not treat these targets as production gates.

- **REQ-MET-001:** Pipeline health (leading). Target: ≥ 1 `Congressional` signal emitted, ≥ 1 paper fill, 0 unhandled exceptions in the strategy loop over 14 days. Measured from `logs/session-pelosi-*.log`.
- **REQ-MET-002:** Signal quality (leading). Target: ≥ 50% of emitted signals survive the staleness gate + dedup (i.e. the generator is not running on a firehose of rejects). Measured from the dedup ledger + structured logs.
- **REQ-MET-003:** Paper PnL (lagging, diagnostic only). Target: break-even net of commissions over 14 days is a pass; `pnl_net > -1%` of `initial_balance` is a soft pass. A hard fail is a loss > 3% of allocation — that triggers a plan revision before Phase C continues.
- **REQ-MET-004:** Commission drag ratio (lagging). Target: `commission / gross_pnl < 40%` — tighter than the Session 18 baseline of 64% because this strategy trades far less frequently.

---

## Architecture Sketch

```
                +-----------------------+
                | External data source  |
                | (see DEC-PELOSI-DATA) |
                +-----------+-----------+
                            |  poll every N minutes
                            v
        +----------------------------------+
        | CongressionalTradeSignal (new)   |
        |  - dedup via SQLite ledger       |
        |  - staleness gate                |
        |  - options-policy transformer    |
        |  - amount-bucket → size hint     |
        +-----------+----------------------+
                    | SignalEvent(source="Congressional")
                    v
        +---------------------------+   filtered by signal_source_filter
        | SignalAggregator          |   ⇒ only pelosi_follow sees these
        | (pelosi_follow instance)  |
        +-----------+---------------+
                    |
                    v
        +---------------------------+
        | RiskManager               |
        |  + MarketHoursGate        |
        |  + commission_gate        |
        +-----------+---------------+
                    |
                    v
        +---------------------------+
        | PaperTradingAdapter       |
        |  (existing, unchanged)    |
        +---------------------------+
                    |
                    v
        +---------------------------+
        | ExitMonitor               |
        |  + EndOfDayFlatten        |
        |  + (P1) disclosed-sale hook
        +---------------------------+
```

**Files (new):**

| Path | Role |
|---|---|
| `cerebrum/signals/congressional.py` | Polling signal generator. Does **not** inherit from `SignalGenerator` (which is tick-driven) — it polls on its own schedule, builds `SignalEvent`s manually, and publishes directly to the bus. Mirrors the pattern used by `intelligence/news.py`. |
| `cerebrum/strategies/pelosi_follow.py` | `PELOSI_FOLLOW_CONFIG = StrategyConfig(...)` — mirror of `orb_stocks` with `signal_source_filter="Congressional"`. |
| `cerebrum/data/congressional_store.py` | SQLite-backed dedup ledger. Schema: `(filing_id PRIMARY KEY, member, ticker, action, amount_bucket, transaction_date, disclosed_date, ingested_at, emitted)`. |
| `scripts/show_congressional.py` (P1) | CLI viewer. |
| `tests/fixtures/congressional/*.json` | Deterministic test fixtures. |
| `tests/unit/test_congressional_signal.py` | ~8 unit tests. |
| `tests/unit/test_congressional_store.py` | ~4 unit tests for dedup. |
| `tests/unit/test_pelosi_follow_config.py` | ~3 unit tests for the strategy config. |
| `tests/unit/test_main_pelosi_wiring.py` | ~3 unit tests for `_maybe_build_congressional_signal`. |

**Files (modified):**

| Path | Change |
|---|---|
| `cerebrum/main.py` | Add `_maybe_build_congressional_signal(config, bus)` helper and `[strategy.pelosi_follow]` registration gate. Pattern: copy from `_maybe_build_kraken_xstocks_adapter` (commits bf614c7, 4f57e84). |
| `config/paper.toml` | Add `[signal.congressional]` and `[strategy.pelosi_follow]` + `[strategy.pelosi_follow.weights]` sections (both `enabled = false`). |
| `cerebrum/adapters/paper.py` | Extend `migrate_state_v3_to_v4` (or equivalent) to add `pelosi_follow` snapshot when missing. |
| `pyproject.toml` | Add optional dep group `pelosi = ["httpx>=0.27"]` if the chosen data source needs a client distinct from what's already in `[project.dependencies]`. |

---

## Decisions (resolved 2026-04-19)

User accepted all planner recommendations on 2026-04-19. Values below are authoritative.

- **DEC-PELOSI-DATA-001:** Data source = **Finnhub free tier** (congressional-trading endpoint, ~60 calls/min, free, non-commercial ToS acceptable for paper-only v1). Addresses REQ-P0-001, REQ-NOGO-005. Revisit Quiver Quantitative if paper result warrants going live.

- **DEC-PELOSI-OPT-001:** Options policy = **call buys → underlying BUY at half size; drop puts and option sells.** Addresses REQ-P0-004, REQ-NOGO-002. Logged to `options_skipped` metric. Recovers ~half of her options volume while respecting equity-only execution constraint.

- **DEC-PELOSI-LAG-001:** Staleness ceiling = **45 days** (statutory max). Addresses REQ-P0-005. Filings older than 45 days from disclosure date are rejected by the `staleness_gate` rule. Tighten later if stale signals prove worthless.

- **DEC-PELOSI-HOLD-001:** Hold policy = **default exit rules only for Phase A/B** (existing stop-loss, take-profit, max-position-age). Phase C adds a disclosed-sale hook with 60-day fallback. Addresses REQ-P1-002.

- **DEC-PELOSI-UNIV-001:** Member universe = **Pelosi only for Phase A/B/C.** Addresses REQ-GOAL-001, REQ-P2-001. Schema must be extensible to top-N without migration — Phase D (optional) enables it.

- **DEC-PELOSI-SIZE-001:** Position sizing = **fixed $-per-trade for Phase A/B** (uses `[strategy.pelosi_follow] position_size_usd`). Phase C switches to amount-bucket midpoint scaled to `position_size_percent` cap using the richer disclosed data.

---

## Phased Plan

### Phase 0: Decision Gate (0.5 days)
**Status:** CLOSED 2026-04-19
**Output:** All six `DEC-PELOSI-*` values resolved above; user accepted planner recommendations.
**DoD:** File updated with resolved choices; Phase A is unblocked.

### Phase A: Data pipeline (disabled) — 1–2 days
**Status:** planned
**Decision IDs:** DEC-PELOSI-DATA-001, DEC-PELOSI-OPT-001, DEC-PELOSI-LAG-001
**Requirements:** REQ-P0-001, REQ-P0-002, REQ-P0-004, REQ-P0-005, REQ-P0-008
**Issues:** TBD — filed in Phase 5 after approval
**Definition of Done:**
- REQ-P0-001 satisfied: `CongressionalTradeSignal` emits `SignalEvent`s from fixture input.
- REQ-P0-002 satisfied: dedup ledger prevents double-emission (test proves it).
- REQ-P0-004 satisfied: options-policy transformer has fixture coverage for all four cases (stock-buy, stock-sell, call-buy, put-buy).
- REQ-P0-005 satisfied: staleness gate rejects a 46-day-old fixture.
- REQ-P0-008 satisfied: unit suite runs offline.
- Feature flags default to `false`. Regression: baseline test count + 18 new = pass.

### Phase B: Strategy wiring (disabled) — 1 day
**Status:** planned
**Decision IDs:** DEC-PELOSI-UNIV-001, DEC-PELOSI-SIZE-001 (partial)
**Requirements:** REQ-P0-003, REQ-P0-006, REQ-P0-007, REQ-GOAL-004
**Issues:** TBD
**Definition of Done:**
- REQ-P0-003: `PELOSI_FOLLOW_CONFIG` registered behind an `enabled` gate, signal isolation verified via `test_signal_aggregator_symbols_filter` extension.
- REQ-P0-006: 8-second startup smoke with flags off shows no pelosi/congressional log lines; regression passes.
- REQ-P0-007: `MarketHoursGate` + `EndOfDayFlatten` cover `pelosi_follow` symbols (no new risk rule code; config extension only).
- REQ-GOAL-004: signal-isolation test proves only `pelosi_follow` receives `Congressional` signals.
- State migration adds `pelosi_follow` snapshot.

### Phase C: Live polling + 2-week paper validation — 14 days wall-clock
**Status:** planned
**Decision IDs:** all (live integration exercises every decision)
**Requirements:** REQ-GOAL-002, REQ-MET-001, REQ-MET-002, REQ-MET-003, REQ-MET-004
**Issues:** TBD
**Definition of Done:**
- Feature flags flipped to `true` in a live paper session (not merged to main until the window closes).
- 14-day window runs. Logged artifacts: `logs/session-pelosi-*.log`, dedup ledger snapshot, daily PnL report.
- REQ-MET-001/002 computed and written to a session report under `data/pelosi_session_report.md`.
- User-reviewable artifact: table of every emitted signal, its decision (executed / denied / skipped), and the resulting fill + PnL.
- Decision point at day-14: merge to main (success), extend window (inconclusive), or revert and close (hard fail).

### Phase D (optional, gated on Phase C pass): Top-N universe — 2–3 days
**Status:** planned
**Decision IDs:** DEC-PELOSI-UNIV-001 (expand)
**Requirements:** REQ-P2-001
**Issues:** TBD (filed only if Phase C passes)
**Definition of Done:**
- Generator accepts `members` config list.
- Per-member weight multiplier in the signal.
- Schema already supports `member_id` from Phase A — no migration needed.
- 30-day paper validation with top-3 members.

### Decision Log
<!-- Guardian appends here after phase completion -->

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Data source ToS / licensing** | Medium | v1 uses a free tier with clear non-commercial terms. Revisit before live. Cite the provider's ToS URL in the first commit's @decision block. |
| **45-day lag makes signals stale** | High | This is the experiment's core hypothesis. Staleness gate + per-filing lag metric let us quantify decay. 2-week paper is too short to conclude — Phase D extends to 30 days. |
| **60% of Pelosi's trades are options → biased sample** | High | REQ-NOGO-002 accepts this in v1. Option-proxy policy (DEC-PELOSI-OPT-001) partially mitigates for calls. Document in the session report so the user knows the PnL is from 40% of her volume. |
| **Signal sparsity** → commission_gate dominates | Medium | REQ-MET-004 tracks this. Start with `commission_gate=3.0` (same as DEC-TUNE-005). If commission > 40% of gross in Phase C, tune up. |
| **Legal / reputational** | Low | Following public disclosures is legal in the US. Plan explicitly scopes this as a market-data experiment, not an endorsement. No distribution of scraped data; ingestion only. |
| **Provider outage / schema drift** | Medium | Generator catches exceptions, logs `congressional_poll_failed`, continues. One outage ≠ system outage. Fixtures cover the expected schema; a schema change needs a fixture update + test. |
| **Main-branch contamination** | Low | Worktree-only (Sacred Practice #2). Phase A–B land feature-flagged; Phase C runs in a live-paper session but flags flip only in the worktree until day-14 review. |
| **Background Session 31 disturbance** | Low | Plan explicitly excludes touching the running paper-trading process (PID 49657). All work in worktree; no main-branch restart required until Phase B merge. |

---

## Verification Strategy

**Phase A (unit):**
- Fixture-driven tests for parser, dedup ledger, options-policy transformer, staleness gate.
- Regression: baseline pass count + 18, zero new failures.
- Offline: tests run with no network.

**Phase B (integration):**
- 8-second startup smoke with flags off: zero pelosi/congressional log lines.
- 15-second startup smoke with flags on (fixtures or mocked source): `congressional_signal_initialized`, `pelosi_follow_strategy_registered`, at least one simulated `Congressional` signal event reaching `pelosi_follow`'s aggregator only.
- `test_signal_aggregator_symbols_filter` extension: a `Congressional` signal for NVDA reaches `pelosi_follow` but **not** `orb_stocks`, `mean_reversion`, or `range_trading`.

**Phase C (live paper):**
- 14-day real-world run with the chosen data provider.
- Daily `tail logs/session-pelosi-*.log | grep congressional` sanity check.
- End-of-window report: number of filings ingested, signals emitted, fills, PnL, commission drag. Written to `data/pelosi_session_report.md`.
- User review gate before merge.

---

## Out-of-Scope

- Live money execution (separate plan + DEC-LIVE-002 gate).
- Options execution.
- Paid data providers (Quiver, Bloomberg).
- Backtesting with any claim of realism (the lag contamination is unfixable without provider-supplied filing timestamps).
- Extending to Senate / House committees / non-US legislative bodies.
- Webhook-push ingestion — polling is sufficient at this signal cadence.
- UI / dashboard work beyond the P1 CLI.
- LLM interpretation of filings (the data is already structured).

---

## Worktree Strategy

- Branch: `worktree-pelosi-follow` off main.
- Worktree path: `.worktrees/worktree-pelosi-follow`.
- Phase A + B land on this branch in small commits (one per task).
- Phase C runs from the worktree with `enabled=true` config; merge to main happens only after the 14-day review passes.
- Session 31 (PID 49657) is undisturbed — that process holds main's current config; the worktree has its own `config/paper.toml`.

---

## References

- `docs/superpowers/plans/2026-04-12-stocks-orb-implementation.md` — the canonical pattern for a new equity strategy.
- `docs/superpowers/plans/2026-04-16-xstocks-path-b-implementation.md` — template for the conditional-adapter wiring helper.
- `cerebrum/strategies/orb_stocks.py` — the exact file layout to mirror.
- `cerebrum/intelligence/news.py` — the exact polling pattern to mirror for `CongressionalTradeSignal` (polling, not tick-driven).
- `project_stocks_strategy.md` — memory directive requiring signal isolation for new equity strategies.
- STOCK Act (Pub. L. 112-105) — the statute mandating congressional trade disclosure.
- Finnhub `/stock/congressional-trading` endpoint — candidate data source for DEC-PELOSI-DATA-001 option A.

---

## Session End Protocol

After the user resolves Phase 0 decisions, this plan is ready for implementation. Next actions:

1. Planner updates the six `DEC-PELOSI-*` entries with the user's choices.
2. Planner or orchestrator files GitHub issues for Phase A tasks (1 per file created).
3. Orchestrator dispatches the implementer to the `worktree-pelosi-follow` worktree with this plan as the driver.
4. Session 31 continues undisturbed throughout.
