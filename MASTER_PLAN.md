# MASTER_PLAN.md — CerebrumCoin

## Original Intent

Build an autonomous adaptive AI trading agent that goes beyond price-chasing bots. CerebrumCoin integrates news, sentiment, and market regime detection into a hybrid AI pipeline (ML signals + rule-based risk + LLM reasoning). The user has a Kraken API key and wants to start with paper trading, graduating to live. The system must be extensible to stocks and future agentic trading systems.

## Project Overview

CerebrumCoin is an autonomous adaptive AI trading agent that integrates news, sentiment, and market regime detection into a hybrid AI pipeline (ML signals + rule-based risk + LLM reasoning). Starts with paper trading on Kraken, graduates to live. Extensible to stocks and future agentic trading systems.

**Architecture**: Async event-driven with central event bus. Every component (data ingestion, signals, risk, execution) is a loosely-coupled subscriber/publisher.

```
[Data Sources] → [Event Bus] → [Signal Pipeline] → [Aggregator] → [Risk Manager] → [Executor]
       ↑                                                                                   |
       └──────────────────── [Closed-Loop Learner] ←───────────────────────────────────────┘
```

**Status**: ALL PHASES COMPLETED — CerebrumCoin is a fully operational autonomous trading agent

## Current Active Work (Session 40, 2026-04-24)

**Live paper session**: Session 40 running since 2026-04-24 03:03:59 UTC (PID 207994) in `.worktrees/live` on branch `live` (= main HEAD `3b993a8`). Launched via `scripts/launch_paper.sh`. Fresh $10,000 state.

**Active strategies (3)**:
- `mean_reversion` — enabled
- `range_trading` — enabled
- `orb_stocks` — enabled

**Paused strategies**:
- `pelosi_follow` — paused. Phase 15A (data pipeline) and Phase 15B (strategy wiring) landed; **Phase 15C BLOCKED** because Finnhub `/stock/congressional-trading` is Business/Premium-tier only. Free-tier key returned 403 for all 7 symbols during Session 32. Replacement path tracked in `im40percentgit/cerebrumcoin#35`: port the House Clerk ZIP scraper from `im40percentgit/StockInsightsTracker` with OpenRouter LLM-based PDF extraction.
- `xstocks_reversion` — paused (Kraken xStocks wiring landed but disabled pending future evaluation).

**Recent production decisions (2026-04-23/24)**:

| ID | Description | Commit | Rationale |
|----|-------------|--------|-----------|
| DEC-TUNE-PHASE-A-001 | Session 34 Phase A tuning: `position_size_percent` 5.0→7.0, `volatility_gate_min_range_pct` 0.5→0.35, `bear_halt_min_confidence` 0.7→0.85 | `a6d2d91` (merged via `3b993a8`) | Session 33 analysis (30h 35m, 2 fills, -$8): 99.9% signal denial rate. 48% of 2,204 position_sizing denials were sub-$100 trade values (signal-strength scaling pushed trades below the $100 floor → bigger base size fixes it). 35% of BEAR-halt denials were false positives in a rising market (BTC +0.91%) → raise halt confidence floor to 0.85. Widening volatility gate captures more low-range entries. |
| DEC-CONDUCTOR-006 | Conductor capital conservation — residual redistribution in `_apply_allocations` | `04c046c` | Per-strategy trackers previously leaked surplus when one had post-fill cash above target. Introduces `global_equity_fn` wiring in `main.py` + `get_global_equity()` on paper adapter + `set_total_capital()` on allocator. Ensures total capital across trackers matches true equity every Conductor cycle. |
| DEC-CONDUCTOR-007 | `_refresh_allocator_performance` called per Conductor cycle; 3-trade minimum gate for Sharpe feed | `b92f9df` | DarwinianAllocator needs Sharpe input to reallocate. Previously wired but never called. 3-trade floor prevents noise-driven reallocation on single-trade strategies. |
| DEC-CONDUCTOR-008 | `_closed_trades` deque on `PortfolioTracker` | `b92f9df` | Sharpe requires historical closed-trade returns. Bounded deque prevents unbounded memory growth while preserving enough history for stable Sharpe. |
| DEC-CONDUCTOR-009 | Equity curve parameter reserved on allocator API | `b92f9df` | Forward-compatible API: pass equity curve alongside closed trades for future Sharpe variants (rolling/weighted). Currently unused but plumbed. |
| DEC-CONDUCTOR-010 | 3-trade minimum gate explicit in allocator | `c24e0e3` (merge) | Duplicates 007 gate at the allocator layer so the floor is enforced regardless of caller. Defense in depth. |
| DEC-CONDUCTOR-011 | Warmup behavior unchanged through DarwinianAllocator wiring | `c24e0e3` (merge) | Explicit no-op decision: warmup still runs on equal allocation until trade history matures. Preserves conservative startup behavior. |
| DEC-CONDUCTOR-012 | v3→v4 state migration, `CURRENT_STATE_VERSION=4` in `paper.py` | `096317d` | Closed-trade history must survive restart for Sharpe to work across sessions. v4 adds `closed_trades` to snapshot; v3 files upgrade with empty deque (no error). |

Detailed analyses in `docs/superpowers/plans/` and `/home/j/.claude/projects/.../memory/project_session34.md`.

## Technology Stack

| Component | Library | Rationale |
|-----------|---------|-----------|
| Exchange connectivity | `ccxt` (async) | Unified API for 100+ exchanges, Kraken WebSocket support |
| Data processing | `pandas`, `numpy` | Industry standard, fast vectorized operations |
| Technical analysis | `pandas-ta` | Pure Python, no C dependency headaches |
| ML / Regime detection | `scikit-learn` | Lightweight, interpretable models (HMM, RF) |
| Sentiment / NLP | `transformers` + `finbert` | Pre-trained financial sentiment |
| LLM reasoning | `anthropic` SDK | Claude for news interpretation and market reasoning |
| State persistence | `SQLite` + `SQLAlchemy` async | Zero-ops, file-based |
| Configuration | `pydantic-settings` | Typed config with env var support |
| Backtesting | `vectorbt` | Fast vectorized backtesting |
| HTTP / async | `aiohttp`, `asyncio` | Native async for non-blocking event loop |
| Testing | `pytest` + `pytest-asyncio` | Async test support |
| Logging | `structlog` | Structured JSON logs |

## Data Sources

| Source | Type | API |
|--------|------|-----|
| Kraken WebSocket | Price, volume, order book | ccxt async |
| CryptoPanic | Crypto news aggregator | Free API (200 req/hr) |
| Reddit (r/cryptocurrency) | Social sentiment | PRAW (free) |
| Fear & Greed Index | Market sentiment | alternative.me API (free) |
| CoinGecko | Market data, trending | Free tier (30 req/min) |
| NewsAPI.org | General financial news | Free tier (100 req/day) |

## Risk & Safety

- **API keys**: `.env` file, never in code, `.gitignore`'d
- **Rate limiting**: Per-adapter rate limiters respecting exchange limits
- **Circuit breakers**: Auto-halt on max drawdown (configurable, default 5%), API errors, data staleness
- **Position limits**: Max position size per asset, max total exposure, max daily loss
- **Graceful degradation**: If sentiment feed dies, continue on price signals with reduced confidence
- **Kill switch**: Manual override to flatten all positions immediately

## Phases

### Phase 1: Foundation (COMPLETED)
**Goal**: Project scaffolding, event bus, Kraken data flowing through the pipeline.

- [x] Initialize git repo, pyproject.toml with dependencies
- [x] Implement `core/bus.py` — async event bus with typed events
- [x] Implement `core/events.py` — MarketData, Signal, Order, Fill event types
- [x] Implement `core/config.py` — Pydantic settings from TOML + env vars
- [x] Implement `core/types.py` — shared type definitions
- [x] Implement `adapters/base.py` — abstract exchange adapter
- [x] Implement `adapters/kraken.py` — connect to Kraken, stream price data
- [x] Implement `adapters/paper.py` — paper trading execution engine
- **Verification**: Live Kraken price data flows through the event bus, paper orders execute -- VERIFIED 2026-02-17

### Phase 2: Signal Engine (COMPLETED)
**Goal**: Technical analysis signals producing actionable trade signals.

- [x] Implement `signals/base.py` — abstract signal generator with data accumulation
- [x] Implement `signals/candles.py` — tick-to-OHLCV candle aggregation
- [x] Implement `signals/technical.py` — RSI, MACD, Bollinger Bands, VWAP
- [x] Implement `signals/aggregator.py` — weighted signal combination with debounce
- [x] Implement `risk/portfolio.py` — position tracking, P&L, drawdown
- [x] Implement `risk/rules.py` — composable risk rules (position sizing, exposure, drawdown)
- [x] Implement `risk/manager.py` — risk orchestration, order emission
- [x] Basic strategy loop: data → candles → signals → aggregator → risk → paper execute
- **Verification**: Bot paper-trades based on technical signals with risk management -- VERIFIED 2026-02-17

### Phase 3: Intelligence Layer (COMPLETED)
**Goal**: News, sentiment, and regime detection augmenting signals.

- [x] Implement `intelligence/news.py` — CryptoPanic + NewsAPI ingestion
- [x] Implement `intelligence/llm.py` — Claude-powered news interpretation
- [x] Implement `intelligence/social.py` — Fear & Greed Index sentiment
- [x] Implement `signals/sentiment.py` — FinBERT sentiment scoring
- [x] Implement `signals/regime.py` — HMM-based regime detection
- [x] Wire intelligence into aggregator with regime-aware weighting
- **Verification**: 67 passed, 2 skipped, 0 failed — intelligence adjusts behavior based on news and regime shifts -- VERIFIED 2026-02-17

### Phase 4: Closed-Loop Learning (COMPLETED)
**Goal**: Agent learns from its own trading outcomes.

- [x] Implement `learning/tracker.py` — trade outcome tracking
- [x] Implement `learning/scorer.py` — signal effectiveness scoring
- [x] Implement `learning/adapter.py` — adaptive weight adjustment
- [x] Implement `core/state.py` — persist learning state across restarts
- **Verification**: Signal weights shift toward better-performing signals over time -- VERIFIED 2026-02-17

### Phase 5: Paper Trading Validation (COMPLETED)
**Goal**: Extended paper trading with monitoring and backtesting.

- [x] Implement `monitoring/stats.py` — performance metrics calculator (Sharpe, Sortino, drawdown)
- [x] Implement `monitoring/dashboard.py` — real-time CLI dashboard
- [x] Implement `monitoring/reporter.py` — session report generator
- [x] Implement `scripts/run_backtest.py` — backtesting runner via ccxt OHLCV replay
- [x] Implement `scripts/show_stats.py` — CLI stats viewer
- [x] Add MonitoringConfig to core config, wire dashboard into main
- **Verification**: 99 passed, 2 skipped, 0 failed — monitoring dashboard, stats calculator, backtesting, and session reporter operational -- VERIFIED 2026-02-17

### Phase 6: Live Trading & Plugin System (COMPLETED)
**Goal**: Graduate to real trading, enable future system integration.

- [x] Switch paper adapter to real Kraken execution with dual safety gate (DEC-LIVE-001)
- [x] Implement `plugins/base.py` — abstract plugin interface with lifecycle hooks (DEC-PLUGIN-001)
- [x] Implement `plugins/registry.py` — plugin discovery, lifecycle, error isolation (DEC-PLUGIN-002)
- [x] Implement `adapters/alpaca.py` — Alpaca stock adapter as multi-asset proof (DEC-ALPACA-001)
- [x] Add `config/live.toml` — conservative live trading configuration
- [x] Wire live/paper mode branching in `main.py`
- **Verification**: 110 tests pass, 3 skipped — live Kraken execution, plugin system, and Alpaca adapter operational -- VERIFIED 2026-02-17

## Decision Log

**Phase 1 & 2 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-BUS-001 | Async event bus with type-based subscriptions | Enables hot-swapping components, plugin architecture, future agentic integration | 2026-02-17 |
| DEC-EVENTS-001 | Immutable frozen dataclasses for all events | Prevents accidental mutation during event propagation; events are facts | 2026-02-17 |
| DEC-TYPES-001 | Decimal for all financial calculations | Avoids floating-point precision errors in financial math | 2026-02-17 |
| DEC-CONFIG-001 | Pydantic Settings with TOML + env var layering | Readable defaults/profiles (paper vs live); env vars for secrets | 2026-02-17 |
| DEC-ADAPTER-001 | Abstract adapter interface for exchange independence | Allows swapping exchanges without changing business logic | 2026-02-17 |
| DEC-KRAKEN-001 | ccxt.pro async WebSocket for real-time data | Unified WebSocket interface across exchanges with automatic reconnect | 2026-02-17 |
| DEC-PAPER-001 | File-based state persistence for paper trading | Simple JSON file persists balances/positions; no database needed yet | 2026-02-17 |
| DEC-MAIN-001 | Graceful shutdown with signal handlers | Proper cleanup prevents dangling WebSocket connections | 2026-02-17 |
| DEC-SIGNAL-001 | Abstract signal generator with automatic data accumulation | All technical signals need historical data; base class handles accumulation, subclasses focus on computation | 2026-02-17 |
| DEC-SIGNAL-002 | Candle aggregator for OHLCV bar construction | Technical indicators require OHLCV bars, not ticks; time-based aggregation from raw market data | 2026-02-17 |
| DEC-SIGNAL-003 | Normalized signal strength [-1.0, 1.0] convention | Uniform scale enables weighted combination in aggregator; -1=strong sell, +1=strong buy | 2026-02-17 |
| DEC-SIGNAL-004 | pandas-ta for technical indicator calculations | Pure Python, no C compilation needed; adequate coverage for RSI, MACD, BB, VWAP | 2026-02-17 |
| DEC-AGG-001 | Signal aggregator with weighted combination and debounce | Multiple signals produce conflicting recommendations; weighted voting with threshold prevents noise | 2026-02-17 |
| DEC-RISK-001 | Composable risk rules architecture | Each risk rule is independent and testable; rules can be enabled/disabled per config | 2026-02-17 |
| DEC-RISK-002 | Portfolio state tracking for exposure calculations | Centralized position tracking enables position sizing, exposure limits, and drawdown monitoring | 2026-02-17 |
| DEC-TEST-001 | Test real implementations, not mocks | Validates actual event behavior; follows Sacred Practice #5 (real implementations over mocks) | 2026-02-17 |
| DEC-TEST-002 | Async test fixtures for event bus validation | Event bus is async; tests must be async to verify queue behavior and subscriber notification | 2026-02-17 |
| DEC-TEST-003 | Test config validation and TOML loading | Validates Pydantic settings: type validation, percentage bounds, TOML composition | 2026-02-17 |
| DEC-TEST-004 | Test paper trading with real event bus integration | Verifies order execution, balance tracking, commission handling through real event flow | 2026-02-17 |
| DEC-TEST-005 | End-to-end pipeline test with mock Kraken data | Integration test verifies complete flow: MarketDataEvent → SignalEvent → OrderEvent → FillEvent | 2026-02-17 |

**Phase 3 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-INT-001 | Graceful degradation over hard failures | Each intelligence source can fail independently; system continues with whatever sources ARE available; technical signals always provide baseline | 2026-02-17 |
| DEC-INT-002 | Async polling with aiohttp | All news sources poll in background tasks; non-blocking event loop; proper timeout and error handling | 2026-02-17 |
| DEC-INT-003 | Optional heavy dependencies | FinBERT (transformers + torch) and hmmlearn in optional extras; simple fallbacks for regime detection; tests mock all models | 2026-02-17 |
| DEC-INT-004 | Rate limiting and cost control | LLM calls rate-limited (default: 10/hour); use claude-haiku-4-5 for cost efficiency; batch news articles to reduce API calls | 2026-02-17 |
| DEC-INT-005 | Regime-aware aggregation | Regime changes trigger weight adjustments; dynamic adaptation to market conditions; existing signals get context-aware weighting | 2026-02-17 |

**Phase 4 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-STATE-001 | SQLite with aiosqlite for async state persistence | Learning system needs durable storage; SQLite provides ACID guarantees with zero ops; aiosqlite enables async access | 2026-02-17 |
| DEC-LEARN-001 | Trade outcome tracking with signal snapshots | Capture signal state at trade entry to attribute P&L to specific signals; enables signal performance scoring | 2026-02-17 |
| DEC-LEARN-002 | Conservative EMA weight adaptation | EMA smoothing (alpha=0.1) prevents oscillation from single bad trades; gradual adaptation based on empirical performance | 2026-02-17 |
| DEC-LEARN-003 | Per-regime signal scoring | Different signals perform differently in different regimes; separate metrics enable context-aware weight adjustment | 2026-02-17 |

**Phase 5 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-MONITOR-001 | Pure function stats calculator | Performance metrics must be deterministic and testable; pure functions with no side effects enable easy unit testing | 2026-02-17 |
| DEC-MONITOR-002 | Event-driven dashboard with periodic updates | Dashboard subscribes to FillEvent/PositionUpdateEvent/TradeClosedEvent for real-time state; asyncio.sleep for periodic console updates (30s); non-blocking | 2026-02-17 |
| DEC-MONITOR-003 | Session reporter with file and console output | Post-session analysis requires comprehensive reports; uses stats.py for calculations; outputs to console and file with all metrics, trade list, regime breakdown | 2026-02-17 |
| DEC-MONITOR-004 | CLI stats viewer with SQLite queries | Quick command-line access to trade history, weight evolution, and regime data from persistent state | 2026-02-17 |
| DEC-MONITOR-005 | Backtesting via OHLCV replay through event bus | Reuses the same event bus pipeline for backtesting; downloads historical OHLCV via ccxt and replays as MarketDataEvents | 2026-02-17 |
| DEC-TEST-006 | Deterministic stats tests with trade fixtures | Performance metrics must be mathematically correct; use known PnL and equity curves to verify calculations | 2026-02-17 |

**Phase 6 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-LIVE-001 | Dual safety check for live trading (mode + env flag) | Requires both TRADING_MODE=live AND KRAKEN_LIVE_ENABLED=true before executing real orders; prevents accidental live trading | 2026-02-17 |
| DEC-PLUGIN-001 | Abstract plugin interface with lifecycle hooks | Plugins need initialize/start/stop hooks for clean lifecycle management; bus reference enables event subscription; metadata properties for discovery | 2026-02-17 |
| DEC-PLUGIN-002 | Error isolation in plugin registry | One failing plugin shouldn't crash the system; registry wraps plugin.start() in try/except, logs failures, continues with healthy plugins | 2026-02-17 |
| DEC-ALPACA-001 | Alpaca adapter for multi-asset proof | Stock trading proves multi-asset extensibility; Alpaca provides free paper trading for stocks; same ExchangeAdapter interface | 2026-02-17 |
| DEC-TEST-007 | Mock external APIs in tests | ccxt and alpaca-py are external dependencies; mock at API boundary (create_order, fetch_order) to test our logic without hitting real exchanges | 2026-02-17 |
| DEC-TEST-008 | Mock Alpaca API at client boundary | Tests mock TradingClient and StockHistoricalDataClient to verify adapter logic without API keys or market hours | 2026-02-17 |

### Phase 7: Paper Trading Improvement (COMPLETED)
**Goal**: Fix poor paper trading performance (25 trades, 20% win rate, Sharpe -1.42) with exit rules and signal quality improvements.

- [x] Add `entry_time` to `Position` dataclass for time-based exits (DEC-EXIT-001)
- [x] Implement `ExitMonitor` in `cerebrum/risk/exit_monitor.py` — stop-loss, take-profit, time-based exits (DEC-EXIT-001)
- [x] Add exit config fields to `RiskConfig` (stop_loss_percent, take_profit_percent, max_position_age_minutes)
- [x] Wire `ExitMonitor` in `cerebrum/main.py`
- [x] Fix signal aggregator consensus normalization — reward agreeing signals (DEC-AGG-002)
- [x] Add VWAP neutral zone to eliminate near-VWAP noise signals (DEC-SIGNAL-005)
- [x] Tune paper.toml: lower threshold to 0.4, increase position size to 3%
- **Verification**: 133 tests pass, 0 failures (2026-02-25)

**Phase 7 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-EXIT-001 | ExitMonitor as separate component from RiskManager | RiskRules evaluate proposed orders; ExitMonitor proactively generates exit orders. Separation of concerns: rules say yes/no, monitor watches and acts | 2026-02-21 |
| DEC-AGG-002 | Consensus multiplier via sqrt(buy_weight_fraction) | Rewards signal agreement without hard requirements. sqrt gives diminishing returns (100% consensus=1.0x, 50%=0.71x). Prevents 4 weak agreeing signals from equaling 1 strong signal when split | 2026-02-21 |
| DEC-SIGNAL-005 | VWAP neutral zone (0.5%) to filter near-VWAP noise | Price within 0.5% of VWAP is statistically insignificant; emitting signals there created noise trades. Neutral zone forces meaningful deviation before firing | 2026-02-21 |
| DEC-TEST-009 | Test exit monitor with real EventBus and PortfolioTracker | ExitMonitor interacts with EventBus and PortfolioTracker via events. Using real implementations validates the full event flow without mocks | 2026-02-21 |
| DEC-TEST-010 | Test PostFillCooldownRule with injectable clock and real EventBus | PostFillCooldownRule self-subscribes to FillEvents via the bus. Testing with a real EventBus validates subscription wiring, async event delivery, and per-symbol cooldown logic together | 2026-02-21 |
| DEC-COOL-001 | Post-fill cooldown rule prevents rapid-fire ordering | After a fill, the same symbol is blocked from new orders for cooldown_seconds (default 300s / paper 600s). Prevents churn in range-bound markets where the bot would otherwise oscillate | 2026-02-25 |
| DEC-DASH-001 | Publish PositionUpdateEvent on fills only (not price ticks) | Keeps dashboard updated without flooding the event bus. Price-tick updates would create thousands of events/minute; fill-driven updates are sparse and sufficient for display | 2026-02-25 |
| DEC-RISK-003 | Short position equity accounting — remove abs() from get_total_equity() | abs(amount) incorrectly added short positions' market value to equity instead of subtracting it, inflating _peak_equity and causing phantom drawdowns (6.2% reported vs 0.03% actual). Fix: signed amount * price so shorts reduce equity | 2026-02-25 |
| DEC-RISK-TEST-001 | Unit tests verify fill-driven PositionUpdateEvent publishing | Portfolio tracker must publish PositionUpdateEvent on every fill so the dashboard displays accurate open positions without requiring direct access to internal state | 2026-02-25 |

### Phase 8: Regime Detector Improvements (COMPLETED)
**Goal**: Fix 0/28 win rate from session 3 by improving regime detection to catch ultra-slow drifts.

- [x] Add three-metric regime detection (mean return + cumulative return + MA slope) — DEC-REGIME-001
- [x] Add buy suppression (0.2x) in high-confidence BEAR regime — DEC-REGIME-002
- [x] Add sentiment dampening in non-trending regimes (SIDEWAYS 0.4x, UNKNOWN 0.6x) — DEC-SENT-001
- [x] Add dual-window regime detection for ultra-slow drift (50-min long window) — DEC-REGIME-003
- **Verification**: All tests pass

**Phase 8 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-REGIME-001 | Cumulative return + MA slope for slow-trend detection | Original np.mean(returns) missed slow bleeds. Two new signals: cumulative_return (total drift) and ma_slope (directional momentum). BEAR/BULL if EITHER mean_return exceeds threshold OR both cumulative and ma_slope agree. Confidence from metric agreement count | 2026-03-11 |
| DEC-REGIME-002 | Buy suppression in high-confidence BEAR regime | Paper session went 0/20 buying into slow downtrend. When regime is BEAR with confidence >= 0.8, buy score is multiplied by 0.2 (configurable), making it nearly impossible to clear the signal threshold | 2026-03-11 |
| DEC-SENT-001 | Sentiment weight reduction in non-trending regimes | SIDEWAYS (0.4x) and UNKNOWN (0.6x) multipliers on base sentiment weight (0.5). Prevents Fear & Greed index from dominating aggregation when market is ranging or regime classifier lacks confidence | 2026-03-11 |
| DEC-REGIME-003 | Dual-window regime detection for ultra-slow drift | Single 5-min window cannot detect drifts slower than ~0.04%/min. 50-min long window (3000 ticks default) catches cumulative drifts as small as 0.1%. Only overrides SIDEWAYS — BULL/BEAR from short window is unchanged. Session 3 evidence: 0/28 win rate, -$128 PnL | 2026-03-12 |
| DEC-REGIME-004 | Trade halt in BEAR regime | Session 4 showed 39% win rate during BEAR vs 73% non-BEAR. Even sell-side trades lose during strong downtrends. Stopping all trading in BEAR would have saved $37+ in losses. The 0.2x buy suppression alone is insufficient — a full halt is needed. RegimeTradeHaltRule subscribes to REGIME_CHANGE events and denies all orders when regime=BEAR and confidence >= min_confidence | 2026-03-13 |
| DEC-TEST-011 | Test RegimeTradeHaltRule with real EventBus and RegimeChangeEvent | Validates that trading halts during high-confidence BEAR and permits during non-BEAR or low-confidence BEAR. Uses real EventBus (same pattern as test_cooldown_rule.py) to validate bus subscription wiring end-to-end | 2026-03-13 |

### Phase 9: Volatility Gate Risk Rule (COMPLETED)
**Goal**: Prevent trading during low-volatility SIDEWAYS markets where price range is insufficient to cover round-trip commissions (GitHub Issue #2).

- [x] Add `VolatilityGateRule` to `cerebrum/risk/rules.py` — deny orders when price range % is below threshold (DEC-VOL-001, DEC-VOL-002, DEC-VOL-003)
- [x] Add config fields `volatility_gate_min_range_pct` and `volatility_gate_window_size` to `RiskConfig` in `cerebrum/core/config.py`
- [x] Wire `VolatilityGateRule` into `cerebrum/main.py` risk rules list
- [x] Add defaults to `config/default.toml` and `config/paper.toml` under `[risk]`
- [x] Add `tests/unit/test_volatility_gate.py` with 7 test scenarios (DEC-TEST-012)
- **Verification**: 166 passed, 1 skipped — volatility gate denies orders in flat markets, approves during cold start, configurable via TOML (2026-03-14)

**Phase 9 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-VOL-001 | Percentage price range `(max - min) / min * 100` as volatility metric | Simple, interpretable, directly models whether price swings are large enough to profit after commission. No look-ahead bias. Avoids statistical complexity of std-dev which can be high even in one-directional trends | 2026-03-14 |
| DEC-VOL-002 | Per-symbol rolling price window via MARKET_DATA event bus subscription | Mirrors PostFillCooldownRule and RegimeTradeHaltRule patterns — self-subscribes in __init__, maintains per-symbol state dict of deque. Decoupled from regime detector; independent risk signal | 2026-03-14 |
| DEC-VOL-003 | Default threshold 0.5%, lookback 300 ticks, both configurable via TOML | 0.5% covers round-trip commission (~0.32%) plus slippage (~0.1%) with margin. 300 ticks (~5 min at 1 tick/sec) matches the regime detector's short window for consistency. APPROVE on cold start (fewer ticks than window) to not block early trades | 2026-03-14 |
| DEC-TEST-012 | Test VolatilityGateRule with real EventBus and injected prices | Self-subscribes to MARKET_DATA events. Real EventBus validates subscription wiring. Cold-start APPROVE tests insufficient-data path. Boundary test at exact threshold verifies >= semantics | 2026-03-14 |

### Phase 10: SIDEWAYS Guards + Adaptive TP + Data Export (COMPLETED)
**Goal**: Fix 0/17 SIDEWAYS loss rate (Issue #3) with four targeted guards, add adaptive take-profit, and add data export scripts for fine-tuning and analysis (Issue #4).

- [x] Add `SidewaysSuppressionRule` to `cerebrum/risk/rules.py` — deny BUY orders in SIDEWAYS+low-vol markets (DEC-REGIME-005)
- [x] Add `MacroVolatilityGateRule` to `cerebrum/risk/rules.py` — session-level (~5h) volatility gate to catch global flatness (DEC-VOL-004)
- [x] Add adaptive take-profit to `ExitMonitor` — scales TP to actual price range (DEC-EXIT-002)
- [x] Add `scripts/export_finetune.py` — LLM fine-tuning JSONL exporter (DEC-EXPORT-001)
- [x] Add `scripts/export_trades_csv.py` — parameter tuning CSV exporter (DEC-EXPORT-002)
- [x] Add `scripts/export_weights.py` — signal weight history CSV exporter (DEC-EXPORT-003)
- [x] Add denial counters to `RiskManager` and Guard Denials display to `Dashboard` (DEC-DENIAL-001)
- [x] Add `tests/unit/test_sideways_suppression.py` (DEC-TEST-013)
- [x] Add `tests/unit/test_macro_volatility_gate.py` (DEC-TEST-014)
- [x] Add `tests/unit/test_exit_monitor_adaptive.py` (DEC-TEST-015)
- [x] Add `tests/unit/test_export_scripts.py` (DEC-TEST-016)
- [x] Add `tests/unit/test_denial_counter.py` (DEC-TEST-017)
- **Verification**: All tests pass — SIDEWAYS guards, adaptive TP, and export scripts operational (2026-03-17)

**Phase 10 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-REGIME-005 | SIDEWAYS suppression: block BUY entries in low-vol sideways markets | Session 5 showed 0/17 win rate (-$61) in SIDEWAYS markets. Take-profit (3%) is unreachable when intraday range is <0.5%. All positions hit the 120-min max-age timeout at a guaranteed loss. Rule denies new BUY entries when regime==SIDEWAYS AND price range < min_range_pct. SELL orders never blocked | 2026-03-17 |
| DEC-VOL-004 | Macro-window volatility gate for session-level flatness detection | The 5-min VolatilityGateRule is fooled by local noise in a globally flat session. MacroVolatilityGateRule uses an 18000-tick window (~5h) to detect session-level flatness. Both gates must approve for a trade to proceed | 2026-03-17 |
| DEC-EXIT-002 | Adaptive take-profit based on recent price range | Session 5: fixed 3% TP unreachable in <0.5% intraday range. Adaptive TP: effective_tp = max(min_tp, range_pct * tp_multiplier). In 0.4% range: effective_tp = max(0.3%, 0.6%) = 0.6% — reachable while covering commission. Backward-compatible (adaptive_tp=False keeps fixed behaviour) | 2026-03-17 |
| DEC-EXPORT-001 | LLM fine-tuning JSONL exporter using raw sqlite3 (no ORM) | Export scripts run outside the full cerebrum system. Raw sqlite3 avoids asyncio-based StateManager and transitive deps. Pure stdlib deployable anywhere Python 3.10+ is available without a venv | 2026-03-17 |
| DEC-EXPORT-002 | Trades CSV exporter with flattened signal_snapshot columns | Parameter tuning requires flat tabular data. JSON signal_snapshot is destructured into signal_strength, signal_confidence, signal_action columns. ISO timestamps for human readability. hold_duration_min derived from entry/exit timestamps | 2026-03-17 |
| DEC-EXPORT-003 | Weight history CSV exporter with ISO timestamps | Signal weight evolution is key data for understanding adaptive learning. Exports weight_history table directly with human-readable timestamps | 2026-03-17 |
| DEC-DENIAL-001 | Denial counters in RiskManager, displayed in Dashboard Guard Denials section | Observability: knowing which rules fire most often guides tuning. Counters are per-rule dicts incremented on every DENY in _apply_rules. Dashboard shows counts when risk_manager is provided (optional parameter) | 2026-03-17 |
| DEC-RISK-MGR-001 | RiskManager as stateful coordinator with per-rule denial counters | RiskManager owns the full order validation pipeline: receives combined SignalEvents, creates proposed orders, applies risk rules in sequence (any DENY blocks; MODIFY adjusts amount), and emits approved OrderEvents. Denial counters accumulated here rather than in individual rules because the manager is the single choke-point that sees every denial. Single-threaded asyncio means no locking needed | 2026-03-17 |
| DEC-TEST-013 | Tests for SidewaysSuppressionRule | Subscribes to both REGIME_CHANGE and MARKET_DATA events. Real EventBus validates subscription wiring. SELL orders must always be approved. BULL/BEAR regimes bypass suppression entirely | 2026-03-17 |
| DEC-TEST-014 | Tests for MacroVolatilityGateRule | Subscribes to MARKET_DATA events. Key test: local noise passes short gate but macro gate blocks when session is globally flat | 2026-03-17 |
| DEC-TEST-015 | Tests for adaptive take-profit in ExitMonitor | White-box test of _compute_effective_tp method. End-to-end test verifies exit orders fire at expected gain level below fixed TP threshold | 2026-03-17 |
| DEC-TEST-016 | Tests for export scripts using in-memory SQLite | Export scripts use raw sqlite3. Tests use in-memory SQLite DB with seeded trade data to verify JSONL validity, CSV headers, and record counts | 2026-03-17 |
| DEC-TEST-017 | Tests for RiskManager denial counters | Verifies counters increment on DENY, remain at zero for APPROVE/MODIFY, and denial_counts returns a copy not the live dict | 2026-03-17 |
| DEC-TEST-LEARN-001 | Test trade tracker lifecycle and sell-fill correlation fix | Covers the Session 6 bug (sell fills creating phantom shorts instead of closing matching open BUY trades). Tests prove: (1) BUY fill opens a trade, (2) subsequent SELL fill closes it via FIFO matching, (3) unmatched SELL fill is silently skipped and does NOT open a phantom SELL record. Real EventBus and in-memory SQLite — no mocks | 2026-03-23 |

### Phase 11A Prerequisites: Multi-Strategy Groundwork (COMPLETED)
**Goal**: Four self-contained changes that lay the groundwork for the multi-strategy agent swarm refactor.

- [x] Fix trade tracker DB bug — unmatched SELL fills must not create phantom short positions (DEC-PREREQ-001)
- [x] Add `strategy_id: str | None = None` to SignalEvent, OrderEvent, FillEvent (DEC-PREREQ-001)
- [x] PaperAdapter propagates strategy_id from OrderEvent to FillEvent
- [x] Refactor RiskManager._apply_rules to use dataclasses.replace() (DEC-RISK-MGR-002)

**Phase 11A Prerequisites Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-PREREQ-001 | strategy_id on SignalEvent/OrderEvent/FillEvent; tracker long-only fix | strategy_id added as last optional field (None default) for forward-compatible multi-strategy routing. Tracker fix: SELL fills with no matching open BUY trade are logged+skipped instead of creating phantom shorts that corrupt subsequent BUY matching (Session 6: 18/19 trades stuck OPEN) | 2026-03-23 |
| DEC-RISK-MGR-002 | Use dataclasses.replace() for frozen OrderEvent mutation in _apply_rules | Manual field-by-field reconstruction breaks when new fields are added to OrderEvent (e.g. strategy_id). replace() copies all unspecified fields automatically, making the MODIFY path forward-compatible | 2026-03-23 |

### Session 7 Bug Fixes (COMPLETED)
**Goal**: Fix three bugs discovered in Session 7 deep analysis that left the multi-strategy system in a severely degraded state.

- [x] TradeTracker subscribes to REGIME_CHANGE so _current_regime stays current (DEC-S7-001)
- [x] get_open_trades() adds ORDER BY entry_time ASC for deterministic FIFO matching (DEC-S7-002)
- [x] strategy_id propagated through TradeRecord, DB schema, save_trade, _row_to_trade_record, RiskManager signal_snapshot, and OrderEvent (DEC-S7-003)

**Session 7 Bug Fix Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-S7-001 | TradeTracker subscribes to REGIME_CHANGE via bus in start() | update_regime() existed but was never called. Tracker initialised with "UNKNOWN" and stayed there for entire session — every trade record stored regime="UNKNOWN". Fix: subscribe in start() using same pattern as RegimeTradeHaltRule. Handler calls update_regime() then logs the transition | 2026-03-24 |
| DEC-S7-002 | get_open_trades() adds ORDER BY entry_time ASC | Without ORDER BY, SQLite returns rows in unspecified order. FIFO matching relied on open_trades[0] being the oldest trade — non-deterministic with multiple open positions. Fix: both symbol-filtered and unfiltered queries now ORDER BY entry_time ASC | 2026-03-24 |
| DEC-S7-003 | strategy_id field on TradeRecord, DB column via ALTER TABLE migration, propagated from FillEvent | Three-layer gap: strategy_id existed on FillEvent but (1) TradeRecord had no field, (2) DB had no column, (3) _open_trade() didn't read it. Fix: add strategy_id: str | None = None to TradeRecord dataclass; ALTER TABLE migration in _create_schema() with try/except for existing DBs; save_trade() includes it in INSERT; _row_to_trade_record() reads via dict(row).get() for old-DB compat; RiskManager propagates signal.strategy_id into both signal_snapshot dict and OrderEvent.strategy_id | 2026-03-24 |
| DEC-TEST-S7-001 | Session 7 bug-fix regression tests — real EventBus + in-memory SQLite | Covers all three bugs: (1) bus subscription wiring for regime updates, (2) ORDER BY correctness with out-of-order insertion times, (3) strategy_id round-trip through fill → tracker → DB → retrieval. No mocks — same pattern as test_learning.py and test_regime_halt.py | 2026-03-24 |

### Phase 11A: Strategy Abstraction Layer (COMPLETED)
**Goal**: Create the multi-strategy foundation — isolated pipelines per strategy, shared signal generators, global guards, backward-compatible single-strategy mode.

- [x] Create `cerebrum/strategies/base.py` — StrategyConfig pure config dataclass (DEC-STRAT-001, DEC-STRAT-002)
- [x] Create `cerebrum/strategies/momentum.py` — MomentumStrategy config matching current paper.toml values exactly (DEC-STRAT-006)
- [x] Create `cerebrum/strategies/registry.py` — StrategyRegistry: creates aggregator+portfolio+exit_monitor+risk_manager per strategy with error isolation (DEC-STRAT-003)
- [x] Create `cerebrum/strategies/global_portfolio.py` — GlobalPortfolio: aggregates equity across all PortfolioTrackers (DEC-STRAT-004)
- [x] Create `cerebrum/risk/global_trade_rate.py` — GlobalTradeRateLimitRule: cross-strategy fill rate limit (DEC-STRAT-007)
- [x] Modify `cerebrum/signals/aggregator.py` — add strategy_id param; tag COMBINED signals (DEC-STRAT-005)
- [x] Modify `cerebrum/risk/manager.py` — add strategy_id filter in _on_signal (DEC-STRAT-005)
- [x] Modify `cerebrum/main.py` — wire StrategyRegistry with backward compat (DEC-STRAT-006, DEC-MAIN-002)
- [x] Add `cerebrum/risk/portfolio.py` — strategy_id filtering on PortfolioTracker (DEC-RISK-004)
- [x] Add `cerebrum/signals/support_resistance.py` — pivot-based S/R signal (DEC-SIGNAL-006)
- [x] Add tests: test_strategy_registry.py, test_global_portfolio.py, test_global_trade_rate.py, test_single_strategy_regression.py
- **Verification**: All tests pass

**Phase 11A Strategy Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-STRAT-001 | StrategyConfig as a pure config dataclass, not an actor | Rejected "Strategy as actor" pattern (generate_signals(), on_fill()). Config object is simpler to test, version, and serialize. StrategyRegistry owns wiring logic; config owns parameters. Enables backward compat: single MomentumStrategy config wraps current main.py wiring | 2026-03-23 |
| DEC-STRAT-002 | Isolated portfolios per strategy — no compound keys | Each strategy gets its own PortfolioTracker with own cash balance and position set. Cross-strategy aggregation via GlobalPortfolio (read-only view). Avoids compound symbol/strategy_id keys, simplifies per-strategy P&L attribution, prevents one strategy's drawdown circuit-breaker from triggering another's | 2026-03-23 |
| DEC-STRAT-003 | StrategyRegistry owns pipeline lifecycle with error isolation | Models plugins/registry.py pattern: register(), start_all(), stop_all() with try/except isolation. One strategy failing to start doesn't block others. Shared global guards (RegimeTradeHaltRule etc.) constructed once and passed to all RiskManagers by reference | 2026-03-23 |
| DEC-STRAT-004 | GlobalPortfolio as read-only aggregation view | Aggregates equity, drawdown, positions across all strategy PortfolioTrackers. Read-only (no event subscriptions). Drawdown computed from global peak across all strategies combined. Provides per-strategy equity delegation for dashboard and monitoring | 2026-03-23 |
| DEC-STRAT-005 | strategy_id tagging on aggregator output, filtering on RiskManager input | Each strategy's SignalAggregator tags COMBINED signals with strategy_id=config.name. Each strategy's RiskManager filters _on_signal to skip signals where event.strategy_id != self._strategy_id. None strategy_id on either side = no filtering (backward compat for legacy callers) | 2026-03-23 |
| DEC-STRAT-006 | Backward-compatible single-strategy mode in main.py | If no strategy config in TOML: create a single MomentumStrategy pipeline identical to current main.py wiring. If strategy configs present: use StrategyRegistry. Zero behavioural change for existing paper/live runs | 2026-03-23 |
| DEC-STRAT-007 | GlobalTradeRateLimitRule for cross-strategy commission control | Session 4 showed 64% commission drag. A per-strategy rate limit doesn't prevent all five strategies from trading simultaneously. Global cap of 40 trades/hour (5 strategies × ~8 trades) enforces system-wide commission budget | 2026-03-23 |
| DEC-RISK-004 | strategy_id filtering in PortfolioTracker prevents double-counting | In multi-strategy mode, each PortfolioTracker must only process fills for its own strategy. strategy_id=None accepts all fills (single-strategy backward compat). Prevents triple-counting positions when multiple trackers share one event bus | 2026-03-23 |
| DEC-SIGNAL-006 | Pivot-based S/R detection with proximity signals | Simple pivot point detection (local highs/lows over N candles) provides structural price levels for range trading. Proximity threshold (0.5% default) determines when price is testing a level. Emits BUY near support, SELL near resistance | 2026-03-23 |
| DEC-MAIN-002 | Multi-strategy mode as default with single-strategy fallback | Multi-strategy pipeline (StrategyRegistry + DarwinianAllocator + Conductor + WebDashboard) is Phase 11 target. CEREBRUM_MULTI_STRATEGY env var controls mode (default true). Single-strategy preserved for backward compatibility | 2026-03-23 |

### Phase 11B: Additional Strategies + Conductor + Dashboard (COMPLETED)
**Goal**: Add range trading, swing trading, and news-driven strategies; LLM Conductor for Darwinian capital allocation; htmx web dashboard.

- [x] Create `cerebrum/strategies/mean_reversion.py` — MeanReversionStrategy config (DEC-STRAT-008)
- [x] Create `cerebrum/strategies/breakout.py` — BreakoutStrategy config (DEC-STRAT-009)
- [x] Create `cerebrum/strategies/range_trading.py` — RangeTradingStrategy: S/R-only signals, SIDEWAYS exemption (DEC-RANGE-006)
- [x] Create `cerebrum/strategies/range_detector.py` — RangeDetector: bounce evidence accumulation (DEC-RANGE-001, DEC-RANGE-002, DEC-RANGE-003)
- [x] Create `cerebrum/risk/range_exit_monitor.py` — structural exits for range trading (DEC-RANGE-004, DEC-RANGE-005)
- [x] Create `cerebrum/strategies/swing_trading.py` — 1h timeframe strategy to reduce commission drag (DEC-SWING-001)
- [x] Create `cerebrum/strategies/news_driven.py` — news-heavy signal weighting (DEC-NEWS-001)
- [x] Create `cerebrum/conductor/allocator.py` — DarwinianAllocator: Sharpe-based capital allocation (DEC-ALLOC-001, DEC-ALLOC-002, DEC-ALLOC-003)
- [x] Create `cerebrum/conductor/conductor.py` — LLM Conductor: event+polling hybrid, graceful degradation (DEC-CONDUCTOR-001, DEC-CONDUCTOR-002, DEC-CONDUCTOR-003, DEC-CONDUCTOR-004)
- [x] Create `cerebrum/dashboard/web.py` — htmx+FastAPI web dashboard with copilot mode (DEC-DASH-002, DEC-DASH-003)
- [x] Add 1h CandleAggregator + timeframe-tagged signal generators to main.py
- [x] Add tests: test_range_detector.py, test_range_exit_monitor.py, test_range_trading_integration.py, test_allocator.py, test_conductor.py, test_web_dashboard.py, test_multi_timeframe.py, test_news_driven.py, test_strategies.py, test_support_resistance.py, test_main_wiring.py
- **Verification**: All tests pass

**Phase 11B Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-STRAT-008 | MEAN_REVERSION_CONFIG as StrategyConfig for StrategyRegistry wiring | MeanReversionStrategy is a documentation dataclass; config is the real artifact consumed by StrategyRegistry. Higher RSI thresholds (oversold=20, overbought=80) target more extreme mean-reversion setups | 2026-03-24 |
| DEC-STRAT-009 | BREAKOUT_CONFIG as StrategyConfig for StrategyRegistry wiring | Mirrors DEC-STRAT-008. Breakout uses MACD + Bollinger Bands heavily, lower RSI weight. Targets strong momentum continuation after consolidation breakouts | 2026-03-24 |
| DEC-RANGE-001 | RangeDetector as a queryable state object, not an event emitter | Range detection requires accumulating bounce evidence over time. State object is simpler to test and reason about than an event-driven accumulator that must handle ordering and timing of bus events | 2026-03-24 |
| DEC-RANGE-002 | Bounce deduplication via proximity zone tracking | Without deduplication, a burst of S/R signals during a single price test counts as multiple bounces, inflating confidence and triggering false range confirmation | 2026-03-24 |
| DEC-RANGE-003 | Regex-based level extraction from signal.reason string | SupportResistanceSignal encodes the actual level price in its reason string. Regex extraction avoids adding a new field to the signal event type, keeping the event schema stable | 2026-03-24 |
| DEC-RANGE-004 | Structural exits over percentage-based for range trading | Fixed % TP is unreachable in tight ranges (Session 5: 0/17). Range trading exits at confirmed S/R levels: take profit near resistance, stop loss below support, giving the trade room to reach the structural target | 2026-03-24 |
| DEC-RANGE-005 | Fallback to percentage-based exits when no confirmed range exists | After regime change or when S/R levels haven't accumulated enough bounces, RangeExitMonitor falls back to standard percentage-based stop-loss and take-profit to prevent unprotected positions | 2026-03-24 |
| DEC-RANGE-006 | Dedicated range strategy with S/R-only signal filtering | Mean reversion uses RSI/MACD with different weights — doesn't model range boundaries explicitly. Range trading uses S/R signals exclusively and is exempt from SidewaysSuppressionRule (range is the target regime) | 2026-03-24 |
| DEC-SWING-001 | 1-hour timeframe swing strategy to reduce commission drag | Session 4: $115 commission on $179 gross (64%). 1h candles → fewer signals → fewer trades → lower commission ratio. Dedicated 1h CandleAggregator; timeframe metadata tag filters swing signals away from 1m strategies | 2026-03-25 |
| DEC-NEWS-001 | News-heavy signal weighting for event-driven trading | LLM news analyzer already generates SignalType.NEWS signals scored [-1,1] by Claude. News-driven strategy ups the weight to 0.6 (vs 0.1 default) making it the dominant signal. Targets crypto market movers driven by news events | 2026-03-25 |
| DEC-NEWS-002 | Test news-driven config values against spec, not behavior | NEWS_DRIVEN_CONFIG is a pure data object (frozen StrategyConfig). Tests assert field values directly rather than mocking aggregator/risk manager. Config fields are the contract; tests enforce stability across refactors | 2026-03-25 |
| DEC-ALLOC-001 | Darwinian capital allocation via rolling Sharpe ratio | Strategies compete for capital based on risk-adjusted returns. Sharpe ratio penalizes both low returns and high volatility. Rolling window (20 trades) adapts to recent performance without over-indexing on one bad trade | 2026-03-25 |
| DEC-ALLOC-002 | Auto-reactivation with exponential backoff prevents permanent deadlock | A paused strategy stays paused forever under naive Darwinian selection. Exponential backoff (2^n hours) gives struggling strategies occasional capital to prove recovery | 2026-03-25 |
| DEC-ALLOC-003 | All-paused edge case: reactivate the least-bad strategy | If every strategy falls below the pause threshold simultaneously, the system would hold 100% cash. Reactivating the strategy with the best Sharpe ratio preserves trading activity | 2026-03-25 |
| DEC-CONDUCTOR-001 | Event-driven + polling hybrid LLM conductor | Pure polling misses immediate regime changes. Pure event-driven cannot enforce periodic rebalancing. Hybrid: poll every N minutes AND rebalance on REGIME_CHANGE events | 2026-03-25 |
| DEC-CONDUCTOR-002 | Freeze allocations on API failure, never reset | A trading system must degrade gracefully. If the Claude API is unavailable, freeze current allocations rather than resetting to equal weight (which would trigger disruptive capital transfers) | 2026-03-25 |
| DEC-CONDUCTOR-003 | Math-only mode when no API key provided | DarwinianAllocator alone is genuinely useful — it adjusts capital based on Sharpe. No API key = math-only mode. LLM reasoning is additive, not required | 2026-03-25 |
| DEC-CONDUCTOR-004 | 50% single-strategy allocation cap to prevent peak-equity spikes | Haiku returned 75% to range_trading at T+90s, injecting $5,000 into a $2,500 portfolio. When Haiku reverted at T+3:44, _peak_equity held $7,500, producing a permanent 66.7% false drawdown that blocked all trading for the rest of the session. Capping at 50% limits the worst-case transient spike to 2x base allocation, keeping false drawdown below any reasonable circuit-breaker threshold | 2026-03-25 |
| DEC-CONDUCTOR-005 | Normalize LLM allocation fractions to percentages | Haiku returns 0.25 instead of 25 for "25%". Rather than relying on prompt engineering, detect sum(allocations) <= 2 and multiply by 100. This is the single normalization point since _apply_allocations() is called by all allocation sources | 2026-03-25 |
| DEC-DASH-002 | htmx + FastAPI web dashboard for multi-strategy visualization | Pure server-side rendering with htmx for partial updates eliminates JavaScript complexity. FastAPI serves JSON API + HTML template. Auto-refresh every 5s via htmx polling | 2026-03-25 |
| DEC-DASH-003 | Copilot mode queues pending allocations rather than blocking the Conductor | When copilot_mode=True, Conductor produces an allocation proposal but does not apply it. Dashboard displays the pending allocation for human review and approval via /approve endpoint | 2026-03-25 |
| DEC-TEST-SR-001 | S/R tests with synthetic candle data for deterministic pivot detection | Pivot detection depends on candle patterns. Tests use synthetic candle sequences with known pivot highs/lows to verify detection, clustering (merging nearby pivots), and proximity signal generation. Async tests verify event bus integration. Sync unit tests use EventBus mock only for pure-computation methods | 2026-03-25 |
| DEC-TEST-STRAT-001 | Strategy config tests covering regime affinity and weight math | Strategy presets drive aggregator weights and risk params. Tests verify mean reversion favors SIDEWAYS with tight TP, breakout favors BULL/VOLATILE with wide TP, and the two strategies cover complementary regimes without overlap | 2026-03-25 |
| DEC-TEST-RANGE-001 | Real EventBus for all RangeDetector tests — no mocks | RangeDetector is an async subscriber that processes events through the bus pipeline. Testing with a real EventBus exercises the actual delivery path (queue, task dispatch, handler invocation) and guards against timing bugs that mocks would hide | 2026-03-25 |
| DEC-TEST-RANGE-INT-001 | Integration tests prove cross-component range trading wiring | Unit tests verify each component in isolation. Integration tests verify SidewaysSuppressionRule + SignalAggregator + RangeDetector interact correctly when wired to a shared EventBus | 2026-03-25 |
| DEC-TEST-MAIN-001 | Test strategy_id filtering and config instances with real implementations | Sacred Practice #5: all tests use real implementations, no internal mocks. PortfolioTracker, StrategyRegistry, and CerebrumCoin._setup_* are exercised directly with in-memory EventBus instances and real Config loaded from paper.toml | 2026-03-25 |

### Phase 11C: Per-Strategy State Persistence (COMPLETED)
**Goal**: Fix dashboard showing stale equity on restart by persisting per-strategy PortfolioTracker state (cash, positions, peak_equity, realized_pnl) in paper_state.json.

- [x] Add `save_snapshot()` / `restore_snapshot()` to `PortfolioTracker` in `cerebrum/risk/portfolio.py` (DEC-PERSIST-001, DEC-RISK-005)
- [x] Extend `PaperTradingAdapter` with v2 state format: `set_strategy_portfolios()`, `get_strategy_snapshot()`, v2 `_save_state()`, backward-compat `_load_state()` (DEC-PERSIST-001)
- [x] Wire restore in `cerebrum/main.py` after `strategy_registry.start_all()`
- [x] Add tests: test_portfolio_persistence.py — 6 scenarios covering roundtrip, positions, v2 format, v1 compat, missing/new strategy
- **Verification**: All tests pass (PR #14 merged)

**Phase 11C Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-PERSIST-001 | Per-strategy PortfolioTracker snapshots in paper_state.json v2 format | Each strategy has isolated PortfolioTracker (cash, positions, peak_equity, realized_pnl). Without per-strategy snapshots, all per-strategy equity is lost on restart and the dashboard shows stale global aggregates. v2 adds strategy_snapshots key alongside v1 fields; v1 files load cleanly (no version key = empty snapshots, no error). initial_balance is not restored — fixed at construction time | 2026-03-25 |
| DEC-RISK-005 | Peak equity lowered on capital withdrawal to prevent false drawdown | Conductor reallocation injects then later withdraws capital. Without peak-lowering on withdrawal, _peak_equity holds a transient high and the drawdown calculation permanently exceeds the circuit-breaker threshold, blocking all trading. Fix: on negative delta, lower peak by the same amount (floor at new equity so real losses are preserved) | 2026-03-25 |

### Phase 12: Proving Ground — Strategy Attribution + Backtest + Tuning (IN PROGRESS)
**Goal**: Build analysis tooling, full multi-strategy backtest, parameter sensitivity, and dashboard upgrades to enable data-driven go-live decisions. Scorecard not yet passing — 30-day proving ground in progress.

- [x] 12A: Strategy attribution analysis (`scripts/analyze.py`) — PR #18
- [x] 12B: Go-live scorecard (in analyze.py) — PR #18
- [x] 12C: Fix Sharpe ratio to use percentage returns — PR #17
- [x] 12D: Full multi-strategy backtest (`scripts/run_backtest.py`) — PR #21
- [x] 12E: Parameter sensitivity analysis (`scripts/sensitivity.py`) — PR #19
- [x] 12F: Dashboard upgrades (equity curves, scorecard, commission, heatmap) — PR #20
- [x] 12G: Backtest fixes (OHLCV pagination, Bitstamp data, virtual clock, paper adapter clock) — PRs #22-27
- [x] 12H: Data-driven SL tuning (momentum 1.0%, breakout 1.5%, range_trading 0.5%) — PR #28
- [ ] 12I: 30-day proving ground evaluation — Session 13 running
- [ ] 12J: Live trading preparation ($100 pilot) — after scorecard passes

**Phase 12 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-ANALYZE-001 | analyze.py replicates stats logic inline (no cerebrum imports) | Follows DEC-EXPORT-001 pattern: standalone scripts avoid importing asyncio-based StateManager and transitive dependencies. Sharpe/Sortino/drawdown logic replicated using raw Decimal arithmetic and sqlite3. Keeps script deployable anywhere Python 3.10+ is available without a full venv | 2026-03-25 |
| DEC-ANALYZE-002 | Tests use in-memory SQLite with seeded trades — no production DB | Follows DEC-TEST-016 pattern. All scorecard logic, commission calculations, and Sharpe correctness are verified against deterministic seeded data before touching production data | 2026-03-25 |
| DEC-ANALYZE-003 | weekly_report.py delegates to generate_report() from analyze.py | Code reuse without circular imports: weekly_report imports the pure generate_report() function from analyze.py rather than shelling out. Week number is auto-detected from earliest trade timestamp to avoid manual tracking | 2026-03-25 |
| DEC-STATS-001 | Sharpe/Sortino use percentage returns | Raw dollar P&L was scale-dependent | 2026-03-25 |
| DEC-SENSITIVITY-001 | sensitivity.py imports TradeRow/calculate_commission from analyze.py; simulation logic is pure (no I/O) | Reuses the stable TradeRow dataclass and commission formula from analyze.py rather than duplicating. Simulation functions (simulate_sl, simulate_tp, simulate_age, simulate_cooldown) are pure functions that accept a TradeRow and return a simulated P&L Decimal — no side effects, trivially testable | 2026-03-25 |
| DEC-SENSITIVITY-002 | Tests use in-memory SQLite + direct TradeRow construction for simulation unit tests | DB tests verify fetch + filter logic; direct TradeRow construction for simulation logic (no DB needed to test math). Follows DEC-ANALYZE-002 pattern. Grid cap test uses warnings.catch_warnings to assert UserWarning emission | 2026-03-25 |
| DEC-DASH-004 | Phase 12F state tracked from FillEvents in-memory — no DB queries | The dashboard runs embedded in the trading process and must stay lightweight. Per-strategy equity history, fill counts, commission totals, and realized P&L are accumulated incrementally from FillEvent callbacks. Authoritative analysis (Sharpe, full attribution) is deferred to scripts/analyze.py which queries the trade DB directly; the scorecard notes this for criteria that cannot be computed inline. | 2026-03-25 |
| DEC-BACKTEST-001 | Backtest reuses entire live pipeline — no separate signal logic | The only change from live is the data source: OHLCV CSV instead of Kraken WebSocket. All signal generators, risk rules, strategy registry, and paper adapter are identical. This validates strategies against real-world data using the same execution path they'll see in production | 2026-03-26 |
| DEC-BACKTEST-002 | News/sentiment and Conductor skipped in backtest | News feeds, LLM analyzer, fear/greed sentiment, and FinBERT are real-time push systems with no historical data equivalent. Conductor uses LLM API with cost. Both are skipped: news_driven strategy gets zero NEWS signals (noted in output), Conductor allocation is replaced by equal static allocation. RegimeDetector is included (feeds on price data alone) | 2026-03-26 |
| DEC-BACKTEST-003 | Scale time-based params for non-1m candles | Guard windows calibrated for 1m ticks | 2026-03-26 |
| DEC-BACKTEST-004 | Injectable BacktestClock for virtual time | SignalAggregator, PostFillCooldownRule, PaperAdapter all used time() | 2026-03-26 |
| DEC-TUNE-002 | Mean reversion position_size 3%→5% | At $1,666.67 capital, 3% = $50 trades where commission eats profits | 2026-03-25 |
| DEC-TUNE-003 | Mean reversion cooldown 600→900s | Match other strategies, reduce churn in SIDEWAYS | 2026-03-25 |
| DEC-TUNE-004 | Tighten SL: momentum 1.0%, breakout 1.5%, range_trading 0.5% | Session 11 sensitivity analysis: SL=1.0% improves AdjPnL by +$118 | 2026-03-26 |
| DEC-SHUTDOWN-001 | Graceful position liquidation on shutdown | Open positions persisted across sessions create phantom P&L. Session 11 showed $10,750.72 equity with unrealized gains never realized. _close_all_positions() publishes OrderEvents to the still-running bus so fills flow through normal PortfolioTracker path, realizing P&L before _save_state() captures clean state | 2026-03-27 |
| DEC-TEST-SHUTDOWN-001 | Test shutdown liquidation with real bus and in-memory PortfolioTracker | Sacred Practice #5: real EventBus + real PortfolioTracker. Verifies SELL for long, BUY for short, zero-amount skip, error isolation, multi-strategy registry path | 2026-03-27 |

---

### Phase 13: Stocks Trading Expansion (IN PROGRESS)
**Goal**: Enable stock trading alongside existing crypto strategies via Alpaca, with proper symbol routing, NYSE market hours awareness, and paper trading validation before any live stocks execution.

**Scope**: Paper-only. Existing strategies trade stock symbols using the same logic as crypto. Stock-specific strategies, backtesting data pipeline, and live stocks execution are deferred to Phase 14.

- [x] 13A: Alpaca + Kraken `strategy_id` fix; `AlpacaConfig` dataclass; commented `[alpaca]` section in paper.toml — `fb4b5fa`
- [x] 13B: `MarketHoursRule` — NYSE calendar guard (9:30–16:00 ET, 10 holidays, 60s cache, crypto always passes) — `fb4b5fa`
- [ ] 13C: WebSocket streaming upgrade for AlpacaAdapter (replace 5s polling; polling remains as fallback)
- [ ] 13D: Multi-adapter routing — `OrderRouter` dispatches stock symbols to Alpaca, crypto to Kraken/paper; wire both adapters + `MarketHoursRule` into `main.py`
- [ ] 13E: Mixed paper session validation — crypto on Kraken + stocks on Alpaca paper; 30-day proving ground before live stocks

**Waiting on**: Phase 12J ($100 live crypto pilot) before proceeding with 13C–13E.

**Phase 13 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-ALPACA-CONFIG-001 | AlpacaConfig as optional BaseSettings with all safe defaults | Alpaca is an optional extension for stock trading. Empty-string API key defaults mean the system boots without Alpaca credentials and only activates stock trading when keys are explicitly configured. Empty default symbols list prevents unexpected stock subscriptions on first boot | 2026-03-28 |
| DEC-ALPACA-FIX-001 | Propagate `strategy_id` from OrderEvent to FillEvent in AlpacaAdapter and KrakenAdapter | All three adapters must propagate strategy_id consistently so Conductor can attribute stock fills to the correct strategy. Omission was a Phase 6 POC gap | 2026-03-28 |
| DEC-HOURS-001 | Local NYSE calendar in MarketHoursRule — no external API | External clock API calls introduce network dependency in the risk path. Local calendar using `zoneinfo` + Anonymous Gregorian Easter algorithm covers all 10 NYSE holidays deterministically. Crypto symbols (`/`) always bypass the rule | 2026-03-28 |
| DEC-TEST-HOURS-001 | Test market hours with datetime monkeypatching at module level | MarketHoursRule._compute_market_open() calls datetime.now(tz=...) internally. Monkeypatching datetime.now on the market_hours module directly is the standard pattern for testing time-dependent logic without exposing test-only seams in production code | 2026-03-28 |
| DEC-COMMISSION-001 | Commission-aware minimum trade viability gate | Session 17 showed trades too small to overcome 0.32% round-trip commission. CommissionGateRule: threshold = commission_percent * 2 * min_profit_to_commission_ratio. If range_pct < threshold, deny. Dynamically computed per-session vs. the static VolatilityGateRule threshold, making the gate self-calibrating as commission rates change | 2026-03-28 |
| DEC-ROUTE-001 | Symbol format determines adapter: `/` → crypto (Kraken/paper), no `/` → stocks (Alpaca) | Crypto pairs use ccxt format (`BTC/USD`); stock tickers have no slash (`AAPL`). Universal convention, no extra config needed for common cases | TBD (13D) |
| DEC-ALPACA-002 | WebSocket streaming via `StockDataStream`, polling as fallback | Alpaca free tier provides IEX data via WebSocket — sufficient for paper validation | TBD (13C) |
| DEC-ALPACA-003 | Alpaca's built-in paper mode serves as the stocks paper adapter | No separate PaperTradingAdapter needed for stocks. `paper=True` config flag controls endpoint | TBD (13D) |

---

### Session 18 Bug Fixes + Tuning (COMPLETED)
**Goal**: Fix infinite exit loop bug discovered in Session 18 (37 rapid-fire SOL sells), add CommissionGateRule to block unprofitable micro-trades, and apply empirical tuning based on 60-hour session data.

- [x] Fix ExitMonitor: carry strategy_id, tag emitted OrderEvents (DEC-EXIT-003)
- [x] Fix ExitMonitor: clear pending_exits only when position is fully gone (DEC-EXIT-004)
- [x] Fix RangeExitMonitor: same strategy_id fix (DEC-RANGE-007)
- [x] Add `CommissionGateRule` — deny orders where range can't cover round-trip commission (DEC-COMMISSION-001)
- [x] Disable swing_trading — Session 18 sole loser (DEC-TUNE-005)
- [x] Remove BTC/USD from momentum strategy (DEC-TUNE-006)
- [x] Remove BTC/USD from breakout strategy (DEC-TUNE-007)
- [x] Normalize LLM allocation fractions to percentages in Conductor (DEC-CONDUCTOR-005)
- [x] Add `fix_orphaned_trades.py` script + tests (DEC-TEST-CLEANUP-001)
- [x] Add `tests/unit/test_commission_gate.py` (DEC-TEST-COMMISSION-001)
- **Verification**: All tests pass (committed in 0c7558d, eb40e8e)

**Session 18 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-EXIT-003 | ExitMonitor carries strategy_id and tags emitted OrderEvents | In multi-strategy mode, PortfolioTracker filters fills by strategy_id. If ExitMonitor emits a SELL OrderEvent without strategy_id, the paper adapter propagates a FillEvent with strategy_id=None, bypassing the per-strategy portfolio. Position never decreases, causing the exit monitor to re-fire on every subsequent tick (infinite exit loop). strategy_id=None (default) preserves backward compat | 2026-03-30 |
| DEC-EXIT-004 | _on_fill clears pending_exits only when position is actually gone | Original implementation cleared pending_exits on any SELL fill, even partial fills or fills for a different strategy's order on the same symbol. Correct invariant: pending flag stays set until portfolio confirms position amount < 0.0001. Prevents second exit order being emitted before portfolio processes the fill | 2026-03-30 |
| DEC-RANGE-007 | RangeExitMonitor carries strategy_id (same rationale as DEC-EXIT-003) | Mirrors the fix in ExitMonitor: without strategy_id on emitted OrderEvents, fills bypass the per-strategy PortfolioTracker routing, leaving positions open and causing infinite re-fire loop on every market tick. _on_fill uses the position-amount guard (DEC-EXIT-004) to prevent premature clearing | 2026-03-30 |
| DEC-COMMISSION-001 | Commission-aware minimum trade viability gate | Session 17 showed trades too small to overcome 0.32% round-trip commission. Expected profit = position_value * recent_range_pct. Commission cost = position_value * commission_pct * 2 (round-trip). If range_pct < round_trip_commission * min_ratio, the trade is denied. Default min_ratio=3.0 requires range to be 3× commission before entering | 2026-03-30 |
| DEC-TUNE-005 | Disable swing_trading — Session 18 sole loser | Session 18: -$51 PnL, zero realized trades, only 1 position held (short DOGE). Only losing strategy of 6. Disabled until tuning is revisited. Re-enable by uncommenting in main.py | 2026-03-30 |
| DEC-TUNE-006 | Remove BTC/USD from momentum strategy | Session 18: momentum bought BTC at $67,653 and sold at $67,342 (-$311 move). Short-timeframe momentum signals not catching BTC trends. BTC exposure remains via mean_reversion (+$877) and news_driven (+$650). Per-pair thresholds tabled for future | 2026-03-30 |
| DEC-TUNE-007 | Remove BTC/USD from breakout strategy | Same as DEC-TUNE-006 — BTC/USD better served by mean_reversion and news_driven. Breakout keeps ETH and SOL where shorter-timeframe signals perform better | 2026-03-30 |
| DEC-CONDUCTOR-005 | Normalize LLM allocation fractions to percentages | Haiku returns 0.25 instead of 25 for "25%". Rather than relying on prompt engineering, detect sum(allocations) <= 2 and multiply by 100. Single normalization point in _apply_allocations() since it is called by all allocation sources | 2026-03-30 |
| DEC-TEST-CLEANUP-001 | Tests for fix_orphaned_trades using in-memory SQLite | fix_orphaned_trades uses raw sqlite3. Tests use an in-memory SQLite DB with seeded trade data to verify mutation correctness without touching the production database. Follows DEC-TEST-016 / DEC-ANALYZE-002 pattern. All assertions read back from DB after function returns | 2026-03-30 |
| DEC-CLEANUP-001 | One-time orphaned-trade fix using pure stdlib sqlite3 | The cleanup must not import any cerebrum module because StateManager is async and requires an event loop. A self-contained stdlib script can be run outside the venv by any operator without spinning up the full system. signal_snapshot JSON is patched in Python (not SQL) so that NULL and malformed values are handled consistently | 2026-03-30 |
| DEC-TEST-COMMISSION-001 | Test CommissionGateRule with real EventBus and injected prices | CommissionGateRule self-subscribes to MARKET_DATA events. Testing with a real EventBus validates subscription wiring and per-symbol deque logic. No wall-clock dependency — tests inject prices directly via bus.publish() and verify evaluate() outcomes | 2026-03-30 |

### Strategy Consolidation: min_trade_value_usd (IN PROGRESS)
**Goal**: Add a minimum trade value floor to PositionSizingRule to prevent commission-dominated micro-trades. Part of the strategy consolidation effort addressing multi-strategy mode generating tiny $20 trades.

- [x] Add `min_trade_value_usd` parameter to `PositionSizingRule` (DEC-SIZING-001)
- [x] Add `tests/unit/test_position_sizing_min_value.py` — 4 test cases (DEC-TEST-SIZING-001)
- [ ] Wire `min_trade_value_usd` into `StrategyRegistry` per-strategy config (Task 4)
- [ ] Disable momentum, breakout, news_driven strategies (Task 2)
- [ ] Update capital allocation and cooldowns (Task 3)
- [ ] Integration test: 2-strategy consolidated pipeline (Task 5)

**Strategy Consolidation Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-SIZING-001 | Minimum trade value floor to prevent commission-killed micro-trades | Multi-strategy mode with $1,666 capital per strategy generates $20 trades where 0.32% round-trip commission eats 33% of wins. Floor at $100 keeps commission below 10%. Check uses strength-adjusted value (not raw target) since that is the actual amount risked. min_trade_value_usd=None default preserves backward compatibility | 2026-03-30 |
| DEC-TEST-SIZING-001 | Tests for PositionSizingRule min_trade_value_usd using simple MockPortfolio | PositionSizingRule.evaluate() is a pure synchronous function — no event bus needed. Tests use a minimal MockPortfolio class (2 methods) rather than MagicMock to keep intent clear. All 4 cases: above-min MODIFY, below-min DENY, None MODIFY (backward compat), strength-adjusted DENY | 2026-03-30 |
| DEC-TRACK-002 | Orphan trade cleanup at startup | Disabled strategies accumulate OPEN trades across sessions. Without cleanup, a future session that re-enabling a strategy would pick up stale OPEN trades and immediately close them on the first SELL fill, producing phantom P&L. Cleaning at startup (before the event loop begins) prevents this contamination | 2026-03-31 |
| DEC-TUNE-008 | Disable momentum, breakout, news_driven — signal cannibalization | Investigation of 219 multi-strategy trades (Mar 24-30) showed all 4 unfiltered strategies (momentum, mean_reversion, breakout, news_driven) consume identical RSI/MACD/BB/VWAP signals. 78 simultaneous entry pairs confirmed: same signal, same symbol, same second, all lost money together. Only mean_reversion and range_trading have differentiated signal sources and survive consolidation | 2026-03-31 |

---

### Phase 14: Hot-Swappable Profile Selector (IN PROGRESS)
**Goal**: Runtime risk profile switching without restarting the process. Three predefined profiles (conservative, moderate, aggressive) cover the spectrum from choppy markets to trending sessions. Dashboard UI in Phase 14B.

- [x] 14A: ProfileConfig schema expansion + ProfileManager + TOML profiles (DEC-PROFILE-001, DEC-PROFILE-002)
- [ ] 14B: Dashboard profile selector UI (dropdown + apply button)
- [ ] 14C: Wire ProfileManager into main.py startup

**Phase 14 Decisions:**

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-PROFILE-001 | Three predefined profiles in paper.toml: conservative / moderate / aggressive | 20 sessions of tuning data confirm the moderate profile as the proven baseline. Conservative tightens all thresholds for choppy/unknown-regime sessions. Aggressive widens them for high-volatility trending sessions. Three discrete profiles are easier to reason about and switch between than a continuous tuning surface | 2026-04-01 |
| DEC-PROFILE-002 | ProfileManager mutates private pipeline attributes for hot-swap | Two options: (A) add public setters to each component class, (B) ProfileManager mutates private attrs directly. Chose B: zero changes to component classes; single choke-point for all overrides; easy to extend. Trade-off: relies on private naming convention; documented in manager.py with full attribute map and rationale | 2026-04-01 |

---

## Decision Catalog
_Auto-indexed from `@decision` annotations in source on 2026-04-24._
_Excludes `DEC-TEST-*` and `DEC-TUNE-*` (code-local, non-architectural — not MASTER_PLAN-level)._
_122 unique decisions across 46 categories._

Each row: ID → short title → source file(s). Grep `@decision DEC-FOO-NNN` in the repo for full rationale and context.

### DEC-ADAPTER-* — Exchange Adapters (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-ADAPTER-001` | Abstract adapter interface for exchange independence | adapters/base.py |

### DEC-AGG-* — Signal Aggregation (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-AGG-001` | Signal aggregator with weighted combination and debounce | signals/aggregator.py |
| `DEC-AGG-002` | Consensus multiplier via sqrt(buy_weight_fraction) | signals/aggregator.py |

### DEC-ALLOC-* — Capital Allocation (3)

| ID | Title | Source |
|----|-------|--------|
| `DEC-ALLOC-001` | Darwinian capital allocation via rolling Sharpe ratio | conductor/allocator.py |
| `DEC-ALLOC-002` | Auto-reactivation with exponential backoff prevents permanent deadlock | conductor/allocator.py |
| `DEC-ALLOC-003` | All-paused edge case: reactivate the least-bad strategy | conductor/allocator.py |

### DEC-ALPACA-* — Alpaca (Stocks) (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-ALPACA-001` | Alpaca adapter for multi-asset proof-of-concept | adapters/alpaca.py |
| `DEC-ALPACA-002` | Conditional Alpaca adapter wiring via raw TOML config | main.py |

### DEC-ANALYZE-* — Session Analysis (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-ANALYZE-001` | analyze.py replicates stats logic inline (no cerebrum imports) | s/analyze.py |
| `DEC-ANALYZE-003` | weekly_report.py delegates to generate_report() from analyze.py | s/weekly_report.py |

### DEC-BACKTEST-* — Backtesting (4)

| ID | Title | Source |
|----|-------|--------|
| `DEC-BACKTEST-001` | Full multi-strategy backtest reuses entire live pipeline | s/run_backtest.py |
| `DEC-BACKTEST-002` | News/sentiment and Conductor skipped; RegimeDetector included | s/run_backtest.py |
| `DEC-BACKTEST-003` | Automatic parameter scaling for non-1m candle intervals | s/run_backtest.py |
| `DEC-BACKTEST-004` | (no title) | signals/aggregator.py, s/run_backtest.py |

### DEC-BUS-* — Event Bus (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-BUS-001` | Async event bus with type-based subscriptions | core/bus.py |

### DEC-CLEANUP-* — Cleanup / Orphans (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-CLEANUP-001` | One-time orphaned-trade fix using pure stdlib sqlite3 | s/fix_orphaned_trades.py |

### DEC-COMMISSION-* — Commission Gate (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-COMMISSION-001` | Commission-aware minimum trade viability gate | risk/rules.py |

### DEC-CONDUCTOR-* — LLM Conductor (12)

| ID | Title | Source |
|----|-------|--------|
| `DEC-CONDUCTOR-001` | Event-driven + polling hybrid LLM conductor | conductor/conductor.py |
| `DEC-CONDUCTOR-002` | Freeze allocations on API failure, never reset | conductor/conductor.py |
| `DEC-CONDUCTOR-003` | Math-only mode when no API key provided | conductor/conductor.py |
| `DEC-CONDUCTOR-004` | 50% single-strategy allocation cap to prevent peak-equity spikes | conductor/conductor.py |
| `DEC-CONDUCTOR-005` | Normalize LLM allocation fractions to percentages | conductor/conductor.py |
| `DEC-CONDUCTOR-006` | Live total_capital refresh + conservation check on every allocation cycle | conductor/conductor.py |
| `DEC-CONDUCTOR-007` | Per-cycle Sharpe refresh: call _refresh_allocator_performance at top of _apply_allocations | conductor/conductor.py |
| `DEC-CONDUCTOR-008` | Rolling closed-trades deque in PortfolioTracker for Darwinian Sharpe feed | risk/portfolio.py |
| `DEC-CONDUCTOR-009` | Equity curve passed as empty list; reserved for future Sortino metrics | conductor/conductor.py |
| `DEC-CONDUCTOR-010` | Minimum 3 closed trades required before calling update_performance | conductor/conductor.py |
| `DEC-CONDUCTOR-011` | Warmup semantics unchanged; _refresh_allocator_performance only populates the dict | conductor/conductor.py |
| `DEC-CONDUCTOR-012` | Atomic v3→v4 migration + _save_state version correctness closes Sharpe persistence | adapters/paper.py |

### DEC-CONFIG-* — Configuration (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-CONFIG-001` | Pydantic Settings with TOML + env var layering | core/config.py |

### DEC-COOL-* — Post-fill Cooldown (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-COOL-001` | (no title) | risk/rules.py |

### DEC-DASH-* — Dashboard (7)

| ID | Title | Source |
|----|-------|--------|
| `DEC-DASH-001` | (no title) | dashboard/web.py, risk/portfolio.py |
| `DEC-DASH-002` | htmx + FastAPI web dashboard for multi-strategy visualization | dashboard/web.py |
| `DEC-DASH-003` | Copilot mode queues pending allocations rather than blocking the Conductor | dashboard/web.py |
| `DEC-DASH-004` | Phase 12F state tracked from FillEvents in-memory — no DB queries | dashboard/web.py |
| `DEC-DASH-005` | ProfileManager wired into WebDashboard as Optional — None means unavailable | dashboard/web.py |
| `DEC-DASH-006` | Seed "Days Trading" from MIN(entry_time) in trades DB on init | dashboard/web.py |
| `DEC-DASH-007` | Periodic equity snapshot task — fill-independent chart population | dashboard/web.py |

### DEC-DENIAL-* — Denial Tracking (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-DENIAL-001` | (no title) | risk/manager.py |

### DEC-EVENTS-* — Event Types (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-EVENTS-001` | Immutable frozen dataclasses for all events | core/events.py |

### DEC-EXIT-* — Exit Monitor (5)

| ID | Title | Source |
|----|-------|--------|
| `DEC-EXIT-001` | ExitMonitor as separate component from RiskManager | risk/exit_monitor.py |
| `DEC-EXIT-002` | Adaptive take-profit based on recent price range | risk/exit_monitor.py |
| `DEC-EXIT-003` | ExitMonitor carries strategy_id and tags emitted OrderEvents | risk/exit_monitor.py |
| `DEC-EXIT-004` | _on_fill clears pending_exits only when position is actually gone | risk/exit_monitor.py |
| `DEC-EXIT-006` | min_hold_minutes prevents premature SL/TP exits on freshly-opened positions | risk/exit_monitor.py, risk/range_exit_monitor.py |

### DEC-EXPORT-* — Data Export (3)

| ID | Title | Source |
|----|-------|--------|
| `DEC-EXPORT-001` | LLM fine-tuning JSONL exporter using raw sqlite3 (no ORM) | s/export_finetune.py |
| `DEC-EXPORT-002` | Trades CSV exporter with flattened signal_snapshot columns | s/export_trades_csv.py |
| `DEC-EXPORT-003` | Weight history CSV exporter with ISO timestamps | s/export_weights.py |

### DEC-HOURS-* — Trading Hours (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-HOURS-001` | Local calendar market hours check — no external API dependency | risk/market_hours.py |

### DEC-INT-* — Intelligence (LLM/Sentiment) (5)

| ID | Title | Source |
|----|-------|--------|
| `DEC-INT-001` | News ingestion with graceful degradation | intelligence/news.py, intelligence/social.py |
| `DEC-INT-002` | Async polling with aiohttp | intelligence/llm.py, intelligence/news.py |
| `DEC-INT-003` | Optional HMM with rule-based fallback for regime detection | signals/regime.py, signals/sentiment.py |
| `DEC-INT-004` | Rate limiting and cost control for LLM | intelligence/llm.py |
| `DEC-INT-005` | Regime-based signal weight adjustment | signals/aggregator.py, signals/regime.py |

### DEC-KRAKEN-* — Kraken (Crypto) (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-KRAKEN-001` | ccxt.pro async WebSocket for real-time data | adapters/kraken.py |

### DEC-LEARN-* — Learning / Weights (3)

| ID | Title | Source |
|----|-------|--------|
| `DEC-LEARN-001` | Trade outcome tracking with signal snapshots | learning/tracker.py |
| `DEC-LEARN-002` | Conservative EMA weight adaptation | learning/adapter.py |
| `DEC-LEARN-003` | Per-regime signal scoring | learning/scorer.py |

### DEC-LIVE-* — Live Trading (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-LIVE-001` | (no title) | adapters/kraken.py |

### DEC-MAIN-* — Main Entrypoint (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-MAIN-001` | Graceful shutdown with signal handlers | main.py |
| `DEC-MAIN-002` | Multi-strategy mode as default with single-strategy fallback | main.py |

### DEC-MONITOR-* — Monitoring / Stats (5)

| ID | Title | Source |
|----|-------|--------|
| `DEC-MONITOR-001` | Pure function stats calculator | monitoring/stats.py |
| `DEC-MONITOR-002` | Event-driven dashboard with periodic updates | monitoring/dashboard.py |
| `DEC-MONITOR-003` | Session reporter with file and console output | monitoring/reporter.py |
| `DEC-MONITOR-004` | CLI stats viewer with filtering | s/show_stats.py |
| `DEC-MONITOR-005` | Backtest runner with OHLCV replay | s/run_backtest.py |

### DEC-NEWS-* — News Signals (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-NEWS-001` | News-heavy signal weighting for event-driven trading | strategies/news_driven.py |

### DEC-PAPER-* — Paper Trading Adapter (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-PAPER-001` | File-based state persistence for paper trading | adapters/paper.py |

### DEC-PERSIST-* — State Persistence (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-PERSIST-001` | Per-strategy PortfolioTracker snapshots in paper_state.json | adapters/paper.py, risk/portfolio.py |

### DEC-PLUGIN-* — Plugin System (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-PLUGIN-001` | Abstract plugin interface with lifecycle hooks | plugins/base.py |
| `DEC-PLUGIN-002` | Error isolation in plugin registry | plugins/registry.py |

### DEC-PROFILE-* — Preset Profiles (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-PROFILE-002` | ProfileManager mutates private pipeline attributes for hot-swap | profiles/manager.py |

### DEC-RANGE-* — Range Trading Strategy (7)

| ID | Title | Source |
|----|-------|--------|
| `DEC-RANGE-001` | RangeDetector as a queryable state object, not an event emitter | strategies/range_detector.py |
| `DEC-RANGE-002` | Bounce deduplication via proximity zone tracking | strategies/range_detector.py |
| `DEC-RANGE-003` | Regex-based level extraction from signal.reason string | strategies/range_detector.py |
| `DEC-RANGE-004` | Structural exits over percentage-based for range trading | risk/range_exit_monitor.py |
| `DEC-RANGE-005` | Fallback to percentage-based exits when no confirmed range exists | risk/range_exit_monitor.py |
| `DEC-RANGE-006` | Dedicated range strategy with S/R-only signal filtering | strategies/range_trading.py |
| `DEC-RANGE-007` | RangeExitMonitor carries strategy_id (same rationale as DEC-EXIT-003) | risk/range_exit_monitor.py |

### DEC-RECONCILE-* — Position Reconciliation (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-RECONCILE-001` | Startup position reconciliation between portfolio trackers and paper adapter | main.py |

### DEC-REGIME-* — Regime Detection (6)

| ID | Title | Source |
|----|-------|--------|
| `DEC-REGIME-001` | Cumulative return + MA slope for slow-trend detection | signals/regime.py |
| `DEC-REGIME-002` | (no title) | signals/aggregator.py |
| `DEC-REGIME-003` | Dual-window regime detection for ultra-slow drift | signals/regime.py |
| `DEC-REGIME-004` | Trade halt in BEAR regime | risk/rules.py |
| `DEC-REGIME-005` | Regime hysteresis — require N consecutive readings before transition | risk/rules.py, signals/regime.py |
| `DEC-REGIME-006` | Block trading during UNKNOWN regime (startup/reconnect window) | risk/rules.py |

### DEC-RISK-* — Risk Management (4)

| ID | Title | Source |
|----|-------|--------|
| `DEC-RISK-001` | Composable risk rules architecture | risk/rules.py |
| `DEC-RISK-002` | Portfolio state tracking for exposure calculations | risk/portfolio.py |
| `DEC-RISK-004` | strategy_id filtering in PortfolioTracker prevents double-counting in multi-strategy mode | risk/portfolio.py |
| `DEC-RISK-005` | Cap open positions per strategy/symbol | risk/portfolio.py, risk/rules.py |

### DEC-SENSITIVITY-* — Parameter Sensitivity (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SENSITIVITY-001` | sensitivity.py imports TradeRow/calculate_commission from analyze.py | s/sensitivity.py |

### DEC-SENT-* — Sentiment Dampening (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SENT-001` | (no title) | signals/aggregator.py |

### DEC-SHUTDOWN-* — Graceful Shutdown (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SHUTDOWN-001` | Graceful position liquidation on shutdown | main.py |

### DEC-SIGNAL-* — Signal Framework (6)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SIGNAL-001` | Abstract signal generator with automatic data accumulation | signals/base.py |
| `DEC-SIGNAL-002` | Source metadata injected by _create_signal | signals/base.py, signals/candles.py |
| `DEC-SIGNAL-003` | Timeframe metadata injected by _create_signal | signals/base.py, signals/technical.py |
| `DEC-SIGNAL-004` | pandas-ta for technical indicator calculations | signals/technical.py |
| `DEC-SIGNAL-005` | VWAP neutral zone (0.5%) to filter near-VWAP noise | signals/technical.py |
| `DEC-SIGNAL-006` | Pivot-based S/R detection with proximity signals | signals/support_resistance.py |

### DEC-SIZING-* — Position Sizing (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SIZING-001` | Minimum trade value floor to prevent commission-killed micro-trades | risk/rules.py |
| `DEC-SIZING-002` | Floor signal multiplier at 0.6 to prevent position starvation | risk/rules.py |

### DEC-STATE-* — State Machine (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-STATE-001` | SQLite with aiosqlite for async state persistence | core/state.py |

### DEC-STOCKS-* — Stocks Strategy (3)

| ID | Title | Source |
|----|-------|--------|
| `DEC-STOCKS-003` | RTH-only enforcement with entry cutoff before close | risk/end_of_day_flatten.py, risk/market_hours_gate.py, ut… |
| `DEC-STOCKS-004` | orb_stocks strategy — breakout entry, RTH-only, auto-flatten | signals/opening_range.py, strategies/orb_stocks.py, s/rec… |
| `DEC-STOCKS-006` | Atomic v2→v3 state migration with .v2.bak backup | adapters/paper.py |

### DEC-STRAT-* — Strategy Framework (8)

| ID | Title | Source |
|----|-------|--------|
| `DEC-STRAT-001` | Mean reversion strategy preset with BB/RSI emphasis | strategies/base.py, strategies/mean_reversion.py |
| `DEC-STRAT-002` | Breakout strategy preset with MACD/VWAP emphasis | strategies/base.py, strategies/breakout.py |
| `DEC-STRAT-003` | StrategyRegistry owns pipeline lifecycle with error isolation | strategies/registry.py |
| `DEC-STRAT-004` | GlobalPortfolio as read-only aggregation view | strategies/global_portfolio.py |
| `DEC-STRAT-006` | Backward-compatible single-strategy mode in main.py | strategies/momentum.py |
| `DEC-STRAT-007` | GlobalTradeRateLimitRule for cross-strategy commission control | risk/global_trade_rate.py |
| `DEC-STRAT-008` | MEAN_REVERSION_CONFIG as StrategyConfig for StrategyRegistry wiring | strategies/mean_reversion.py |
| `DEC-STRAT-009` | BREAKOUT_CONFIG as StrategyConfig for StrategyRegistry wiring | strategies/breakout.py |

### DEC-SWING-* — Swing Trading Strategy (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-SWING-001` | 1-hour timeframe swing strategy to reduce commission drag | strategies/swing_trading.py |

### DEC-TRACK-* — Trade Tracking (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-TRACK-002` | Orphan trade cleanup at startup | learning/tracker.py |

### DEC-TYPES-* — Core Types (1)

| ID | Title | Source |
|----|-------|--------|
| `DEC-TYPES-001` | Use Decimal for all financial calculations | core/types.py |

### DEC-VOL-* — Volatility Gate (4)

| ID | Title | Source |
|----|-------|--------|
| `DEC-VOL-001` | Percentage price range as volatility metric | risk/rules.py |
| `DEC-VOL-002` | Per-symbol rolling price window via MARKET_DATA event bus subscription | risk/rules.py |
| `DEC-VOL-003` | Default threshold 0.5%, lookback 300 ticks, both configurable via TOML | risk/rules.py |
| `DEC-VOL-004` | Macro-window volatility gate for session-level flatness detection | risk/rules.py |

### DEC-XSTOCKS-* — Kraken xStocks (2)

| ID | Title | Source |
|----|-------|--------|
| `DEC-XSTOCKS-001` | Conditional KrakenXStocks adapter wiring via raw TOML config | adapters/kraken_xstocks.py, main.py |
| `DEC-XSTOCKS-002` | xstocks_reversion — 24/7 mean-reversion on Kraken tokenized equities | strategies/xstocks_reversion.py |


---

## Resources

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `config/default.toml` | Default runtime configuration |
| `config/paper.toml` | Paper trading overrides (includes commented `[alpaca]` section) |
| `config/live.toml` | Live trading configuration (conservative) |
| `.env.example` | API key template |
