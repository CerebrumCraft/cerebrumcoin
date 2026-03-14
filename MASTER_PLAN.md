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

## Resources

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `config/default.toml` | Default runtime configuration |
| `config/paper.toml` | Paper trading overrides |
| `config/live.toml` | Live trading configuration (conservative) |
| `.env.example` | API key template |
