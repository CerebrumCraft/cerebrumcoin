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

## Resources

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `config/default.toml` | Default runtime configuration |
| `config/paper.toml` | Paper trading overrides |
| `config/live.toml` | Live trading configuration (conservative) |
| `.env.example` | API key template |
