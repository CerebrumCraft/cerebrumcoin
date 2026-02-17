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

**Status**: Phase 2 — Signal Engine (completed) | Phase 3 — Intelligence Layer (planned)

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

### Phase 3: Intelligence Layer
**Goal**: News, sentiment, and regime detection augmenting signals.

- [ ] Implement `intelligence/news.py` — CryptoPanic + NewsAPI ingestion
- [ ] Implement `intelligence/llm.py` — Claude-powered news interpretation
- [ ] Implement `signals/sentiment.py` — FinBERT sentiment scoring
- [ ] Implement `signals/regime.py` — HMM-based regime detection
- [ ] Wire intelligence into aggregator with regime-aware weighting
- **Verification**: Agent adjusts behavior based on news and regime shifts

### Phase 4: Closed-Loop Learning
**Goal**: Agent learns from its own trading outcomes.

- [ ] Implement `learning/tracker.py` — trade outcome tracking
- [ ] Implement `learning/scorer.py` — signal effectiveness scoring
- [ ] Implement `learning/adapter.py` — adaptive weight adjustment
- [ ] Implement `core/state.py` — persist learning state across restarts
- **Verification**: Signal weights shift toward better-performing signals over time

### Phase 5: Paper Trading Validation
**Goal**: Extended paper trading with monitoring and backtesting.

- [ ] Run agent on paper mode for 2+ weeks
- [ ] Monitoring dashboard (structlog → CLI stats)
- [ ] Backtests against historical data via vectorbt
- [ ] Tune risk parameters, signal weights, regime thresholds
- **Verification**: Positive expectancy on paper + backtests

### Phase 6: Live Trading & Plugin System
**Goal**: Graduate to real trading, enable future system integration.

- [ ] Switch paper adapter to real Kraken execution
- [ ] Implement `plugins/base.py` — plugin interface
- [ ] Implement `plugins/registry.py` — plugin discovery and lifecycle
- [ ] Stock adapter (Alpaca) as multi-asset proof
- **Verification**: Live trades on Kraken, plugin system accepting connections

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
| DEC-TEST-001 | Test real implementations, not mocks | Validates actual event behavior, not simulated behavior | 2026-02-17 |
| DEC-TEST-002 | Async test fixtures for event bus validation | Event bus is async; tests must be async to verify queue behavior | 2026-02-17 |
| DEC-TEST-003 | Test config validation and TOML loading | Validates Pydantic settings: type validation, percentage bounds, composition | 2026-02-17 |
| DEC-TEST-004 | Test paper trading with real event bus integration | Verifies order execution, balance tracking, commission handling | 2026-02-17 |
| DEC-TEST-005 | End-to-end pipeline test with mock Kraken data | Integration test verifies complete flow: MarketDataEvent to FillEvent | 2026-02-17 |

## Resources

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `config/default.toml` | Default runtime configuration |
| `config/paper.toml` | Paper trading overrides |
| `.env.example` | API key template |
