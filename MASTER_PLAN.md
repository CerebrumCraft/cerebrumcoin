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

**Status**: Phase 1 — Foundation (in progress)

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

### Phase 1: Foundation ← CURRENT
**Goal**: Project scaffolding, event bus, Kraken data flowing through the pipeline.

- [x] Initialize git repo, pyproject.toml with dependencies
- [ ] Implement `core/bus.py` — async event bus with typed events
- [ ] Implement `core/events.py` — MarketData, Signal, Order, Fill event types
- [ ] Implement `core/config.py` — Pydantic settings from TOML + env vars
- [ ] Implement `core/types.py` — shared type definitions
- [ ] Implement `adapters/base.py` — abstract exchange adapter
- [ ] Implement `adapters/kraken.py` — connect to Kraken, stream price data
- [ ] Implement `adapters/paper.py` — paper trading execution engine
- **Verification**: Live Kraken price data flows through the event bus, paper orders execute

### Phase 2: Signal Engine
**Goal**: Technical analysis signals producing actionable trade signals.

- [ ] Implement `signals/technical.py` — RSI, MACD, Bollinger Bands, VWAP
- [ ] Implement `signals/aggregator.py` — weighted signal combination
- [ ] Implement `risk/manager.py` — position sizing, stop-loss, exposure limits
- [ ] Basic strategy loop: data → signals → risk check → paper execute
- **Verification**: Bot paper-trades based on technical signals with risk management

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

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| DEC-001 | ccxt over raw Kraken API | Unified interface for multi-exchange future; WebSocket support adequate | 2026-02-17 |
| DEC-002 | pandas-ta over ta-lib | No C compilation headaches; pure Python; adequate indicator coverage | 2026-02-17 |
| DEC-003 | SQLite over Postgres | Zero-ops for solo project; migrate when scale demands it | 2026-02-17 |
| DEC-004 | Event bus over direct coupling | Enables hot-swap, plugin system, and agentic integration | 2026-02-17 |

## Resources

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project config |
| `config/default.toml` | Default runtime configuration |
| `config/paper.toml` | Paper trading overrides |
| `.env.example` | API key template |
