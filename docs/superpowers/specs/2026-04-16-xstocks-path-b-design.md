# xStocks Path B — Kraken Tokenized Equities (24/7 Mean-Reversion)

## Context

CerebrumCoin already supports stocks via Alpaca (Path A: ORB strategy, RTH-only,
implemented in commits `bd1847d`–`39523af` on branch `worktree-stocks-orb`).
Path B adds a second stocks adapter using **Kraken's xStocks** — tokenized 1:1
representations of US equities that trade 24/7 like crypto.

The user already has a Kraken account with API keys. Using Kraken for stocks
simplifies the eventual live-trading migration (one broker, one credential set,
unified fee structure). Alpaca remains for RTH market-data and the ORB strategy;
xStocks runs mean-reversion on tokenized equities around the clock.

**Key API discovery (2026-04-16):**
- xStock assets: 130 available (`aclass=tokenized_asset`), 254 trading pairs
- Pair format: `AAPLxUSD` (REST) / `AAPLx/USD` (WebSocket)
- Endpoint: same Kraken Spot API, filtered via `aclass_base=tokenized_asset`
- Assets queried via `https://api.kraken.com/0/public/Assets?aclass=tokenized_asset`
- Pairs queried via `https://api.kraken.com/0/public/AssetPairs?aclass_base=tokenized_asset`
- All target symbols confirmed: AAPLx, MSFTx, NVDAx (all `status=online`)
- Min order: 0.00000001 (fractional), tick size: $0.01
- `token_multiplier` field exists (~1.00x for most) — indicates minor dividend/rebalancing drift vs underlying

**Brainstorming decisions (2026-04-16):**

| Dimension | Choice |
|---|---|
| SDK | `python-kraken-sdk` (explicit xStocks support, Spot + WS) |
| Strategy | New `xstocks_reversion` ($5k, own tuning, same signal mix as crypto mean_reversion) |
| Credentials | Reuse existing `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` |
| Trading hours | 24/7 (no MarketHoursGate, no EndOfDayFlatten) |
| Symbols | AAPLx/USD, MSFTx/USD, NVDAx/USD |

## Non-Goals

- No new signal generators (xStocks use existing RSI/MACD/BB/VWAP/SR)
- No new risk rules (existing regime_trade_halt, volatility_gate, commission_gate, cooldown apply)
- No new exit monitors (standard ExitMonitor covers xStocks like crypto)
- No live Kraken order execution in v1 (paper_trading adapter simulates fills)
- No cross-asset correlation signals (BTC↔AAPL) — deferred
- No xStocks perpetuals/derivatives (spot only)

## Architecture

### New file (1)

| File | Purpose | ~LOC |
|---|---|---|
| `cerebrum/adapters/kraken_xstocks.py` | `KrakenXStocksAdapter` using `python-kraken-sdk`. Implements `ExchangeAdapter` interface. Streams `AAPLx/USD`, `MSFTx/USD`, `NVDAx/USD` tickers via Kraken SpotWSClient. Publishes `MarketDataEvent` to shared bus. | ~180 |

### Modified files (4)

| File | Change | ~Lines |
|---|---|---|
| `cerebrum/main.py` | `_maybe_build_kraken_xstocks_adapter()` helper (follows Task 6 Alpaca pattern). Detect `[kraken_xstocks]` config, instantiate + connect + subscribe. | ~30 |
| `cerebrum/strategies/xstocks_reversion.py` | New file: `XSTOCKS_REVERSION_CONFIG = StrategyConfig(...)` following `orb_stocks.py` pattern. | ~40 |
| `config/paper.toml` | `[kraken_xstocks]` + `[strategy.xstocks_reversion]` + `[strategy.xstocks_reversion.weights]` sections. | ~25 |
| `pyproject.toml` | `xstocks = ["python-kraken-sdk>=2.0.0"]` in optional-dependencies. | ~2 |

### Unchanged (reused from existing infrastructure)

- EventBus, SignalAggregator (DEC-STOCKS-005 symbols filter), all 5 signal generators
- RiskManager with regime_trade_halt, volatility_gate, commission_gate, cooldown, sideways_suppression
- ExitMonitor (standard: stop-loss, take-profit, max-age)
- PortfolioTracker, paper_trading adapter, dashboard (`/api/strategies` auto-discovers)
- State file v3 (additive snapshot — no version bump)
- No MarketHoursGate (24/7), no EndOfDayFlatten (24/7), no OpeningRangeSignal

## Configuration

### `config/paper.toml` additions

```toml
[kraken_xstocks]
enabled = false
symbols = ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]

[strategy.xstocks_reversion]
enabled = false
initial_balance = 5000.0
symbols = ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]
signal_source_filter = null
aggregation_threshold = 0.4
position_size_percent = 20.0
stop_loss_percent = 1.0
take_profit_percent = 1.5
max_position_age_minutes = 120
min_hold_minutes = 15
post_fill_cooldown_seconds = 1800
min_signal_strength = 0.65

[strategy.xstocks_reversion.weights]
technical = 1.2
sentiment = 0.0
news = 0.0
regime = 0.5
```

### Credentials

Reuses existing `.env` variables:
```
EXCHANGE_API_KEY=...       # already set for Kraken crypto
EXCHANGE_API_SECRET=...    # already set for Kraken crypto
```

No new env vars. If xStocks requires separate permissions, the adapter catches
the auth error and logs `kraken_xstocks_auth_failed`.

### Portfolio

| Strategy | Balance | Asset | Market |
|---|---|---|---|
| mean_reversion | $5,000 | crypto | 24/7 |
| range_trading | $5,000 | crypto | 24/7 |
| orb_stocks | $5,000 | equities (Alpaca) | RTH |
| xstocks_reversion | $5,000 | tokenized equities (Kraken) | 24/7 |
| **Total** | **$20,000** | | |

### State migration

No version bump. Existing `migrate_state_v2_to_v3` pattern extended: on startup,
if `xstocks_reversion` is absent from `strategy_snapshots`, add it with $5k cash
and empty positions. Idempotent.

## Data Flow

Minimal — xStocks follow the exact same path as crypto:

1. `KrakenXStocksAdapter` streams `AAPLx/USD` ticks via SpotWSClient
2. Publishes `MarketDataEvent(symbol="AAPLx/USD", price=..., timestamp=...)` to bus
3. All signal generators (RSI, MACD, BB, VWAP, SR) receive the tick and emit signals
4. `signal_aggregator_xstocks_reversion` filters by `symbols=["AAPLx/USD","MSFTx/USD","NVDAx/USD"]` (DEC-STOCKS-005)
5. Aggregator threshold met → RiskManager evaluates → OrderEvent → PaperTradingAdapter fills
6. PortfolioTracker updates `xstocks_reversion` snapshot

No special gating, no special exit logic. The existing mean-reversion + risk
layering runs identically on xStock symbols as it does on BTC/USD.

## Error Handling

| Failure | Detection | Containment | Log event |
|---|---|---|---|
| `python-kraken-sdk` not installed | `ImportError` at adapter ctor | Skip, continue crypto+Alpaca-only | `kraken_xstocks_unavailable` |
| Credentials rejected for xStocks | WS auth returns error | Disable for this run, continue | `kraken_xstocks_auth_failed` |
| xStocks geo-blocked | API permission error | Disable, continue | `kraken_xstocks_geo_blocked` |
| WS disconnect | Ping timeout / close frame | Exponential backoff reconnect | `kraken_xstocks_disconnected` / `_reconnected` |
| `token_multiplier` drift | Not detected in v1 | Paper PnL tracks xStock price directly (not NYSE). Acceptable — drift is <0.3% per the API data | Document in spec |

## Testing

### Unit tests (~12 new)

| File | Coverage |
|---|---|
| `tests/unit/test_kraken_xstocks_adapter.py` | Constructor, config handling, credential loading from env, connect/disconnect mock, MarketDataEvent shape. ~6 tests. |
| `tests/unit/test_xstocks_reversion_config.py` | Config imports cleanly, symbols match, weights match, signal_source_filter is null. ~3 tests. |
| `tests/unit/test_main_xstocks_wiring.py` | Wiring helper: disabled → None, enabled + creds → builds, missing module → graceful skip. ~3 tests. |

### Regression

All existing tests (~768 + new) must pass with `[kraken_xstocks].enabled = false`.
No regressions on crypto or Alpaca paths.

### Live smoke (manual, after enabling)

Start session with `[kraken_xstocks].enabled = true`. Verify:
- `kraken_xstocks_connected` in log
- `AAPLx/USD` market data flowing (if xStocks are streaming)
- Dashboard shows 4 strategies

## Decision IDs

- **DEC-XSTOCKS-001** — Kraken xStocks via `python-kraken-sdk` for 24/7 tokenized
  equity trading. Reuses existing Kraken credentials.
- **DEC-XSTOCKS-002** — Dedicated `xstocks_reversion` strategy with $5k allocation,
  consuming all signal types (no source filter), scoped to `AAPLx/MSFTx/NVDAx`
  via symbols filter (DEC-STOCKS-005).
- **DEC-XSTOCKS-003** — No new risk rules or exit monitors. Standard crypto-style
  mean-reversion lifecycle (24/7, same ExitMonitor + RiskManager stack). Rationale:
  xStocks ARE crypto from a trading-mechanics perspective.

## Rollout

5 tasks, each ships independently:

1. **KrakenXStocksAdapter** + 6 unit tests
2. **XSTOCKS_REVERSION_CONFIG** + 3 unit tests
3. **main.py wiring** + 3 unit tests
4. **Config + pyproject.toml + state migration extension**
5. **Regression sweep + live smoke test**

## Out of Scope / Future

- Live Kraken xStocks execution (paper-sim only in v1)
- Cross-asset correlation signals (BTC↔AAPLx)
- xStocks perpetuals/derivatives
- Extended symbol list beyond 3 initial names
- `token_multiplier` tracking / NAV reconciliation vs NYSE price
- Replacing Alpaca entirely with Kraken xStocks (keep both paths)
