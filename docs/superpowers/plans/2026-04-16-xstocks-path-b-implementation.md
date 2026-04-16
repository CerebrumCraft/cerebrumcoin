# xStocks Path B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `KrakenXStocksAdapter` using `python-kraken-sdk` to stream tokenized US equity tickers (AAPLx/USD, MSFTx/USD, NVDAx/USD) 24/7, feeding a dedicated `xstocks_reversion` mean-reversion strategy through the existing event bus.

**Architecture:** New adapter publishes `MarketDataEvent` to the shared bus — identical to how `KrakenAdapter` handles crypto. A new `xstocks_reversion` strategy config (same signal weights as crypto `mean_reversion`) consumes only xStock symbols via the DEC-STOCKS-005 aggregator filter. No new signal generators, risk rules, or exit monitors needed.

**Tech Stack:** `python-kraken-sdk>=3.2.0` (`SpotWSClient` for WebSocket, `SpotAsyncClient` for REST), existing `EventBus`/`SignalGenerator`/`RiskRule`/`ExitMonitor` infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-16-xstocks-path-b-design.md`

**Worktree:** `.worktrees/worktree-stocks-orb` branch `worktree-stocks-orb` (continues from Path A work).

---

## One-Time Setup

- [ ] **Install python-kraken-sdk in the worktree venv (if not already done)**

```bash
cd /home/j/CerebrumCraft/CerebrumCoin/.worktrees/worktree-stocks-orb
.venv/bin/pip install python-kraken-sdk
```

Verify: `.venv/bin/python -c "from kraken.spot import SpotWSClient; print('OK')"`

---

## File Structure

### New files

| Path | Responsibility | Task |
|---|---|---|
| `cerebrum/adapters/kraken_xstocks.py` | `KrakenXStocksAdapter` — WS ticker stream for xStock pairs via `SpotWSClient`. Same `ExchangeAdapter` interface. | 1 |
| `cerebrum/strategies/xstocks_reversion.py` | `XSTOCKS_REVERSION_CONFIG = StrategyConfig(...)` — mean-reversion on xStocks. | 2 |
| `tests/unit/test_kraken_xstocks_adapter.py` | 6 unit tests for the adapter. | 1 |
| `tests/unit/test_xstocks_reversion_config.py` | 3 unit tests for the strategy config. | 2 |
| `tests/unit/test_main_xstocks_wiring.py` | 3 unit tests for the wiring helper. | 3 |

### Modified files

| Path | Change | Task |
|---|---|---|
| `cerebrum/main.py` | `_maybe_build_kraken_xstocks_adapter()` + startup call + shutdown disconnect. | 3 |
| `config/paper.toml` | `[kraken_xstocks]` + `[strategy.xstocks_reversion]` sections. | 4 |
| `pyproject.toml` | `xstocks = ["python-kraken-sdk>=3.2.0"]` optional dep group. | 4 |
| `cerebrum/adapters/paper.py` | Extend `migrate_state_v2_to_v3` to also add `xstocks_reversion` snapshot if missing. | 4 |

---

## SDK Reference (from live inspection)

```python
from kraken.spot import SpotWSClient, SpotAsyncClient, Market

# WebSocket: override on_message to receive ticker data
class MyWS(SpotWSClient):
    def on_message(self, message: dict | list) -> None:
        ...  # handle {"channel": "ticker", "type": "update", "data": [...]}

# Constructor:
ws = MyWS(key="...", secret="...", callback=None, no_public=False)

# Subscribe:
ws.subscribe(params={"channel": "ticker", "symbol": ["AAPLx/USD", "MSFTx/USD"]})

# REST (for initial price snapshot):
market = Market()
ticker = market.get_ticker(pair="AAPLxUSD")
```

---

# Task 1: KrakenXStocksAdapter + 6 unit tests

**Files:**
- Create: `cerebrum/adapters/kraken_xstocks.py`
- Create: `tests/unit/test_kraken_xstocks_adapter.py`

- [ ] **Step 1: Read the existing adapters to match patterns**

```bash
cd /home/j/CerebrumCraft/CerebrumCoin/.worktrees/worktree-stocks-orb
head -80 cerebrum/adapters/alpaca.py   # Task 6 built this — same interface
head -50 cerebrum/adapters/base.py     # ExchangeAdapter base class
head -40 cerebrum/adapters/kraken.py   # existing crypto adapter for reference
```

Note: the constructor signature, `connect()` / `disconnect()` / `subscribe_market_data()` shape, how `MarketDataEvent` is built and published.

- [ ] **Step 2: Write the 6 failing tests**

Create `tests/unit/test_kraken_xstocks_adapter.py`:

```python
"""Unit tests for KrakenXStocksAdapter (DEC-XSTOCKS-001)."""
import os
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from cerebrum.adapters.kraken_xstocks import KrakenXStocksAdapter


@pytest.fixture
def config():
    return {
        "symbols": ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"],
    }


@pytest.fixture
def bus():
    b = MagicMock()
    b.publish = AsyncMock()
    return b


def test_constructor_stores_symbols(config, bus):
    adapter = KrakenXStocksAdapter(bus=bus, config=config)
    assert adapter._symbols == ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]


def test_constructor_reads_credentials_from_env(config, bus, monkeypatch):
    monkeypatch.setenv("EXCHANGE_API_KEY", "test_key")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "test_secret")
    adapter = KrakenXStocksAdapter(bus=bus, config=config)
    assert adapter._api_key == "test_key"
    assert adapter._api_secret == "test_secret"


def test_constructor_raises_without_credentials(config, bus, monkeypatch):
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="kraken_xstocks_credentials_missing"):
        KrakenXStocksAdapter(bus=bus, config=config)


def test_parse_ticker_message_publishes_market_data(config, bus, monkeypatch):
    """Simulate a SpotWSClient ticker update and verify MarketDataEvent."""
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    adapter = KrakenXStocksAdapter(bus=bus, config=config)
    # Simulate the shape of a Kraken WS v2 ticker message
    # Adjust after inspecting real WS messages in Step 1
    adapter._handle_ticker_update(
        symbol="AAPLx/USD",
        bid=Decimal("182.50"),
        ask=Decimal("182.55"),
        last=Decimal("182.52"),
        volume=Decimal("12345.67"),
    )
    assert bus.publish.call_count >= 1


def test_symbol_normalization(config, bus, monkeypatch):
    """Kraken WS may use 'AAPLx/USD'; verify adapter preserves this format."""
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    adapter = KrakenXStocksAdapter(bus=bus, config=config)
    assert adapter._normalize_symbol("AAPLx/USD") == "AAPLx/USD"


def test_disabled_when_sdk_missing(config, bus, monkeypatch):
    """If python-kraken-sdk is not installed, adapter should not be importable
    and the wiring helper (Task 3) handles the ImportError gracefully."""
    # This test just confirms the module structure is correct
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    adapter = KrakenXStocksAdapter(bus=bus, config=config)
    assert adapter is not None
```

- [ ] **Step 3: Run tests → fail (module doesn't exist)**

```bash
.venv/bin/pytest tests/unit/test_kraken_xstocks_adapter.py -v
```

- [ ] **Step 4: Implement `cerebrum/adapters/kraken_xstocks.py`**

Read the adapters from Step 1, then write the adapter following the same interface. Key points:
- Constructor: `(bus, config)` matching Alpaca adapter pattern
- Reads `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` from `os.environ`
- Creates a `SpotWSClient` subclass internally that overrides `on_message`
- `connect()`: starts the WS connection
- `disconnect()`: closes the WS
- `subscribe_market_data(symbols)`: calls `ws.subscribe({"channel": "ticker", "symbol": symbols})`
- On each ticker message: parses bid/ask/last/volume, builds `MarketDataEvent`, publishes to bus
- `@decision DEC-XSTOCKS-001` annotation required

The `_handle_ticker_update` method should be the internal handler that the tests can call directly without a live WS connection.

IMPORTANT: inspect how `SpotWSClient.on_message` delivers ticker data — the message format may be:
```python
{"channel": "ticker", "type": "update", "data": [{"symbol": "AAPLx/USD", "bid": ..., "ask": ..., "last": ...}]}
```
or a different shape. Read the SDK source at `.venv/lib/python3.12/site-packages/kraken/spot/websocket/` to confirm the exact message structure before writing the parser.

- [ ] **Step 5: Run tests → all 6 pass**

```bash
.venv/bin/pytest tests/unit/test_kraken_xstocks_adapter.py -v
```

- [ ] **Step 6: Commit**

```bash
git add cerebrum/adapters/kraken_xstocks.py tests/unit/test_kraken_xstocks_adapter.py
git commit --no-gpg-sign -m "feat(adapters): KrakenXStocksAdapter for 24/7 tokenized equities (DEC-XSTOCKS-001)"
```

---

# Task 2: XSTOCKS_REVERSION_CONFIG + 3 unit tests

**Files:**
- Create: `cerebrum/strategies/xstocks_reversion.py`
- Create: `tests/unit/test_xstocks_reversion_config.py`

- [ ] **Step 1: Read existing strategy config for pattern**

```bash
cat cerebrum/strategies/orb_stocks.py   # created in Task 23 — latest pattern
cat cerebrum/strategies/mean_reversion.py  # the crypto version we're mirroring
```

Note the `StrategyConfig` import path, field names, and construction pattern.

- [ ] **Step 2: Write the 3 failing tests**

Create `tests/unit/test_xstocks_reversion_config.py`:

```python
"""Unit tests for xstocks_reversion strategy config (DEC-XSTOCKS-002)."""
from decimal import Decimal

import pytest

from cerebrum.strategies.xstocks_reversion import XSTOCKS_REVERSION_CONFIG


def test_config_name():
    assert XSTOCKS_REVERSION_CONFIG.name == "xstocks_reversion"


def test_config_symbols():
    assert XSTOCKS_REVERSION_CONFIG.symbols == ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]


def test_config_no_signal_source_filter():
    """xstocks_reversion consumes ALL signal types (RSI/MACD/BB/VWAP/SR)."""
    assert XSTOCKS_REVERSION_CONFIG.signal_source_filter is None
```

- [ ] **Step 3: Run → fail**

- [ ] **Step 4: Implement `cerebrum/strategies/xstocks_reversion.py`**

```python
"""xStocks mean-reversion strategy config.

@decision DEC-XSTOCKS-002
@title xstocks_reversion — 24/7 mean-reversion on Kraken tokenized equities
@status accepted
@rationale Dedicated strategy consuming all signal types (RSI/MACD/BB/VWAP/SR)
scoped to AAPLx/MSFTx/NVDAx via symbols filter (DEC-STOCKS-005). Same mechanics
as crypto mean_reversion but independently tunable. $5,000 allocation. No
MarketHoursGate (24/7), no EndOfDayFlatten (24/7). Uses existing Kraken
credentials (DEC-XSTOCKS-003).
"""

from decimal import Decimal

from cerebrum.strategies.base import StrategyConfig  # ADJUST import to match orb_stocks.py


XSTOCKS_REVERSION_CONFIG = StrategyConfig(
    name="xstocks_reversion",
    initial_balance=Decimal("5000.0"),
    symbols=["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"],
    signal_source_filter=None,  # consumes ALL signal types
    aggregator_weights={
        "technical": Decimal("1.2"),
        "sentiment": Decimal("0"),
        "news": Decimal("0"),
        "regime": Decimal("0.5"),
    },
    aggregator_threshold=Decimal("0.4"),
    risk_overrides={
        "position_size_percent": Decimal("20.0"),
        "stop_loss_percent": Decimal("1.0"),
        "take_profit_percent": Decimal("1.5"),
        "max_position_age_minutes": 120,
        "min_hold_minutes": 15,
        "post_fill_cooldown_seconds": 1800,
        "min_signal_strength": Decimal("0.65"),
    },
    # exit_config — match whatever orb_stocks.py or mean_reversion.py sets
)
```

ADJUST field names and any additional required fields to match what you see in `orb_stocks.py` (the most recently created strategy config).

- [ ] **Step 5: Run → 3 pass**

- [ ] **Step 6: Commit**

```bash
git add cerebrum/strategies/xstocks_reversion.py tests/unit/test_xstocks_reversion_config.py
git commit --no-gpg-sign -m "feat(strategies): XSTOCKS_REVERSION_CONFIG for 24/7 tokenized equities (DEC-XSTOCKS-002)"
```

---

# Task 3: main.py wiring + 3 unit tests

**Files:**
- Modify: `cerebrum/main.py`
- Create: `tests/unit/test_main_xstocks_wiring.py`

- [ ] **Step 1: Read the existing Alpaca wiring as template**

```bash
grep -n "alpaca\|_maybe_build_alpaca" cerebrum/main.py | head -20
```

The `_maybe_build_alpaca_adapter` helper (Task 6) is the exact pattern to follow.

- [ ] **Step 2: Write the 3 failing tests**

Create `tests/unit/test_main_xstocks_wiring.py`:

```python
"""Test KrakenXStocksAdapter wiring in main.py (DEC-XSTOCKS-001)."""
import pytest
from unittest.mock import MagicMock

from cerebrum.main import _maybe_build_kraken_xstocks_adapter


def test_returns_none_when_disabled():
    config = {"kraken_xstocks": {"enabled": False}}
    assert _maybe_build_kraken_xstocks_adapter(config, MagicMock()) is None


def test_returns_none_when_section_missing():
    config = {}
    assert _maybe_build_kraken_xstocks_adapter(config, MagicMock()) is None


def test_returns_none_when_sdk_missing(monkeypatch):
    config = {"kraken_xstocks": {"enabled": True, "symbols": ["AAPLx/USD"]}}
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    # Simulate missing SDK by patching the import
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if "kraken_xstocks" in name or "kraken.spot" in name:
            raise ImportError("no kraken sdk")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = _maybe_build_kraken_xstocks_adapter(config, MagicMock())
    assert result is None
```

- [ ] **Step 3: Run → fail**

- [ ] **Step 4: Add `_maybe_build_kraken_xstocks_adapter` to `cerebrum/main.py`**

Follow the exact same pattern as `_maybe_build_alpaca_adapter`:

```python
def _maybe_build_kraken_xstocks_adapter(config: dict, event_bus) -> Any | None:
    """Conditionally build the Kraken xStocks adapter.

    Returns None when disabled or SDK not installed.
    Uses existing EXCHANGE_API_KEY / EXCHANGE_API_SECRET (same Kraken account).
    """
    xstocks_cfg = config.get("kraken_xstocks", {})
    if not xstocks_cfg.get("enabled", False):
        return None

    try:
        from cerebrum.adapters.kraken_xstocks import KrakenXStocksAdapter
    except ImportError as e:
        log.warning("kraken_xstocks_unavailable", extra={"reason": str(e)})
        return None

    try:
        adapter = KrakenXStocksAdapter(
            bus=event_bus,
            config=xstocks_cfg,
        )
    except RuntimeError as e:
        if "credentials" in str(e):
            log.warning("kraken_xstocks_auth_failed", extra={"error": str(e)})
            return None
        raise

    log.info("kraken_xstocks_adapter_built", extra={"symbols": xstocks_cfg.get("symbols", [])})
    return adapter
```

Then in the startup flow (near the Alpaca wiring):

```python
xstocks_adapter = _maybe_build_kraken_xstocks_adapter(config, event_bus)
if xstocks_adapter is not None:
    await xstocks_adapter.connect()
    await xstocks_adapter.subscribe_market_data(config["kraken_xstocks"]["symbols"])
```

And in shutdown:
```python
if xstocks_adapter is not None:
    await xstocks_adapter.disconnect()
```

Also add the `xstocks_reversion` strategy registration gate (same pattern as orb_stocks):

```python
_xstocks_cfg = self._raw_toml.get("strategy", {}).get("xstocks_reversion", {})
if _xstocks_cfg.get("enabled", False):
    try:
        from cerebrum.strategies.xstocks_reversion import XSTOCKS_REVERSION_CONFIG
        self.strategy_registry.register(XSTOCKS_REVERSION_CONFIG)
        self._log.info("xstocks_reversion_strategy_registered")
    except ImportError:
        self._log.warning("xstocks_reversion_unavailable")
```

- [ ] **Step 5: Run → 3 pass. Also run regression:**

```bash
.venv/bin/pytest tests/unit/ tests/integration/ --ignore=tests/live -q --tb=no 2>&1 | tail -3
```

Expected: 780+ passed (768 baseline + 12 new) + 2 pre-existing failed. No new failures.

- [ ] **Step 6: Commit**

```bash
git add cerebrum/main.py tests/unit/test_main_xstocks_wiring.py
git commit --no-gpg-sign -m "feat(main): conditional Kraken xStocks wiring (DEC-XSTOCKS-001)"
```

---

# Task 4: Config + pyproject.toml + state migration

**Files:**
- Modify: `config/paper.toml`
- Modify: `pyproject.toml`
- Modify: `cerebrum/adapters/paper.py` (migration function)

- [ ] **Step 1: Add config sections to `config/paper.toml`**

Append at end of file:

```toml
# === Kraken xStocks adapter (tokenized equities, 24/7) ===
[kraken_xstocks]
enabled = false                              # DEC-XSTOCKS-001 — flip after live verification
symbols = ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]

# === xStocks mean-reversion strategy ===
[strategy.xstocks_reversion]
enabled = false                              # gate until adapter also enabled
initial_balance = 5000.0
symbols = ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]
signal_source_filter = ""
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

- [ ] **Step 2: Add optional dep to `pyproject.toml`**

Find the `[project.optional-dependencies]` section. Add:

```toml
xstocks = [
    "python-kraken-sdk>=3.2.0",
]
```

- [ ] **Step 3: Extend state migration to handle `xstocks_reversion`**

In `cerebrum/adapters/paper.py`, find `migrate_state_v2_to_v3`. After the block that adds `orb_stocks`, add:

```python
if "xstocks_reversion" not in snapshots:
    snapshots["xstocks_reversion"] = {
        "cash_balance": str(initial_balance_orb),  # reuse param or add new one
        "initial_balance": str(initial_balance_orb),
        "peak_equity": str(initial_balance_orb),
        "total_realized_pnl": "0",
        "positions": {},
    }
```

Or better: make the migration function accept a list of `(strategy_name, initial_balance)` tuples to add, keeping it general. But the simplest approach: just add another block for `xstocks_reversion` with $5000 default. This is idempotent — if the snapshot already exists, it's skipped.

- [ ] **Step 4: Verify config parses**

```bash
.venv/bin/python -c "
import tomllib
with open('config/paper.toml', 'rb') as f:
    c = tomllib.load(f)
print('kraken_xstocks:', c.get('kraken_xstocks', {}).get('enabled'))
print('xstocks_reversion:', c.get('strategy', {}).get('xstocks_reversion', {}).get('enabled'))
print('xstocks symbols:', c.get('kraken_xstocks', {}).get('symbols'))
"
```

Expected: `False`, `False`, `['AAPLx/USD', 'MSFTx/USD', 'NVDAx/USD']`

- [ ] **Step 5: Regression check**

```bash
.venv/bin/pytest tests/unit/ tests/integration/ --ignore=tests/live -q --tb=no 2>&1 | tail -3
```

Expected: same pass count, no new failures. Config changes are `enabled = false` → runtime no-op.

- [ ] **Step 6: Commit**

```bash
git add config/paper.toml pyproject.toml cerebrum/adapters/paper.py
git commit --no-gpg-sign -m "config: add kraken_xstocks + xstocks_reversion sections; extend state migration"
```

---

# Task 5: Final regression + live smoke test

- [ ] **Step 1: Full regression**

```bash
.venv/bin/pytest tests/unit/ tests/integration/ --ignore=tests/live -q --tb=short 2>&1 | tail -10
```

Expected: 780+ passed + 2 pre-existing failed + 7 skipped. Zero new failures.

- [ ] **Step 2: Startup smoke with xStocks disabled**

```bash
timeout 8 .venv/bin/python -m cerebrum --mode paper --config config/paper.toml 2>&1 | grep -iE 'starting|strategy|adapter|xstocks|error' | head -20
```

Expected: 4 strategies registered (mean_reversion, range_trading, orb_stocks, xstocks_reversion — wait, xstocks_reversion is `enabled=false` so it should NOT register). Should see 3 strategies: mean_reversion, range_trading, orb_stocks. NO xstocks lines. NO errors.

- [ ] **Step 3: Flip flags + test with xStocks enabled (quick live check)**

Temporarily edit `config/paper.toml`:
- `[kraken_xstocks].enabled = true`
- `[strategy.xstocks_reversion].enabled = true`

```bash
timeout 15 .venv/bin/python -m cerebrum --mode paper --config config/paper.toml 2>&1 | grep -iE 'starting|strategy|xstocks|adapter|connected|error|market_data' | head -30
```

Expected:
- `kraken_xstocks_adapter_built` with symbols
- `xstocks_reversion_strategy_registered`
- Either `kraken_xstocks_connected` (if your Kraken account supports xStocks)
- Or `kraken_xstocks_auth_failed` / `kraken_xstocks_geo_blocked` (graceful fallback)
- If connected: `AAPLx/USD` market data flowing (if market is streaming)
- 4 strategies total in the registry

**Revert the flag flips after testing** (or leave them enabled if everything worked and you want the live session to include xStocks).

- [ ] **Step 4: Branch summary**

```bash
git log --oneline main..HEAD | head -25
echo "---"
git diff --stat main..HEAD | tail -5
```

- [ ] **Step 5: Commit any remaining changes** (e.g., if you left xStocks enabled in config)

```bash
git add -A
git commit --no-gpg-sign -m "feat: enable kraken_xstocks + xstocks_reversion in paper.toml"
```

---

## Self-Review

**Spec coverage:**
- DEC-XSTOCKS-001 (adapter) → Task 1
- DEC-XSTOCKS-002 (strategy config) → Task 2
- DEC-XSTOCKS-003 (credential reuse) → Task 1 (reads EXCHANGE_API_KEY)
- Config + pyproject → Task 4
- State migration → Task 4
- Error handling (missing SDK, auth fail, geo-block) → Task 1 adapter + Task 3 wiring
- Regression + smoke → Task 5

All spec requirements covered.

**Type consistency:**
- `KrakenXStocksAdapter(bus, config)` — consistent across Tasks 1 and 3
- `XSTOCKS_REVERSION_CONFIG` — consistent name in Tasks 2 and 3
- `_maybe_build_kraken_xstocks_adapter(config, event_bus)` — consistent in Tasks 3

**Known open items:**
- SpotWSClient ticker message shape: Task 1 Step 4 instructs the implementer to read the SDK source first. The test in Step 2 uses a `_handle_ticker_update()` helper that abstracts the raw WS message parsing.
- `signal_source_filter = ""` vs `null` in TOML: TOML doesn't have null. Use empty string and have the config loader treat `""` as `None`. Or omit the key entirely and let the default apply. The implementer should check how existing strategies handle this.
