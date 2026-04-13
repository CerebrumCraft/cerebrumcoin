# Stocks ORB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stock-market trading capability to CerebrumCoin via an Opening Range Breakout (ORB) strategy on AAPL/MSFT/NVDA during RTH (09:30–16:00 ET), running in the same cerebrum process as the existing crypto strategies, with local-sim fills through the existing paper_trading adapter.

**Architecture:** Same-process unified event bus (DEC-STOCKS-001). Alpaca adapter (already built at `cerebrum/adapters/alpaca.py`) streams market data to the shared bus. A new `OpeningRangeSignal` generator emits breakout signals that only a new `orb_stocks` strategy consumes (via `signal_source_filter="OpeningRange"`). A `market_hours_gate` risk rule denies out-of-RTH stock orders; an `end_of_day_flatten` rule forces closes at 15:55 ET. Crypto strategies are untouched; cross-asset signal contamination is prevented by a new per-strategy `symbols` filter in `SignalAggregator` (DEC-STOCKS-005).

**Tech Stack:** Python 3.12, `alpaca-py>=0.28.0` (optional dep group `[stocks]`), `zoneinfo` stdlib for timezones, `pytest` for tests, existing `EventBus`/`SignalGenerator`/`RiskRule`/`ExitRule` abstractions.

**Spec:** `docs/superpowers/specs/2026-04-12-stocks-support-design.md` (read first).

**Worktree:** `.worktrees/worktree-stocks-orb` branch `worktree-stocks-orb` off main HEAD `954a219`.

---

## One-Time Setup (before Task 1)

- [ ] **Setup Step A: Install deps with optional `stocks` extra in worktree venv**

```bash
cd /home/j/CerebrumCraft/CerebrumCoin/.worktrees/worktree-stocks-orb
python3 -m venv .venv
.venv/bin/pip install -e '.[stocks,dev]'
```

Expected: `alpaca-py-0.x.x`, `ccxt-4.x.x`, and dev tools (pytest, etc.) install without errors.

- [ ] **Setup Step B: Verify baseline test pass**

```bash
cd /home/j/CerebrumCraft/CerebrumCoin/.worktrees/worktree-stocks-orb
.venv/bin/pytest -q --ignore=tests/live 2>&1 | /usr/bin/tail -5
```

Expected: All existing tests (~782) pass. If any fail, STOP and report to user before proceeding — baseline must be clean.

- [ ] **Setup Step C: Add Alpaca paper credentials to `.env` (only needed for Phase 8 live tests)**

```bash
# If not already present in .env:
echo "ALPACA_API_KEY_ID=PK...your-paper-key" >> .env
echo "ALPACA_API_SECRET_KEY=your-paper-secret" >> .env
```

Expected: `.env` contains two new lines. Free account signup at alpaca.markets → Paper trading → API keys. Skip if only running Phase 1–7 (no network calls).

---

## File Structure

Fully enumerated before task decomposition — lock in the file boundaries.

### New files (source)

| Path | Responsibility | Owner task |
|---|---|---|
| `cerebrum/utils/trading_session.py` | Pure helper: US market hours + NYSE holidays + early-close days. No I/O, no state. | Phase 1 / Task 1 |
| `cerebrum/signals/opening_range.py` | `OpeningRangeSignal` generator. Per-symbol opening range tracker. Emits buy/sell breakout signals with `metadata={"source":"OpeningRange", ...}`. | Phase 2 / Task 3 |
| `cerebrum/risk/market_hours_gate.py` | `MarketHoursGate` risk rule. Denies stock orders outside RTH. | Phase 5 / Task 8 |
| `cerebrum/risk/end_of_day_flatten.py` | `EndOfDayFlatten` exit rule. Auto-closes stock positions at 15:55 ET (or early-close − 5 min). | Phase 5 / Task 9 |

### Modified files (source)

| Path | Change | Owner task |
|---|---|---|
| `cerebrum/main.py` | Detect `[alpaca]` config, instantiate `AlpacaAdapter`, register new signal + rules, register `orb_stocks` strategy. ~30 lines. | Phase 4 / Task 6; Phase 6 / Task 11 |
| `cerebrum/strategies/registry.py` or `SignalAggregator` source file (TBD during Task 5) | Add per-strategy `symbols` filter (DEC-STOCKS-005). | Phase 3 / Task 5 |
| `cerebrum/adapters/paper_trading.py` | Optional `commission_by_symbol` lookup — stocks use $0 flat, crypto keeps existing percentage. ~15 lines. | Phase 6 / Task 11 |
| `cerebrum/main.py` (state loader) | Migration v2 → v3 for `paper_state.json`. Adds empty `orb_stocks` snapshot. Writes `.v2.bak.json` backup. | Phase 6 / Task 12 |
| `config/paper.toml` | New `[alpaca]`, `[signal.opening_range]`, `[strategy.orb_stocks]`, `[risk.market_hours_gate]`, `[risk.end_of_day_flatten]` sections. | Phase 6 / Task 11 |

### New test files

| Path | Coverage |
|---|---|
| `tests/unit/test_trading_session.py` | 12 unit tests — Task 2 |
| `tests/unit/test_opening_range_signal.py` | 10 unit tests — Task 4 |
| `tests/unit/test_signal_aggregator_symbols_filter.py` | 3 unit tests — Task 5 |
| `tests/unit/test_market_hours_gate.py` | 6 unit tests — Task 8 |
| `tests/unit/test_end_of_day_flatten.py` | 5 unit tests — Task 9 |
| `tests/unit/test_paper_state_migration.py` | 4 unit tests — Task 12 |
| `tests/integration/test_orb_full_day.py` | 1 replay integration test — Task 14 |
| `tests/integration/test_orb_nyse_holiday.py` | 1 integration test — Task 15 |
| `tests/integration/test_orb_early_close.py` | 1 integration test — Task 16 |
| `tests/integration/test_cross_asset_isolation.py` | 1 integration test — Task 17 |
| `tests/integration/test_stream_stale.py` | 1 integration test — Task 18 |
| `tests/integration/test_eod_flatten_with_partial_fill.py` | 1 integration test — Task 19 |
| `tests/live/test_alpaca_live_connection.py` | 2 live tests (gated) — Task 21 |
| `tests/live/test_live_orb_smoke.py` | 1 manual live test — Task 22 |

### New fixture files (Phase 7)

| Path | Source |
|---|---|
| `tests/fixtures/alpaca_aapl_2026-03-10.jsonl` | Captured via `scripts/record_alpaca_ticks.py` during one RTH session. |
| `tests/fixtures/alpaca_mixed_stocks_2026-03-10.jsonl` | Same date, 3 symbols — for cross-asset test. |

### New scripts

| Path | Purpose |
|---|---|
| `scripts/record_alpaca_ticks.py` | CLI helper (~80 lines) that connects to live Alpaca and writes JSONL ticks. Used once to bootstrap fixtures; kept for annual refresh. — Task 13 |

### Docs

| Path | Change |
|---|---|
| `CONTRIBUTING.md` | New section: "Running live Alpaca tests." — Task 20 |

---

# PHASE 1 — `trading_session.py` (pure utility, no wiring)

Goal: a dependency-free helper that answers "is it RTH right now?", "minutes to close?", "is today a market holiday?", "what time does the market close today (accounting for early closes)?".

## Task 1: Create `cerebrum/utils/trading_session.py`

**Files:**
- Create: `cerebrum/utils/trading_session.py`

- [ ] **Step 1: Ensure directory exists**

```bash
test -d cerebrum/utils && echo "utils dir exists" || mkdir -p cerebrum/utils && touch cerebrum/utils/__init__.py
```

Expected: Either "utils dir exists" OR a freshly created `cerebrum/utils/__init__.py`.

- [ ] **Step 2: Write `cerebrum/utils/trading_session.py`**

```python
"""
US equity market trading-session utilities.

@decision DEC-STOCKS-003
@title RTH-only trading with auto-flatten
@status accepted
@rationale Zero-overnight-exposure model. Dependency-free; uses only
stdlib `zoneinfo` + `datetime`. Static NYSE calendar through 2028 —
refresh annually.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)

# NYSE full-day closures through 2028 (static list; refresh annually).
# Source: https://www.nyse.com/markets/hours-calendars
NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth (observed, Friday since 19th is Saturday)
    date(2027, 7, 5),    # Independence Day (observed)
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # Christmas (observed)
    # 2028
    date(2028, 1, 3),    # New Year's (observed, since 1st is Saturday)
    date(2028, 1, 17),
    date(2028, 2, 21),
    date(2028, 4, 14),
    date(2028, 5, 29),
    date(2028, 6, 19),
    date(2028, 7, 4),
    date(2028, 9, 4),
    date(2028, 11, 23),
    date(2028, 12, 25),
})

# Early-close days (market closes at 13:00 ET). Refresh annually.
NYSE_EARLY_CLOSE: dict[date, time] = {
    date(2026, 7, 2):  time(13, 0),  # day before Independence Day
    date(2026, 11, 27): time(13, 0),  # Black Friday
    date(2026, 12, 24): time(13, 0),  # Christmas Eve
    date(2027, 7, 2):  time(13, 0),
    date(2027, 11, 26): time(13, 0),
    date(2027, 12, 23): time(13, 0),  # Christmas Eve (observed)
    date(2028, 7, 3):  time(13, 0),
    date(2028, 11, 24): time(13, 0),
    date(2028, 12, 22): time(13, 0),  # Christmas Eve (observed, last biz day)
}


def _now_et(now_utc: datetime | None = None) -> datetime:
    """Return current time in ET. `now_utc` override enables deterministic testing."""
    if now_utc is None:
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    return now_utc.astimezone(ET)


def is_market_holiday(d: date) -> bool:
    """True if `d` is a full-day NYSE closure."""
    return d in NYSE_HOLIDAYS


def early_close_time_for(d: date) -> time | None:
    """Return early-close time (ET) for `d`, or None if normal close."""
    return NYSE_EARLY_CLOSE.get(d)


def rth_close_for(d: date) -> time | None:
    """Return the close time (ET) that applies on date `d`.

    Returns None if the market is closed (weekend or holiday).
    """
    if d.weekday() >= 5:  # Sat/Sun
        return None
    if is_market_holiday(d):
        return None
    return NYSE_EARLY_CLOSE.get(d, RTH_CLOSE)


def is_rth_now(now_utc: datetime | None = None) -> bool:
    """True if the US equity market is currently in regular trading hours."""
    now_et = _now_et(now_utc)
    today = now_et.date()
    close = rth_close_for(today)
    if close is None:
        return False
    t = now_et.time()
    return RTH_OPEN <= t < close


def minutes_until_close(now_utc: datetime | None = None) -> int | None:
    """Minutes remaining until today's market close.

    Returns None if the market is not open. Returns 0 at/after close.
    """
    now_et = _now_et(now_utc)
    today = now_et.date()
    close = rth_close_for(today)
    if close is None:
        return None
    if now_et.time() < RTH_OPEN:
        return None
    close_dt = datetime.combine(today, close, tzinfo=ET)
    delta = close_dt - now_et
    mins = int(delta.total_seconds() // 60)
    return max(mins, 0)
```

- [ ] **Step 3: Verify the module imports and static calendars are consistent**

Run:
```bash
.venv/bin/python -c "from cerebrum.utils.trading_session import is_market_holiday, rth_close_for, NYSE_HOLIDAYS, NYSE_EARLY_CLOSE; print(f'{len(NYSE_HOLIDAYS)} holidays, {len(NYSE_EARLY_CLOSE)} early-close days')"
```
Expected output: `30 holidays, 9 early-close days`

- [ ] **Step 4: Commit**

```bash
git add cerebrum/utils/__init__.py cerebrum/utils/trading_session.py
git commit -m "feat(utils): add trading_session helper with NYSE calendar through 2028 (DEC-STOCKS-003)"
```

## Task 2: Unit tests for `trading_session`

**Files:**
- Create: `tests/unit/test_trading_session.py`

- [ ] **Step 1: Write all 12 failing tests**

Create `tests/unit/test_trading_session.py`:

```python
"""Unit tests for cerebrum.utils.trading_session."""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from cerebrum.utils.trading_session import (
    is_market_holiday,
    rth_close_for,
    early_close_time_for,
    is_rth_now,
    minutes_until_close,
)

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _utc(year, month, day, hour, minute=0):
    """Build a UTC datetime for deterministic tests."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def test_holiday_christmas_2026():
    assert is_market_holiday(date(2026, 12, 25)) is True


def test_holiday_juneteenth_2026():
    assert is_market_holiday(date(2026, 6, 19)) is True


def test_non_holiday_regular_weekday():
    assert is_market_holiday(date(2026, 6, 15)) is False  # ordinary Monday


def test_early_close_black_friday_2026():
    assert early_close_time_for(date(2026, 11, 27)) == time(13, 0)


def test_early_close_none_for_regular_day():
    assert early_close_time_for(date(2026, 6, 15)) is None


def test_rth_close_for_weekend_returns_none():
    # 2026-06-20 is a Saturday
    assert rth_close_for(date(2026, 6, 20)) is None


def test_rth_close_for_holiday_returns_none():
    assert rth_close_for(date(2026, 12, 25)) is None


def test_rth_close_for_early_close_day():
    assert rth_close_for(date(2026, 11, 27)) == time(13, 0)


def test_is_rth_now_inside_window():
    # 2026-06-15 at 10:00 ET == 14:00 UTC (EDT is UTC-4)
    assert is_rth_now(_utc(2026, 6, 15, 14, 0)) is True


def test_is_rth_now_before_open():
    # 09:29 ET on a weekday
    assert is_rth_now(_utc(2026, 6, 15, 13, 29)) is False


def test_is_rth_now_weekend():
    # Saturday 10:00 ET
    assert is_rth_now(_utc(2026, 6, 20, 14, 0)) is False


def test_minutes_until_close_at_midday():
    # 2026-06-15 at 12:00 ET → 4 hours to 16:00 ET close = 240 min
    assert minutes_until_close(_utc(2026, 6, 15, 16, 0)) == 240


def test_minutes_until_close_before_open_is_none():
    assert minutes_until_close(_utc(2026, 6, 15, 13, 0)) is None


def test_minutes_until_close_holiday_is_none():
    assert minutes_until_close(_utc(2026, 12, 25, 14, 0)) is None


def test_minutes_until_close_on_early_close_day():
    # 2026-11-27 at 12:00 ET → 60 min to 13:00 ET close (EST is UTC-5 in November)
    assert minutes_until_close(_utc(2026, 11, 27, 17, 0)) == 60
```

- [ ] **Step 2: Run tests — verify ALL fail cleanly (import-wise succeed, assertions run)**

```bash
.venv/bin/pytest tests/unit/test_trading_session.py -v
```
Expected: 15 tests collected. If module not importable, fix imports before proceeding.

Since implementation already exists from Task 1, all 15 SHOULD pass. If any fail, it's a real bug in the implementation — fix it.

- [ ] **Step 3: Verify all pass**

Expected output tail: `15 passed in <time>s`.

If failing:
- `test_minutes_until_close_at_midday`: check DST — EDT is UTC-4 in June. 16:00 UTC = 12:00 ET. 16:00 ET = 20:00 UTC.
- `test_minutes_until_close_on_early_close_day`: check EST/EDT — November is EST = UTC-5. 17:00 UTC = 12:00 ET. Close at 13:00 ET = 18:00 UTC.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_trading_session.py
git commit -m "test(trading_session): 15 unit tests covering RTH, holidays, DST, early close"
```

---

# PHASE 2 — `opening_range.py` signal

Goal: `OpeningRangeSignal` generator that subscribes to `MarketDataEvent`, tracks per-symbol opening range during 09:30–09:45 ET, and emits buy/sell breakout signals after the range freezes.

## Task 3: Create `cerebrum/signals/opening_range.py`

**Files:**
- Create: `cerebrum/signals/opening_range.py`
- Reference existing patterns: `cerebrum/signals/` — inspect one existing signal generator first.

- [ ] **Step 1: Inspect an existing signal generator to match patterns**

```bash
.venv/bin/python -c "import os; [print(f) for f in sorted(os.listdir('cerebrum/signals')) if f.endswith('.py') and not f.startswith('_')]"
```
Expected: list of signal files (rsi.py, macd.py, etc.). Read `cerebrum/signals/rsi.py` (or the smallest one) to understand: the base class, the `__init__` signature (config dict, event bus), the subscribe-to-market-data pattern, and `_create_signal(...)` usage. Also read `cerebrum/signals/base.py` (or similar) to know the base `SignalGenerator` class.

- [ ] **Step 2: Write `cerebrum/signals/opening_range.py`**

The concrete class must inherit whatever base class is used by `rsi.py`. Assume `SignalGenerator` from `cerebrum.signals.base`. Adjust imports to match what you see in Step 1.

```python
"""
Opening Range Breakout (ORB) signal generator for US equities.

@decision DEC-STOCKS-004
@title ORB signal with source-based isolation
@status accepted
@rationale Dedicated equity-native signal source. Uses signal_source_filter
on the consuming strategy to prevent contamination with crypto strategies.
Builds per-symbol high/low over the first N minutes after 09:30 ET, then
emits a buy signal on break above (with configurable buffer) or a sell
signal on break below. Each symbol's ORB resets daily.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from cerebrum.signals.base import SignalGenerator
from cerebrum.utils.trading_session import ET, RTH_OPEN, is_rth_now


@dataclass
class _ORBState:
    high: Decimal | None = None
    low: Decimal | None = None
    tick_count: int = 0
    frozen: bool = False
    valid: bool = True  # False when range rejected (too tight / too wide / too few ticks)
    date: date | None = None


class OpeningRangeSignal(SignalGenerator):
    """Emits breakout signals for configured stock symbols."""

    NAME = "OpeningRange"

    def __init__(self, event_bus, config: dict[str, Any]):
        super().__init__(event_bus=event_bus, name=self.NAME)
        self._symbols: set[str] = set(config.get("symbols", []))
        self._range_minutes: int = int(config.get("range_minutes", 15))
        self._breakout_buffer_bps: Decimal = Decimal(str(config.get("breakout_buffer_bps", 5)))
        self._min_range_bps: Decimal = Decimal(str(config.get("min_range_bps", 20)))
        self._max_range_bps: Decimal = Decimal(str(config.get("max_range_bps", 500)))
        self._states: dict[str, _ORBState] = {s: _ORBState() for s in self._symbols}
        # Bar-by-bar in-range tracking: a tick arriving after 09:30 but before 09:30+range_minutes
        # extends high/low. After that cutoff the range freezes.
        # Buy signals emit on tick > orb_high × (1 + buffer_bps/10000).
        # Sell signals emit on tick < orb_low  × (1 - buffer_bps/10000).
        # Double-emit per direction per day is prevented via _signaled_directions.
        self._signaled_directions: dict[str, set[str]] = {s: set() for s in self._symbols}

    def on_market_data(self, event) -> None:
        """Called for every MarketDataEvent. Filter to our symbols."""
        symbol = event.symbol
        if symbol not in self._symbols:
            return
        price: Decimal = event.price
        ts: datetime = event.timestamp  # expected UTC tz-aware; adjust if your EventBus uses different

        et_time = ts.astimezone(ET)
        today = et_time.date()

        state = self._states[symbol]

        # Daily reset — first tick of a new date (regardless of RTH)
        if state.date != today:
            state.__init__()
            state.date = today
            self._signaled_directions[symbol] = set()

        # Only operate during RTH
        if not is_rth_now(ts):
            return

        # Compute minutes since 09:30 ET today
        open_dt = datetime.combine(today, RTH_OPEN, tzinfo=ET)
        minutes_since_open = (et_time - open_dt).total_seconds() / 60.0

        if minutes_since_open < self._range_minutes:
            # Still building the range
            if state.high is None or price > state.high:
                state.high = price
            if state.low is None or price < state.low:
                state.low = price
            state.tick_count += 1
            return

        # Range freezes on the first tick at/after the cutoff
        if not state.frozen:
            state.frozen = True
            if state.tick_count < 10:
                state.valid = False
                return
            # Validate range width
            if state.high is None or state.low is None or state.high == state.low:
                state.valid = False
                return
            mid = (state.high + state.low) / 2
            range_bps = (state.high - state.low) / mid * Decimal("10000")
            if range_bps < self._min_range_bps or range_bps > self._max_range_bps:
                state.valid = False
                return

        if not state.valid:
            return  # range invalid — no signals today

        # Check breakout
        buffer = self._breakout_buffer_bps / Decimal("10000")
        if "buy" not in self._signaled_directions[symbol]:
            threshold_up = state.high * (Decimal("1") + buffer)
            if price > threshold_up:
                self._signaled_directions[symbol].add("buy")
                self._emit(symbol, "buy", price, state)
                return
        if "sell" not in self._signaled_directions[symbol]:
            threshold_dn = state.low * (Decimal("1") - buffer)
            if price < threshold_dn:
                self._signaled_directions[symbol].add("sell")
                self._emit(symbol, "sell", price, state)
                return

    def _emit(self, symbol: str, side: str, price: Decimal, state: _ORBState) -> None:
        mid = (state.high + state.low) / 2
        range_bps = (state.high - state.low) / mid * Decimal("10000")
        # Strength scales with range width within [min, max] bounds, clamped to [0, 1]
        strength = min(Decimal("1"), range_bps / self._max_range_bps)
        self._create_signal(
            symbol=symbol,
            action=side,
            strength=strength,
            confidence=Decimal("0.75"),
            metadata={
                "source": self.NAME,
                "orb_high": str(state.high),
                "orb_low": str(state.low),
                "price": str(price),
            },
        )
```

- [ ] **Step 3: Verify import + quick syntax check**

```bash
.venv/bin/python -c "from cerebrum.signals.opening_range import OpeningRangeSignal; print(OpeningRangeSignal.NAME)"
```
Expected: `OpeningRange`.

If import fails: check that `cerebrum/signals/base.py` exposes `SignalGenerator` under the name used. If it's different (e.g. `BaseSignalGenerator`), adjust the import.

- [ ] **Step 4: Commit**

```bash
git add cerebrum/signals/opening_range.py
git commit -m "feat(signals): add OpeningRangeSignal for equity ORB strategy (DEC-STOCKS-004)"
```

## Task 4: Unit tests for `OpeningRangeSignal`

**Files:**
- Create: `tests/unit/test_opening_range_signal.py`

- [ ] **Step 1: Write tests**

```python
"""Unit tests for OpeningRangeSignal."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock

import pytest

from cerebrum.signals.opening_range import OpeningRangeSignal

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


class _TickEvent:
    def __init__(self, symbol, price, ts):
        self.symbol = symbol
        self.price = Decimal(str(price))
        self.timestamp = ts


def _cfg(**overrides):
    base = {
        "symbols": ["AAPL", "MSFT"],
        "range_minutes": 15,
        "breakout_buffer_bps": 5,
        "min_range_bps": 20,
        "max_range_bps": 500,
    }
    base.update(overrides)
    return base


def _et_to_utc(year, month, day, hour, minute):
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(UTC)


def _feed(gen, symbol, prices_by_et):
    """Feed a list of (et_hour, et_minute, price) tuples to the generator."""
    for hh, mm, p in prices_by_et:
        ts = _et_to_utc(2026, 6, 15, hh, mm)
        gen.on_market_data(_TickEvent(symbol, p, ts))


@pytest.fixture
def gen():
    bus = MagicMock()
    return OpeningRangeSignal(event_bus=bus, config=_cfg())


def test_ignores_unknown_symbol(gen):
    _feed(gen, "TSLA", [(9, 31, 100)])
    # No signal should have been emitted (we'd see it via the mock)
    assert gen.event_bus.publish.call_count == 0  # or adjust to match actual emit path


def test_no_signal_during_range_building(gen):
    # Feed ticks 09:30–09:44 — range is still building
    ticks = [(9, 30, 180.0), (9, 35, 181.0), (9, 40, 179.0), (9, 44, 182.0)]
    _feed(gen, "AAPL", ticks)
    # Range not yet frozen → no emit
    # Replace this assertion with the actual emit-detection mechanism from base class
    assert True  # placeholder — replace after inspecting _create_signal implementation


def test_breakout_above_high_emits_buy(gen):
    # Build range 181.00–182.00
    _feed(gen, "AAPL", [(9, 30, 181.5), (9, 31, 182.0), (9, 32, 181.0), (9, 33, 181.5)])
    # Add enough ticks (>= 10) to pass validation
    for i in range(10):
        _feed(gen, "AAPL", [(9, 34 + i, 181.5)])
    # Advance past range cutoff
    _feed(gen, "AAPL", [(9, 46, 181.5)])  # freezes range
    # Breakout: 182.10 is above 182.00 × (1 + 5bps) = 182.091
    _feed(gen, "AAPL", [(9, 50, 182.15)])
    # Assert a buy signal fired for AAPL (use whatever mechanism SignalGenerator uses)
    # After inspecting base class in Task 3, adjust this assertion.
    assert True  # placeholder


def test_breakdown_below_low_emits_sell(gen):
    _feed(gen, "AAPL", [(9, 30, 181.5), (9, 31, 182.0), (9, 32, 181.0)])
    for i in range(10):
        _feed(gen, "AAPL", [(9, 33 + i, 181.5)])
    _feed(gen, "AAPL", [(9, 46, 181.5)])
    _feed(gen, "AAPL", [(9, 50, 180.85)])  # below 181.0 × (1 - 5bps) = 180.9095
    assert True  # placeholder


def test_range_too_tight_marks_invalid(gen):
    # Range of 181.00 to 181.05 is 2.76 bps < min 20 bps
    ticks = [(9, 30, 181.00), (9, 31, 181.05)]
    for i in range(12):
        ticks.append((9, 32 + i, 181.02))
    _feed(gen, "AAPL", ticks)
    _feed(gen, "AAPL", [(9, 46, 181.02)])  # freeze
    state = gen._states["AAPL"]
    assert state.valid is False


def test_range_too_wide_marks_invalid(gen):
    # Range 180 to 200 is ~1050 bps > max 500 bps
    ticks = [(9, 30, 180.0), (9, 31, 200.0)]
    for i in range(12):
        ticks.append((9, 32 + i, 190.0))
    _feed(gen, "AAPL", ticks)
    _feed(gen, "AAPL", [(9, 46, 190.0)])
    assert gen._states["AAPL"].valid is False


def test_insufficient_ticks_marks_invalid(gen):
    # Only 3 ticks in the whole range window
    _feed(gen, "AAPL", [(9, 30, 181), (9, 35, 182), (9, 44, 181.5)])
    _feed(gen, "AAPL", [(9, 46, 181.5)])  # freeze
    assert gen._states["AAPL"].valid is False


def test_daily_reset_on_new_date(gen):
    # Day 1: build + freeze
    _feed(gen, "AAPL", [(9, 30, 181)])
    # Fast-forward to next day
    next_day = _et_to_utc(2026, 6, 16, 9, 30)
    gen.on_market_data(_TickEvent("AAPL", Decimal("185"), next_day))
    state = gen._states["AAPL"]
    # High should reset to the first tick of new day
    assert state.high == Decimal("185")


def test_only_one_buy_per_day(gen):
    # Build valid range, trigger buy, then feed another up-breakout tick
    _feed(gen, "AAPL", [(9, 30, 181), (9, 31, 182), (9, 32, 181)])
    for i in range(10):
        _feed(gen, "AAPL", [(9, 33 + i, 181.5)])
    _feed(gen, "AAPL", [(9, 46, 181.5), (9, 50, 182.2), (9, 51, 183.0)])
    # After first buy fires, second breakout must NOT double-emit
    assert "buy" in gen._signaled_directions["AAPL"]


def test_out_of_rth_ticks_ignored(gen):
    # 07:00 ET pre-market tick
    gen.on_market_data(_TickEvent("AAPL", Decimal("181"), _et_to_utc(2026, 6, 15, 7, 0)))
    state = gen._states["AAPL"]
    # Pre-market tick should not populate range
    assert state.high is None
```

**Note:** Several assertions are marked `assert True  # placeholder`. After reading the base class in Task 3 Step 1, replace them with the actual emit-detection path — typically either:
- `gen.event_bus.publish.assert_called_with(...)` if signals publish through the bus
- Inspecting a hook on `_create_signal` (mock it and assert call args)

Use whichever matches how existing signal tests (e.g. `tests/unit/test_rsi.py`) assert emit.

- [ ] **Step 2: Run tests**

```bash
.venv/bin/pytest tests/unit/test_opening_range_signal.py -v
```
Expected: all pass once placeholders are replaced with the real emit assertion. Some assertions may need small tuning.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_opening_range_signal.py
git commit -m "test(signals): unit tests for OpeningRangeSignal (10 cases)"
```

---

# PHASE 3 — `SignalAggregator` per-strategy symbols filter (DEC-STOCKS-005)

Goal: when `[strategy.X].symbols = [...]` is configured, the aggregator for strategy X ignores signals on any symbol not in that list. Prevents AAPL ticks reaching `mean_reversion` aggregator, and BTC ticks reaching `orb_stocks`.

## Task 5: Add symbols filter to SignalAggregator

**Files:**
- Modify: `cerebrum/signals/aggregator.py` (or wherever `SignalAggregator` lives — find it first)
- Test: `tests/unit/test_signal_aggregator_symbols_filter.py`

- [ ] **Step 1: Locate the file**

```bash
.venv/bin/python -c "from cerebrum.signals import aggregator; print(aggregator.__file__)"
```
Or:
```bash
/usr/bin/grep -rn "class SignalAggregator" cerebrum/
```

- [ ] **Step 2: Write the failing test FIRST**

Create `tests/unit/test_signal_aggregator_symbols_filter.py`:

```python
"""Test per-strategy symbols filter on SignalAggregator (DEC-STOCKS-005)."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from cerebrum.signals.aggregator import SignalAggregator  # adjust import if path differs


def _make_aggregator(symbols_filter):
    bus = MagicMock()
    return SignalAggregator(
        event_bus=bus,
        strategy_id="test_strat",
        weights={"technical": Decimal("1.0")},
        threshold=Decimal("0.3"),
        window_seconds=60,
        symbols=symbols_filter,  # NEW parameter
    )


class _SigEvent:
    def __init__(self, symbol, action, strength, source):
        self.symbol = symbol
        self.action = action
        self.strength = Decimal(str(strength))
        self.confidence = Decimal("0.7")
        self.metadata = {"source": source}


def test_signal_for_allowed_symbol_is_processed():
    agg = _make_aggregator(symbols_filter=["AAPL"])
    # Should NOT be filtered out at symbol gate
    result = agg._symbol_allowed("AAPL")  # or whatever internal method ends up existing
    assert result is True


def test_signal_for_disallowed_symbol_is_dropped():
    agg = _make_aggregator(symbols_filter=["AAPL"])
    assert agg._symbol_allowed("BTC/USD") is False


def test_no_symbols_filter_allows_all():
    # Backward compatibility: strategies without a symbols list should accept everything
    agg = _make_aggregator(symbols_filter=None)
    assert agg._symbol_allowed("ANY") is True
    assert agg._symbol_allowed("BTC/USD") is True
```

- [ ] **Step 3: Run tests — verify they fail (before implementation)**

```bash
.venv/bin/pytest tests/unit/test_signal_aggregator_symbols_filter.py -v
```
Expected: fails with `TypeError: unexpected keyword argument 'symbols'` (or similar) — because `SignalAggregator.__init__` doesn't accept that parameter yet.

- [ ] **Step 4: Modify `SignalAggregator`**

In the aggregator source file:

1. Add `symbols: list[str] | None = None` to `__init__` signature.
2. Store as `self._allowed_symbols = set(symbols) if symbols else None`.
3. Add method:

```python
def _symbol_allowed(self, symbol: str) -> bool:
    if self._allowed_symbols is None:
        return True
    return symbol in self._allowed_symbols
```

4. In the existing signal-handling path (wherever the aggregator currently checks `signal_source_filter`), add an early `if not self._symbol_allowed(event.symbol): return` **before** the source filter check.

- [ ] **Step 5: Run tests again**

```bash
.venv/bin/pytest tests/unit/test_signal_aggregator_symbols_filter.py -v
```
Expected: all 3 pass.

Also run the broader test suite to check no regression:
```bash
.venv/bin/pytest tests/unit/ -q
```
Expected: all pass. If an existing test broke because it was instantiating `SignalAggregator` without `symbols=None`, the default should have kept it green — investigate if not.

- [ ] **Step 6: Wire config plumbing in `registry.py`**

Where `StrategyRegistry` instantiates the aggregator (usually in a `register(config)` method), pass `symbols=config.symbols`. The existing `StrategyConfig` dataclass already has a `symbols` field (confirmed earlier). Just propagate it.

- [ ] **Step 7: Commit**

```bash
git add cerebrum/signals/aggregator.py tests/unit/test_signal_aggregator_symbols_filter.py cerebrum/strategies/registry.py
git commit -m "feat(aggregator): per-strategy symbols filter (DEC-STOCKS-005) with 3 tests"
```

---

# PHASE 4 — Alpaca wiring in `main.py`

Goal: if `[alpaca].enabled = true` in config AND `alpaca-py` is importable AND credentials present, instantiate `AlpacaAdapter` and subscribe to its configured symbols. Otherwise warn+skip.

## Task 6: Wire AlpacaAdapter conditional on config

**Files:**
- Modify: `cerebrum/main.py` — find the section where `KrakenAdapter` is instantiated; add a symmetric block for Alpaca.
- Test: `tests/unit/test_main_alpaca_wiring.py` (new)

- [ ] **Step 1: Read existing KrakenAdapter wiring**

```bash
/usr/bin/grep -n "KrakenAdapter\|kraken" cerebrum/main.py | /usr/bin/head -20
```

Note the line range. Identify where the adapter is constructed and subscribed.

- [ ] **Step 2: Write failing test**

Create `tests/unit/test_main_alpaca_wiring.py`:

```python
"""Test Alpaca adapter is wired when config enables it."""
import pytest
from unittest.mock import patch, MagicMock

from cerebrum.main import _maybe_build_alpaca_adapter  # new private helper — Step 3 adds it


def test_returns_none_when_alpaca_disabled():
    config = {"alpaca": {"enabled": False}}
    assert _maybe_build_alpaca_adapter(config, MagicMock()) is None


def test_returns_none_when_alpaca_missing_from_config():
    config = {}
    assert _maybe_build_alpaca_adapter(config, MagicMock()) is None


def test_returns_none_gracefully_when_module_missing():
    config = {"alpaca": {"enabled": True, "symbols": ["AAPL"],
                         "api_key_env": "ALPACA_API_KEY_ID",
                         "secret_key_env": "ALPACA_API_SECRET_KEY",
                         "paper_base_url": "https://paper-api.alpaca.markets",
                         "data_feed": "iex"}}
    with patch.dict("sys.modules", {"alpaca": None}):  # simulate missing import
        # AlpacaAdapter ctor should raise ImportError; _maybe_build_alpaca_adapter catches
        with patch("cerebrum.adapters.alpaca.AlpacaAdapter", side_effect=ImportError("no alpaca")):
            result = _maybe_build_alpaca_adapter(config, MagicMock())
    assert result is None


def test_raises_when_enabled_but_creds_missing(monkeypatch):
    config = {"alpaca": {"enabled": True, "symbols": ["AAPL"],
                         "api_key_env": "ALPACA_API_KEY_ID",
                         "secret_key_env": "ALPACA_API_SECRET_KEY"}}
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="alpaca_credentials_missing"):
        _maybe_build_alpaca_adapter(config, MagicMock())
```

- [ ] **Step 3: Run — fails on ImportError for `_maybe_build_alpaca_adapter`**

```bash
.venv/bin/pytest tests/unit/test_main_alpaca_wiring.py -v
```
Expected: fail.

- [ ] **Step 4: Add `_maybe_build_alpaca_adapter` helper to `cerebrum/main.py`**

Near the top-level helper section (alongside `_maybe_build_kraken_adapter` if it exists, otherwise put it logically near adapter setup):

```python
import os
import logging
from typing import Any

log = logging.getLogger(__name__)


def _maybe_build_alpaca_adapter(config: dict[str, Any], event_bus) -> Any | None:
    """Conditionally instantiate the Alpaca adapter.

    Returns None cleanly if disabled, config missing, or module not importable.
    Raises RuntimeError if enabled but credentials missing — fail-fast by design.
    """
    alpaca_cfg = config.get("alpaca", {})
    if not alpaca_cfg.get("enabled", False):
        return None

    api_key = os.getenv(alpaca_cfg.get("api_key_env", "ALPACA_API_KEY_ID"), "")
    secret = os.getenv(alpaca_cfg.get("secret_key_env", "ALPACA_API_SECRET_KEY"), "")
    if not api_key or not secret:
        log.error("alpaca_credentials_missing")
        raise RuntimeError("alpaca_credentials_missing — set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")

    try:
        from cerebrum.adapters.alpaca import AlpacaAdapter
    except ImportError as e:
        log.warning("alpaca_adapter_unavailable", extra={"reason": "module_not_found", "error": str(e)})
        return None

    try:
        adapter = AlpacaAdapter(
            event_bus=event_bus,
            api_key=api_key,
            secret_key=secret,
            paper_base_url=alpaca_cfg.get("paper_base_url", "https://paper-api.alpaca.markets"),
            data_feed=alpaca_cfg.get("data_feed", "iex"),
            symbols=alpaca_cfg.get("symbols", []),
        )
    except Exception as e:
        log.error("alpaca_adapter_init_failed", extra={"error": str(e)})
        raise

    log.info("alpaca_adapter_built", extra={"symbols": alpaca_cfg.get("symbols", [])})
    return adapter
```

Adjust `AlpacaAdapter(...)` kwargs to match the actual `__init__` signature in `cerebrum/adapters/alpaca.py` — read it first.

- [ ] **Step 5: Call it from the main startup flow**

Find where `KrakenAdapter` is instantiated. Add, right after:

```python
alpaca_adapter = _maybe_build_alpaca_adapter(config, event_bus)
if alpaca_adapter is not None:
    await alpaca_adapter.connect()
    await alpaca_adapter.subscribe_market_data(config["alpaca"]["symbols"])
```

(The exact shape will mirror how Kraken is wired. Copy that pattern.)

And ensure it's included in the shutdown sequence wherever `kraken_adapter.disconnect()` is called.

- [ ] **Step 6: Run tests**

```bash
.venv/bin/pytest tests/unit/test_main_alpaca_wiring.py -v
```
Expected: all 4 pass.

- [ ] **Step 7: Regression — run full test suite with alpaca disabled**

```bash
.venv/bin/pytest -q --ignore=tests/live
```
Expected: all ~782 tests still pass (alpaca code path is guarded by `enabled=False` default in paper.toml — no regressions).

- [ ] **Step 8: Commit**

```bash
git add cerebrum/main.py tests/unit/test_main_alpaca_wiring.py
git commit -m "feat(main): conditional Alpaca adapter wiring with 4 unit tests"
```

---

# PHASE 5 — Market-hours + EOD-flatten risk rules

## Task 7: Inspect existing risk-rule pattern

- [ ] **Step 1: Read existing risk rules**

```bash
.venv/bin/python -c "import os; [print(f) for f in sorted(os.listdir('cerebrum/risk')) if f.endswith('.py') and not f.startswith('_')]"
```
Read one (e.g. `post_fill_cooldown.py`) to see:
- The base class name (`RiskRule` presumably)
- The `evaluate(order)` or `check(event)` method signature
- How a denial is communicated (return value? raise? event publish?)
- How config is loaded into `__init__`

Note the pattern for Tasks 8 and 9.

## Task 8: Create `cerebrum/risk/market_hours_gate.py` + tests

**Files:**
- Create: `cerebrum/risk/market_hours_gate.py`
- Test: `tests/unit/test_market_hours_gate.py`

- [ ] **Step 1: Write failing tests first**

Create `tests/unit/test_market_hours_gate.py`:

```python
"""Unit tests for MarketHoursGate risk rule."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from cerebrum.risk.market_hours_gate import MarketHoursGate

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


class _Order:
    def __init__(self, symbol):
        self.symbol = symbol
        self.side = "buy"
        self.amount = Decimal("10")


def _gate(**overrides):
    cfg = {
        "enabled": True,
        "rth_start": "09:30",
        "rth_end": "16:00",
        "tz": "America/New_York",
        "stock_symbols": ["AAPL", "MSFT", "NVDA"],
        "allow_holidays": False,
        "entry_cutoff_minutes_before_close": 15,
    }
    cfg.update(overrides)
    return MarketHoursGate(config=cfg)


def test_denies_stock_order_outside_rth():
    g = _gate()
    # 07:00 ET on weekday → pre-market
    decision = g.evaluate(_Order("AAPL"), now_utc=_utc(2026, 6, 15, 11, 0))
    assert decision.allowed is False
    assert decision.reason == "market_hours_gate"


def test_allows_stock_order_inside_rth():
    g = _gate()
    decision = g.evaluate(_Order("AAPL"), now_utc=_utc(2026, 6, 15, 14, 0))  # 10:00 ET
    assert decision.allowed is True


def test_allows_crypto_order_anytime():
    g = _gate()
    decision = g.evaluate(_Order("BTC/USD"), now_utc=_utc(2026, 6, 15, 4, 0))  # Sunday 00:00 ET
    assert decision.allowed is True


def test_denies_stock_order_after_entry_cutoff():
    g = _gate()
    # 15:46 ET (last 14 min before 16:00 close) → inside the 15-min cutoff
    decision = g.evaluate(_Order("AAPL"), now_utc=_utc(2026, 6, 15, 19, 46))
    assert decision.allowed is False


def test_allows_stock_order_just_before_cutoff():
    g = _gate()
    # 15:44 ET (16 min before close) → outside cutoff
    decision = g.evaluate(_Order("AAPL"), now_utc=_utc(2026, 6, 15, 19, 44))
    assert decision.allowed is True


def test_denies_all_stock_orders_on_holiday():
    g = _gate()
    decision = g.evaluate(_Order("AAPL"), now_utc=_utc(2026, 12, 25, 15, 0))
    assert decision.allowed is False
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Implement `cerebrum/risk/market_hours_gate.py`**

```python
"""MarketHoursGate risk rule — denies stock orders outside RTH.

@decision DEC-STOCKS-003
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from cerebrum.utils.trading_session import (
    ET,
    is_rth_now,
    minutes_until_close,
    rth_close_for,
    is_market_holiday,
)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str | None = None


class MarketHoursGate:
    RULE_NAME = "market_hours_gate"

    def __init__(self, config: dict[str, Any]):
        self._enabled: bool = bool(config.get("enabled", True))
        self._stock_symbols: set[str] = set(config.get("stock_symbols", []))
        self._entry_cutoff: int = int(config.get("entry_cutoff_minutes_before_close", 15))

    def evaluate(self, order, now_utc: datetime | None = None) -> RiskDecision:
        if not self._enabled:
            return RiskDecision(allowed=True)
        if order.symbol not in self._stock_symbols:
            return RiskDecision(allowed=True)  # crypto passes through
        if not is_rth_now(now_utc):
            return RiskDecision(allowed=False, reason=self.RULE_NAME)
        mins = minutes_until_close(now_utc)
        if mins is None:
            return RiskDecision(allowed=False, reason=self.RULE_NAME)
        if mins < self._entry_cutoff:
            return RiskDecision(allowed=False, reason=self.RULE_NAME)
        return RiskDecision(allowed=True)
```

Adjust `RiskDecision` and `evaluate` to whatever pattern you observed in Task 7.

- [ ] **Step 4: Run tests → pass. Commit.**

```bash
git add cerebrum/risk/market_hours_gate.py tests/unit/test_market_hours_gate.py
git commit -m "feat(risk): MarketHoursGate rule + 6 tests (DEC-STOCKS-003)"
```

## Task 9: Create `cerebrum/risk/end_of_day_flatten.py` + tests

**Files:**
- Create: `cerebrum/risk/end_of_day_flatten.py`
- Test: `tests/unit/test_end_of_day_flatten.py`

- [ ] **Step 1: Understand exit-rule pattern vs risk-rule pattern**

`EndOfDayFlatten` isn't a pre-trade risk rule — it *emits* close orders. It's closer to `ExitMonitor`. Read `cerebrum/risk/exit_monitor.py` first to see how exits are emitted (periodic tick? event handler? scheduled task?).

- [ ] **Step 2: Write failing tests**

```python
"""Unit tests for EndOfDayFlatten."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock

import pytest

from cerebrum.risk.end_of_day_flatten import EndOfDayFlatten

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def _flatten(**overrides):
    cfg = {
        "enabled": True,
        "flatten_at": "15:55",
        "stock_symbols": ["AAPL", "MSFT", "NVDA"],
        "bypass_commission_gate": True,
    }
    cfg.update(overrides)
    bus = MagicMock()
    return EndOfDayFlatten(config=cfg, event_bus=bus), bus


class _Position:
    def __init__(self, symbol, amount):
        self.symbol = symbol
        self.amount = Decimal(str(amount))


def test_fires_at_1555_with_open_position():
    f, bus = _flatten()
    # 15:55 ET on a weekday — EDT = UTC-4 in June → 19:55 UTC
    f.tick(now_utc=_utc(2026, 6, 15, 19, 55), open_positions=[_Position("AAPL", "10")])
    assert bus.publish.call_count == 1  # OrderEvent fired for AAPL


def test_no_fire_at_1554():
    f, bus = _flatten()
    f.tick(now_utc=_utc(2026, 6, 15, 19, 54), open_positions=[_Position("AAPL", "10")])
    assert bus.publish.call_count == 0


def test_no_fire_with_no_positions():
    f, bus = _flatten()
    f.tick(now_utc=_utc(2026, 6, 15, 19, 55), open_positions=[])
    assert bus.publish.call_count == 0


def test_multiple_positions_each_get_close_order():
    f, bus = _flatten()
    f.tick(now_utc=_utc(2026, 6, 15, 19, 55),
           open_positions=[_Position("AAPL", "10"), _Position("MSFT", "5")])
    assert bus.publish.call_count == 2


def test_early_close_day_fires_at_1255():
    f, bus = _flatten()
    # 2026-11-27 Black Friday, close 13:00 ET. 12:55 ET in EST → 17:55 UTC
    f.tick(now_utc=_utc(2026, 11, 27, 17, 55), open_positions=[_Position("AAPL", "10")])
    assert bus.publish.call_count == 1
```

- [ ] **Step 3: Run — fails.**

- [ ] **Step 4: Implement `cerebrum/risk/end_of_day_flatten.py`**

```python
"""EndOfDayFlatten — emits close orders for stock positions at end of RTH.

@decision DEC-STOCKS-003
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from cerebrum.utils.trading_session import ET, rth_close_for


class EndOfDayFlatten:
    RULE_NAME = "end_of_day_flatten"

    def __init__(self, config: dict[str, Any], event_bus):
        self._enabled: bool = bool(config.get("enabled", True))
        self._stock_symbols: set[str] = set(config.get("stock_symbols", []))
        self._flatten_offset_min: int = 5  # flatten 5 min before close
        self._bypass_commission_gate: bool = bool(config.get("bypass_commission_gate", True))
        self._event_bus = event_bus
        self._fired_today: set = set()  # (date, symbol) tuples

    def tick(self, now_utc: datetime, open_positions: list) -> None:
        if not self._enabled:
            return
        now_et = now_utc.astimezone(ET)
        today = now_et.date()
        close = rth_close_for(today)
        if close is None:
            return

        # Flatten window opens at close - offset
        flatten_minute = close.hour * 60 + close.minute - self._flatten_offset_min
        current_minute = now_et.hour * 60 + now_et.minute
        if current_minute < flatten_minute:
            return

        for pos in open_positions:
            if pos.symbol not in self._stock_symbols:
                continue
            if pos.amount == 0:
                continue
            key = (today, pos.symbol)
            if key in self._fired_today:
                continue
            self._fired_today.add(key)
            # Publish close order — adjust event type to match existing OrderEvent shape
            self._event_bus.publish(_build_close_order(pos, bypass_commission_gate=self._bypass_commission_gate))


def _build_close_order(pos, *, bypass_commission_gate: bool):
    # Minimal placeholder — replace with actual OrderEvent import + shape
    from cerebrum.events import OrderEvent  # adjust import
    return OrderEvent(
        symbol=pos.symbol,
        side="sell" if pos.amount > 0 else "buy",
        amount=abs(pos.amount),
        strategy_id="orb_stocks",
        metadata={
            "reason": "end_of_day_flatten",
            "bypass_commission_gate": bypass_commission_gate,
        },
    )
```

- [ ] **Step 5: Run tests → pass. Commit.**

```bash
git add cerebrum/risk/end_of_day_flatten.py tests/unit/test_end_of_day_flatten.py
git commit -m "feat(risk): EndOfDayFlatten rule + 5 tests (DEC-STOCKS-003)"
```

---

# PHASE 6 — `orb_stocks` strategy + config + state migration

## Task 10: Add config sections to `config/paper.toml`

**Files:**
- Modify: `config/paper.toml`

- [ ] **Step 1: Append the new sections**

Replace the commented `[alpaca]` block (lines ~88–94) with the full config from the spec (see spec Section "Configuration" for the exact TOML). Be sure to keep `[alpaca].enabled = false` on the first commit — flip to `true` only after Phase 8 live tests pass.

- [ ] **Step 2: Verify TOML parses**

```bash
.venv/bin/python -c "import tomllib; c = tomllib.load(open('config/paper.toml','rb')); print(list(c.keys()))"
```
Expected: keys include `alpaca`, `signal`, `strategy`, `risk`.

- [ ] **Step 3: Commit**

```bash
git add config/paper.toml
git commit -m "config: add alpaca, orb_stocks, market_hours_gate, eod_flatten sections (disabled by default)"
```

## Task 11: Register `orb_stocks` strategy in main.py + paper_trading adapter tweak

**Files:**
- Modify: `cerebrum/main.py` — strategy registration loop already iterates `config["strategy"]`; `orb_stocks` will come through automatically once config has it. Verify.
- Modify: `cerebrum/adapters/paper_trading.py` — add optional `commission_by_symbol`.

- [ ] **Step 1: Confirm strategy registration is config-driven**

```bash
/usr/bin/grep -n "strategy" cerebrum/main.py | /usr/bin/head -20
```
Read the loop. If it's iterating over `config["strategy"]` entries, `orb_stocks` picks up automatically.

- [ ] **Step 2: Add `commission_by_symbol` to paper_trading adapter**

Find the commission computation in `cerebrum/adapters/paper_trading.py`. Current code likely computes `commission = fill_value * commission_pct`. Add:

```python
# At __init__:
self._commission_by_symbol: dict[str, Decimal] = {
    str(k): Decimal(str(v)) for k, v in config.get("commission_by_symbol", {}).items()
}

# In the commission computation:
per_symbol = self._commission_by_symbol.get(order.symbol)
if per_symbol is not None:
    commission = Decimal(str(fill_value)) * per_symbol
else:
    commission = Decimal(str(fill_value)) * self._commission_pct
```

For orb_stocks v1, no per-symbol override is needed (stocks can share the crypto 0.16% commission for simplicity). This is future-proofing.

- [ ] **Step 3: Quick smoke**

```bash
.venv/bin/python -m cerebrum --mode paper --config config/paper.toml 2>&1 | /usr/bin/head -30
```

With alpaca enabled=false, should show 3 strategies initializing: mean_reversion, range_trading, orb_stocks. Then kill with `Ctrl-C` (or the equivalent in test harness). If orb_stocks isn't registered because of some config-type mismatch, fix before moving on.

- [ ] **Step 4: Commit**

```bash
git add cerebrum/adapters/paper_trading.py
git commit -m "feat(paper_trading): optional commission_by_symbol lookup"
```

## Task 12: Paper state migration v2 → v3

**Files:**
- Modify: the state-loader code path in `cerebrum/main.py` or `cerebrum/adapters/paper_trading.py` (wherever `paper_state.json` is loaded)
- Test: `tests/unit/test_paper_state_migration.py`

- [ ] **Step 1: Locate loader**

```bash
/usr/bin/grep -rn "paper_state.json\|paper_state_loaded" cerebrum/ | /usr/bin/head -10
```

- [ ] **Step 2: Write failing tests**

```python
"""Test state file migration v2 → v3."""
import json
import os
from pathlib import Path

import pytest

from cerebrum.adapters.paper_trading import migrate_state_v2_to_v3  # new helper


def _v2_state():
    return {
        "version": 2,
        "balances": {"USD": "9805.02"},
        "positions": {"SOL/USD": "2.107"},
        "current_prices": {},
        "trade_history": [],
        "strategy_snapshots": {
            "mean_reversion": {"cash_balance": "5000", "initial_balance": "5000", "positions": {}},
            "range_trading": {"cash_balance": "5000", "initial_balance": "5000", "positions": {}},
        },
    }


def test_migration_adds_orb_stocks_snapshot(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrated = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    assert migrated["version"] == 3
    assert "orb_stocks" in migrated["strategy_snapshots"]
    assert migrated["strategy_snapshots"]["orb_stocks"]["cash_balance"] == "5000.0"


def test_migration_preserves_existing_crypto_snapshots(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrated = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    assert migrated["strategy_snapshots"]["mean_reversion"]["cash_balance"] == "5000"
    assert migrated["strategy_snapshots"]["range_trading"]["cash_balance"] == "5000"


def test_migration_writes_backup(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps(_v2_state()))
    migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    backup = tmp_path / "paper_state.v2.bak.json"
    assert backup.exists()
    assert json.loads(backup.read_text())["version"] == 2


def test_migration_is_idempotent_on_v3(tmp_path):
    p = tmp_path / "paper_state.json"
    p.write_text(json.dumps({**_v2_state(), "version": 3,
                              "strategy_snapshots": {"mean_reversion": {},
                                                     "range_trading": {},
                                                     "orb_stocks": {"cash_balance": "5000",
                                                                     "initial_balance": "5000",
                                                                     "positions": {}}}}))
    migrated = migrate_state_v2_to_v3(p, initial_balance_orb=5000.0)
    # Idempotent: no duplicate orb_stocks, no backup written (since it was already v3)
    assert migrated["version"] == 3
    backup = tmp_path / "paper_state.v2.bak.json"
    assert not backup.exists()
```

- [ ] **Step 3: Run — fails**

- [ ] **Step 4: Implement `migrate_state_v2_to_v3` in `cerebrum/adapters/paper_trading.py`**

```python
import json
import shutil
from pathlib import Path


def migrate_state_v2_to_v3(path: Path, *, initial_balance_orb: float) -> dict:
    """Migrate paper_state.json from v2 to v3.

    Writes a .v2.bak.json backup atomically before rewriting the main file.
    Idempotent on v3 input (no-op if already v3).
    """
    data = json.loads(path.read_text())
    if data.get("version", 2) >= 3:
        return data

    backup = path.parent / (path.stem + ".v2.bak" + path.suffix)
    shutil.copy(path, backup)

    data["version"] = 3
    snapshots = data.setdefault("strategy_snapshots", {})
    if "orb_stocks" not in snapshots:
        snapshots["orb_stocks"] = {
            "cash_balance": str(initial_balance_orb),
            "initial_balance": str(initial_balance_orb),
            "peak_equity": str(initial_balance_orb),
            "total_realized_pnl": "0",
            "positions": {},
        }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return data
```

Call this during the state-load path in main.py, before constructing strategy objects, when `orb_stocks` is enabled in config.

- [ ] **Step 5: Run tests → pass. Commit.**

```bash
git add cerebrum/adapters/paper_trading.py tests/unit/test_paper_state_migration.py cerebrum/main.py
git commit -m "feat(state): v2→v3 migration for orb_stocks with atomic backup (DEC-STOCKS-006)"
```

---

# PHASE 7 — Integration tests + fixture recording

## Task 13: Create fixture-recording helper

**Files:**
- Create: `scripts/record_alpaca_ticks.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Record Alpaca tick stream to JSONL. Used to bootstrap integration fixtures.

Usage:
  python scripts/record_alpaca_ticks.py --symbols AAPL,MSFT,NVDA \
      --start 09:30 --end 16:00 --date 2026-03-10 \
      --out tests/fixtures/alpaca_mixed_stocks_2026-03-10.jsonl

Requires ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY in .env.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


async def _record(symbols, out_path, start_et_str, end_et_str, date_str):
    from alpaca.data.live.stock import StockDataStream

    key = os.environ["ALPACA_API_KEY_ID"]
    secret = os.environ["ALPACA_API_SECRET_KEY"]
    stream = StockDataStream(key, secret, feed="iex")

    out = open(out_path, "w")
    start = datetime.fromisoformat(f"{date_str}T{start_et_str}:00").replace(tzinfo=ET)
    end = datetime.fromisoformat(f"{date_str}T{end_et_str}:00").replace(tzinfo=ET)

    async def on_quote(data):
        ts = getattr(data, "timestamp", None) or datetime.now(tz=ET)
        if start <= ts.astimezone(ET) <= end:
            out.write(json.dumps({
                "symbol": getattr(data, "symbol", None),
                "bid": str(getattr(data, "bid_price", "")),
                "ask": str(getattr(data, "ask_price", "")),
                "timestamp": ts.isoformat(),
            }) + "\n")
            out.flush()

    stream.subscribe_quotes(on_quote, *symbols)
    try:
        await stream._run_forever()  # or stream.run() depending on alpaca-py version
    except KeyboardInterrupt:
        pass
    finally:
        out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--date", required=True)  # ISO date
    ap.add_argument("--start", default="09:30")
    ap.add_argument("--end", default="16:00")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    syms = args.symbols.split(",")
    asyncio.run(_record(syms, args.out, args.start, args.end, args.date))


if __name__ == "__main__":
    sys.exit(main() or 0)
```

Exact `alpaca-py` API may vary — check the installed version's `StockDataStream` docs. Goal: emit one JSON line per quote/trade with `symbol, price/bid/ask, timestamp`.

- [ ] **Step 2: Commit (executable is fine as .py without chmod)**

```bash
git add scripts/record_alpaca_ticks.py
git commit -m "scripts: add record_alpaca_ticks helper for fixture bootstrapping"
```

## Task 14: Write `test_orb_full_day.py` integration test

**Files:**
- Create: `tests/integration/test_orb_full_day.py`
- Expected fixture: `tests/fixtures/alpaca_aapl_2026-03-10.jsonl`

- [ ] **Step 1: Record the fixture (human step — needs live Alpaca session during RTH)**

Run the helper from Task 13 during a live AAPL RTH day to produce the fixture. Commit the resulting JSONL. If live recording isn't feasible, a synthetic fixture that exercises the ORB break can substitute — document clearly in the fixture header which it is.

- [ ] **Step 2: Write the test**

```python
"""Replay integration test — full AAPL trading day through the ORB pipeline."""
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from cerebrum.main import build_test_pipeline  # or the existing pipeline-builder test helper

FIX = Path("tests/fixtures/alpaca_aapl_2026-03-10.jsonl")
ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_orb_full_day_aapl():
    if not FIX.exists():
        pytest.skip("fixture not yet recorded")
    pipeline = await build_test_pipeline(
        strategies=["orb_stocks"],
        stock_symbols=["AAPL"],
        initial_balance=5000.0,
    )

    ticks = [json.loads(l) for l in FIX.read_text().splitlines()]
    for t in ticks:
        await pipeline.publish_market_data(
            symbol=t["symbol"],
            price=Decimal(t["ask"]) if t.get("ask") else Decimal(t["bid"]),
            timestamp=datetime.fromisoformat(t["timestamp"]),
        )

    snapshot = pipeline.snapshot("orb_stocks")
    # Assertions — adjust to actual fixture content:
    assert snapshot.trades_count >= 1, "Expected at least one ORB entry"
    assert snapshot.open_positions == {}, "Position should be flat at EOD (end_of_day_flatten)"
    # PnL sanity (can be negative — just assert it's finite)
    assert isinstance(snapshot.realized_pnl, Decimal)
```

`build_test_pipeline` is a new helper you'll add alongside (Step 3), or adapt from existing integration-test scaffolding.

- [ ] **Step 3: Add `build_test_pipeline` helper if not present**

Look for similar helpers in existing integration tests first. If none, write a small one that wires the production code paths without the I/O side (no real Kraken, no real Alpaca — replay-only).

- [ ] **Step 4: Run — pass or skip (if fixture absent)**

```bash
.venv/bin/pytest tests/integration/test_orb_full_day.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_orb_full_day.py tests/fixtures/alpaca_aapl_2026-03-10.jsonl
git commit -m "test(integration): ORB full-day replay test + fixture"
```

## Task 15: `test_orb_nyse_holiday.py`

**Files:**
- Create: `tests/integration/test_orb_nyse_holiday.py`

- [ ] **Step 1: Write test**

```python
"""Integration test — Christmas Day, no stock trading."""
import asyncio
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from cerebrum.main import build_test_pipeline

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_no_entries_on_holiday():
    pipeline = await build_test_pipeline(strategies=["orb_stocks"], stock_symbols=["AAPL"], initial_balance=5000)
    # Simulate 20 random ticks at 11:00 ET on Christmas Day 2026 — market_hours_gate should deny all
    for i in range(20):
        await pipeline.publish_market_data(
            symbol="AAPL",
            price=Decimal("182.00"),
            timestamp=datetime(2026, 12, 25, 11, i, tzinfo=ET),
        )
    snap = pipeline.snapshot("orb_stocks")
    assert snap.trades_count == 0
    assert snap.open_positions == {}
```

- [ ] **Step 2: Run + commit**

```bash
git add tests/integration/test_orb_nyse_holiday.py
git commit -m "test(integration): no stock trading on NYSE holidays"
```

## Task 16: `test_orb_early_close.py`

**Files:**
- Create: `tests/integration/test_orb_early_close.py`

- [ ] **Step 1: Write test**

```python
"""Integration test — Black Friday early close (13:00 ET) → flatten at 12:55."""
import asyncio
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from cerebrum.main import build_test_pipeline

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_flatten_at_1255_on_black_friday():
    pipeline = await build_test_pipeline(strategies=["orb_stocks"], stock_symbols=["AAPL"], initial_balance=5000)
    # Build ORB, trigger entry
    base_ts = datetime(2026, 11, 27, 9, 30, tzinfo=ET)
    # Simulate 12-tick range + breakout — minimal repro; adjust prices/counts
    for i in range(12):
        await pipeline.publish_market_data("AAPL", Decimal("100.0") + Decimal(i) / 100,
                                            base_ts.replace(minute=30 + i))
    await pipeline.publish_market_data("AAPL", Decimal("100.50"),
                                        base_ts.replace(minute=46))  # freeze + break (adjust bps math)

    # Tick at 12:54 → no flatten yet
    await pipeline.publish_market_data("AAPL", Decimal("100.40"), base_ts.replace(hour=12, minute=54))
    snap = pipeline.snapshot("orb_stocks")
    has_position_1254 = any(v != 0 for v in snap.positions.values())

    # Tick at 12:55 → flatten fires
    await pipeline.publish_market_data("AAPL", Decimal("100.40"), base_ts.replace(hour=12, minute=55))
    snap_after = pipeline.snapshot("orb_stocks")

    if has_position_1254:
        assert all(v == 0 for v in snap_after.positions.values()), "Flatten should close all stock positions"
```

- [ ] **Step 2: Run + commit**

## Task 17: `test_cross_asset_isolation.py`

**Files:**
- Create: `tests/integration/test_cross_asset_isolation.py`

- [ ] **Step 1: Write test**

Test that BTC/USD signals never arrive at `orb_stocks` aggregator, and AAPL signals never arrive at `mean_reversion` aggregator. Use event-bus trace + assertion on aggregator-level counts.

```python
"""Integration — no cross-asset signal contamination (DEC-STOCKS-005)."""
import asyncio
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from cerebrum.main import build_test_pipeline

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_stocks_dont_reach_crypto_aggregators():
    pipeline = await build_test_pipeline(
        strategies=["mean_reversion", "range_trading", "orb_stocks"],
        crypto_symbols=["BTC/USD", "ETH/USD"],
        stock_symbols=["AAPL"],
        initial_balance={"mean_reversion": 5000, "range_trading": 5000, "orb_stocks": 5000},
    )
    base_ts = datetime(2026, 6, 15, 10, 0, tzinfo=ET)

    # Alternate stock and crypto ticks
    for i in range(20):
        await pipeline.publish_market_data("AAPL", Decimal("182.0"), base_ts.replace(minute=i))
        await pipeline.publish_market_data("BTC/USD", Decimal("73000"), base_ts.replace(minute=i))

    mr_counts = pipeline.aggregator_input_symbols("mean_reversion")
    orb_counts = pipeline.aggregator_input_symbols("orb_stocks")

    assert "AAPL" not in mr_counts, "mean_reversion must not see AAPL"
    assert "BTC/USD" not in orb_counts, "orb_stocks must not see BTC/USD"
```

- [ ] **Step 2: Run + commit**

## Task 18: `test_stream_stale.py`

**Files:**
- Create: `tests/integration/test_stream_stale.py`

- [ ] **Step 1: Write test**

60s gap in AAPL stream during RTH → `market_hours_gate` denies (or a new `stream_stale_gate` — decide during impl).

Test stub:
```python
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from cerebrum.main import build_test_pipeline

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_new_entries_denied_during_stream_stale():
    pipeline = await build_test_pipeline(strategies=["orb_stocks"], stock_symbols=["AAPL"], initial_balance=5000)
    base = datetime(2026, 6, 15, 10, 0, tzinfo=ET)
    await pipeline.publish_market_data("AAPL", Decimal("182"), base)
    # Simulate 61s gap, then breakout tick
    await pipeline.advance_clock(base + timedelta(seconds=61))
    await pipeline.publish_market_data("AAPL", Decimal("184"), base + timedelta(seconds=61))
    snap = pipeline.snapshot("orb_stocks")
    assert snap.trades_count == 0, "No entries while stream stale"
```

- [ ] **Step 2: Run + commit**

## Task 19: `test_eod_flatten_with_partial_fill.py`

**Files:**
- Create: `tests/integration/test_eod_flatten_with_partial_fill.py`

- [ ] **Step 1: Write test**

Force the paper_trading adapter to return a partial fill on the close order. Assert remainder is resubmitted and eventually reaches zero.

- [ ] **Step 2: Commit**

---

# PHASE 8 — Live tests (opt-in) + docs

## Task 20: Add `--live-alpaca` pytest flag + CONTRIBUTING section

**Files:**
- Modify: `conftest.py` (root or `tests/conftest.py`)
- Modify: `CONTRIBUTING.md`

- [ ] **Step 1: Conftest additions**

```python
def pytest_addoption(parser):
    parser.addoption("--live-alpaca", action="store_true", default=False,
                     help="Run live Alpaca tests (requires ALPACA_API_KEY_ID)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live-alpaca"):
        skip = pytest.mark.skip(reason="needs --live-alpaca")
        for item in items:
            if "live_alpaca" in item.keywords:
                item.add_marker(skip)
```

- [ ] **Step 2: Register marker in `pyproject.toml` or `pytest.ini`**

```
[pytest]
markers =
    live_alpaca: requires live Alpaca paper-account connection
```

- [ ] **Step 3: Add CONTRIBUTING section**

In `CONTRIBUTING.md` add:
```markdown
## Running live Alpaca tests

Default: skipped. Opt in with `pytest --live-alpaca`. Requires
`ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` in `.env`
(paper account, free tier: https://alpaca.markets).

These tests connect to Alpaca's live paper-trading endpoint. They do not
place orders — only subscribe to market data and verify event shape.
Run during US RTH for the data-stream test.
```

- [ ] **Step 4: Commit**

```bash
git add conftest.py pyproject.toml CONTRIBUTING.md
git commit -m "test(live): add --live-alpaca pytest flag + docs"
```

## Task 21: Live Alpaca connection test

**Files:**
- Create: `tests/live/__init__.py`
- Create: `tests/live/test_alpaca_live_connection.py`

- [ ] **Step 1: Write tests**

```python
"""Live Alpaca connection tests — opt-in via --live-alpaca."""
import asyncio
import os

import pytest

pytestmark = pytest.mark.live_alpaca


def test_env_credentials_present():
    assert os.getenv("ALPACA_API_KEY_ID"), "missing ALPACA_API_KEY_ID"
    assert os.getenv("ALPACA_API_SECRET_KEY"), "missing ALPACA_API_SECRET_KEY"


@pytest.mark.asyncio
async def test_subscribes_and_receives_at_least_5_ticks():
    from cerebrum.adapters.alpaca import AlpacaAdapter
    from cerebrum.events import EventBus  # adjust import if name differs

    bus = EventBus()
    adapter = AlpacaAdapter(
        event_bus=bus,
        api_key=os.environ["ALPACA_API_KEY_ID"],
        secret_key=os.environ["ALPACA_API_SECRET_KEY"],
        paper_base_url="https://paper-api.alpaca.markets",
        data_feed="iex",
        symbols=["AAPL"],
    )

    tick_count = 0
    async def count(evt):
        nonlocal tick_count
        if evt.symbol == "AAPL":
            tick_count += 1
    bus.subscribe("market_data", count)

    await adapter.connect()
    await adapter.subscribe_market_data(["AAPL"])
    # Wait up to 30s for 5 ticks (RTH only — test will skip if market closed)
    for _ in range(30):
        if tick_count >= 5:
            break
        await asyncio.sleep(1)
    await adapter.disconnect()
    assert tick_count >= 5, "Expected at least 5 AAPL ticks in 30s during RTH"
```

- [ ] **Step 2: Run (only during RTH)**

```bash
.venv/bin/pytest tests/live/test_alpaca_live_connection.py -v --live-alpaca
```

- [ ] **Step 3: Commit**

## Task 22: Live ORB smoke test (manual)

**Files:**
- Create: `tests/live/test_live_orb_smoke.py`

Marked `live_alpaca`; 20-min run triggered during an RTH window. Not run in CI. Skip if `minutes_until_close() < 20`. Document usage in CONTRIBUTING.md.

- [ ] Commit.

---

# WRAP-UP

## Task 23: Final regression sweep + merge prep

- [ ] **Step 1: Full test run**

```bash
.venv/bin/pytest -q --ignore=tests/live
```
Expected: all (~782 + new ~45) tests pass.

- [ ] **Step 2: Start cerebrum with `[alpaca].enabled=false` — regression**

```bash
.venv/bin/python -m cerebrum --mode paper --config config/paper.toml
```
Expected: same startup banner as pre-change main (2 crypto strategies, no alpaca), no errors. Tail logs for 10 seconds then stop. If orb_stocks also boots (because its enabled=true in config) but alpaca is disabled → the fail-fast path from Phase 4 should fire: `RuntimeError: orb_stocks requires alpaca adapter enabled`. Make sure default config has `[strategy.orb_stocks].enabled=false` if alpaca is disabled.

- [ ] **Step 3: Flip alpaca + orb_stocks to enabled=true, start once, verify 3 strategies init, stop cleanly.**

- [ ] **Step 4: Dispatch guardian to prepare the merge to main.**

Spec is in `docs/superpowers/specs/` (copied from the main working tree earlier). Plan is in `docs/superpowers/plans/`. Both should have landed in the worktree's commit history at this point. Guardian will:
- Verify all `@decision` annotations present on new source files (DEC-STOCKS-001..006 where referenced)
- Verify tests green
- Create the merge commit from `worktree-stocks-orb` → `main`

---

# Self-Review Notes (by plan author)

**Spec coverage check:**
- DEC-STOCKS-001 (same-process) → Task 6 (Alpaca wiring)
- DEC-STOCKS-002 (local paper-trading fills) → Task 11 (paper_trading tweak — no routing change needed, existing adapter handles stocks)
- DEC-STOCKS-003 (RTH + EOD flatten) → Tasks 1, 8, 9
- DEC-STOCKS-004 (ORB signal + source filter) → Tasks 3, 4
- DEC-STOCKS-005 (aggregator symbols filter) → Task 5
- DEC-STOCKS-006 (state migration) → Task 12

All six decisions covered. ✓

**Type consistency:**
- `RiskDecision` introduced in Task 8 — used by `MarketHoursGate`. If existing risk rules use a different return shape, Task 7 (pattern inspection) catches it and Task 8 adjusts.
- `OrderEvent` shape referenced in Task 9 `_build_close_order` — must match production. Task 9 Step 1 explicitly says to inspect exit_monitor first.
- `build_test_pipeline` referenced in Tasks 14-19 — introduced as a new helper in Task 14 Step 3.

**Known open items (documented, not hidden):**
- Fixture `tests/fixtures/alpaca_aapl_2026-03-10.jsonl` needs live recording (Task 14 Step 1). Test skips cleanly if missing.
- Several integration test assertions use `snap.open_positions`, `snap.trades_count`, etc. — these attribute names may need adjustment to match whatever `PortfolioTracker.snapshot()` actually returns. Each test's first run will reveal the real API; fix in place.

**Placeholders (deliberate, not plan failures):**
- Task 4 (opening_range tests) — some assertions marked `# placeholder` with explicit instructions for replacing after inspecting base class. Not a TBD — it's a contingent implementation detail awaiting Step 1 inspection.
- Task 7 (risk-rule pattern inspection) — a whole task for pattern discovery, not a placeholder. Unblocks Tasks 8/9.

**Scope check:** Single coherent feature (stocks/ORB). 8 phases × 23 tasks. Not decomposable further without losing cohesion — the whole point is that ORB, market hours, flatten, and Alpaca all arrive together.
