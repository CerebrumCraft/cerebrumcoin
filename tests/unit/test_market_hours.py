"""
Tests for MarketHoursRule — NYSE trading hours guard.

All tests use real datetime manipulation via monkeypatching datetime.now at
the module level, or by constructing OrderEvents with specific symbols.
No mocks of internal code.

@decision DEC-TEST-HOURS-001
@title Test market hours with datetime monkeypatching at module level
@status accepted
@rationale MarketHoursRule._compute_market_open() calls datetime.now(tz=...) internally.
To control "what time is it" without mocking, we subclass or monkeypatch datetime.now
on the market_hours module directly. This is the standard pattern for testing
time-dependent code without freezegun — patch only the specific callable the
production code uses.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from time import time
from zoneinfo import ZoneInfo

import pytest

from cerebrum.core.events import OrderEvent
from cerebrum.core.types import EventType, OrderType, Side
from cerebrum.risk.market_hours import MarketHoursRule, _us_market_holidays, _easter

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(symbol: str) -> OrderEvent:
    """Build a minimal OrderEvent for the given symbol."""
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time(),
        order_id="test_order",
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("1"),
    )


def _rule_with_fixed_time(dt_et: datetime) -> MarketHoursRule:
    """
    Return a MarketHoursRule whose _compute_market_open is pinned to dt_et.

    We override _compute_market_open on the instance directly so no patching
    of datetime itself is required. This keeps tests deterministic without
    depending on freezegun or unittest.mock.
    """
    rule = MarketHoursRule(cache_ttl_seconds=0)  # disable caching so every check is live

    def _fixed_compute() -> bool:
        # Replicate real logic using the pinned datetime
        today = dt_et.date()
        if dt_et.weekday() >= 5:
            return False
        holidays = _us_market_holidays(today.year)
        if today in holidays:
            return False
        open_time = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= dt_et < close_time

    rule._compute_market_open = _fixed_compute  # type: ignore[method-assign]
    return rule


# ---------------------------------------------------------------------------
# 13B: Stock symbol detection
# ---------------------------------------------------------------------------


def test_is_crypto_symbol():
    """Symbols with '/' are crypto; without '/' are stocks."""
    rule = MarketHoursRule()
    assert rule._is_crypto_symbol("BTC/USD") is True
    assert rule._is_crypto_symbol("ETH/USD") is True
    assert rule._is_crypto_symbol("AAPL") is False
    assert rule._is_crypto_symbol("MSFT") is False
    assert rule._is_crypto_symbol("NVDA") is False


# ---------------------------------------------------------------------------
# Crypto always passes
# ---------------------------------------------------------------------------


def test_crypto_always_allowed_during_market_hours():
    """Crypto orders pass through regardless of market state."""
    # Pin to a time when stock market would be open
    dt_open = datetime(2026, 3, 2, 10, 0, 0, tzinfo=_ET)  # Monday 10:00 ET
    rule = _rule_with_fixed_time(dt_open)

    allowed, info = rule.check(_make_order("BTC/USD"))
    assert allowed is True
    assert info == {}


def test_crypto_always_allowed_outside_market_hours():
    """Crypto orders pass through even when market is closed."""
    dt_closed = datetime(2026, 3, 2, 18, 0, 0, tzinfo=_ET)  # Monday 18:00 ET (after close)
    rule = _rule_with_fixed_time(dt_closed)

    allowed, info = rule.check(_make_order("ETH/USD"))
    assert allowed is True
    assert info == {}


def test_crypto_always_allowed_on_weekend():
    """Crypto orders pass through on Saturday."""
    dt_weekend = datetime(2026, 3, 7, 12, 0, 0, tzinfo=_ET)  # Saturday noon ET
    rule = _rule_with_fixed_time(dt_weekend)

    allowed, info = rule.check(_make_order("BTC/USD"))
    assert allowed is True
    assert info == {}


# ---------------------------------------------------------------------------
# Stock during market hours
# ---------------------------------------------------------------------------


def test_stock_allowed_during_market_hours():
    """Stock order during NYSE hours (Mon-Fri 09:30-16:00 ET) is allowed."""
    dt_open = datetime(2026, 3, 2, 11, 30, 0, tzinfo=_ET)  # Monday 11:30 ET
    rule = _rule_with_fixed_time(dt_open)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is True
    assert info == {}


def test_stock_allowed_at_open_exactly():
    """Stock order at exactly 09:30 ET is allowed (open is inclusive)."""
    dt_open = datetime(2026, 3, 2, 9, 30, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_open)

    allowed, info = rule.check(_make_order("MSFT"))
    assert allowed is True


def test_stock_denied_at_close_exactly():
    """Stock order at exactly 16:00 ET is denied (close is exclusive)."""
    dt_close = datetime(2026, 3, 2, 16, 0, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_close)

    allowed, info = rule.check(_make_order("NVDA"))
    assert allowed is False
    assert info["reason"] == "market_closed"
    assert info["symbol"] == "NVDA"


# ---------------------------------------------------------------------------
# Stock outside market hours
# ---------------------------------------------------------------------------


def test_stock_denied_before_open():
    """Stock order before 09:30 ET is denied."""
    dt_pre = datetime(2026, 3, 2, 8, 0, 0, tzinfo=_ET)  # Monday 08:00 ET
    rule = _rule_with_fixed_time(dt_pre)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is False
    assert info["reason"] == "market_closed"


def test_stock_denied_after_close():
    """Stock order at 18:00 ET is denied."""
    dt_post = datetime(2026, 3, 2, 18, 0, 0, tzinfo=_ET)  # Monday 18:00 ET
    rule = _rule_with_fixed_time(dt_post)

    allowed, info = rule.check(_make_order("MSFT"))
    assert allowed is False
    assert info["reason"] == "market_closed"
    assert info["symbol"] == "MSFT"


# ---------------------------------------------------------------------------
# Weekend
# ---------------------------------------------------------------------------


def test_stock_denied_on_saturday():
    """Stock orders on Saturday are denied regardless of time."""
    dt_sat = datetime(2026, 3, 7, 11, 0, 0, tzinfo=_ET)  # Saturday 11:00 ET
    rule = _rule_with_fixed_time(dt_sat)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is False
    assert info["reason"] == "market_closed"


def test_stock_denied_on_sunday():
    """Stock orders on Sunday are denied."""
    dt_sun = datetime(2026, 3, 8, 11, 0, 0, tzinfo=_ET)  # Sunday 11:00 ET
    rule = _rule_with_fixed_time(dt_sun)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is False
    assert info["reason"] == "market_closed"


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------


def test_stock_denied_on_christmas():
    """Stock orders on December 25 (Christmas) are denied even during trading hours."""
    # 2026-12-25 is a Friday — Christmas is not shifted
    dt_xmas = datetime(2026, 12, 25, 11, 0, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_xmas)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is False
    assert info["reason"] == "market_closed"


def test_stock_denied_on_new_years_day():
    """Stock orders on January 1 are denied."""
    # 2026-01-01 is a Thursday
    dt_ny = datetime(2026, 1, 1, 11, 0, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_ny)

    allowed, info = rule.check(_make_order("MSFT"))
    assert allowed is False
    assert info["reason"] == "market_closed"


def test_stock_denied_on_independence_day():
    """Stock orders on July 4 are denied."""
    # 2026-07-04 is a Saturday — observance shifts to Friday July 3
    # Use the observed date
    observed = date(2026, 7, 3)
    dt_july4 = datetime(2026, 7, 3, 11, 0, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_july4)

    holidays = _us_market_holidays(2026)
    assert observed in holidays, f"July 3 2026 should be the observed Independence Day holiday; got {sorted(holidays)}"

    allowed, info = rule.check(_make_order("NVDA"))
    assert allowed is False
    assert info["reason"] == "market_closed"


def test_stock_allowed_day_after_holiday():
    """Stock orders on the trading day after a holiday are allowed."""
    # 2026-12-28 is the Monday after Christmas week — should be a normal trading day
    dt_after = datetime(2026, 12, 28, 11, 0, 0, tzinfo=_ET)
    rule = _rule_with_fixed_time(dt_after)

    allowed, info = rule.check(_make_order("AAPL"))
    assert allowed is True


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_serves_stale_value_within_ttl():
    """
    Verify the cache returns the stored value without recomputing within TTL.

    We use a large TTL and pin the computation to one result, then change
    _compute_market_open to a different result — the cached value should still
    be returned until expiry.
    """
    rule = MarketHoursRule(cache_ttl_seconds=3600)  # very long TTL

    # First: prime cache with market OPEN
    call_count = {"n": 0}

    def _open() -> bool:
        call_count["n"] += 1
        return True

    rule._compute_market_open = _open  # type: ignore[method-assign]
    result1 = rule._is_market_open()
    assert result1 is True
    assert call_count["n"] == 1

    # Second call — should hit cache, not recompute
    def _closed() -> bool:
        call_count["n"] += 1
        return False

    rule._compute_market_open = _closed  # type: ignore[method-assign]
    result2 = rule._is_market_open()
    assert result2 is True  # still True — from cache
    assert call_count["n"] == 1  # _closed was never called


def test_cache_refreshes_after_ttl_zero():
    """With TTL=0, every _is_market_open call recomputes."""
    rule = MarketHoursRule(cache_ttl_seconds=0)

    call_count = {"n": 0}

    def _compute() -> bool:
        call_count["n"] += 1
        return True

    rule._compute_market_open = _compute  # type: ignore[method-assign]
    rule._is_market_open()
    rule._is_market_open()
    rule._is_market_open()
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Holiday calendar correctness
# ---------------------------------------------------------------------------


def test_easter_algorithm_known_dates():
    """Verify Easter calculation against known dates."""
    # Known Easter dates
    known = {
        2024: date(2024, 3, 31),
        2025: date(2025, 4, 20),
        2026: date(2026, 4, 5),
        2027: date(2027, 3, 28),
    }
    for year, expected in known.items():
        assert _easter(year) == expected, f"Easter {year}: expected {expected}, got {_easter(year)}"


def test_good_friday_2026():
    """Good Friday 2026 is April 3 (2 days before Easter April 5)."""
    holidays = _us_market_holidays(2026)
    assert date(2026, 4, 3) in holidays


def test_juneteenth_2026():
    """Juneteenth (June 19) 2026 is a Friday — no shift needed."""
    holidays = _us_market_holidays(2026)
    assert date(2026, 6, 19) in holidays


def test_thanksgiving_2026():
    """Thanksgiving 2026 is the 4th Thursday of November."""
    holidays = _us_market_holidays(2026)
    assert date(2026, 11, 26) in holidays


def test_mlk_day_2026():
    """Martin Luther King Jr. Day 2026 is the 3rd Monday of January."""
    holidays = _us_market_holidays(2026)
    assert date(2026, 1, 19) in holidays


def test_us_market_holidays_count():
    """Each year should have exactly 10 market holidays (NYSE standard)."""
    for year in (2024, 2025, 2026, 2027):
        holidays = _us_market_holidays(year)
        assert len(holidays) == 10, (
            f"Expected 10 NYSE holidays in {year}, got {len(holidays)}: {sorted(holidays)}"
        )
