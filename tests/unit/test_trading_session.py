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
