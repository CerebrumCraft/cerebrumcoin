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
