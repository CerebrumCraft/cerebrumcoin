"""
Market hours guard for stock symbol orders.

Denies orders for US stock symbols outside NYSE trading hours (Mon-Fri 9:30-16:00 ET)
and on US market holidays. Crypto symbols (containing '/') are always allowed through.

@decision DEC-HOURS-001
@title Local calendar market hours check — no external API dependency
@status accepted
@rationale Using an external API (e.g. Alpaca clock endpoint) for market hours would
add a network dependency and a failure mode during startup. A local calendar
implementation using Python's zoneinfo module is deterministic, fast (microseconds),
and requires no credentials. The trade-off is that ad-hoc market closures (e.g.
national mourning days) require a code update, but these are rare and announcements
are made days in advance. The cache (60s TTL) avoids calling datetime on every order,
while keeping the state fresh enough that transitions are noticed within one minute.
The rule is NOT wired into the live trading path yet (Phase 13D) — it only exists
as a standalone guard that can be unit-tested and optionally applied.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import structlog

from cerebrum.core.events import OrderEvent
from cerebrum.core.types import Symbol

logger = structlog.get_logger()

# Eastern Time zone used for all NYSE schedule calculations
_ET = ZoneInfo("America/New_York")


def _us_market_holidays(year: int) -> frozenset[date]:
    """
    Compute the set of US market holidays for a given year.

    Covers the ten NYSE-observed holidays:
      New Year's Day, Martin Luther King Jr. Day, Presidents' Day,
      Good Friday, Memorial Day, Juneteenth, Independence Day,
      Labor Day, Thanksgiving Day, Christmas Day.

    Rules:
    - If the calendar date falls on Saturday, the observance moves to Friday.
    - If the calendar date falls on Sunday, the observance moves to Monday.
    """

    def _observed(d: date) -> date:
        """Shift a holiday date to the nearest weekday if it falls on a weekend."""
        wd = d.weekday()  # Monday=0, Sunday=6
        if wd == 5:  # Saturday -> Friday
            from datetime import timedelta
            return date(d.year, d.month, d.day - 1)
        if wd == 6:  # Sunday -> Monday
            from datetime import timedelta
            return date(d.year, d.month, d.day + 1)
        return d

    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Return the nth occurrence of weekday (0=Mon) in (year, month)."""
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return date(year, month, 1 + offset + (n - 1) * 7)

    def _last_weekday(year: int, month: int, weekday: int) -> date:
        """Return the last occurrence of weekday in (year, month)."""
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        last = date(year, month, last_day)
        offset = (last.weekday() - weekday) % 7
        return date(year, month, last_day - offset)

    # Easter (Good Friday is 2 days before Easter Sunday)
    easter_sunday = _easter(year)
    from datetime import timedelta
    good_friday = easter_sunday - timedelta(days=2)

    holidays = {
        # New Year's Day — January 1
        _observed(date(year, 1, 1)),
        # Martin Luther King Jr. Day — 3rd Monday in January
        _nth_weekday(year, 1, 0, 3),
        # Presidents' Day — 3rd Monday in February
        _nth_weekday(year, 2, 0, 3),
        # Good Friday — 2 days before Easter Sunday (no weekend shift needed)
        good_friday,
        # Memorial Day — last Monday in May
        _last_weekday(year, 5, 0),
        # Juneteenth — June 19 (observed since 2022)
        _observed(date(year, 6, 19)),
        # Independence Day — July 4
        _observed(date(year, 7, 4)),
        # Labor Day — 1st Monday in September
        _nth_weekday(year, 9, 0, 1),
        # Thanksgiving Day — 4th Thursday in November
        _nth_weekday(year, 11, 3, 4),
        # Christmas Day — December 25
        _observed(date(year, 12, 25)),
    }
    return frozenset(holidays)


def _easter(year: int) -> date:
    """
    Compute Easter Sunday for a given year using the Anonymous Gregorian algorithm.

    This is a pure-Python implementation — no external dependencies.
    Reference: https://en.wikipedia.org/wiki/Date_of_Easter#Anonymous_Gregorian_algorithm
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


class MarketHoursRule:
    """
    Risk rule that denies stock orders placed outside NYSE trading hours.

    NYSE schedule: Monday-Friday, 09:30-16:00 Eastern Time, excluding US market holidays.

    Crypto symbols (those containing '/') bypass this rule entirely — crypto markets
    trade 24/7 and have no exchange-imposed close.

    Cache: The "is market open" state is recomputed at most once every 60 seconds
    to avoid datetime overhead on every order check. The cached state includes the
    timestamp so callers can override the cache TTL for testing.

    Usage (not yet wired into main.py — Phase 13D):
        rule = MarketHoursRule()
        allowed, info = rule.check(order)
        if not allowed:
            print(info["reason"])  # "market_closed"
    """

    # NYSE open/close times in Eastern Time
    _MARKET_OPEN_HOUR = 9
    _MARKET_OPEN_MINUTE = 30
    _MARKET_CLOSE_HOUR = 16
    _MARKET_CLOSE_MINUTE = 0

    # Cache TTL in seconds — refresh "is market open" state at most this often
    _CACHE_TTL_SECONDS = 60

    def __init__(self, cache_ttl_seconds: int = _CACHE_TTL_SECONDS) -> None:
        """
        Initialize MarketHoursRule.

        Args:
            cache_ttl_seconds: How often to recompute the market-open state.
                               Default 60s. Pass a smaller value in tests if needed.
        """
        self._cache_ttl = cache_ttl_seconds
        self._cached_open: bool | None = None
        self._cache_expires: float = 0.0
        self._log = logger.bind(component="market_hours_rule")

    def check(self, order: OrderEvent) -> tuple[bool, dict]:
        """
        Check whether the order is allowed based on market hours.

        Args:
            order: The order to evaluate.

        Returns:
            (True, {}) if the order is allowed.
            (False, {"reason": "market_closed", "symbol": symbol}) if denied.
        """
        symbol: Symbol = order.symbol

        # Crypto symbols always pass through
        if self._is_crypto_symbol(symbol):
            self._log.debug("market_hours_crypto_pass", symbol=symbol)
            return True, {}

        # Stock symbol — check market hours
        if self._is_market_open():
            self._log.debug("market_hours_stock_open", symbol=symbol)
            return True, {}

        self._log.info(
            "market_hours_stock_denied",
            symbol=symbol,
            reason="market_closed",
        )
        return False, {"reason": "market_closed", "symbol": symbol}

    def _is_crypto_symbol(self, symbol: Symbol) -> bool:
        """
        Determine if a symbol is a crypto trading pair.

        Convention: crypto pairs contain '/' (e.g. "BTC/USD", "ETH/USD").
        Stock symbols do not contain '/' (e.g. "AAPL", "MSFT", "NVDA").
        """
        return "/" in symbol

    def _is_market_open(self) -> bool:
        """
        Return True if NYSE is currently open, using a cached value refreshed
        every cache_ttl_seconds.
        """
        now = time.monotonic()
        if self._cached_open is not None and now < self._cache_expires:
            return self._cached_open

        result = self._compute_market_open()
        self._cached_open = result
        self._cache_expires = now + self._cache_ttl

        self._log.debug(
            "market_hours_cache_refreshed",
            is_open=result,
            cache_ttl=self._cache_ttl,
        )
        return result

    def _compute_market_open(self) -> bool:
        """
        Compute the current NYSE open/closed state from wall-clock time.

        NYSE is open when ALL of:
        1. The current day (ET) is a weekday (Mon-Fri).
        2. The current date is not a US market holiday.
        3. The current time (ET) is between 09:30 and 16:00 inclusive of open,
           exclusive of close (i.e. 09:30:00 <= t < 16:00:00).
        """
        now_et = datetime.now(tz=_ET)
        today = now_et.date()

        # Check weekend
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check market holiday
        holidays = _us_market_holidays(today.year)
        if today in holidays:
            return False

        # Check trading hours
        open_time = now_et.replace(
            hour=self._MARKET_OPEN_HOUR,
            minute=self._MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        close_time = now_et.replace(
            hour=self._MARKET_CLOSE_HOUR,
            minute=self._MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return open_time <= now_et < close_time
