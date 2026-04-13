"""Unit tests for MarketHoursGateRule (DEC-STOCKS-003)."""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock

import pytest

from cerebrum.risk.market_hours_gate import MarketHoursGateRule
from cerebrum.risk.rules import RuleDecision

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def _make_order(symbol: str):
    """Minimal order stub — match whatever OrderEvent fields the rule reads (should be `symbol` only)."""
    order = MagicMock()
    order.symbol = symbol
    order.side = "buy"
    order.amount = Decimal("10")
    return order


def _make_signal():
    return MagicMock()


def _make_portfolio():
    return MagicMock()


def _rule(**overrides):
    defaults = {
        "stock_symbols": ["AAPL", "MSFT", "NVDA"],
        "entry_cutoff_minutes_before_close": 15,
        "now_utc_provider": None,  # optional — inject clock for testability
    }
    defaults.update(overrides)
    return MarketHoursGateRule(**defaults)


def _evaluate_at(rule, symbol, now_utc):
    """Helper that lets tests inject the current time."""
    order = _make_order(symbol)
    signal = _make_signal()
    portfolio = _make_portfolio()
    # Inject clock via the rule's _now_utc attribute or constructor arg
    rule._now_utc_override = now_utc  # adjust to whatever override mechanism the rule provides
    return rule.evaluate(signal, order, portfolio)


def test_denies_stock_order_outside_rth():
    r = _rule()
    # 07:00 ET pre-market on a weekday (June, EDT = UTC-4 → 11:00 UTC)
    result = _evaluate_at(r, "AAPL", _utc(2026, 6, 15, 11, 0))
    assert result.decision == RuleDecision.DENY
    assert "market_hours_gate" in result.reason or "rth" in result.reason.lower() or "hours" in result.reason.lower()


def test_allows_stock_order_inside_rth():
    r = _rule()
    # 10:00 ET on a weekday → 14:00 UTC
    result = _evaluate_at(r, "AAPL", _utc(2026, 6, 15, 14, 0))
    assert result.decision == RuleDecision.APPROVE


def test_allows_crypto_order_anytime():
    r = _rule()
    # Sunday 00:00 ET → Saturday 04:00 UTC. Crypto not in stock_symbols list → always passes.
    result = _evaluate_at(r, "BTC/USD", _utc(2026, 6, 20, 4, 0))
    assert result.decision == RuleDecision.APPROVE


def test_denies_stock_order_inside_entry_cutoff():
    r = _rule()
    # 15:46 ET (14 min before 16:00 close) — inside the 15-min cutoff window → deny
    result = _evaluate_at(r, "AAPL", _utc(2026, 6, 15, 19, 46))
    assert result.decision == RuleDecision.DENY


def test_allows_stock_order_just_before_cutoff():
    r = _rule()
    # 15:44 ET (16 min before close) → outside cutoff → approve
    result = _evaluate_at(r, "AAPL", _utc(2026, 6, 15, 19, 44))
    assert result.decision == RuleDecision.APPROVE


def test_denies_all_stock_orders_on_holiday():
    r = _rule()
    # Christmas 2026 at 15:00 ET → 20:00 UTC (EST = UTC-5)
    result = _evaluate_at(r, "AAPL", _utc(2026, 12, 25, 20, 0))
    assert result.decision == RuleDecision.DENY
