"""Test per-strategy symbols filter on SignalAggregator (DEC-STOCKS-005)."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from cerebrum.signals.aggregator import SignalAggregator


def _make_aggregator(symbols_filter):
    bus = MagicMock()
    return SignalAggregator(
        bus=bus,
        strategy_id="test_strat",
        weights={},
        threshold=Decimal("0.3"),
        window_seconds=60,
        symbols=symbols_filter,  # NEW parameter
    )


def test_allowed_symbol_passes_filter():
    agg = _make_aggregator(symbols_filter=["AAPL", "MSFT"])
    assert agg._symbol_allowed("AAPL") is True
    assert agg._symbol_allowed("MSFT") is True


def test_disallowed_symbol_rejected():
    agg = _make_aggregator(symbols_filter=["AAPL"])
    assert agg._symbol_allowed("BTC/USD") is False


def test_none_filter_allows_all_backward_compat():
    agg = _make_aggregator(symbols_filter=None)
    assert agg._symbol_allowed("ANY/SYMBOL") is True
    assert agg._symbol_allowed("BTC/USD") is True
