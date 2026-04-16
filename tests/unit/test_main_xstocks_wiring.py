"""Test KrakenXStocksAdapter wiring (DEC-XSTOCKS-001)."""
import pytest
from unittest.mock import MagicMock
from cerebrum.main import _maybe_build_kraken_xstocks_adapter


def test_returns_none_when_disabled():
    assert _maybe_build_kraken_xstocks_adapter({"kraken_xstocks": {"enabled": False}}, MagicMock()) is None


def test_returns_none_when_section_missing():
    assert _maybe_build_kraken_xstocks_adapter({}, MagicMock()) is None


def test_returns_none_when_sdk_missing(monkeypatch):
    config = {"kraken_xstocks": {"enabled": True, "symbols": ["AAPLx/USD"]}}
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    # Patch the adapter import to simulate missing SDK
    from unittest.mock import patch
    with patch("cerebrum.adapters.kraken_xstocks.KrakenXStocksAdapter", side_effect=ImportError("no sdk")):
        result = _maybe_build_kraken_xstocks_adapter(config, MagicMock())
    assert result is None
