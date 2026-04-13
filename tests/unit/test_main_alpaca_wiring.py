"""Test Alpaca adapter is wired when config enables it."""
import pytest
from unittest.mock import patch, MagicMock

from cerebrum.main import _maybe_build_alpaca_adapter  # private helper you will add


def test_returns_none_when_alpaca_disabled():
    config = {"alpaca": {"enabled": False}}
    assert _maybe_build_alpaca_adapter(config, MagicMock()) is None


def test_returns_none_when_alpaca_missing_from_config():
    config = {}
    assert _maybe_build_alpaca_adapter(config, MagicMock()) is None


def test_returns_none_gracefully_when_module_missing(monkeypatch):
    config = {
        "alpaca": {
            "enabled": True,
            "symbols": ["AAPL"],
            "api_key_env": "ALPACA_API_KEY_ID",
            "secret_key_env": "ALPACA_API_SECRET_KEY",
            "paper_base_url": "https://paper-api.alpaca.markets",
            "data_feed": "iex",
        }
    }
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKXXX")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "yyy")
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
