"""
Unit tests for configuration management.

@decision DEC-TEST-003
@title Test config validation and TOML loading
@status accepted
@rationale Validates Pydantic settings behavior: type validation, percentage bounds,
TOML parsing. Uses temp files for TOML tests to avoid polluting the repo. Tests prove
config catches invalid values before runtime.
"""

from decimal import Decimal
from pathlib import Path
import tempfile

import pytest

from cerebrum.core.config import (
    Config,
    ExchangeConfig,
    LoggingConfig,
    PaperTradingConfig,
    ProfileConfig,
    RiskConfig,
    SignalConfig,
    TradingConfig,
)
from cerebrum.core.types import TradingMode


def test_exchange_config_defaults(monkeypatch):
    """Test ExchangeConfig default values.

    Isolates from real .env file and environment variables by passing
    _env_file=None and clearing EXCHANGE_* env vars via monkeypatch.
    """
    # Isolate from real .env file and any EXCHANGE_* env vars
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)

    config = ExchangeConfig(_env_file=None)

    assert config.name == "kraken"
    assert config.api_key == ""
    assert config.api_secret == ""
    assert config.testnet is False
    assert config.rate_limit_per_minute == 60
    assert config.websocket_enabled is True


def test_risk_config_defaults():
    """Test RiskConfig default values."""
    config = RiskConfig()

    assert config.max_position_size_usd == Decimal("1000.0")
    assert config.max_total_exposure_usd == Decimal("5000.0")
    assert config.max_drawdown_percent == Decimal("5.0")
    assert config.max_daily_loss_usd == Decimal("500.0")
    assert config.position_size_percent == Decimal("2.0")


def test_risk_config_percentage_validation():
    """Test that percentage fields validate bounds."""
    # Valid percentage
    config = RiskConfig(max_drawdown_percent=Decimal("10.0"))
    assert config.max_drawdown_percent == Decimal("10.0")

    # Invalid: negative
    with pytest.raises(ValueError, match="between 0 and 100"):
        RiskConfig(max_drawdown_percent=Decimal("-5.0"))

    # Invalid: > 100
    with pytest.raises(ValueError, match="between 0 and 100"):
        RiskConfig(max_drawdown_percent=Decimal("150.0"))


def test_paper_trading_config_defaults():
    """Test PaperTradingConfig default values."""
    config = PaperTradingConfig()

    assert config.initial_balance_usd == Decimal("10000.0")
    assert config.commission_percent == Decimal("0.1")
    assert config.slippage_percent == Decimal("0.05")
    assert config.state_file == Path("data/paper_state.json")


def test_trading_config_defaults():
    """Test TradingConfig default values."""
    config = TradingConfig()

    assert config.mode == TradingMode.PAPER
    assert config.symbols == ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    assert config.data_refresh_seconds == 1


def test_master_config_composition():
    """Test that Config composes all sub-configs."""
    config = Config()

    assert isinstance(config.exchange, ExchangeConfig)
    assert isinstance(config.risk, RiskConfig)
    assert isinstance(config.paper, PaperTradingConfig)
    assert isinstance(config.trading, TradingConfig)


def test_config_from_toml():
    """Test loading configuration from TOML file."""
    toml_content = """
[exchange]
name = "kraken"
testnet = true

[risk]
max_position_size_usd = "2000.0"
max_drawdown_percent = "10.0"

[paper]
initial_balance_usd = "50000.0"

[trading]
mode = "paper"
symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = Path(f.name)

    try:
        config, raw_toml = Config.from_toml(temp_path)

        assert config.exchange.name == "kraken"
        assert config.exchange.testnet is True
        assert config.risk.max_position_size_usd == Decimal("2000.0")
        assert config.risk.max_drawdown_percent == Decimal("10.0")
        assert config.paper.initial_balance_usd == Decimal("50000.0")
        assert config.trading.symbols == ["BTC/USD", "ETH/USD", "SOL/USD"]
        assert isinstance(raw_toml, dict)
        assert raw_toml["exchange"]["name"] == "kraken"

    finally:
        temp_path.unlink()


def test_config_from_nonexistent_toml():
    """Test loading from non-existent file returns defaults."""
    config, raw_toml = Config.from_toml(Path("/nonexistent/path.toml"))

    # Should return default config
    assert config.exchange.name == "kraken"
    assert config.paper.initial_balance_usd == Decimal("10000.0")
    assert raw_toml == {}


def test_config_partial_toml():
    """Test that TOML can override subset of values."""
    toml_content = """
[risk]
max_drawdown_percent = "3.0"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = Path(f.name)

    try:
        config, _raw_toml = Config.from_toml(temp_path)

        # Overridden value
        assert config.risk.max_drawdown_percent == Decimal("3.0")

        # Default values still present
        assert config.risk.max_position_size_usd == Decimal("1000.0")
        assert config.exchange.name == "kraken"

    finally:
        temp_path.unlink()


def test_logging_config_defaults():
    """Test LoggingConfig default values."""
    config = LoggingConfig()

    assert config.level == "INFO"
    assert config.format == "json"
    assert config.file is None


def test_logging_config_level_parsed_from_toml():
    """Test that LoggingConfig.level is correctly parsed from TOML."""
    toml_content = """
[logging]
level = "DEBUG"
format = "console"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        temp_path = Path(f.name)

    try:
        config, _raw_toml = Config.from_toml(temp_path)

        assert config.logging.level == "DEBUG"
        assert config.logging.format == "console"
    finally:
        temp_path.unlink()


def test_logging_config_level_variety():
    """Test that LoggingConfig accepts all standard log levels."""
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        config = LoggingConfig(level=level)
        assert config.level == level


def test_paper_toml_tuned_parameters():
    """Test that paper.toml has tuned signal and risk parameters."""
    paper_config_path = Path(__file__).parent.parent.parent / "config" / "paper.toml"

    if not paper_config_path.exists():
        pytest.skip("paper.toml not found - may be in different environment")

    config, _raw_toml = Config.from_toml(paper_config_path)

    # Verify tuned signal parameters (tighter than defaults)
    assert config.signals.aggregation_threshold == Decimal("0.4"), \
        "aggregation_threshold should be 0.4 — consensus multiplier makes 0.4 equivalent to old 0.5"
    assert config.signals.aggregation_window_seconds == 120, \
        "aggregation_window should be 120s (tuned in session 2/3 to reduce signal spam)"
    assert config.signals.rsi_oversold == 25, \
        "rsi_oversold should be 25 (down from 30) for more extreme signals"
    assert config.signals.rsi_overbought == 75, \
        "rsi_overbought should be 75 (up from 70) for more extreme signals"

    # Verify tuned risk parameters
    assert config.risk.min_signal_strength == Decimal("0.65"), \
        "min_signal_strength should be 0.65 (up from 0.6, session 13 tuning)"
    assert config.risk.stop_loss_percent == Decimal("1.0"), \
        "stop_loss_percent should be 1.0 (down from 1.5, session 13 0% WR — cut losses faster)"
    assert config.risk.take_profit_percent == Decimal("3.0"), \
        "take_profit_percent should be 3.0 (new exit rule)"
    assert config.risk.max_position_age_minutes == 120, \
        "max_position_age_minutes should be 120 (2 hour time-based exit)"
    assert config.risk.position_size_percent == Decimal("5.0"), \
        "position_size_percent should be 5.0 (tuned in session 4 to reduce commission drag on larger trades)"
    assert config.risk.post_fill_cooldown_seconds == 1800, \
        "post_fill_cooldown_seconds should be 1800 (30 min, session 13 tuning for choppy markets)"

    # Verify volatility gate parameters (Issue #2)
    assert config.risk.volatility_gate_min_range_pct == Decimal("0.5"), \
        "volatility_gate_min_range_pct should be 0.5% (covers commission 0.32% + slippage 0.1%)"
    assert config.risk.volatility_gate_window_size == 300, \
        "volatility_gate_window_size should be 300 ticks (~5 min rolling window)"

    # Verify paper trading parameters remain realistic
    assert config.paper.commission_percent == Decimal("0.16"), \
        "commission should match Kraken's maker fee"
    assert config.paper.slippage_percent == Decimal("0.1"), \
        "slippage should be conservative"


def test_paper_toml_loads_four_symbols():
    """Test that paper.toml explicitly declares all four trading pairs (Phase 13 expansion).

    paper.toml loads standalone (not layered on default.toml), so symbols must
    be present in the [trading] section rather than relying on TradingConfig defaults.
    """
    paper_config_path = Path(__file__).parent.parent.parent / "config" / "paper.toml"

    if not paper_config_path.exists():
        pytest.skip("paper.toml not found")

    config, _raw_toml = Config.from_toml(paper_config_path)

    expected = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    assert config.trading.symbols == expected, (
        f"paper.toml should declare all 4 trading pairs; got {config.trading.symbols}"
    )


def test_default_toml_loads_four_symbols():
    """Test that config/default.toml declares all four trading pairs (Phase 13 expansion)."""
    default_config_path = Path(__file__).parent.parent.parent / "config" / "default.toml"

    if not default_config_path.exists():
        pytest.skip("config/default.toml not found")

    config, _raw_toml = Config.from_toml(default_config_path)

    expected = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
    assert config.trading.symbols == expected, (
        f"default.toml should declare all 4 trading pairs; got {config.trading.symbols}"
    )
