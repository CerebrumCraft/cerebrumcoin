"""Unit tests for xstocks_reversion strategy config (DEC-XSTOCKS-002)."""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.xstocks_reversion import XSTOCKS_REVERSION_CONFIG


def test_config_name():
    """Test strategy name is xstocks_reversion."""
    assert XSTOCKS_REVERSION_CONFIG.name == "xstocks_reversion"


def test_config_symbols():
    """Test symbols are tokenized equities on Kraken."""
    assert XSTOCKS_REVERSION_CONFIG.symbols == ["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"]


def test_config_no_signal_source_filter():
    """Test signal_source_filter is None (consume all signal types)."""
    assert XSTOCKS_REVERSION_CONFIG.signal_source_filter is None


def test_config_aggregator_weights():
    """Test aggregator weights: technical 1.2, regime 0.5, sentiment/news 0."""
    assert XSTOCKS_REVERSION_CONFIG.aggregator_weights[SignalType.TECHNICAL] == Decimal("1.2")
    assert XSTOCKS_REVERSION_CONFIG.aggregator_weights[SignalType.SENTIMENT] == Decimal("0")
    assert XSTOCKS_REVERSION_CONFIG.aggregator_weights[SignalType.NEWS] == Decimal("0")
    assert XSTOCKS_REVERSION_CONFIG.aggregator_weights[SignalType.REGIME] == Decimal("0.5")


def test_config_initial_balance():
    """Test initial balance is $5,000."""
    assert XSTOCKS_REVERSION_CONFIG.initial_balance == Decimal("5000.0")


def test_config_aggregator_threshold():
    """Test aggregator threshold is 0.4."""
    assert XSTOCKS_REVERSION_CONFIG.aggregator_threshold == Decimal("0.4")


def test_config_risk_overrides():
    """Test risk management parameters."""
    assert XSTOCKS_REVERSION_CONFIG.risk_overrides["position_size_percent"] == "20.0"
    assert XSTOCKS_REVERSION_CONFIG.risk_overrides["stop_loss_percent"] == "1.0"
    assert XSTOCKS_REVERSION_CONFIG.risk_overrides["take_profit_percent"] == "1.5"
    assert XSTOCKS_REVERSION_CONFIG.risk_overrides["min_signal_strength"] == "0.65"
    assert XSTOCKS_REVERSION_CONFIG.risk_overrides["post_fill_cooldown_seconds"] == 1800


def test_config_exit_config():
    """Test exit configuration parameters."""
    assert XSTOCKS_REVERSION_CONFIG.exit_config["max_position_age_minutes"] == 120
    assert XSTOCKS_REVERSION_CONFIG.exit_config["min_hold_minutes"] == 15
