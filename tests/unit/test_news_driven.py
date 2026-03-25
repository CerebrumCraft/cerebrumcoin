"""
Tests for the news-driven trading strategy configuration.

Validates that NEWS_DRIVEN_CONFIG has the expected weights, exits, and
cooldown parameters that make it function as an event-driven strategy.

@decision DEC-NEWS-002
@title Test news-driven config values against spec, not behavior
@status accepted
@rationale NEWS_DRIVEN_CONFIG is a pure data object (frozen StrategyConfig).
The correct test strategy is asserting field values directly — not mocking
the aggregator or risk manager. This follows Sacred Practice #5: test against
real implementations, no internal mocks. The config fields are the contract;
these tests enforce that contract remains stable across refactors.
"""

import pytest
from decimal import Decimal
from cerebrum.strategies.news_driven import NEWS_DRIVEN_CONFIG
from cerebrum.core.types import SignalType


def test_news_driven_config_weights():
    """News-driven strategy heavily weights NEWS signals."""
    assert NEWS_DRIVEN_CONFIG.name == "news_driven"
    assert NEWS_DRIVEN_CONFIG.aggregator_weights[SignalType.NEWS] == Decimal("2.0")
    assert NEWS_DRIVEN_CONFIG.aggregator_weights[SignalType.TECHNICAL] == Decimal("0.2")


def test_news_driven_wider_exits():
    """News-driven strategy uses wider exits for event-driven moves."""
    assert NEWS_DRIVEN_CONFIG.exit_config["take_profit_percent"] == "4.0"
    assert NEWS_DRIVEN_CONFIG.exit_config["stop_loss_percent"] == "2.5"
    assert NEWS_DRIVEN_CONFIG.exit_config["max_position_age_minutes"] == 240


def test_news_driven_longer_cooldown():
    """30-min cooldown prevents overreaction to follow-up commentary."""
    assert NEWS_DRIVEN_CONFIG.risk_overrides["post_fill_cooldown_seconds"] == 1800


def test_news_driven_initial_balance():
    """Balance is 1/6 of $10k for the 6-strategy equal split."""
    assert NEWS_DRIVEN_CONFIG.initial_balance == Decimal("1666.67")


def test_news_driven_sentiment_weight_above_technical():
    """Sentiment (0.8) outweighs technical (0.2) — confirms news direction."""
    assert (
        NEWS_DRIVEN_CONFIG.aggregator_weights[SignalType.SENTIMENT]
        > NEWS_DRIVEN_CONFIG.aggregator_weights[SignalType.TECHNICAL]
    )


def test_news_driven_symbols():
    """Strategy trades BTC/USD and ETH/USD."""
    assert "BTC/USD" in NEWS_DRIVEN_CONFIG.symbols
    assert "ETH/USD" in NEWS_DRIVEN_CONFIG.symbols


def test_news_driven_is_immutable():
    """frozen=True prevents accidental runtime mutation."""
    with pytest.raises(AttributeError):
        NEWS_DRIVEN_CONFIG.name = "hacked"  # type: ignore[misc]
