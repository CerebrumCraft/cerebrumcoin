"""
Root pytest configuration.

Registers custom CLI options and markers used across the test suite.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-alpaca",
        action="store_true",
        default=False,
        help="Run live Alpaca tests (requires ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--live-alpaca"):
        skip_marker = pytest.mark.skip(reason="needs --live-alpaca flag")
        for item in items:
            if "live_alpaca" in item.keywords:
                item.add_marker(skip_marker)
