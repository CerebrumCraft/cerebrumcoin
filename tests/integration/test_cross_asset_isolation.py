"""Integration tests — no cross-asset signal contamination (DEC-STOCKS-005).

Validates that SignalAggregator's `symbols` filter correctly isolates crypto
aggregators from stock signals and vice versa. Uses synthetic SignalEvents
injected directly onto the bus — no live fixtures required.

@decision DEC-STOCKS-005
@title Symbol-scoped SignalAggregator prevents cross-asset contamination
@status accepted
@rationale Crypto strategies (mean_reversion, range_trading) must never act
on equity signals (AAPL, SPY) and the orb_stocks strategy must never act on
crypto signals (BTC/USD, ETH/USD). The `symbols` param added to
SignalAggregator in Task 5 enforces this at the aggregator level. These tests
prove the filter holds end-to-end through the real EventBus dispatch path.
"""

import asyncio
from decimal import Decimal

import pytest

from cerebrum.core.types import SignalAction, SignalType
from tests.integration.conftest import build_test_pipeline


async def test_crypto_strategies_dont_see_stock_ticks():
    """DEC-STOCKS-005: mean_reversion and range_trading aggregators must never
    receive AAPL signals even when AAPL ticks are published on the shared bus.
    """
    pipeline = await build_test_pipeline(
        crypto_strategies=["mean_reversion", "range_trading"],
        stock_strategies=["orb_stocks"],
        crypto_symbols=["BTC/USD", "ETH/USD"],
        stock_symbols=["AAPL"],
    )

    try:
        # Publish alternating stock and crypto signals — 10 of each.
        for _ in range(10):
            await pipeline.publish_signal("AAPL", action=SignalAction.BUY)
            await pipeline.publish_signal("BTC/USD", action=SignalAction.BUY)

        mr_seen = pipeline.aggregator_input_symbols("mean_reversion")
        rt_seen = pipeline.aggregator_input_symbols("range_trading")
        orb_seen = pipeline.aggregator_input_symbols("orb_stocks")

        # Crypto aggregators must NOT have seen AAPL
        assert "AAPL" not in mr_seen, (
            f"mean_reversion aggregator admitted AAPL signal — "
            f"DEC-STOCKS-005 filter is broken. seen={mr_seen}"
        )
        assert "AAPL" not in rt_seen, (
            f"range_trading aggregator admitted AAPL signal — "
            f"DEC-STOCKS-005 filter is broken. seen={rt_seen}"
        )

        # Crypto aggregators SHOULD have seen BTC/USD
        assert "BTC/USD" in mr_seen, (
            f"mean_reversion aggregator missed BTC/USD — symbols filter too aggressive. seen={mr_seen}"
        )
        assert "BTC/USD" in rt_seen, (
            f"range_trading aggregator missed BTC/USD — symbols filter too aggressive. seen={rt_seen}"
        )

        # Stock aggregator should have seen AAPL
        assert "AAPL" in orb_seen, (
            f"orb_stocks aggregator missed AAPL — symbols filter too aggressive. seen={orb_seen}"
        )

    finally:
        await pipeline.stop()


async def test_stock_strategy_doesnt_see_crypto_ticks():
    """Inverse of above — orb_stocks aggregator must never receive BTC/USD signals."""
    pipeline = await build_test_pipeline(
        crypto_strategies=["mean_reversion"],
        stock_strategies=["orb_stocks"],
        crypto_symbols=["BTC/USD"],
        stock_symbols=["AAPL"],
    )

    try:
        for _ in range(10):
            await pipeline.publish_signal("BTC/USD", action=SignalAction.SELL)

        orb_seen = pipeline.aggregator_input_symbols("orb_stocks")
        mr_seen = pipeline.aggregator_input_symbols("mean_reversion")

        assert "BTC/USD" not in orb_seen, (
            f"orb_stocks aggregator admitted BTC/USD signal — "
            f"DEC-STOCKS-005 filter is broken. seen={orb_seen}"
        )

        # Crypto aggregator should have seen BTC/USD
        assert "BTC/USD" in mr_seen, (
            f"mean_reversion aggregator missed BTC/USD — symbols filter too aggressive. seen={mr_seen}"
        )

    finally:
        await pipeline.stop()


async def test_multi_symbol_crypto_isolation():
    """Multiple crypto symbols are admitted; all stock symbols are rejected."""
    pipeline = await build_test_pipeline(
        crypto_strategies=["mean_reversion"],
        stock_strategies=["orb_stocks"],
        crypto_symbols=["BTC/USD", "ETH/USD"],
        stock_symbols=["AAPL", "SPY"],
    )

    try:
        for sym in ["BTC/USD", "ETH/USD", "AAPL", "SPY"]:
            await pipeline.publish_signal(sym, action=SignalAction.BUY)

        mr_seen = pipeline.aggregator_input_symbols("mean_reversion")
        orb_seen = pipeline.aggregator_input_symbols("orb_stocks")

        assert "AAPL" not in mr_seen, f"mean_reversion saw AAPL: {mr_seen}"
        assert "SPY" not in mr_seen, f"mean_reversion saw SPY: {mr_seen}"
        assert "BTC/USD" not in orb_seen, f"orb_stocks saw BTC/USD: {orb_seen}"
        assert "ETH/USD" not in orb_seen, f"orb_stocks saw ETH/USD: {orb_seen}"

        assert "BTC/USD" in mr_seen, f"mean_reversion missed BTC/USD: {mr_seen}"
        assert "ETH/USD" in mr_seen, f"mean_reversion missed ETH/USD: {mr_seen}"
        assert "AAPL" in orb_seen, f"orb_stocks missed AAPL: {orb_seen}"
        assert "SPY" in orb_seen, f"orb_stocks missed SPY: {orb_seen}"

    finally:
        await pipeline.stop()


async def test_no_symbols_filter_admits_all():
    """When symbols=None (no filter), the aggregator admits every symbol — backward compat."""
    pipeline = await build_test_pipeline(
        crypto_strategies=[],
        stock_strategies=[],
        crypto_symbols=[],
        stock_symbols=[],
    )

    # Build an unscoped aggregator manually
    from cerebrum.signals.aggregator import SignalAggregator
    from cerebrum.core.types import SignalType
    from decimal import Decimal

    unscoped = SignalAggregator(
        bus=pipeline.bus,
        strategy_id="unscoped",
        weights={SignalType.TECHNICAL: Decimal("1.0")},
        threshold=Decimal("0.3"),
        window_seconds=60,
        symbols=None,   # no filter
    )
    pipeline.aggregators["unscoped"] = unscoped

    try:
        for sym in ["BTC/USD", "AAPL", "ETH/USD", "SPY"]:
            await pipeline.publish_signal(sym, action=SignalAction.BUY)

        seen = pipeline.aggregator_input_symbols("unscoped")
        assert "BTC/USD" in seen, f"unscoped missed BTC/USD: {seen}"
        assert "AAPL" in seen, f"unscoped missed AAPL: {seen}"
        assert "ETH/USD" in seen, f"unscoped missed ETH/USD: {seen}"
        assert "SPY" in seen, f"unscoped missed SPY: {seen}"

    finally:
        await pipeline.stop()
