"""
Unit tests for SidewaysSuppressionRule.

Tests cover:
1. DENY BUY in SIDEWAYS + low-vol (primary case — Issue #3)
2. APPROVE BUY in SIDEWAYS + high-vol (enough range to reach TP)
3. APPROVE SELL in SIDEWAYS + low-vol (exits must always work)
4. APPROVE BUY in BULL regime regardless of volatility
5. APPROVE BUY in BEAR regime regardless of volatility
6. Cold start (no regime data) APPROVE
7. Cold start (insufficient price data) APPROVE
8. Per-symbol independence
9. Bus subscription wiring — regime change and market data events update state
10. Deny reason includes symbol, range, and threshold for observability

@decision DEC-TEST-013
@title Tests for SidewaysSuppressionRule
@status accepted
@rationale SidewaysSuppressionRule subscribes to both REGIME_CHANGE and MARKET_DATA
events. Tests use a real EventBus to validate end-to-end subscription wiring.
SELL orders must always be approved to prevent blocking position exits. BULL/BEAR
regimes bypass the suppression entirely — only SIDEWAYS triggers the volatility check.
"""

import asyncio
import time
from decimal import Decimal
from uuid import uuid4

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, OrderEvent, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderStatus,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import RuleDecision, SidewaysSuppressionRule


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Create and start a real EventBus."""
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


def _make_rule(
    bus: EventBus,
    min_range_pct: Decimal = Decimal("1.0"),
    window_size: int = 10,
) -> SidewaysSuppressionRule:
    """
    Create a SidewaysSuppressionRule with test-friendly defaults.

    Uses a small window_size (default 10) so tests don't need to publish
    hundreds of events to fill the window.
    """
    return SidewaysSuppressionRule(
        min_range_pct=min_range_pct,
        window_size=window_size,
        bus=bus,
    )


def _make_regime_event(
    symbol: str,
    to_regime: str,
    confidence: str = "0.7",
    from_regime: str = "UNKNOWN",
    ts: float = 1.0,
) -> RegimeChangeEvent:
    """Construct a RegimeChangeEvent with symbol embedded in indicators."""
    return RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=ts,
        from_regime=from_regime,
        to_regime=to_regime,
        confidence=Decimal(confidence),
        indicators={"symbol": symbol},
    )


def _make_market_data(symbol: str, price: Decimal) -> MarketDataEvent:
    """Create a MarketDataEvent for the given symbol and price."""
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time.time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )


def _make_buy_order(symbol: str = "BTC/USD") -> OrderEvent:
    """Create a minimal BUY OrderEvent."""
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
        status=OrderStatus.PENDING,
    )


def _make_sell_order(symbol: str = "BTC/USD") -> OrderEvent:
    """Create a minimal SELL OrderEvent."""
    return OrderEvent(
        event_type=EventType.ORDER,
        timestamp=time.time(),
        order_id=str(uuid4()),
        symbol=symbol,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        amount=Decimal("0.01"),
        status=OrderStatus.PENDING,
    )


def _make_signal(symbol: str = "BTC/USD") -> SignalEvent:
    """Create a minimal SignalEvent."""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.7"),
    )


async def _publish_prices(
    bus: EventBus,
    symbol: str,
    prices: list,
    settle_delay: float = 0.15,
) -> None:
    """Publish a sequence of MarketDataEvents and wait for delivery."""
    for price in prices:
        await bus.publish(_make_market_data(symbol, price))
    await asyncio.sleep(settle_delay)


# ---------------------------------------------------------------------------
# Test 1: DENY BUY in SIDEWAYS + low-vol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_buy_sideways_low_vol(bus):
    """
    PRIMARY CASE (Issue #3): BUY denied when regime is SIDEWAYS and price
    range is below threshold. Flat market + SIDEWAYS = guaranteed max-age exit.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    # Set regime to SIDEWAYS
    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))

    # Publish flat prices — range = 0%
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY, (
        f"Expected DENY for SIDEWAYS+flat prices, got {result.decision}: {result.reason}"
    )
    assert result.risk_level == RiskLevel.MEDIUM
    assert "SIDEWAYS" in result.reason


# ---------------------------------------------------------------------------
# Test 2: APPROVE BUY in SIDEWAYS + high-vol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_buy_sideways_high_vol(bus):
    """
    BUY approved in SIDEWAYS when price range exceeds threshold.
    A 2% range in SIDEWAYS means TP is reachable — trading is profitable.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))

    # 2% range: 50000 to 51000
    prices = [Decimal("50000")] * 5 + [Decimal("51000")] * 5
    await _publish_prices(bus, "BTC/USD", prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE for SIDEWAYS+2% range, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 3: APPROVE SELL in SIDEWAYS + low-vol (exits must work)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_sell_sideways_low_vol(bus):
    """
    SELL orders are always approved regardless of regime or volatility.
    Blocking exits would trap losing positions — must never happen.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))

    # Flat prices — would deny a BUY
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_sell_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE for SELL (exits must always work), got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 4: APPROVE BUY in BULL regardless of volatility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_buy_bull_low_vol(bus):
    """
    BULL regime bypasses SIDEWAYS suppression — only SIDEWAYS triggers the check.
    Directional regimes have momentum where TP is reachable.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "BULL"))

    # Flat prices that would deny in SIDEWAYS
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE in BULL regime, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 5: APPROVE BUY in BEAR regardless of volatility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_buy_bear_low_vol(bus):
    """
    BEAR regime bypasses SIDEWAYS suppression. RegimeTradeHaltRule handles BEAR —
    SidewaysSuppressionRule should not interfere with that logic.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "BEAR"))

    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE in BEAR regime, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 6: Cold start — no regime data APPROVE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_no_regime_data_approve(bus):
    """
    No regime data recorded yet (symbol never seen) — approve to avoid
    blocking early trades during warm-up.
    """
    rule = _make_rule(bus)

    result = rule.evaluate(_make_signal("XRP/USD"), _make_buy_order("XRP/USD"), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE with no regime data, got {result.decision}: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 7: Cold start — insufficient price data APPROVE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_start_insufficient_price_data_approve(bus):
    """
    Regime is SIDEWAYS but price window not yet full — approve during warm-up.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=100)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))

    # Publish only 5 prices — window_size is 100
    partial_prices = [Decimal("50000")] * 5
    await _publish_prices(bus, "BTC/USD", partial_prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE during cold start (5/100 ticks), got {result.decision}: {result.reason}"
    )
    assert "warming" in result.reason.lower() or "insufficient" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 8: Per-symbol independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_symbol_independence(bus):
    """
    BTC/USD SIDEWAYS+flat should not affect ETH/USD in BULL.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))
    await rule._on_regime_change(_make_regime_event("ETH/USD", "BULL"))

    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    eth_prices = [Decimal("3000")] * 5 + [Decimal("3030")] * 5
    await _publish_prices(bus, "ETH/USD", eth_prices)

    btc_result = rule.evaluate(_make_signal("BTC/USD"), _make_buy_order("BTC/USD"), portfolio=None)  # type: ignore[arg-type]
    eth_result = rule.evaluate(_make_signal("ETH/USD"), _make_buy_order("ETH/USD"), portfolio=None)  # type: ignore[arg-type]

    assert btc_result.decision == RuleDecision.DENY, (
        f"BTC/USD SIDEWAYS+flat should DENY, got {btc_result.decision}: {btc_result.reason}"
    )
    assert eth_result.decision == RuleDecision.APPROVE, (
        f"ETH/USD BULL should APPROVE, got {eth_result.decision}: {eth_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 9: Bus subscription wiring — regime + market data events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bus_subscription_wiring(bus):
    """
    Publishing RegimeChangeEvent and MarketDataEvent via bus should update
    the rule's state and trigger DENY for SIDEWAYS+flat.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=5)

    # Before any events: cold start — approve
    result_before = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]
    assert result_before.decision == RuleDecision.APPROVE

    # Publish SIDEWAYS regime via bus
    regime_event = _make_regime_event("BTC/USD", "SIDEWAYS")
    await bus.publish(regime_event)
    await asyncio.sleep(0.15)

    # Publish flat prices via bus
    for _ in range(5):
        await bus.publish(_make_market_data("BTC/USD", Decimal("50000")))
    await asyncio.sleep(0.15)

    # Now should DENY
    result_after = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]
    assert result_after.decision == RuleDecision.DENY, (
        f"Expected DENY after SIDEWAYS+flat via bus, got {result_after.decision}: {result_after.reason}"
    )

    # Verify internal state was updated
    assert "BTC/USD" in rule._regimes
    assert rule._regimes["BTC/USD"][0] == "SIDEWAYS"


# ---------------------------------------------------------------------------
# Test 10: Deny reason includes symbol, range, and threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deny_reason_includes_values(bus):
    """
    Denial reason should include symbol, current range, and threshold for
    operator observability.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))

    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    result = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]

    assert result.decision == RuleDecision.DENY
    assert "BTC/USD" in result.reason, f"Expected symbol in reason: {result.reason}"
    assert "1.0" in result.reason or "threshold" in result.reason.lower(), (
        f"Expected threshold in reason: {result.reason}"
    )
    assert "SIDEWAYS" in result.reason, f"Expected 'SIDEWAYS' in reason: {result.reason}"


# ---------------------------------------------------------------------------
# Test 11: Regime transition from SIDEWAYS to BULL resumes trading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_transition_resumes_trading(bus):
    """
    When regime transitions from SIDEWAYS to BULL, BUY orders should be approved
    even if the price window is still flat.
    """
    rule = _make_rule(bus, min_range_pct=Decimal("1.0"), window_size=10)

    await rule._on_regime_change(_make_regime_event("BTC/USD", "SIDEWAYS"))
    flat_prices = [Decimal("50000")] * 10
    await _publish_prices(bus, "BTC/USD", flat_prices)

    denied = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]
    assert denied.decision == RuleDecision.DENY

    # Regime transitions to BULL
    await rule._on_regime_change(_make_regime_event("BTC/USD", "BULL", from_regime="SIDEWAYS"))

    approved = rule.evaluate(_make_signal(), _make_buy_order(), portfolio=None)  # type: ignore[arg-type]
    assert approved.decision == RuleDecision.APPROVE, (
        f"Expected APPROVE after regime transition to BULL, got {approved.decision}: {approved.reason}"
    )
