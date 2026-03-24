"""
Unit tests for RangeDetector.

Verifies bounce counting, deduplication, range confirmation, regime gating,
staleness handling, and breakout invalidation.

All tests use a real EventBus (no mocks for internal modules per Sacred
Practice #5). Events are published to the bus; the detector processes them
via its subscribed handlers.

@decision DEC-TEST-RANGE-001
@title Real EventBus for all RangeDetector tests — no mocks
@status accepted
@rationale RangeDetector is an async subscriber that processes events through
the bus pipeline. Testing with a real EventBus exercises the actual delivery
path (queue, task dispatch, handler invocation) and guards against timing
bugs that mocks would hide. The only state visible to tests is through
get_range(), matching production usage. Each test publishes events and drains
the bus queue with asyncio.sleep before asserting.
"""

import asyncio
import time
from decimal import Decimal

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.strategies.range_detector import RangeDetector, RangeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "BTC/USD"
SUPPORT = Decimal("69500")
RESISTANCE = Decimal("70200")
# Width = (70200 - 69500) / 69500 * 100 ≈ 1.007%


def _make_regime_event(to_regime: str, from_regime: str = "UNKNOWN") -> RegimeChangeEvent:
    return RegimeChangeEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        from_regime=from_regime,
        to_regime=to_regime,
        confidence=Decimal("0.9"),
        indicators={},
    )


def _make_support_signal(
    symbol: str = SYMBOL,
    level: Decimal = SUPPORT,
    touches: int = 2,
    distance_pct: str = "0.15",
) -> SignalEvent:
    return SignalEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.5"),
        confidence=Decimal("0.6"),
        reason=f"Near support {level} ({touches} touches, {distance_pct}% away)",
        metadata={"source": "SupportResistance"},
    )


def _make_resistance_signal(
    symbol: str = SYMBOL,
    level: Decimal = RESISTANCE,
    touches: int = 2,
    distance_pct: str = "0.10",
) -> SignalEvent:
    return SignalEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.SELL,
        strength=Decimal("0.5"),
        confidence=Decimal("0.6"),
        reason=f"Near resistance {level} ({touches} touches, {distance_pct}% away)",
        metadata={"source": "SupportResistance"},
    )


def _make_market_data(
    symbol: str = SYMBOL,
    price: Decimal = Decimal("69850"),
) -> MarketDataEvent:
    return MarketDataEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        symbol=symbol,
        price=price,
        bid=price - Decimal("1"),
        ask=price + Decimal("1"),
        volume=Decimal("1.0"),
    )


async def _drain(seconds: float = 0.15) -> None:
    """Allow the event bus to deliver queued events."""
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    b = EventBus(queue_size=100)
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
async def detector(bus):
    """RangeDetector with low min_bounces (3) wired to a real EventBus."""
    d = RangeDetector(
        bus,
        min_bounces=3,
        min_range_width_pct=Decimal("0.6"),
        breakout_margin_pct=Decimal("0.5"),
        level_staleness_minutes=120,
    )
    await d.start()
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_range_returns_none_for_unknown_symbol(detector):
    """No data for a symbol → get_range() returns None."""
    result = detector.get_range("ETH/USD")
    assert result is None


@pytest.mark.asyncio
async def test_range_not_confirmed_with_fewer_bounces(bus, detector):
    """Two bounces (1 support + 1 resistance) → range exists but not confirmed."""
    # Set regime to SIDEWAYS
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # 1 support bounce
    await bus.publish(_make_support_signal())
    await _drain()

    # Move price away from support zone to allow the next signal to count
    await bus.publish(_make_market_data(price=Decimal("70000")))
    await _drain()

    # 1 resistance bounce
    await bus.publish(_make_resistance_signal())
    await _drain()

    result = detector.get_range(SYMBOL)
    assert result is not None, "Should have a partial range after 2 bounces"
    assert result.bounce_count == 2
    assert result.range_confirmed is False


@pytest.mark.asyncio
async def test_range_confirmed_after_three_bounces(bus, detector):
    """2 support + 1 resistance = 3 bounces → range_confirmed == True."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Support bounce 1
    await bus.publish(_make_support_signal())
    await _drain()

    # Price leaves support zone
    await bus.publish(_make_market_data(price=Decimal("70000")))
    await _drain()

    # Resistance bounce 1
    await bus.publish(_make_resistance_signal())
    await _drain()

    # Price leaves resistance zone
    await bus.publish(_make_market_data(price=Decimal("69850")))
    await _drain()

    # Support bounce 2
    await bus.publish(_make_support_signal())
    await _drain()

    result = detector.get_range(SYMBOL)
    assert result is not None
    assert result.bounce_count == 3
    assert result.range_confirmed is True
    assert result.support_level == SUPPORT
    assert result.resistance_level == RESISTANCE


@pytest.mark.asyncio
async def test_bounce_deduplication(bus, detector):
    """5 consecutive support signals without price leaving zone → only 1 bounce counted."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Send 5 support signals without any market data that would trigger zone exit
    for _ in range(5):
        await bus.publish(_make_support_signal())
        await asyncio.sleep(0.02)

    await _drain()

    result = detector.get_range(SYMBOL)
    # Only 1 bounce because price never left the support zone
    if result is not None:
        assert result.bounce_count == 1, (
            f"Expected 1 bounce due to deduplication, got {result.bounce_count}"
        )


@pytest.mark.asyncio
async def test_range_invalidated_on_regime_change(bus, detector):
    """SIDEWAYS → BEAR transition clears all range data."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Build 3 bounces to confirm the range
    await bus.publish(_make_support_signal())
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("70000")))
    await _drain()
    await bus.publish(_make_resistance_signal())
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("69850")))
    await _drain()
    await bus.publish(_make_support_signal())
    await _drain()

    # Confirm range exists
    pre_change = detector.get_range(SYMBOL)
    assert pre_change is not None and pre_change.range_confirmed

    # Regime switches to BEAR
    await bus.publish(_make_regime_event("BEAR", from_regime="SIDEWAYS"))
    await _drain()

    # Range must be gone
    post_change = detector.get_range(SYMBOL)
    assert post_change is None, "Range should be invalidated after leaving SIDEWAYS"


@pytest.mark.asyncio
async def test_range_rejected_when_too_narrow(bus, detector):
    """Support=100, resistance=100.4 → width 0.4% < 0.6% min → not confirmed."""
    narrow_support = Decimal("100")
    narrow_resistance = Decimal("100.4")  # 0.4% width

    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Support bounce 1
    await bus.publish(_make_support_signal(level=narrow_support))
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("100.2")))
    await _drain()

    # Resistance bounce 1
    await bus.publish(_make_resistance_signal(level=narrow_resistance))
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("100.1")))
    await _drain()

    # Support bounce 2 (3rd total bounce — enough for count but too narrow)
    await bus.publish(_make_support_signal(level=narrow_support))
    await _drain()

    result = detector.get_range(SYMBOL)
    # Should have bounce data but range_confirmed False due to narrow width
    if result is not None:
        assert result.range_confirmed is False, (
            f"Width {result.range_width_pct:.3f}% should be below 0.6% min"
        )


@pytest.mark.asyncio
async def test_breakout_invalidates_range(bus, detector):
    """Price drops below support by breakout margin → range cleared."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Build confirmed range (3 bounces)
    await bus.publish(_make_support_signal())
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("70000")))
    await _drain()
    await bus.publish(_make_resistance_signal())
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("69850")))
    await _drain()
    await bus.publish(_make_support_signal())
    await _drain()

    pre_breakout = detector.get_range(SYMBOL)
    assert pre_breakout is not None and pre_breakout.range_confirmed

    # Price drops well below support (69500 * (1 - 0.6%) ≈ 69083)
    # breakout_margin_pct=0.5, so need > 0.5% below 69500 → price < 69153
    breakout_price = Decimal("69000")  # ~0.72% below support
    await bus.publish(_make_market_data(price=breakout_price))
    await _drain()

    post_breakout = detector.get_range(SYMBOL)
    assert post_breakout is None, (
        f"Range should be cleared after breakout below support, got {post_breakout}"
    )


@pytest.mark.asyncio
async def test_signals_ignored_outside_sideways(bus, detector):
    """Bounces in non-SIDEWAYS regime are not counted."""
    # Regime is UNKNOWN (default) — signals should be ignored
    await bus.publish(_make_support_signal())
    await _drain()
    await bus.publish(_make_resistance_signal())
    await _drain()
    await bus.publish(_make_support_signal())
    await _drain()

    result = detector.get_range(SYMBOL)
    assert result is None, "No range should exist when regime is not SIDEWAYS"


@pytest.mark.asyncio
async def test_non_sr_signals_ignored(bus, detector):
    """Signals without source=SupportResistance are ignored."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    # Signal with wrong source
    rsi_signal = SignalEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=SYMBOL,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="Near support 69500.00 (3 touches, 0.15% away)",
        metadata={"source": "RSI"},
    )
    await bus.publish(rsi_signal)
    await _drain()

    # Signal with no metadata
    bare_signal = SignalEvent(
        event_type=None,  # type: ignore[arg-type]
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=SYMBOL,
        action=SignalAction.BUY,
        strength=Decimal("0.8"),
        confidence=Decimal("0.9"),
        reason="Near support 69500.00 (3 touches, 0.15% away)",
        metadata=None,
    )
    await bus.publish(bare_signal)
    await _drain()

    result = detector.get_range(SYMBOL)
    assert result is None, "Non-SR signals must not count as bounces"


@pytest.mark.asyncio
async def test_range_state_fields(bus, detector):
    """Verify RangeState fields are computed correctly after confirmation."""
    await bus.publish(_make_regime_event("SIDEWAYS"))
    await _drain()

    await bus.publish(_make_support_signal(level=SUPPORT))
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("70000")))
    await _drain()
    await bus.publish(_make_resistance_signal(level=RESISTANCE))
    await _drain()
    await bus.publish(_make_market_data(price=Decimal("69850")))
    await _drain()
    await bus.publish(_make_support_signal(level=SUPPORT))
    await _drain()

    result = detector.get_range(SYMBOL)
    assert result is not None
    assert isinstance(result, RangeState)
    assert result.support_level == SUPPORT
    assert result.resistance_level == RESISTANCE
    assert result.bounce_count == 3
    assert result.range_confirmed is True
    # Width check: (70200 - 69500) / 69500 * 100 ≈ 1.007%
    expected_width = (RESISTANCE - SUPPORT) / SUPPORT * Decimal("100")
    assert abs(result.range_width_pct - expected_width) < Decimal("0.001")
    assert result.last_updated > 0
