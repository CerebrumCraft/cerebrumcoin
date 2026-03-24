"""
End-to-end integration tests for the range trading feature.

Proves that all three components of the range trading pipeline interact
correctly as a system:

1. SidewaysSuppressionRule with exempt_strategies={"range_trading"} —
   range_trading signals pass through while momentum signals are blocked.
2. SignalAggregator with signal_source_filter="SupportResistance" —
   RSI/MACD signals are dropped before reaching the buffer.
3. RangeDetector — confirmed range after 3 S/R bounces with inter-bounce
   zone exits.

These are system-level integration tests, not unit tests. Each test wires
together multiple real components via a shared EventBus and verifies the
cross-component behavior. The unit tests for each component live in
test_sideways_suppression.py, test_signal_source_metadata.py, and
test_range_detector.py respectively.

@decision DEC-TEST-RANGE-INT-001
@title Integration tests prove cross-component range trading wiring
@status accepted
@rationale Unit tests verify each component in isolation. Integration tests
verify that SidewaysSuppressionRule + SignalAggregator + RangeDetector interact
correctly when wired to a shared EventBus. The integration test proves:
(a) exempt_strategies bypass survives the full evaluation call path with real
    OrderEvent construction, not just the signal path;
(b) signal_source_filter drop occurs on the bus-delivered event, not just via
    direct handler calls;
(c) RangeDetector bounce counting works end-to-end via bus-published S/R signals
    and MarketDataEvents for zone exits, confirming the full delivery pipeline.
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
    Side,
    SignalAction,
    SignalType,
)
from cerebrum.risk.rules import RuleDecision, SidewaysSuppressionRule
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.strategies.range_detector import RangeDetector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_regime_event(
    to_regime: str,
    symbol: str = "BTC/USD",
    from_regime: str = "UNKNOWN",
    confidence: str = "0.7",
) -> RegimeChangeEvent:
    """Construct a RegimeChangeEvent. symbol is stored in indicators for SidewaysSuppressionRule."""
    return RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=time.time(),
        from_regime=from_regime,
        to_regime=to_regime,
        confidence=Decimal(confidence),
        indicators={"symbol": symbol},
    )


def _make_market_data(symbol: str, price: Decimal) -> MarketDataEvent:
    return MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=time.time(),
        symbol=symbol,
        price=price,
        volume=Decimal("1.0"),
    )


def _make_buy_order(symbol: str = "BTC/USD") -> OrderEvent:
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


def _make_signal(
    symbol: str = "BTC/USD",
    strategy_id: str | None = None,
    source: str | None = None,
    action: SignalAction = SignalAction.BUY,
) -> SignalEvent:
    """Build a SignalEvent with optional strategy_id and metadata source."""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=action,
        strength=Decimal("0.8"),
        confidence=Decimal("0.7"),
        metadata={"source": source} if source else None,
        strategy_id=strategy_id,
    )


def _make_sr_support_signal(
    symbol: str = "BTC/USD",
    level: Decimal = Decimal("69500"),
    touches: int = 3,
    distance_pct: str = "0.15",
) -> SignalEvent:
    """Build an S/R BUY signal near support, formatted for RangeDetector parsing."""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.BUY,
        strength=Decimal("0.7"),
        confidence=Decimal("0.8"),
        reason=f"Near support {level} ({touches} touches, {distance_pct}% away)",
        metadata={"source": "SupportResistance"},
    )


def _make_sr_resistance_signal(
    symbol: str = "BTC/USD",
    level: Decimal = Decimal("70200"),
    touches: int = 2,
    distance_pct: str = "0.10",
) -> SignalEvent:
    """Build an S/R SELL signal near resistance, formatted for RangeDetector parsing."""
    return SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=time.time(),
        signal_type=SignalType.TECHNICAL,
        symbol=symbol,
        action=SignalAction.SELL,
        strength=Decimal("0.7"),
        confidence=Decimal("0.8"),
        reason=f"Near resistance {level} ({touches} touches, {distance_pct}% away)",
        metadata={"source": "SupportResistance"},
    )


async def _drain(seconds: float = 0.15) -> None:
    """Allow the EventBus to drain its queue and deliver pending events."""
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def bus():
    """Real EventBus started and stopped around each test."""
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


# ---------------------------------------------------------------------------
# Test 1: SidewaysSuppressionRule exempts range_trading, blocks momentum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_range_trading_exempted_from_sideways_suppression(bus):
    """
    Range trading strategy can enter trades in SIDEWAYS while momentum is blocked.

    Integration proof:
    - SidewaysSuppressionRule is wired to the bus via subscriptions
    - Regime is set to SIDEWAYS via bus-published RegimeChangeEvent
    - Flat prices are published via bus-published MarketDataEvents
    - An evaluate() call with strategy_id="range_trading" → APPROVE
    - An evaluate() call with strategy_id="momentum" → DENY
    - Both evaluated on the same order under identical market conditions
    """
    rule = SidewaysSuppressionRule(
        min_range_pct=Decimal("1.0"),
        window_size=10,
        bus=bus,
        exempt_strategies={"range_trading"},
    )

    # Publish SIDEWAYS regime via bus (tests bus subscription wiring)
    await bus.publish(_make_regime_event("SIDEWAYS", symbol="BTC/USD"))
    await _drain()

    # Publish flat prices — range = 0%, below the 1.0% threshold
    for _ in range(10):
        await bus.publish(_make_market_data("BTC/USD", Decimal("50000")))
    await _drain()

    order = _make_buy_order("BTC/USD")

    # range_trading is in exempt_strategies — must APPROVE even in SIDEWAYS+flat
    range_signal = _make_signal(strategy_id="range_trading")
    range_result = rule.evaluate(range_signal, order, portfolio=None)  # type: ignore[arg-type]

    assert range_result.decision == RuleDecision.APPROVE, (
        f"range_trading strategy must be exempt from SIDEWAYS suppression, "
        f"got {range_result.decision}: {range_result.reason}"
    )
    assert "range_trading" in range_result.reason, (
        f"Exemption reason must identify the exempt strategy, got: {range_result.reason}"
    )

    # momentum is NOT in exempt_strategies — must DENY under SIDEWAYS+flat
    momentum_signal = _make_signal(strategy_id="momentum")
    momentum_result = rule.evaluate(momentum_signal, order, portfolio=None)  # type: ignore[arg-type]

    assert momentum_result.decision == RuleDecision.DENY, (
        f"momentum strategy must be blocked by SIDEWAYS suppression, "
        f"got {momentum_result.decision}: {momentum_result.reason}"
    )
    assert "SIDEWAYS" in momentum_result.reason, (
        f"Denial reason must reference SIDEWAYS regime, got: {momentum_result.reason}"
    )


# ---------------------------------------------------------------------------
# Test 2: SignalAggregator source filter drops non-SR signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_source_filter_drops_non_sr_signals(bus):
    """
    Range trading aggregator only processes SupportResistance signals.

    Integration proof:
    - SignalAggregator is wired to the bus with signal_source_filter="SupportResistance"
    - A signal published with metadata={"source": "RSI"} must not appear in the buffer
    - A signal published with metadata={"source": "MACD"} must not appear in the buffer
    - A signal published with metadata={"source": "SupportResistance"} must appear in the buffer

    This tests bus-delivered filtering (events flow through the bus queue and
    handler before reaching the buffer), not just direct handler invocation.
    """
    agg = SignalAggregator(
        bus,
        signal_source_filter="SupportResistance",
        strategy_id="range_trading",
        threshold=Decimal("0.1"),  # low threshold — accepted signals would aggregate
    )

    # Publish RSI signal — must be dropped
    rsi_signal = _make_signal(source="RSI", action=SignalAction.BUY)
    await bus.publish(rsi_signal)
    await _drain()

    assert len(agg._signal_buffer["BTC/USD"]) == 0, (
        f"RSI signal must not enter the buffer, "
        f"buffer has {len(agg._signal_buffer['BTC/USD'])} entries"
    )

    # Publish MACD signal — must be dropped
    macd_signal = _make_signal(source="MACD", action=SignalAction.BUY)
    await bus.publish(macd_signal)
    await _drain()

    assert len(agg._signal_buffer["BTC/USD"]) == 0, (
        f"MACD signal must not enter the buffer, "
        f"buffer has {len(agg._signal_buffer['BTC/USD'])} entries"
    )

    # Publish SupportResistance signal — must be accepted
    sr_signal = _make_signal(source="SupportResistance", action=SignalAction.BUY)
    await bus.publish(sr_signal)
    await _drain()

    assert len(agg._signal_buffer["BTC/USD"]) == 1, (
        f"SupportResistance signal must enter the buffer, "
        f"buffer has {len(agg._signal_buffer['BTC/USD'])} entries"
    )

    # Confirm the buffer holds exactly the S/R signal (not the dropped ones)
    buffered = agg._signal_buffer["BTC/USD"][0]
    assert buffered.metadata is not None
    assert buffered.metadata.get("source") == "SupportResistance", (
        f"Buffered signal source must be 'SupportResistance', "
        f"got {buffered.metadata.get('source')}"
    )


# ---------------------------------------------------------------------------
# Test 3: RangeDetector confirms range after 3 S/R bounces via bus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_range_detector_confirms_range_after_bounces(bus):
    """
    RangeDetector confirms a range after 3 S/R bounces in SIDEWAYS.

    Integration proof:
    - RangeDetector subscribes to the bus via start()
    - Regime is set to SIDEWAYS via bus-published RegimeChangeEvent
    - 2 support bounces + 1 resistance bounce = 3 total bounces
    - Between bounces, market data is published so price "leaves" the zone,
      allowing the next signal to be counted (deduplication proof)
    - get_range() returns a confirmed RangeState with bounce_count == 3

    This proves the full event delivery pipeline: bus publish → queue → handler
    → state update → get_range() query.
    """
    support = Decimal("69500")
    resistance = Decimal("70200")
    # Range width = (70200 - 69500) / 69500 * 100 ≈ 1.007% → above min_range_width_pct=0.6%
    symbol = "BTC/USD"

    detector = RangeDetector(
        bus,
        min_bounces=3,
        min_range_width_pct=Decimal("0.6"),
        breakout_margin_pct=Decimal("0.5"),
        level_staleness_minutes=120,
    )
    await detector.start()

    # Set regime to SIDEWAYS — RangeDetector only counts bounces in SIDEWAYS
    await bus.publish(_make_regime_event("SIDEWAYS", symbol=symbol))
    await _drain()

    # --- Support bounce 1 ---
    # Price enters support proximity zone
    await bus.publish(_make_sr_support_signal(symbol=symbol, level=support))
    await _drain()

    # Price leaves support zone (distance > 0.5% threshold for zone exit)
    # 69500 + 1% = 70195 — comfortably outside the 0.5% zone
    await bus.publish(_make_market_data(symbol, Decimal("70195")))
    await _drain()

    # After only 1 support bounce, resistance is still unknown — get_range() returns None.
    # This is correct: RangeDetector requires both support AND resistance to be known
    # before it can return any RangeState snapshot. We verify the bounce was recorded
    # by checking the cumulative count once resistance is also observed (below).

    # --- Resistance bounce 1 ---
    # Price enters resistance proximity zone (near 70200)
    await bus.publish(_make_sr_resistance_signal(symbol=symbol, level=resistance))
    await _drain()

    # Price leaves resistance zone back toward the middle
    await bus.publish(_make_market_data(symbol, Decimal("69850")))
    await _drain()

    # Now both support and resistance are known — get_range() must return a partial range
    partial2 = detector.get_range(symbol)
    assert partial2 is not None, (
        "Range state must be available after 1 support + 1 resistance bounce"
    )
    # The 1 support bounce from before + 1 resistance bounce = 2 total
    assert partial2.bounce_count == 2, f"Expected 2 bounces, got {partial2.bounce_count}"
    assert partial2.range_confirmed is False, "Range not yet confirmed at 2 bounces"

    # --- Support bounce 2 (3rd total bounce — triggers confirmation) ---
    # Price enters support proximity zone again
    await bus.publish(_make_sr_support_signal(symbol=symbol, level=support))
    await _drain()

    confirmed = detector.get_range(symbol)
    assert confirmed is not None, (
        "get_range() must return a RangeState after 3 bounces"
    )
    assert confirmed.bounce_count == 3, (
        f"Expected 3 total bounces (2 support + 1 resistance), got {confirmed.bounce_count}"
    )
    assert confirmed.range_confirmed is True, (
        f"Range must be confirmed after 3 bounces with sufficient width, "
        f"range_confirmed={confirmed.range_confirmed}, width={confirmed.range_width_pct:.3f}%"
    )
    assert confirmed.support_level == support, (
        f"Expected support_level={support}, got {confirmed.support_level}"
    )
    assert confirmed.resistance_level == resistance, (
        f"Expected resistance_level={resistance}, got {confirmed.resistance_level}"
    )
    # Width sanity check
    expected_width = (resistance - support) / support * Decimal("100")
    assert abs(confirmed.range_width_pct - expected_width) < Decimal("0.01"), (
        f"Range width {confirmed.range_width_pct:.4f}% should be ~{expected_width:.4f}%"
    )
