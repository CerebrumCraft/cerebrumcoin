"""
Unit tests for CongressionalTradeSignal.

All tests use fixture JSON under tests/fixtures/congressional/ — no network
calls are made. The ``_fetch`` method is overridden on the signal instance to
return fixture data, following the pattern established in test_news.py where
only external HTTP calls are mocked (Sacred Practice #5).

Coverage:
  1. stock_buy fixture → BUY signal emitted with size_multiplier=1.0
  2. stock_sell fixture → SELL signal emitted
  3. call_buy fixture → BUY signal emitted with size_multiplier=0.5 (half strength)
  4. put_buy fixture → no signal emitted; options_skipped incremented
  5. source tag = "Congressional" on all emitted signals
  6. filing_id carried in metadata
  7. filing_date carried in metadata
  8. Dedup: second call with same filing_id returns False and does NOT re-emit
  9. start() is a no-op without API key
 10. Multiple symbols polled independently
 11. _classify_transaction edge cases: partial sale, full sale, LT call
 12. Empty response → no signals
"""

# @mock-exempt: Mocking _fetch (external HTTP stub) and CongressionalLedger
# (SQLite external boundary) per Sacred Practice #5.

from __future__ import annotations

import asyncio
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction
from cerebrum.signals.congressional import CongressionalTradeSignal, _classify_transaction

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures" / "congressional"


def _load(name: str) -> list[dict[str, Any]]:
    """Load a fixture file and return the list of filings."""
    return json.loads((FIXTURES / name).read_text())


def _make_ledger() -> MagicMock:
    """Return an in-memory mock ledger that starts empty."""
    ledger = MagicMock()
    seen: set[str] = set()

    def has_seen(filing_id: str) -> bool:
        return filing_id in seen

    def record(filing_id: str, symbol: str, filing_date: str, action: str) -> bool:
        if filing_id in seen:
            return False
        seen.add(filing_id)
        return True

    ledger.has_seen.side_effect = has_seen
    ledger.record.side_effect = record
    return ledger


@pytest.fixture
async def bus() -> EventBus:
    b = EventBus(queue_size=1000)
    await b.start()
    yield b
    await b.stop()


def _make_signal_gen(
    bus: EventBus,
    filings: list[dict[str, Any]],
    symbols: list[str] | None = None,
    ledger: Any = None,
) -> CongressionalTradeSignal:
    """
    Build a CongressionalTradeSignal with ``_fetch`` overridden to return
    ``filings`` for every symbol. Uses a mock ledger by default.
    """
    gen = CongressionalTradeSignal(
        bus=bus,
        symbols=symbols or ["NVDA"],
        api_key="test-key",
        ledger=ledger or _make_ledger(),
    )

    async def _fake_fetch(symbol: str) -> list[dict[str, Any]]:
        return filings

    gen._fetch = _fake_fetch  # type: ignore[method-assign]
    return gen


# ---------------------------------------------------------------------------
# 1. stock_buy → BUY signal, size_multiplier 1.0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stock_buy_emits_buy_signal(bus: EventBus) -> None:
    """stock_buy.json should produce a BUY SignalEvent."""
    filings = _load("stock_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings)
    await gen._process_symbol("NVDA")
    await asyncio.sleep(0.05)  # let bus queue drain

    assert len(received) == 1
    sig = received[0]
    assert sig.action == SignalAction.BUY
    assert sig.symbol == "NVDA"


# ---------------------------------------------------------------------------
# 2. stock_sell → SELL signal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stock_sell_emits_sell_signal(bus: EventBus) -> None:
    """stock_sell.json should produce a SELL SignalEvent."""
    filings = _load("stock_sell.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings, symbols=["MSFT"])
    await gen._process_symbol("MSFT")
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].action == SignalAction.SELL
    assert received[0].symbol == "MSFT"


# ---------------------------------------------------------------------------
# 3. call_buy → BUY signal at half strength
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_buy_emits_buy_at_half_strength(bus: EventBus) -> None:
    """call_buy.json should produce a BUY with strength = 0.75 * 0.5 = 0.375."""
    filings = _load("call_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings, symbols=["AVGO"])
    await gen._process_symbol("AVGO")
    await asyncio.sleep(0.05)

    assert len(received) == 1
    sig = received[0]
    assert sig.action == SignalAction.BUY
    # Strength = base(0.75) * call_proxy_multiplier(0.5)
    assert sig.strength == Decimal("0.375")
    # size_multiplier recorded in metadata
    assert sig.metadata is not None
    assert sig.metadata["size_multiplier"] == "0.5"


# ---------------------------------------------------------------------------
# 4. put_buy → no signal emitted; options_skipped counter incremented
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_buy_dropped_not_emitted(bus: EventBus) -> None:
    """put_buy.json should produce no SignalEvent; options_skipped should be 1."""
    filings = _load("put_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings, symbols=["AAPL"])
    await gen._process_symbol("AAPL")
    await asyncio.sleep(0.05)

    assert len(received) == 0
    assert gen._options_skipped == 1


# ---------------------------------------------------------------------------
# 5. source tag = "Congressional"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signal_source_tag_is_congressional(bus: EventBus) -> None:
    """All emitted signals must carry metadata['source'] == 'Congressional'."""
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    for fixture, symbol in [("stock_buy.json", "NVDA"), ("stock_sell.json", "MSFT"), ("call_buy.json", "AVGO")]:
        filings = _load(fixture)
        gen = _make_signal_gen(bus, filings, symbols=[symbol])
        await gen._process_symbol(symbol)

    await asyncio.sleep(0.05)

    assert len(received) == 3
    for sig in received:
        assert sig.metadata is not None
        assert sig.metadata["source"] == "Congressional"


# ---------------------------------------------------------------------------
# 6. filing_id in metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filing_id_in_metadata(bus: EventBus) -> None:
    """filing_id from fixture must appear in signal metadata."""
    filings = _load("stock_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings)
    await gen._process_symbol("NVDA")
    await asyncio.sleep(0.05)

    assert received[0].metadata is not None
    assert received[0].metadata["filing_id"] == "filing-001"


# ---------------------------------------------------------------------------
# 7. filing_date in metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filing_date_in_metadata(bus: EventBus) -> None:
    """filing_date from fixture must appear in signal metadata."""
    filings = _load("stock_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings)
    await gen._process_symbol("NVDA")
    await asyncio.sleep(0.05)

    assert received[0].metadata is not None
    assert received[0].metadata["filing_date"] == "2026-04-01"


# ---------------------------------------------------------------------------
# 8. Dedup: second call with same filing_id does not re-emit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_second_call_no_signal(bus: EventBus) -> None:
    """Processing the same filing twice must emit only one signal (no duplicate)."""
    filings = _load("stock_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    ledger = _make_ledger()
    gen = _make_signal_gen(bus, filings, ledger=ledger)

    await gen._process_symbol("NVDA")
    await gen._process_symbol("NVDA")  # second pass — same fixture, same filing_id
    await asyncio.sleep(0.05)

    assert len(received) == 1, (
        f"Expected exactly 1 signal but got {len(received)} — dedup failed"
    )


# ---------------------------------------------------------------------------
# 9. start() no-op without API key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_noop_without_api_key(bus: EventBus) -> None:
    """start() must not spawn any tasks when api_key is empty."""
    gen = CongressionalTradeSignal(
        bus=bus,
        symbols=["NVDA"],
        api_key="",
        ledger=_make_ledger(),
    )
    await gen.start()
    assert len(gen._tasks) == 0
    assert not gen._running


# ---------------------------------------------------------------------------
# 10. Multiple distinct filings processed independently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_filings_all_emitted(bus: EventBus) -> None:
    """stock_buy + call_buy together should emit 2 signals."""
    filings = _load("stock_buy.json") + _load("call_buy.json")
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, filings, symbols=["NVDA"])
    await gen._process_symbol("NVDA")
    await asyncio.sleep(0.05)

    # Both should emit; stock_buy is BUY@1.0, call_buy is BUY@0.5
    assert len(received) == 2
    actions = {s.action for s in received}
    assert actions == {SignalAction.BUY}


# ---------------------------------------------------------------------------
# 11. _classify_transaction edge cases
# ---------------------------------------------------------------------------

def test_classify_partial_sale() -> None:
    action, sig_action, mult = _classify_transaction("Stock Sale (Partial)")
    assert action == "stock_sell"
    assert sig_action == SignalAction.SELL
    assert mult == Decimal("1.0")


def test_classify_full_sale() -> None:
    action, sig_action, mult = _classify_transaction("Stock Sale (Full)")
    assert action == "stock_sell"
    assert sig_action == SignalAction.SELL


def test_classify_lt_call_buy() -> None:
    action, sig_action, mult = _classify_transaction("Call (LT) Purchase")
    assert action == "call_buy"
    assert sig_action == SignalAction.BUY
    assert mult == Decimal("0.5")


def test_classify_put_dropped() -> None:
    action, sig_action, mult = _classify_transaction("Put (ST) Purchase")
    assert action == "put_buy"
    assert sig_action is None


def test_classify_option_sale_dropped() -> None:
    action, sig_action, mult = _classify_transaction("Option Sale")
    assert action == "option_sell"
    assert sig_action is None


# ---------------------------------------------------------------------------
# 12. Empty response → no signals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_response_no_signals(bus: EventBus) -> None:
    """An empty Finnhub response should produce zero signals."""
    received: list[SignalEvent] = []

    async def _capture(event: Any) -> None:
        if isinstance(event, SignalEvent):
            received.append(event)

    bus.subscribe(EventType.SIGNAL, _capture, "test_capture")

    gen = _make_signal_gen(bus, [], symbols=["NVDA"])
    await gen._process_symbol("NVDA")

    assert len(received) == 0
