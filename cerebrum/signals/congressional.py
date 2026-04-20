"""
Congressional trade signal generator for CerebrumCoin.

Polls the Finnhub /stock/congressional-trading endpoint on a configurable
interval (default 300 s) and publishes SignalEvents to the event bus.

This is NOT a tick-driven signal generator — it does NOT inherit from
SignalGenerator (which is tick-driven via MarketDataEvent subscriptions).
It mirrors the polling pattern used by ``cerebrum/intelligence/news.py``:
spawns a background asyncio Task in ``start()``, loops with ``asyncio.sleep``,
and publishes directly to the bus on new filings.

Network calls are isolated in ``_fetch(symbol)`` (overridable in tests) so
unit tests can inject fixture data without hitting the network.

Options policy (DEC-PELOSI-OPT-001)
-------------------------------------
Finnhub transactionType values observed in practice:

  "Stock Purchase"        → stock_buy  → BUY signal, size_multiplier=1.0
  "Stock Sale (Partial)"  → stock_sell → SELL signal, size_multiplier=1.0
  "Stock Sale (Full)"     → stock_sell → SELL signal, size_multiplier=1.0
  "Call (ST) Purchase"    → call_buy   → BUY signal, size_multiplier=0.5
  "Call (LT) Purchase"    → call_buy   → BUY signal, size_multiplier=0.5
  "Put (ST) Purchase"     → put_buy    → DROP + log options_skipped
  "Put (LT) Purchase"     → put_buy    → DROP + log options_skipped
  "Option Sale"           → option_sell → DROP + log options_skipped
  (anything else)                      → DROP + log options_skipped

# @decision DEC-PELOSI-DATA-001
# @title Finnhub free-tier congressional-trading endpoint
# @status accepted
# @rationale Zero cost, stable JSON contract, filing_id exposed. Non-commercial ToS
# acceptable for paper-only v1. Revisit Quiver if paper validates and we want live.
# Finnhub ToS: https://finnhub.io/terms (non-commercial use permitted on free tier).

# @decision DEC-PELOSI-OPT-001
# @title Call buys → underlying BUY at half size; drop puts and option sells
# @status accepted
# @rationale ~60% of Pelosi's disclosed trades are options. The paper adapter
# executes equities only. Converting call buys to underlying BUY at 0.5×
# position size partially recovers her bullish options volume. Puts and
# option sells are dropped (logged to options_skipped metric) because there
# is no clean equity-side proxy — a put buy is bearish, which would require
# a short or a sell of an existing position we may not hold.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Coroutine, Dict, List, Optional

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOURCE = "Congressional"
_STRENGTH = Decimal("0.75")   # fixed strength for all congressional signals
_CONFIDENCE = Decimal("0.65") # conservative confidence — lag makes certainty lower

# Options policy multipliers (DEC-PELOSI-OPT-001)
_SIZE_STOCK = Decimal("1.0")
_SIZE_CALL_PROXY = Decimal("0.5")

# Finnhub congressional-trading base URL
_FINNHUB_BASE = "https://finnhub.io/api/v1/stock/congressional-trading"

# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------


def _classify_transaction(transaction_type: str) -> tuple[str, SignalAction | None, Decimal]:
    """
    Map a Finnhub transactionType string to (action_code, SignalAction, size_multiplier).

    Returns:
        (action_code, signal_action, size_multiplier) where signal_action is
        None if the filing should be dropped (put buys, option sells, unknown).

    action_code is the normalised string stored in the ledger.
    """
    t = transaction_type.lower()

    if "stock" in t and ("purchase" in t or "buy" in t):
        return ("stock_buy", SignalAction.BUY, _SIZE_STOCK)

    if "stock" in t and "sale" in t:
        return ("stock_sell", SignalAction.SELL, _SIZE_STOCK)

    if "call" in t and ("purchase" in t or "buy" in t):
        return ("call_buy", SignalAction.BUY, _SIZE_CALL_PROXY)

    if "put" in t and ("purchase" in t or "buy" in t):
        return ("put_buy", None, Decimal("0"))

    if "option" in t and "sale" in t:
        return ("option_sell", None, Decimal("0"))

    # Unknown / catch-all
    return ("unknown", None, Decimal("0"))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CongressionalTradeSignal:
    """
    Poll-and-publish congressional trade signal generator.

    Polls Finnhub for each symbol in ``symbols`` on a configurable interval.
    New filings (not yet in the dedup ledger) are transformed via the options
    policy and published as SignalEvents to the event bus.

    Features:
    - Configurable poll interval (default 300 s)
    - Per-symbol polling so we fetch only the universe we care about
    - Options policy transformer (DEC-PELOSI-OPT-001)
    - Dedup via CongressionalLedger (prevents double-emission across restarts)
    - Injectable ``_fetch`` method — override in tests to inject fixtures
    - Graceful error handling: one failed poll does not stop the loop

    This class does NOT inherit from SignalGenerator because SignalGenerator is
    tick-driven (subscribes to MarketDataEvents). Congressional signals are
    event-driven off an external HTTP poll — a fundamentally different cadence.
    """

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str],
        api_key: str = "",
        poll_interval_seconds: int = 300,
        ledger: Any = None,
    ) -> None:
        """
        Initialize the congressional trade signal generator.

        Args:
            bus: Event bus for publishing SignalEvents.
            symbols: List of ticker symbols to poll (e.g. ["NVDA", "AVGO"]).
            api_key: Finnhub API key. Empty string disables live polling
                     (generator will log a warning and not start tasks).
            poll_interval_seconds: Seconds between polls per symbol.
                                   Default 300 (5 min). At 60 symbols and
                                   ~60 free-tier calls/min this is safe.
            ledger: Optional CongressionalLedger instance.  When None, a
                    real ledger is created (connects to data/cerebrum.db).
                    Inject a custom ledger in tests to use an in-memory DB.
        """
        self._bus = bus
        self._symbols = list(symbols)
        self._api_key = api_key
        self._interval = poll_interval_seconds

        # Dedup ledger — import lazily to allow test injection before import
        if ledger is None:
            from cerebrum.data.congressional_ledger import CongressionalLedger
            self._ledger = CongressionalLedger()
        else:
            self._ledger = ledger

        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._options_skipped: int = 0

        self._log = logger.bind(
            component="congressional_trade_signal",
            symbols=self._symbols,
            poll_interval=poll_interval_seconds,
        )

        if not self._api_key:
            self._log.warning(
                "congressional_signal_no_api_key",
                message="No Finnhub API key provided. Live polling disabled.",
            )
        else:
            self._log.info("congressional_signal_initialized")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start per-symbol polling tasks."""
        if self._running:
            return

        if not self._api_key:
            self._log.warning(
                "congressional_signal_start_skipped",
                reason="no API key — skipping polling tasks",
            )
            return

        self._running = True
        for symbol in self._symbols:
            task = asyncio.create_task(self._poll_loop(symbol))
            self._tasks.append(task)
            self._log.info(
                "congressional_poll_started",
                symbol=symbol,
                interval=self._interval,
            )

    async def stop(self) -> None:
        """Cancel all polling tasks and wait for them to finish."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._log.info("congressional_signal_stopped")

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self, symbol: str) -> None:
        """Continuously poll for a single symbol on ``_interval`` cadence."""
        while self._running:
            try:
                await self._process_symbol(symbol)
            except Exception as exc:
                self._log.error(
                    "congressional_poll_failed",
                    symbol=symbol,
                    error=str(exc),
                )
            await asyncio.sleep(self._interval)

    async def _process_symbol(self, symbol: str) -> None:
        """Fetch and process all filings for a symbol, emitting new ones."""
        filings: list[dict[str, Any]] = await self._fetch(symbol)

        for filing in filings:
            await self._handle_filing(symbol, filing)

    # ------------------------------------------------------------------
    # Network stub — override in tests
    # ------------------------------------------------------------------

    async def _fetch(self, symbol: str) -> list[dict[str, Any]]:
        """
        Fetch congressional-trading filings for ``symbol`` from Finnhub.

        Returns a list of filing dicts with at least:
            id             — str, unique filing identifier
            transactionDate — str, ISO-8601 date (e.g. "2026-03-01")
            filingDate      — str, ISO-8601 date of public disclosure
            transactionType — str, e.g. "Stock Purchase"

        Subclass or monkeypatch this method in tests to inject fixture data
        without making live network calls.
        """
        import aiohttp  # optional at import time — only imported when polling

        url = _FINNHUB_BASE
        params = {
            "symbol": symbol,
            "token": self._api_key,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        self._log.warning(
                            "congressional_http_error",
                            symbol=symbol,
                            status=resp.status,
                        )
                        return []

                    data = await resp.json()
                    # Finnhub wraps results in a "data" key
                    return data.get("data", [])

            except asyncio.TimeoutError:
                self._log.warning("congressional_timeout", symbol=symbol)
                return []

    # ------------------------------------------------------------------
    # Filing handler
    # ------------------------------------------------------------------

    async def _handle_filing(
        self,
        symbol: str,
        filing: dict[str, Any],
    ) -> bool:
        """
        Process a single filing dict and emit a SignalEvent if appropriate.

        Returns True if a signal was emitted, False if dropped (dedup, options
        policy, or missing data).
        """
        filing_id: str = filing.get("id", "")
        transaction_type: str = filing.get("transactionType", "")
        transaction_date: str = filing.get("transactionDate", "")
        filing_date: str = filing.get("filingDate", "")

        if not filing_id:
            self._log.warning(
                "congressional_filing_no_id",
                symbol=symbol,
                transaction_type=transaction_type,
            )
            return False

        # Dedup check — skip if we've already emitted this filing
        if self._ledger.has_seen(filing_id):
            return False

        # Options policy classification
        action_code, signal_action, size_multiplier = _classify_transaction(
            transaction_type
        )

        if signal_action is None:
            # Log as skipped (put buys, option sells, unknowns)
            self._options_skipped += 1
            self._log.info(
                "options_skipped",
                filing_id=filing_id,
                symbol=symbol,
                transaction_type=transaction_type,
                action_code=action_code,
                total_skipped=self._options_skipped,
            )
            # Record in ledger so we don't log it on every poll
            self._ledger.record(filing_id, symbol, filing_date, action_code)
            return False

        # Record in ledger before emitting — prevents duplicate on restart
        is_new = self._ledger.record(filing_id, symbol, filing_date, action_code)
        if not is_new:
            # Race: another process recorded it between has_seen() and record()
            return False

        # Build and publish the signal
        signal = self._build_signal(
            symbol=symbol,
            action=signal_action,
            size_multiplier=size_multiplier,
            filing_id=filing_id,
            filing_date=filing_date,
            transaction_date=transaction_date,
            transaction_type=transaction_type,
        )
        await self._bus.publish(signal)

        self._log.info(
            "congressional_signal_emitted",
            filing_id=filing_id,
            symbol=symbol,
            action=signal_action.value,
            size_multiplier=str(size_multiplier),
            filing_date=filing_date,
        )
        return True

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        action: SignalAction,
        size_multiplier: Decimal,
        filing_id: str,
        filing_date: str,
        transaction_date: str,
        transaction_type: str,
    ) -> SignalEvent:
        """
        Construct a SignalEvent for a congressional filing.

        metadata["source"] = "Congressional" enables signal_source_filter
        routing so only the pelosi_follow strategy aggregator receives these.

        metadata["filing_id"] and metadata["filing_date"] are carried through
        so downstream rules (e.g. StalenessGateRule) can read them without
        needing to re-query the ledger.
        """
        # Adjust strength by the options-proxy size multiplier so the position
        # sizer naturally produces a smaller order for call proxies (0.5×).
        effective_strength = _STRENGTH * size_multiplier

        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=datetime.now(timezone.utc).timestamp(),
            signal_type=SignalType.NEWS,   # No dedicated CONGRESSIONAL type — NEWS is closest
            symbol=symbol,
            action=action,
            strength=effective_strength,
            confidence=_CONFIDENCE,
            reason=(
                f"Congressional {transaction_type} by Pelosi "
                f"(filing_id={filing_id}, filed={filing_date})"
            ),
            metadata={
                "source": _SOURCE,
                "filing_id": filing_id,
                "filing_date": filing_date,
                "transaction_date": transaction_date,
                "transaction_type": transaction_type,
                "size_multiplier": str(size_multiplier),
            },
        )
