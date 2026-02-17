"""
Social sentiment indicators for CerebrumCoin.

Fetches Fear & Greed Index and other social sentiment signals.

@decision DEC-INT-001
@title Graceful degradation for social sentiment
@status accepted
@rationale Fear & Greed Index is a free public API with no authentication.
If it fails, system continues with technical signals only. Contrarian strategy:
high fear (low index) = buying opportunity, high greed (high index) = selling signal.
"""

import asyncio
from decimal import Decimal
from time import time

import aiohttp
import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType

logger = structlog.get_logger()


class FearGreedSentiment:
    """
    Fear & Greed Index sentiment signal generator.
    
    Uses alternative.me API (free, no auth required).
    Contrarian strategy: fear = buy, greed = sell.
    """

    def __init__(
        self,
        bus: EventBus,
        poll_interval: int = 3600,  # 1 hour
        symbol: str = "BTC/USD",
    ) -> None:
        """
        Initialize Fear & Greed sentiment tracker.

        Args:
            bus: Event bus for publishing SignalEvents
            poll_interval: Seconds between polls
            symbol: Symbol to emit signals for (default: BTC/USD)
        """
        self._bus = bus
        self._poll_interval = poll_interval
        self._symbol = symbol
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._log = logger.bind(component="fear_greed")

    async def start(self) -> None:
        """Start polling Fear & Greed Index."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        self._log.info("fear_greed_polling_started", interval=self._poll_interval)

    async def stop(self) -> None:
        """Stop polling."""
        self._running = False

        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        self._log.info("fear_greed_stopped")

    async def _poll_loop(self) -> None:
        """Poll Fear & Greed Index in a loop."""
        while self._running:
            try:
                await self._fetch_and_emit()
            except Exception as e:
                self._log.error("fear_greed_fetch_error", error=str(e))

            await asyncio.sleep(self._poll_interval)

    async def _fetch_and_emit(self) -> None:
        """Fetch Fear & Greed Index and emit signal."""
        url = "https://api.alternative.me/fng/"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        self._log.warning("fear_greed_http_error", status=resp.status)
                        return

                    data = await resp.json()
                    
                    if not data.get("data"):
                        return

                    # Extract latest reading
                    latest = data["data"][0]
                    value = int(latest.get("value", 50))  # 0-100
                    classification = latest.get("value_classification", "Neutral")

                    # Convert to signal (contrarian strategy)
                    # 0-25 = Extreme Fear -> Strong BUY
                    # 25-45 = Fear -> BUY
                    # 45-55 = Neutral -> HOLD
                    # 55-75 = Greed -> SELL
                    # 75-100 = Extreme Greed -> Strong SELL

                    if value < 25:
                        action = SignalAction.BUY
                        strength = Decimal("0.8")
                    elif value < 45:
                        action = SignalAction.BUY
                        strength = Decimal("0.5")
                    elif value < 55:
                        action = SignalAction.HOLD
                        strength = Decimal("0.0")
                    elif value < 75:
                        action = SignalAction.SELL
                        strength = Decimal("0.5")
                    else:
                        action = SignalAction.SELL
                        strength = Decimal("0.8")

                    # Confidence based on how extreme the reading is
                    # More extreme = higher confidence
                    distance_from_neutral = abs(value - 50)
                    confidence = Decimal(str(min(1.0, distance_from_neutral / 50.0)))

                    event = SignalEvent(
                        event_type=EventType.SIGNAL,
                        timestamp=time(),
                        signal_type=SignalType.SENTIMENT,
                        symbol=self._symbol,
                        action=action,
                        strength=strength,
                        confidence=confidence,
                        reason=f"Fear & Greed Index: {value} ({classification})",
                    )

                    await self._bus.publish(event)

                    self._log.info(
                        "fear_greed_signal_emitted",
                        value=value,
                        classification=classification,
                        action=action.value,
                        strength=str(strength),
                    )

            except asyncio.TimeoutError:
                self._log.warning("fear_greed_timeout")
            except Exception as e:
                self._log.error("fear_greed_error", error=str(e))
