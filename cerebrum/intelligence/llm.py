"""
LLM-powered news reasoning for CerebrumCoin.

Uses Claude to interpret batches of news articles and generate trading signals.

@decision DEC-INT-002
@title Async LLM news analysis with batching
@status accepted
@rationale Batching news articles reduces API calls and cost. Rate limiting prevents
excessive spending. Claude SDK provides structured output for reliable signal extraction.

@decision DEC-INT-004
@title Rate limiting and cost control for LLM
@status accepted
@rationale LLM calls are expensive. Rate limit (default: 10/hour) prevents runaway costs.
Use claude-haiku-4-5 for cost efficiency. Batch news items to get more value per call.
"""

import asyncio
import json
from collections import deque
from decimal import Decimal
from time import time
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, NewsEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType

logger = structlog.get_logger()


class LLMNewsAnalyzer:
    """
    Claude-powered news analysis and signal generation.
    
    Features:
    - Batches news articles to reduce API calls
    - Rate limiting to control costs
    - Structured output for reliable signal extraction
    - Graceful degradation if API key missing
    """

    def __init__(
        self,
        bus: EventBus,
        anthropic_api_key: str = "",
        model: str = "claude-haiku-4-5",
        max_calls_per_hour: int = 10,
        batch_size: int = 5,
        batch_window_seconds: int = 300,  # 5 minutes
        timeout_seconds: int = 30,
    ) -> None:
        """
        Initialize LLM news analyzer.

        Args:
            bus: Event bus
            anthropic_api_key: Anthropic API key (optional)
            model: Claude model to use
            max_calls_per_hour: Rate limit for API calls
            batch_size: Number of news items per batch
            batch_window_seconds: Time window for batching
            timeout_seconds: API timeout
        """
        self._bus = bus
        self._api_key = anthropic_api_key
        self._model = model
        self._max_calls_per_hour = max_calls_per_hour
        self._batch_size = batch_size
        self._batch_window = batch_window_seconds
        self._timeout = timeout_seconds

        # News buffer for batching
        self._news_buffer: Deque[NewsEvent] = deque(maxlen=50)

        # Rate limiting
        self._call_times: Deque[float] = deque(maxlen=max_calls_per_hour)

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._log = logger.bind(component="llm_news_analyzer")

        self._enabled = bool(anthropic_api_key)
        if not self._enabled:
            self._log.warning("llm_disabled", message="No Anthropic API key. LLM news analysis disabled.")
        else:
            self._log.info("llm_enabled", model=model, max_calls_per_hour=max_calls_per_hour)

    async def start(self) -> None:
        """Start news analysis."""
        if not self._enabled or self._running:
            return

        self._running = True

        # Subscribe to news events
        self._bus.subscribe(EventType.NEWS, self._on_news, subscriber_name="llm_news_analyzer")

        # Start batch processing task
        self._task = asyncio.create_task(self._process_batches())

        self._log.info("llm_news_analyzer_started")

    async def stop(self) -> None:
        """Stop news analysis."""
        self._running = False

        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        self._log.info("llm_news_analyzer_stopped")

    async def _on_news(self, event: Event) -> None:
        """Handle incoming news events."""
        if not isinstance(event, NewsEvent):
            return

        self._news_buffer.append(event)

    async def _process_batches(self) -> None:
        """Process news batches periodically."""
        while self._running:
            try:
                if len(self._news_buffer) >= self._batch_size:
                    await self._analyze_batch()
            except Exception as e:
                self._log.error("batch_processing_error", error=str(e))

            # Wait before next check
            await asyncio.sleep(self._batch_window)

    async def _analyze_batch(self) -> None:
        """Analyze a batch of news and emit signals."""
        # Check rate limit
        if not self._check_rate_limit():
            self._log.warning("rate_limit_exceeded", calls_in_last_hour=len(self._call_times))
            return

        # Extract batch
        batch = [self._news_buffer.popleft() for _ in range(min(self._batch_size, len(self._news_buffer)))]

        if not batch:
            return

        try:
            # Import here to allow graceful degradation
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)

            # Build prompt
            news_text = "\n\n".join([
                f"[{i+1}] {item.title} (Source: {item.source}, Symbols: {item.symbols})"
                for i, item in enumerate(batch)
            ])

            prompt = f"""You are a cryptocurrency trading analyst. Analyze these recent news headlines and provide a trading signal.

News:
{news_text}

Provide your analysis in JSON format:
{{
  "action": "buy" | "sell" | "hold",
  "strength": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "affected_symbols": ["BTC/USD", ...]
}}

Focus on market-moving news. Ignore noise. Consider regulatory changes, major partnerships, security incidents, and macro trends."""

            # Make API call
            response = await client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            # Record call time for rate limiting
            self._call_times.append(time())

            # Parse response
            content = response.content[0].text
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Emit signals for affected symbols
            symbols = result.get("affected_symbols", ["BTC/USD"])
            action_str = result.get("action", "hold").lower()
            action = SignalAction.BUY if action_str == "buy" else (
                SignalAction.SELL if action_str == "sell" else SignalAction.HOLD
            )

            strength = Decimal(str(min(1.0, max(0.0, result.get("strength", 0.5)))))
            confidence = Decimal(str(min(1.0, max(0.0, result.get("confidence", 0.5)))))
            reasoning = result.get("reasoning", "LLM news analysis")

            for symbol in symbols:
                event = SignalEvent(
                    event_type=EventType.SIGNAL,
                    timestamp=time(),
                    signal_type=SignalType.NEWS,
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    reason=reasoning,
                )

                await self._bus.publish(event)

            self._log.info(
                "llm_signal_generated",
                news_count=len(batch),
                action=action.value,
                symbols=symbols,
                strength=str(strength),
                confidence=str(confidence),
            )

        except ImportError:
            self._log.error("anthropic_sdk_not_installed",
                          message="Install with: pip install anthropic")
            self._enabled = False
        except Exception as e:
            self._log.error("llm_analysis_error", error=str(e))

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limit."""
        current_time = time()
        one_hour_ago = current_time - 3600

        # Remove calls older than 1 hour
        while self._call_times and self._call_times[0] < one_hour_ago:
            self._call_times.popleft()

        return len(self._call_times) < self._max_calls_per_hour
