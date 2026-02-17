"""
FinBERT sentiment analysis for CerebrumCoin.

Uses pre-trained FinBERT model for financial news sentiment scoring.

@decision DEC-INT-003
@title Optional FinBERT with graceful fallback
@status accepted
@rationale FinBERT (ProsusAI/finbert) requires transformers + torch (large dependencies).
Made optional via enable_finbert config flag. If disabled or unavailable, system
continues with LLM-based sentiment and social signals only.
"""

from collections import deque
from decimal import Decimal
from time import time
from typing import Deque

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, NewsEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction, SignalType

logger = structlog.get_logger()


class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis for news.
    
    Features:
    - Pre-trained financial sentiment model
    - Aggregates sentiment across recent headlines
    - Optional dependency (transformers + torch)
    - Graceful degradation if unavailable
    """

    def __init__(
        self,
        bus: EventBus,
        enabled: bool = False,
        window_size: int = 10,  # Recent news to aggregate
        model_name: str = "ProsusAI/finbert",
    ) -> None:
        """
        Initialize FinBERT sentiment analyzer.

        Args:
            bus: Event bus
            enabled: Enable FinBERT (requires transformers)
            window_size: Number of recent news items to aggregate
            model_name: HuggingFace model name
        """
        self._bus = bus
        self._enabled = enabled
        self._window_size = window_size
        self._model_name = model_name

        # News buffer
        self._news_buffer: Deque[NewsEvent] = deque(maxlen=window_size)

        self._model = None
        self._tokenizer = None
        self._log = logger.bind(component="finbert_sentiment")

        if enabled:
            self._load_model()

        if not self._enabled:
            self._log.info("finbert_disabled")
        else:
            # Subscribe to news
            bus.subscribe(EventType.NEWS, self._on_news, subscriber_name="finbert_sentiment")
            self._log.info("finbert_enabled", model=model_name)

    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()

            self._log.info("finbert_model_loaded")

        except ImportError:
            self._log.warning("finbert_unavailable",
                            message="transformers not installed. Install with: pip install transformers torch")
            self._enabled = False
        except Exception as e:
            self._log.error("finbert_load_failed", error=str(e))
            self._enabled = False

    async def _on_news(self, event: Event) -> None:
        """Handle news events and analyze sentiment."""
        if not isinstance(event, NewsEvent) or not self._enabled:
            return

        self._news_buffer.append(event)

        # Analyze when we have enough news
        if len(self._news_buffer) >= 3:
            await self._analyze_and_emit()

    async def _analyze_and_emit(self) -> None:
        """Analyze sentiment of recent news and emit signal."""
        if not self._model or not self._tokenizer:
            return

        try:
            import torch

            # Aggregate sentiment scores
            sentiment_scores = []

            for news in self._news_buffer:
                # Tokenize
                inputs = self._tokenizer(
                    news.title,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )

                # Predict
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # FinBERT outputs: [positive, negative, neutral]
                pos_score = float(predictions[0][0])
                neg_score = float(predictions[0][1])
                
                # Convert to -1 to 1 scale
                sentiment = pos_score - neg_score
                sentiment_scores.append(sentiment)

            # Average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Convert to signal
            if avg_sentiment > 0.2:
                action = SignalAction.BUY
                strength = Decimal(str(min(1.0, avg_sentiment)))
            elif avg_sentiment < -0.2:
                action = SignalAction.SELL
                strength = Decimal(str(min(1.0, abs(avg_sentiment))))
            else:
                action = SignalAction.HOLD
                strength = Decimal("0.0")

            # Confidence based on agreement across news items
            variance = sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
            confidence = Decimal(str(max(0.3, 1.0 - variance)))

            # Emit signal for most mentioned symbol
            symbols = []
            for news in self._news_buffer:
                if news.symbols:
                    symbols.extend(news.symbols)
            
            target_symbol = symbols[0] if symbols else "BTC/USD"

            event = SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=time(),
                signal_type=SignalType.SENTIMENT,
                symbol=target_symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                reason=f"FinBERT sentiment: {avg_sentiment:.2f} across {len(self._news_buffer)} headlines",
            )

            await self._bus.publish(event)

            self._log.info(
                "finbert_signal_emitted",
                avg_sentiment=avg_sentiment,
                action=action.value,
                strength=str(strength),
                confidence=str(confidence),
            )

        except Exception as e:
            self._log.error("finbert_analysis_error", error=str(e))
