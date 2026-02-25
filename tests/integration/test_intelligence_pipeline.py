# @mock-exempt: Mocking external HTTP APIs (CryptoPanic, NewsAPI, Fear & Greed, Anthropic)
"""
Integration test for full intelligence pipeline.

@decision DEC-TEST-001
@title End-to-end intelligence pipeline test
@status accepted
@rationale Integration test verifies complete flow: NewsEvent → LLM/FinBERT/Social → 
SignalEvent → Aggregator → regime-aware weighting. Mocks only external APIs (HTTP, LLM).
Tests prove intelligence components integrate correctly via event bus.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import NewsEvent, SignalEvent, RegimeChangeEvent, MarketDataEvent
from cerebrum.core.types import EventType, SignalAction, SignalType
from cerebrum.intelligence.news import NewsIngestionPipeline
from cerebrum.intelligence.llm import LLMNewsAnalyzer
from cerebrum.intelligence.social import FearGreedSentiment
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.regime import RegimeDetector


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_full_intelligence_pipeline(event_bus):
    """Test complete intelligence pipeline integration."""
    pytest.importorskip("anthropic")
    
    # Mock CryptoPanic response
    cryptopanic_response = {
        "results": [
            {
                "title": "Bitcoin adoption grows",
                "url": "https://example.com/btc-news",
                "source": {"title": "CryptoNews"},
                "published_at": "2026-02-17T10:00:00Z",
                "currencies": [{"code": "BTC"}],
                "votes": {},
            }
        ]
    }
    
    # Mock Fear & Greed response (extreme fear = buy signal)
    fear_greed_response = {
        "data": [{"value": "15", "value_classification": "Extreme Fear"}]
    }
    
    # Mock LLM response
    mock_llm_client = AsyncMock()
    mock_llm_response = MagicMock()
    mock_llm_response.content = [MagicMock(text='{"action": "buy", "strength": 0.8, "confidence": 0.7, "reasoning": "Bullish adoption", "affected_symbols": ["BTC/USD"]}')]
    mock_llm_client.messages.create = AsyncMock(return_value=mock_llm_response)

    # Both news.py and social.py do `import aiohttp` at the top level, so they
    # share the same aiohttp module object. Patching each sub-path separately
    # (news.aiohttp.ClientSession, social.aiohttp.ClientSession) targets the same
    # underlying attribute — the second patch wins and the first mock never fires.
    # Fix: patch aiohttp.ClientSession once at the source with a URL-dispatching
    # side_effect so both callers get the right response.
    def make_session_for_url(url, response_data):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=response_data)
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        mock_sess_inst = MagicMock()
        mock_sess_inst.get = MagicMock(return_value=mock_get_cm)
        mock_sess_cm = MagicMock()
        mock_sess_cm.__aenter__ = AsyncMock(return_value=mock_sess_inst)
        mock_sess_cm.__aexit__ = AsyncMock(return_value=None)
        return mock_sess_cm

    cryptopanic_session_cm = make_session_for_url("cryptopanic", cryptopanic_response)
    fear_greed_session_cm = make_session_for_url("fear_greed", fear_greed_response)

    # Alternate which session mock to return on successive calls:
    # first call → news pipeline (CryptoPanic), second → social (Fear & Greed)
    session_call_count = [0]
    def session_side_effect(*args, **kwargs):
        idx = session_call_count[0]
        session_call_count[0] += 1
        return [cryptopanic_session_cm, fear_greed_session_cm][idx % 2]

    with patch("aiohttp.ClientSession", side_effect=session_side_effect), \
         patch("anthropic.AsyncAnthropic", return_value=mock_llm_client):
        
        # Initialize components
        news_pipeline = NewsIngestionPipeline(
            event_bus,
            cryptopanic_api_key="test_key",
            cryptopanic_poll_interval=1,
        )
        
        llm_analyzer = LLMNewsAnalyzer(
            event_bus,
            anthropic_api_key="test_key",
            batch_size=1,
            batch_window_seconds=1,
        )
        await llm_analyzer.start()
        
        fear_greed = FearGreedSentiment(event_bus, poll_interval=1)
        
        aggregator = SignalAggregator(event_bus, threshold=Decimal("0.2"))
        
        regime_detector = RegimeDetector(event_bus, window_size=50, update_interval=10)
        
        # Collect events
        all_news = []
        all_signals = []
        all_regimes = []
        
        async def collect_news(event):
            if isinstance(event, NewsEvent):
                all_news.append(event)
        
        async def collect_signals(event):
            if isinstance(event, SignalEvent):
                all_signals.append(event)
        
        async def collect_regimes(event):
            if isinstance(event, RegimeChangeEvent):
                all_regimes.append(event)
        
        event_bus.subscribe(EventType.NEWS, collect_news, subscriber_name="test_news")
        event_bus.subscribe(EventType.SIGNAL, collect_signals, subscriber_name="test_signals")
        event_bus.subscribe(EventType.REGIME_CHANGE, collect_regimes, subscriber_name="test_regime")

        # Yield to event loop so subscriber consumer tasks are started before we
        # publish events — without this, news published by _fetch_cryptopanic()
        # is delivered before test_news's queue task is running.
        await asyncio.sleep(0)

        # Trigger pipeline
        await news_pipeline._fetch_cryptopanic()
        await fear_greed._fetch_and_emit()
        
        # Give LLM time to process news batch
        await asyncio.sleep(0.5)
        
        # Process batch
        if len(llm_analyzer._news_buffer) > 0:
            await llm_analyzer._analyze_batch()
        
        await asyncio.sleep(0.3)
        
        # Verify pipeline
        assert len(all_news) >= 1, "News should be ingested"
        assert len(all_signals) >= 2, "Should have signals from Fear&Greed and LLM"
        
        # Check signal types
        signal_types = {s.signal_type for s in all_signals}
        assert SignalType.SENTIMENT in signal_types or SignalType.NEWS in signal_types
        
        # Cleanup
        await llm_analyzer.stop()
        await news_pipeline.stop()
        await fear_greed.stop()


@pytest.mark.asyncio
async def test_regime_aware_aggregation(event_bus):
    """Test that aggregator adjusts to regime changes."""
    aggregator = SignalAggregator(event_bus, threshold=Decimal("0.2"))
    
    # Generate market data for regime detection
    regime_detector = RegimeDetector(event_bus, window_size=50, update_interval=10, use_hmm=False)
    
    regime_changes = []
    combined_signals = []
    
    async def collect_regimes(event):
        if isinstance(event, RegimeChangeEvent):
            regime_changes.append(event)
    
    async def collect_combined(event):
        if isinstance(event, SignalEvent) and event.signal_type == SignalType.COMBINED:
            combined_signals.append(event)
    
    event_bus.subscribe(EventType.REGIME_CHANGE, collect_regimes, subscriber_name="test_regime")
    event_bus.subscribe(EventType.SIGNAL, collect_combined, subscriber_name="test_combined")
    
    # Emit regime change manually (simpler than generating 50+ market data points)
    regime_event = RegimeChangeEvent(
        event_type=EventType.REGIME_CHANGE,
        timestamp=asyncio.get_event_loop().time(),
        from_regime="UNKNOWN",
        to_regime="BULL",
        confidence=Decimal("0.8"),
        indicators={"test": "bull"},
    )
    await event_bus.publish(regime_event)
    await asyncio.sleep(0.1)
    
    # Emit technical signal
    tech_signal = SignalEvent(
        event_type=EventType.SIGNAL,
        timestamp=asyncio.get_event_loop().time(),
        signal_type=SignalType.TECHNICAL,
        symbol="BTC/USD",
        action=SignalAction.BUY,
        strength=Decimal("0.6"),
        confidence=Decimal("0.8"),
    )
    await event_bus.publish(tech_signal)
    await asyncio.sleep(0.1)
    
    # Verify regime was applied
    assert aggregator._current_regime == "BULL"
    assert aggregator._weights[SignalType.TECHNICAL] > aggregator._base_weights[SignalType.TECHNICAL]
