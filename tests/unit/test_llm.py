# @mock-exempt: Mocking external Anthropic API calls
"""
Unit tests for LLM news analyzer.

Tests verify batching, rate limiting, and graceful degradation.

@decision DEC-TEST-004
@title Mock at anthropic module boundary, not cerebrum re-export
@status accepted
@rationale LLMNewsAnalyzer imports AsyncAnthropic locally inside _analyze_batch()
via `from anthropic import AsyncAnthropic`. Module-level patching of
`cerebrum.intelligence.llm.AsyncAnthropic` does nothing because no attribute by
that name exists at module scope. Patching `anthropic.AsyncAnthropic` intercepts
the import at the source, which is the correct boundary for external API mocks.
"""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import NewsEvent, SignalEvent
from cerebrum.core.types import EventType, SignalAction
from cerebrum.intelligence.llm import LLMNewsAnalyzer


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_llm_disabled_without_key(event_bus):
    """Test graceful degradation when API key missing."""
    analyzer = LLMNewsAnalyzer(event_bus)
    assert not analyzer._enabled


@pytest.mark.asyncio
async def test_rate_limiting(event_bus):
    """Test rate limit enforcement."""
    from time import time

    analyzer = LLMNewsAnalyzer(event_bus, anthropic_api_key="test", max_calls_per_hour=2)

    # Initially under limit
    assert analyzer._check_rate_limit()

    # Simulate 2 calls (at limit) - use time() not asyncio time
    current_time = time()
    analyzer._call_times.append(current_time)
    analyzer._call_times.append(current_time)

    # Should be at limit now (2 calls with max=2)
    assert not analyzer._check_rate_limit()

    # Simulate time passing (calls expire after 1 hour)
    analyzer._call_times.clear()
    analyzer._call_times.append(current_time - 3700)  # Older than 1 hour

    # Old call should be cleaned up, back under limit
    assert analyzer._check_rate_limit()


@pytest.mark.asyncio
async def test_news_batching(event_bus):
    """Test news batching logic."""
    pytest.importorskip("anthropic")
    
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"action": "buy", "strength": 0.8, "confidence": 0.7, "reasoning": "Bullish", "affected_symbols": ["BTC/USD"]}')]
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        analyzer = LLMNewsAnalyzer(event_bus, anthropic_api_key="test", batch_size=2)
        await analyzer.start()
        
        # Add news to buffer
        news1 = NewsEvent(
            event_type=EventType.NEWS,
            timestamp=asyncio.get_event_loop().time(),
            title="Bitcoin rises",
            source="Test",
            url="http://test.com/1",
            published_at=asyncio.get_event_loop().time(),
        )
        news2 = NewsEvent(
            event_type=EventType.NEWS,
            timestamp=asyncio.get_event_loop().time(),
            title="Ethereum up",
            source="Test",
            url="http://test.com/2",
            published_at=asyncio.get_event_loop().time(),
        )
        
        analyzer._news_buffer.append(news1)
        analyzer._news_buffer.append(news2)
        
        signals_received = []
        async def collect(event):
            if isinstance(event, SignalEvent):
                signals_received.append(event)
        
        event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test")
        
        await analyzer._analyze_batch()
        await asyncio.sleep(0.1)
        
        assert len(signals_received) > 0
        await analyzer.stop()
