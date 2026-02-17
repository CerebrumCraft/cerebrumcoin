# @mock-exempt: Mocking external HTTP API calls (aiohttp) to CryptoPanic/NewsAPI services
"""
Unit tests for news ingestion pipeline.

@decision DEC-TEST-001
@title Test real implementations with mocked HTTP
@status accepted
@rationale Tests verify real NewsIngestionPipeline behavior with mocked HTTP responses.
No mocking of internal logic—only external API calls are mocked per Sacred Practice #5.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import NewsEvent
from cerebrum.core.types import EventType
from cerebrum.intelligence.news import NewsIngestionPipeline


@pytest.fixture
async def event_bus():
    """Create and start event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_news_pipeline_disabled_without_keys(event_bus):
    """Test that pipeline logs warning when no API keys provided."""
    pipeline = NewsIngestionPipeline(event_bus)
    assert not pipeline._cryptopanic_enabled
    assert not pipeline._newsapi_enabled


@pytest.mark.asyncio
async def test_cryptopanic_integration(event_bus):
    """Test CryptoPanic news fetch and deduplication."""
    mock_response = {
        "results": [
            {
                "title": "Bitcoin hits new high",
                "url": "https://example.com/btc1",
                "source": {"title": "CryptoNews"},
                "published_at": "2026-02-17T10:00:00Z",
                "currencies": [{"code": "BTC"}],
                "votes": {"positive": 10},
            },
            {
                "title": "Ethereum update released",
                "url": "https://example.com/eth1",
                "source": {"title": "EthNews"},
                "published_at": "2026-02-17T11:00:00Z",
                "currencies": [{"code": "ETH"}],
                "votes": {},
            },
        ]
    }

    # Mock aiohttp with proper async context manager support
    with patch("cerebrum.intelligence.news.aiohttp.ClientSession") as mock_session_cls:
        # Create the response mock
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        # Create a context manager for session.get()
        # Key: get() returns a context manager synchronously, not a coroutine
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        # Create session instance
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        # Create session context manager
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session_cm

        pipeline = NewsIngestionPipeline(
            event_bus,
            cryptopanic_api_key="test_key",
            cryptopanic_poll_interval=1,
        )

        # Collect events
        received_events = []

        async def collect_events(event):
            if isinstance(event, NewsEvent):
                received_events.append(event)

        event_bus.subscribe(EventType.NEWS, collect_events, subscriber_name="test")

        # Fetch once
        await pipeline._fetch_cryptopanic()

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify
        assert len(received_events) == 2
        assert received_events[0].title == "Bitcoin hits new high"
        assert received_events[0].symbols == ["BTC/USD"]
        assert received_events[1].title == "Ethereum update released"


@pytest.mark.asyncio
async def test_deduplication(event_bus):
    """Test URL-based deduplication."""
    mock_response = {
        "results": [
            {
                "title": "Same news",
                "url": "https://example.com/duplicate",
                "source": {"title": "Source"},
                "published_at": "2026-02-17T10:00:00Z",
                "currencies": [],
                "votes": {},
            }
        ]
    }

    with patch("cerebrum.intelligence.news.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session_cm

        pipeline = NewsIngestionPipeline(event_bus, cryptopanic_api_key="test")

        received_events = []

        async def collect(event):
            if isinstance(event, NewsEvent):
                received_events.append(event)

        event_bus.subscribe(EventType.NEWS, collect, subscriber_name="test")

        # Fetch twice
        await pipeline._fetch_cryptopanic()
        await pipeline._fetch_cryptopanic()
        
        await asyncio.sleep(0.1)

        # Should only emit once (deduplicated)
        assert len(received_events) == 1


@pytest.mark.asyncio
async def test_newsapi_integration(event_bus):
    """Test NewsAPI fetch."""
    mock_response = {
        "articles": [
            {
                "title": "Crypto market update",
                "url": "https://example.com/news1",
                "source": {"name": "Financial Times"},
                "publishedAt": "2026-02-17T12:00:00Z",
                "description": "Market analysis",
            }
        ]
    }

    with patch("cerebrum.intelligence.news.aiohttp.ClientSession") as mock_session_cls:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session_cm

        pipeline = NewsIngestionPipeline(event_bus, newsapi_api_key="test")

        received_events = []

        async def collect(event):
            if isinstance(event, NewsEvent):
                received_events.append(event)

        event_bus.subscribe(EventType.NEWS, collect, subscriber_name="test")

        await pipeline._fetch_newsapi()
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].title == "Crypto market update"
        assert received_events[0].source == "Financial Times"
