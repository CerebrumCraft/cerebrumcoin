# @mock-exempt: Mocking external HTTP API calls (aiohttp) to Fear & Greed Index API
"""
Unit tests for social sentiment (Fear & Greed Index).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.events import SignalEvent
from cerebrum.core.types import EventType, SignalAction
from cerebrum.intelligence.social import FearGreedSentiment


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_fear_greed_extreme_fear(event_bus):
    """Test extreme fear generates strong buy signal."""
    mock_response = {
        "data": [
            {
                "value": "20",
                "value_classification": "Extreme Fear"
            }
        ]
    }
    
    with patch("cerebrum.intelligence.social.aiohttp.ClientSession") as mock_session_cls:
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
        
        sentiment = FearGreedSentiment(event_bus, poll_interval=1)
        
        signals = []
        async def collect(event):
            if isinstance(event, SignalEvent):
                signals.append(event)
        
        event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test")
        
        await sentiment._fetch_and_emit()
        await asyncio.sleep(0.1)
        
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY
        assert float(signals[0].strength) >= 0.7


@pytest.mark.asyncio
async def test_fear_greed_extreme_greed(event_bus):
    """Test extreme greed generates strong sell signal."""
    mock_response = {
        "data": [
            {
                "value": "85",
                "value_classification": "Extreme Greed"
            }
        ]
    }
    
    with patch("cerebrum.intelligence.social.aiohttp.ClientSession") as mock_session_cls:
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
        
        sentiment = FearGreedSentiment(event_bus)
        
        signals = []
        async def collect(event):
            if isinstance(event, SignalEvent):
                signals.append(event)
        
        event_bus.subscribe(EventType.SIGNAL, collect, subscriber_name="test")
        
        await sentiment._fetch_and_emit()
        await asyncio.sleep(0.1)
        
        assert len(signals) == 1
        assert signals[0].action == SignalAction.SELL
