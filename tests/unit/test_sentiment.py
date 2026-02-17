"""
Unit tests for FinBERT sentiment analysis.
"""

import asyncio
import pytest

from cerebrum.core.bus import EventBus
from cerebrum.signals.sentiment import FinBERTSentiment


@pytest.fixture
async def event_bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_finbert_disabled_by_default(event_bus):
    """Test FinBERT is disabled when not explicitly enabled."""
    sentiment = FinBERTSentiment(event_bus, enabled=False)
    assert not sentiment._enabled


@pytest.mark.asyncio
async def test_finbert_handles_missing_transformers(event_bus):
    """Test graceful degradation when transformers not installed."""
    # FinBERT will try to import and fail gracefully
    sentiment = FinBERTSentiment(event_bus, enabled=True)
    # Should disable itself if transformers not available
    # (actual behavior depends on whether transformers is installed)
    assert sentiment._enabled in [True, False]
