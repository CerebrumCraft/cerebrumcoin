"""
News ingestion pipeline for CerebrumCoin.

Polls multiple news sources (CryptoPanic, NewsAPI) and publishes NewsEvents to the event bus.
Implements deduplication, rate limiting, and graceful degradation.

@decision DEC-INT-001
@title News ingestion with graceful degradation
@status accepted
@rationale Multiple news sources provide redundancy. If CryptoPanic is down, NewsAPI continues.
If both fail, the system continues with technical signals only. URL-based deduplication prevents
duplicate processing. Each source runs in its own async task with independent error handling.

@decision DEC-INT-002
@title Async polling with aiohttp
@status accepted
@rationale Non-blocking HTTP requests keep the event loop responsive. Background tasks poll
independently without blocking market data or signal processing. Proper timeout and retry
handling prevents cascading failures.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Set

import aiohttp
import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import NewsEvent
from cerebrum.core.types import EventType

logger = structlog.get_logger()


class NewsIngestionPipeline:
    """
    News ingestion from multiple sources with deduplication and graceful degradation.

    Sources:
    - CryptoPanic (crypto news aggregator, 200 req/hr free tier)
    - NewsAPI.org (general financial news, 100 req/day free tier)

    Features:
    - URL-based deduplication across all sources
    - Independent polling tasks per source
    - Graceful degradation if API keys missing or sources fail
    - Configurable poll intervals
    """

    def __init__(
        self,
        bus: EventBus,
        cryptopanic_api_key: str = "",
        cryptopanic_poll_interval: int = 300,  # 5 minutes
        newsapi_api_key: str = "",
        newsapi_poll_interval: int = 1800,  # 30 minutes
    ) -> None:
        """
        Initialize news ingestion pipeline.

        Args:
            bus: Event bus for publishing NewsEvents
            cryptopanic_api_key: CryptoPanic API key (optional)
            cryptopanic_poll_interval: Seconds between CryptoPanic polls
            newsapi_api_key: NewsAPI.org API key (optional)
            newsapi_poll_interval: Seconds between NewsAPI polls
        """
        self._bus = bus
        self._cryptopanic_key = cryptopanic_api_key
        self._cryptopanic_interval = cryptopanic_poll_interval
        self._newsapi_key = newsapi_api_key
        self._newsapi_interval = newsapi_poll_interval

        # Deduplication: track seen URLs
        self._seen_urls: Set[str] = set()
        self._max_seen_urls = 1000  # Limit memory growth

        # Background tasks
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

        self._log = logger.bind(component="news_ingestion")

        # Check which sources are enabled
        self._cryptopanic_enabled = bool(cryptopanic_api_key)
        self._newsapi_enabled = bool(newsapi_api_key)

        if not self._cryptopanic_enabled and not self._newsapi_enabled:
            self._log.warning("no_news_sources_configured",
                            message="Both CryptoPanic and NewsAPI keys missing. News ingestion disabled.")
        else:
            enabled_sources = []
            if self._cryptopanic_enabled:
                enabled_sources.append("CryptoPanic")
            if self._newsapi_enabled:
                enabled_sources.append("NewsAPI")
            self._log.info("news_sources_configured", sources=enabled_sources)

    async def start(self) -> None:
        """Start news ingestion tasks."""
        if self._running:
            return

        self._running = True

        # Start CryptoPanic polling if enabled
        if self._cryptopanic_enabled:
            task = asyncio.create_task(self._poll_cryptopanic())
            self._tasks.append(task)
            self._log.info("cryptopanic_polling_started", interval=self._cryptopanic_interval)

        # Start NewsAPI polling if enabled
        if self._newsapi_enabled:
            task = asyncio.create_task(self._poll_newsapi())
            self._tasks.append(task)
            self._log.info("newsapi_polling_started", interval=self._newsapi_interval)

    async def stop(self) -> None:
        """Stop news ingestion tasks."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        self._log.info("news_ingestion_stopped")

    async def _poll_cryptopanic(self) -> None:
        """Poll CryptoPanic API in a loop."""
        while self._running:
            try:
                await self._fetch_cryptopanic()
            except Exception as e:
                self._log.error("cryptopanic_fetch_error", error=str(e))

            # Wait for next poll
            await asyncio.sleep(self._cryptopanic_interval)

    async def _poll_newsapi(self) -> None:
        """Poll NewsAPI in a loop."""
        while self._running:
            try:
                await self._fetch_newsapi()
            except Exception as e:
                self._log.error("newsapi_fetch_error", error=str(e))

            # Wait for next poll
            await asyncio.sleep(self._newsapi_interval)

    async def _fetch_cryptopanic(self) -> None:
        """Fetch news from CryptoPanic API."""
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self._cryptopanic_key,
            "public": "true",
            "kind": "news",  # Only news, not social media
            "filter": "rising",  # Rising stories
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        self._log.warning("cryptopanic_http_error", status=resp.status)
                        return

                    data = await resp.json()
                    results = data.get("results", [])

                    news_count = 0
                    for item in results:
                        if await self._process_cryptopanic_item(item):
                            news_count += 1

                    if news_count > 0:
                        self._log.info("cryptopanic_news_fetched", count=news_count)

            except asyncio.TimeoutError:
                self._log.warning("cryptopanic_timeout")
            except Exception as e:
                self._log.error("cryptopanic_error", error=str(e))

    async def _process_cryptopanic_item(self, item: Dict[str, Any]) -> bool:
        """
        Process a CryptoPanic news item.

        Returns:
            True if news was published, False if deduplicated/skipped
        """
        url = item.get("url", "")
        if not url or url in self._seen_urls:
            return False

        # Extract data
        title = item.get("title", "")
        source_name = item.get("source", {}).get("title", "CryptoPanic")
        published_str = item.get("published_at", "")

        # Parse timestamp
        try:
            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = datetime.utcnow()

        # Extract relevant crypto symbols
        currencies = item.get("currencies", [])
        symbols = [f"{c['code']}/USD" for c in currencies if c.get("code")] if currencies else None

        # Create and publish event
        event = NewsEvent(
            event_type=EventType.NEWS,
            timestamp=datetime.utcnow().timestamp(),
            title=title,
            source=source_name,
            url=url,
            published_at=published_at,
            symbols=symbols,
            metadata={"votes": item.get("votes", {})},
        )

        await self._bus.publish(event)

        # Mark as seen
        self._seen_urls.add(url)
        self._cleanup_seen_urls()

        return True

    async def _fetch_newsapi(self) -> None:
        """Fetch news from NewsAPI.org."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": self._newsapi_key,
            "q": "cryptocurrency OR bitcoin OR ethereum",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        self._log.warning("newsapi_http_error", status=resp.status)
                        return

                    data = await resp.json()
                    articles = data.get("articles", [])

                    news_count = 0
                    for article in articles:
                        if await self._process_newsapi_article(article):
                            news_count += 1

                    if news_count > 0:
                        self._log.info("newsapi_news_fetched", count=news_count)

            except asyncio.TimeoutError:
                self._log.warning("newsapi_timeout")
            except Exception as e:
                self._log.error("newsapi_error", error=str(e))

    async def _process_newsapi_article(self, article: Dict[str, Any]) -> bool:
        """
        Process a NewsAPI article.

        Returns:
            True if news was published, False if deduplicated/skipped
        """
        url = article.get("url", "")
        if not url or url in self._seen_urls:
            return False

        # Extract data
        title = article.get("title", "")
        source_name = article.get("source", {}).get("name", "NewsAPI")
        published_str = article.get("publishedAt", "")

        # Parse timestamp
        try:
            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = datetime.utcnow()

        # Create and publish event
        event = NewsEvent(
            event_type=EventType.NEWS,
            timestamp=datetime.utcnow().timestamp(),
            title=title,
            source=source_name,
            url=url,
            published_at=published_at,
            symbols=None,  # NewsAPI doesn't provide crypto symbol tagging
            metadata={"description": article.get("description")},
        )

        await self._bus.publish(event)

        # Mark as seen
        self._seen_urls.add(url)
        self._cleanup_seen_urls()

        return True

    def _cleanup_seen_urls(self) -> None:
        """Prevent unbounded memory growth of seen URLs set."""
        if len(self._seen_urls) > self._max_seen_urls:
            # Remove oldest half (set doesn't preserve order, but that's OK for dedup)
            urls_list = list(self._seen_urls)
            self._seen_urls = set(urls_list[-self._max_seen_urls // 2:])
            self._log.debug("seen_urls_cleaned", remaining=len(self._seen_urls))
