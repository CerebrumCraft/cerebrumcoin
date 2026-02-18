"""
Base plugin interface for CerebrumCoin extensibility.

All plugins implement this interface to integrate with the event bus and trading system.

@decision DEC-PLUGIN-001
@title Abstract plugin interface with lifecycle hooks
@status accepted
@rationale Plugins need initialize/start/stop hooks for clean lifecycle management.
Bus reference enables event subscription/publication. Metadata properties (name, version,
description, supported_events) enable discovery and documentation. Abstract base class
enforces interface contract across all plugins.
"""

from abc import ABC, abstractmethod

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event

logger = structlog.get_logger()


class Plugin(ABC):
    """
    Abstract base class for all CerebrumCoin plugins.

    Plugins extend the trading system with custom functionality:
    - Custom signal generators
    - Alternative execution venues
    - External integrations (webhooks, alerts, etc.)
    - Custom risk rules
    - Data exporters

    Lifecycle:
        1. __init__() — Plugin instantiation
        2. initialize(bus, config) — Setup with bus reference and config
        3. start() — Begin operations (subscribe to events, start tasks)
        4. stop() — Clean shutdown (unsubscribe, cancel tasks)
    """

    def __init__(self) -> None:
        """Initialize plugin (before bus connection)."""
        self.bus: EventBus | None = None
        self.config: dict = {}
        self._log = logger.bind(plugin=self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name (unique identifier)."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version (semver recommended)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable plugin description."""
        pass

    @property
    @abstractmethod
    def supported_events(self) -> list[type[Event]]:
        """
        Event types this plugin subscribes to or publishes.

        Used for documentation and dependency resolution.
        """
        pass

    async def initialize(self, bus: EventBus, config: dict) -> None:
        """
        Initialize plugin with bus reference and configuration.

        Args:
            bus: Event bus for subscribing/publishing
            config: Plugin-specific configuration dict

        Raises:
            ValueError: If config is invalid
        """
        self.bus = bus
        self.config = config
        self._log.info("plugin_initialized", config=config)

    @abstractmethod
    async def start(self) -> None:
        """
        Start plugin operations.

        Called after all plugins are initialized. Subscribe to events,
        start background tasks, open connections here.

        Raises:
            RuntimeError: If plugin fails to start
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop plugin operations gracefully.

        Unsubscribe from events, cancel tasks, close connections.
        """
        pass
