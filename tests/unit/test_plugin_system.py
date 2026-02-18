"""
Tests for plugin system (base plugin interface and registry).

Tests the plugin lifecycle, discovery, and error isolation.

@decision DEC-TEST-007
@title Mock external APIs in plugin and adapter tests
@status accepted
@rationale Plugin system tests use mock plugins to verify lifecycle and error isolation
without depending on real external services. Tests verify: plugin lifecycle (init → start
→ stop), error isolation (failing plugins don't break others), config passing, and event
subscription via bus reference.
"""

import asyncio
from dataclasses import dataclass

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.types import EventType
from cerebrum.core.events import Event
from cerebrum.plugins.base import Plugin
from cerebrum.plugins.registry import PluginRegistry


@dataclass(frozen=True)
class MockEvent(Event):
    """Mock event for testing."""
    event_type: EventType = EventType.MARKET_DATA
    timestamp: float = 0.0
    value: int = 0


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mock plugin for testing"

    @property
    def supported_events(self) -> list[type[Event]]:
        return [MockEvent]

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.started = False
        self.stopped = False
        self.events_received = []

    async def initialize(self, bus: EventBus, config: dict) -> None:
        """Initialize plugin."""
        await super().initialize(bus, config)
        self.initialized = True

        # Subscribe to MockEvent
        async def handler(event: MockEvent) -> None:
            self.events_received.append(event)

        self.bus.subscribe(EventType.MARKET_DATA, handler, "mock_plugin")

    async def start(self) -> None:
        """Start plugin."""
        self.started = True

    async def stop(self) -> None:
        """Stop plugin."""
        self.stopped = True


class FailingPlugin(Plugin):
    """Plugin that fails during start."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Failing plugin"

    @property
    def supported_events(self) -> list[type[Event]]:
        return []

    async def initialize(self, bus: EventBus, config: dict) -> None:
        """Initialize plugin."""
        await super().initialize(bus, config)

    async def start(self) -> None:
        """Start plugin - fails."""
        raise RuntimeError("Plugin start failed")

    async def stop(self) -> None:
        """Stop plugin."""
        pass


@pytest.mark.asyncio
async def test_plugin_lifecycle():
    """Test plugin initialization, start, and stop."""
    bus = EventBus()
    await bus.start()

    plugin = MockPlugin()
    assert not plugin.initialized
    assert not plugin.started
    assert not plugin.stopped

    # Initialize
    await plugin.initialize(bus, {"setting": "value"})
    assert plugin.initialized
    assert plugin.config["setting"] == "value"
    assert plugin.bus is bus

    # Start
    await plugin.start()
    assert plugin.started

    # Stop
    await plugin.stop()
    assert plugin.stopped

    await bus.stop()


@pytest.mark.asyncio
async def test_plugin_event_subscription():
    """Test plugin can subscribe to events via bus."""
    bus = EventBus()
    await bus.start()

    plugin = MockPlugin()
    await plugin.initialize(bus, {})
    await plugin.start()

    # Publish event
    event = MockEvent(value=42)
    await bus.publish(event)
    await asyncio.sleep(0.01)  # Let event propagate

    # Plugin should have received it
    assert len(plugin.events_received) == 1
    assert plugin.events_received[0].value == 42

    await plugin.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_registry_manual_registration():
    """Test manual plugin registration."""
    bus = EventBus()
    await bus.start()

    registry = PluginRegistry(bus)

    # Register plugin
    registry.register("mock", MockPlugin)
    assert "mock" in registry.list_plugins()

    # Initialize and start
    await registry.initialize_all({})
    await registry.start_all()

    # Get plugin instance
    plugin = registry.get_plugin("mock")
    assert isinstance(plugin, MockPlugin)
    assert plugin.initialized
    assert plugin.started

    # Stop all
    await registry.stop_all()
    assert plugin.stopped

    await bus.stop()


@pytest.mark.asyncio
async def test_registry_error_isolation():
    """Test that one failing plugin doesn't break others."""
    bus = EventBus()
    await bus.start()

    registry = PluginRegistry(bus)
    registry.register("mock", MockPlugin)
    registry.register("failing", FailingPlugin)

    await registry.initialize_all({})

    # Start all - failing plugin should be isolated
    await registry.start_all()

    # Mock plugin should still work
    mock_plugin = registry.get_plugin("mock")
    assert mock_plugin.started

    # Failing plugin should not be in active plugins
    assert registry.get_plugin("failing") is None

    await registry.stop_all()
    await bus.stop()


@pytest.mark.asyncio
async def test_registry_plugin_config():
    """Test plugin-specific config passing."""
    bus = EventBus()
    await bus.start()

    registry = PluginRegistry(bus)
    registry.register("mock", MockPlugin)

    # Pass plugin-specific config
    config = {
        "mock": {"custom_setting": "custom_value"}
    }
    await registry.initialize_all(config)

    plugin = registry.get_plugin("mock")
    assert plugin.config["custom_setting"] == "custom_value"

    await registry.stop_all()
    await bus.stop()


@pytest.mark.asyncio
async def test_plugin_metadata():
    """Test plugin metadata properties."""
    plugin = MockPlugin()

    assert plugin.name == "mock"
    assert plugin.version == "1.0.0"
    assert plugin.description == "Mock plugin for testing"
    assert MockEvent in plugin.supported_events
