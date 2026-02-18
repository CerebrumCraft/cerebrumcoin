"""
Plugin registry for lifecycle management and discovery.

@decision DEC-PLUGIN-002
@title Error isolation in plugin registry
@status accepted
@rationale One failing plugin shouldn't crash the system. Registry wraps each plugin.start()
in try/except, logs failures with full traceback, removes failed plugins from active set,
and continues with healthy plugins. This enables graceful degradation and easier debugging.
"""

import structlog
from pathlib import Path
from typing import Type

from cerebrum.core.bus import EventBus
from cerebrum.plugins.base import Plugin

logger = structlog.get_logger()


class PluginRegistry:
    """
    Manages plugin lifecycle and discovery.

    Responsibilities:
    - Register plugins (manual or auto-discovery)
    - Initialize all plugins with bus and config
    - Start/stop all plugins
    - Error isolation (failing plugins don't break others)
    """

    def __init__(self, bus: EventBus) -> None:
        """
        Initialize registry.

        Args:
            bus: Event bus for plugins to use
        """
        self.bus = bus
        self._plugin_classes: dict[str, Type[Plugin]] = {}
        self._plugin_instances: dict[str, Plugin] = {}
        self._log = logger.bind(component="plugin_registry")

    def register(self, name: str, plugin_class: Type[Plugin]) -> None:
        """
        Register a plugin class.

        Args:
            name: Unique plugin identifier
            plugin_class: Plugin class (not instance)

        Raises:
            ValueError: If name already registered
        """
        if name in self._plugin_classes:
            raise ValueError(f"Plugin {name} already registered")

        self._plugin_classes[name] = plugin_class
        self._log.info("plugin_registered", name=name, plugin_class=plugin_class.__name__)

    def list_plugins(self) -> list[str]:
        """
        List all registered plugin names.

        Returns:
            List of plugin names
        """
        return list(self._plugin_classes.keys())

    def get_plugin(self, name: str) -> Plugin | None:
        """
        Get active plugin instance by name.

        Returns:
            Plugin instance or None if not active
        """
        return self._plugin_instances.get(name)

    async def initialize_all(self, config: dict) -> None:
        """
        Initialize all registered plugins.

        Args:
            config: Dict of plugin-specific configs, keyed by plugin name
                    Example: {"my_plugin": {"setting": "value"}}

        Note: Failing plugins are logged but don't stop initialization of others.
        """
        for name, plugin_class in self._plugin_classes.items():
            try:
                plugin = plugin_class()
                plugin_config = config.get(name, {})
                await plugin.initialize(self.bus, plugin_config)
                self._plugin_instances[name] = plugin
                self._log.info(
                    "plugin_initialized",
                    name=name,
                    version=plugin.version,
                    description=plugin.description,
                )
            except Exception as e:
                self._log.error(
                    "plugin_initialize_failed",
                    name=name,
                    error=str(e),
                    exc_info=True,
                )

    async def start_all(self) -> None:
        """
        Start all initialized plugins.

        Error isolation: If a plugin fails to start, it's removed from
        active plugins and logged, but others continue.
        """
        failed_plugins = []

        for name, plugin in list(self._plugin_instances.items()):
            try:
                await plugin.start()
                self._log.info("plugin_started", name=name)
            except Exception as e:
                self._log.error(
                    "plugin_start_failed",
                    name=name,
                    error=str(e),
                    exc_info=True,
                )
                failed_plugins.append(name)

        # Remove failed plugins from active set
        for name in failed_plugins:
            del self._plugin_instances[name]

        if failed_plugins:
            self._log.warning(
                "plugins_failed_to_start",
                failed=failed_plugins,
                active=list(self._plugin_instances.keys()),
            )

    async def stop_all(self) -> None:
        """Stop all active plugins gracefully."""
        for name, plugin in list(self._plugin_instances.items()):
            try:
                await plugin.stop()
                self._log.info("plugin_stopped", name=name)
            except Exception as e:
                self._log.error(
                    "plugin_stop_failed",
                    name=name,
                    error=str(e),
                    exc_info=True,
                )

        self._plugin_instances.clear()

    def discover(self, directory: Path) -> None:
        """
        Auto-discover plugins from a directory.

        Looks for *_plugin.py files and imports them.
        Each file should define a Plugin subclass.

        Args:
            directory: Directory to scan for plugins

        Note: Not implemented in Phase 6. Manual registration is sufficient.
        Future enhancement for extensibility.
        """
        raise NotImplementedError("Plugin auto-discovery coming in future release")
