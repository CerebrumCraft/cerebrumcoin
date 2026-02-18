"""Plugin system for CerebrumCoin extensibility."""

from cerebrum.plugins.base import Plugin
from cerebrum.plugins.registry import PluginRegistry

__all__ = ["Plugin", "PluginRegistry"]
