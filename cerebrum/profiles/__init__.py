"""
cerebrum.profiles — hot-swappable risk profile management.

Public API:
    ProfileManager  — parse profiles from TOML, apply overrides to live pipelines
"""

from cerebrum.profiles.manager import ProfileManager

__all__ = ["ProfileManager"]
