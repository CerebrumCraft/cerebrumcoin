"""
Conductor package: LLM-powered capital allocation for multi-strategy trading.

The Conductor combines Darwinian (math-based) allocation with LLM reasoning
to decide how much capital each strategy receives at any given time.

Components:
    DarwinianAllocator  — Pure-math, rolling-Sharpe-based capital allocator.
                          No LLM calls. Testable in isolation.
    Conductor           — Orchestrator that subscribes to market events, runs
                          the allocator, and optionally calls Claude for overrides.
"""

from cerebrum.conductor.allocator import DarwinianAllocator
from cerebrum.conductor.conductor import Conductor

__all__ = ["DarwinianAllocator", "Conductor"]
