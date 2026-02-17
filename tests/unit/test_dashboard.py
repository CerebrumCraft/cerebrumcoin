"""Tests for monitoring dashboard."""

import asyncio
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from cerebrum.core.bus import EventBus
from cerebrum.core.state import StateManager
from cerebrum.monitoring.dashboard import Dashboard


@pytest.fixture
async def event_bus():
    """Create event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def state_manager():
    """Create temporary state manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        state = StateManager(db_path)
        await state.initialize()
        yield state
        await state.close()


@pytest.mark.asyncio
async def test_dashboard_initialization(event_bus, state_manager):
    """Test dashboard initializes without errors."""
    dashboard = Dashboard(
        event_bus,
        state_manager,
        update_interval_seconds=60,
        initial_balance=Decimal("10000.0"),
    )
    
    await dashboard.start()
    await asyncio.sleep(0.1)
    await dashboard.stop()


@pytest.mark.asyncio
async def test_dashboard_handles_no_trades(event_bus, state_manager):
    """Test dashboard displays correctly with no trades."""
    dashboard = Dashboard(
        event_bus,
        state_manager,
        update_interval_seconds=60,
    )
    
    await dashboard.start()
    
    # Trigger manual display (don't wait for timer)
    await dashboard._display_stats()
    
    await dashboard.stop()
