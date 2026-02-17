"""
Adaptive weight adjustment for signal aggregator.

Adjusts signal weights based on performance scores using EMA smoothing.
Updates aggregator weights when ScoreUpdateEvent is published.

@decision DEC-LEARN-002
@title Conservative EMA weight adaptation
@status accepted
@rationale EMA smoothing (alpha=0.1) prevents oscillation from single bad trades.
Floor/ceiling bounds (0.1-2.0) prevent extreme weights. Gradual adaptation is more
stable than reactive changes. Weights persist across restarts via StateManager.
"""

import time
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import ScoreUpdateEvent
from cerebrum.core.state import StateManager
from cerebrum.core.types import SignalType

logger = structlog.get_logger()


class WeightAdapter:
    """
    Adapts signal weights based on performance.

    Subscribes to: ScoreUpdateEvent
    Updates: Signal aggregator weights (via callback)
    """

    def __init__(
        self,
        bus: EventBus,
        state: StateManager,
        weight_callback: callable,
        alpha: Decimal = Decimal("0.1"),
        floor: Decimal = Decimal("0.1"),
        ceiling: Decimal = Decimal("2.0"),
    ) -> None:
        """
        Initialize weight adapter.

        Args:
            bus: Event bus for pub/sub
            state: State manager for persistence
            weight_callback: Function to call with updated weights: callback(signal_type, regime, weight)
            alpha: EMA smoothing factor (0.0-1.0)
            floor: Minimum weight
            ceiling: Maximum weight
        """
        self._bus = bus
        self._state = state
        self._weight_callback = weight_callback
        self._alpha = alpha
        self._floor = floor
        self._ceiling = ceiling
        self._log = logger.bind(component="weight_adapter")

    async def start(self) -> None:
        """Start listening to events."""
        from cerebrum.core.types import EventType
        self._bus.subscribe(EventType.SCORE_UPDATE, self._on_score_update, "weight_adapter")
        self._log.info("weight_adapter_started")

    async def _on_score_update(self, event: ScoreUpdateEvent) -> None:
        """Adjust weights when scores update."""
        for signal_type, metrics in event.scores.items():
            # Calculate new weight based on composite score
            # Using profit factor as primary metric (balanced with win rate)
            profit_factor = metrics.get("profit_factor", Decimal(1))
            win_rate = metrics.get("win_rate", Decimal("0.5"))

            # Composite score: weighted average of profit factor and win rate
            # Profit factor dominates (70%), win rate stabilizes (30%)
            raw_score = (profit_factor * Decimal("0.7")) + (win_rate * Decimal("2") * Decimal("0.3"))

            # Get historical weight
            history = await self._state.get_weight_history(signal_type, event.regime, limit=1)
            if history:
                old_weight = history[0][1]
            else:
                old_weight = Decimal(1)  # Default neutral weight

            # EMA smoothing
            new_weight = (self._alpha * raw_score) + ((Decimal(1) - self._alpha) * old_weight)

            # Apply bounds
            new_weight = max(self._floor, min(self._ceiling, new_weight))

            # Save weight history
            await self._state.save_weight(signal_type, event.regime, new_weight, time.time())

            # Update aggregator weights
            self._weight_callback(signal_type, event.regime, new_weight)

            self._log.info(
                "weight_adjusted",
                signal_type=signal_type.value,
                regime=event.regime,
                old_weight=float(old_weight),
                new_weight=float(new_weight),
                profit_factor=float(profit_factor),
                win_rate=float(win_rate),
            )
