"""
Signal effectiveness scorer for learning system.

Calculates performance metrics (win rate, profit factor, Sharpe ratio)
per signal type per regime. Publishes ScoreUpdateEvent.

@decision DEC-LEARN-003
@title Per-regime signal scoring
@status accepted
@rationale Different signals perform differently in different regimes. Separate metrics
(win rate, profit factor, Sharpe ratio) per signal type per regime enable regime-aware
weight adaptation. Minimum sample size (10 trades) prevents scoring on insufficient data.
"""

import time
from datetime import datetime
from decimal import Decimal

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import ScoreUpdateEvent, TradeClosedEvent
from cerebrum.core.state import SignalScore, StateManager
from cerebrum.core.types import SignalType

logger = structlog.get_logger()


class SignalScorer:
    """
    Calculates signal performance metrics per regime.

    Subscribes to: TradeClosedEvent
    Publishes: ScoreUpdateEvent
    """

    def __init__(self, bus: EventBus, state: StateManager, min_sample_size: int = 10) -> None:
        """
        Initialize signal scorer.

        Args:
            bus: Event bus for pub/sub
            state: State manager for persistence
            min_sample_size: Minimum trades required before scoring
        """
        self._bus = bus
        self._state = state
        self._min_sample_size = min_sample_size
        self._log = logger.bind(component="signal_scorer")

    async def start(self) -> None:
        """Start listening to events."""
        from cerebrum.core.types import EventType
        self._bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed, "signal_scorer")
        self._log.info("signal_scorer_started")

    async def _on_trade_closed(self, event: TradeClosedEvent) -> None:
        """Recalculate scores when a trade closes."""
        # Get all closed trades for this regime
        trades = await self._state.get_closed_trades(regime=event.regime)

        if len(trades) < self._min_sample_size:
            self._log.debug("insufficient_sample_size", regime=event.regime, count=len(trades))
            return

        # Group trades by signal type (based on signal_snapshot)
        signal_trades: dict[SignalType, list] = {}

        for trade in trades:
            # Determine dominant signal from snapshot
            if not trade.signal_snapshot:
                continue

            # Find signal with highest strength
            dominant_signal = None
            max_strength = Decimal(0)

            for signal_key, signal_data in trade.signal_snapshot.items():
                try:
                    signal_type = SignalType(signal_key)
                    strength = abs(Decimal(str(signal_data.get("strength", 0))))
                    if strength > max_strength:
                        max_strength = strength
                        dominant_signal = signal_type
                except (ValueError, KeyError):
                    continue

            if dominant_signal:
                signal_trades.setdefault(dominant_signal, []).append(trade)

        # Calculate scores for each signal type
        scores = {}

        for signal_type, signal_trade_list in signal_trades.items():
            if len(signal_trade_list) < self._min_sample_size:
                continue

            score = self._calculate_score(signal_trade_list)
            if score:
                scores[signal_type] = score
                await self._state.save_signal_score(SignalScore(
                    signal_type=signal_type,
                    regime=event.regime,
                    win_rate=score["win_rate"],
                    profit_factor=score["profit_factor"],
                    sharpe_ratio=score["sharpe_ratio"],
                    sample_size=len(signal_trade_list),
                    updated_at=datetime.now(),
                ))

        # Publish score update
        if scores:
            await self._bus.publish(ScoreUpdateEvent(
                event_type=None,  # Will be set by __post_init__
                timestamp=time.time(),
                regime=event.regime,
                scores=scores,
            ))

            self._log.info("scores_updated", regime=event.regime, signal_count=len(scores))

    def _calculate_score(self, trades: list) -> dict[str, Decimal] | None:
        """Calculate performance metrics for a list of trades."""
        if not trades:
            return None

        # Win rate
        winners = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = Decimal(len(winners)) / Decimal(len(trades))

        # Profit factor
        total_wins = sum(t.pnl for t in winners if t.pnl)
        losers = [t for t in trades if t.pnl and t.pnl < 0]
        total_losses = abs(sum(t.pnl for t in losers if t.pnl))

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = Decimal(999) if total_wins > 0 else Decimal(1)

        # Sharpe ratio (simplified: returns / std dev of returns)
        returns = [t.pnl / t.entry_price / t.quantity for t in trades if t.pnl and t.entry_price and t.quantity]

        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** Decimal("0.5")

            if std_dev > 0:
                sharpe_ratio = mean_return / std_dev * Decimal("15.87")  # Annualized (sqrt(252))
            else:
                sharpe_ratio = Decimal(0)
        else:
            sharpe_ratio = Decimal(0)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
        }
