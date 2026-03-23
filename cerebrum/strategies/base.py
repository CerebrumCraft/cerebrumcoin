"""
Strategy configuration dataclass for the multi-strategy abstraction layer.

A StrategyConfig is a pure data object — it describes how a strategy should
be wired (weights, thresholds, risk overrides, capital allocation) but it
is NOT an actor. It does not generate signals, hold state, or subscribe to
events. The StrategyRegistry reads these configs and constructs the actual
pipeline components (SignalAggregator, PortfolioTracker, ExitMonitor,
RiskManager) from them.

@decision DEC-STRAT-001
@title StrategyConfig as a pure config dataclass, not an actor
@status accepted
@rationale The eng review explicitly rejected the "Strategy as actor" pattern
(generate_signals(), on_fill(), etc.) because it duplicates the event bus
routing and creates hidden coupling between strategies. A config object is
simpler to test, version, and serialize. The StrategyRegistry owns the
wiring logic; the config owns the parameters. This also enables the
single-strategy backward compatibility path: wrap the current main.py
wiring in a MomentumStrategy config and the behaviour is identical.

@decision DEC-STRAT-002
@title Isolated portfolios per strategy — no compound keys
@status accepted
@rationale Each strategy gets its own PortfolioTracker with its own cash
balance and position set. Cross-strategy aggregation is handled by
GlobalPortfolio (read-only view). This avoids compound symbol/strategy_id
keys in the portfolio, simplifies per-strategy P&L attribution, and
prevents one strategy's drawdown circuit-breaker from triggering another's.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from cerebrum.core.types import SignalType


@dataclass(frozen=True)
class StrategyConfig:
    """
    Configuration for a single trading strategy.

    This is a pure data object describing how the StrategyRegistry should
    wire the strategy's pipeline. All fields are immutable after creation
    (frozen=True) to prevent accidental mutation during runtime.

    Args:
        name: Unique strategy identifier. Used as the strategy_id tag on
              COMBINED SignalEvents and for pipeline isolation routing.
        aggregator_weights: Signal type weights for this strategy's
              SignalAggregator. Keys are SignalType enum values; values are
              Decimal multipliers applied during weighted voting.
        aggregator_threshold: Minimum aggregate signal strength required for
              the aggregator to emit a COMBINED signal. Lower = more signals,
              higher = fewer but stronger signals.
        risk_overrides: Per-strategy risk rule parameter overrides, keyed by
              the config attribute name. Example: {"stop_loss_percent": "1.0"}.
              Applied as Decimal conversions over the config defaults.
        exit_config: Per-strategy exit monitor parameter overrides. Example:
              {"take_profit_percent": "2.0", "max_position_age_minutes": 60}.
        initial_balance: Capital allocated to this strategy in USD. The
              StrategyRegistry creates a PortfolioTracker with this balance.
              Default is ~1/3 of $10k for a 3-strategy split.
        symbols: Trading symbols this strategy is active on. Used for
              position tracking and exit monitoring. Defaults to BTC/USD
              and ETH/USD (the current paper trading symbols).
    """

    name: str
    aggregator_weights: dict[SignalType, Decimal]
    aggregator_threshold: Decimal = Decimal("0.4")
    risk_overrides: dict[str, Any] = field(default_factory=dict)
    exit_config: dict[str, Any] = field(default_factory=dict)
    initial_balance: Decimal = Decimal("3333.33")
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
