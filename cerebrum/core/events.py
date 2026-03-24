"""
Event definitions for the CerebrumCoin event bus.

All events are immutable dataclasses that flow through the central event bus.
Components communicate solely through these events—no direct coupling.

@decision DEC-EVENTS-001
@title Immutable frozen dataclasses for all events
@status accepted
@rationale Prevents accidental mutation during event propagation. Events are facts
that happened at a point in time—they should never change. Using frozen=True
enforces immutability and enables safe concurrent access across async tasks.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from cerebrum.core.types import (
    Amount,
    EventType,
    OrderStatus,
    OrderType,
    Price,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
    Symbol,
    Timestamp,
    Volume,
)


@dataclass(frozen=True)
class Event:
    """Base event class with timestamp and type."""
    event_type: EventType
    timestamp: Timestamp


@dataclass(frozen=True)
class MarketDataEvent(Event):
    """Real-time market data from exchange."""
    symbol: Symbol
    price: Price
    volume: Volume
    bid: Price | None = None
    ask: Price | None = None
    spread: Price | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Ensure event_type is set correctly
        object.__setattr__(self, 'event_type', EventType.MARKET_DATA)


@dataclass(frozen=True)
class SignalEvent(Event):
    """Trading signal from any signal generator."""
    signal_type: SignalType
    symbol: Symbol
    action: SignalAction
    strength: Decimal  # 0.0 to 1.0
    confidence: Decimal  # 0.0 to 1.0
    target_price: Price | None = None
    stop_loss: Price | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None
    strategy_id: str | None = None  # Optional strategy identifier for multi-strategy routing

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.SIGNAL)


@dataclass(frozen=True)
class OrderEvent(Event):
    """Order request to be executed."""
    order_id: str
    symbol: Symbol
    side: Side
    order_type: OrderType
    amount: Amount
    price: Price | None = None  # None for market orders
    stop_price: Price | None = None
    status: OrderStatus = OrderStatus.PENDING
    metadata: dict[str, Any] | None = None
    strategy_id: str | None = None  # Optional strategy identifier for multi-strategy routing

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.ORDER)


@dataclass(frozen=True)
class FillEvent(Event):
    """Order execution confirmation."""
    order_id: str
    symbol: Symbol
    side: Side
    filled_amount: Amount
    fill_price: Price
    commission: Decimal
    commission_asset: str
    exchange_order_id: str | None = None
    metadata: dict[str, Any] | None = None
    strategy_id: str | None = None  # Optional strategy identifier for multi-strategy routing

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.FILL)


@dataclass(frozen=True)
class PositionUpdateEvent(Event):
    """Position state change."""
    symbol: Symbol
    amount: Amount
    average_entry_price: Price
    current_price: Price
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.POSITION_UPDATE)


@dataclass(frozen=True)
class RiskAlertEvent(Event):
    """Risk management alert."""
    level: RiskLevel
    message: str
    symbol: Symbol | None = None
    action_required: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.RISK_ALERT)


@dataclass(frozen=True)
class NewsEvent(Event):
    """News article or headline."""
    title: str
    source: str
    url: str
    published_at: datetime
    symbols: list[Symbol] | None = None
    sentiment_score: Decimal | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.NEWS)


@dataclass(frozen=True)
class SentimentEvent(Event):
    """Aggregated sentiment signal."""
    symbol: Symbol
    score: Decimal  # -1.0 (bearish) to 1.0 (bullish)
    source: str
    confidence: Decimal
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.SENTIMENT)


@dataclass(frozen=True)
class RegimeChangeEvent(Event):
    """Market regime transition."""
    from_regime: str
    to_regime: str
    confidence: Decimal
    indicators: dict[str, Any]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.REGIME_CHANGE)


@dataclass(frozen=True)
class TradeOpenedEvent(Event):
    """Published when a trade is opened."""
    trade_id: int
    symbol: Symbol
    side: Side
    entry_price: Decimal
    quantity: Decimal
    signal_snapshot: dict[str, Any]
    regime: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.TRADE_OPENED)


@dataclass(frozen=True)
class TradeClosedEvent(Event):
    """Published when a trade is closed."""
    trade_id: int
    symbol: Symbol
    side: Side
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    signal_snapshot: dict[str, Any]
    regime: str
    entry_time: float
    exit_time: float

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.TRADE_CLOSED)


@dataclass(frozen=True)
class ScoreUpdateEvent(Event):
    """Published when signal scores are recalculated."""
    regime: str
    scores: dict[SignalType, dict[str, Decimal]]  # {signal_type: {metric: value}}

    def __post_init__(self) -> None:
        object.__setattr__(self, 'event_type', EventType.SCORE_UPDATE)
