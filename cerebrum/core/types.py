"""
Core type definitions for CerebrumCoin.

Provides enums, type aliases, and shared types used across all modules.

@decision DEC-TYPES-001
@title Use Decimal for all financial calculations
@status accepted
@rationale Avoid floating-point precision errors in financial math. All price, volume,
and amount calculations use Python's Decimal type. Type aliases (Price, Volume, Amount)
enforce this convention across the codebase.
"""

from decimal import Decimal
from enum import Enum
from typing import Literal, TypeAlias

# Type aliases for clarity
Price: TypeAlias = Decimal
Volume: TypeAlias = Decimal
Amount: TypeAlias = Decimal
Timestamp: TypeAlias = float  # Unix epoch seconds


class Side(str, Enum):
    """Order side: buy or sell."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type for exchange execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EventType(str, Enum):
    """Event types flowing through the bus."""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    POSITION_UPDATE = "position_update"
    RISK_ALERT = "risk_alert"
    NEWS = "news"
    SENTIMENT = "sentiment"
    REGIME_CHANGE = "regime_change"


class TradingMode(str, Enum):
    """Trading execution mode."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class SignalType(str, Enum):
    """Signal classification."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    REGIME = "regime"
    NEWS = "news"
    COMBINED = "combined"


class SignalAction(str, Enum):
    """Signal recommendation."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class RiskLevel(str, Enum):
    """Risk assessment level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Literal types for stricter validation
Exchange = Literal["kraken", "binance", "coinbase"]
Symbol = str  # e.g., "BTC/USD"
