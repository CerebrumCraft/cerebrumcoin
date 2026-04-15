"""ORB Stocks strategy config.

@decision DEC-STOCKS-004
@title orb_stocks strategy — breakout entry, RTH-only, auto-flatten
@status accepted
@rationale Dedicated stocks-native strategy consuming only OpeningRange
signals via signal_source_filter (DEC-STOCKS-005 symbol+source isolation
from crypto strategies). $5,000 allocation on AAPL/MSFT/NVDA. Positions
flatten at 15:55 ET via end_of_day_flatten (DEC-STOCKS-003).
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig


ORB_STOCKS_CONFIG = StrategyConfig(
    name="orb_stocks",
    initial_balance=Decimal("5000.0"),
    symbols=["AAPL", "MSFT", "NVDA"],
    signal_source_filter="OpeningRange",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.0"),
        SignalType.SENTIMENT: Decimal("0"),
        SignalType.NEWS: Decimal("0"),
        SignalType.REGIME: Decimal("0"),
    },
    aggregator_threshold=Decimal("0.4"),
    risk_overrides={
        "position_size_percent": "20.0",
        "min_signal_strength": "0.6",
        "post_fill_cooldown_seconds": 600,
    },
    exit_config={
        "stop_loss_percent": "0.5",
        "take_profit_percent": "1.0",
        "max_position_age_minutes": 390,
        "min_hold_minutes": 5,
    },
)
