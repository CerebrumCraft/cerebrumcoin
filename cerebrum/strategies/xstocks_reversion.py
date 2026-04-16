"""xStocks mean-reversion strategy config.

@decision DEC-XSTOCKS-002
@title xstocks_reversion — 24/7 mean-reversion on Kraken tokenized equities
@status accepted
@rationale Dedicated strategy consuming all signal types (RSI/MACD/BB/VWAP/SR)
scoped to AAPLx/MSFTx/NVDAx via symbols filter. Same mechanics as crypto
mean_reversion but independently tunable. $5,000 allocation. 24/7 operation.
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig


XSTOCKS_REVERSION_CONFIG = StrategyConfig(
    name="xstocks_reversion",
    initial_balance=Decimal("5000.0"),
    symbols=["AAPLx/USD", "MSFTx/USD", "NVDAx/USD"],
    signal_source_filter=None,
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.2"),
        SignalType.SENTIMENT: Decimal("0"),
        SignalType.NEWS: Decimal("0"),
        SignalType.REGIME: Decimal("0.5"),
    },
    aggregator_threshold=Decimal("0.4"),
    risk_overrides={
        "position_size_percent": "20.0",
        "stop_loss_percent": "1.0",
        "take_profit_percent": "1.5",
        "min_signal_strength": "0.65",
        "post_fill_cooldown_seconds": 1800,
    },
    exit_config={
        "max_position_age_minutes": 120,
        "min_hold_minutes": 15,
    },
)
