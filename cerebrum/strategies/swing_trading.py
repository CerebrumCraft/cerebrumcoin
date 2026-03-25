"""
Swing trading strategy configuration.

Trades on 1-hour candles for fewer, higher-conviction trades.
Directly addresses commission drag (#1 enemy from Session 4: 64% of gross).

@decision DEC-SWING-001
@title 1-hour timeframe swing strategy to reduce commission drag
@status accepted
@rationale Session 4 showed $115 commission on $179 gross (64%). Most profitable
trades held 1-4 hours. 1h candles capture multi-hour trends, reducing trade
frequency while increasing per-trade profit. Same RSI/MACD/BB pipeline,
different timeframe. signal_timeframe_filter="1h" ensures the swing aggregator
only consumes 1h signals and ignores the 1m scalp signals from other generators.
Wider TP (5%) and longer max_position_age (8h) match the slower 1h rhythm.
Capital allocation is 1/5 of $10k for the 5-strategy equal split.
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig

SWING_TRADING_CONFIG = StrategyConfig(
    name="swing_trading",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("1.5"),
        SignalType.SENTIMENT: Decimal("0.3"),
        SignalType.NEWS: Decimal("0.4"),
        SignalType.REGIME: Decimal("0.8"),
    },
    aggregator_threshold=Decimal("0.5"),
    signal_timeframe_filter="1h",
    risk_overrides={
        "min_signal_strength": "0.5",
        "position_size_percent": "5.0",
        "post_fill_cooldown_seconds": 3600,
    },
    exit_config={
        "stop_loss_percent": "3.0",
        "take_profit_percent": "5.0",
        "max_position_age_minutes": 480,
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "1.0",
    },
    initial_balance=Decimal("2000.00"),
    symbols=["BTC/USD", "ETH/USD"],
)
