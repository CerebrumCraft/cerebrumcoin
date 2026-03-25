"""
News-driven trading strategy configuration.

Trades primarily on LLM-analyzed news signals. When Claude Haiku identifies
significant market-moving news (regulatory changes, exchange hacks, ETF
approvals, partnerships), this strategy acts on it. Technical signals are
suppressed to near-zero — the thesis comes from news, not charts.

@decision DEC-NEWS-001
@title News-heavy signal weighting for event-driven trading
@status accepted
@rationale The LLM news analyzer already generates SignalType.NEWS signals
with action/strength/confidence from Claude Haiku. A dedicated strategy
with NEWS weight 2.0 (vs 0.3 default) ensures news-driven entries aren't
diluted by conflicting technical signals. Wider exits (4% TP / 2.5% SL)
because news-driven moves tend to be larger and more sustained than
technical patterns. Longer cooldown (1800s) prevents overreaction to
follow-up commentary on the same event.
"""
from decimal import Decimal
from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig

NEWS_DRIVEN_CONFIG = StrategyConfig(
    name="news_driven",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("0.2"),   # Suppress technicals — news is the thesis
        SignalType.SENTIMENT: Decimal("0.8"),   # Sentiment confirms news direction
        SignalType.NEWS: Decimal("2.0"),         # NEWS is THE signal
        SignalType.REGIME: Decimal("0.5"),       # Regime provides context but doesn't drive
    },
    aggregator_threshold=Decimal("0.3"),         # Moderate threshold — NEWS signals are infrequent
    risk_overrides={
        "min_signal_strength": "0.4",            # Accept moderate NEWS signals
        "position_size_percent": "4.0",          # Larger positions — news-driven moves are bigger
        "post_fill_cooldown_seconds": 1800,      # 30 min — don't overtrade on same event
    },
    exit_config={
        "stop_loss_percent": "2.5",              # Medium stop — news moves can be volatile
        "take_profit_percent": "4.0",            # Wide target — news moves are larger
        "max_position_age_minutes": 240,         # 4 hours — news impact decays
        "adaptive_tp": True,
        "tp_multiplier": "1.5",
        "min_tp_percent": "0.5",
    },
    initial_balance=Decimal("1666.67"),           # 6-strategy split
    symbols=["BTC/USD", "ETH/USD"],
)
