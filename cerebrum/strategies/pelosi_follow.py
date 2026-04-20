"""Pelosi-follow strategy config.

Mirror of orb_stocks.py for the congressional-trade signal pipeline.

# @decision DEC-PELOSI-UNIV-001
# @title Pelosi-only universe, schema extensible for top-N members
# @status accepted
# @rationale Narrow wedge for v1 validation. The signal generator already loops
# over a ``members`` config list, so Phase D (top-N congress members) is purely
# a config change — no migration or new code required. Universe = 7 liquid
# large-caps that Pelosi has historically traded (NVDA, AAPL, MSFT, GOOGL,
# AVGO, TEM, PANW). Widening the universe requires only a config change.

# @decision DEC-PELOSI-SIZE-001
# @title Fixed-dollar position sizing for Phase A/B
# @status accepted
# @rationale STOCK Act discloses amount ranges ($1k-$15k, $15k-$50k, etc.)
# but a clean bucket→size mapping requires Phase C investment. For Phase A/B
# we use a flat position_size_usd = $500 so any signal produces a
# consistent-size equity order the paper adapter can fill without special-casing.
# Phase C switches to bucket-midpoint sizing with a position_size_percent cap.
# The ``position_size_usd`` field is stored as a risk_override key understood by
# StrategyRegistry's PositionSizingRule when position_size_usd is set.
"""

from decimal import Decimal

from cerebrum.core.types import SignalType
from cerebrum.strategies.base import StrategyConfig


# ---------------------------------------------------------------------------
# Universe (DEC-PELOSI-UNIV-001)
# ---------------------------------------------------------------------------

_PELOSI_SYMBOLS = [
    "NVDA",   # Pelosi's most-disclosed stock (NVDA call buys, 2023-2026)
    "AAPL",   # Long-running holding via calls and stock
    "MSFT",   # Disclosed stock purchases 2024-2025
    "GOOGL",  # Class A shares, multiple disclosures
    "AVGO",   # Call purchases 2024
    "TEM",    # Tempus AI — newer addition, stock purchases 2025
    "PANW",   # Palo Alto Networks — cybersecurity holding
]


# ---------------------------------------------------------------------------
# Strategy config (DEC-PELOSI-SIZE-001)
# ---------------------------------------------------------------------------

PELOSI_FOLLOW_CONFIG = StrategyConfig(
    name="pelosi_follow",
    initial_balance=Decimal("5000.0"),
    symbols=_PELOSI_SYMBOLS,
    signal_source_filter="Congressional",
    aggregator_weights={
        SignalType.TECHNICAL: Decimal("0"),
        SignalType.SENTIMENT: Decimal("0"),
        SignalType.NEWS: Decimal("1.0"),  # Congressional signals arrive as SignalType.NEWS
        SignalType.REGIME: Decimal("0"),
    },
    aggregator_threshold=Decimal("0.4"),
    risk_overrides={
        # Fixed $500 per trade (DEC-PELOSI-SIZE-001).
        # Overrides percentage-based sizing so a $5k allocation doesn't auto-size
        # into $250 trades (5%) — we want flat dollar amounts for sparse signals.
        "position_size_usd": "500.0",
        "min_signal_strength": "0.5",   # Lower than crypto — filing lag already filters noise
        "post_fill_cooldown_seconds": 3600,  # 1h cooldown — sparse strategy, avoid churn
    },
    exit_config={
        "stop_loss_percent": "3.0",   # Wider SL — congressional signals are slow-moving
        "take_profit_percent": "10.0",  # Wide TP — lag means we ride the trend, not scalp
        "max_position_age_minutes": 43200,  # 30 days (in minutes) — hold until disclosed sell (Phase C)
        "min_hold_minutes": 60,         # 1h minimum — these are not scalp trades
    },
)
