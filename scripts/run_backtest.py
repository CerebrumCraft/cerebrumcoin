#!/usr/bin/env python3
"""
Full multi-strategy backtester for CerebrumCoin.

Replays historical OHLCV data through the REAL trading pipeline — same code as live.
The only difference from live trading is the data source: OHLCV CSV/ccxt instead of
Kraken WebSocket.

Pipeline: OHLCV data -> MarketDataEvents -> EventBus -> CandleAggregator ->
          Signal Generators -> StrategyRegistry (6 strategies) -> RiskManagers ->
          PaperTradingAdapter -> PortfolioTrackers -> Results

What is skipped in backtest mode (DEC-BACKTEST-002):
  - NewsIngestionPipeline, LLMNewsAnalyzer, FearGreedSentiment — real-time push
    feeds with no historical equivalent; news_driven strategy gets zero NEWS signals
  - Conductor/DarwinianAllocator — uses LLM API; equal allocation used instead
  - WebDashboard — not needed for batch analysis

What IS included:
  - RegimeDetector (feeds on price data; runs from OHLCV candles normally)
  - All 6 StrategyRegistry strategies with their real risk rules
  - Global guards (RegimeTradeHaltRule, VolatilityGateRule, etc.)
  - PaperTradingAdapter with commission + slippage simulation

Usage::

    # Default: BTC/USD + ETH/USD, last 7 days, all 6 strategies
    python3 scripts/run_backtest.py

    # Custom
    python3 scripts/run_backtest.py --symbols BTC/USD,ETH/USD --days 14
    python3 scripts/run_backtest.py --symbols BTC/USD --days 30 --output data/backtest_results.json

@decision DEC-MONITOR-005
@title Backtest runner with OHLCV replay
@status accepted
@rationale Validates strategy on historical data. Uses ccxt to fetch OHLCV data,
caches to CSV for reuse. Replays data through existing event pipeline (same code
as live trading). Configurable date ranges and speedup factors.

@decision DEC-BACKTEST-001
@title Full multi-strategy backtest reuses entire live pipeline
@status accepted
@rationale The only change from live is the data source. All signal generators,
risk rules, strategy registry, and paper adapter are identical. This validates
strategies against real-world data using the same execution path they'll see in
production. The backtest is not a simulation of the pipeline — it IS the pipeline.

@decision DEC-BACKTEST-002
@title News/sentiment and Conductor skipped; RegimeDetector included
@status accepted
@rationale News feeds, LLM analyzer, fear/greed sentiment, and FinBERT are
real-time push systems with no historical data equivalent. Conductor uses LLM API
with cost and non-deterministic output. Both are skipped with notes in output.
RegimeDetector is included because it runs purely from price data (MarketDataEvents)
and provides regime-aware signal weighting that is part of the core strategy logic.
Equal static allocation replaces Conductor during backtest.
"""

import argparse
import asyncio
import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt

# Add project root to path for script invocation
sys.path.insert(0, str(Path(__file__).parents[1]))

from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import MarketDataEvent
from cerebrum.core.types import EventType
from cerebrum.risk.global_trade_rate import GlobalTradeRateLimitRule
from cerebrum.risk.rules import (
    MacroVolatilityGateRule,
    RegimeTradeHaltRule,
    SidewaysSuppressionRule,
    VolatilityGateRule,
)
from cerebrum.signals.candles import CandleAggregator
from cerebrum.signals.regime import RegimeDetector
from cerebrum.signals.technical import (
    BollingerBandsSignal,
    MACDSignal,
    RSISignal,
    VWAPSignal,
)
from cerebrum.signals.support_resistance import SupportResistanceSignal
from cerebrum.strategies.breakout import BREAKOUT_CONFIG
from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
from cerebrum.strategies.momentum import MOMENTUM_CONFIG
from cerebrum.strategies.news_driven import NEWS_DRIVEN_CONFIG
from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
from cerebrum.strategies.registry import StrategyRegistry
from cerebrum.strategies.swing_trading import SWING_TRADING_CONFIG

# ---------------------------------------------------------------------------
# OHLCV fetch with CSV caching (original implementation — kept intact)
# ---------------------------------------------------------------------------


async def fetch_ohlcv_data(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Path,
) -> list[dict]:
    """
    Fetch OHLCV data from exchange with caching.

    Returns list of candles with keys: timestamp, open, high, low, close, volume
    """
    # Generate cache filename
    cache_file = cache_dir / f"{exchange_name}_{symbol.replace('/', '_')}_{timeframe}_{start_date.date()}_{end_date.date()}.csv"

    # Check cache first
    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        candles = []
        with open(cache_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append({
                    "timestamp": float(row["timestamp"]),
                    "open": Decimal(row["open"]),
                    "high": Decimal(row["high"]),
                    "low": Decimal(row["low"]),
                    "close": Decimal(row["close"]),
                    "volume": Decimal(row["volume"]),
                })
        return candles

    # Fetch from exchange
    print(f"Fetching {symbol} data from {exchange_name}...")
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})

    try:
        # Convert dates to milliseconds
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        all_candles = []
        current = since

        while current < end_ms:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not ohlcv:
                break

            all_candles.extend(ohlcv)
            current = ohlcv[-1][0] + 1

            if len(ohlcv) < 1000:
                break

        # Convert to our format
        candles = []
        for candle in all_candles:
            if candle[0] >= end_ms:
                break
            candles.append({
                "timestamp": float(candle[0] / 1000),
                "open": Decimal(str(candle[1])),
                "high": Decimal(str(candle[2])),
                "low": Decimal(str(candle[3])),
                "close": Decimal(str(candle[4])),
                "volume": Decimal(str(candle[5])),
            })

        # Cache results
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", newline="") as f:
            if candles:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
                writer.writeheader()
                for candle in candles:
                    writer.writerow({k: str(v) for k, v in candle.items()})

        print(f"Cached {len(candles)} candles to {cache_file}")
        return candles

    finally:
        await exchange.close()


# ---------------------------------------------------------------------------
# Candle merge
# ---------------------------------------------------------------------------


def merge_candles_by_timestamp(
    candles_by_symbol: dict[str, list[dict]],
) -> list[dict]:
    """
    Merge per-symbol OHLCV candle lists into a single chronological stream.

    Each candle in the output carries a "symbol" key identifying its origin.
    Candles with equal timestamps are ordered by symbol name (deterministic).

    Args:
        candles_by_symbol: Mapping of symbol -> list of candle dicts.
            Each candle must have a "timestamp" key (float, Unix seconds).

    Returns:
        Flat list of candles sorted by timestamp ascending.
    """
    tagged: list[dict] = []
    for symbol, candles in candles_by_symbol.items():
        for candle in candles:
            tagged.append({**candle, "symbol": symbol})

    tagged.sort(key=lambda c: (c["timestamp"], c["symbol"]))
    return tagged


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


async def build_backtest_pipeline(
    config: Config,
    symbols: list[str],
    state_file: Path | None = None,
) -> tuple[EventBus, StrategyRegistry, PaperTradingAdapter, list[Any]]:
    """
    Wire the full multi-strategy backtest pipeline.

    Mirrors main.py:_setup_multi_strategy() exactly, except:
    - No news/LLM/sentiment/FinBERT (no real-time data in backtest)
    - No Conductor/DarwinianAllocator (equal allocation throughout)
    - No WebDashboard (not useful for batch analysis)
    - RegimeDetector IS included (feeds on price data, works from OHLCV)
    - PaperTradingAdapter uses a fresh temp state file (no corruption of live state)

    Args:
        config: Loaded Config (e.g. from paper.toml).
        symbols: List of symbol strings for logging/context.
        state_file: Path for paper adapter state. If None, a temp file is used.

    Returns:
        (bus, registry, paper_adapter, signal_generators)
        Caller must call registry.stop_all() + bus.stop() when done.
    """
    bus = EventBus()
    await bus.start()

    # Use a fresh state file so backtest doesn't corrupt live paper state.
    _owns_state_file = False
    if state_file is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        state_file = Path(tmp.name)
        tmp.close()
        _owns_state_file = True

    # --- PaperTradingAdapter (simulates fills) ---
    paper_adapter = PaperTradingAdapter(
        bus=bus,
        config={},
        initial_balance=config.paper.initial_balance_usd,
        commission_percent=config.paper.commission_percent,
        slippage_percent=config.paper.slippage_percent,
        state_file=state_file,
    )
    await paper_adapter.connect()

    # --- 1m CandleAggregator (shared across momentum/mean_reversion/breakout/
    #     range_trading/news_driven strategies) ---
    candle_agg = CandleAggregator(
        bus,
        interval_seconds=config.signals.candle_interval_seconds,
    )

    # --- 1h CandleAggregator (dedicated to swing_trading — DEC-SWING-001) ---
    candle_agg_1h = CandleAggregator(
        bus,
        interval_seconds=3600,
    )

    # --- 1m technical signal generators ---
    signal_generators: list[Any] = [
        RSISignal(
            bus,
            candle_agg,
            period=config.signals.rsi_period,
            oversold=config.signals.rsi_oversold,
            overbought=config.signals.rsi_overbought,
        ),
        MACDSignal(
            bus,
            candle_agg,
            fast=config.signals.macd_fast,
            slow=config.signals.macd_slow,
            signal=config.signals.macd_signal,
        ),
        BollingerBandsSignal(
            bus,
            candle_agg,
            period=config.signals.bb_period,
            std_dev=config.signals.bb_std_dev,
        ),
        VWAPSignal(
            bus,
            candle_agg,
            period=config.signals.vwap_period,
        ),
        SupportResistanceSignal(
            bus,
            candle_agg,
            pivot_lookback=config.signals.sr_pivot_lookback,
            min_touches=config.signals.sr_min_touches,
            proximity_pct=config.signals.sr_proximity_pct,
        ),
    ]

    # --- 1h signal generators (swing_trading only — DEC-SWING-001) ---
    signal_generators_1h: list[Any] = [
        RSISignal(
            bus,
            candle_agg_1h,
            period=config.signals.rsi_period,
            oversold=config.signals.rsi_oversold,
            overbought=config.signals.rsi_overbought,
            timeframe="1h",
        ),
        MACDSignal(
            bus,
            candle_agg_1h,
            fast=config.signals.macd_fast,
            slow=config.signals.macd_slow,
            signal=config.signals.macd_signal,
            timeframe="1h",
        ),
        BollingerBandsSignal(
            bus,
            candle_agg_1h,
            period=config.signals.bb_period,
            std_dev=config.signals.bb_std_dev,
            timeframe="1h",
        ),
        VWAPSignal(
            bus,
            candle_agg_1h,
            period=config.signals.vwap_period,
            timeframe="1h",
        ),
    ]
    signal_generators.extend(signal_generators_1h)

    # --- RegimeDetector (included — runs from price data, no external API) ---
    _regime_detector = RegimeDetector(
        bus,
        window_size=config.regime.window_size,
        update_interval=config.regime.update_interval,
        use_hmm=False,  # No hmmlearn dependency required for backtest
        cumulative_trend_threshold=config.regime.cumulative_trend_threshold,
        ma_slope_threshold=config.regime.ma_slope_threshold,
        mean_return_threshold=config.regime.mean_return_threshold,
        volatility_threshold=config.regime.volatility_threshold,
        ma_period=config.regime.ma_period,
        long_window_size=config.regime.long_window_size,
        long_cumulative_threshold=config.regime.long_cumulative_threshold,
    )

    # --- Shared global guards (constructed once, shared by reference — DEC-STRAT-003) ---
    global_guards = [
        RegimeTradeHaltRule(
            min_confidence=Decimal(str(config.regime.bear_halt_min_confidence)),
            bus=bus,
        ),
        VolatilityGateRule(
            min_range_pct=config.risk.volatility_gate_min_range_pct,
            window_size=config.risk.volatility_gate_window_size,
            bus=bus,
        ),
        MacroVolatilityGateRule(
            min_range_pct=config.risk.macro_volatility_min_range_pct,
            window_size=config.risk.macro_volatility_window_size,
            bus=bus,
        ),
        SidewaysSuppressionRule(
            min_range_pct=config.risk.sideways_suppression_min_range_pct,
            window_size=config.risk.sideways_suppression_window_size,
            bus=bus,
            exempt_strategies={"range_trading"},
        ),
        GlobalTradeRateLimitRule(
            max_trades_per_hour=40,
            bus=bus,
        ),
    ]

    # --- StrategyRegistry: all 6 strategies ---
    registry = StrategyRegistry(bus=bus, config=config)
    registry.register(MOMENTUM_CONFIG)
    registry.register(MEAN_REVERSION_CONFIG)
    registry.register(BREAKOUT_CONFIG)
    registry.register(RANGE_TRADING_CONFIG)
    registry.register(SWING_TRADING_CONFIG)
    registry.register(NEWS_DRIVEN_CONFIG)
    await registry.start_all(shared_global_rules=global_guards)

    return bus, registry, paper_adapter, signal_generators


# ---------------------------------------------------------------------------
# Per-candle publish helper
# ---------------------------------------------------------------------------


async def publish_candle(bus: EventBus, candle: dict) -> None:
    """
    Publish a single OHLCV candle as a MarketDataEvent.

    The candle dict must have keys: symbol, timestamp, close, volume.

    Args:
        bus: Running EventBus.
        candle: OHLCV candle dict with "symbol" tag set.
    """
    event = MarketDataEvent(
        event_type=EventType.MARKET_DATA,
        timestamp=candle["timestamp"],
        symbol=candle["symbol"],
        price=candle["close"],
        volume=candle["volume"],
    )
    await bus.publish(event)


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------


def collect_results(
    registry: StrategyRegistry,
    initial_balance_per_strategy: Decimal,
) -> dict[str, dict[str, Any]]:
    """
    Collect per-strategy performance metrics from PortfolioTrackers.

    Called after replay completes and bus queues have drained.

    Args:
        registry: StrategyRegistry with active pipelines.
        initial_balance_per_strategy: Starting balance per strategy (for P&L basis).

    Returns:
        Dict mapping strategy name -> stats dict with keys:
            total_pnl, realized_pnl, unrealized_pnl, drawdown_pct, cash_balance,
            total_equity, initial_balance, return_pct
    """
    results: dict[str, dict[str, Any]] = {}
    for name in registry.active_strategy_names():
        portfolio = registry.get_portfolio(name)
        if portfolio is None:
            continue

        realized_pnl, unrealized_pnl = portfolio.get_pnl()
        total_pnl = realized_pnl + unrealized_pnl
        drawdown_pct = portfolio.get_drawdown_percent()
        cash_balance = portfolio.get_cash_balance()
        total_equity = portfolio.get_total_equity()

        results[name] = {
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "drawdown_pct": drawdown_pct,
            "cash_balance": cash_balance,
            "total_equity": total_equity,
            "initial_balance": initial_balance_per_strategy,
            "return_pct": (
                total_pnl / initial_balance_per_strategy * Decimal("100")
                if initial_balance_per_strategy > 0
                else Decimal("0")
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Full backtest orchestrator
# ---------------------------------------------------------------------------


async def run_backtest(
    symbols: list[str],
    candles_by_symbol: dict[str, list[dict]],
    config: Config,
    state_file: Path | None = None,
) -> dict[str, Any]:
    """
    Run the full multi-strategy backtest.

    Wires the pipeline, replays candles, drains the event bus, collects results.

    Args:
        symbols: Symbol list (for reporting).
        candles_by_symbol: Per-symbol OHLCV data (already fetched/loaded).
        config: Config loaded from TOML.
        state_file: Optional path for paper adapter state (temp file if None).

    Returns:
        Dict with keys: "period", "symbols", "candle_counts", "per_strategy",
        "aggregate", "notes" — suitable for JSON serialization.
    """
    if not candles_by_symbol or not any(candles_by_symbol.values()):
        return {
            "period": {"start": None, "end": None},
            "symbols": symbols,
            "candle_counts": {},
            "per_strategy": {},
            "aggregate": {},
            "notes": ["No candle data available."],
        }

    # Merge all symbol streams into a single chronological list
    merged = merge_candles_by_timestamp(candles_by_symbol)
    total_candles = len(merged)

    start_ts = merged[0]["timestamp"]
    end_ts = merged[-1]["timestamp"]
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)

    print(f"\nBacktest period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Total candles: {total_candles:,}")
    for sym, candles in candles_by_symbol.items():
        print(f"  {sym}: {len(candles):,} candles")
    print()

    # Build pipeline
    bus, registry, paper_adapter, _signal_generators = await build_backtest_pipeline(
        config=config,
        symbols=symbols,
        state_file=state_file,
    )

    # --- Replay ---
    print("Replaying candles through pipeline...")

    # 1ms drain after each candle gives all async subscriber tasks (CandleAggregator,
    # signal generators, risk managers) time to process before the next tick arrives.
    # At 10,000 candles this adds ~10s of real time — acceptable for analysis use.
    DRAIN_INTERVAL = 0.001
    PROGRESS_EVERY = max(1, total_candles // 20)  # ~5% increments

    try:
        for i, candle in enumerate(merged):
            await publish_candle(bus, candle)
            await asyncio.sleep(DRAIN_INTERVAL)

            if (i + 1) % PROGRESS_EVERY == 0 or i == total_candles - 1:
                pct = (i + 1) / total_candles * 100
                print(f"  {i + 1:,}/{total_candles:,} candles ({pct:.0f}%)")

        # Final drain: allow last candle's event chain to fully propagate
        await asyncio.sleep(0.05)

    finally:
        await registry.stop_all()
        await bus.stop()

    print("\nReplay complete. Collecting results...")

    # --- Results ---
    num_strategies = len(registry.active_strategy_names())
    initial_balance_per = (
        config.paper.initial_balance_usd / Decimal(str(max(1, num_strategies)))
    )

    per_strategy = collect_results(registry, initial_balance_per_strategy=initial_balance_per)

    # Aggregate across all strategies
    total_pnl = sum(s["total_pnl"] for s in per_strategy.values())
    total_realized = sum(s["realized_pnl"] for s in per_strategy.values())
    total_unrealized = sum(s["unrealized_pnl"] for s in per_strategy.values())
    max_drawdown = max(
        (s["drawdown_pct"] for s in per_strategy.values()),
        default=Decimal("0"),
    )

    return {
        "period": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "days": (end_dt - start_dt).days + 1,
        },
        "symbols": symbols,
        "candle_counts": {
            sym: len(candles) for sym, candles in candles_by_symbol.items()
        },
        "per_strategy": {
            name: {k: str(v) if isinstance(v, Decimal) else v for k, v in stats.items()}
            for name, stats in per_strategy.items()
        },
        "aggregate": {
            "total_pnl": str(total_pnl),
            "realized_pnl": str(total_realized),
            "unrealized_pnl": str(total_unrealized),
            "max_drawdown_pct": str(max_drawdown),
            "initial_balance": str(config.paper.initial_balance_usd),
        },
        "notes": [
            "News/sentiment feeds not available in backtest mode (news_driven strategy gets zero NEWS signals).",
            "Equal allocation used (no Conductor/DarwinianAllocator).",
            f"Commission: {config.paper.commission_percent}% per side (Kraken maker fee).",
        ],
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(results: dict[str, Any]) -> None:
    """Print a human-readable backtest summary to stdout."""
    period = results.get("period", {})
    symbols = results.get("symbols", [])
    candle_counts = results.get("candle_counts", {})
    per_strategy = results.get("per_strategy", {})
    aggregate = results.get("aggregate", {})
    notes = results.get("notes", [])

    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    if period.get("start") and period.get("end"):
        print(f"Period:  {period['start']} to {period['end']} ({period.get('days', '?')} days)")
    print(f"Symbols: {', '.join(symbols)}")
    total_candles = sum(candle_counts.values())
    counts_str = ", ".join(f"{s}: {n:,}" for s, n in candle_counts.items())
    print(f"Candles: {total_candles:,} ({counts_str})")

    if per_strategy:
        print()
        print("Per-Strategy Performance:")
        header = f"  {'Strategy':<18} {'TotalPnL':>10} {'ReturnPct':>10} {'Drawdown':>10} {'Cash':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for name, stats in sorted(per_strategy.items()):
            total_pnl_val = Decimal(stats["total_pnl"])
            return_pct = Decimal(stats["return_pct"])
            drawdown = Decimal(stats["drawdown_pct"])
            cash = Decimal(stats["cash_balance"])
            print(
                f"  {name:<18} "
                f"{float(total_pnl_val):>+10.2f} "
                f"{float(return_pct):>9.2f}% "
                f"{float(drawdown):>9.2f}% "
                f"${float(cash):>11.2f}"
            )

    if aggregate:
        print()
        print("Aggregate:")
        total = Decimal(aggregate.get("total_pnl", "0"))
        realized = Decimal(aggregate.get("realized_pnl", "0"))
        unrealized = Decimal(aggregate.get("unrealized_pnl", "0"))
        drawdown = Decimal(aggregate.get("max_drawdown_pct", "0"))
        initial = Decimal(aggregate.get("initial_balance", "10000"))
        return_pct = total / initial * 100 if initial > 0 else Decimal("0")

        print(f"  Net P&L:        ${float(total):>+10.2f}")
        print(f"  Realized P&L:   ${float(realized):>+10.2f}")
        print(f"  Unrealized P&L: ${float(unrealized):>+10.2f}")
        print(f"  Return:         {float(return_pct):>+9.2f}%")
        print(f"  Max Drawdown:   {float(drawdown):>9.2f}%")

    if notes:
        print()
        print("Notes:")
        for note in notes:
            print(f"  - {note}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse backtest CLI arguments.

    Separated from main() so it can be tested without running the full script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed argparse.Namespace with attributes:
            symbols (list[str]), days (int), speedup (float),
            output (Path | None), exchange (str), timeframe (str), cache_dir (Path)
    """
    parser = argparse.ArgumentParser(
        description="Run multi-strategy backtest on historical OHLCV data"
    )
    parser.add_argument(
        "--symbols",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["BTC/USD", "ETH/USD"],
        help="Comma-separated trading pairs (default: BTC/USD,ETH/USD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to backtest (default: 7)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=10000.0,
        help="Replay speedup factor — kept for CLI compatibility but backtest runs as fast as possible (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this file (optional)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="kraken",
        help="Exchange name for OHLCV fetch (default: kraken)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Candle timeframe (default: 1m)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/backtest_cache"),
        help="Cache directory for OHLCV data (default: data/backtest_cache)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/paper.toml"),
        help="Config file (default: config/paper.toml)",
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    config = Config.from_toml(args.config)

    async def _run() -> None:
        # Fetch OHLCV data for all symbols
        candles_by_symbol: dict[str, list[dict]] = {}
        for symbol in args.symbols:
            candles = await fetch_ohlcv_data(
                args.exchange,
                symbol,
                args.timeframe,
                start_date,
                end_date,
                args.cache_dir,
            )
            if candles:
                candles_by_symbol[symbol] = candles
            else:
                print(f"WARNING: No data for {symbol} — skipping")

        if not candles_by_symbol:
            print("No data available for any symbol. Exiting.")
            return

        # Run full backtest
        results = await run_backtest(
            symbols=list(candles_by_symbol.keys()),
            candles_by_symbol=candles_by_symbol,
            config=config,
        )

        # Print results
        print_results(results)

        # Write JSON output if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults written to {args.output}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
