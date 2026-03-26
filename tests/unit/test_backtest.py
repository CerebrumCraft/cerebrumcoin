"""
Tests for the full multi-strategy backtest script.

@decision DEC-BACKTEST-001
@title Test-first backtest validation with synthetic OHLCV data
@status accepted
@rationale The backtest replays the real trading pipeline — not a stub. Tests must
verify that the pipeline wires correctly, candles are merged chronologically across
symbols, MarketDataEvents are published for each candle, and per-strategy stats are
collected after replay. A small synthetic dataset (30 candles with clear trends)
is used so tests are deterministic and fast without requiring exchange access.

Sacred Practice #5: no internal mocks. EventBus, StrategyRegistry, PortfolioTracker,
PaperTradingAdapter are used directly. Only the OHLCV fetch step (external API) is
replaced by supplying synthetic candles directly.
"""

import asyncio
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

BASE_TIMESTAMP = 1_700_000_000.0  # fixed epoch for reproducibility
CANDLE_INTERVAL_S = 60            # 1-minute candles


def make_candle(
    symbol: str,
    i: int,
    base_price: float = 50_000.0,
    trend: float = 0.0,
) -> dict[str, Any]:
    """
    Build a synthetic OHLCV candle.

    Args:
        symbol: Trading pair string, e.g. "BTC/USD"
        i: Candle index (0-based). Timestamp advances i * CANDLE_INTERVAL_S.
        base_price: Starting price.
        trend: Per-candle price change (positive = uptrend).
    """
    price = Decimal(str(base_price + trend * i))
    high = price * Decimal("1.002")
    low = price * Decimal("0.998")
    return {
        "symbol": symbol,
        "timestamp": BASE_TIMESTAMP + i * CANDLE_INTERVAL_S,
        "open": price,
        "high": high,
        "low": low,
        "close": price,
        "volume": Decimal("10.0"),
    }


def synthetic_candles(
    symbol: str,
    count: int = 30,
    base_price: float = 50_000.0,
    trend: float = 50.0,
) -> list[dict[str, Any]]:
    """Return a list of count candles with a linear uptrend."""
    return [make_candle(symbol, i, base_price, trend) for i in range(count)]


# ---------------------------------------------------------------------------
# 1. OHLCV merge — candles from 2 symbols merged chronologically
# ---------------------------------------------------------------------------

class TestMergeCandles:
    """merge_candles_by_timestamp() must interleave two symbol streams by time."""

    def test_merge_two_symbols_interleaved(self) -> None:
        """Merged stream interleaves BTC and ETH candles in timestamp order."""
        from scripts.run_backtest import merge_candles_by_timestamp

        btc = synthetic_candles("BTC/USD", count=3, base_price=50_000.0)
        eth = synthetic_candles("ETH/USD", count=3, base_price=3_000.0)

        merged = merge_candles_by_timestamp({"BTC/USD": btc, "ETH/USD": eth})

        # All 6 candles present
        assert len(merged) == 6

        # Strictly non-decreasing timestamps
        for prev, curr in zip(merged, merged[1:]):
            assert curr["timestamp"] >= prev["timestamp"]

    def test_merge_preserves_symbol_tag(self) -> None:
        """Each candle in the merged stream carries its originating symbol."""
        from scripts.run_backtest import merge_candles_by_timestamp

        btc = synthetic_candles("BTC/USD", count=2)
        eth = synthetic_candles("ETH/USD", count=2)
        merged = merge_candles_by_timestamp({"BTC/USD": btc, "ETH/USD": eth})

        symbols = {c["symbol"] for c in merged}
        assert "BTC/USD" in symbols
        assert "ETH/USD" in symbols

    def test_merge_empty_input(self) -> None:
        """Empty input produces empty output without error."""
        from scripts.run_backtest import merge_candles_by_timestamp

        assert merge_candles_by_timestamp({}) == []

    def test_merge_single_symbol(self) -> None:
        """Single symbol passes through unchanged."""
        from scripts.run_backtest import merge_candles_by_timestamp

        btc = synthetic_candles("BTC/USD", count=5)
        merged = merge_candles_by_timestamp({"BTC/USD": btc})
        assert len(merged) == 5
        assert all(c["symbol"] == "BTC/USD" for c in merged)


# ---------------------------------------------------------------------------
# 2. Pipeline wiring — all components start and bus has subscribers
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    """build_backtest_pipeline() creates a wired pipeline with active subscribers."""

    @pytest.mark.asyncio
    async def test_pipeline_starts_and_stops(self) -> None:
        """build_backtest_pipeline() returns a running registry and bus."""
        from scripts.run_backtest import build_backtest_pipeline
        from cerebrum.core.config import Config
        from cerebrum.core.types import EventType

        config = Config.from_toml(Path("config/paper.toml"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            bus, registry, paper_adapter, _ = await build_backtest_pipeline(
                config=config,
                symbols=["BTC/USD"],
                state_file=state_file,
            )

            # Bus is running and has subscribers for MARKET_DATA
            assert bus.get_subscriber_count(EventType.MARKET_DATA) > 0
            # Registry has active strategies
            assert len(registry.active_strategy_names()) > 0

            await registry.stop_all()
            await bus.stop()
        finally:
            state_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_all_six_strategies_registered(self) -> None:
        """All six strategies are active after pipeline wiring."""
        from scripts.run_backtest import build_backtest_pipeline
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            bus, registry, paper_adapter, _ = await build_backtest_pipeline(
                config=config,
                symbols=["BTC/USD"],
                state_file=state_file,
            )

            active = registry.active_strategy_names()
            expected = {
                "momentum", "mean_reversion", "breakout",
                "range_trading", "swing_trading", "news_driven",
            }
            assert expected.issubset(set(active)), (
                f"Missing strategies: {expected - set(active)}"
            )

            await registry.stop_all()
            await bus.stop()
        finally:
            state_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Single-candle replay — MarketDataEvent published per candle
# ---------------------------------------------------------------------------

class TestSingleCandleReplay:
    """Publishing one candle emits exactly one MarketDataEvent to the bus."""

    @pytest.mark.asyncio
    async def test_candle_emits_market_data_event(self) -> None:
        """A single candle replay produces a MarketDataEvent with correct price."""
        from cerebrum.core.bus import EventBus
        from cerebrum.core.events import MarketDataEvent
        from cerebrum.core.types import EventType

        bus = EventBus(queue_size=100)
        await bus.start()

        received: list[MarketDataEvent] = []

        async def capture(event: MarketDataEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.MARKET_DATA, capture, "test_capture")

        candle = make_candle("BTC/USD", 0, base_price=50_000.0)

        from scripts.run_backtest import publish_candle
        await publish_candle(bus, candle)

        # Yield to event loop so the subscriber task processes the queue
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].symbol == "BTC/USD"
        assert received[0].price == candle["close"]
        assert received[0].volume == candle["volume"]

        await bus.stop()


# ---------------------------------------------------------------------------
# 4. Results collection — per-strategy stats after replay
# ---------------------------------------------------------------------------

class TestResultsCollection:
    """collect_results() returns per-strategy stats after a replay completes."""

    @pytest.mark.asyncio
    async def test_results_have_all_strategies(self) -> None:
        """collect_results() returns an entry for every active strategy."""
        from scripts.run_backtest import build_backtest_pipeline, collect_results
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            bus, registry, paper_adapter, _ = await build_backtest_pipeline(
                config=config,
                symbols=["BTC/USD"],
                state_file=state_file,
            )

            # No candles replayed — expect zero-trade results but all strategies present
            results = collect_results(registry, initial_balance_per_strategy=Decimal("1666.67"))

            active = set(registry.active_strategy_names())
            result_keys = set(results.keys())
            assert active == result_keys

            await registry.stop_all()
            await bus.stop()
        finally:
            state_file.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_results_keys_present(self) -> None:
        """Each strategy result contains required metric keys."""
        from scripts.run_backtest import build_backtest_pipeline, collect_results
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = Path(f.name)

        try:
            bus, registry, paper_adapter, _ = await build_backtest_pipeline(
                config=config,
                symbols=["BTC/USD"],
                state_file=state_file,
            )

            results = collect_results(registry, initial_balance_per_strategy=Decimal("1666.67"))

            required_keys = {"total_pnl", "realized_pnl", "unrealized_pnl",
                             "drawdown_pct", "cash_balance"}
            for name, stats in results.items():
                assert required_keys.issubset(set(stats.keys())), (
                    f"Strategy '{name}' missing keys: {required_keys - set(stats.keys())}"
                )

            await registry.stop_all()
            await bus.stop()
        finally:
            state_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. No-data handling — empty candles list produces empty results
# ---------------------------------------------------------------------------

class TestNoDataHandling:
    """Empty candle input is handled gracefully."""

    def test_merge_empty_candles_per_symbol(self) -> None:
        """Symbol with empty candle list contributes nothing to the merge."""
        from scripts.run_backtest import merge_candles_by_timestamp

        merged = merge_candles_by_timestamp({"BTC/USD": [], "ETH/USD": []})
        assert merged == []

    def test_merge_one_empty_one_full(self) -> None:
        """Only non-empty symbol contributes candles."""
        from scripts.run_backtest import merge_candles_by_timestamp

        btc = synthetic_candles("BTC/USD", count=5)
        merged = merge_candles_by_timestamp({"BTC/USD": btc, "ETH/USD": []})
        assert len(merged) == 5


# ---------------------------------------------------------------------------
# 6. CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    """parse_args() correctly handles --symbols, --days, --speedup, --output."""

    def test_defaults(self) -> None:
        """Default values match the spec: BTC/USD+ETH/USD, 7 days, no output."""
        from scripts.run_backtest import parse_args

        args = parse_args([])
        assert "BTC/USD" in args.symbols
        assert "ETH/USD" in args.symbols
        assert args.days == 7
        assert args.output is None

    def test_custom_symbols(self) -> None:
        """--symbols accepts comma-separated pairs."""
        from scripts.run_backtest import parse_args

        args = parse_args(["--symbols", "BTC/USD,ETH/USD"])
        assert args.symbols == ["BTC/USD", "ETH/USD"]

    def test_custom_days(self) -> None:
        """--days overrides the default."""
        from scripts.run_backtest import parse_args

        args = parse_args(["--days", "14"])
        assert args.days == 14

    def test_speedup_flag(self) -> None:
        """--speedup is parsed as float."""
        from scripts.run_backtest import parse_args

        args = parse_args(["--speedup", "10000"])
        assert args.speedup == 10000.0

    def test_output_flag(self) -> None:
        """--output stores a Path."""
        from scripts.run_backtest import parse_args

        args = parse_args(["--output", "data/backtest_results.json"])
        assert args.output == Path("data/backtest_results.json")

    def test_single_symbol(self) -> None:
        """Single symbol without comma parses to a one-element list."""
        from scripts.run_backtest import parse_args

        args = parse_args(["--symbols", "BTC/USD"])
        assert args.symbols == ["BTC/USD"]
