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
            bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
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
            bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
                config=config,
                symbols=["BTC/USD"],
                state_file=state_file,
            )

            active = registry.active_strategy_names()
            expected = {
                # swing_trading disabled (DEC-TUNE-005)
                "momentum", "mean_reversion", "breakout",
                "range_trading", "news_driven",
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
            bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
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
            bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
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


# ---------------------------------------------------------------------------
# 7. Parameter scaling — scale_backtest_params() returns correct values
# ---------------------------------------------------------------------------

class TestScaleBacktestParams:
    """
    scale_backtest_params() returns correctly scaled component parameters for
    non-1m candle intervals.

    The live system is tuned for ~1 tick/sec MarketDataEvent rate. Backtesting
    with 15m candles means 1 tick per 900 seconds. All time-sensitive parameters
    must be adjusted so the backtest covers the same real-time durations.
    """

    def test_1m_candles_identity(self) -> None:
        """1m candles (60s interval) produce identity scaling — params unchanged."""
        from scripts.run_backtest import scale_backtest_params

        params = scale_backtest_params(candle_interval_seconds=60)

        # Guard windows: 300 ticks at 1/sec = 5 min → at 60s/tick: ~5 ticks
        # But we cap at min values so results should stay reasonable
        assert "volatility_gate_window" in params
        assert "macro_volatility_window" in params
        assert "sideways_suppression_window" in params
        assert "aggregation_window_seconds" in params
        assert "post_fill_cooldown_seconds" in params

    def test_15m_candles_vol_gate_scaled_down(self) -> None:
        """15m candles shrink the volatility gate window to maintain 5-min coverage."""
        from scripts.run_backtest import scale_backtest_params

        params = scale_backtest_params(candle_interval_seconds=900)

        # Live: 300 ticks at 1 tick/sec = 5 min of data.
        # At 15m (900s/tick): 5 min = 300s → 300/900 < 1 tick → use min of 3.
        assert params["volatility_gate_window"] >= 1
        # Must be much smaller than the original 300 ticks
        assert params["volatility_gate_window"] < 300

    def test_15m_candles_macro_window_scaled_down(self) -> None:
        """15m candles shrink the macro window but still cover ~5 hours."""
        from scripts.run_backtest import scale_backtest_params

        params = scale_backtest_params(candle_interval_seconds=900)

        # Live: 18000 ticks at 1 tick/sec = 5 hours.
        # At 15m: 5h = 18000s → 18000/900 = 20 ticks.
        assert params["macro_volatility_window"] == 20
        # Sideways window mirrors macro window
        assert params["sideways_suppression_window"] == 20

    def test_1h_candles_macro_window(self) -> None:
        """1h candles: 18000s / 3600s = 5 ticks for ~5-hour macro window."""
        from scripts.run_backtest import scale_backtest_params

        params = scale_backtest_params(candle_interval_seconds=3600)

        assert params["macro_volatility_window"] == 5
        assert params["sideways_suppression_window"] == 5

    def test_aggregation_window_larger_than_candle_interval(self) -> None:
        """aggregation_window_seconds must be >= candle_interval to allow signal combination."""
        from scripts.run_backtest import scale_backtest_params

        for candle_secs in [60, 300, 900, 3600]:
            params = scale_backtest_params(candle_interval_seconds=candle_secs)
            assert params["aggregation_window_seconds"] >= candle_secs, (
                f"aggregation_window {params['aggregation_window_seconds']} < "
                f"candle_interval {candle_secs} — signals would always expire"
            )

    def test_post_fill_cooldown_at_least_one_second(self) -> None:
        """post_fill_cooldown_seconds must be >= 1 to allow at least one trade per session."""
        from scripts.run_backtest import scale_backtest_params

        for candle_secs in [60, 900, 3600]:
            params = scale_backtest_params(candle_interval_seconds=candle_secs)
            assert params["post_fill_cooldown_seconds"] >= 1, (
                f"cooldown {params['post_fill_cooldown_seconds']} would block all trades"
            )

    def test_15m_pipeline_uses_scaled_aggregation_window(self) -> None:
        """build_backtest_pipeline() at 15m uses aggregation_window > 900s."""
        # Verifies the pipeline constructor respects scale_backtest_params output.
        # We check by inspecting the SignalAggregator window_seconds attribute.
        import asyncio
        import tempfile
        from pathlib import Path
        from scripts.run_backtest import build_backtest_pipeline
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))

        async def _check() -> None:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                state_file = Path(f.name)

            try:
                bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
                    config=config,
                    symbols=["BTC/USD"],
                    state_file=state_file,
                    candle_interval_seconds=900,
                )

                # Every strategy's aggregator must have window > candle interval
                for name in registry.active_strategy_names():
                    agg = registry.get_aggregator(name)
                    assert agg is not None, f"No aggregator for {name}"
                    assert agg._window_seconds >= 900, (
                        f"Strategy {name}: aggregation_window={agg._window_seconds} "
                        f"< candle_interval=900 — signals will always expire"
                    )

                await registry.stop_all()
                await bus.stop()
            finally:
                state_file.unlink(missing_ok=True)

        asyncio.get_event_loop().run_until_complete(_check())

    def test_1m_pipeline_preserves_config_aggregation_window(self) -> None:
        """build_backtest_pipeline() at 1m leaves aggregation_window at config value."""
        import asyncio
        import tempfile
        from pathlib import Path
        from scripts.run_backtest import build_backtest_pipeline
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))
        config_window = config.signals.aggregation_window_seconds  # 120

        async def _check() -> None:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                state_file = Path(f.name)

            try:
                bus, registry, paper_adapter, *_ = await build_backtest_pipeline(
                    config=config,
                    symbols=["BTC/USD"],
                    state_file=state_file,
                    candle_interval_seconds=60,
                )

                for name in registry.active_strategy_names():
                    agg = registry.get_aggregator(name)
                    assert agg is not None
                    # At 1m, window should equal config value (120s) since 120 >= 60
                    assert agg._window_seconds == config_window, (
                        f"Strategy {name}: window {agg._window_seconds} != config {config_window}"
                    )

                await registry.stop_all()
                await bus.stop()
            finally:
                state_file.unlink(missing_ok=True)

        asyncio.get_event_loop().run_until_complete(_check())


# ---------------------------------------------------------------------------
# 8. BacktestClock — virtual time source for backtest mode
# ---------------------------------------------------------------------------


class TestBacktestClock:
    """
    BacktestClock tracks the latest candle timestamp and never goes backwards.

    @decision DEC-BACKTEST-004
    @title BacktestClock injectable virtual clock for backtest mode
    @status accepted
    @rationale Replaces wall-clock time.time() in SignalAggregator and
    PostFillCooldownRule during backtest so historical signals are compared
    against historical time, not today's wall-clock.
    """

    def test_initial_time_is_zero(self) -> None:
        """BacktestClock starts at 0.0 before any candle is processed."""
        from scripts.run_backtest import BacktestClock

        clock = BacktestClock()
        assert clock() == 0.0

    def test_advance_sets_time(self) -> None:
        """advance() moves the clock forward to the given timestamp."""
        from scripts.run_backtest import BacktestClock

        clock = BacktestClock()
        clock.advance(1_742_000_000.0)
        assert clock() == 1_742_000_000.0

    def test_advance_never_goes_backwards(self) -> None:
        """advance() with a smaller timestamp does not decrease the clock."""
        from scripts.run_backtest import BacktestClock

        clock = BacktestClock()
        clock.advance(1_742_000_100.0)
        clock.advance(1_742_000_050.0)  # earlier — should be ignored
        assert clock() == 1_742_000_100.0

    def test_advance_multiple_candles(self) -> None:
        """Clock advances monotonically across a series of candles."""
        from scripts.run_backtest import BacktestClock

        clock = BacktestClock()
        timestamps = [
            BASE_TIMESTAMP + i * CANDLE_INTERVAL_S for i in range(5)
        ]
        for ts in timestamps:
            clock.advance(ts)
        assert clock() == timestamps[-1]

    def test_clock_callable_returns_float(self) -> None:
        """BacktestClock() returns a float — compatible with time.time() signature."""
        from scripts.run_backtest import BacktestClock

        clock = BacktestClock()
        clock.advance(1_742_000_000.123)
        result = clock()
        assert isinstance(result, float)

    def test_aggregator_uses_backtest_clock(self) -> None:
        """
        SignalAggregator respects an injected BacktestClock: signals are NOT
        expired when the clock is set to match their historical timestamps.

        Without BacktestClock, _clean_old_signals() would compare historical
        signal timestamps against time.time() (today), instantly expiring them.
        With BacktestClock advanced to match, signals survive.
        """
        import asyncio
        from decimal import Decimal
        from scripts.run_backtest import BacktestClock
        from cerebrum.signals.aggregator import SignalAggregator
        from cerebrum.core.bus import EventBus
        from cerebrum.core.events import SignalEvent
        from cerebrum.core.types import EventType, SignalAction, SignalType

        async def _run() -> None:
            bus = EventBus()
            await bus.start()

            clock = BacktestClock()
            historical_ts = BASE_TIMESTAMP  # 1_700_000_000.0 — far in the past

            # Advance clock to match: signals at this timestamp are "current"
            clock.advance(historical_ts)

            agg = SignalAggregator(
                bus=bus,
                window_seconds=120,  # 2-minute window
                clock=clock,
            )

            # Inject a signal with the historical timestamp directly into buffer
            signal = SignalEvent(
                event_type=EventType.SIGNAL,
                timestamp=historical_ts,  # matches the clock
                signal_type=SignalType.TECHNICAL,
                symbol="BTC/USD",
                action=SignalAction.BUY,
                strength=Decimal("0.8"),
                confidence=Decimal("0.7"),
            )
            agg._signal_buffer["BTC/USD"].append(signal)

            # With clock at historical_ts, the signal should survive
            agg._clean_old_signals("BTC/USD", clock())
            assert agg.get_signal_count("BTC/USD") == 1, (
                "Signal at historical_ts should NOT be expired when clock is at historical_ts"
            )

            # Advance 8 days past the signal — it should now expire
            clock.advance(historical_ts + 8 * 86400)
            agg._clean_old_signals("BTC/USD", clock())
            assert agg.get_signal_count("BTC/USD") == 0, (
                "Signal should be expired 8 days after its timestamp"
            )

            await bus.stop()

        asyncio.get_event_loop().run_until_complete(_run())

    def test_pipeline_clock_advances_with_candles(self) -> None:
        """
        After build_backtest_pipeline(), clock.advance() updates the clock
        returned from the pipeline. Verifies the BacktestClock is the same
        instance wired into each strategy's SignalAggregator.
        """
        import asyncio
        import tempfile
        from pathlib import Path
        from scripts.run_backtest import build_backtest_pipeline, BacktestClock
        from cerebrum.core.config import Config

        config = Config.from_toml(Path("config/paper.toml"))

        async def _check() -> None:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                state_file = Path(f.name)

            try:
                bus, registry, paper_adapter, _sigs, clock = await build_backtest_pipeline(
                    config=config,
                    symbols=["BTC/USD"],
                    state_file=state_file,
                    candle_interval_seconds=60,
                )

                assert isinstance(clock, BacktestClock), (
                    "build_backtest_pipeline must return a BacktestClock as 5th element"
                )
                assert clock() == 0.0, "Clock should start at 0"

                clock.advance(BASE_TIMESTAMP)
                assert clock() == BASE_TIMESTAMP

                # The aggregator's _clock must be the same object (not a copy)
                agg = registry.get_aggregator("momentum")
                assert agg is not None
                assert agg._clock is clock, (
                    "SignalAggregator._clock must be the same BacktestClock instance "
                    "passed to StrategyRegistry — not a copy"
                )

                await registry.stop_all()
                await bus.stop()
            finally:
                state_file.unlink(missing_ok=True)

        asyncio.get_event_loop().run_until_complete(_check())
