"""
CerebrumCoin main entry point.

Orchestrates the entire trading system: event bus, adapters, signal pipeline.

@decision DEC-MAIN-001
@title Graceful shutdown with signal handlers
@status accepted
@rationale Proper cleanup on Ctrl+C prevents dangling WebSocket connections and ensures
state persistence. asyncio signal handlers trigger bus.stop() which drains all queues
before exiting. Paper trading state saves on shutdown.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

import structlog

from cerebrum.adapters.kraken import KrakenAdapter
from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.types import EventType, TradingMode
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    MaxDrawdownRule,
    MaxPositionSizeRule,
    MaxTotalExposureRule,
    MinSignalStrengthRule,
    PositionSizingRule,
    PostFillCooldownRule,
)
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.candles import CandleAggregator
from cerebrum.signals.technical import (
    BollingerBandsSignal,
    MACDSignal,
    RSISignal,
    VWAPSignal,
)
from cerebrum.intelligence.news import NewsIngestionPipeline
from cerebrum.intelligence.llm import LLMNewsAnalyzer
from cerebrum.intelligence.social import FearGreedSentiment
from cerebrum.signals.sentiment import FinBERTSentiment
from cerebrum.signals.regime import RegimeDetector
from cerebrum.core.state import StateManager
from cerebrum.learning.tracker import TradeTracker
from cerebrum.learning.scorer import SignalScorer
from cerebrum.learning.adapter import WeightAdapter
from cerebrum.monitoring.dashboard import Dashboard

def _configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog with the given minimum log level.

    Called twice: once at module load with "INFO" default (so logging works
    before config is available), and again after config loads to apply the
    user's preferred level from config.logging.level.

    cache_logger_on_first_use=False is required to allow reconfiguration
    after the initial module-level call.

    Args:
        level: Minimum log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(min_level=level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


# Configure logging at module load with INFO default so that any module-level
# log statements work before the config file is read.
_configure_logging("INFO")

logger = structlog.get_logger()


class CerebrumCoin:
    """Main application controller."""

    def __init__(self, config: Config) -> None:
        """
        Initialize CerebrumCoin.

        Args:
            config: Application configuration
        """
        self.config = config
        self.bus = EventBus()
        self.kraken_adapter: KrakenAdapter | None = None
        self.paper_adapter: PaperTradingAdapter | None = None
        self.candle_agg: CandleAggregator | None = None
        self.portfolio: PortfolioTracker | None = None
        self.risk_manager: RiskManager | None = None
        self.exit_monitor: ExitMonitor | None = None
        self.signal_agg: SignalAggregator | None = None
        self._signal_generators: list = []
        self._intelligence_components: list = []
        self.state_manager: StateManager | None = None
        self.trade_tracker: TradeTracker | None = None
        self.signal_scorer: SignalScorer | None = None
        self.weight_adapter: WeightAdapter | None = None
        self.dashboard: Dashboard | None = None
        self._shutdown_event = asyncio.Event()
        self._log = logger.bind(component="main")

    async def start(self) -> None:
        """Start the trading system."""
        self._log.info("cerebrumcoin_starting", mode=self.config.trading.mode.value)

        # Start event bus
        await self.bus.start()

        # Initialize Kraken adapter for market data
        self.kraken_adapter = KrakenAdapter(
            self.bus,
            {
                "api_key": self.config.exchange.api_key,
                "api_secret": self.config.exchange.api_secret,
                "rate_limit_per_minute": self.config.exchange.rate_limit_per_minute,
            }
        )
        await self.kraken_adapter.connect()

        # Initialize execution adapter based on trading mode
        if self.config.trading.mode == TradingMode.LIVE:
            self._log.warning(
                "LIVE_MODE_ACTIVE",
                message="*** REAL MONEY TRADING ENABLED — Orders will execute on live exchange ***",
            )
            # In live mode, Kraken handles both data AND execution
            self.bus.subscribe(
                EventType.ORDER,
                self.kraken_adapter.execute_order,
                "kraken_live_executor",
            )
        else:
            # Paper mode (default) — use paper adapter for execution
            self.paper_adapter = PaperTradingAdapter(
                self.bus,
                {},
                initial_balance=self.config.paper.initial_balance_usd,
                commission_percent=self.config.paper.commission_percent,
                slippage_percent=self.config.paper.slippage_percent,
                state_file=self.config.paper.state_file,
            )
            await self.paper_adapter.connect()

        # Initialize candle aggregator
        self.candle_agg = CandleAggregator(
            self.bus,
            interval_seconds=self.config.signals.candle_interval_seconds,
        )

        # Initialize portfolio tracker
        self.portfolio = PortfolioTracker(
            self.bus,
            initial_balance=self.config.paper.initial_balance_usd,
        )

        # Initialize technical signal generators
        self._signal_generators = [
            RSISignal(
                self.bus,
                self.candle_agg,
                period=self.config.signals.rsi_period,
                oversold=self.config.signals.rsi_oversold,
                overbought=self.config.signals.rsi_overbought,
            ),
            MACDSignal(
                self.bus,
                self.candle_agg,
                fast=self.config.signals.macd_fast,
                slow=self.config.signals.macd_slow,
                signal=self.config.signals.macd_signal,
            ),
            BollingerBandsSignal(
                self.bus,
                self.candle_agg,
                period=self.config.signals.bb_period,
                std_dev=self.config.signals.bb_std_dev,
            ),
            VWAPSignal(
                self.bus,
                self.candle_agg,
                period=self.config.signals.vwap_period,
            ),
        ]

        # Initialize signal aggregator
        self.signal_agg = SignalAggregator(
            self.bus,
            threshold=self.config.signals.aggregation_threshold,
            window_seconds=self.config.signals.aggregation_window_seconds,
            buy_suppression_factor=self.config.regime.buy_suppression_factor,
            buy_suppression_min_confidence=self.config.regime.buy_suppression_min_confidence,
        )

        # Initialize intelligence layer components
        news_pipeline = NewsIngestionPipeline(
            self.bus,
            cryptopanic_api_key=self.config.intelligence.cryptopanic_api_key,
            cryptopanic_poll_interval=self.config.intelligence.cryptopanic_poll_interval_seconds,
            newsapi_api_key=self.config.intelligence.newsapi_api_key,
            newsapi_poll_interval=self.config.intelligence.newsapi_poll_interval_seconds,
        )
        await news_pipeline.start()
        self._intelligence_components.append(news_pipeline)

        llm_analyzer = LLMNewsAnalyzer(
            self.bus,
            anthropic_api_key=self.config.llm.anthropic_api_key,
            model=self.config.llm.model,
            max_calls_per_hour=self.config.llm.max_calls_per_hour,
            batch_size=self.config.llm.news_batch_size,
            batch_window_seconds=self.config.llm.news_batch_window_seconds,
            timeout_seconds=self.config.llm.timeout_seconds,
        )
        await llm_analyzer.start()
        self._intelligence_components.append(llm_analyzer)

        fear_greed = FearGreedSentiment(
            self.bus,
            poll_interval=self.config.intelligence.fear_greed_poll_interval_seconds,
        )
        await fear_greed.start()
        self._intelligence_components.append(fear_greed)

        if self.config.intelligence.enable_finbert:
            finbert = FinBERTSentiment(
                self.bus,
                enabled=True,
            )
            self._intelligence_components.append(finbert)

        regime_detector = RegimeDetector(
            self.bus,
            window_size=self.config.regime.window_size,
            update_interval=self.config.regime.update_interval,
            use_hmm=self.config.intelligence.enable_hmm_regime,
            cumulative_trend_threshold=self.config.regime.cumulative_trend_threshold,
            ma_slope_threshold=self.config.regime.ma_slope_threshold,
            mean_return_threshold=self.config.regime.mean_return_threshold,
            volatility_threshold=self.config.regime.volatility_threshold,
            ma_period=self.config.regime.ma_period,
            long_window_size=self.config.regime.long_window_size,
            long_cumulative_threshold=self.config.regime.long_cumulative_threshold,
        )
        self._intelligence_components.append(regime_detector)

        # Initialize risk manager with rules
        risk_rules = [
            PositionSizingRule(self.config.risk.position_size_percent),
            MaxPositionSizeRule(self.config.risk.max_position_size_usd),
            MaxTotalExposureRule(self.config.risk.max_total_exposure_usd),
            MaxDrawdownRule(self.config.risk.max_drawdown_percent),
            MinSignalStrengthRule(self.config.risk.min_signal_strength),
            PostFillCooldownRule(
                cooldown_seconds=self.config.risk.post_fill_cooldown_seconds,
                bus=self.bus,
            ),
        ]
        self.risk_manager = RiskManager(
            self.bus,
            self.portfolio,
            rules=risk_rules,
        )

        # Initialize exit monitor (stop-loss, take-profit, time-based exits)
        self.exit_monitor = ExitMonitor(
            self.bus,
            self.portfolio,
            stop_loss_percent=self.config.risk.stop_loss_percent,
            take_profit_percent=self.config.risk.take_profit_percent,
            max_position_age_minutes=self.config.risk.max_position_age_minutes,
        )

        # Initialize learning system
        db_path = Path("data/cerebrum.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_manager = StateManager(db_path)
        await self.state_manager.initialize()

        self.trade_tracker = TradeTracker(self.bus, self.state_manager, "UNKNOWN")
        await self.trade_tracker.start()

        self.signal_scorer = SignalScorer(self.bus, self.state_manager)
        await self.signal_scorer.start()

        def weight_callback(signal_type, regime, weight):
            self.signal_agg.set_regime_weight(signal_type, regime, weight)

        self.weight_adapter = WeightAdapter(self.bus, self.state_manager, weight_callback)
        await self.weight_adapter.start()

        self._log.info("learning_system_initialized", db_path=str(db_path))

        # Initialize dashboard if enabled
        if self.config.monitoring.dashboard_enabled:
            self.dashboard = Dashboard(
                self.bus,
                self.state_manager,
                update_interval_seconds=self.config.monitoring.update_interval_seconds,
                initial_balance=self.config.paper.initial_balance_usd,
            )
            await self.dashboard.start()
            self._log.info("dashboard_started")

        # Subscribe to market data
        await self.kraken_adapter.subscribe_market_data(self.config.trading.symbols)

        self._log.info(
            "cerebrumcoin_started",
            symbols=self.config.trading.symbols,
            initial_balance=str(self.config.paper.initial_balance_usd),
            signal_generators=len(self._signal_generators),
            risk_rules=len(risk_rules),
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the trading system gracefully."""
        self._log.info("cerebrumcoin_stopping")

        # Stop dashboard
        if self.dashboard:
            await self.dashboard.stop()


        # Close learning system
        if self.state_manager:
            await self.state_manager.close()

        # Stop intelligence components
        for component in self._intelligence_components:
            if hasattr(component, 'stop'):
                await component.stop()

        # Disconnect adapters
        if self.kraken_adapter:
            await self.kraken_adapter.disconnect()

        if self.paper_adapter:
            await self.paper_adapter.disconnect()

        # Stop event bus (drains queues)
        await self.bus.stop()

        self._log.info("cerebrumcoin_stopped")

    def trigger_shutdown(self) -> None:
        """Trigger graceful shutdown."""
        self._shutdown_event.set()


async def async_main(config_path: Path) -> None:
    """
    Async main function.

    Args:
        config_path: Path to TOML configuration file
    """
    # Load configuration
    config = Config.from_toml(config_path)

    # Reconfigure logging with user's preferred level from config.
    # This supersedes the module-level INFO default.
    _configure_logging(config.logging.level)

    # Create application
    app = CerebrumCoin(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler(sig: int) -> None:
        logger.info("shutdown_signal_received", signal=sig)
        app.trigger_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    try:
        await app.start()
    finally:
        await app.stop()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CerebrumCoin - Autonomous AI Trading Agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.toml"),
        help="Path to configuration file (default: config/default.toml)"
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode (default: paper)"
    )

    args = parser.parse_args()

    # Override config path based on mode
    if args.mode == "paper" and args.config == Path("config/default.toml"):
        args.config = Path("config/paper.toml")

    # Run async main
    try:
        asyncio.run(async_main(args.config))
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("fatal_error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
