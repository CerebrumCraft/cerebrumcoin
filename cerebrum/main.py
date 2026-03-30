"""
CerebrumCoin main entry point.

Orchestrates the entire trading system: event bus, adapters, signal pipeline.

@decision DEC-MAIN-001
@title Graceful shutdown with signal handlers
@status accepted
@rationale Proper cleanup on Ctrl+C prevents dangling WebSocket connections and ensures
state persistence. asyncio signal handlers trigger bus.stop() which drains all queues
before exiting. Paper trading state saves on shutdown.

@decision DEC-MAIN-002
@title Multi-strategy mode as default with single-strategy fallback
@status accepted
@rationale The multi-strategy pipeline (StrategyRegistry + DarwinianAllocator +
Conductor + WebDashboard) is the Phase 11 target architecture. Single-strategy mode
is preserved for backward compatibility — it produces behaviour identical to pre-Phase-11
sessions (same paper.toml tuning, same risk rules). The mode is controlled by the
CEREBRUM_MULTI_STRATEGY env var (default "true"). Shared global guards (regime halt,
volatility gate, macro gate, sideways suppression, global rate limit) are constructed
once and passed to StrategyRegistry.start_all() so they are shared by reference across
all per-strategy RiskManagers — avoiding duplicate event bus subscriptions for guards
that observe global market state (DEC-STRAT-003).

@decision DEC-SHUTDOWN-001
@title Graceful position liquidation on shutdown
@status accepted
@rationale Open positions persisted across sessions create phantom P&L. Session 11
showed $10,750.72 equity with unrealized gains that were never realized. Closing all
positions at market during graceful shutdown ensures accurate realized P&L and clean
state for the next session. _close_all_positions() publishes OrderEvents to the still-
running event bus (and hence the paper adapter's execute_order handler) so fills flow
through the normal PortfolioTracker path, realizing P&L correctly. Called in stop()
after the web dashboard is stopped but before the conductor/strategies/bus are torn
down so the event loop is still live. Each position close is isolated with try/except
so one failure cannot block the others.
"""

import argparse
import asyncio
import os
import signal
import sys
import uuid
from decimal import Decimal
from pathlib import Path
from time import time as _time
from typing import Any

import structlog

from cerebrum.adapters.kraken import KrakenAdapter
from cerebrum.adapters.paper import PaperTradingAdapter
from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
from cerebrum.core.events import OrderEvent
from cerebrum.core.types import EventType, OrderStatus, OrderType, Side, TradingMode
from cerebrum.risk.exit_monitor import ExitMonitor
from cerebrum.risk.global_trade_rate import GlobalTradeRateLimitRule
from cerebrum.risk.manager import RiskManager
from cerebrum.risk.portfolio import PortfolioTracker
from cerebrum.risk.rules import (
    CommissionGateRule,
    MacroVolatilityGateRule,
    MaxDrawdownRule,
    MaxPositionSizeRule,
    MaxTotalExposureRule,
    MinSignalStrengthRule,
    PositionSizingRule,
    PostFillCooldownRule,
    RegimeTradeHaltRule,
    SidewaysSuppressionRule,
    VolatilityGateRule,
)
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.signals.candles import CandleAggregator
from cerebrum.signals.support_resistance import SupportResistanceSignal
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
    """
    Main application controller.

    Supports two wiring modes:
    - Multi-strategy (default): StrategyRegistry + DarwinianAllocator + Conductor
      + WebDashboard. Three isolated strategy pipelines share global guards.
    - Single-strategy (legacy): one SignalAggregator + RiskManager + PortfolioTracker,
      behaviorally identical to pre-Phase-11 sessions. Activated by setting
      CEREBRUM_MULTI_STRATEGY=false.

    See DEC-MAIN-002 for the rationale.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize CerebrumCoin.

        Args:
            config: Application configuration
        """
        self.config = config
        self.bus = EventBus()

        # Shared infrastructure (both modes)
        self.kraken_adapter: KrakenAdapter | None = None
        self.paper_adapter: PaperTradingAdapter | None = None
        self.candle_agg: CandleAggregator | None = None
        self.candle_agg_1h: CandleAggregator | None = None
        self._signal_generators: list[Any] = []
        self._intelligence_components: list[Any] = []
        self.state_manager: StateManager | None = None
        self.trade_tracker: TradeTracker | None = None
        self.signal_scorer: SignalScorer | None = None
        self.weight_adapter: WeightAdapter | None = None
        self.dashboard: Dashboard | None = None

        # Single-strategy mode components (legacy path)
        self.portfolio: PortfolioTracker | None = None
        self.risk_manager: RiskManager | None = None
        self.exit_monitor: ExitMonitor | None = None
        self.signal_agg: SignalAggregator | None = None

        # Multi-strategy mode components
        self.strategy_registry: Any | None = None   # StrategyRegistry
        self.allocator: Any | None = None            # DarwinianAllocator
        self.conductor: Any | None = None            # Conductor
        self.web_dashboard: Any | None = None        # WebDashboard | None

        self._shutdown_event = asyncio.Event()
        self._log = logger.bind(component="main")

    # ------------------------------------------------------------------
    # Shared setup helpers (both modes)
    # ------------------------------------------------------------------

    def _build_signal_generators(self) -> list[Any]:
        """Build the shared technical signal generators."""
        config = self.config
        return [
            RSISignal(
                self.bus,
                self.candle_agg,
                period=config.signals.rsi_period,
                oversold=config.signals.rsi_oversold,
                overbought=config.signals.rsi_overbought,
            ),
            MACDSignal(
                self.bus,
                self.candle_agg,
                fast=config.signals.macd_fast,
                slow=config.signals.macd_slow,
                signal=config.signals.macd_signal,
            ),
            BollingerBandsSignal(
                self.bus,
                self.candle_agg,
                period=config.signals.bb_period,
                std_dev=config.signals.bb_std_dev,
            ),
            VWAPSignal(
                self.bus,
                self.candle_agg,
                period=config.signals.vwap_period,
            ),
            SupportResistanceSignal(
                self.bus,
                self.candle_agg,
                pivot_lookback=config.signals.sr_pivot_lookback,
                min_touches=config.signals.sr_min_touches,
                proximity_pct=config.signals.sr_proximity_pct,
            ),
        ]

    def _build_signal_generators_1h(self) -> list[Any]:
        """
        Build 1-hour technical signal generators for the swing trading strategy.

        These are parallel to the 1m generators but consume the 1h CandleAggregator
        and stamp metadata["timeframe"] = "1h" on every emitted signal. The swing
        trading SignalAggregator filters for this tag via signal_timeframe_filter="1h",
        so 1h signals feed only swing_trading while 1m signals feed the other four
        strategies (DEC-SWING-001).
        """
        config = self.config
        return [
            RSISignal(
                self.bus,
                self.candle_agg_1h,
                period=config.signals.rsi_period,
                oversold=config.signals.rsi_oversold,
                overbought=config.signals.rsi_overbought,
                timeframe="1h",
            ),
            MACDSignal(
                self.bus,
                self.candle_agg_1h,
                fast=config.signals.macd_fast,
                slow=config.signals.macd_slow,
                signal=config.signals.macd_signal,
                timeframe="1h",
            ),
            BollingerBandsSignal(
                self.bus,
                self.candle_agg_1h,
                period=config.signals.bb_period,
                std_dev=config.signals.bb_std_dev,
                timeframe="1h",
            ),
            VWAPSignal(
                self.bus,
                self.candle_agg_1h,
                period=config.signals.vwap_period,
                timeframe="1h",
            ),
        ]

    async def _start_intelligence_components(self) -> None:
        """Start news, LLM, fear/greed, FinBERT, and regime components."""
        config = self.config

        news_pipeline = NewsIngestionPipeline(
            self.bus,
            cryptopanic_api_key=config.intelligence.cryptopanic_api_key,
            cryptopanic_poll_interval=config.intelligence.cryptopanic_poll_interval_seconds,
            newsapi_api_key=config.intelligence.newsapi_api_key,
            newsapi_poll_interval=config.intelligence.newsapi_poll_interval_seconds,
        )
        await news_pipeline.start()
        self._intelligence_components.append(news_pipeline)

        llm_analyzer = LLMNewsAnalyzer(
            self.bus,
            anthropic_api_key=config.llm.anthropic_api_key,
            model=config.llm.model,
            max_calls_per_hour=config.llm.max_calls_per_hour,
            batch_size=config.llm.news_batch_size,
            batch_window_seconds=config.llm.news_batch_window_seconds,
            timeout_seconds=config.llm.timeout_seconds,
        )
        await llm_analyzer.start()
        self._intelligence_components.append(llm_analyzer)

        fear_greed = FearGreedSentiment(
            self.bus,
            poll_interval=config.intelligence.fear_greed_poll_interval_seconds,
        )
        await fear_greed.start()
        self._intelligence_components.append(fear_greed)

        if config.intelligence.enable_finbert:
            finbert = FinBERTSentiment(self.bus, enabled=True)
            self._intelligence_components.append(finbert)

        regime_detector = RegimeDetector(
            self.bus,
            window_size=config.regime.window_size,
            update_interval=config.regime.update_interval,
            use_hmm=config.intelligence.enable_hmm_regime,
            cumulative_trend_threshold=config.regime.cumulative_trend_threshold,
            ma_slope_threshold=config.regime.ma_slope_threshold,
            mean_return_threshold=config.regime.mean_return_threshold,
            volatility_threshold=config.regime.volatility_threshold,
            ma_period=config.regime.ma_period,
            long_window_size=config.regime.long_window_size,
            long_cumulative_threshold=config.regime.long_cumulative_threshold,
            min_hold_count=config.regime.min_hold_count,
        )
        self._intelligence_components.append(regime_detector)

    async def _start_learning_system(self, db_path: Path) -> None:
        """Start StateManager, TradeTracker, SignalScorer, WeightAdapter."""
        self.state_manager = StateManager(db_path)
        await self.state_manager.initialize()

        self.trade_tracker = TradeTracker(self.bus, self.state_manager, "UNKNOWN")
        await self.trade_tracker.start()

        self.signal_scorer = SignalScorer(self.bus, self.state_manager)
        await self.signal_scorer.start()

        # In single-strategy mode signal_agg is available for weight updates.
        # In multi-strategy mode there is no single aggregator — weight updates
        # are a no-op until per-strategy weight routing is implemented.
        def weight_callback(signal_type: Any, regime: Any, weight: Any) -> None:
            if self.signal_agg is not None:
                self.signal_agg.set_regime_weight(signal_type, regime, weight)

        self.weight_adapter = WeightAdapter(self.bus, self.state_manager, weight_callback)
        await self.weight_adapter.start()

        self._log.info("learning_system_initialized", db_path=str(db_path))

    # ------------------------------------------------------------------
    # Single-strategy setup (legacy / backward-compat path)
    # ------------------------------------------------------------------

    def _setup_single_strategy(self) -> None:
        """
        Wire the single-strategy pipeline — behaviorally identical to pre-Phase-11.

        Creates one SignalAggregator, PortfolioTracker, ExitMonitor, and
        RiskManager using the paper.toml risk parameters directly.
        """
        config = self.config

        self.portfolio = PortfolioTracker(
            self.bus,
            initial_balance=config.paper.initial_balance_usd,
            # No strategy_id — accepts all fills (DEC-RISK-004 backward compat)
        )

        self.signal_agg = SignalAggregator(
            self.bus,
            threshold=config.signals.aggregation_threshold,
            window_seconds=config.signals.aggregation_window_seconds,
            buy_suppression_factor=config.regime.buy_suppression_factor,
            buy_suppression_min_confidence=config.regime.buy_suppression_min_confidence,
        )

        risk_rules = [
            PositionSizingRule(config.risk.position_size_percent),
            MaxPositionSizeRule(config.risk.max_position_size_usd),
            MaxTotalExposureRule(config.risk.max_total_exposure_usd),
            MaxDrawdownRule(config.risk.max_drawdown_percent),
            MinSignalStrengthRule(config.risk.min_signal_strength),
            RegimeTradeHaltRule(
                min_confidence=Decimal(str(config.regime.bear_halt_min_confidence)),
                bus=self.bus,
            ),
            PostFillCooldownRule(
                cooldown_seconds=config.risk.post_fill_cooldown_seconds,
                bus=self.bus,
            ),
            VolatilityGateRule(
                min_range_pct=config.risk.volatility_gate_min_range_pct,
                window_size=config.risk.volatility_gate_window_size,
                bus=self.bus,
            ),
            SidewaysSuppressionRule(
                min_range_pct=config.risk.sideways_suppression_min_range_pct,
                window_size=config.risk.sideways_suppression_window_size,
                bus=self.bus,
            ),
            MacroVolatilityGateRule(
                min_range_pct=config.risk.macro_volatility_min_range_pct,
                window_size=config.risk.macro_volatility_window_size,
                bus=self.bus,
            ),
            CommissionGateRule(
                commission_percent=config.paper.commission_percent,
                min_profit_to_commission_ratio=config.risk.min_profit_to_commission_ratio,
                window_size=config.risk.volatility_gate_window_size,
                bus=self.bus,
            ),
        ]
        self.risk_manager = RiskManager(
            self.bus,
            self.portfolio,
            rules=risk_rules,
        )

        self.exit_monitor = ExitMonitor(
            self.bus,
            self.portfolio,
            stop_loss_percent=config.risk.stop_loss_percent,
            take_profit_percent=config.risk.take_profit_percent,
            max_position_age_minutes=config.risk.max_position_age_minutes,
            adaptive_tp=config.risk.adaptive_tp,
            tp_multiplier=config.risk.tp_multiplier,
            min_tp_percent=config.risk.min_tp_percent,
        )

        self._log.info(
            "single_strategy_pipeline_wired",
            risk_rules=len(risk_rules),
        )

    # ------------------------------------------------------------------
    # Multi-strategy setup (Phase 11 default path)
    # ------------------------------------------------------------------

    async def _setup_multi_strategy(self) -> None:
        """
        Wire the multi-strategy pipeline.

        Creates shared global guards (one instance per guard type, shared by
        reference across all strategy RiskManagers via StrategyRegistry.start_all).
        Registers six strategies: momentum, mean_reversion, breakout, range_trading, swing_trading, news_driven.
        Creates DarwinianAllocator, Conductor, and WebDashboard.

        See DEC-MAIN-002, DEC-STRAT-003.
        """
        from cerebrum.strategies.registry import StrategyRegistry
        from cerebrum.strategies.momentum import MOMENTUM_CONFIG
        from cerebrum.strategies.mean_reversion import MEAN_REVERSION_CONFIG
        from cerebrum.strategies.breakout import BREAKOUT_CONFIG
        from cerebrum.strategies.range_trading import RANGE_TRADING_CONFIG
        from cerebrum.strategies.swing_trading import SWING_TRADING_CONFIG
        from cerebrum.strategies.news_driven import NEWS_DRIVEN_CONFIG
        from cerebrum.conductor.allocator import DarwinianAllocator
        from cerebrum.conductor.conductor import Conductor

        config = self.config

        # --- Shared global guards (constructed once, shared across all strategies) ---
        # Each guard subscribes to bus events exactly once — no duplicate subscriptions.
        global_guards = [
            RegimeTradeHaltRule(
                min_confidence=Decimal(str(config.regime.bear_halt_min_confidence)),
                bus=self.bus,
            ),
            VolatilityGateRule(
                min_range_pct=config.risk.volatility_gate_min_range_pct,
                window_size=config.risk.volatility_gate_window_size,
                bus=self.bus,
            ),
            MacroVolatilityGateRule(
                min_range_pct=config.risk.macro_volatility_min_range_pct,
                window_size=config.risk.macro_volatility_window_size,
                bus=self.bus,
            ),
            CommissionGateRule(
                commission_percent=config.paper.commission_percent,
                min_profit_to_commission_ratio=config.risk.min_profit_to_commission_ratio,
                window_size=config.risk.volatility_gate_window_size,
                bus=self.bus,
            ),
            SidewaysSuppressionRule(
                min_range_pct=config.risk.sideways_suppression_min_range_pct,
                window_size=config.risk.sideways_suppression_window_size,
                bus=self.bus,
                # range_trading targets SIDEWAYS — exempt so it can enter when
                # other strategies are suppressed (DEC-RANGE-006).
                exempt_strategies={"range_trading"},
            ),
            # ~10 trades/hour per strategy * 5 strategies = 50 global cap
            # swing_trading targets ~2-4 trades/day so the 1h strategy contributes
            # very few; the cap is kept at 40 to maintain pre-swing budget.
            GlobalTradeRateLimitRule(
                max_trades_per_hour=40,
                bus=self.bus,
            ),
        ]

        # --- StrategyRegistry ---
        self.strategy_registry = StrategyRegistry(bus=self.bus, config=config)
        self.strategy_registry.register(MOMENTUM_CONFIG)
        self.strategy_registry.register(MEAN_REVERSION_CONFIG)
        self.strategy_registry.register(BREAKOUT_CONFIG)
        self.strategy_registry.register(RANGE_TRADING_CONFIG)
        # @decision DEC-TUNE-005
        # @title Disable swing_trading — Session 18 sole loser
        # @status accepted
        # @rationale Session 18: -$51 PnL, zero realized trades, only 1 position held (short DOGE).
        #   Only losing strategy of 6. Disable until tuning is revisited. Re-enable by uncommenting.
        # self.strategy_registry.register(SWING_TRADING_CONFIG)
        self.strategy_registry.register(NEWS_DRIVEN_CONFIG)

        # Build and start all strategy pipelines, injecting shared global guards
        await self.strategy_registry.start_all(shared_global_rules=global_guards)

        # --- Per-strategy state restore (DEC-PERSIST-001) ---
        # After pipelines are started, restore each strategy's PortfolioTracker
        # from the saved snapshot (if any). Then register the live portfolios
        # so _save_state() embeds v2 snapshots on every subsequent trade.
        # New strategies (no snapshot) start fresh at their configured balance.
        # Removed strategies (snapshot exists but not in registry) are ignored.
        if self.paper_adapter is not None:
            portfolios: dict[str, PortfolioTracker] = {}
            for name in self.strategy_registry.active_strategy_names():
                portfolio = self.strategy_registry.get_portfolio(name)
                if portfolio is None:
                    continue
                snapshot = self.paper_adapter.get_strategy_snapshot(name)
                if snapshot is not None:
                    portfolio.restore_snapshot(snapshot)
                    self._log.info(
                        "strategy_portfolio_restored",
                        strategy=name,
                        cash_balance=snapshot.get("cash_balance"),
                    )
                portfolios[name] = portfolio
            self.paper_adapter.set_strategy_portfolios(portfolios)

        active = self.strategy_registry.active_strategy_names()
        if not active:
            raise RuntimeError(
                "No strategy pipelines started successfully — cannot continue."
            )

        # --- DarwinianAllocator ---
        self.allocator = DarwinianAllocator(
            strategy_names=active,
            total_capital=Decimal(str(config.paper.initial_balance_usd)),
        )

        # --- Conductor ---
        anthropic_key = getattr(config, "llm", None)
        anthropic_key = anthropic_key.anthropic_api_key if anthropic_key else None

        self.conductor = Conductor(
            bus=self.bus,
            registry=self.strategy_registry,
            allocator=self.allocator,
            anthropic_api_key=anthropic_key,
        )
        await self.conductor.start()

        # --- WebDashboard (optional — requires fastapi/uvicorn) ---
        try:
            from cerebrum.dashboard.web import WebDashboard
            self.web_dashboard = WebDashboard(
                bus=self.bus,
                registry=self.strategy_registry,
                conductor=self.conductor,
                global_portfolio=self.strategy_registry.global_portfolio,
                paper_adapter=self.paper_adapter,
            )
            await self.web_dashboard.start()
            self._log.info("web_dashboard_started", url="http://127.0.0.1:8080")
        except ImportError:
            self._log.warning(
                "web_dashboard_unavailable",
                reason="fastapi/uvicorn not installed — run: pip install cerebrumcoin[dashboard]",
            )
            self.web_dashboard = None

        self._log.info(
            "multi_strategy_pipeline_wired",
            active_strategies=active,
            global_guards=len(global_guards),
        )

    # ------------------------------------------------------------------
    # Main lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the trading system."""
        config = self.config
        self._log.info("cerebrumcoin_starting", mode=config.trading.mode.value)

        # Start event bus
        await self.bus.start()

        # Initialize Kraken adapter for market data
        self.kraken_adapter = KrakenAdapter(
            self.bus,
            {
                "api_key": config.exchange.api_key,
                "api_secret": config.exchange.api_secret,
                "rate_limit_per_minute": config.exchange.rate_limit_per_minute,
            },
        )
        await self.kraken_adapter.connect()

        # Initialize execution adapter based on trading mode
        if config.trading.mode == TradingMode.LIVE:
            self._log.warning(
                "LIVE_MODE_ACTIVE",
                message="*** REAL MONEY TRADING ENABLED — Orders will execute on live exchange ***",
            )
            self.bus.subscribe(
                EventType.ORDER,
                self.kraken_adapter.execute_order,
                "kraken_live_executor",
            )
        else:
            self.paper_adapter = PaperTradingAdapter(
                self.bus,
                {},
                initial_balance=config.paper.initial_balance_usd,
                commission_percent=config.paper.commission_percent,
                slippage_percent=config.paper.slippage_percent,
                state_file=config.paper.state_file,
            )
            await self.paper_adapter.connect()

        # 1m candle aggregator — shared across all non-swing strategies
        self.candle_agg = CandleAggregator(
            self.bus,
            interval_seconds=config.signals.candle_interval_seconds,
        )

        # 1h candle aggregator — dedicated to swing trading strategy (DEC-SWING-001).
        # Independent of candle_agg: separate state, separate interval boundary.
        self.candle_agg_1h = CandleAggregator(
            self.bus,
            interval_seconds=3600,
        )

        # 1m technical signal generators — shared across momentum/mean_reversion/
        # breakout/range_trading strategies. Each generator stamps metadata["timeframe"]
        # = "1m" so swing_trading's aggregator (filter="1h") ignores them.
        self._signal_generators = self._build_signal_generators()

        # 1h technical signal generators — exclusively consumed by swing_trading.
        # Stamp metadata["timeframe"] = "1h" on every emitted signal so other
        # strategy aggregators (no timeframe filter, or filter != "1h") ignore them.
        signal_generators_1h = self._build_signal_generators_1h()
        self._signal_generators.extend(signal_generators_1h)

        # Intelligence layer (shared regime + news + sentiment)
        await self._start_intelligence_components()

        # Strategy pipeline — multi or single depending on env var
        multi_strategy = (
            os.environ.get("CEREBRUM_MULTI_STRATEGY", "true").lower() == "true"
        )

        if multi_strategy:
            self._log.info("startup_mode", mode="multi_strategy")
            await self._setup_multi_strategy()
        else:
            self._log.info("startup_mode", mode="single_strategy_legacy")
            self._setup_single_strategy()

        # Learning system (shared — tracks signals and outcomes)
        db_path = Path("data/cerebrum.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        await self._start_learning_system(db_path)

        # Legacy terminal dashboard (single-strategy mode only — it needs a
        # single RiskManager reference; multi-strategy uses WebDashboard)
        if not multi_strategy and config.monitoring.dashboard_enabled:
            self.dashboard = Dashboard(
                self.bus,
                self.state_manager,
                update_interval_seconds=config.monitoring.update_interval_seconds,
                initial_balance=config.paper.initial_balance_usd,
                risk_manager=self.risk_manager,
            )
            await self.dashboard.start()
            self._log.info("terminal_dashboard_started")

        # Subscribe to market data (after all components are set up)
        await self.kraken_adapter.subscribe_market_data(config.trading.symbols)

        self._log.info(
            "cerebrumcoin_started",
            symbols=config.trading.symbols,
            initial_balance=str(config.paper.initial_balance_usd),
            signal_generators=len(self._signal_generators),
            multi_strategy=multi_strategy,
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def _close_all_positions(self) -> None:
        """
        Liquidate all open positions at market price before shutdown.

        Collects positions from both single-strategy (self.portfolio) and
        multi-strategy (self.strategy_registry) modes. For each open position
        publishes an OrderEvent to the still-running event bus so the paper
        adapter's execute_order handler processes it normally, realizing P&L
        through PortfolioTracker._on_fill().

        Must be called while the event bus and paper adapter are still running
        (i.e. before conductor/strategies/bus teardown). Each position closure
        is isolated with try/except so one failure cannot block the others.

        See DEC-SHUTDOWN-001.
        """
        # Gather (strategy_id, symbol, amount) tuples for every open position.
        pending: list[tuple[str | None, str, Decimal]] = []

        # Single-strategy mode
        if self.portfolio is not None:
            for symbol, pos in self.portfolio.get_all_positions().items():
                if pos.amount != Decimal("0"):
                    pending.append((None, symbol, pos.amount))

        # Multi-strategy mode
        if self.strategy_registry is not None:
            for name in self.strategy_registry.active_strategy_names():
                portfolio = self.strategy_registry.get_portfolio(name)
                if portfolio is None:
                    continue
                for symbol, pos in portfolio.get_all_positions().items():
                    if pos.amount != Decimal("0"):
                        pending.append((name, symbol, pos.amount))

        if not pending:
            self._log.info("shutdown_liquidation_no_positions")
            return

        self._log.info(
            "shutdown_liquidation_starting",
            position_count=len(pending),
        )

        for strategy_id, symbol, amount in pending:
            try:
                # Long position → SELL to close; short position → BUY to cover.
                side = Side.SELL if amount > Decimal("0") else Side.BUY
                close_amount = abs(amount)

                order = OrderEvent(
                    event_type=EventType.ORDER,
                    timestamp=_time(),
                    order_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    amount=close_amount,
                    price=None,
                    status=OrderStatus.PENDING,
                    metadata={"exit_reason": "shutdown_liquidation", "source": "shutdown"},
                    strategy_id=strategy_id,
                )
                await self.bus.publish(order)
                self._log.info(
                    "shutdown_liquidation_order_sent",
                    strategy_id=strategy_id or "single",
                    symbol=symbol,
                    side=side.value,
                    amount=str(close_amount),
                )
            except Exception as exc:
                self._log.warning(
                    "shutdown_liquidation_order_failed",
                    strategy_id=strategy_id or "single",
                    symbol=symbol,
                    error=str(exc),
                )

        # Yield control so the event bus can dispatch the published orders
        # to the paper adapter's execute_order handler before we proceed
        # with teardown. A single asyncio.sleep(0) is enough because the
        # bus processes events in the same event loop iteration.
        await asyncio.sleep(0)
        self._log.info("shutdown_liquidation_complete")

    async def stop(self) -> None:
        """Stop the trading system gracefully (reverse startup order)."""
        self._log.info("cerebrumcoin_stopping")

        # Stop WebDashboard first (closes HTTP server / WS connections)
        if self.web_dashboard is not None:
            await self.web_dashboard.stop()

        # Liquidate all open positions before tearing down the event bus.
        # The bus and paper adapter must still be running for fills to process.
        # See DEC-SHUTDOWN-001.
        await self._close_all_positions()

        # Stop Conductor (cancels poll task)
        if self.conductor is not None:
            await self.conductor.stop()

        # Stop strategy registry (logs each pipeline stopped)
        if self.strategy_registry is not None:
            await self.strategy_registry.stop_all()

        # Stop legacy terminal dashboard
        if self.dashboard:
            await self.dashboard.stop()

        # Close learning system
        if self.state_manager:
            await self.state_manager.close()

        # Stop intelligence components
        for component in self._intelligence_components:
            if hasattr(component, "stop"):
                await component.stop()

        # Disconnect adapters (paper adapter calls _save_state() here,
        # capturing the post-liquidation state with no open positions).
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
    config = Config.from_toml(config_path)

    # Reconfigure logging with user's preferred level from config.
    # This supersedes the module-level INFO default.
    _configure_logging(config.logging.level)

    app = CerebrumCoin(config)

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
        help="Path to configuration file (default: config/default.toml)",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode (default: paper)",
    )

    args = parser.parse_args()

    if args.mode == "paper" and args.config == Path("config/default.toml"):
        args.config = Path("config/paper.toml")

    try:
        asyncio.run(async_main(args.config))
    except KeyboardInterrupt:
        logger.info("interrupted_by_user")
    except Exception as e:
        logger.error("fatal_error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
