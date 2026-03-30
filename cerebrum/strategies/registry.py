"""
StrategyRegistry: lifecycle manager for multi-strategy pipelines.

For each registered StrategyConfig, the registry creates an isolated pipeline:
  - SignalAggregator (tagged with strategy_id)
  - PortfolioTracker (with strategy's initial_balance)
  - ExitMonitor (with strategy's exit_config)
  - RiskManager (filtered to strategy's strategy_id, with per-strategy rules)

Signal generators (RSI, MACD, BB, VWAP) remain global and emit raw signals to
the shared event bus. Each strategy's SignalAggregator subscribes to those raw
signals, applies its own weights, and emits COMBINED signals tagged with its
strategy_id. Each strategy's RiskManager filters on that strategy_id, ensuring
pipeline isolation without requiring separate event buses.

Lifecycle pattern mirrors plugins/registry.py: register() → start_all() →
stop_all(), with error isolation (one strategy failing doesn't break others).

@decision DEC-STRAT-003
@title StrategyRegistry owns pipeline lifecycle with error isolation
@status accepted
@rationale Models plugins/registry.py pattern: register(), start_all(),
stop_all() with try/except isolation. One strategy failing to start doesn't
block others. Shared global guards (RegimeTradeHaltRule etc.) are constructed
once by the caller and passed into build_rules() — this avoids duplicate
subscriptions on the event bus for guards that observe global market state.
Per-strategy rules (PositionSizingRule, MaxDrawdownRule etc.) are instantiated
independently per strategy so each strategy has its own thresholds and counters.
"""

from decimal import Decimal
from typing import Any, Callable

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.config import Config
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
    RiskRule,
)
from cerebrum.signals.aggregator import SignalAggregator
from cerebrum.strategies.base import StrategyConfig
from cerebrum.strategies.global_portfolio import GlobalPortfolio

logger = structlog.get_logger()


class _StrategyPipeline:
    """
    Internal container for a single strategy's pipeline components.

    Not part of the public API — callers access components via StrategyRegistry
    accessor methods. Bundled here so start_all/stop_all can iterate cleanly.
    """

    __slots__ = (
        "config",
        "aggregator",
        "portfolio",
        "exit_monitor",
        "risk_manager",
    )

    def __init__(
        self,
        config: StrategyConfig,
        aggregator: SignalAggregator,
        portfolio: PortfolioTracker,
        exit_monitor: ExitMonitor,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.aggregator = aggregator
        self.portfolio = portfolio
        self.exit_monitor = exit_monitor
        self.risk_manager = risk_manager


class StrategyRegistry:
    """
    Creates, starts, and stops isolated pipelines for each registered strategy.

    Usage::

        registry = StrategyRegistry(bus, app_config)
        registry.register(MOMENTUM_CONFIG)
        await registry.start_all(shared_global_rules)
        # ... trading runs ...
        await registry.stop_all()

    After start_all(), access components via:
        registry.get_risk_manager("momentum")
        registry.get_portfolio("momentum")
        registry.global_portfolio  # aggregate view
    """

    def __init__(
        self,
        bus: EventBus,
        config: Config,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize the strategy registry.

        Args:
            bus: Shared event bus. All strategy pipelines share this bus —
                 isolation is achieved via strategy_id filtering, not separate
                 buses.
            config: Application config. Used as the base for per-strategy
                    parameter overrides defined in StrategyConfig.risk_overrides.
            clock: Callable returning current time as float (Unix epoch seconds).
                   Default: None (uses time.time in each component). Pass a
                   BacktestClock instance in backtest mode to inject simulated
                   historical time into SignalAggregator and PostFillCooldownRule.
                   Live mode always passes None — no behavioral change.
                   (DEC-BACKTEST-004)
        """
        self._bus = bus
        self._config = config
        # Injectable clock threaded to per-strategy components that use wall-clock time.
        # None = time.time default (live). BacktestClock in backtest mode.
        self._clock: Callable[[], float] | None = clock
        self._strategy_configs: dict[str, StrategyConfig] = {}
        self._pipelines: dict[str, _StrategyPipeline] = {}
        self._log = logger.bind(component="strategy_registry")

    def register(self, strategy_config: StrategyConfig) -> None:
        """
        Register a strategy configuration.

        Must be called before start_all(). Registering the same name twice
        raises ValueError.

        Args:
            strategy_config: Strategy configuration to register.
        """
        name = strategy_config.name
        if name in self._strategy_configs:
            raise ValueError(f"Strategy '{name}' already registered")
        self._strategy_configs[name] = strategy_config
        self._log.info("strategy_registered", name=name)

    def list_strategies(self) -> list[str]:
        """Return registered strategy names."""
        return list(self._strategy_configs.keys())

    async def start_all(
        self,
        shared_global_rules: list[RiskRule] | None = None,
    ) -> None:
        """
        Build and start all registered strategy pipelines.

        For each strategy:
        1. Creates SignalAggregator with strategy weights + strategy_id tag
        2. Creates PortfolioTracker with strategy initial_balance
        3. Creates ExitMonitor with strategy exit_config overrides
        4. Creates RiskManager with per-strategy rules + shared global rules,
           filtered to this strategy's strategy_id

        Error isolation: if one strategy fails to initialize, it is logged
        and skipped; other strategies continue.

        Args:
            shared_global_rules: Risk rules that observe global market state
                                 (e.g. RegimeTradeHaltRule, VolatilityGateRule).
                                 These are shared by reference across all strategy
                                 RiskManagers — they must already be subscribed to
                                 the bus. Pass None or [] for no shared rules.
        """
        shared_rules = shared_global_rules or []
        failed: list[str] = []

        for name, cfg in self._strategy_configs.items():
            try:
                pipeline = self._build_pipeline(cfg, shared_rules)
                self._pipelines[name] = pipeline
                self._log.info(
                    "strategy_pipeline_started",
                    name=name,
                    initial_balance=str(cfg.initial_balance),
                    symbols=cfg.symbols,
                )
            except Exception as exc:
                self._log.error(
                    "strategy_pipeline_start_failed",
                    name=name,
                    error=str(exc),
                    exc_info=True,
                )
                failed.append(name)

        if failed:
            self._log.warning(
                "strategies_failed_to_start",
                failed=failed,
                active=list(self._pipelines.keys()),
            )

    async def stop_all(self) -> None:
        """Stop all active strategy pipelines gracefully."""
        for name in list(self._pipelines.keys()):
            try:
                # ExitMonitor and RiskManager have no explicit stop — they
                # stop reacting when the event bus drains. Pipeline teardown
                # is handled by bus.stop() in the outer application shutdown.
                self._log.info("strategy_pipeline_stopped", name=name)
            except Exception as exc:
                self._log.error(
                    "strategy_pipeline_stop_failed",
                    name=name,
                    error=str(exc),
                    exc_info=True,
                )
        self._pipelines.clear()

    # --- Component accessors ---

    def get_risk_manager(self, name: str) -> RiskManager | None:
        """Get the RiskManager for a strategy, or None if not active."""
        pipeline = self._pipelines.get(name)
        return pipeline.risk_manager if pipeline else None

    def get_portfolio(self, name: str) -> PortfolioTracker | None:
        """Get the PortfolioTracker for a strategy, or None if not active."""
        pipeline = self._pipelines.get(name)
        return pipeline.portfolio if pipeline else None

    def get_aggregator(self, name: str) -> SignalAggregator | None:
        """Get the SignalAggregator for a strategy, or None if not active."""
        pipeline = self._pipelines.get(name)
        return pipeline.aggregator if pipeline else None

    def get_exit_monitor(self, name: str) -> ExitMonitor | None:
        """Get the ExitMonitor for a strategy, or None if not active."""
        pipeline = self._pipelines.get(name)
        return pipeline.exit_monitor if pipeline else None

    @property
    def global_portfolio(self) -> GlobalPortfolio:
        """
        Aggregate read-only view across all active strategy portfolios.

        Returns a new GlobalPortfolio instance reflecting current active
        pipelines. Call after start_all() for accurate results.
        """
        portfolios = {
            name: p.portfolio for name, p in self._pipelines.items()
        }
        return GlobalPortfolio(portfolios)

    def active_strategy_names(self) -> list[str]:
        """Return names of strategies with active pipelines."""
        return list(self._pipelines.keys())

    # --- Internal pipeline construction ---

    def _build_pipeline(
        self,
        cfg: StrategyConfig,
        shared_global_rules: list[RiskRule],
    ) -> _StrategyPipeline:
        """
        Construct all pipeline components for a single strategy.

        Args:
            cfg: Strategy configuration.
            shared_global_rules: Already-constructed global rules to append
                                 after per-strategy rules.

        Returns:
            Fully wired _StrategyPipeline.
        """
        # --- SignalAggregator ---
        # Pass clock if provided (backtest mode). None = time.time default (live).
        aggregator = SignalAggregator(
            bus=self._bus,
            weights=dict(cfg.aggregator_weights),
            threshold=cfg.aggregator_threshold,
            window_seconds=self._config.signals.aggregation_window_seconds,
            buy_suppression_factor=self._config.regime.buy_suppression_factor,
            buy_suppression_min_confidence=self._config.regime.buy_suppression_min_confidence,
            strategy_id=cfg.name,
            signal_source_filter=cfg.signal_source_filter,
            signal_timeframe_filter=cfg.signal_timeframe_filter,
            clock=self._clock,
        )

        # --- PortfolioTracker (strategy_id-filtered — DEC-RISK-004) ---
        portfolio = PortfolioTracker(
            bus=self._bus,
            initial_balance=cfg.initial_balance,
            strategy_id=cfg.name,
        )

        # --- ExitMonitor — use factory if provided, else default ---
        if cfg.exit_monitor_factory is not None:
            exit_monitor = cfg.exit_monitor_factory(
                bus=self._bus,
                portfolio=portfolio,
                config=cfg,
                app_config=self._config,
            )
        else:
            exit_overrides = cfg.exit_config
            exit_monitor = ExitMonitor(
                bus=self._bus,
                portfolio=portfolio,
                stop_loss_percent=Decimal(
                    exit_overrides.get("stop_loss_percent",
                                       str(self._config.risk.stop_loss_percent))
                ),
                take_profit_percent=Decimal(
                    exit_overrides.get("take_profit_percent",
                                       str(self._config.risk.take_profit_percent))
                ),
                max_position_age_minutes=int(
                    exit_overrides.get("max_position_age_minutes",
                                       self._config.risk.max_position_age_minutes)
                ),
                adaptive_tp=bool(
                    exit_overrides.get("adaptive_tp", self._config.risk.adaptive_tp)
                ),
                tp_multiplier=Decimal(
                    exit_overrides.get("tp_multiplier",
                                       str(self._config.risk.tp_multiplier))
                ),
                min_tp_percent=Decimal(
                    exit_overrides.get("min_tp_percent",
                                       str(self._config.risk.min_tp_percent))
                ),
                # DEC-EXIT-003: tag emitted orders with strategy_id for per-strategy routing
                strategy_id=cfg.name,
            )

        # --- Per-strategy risk rules ---
        overrides = cfg.risk_overrides
        per_strategy_rules: list[RiskRule] = [
            PositionSizingRule(
                Decimal(overrides.get("position_size_percent",
                                      str(self._config.risk.position_size_percent)))
            ),
            MaxPositionSizeRule(self._config.risk.max_position_size_usd),
            MaxTotalExposureRule(self._config.risk.max_total_exposure_usd),
            MaxDrawdownRule(self._config.risk.max_drawdown_percent),
            MinSignalStrengthRule(
                Decimal(overrides.get("min_signal_strength",
                                      str(self._config.risk.min_signal_strength)))
            ),
            PostFillCooldownRule(
                cooldown_seconds=int(
                    overrides.get("post_fill_cooldown_seconds",
                                  self._config.risk.post_fill_cooldown_seconds)
                ),
                bus=self._bus,
                _clock=self._clock,  # None = time.time (live); BacktestClock in backtest
            ),
        ]

        # Combine per-strategy rules + shared global rules
        all_rules = per_strategy_rules + shared_global_rules

        # --- RiskManager (strategy_id filtered) ---
        risk_manager = RiskManager(
            bus=self._bus,
            portfolio=portfolio,
            rules=all_rules,
            strategy_id=cfg.name,
        )

        return _StrategyPipeline(
            config=cfg,
            aggregator=aggregator,
            portfolio=portfolio,
            exit_monitor=exit_monitor,
            risk_manager=risk_manager,
        )
