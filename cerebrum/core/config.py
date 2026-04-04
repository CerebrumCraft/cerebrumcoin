"""
Configuration management for CerebrumCoin.

Loads settings from TOML files and environment variables using Pydantic.
Environment variables override TOML settings for secrets and deployment-specific config.

@decision DEC-CONFIG-001
@title Pydantic Settings with TOML + env var layering
@status accepted
@rationale TOML provides readable defaults and profiles (paper vs live). Environment
variables enable secrets (.env) and deployment overrides without touching config files.
Pydantic validates types and provides auto-completion.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cerebrum.core.types import TradingMode


class ExchangeConfig(BaseSettings):
    """Exchange API configuration."""
    name: str = "kraken"
    api_key: str = Field(default="", description="Exchange API key")
    api_secret: str = Field(default="", description="Exchange API secret")
    testnet: bool = False
    rate_limit_per_minute: int = 60
    websocket_enabled: bool = True

    model_config = SettingsConfigDict(
        env_prefix="EXCHANGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class RiskConfig(BaseSettings):
    """Risk management parameters."""
    max_position_size_usd: Decimal = Field(
        default=Decimal("1000.0"),
        description="Maximum position size per asset in USD"
    )
    max_total_exposure_usd: Decimal = Field(
        default=Decimal("5000.0"),
        description="Maximum total portfolio exposure"
    )
    max_drawdown_percent: Decimal = Field(
        default=Decimal("5.0"),
        description="Circuit breaker: halt trading if drawdown exceeds this %"
    )
    max_daily_loss_usd: Decimal = Field(
        default=Decimal("500.0"),
        description="Maximum loss allowed in a single day"
    )
    position_size_percent: Decimal = Field(
        default=Decimal("2.0"),
        description="Default position size as % of portfolio"
    )
    min_signal_strength: Decimal = Field(
        default=Decimal("0.3"),
        description="Minimum signal strength required to execute trades"
    )
    stop_loss_percent: Decimal = Field(
        default=Decimal("2.0"),
        description="Close position if unrealized loss exceeds this % of entry value"
    )
    take_profit_percent: Decimal = Field(
        default=Decimal("3.0"),
        description="Close position if unrealized gain exceeds this % of entry value"
    )
    max_position_age_minutes: int = Field(
        default=120,
        description="Close position if open longer than this many minutes"
    )
    post_fill_cooldown_seconds: int = Field(
        default=300,
        description="Minimum seconds between fills per symbol to prevent rapid-fire ordering"
    )
    volatility_gate_min_range_pct: Decimal = Field(
        default=Decimal("0.5"),
        description=(
            "Minimum price range % required to allow trading. "
            "Orders denied when (max-min)/min*100 < this value. "
            "Default 0.5% covers round-trip commission + slippage."
        )
    )
    volatility_gate_window_size: int = Field(
        default=300,
        description=(
            "Number of recent price ticks per symbol used to calculate volatility. "
            "~5 minutes at 1 tick/sec. Rule approves during cold start (window not yet full)."
        )
    )
    sideways_suppression_min_range_pct: Decimal = Field(
        default=Decimal("1.0"),
        description=(
            "Minimum price range % required to allow BUY entries in SIDEWAYS regime. "
            "Higher than volatility_gate (1.0% vs 0.5%) because TP must be reachable "
            "above commission. Session 5: 3% TP unreachable in <0.5% range markets."
        )
    )
    sideways_suppression_window_size: int = Field(
        default=18000,
        description=(
            "Number of recent price ticks for SIDEWAYS suppression range check. "
            "~5 hours at 1 tick/sec — same as macro gate to detect session-level flatness."
        )
    )
    macro_volatility_min_range_pct: Decimal = Field(
        default=Decimal("0.8"),
        description=(
            "Minimum session-level price range % to allow trading. "
            "MacroVolatilityGateRule uses a 5-hour window to catch sessions that are "
            "globally flat even if the 5-min window shows local noise."
        )
    )
    macro_volatility_window_size: int = Field(
        default=18000,
        description=(
            "Number of recent price ticks for macro volatility gate (~5 hours). "
            "Must be much larger than volatility_gate_window_size to distinguish "
            "session-level from tick-level flatness."
        )
    )
    adaptive_tp: bool = Field(
        default=False,
        description=(
            "Enable adaptive take-profit based on recent price range. "
            "When True: effective_tp = max(min_tp_percent, range_pct * tp_multiplier). "
            "When False: use fixed take_profit_percent (backward compatible)."
        )
    )
    tp_multiplier: Decimal = Field(
        default=Decimal("1.5"),
        description=(
            "Multiplier applied to recent range_pct to compute adaptive TP. "
            "effective_tp = range_pct * tp_multiplier. Higher = more ambitious target."
        )
    )
    min_tp_percent: Decimal = Field(
        default=Decimal("0.3"),
        description=(
            "Floor for adaptive take-profit — never target less than this %. "
            "Should exceed round-trip commission cost (~0.32%). "
            "Prevents TP being set so low that commission guarantees a loss."
        )
    )
    min_profit_to_commission_ratio: Decimal = Field(
        default=Decimal("2.0"),
        description=(
            "Minimum ratio of expected price range to round-trip commission cost. "
            "CommissionGateRule denies orders when range_pct < commission_pct * 2 * ratio. "
            "Default 2.0 requires the recent range to be at least 2x the round-trip "
            "commission (e.g. 0.64% minimum range for Kraken 0.16% maker fee). "
            "Raise to be more conservative; lower to allow thinner-margin trades."
        )
    )
    max_open_positions_per_symbol: int = Field(
        default=2,
        description=(
            "Maximum concurrent open positions per (strategy, symbol) pair. "
            "Prevents position pile-up."
        ),
    )
    min_hold_minutes: int = Field(
        default=0,
        description=(
            "Minimum minutes to hold a position before SL/TP/adaptive-TP exits are "
            "evaluated. Time-based (max_position_age) exits still fire regardless. "
            "Default 0 = no minimum hold (backward-compatible). DEC-EXIT-006."
        ),
    )

    @field_validator("max_drawdown_percent", "position_size_percent")
    @classmethod
    def validate_percentage(cls, v: Decimal) -> Decimal:
        if v < 0 or v > 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v

    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        extra="ignore",
    )


class PaperTradingConfig(BaseSettings):
    """Paper trading simulation settings."""
    initial_balance_usd: Decimal = Field(
        default=Decimal("10000.0"),
        description="Starting balance for paper trading"
    )
    commission_percent: Decimal = Field(
        default=Decimal("0.1"),
        description="Simulated commission as % of trade value"
    )
    slippage_percent: Decimal = Field(
        default=Decimal("0.05"),
        description="Simulated slippage as % of trade value"
    )
    state_file: Path = Field(
        default=Path("data/paper_state.json"),
        description="Persistence file for paper trading state"
    )

    model_config = SettingsConfigDict(
        env_prefix="PAPER_",
        extra="ignore",
    )


class SignalConfig(BaseSettings):
    """Signal generation configuration."""
    candle_interval_seconds: int = Field(
        default=60,
        description="Candle aggregation interval (60 = 1 minute)"
    )
    rsi_period: int = Field(default=14, description="RSI period")
    rsi_oversold: int = Field(default=30, description="RSI oversold threshold")
    rsi_overbought: int = Field(default=70, description="RSI overbought threshold")
    macd_fast: int = Field(default=12, description="MACD fast period")
    macd_slow: int = Field(default=26, description="MACD slow period")
    macd_signal: int = Field(default=9, description="MACD signal period")
    bb_period: int = Field(default=20, description="Bollinger Bands period")
    bb_std_dev: float = Field(default=2.0, description="Bollinger Bands std dev")
    vwap_period: int = Field(default=20, description="VWAP period")
    aggregation_threshold: Decimal = Field(
        default=Decimal("0.3"),
        description="Minimum aggregate signal strength to emit"
    )
    aggregation_window_seconds: int = Field(
        default=5,
        description="Time window for signal aggregation"
    )
    sr_pivot_lookback: int = Field(
        default=5,
        description="Candles on each side to confirm a pivot high/low for S/R detection"
    )
    sr_min_touches: int = Field(
        default=2,
        description="Minimum touch count for a valid support/resistance level"
    )
    sr_proximity_pct: float = Field(
        default=0.3,
        description="Price distance (%) to S/R level that triggers a signal"
    )

    model_config = SettingsConfigDict(
        env_prefix="SIGNAL_",
        extra="ignore",
    )


class TradingConfig(BaseSettings):
    """Core trading configuration."""
    mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode: paper, live, or backtest"
    )
    symbols: list[str] = Field(
        default=["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"],
        description="Trading pairs to monitor"
    )
    data_refresh_seconds: int = Field(
        default=1,
        description="Market data refresh interval in seconds"
    )

    model_config = SettingsConfigDict(
        env_prefix="TRADING_",
        extra="ignore",
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format: json or console")
    file: Path | None = Field(default=None, description="Log file path")

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        extra="ignore",
    )


class RegimeConfig(BaseSettings):
    """Regime detection configuration."""
    window_size: int = Field(default=100, description="Number of price points for regime calculation")
    update_interval: int = Field(default=20, description="Update regime every N market data events")
    cumulative_trend_threshold: float = Field(
        default=0.005,
        description="Cumulative return threshold to detect slow trends (0.5%)"
    )
    ma_slope_threshold: float = Field(
        default=0.00005,
        description="MA slope threshold for directional momentum"
    )
    mean_return_threshold: float = Field(
        default=0.002,
        description="Mean return threshold for strong trends (0.2%)"
    )
    volatility_threshold: float = Field(
        default=0.03,
        description="Volatility threshold for VOLATILE regime (3%)"
    )
    ma_period: int = Field(default=10, description="Moving average period for slope calculation")
    long_window_size: int = Field(
        default=3000,
        description="Long-term price window for slow drift detection (~50 min at 1 tick/sec)"
    )
    long_cumulative_threshold: float = Field(
        default=0.001,
        description="Cumulative return threshold for long window (0.1%)"
    )
    buy_suppression_factor: str = Field(
        default="0.2",
        description="Buy score multiplier in high-confidence BEAR regime"
    )
    buy_suppression_min_confidence: str = Field(
        default="0.8",
        description="Minimum regime confidence to trigger buy suppression"
    )
    bear_halt_min_confidence: str = Field(
        default="0.7",
        description="Minimum BEAR confidence to halt all trading for a symbol"
    )
    min_hold_count: int = Field(
        default=3,
        description="Consecutive readings required before committing to a regime transition (DEC-REGIME-005)"
    )
    halt_regimes: list[str] = Field(
        default=["BEAR", "UNKNOWN"],
        description=(
            "Regimes that trigger full trade halt. UNKNOWN blocks trading "
            "before the regime detector has enough data (startup/reconnect). "
            "DEC-REGIME-006."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="REGIME_",
        extra="ignore",
    )


class IntelligenceConfig(BaseSettings):
    """Intelligence layer configuration."""
    cryptopanic_api_key: str = Field(default="", description="CryptoPanic API key")
    cryptopanic_poll_interval_seconds: int = Field(
        default=300,
        description="CryptoPanic poll interval (5 minutes)"
    )
    newsapi_api_key: str = Field(default="", description="NewsAPI.org API key")
    newsapi_poll_interval_seconds: int = Field(
        default=1800,
        description="NewsAPI poll interval (30 minutes)"
    )
    fear_greed_poll_interval_seconds: int = Field(
        default=3600,
        description="Fear & Greed Index poll interval (1 hour)"
    )
    enable_finbert: bool = Field(
        default=False,
        description="Enable FinBERT sentiment analysis (requires transformers)"
    )
    enable_hmm_regime: bool = Field(
        default=False,
        description="Enable HMM-based regime detection (requires hmmlearn)"
    )

    model_config = SettingsConfigDict(
        env_prefix="INTELLIGENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMConfig(BaseSettings):
    """LLM configuration for news reasoning."""
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    model: str = Field(default="claude-haiku-4-5", description="Claude model to use")
    max_calls_per_hour: int = Field(default=10, description="Rate limit for LLM calls")
    news_batch_size: int = Field(default=5, description="Number of news items per LLM call")
    news_batch_window_seconds: int = Field(
        default=300,
        description="Time window for batching news (5 minutes)"
    )
    timeout_seconds: int = Field(default=30, description="API timeout")

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MonitoringConfig(BaseSettings):
    """Monitoring and statistics configuration."""
    dashboard_enabled: bool = Field(
        default=True,
        description="Enable real-time dashboard display"
    )
    update_interval_seconds: int = Field(
        default=30,
        description="Dashboard update interval in seconds"
    )
    report_file_path: Path | None = Field(
        default=Path("data/session_report.txt"),
        description="Path to save session reports"
    )
    backtest_cache_dir: Path = Field(
        default=Path("data/backtest_cache"),
        description="Directory for cached backtest data"
    )

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_",
        extra="ignore",
    )


class AlpacaConfig(BaseSettings):
    """
    Alpaca stock trading API configuration.

    Used when the system operates in multi-asset mode (stocks + crypto).
    The [alpaca] section in TOML is optional — omitting it uses all defaults,
    so existing configs without an [alpaca] section continue to work.

    @decision DEC-ALPACA-CONFIG-001
    @title AlpacaConfig as optional BaseSettings with all safe defaults
    @status accepted
    @rationale Alpaca is an optional extension for stock trading. Adding it as
    a separate config class with empty-string API key defaults means the system
    boots without Alpaca credentials and only activates stock trading when the
    user explicitly configures keys. The default symbols list is empty so no
    unexpected stock subscriptions occur on first boot.
    """

    api_key: str = Field(default="", description="Alpaca API key")
    secret_key: str = Field(default="", description="Alpaca secret key")
    paper: bool = Field(
        default=True,
        description="Use Alpaca paper trading endpoint (True) or live (False)"
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="Stock symbols to trade (e.g. ['AAPL', 'MSFT', 'NVDA'])"
    )
    poll_interval_seconds: int = Field(
        default=5,
        description="Market data polling interval in seconds"
    )
    websocket_enabled: bool = Field(
        default=True,
        description="Enable WebSocket streaming for real-time data (future use)"
    )

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ProfileConfig(BaseSettings):
    """
    User-facing risk profile selection.

    All override fields are Optional — a profile only changes what it explicitly
    specifies. Fields left as None fall through to the base config values.
    Profiles are applied at runtime via ProfileManager.apply_profile().
    """
    name: str = ""
    symbols: list[str] = Field(default_factory=list)
    # Risk overrides
    position_size_percent: Decimal | None = None
    stop_loss_percent: Decimal | None = None
    take_profit_percent: Decimal | None = None
    max_position_age_minutes: int | None = None
    post_fill_cooldown_seconds: int | None = None
    min_signal_strength: Decimal | None = None
    # Signal overrides
    aggregation_threshold: Decimal | None = None
    # Exit monitor overrides
    adaptive_tp: bool | None = None
    tp_multiplier: Decimal | None = None
    min_tp_percent: Decimal | None = None

    model_config = SettingsConfigDict(
        env_prefix="PROFILE_",
        extra="ignore",
    )


class Config(BaseSettings):
    """Master configuration combining all subsystems."""
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def from_toml(cls, toml_path: Path) -> tuple["Config", dict]:
        """Load configuration from a TOML file, with env var overrides.

        Returns:
            Tuple of (Config, raw_toml_dict).
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        if not toml_path.exists():
            return cls(), {}

        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)

        return cls(**toml_data), toml_data
