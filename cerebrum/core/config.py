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
        default=["BTC/USD", "ETH/USD"],
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


class Config(BaseSettings):
    """Master configuration combining all subsystems."""
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def from_toml(cls, toml_path: Path) -> "Config":
        """Load configuration from a TOML file, with env var overrides."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore

        if not toml_path.exists():
            return cls()

        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)

        return cls(**toml_data)
