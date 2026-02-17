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

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(min_level="INFO"),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

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

        # Initialize paper trading adapter for execution
        self.paper_adapter = PaperTradingAdapter(
            self.bus,
            {},
            initial_balance=self.config.paper.initial_balance_usd,
            commission_percent=self.config.paper.commission_percent,
            slippage_percent=self.config.paper.slippage_percent,
            state_file=self.config.paper.state_file,
        )
        await self.paper_adapter.connect()

        # Subscribe to market data
        await self.kraken_adapter.subscribe_market_data(self.config.trading.symbols)

        self._log.info(
            "cerebrumcoin_started",
            symbols=self.config.trading.symbols,
            initial_balance=str(self.config.paper.initial_balance_usd),
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the trading system gracefully."""
        self._log.info("cerebrumcoin_stopping")

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
