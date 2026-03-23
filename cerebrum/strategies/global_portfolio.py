"""
GlobalPortfolio: read-only aggregation view across all strategy portfolios.

Aggregates equity, drawdown, and positions from each strategy's individual
PortfolioTracker. This is a pure query object — it has no event subscriptions
and mutates no state. The individual PortfolioTrackers are the source of truth;
GlobalPortfolio merely provides a consolidated view for monitoring and
circuit-breaker decisions.

@decision DEC-STRAT-004
@title GlobalPortfolio as read-only aggregation view
@status accepted
@rationale Each strategy owns its isolated PortfolioTracker (DEC-STRAT-002).
The dashboard and cross-strategy risk checks (e.g., global drawdown circuit
breaker) need a unified view without breaking that isolation. GlobalPortfolio
provides this by reading from all trackers at call time — no caching, no
events, no mutation. Drawdown is computed from the global peak across all
strategies combined, so a drawdown in one strategy is visible at the global
level even if the other strategies are flat.
"""

from decimal import Decimal
from typing import Any

import structlog

from cerebrum.risk.portfolio import PortfolioTracker

logger = structlog.get_logger()


class GlobalPortfolio:
    """
    Read-only aggregate view of all strategy portfolios.

    Features:
    - Total equity across all strategies
    - Global drawdown from combined peak
    - Per-strategy equity lookup
    - Merged position view (all open positions across strategies)

    All methods compute on-demand from the live PortfolioTracker instances.
    There is no internal state beyond the tracker references and the recorded
    global equity peak.
    """

    def __init__(self, strategy_portfolios: dict[str, PortfolioTracker]) -> None:
        """
        Initialize global portfolio view.

        Args:
            strategy_portfolios: Mapping of strategy name → PortfolioTracker.
                                 These are the authoritative portfolio objects
                                 owned by the StrategyRegistry.
        """
        self._portfolios = strategy_portfolios
        self._peak_equity: Decimal = Decimal("0.0")
        self._log = logger.bind(component="global_portfolio")
        self._log.info(
            "global_portfolio_initialized",
            strategies=list(strategy_portfolios.keys()),
        )

    def get_total_equity(self) -> Decimal:
        """
        Sum equity across all strategy portfolios.

        Updates the global peak if the current total exceeds it, enabling
        accurate global drawdown calculation.

        Returns:
            Total equity in USD across all strategies.
        """
        total = sum(
            portfolio.get_total_equity()
            for portfolio in self._portfolios.values()
        )
        # Track global equity peak for drawdown calculation
        if total > self._peak_equity:
            self._peak_equity = total
        return total

    def get_total_drawdown(self) -> Decimal:
        """
        Global drawdown from the combined equity peak.

        Returns the percentage drawdown from the highest total equity
        ever seen across all strategies combined.

        Returns:
            Drawdown percentage (0.0 = no drawdown, 10.0 = 10% below peak).
        """
        if self._peak_equity == Decimal("0.0"):
            return Decimal("0.0")
        current = self.get_total_equity()
        drawdown = (self._peak_equity - current) / self._peak_equity * Decimal("100")
        return max(Decimal("0.0"), drawdown)

    def get_strategy_equity(self, name: str) -> Decimal:
        """
        Get equity for a specific strategy.

        Args:
            name: Strategy name as registered in strategy_portfolios.

        Returns:
            Strategy equity in USD, or Decimal("0.0") if strategy not found.
        """
        portfolio = self._portfolios.get(name)
        if portfolio is None:
            self._log.warning("strategy_not_found", name=name)
            return Decimal("0.0")
        return portfolio.get_total_equity()

    def get_all_positions(self) -> dict[str, Any]:
        """
        Merged view of all open positions across all strategies.

        Returns a dict keyed by "strategy_name:symbol" to avoid collisions
        when multiple strategies hold positions in the same symbol.

        Returns:
            Dict mapping "strategy:symbol" → Position object.
        """
        merged: dict[str, Any] = {}
        for strategy_name, portfolio in self._portfolios.items():
            for symbol, position in portfolio.get_all_positions().items():
                key = f"{strategy_name}:{symbol}"
                merged[key] = position
        return merged

    def get_strategy_names(self) -> list[str]:
        """Return list of all registered strategy names."""
        return list(self._portfolios.keys())
