"""
Composable risk rules for trade validation.

Each rule is an independent check that can approve, reject, or modify orders.

@decision DEC-RISK-001
@title Composable risk rules architecture
@status accepted
@rationale Each risk rule is independent and testable. Rules can be enabled/disabled
in config. Manager applies rules in sequence: any DENY blocks the order, MODIFY
adjusts parameters (position size), APPROVE allows it through. This enables flexible
risk profiles (conservative = more rules, aggressive = fewer rules).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

import structlog

from cerebrum.core.events import OrderEvent, SignalEvent
from cerebrum.core.types import Amount, RiskLevel, Side

from .portfolio import PortfolioTracker

logger = structlog.get_logger()


class RuleDecision(str, Enum):
    """Risk rule decision."""
    APPROVE = "approve"
    DENY = "deny"
    MODIFY = "modify"


@dataclass
class RuleResult:
    """Result of a risk rule evaluation."""
    decision: RuleDecision
    reason: str
    risk_level: RiskLevel
    modified_amount: Amount | None = None


class RiskRule(ABC):
    """Abstract base class for risk rules."""
    
    def __init__(self, name: str) -> None:
        """
        Initialize risk rule.
        
        Args:
            name: Rule identifier
        """
        self.name = name
        self._log = logger.bind(component=f"risk_rule_{name}")
    
    @abstractmethod
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Evaluate if an order should be allowed.
        
        Args:
            signal: Triggering signal
            order: Proposed order
            portfolio: Current portfolio state
        
        Returns:
            RuleResult with decision
        """
        pass


class MaxPositionSizeRule(RiskRule):
    """Limit maximum position size per symbol."""
    
    def __init__(self, max_position_usd: Decimal) -> None:
        """
        Initialize max position size rule.
        
        Args:
            max_position_usd: Maximum position value in USD
        """
        super().__init__("max_position_size")
        self._max_position = max_position_usd
    
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Check if order exceeds max position size."""
        if order.price is None:
            # Can't evaluate market orders without price
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="Market order - size check skipped",
                risk_level=RiskLevel.LOW,
            )
        
        # Calculate order value
        order_value = order.amount * order.price
        
        # Check existing position
        existing_pos = portfolio.get_position(order.symbol)
        if existing_pos:
            # Calculate new position size
            if order.side == Side.BUY:
                new_amount = existing_pos.amount + order.amount
            else:
                new_amount = existing_pos.amount - order.amount
            
            new_value = abs(new_amount * order.price)
        else:
            new_value = order_value
        
        if new_value > self._max_position:
            # Reduce order size to fit limit
            max_amount = self._max_position / order.price
            
            return RuleResult(
                decision=RuleDecision.MODIFY,
                reason=f"Position size reduced: {new_value} > {self._max_position} USD",
                risk_level=RiskLevel.MEDIUM,
                modified_amount=max_amount,
            )
        
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason="Position size within limit",
            risk_level=RiskLevel.LOW,
        )


class MaxTotalExposureRule(RiskRule):
    """Limit total portfolio exposure."""
    
    def __init__(self, max_exposure_usd: Decimal) -> None:
        """
        Initialize max exposure rule.
        
        Args:
            max_exposure_usd: Maximum total exposure in USD
        """
        super().__init__("max_total_exposure")
        self._max_exposure = max_exposure_usd
    
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Check if order exceeds total exposure limit."""
        current_exposure = portfolio.get_total_exposure()
        
        if order.price is None:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="Market order - exposure check approximate",
                risk_level=RiskLevel.LOW,
            )
        
        order_value = order.amount * order.price
        new_exposure = current_exposure + order_value
        
        if new_exposure > self._max_exposure:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"Total exposure limit exceeded: {new_exposure} > {self._max_exposure} USD",
                risk_level=RiskLevel.HIGH,
            )
        
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason="Total exposure within limit",
            risk_level=RiskLevel.LOW,
        )


class MaxDrawdownRule(RiskRule):
    """Circuit breaker: halt trading on excessive drawdown."""
    
    def __init__(self, max_drawdown_percent: Decimal) -> None:
        """
        Initialize max drawdown rule.
        
        Args:
            max_drawdown_percent: Maximum allowed drawdown (%)
        """
        super().__init__("max_drawdown")
        self._max_drawdown = max_drawdown_percent
    
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Check if drawdown exceeds limit."""
        drawdown = portfolio.get_drawdown_percent()
        
        if drawdown > self._max_drawdown:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"Circuit breaker: drawdown {drawdown:.1f}% > {self._max_drawdown}%",
                risk_level=RiskLevel.CRITICAL,
            )
        
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=f"Drawdown {drawdown:.1f}% within limit",
            risk_level=RiskLevel.LOW,
        )


class MinSignalStrengthRule(RiskRule):
    """Require minimum signal strength to trade."""
    
    def __init__(self, min_strength: Decimal = Decimal("0.4")) -> None:
        """
        Initialize min strength rule.
        
        Args:
            min_strength: Minimum signal strength required
        """
        super().__init__("min_signal_strength")
        self._min_strength = min_strength
    
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Check if signal meets minimum strength."""
        if signal.strength < self._min_strength:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=f"Signal too weak: {signal.strength} < {self._min_strength}",
                risk_level=RiskLevel.LOW,
            )
        
        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason="Signal strength sufficient",
            risk_level=RiskLevel.LOW,
        )


class PositionSizingRule(RiskRule):
    """Calculate position size as percentage of portfolio."""
    
    def __init__(self, position_size_percent: Decimal = Decimal("2.0")) -> None:
        """
        Initialize position sizing rule.
        
        Args:
            position_size_percent: Position size as % of equity
        """
        super().__init__("position_sizing")
        self._size_percent = position_size_percent
    
    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Calculate appropriate position size."""
        equity = portfolio.get_total_equity()
        target_value = equity * (self._size_percent / 100)

        # Get price (either from order or from latest market data)
        price = order.price
        if price is None:
            # For market orders, use latest known price from portfolio tracker
            price = portfolio.get_latest_price(order.symbol)
            if price is None:
                # No price data available - can't safely size the order.
                # Returning DENY prevents the 1.0 BTC placeholder amount from
                # reaching the exchange, which would be rejected as insufficient_balance.
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason="Cannot size order: no price data available",
                    risk_level=RiskLevel.MEDIUM,
                )

        target_amount = target_value / price

        # Adjust by signal strength
        adjusted_amount = target_amount * signal.strength

        return RuleResult(
            decision=RuleDecision.MODIFY,
            reason=f"Position sized at {self._size_percent}% of equity, adjusted by signal strength",
            risk_level=RiskLevel.LOW,
            modified_amount=adjusted_amount,
        )
