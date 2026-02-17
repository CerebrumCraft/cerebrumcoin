"""
Risk manager coordinating all risk rules and order validation.

Applies risk rules to signals before converting to orders, ensuring safe trading.
"""

from decimal import Decimal
from time import time
from uuid import uuid4

import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import Event, OrderEvent, RiskAlertEvent, SignalEvent
from cerebrum.core.types import (
    EventType,
    OrderType,
    RiskLevel,
    Side,
    SignalAction,
    SignalType,
)

from .portfolio import PortfolioTracker
from .rules import RiskRule, RuleDecision

logger = structlog.get_logger()


class RiskManager:
    """
    Risk manager for trade validation and position sizing.
    
    Features:
    - Applies risk rules to signals
    - Converts approved signals to orders
    - Emits risk alerts
    - Configurable rule set
    """
    
    def __init__(
        self,
        bus: EventBus,
        portfolio: PortfolioTracker,
        rules: list[RiskRule] | None = None,
    ) -> None:
        """
        Initialize risk manager.
        
        Args:
            bus: Event bus
            portfolio: Portfolio tracker
            rules: List of risk rules to apply
        """
        self._bus = bus
        self._portfolio = portfolio
        self._rules = rules or []
        self._log = logger.bind(component="risk_manager")
        
        # Subscribe to combined signals only
        bus.subscribe(
            EventType.SIGNAL,
            self._on_signal,
            subscriber_name="risk_manager",
        )
        
        self._log.info(
            "risk_manager_initialized",
            rule_count=len(self._rules),
            rules=[r.name for r in self._rules],
        )
    
    async def _on_signal(self, event: Event) -> None:
        """Handle signals and apply risk checks."""
        if not isinstance(event, SignalEvent):
            return
        
        # Only process combined signals (output of aggregator)
        if event.signal_type != SignalType.COMBINED:
            return
        
        # Ignore HOLD and CLOSE actions
        if event.action not in (SignalAction.BUY, SignalAction.SELL):
            return
        
        # Create proposed order
        order = self._create_order_from_signal(event)
        
        # Apply risk rules
        approved, final_order, risk_level = await self._apply_rules(event, order)
        
        if approved and final_order is not None:
            # Emit approved order
            await self._bus.publish(final_order)
            
            self._log.info(
                "order_approved",
                symbol=event.symbol,
                side=final_order.side.value,
                amount=str(final_order.amount),
                risk_level=risk_level.value,
            )
        else:
            self._log.warning(
                "order_rejected",
                symbol=event.symbol,
                risk_level=risk_level.value,
            )
    
    def _create_order_from_signal(self, signal: SignalEvent) -> OrderEvent:
        """Create a proposed order from a signal."""
        # Determine side
        side = Side.BUY if signal.action == SignalAction.BUY else Side.SELL
        
        # Initial amount (will be sized by rules)
        amount = Decimal("1.0")  # Placeholder
        
        return OrderEvent(
            event_type=EventType.ORDER,
            timestamp=time(),
            order_id=str(uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            price=None,  # Market order
        )
    
    async def _apply_rules(
        self,
        signal: SignalEvent,
        order: OrderEvent,
    ) -> tuple[bool, OrderEvent | None, RiskLevel]:
        """
        Apply all risk rules to an order.
        
        Returns:
            (approved, final_order, risk_level)
        """
        current_order = order
        highest_risk = RiskLevel.LOW
        
        for rule in self._rules:
            result = rule.evaluate(signal, current_order, self._portfolio)
            
            # Track highest risk level
            if result.risk_level.value > highest_risk.value:
                highest_risk = result.risk_level
            
            if result.decision == RuleDecision.DENY:
                # Any deny blocks the order
                await self._emit_risk_alert(
                    signal.symbol,
                    result.risk_level,
                    f"Order denied by {rule.name}: {result.reason}",
                )
                return False, None, result.risk_level
            
            elif result.decision == RuleDecision.MODIFY:
                # Modify order parameters
                if result.modified_amount is not None:
                    # Create modified order (OrderEvent is frozen)
                    current_order = OrderEvent(
                        event_type=current_order.event_type,
                        timestamp=current_order.timestamp,
                        order_id=current_order.order_id,
                        symbol=current_order.symbol,
                        side=current_order.side,
                        order_type=current_order.order_type,
                        amount=result.modified_amount,
                        price=current_order.price,
                        stop_price=current_order.stop_price,
                        status=current_order.status,
                        metadata=current_order.metadata,
                    )
                
                self._log.debug(
                    "rule_modified_order",
                    rule=rule.name,
                    reason=result.reason,
                    new_amount=str(current_order.amount),
                )
        
        # All rules passed or modified
        return True, current_order, highest_risk
    
    async def _emit_risk_alert(
        self,
        symbol: str,
        level: RiskLevel,
        message: str,
    ) -> None:
        """Emit a risk alert event."""
        alert = RiskAlertEvent(
            event_type=EventType.RISK_ALERT,
            timestamp=time(),
            level=level,
            message=message,
            symbol=symbol,
        )
        await self._bus.publish(alert)
    
    def add_rule(self, rule: RiskRule) -> None:
        """Add a risk rule."""
        self._rules.append(rule)
        self._log.info("rule_added", rule=rule.name)
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove a risk rule by name."""
        self._rules = [r for r in self._rules if r.name != rule_name]
        self._log.info("rule_removed", rule=rule_name)
