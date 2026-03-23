"""
Risk manager coordinating all risk rules and order validation.

Applies risk rules to signals before converting to orders, ensuring safe trading.

@decision DEC-RISK-MGR-001
@title RiskManager as stateful coordinator with per-rule denial counters
@status accepted
@rationale RiskManager owns the full order validation pipeline: it receives
combined SignalEvents, creates proposed orders, applies risk rules in sequence
(any DENY blocks; MODIFY adjusts amount), and emits approved OrderEvents.
Denial counters (DEC-DENIAL-001) are accumulated here rather than in individual
rules because the manager is the single choke-point that sees every denial.
Single-threaded asyncio means no locking is needed for the counter dict.
"""

from dataclasses import replace
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

        # Per-rule denial counters — incremented each time a rule returns DENY.
        # Single-threaded asyncio means no lock needed.
        # @decision DEC-DENIAL-001: denial counters for rule observability
        self._denial_counts: dict[str, int] = {}

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

        # Capture signal snapshot for learning
        signal_snapshot = {
            signal.signal_type.value: {
                "strength": float(signal.strength),
                "confidence": float(signal.confidence),
                "action": signal.action.value,
            }
        }

        return OrderEvent(
            event_type=EventType.ORDER,
            timestamp=time(),
            order_id=str(uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            amount=amount,
            price=None,  # Market order
            metadata={"signals": signal_snapshot},
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
                # Increment per-rule denial counter for observability
                self._denial_counts[rule.name] = self._denial_counts.get(rule.name, 0) + 1

                # Any deny blocks the order
                self._log.warning(
                    "order_denied_by_rule",
                    rule=rule.name,
                    reason=result.reason,
                    risk_level=result.risk_level.value,
                    symbol=signal.symbol,
                    denial_counts=dict(self._denial_counts),
                )
                await self._emit_risk_alert(
                    signal.symbol,
                    result.risk_level,
                    f"Order denied by {rule.name}: {result.reason}",
                )
                return False, None, result.risk_level
            
            elif result.decision == RuleDecision.MODIFY:
                # Modify order parameters
                if result.modified_amount is not None:
                    # Create modified order using replace() — forward-compatible with new fields
                    # @decision DEC-RISK-MGR-002: use dataclasses.replace() for frozen OrderEvent mutation
                    # Rationale: listing all fields manually breaks whenever a new field is added
                    # (e.g. strategy_id). replace() copies all unspecified fields automatically.
                    current_order = replace(current_order, amount=result.modified_amount)
                
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
    
    @property
    def denial_counts(self) -> dict[str, int]:
        """
        Return a copy of per-rule denial counters.

        Returns a copy so callers cannot mutate the live dict.
        Keys are rule names; values are the number of times that rule
        has issued a DENY since this RiskManager was created.
        """
        return dict(self._denial_counts)

    def add_rule(self, rule: RiskRule) -> None:
        """Add a risk rule."""
        self._rules.append(rule)
        self._log.info("rule_added", rule=rule.name)
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove a risk rule by name."""
        self._rules = [r for r in self._rules if r.name != rule_name]
        self._log.info("rule_removed", rule=rule_name)
