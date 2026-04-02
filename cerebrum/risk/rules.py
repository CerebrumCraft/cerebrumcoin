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
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from cerebrum.core.events import Event, FillEvent, MarketDataEvent, OrderEvent, RegimeChangeEvent, SignalEvent
from cerebrum.core.types import Amount, EventType, Price, RiskLevel, Side, Symbol

from .portfolio import PortfolioTracker

if TYPE_CHECKING:
    from cerebrum.core.bus import EventBus

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
    """Calculate position size as percentage of portfolio.

    # @decision DEC-SIZING-001
    # @title Minimum trade value floor to prevent commission-killed micro-trades
    # @status accepted
    # @rationale Investigation showed $20 range_trading trades where 0.32% round-trip
    # commission ate 33% of wins. Floor at $100 keeps commission below 10%.
    # min_trade_value_usd=None (default) preserves full backward compatibility.
    #
    # @decision DEC-SIZING-002
    # @title Floor signal multiplier at 0.6 to prevent position starvation
    # @status accepted
    # @rationale At 2% sizing × $5k capital, base target is exactly $100 — the
    # min_trade_value_usd floor. Any signal_strength < 1.0 shrinks adjusted_amount
    # below the floor, causing blanket DENY. The min_signal_strength rule already
    # gates weak signals; the sizer's job is to size viable trades, not re-filter.
    # Fix: clamp multiplier to max(signal.strength, 0.6). Range_trading also raised
    # from 2% → 5% so 0.6 floor × 5% × $5k = $150, comfortably above $100 min.
    """

    def __init__(
        self,
        position_size_percent: Decimal = Decimal("2.0"),
        min_trade_value_usd: Decimal | None = None,
    ) -> None:
        """
        Initialize position sizing rule.

        Args:
            position_size_percent: Position size as % of equity
            min_trade_value_usd: Minimum trade value in USD. Orders whose
                strength-adjusted value falls below this floor are denied.
                None (default) disables the check for backward compatibility.
        """
        super().__init__("position_sizing")
        self._size_percent = position_size_percent
        self._min_trade_value = min_trade_value_usd

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

        # DEC-SIZING-002: Floor signal multiplier at 0.6 to prevent position
        # starvation. Without this, low-strength signals (e.g., 0.3) shrink
        # position size below the min_trade_value_usd floor, causing blanket
        # denials. The min_signal_strength rule already filters weak signals;
        # the position sizer should size viable trades, not double-filter.
        strength_multiplier = max(signal.strength, Decimal("0.6"))
        adjusted_amount = target_amount * strength_multiplier

        # DEC-SIZING-001: check strength-adjusted trade value against floor
        if self._min_trade_value is not None:
            actual_value = adjusted_amount * price
            if actual_value < self._min_trade_value:
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason=(
                        f"Trade value ${actual_value:.2f} below minimum "
                        f"${self._min_trade_value}. Commission would dominate."
                    ),
                    risk_level=RiskLevel.LOW,
                )

        return RuleResult(
            decision=RuleDecision.MODIFY,
            reason=f"Position sized at {self._size_percent}% of equity, adjusted by signal strength (multiplier={strength_multiplier:.2f})",
            risk_level=RiskLevel.LOW,
            modified_amount=adjusted_amount,
        )


class PostFillCooldownRule(RiskRule):
    """
    Rate-limit new orders after a fill to prevent rapid-fire position accumulation.

    # @decision DEC-COOL-001: Post-fill cooldown prevents rapid-fire ordering.
    # Self-subscribes to FillEvents via bus (like ExitMonitor pattern).
    # Per-symbol tracking ensures independent cooldowns for each trading pair.
    # Default 300s (5 min) configurable via risk.post_fill_cooldown_seconds.
    #
    # Problem solved: without cooldown, the 60s aggregation window allows a new
    # BUY every minute — observed as 28 fills in 20 minutes during paper trading.
    # The cooldown enforces a minimum gap between fills per symbol, regardless of
    # how many buy signals the aggregator emits in that window.
    #
    # Design: _last_fill_time is updated on every FillEvent (both BUY and SELL).
    # This means selling also resets the cooldown, so the system won't immediately
    # re-enter a position after an exit fills. This is intentional — re-entries
    # should wait for a fresh signal window.
    """

    def __init__(
        self,
        cooldown_seconds: int,
        bus: "EventBus",
        _clock=None,
    ) -> None:
        """
        Initialize post-fill cooldown rule.

        Args:
            cooldown_seconds: Minimum seconds between fills per symbol.
                              Orders arriving sooner than this after the last
                              fill for the same symbol will be DENY'd.
            bus: Event bus to subscribe to FillEvents.
            _clock: Callable returning current time as float (default: time.time).
                    Injectable for testing without mocking or long sleeps.
        """
        import time as _time_module
        super().__init__("post_fill_cooldown")
        self._cooldown_seconds = cooldown_seconds
        self._clock = _clock if _clock is not None else _time_module.time
        # Per-symbol timestamp of the most recent fill (unix epoch float)
        self._last_fill_time: dict[Symbol, float] = {}

        bus.subscribe(
            EventType.FILL,
            self._on_fill,
            subscriber_name="post_fill_cooldown_rule",
        )

        self._log.info(
            "post_fill_cooldown_initialized",
            cooldown_seconds=cooldown_seconds,
        )

    async def _on_fill(self, event: Event) -> None:
        """Record fill timestamp per symbol when a FillEvent is received."""
        if not isinstance(event, FillEvent):
            return
        self._last_fill_time[event.symbol] = event.timestamp
        self._log.debug(
            "fill_recorded",
            symbol=event.symbol,
            fill_time=event.timestamp,
            cooldown_seconds=self._cooldown_seconds,
        )

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Deny the order if a fill for this symbol occurred within the cooldown window."""
        last_fill = self._last_fill_time.get(order.symbol)
        if last_fill is not None:
            elapsed = self._clock() - last_fill
            if elapsed < self._cooldown_seconds:
                remaining = int(self._cooldown_seconds - elapsed)
                return RuleResult(
                    decision=RuleDecision.DENY,
                    reason=(
                        f"Post-fill cooldown active for {order.symbol}: "
                        f"{remaining}s remaining (cooldown={self._cooldown_seconds}s)"
                    ),
                    risk_level=RiskLevel.LOW,
                )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason="No active cooldown",
            risk_level=RiskLevel.LOW,
        )


class RegimeTradeHaltRule(RiskRule):
    """
    Halt all trading for a symbol when regime is BEAR with high confidence.

    @decision DEC-REGIME-004
    @title Trade halt in BEAR regime
    @status accepted
    @rationale Session 4 showed 39% win rate during BEAR vs 73% non-BEAR.
    Even sell-side trades lose during strong downtrends. Stopping all trading
    in BEAR would have saved $37+ in losses. The 0.2x buy suppression alone
    is insufficient — a full halt is needed. The rule subscribes to
    REGIME_CHANGE events via the event bus (same pattern as PostFillCooldownRule)
    and maintains a per-symbol registry of the current regime and confidence.
    Orders for any symbol currently in BEAR with confidence >= min_confidence
    are denied regardless of direction (buy or sell).
    """

    def __init__(self, min_confidence: Decimal, bus: "EventBus") -> None:
        """
        Initialize regime trade halt rule.

        Args:
            min_confidence: Minimum BEAR confidence required to halt trading.
                            BEAR detections below this threshold are ignored.
            bus: Event bus to subscribe to REGIME_CHANGE events.
        """
        super().__init__("regime_trade_halt")
        self._min_confidence = min_confidence
        # symbol -> (regime, confidence)
        self._regimes: dict[str, tuple[str, Decimal]] = {}

        bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            subscriber_name="regime_halt_rule",
        )

        self._log.info(
            "regime_halt_rule_initialized",
            min_confidence=float(min_confidence),
        )

    async def _on_regime_change(self, event: Event) -> None:
        """Record the latest regime and confidence for the affected symbol."""
        if not isinstance(event, RegimeChangeEvent):
            return
        symbol = event.indicators.get("symbol", "")
        if symbol:
            self._regimes[symbol] = (event.to_regime, event.confidence)
            self._log.debug(
                "regime_updated",
                symbol=symbol,
                regime=event.to_regime,
                confidence=float(event.confidence),
            )

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """Deny the order if the symbol is currently in a high-confidence BEAR regime."""
        symbol = order.symbol
        regime_info = self._regimes.get(symbol)
        if regime_info is None:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="No regime data — trading allowed",
                risk_level=RiskLevel.LOW,
            )

        regime, confidence = regime_info
        if regime == "BEAR" and confidence >= self._min_confidence:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Trading halted: {symbol} in BEAR regime "
                    f"(confidence={float(confidence):.2f}, "
                    f"threshold={float(self._min_confidence):.2f})"
                ),
                risk_level=RiskLevel.HIGH,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=f"Regime {regime} does not trigger halt",
            risk_level=RiskLevel.LOW,
        )


class VolatilityGateRule(RiskRule):
    """
    Deny orders when recent price range is too small to cover round-trip commissions.

    @decision DEC-VOL-001
    @title Percentage price range as volatility metric
    @status accepted
    @rationale (max - min) / min * 100 is simple, interpretable, and directly
    models whether price swings are large enough to profit after commission.
    No look-ahead bias. Avoids statistical complexity of std-dev which can be
    high even in one-directional trends.

    @decision DEC-VOL-002
    @title Per-symbol rolling price window via MARKET_DATA event bus subscription
    @status accepted
    @rationale Mirrors PostFillCooldownRule and RegimeTradeHaltRule patterns —
    self-subscribes in __init__, maintains per-symbol dict of deque. Decoupled
    from regime detector; independent risk signal.

    @decision DEC-VOL-003
    @title Default threshold 0.5%, lookback 300 ticks, both configurable via TOML
    @status accepted
    @rationale 0.5% covers round-trip commission (~0.32%) plus slippage (~0.1%)
    with margin. 300 ticks (~5 min at 1 tick/sec) matches the regime detector's
    short window. APPROVE on cold start (fewer ticks than window) to not block
    early trades.
    """

    def __init__(
        self,
        min_range_pct: Decimal,
        window_size: int,
        bus: "EventBus",
    ) -> None:
        """
        Initialize volatility gate rule.

        Args:
            min_range_pct: Minimum price range percentage required to allow trading.
                           Orders are denied when (max - min) / min * 100 < min_range_pct.
            window_size: Number of recent price ticks to consider per symbol.
                         The window is a rolling deque — old prices fall off automatically.
            bus: Event bus to subscribe to MARKET_DATA events.
        """
        super().__init__("volatility_gate")
        self._min_range_pct = min_range_pct
        self._window_size = window_size
        # Per-symbol rolling price windows. deque(maxlen=N) auto-evicts oldest entries.
        self._price_windows: dict[Symbol, deque[Price]] = {}

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="volatility_gate_rule",
        )

        self._log.info(
            "volatility_gate_initialized",
            min_range_pct=float(min_range_pct),
            window_size=window_size,
        )

    async def _on_market_data(self, event: Event) -> None:
        """Append the latest price to the per-symbol rolling window."""
        if not isinstance(event, MarketDataEvent):
            return
        symbol = event.symbol
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque(maxlen=self._window_size)
        self._price_windows[symbol].append(event.price)
        self._log.debug(
            "price_recorded",
            symbol=symbol,
            price=float(event.price),
            window_len=len(self._price_windows[symbol]),
        )

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Deny the order if recent price range for the symbol is below the minimum threshold.

        Returns APPROVE during cold start (window not yet full) to avoid blocking
        early trades when the system has just started.
        """
        symbol = order.symbol
        window = self._price_windows.get(symbol)

        # Cold start: insufficient data to evaluate — allow trading
        if window is None or len(window) < self._window_size:
            current = len(window) if window is not None else 0
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=(
                    f"Insufficient data for {symbol}: warming up "
                    f"({current}/{self._window_size} ticks)"
                ),
                risk_level=RiskLevel.LOW,
            )

        # Calculate price range as a percentage of the minimum price
        price_min = min(window)
        price_max = max(window)
        range_pct = (price_max - price_min) / price_min * Decimal("100")

        if range_pct < self._min_range_pct:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Volatility gate: {symbol} range {float(range_pct):.4f}% "
                    f"< threshold {float(self._min_range_pct):.4f}% "
                    f"(too flat to cover commission)"
                ),
                risk_level=RiskLevel.LOW,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"Volatility sufficient: {symbol} range {float(range_pct):.4f}% "
                f">= {float(self._min_range_pct):.4f}%"
            ),
            risk_level=RiskLevel.LOW,
        )


class SidewaysSuppressionRule(RiskRule):
    """
    Deny BUY orders when the market is in a SIDEWAYS regime AND price range
    is below the minimum threshold (insufficient volatility to reach take-profit).

    @decision DEC-REGIME-005
    @title SIDEWAYS suppression: block BUY entries in low-vol sideways markets
    @status accepted
    @rationale Session 5 showed 0/17 win rate (-$61) in SIDEWAYS markets because
    take-profit (3%) is unreachable when intraday range is <0.5%. All positions hit
    the 120-min max-age timeout at a guaranteed loss. This rule denies new BUY entries
    when regime==SIDEWAYS AND the price range over a configurable window is below
    min_range_pct, preventing accumulation of guaranteed-loss positions.

    SELL orders (exits) are never blocked — the rule only suppresses new entries.
    This is intentional: if we're already in a position when the regime becomes
    SIDEWAYS+low-vol, we need exits to work to close the losing position.

    The rule subscribes to both REGIME_CHANGE (to track per-symbol regime) and
    MARKET_DATA (to maintain a rolling price window for range calculation), following
    the combined pattern of RegimeTradeHaltRule and VolatilityGateRule.

    APPROVE on cold start (insufficient data) and for BULL/BEAR regimes regardless
    of volatility — those regimes have directional momentum where TP is reachable.
    """

    def __init__(
        self,
        min_range_pct: Decimal,
        window_size: int,
        bus: "EventBus",
        exempt_strategies: set[str] | None = None,
    ) -> None:
        """
        Initialize SIDEWAYS suppression rule.

        Args:
            min_range_pct: Minimum price range percentage required to allow BUY orders
                           in SIDEWAYS regime. BUYs denied when (max-min)/min*100 < this.
            window_size: Number of recent price ticks to consider per symbol.
            bus: Event bus to subscribe to REGIME_CHANGE and MARKET_DATA events.
            exempt_strategies: Set of strategy_id values that bypass this rule entirely.
                               Range trading strategies need to trade in SIDEWAYS markets,
                               so they must be exempted from suppression.
        """
        super().__init__("sideways_suppression")
        self._min_range_pct = min_range_pct
        self._window_size = window_size
        self._exempt_strategies: set[str] = exempt_strategies or set()
        # Per-symbol regime tracking: symbol -> (regime, confidence)
        self._regimes: dict[str, tuple[str, Decimal]] = {}
        # Per-symbol rolling price windows for range calculation
        self._price_windows: dict[Symbol, deque[Price]] = {}

        bus.subscribe(
            EventType.REGIME_CHANGE,
            self._on_regime_change,
            subscriber_name="sideways_suppression_regime",
        )
        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="sideways_suppression_market_data",
        )

        self._log.info(
            "sideways_suppression_initialized",
            min_range_pct=float(min_range_pct),
            window_size=window_size,
        )

    async def _on_regime_change(self, event: Event) -> None:
        """Record the latest regime and confidence for the affected symbol."""
        if not isinstance(event, RegimeChangeEvent):
            return
        symbol = event.indicators.get("symbol", "")
        if symbol:
            self._regimes[symbol] = (event.to_regime, event.confidence)
            self._log.debug(
                "sideways_suppression_regime_updated",
                symbol=symbol,
                regime=event.to_regime,
                confidence=float(event.confidence),
            )

    async def _on_market_data(self, event: Event) -> None:
        """Append the latest price to the per-symbol rolling window."""
        if not isinstance(event, MarketDataEvent):
            return
        symbol = event.symbol
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque(maxlen=self._window_size)
        self._price_windows[symbol].append(event.price)

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Deny BUY orders when regime is SIDEWAYS and price range is below threshold.

        SELL orders are always approved — exits must never be blocked.
        BULL/BEAR regimes are always approved — directional momentum makes TP reachable.
        Cold start (insufficient price data) approves to avoid blocking early trades.
        Strategies in exempt_strategies bypass this rule — range trading strategies
        are designed to profit in SIDEWAYS markets and must not be suppressed.
        """
        # Exempt strategies bypass suppression entirely — range_trading is designed
        # to operate in SIDEWAYS markets and must not be blocked by this rule.
        if signal.strategy_id and signal.strategy_id in self._exempt_strategies:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=f"Strategy {signal.strategy_id} exempt from sideways suppression",
                risk_level=RiskLevel.LOW,
            )

        # Never block exits — allows positions to close even in sideways+low-vol
        if order.side == Side.SELL:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="SELL order: exits always allowed by SidewaysSuppressionRule",
                risk_level=RiskLevel.LOW,
            )

        symbol = order.symbol
        regime_info = self._regimes.get(symbol)

        # No regime data yet — approve (cold start or unknown symbol)
        if regime_info is None:
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason="No regime data — trading allowed (cold start)",
                risk_level=RiskLevel.LOW,
            )

        regime, confidence = regime_info

        # Only suppress in SIDEWAYS regime; BULL/BEAR/VOLATILE are directional
        if regime != "SIDEWAYS":
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=f"Regime {regime} does not trigger SIDEWAYS suppression",
                risk_level=RiskLevel.LOW,
            )

        # SIDEWAYS detected — check if there's enough volatility to reach take-profit
        window = self._price_windows.get(symbol)

        # Cold start: insufficient price data — approve
        if window is None or len(window) < self._window_size:
            current = len(window) if window is not None else 0
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=(
                    f"SIDEWAYS but insufficient data for {symbol}: warming up "
                    f"({current}/{self._window_size} ticks)"
                ),
                risk_level=RiskLevel.LOW,
            )

        # Calculate price range percentage
        price_min = min(window)
        price_max = max(window)
        range_pct = (price_max - price_min) / price_min * Decimal("100")

        if range_pct < self._min_range_pct:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"SIDEWAYS suppression: {symbol} in SIDEWAYS regime with range "
                    f"{float(range_pct):.4f}% < threshold {float(self._min_range_pct):.4f}% "
                    f"(take-profit unreachable in current conditions)"
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"SIDEWAYS regime but sufficient range: {symbol} range "
                f"{float(range_pct):.4f}% >= {float(self._min_range_pct):.4f}%"
            ),
            risk_level=RiskLevel.LOW,
        )


class MacroVolatilityGateRule(RiskRule):
    """
    Deny orders when session-level (macro) price range is too small to cover
    round-trip commissions. Identical logic to VolatilityGateRule but uses a
    much larger default window (~5 hours) to catch session-level flatness that
    the short-window gate misses.

    @decision DEC-VOL-004
    @title Macro-window volatility gate for session-level flatness detection
    @status accepted
    @rationale The 5-min VolatilityGateRule can be fooled by local noise in a
    globally flat session — a small local price swing can pass the short gate
    even though the overall session is ranging <0.5%. Session 5 showed all 17
    trades were losers in a market moving <0.5% intraday. The macro gate uses
    an 18000-tick window (~5h at 1 tick/sec) to detect session-level flatness
    that the 5-min window cannot catch. Both gates must approve for a trade to
    proceed: the short gate blocks local flatness, the macro gate blocks global
    flatness.
    """

    def __init__(
        self,
        min_range_pct: Decimal,
        window_size: int,
        bus: "EventBus",
    ) -> None:
        """
        Initialize macro volatility gate rule.

        Args:
            min_range_pct: Minimum price range percentage required to allow trading.
                           Orders denied when (max-min)/min*100 < min_range_pct
                           over the macro window.
            window_size: Number of recent price ticks (default ~18000 = 5 hours).
            bus: Event bus to subscribe to MARKET_DATA events.
        """
        super().__init__("macro_volatility_gate")
        self._min_range_pct = min_range_pct
        self._window_size = window_size
        # Per-symbol rolling price windows
        self._price_windows: dict[Symbol, deque[Price]] = {}

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="macro_volatility_gate_rule",
        )

        self._log.info(
            "macro_volatility_gate_initialized",
            min_range_pct=float(min_range_pct),
            window_size=window_size,
        )

    async def _on_market_data(self, event: Event) -> None:
        """Append the latest price to the per-symbol rolling window."""
        if not isinstance(event, MarketDataEvent):
            return
        symbol = event.symbol
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque(maxlen=self._window_size)
        self._price_windows[symbol].append(event.price)

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Deny the order if the macro price range for the symbol is below threshold.

        Returns APPROVE during cold start (window not yet full) to avoid blocking
        early trades when the system has just started.
        """
        symbol = order.symbol
        window = self._price_windows.get(symbol)

        # Cold start: insufficient data — allow trading
        if window is None or len(window) < self._window_size:
            current = len(window) if window is not None else 0
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=(
                    f"Macro gate insufficient data for {symbol}: warming up "
                    f"({current}/{self._window_size} ticks)"
                ),
                risk_level=RiskLevel.LOW,
            )

        # Calculate price range as a percentage of the minimum price
        price_min = min(window)
        price_max = max(window)
        range_pct = (price_max - price_min) / price_min * Decimal("100")

        if range_pct < self._min_range_pct:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"Macro volatility gate: {symbol} session range {float(range_pct):.4f}% "
                    f"< threshold {float(self._min_range_pct):.4f}% "
                    f"(session-level flatness — take-profit unreachable)"
                ),
                risk_level=RiskLevel.LOW,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"Macro volatility sufficient: {symbol} session range {float(range_pct):.4f}% "
                f">= {float(self._min_range_pct):.4f}%"
            ),
            risk_level=RiskLevel.LOW,
        )


class CommissionGateRule(RiskRule):
    """
    Deny orders when expected profit cannot cover round-trip commission.

    @decision DEC-COMMISSION-001
    @title Commission-aware minimum trade viability gate
    @status accepted
    @rationale Session 17 showed trades too small to overcome 0.32% round-trip
    commission. Expected profit = position_value * recent_range_pct. Commission
    cost = position_value * commission_pct * 2 (round-trip). If range_pct
    < round_trip_commission * min_ratio, the trade is denied.

    Unlike VolatilityGateRule (which uses a static config threshold), this rule
    computes the threshold dynamically from commission_percent so it stays
    self-calibrating if fee rates change. The two rules are complementary:
    VolatilityGateRule guards against absolute flatness; CommissionGateRule
    guards against commission-relative unprofitability.

    @decision DEC-VOL-002
    @title Per-symbol rolling price window via MARKET_DATA event bus subscription
    @status accepted
    @rationale Reuses the same per-symbol deque pattern established by
    VolatilityGateRule — self-subscribes in __init__, maintains per-symbol dict
    of deque(maxlen=window_size). Decoupled from regime detector and other gates.
    """

    def __init__(
        self,
        commission_percent: Decimal,
        min_profit_to_commission_ratio: Decimal,
        window_size: int,
        bus: "EventBus",
    ) -> None:
        """
        Initialize commission gate rule.

        Args:
            commission_percent: One-way commission rate as a percentage
                (e.g. Decimal("0.16") for Kraken's 0.16% maker fee).
                Round-trip cost = commission_percent * 2.
            min_profit_to_commission_ratio: Required multiple of round-trip
                commission that the recent price range must exceed before a
                trade is allowed. Decimal("2.0") means the range must be at
                least 2x the round-trip commission cost.
            window_size: Number of recent price ticks to consider per symbol.
                The window is a rolling deque — old prices fall off automatically.
            bus: Event bus to subscribe to MARKET_DATA events.
        """
        super().__init__("commission_gate")
        self._commission_percent = commission_percent
        self._min_ratio = min_profit_to_commission_ratio
        self._window_size = window_size
        # Compute threshold once at construction — invariant for the session.
        # threshold = commission_percent * 2 (round-trip) * min_ratio
        self._threshold = commission_percent * Decimal("2") * min_profit_to_commission_ratio
        # Per-symbol rolling price windows. deque(maxlen=N) auto-evicts oldest entries.
        self._price_windows: dict[Symbol, deque[Price]] = {}

        bus.subscribe(
            EventType.MARKET_DATA,
            self._on_market_data,
            subscriber_name="commission_gate_rule",
        )

        self._log.info(
            "commission_gate_initialized",
            commission_percent=float(commission_percent),
            min_profit_to_commission_ratio=float(min_profit_to_commission_ratio),
            threshold_pct=float(self._threshold),
            window_size=window_size,
        )

    async def _on_market_data(self, event: Event) -> None:
        """Append the latest price to the per-symbol rolling window."""
        if not isinstance(event, MarketDataEvent):
            return
        symbol = event.symbol
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque(maxlen=self._window_size)
        self._price_windows[symbol].append(event.price)

    def evaluate(
        self,
        signal: SignalEvent,
        order: OrderEvent,
        portfolio: PortfolioTracker,
    ) -> RuleResult:
        """
        Deny the order if the recent price range for the symbol is below
        the minimum commission-coverage threshold.

        Returns APPROVE during cold start (window not yet full) to avoid
        blocking early trades when the system has just started.
        """
        symbol = order.symbol
        window = self._price_windows.get(symbol)

        # Cold start: insufficient data to evaluate — allow trading
        if window is None or len(window) < self._window_size:
            current = len(window) if window is not None else 0
            return RuleResult(
                decision=RuleDecision.APPROVE,
                reason=(
                    f"Insufficient data for {symbol}: warming up "
                    f"({current}/{self._window_size} ticks)"
                ),
                risk_level=RiskLevel.LOW,
            )

        # Calculate price range as a percentage of the minimum price
        price_min = min(window)
        price_max = max(window)
        range_pct = (price_max - price_min) / price_min * Decimal("100")

        if range_pct < self._threshold:
            return RuleResult(
                decision=RuleDecision.DENY,
                reason=(
                    f"commission_gate: {symbol} range {float(range_pct):.4f}% "
                    f"< {float(self._threshold):.4f}% threshold "
                    f"(commission={float(self._commission_percent):.2f}% "
                    f"x2 x{float(self._min_ratio):.1f} ratio)"
                ),
                risk_level=RiskLevel.LOW,
            )

        return RuleResult(
            decision=RuleDecision.APPROVE,
            reason=(
                f"Commission gate passed: {symbol} range {float(range_pct):.4f}% "
                f">= {float(self._threshold):.4f}% threshold"
            ),
            risk_level=RiskLevel.LOW,
        )
