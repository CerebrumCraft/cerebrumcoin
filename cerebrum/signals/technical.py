"""
Technical analysis signal generators using pandas-ta.

Implements RSI, MACD, Bollinger Bands, and VWAP indicators with normalized
signal output for weighted combination.

@decision DEC-SIGNAL-003
@title Normalized signal strength [-1.0, 1.0] convention
@status accepted
@rationale Uniform scale enables weighted combination in aggregator. -1=strong sell,
0=neutral, +1=strong buy. Each indicator maps its values to this scale using
domain-specific thresholds (e.g., RSI < 30 = oversold/buy, > 70 = overbought/sell).

@decision DEC-SIGNAL-004
@title pandas-ta for technical indicator calculations
@status accepted
@rationale Pure Python library, no C compilation needed. Adequate coverage for
common indicators. Alternative (ta-lib) requires compilation and OS dependencies.
"""

from decimal import Decimal

import pandas as pd
import pandas_ta as ta
import structlog

from cerebrum.core.bus import EventBus
from cerebrum.core.events import MarketDataEvent, SignalEvent
from cerebrum.core.types import SignalAction, SignalType, Symbol

from .base import SignalGenerator
from .candles import Candle, CandleAggregator

logger = structlog.get_logger()


class RSISignal(SignalGenerator):
    """
    Relative Strength Index (RSI) signal generator.
    
    RSI measures momentum. Values < 30 indicate oversold (buy signal),
    values > 70 indicate overbought (sell signal).
    """
    
    def __init__(
        self,
        bus: EventBus,
        candle_agg: CandleAggregator,
        period: int = 14,
        oversold: int = 30,
        overbought: int = 70,
    ) -> None:
        """
        Initialize RSI signal generator.
        
        Args:
            bus: Event bus
            candle_agg: Candle aggregator for OHLCV data
            period: RSI calculation period
            oversold: Oversold threshold (buy signal)
            overbought: Overbought threshold (sell signal)
        """
        super().__init__(bus, SignalType.TECHNICAL, window_size=period + 50, name="RSI")
        self._candle_agg = candle_agg
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
        self._log = logger.bind(component="signal_rsi")
    
    def _get_min_periods(self) -> int:
        return self._period + 1
    
    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """Generate RSI-based signal."""
        candles = self._candle_agg.get_candles(symbol, count=self._period + 20)
        
        if len(candles) < self._period + 1:
            return None
        
        # Convert candles to DataFrame
        df = pd.DataFrame([
            {
                "close": float(c.close),
                "timestamp": c.timestamp,
            }
            for c in candles
        ])
        
        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=self._period)
        
        latest_rsi = df["rsi"].iloc[-1]
        
        if pd.isna(latest_rsi):
            return None
        
        # Map RSI to signal strength [-1.0, 1.0]
        # RSI < 30: strong buy (positive strength)
        # RSI > 70: strong sell (negative strength)
        # RSI 30-70: neutral (near zero strength)
        
        if latest_rsi <= self._oversold:
            # Oversold: buy signal
            # Map [0, 30] -> [1.0, 0.3]
            strength = Decimal(str(1.0 - (latest_rsi / self._oversold) * 0.7))
            action = SignalAction.BUY
        elif latest_rsi >= self._overbought:
            # Overbought: sell signal
            # Map [70, 100] -> [-0.3, -1.0]
            strength = Decimal(str(-0.3 - ((latest_rsi - self._overbought) / (100 - self._overbought)) * 0.7))
            action = SignalAction.SELL
        else:
            # Neutral zone
            action = SignalAction.HOLD
            # Map [30, 70] -> [-0.2, 0.2] with center at 50
            normalized = (latest_rsi - 50) / 50  # -0.4 to 0.4
            strength = Decimal(str(normalized * 0.5))
        
        confidence = Decimal("0.7")  # RSI is moderately reliable
        
        return self._create_signal(
            symbol=symbol,
            action=action,
            strength=abs(strength),
            confidence=confidence,
            timestamp=data[-1].timestamp,
            reason=f"RSI={latest_rsi:.1f}",
        )


class MACDSignal(SignalGenerator):
    """
    Moving Average Convergence Divergence (MACD) signal generator.
    
    MACD shows trend changes. Bullish crossover (MACD > signal) = buy,
    bearish crossover (MACD < signal) = sell.
    """
    
    def __init__(
        self,
        bus: EventBus,
        candle_agg: CandleAggregator,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        """
        Initialize MACD signal generator.
        
        Args:
            bus: Event bus
            candle_agg: Candle aggregator
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        super().__init__(bus, SignalType.TECHNICAL, window_size=slow + signal + 50, name="MACD")
        self._candle_agg = candle_agg
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._log = logger.bind(component="signal_macd")
    
    def _get_min_periods(self) -> int:
        return self._slow + self._signal + 1
    
    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """Generate MACD-based signal."""
        candles = self._candle_agg.get_candles(symbol, count=self._slow + self._signal + 20)
        
        if len(candles) < self._slow + self._signal + 1:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{"close": float(c.close)} for c in candles])
        
        # Calculate MACD
        macd_result = ta.macd(df["close"], fast=self._fast, slow=self._slow, signal=self._signal)
        
        if macd_result is None or macd_result.empty:
            return None
        
        macd_line = macd_result[f"MACD_{self._fast}_{self._slow}_{self._signal}"].iloc[-1]
        signal_line = macd_result[f"MACDs_{self._fast}_{self._slow}_{self._signal}"].iloc[-1]
        histogram = macd_result[f"MACDh_{self._fast}_{self._slow}_{self._signal}"].iloc[-1]
        
        if pd.isna(macd_line) or pd.isna(signal_line):
            return None
        
        # Signal based on MACD crossover and histogram
        diff = macd_line - signal_line
        
        if diff > 0:
            # Bullish: MACD above signal line
            action = SignalAction.BUY
            # Strength based on histogram magnitude
            strength = Decimal(str(min(abs(histogram) * 10, 1.0)))
        elif diff < 0:
            # Bearish: MACD below signal line
            action = SignalAction.SELL
            strength = Decimal(str(min(abs(histogram) * 10, 1.0)))
        else:
            action = SignalAction.HOLD
            strength = Decimal("0.0")
        
        confidence = Decimal("0.75")  # MACD is fairly reliable for trends
        
        return self._create_signal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            timestamp=data[-1].timestamp,
            reason=f"MACD={macd_line:.2f}, Signal={signal_line:.2f}",
        )


class BollingerBandsSignal(SignalGenerator):
    """
    Bollinger Bands signal generator.
    
    Bollinger Bands show volatility and potential reversals.
    Price near lower band = oversold/buy, near upper band = overbought/sell.
    """
    
    def __init__(
        self,
        bus: EventBus,
        candle_agg: CandleAggregator,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> None:
        """
        Initialize Bollinger Bands signal generator.
        
        Args:
            bus: Event bus
            candle_agg: Candle aggregator
            period: Moving average period
            std_dev: Standard deviation multiplier
        """
        super().__init__(bus, SignalType.TECHNICAL, window_size=period + 50, name="BollingerBands")
        self._candle_agg = candle_agg
        self._period = period
        self._std_dev = std_dev
        self._log = logger.bind(component="signal_bb")
    
    def _get_min_periods(self) -> int:
        return self._period + 1
    
    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """Generate Bollinger Bands signal."""
        candles = self._candle_agg.get_candles(symbol, count=self._period + 20)
        
        if len(candles) < self._period + 1:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([{"close": float(c.close)} for c in candles])
        
        # Calculate Bollinger Bands
        bb = ta.bbands(df["close"], length=self._period, std=self._std_dev)

        if bb is None or bb.empty:
            return None

        # Detect column names dynamically (pandas-ta may format std differently)
        bb_cols = bb.columns.tolist()
        lower_col = [c for c in bb_cols if c.startswith("BBL_")][0]
        middle_col = [c for c in bb_cols if c.startswith("BBM_")][0]
        upper_col = [c for c in bb_cols if c.startswith("BBU_")][0]

        lower = bb[lower_col].iloc[-1]
        middle = bb[middle_col].iloc[-1]
        upper = bb[upper_col].iloc[-1]
        current_price = df["close"].iloc[-1]
        
        if pd.isna(lower) or pd.isna(upper):
            return None
        
        # Calculate position within bands [0, 1]
        band_width = upper - lower
        if band_width == 0:
            return None
        
        position = (current_price - lower) / band_width
        
        # Map position to signal
        # position < 0.2: near lower band (oversold/buy)
        # position > 0.8: near upper band (overbought/sell)
        
        if position < 0.2:
            action = SignalAction.BUY
            strength = Decimal(str((0.2 - position) * 5))  # [0, 1]
        elif position > 0.8:
            action = SignalAction.SELL
            strength = Decimal(str((position - 0.8) * 5))  # [0, 1]
        else:
            action = SignalAction.HOLD
            strength = Decimal(str(abs(0.5 - position)))  # Low strength in middle
        
        confidence = Decimal("0.65")  # BB is moderately reliable
        
        return self._create_signal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            timestamp=data[-1].timestamp,
            reason=f"Price={current_price:.2f}, BB=[{lower:.2f}, {upper:.2f}]",
        )


class VWAPSignal(SignalGenerator):
    """
    Volume Weighted Average Price (VWAP) signal generator.
    
    VWAP shows the average price weighted by volume. Price above VWAP = bullish,
    below VWAP = bearish.
    """
    
    def __init__(
        self,
        bus: EventBus,
        candle_agg: CandleAggregator,
        period: int = 20,
    ) -> None:
        """
        Initialize VWAP signal generator.
        
        Args:
            bus: Event bus
            candle_agg: Candle aggregator
            period: VWAP calculation period
        """
        super().__init__(bus, SignalType.TECHNICAL, window_size=period + 50, name="VWAP")
        self._candle_agg = candle_agg
        self._period = period
        self._log = logger.bind(component="signal_vwap")
    
    def _get_min_periods(self) -> int:
        return self._period
    
    def _generate_signal(
        self,
        symbol: Symbol,
        data: list[MarketDataEvent],
    ) -> SignalEvent | None:
        """Generate VWAP-based signal."""
        candles = self._candle_agg.get_candles(symbol, count=self._period + 20)

        if len(candles) < self._period:
            return None

        # Convert to DataFrame with timestamp as index
        df = pd.DataFrame([
            {
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
                "timestamp": pd.Timestamp(c.timestamp, unit="s"),
            }
            for c in candles
        ])

        # Set DatetimeIndex (required by pandas-ta vwap)
        df.set_index("timestamp", inplace=True)

        # Calculate VWAP
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        
        if vwap is None or vwap.empty:
            return None
        
        vwap_value = vwap.iloc[-1]
        current_price = df["close"].iloc[-1]
        
        if pd.isna(vwap_value):
            return None
        
        # Calculate distance from VWAP as percentage
        distance_pct = ((current_price - vwap_value) / vwap_value) * 100
        
        # Map distance to signal
        # distance > 2%: price well above VWAP (bullish)
        # distance < -2%: price well below VWAP (bearish)
        
        if distance_pct > 0:
            action = SignalAction.BUY
            strength = Decimal(str(min(distance_pct / 5, 1.0)))  # Cap at 5%
        elif distance_pct < 0:
            action = SignalAction.SELL
            strength = Decimal(str(min(abs(distance_pct) / 5, 1.0)))
        else:
            action = SignalAction.HOLD
            strength = Decimal("0.0")
        
        confidence = Decimal("0.70")  # VWAP is fairly reliable
        
        return self._create_signal(
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=confidence,
            timestamp=data[-1].timestamp,
            reason=f"Price={current_price:.2f}, VWAP={vwap_value:.2f} ({distance_pct:+.1f}%)",
        )
