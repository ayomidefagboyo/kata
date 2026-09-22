"""
Utility functions for Multi-Level Learning Service

This module contains helper functions and implementations for:
- Market regime detection
- Pattern recognition algorithms
- Performance metrics calculations
- Statistical analysis functions
"""

import numpy as np
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection system."""

    def __init__(self):
        self.price_window = 50
        self.volume_window = 20
        self.volatility_window = 30

    def detect_regime(self, price_data: List[float], volume_data: List[float] = None) -> Dict[str, Any]:
        """
        Detect current market regime based on price and volume data.

        Returns:
            Dict with regime_type, strength, confidence, and supporting_factors
        """
        try:
            if len(price_data) < self.price_window:
                return self._default_regime()

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(price_data)

            # Calculate volatility regime
            volatility_regime = self._calculate_volatility_regime(price_data)

            # Calculate momentum
            momentum = self._calculate_momentum(price_data)

            # Volume analysis (if available)
            volume_profile = "normal"
            if volume_data and len(volume_data) >= self.volume_window:
                volume_profile = self._analyze_volume_profile(volume_data)

            # Regime classification logic
            regime_type = self._classify_regime(trend_strength, volatility_regime, momentum, volume_profile)

            # Calculate regime strength and confidence
            strength = self._calculate_regime_strength(trend_strength, volatility_regime, momentum)
            confidence = self._calculate_regime_confidence(price_data, trend_strength, volatility_regime)

            # Supporting factors
            supporting_factors = self._identify_supporting_factors(
                trend_strength, volatility_regime, momentum, volume_profile
            )

            return {
                'regime_type': regime_type,
                'strength': strength,
                'confidence': confidence,
                'supporting_factors': supporting_factors,
                'trend_strength': trend_strength,
                'volatility_regime': volatility_regime,
                'momentum': momentum,
                'volume_profile': volume_profile
            }

        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return self._default_regime()

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using multiple indicators."""
        if len(prices) < 20:
            return 0.0

        # Simple moving averages
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20

        # Trend alignment
        current_price = prices[-1]

        # Bullish alignment: price > sma_10 > sma_20 > sma_50
        if current_price > sma_10 > sma_20 > sma_50:
            return 1.0  # Strong bullish trend
        elif current_price > sma_10 > sma_20:
            return 0.7  # Moderate bullish trend
        elif current_price > sma_10:
            return 0.4  # Weak bullish trend

        # Bearish alignment: price < sma_10 < sma_20 < sma_50
        elif current_price < sma_10 < sma_20 < sma_50:
            return -1.0  # Strong bearish trend
        elif current_price < sma_10 < sma_20:
            return -0.7  # Moderate bearish trend
        elif current_price < sma_10:
            return -0.4  # Weak bearish trend

        return 0.0  # No clear trend

    def _calculate_volatility_regime(self, prices: List[float]) -> str:
        """Calculate volatility regime classification."""
        if len(prices) < self.volatility_window:
            return "normal"

        # Calculate rolling volatility
        returns = np.diff(prices) / prices[:-1]
        current_vol = np.std(returns[-self.volatility_window:])
        baseline_vol = np.std(returns[:-self.volatility_window]) if len(returns) > self.volatility_window else current_vol

        vol_ratio = current_vol / max(baseline_vol, 0.001)

        if vol_ratio > 2.0:
            return "extreme"
        elif vol_ratio > 1.5:
            return "high"
        elif vol_ratio > 1.2:
            return "elevated"
        elif vol_ratio < 0.7:
            return "low"
        else:
            return "normal"

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum indicator."""
        if len(prices) < 20:
            return 0.0

        # Rate of change over different periods
        roc_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        roc_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        roc_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0

        # Weighted momentum
        momentum = (roc_5 * 0.5 + roc_10 * 0.3 + roc_20 * 0.2)

        # Normalize to -1 to 1 scale
        return max(-1.0, min(1.0, momentum * 10))

    def _analyze_volume_profile(self, volume_data: List[float]) -> str:
        """Analyze volume profile characteristics."""
        if len(volume_data) < self.volume_window:
            return "normal"

        recent_avg = np.mean(volume_data[-5:])
        baseline_avg = np.mean(volume_data[-self.volume_window:-5])

        ratio = recent_avg / max(baseline_avg, 1)

        if ratio > 3.0:
            return "spike"
        elif ratio > 1.5:
            return "high"
        elif ratio > 1.2:
            return "elevated"
        elif ratio < 0.7:
            return "low"
        else:
            return "normal"

    def _classify_regime(self, trend_strength: float, volatility_regime: str, momentum: float, volume_profile: str) -> str:
        """Classify overall market regime."""
        # Strong trending markets
        if abs(trend_strength) > 0.7:
            if trend_strength > 0:
                if volume_profile in ["high", "spike"]:
                    return "TRENDING_BULL_STRONG"
                else:
                    return "TRENDING_BULL"
            else:
                if volume_profile in ["high", "spike"]:
                    return "TRENDING_BEAR_STRONG"
                else:
                    return "TRENDING_BEAR"

        # High volatility regimes
        if volatility_regime in ["extreme", "high"]:
            if abs(momentum) > 0.5:
                return "HIGH_VOLATILITY_TRENDING"
            else:
                return "HIGH_VOLATILITY_CHOPPY"

        # Breakout detection
        if volume_profile == "spike" and abs(momentum) > 0.3:
            return "BREAKOUT"

        # Range-bound market
        if abs(trend_strength) < 0.3 and volatility_regime in ["normal", "low"]:
            return "RANGE_BOUND"

        # Default trending classification
        if trend_strength > 0.3:
            return "TRENDING_BULL_WEAK"
        elif trend_strength < -0.3:
            return "TRENDING_BEAR_WEAK"
        else:
            return "NEUTRAL"

    def _calculate_regime_strength(self, trend_strength: float, volatility_regime: str, momentum: float) -> float:
        """Calculate regime strength (0.0 to 1.0)."""
        # Base strength from trend
        base_strength = abs(trend_strength)

        # Momentum confirmation
        momentum_confirmation = min(abs(momentum), 0.5) * 2  # Scale to 0-1

        # Volatility adjustment
        vol_multiplier = {
            "extreme": 0.8,  # High vol reduces strength reliability
            "high": 0.9,
            "elevated": 0.95,
            "normal": 1.0,
            "low": 0.9       # Low vol also reduces strength
        }.get(volatility_regime, 1.0)

        strength = (base_strength * 0.6 + momentum_confirmation * 0.4) * vol_multiplier
        return max(0.0, min(1.0, strength))

    def _calculate_regime_confidence(self, prices: List[float], trend_strength: float, volatility_regime: str) -> float:
        """Calculate confidence in regime detection."""
        if len(prices) < 20:
            return 0.5

        # Data quality factor
        data_quality = min(len(prices) / 100.0, 1.0)  # More data = higher confidence

        # Trend consistency
        trend_consistency = self._calculate_trend_consistency(prices)

        # Volatility stability
        vol_stability = 1.0 if volatility_regime == "normal" else 0.7

        confidence = (data_quality * 0.3 + trend_consistency * 0.5 + vol_stability * 0.2)
        return max(0.0, min(1.0, confidence))

    def _calculate_trend_consistency(self, prices: List[float]) -> float:
        """Calculate how consistent the trend direction is."""
        if len(prices) < 10:
            return 0.5

        # Count directional changes
        changes = np.diff(prices)
        positive_changes = len([c for c in changes if c > 0])
        negative_changes = len([c for c in changes if c < 0])

        total_changes = len(changes)
        dominant_direction = max(positive_changes, negative_changes)

        consistency = dominant_direction / total_changes if total_changes > 0 else 0.5
        return consistency

    def _identify_supporting_factors(self, trend_strength: float, volatility_regime: str, momentum: float, volume_profile: str) -> List[str]:
        """Identify factors supporting the regime classification."""
        factors = []

        if abs(trend_strength) > 0.5:
            factors.append(f"{'Strong bullish' if trend_strength > 0 else 'Strong bearish'} trend alignment")

        if abs(momentum) > 0.3:
            factors.append(f"{'Positive' if momentum > 0 else 'Negative'} momentum confirmation")

        if volume_profile in ["high", "spike"]:
            factors.append("Elevated volume supporting move")

        if volatility_regime == "low":
            factors.append("Low volatility indicates stability")
        elif volatility_regime in ["high", "extreme"]:
            factors.append("High volatility indicates uncertainty")

        return factors

    def _default_regime(self) -> Dict[str, Any]:
        """Return default regime when detection fails."""
        return {
            'regime_type': 'NEUTRAL',
            'strength': 0.5,
            'confidence': 0.3,
            'supporting_factors': ['Insufficient data for regime detection'],
            'trend_strength': 0.0,
            'volatility_regime': 'normal',
            'momentum': 0.0,
            'volume_profile': 'normal'
        }


class PatternRecognitionEngine:
    """Advanced pattern recognition for trading signals."""

    def __init__(self):
        self.min_pattern_length = 5
        self.similarity_threshold = 0.7

    def extract_patterns(self, signals: List[Dict]) -> List[Dict]:
        """Extract trading patterns from historical signals."""
        patterns = []

        try:
            # Group signals by similar market conditions
            grouped_signals = self._group_by_market_conditions(signals)

            for group_name, group_signals in grouped_signals.items():
                if len(group_signals) < self.min_pattern_length:
                    continue

                pattern = self._analyze_pattern_group(group_name, group_signals)
                if pattern:
                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"❌ Pattern extraction failed: {e}")

        return patterns

    def _group_by_market_conditions(self, signals: List[Dict]) -> Dict[str, List[Dict]]:
        """Group signals by similar market conditions."""
        groups = defaultdict(list)

        for signal in signals:
            # Create condition key based on signal characteristics
            condition_key = self._create_condition_key(signal)
            groups[condition_key].append(signal)

        return dict(groups)

    def _create_condition_key(self, signal: Dict) -> str:
        """Create a key representing market conditions for grouping."""
        try:
            # Extract key characteristics
            direction = signal.get('direction', 'UNKNOWN')
            confidence_range = self._discretize_confidence(signal.get('confidence', 0.5))
            leverage_range = self._discretize_leverage(signal.get('leverage', 1))

            # Time-based features
            timestamp = signal.get('analysis_timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                hour = dt.hour
                weekday = dt.weekday()
                time_category = self._categorize_time(hour, weekday)
            else:
                time_category = "unknown"

            return f"{direction}_{confidence_range}_{leverage_range}_{time_category}"

        except Exception as e:
            logger.error(f"❌ Failed to create condition key: {e}")
            return "unknown"

    def _discretize_confidence(self, confidence: float) -> str:
        """Discretize confidence into ranges."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.65:
            return "medium_high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def _discretize_leverage(self, leverage: int) -> str:
        """Discretize leverage into ranges."""
        if leverage >= 10:
            return "very_high"
        elif leverage >= 5:
            return "high"
        elif leverage >= 3:
            return "medium"
        else:
            return "low"

    def _categorize_time(self, hour: int, weekday: int) -> str:
        """Categorize time of day and week."""
        # Weekend
        if weekday in [5, 6]:
            return "weekend"

        # Trading session categorization
        if 0 <= hour < 6:
            return "asian_session"
        elif 6 <= hour < 14:
            return "european_session"
        elif 14 <= hour < 22:
            return "us_session"
        else:
            return "asian_session"

    def _analyze_pattern_group(self, group_name: str, signals: List[Dict]) -> Optional[Dict]:
        """Analyze a group of signals to extract pattern characteristics."""
        try:
            if len(signals) < 3:
                return None

            # Calculate performance metrics
            outcomes = [s.get('pnl_percentage', 0) for s in signals if s.get('pnl_percentage') is not None]

            if not outcomes:
                return None

            win_rate = len([o for o in outcomes if o > 0]) / len(outcomes)
            avg_pnl = statistics.mean(outcomes)

            # Risk metrics
            volatility = statistics.stdev(outcomes) if len(outcomes) > 1 else 0
            sharpe_ratio = avg_pnl / max(volatility, 0.01) if volatility > 0 else 0
            max_drawdown = min(outcomes)

            # Extract common features
            avg_confidence = statistics.mean([s.get('confidence', 0.5) for s in signals])
            avg_leverage = statistics.mean([s.get('leverage', 1) for s in signals])

            # Time analysis
            execution_times = []
            for signal in signals:
                if signal.get('created_at') and signal.get('updated_at'):
                    # Calculate time to completion (simplified)
                    execution_times.append(30)  # Placeholder

            avg_execution_time = statistics.mean(execution_times) if execution_times else 30

            return {
                'pattern_name': group_name,
                'sample_size': len(signals),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_confidence': avg_confidence,
                'avg_leverage': avg_leverage,
                'avg_execution_time': avg_execution_time,
                'risk_adjusted_return': avg_pnl / max(abs(max_drawdown), 0.01),
                'last_updated': datetime.now()
            }

        except Exception as e:
            logger.error(f"❌ Pattern analysis failed for {group_name}: {e}")
            return None


# Performance calculation utilities
class PerformanceCalculator:
    """Calculate advanced performance metrics."""

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not returns or len(returns) < 2:
                return 0.0

            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            if std_return == 0:
                return 0.0

            return (mean_return - risk_free_rate) / std_return

        except Exception:
            return 0.0

    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not returns:
                return 0.0

            cumulative = 0.0
            max_cumulative = 0.0
            max_drawdown = 0.0

            for ret in returns:
                cumulative += ret
                max_cumulative = max(max_cumulative, cumulative)
                drawdown = max_cumulative - cumulative
                max_drawdown = max(max_drawdown, drawdown)

            return max_drawdown

        except Exception:
            return 0.0

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            if not returns:
                return 0.0

            mean_return = statistics.mean(returns)
            negative_returns = [r for r in returns if r < target_return]

            if not negative_returns:
                return float('inf') if mean_return > target_return else 0.0

            downside_deviation = statistics.stdev(negative_returns)

            if downside_deviation == 0:
                return 0.0

            return (mean_return - target_return) / downside_deviation

        except Exception:
            return 0.0