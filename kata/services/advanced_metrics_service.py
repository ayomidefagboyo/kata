"""
Advanced Financial Metrics Service

Implements sophisticated financial indicators and ratios for cryptocurrency analysis:
- Sharpe Ratio
- Sortino Ratio  
- Liquidity Ratios
- Market Efficiency Metrics
- Risk-Adjusted Returns
- On-chain Metrics
"""

import asyncio
import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class AdvancedMetrics:
    """Advanced financial metrics for token analysis."""
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Liquidity Metrics
    bid_ask_spread_ratio: float
    market_impact_ratio: float
    turnover_ratio: float
    amihud_illiquidity: float
    
    # Market Efficiency
    hurst_exponent: float
    autocorrelation: float
    variance_ratio: float
    
    # Volatility Metrics
    realized_volatility: float
    garch_volatility: float
    volatility_of_volatility: float
    
    # On-chain Metrics (when available)
    network_value_to_transactions: Optional[float] = None
    mvrv_ratio: Optional[float] = None
    active_addresses_ratio: Optional[float] = None
    
    # Composite Scores
    liquidity_score: float = 0.0
    technical_score: float = 0.0
    risk_score: float = 0.0
    efficiency_score: float = 0.0


class AdvancedMetricsService:
    """Service for calculating advanced financial metrics."""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% annual risk-free rate (US Treasury)
        logger.info("AdvancedMetricsService initialized")
    
    async def calculate_advanced_metrics(
        self, 
        price_data: List[float], 
        volume_data: List[float],
        market_data: Dict[str, Any],
        timeframe: str = "1h"
    ) -> AdvancedMetrics:
        """
        Calculate comprehensive advanced metrics for a token.
        
        Args:
            price_data: List of historical prices
            volume_data: List of historical volumes
            market_data: Current market data (spread, market cap, etc.)
            timeframe: Data timeframe (1h, 1d, etc.)
            
        Returns:
            AdvancedMetrics object with all calculated metrics
        """
        try:
            if len(price_data) < 30:
                logger.warning("Insufficient price data for advanced metrics")
                return self._fallback_metrics()
            
            # Calculate returns
            returns = self._calculate_returns(price_data)
            
            # Risk-adjusted return metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns, timeframe)
            sortino_ratio = self._calculate_sortino_ratio(returns, timeframe)
            calmar_ratio = self._calculate_calmar_ratio(returns, price_data)
            
            # Liquidity metrics
            bid_ask_spread = market_data.get('bid_ask_spread', 0.001)
            
            # Handle different price_data types (list of floats vs CoinPrice object)
            if isinstance(price_data, list) and len(price_data) > 0:
                current_price = price_data[-1]
            elif hasattr(price_data, 'current_price'):
                current_price = price_data.current_price
            elif price_data:
                current_price = float(price_data)  # Try to convert to float
            else:
                current_price = 1
                
            bid_ask_spread_ratio = bid_ask_spread / current_price if current_price > 0 else 0
            
            market_impact_ratio = self._calculate_market_impact_ratio(price_data, volume_data)
            turnover_ratio = self._calculate_turnover_ratio(volume_data, market_data)
            amihud_illiquidity = self._calculate_amihud_illiquidity(returns, volume_data)
            
            # Market efficiency metrics
            hurst_exponent = self._calculate_hurst_exponent(price_data)
            autocorrelation = self._calculate_autocorrelation(returns)
            variance_ratio = self._calculate_variance_ratio(returns)
            
            # Volatility metrics
            realized_volatility = self._calculate_realized_volatility(returns, timeframe)
            garch_volatility = self._calculate_garch_volatility(returns)
            volatility_of_volatility = self._calculate_volatility_of_volatility(returns)
            
            # On-chain metrics (if available)
            nvt_ratio = self._calculate_nvt_ratio(market_data)
            mvrv_ratio = market_data.get('mvrv_ratio')
            active_addresses_ratio = market_data.get('active_addresses_ratio')
            
            # Calculate composite scores
            liquidity_score = self._calculate_liquidity_score(
                bid_ask_spread_ratio, market_impact_ratio, turnover_ratio, amihud_illiquidity
            )
            
            technical_score = self._calculate_technical_score(
                sharpe_ratio, sortino_ratio, realized_volatility, hurst_exponent
            )
            
            risk_score = self._calculate_risk_score(
                realized_volatility, garch_volatility, calmar_ratio
            )
            
            efficiency_score = self._calculate_efficiency_score(
                hurst_exponent, autocorrelation, variance_ratio
            )
            
            return AdvancedMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                bid_ask_spread_ratio=bid_ask_spread_ratio,
                market_impact_ratio=market_impact_ratio,
                turnover_ratio=turnover_ratio,
                amihud_illiquidity=amihud_illiquidity,
                hurst_exponent=hurst_exponent,
                autocorrelation=autocorrelation,
                variance_ratio=variance_ratio,
                realized_volatility=realized_volatility,
                garch_volatility=garch_volatility,
                volatility_of_volatility=volatility_of_volatility,
                network_value_to_transactions=nvt_ratio,
                mvrv_ratio=mvrv_ratio,
                active_addresses_ratio=active_addresses_ratio,
                liquidity_score=liquidity_score,
                technical_score=technical_score,
                risk_score=risk_score,
                efficiency_score=efficiency_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return self._fallback_metrics()
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate log returns from price data."""
        if len(prices) < 2:
            return [0.0]
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i-1]))
            else:
                returns.append(0.0)
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], timeframe: str) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        if len(returns) < 2:
            return 0.0
        
        # Annualize returns and volatility based on timeframe
        periods_per_year = self._get_periods_per_year(timeframe)
        
        mean_return = np.mean(returns) * periods_per_year
        std_return = np.std(returns) * math.sqrt(periods_per_year)
        
        if std_return == 0:
            return 0.0
        
        # Risk-free rate already annualized
        excess_return = mean_return - self.risk_free_rate
        return excess_return / std_return
    
    def _calculate_sortino_ratio(self, returns: List[float], timeframe: str) -> float:
        """Calculate Sortino ratio (downside deviation adjusted return)."""
        if len(returns) < 2:
            return 0.0
        
        periods_per_year = self._get_periods_per_year(timeframe)
        
        mean_return = np.mean(returns) * periods_per_year
        
        # Calculate downside deviation
        downside_returns = [r for r in returns if r < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns) * math.sqrt(periods_per_year)
        
        if downside_std == 0:
            return float('inf')
        
        excess_return = mean_return - self.risk_free_rate
        return excess_return / downside_std
    
    def _calculate_calmar_ratio(self, returns: List[float], prices: List[float]) -> float:
        """Calculate Calmar ratio (return over maximum drawdown)."""
        if len(returns) < 2 or len(prices) < 2:
            return 0.0
        
        # Calculate annual return
        total_return = (prices[-1] / prices[0]) - 1
        periods = len(prices)
        annual_return = (1 + total_return) ** (252 / periods) - 1
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(prices)
        
        if max_drawdown == 0:
            return float('inf')
        
        return annual_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown from price series."""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            peak = max(peak, price)
            drawdown = (peak - price) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_market_impact_ratio(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate market impact ratio (price change per unit volume)."""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                price_changes.append(abs(prices[i] - prices[i-1]) / prices[i-1])
        
        if len(price_changes) == 0 or len(volumes) == 0:
            return 0.0
        
        avg_price_change = np.mean(price_changes)
        avg_volume = np.mean(volumes)
        
        if avg_volume == 0:
            return float('inf')
        
        return avg_price_change / (avg_volume / 1000000)  # Normalize by millions
    
    def _calculate_turnover_ratio(self, volumes: List[float], market_data: Dict[str, Any]) -> float:
        """Calculate turnover ratio (volume relative to market cap)."""
        if len(volumes) == 0:
            return 0.0
        
        avg_volume = np.mean(volumes)
        market_cap = market_data.get('market_cap', 0)
        
        if market_cap == 0:
            return 0.0
        
        return avg_volume / market_cap
    
    def _calculate_amihud_illiquidity(self, returns: List[float], volumes: List[float]) -> float:
        """Calculate Amihud illiquidity measure."""
        if len(returns) != len(volumes) or len(returns) == 0:
            return 0.0
        
        illiquidity_measures = []
        for i in range(len(returns)):
            if volumes[i] > 0:
                illiquidity_measures.append(abs(returns[i]) / volumes[i])
        
        if len(illiquidity_measures) == 0:
            return 0.0
        
        return np.mean(illiquidity_measures) * 1000000  # Scale for readability
    
    def _calculate_hurst_exponent(self, prices: List[float]) -> float:
        """Calculate Hurst exponent for trend persistence."""
        if len(prices) < 20:
            return 0.5  # Random walk
        
        try:
            # Convert to numpy array
            ts = np.array(prices)
            
            # Calculate log returns
            log_returns = np.diff(np.log(ts))
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(len(log_returns)//4, 50))
            rs_values = []
            
            for lag in lags:
                # Divide series into non-overlapping sub-periods
                n_periods = len(log_returns) // lag
                if n_periods < 2:
                    continue
                
                rs_period = []
                for i in range(n_periods):
                    start_idx = i * lag
                    end_idx = start_idx + lag
                    period_returns = log_returns[start_idx:end_idx]
                    
                    # Calculate mean
                    mean_return = np.mean(period_returns)
                    
                    # Calculate cumulative deviations
                    deviations = np.cumsum(period_returns - mean_return)
                    
                    # Calculate range
                    R = np.max(deviations) - np.min(deviations)
                    
                    # Calculate standard deviation
                    S = np.std(period_returns)
                    
                    if S > 0:
                        rs_period.append(R / S)
                
                if rs_period:
                    rs_values.append(np.mean(rs_period))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression to find Hurst exponent
            log_lags = np.log(list(lags[:len(rs_values)]))
            log_rs = np.log(rs_values)
            
            # Remove any infinite or NaN values
            valid_indices = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_indices) < 3:
                return 0.5
            
            # Fit line
            coeffs = np.polyfit(log_lags[valid_indices], log_rs[valid_indices], 1)
            hurst = coeffs[0]
            
            # Constrain to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception as e:
            logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def _calculate_autocorrelation(self, returns: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation of returns."""
        if len(returns) < lag + 10:
            return 0.0
        
        try:
            returns_array = np.array(returns)
            return np.corrcoef(returns_array[:-lag], returns_array[lag:])[0, 1]
        except:
            return 0.0
    
    def _calculate_variance_ratio(self, returns: List[float]) -> float:
        """Calculate variance ratio test for random walk."""
        if len(returns) < 10:
            return 1.0
        
        try:
            returns_array = np.array(returns)
            n = len(returns_array)
            
            # Calculate variance of 1-period returns
            var_1 = np.var(returns_array)
            
            # Calculate variance of 2-period returns (ensure same length arrays)
            # Take non-overlapping 2-period returns
            if n < 4:
                return 1.0
                
            # Create non-overlapping 2-period returns
            n_pairs = n // 2
            returns_2 = []
            for i in range(0, n_pairs * 2, 2):
                if i + 1 < len(returns_array):
                    returns_2.append(returns_array[i] + returns_array[i + 1])
            
            if len(returns_2) < 2:
                return 1.0
            
            var_2 = np.var(returns_2)
            
            if var_1 == 0:
                return 1.0
            
            # Variance ratio should be close to 2 for random walk
            variance_ratio = var_2 / (2 * var_1)
            return variance_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating variance ratio: {e}")
            return 1.0
    
    def _calculate_realized_volatility(self, returns: List[float], timeframe: str) -> float:
        """Calculate realized volatility (annualized)."""
        if len(returns) < 2:
            return 0.0
        
        periods_per_year = self._get_periods_per_year(timeframe)
        return np.std(returns) * math.sqrt(periods_per_year)
    
    def _calculate_garch_volatility(self, returns: List[float]) -> float:
        """Simplified GARCH(1,1) volatility estimate."""
        if len(returns) < 10:
            return self._calculate_realized_volatility(returns, "1d")
        
        try:
            # Simplified GARCH parameters (typical values)
            omega = 0.000001
            alpha = 0.1
            beta = 0.85
            
            returns_array = np.array(returns)
            n = len(returns_array)
            
            # Initialize variance
            variance = np.var(returns_array)
            garch_variances = []
            
            for i in range(1, n):
                # GARCH(1,1): σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
                variance = omega + alpha * (returns_array[i-1] ** 2) + beta * variance
                garch_variances.append(variance)
            
            if len(garch_variances) == 0:
                return 0.0
            
            # Return annualized volatility
            return math.sqrt(np.mean(garch_variances)) * math.sqrt(252)
            
        except Exception as e:
            logger.warning(f"Error calculating GARCH volatility: {e}")
            return self._calculate_realized_volatility(returns, "1d")
    
    def _calculate_volatility_of_volatility(self, returns: List[float], window: int = 10) -> float:
        """Calculate volatility of volatility (second-order risk)."""
        if len(returns) < window * 2:
            return 0.0
        
        try:
            returns_array = np.array(returns)
            rolling_vols = []
            
            for i in range(window, len(returns_array)):
                window_returns = returns_array[i-window:i]
                rolling_vol = np.std(window_returns)
                rolling_vols.append(rolling_vol)
            
            if len(rolling_vols) < 2:
                return 0.0
            
            return np.std(rolling_vols)
            
        except Exception as e:
            logger.warning(f"Error calculating volatility of volatility: {e}")
            return 0.0
    
    def _calculate_nvt_ratio(self, market_data: Dict[str, Any]) -> Optional[float]:
        """Calculate Network Value to Transactions ratio."""
        market_cap = market_data.get('market_cap', 0)
        daily_volume = market_data.get('volume_24h', 0)
        
        if daily_volume == 0:
            return None
        
        return market_cap / daily_volume
    
    def _calculate_liquidity_score(self, spread_ratio: float, impact_ratio: float, 
                                 turnover_ratio: float, amihud: float) -> float:
        """Calculate composite liquidity score (0-1, higher is better)."""
        try:
            # Normalize each component (lower is better for spread, impact, amihud)
            spread_score = max(0, 1 - min(1, spread_ratio * 1000))  # Scale spread
            impact_score = max(0, 1 - min(1, impact_ratio * 10))    # Scale impact
            turnover_score = min(1, turnover_ratio * 100)           # Higher turnover is better
            amihud_score = max(0, 1 - min(1, amihud / 100))        # Lower illiquidity is better
            
            # Weighted average
            liquidity_score = (
                spread_score * 0.3 +
                impact_score * 0.25 +
                turnover_score * 0.25 +
                amihud_score * 0.2
            )
            
            return max(0.0, min(1.0, liquidity_score))
        except:
            return 0.5
    
    def _calculate_technical_score(self, sharpe: float, sortino: float, 
                                 volatility: float, hurst: float) -> float:
        """Calculate composite technical score (0-1, higher is better)."""
        try:
            # Normalize Sharpe ratio (good range: -1 to 3)
            sharpe_score = max(0, min(1, (sharpe + 1) / 4))
            
            # Normalize Sortino ratio (good range: -1 to 5)
            sortino_score = max(0, min(1, (sortino + 1) / 6))
            
            # Volatility score (moderate volatility is preferred: 0.1-0.4)
            optimal_vol = 0.25
            vol_diff = abs(volatility - optimal_vol)
            vol_score = max(0, 1 - vol_diff * 2)
            
            # Hurst score (trending markets: 0.5-0.8)
            if hurst < 0.5:
                hurst_score = hurst  # Anti-persistent
            elif hurst > 0.8:
                hurst_score = 1 - (hurst - 0.8) * 5  # Over-trending
            else:
                hurst_score = (hurst - 0.5) * 2 + 0.5  # Good trending
            
            hurst_score = max(0, min(1, hurst_score))
            
            # Weighted average
            technical_score = (
                sharpe_score * 0.35 +
                sortino_score * 0.25 +
                vol_score * 0.25 +
                hurst_score * 0.15
            )
            
            return max(0.0, min(1.0, technical_score))
        except:
            return 0.5
    
    def _calculate_risk_score(self, realized_vol: float, garch_vol: float, calmar: float) -> float:
        """Calculate composite risk score (0-1, lower is better)."""
        try:
            # Higher volatility = higher risk
            vol_risk = min(1, realized_vol * 2)  # Scale volatility
            
            # GARCH vs realized vol difference (instability indicator)
            vol_instability = min(1, abs(garch_vol - realized_vol) * 5)
            
            # Calmar ratio (higher is better, so invert for risk)
            if calmar <= 0:
                calmar_risk = 1.0
            else:
                calmar_risk = max(0, 1 - min(1, calmar / 2))
            
            # Weighted average
            risk_score = (
                vol_risk * 0.4 +
                vol_instability * 0.3 +
                calmar_risk * 0.3
            )
            
            return max(0.0, min(1.0, risk_score))
        except:
            return 0.5
    
    def _calculate_efficiency_score(self, hurst: float, autocorr: float, var_ratio: float) -> float:
        """Calculate market efficiency score (0-1, higher is more efficient)."""
        try:
            # Hurst close to 0.5 indicates efficiency
            hurst_eff = 1 - abs(hurst - 0.5) * 2
            
            # Low autocorrelation indicates efficiency
            autocorr_eff = 1 - abs(autocorr)
            
            # Variance ratio close to 1 indicates efficiency
            var_ratio_eff = 1 - abs(var_ratio - 1) * 0.5
            
            # Weighted average
            efficiency_score = (
                hurst_eff * 0.4 +
                autocorr_eff * 0.3 +
                var_ratio_eff * 0.3
            )
            
            return max(0.0, min(1.0, efficiency_score))
        except:
            return 0.5
    
    def _get_periods_per_year(self, timeframe: str) -> int:
        """Get number of periods per year for different timeframes."""
        timeframe_map = {
            "1m": 525600,    # 1 minute
            "5m": 105120,    # 5 minutes
            "15m": 35040,    # 15 minutes
            "1h": 8760,      # 1 hour
            "4h": 2190,      # 4 hours
            "1d": 365,       # 1 day
            "1w": 52,        # 1 week
        }
        return timeframe_map.get(timeframe, 365)
    
    def _fallback_metrics(self) -> AdvancedMetrics:
        """Return fallback metrics when calculation fails."""
        return AdvancedMetrics(
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            bid_ask_spread_ratio=0.001,
            market_impact_ratio=0.1,
            turnover_ratio=0.1,
            amihud_illiquidity=1.0,
            hurst_exponent=0.5,
            autocorrelation=0.0,
            variance_ratio=1.0,
            realized_volatility=0.2,
            garch_volatility=0.2,
            volatility_of_volatility=0.05,
            liquidity_score=0.5,
            technical_score=0.5,
            risk_score=0.5,
            efficiency_score=0.5
        )


# Factory function
def create_advanced_metrics_service() -> AdvancedMetricsService:
    """Create and return AdvancedMetricsService instance."""
    return AdvancedMetricsService()