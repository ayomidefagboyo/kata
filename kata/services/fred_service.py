"""
FRED API Service for Yuki Agent

Provides macro economic data from the Federal Reserve Economic Data (FRED) including:
- Interest rates, inflation, and monetary policy indicators
- Economic growth and employment data
- Market volatility and risk indicators
- Economic sentiment and business confidence metrics
- Macro correlations with cryptocurrency markets
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EconomicIndicator:
    """Economic indicator data structure."""
    series_id: str
    title: str
    units: str
    frequency: str
    seasonal_adjustment: str
    last_updated: datetime
    value: float
    previous_value: float
    change: float
    change_percent: float
    trend: str  # 'rising', 'falling', 'stable'
    significance: str  # 'high', 'medium', 'low'
    market_impact: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class MacroData:
    """Comprehensive macro economic data structure."""
    monetary_policy: Dict[str, EconomicIndicator]
    growth_indicators: Dict[str, EconomicIndicator]
    inflation_metrics: Dict[str, EconomicIndicator]
    employment_data: Dict[str, EconomicIndicator]
    market_indicators: Dict[str, EconomicIndicator]
    sentiment_indices: Dict[str, EconomicIndicator]
    risk_indicators: Dict[str, EconomicIndicator]
    macro_score: float  # -1 to 1 (bearish to bullish for risk assets)
    risk_environment: str  # 'risk_on', 'risk_off', 'neutral'
    market_regime: str  # 'growth', 'recession', 'recovery', 'peak'
    crypto_correlation: float  # Historical correlation with crypto
    timestamp: datetime


@dataclass
class EconomicEvent:
    """Economic event or data release."""
    title: str
    description: str
    release_date: datetime
    series_id: str
    value: float
    forecast: Optional[float]
    previous: float
    importance: str  # 'high', 'medium', 'low'
    market_impact: str  # 'bullish', 'bearish', 'neutral'
    surprise_factor: float  # How much actual vs forecast differed


class FREDService:
    """
    Federal Reserve Economic Data (FRED) service for macro economic analysis.
    
    Provides comprehensive economic indicators and their potential impact
    on risk assets including cryptocurrencies.
    """
    
    def __init__(self, api_key: str):
        """Initialize FRED service."""
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache settings
        self.cache_duration = {
            'indicators': 3600,  # 1 hour for economic indicators
            'macro_data': 1800,  # 30 minutes for macro analysis
            'events': 3600  # 1 hour for economic events
        }
        self.cache = {}
        
        # Key economic indicators to track
        self.key_indicators = {
            'monetary_policy': {
                'FEDFUNDS': 'Federal Funds Rate',
                'DFF': 'Effective Federal Funds Rate',
                'DGS10': '10-Year Treasury Rate',
                'DGS2': '2-Year Treasury Rate',
                'T10Y2Y': '10-Year Treasury minus 2-Year',
                'WALCL': 'Fed Balance Sheet Total Assets'
            },
            'growth_indicators': {
                'GDP': 'Gross Domestic Product',
                'GDPC1': 'Real GDP',
                'INDPRO': 'Industrial Production Index',
                'PAYEMS': 'Nonfarm Payrolls',
                'HOUST': 'Housing Starts',
                'RRSFS': 'Retail Sales'
            },
            'inflation_metrics': {
                'CPIAUCSL': 'Consumer Price Index',
                'CPILFESL': 'Core CPI',
                'PCEPI': 'PCE Price Index',
                'PCEPILFE': 'Core PCE Price Index',
                'T5YIE': '5-Year Breakeven Inflation Rate',
                'T10YIE': '10-Year Breakeven Inflation Rate'
            },
            'employment_data': {
                'UNRATE': 'Unemployment Rate',
                'CIVPART': 'Labor Force Participation Rate',
                'AHETPI': 'Average Hourly Earnings',
                'ICSA': 'Initial Jobless Claims',
                'JTSJOL': 'Job Openings'
            },
            'market_indicators': {
                'VIXCLS': 'VIX Volatility Index',
                'DEXUSEU': 'USD/EUR Exchange Rate',
                'DTWEXBGS': 'Trade Weighted USD Index',
                'DCOILWTICO': 'WTI Crude Oil Price',
                'GOLDAMGBD228NLBM': 'Gold Price'
            },
            'sentiment_indices': {
                'UMCSENT': 'University of Michigan Consumer Sentiment',
                'BSCICP03USM665S': 'Business Confidence Index',
                'STLFSI4': 'Financial Stress Index',
                'NFCI': 'National Financial Conditions Index'
            },
            'risk_indicators': {
                'TEDRATE': 'TED Spread',
                'DFII10': '10-Year TIPS Spread',
                'BAMLH0A0HYM2': 'High Yield Credit Spread',
                'MORTGAGE30US': '30-Year Mortgage Rate'
            }
        }
        
        logger.info("FREDService initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("FRED HTTP session started")
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("FRED HTTP session closed")
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_duration.get(cache_type, 3600)
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_macro_data(self) -> MacroData:
        """
        Get comprehensive macro economic data and analysis.
        
        Returns:
            MacroData object with all key indicators
        """
        try:
            cache_key = "macro_data_comprehensive"
            
            if self._is_cache_valid(cache_key, 'macro_data'):
                return self.cache[cache_key]['data']
            
            if not self.session:
                await self.start_session()
            
            # Collect all indicators by category
            macro_categories = {}
            
            for category, indicators in self.key_indicators.items():
                category_data = {}
                for series_id, title in indicators.items():
                    try:
                        indicator = await self.get_economic_indicator(series_id, title)
                        if indicator:
                            category_data[series_id] = indicator
                    except Exception as e:
                        logger.warning(f"Error fetching indicator {series_id}: {e}")
                        continue
                
                macro_categories[category] = category_data
            
            # Calculate macro score and regime
            macro_score, risk_environment, market_regime = self._analyze_macro_environment(macro_categories)
            
            # Calculate crypto correlation (simplified)
            crypto_correlation = self._estimate_crypto_correlation(macro_categories)
            
            macro_data = MacroData(
                monetary_policy=macro_categories.get('monetary_policy', {}),
                growth_indicators=macro_categories.get('growth_indicators', {}),
                inflation_metrics=macro_categories.get('inflation_metrics', {}),
                employment_data=macro_categories.get('employment_data', {}),
                market_indicators=macro_categories.get('market_indicators', {}),
                sentiment_indices=macro_categories.get('sentiment_indices', {}),
                risk_indicators=macro_categories.get('risk_indicators', {}),
                macro_score=macro_score,
                risk_environment=risk_environment,
                market_regime=market_regime,
                crypto_correlation=crypto_correlation,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, macro_data)
            return macro_data
            
        except Exception as e:
            logger.error(f"Error getting macro data: {e}")
            return self._empty_macro_data()
    
    async def get_economic_indicator(self, series_id: str, title: str = "") -> Optional[EconomicIndicator]:
        """
        Get specific economic indicator data.
        
        Args:
            series_id: FRED series ID
            title: Human-readable title
            
        Returns:
            EconomicIndicator object or None
        """
        try:
            cache_key = f"indicator_{series_id}"
            
            if self._is_cache_valid(cache_key, 'indicators'):
                return self.cache[cache_key]['data']
            
            if not self.session:
                await self.start_session()
            
            # Get series metadata
            metadata = await self._fetch_series_metadata(series_id)
            if not metadata:
                return None
            
            # Get recent observations
            observations = await self._fetch_series_observations(series_id, limit=2)
            if not observations or len(observations) < 1:
                return None
            
            # Parse current and previous values
            current_obs = observations[0]
            current_value = float(current_obs['value'])
            
            previous_value = current_value
            if len(observations) > 1:
                try:
                    previous_value = float(observations[1]['value'])
                except (ValueError, TypeError):
                    previous_value = current_value
            
            # Calculate change
            change = current_value - previous_value
            change_percent = (change / previous_value * 100) if previous_value != 0 else 0
            
            # Determine trend
            if abs(change_percent) < 0.1:
                trend = "stable"
            elif change > 0:
                trend = "rising"
            else:
                trend = "falling"
            
            # Determine significance and market impact
            significance, market_impact = self._assess_indicator_impact(series_id, change_percent, current_value)
            
            indicator = EconomicIndicator(
                series_id=series_id,
                title=title or metadata.get('title', series_id),
                units=metadata.get('units', ''),
                frequency=metadata.get('frequency', ''),
                seasonal_adjustment=metadata.get('seasonal_adjustment', ''),
                last_updated=datetime.fromisoformat(current_obs['date']),
                value=current_value,
                previous_value=previous_value,
                change=change,
                change_percent=change_percent,
                trend=trend,
                significance=significance,
                market_impact=market_impact
            )
            
            self._cache_data(cache_key, indicator)
            return indicator
            
        except Exception as e:
            logger.error(f"Error fetching indicator {series_id}: {e}")
            return None
    
    async def _fetch_series_metadata(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch series metadata from FRED API."""
        try:
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    series_list = data.get('seriess', [])
                    if series_list:
                        return series_list[0]
                
                logger.warning(f"Failed to fetch metadata for {series_id}: {response.status}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching metadata for {series_id}: {e}")
            return None
    
    async def _fetch_series_observations(self, series_id: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Fetch recent observations for a series."""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get('observations', [])
                    
                    # Filter out invalid values
                    valid_observations = []
                    for obs in observations:
                        if obs.get('value') != '.' and obs.get('value') is not None:
                            valid_observations.append(obs)
                    
                    return valid_observations
                
                logger.warning(f"Failed to fetch observations for {series_id}: {response.status}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching observations for {series_id}: {e}")
            return []
    
    def _assess_indicator_impact(self, series_id: str, change_percent: float, current_value: float) -> tuple[str, str]:
        """Assess the significance and market impact of an indicator change."""
        # Default values
        significance = "medium"
        market_impact = "neutral"
        
        # Assess significance based on magnitude of change
        abs_change = abs(change_percent)
        if abs_change > 5:
            significance = "high"
        elif abs_change > 1:
            significance = "medium"
        else:
            significance = "low"
        
        # Assess market impact based on indicator type and direction
        if series_id in ['FEDFUNDS', 'DFF']:  # Fed Funds Rate
            if change_percent > 0:
                market_impact = "bearish"  # Higher rates = bearish for risk assets
            elif change_percent < 0:
                market_impact = "bullish"  # Lower rates = bullish for risk assets
                
        elif series_id in ['VIXCLS']:  # VIX
            if change_percent > 10:
                market_impact = "bearish"  # High volatility = risk off
            elif change_percent < -10:
                market_impact = "bullish"  # Low volatility = risk on
                
        elif series_id in ['UNRATE']:  # Unemployment
            if change_percent > 5:
                market_impact = "bearish"  # Rising unemployment = economic weakness
            elif change_percent < -5:
                market_impact = "bullish"  # Falling unemployment = economic strength
                
        elif series_id in ['CPIAUCSL', 'CPILFESL']:  # Inflation
            if current_value > 4 and change_percent > 0:
                market_impact = "bearish"  # High rising inflation = hawkish Fed
            elif current_value < 2 and change_percent < 0:
                market_impact = "bullish"  # Low falling inflation = dovish Fed
                
        elif series_id in ['GDP', 'GDPC1', 'INDPRO']:  # Growth indicators
            if change_percent > 0:
                market_impact = "bullish"  # Economic growth = bullish
            elif change_percent < -2:
                market_impact = "bearish"  # Economic contraction = bearish
        
        return significance, market_impact
    
    def _analyze_macro_environment(self, macro_categories: Dict[str, Dict[str, EconomicIndicator]]) -> tuple[float, str, str]:
        """Analyze overall macro environment and return score, risk environment, and regime."""
        scores = []
        
        # Analyze each category
        for category, indicators in macro_categories.items():
            category_score = 0
            weight = 1.0
            
            # Weight different categories
            if category == 'monetary_policy':
                weight = 1.5  # Monetary policy has high impact
            elif category == 'market_indicators':
                weight = 1.2  # Market indicators are important
            elif category == 'risk_indicators':
                weight = 1.3  # Risk indicators are crucial
            
            for indicator in indicators.values():
                if indicator.market_impact == "bullish":
                    category_score += 1 * weight
                elif indicator.market_impact == "bearish":
                    category_score -= 1 * weight
                # neutral adds 0
            
            if indicators:  # Avoid division by zero
                category_score /= len(indicators)
                scores.append(category_score)
        
        # Calculate overall macro score
        macro_score = sum(scores) / len(scores) if scores else 0
        macro_score = max(-1.0, min(1.0, macro_score))  # Clamp to [-1, 1]
        
        # Determine risk environment
        if macro_score > 0.3:
            risk_environment = "risk_on"
        elif macro_score < -0.3:
            risk_environment = "risk_off"
        else:
            risk_environment = "neutral"
        
        # Determine market regime (simplified)
        growth_indicators = macro_categories.get('growth_indicators', {})
        employment_data = macro_categories.get('employment_data', {})
        
        growth_score = 0
        employment_score = 0
        
        for indicator in growth_indicators.values():
            if indicator.trend == "rising":
                growth_score += 1
            elif indicator.trend == "falling":
                growth_score -= 1
        
        for indicator in employment_data.values():
            if indicator.series_id == 'UNRATE':  # Unemployment rate (inverse)
                if indicator.trend == "falling":
                    employment_score += 1
                elif indicator.trend == "rising":
                    employment_score -= 1
            else:  # Other employment indicators
                if indicator.trend == "rising":
                    employment_score += 1
                elif indicator.trend == "falling":
                    employment_score -= 1
        
        total_regime_score = growth_score + employment_score
        
        if total_regime_score > 2:
            market_regime = "growth"
        elif total_regime_score > 0:
            market_regime = "recovery"
        elif total_regime_score > -2:
            market_regime = "peak"
        else:
            market_regime = "recession"
        
        return macro_score, risk_environment, market_regime
    
    def _estimate_crypto_correlation(self, macro_categories: Dict[str, Dict[str, EconomicIndicator]]) -> float:
        """Estimate correlation between macro environment and crypto markets."""
        # Simplified correlation estimate based on risk environment
        risk_indicators = macro_categories.get('risk_indicators', {})
        market_indicators = macro_categories.get('market_indicators', {})
        
        risk_off_signals = 0
        total_signals = 0
        
        for indicator in risk_indicators.values():
            total_signals += 1
            if indicator.market_impact == "bearish":
                risk_off_signals += 1
        
        for indicator in market_indicators.values():
            if indicator.series_id == 'VIXCLS':  # VIX
                total_signals += 1
                if indicator.value > 20:  # High volatility
                    risk_off_signals += 1
        
        if total_signals == 0:
            return 0.0
        
        # Crypto typically has positive correlation with risk assets
        # When risk is off, crypto tends to fall (negative correlation with risk-off indicators)
        risk_off_ratio = risk_off_signals / total_signals
        correlation = 0.7 - (risk_off_ratio * 1.4)  # Range from roughly 0.7 to -0.7
        
        return max(-1.0, min(1.0, correlation))
    
    def _empty_macro_data(self) -> MacroData:
        """Return empty macro data structure."""
        return MacroData(
            monetary_policy={},
            growth_indicators={},
            inflation_metrics={},
            employment_data={},
            market_indicators={},
            sentiment_indices={},
            risk_indicators={},
            macro_score=0.0,
            risk_environment="neutral",
            market_regime="unknown",
            crypto_correlation=0.0,
            timestamp=datetime.now()
        )


# Factory function
def create_fred_service(api_key: str) -> FREDService:
    """Create and return FREDService instance."""
    return FREDService(api_key)