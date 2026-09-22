"""
Enhanced Market Data Service
Provides sophisticated market analysis with better data sources and market regime detection.
"""

import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class MarketRegimeData:
    """Comprehensive market regime information."""
    regime: str  # 'BULL', 'BEAR', 'UNCERTAIN', 'TRANSITIONING'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timeframe: str  # 'short', 'medium', 'long'
    indicators: Dict[str, float]
    trend_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    volatility_regime: str  # 'LOW', 'MEDIUM', 'HIGH'

@dataclass
class TokenSpecificSentiment:
    """Token-specific sentiment analysis."""
    symbol: str
    overall_sentiment: float  # -1.0 to 1.0
    social_sentiment: float
    news_sentiment: float
    technical_sentiment: float
    volume_sentiment: float
    whale_sentiment: float
    sentiment_sources: Dict[str, Any]
    confidence: float

@dataclass
class EnhancedMarketData:
    """Enhanced market data with sophisticated analysis."""
    symbol: str
    price: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    
    # Technical indicators
    rsi: float
    macd_signal: str
    bollinger_position: float
    volume_profile: str
    
    # Market structure
    market_regime: MarketRegimeData
    support_resistance: Dict[str, float]
    trend_strength: float
    
    # Sentiment
    token_sentiment: TokenSpecificSentiment
    fear_greed_index: float
    
    # Risk metrics
    volatility_percentile: float
    drawdown_risk: float
    liquidity_score: float
    
    timestamp: datetime

class EnhancedMarketDataService:
    """
    Enhanced market data service with sophisticated analysis capabilities.
    
    Features:
    - Real-time token-specific sentiment
    - Advanced market regime detection
    - Multiple data source integration
    - Dynamic fear & greed calculation
    """
    
    def __init__(self):
        self.session = None
        self._session_lock = asyncio.Lock()
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # API endpoints
        self.endpoints = {
            'coingecko_base': 'https://api.coingecko.com/api/v3',
            'fear_greed': 'https://api.alternative.me/fng/',
            'crypto_panic': 'https://cryptopanic.com/api/v1',
            'binance': 'https://api.binance.com/api/v3',
            'messari': 'https://data.messari.io/api/v1'
        }
        
        # Market regime indicators
        self.regime_indicators = {
            'btc_dominance_threshold': {'bull_min': 45, 'bear_max': 60},
            'volume_ma_ratio': {'bull_min': 1.2, 'bear_max': 0.8},
            'rsi_regime': {'overbought': 70, 'oversold': 30},
            'volatility_percentile': {'low': 25, 'high': 75}
        }
        
        logger.info("EnhancedMarketDataService initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start_session(self):
        """Lazily create one reusable session without racing concurrent requests."""
        if self.session and not self.session.closed:
            return
        async with self._session_lock:
            if self.session and not self.session.closed:
                return
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'Flow-AI-Trading-Platform/1.0'}
            )

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
    
    async def get_enhanced_token_analysis(
        self,
        symbol: str,
        coingecko_id: Optional[str] = None,
        price_snapshot: Optional[Dict[str, Any]] = None,
    ) -> EnhancedMarketData:
        """Get comprehensive enhanced analysis for a token."""
        try:
            # Check cache first
            cache_key = f"enhanced_{symbol}_{coingecko_id or 'symbol'}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            await self.start_session()
            return await self._fetch_enhanced_data(
                symbol,
                coingecko_id=coingecko_id,
                price_snapshot=price_snapshot,
            )
                
        except Exception as e:
            logger.error(f"Error getting enhanced analysis for {symbol}: {e}")
            return await self._get_fallback_data(symbol)
    
    async def _fetch_enhanced_data(
        self,
        symbol: str,
        coingecko_id: Optional[str] = None,
        price_snapshot: Optional[Dict[str, Any]] = None,
    ) -> EnhancedMarketData:
        """Fetch comprehensive data from multiple sources."""
        tasks = [
            self._get_technical_indicators(symbol, coingecko_id=coingecko_id),
            self._get_global_market_snapshot(),
            self._get_volatility_metrics(),
            self._get_fear_greed_index(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        price_data = price_snapshot or await self._get_price_data(symbol, coingecko_id=coingecko_id)
        technical_data = results[0] if not isinstance(results[0], Exception) else {}
        global_snapshot = results[1] if not isinstance(results[1], Exception) else {}
        volatility_data = results[2] if not isinstance(results[2], Exception) else {'volatility_percentile': 50.0}
        fear_greed = results[3] if not isinstance(results[3], Exception) else 50.0
        try:
            regime_data = await self._calculate_market_regime(
                float(global_snapshot.get('btc_dominance') or 50.0),
                global_snapshot,
                volatility_data,
                fear_greed=float(fear_greed),
            )
        except Exception as exc:
            logger.warning("Error calculating consolidated market regime: %s", exc)
            regime_data = self._get_fallback_regime()
        sentiment_data = self._derive_token_sentiment(symbol, price_data, technical_data, regime_data)
        volume_sentiment = sentiment_data.volume_sentiment
        
        # Combine data
        enhanced_data = await self._combine_data(
            symbol, price_data, technical_data, regime_data,
            sentiment_data, fear_greed, volume_sentiment,
            coingecko_id=coingecko_id,
        )
        
        # Cache result
        self._cache_data(f"enhanced_{symbol}", enhanced_data)
        
        return enhanced_data
    
    async def _get_price_data(self, symbol: str, coingecko_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current price and basic market data."""
        try:
            url = f"{self.endpoints['coingecko_base']}/simple/price"
            params = {
                'ids': coingecko_id or self._symbol_to_coingecko_id(symbol),
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    coin_id = coingecko_id or self._symbol_to_coingecko_id(symbol)
                    if coin_id in data:
                        coin_data = data[coin_id]
                        return {
                            'price': coin_data.get('usd', 0),
                            'price_change_24h': coin_data.get('usd_24h_change', 0),
                            'volume_24h': coin_data.get('usd_24h_vol', 0),
                            'market_cap': coin_data.get('usd_market_cap', 0)
                        }
        except Exception as e:
            logger.warning(f"Error fetching price data for {symbol}: {e}")
        
        return self._get_fallback_price_data(symbol)
    
    async def _get_technical_indicators(self, symbol: str, coingecko_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            # Get historical data for calculations
            coin_id = coingecko_id or self._symbol_to_coingecko_id(symbol)
            url = f"{self.endpoints['coingecko_base']}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30',
                'interval': 'hourly'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [point[1] for point in data.get('prices', [])]
                    volumes = [point[1] for point in data.get('total_volumes', [])]
                    
                    return await self._calculate_technical_indicators(prices, volumes)
        except Exception as e:
            logger.warning(f"Error fetching technical data for {symbol}: {e}")
        
        return self._get_fallback_technical_data()
    
    async def _calculate_technical_indicators(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Calculate various technical indicators."""
        if len(prices) < 14:
            return self._get_fallback_technical_data()
        
        # RSI calculation
        rsi = self._calculate_rsi(prices[-14:])
        
        # MACD signal
        macd_signal = self._calculate_macd_signal(prices)
        
        # Bollinger Bands position
        bb_position = self._calculate_bollinger_position(prices[-20:])
        
        # Volume analysis
        volume_profile = self._analyze_volume_profile(volumes[-24:])
        
        # Trend strength
        trend_strength = self._calculate_trend_strength(prices[-20:])
        
        return {
            'rsi': rsi,
            'macd_signal': macd_signal,
            'bollinger_position': bb_position,
            'volume_profile': volume_profile,
            'trend_strength': trend_strength
        }
    
    async def _get_market_regime_data(self) -> MarketRegimeData:
        """Determine current market regime using multiple indicators."""
        try:
            global_snapshot, volatility_data, fear_greed = await asyncio.gather(
                self._get_global_market_snapshot(),
                self._get_volatility_metrics(),
                self._get_fear_greed_index(),
            )
            
            # Calculate regime
            regime = await self._calculate_market_regime(
                float(global_snapshot.get('btc_dominance') or 50.0),
                global_snapshot,
                volatility_data,
                fear_greed=float(fear_greed),
            )
            
            return regime
            
        except Exception as e:
            logger.warning(f"Error calculating market regime: {e}")
            return self._get_fallback_regime()
    
    async def _calculate_market_regime(self, btc_dominance: float, 
                                     market_metrics: Dict[str, Any],
                                     volatility_data: Dict[str, Any],
                                     fear_greed: float = 50.0) -> MarketRegimeData:
        """Calculate market regime based on multiple factors."""
        
        # Regime scoring system
        bull_score = 0
        bear_score = 0
        uncertainty_score = 0
        
        # BTC Dominance analysis
        if btc_dominance < 45:  # Alt season indicator
            bull_score += 2
        elif btc_dominance > 60:  # Flight to safety
            bear_score += 2
        else:
            uncertainty_score += 1
        
        # Market cap growth
        market_cap_change = market_metrics.get('market_cap_change_24h', 0)
        if market_cap_change > 3:
            bull_score += 2
        elif market_cap_change < -3:
            bear_score += 2
        else:
            uncertainty_score += 1
        
        # Volume analysis
        volume_ratio = market_metrics.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            if market_cap_change > 0:
                bull_score += 1
            else:
                bear_score += 1
        else:
            uncertainty_score += 1
        
        # Volatility regime
        volatility_percentile = volatility_data.get('volatility_percentile', 50)
        if volatility_percentile > 80:  # Very high volatility
            bear_score += 1
            uncertainty_score += 1
        elif volatility_percentile < 20:  # Very low volatility
            bull_score += 1
        
        # Fear & Greed is fetched independently for the response. Use the supplied
        # snapshot here so regime calculation does not duplicate that network call.
        if fear_greed > 75:  # Extreme greed
            bear_score += 1  # Contrarian indicator
            uncertainty_score += 1
        elif fear_greed < 25:  # Extreme fear
            bull_score += 1  # Buying opportunity
        
        # Determine regime
        total_score = bull_score + bear_score + uncertainty_score
        if total_score == 0:
            regime = 'UNCERTAIN'
            strength = 0.3
        elif bull_score > bear_score and bull_score > uncertainty_score:
            regime = 'BULL'
            strength = bull_score / total_score
        elif bear_score > bull_score and bear_score > uncertainty_score:
            regime = 'BEAR'
            strength = bear_score / total_score
        else:
            regime = 'UNCERTAIN'
            strength = uncertainty_score / total_score
        
        # Confidence calculation
        max_score = max(bull_score, bear_score, uncertainty_score)
        confidence = max_score / total_score if total_score > 0 else 0.3
        
        # Determine volatility regime
        if volatility_percentile > 75:
            vol_regime = 'HIGH'
        elif volatility_percentile < 25:
            vol_regime = 'LOW'
        else:
            vol_regime = 'MEDIUM'
        
        # Trend direction
        if market_cap_change > 2:
            trend_direction = 'UP'
        elif market_cap_change < -2:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'SIDEWAYS'
        
        return MarketRegimeData(
            regime=regime,
            strength=min(strength, 1.0),
            confidence=min(confidence, 1.0),
            timeframe='medium',
            indicators={
                'btc_dominance': btc_dominance,
                'market_cap_change': market_cap_change,
                'volume_ratio': volume_ratio,
                'volatility_percentile': volatility_percentile,
                'fear_greed': fear_greed,
                'bull_score': bull_score,
                'bear_score': bear_score,
                'uncertainty_score': uncertainty_score
            },
            trend_direction=trend_direction,
            volatility_regime=vol_regime
        )

    def _derive_token_sentiment(
        self,
        symbol: str,
        price_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        regime_data: Optional[MarketRegimeData],
    ) -> TokenSpecificSentiment:
        """Derive transparent sentiment from already-fetched data without more HTTP fan-out."""
        price_change = float(price_data.get('price_change_24h') or 0.0)
        social = max(-0.4, min(0.4, price_change / 20.0))
        rsi = float(technical_data.get('rsi') or 50.0)
        macd_signal = str(technical_data.get('macd_signal') or 'NEUTRAL').upper()
        technical = (0.2 if rsi < 30 else -0.2 if rsi > 70 else 0.0)
        technical += 0.1 if macd_signal == 'BULLISH' else -0.1 if macd_signal == 'BEARISH' else 0.0
        technical = max(-0.4, min(0.4, technical))
        volume_profile = str(technical_data.get('volume_profile') or 'NORMAL').upper()
        volume = 0.2 if volume_profile == 'HIGH' else -0.1 if volume_profile == 'LOW' else 0.0
        regime_name = str(getattr(regime_data, 'regime', 'UNCERTAIN')).upper()
        news_proxy = 0.1 if regime_name == 'BULL' else -0.1 if regime_name == 'BEAR' else 0.0
        market_cap = float(price_data.get('market_cap') or 0.0)
        whale = 0.1 if market_cap >= 50_000_000_000 else 0.05 if market_cap >= 10_000_000_000 else -0.05
        values = [social, news_proxy, technical, volume, whale]
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        overall = sum(value * weight for value, weight in zip(values, weights))
        confidence = max(0.3, 1.0 - self._calculate_standard_deviation(values))
        return TokenSpecificSentiment(
            symbol=symbol,
            overall_sentiment=overall,
            social_sentiment=social,
            news_sentiment=news_proxy,
            technical_sentiment=technical,
            volume_sentiment=volume,
            whale_sentiment=whale,
            sentiment_sources={
                'source': 'derived_from_price_technical_and_regime',
                'synthetic_social': True,
                'synthetic_news': True,
            },
            confidence=confidence,
        )
    
    async def _get_token_sentiment(self, symbol: str) -> TokenSpecificSentiment:
        """Get token-specific sentiment from multiple sources."""
        try:
            tasks = [
                self._get_social_sentiment(symbol),
                self._get_news_sentiment(symbol),
                self._get_technical_sentiment(symbol),
                self._get_volume_sentiment(symbol),
                self._get_whale_sentiment(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            social_sentiment = results[0] if not isinstance(results[0], Exception) else 0.0
            news_sentiment = results[1] if not isinstance(results[1], Exception) else 0.0
            technical_sentiment = results[2] if not isinstance(results[2], Exception) else 0.0
            volume_sentiment = results[3] if not isinstance(results[3], Exception) else 0.0
            whale_sentiment = results[4] if not isinstance(results[4], Exception) else 0.0
            
            # Weighted average
            weights = {
                'social': 0.25,
                'news': 0.25,
                'technical': 0.20,
                'volume': 0.15,
                'whale': 0.15
            }
            
            overall_sentiment = (
                social_sentiment * weights['social'] +
                news_sentiment * weights['news'] +
                technical_sentiment * weights['technical'] +
                volume_sentiment * weights['volume'] +
                whale_sentiment * weights['whale']
            )
            
            # Calculate confidence based on agreement
            sentiments = [social_sentiment, news_sentiment, technical_sentiment, volume_sentiment, whale_sentiment]
            sentiment_std = self._calculate_standard_deviation(sentiments)
            confidence = max(0.3, 1.0 - sentiment_std)
            
            return TokenSpecificSentiment(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                social_sentiment=social_sentiment,
                news_sentiment=news_sentiment,
                technical_sentiment=technical_sentiment,
                volume_sentiment=volume_sentiment,
                whale_sentiment=whale_sentiment,
                sentiment_sources={
                    'social_weight': weights['social'],
                    'news_weight': weights['news'],
                    'technical_weight': weights['technical']
                },
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Error getting token sentiment for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)
    
    async def _get_fear_greed_index(self) -> float:
        """Get real Crypto Fear & Greed Index."""
        try:
            async with self.session.get(self.endpoints['fear_greed']) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and len(data['data']) > 0:
                        return float(data['data'][0]['value'])
        except Exception as e:
            logger.warning(f"Error fetching Fear & Greed Index: {e}")
        
        # Fallback calculation
        return await self._calculate_synthetic_fear_greed()
    
    async def _combine_data(self, symbol: str, price_data: Dict, technical_data: Dict,
                          regime_data: MarketRegimeData, sentiment_data: TokenSpecificSentiment,
                          fear_greed: float, volume_sentiment: float,
                          coingecko_id: Optional[str] = None) -> EnhancedMarketData:
        """Combine all data sources into enhanced market data."""

        # Independent enrichments run together. Market-regime volatility is already a
        # BTC-volatility snapshot, so reuse it instead of fetching the same chart twice.
        support_resistance, drawdown_risk = await asyncio.gather(
            self._calculate_support_resistance(
                symbol,
                price_data.get('price', 0),
                coingecko_id=coingecko_id,
            ),
            self._calculate_drawdown_risk(symbol, coingecko_id=coingecko_id),
        )
        regime_indicators = getattr(regime_data, 'indicators', {}) or {}
        volatility_percentile = float(regime_indicators.get('volatility_percentile') or 50.0)
        liquidity_score = self._calculate_liquidity_score_from_price(price_data, volume_sentiment)
        
        return EnhancedMarketData(
            symbol=symbol,
            price=price_data.get('price', 0),
            price_change_24h=price_data.get('price_change_24h', 0),
            volume_24h=price_data.get('volume_24h', 0),
            market_cap=price_data.get('market_cap', 0),
            
            rsi=technical_data.get('rsi', 50),
            macd_signal=technical_data.get('macd_signal', 'NEUTRAL'),
            bollinger_position=technical_data.get('bollinger_position', 0.5),
            volume_profile=technical_data.get('volume_profile', 'NORMAL'),
            
            market_regime=regime_data,
            support_resistance=support_resistance,
            trend_strength=technical_data.get('trend_strength', 0),
            
            token_sentiment=sentiment_data,
            fear_greed_index=fear_greed,
            
            volatility_percentile=volatility_percentile,
            drawdown_risk=drawdown_risk,
            liquidity_score=liquidity_score,
            
            timestamp=datetime.now()
        )
    
    # Helper methods for calculations
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_signal(self, prices: List[float]) -> str:
        """Calculate MACD signal."""
        if len(prices) < 26:
            return 'NEUTRAL'
        
        # Simple MACD approximation
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        if macd_line > 0:
            return 'BULLISH'
        elif macd_line < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[-period:]) / period  # Start with SMA
        
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_standard_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # Cache management
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_age = (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds()
        return cache_age < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    # Fallback methods
    async def _get_fallback_data(self, symbol: str) -> EnhancedMarketData:
        """Provide fallback data when APIs fail."""
        logger.warning(f"Using fallback data for {symbol}")
        
        fallback_regime = MarketRegimeData(
            regime='UNCERTAIN',
            strength=0.3,
            confidence=0.3,
            timeframe='medium',
            indicators={},
            trend_direction='SIDEWAYS',
            volatility_regime='MEDIUM'
        )
        
        fallback_sentiment = TokenSpecificSentiment(
            symbol=symbol,
            overall_sentiment=0.0,
            social_sentiment=0.0,
            news_sentiment=0.0,
            technical_sentiment=0.0,
            volume_sentiment=0.0,
            whale_sentiment=0.0,
            sentiment_sources={},
            confidence=0.3
        )
        
        return EnhancedMarketData(
            symbol=symbol,
            # Missing data must stay missing. Plausible fabricated prices leak into
            # scoring and can anchor an LLM trade plan to the wrong market.
            price=0.0,
            price_change_24h=0.0,
            volume_24h=0.0,
            market_cap=0.0,
            rsi=50.0,
            macd_signal='NEUTRAL',
            bollinger_position=0.5,
            volume_profile='NORMAL',
            market_regime=fallback_regime,
            support_resistance={},
            trend_strength=0.0,
            token_sentiment=fallback_sentiment,
            fear_greed_index=50.0,
            volatility_percentile=50.0,
            drawdown_risk=0.5,
            liquidity_score=0.5,
            timestamp=datetime.now()
        )
    
    # Utility methods
    def _symbol_to_coingecko_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko ID."""
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'AVAX': 'avalanche-2',
            'MATIC': 'matic-network',
            'USDC': 'usd-coin',
            'USDT': 'tether',
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'H': 'humanity-protocol'
        }
        return symbol_map.get(symbol.upper(), symbol.lower())
    
    # Additional methods with basic implementations
    async def _get_global_market_snapshot(self) -> Dict[str, Any]:
        """Fetch CoinGecko global data once for dominance, market cap and volume."""
        try:
            url = f"{self.endpoints['coingecko_base']}/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    payload = await response.json()
                    market_data = payload.get('data') or {}
                    total_market_cap = float((market_data.get('total_market_cap') or {}).get('usd') or 0.0)
                    total_volume = float((market_data.get('total_volume') or {}).get('usd') or 0.0)
                    return {
                        'btc_dominance': float((market_data.get('market_cap_percentage') or {}).get('btc') or 50.0),
                        'market_cap_change_24h': float(market_data.get('market_cap_change_percentage_24h_usd') or 0.0),
                        'volume_ratio': total_volume / total_market_cap if total_market_cap > 0 else 1.0,
                    }
        except Exception as exc:
            logger.warning("Error fetching global market snapshot: %s", exc)
        return {'btc_dominance': 50.0, 'market_cap_change_24h': 0.0, 'volume_ratio': 1.0}

    async def _get_btc_dominance(self) -> float:
        """Get BTC dominance percentage."""
        snapshot = await self._get_global_market_snapshot()
        return float(snapshot.get('btc_dominance') or 50.0)
    
    async def _get_market_metrics(self) -> Dict[str, Any]:
        """Get market-wide metrics."""
        snapshot = await self._get_global_market_snapshot()
        return {
            'market_cap_change_24h': float(snapshot.get('market_cap_change_24h') or 0.0),
            'volume_ratio': float(snapshot.get('volume_ratio') or 1.0),
        }
    
    async def _get_volatility_metrics(self) -> Dict[str, Any]:
        """Get volatility metrics."""
        try:
            # Simple volatility calculation based on BTC price
            url = f"{self.endpoints['coingecko_base']}/coins/bitcoin/market_chart"
            params = {'vs_currency': 'usd', 'days': '30', 'interval': 'daily'}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [point[1] for point in data.get('prices', [])]
                    if len(prices) > 1:
                        # Calculate daily returns
                        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                        volatility = self._calculate_standard_deviation(returns) * (365 ** 0.5)  # Annualized
                        
                        # Convert to percentile (simplified)
                        if volatility > 0.6:
                            percentile = 90
                        elif volatility > 0.4:
                            percentile = 70
                        elif volatility > 0.2:
                            percentile = 50
                        elif volatility > 0.1:
                            percentile = 30
                        else:
                            percentile = 10
                        
                        return {'volatility_percentile': percentile}
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
        return {'volatility_percentile': 50.0}
    
    # Additional sentiment methods with basic implementations
    async def _get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment for token."""
        # Generate synthetic sentiment based on price action and volume
        try:
            price_data = await self._get_price_data(symbol)
            price_change = price_data.get('price_change_24h', 0)
            
            # Simple sentiment based on price momentum
            if price_change > 5:
                return 0.3  # Positive sentiment
            elif price_change > 0:
                return 0.1
            elif price_change > -5:
                return -0.1
            else:
                return -0.3  # Negative sentiment
        except:
            return 0.0
    
    async def _get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment for token."""
        # Generate sentiment based on market conditions
        try:
            market_metrics = await self._get_market_metrics()
            market_change = market_metrics.get('market_cap_change_24h', 0)
            
            # News sentiment follows market sentiment
            return max(-0.5, min(0.5, market_change / 10))
        except:
            return 0.0
    
    async def _get_technical_sentiment(self, symbol: str) -> float:
        """Get technical analysis sentiment."""
        try:
            technical_data = await self._get_technical_indicators(symbol)
            rsi = technical_data.get('rsi', 50)
            macd_signal = technical_data.get('macd_signal', 'NEUTRAL')
            
            # Convert RSI to sentiment
            if rsi > 70:
                rsi_sentiment = -0.2  # Overbought
            elif rsi < 30:
                rsi_sentiment = 0.2   # Oversold (positive for buying)
            else:
                rsi_sentiment = 0.0
            
            # Convert MACD to sentiment
            macd_sentiment = 0.1 if macd_signal == 'BULLISH' else -0.1 if macd_signal == 'BEARISH' else 0.0
            
            return (rsi_sentiment + macd_sentiment) / 2
        except:
            return 0.0
    
    async def _get_volume_sentiment(self, symbol: str) -> float:
        """Get volume-based sentiment."""
        try:
            technical_data = await self._get_technical_indicators(symbol)
            volume_profile = technical_data.get('volume_profile', 'NORMAL')
            
            if volume_profile == 'HIGH':
                return 0.2  # High volume is positive
            elif volume_profile == 'LOW':
                return -0.1  # Low volume is negative
            else:
                return 0.0
        except:
            return 0.0
    
    async def _get_whale_sentiment(self, symbol: str) -> float:
        """Get whale activity sentiment."""
        # Generate based on market cap and volatility
        try:
            price_data = await self._get_price_data(symbol)
            market_cap = price_data.get('market_cap', 0)
            
            # Larger market caps tend to have more institutional interest
            if market_cap > 50_000_000_000:  # > $50B
                return 0.1
            elif market_cap > 10_000_000_000:  # > $10B
                return 0.05
            else:
                return -0.05
        except:
            return 0.0
    
    # Additional required methods
    async def _calculate_support_resistance(
        self,
        symbol: str,
        current_price: float,
        coingecko_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate sophisticated support and resistance levels using multiple methods."""
        try:
            # Get historical data for pivot point calculation
            historical_data = await self._get_historical_ohlc_data(
                symbol,
                days=30,
                coingecko_id=coingecko_id,
            )
            
            # Calculate traditional pivot points
            pivot_levels = self._calculate_pivot_points(historical_data, current_price)
            
            # Calculate psychological levels
            psychological_levels = self._calculate_psychological_levels(current_price)
            
            # Calculate Fibonacci retracement levels
            fibonacci_levels = self._calculate_fibonacci_levels(historical_data, current_price)
            
            # Calculate volume-based support/resistance
            volume_levels = await self._calculate_volume_based_levels(symbol, historical_data)
            
            # Combine all levels and find strongest ones
            all_levels = {
                **pivot_levels,
                **psychological_levels, 
                **fibonacci_levels,
                **volume_levels
            }
            
            # Return the most significant levels
            return self._select_primary_levels(all_levels, current_price)
            
        except Exception as e:
            logger.error(f"Error calculating advanced S/R levels for {symbol}: {e}")
            # Fallback to simple calculation
            return {
                'support': current_price * 0.95,
                'resistance': current_price * 1.05,
                'pivot_point': current_price,
                'support_1': current_price * 0.97,
                'resistance_1': current_price * 1.03
            }
    
    async def _calculate_volatility_percentile(self, symbol: str) -> float:
        """Calculate volatility percentile for the token."""
        volatility_data = await self._get_volatility_metrics()
        return volatility_data.get('volatility_percentile', 50.0)
    
    async def _calculate_drawdown_risk(self, symbol: str, coingecko_id: Optional[str] = None) -> float:
        """Calculate maximum drawdown risk."""
        try:
            # Use historical price data to estimate drawdown risk
            coin_id = coingecko_id or self._symbol_to_coingecko_id(symbol)
            url = f"{self.endpoints['coingecko_base']}/coins/{coin_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': '90', 'interval': 'daily'}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [point[1] for point in data.get('prices', [])]
                    if len(prices) > 1:
                        # Calculate maximum drawdown
                        peak = prices[0]
                        max_drawdown = 0
                        for price in prices:
                            if price > peak:
                                peak = price
                            else:
                                drawdown = (peak - price) / peak
                                max_drawdown = max(max_drawdown, drawdown)
                        return min(max_drawdown, 1.0)
        except:
            pass
        return 0.3  # Default moderate risk
    
    async def _calculate_liquidity_score(self, symbol: str, volume_sentiment: float) -> float:
        """Calculate liquidity score based on volume sentiment."""
        try:
            # Use volume sentiment directly as a base score
            base_score = volume_sentiment
            
            # Adjust based on market cap (larger caps tend to be more liquid)
            price_data = await self._get_price_data(symbol)
            market_cap = price_data.get('market_cap', 1)
            
            # Market cap adjustment
            if market_cap > 10000000000:  # > $10B
                cap_adjustment = 0.2
            elif market_cap > 1000000000:  # > $1B
                cap_adjustment = 0.1
            else:
                cap_adjustment = 0.0
            
            final_score = min(1.0, base_score + cap_adjustment)
            return max(0.0, final_score)
            
        except:
            return volume_sentiment  # Return the sentiment value if calculation fails

    def _calculate_liquidity_score_from_price(self, price_data: Dict[str, Any], volume_sentiment: float) -> float:
        """Calculate liquidity from the price snapshot already in memory."""
        market_cap = float(price_data.get('market_cap') or 0.0)
        volume_24h = float(price_data.get('volume_24h') or 0.0)
        turnover = volume_24h / market_cap if market_cap > 0 else 0.0
        cap_score = 0.75 if market_cap >= 10_000_000_000 else 0.6 if market_cap >= 1_000_000_000 else 0.4
        turnover_score = max(0.0, min(1.0, turnover * 10.0))
        sentiment_score = max(0.0, min(1.0, 0.5 + float(volume_sentiment or 0.0)))
        return max(0.0, min(1.0, cap_score * 0.45 + turnover_score * 0.40 + sentiment_score * 0.15))
    
    async def _calculate_synthetic_fear_greed(self) -> float:
        """Calculate synthetic fear & greed index when API is unavailable."""
        try:
            market_metrics = await self._get_market_metrics()
            btc_dominance = await self._get_btc_dominance()
            volatility_data = await self._get_volatility_metrics()
            
            market_change = market_metrics.get('market_cap_change_24h', 0)
            volatility_percentile = volatility_data.get('volatility_percentile', 50)
            
            # Base score from market change
            base_score = 50 + (market_change * 2)
            
            # Adjust for volatility (high volatility = more fear)
            volatility_adjustment = (50 - volatility_percentile) * 0.3
            
            # Adjust for BTC dominance (high dominance = more fear)
            dominance_adjustment = (50 - btc_dominance) * 0.2
            
            fear_greed = base_score + volatility_adjustment + dominance_adjustment
            
            return max(0, min(100, fear_greed))
        except:
            return 50.0
    
    # Fallback methods
    def _get_fallback_price_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback price data when API fails."""
        return {
            'price': 0.0,
            'price_change_24h': 0.0,
            'volume_24h': 0.0,
            'market_cap': 0.0,
            'data_available': False,
        }
    
    def _get_fallback_technical_data(self) -> Dict[str, Any]:
        """Fallback technical data."""
        return {
            'rsi': 50.0,
            'macd_signal': 'NEUTRAL',
            'bollinger_position': 0.5,
            'volume_profile': 'NORMAL',
            'trend_strength': 0.0
        }
    
    def _get_fallback_regime(self) -> MarketRegimeData:
        """Fallback market regime."""
        return MarketRegimeData(
            regime='UNCERTAIN',
            strength=0.3,
            confidence=0.3,
            timeframe='medium',
            indicators={},
            trend_direction='SIDEWAYS',
            volatility_regime='MEDIUM'
        )
    
    def _get_fallback_sentiment(self, symbol: str) -> TokenSpecificSentiment:
        """Fallback sentiment data."""
        return TokenSpecificSentiment(
            symbol=symbol,
            overall_sentiment=0.0,
            social_sentiment=0.0,
            news_sentiment=0.0,
            technical_sentiment=0.0,
            volume_sentiment=0.0,
            whale_sentiment=0.0,
            sentiment_sources={},
            confidence=0.3
        )
    
    # Missing method implementations
    def _calculate_bollinger_position(self, prices: List[float]) -> float:
        """Calculate position within Bollinger Bands."""
        if len(prices) < 20:
            return 0.5
        
        # Simple moving average
        sma = sum(prices[-20:]) / 20
        
        # Standard deviation
        variance = sum((price - sma) ** 2 for price in prices[-20:]) / 20
        std_dev = variance ** 0.5
        
        # Bollinger bands
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        
        current_price = prices[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        # Position within bands (0 = lower band, 1 = upper band)
        position = (current_price - lower_band) / (upper_band - lower_band)
        return max(0, min(1, position))
    
    def _analyze_volume_profile(self, volumes: List[float]) -> str:
        """Analyze volume profile."""
        if len(volumes) < 10:
            return 'NORMAL'
        
        recent_avg = sum(volumes[-5:]) / 5
        historical_avg = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) > 5 else recent_avg
        
        if historical_avg == 0:
            return 'NORMAL'
        
        ratio = recent_avg / historical_avg
        
        if ratio > 2.0:
            return 'HIGH'
        elif ratio < 0.5:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength."""
        if len(prices) < 10:
            return 0.0
        
        # Calculate price momentum
        recent_prices = prices[-5:]
        older_prices = prices[-10:-5]
        
        recent_avg = sum(recent_prices) / len(recent_prices)
        older_avg = sum(older_prices) / len(older_prices)
        
        if older_avg == 0:
            return 0.0
        
        momentum = (recent_avg - older_avg) / older_avg
        
        # Normalize to -1 to 1 range
        return max(-1, min(1, momentum * 10))


    # Advanced Support/Resistance Analysis Methods
    async def _get_historical_ohlc_data(
        self,
        symbol: str,
        days: int = 30,
        coingecko_id: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Get historical OHLC data for pivot point calculations."""
        try:
            # Get data from CoinGecko
            resolved_coin_id = coingecko_id or self._symbol_to_coingecko_id(symbol)
            url = f"{self.endpoints['coingecko_base']}/coins/{resolved_coin_id}/ohlc"
            params = {'vs_currency': 'usd', 'days': days}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to OHLC format
                    ohlc_data = []
                    for candle in data:
                        ohlc_data.append({
                            'timestamp': candle[0],
                            'open': float(candle[1]),
                            'high': float(candle[2]), 
                            'low': float(candle[3]),
                            'close': float(candle[4])
                        })
                    return ohlc_data
            return []
        except Exception as e:
            logger.error(f"Error fetching OHLC data for {symbol}: {e}")
            return []
    
    def _calculate_pivot_points(self, historical_data: List[Dict[str, float]], current_price: float) -> Dict[str, float]:
        """Calculate traditional pivot points from recent high, low, close."""
        try:
            if not historical_data:
                return {}
            
            # Use last 5 days for pivot calculation
            recent_data = historical_data[-5:] if len(historical_data) >= 5 else historical_data
            
            # Calculate pivot point from recent data
            highs = [candle['high'] for candle in recent_data]
            lows = [candle['low'] for candle in recent_data]
            closes = [candle['close'] for candle in recent_data]
            
            high = max(highs)
            low = min(lows)
            close = closes[-1]  # Most recent close
            
            # Standard pivot point calculation
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot_point': pivot,
                'pivot_resistance_1': r1,
                'pivot_support_1': s1,
                'pivot_resistance_2': r2,
                'pivot_support_2': s2,
                'pivot_resistance_3': r3,
                'pivot_support_3': s3
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    def _calculate_psychological_levels(self, current_price: float) -> Dict[str, float]:
        """Calculate psychological support/resistance levels (round numbers)."""
        try:
            levels = {}
            
            # Determine the magnitude of the price for appropriate rounding
            if current_price >= 1000:
                # For high prices, use hundreds
                round_factor = 100
            elif current_price >= 100:
                # For medium prices, use tens  
                round_factor = 10
            elif current_price >= 10:
                # For lower prices, use whole numbers
                round_factor = 1
            elif current_price >= 1:
                # For very low prices, use 0.1
                round_factor = 0.1
            else:
                # For very small prices, use 0.01
                round_factor = 0.01
            
            # Find nearest psychological levels
            base_level = round(current_price / round_factor) * round_factor
            
            levels['psychological_support'] = base_level - round_factor
            levels['psychological_resistance'] = base_level + round_factor
            
            # Add additional levels
            if current_price > base_level + (round_factor * 0.5):
                levels['psychological_support'] = base_level
                levels['psychological_resistance'] = base_level + round_factor
            else:
                levels['psychological_support'] = base_level - round_factor
                levels['psychological_resistance'] = base_level
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating psychological levels: {e}")
            return {}
    
    def _calculate_fibonacci_levels(self, historical_data: List[Dict[str, float]], current_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels from recent swing high/low."""
        try:
            if len(historical_data) < 10:
                return {}
            
            # Find swing high and low from recent data
            recent_data = historical_data[-20:] if len(historical_data) >= 20 else historical_data
            
            highs = [candle['high'] for candle in recent_data]
            lows = [candle['low'] for candle in recent_data]
            
            swing_high = max(highs)
            swing_low = min(lows)
            
            # Calculate Fibonacci retracement levels
            diff = swing_high - swing_low
            
            fib_levels = {
                'fib_0': swing_high,
                'fib_23.6': swing_high - (diff * 0.236),
                'fib_38.2': swing_high - (diff * 0.382),
                'fib_50': swing_high - (diff * 0.5),
                'fib_61.8': swing_high - (diff * 0.618),
                'fib_78.6': swing_high - (diff * 0.786),
                'fib_100': swing_low,
                # Extension levels
                'fib_161.8': swing_low - (diff * 0.618),
                'fib_261.8': swing_low - (diff * 1.618)
            }
            
            # Only return levels near current price (within 20%)
            price_range = current_price * 0.2
            relevant_levels = {}
            
            for level_name, level_price in fib_levels.items():
                if abs(level_price - current_price) <= price_range:
                    relevant_levels[level_name] = level_price
            
            return relevant_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    async def _calculate_volume_based_levels(self, symbol: str, historical_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate volume-based support/resistance using volume profile concept."""
        try:
            if not historical_data:
                return {}
            
            # Create price-volume histogram
            price_volume_map = {}
            
            for candle in historical_data:
                # Use typical price (HLC/3) as the representative price
                typical_price = (candle['high'] + candle['low'] + candle['close']) / 3
                
                # Round to reasonable precision
                price_bucket = round(typical_price, 4)
                
                # Estimate volume (since we may not have volume data)
                # Use price range as proxy for volume/activity
                estimated_volume = candle['high'] - candle['low']
                
                if price_bucket in price_volume_map:
                    price_volume_map[price_bucket] += estimated_volume
                else:
                    price_volume_map[price_bucket] = estimated_volume
            
            # Find high volume areas (POC - Point of Control)
            if not price_volume_map:
                return {}
            
            # Sort by volume
            volume_sorted = sorted(price_volume_map.items(), key=lambda x: x[1], reverse=True)
            
            # Get top volume levels as support/resistance
            levels = {}
            if len(volume_sorted) >= 1:
                levels['volume_poc'] = volume_sorted[0][0]  # Point of Control
            if len(volume_sorted) >= 2:
                levels['volume_high_1'] = volume_sorted[1][0]
            if len(volume_sorted) >= 3:
                levels['volume_high_2'] = volume_sorted[2][0]
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating volume-based levels: {e}")
            return {}
    
    def _select_primary_levels(self, all_levels: Dict[str, float], current_price: float) -> Dict[str, float]:
        """Select the most significant support and resistance levels."""
        try:
            if not all_levels:
                return {
                    'support': current_price * 0.95,
                    'resistance': current_price * 1.05
                }
            
            # Separate levels into support (below price) and resistance (above price)
            support_levels = []
            resistance_levels = []
            
            for name, price in all_levels.items():
                if price < current_price:
                    support_levels.append((name, price))
                elif price > current_price:
                    resistance_levels.append((name, price))
            
            # Sort support levels (highest first - closest to current price)
            support_levels.sort(key=lambda x: x[1], reverse=True)
            
            # Sort resistance levels (lowest first - closest to current price)
            resistance_levels.sort(key=lambda x: x[1])
            
            # Select primary levels
            result = {}
            
            # Primary support (closest below current price)
            if support_levels:
                result['support'] = support_levels[0][1]
                if len(support_levels) > 1:
                    result['support_2'] = support_levels[1][1]
            else:
                result['support'] = current_price * 0.95
            
            # Primary resistance (closest above current price)  
            if resistance_levels:
                result['resistance'] = resistance_levels[0][1]
                if len(resistance_levels) > 1:
                    result['resistance_2'] = resistance_levels[1][1]
            else:
                result['resistance'] = current_price * 1.05
            
            # Add pivot point if available
            if 'pivot_point' in all_levels:
                result['pivot_point'] = all_levels['pivot_point']
            
            # Add volume POC if available
            if 'volume_poc' in all_levels:
                result['volume_poc'] = all_levels['volume_poc']
            
            return result
            
        except Exception as e:
            logger.error(f"Error selecting primary levels: {e}")
            return {
                'support': current_price * 0.95,
                'resistance': current_price * 1.05
            }

    # BTC Correlation Analysis
    async def calculate_btc_correlation(self, symbol: str, days: int = 30) -> Dict[str, float]:
        """Calculate correlation with BTC and market-wide factors."""
        try:
            if symbol.upper() == 'BTC':
                return {
                    'btc_correlation': 1.0,
                    'correlation_strength': 'PERFECT',
                    'beta': 1.0,
                    'relative_strength': 0.0,
                    'market_dependence': 'INDEPENDENT'
                }
            
            # Get historical data for both assets
            symbol_data = await self._get_historical_ohlc_data(symbol, days)
            btc_data = await self._get_historical_ohlc_data('BTC', days)
            
            if not symbol_data or not btc_data:
                return self._default_correlation_data()
            
            # Align data by timestamp and calculate returns
            correlation_data = self._calculate_price_correlation(symbol_data, btc_data)
            
            # Calculate additional market metrics
            market_metrics = await self._calculate_market_metrics(symbol, correlation_data)
            
            return {
                **correlation_data,
                **market_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating BTC correlation for {symbol}: {e}")
            return self._default_correlation_data()
    
    def _calculate_price_correlation(self, symbol_data: List[Dict], btc_data: List[Dict]) -> Dict[str, float]:
        """Calculate statistical correlation between two price series."""
        try:
            # Extract price changes
            symbol_returns = []
            btc_returns = []
            
            # Calculate daily returns
            for i in range(1, min(len(symbol_data), len(btc_data))):
                symbol_return = (symbol_data[i]['close'] - symbol_data[i-1]['close']) / symbol_data[i-1]['close']
                btc_return = (btc_data[i]['close'] - btc_data[i-1]['close']) / btc_data[i-1]['close']
                
                symbol_returns.append(symbol_return)
                btc_returns.append(btc_return)
            
            if len(symbol_returns) < 5:
                return self._default_correlation_data()
            
            # Calculate correlation coefficient
            correlation = self._pearson_correlation(symbol_returns, btc_returns)
            
            # Calculate beta (sensitivity to BTC moves)
            beta = self._calculate_beta(symbol_returns, btc_returns)
            
            # Calculate relative strength
            symbol_total_return = sum(symbol_returns)
            btc_total_return = sum(btc_returns)
            relative_strength = symbol_total_return - btc_total_return
            
            # Determine correlation strength
            strength = self._correlation_strength(abs(correlation))
            
            # Determine market dependence
            dependence = self._market_dependence(correlation, beta)
            
            return {
                'btc_correlation': round(correlation, 3),
                'correlation_strength': strength,
                'beta': round(beta, 3),
                'relative_strength': round(relative_strength * 100, 2),  # As percentage
                'market_dependence': dependence
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return self._default_correlation_data()
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi * xi for xi in x)
        sum_yy = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_xx - sum_x**2) * (n * sum_yy - sum_y**2))**0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_beta(self, symbol_returns: List[float], btc_returns: List[float]) -> float:
        """Calculate beta (systematic risk measure)."""
        try:
            if not btc_returns:
                return 1.0
            
            # Calculate variance of BTC returns
            btc_mean = sum(btc_returns) / len(btc_returns)
            btc_variance = sum((r - btc_mean)**2 for r in btc_returns) / len(btc_returns)
            
            if btc_variance == 0:
                return 1.0
            
            # Calculate covariance
            symbol_mean = sum(symbol_returns) / len(symbol_returns)
            covariance = sum((symbol_returns[i] - symbol_mean) * (btc_returns[i] - btc_mean) 
                           for i in range(len(symbol_returns))) / len(symbol_returns)
            
            return covariance / btc_variance
            
        except:
            return 1.0
    
    def _correlation_strength(self, abs_correlation: float) -> str:
        """Classify correlation strength."""
        if abs_correlation >= 0.8:
            return 'VERY_STRONG'
        elif abs_correlation >= 0.6:
            return 'STRONG'
        elif abs_correlation >= 0.4:
            return 'MODERATE'
        elif abs_correlation >= 0.2:
            return 'WEAK'
        else:
            return 'VERY_WEAK'
    
    def _market_dependence(self, correlation: float, beta: float) -> str:
        """Determine market dependence level."""
        if abs(correlation) < 0.3 and abs(beta) < 0.5:
            return 'INDEPENDENT'
        elif correlation > 0.7 and beta > 1.2:
            return 'HIGHLY_DEPENDENT'
        elif correlation > 0.5:
            return 'DEPENDENT'
        else:
            return 'MODERATELY_DEPENDENT'
    
    async def _calculate_market_metrics(self, symbol: str, correlation_data: Dict) -> Dict[str, Any]:
        """Calculate additional market-wide metrics."""
        try:
            # Get market cap dominance data if available
            btc_dominance = 50.0  # Default
            
            # Calculate risk-adjusted metrics
            correlation = correlation_data.get('btc_correlation', 0.0)
            beta = correlation_data.get('beta', 1.0)
            
            # Risk score based on correlation and beta
            risk_score = min(100, max(0, 50 + (abs(correlation) * 30) + (abs(beta - 1) * 20)))
            
            # Diversification benefit (inverse of correlation)
            diversification_score = max(0, 100 - (abs(correlation) * 100))
            
            return {
                'btc_dominance': btc_dominance,
                'systematic_risk_score': round(risk_score, 1),
                'diversification_benefit': round(diversification_score, 1),
                'market_regime_alignment': self._assess_regime_alignment(correlation, beta)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market metrics: {e}")
            return {
                'btc_dominance': 50.0,
                'systematic_risk_score': 50.0,
                'diversification_benefit': 50.0,
                'market_regime_alignment': 'NEUTRAL'
            }
    
    def _assess_regime_alignment(self, correlation: float, beta: float) -> str:
        """Assess how well the asset aligns with current market regime."""
        if correlation > 0.6 and beta > 1.0:
            return 'BULL_ALIGNED'
        elif correlation > 0.6 and beta < 1.0:
            return 'DEFENSIVE'
        elif correlation < -0.3:
            return 'CONTRARIAN'
        else:
            return 'NEUTRAL'
    
    def _default_correlation_data(self) -> Dict[str, Any]:
        """Default correlation data when calculation fails."""
        return {
            'btc_correlation': 0.5,
            'correlation_strength': 'MODERATE',
            'beta': 1.0,
            'relative_strength': 0.0,
            'market_dependence': 'DEPENDENT',
            'btc_dominance': 50.0,
            'systematic_risk_score': 50.0,
            'diversification_benefit': 50.0,
            'market_regime_alignment': 'NEUTRAL'
        }


# Factory function
def create_enhanced_market_data_service() -> EnhancedMarketDataService:
    """Create enhanced market data service instance."""
    return EnhancedMarketDataService()
