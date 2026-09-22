"""
Unified Market Data Service for Yuki Agent

Aggregates data from multiple sources:
- CoinGecko: Price data, market trends, sentiment
- Binance: Technical indicators, orderbook, funding rates
- DeFiLlama: DeFi TVL, yield farming, protocol data

Provides a single interface for comprehensive market analysis.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from .coingecko_service import CoinGeckoService, CoinPrice, MarketTrend
from .binance_service import BinanceService, TechnicalIndicators, OrderBookData, FundingData
from .defillama_service import DeFiLlamaService, ProtocolData, YieldData, DeFiMetrics
from .sentiment_service import SentimentService
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal enumeration."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class MarketSignal:
    """Comprehensive market signal structure."""
    symbol: str
    signal: Signal
    confidence: float  # 0-100
    price_target: Optional[float]
    stop_loss: Optional[float]
    timeframe: str
    reasoning: List[str]
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    sentiment_details: Optional[Dict[str, Any]]  # Detailed sentiment breakdown
    risk_score: float  # Risk assessment from sentiment
    timestamp: datetime


@dataclass
class ComprehensiveMarketData:
    """Unified market data structure."""
    symbol: str
    
    # Price data (CoinGecko)
    current_price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_change_7d: float
    
    # Technical indicators (Binance)
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    ema_20: float
    ema_50: float
    
    # Market microstructure (Binance)
    bid_ask_spread: float
    orderbook_depth: Dict[str, float]
    funding_rate: Optional[float]
    long_short_ratio: Optional[float]  # Global long/short ratio
    top_trader_long_ratio: Optional[float]  # Top traders long ratio  
    open_interest: Optional[float]
    open_interest_change_24h: Optional[float]
    
    # DeFi data (DeFiLlama) - if applicable
    defi_tvl: Optional[float]
    yield_opportunities: List[YieldData]
    
    # Aggregated signal
    market_signal: MarketSignal
    
    timestamp: datetime


@dataclass
class MarketOverview:
    """Market overview structure."""
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    defi_tvl: float
    trending_coins: List[str]
    market_sentiment: str
    fear_greed_index: Optional[int]
    sentiment_breakdown: Optional[Dict[str, Any]]  # Detailed sentiment analysis
    market_regime: Optional[str]  # Bull/bear/sideways/crisis/euphoria
    risk_environment: Optional[str]  # Risk-on/risk-off/neutral
    top_gainers: List[CoinPrice]
    top_losers: List[CoinPrice]
    yield_opportunities: List[YieldData]
    timestamp: datetime


class MarketDataService:
    """
    Unified market data service that aggregates data from multiple sources
    and provides comprehensive market analysis for trading decisions.
    """
    
    def __init__(self, 
                 coingecko_api_key: Optional[str] = None,
                 binance_api_key: Optional[str] = None,
                 binance_api_secret: Optional[str] = None,
                 testnet: bool = False,
                 sentiment_service: Optional[SentimentService] = None,
                 sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """Initialize market data service."""
        
        # Initialize individual services
        self.coingecko = CoinGeckoService(coingecko_api_key)
        self.binance = BinanceService(binance_api_key, binance_api_secret, testnet)
        self.defillama = DeFiLlamaService()
        
        # Sentiment services for enhanced analysis
        self.sentiment_service = sentiment_service
        self.sentiment_analyzer = sentiment_analyzer
        
        # Cache for aggregated data
        self.cache = {}
        self.cache_duration = 60  # 1 minute for aggregated data
        
        logger.info("MarketDataService initialized with sentiment integration")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.coingecko.start_session()
        # Binance and DeFiLlama sessions are managed internally
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.coingecko.close_session()
        await self.binance.__aexit__(exc_type, exc_val, exc_tb)
        await self.defillama.close_session()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_comprehensive_data(self, symbol: str, include_defi: bool = True) -> ComprehensiveMarketData:
        """
        Get comprehensive market data for a symbol from all sources.
        
        Args:
            symbol: Symbol to analyze (e.g., 'BTC', 'ETH')
            include_defi: Whether to include DeFi-related data
            
        Returns:
            ComprehensiveMarketData object
        """
        try:
            cache_key = f"comprehensive_{symbol}_{include_defi}"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch independent market, technical, microstructure and derivatives
            # stages together. CoinGecko is only called as a price fallback if the
            # Binance market snapshot is unavailable.
            binance_symbol = f"{symbol}/USDT"
            task_names = [
                'market',
                'technical',
                'orderbook',
                'funding',
                'long_short',
                'open_interest',
                'yield',
            ]
            task_calls = [
                self.binance.get_market_data([symbol]),
                self.binance.get_technical_indicators(binance_symbol),
                self.binance.get_orderbook(binance_symbol),
                self.binance.get_funding_rates([f"{symbol}USDT"]),
                self.binance.get_long_short_ratios([f"{symbol}USDT"]),
                self.binance.get_open_interest_data([f"{symbol}USDT"]),
                self.defillama.get_yields(min_tvl=1000000, limit=10)
                if include_defi else self._empty_yield_data(),
            ]
            gathered = await asyncio.gather(*task_calls, return_exceptions=True)
            resolved = {
                name: None if isinstance(value, Exception) else value
                for name, value in zip(task_names, gathered)
            }

            market_rows = resolved.get('market') or []
            market_data = market_rows[0] if market_rows else None
            if not market_data and isinstance(gathered[0], Exception):
                logger.warning("Binance market data failed for %s: %s", symbol, gathered[0])

            coin_prices = []
            if not market_data:
                try:
                    coin_prices = await self.coingecko.get_coin_prices([symbol])
                except Exception as exc:
                    logger.warning("CoinGecko price fallback failed for %s: %s", symbol, exc)

            technical_data = resolved.get('technical')
            orderbook_data = resolved.get('orderbook')
            funding_data = resolved.get('funding') or []
            long_short_data = resolved.get('long_short') or []
            open_interest_data = resolved.get('open_interest') or []
            yield_data = resolved.get('yield') or []
            
            # Extract price data - use Binance data if available, otherwise CoinGecko
            if market_data:
                # Convert Binance market data to CoinPrice format
                coin_price = CoinPrice(
                    symbol=market_data['symbol'],
                    name=market_data['symbol'],
                    current_price=market_data['current_price'],
                    market_cap=market_data['market_cap'],
                    volume_24h=market_data['volume_24h'],
                    price_change_24h=market_data['price_change_percentage_24h'], # Use percentage instead of absolute change
                    price_change_percentage_24h=market_data['price_change_percentage_24h'],
                    price_change_percentage_7d=market_data['price_change_percentage_7d'],
                    market_cap_rank=market_data['market_cap_rank'],
                    timestamp=market_data['timestamp']
                )
            else:
                coin_price = coin_prices[0] if coin_prices else None
                if not coin_price:
                    # Fallback: create minimal price data
                    coin_price = CoinPrice(
                        symbol=symbol, name=symbol, current_price=0.0, market_cap=0.0,
                        volume_24h=0.0, price_change_24h=0.0, price_change_percentage_24h=0.0,
                        price_change_percentage_7d=0.0, market_cap_rank=0, timestamp=datetime.now()
                    )
            
            # Extract technical data
            if not technical_data:
                technical_data = TechnicalIndicators(
                    symbol=symbol, rsi=50.0, macd={'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
                    bollinger_bands={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                    ema_20=0.0, ema_50=0.0, volume_sma=0.0, atr=0.0, timestamp=datetime.now()
                )
            
            # Extract orderbook data
            if not orderbook_data:
                orderbook_data = OrderBookData(
                    symbol=symbol, bids=[], asks=[], bid_price=0.0, ask_price=0.0,
                    spread=0.0, spread_percentage=0.0, timestamp=datetime.now()
                )
            
            # Calculate orderbook depth
            orderbook_depth = self._calculate_orderbook_depth(orderbook_data)
            
            # Extract funding rate
            funding_rate = funding_data[0].funding_rate if funding_data else None
            
            # Extract sentiment data
            long_short_ratio = None
            top_trader_long_ratio = None
            if long_short_data:
                ls_data = long_short_data[0]
                long_short_ratio = ls_data.global_long_ratio
                top_trader_long_ratio = ls_data.top_trader_long_ratio
            
            # Extract open interest data  
            open_interest = None
            open_interest_change_24h = None
            if open_interest_data:
                oi_data = open_interest_data[0]
                open_interest = oi_data.open_interest
                open_interest_change_24h = oi_data.open_interest_change_percentage
            
            # Filter relevant yield opportunities
            relevant_yields = [y for y in yield_data if symbol.lower() in y.symbol.lower()][:5]
            
            # Calculate DeFi TVL for this asset (simplified)
            defi_tvl = sum(y.tvl for y in relevant_yields) if relevant_yields else None
            
            # Generate market signal
            market_signal = await self._generate_market_signal(
                symbol, coin_price, technical_data, orderbook_data, funding_rate,
                long_short_ratio, top_trader_long_ratio, open_interest_change_24h
            )
            
            # Create comprehensive data
            comprehensive_data = ComprehensiveMarketData(
                symbol=symbol,
                current_price=coin_price.current_price,
                market_cap=coin_price.market_cap,
                volume_24h=coin_price.volume_24h,
                price_change_24h=coin_price.price_change_24h,
                price_change_7d=coin_price.price_change_percentage_7d,
                rsi=technical_data.rsi,
                macd=technical_data.macd,
                bollinger_bands=technical_data.bollinger_bands,
                ema_20=technical_data.ema_20,
                ema_50=technical_data.ema_50,
                bid_ask_spread=orderbook_data.spread_percentage,
                orderbook_depth=orderbook_depth,
                funding_rate=funding_rate,
                long_short_ratio=long_short_ratio,
                top_trader_long_ratio=top_trader_long_ratio,
                open_interest=open_interest,
                open_interest_change_24h=open_interest_change_24h,
                defi_tvl=defi_tvl,
                yield_opportunities=relevant_yields,
                market_signal=market_signal,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, comprehensive_data)
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data for {symbol}: {e}")
            raise
    
    async def get_market_overview(self) -> MarketOverview:
        """
        Get comprehensive market overview from all sources.
        
        Returns:
            MarketOverview object
        """
        try:
            cache_key = "market_overview"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Try to get trending data from Binance first (faster, higher rate limits)
            trending_data = None
            market_movers = {'gainers': [], 'losers': []}
            
            try:
                binance_trending = await self.binance.get_trending_symbols(10)
                # Convert to MarketTrend format
                if binance_trending:
                    trending_data = MarketTrend(
                        trending_coins=[{
                            'id': item['symbol'].lower(),
                            'name': item['symbol'],
                            'symbol': item['symbol'],
                            'price_change_percentage_24h': item['price_change_percentage_24h']
                        } for item in binance_trending[:7]],
                        total_market_cap=0,  # Not available from Binance
                        total_volume=sum(item['volume_24h'] for item in binance_trending),
                        market_cap_change_percentage_24h=0,
                        market_sentiment="neutral",
                        fear_greed_index=50
                    )
                    
                    # Separate gainers and losers
                    for item in binance_trending:
                        if item['price_change_percentage_24h'] > 0:
                            market_movers['gainers'].append(item)
                        else:
                            market_movers['losers'].append(item)
                    
                    # Sort by percentage change
                    market_movers['gainers'].sort(key=lambda x: x['price_change_percentage_24h'], reverse=True)
                    market_movers['losers'].sort(key=lambda x: x['price_change_percentage_24h'])
                    
            except Exception as e:
                logger.warning(f"Binance trending data failed: {e}")
            
            # Fetch remaining data from other sources
            tasks = [
                self.defillama.get_defi_metrics(),
                self.defillama.get_yields(min_tvl=5000000, limit=20)
            ]
            
            # Add CoinGecko tasks only if Binance data failed
            if not trending_data:
                tasks.insert(0, self.coingecko.get_trending_data())
                tasks.insert(1, self.coingecko.get_market_movers(10))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Parse results based on what tasks were added
            task_index = 0
            
            # If trending_data is None, we added CoinGecko tasks
            if not trending_data:
                trending_data = results[task_index] if not isinstance(results[task_index], Exception) else None
                task_index += 1
                if not market_movers['gainers'] and not market_movers['losers']:
                    market_movers = results[task_index] if not isinstance(results[task_index], Exception) else {'gainers': [], 'losers': []}
                    task_index += 1
            
            # DeFi metrics and yield data
            defi_metrics = results[task_index] if task_index < len(results) and not isinstance(results[task_index], Exception) else None
            task_index += 1
            yield_data = results[task_index] if task_index < len(results) and not isinstance(results[task_index], Exception) else []
            
            # Get sentiment data if available
            sentiment_breakdown = None
            market_regime = None
            risk_environment = None
            fear_greed_index = trending_data.fear_greed_index if trending_data else None
            market_sentiment = trending_data.market_sentiment if trending_data else "neutral"
            
            if self.sentiment_service:
                try:
                    unified_sentiment = await self.sentiment_service.get_unified_sentiment(24)
                    
                    sentiment_breakdown = {
                        'overall_sentiment': unified_sentiment.overall_sentiment,
                        'sentiment_label': unified_sentiment.sentiment_label,
                        'confidence': unified_sentiment.confidence,
                        'fear_greed_index': unified_sentiment.fear_greed_index,
                        'market_mood': unified_sentiment.market_mood.primary_mood,
                        'trading_bias': unified_sentiment.trading_bias,
                        'source_agreement': unified_sentiment.source_agreement,
                        'market_stress': unified_sentiment.market_stress
                    }
                    
                    # Override with sentiment-based values
                    fear_greed_index = unified_sentiment.fear_greed_index
                    market_sentiment = unified_sentiment.sentiment_label
                    
                    # Get advanced analysis if available
                    if self.sentiment_analyzer:
                        advanced_analysis = self.sentiment_analyzer.analyze_sentiment(unified_sentiment)
                        market_regime = advanced_analysis.regime.regime
                        risk_environment = advanced_analysis.risk.stress_level
                        
                        # Update sentiment breakdown with advanced metrics
                        sentiment_breakdown.update({
                            'regime': market_regime,
                            'risk_environment': risk_environment,
                            'volatility_regime': advanced_analysis.risk.volatility_regime,
                            'momentum': advanced_analysis.temporal.momentum
                        })
                        
                except Exception as e:
                    logger.warning(f"Error getting sentiment data for market overview: {e}")
            
            # Create market overview
            overview = MarketOverview(
                total_market_cap=trending_data.global_market_cap if trending_data else 0.0,
                total_volume_24h=trending_data.total_volume if trending_data else 0.0,
                btc_dominance=trending_data.btc_dominance if trending_data else 0.0,
                defi_tvl=defi_metrics.total_tvl if defi_metrics else 0.0,
                trending_coins=trending_data.trending_coins if trending_data else [],
                market_sentiment=market_sentiment,
                fear_greed_index=fear_greed_index,
                sentiment_breakdown=sentiment_breakdown,
                market_regime=market_regime,
                risk_environment=risk_environment,
                top_gainers=market_movers['gainers'][:5],
                top_losers=market_movers['losers'][:5],
                yield_opportunities=yield_data[:10],
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, overview)
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            raise
    
    async def _generate_market_signal(self, 
                                    symbol: str,
                                    price_data: CoinPrice,
                                    technical_data: TechnicalIndicators,
                                    orderbook_data: OrderBookData,
                                    funding_rate: Optional[float],
                                    long_short_ratio: Optional[float] = None,
                                    top_trader_long_ratio: Optional[float] = None,
                                    open_interest_change_24h: Optional[float] = None) -> MarketSignal:
        """Generate comprehensive market signal based on all available data."""
        try:
            # Technical analysis scoring
            technical_score = self._calculate_technical_score(technical_data)
            
            # Fundamental analysis scoring (with sentiment data)
            fundamental_score = self._calculate_fundamental_score(
                price_data, funding_rate, long_short_ratio, top_trader_long_ratio, open_interest_change_24h
            )
            
            # Enhanced sentiment analysis scoring
            sentiment_score, sentiment_details, risk_score = await self._calculate_enhanced_sentiment_score(
                symbol, price_data, orderbook_data
            )
            
            # Weight scores based on market conditions
            weights = self._calculate_score_weights(sentiment_details, risk_score)
            
            # Overall weighted score
            overall_score = (
                technical_score * weights['technical'] + 
                fundamental_score * weights['fundamental'] + 
                sentiment_score * weights['sentiment']
            )
            
            # Adjust confidence based on sentiment agreement and risk
            base_confidence = min(abs(overall_score - 50) * 2, 100)
            sentiment_confidence_factor = sentiment_details.get('source_agreement', 0.5) if sentiment_details else 0.5
            risk_confidence_factor = max(0.3, 1 - risk_score)  # Higher risk = lower confidence
            
            confidence = base_confidence * sentiment_confidence_factor * risk_confidence_factor
            
            # Determine signal with sentiment considerations
            signal = self._determine_signal_with_sentiment(overall_score, sentiment_details, risk_score)
            
            # Generate enhanced reasoning
            reasoning = self._generate_enhanced_reasoning(
                technical_data, price_data, overall_score, sentiment_details, risk_score
            )
            
            # Calculate price targets with sentiment-based adjustments
            current_price = price_data.current_price
            price_target, stop_loss = self._calculate_targets_with_sentiment(
                current_price, signal, sentiment_details, risk_score
            )
            
            return MarketSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                timeframe="4h",  # Default timeframe
                reasoning=reasoning,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                sentiment_score=sentiment_score,
                sentiment_details=sentiment_details,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating market signal: {e}")
            return MarketSignal(
                symbol=symbol, signal=Signal.NEUTRAL, confidence=0.0,
                price_target=None, stop_loss=None, timeframe="4h",
                reasoning=["Error in signal generation"], technical_score=50.0,
                fundamental_score=50.0, sentiment_score=50.0, 
                sentiment_details=None, risk_score=0.5, timestamp=datetime.now()
            )
    
    def _calculate_technical_score(self, technical_data: TechnicalIndicators) -> float:
        """Calculate technical analysis score (0-100)."""
        score = 50.0  # Start neutral
        
        # RSI analysis
        if technical_data.rsi < 30:
            score += 15  # Oversold - bullish
        elif technical_data.rsi > 70:
            score -= 15  # Overbought - bearish
        elif 40 <= technical_data.rsi <= 60:
            score += 5   # Neutral zone - slightly positive
        
        # MACD analysis
        macd = technical_data.macd
        if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
            score += 10  # Bullish MACD
        elif macd['macd'] < macd['signal'] and macd['histogram'] < 0:
            score -= 10  # Bearish MACD
        
        # EMA analysis
        if technical_data.ema_20 > technical_data.ema_50:
            score += 10  # Bullish trend
        else:
            score -= 10  # Bearish trend
        
        # Bollinger Bands analysis
        bb = technical_data.bollinger_bands
        current_price = (bb['upper'] + bb['lower']) / 2  # Approximate current price
        if current_price <= bb['lower']:
            score += 5   # Near lower band - potential bounce
        elif current_price >= bb['upper']:
            score -= 5   # Near upper band - potential rejection
        
        return max(0, min(100, score))
    
    def _calculate_fundamental_score(self, price_data: CoinPrice, funding_rate: Optional[float], long_short_ratio: Optional[float] = None, top_trader_long_ratio: Optional[float] = None, open_interest_change_24h: Optional[float] = None) -> float:
        """Calculate fundamental analysis score (0-100)."""
        score = 50.0  # Start neutral
        
        # Price momentum
        if price_data.price_change_percentage_24h > 5:
            score += 10
        elif price_data.price_change_percentage_24h < -5:
            score -= 10
        
        if price_data.price_change_percentage_7d > 10:
            score += 10
        elif price_data.price_change_percentage_7d < -10:
            score -= 10
        
        # Volume analysis
        # (This would be more sophisticated with historical volume data)
        if price_data.volume_24h > 0:
            score += 5  # Positive volume presence
        
        # Funding rate analysis (for perpetuals)
        if funding_rate is not None:
            if funding_rate > 0.01:  # High positive funding (shorts pay longs)
                score += 5
            elif funding_rate < -0.01:  # High negative funding (longs pay shorts)
                score -= 5
        
        # Long/short ratio analysis
        if long_short_ratio is not None:
            if long_short_ratio > 1.5:  # More longs than shorts
                score += 3
            elif long_short_ratio < 0.7:  # More shorts than longs
                score -= 3
        
        # Top trader sentiment
        if top_trader_long_ratio is not None:
            if top_trader_long_ratio > 0.6:  # Top traders are long
                score += 5
            elif top_trader_long_ratio < 0.4:  # Top traders are short
                score -= 5
        
        # Open interest analysis
        if open_interest_change_24h is not None:
            if open_interest_change_24h > 10:  # Increasing interest
                score += 3
            elif open_interest_change_24h < -10:  # Decreasing interest
                score -= 3
        
        return max(0, min(100, score))
    
    def _calculate_sentiment_score(self, price_data: CoinPrice, orderbook_data: OrderBookData) -> float:
        """Calculate sentiment analysis score (0-100)."""
        score = 50.0  # Start neutral
        
        # Market cap rank influence
        if price_data.market_cap_rank <= 10:
            score += 10  # Top 10 coins get sentiment boost
        elif price_data.market_cap_rank <= 50:
            score += 5   # Top 50 coins get smaller boost
        
        # Spread analysis
        if orderbook_data.spread_percentage < 0.1:
            score += 5   # Tight spread indicates healthy market
        elif orderbook_data.spread_percentage > 0.5:
            score -= 5   # Wide spread indicates poor liquidity
        
        # Recent performance sentiment
        if price_data.price_change_percentage_24h > 0:
            score += min(price_data.price_change_percentage_24h * 2, 20)  # Cap at 20 points
        else:
            score += max(price_data.price_change_percentage_24h * 2, -20)  # Cap at -20 points
        
        return max(0, min(100, score))
    
    def _generate_reasoning(self, technical_data: TechnicalIndicators, price_data: CoinPrice, overall_score: float) -> List[str]:
        """Generate human-readable reasoning for the signal."""
        reasoning = []
        
        # Technical reasons
        if technical_data.rsi < 30:
            reasoning.append("RSI indicates oversold conditions")
        elif technical_data.rsi > 70:
            reasoning.append("RSI indicates overbought conditions")
        
        if technical_data.macd['macd'] > technical_data.macd['signal']:
            reasoning.append("MACD showing bullish momentum")
        else:
            reasoning.append("MACD showing bearish momentum")
        
        if technical_data.ema_20 > technical_data.ema_50:
            reasoning.append("Short-term trend above long-term trend")
        else:
            reasoning.append("Short-term trend below long-term trend")
        
        # Price action reasons
        if price_data.price_change_percentage_24h > 5:
            reasoning.append("Strong positive 24h price momentum")
        elif price_data.price_change_percentage_24h < -5:
            reasoning.append("Strong negative 24h price momentum")
        
        # Overall assessment
        if overall_score >= 70:
            reasoning.append("Multiple bullish indicators align")
        elif overall_score <= 30:
            reasoning.append("Multiple bearish indicators align")
        else:
            reasoning.append("Mixed signals suggest caution")
        
        return reasoning
    
    def _calculate_orderbook_depth(self, orderbook_data: OrderBookData) -> Dict[str, float]:
        """Calculate orderbook depth metrics."""
        try:
            # Calculate depth within certain percentage of mid price
            mid_price = (orderbook_data.bid_price + orderbook_data.ask_price) / 2 if orderbook_data.bid_price and orderbook_data.ask_price else 0
            
            # 1% depth
            depth_1pct_bids = sum(size for price, size in orderbook_data.bids if price >= mid_price * 0.99)
            depth_1pct_asks = sum(size for price, size in orderbook_data.asks if price <= mid_price * 1.01)
            
            # 5% depth
            depth_5pct_bids = sum(size for price, size in orderbook_data.bids if price >= mid_price * 0.95)
            depth_5pct_asks = sum(size for price, size in orderbook_data.asks if price <= mid_price * 1.05)
            
            return {
                'depth_1pct_bids': depth_1pct_bids,
                'depth_1pct_asks': depth_1pct_asks,
                'depth_5pct_bids': depth_5pct_bids,
                'depth_5pct_asks': depth_5pct_asks,
                'bid_ask_ratio': depth_1pct_bids / depth_1pct_asks if depth_1pct_asks > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating orderbook depth: {e}")
            return {'depth_1pct_bids': 0, 'depth_1pct_asks': 0, 'depth_5pct_bids': 0, 'depth_5pct_asks': 0, 'bid_ask_ratio': 0}
    
    async def _empty_funding_rate(self) -> List:
        """Return empty funding rate data."""
        return []

    async def _empty_long_short_data(self) -> List:
        """Return empty long/short ratio data."""
        return []

    async def _empty_open_interest_data(self) -> List:
        """Return empty open interest data."""
        return []
    
    async def _empty_yield_data(self) -> List:
        """Return empty yield data."""
        return []
    
    async def get_arbitrage_opportunities(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Identify arbitrage opportunities across different exchanges/protocols.
        
        Args:
            symbols: List of symbols to check for arbitrage
            
        Returns:
            List of arbitrage opportunities
        """
        try:
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL', 'AVAX']
            
            opportunities = []
            
            for symbol in symbols:
                # Get funding rates for potential funding arbitrage
                funding_rates = await self.binance.get_funding_rates([f"{symbol}USDT"])
                
                if funding_rates:
                    funding_rate = funding_rates[0].funding_rate
                    
                    # Look for high funding rates (>0.1% per 8 hours)
                    if abs(funding_rate) > 0.001:
                        opportunities.append({
                            'type': 'funding_arbitrage',
                            'symbol': symbol,
                            'funding_rate': funding_rate,
                            'annual_rate': funding_rate * 365 * 3,  # 3 funding periods per day
                            'direction': 'short_perp_long_spot' if funding_rate > 0 else 'long_perp_short_spot',
                            'exchange': 'Binance',
                            'risk_level': 'medium',
                            'timestamp': datetime.now()
                        })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    async def _calculate_enhanced_sentiment_score(self, symbol: str, price_data: CoinPrice, 
                                               orderbook_data: OrderBookData) -> tuple[float, Optional[Dict[str, Any]], float]:
        """Calculate enhanced sentiment score using sentiment services."""
        try:
            sentiment_score = 50.0  # Default neutral
            sentiment_details = None
            risk_score = 0.5  # Default medium risk
            
            if self.sentiment_service:
                # Get unified sentiment
                unified_sentiment = await self.sentiment_service.get_unified_sentiment(24)
                
                # Convert sentiment to 0-100 scale
                sentiment_score = (unified_sentiment.overall_sentiment + 1) * 50
                
                # Extract sentiment details
                sentiment_details = {
                    'overall_sentiment': unified_sentiment.overall_sentiment,
                    'sentiment_label': unified_sentiment.sentiment_label,
                    'confidence': unified_sentiment.confidence,
                    'fear_greed_index': unified_sentiment.fear_greed_index,
                    'market_mood': unified_sentiment.market_mood.primary_mood,
                    'trading_bias': unified_sentiment.trading_bias,
                    'source_agreement': unified_sentiment.source_agreement,
                    'market_stress': unified_sentiment.market_stress,
                    'risk_level': unified_sentiment.risk_level
                }
                
                # Calculate risk score from sentiment data
                risk_score = unified_sentiment.market_stress
                
                # Use advanced analysis if available
                if self.sentiment_analyzer:
                    advanced_analysis = self.sentiment_analyzer.analyze_sentiment(unified_sentiment)
                    
                    # Update sentiment details with advanced metrics
                    sentiment_details.update({
                        'regime': advanced_analysis.regime.regime,
                        'risk_appetite': advanced_analysis.risk.risk_appetite,
                        'volatility_regime': advanced_analysis.risk.volatility_regime,
                        'momentum': advanced_analysis.temporal.momentum,
                        'trend_strength': unified_sentiment.trend_strength
                    })
                    
                    # Update risk score with advanced risk metrics
                    risk_score = (
                        advanced_analysis.risk.systemic_risk * 0.3 +
                        advanced_analysis.risk.tail_risk * 0.2 +
                        advanced_analysis.risk.correlation_risk * 0.2 +
                        advanced_analysis.risk.liquidity_risk * 0.3
                    )
            else:
                # Fallback: basic sentiment from price action
                sentiment_score = self._calculate_sentiment_score(price_data, orderbook_data)
            
            return sentiment_score, sentiment_details, risk_score
            
        except Exception as e:
            logger.error(f"Error calculating enhanced sentiment score: {e}")
            return 50.0, None, 0.5
    
    def _calculate_score_weights(self, sentiment_details: Optional[Dict[str, Any]], risk_score: float) -> Dict[str, float]:
        """Calculate dynamic weights for different score components based on market conditions."""
        # Default weights
        weights = {
            'technical': 0.4,
            'fundamental': 0.3,
            'sentiment': 0.3
        }
        
        if sentiment_details:
            # Adjust weights based on market regime
            regime = sentiment_details.get('regime', 'sideways')
            confidence = sentiment_details.get('confidence', 0.5)
            source_agreement = sentiment_details.get('source_agreement', 0.5)
            
            # In high-confidence, high-agreement situations, increase sentiment weight
            if confidence > 0.7 and source_agreement > 0.7:
                weights['sentiment'] = 0.5
                weights['technical'] = 0.3
                weights['fundamental'] = 0.2
            
            # In crisis regimes, prioritize risk/fundamental analysis
            elif regime in ['crisis', 'bear_market'] or risk_score > 0.7:
                weights['fundamental'] = 0.5
                weights['sentiment'] = 0.2
                weights['technical'] = 0.3
            
            # In euphoric regimes, be more cautious with sentiment
            elif regime == 'euphoria':
                weights['technical'] = 0.5
                weights['fundamental'] = 0.3
                weights['sentiment'] = 0.2
        
        return weights
    
    def _determine_signal_with_sentiment(self, overall_score: float, sentiment_details: Optional[Dict[str, Any]], 
                                       risk_score: float) -> Signal:
        """Determine trading signal with sentiment considerations."""
        # Base signal from score
        if overall_score >= 70:
            base_signal = Signal.STRONG_BUY
        elif overall_score >= 60:
            base_signal = Signal.BUY
        elif overall_score <= 30:
            base_signal = Signal.STRONG_SELL
        elif overall_score <= 40:
            base_signal = Signal.SELL
        else:
            base_signal = Signal.NEUTRAL
        
        if not sentiment_details:
            return base_signal
        
        # Sentiment-based adjustments
        regime = sentiment_details.get('regime', 'sideways')
        market_mood = sentiment_details.get('market_mood', 'neutral')
        trading_bias = sentiment_details.get('trading_bias', 'neutral')
        
        # Downgrade signals in high-risk environments
        if risk_score > 0.8 or regime == 'crisis':
            if base_signal == Signal.STRONG_BUY:
                return Signal.BUY
            elif base_signal == Signal.BUY:
                return Signal.NEUTRAL
            elif base_signal == Signal.STRONG_SELL:
                return Signal.SELL
        
        # Be cautious during extreme sentiment
        if market_mood in ['euphoric', 'fearful']:
            if base_signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]:
                # Downgrade extreme signals during extreme sentiment
                return Signal.BUY if base_signal == Signal.STRONG_BUY else Signal.SELL
        
        # Consider trading bias alignment
        if trading_bias == 'caution':
            if base_signal in [Signal.STRONG_BUY, Signal.STRONG_SELL]:
                return Signal.NEUTRAL  # Be cautious when sentiment suggests caution
        
        return base_signal
    
    def _generate_enhanced_reasoning(self, technical_data: TechnicalIndicators, price_data: CoinPrice,
                                   overall_score: float, sentiment_details: Optional[Dict[str, Any]], 
                                   risk_score: float) -> List[str]:
        """Generate enhanced reasoning including sentiment insights."""
        reasoning = []
        
        # Technical reasoning
        if technical_data.rsi < 30:
            reasoning.append("RSI indicates oversold conditions")
        elif technical_data.rsi > 70:
            reasoning.append("RSI indicates overbought conditions")
        
        if technical_data.macd['macd'] > technical_data.macd['signal']:
            reasoning.append("MACD shows bullish momentum")
        else:
            reasoning.append("MACD shows bearish momentum")
        
        # Price action reasoning
        if price_data.price_change_percentage_24h > 5:
            reasoning.append("Strong positive price momentum (24h)")
        elif price_data.price_change_percentage_24h < -5:
            reasoning.append("Strong negative price momentum (24h)")
        
        # Sentiment reasoning
        if sentiment_details:
            sentiment_label = sentiment_details.get('sentiment_label', 'neutral')
            confidence = sentiment_details.get('confidence', 0)
            regime = sentiment_details.get('regime', 'sideways')
            market_mood = sentiment_details.get('market_mood', 'neutral')
            
            if confidence > 0.7:
                reasoning.append(f"High-confidence {sentiment_label} sentiment ({confidence:.1%})")
            
            if regime != 'sideways':
                reasoning.append(f"Market regime: {regime}")
            
            if market_mood in ['euphoric', 'fearful']:
                reasoning.append(f"Extreme market mood: {market_mood}")
            
            fear_greed = sentiment_details.get('fear_greed_index', 50)
            if fear_greed > 75:
                reasoning.append("Fear & Greed Index shows extreme greed")
            elif fear_greed < 25:
                reasoning.append("Fear & Greed Index shows extreme fear")
        
        # Risk reasoning
        if risk_score > 0.7:
            reasoning.append("High market risk environment")
        elif risk_score < 0.3:
            reasoning.append("Low market risk environment")
        
        return reasoning
    
    def _calculate_targets_with_sentiment(self, current_price: float, signal: Signal,
                                        sentiment_details: Optional[Dict[str, Any]], 
                                        risk_score: float) -> tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop losses with sentiment-based adjustments."""
        if signal == Signal.NEUTRAL:
            return None, None
        
        # Base targets
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            base_target_pct = 0.10 if signal == Signal.BUY else 0.15
            base_stop_pct = 0.05
        else:  # SELL signals
            base_target_pct = -0.10 if signal == Signal.SELL else -0.15
            base_stop_pct = 0.05
        
        # Sentiment-based adjustments
        if sentiment_details:
            regime = sentiment_details.get('regime', 'sideways')
            volatility_regime = sentiment_details.get('volatility_regime', 'medium')
            trend_strength = sentiment_details.get('trend_strength', 0.5)
            
            # Adjust targets based on regime
            if regime == 'bull_market' and signal in [Signal.BUY, Signal.STRONG_BUY]:
                base_target_pct *= 1.5  # More aggressive targets in bull markets
            elif regime == 'bear_market' and signal in [Signal.SELL, Signal.STRONG_SELL]:
                base_target_pct *= 1.5  # More aggressive targets in bear markets
            
            # Adjust stops based on volatility
            if volatility_regime == 'high':
                base_stop_pct *= 1.5  # Wider stops in high volatility
            elif volatility_regime == 'low':
                base_stop_pct *= 0.7  # Tighter stops in low volatility
            
            # Adjust based on trend strength
            if trend_strength > 0.7:
                base_target_pct *= (1 + trend_strength * 0.5)  # Stronger trends = higher targets
        
        # Risk-based adjustments
        if risk_score > 0.7:
            base_target_pct *= 0.7  # More conservative targets in high risk
            base_stop_pct *= 0.8   # Tighter stops in high risk
        
        # Calculate final prices
        if signal in [Signal.BUY, Signal.STRONG_BUY]:
            price_target = current_price * (1 + base_target_pct)
            stop_loss = current_price * (1 - base_stop_pct)
        else:
            price_target = current_price * (1 + base_target_pct)  # base_target_pct is negative
            stop_loss = current_price * (1 + base_stop_pct)
        
        return price_target, stop_loss


# Factory function
def create_market_data_service(
    coingecko_api_key: Optional[str] = None,
    binance_api_key: Optional[str] = None,
    binance_api_secret: Optional[str] = None,
    testnet: bool = False,
    sentiment_service: Optional[SentimentService] = None,
    sentiment_analyzer: Optional[SentimentAnalyzer] = None
) -> MarketDataService:
    """Create and return MarketDataService instance."""
    return MarketDataService(
        coingecko_api_key, binance_api_key, binance_api_secret, testnet,
        sentiment_service, sentiment_analyzer
    )
