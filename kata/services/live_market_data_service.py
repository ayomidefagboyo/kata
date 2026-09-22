"""
Live Market Data Service for Yuki Agent

This service provides real-time market data from Hyperliquid for the Yuki agent,
combining it with sentiment analysis and technical indicators.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import json

from hyperliquid.info import Info
from hyperliquid.utils import constants

from kata.services.hyperliquid_service import HyperliquidService
from kata.services.sentiment_service import SentimentService
from kata.agents.base_agent import MarketData

logger = logging.getLogger(__name__)


class LiveMarketDataService:
    """
    Service for fetching and processing live market data for trading agents.
    
    Provides real-time data from Hyperliquid combined with sentiment analysis
    and technical indicators for enhanced trading decisions.
    """
    
    def __init__(self, testnet: bool = False):
        """Initialize live market data service."""
        self.testnet = testnet
        self.info_client = None
        self.sentiment_service = None
        
        # Data cache
        self.market_cache = {}
        self.cache_duration = 30  # 30 seconds cache
        self.last_update = {}
        
        # Symbol mapping (Hyperliquid uses different format)
        self.symbol_map = {
            'BTC': 'BTC',
            'ETH': 'ETH', 
            'SOL': 'SOL',
            'AVAX': 'AVAX',
            'MATIC': 'MATIC',
            'DOGE': 'DOGE',
            'ADA': 'ADA',
            'DOT': 'DOT',
            'LINK': 'LINK',
            'UNI': 'UNI'
        }
        
        logger.info(f"LiveMarketDataService initialized (testnet: {testnet})")
    
    async def initialize(self) -> bool:
        """Initialize market data connections."""
        try:
            # Initialize Hyperliquid info client
            from kata.services.hyperliquid_client_factory import create_info_client
            base_url = "https://api.hyperliquid-testnet.xyz" if self.testnet else None
            self.info_client = create_info_client(base_url=base_url)
            
            # Test connection
            try:
                meta = self.info_client.meta()
                logger.info(f"Connected to Hyperliquid: {len(meta.get('universe', []))} markets available")
            except Exception as e:
                logger.warning(f"Hyperliquid connection test failed: {e}")
                return False
            
            # Initialize sentiment service (optional)
            try:
                from kata.services.sentiment_service import get_sentiment_service
                self.sentiment_service = get_sentiment_service()
                logger.info("Sentiment service integrated")
            except Exception as e:
                logger.warning(f"Sentiment service not available: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize live market data service: {e}")
            return False
    
    async def get_live_market_data(self, symbols: List[str] = None) -> MarketData:
        """
        Fetch live market data for specified symbols.
        
        Args:
            symbols: List of symbols to fetch (defaults to major cryptos)
            
        Returns:
            MarketData object with live data
        """
        try:
            if not self.info_client:
                await self.initialize()
            
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
            
            # Check cache first
            cache_key = f"market_data_{'-'.join(symbols)}"
            now = datetime.now()
            
            if (cache_key in self.market_cache and 
                cache_key in self.last_update and
                (now - self.last_update[cache_key]).seconds < self.cache_duration):
                logger.debug(f"Returning cached market data for {symbols}")
                return self.market_cache[cache_key]
            
            # Fetch fresh data
            logger.info(f"Fetching live market data for {symbols}")
            
            # Get market metadata
            meta = self.info_client.meta()
            universe = meta.get('universe', [])
            
            # Get current prices
            all_mids = self.info_client.all_mids()
            
            # Process each symbol
            token_prices = {}
            price_changes = {}
            volatility = {}
            volume_24h = {}
            market_cap = {}
            
            for symbol in symbols:
                try:
                    hl_symbol = self.symbol_map.get(symbol, symbol)
                    
                    # Find symbol in universe
                    symbol_index = None
                    symbol_info = None
                    for i, asset in enumerate(universe):
                        if asset['name'] == hl_symbol:
                            symbol_index = i
                            symbol_info = asset
                            break
                    
                    if symbol_index is None:
                        logger.warning(f"Symbol {hl_symbol} not found in Hyperliquid universe")
                        continue
                    
                    # Current price
                    current_price = float(all_mids[symbol_index]) if symbol_index < len(all_mids) else 0.0
                    token_prices[symbol] = Decimal(str(current_price))
                    
                    # Get 24h data
                    try:
                        candles = self.info_client.candles_snapshot(hl_symbol, "1d", 2)
                        if len(candles) >= 2:
                            prev_close = float(candles[-2]['c'])
                            if prev_close > 0:
                                change_24h = ((current_price - prev_close) / prev_close) * 100
                                price_changes[symbol] = Decimal(str(change_24h))
                                
                                # Calculate volatility from high/low
                                high_24h = float(candles[-1]['h'])
                                low_24h = float(candles[-1]['l'])
                                if current_price > 0:
                                    volatility[symbol] = (high_24h - low_24h) / current_price
                                
                                # 24h volume
                                volume_24h[symbol] = Decimal(str(float(candles[-1]['v']) * current_price))
                            else:
                                price_changes[symbol] = Decimal('0')
                                volatility[symbol] = 0.1
                                volume_24h[symbol] = Decimal('0')
                        else:
                            price_changes[symbol] = Decimal('0')
                            volatility[symbol] = 0.1
                            volume_24h[symbol] = Decimal('0')
                    except Exception as e:
                        logger.warning(f"Error fetching candle data for {hl_symbol}: {e}")
                        price_changes[symbol] = Decimal('0')
                        volatility[symbol] = 0.1
                        volume_24h[symbol] = Decimal('0')
                    
                    # Estimate market cap (simplified)
                    market_cap[symbol] = volume_24h[symbol] * Decimal('50')  # Rough multiplier
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    continue
            
            # Get sentiment data
            trend_indicators = await self._get_sentiment_indicators()
            
            # Create market data object
            market_data = MarketData(
                timestamp=now,
                token_prices=token_prices,
                price_changes=price_changes,
                volatility=volatility,
                volume_24h=volume_24h,
                market_cap=market_cap,
                trend_indicators=trend_indicators
            )
            
            # Cache the result
            self.market_cache[cache_key] = market_data
            self.last_update[cache_key] = now
            
            logger.info(f"Live market data fetched: {len(token_prices)} symbols, "
                       f"avg change: {sum(float(pc) for pc in price_changes.values()) / len(price_changes) if price_changes else 0:.2f}%")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching live market data: {e}")
            # Return fallback data
            return self._get_fallback_market_data(symbols or ['BTC', 'ETH', 'SOL'])
    
    async def _get_sentiment_indicators(self) -> Dict[str, Any]:
        """Get current sentiment indicators."""
        try:
            if not self.sentiment_service:
                return self._get_default_sentiment()
            
            # Get unified sentiment
            unified_sentiment = await self.sentiment_service.get_unified_sentiment(24)
            
            if not unified_sentiment:
                return self._get_default_sentiment()
            
            # Convert to trend indicators
            return {
                'market_trend': 'bullish' if unified_sentiment.overall_sentiment > 0.2 else 'bearish' if unified_sentiment.overall_sentiment < -0.2 else 'neutral',
                'overall_sentiment': 'bullish' if unified_sentiment.overall_sentiment > 0 else 'bearish' if unified_sentiment.overall_sentiment < 0 else 'neutral',
                'fear_greed_index': unified_sentiment.fear_greed_index,
                'social_sentiment': unified_sentiment.market_mood.primary_mood if hasattr(unified_sentiment, 'market_mood') else 'neutral',
                'news_sentiment': unified_sentiment.sentiment_label,
                'market_stress': unified_sentiment.market_stress if hasattr(unified_sentiment, 'market_stress') else 0.5,
                'confidence': unified_sentiment.confidence
            }
            
        except Exception as e:
            logger.warning(f"Error fetching sentiment indicators: {e}")
            return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Get default sentiment indicators when live data unavailable."""
        return {
            'market_trend': 'neutral',
            'overall_sentiment': 'neutral',
            'fear_greed_index': 50,
            'social_sentiment': 'neutral',
            'news_sentiment': 'neutral',
            'market_stress': 0.5,
            'confidence': 0.5
        }
    
    def _get_fallback_market_data(self, symbols: List[str]) -> MarketData:
        """Get fallback market data when live feeds fail."""
        # Basic fallback with reasonable defaults
        base_prices = {
            'BTC': 65000, 'ETH': 3200, 'SOL': 140, 'AVAX': 35, 'MATIC': 0.85,
            'DOGE': 0.08, 'ADA': 0.45, 'DOT': 6.5, 'LINK': 14.5, 'UNI': 8.2
        }
        
        token_prices = {}
        price_changes = {}
        volatility = {}
        volume_24h = {}
        market_cap = {}
        
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            token_prices[symbol] = Decimal(str(base_price))
            price_changes[symbol] = Decimal('0')  # No change data
            volatility[symbol] = 0.15  # Default volatility
            volume_24h[symbol] = Decimal('1000000')  # Default volume
            market_cap[symbol] = Decimal('1000000000')  # Default market cap
        
        return MarketData(
            timestamp=datetime.now(),
            token_prices=token_prices,
            price_changes=price_changes,
            volatility=volatility,
            volume_24h=volume_24h,
            market_cap=market_cap,
            trend_indicators=self._get_default_sentiment()
        )
    
    async def get_funding_rates(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current funding rates for futures symbols."""
        try:
            if not self.info_client:
                await self.initialize()
            
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL']
            
            funding_rates = {}
            
            for symbol in symbols:
                try:
                    hl_symbol = self.symbol_map.get(symbol, symbol)
                    
                    # Get funding history (latest entry has current rate)
                    funding_history = self.info_client.funding_history(hl_symbol, startTime=0, endTime=None)
                    
                    if funding_history:
                        current_funding = float(funding_history[-1].get('fundingRate', 0))
                        funding_rates[symbol] = current_funding
                    else:
                        funding_rates[symbol] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Error fetching funding rate for {symbol}: {e}")
                    funding_rates[symbol] = 0.0
            
            return funding_rates
            
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return {symbol: 0.0 for symbol in (symbols or [])}
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get overall market summary for dashboard."""
        try:
            market_data = await self.get_live_market_data()
            
            # Calculate summary metrics
            total_symbols = len(market_data.token_prices)
            positive_changes = sum(1 for pc in market_data.price_changes.values() if pc > 0)
            negative_changes = sum(1 for pc in market_data.price_changes.values() if pc < 0)
            
            avg_change = float(sum(market_data.price_changes.values()) / len(market_data.price_changes)) if market_data.price_changes else 0
            
            total_volume = float(sum(market_data.volume_24h.values()))
            
            return {
                'timestamp': market_data.timestamp.isoformat(),
                'total_symbols': total_symbols,
                'positive_changes': positive_changes,
                'negative_changes': negative_changes,
                'average_change_24h': avg_change,
                'total_volume_24h': total_volume,
                'market_sentiment': market_data.trend_indicators.get('overall_sentiment', 'neutral'),
                'fear_greed_index': market_data.trend_indicators.get('fear_greed_index', 50),
                'market_trend': market_data.trend_indicators.get('market_trend', 'neutral')
            }
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            }


# Global instance
live_market_service = LiveMarketDataService()


async def get_live_market_data(symbols: List[str] = None) -> MarketData:
    """Convenience function to get live market data."""
    if not live_market_service.info_client:
        await live_market_service.initialize()
    return await live_market_service.get_live_market_data(symbols)


async def get_market_summary() -> Dict[str, Any]:
    """Convenience function to get market summary."""
    return await live_market_service.get_market_summary()