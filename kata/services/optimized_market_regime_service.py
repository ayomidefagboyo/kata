#!/usr/bin/env python3
"""
Optimized Market Regime Service for Flow AI Trading Platform

Efficiently calculates market regime using Binance batch data only.
Eliminates trending data failures and symbol format issues.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketRegimeSignal:
    """Market regime signal data."""
    signal: str  # bull_market, bear_market, sideways, crisis
    strength: float  # 0.0 to 1.0
    value: float
    reason: str

@dataclass 
class MarketRegimeData:
    """Complete market regime analysis."""
    regime: str  # Primary regime
    confidence: float
    signals: Dict[str, MarketRegimeSignal]
    timestamp: datetime

class OptimizedMarketRegimeService:
    """Memory-optimized market regime calculator using Binance batch data."""
    
    def __init__(self):
        self.binance_service = None
        logger.info("📊 Optimized Market Regime Service initialized")
    
    def _get_binance_service(self):
        """Lazy initialization of Binance service."""
        if not self.binance_service:
            from .binance_service import create_binance_service
            # Create Binance service without API keys (for public market data only)
            self.binance_service = create_binance_service()
            logger.info("Initialized Binance service for market regime analysis")
        return self.binance_service
    
    async def calculate_market_regime(self) -> MarketRegimeData:
        """Calculate market regime efficiently using Binance batch data."""
        try:
            logger.info("🌊 Calculating market regime using optimized Binance data...")
            
            # Get Binance service
            binance = self._get_binance_service()
            
            # Batch fetch all ticker data at once (most efficient)
            all_tickers = await binance.exchange.fetch_tickers()
            
            # Extract key market indicators from batch data
            btc_ticker = all_tickers.get('BTC/USDT', {})
            eth_ticker = all_tickers.get('ETH/USDT', {})
            
            # Calculate regime signals
            signals = {}
            
            # 1. BTC Dominance Signal (simplified using volume ratios)
            signals['dominance'] = await self._calculate_dominance_signal(all_tickers)
            
            # 2. Fear & Greed (simplified using volatility)
            signals['fear_greed'] = await self._calculate_fear_greed_signal(btc_ticker)
            
            # 3. BTC Trend Signal
            signals['btc_trend'] = await self._calculate_btc_trend_signal(btc_ticker)
            
            # 4. Sector Rotation Signal (using ETH/BTC ratio)
            signals['sector'] = await self._calculate_sector_signal(btc_ticker, eth_ticker)
            
            # 5. Market Structure (using overall market movement)
            signals['structure'] = await self._calculate_structure_signal(all_tickers)
            
            # Determine overall regime
            regime, confidence = self._determine_regime(signals)
            
            result = MarketRegimeData(
                regime=regime,
                confidence=confidence,
                signals=signals,
                timestamp=datetime.now()
            )
            
            logger.info(f"🌊 Market regime calculated: {regime} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Market regime calculation failed: {e}")
            
            # Return safe default
            return MarketRegimeData(
                regime='sideways',
                confidence=0.5,
                signals={'default': MarketRegimeSignal('sideways', 0.5, 0, 'Default due to calculation error')},
                timestamp=datetime.now()
            )
    
    async def _calculate_dominance_signal(self, all_tickers: Dict) -> MarketRegimeSignal:
        """Calculate BTC dominance signal from volume data."""
        try:
            btc_volume = all_tickers.get('BTC/USDT', {}).get('quoteVolume', 0)
            eth_volume = all_tickers.get('ETH/USDT', {}).get('quoteVolume', 0)
            
            # Get top altcoin volumes
            alt_volumes = []
            alt_symbols = ['ADA/USDT', 'SOL/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT']
            
            for symbol in alt_symbols:
                volume = all_tickers.get(symbol, {}).get('quoteVolume', 0)
                if volume > 0:
                    alt_volumes.append(volume)
            
            total_alt_volume = sum(alt_volumes)
            total_volume = btc_volume + eth_volume + total_alt_volume
            
            if total_volume > 0:
                btc_dominance = (btc_volume / total_volume) * 100
            else:
                btc_dominance = 45  # Default
            
            # Determine signal
            if btc_dominance < 40:
                return MarketRegimeSignal('bull_market', 0.9, btc_dominance, 'Very low BTC dominance - Alt season')
            elif btc_dominance > 65:
                return MarketRegimeSignal('bear_market', 0.8, btc_dominance, 'High BTC dominance - Risk-off')
            else:
                return MarketRegimeSignal('sideways', 0.6, btc_dominance, 'Balanced BTC dominance')
                
        except Exception as e:
            logger.warning(f"Dominance calculation failed: {e}")
            return MarketRegimeSignal('sideways', 0.2, 45, 'Default BTC dominance')
    
    async def _calculate_fear_greed_signal(self, btc_ticker: Dict) -> MarketRegimeSignal:
        """Calculate fear/greed signal from BTC price volatility."""
        try:
            price_change_24h = btc_ticker.get('percentage', 0)
            
            # Simple fear/greed based on daily change
            if price_change_24h > 5:
                return MarketRegimeSignal('bull_market', 0.7, price_change_24h, 'Strong positive momentum - Greed')
            elif price_change_24h < -5:
                return MarketRegimeSignal('bear_market', 0.7, price_change_24h, 'Strong negative momentum - Fear')
            else:
                return MarketRegimeSignal('sideways', 0.5, price_change_24h, 'Neutral momentum')
                
        except Exception as e:
            logger.warning(f"Fear/greed calculation failed: {e}")
            return MarketRegimeSignal('sideways', 0.2, 0, 'Default fear/greed')
    
    async def _calculate_btc_trend_signal(self, btc_ticker: Dict) -> MarketRegimeSignal:
        """Calculate BTC trend signal from price data."""
        try:
            price_change_24h = btc_ticker.get('percentage', 0)
            volume_24h = btc_ticker.get('quoteVolume', 0)
            
            # Strong trend requires both price movement and volume
            if price_change_24h > 3 and volume_24h > 1000000000:  # $1B volume
                return MarketRegimeSignal('bull_market', 0.8, price_change_24h, 'Strong BTC uptrend with volume')
            elif price_change_24h < -3 and volume_24h > 1000000000:
                return MarketRegimeSignal('bear_market', 0.8, price_change_24h, 'Strong BTC downtrend with volume')
            else:
                return MarketRegimeSignal('sideways', 0.6, price_change_24h, 'BTC consolidation or low volume')
                
        except Exception as e:
            logger.warning(f"BTC trend calculation failed: {e}")
            return MarketRegimeSignal('sideways', 0.3, 0, 'Default BTC trend')
    
    async def _calculate_sector_signal(self, btc_ticker: Dict, eth_ticker: Dict) -> MarketRegimeSignal:
        """Calculate sector rotation signal using ETH/BTC performance."""
        try:
            btc_change = btc_ticker.get('percentage', 0)
            eth_change = eth_ticker.get('percentage', 0)
            
            # ETH outperforming BTC suggests alt season
            relative_performance = eth_change - btc_change
            
            if relative_performance > 5:
                return MarketRegimeSignal('bull_market', 0.7, relative_performance, 'ETH outperforming - Alt season')
            elif relative_performance < -5:
                return MarketRegimeSignal('bear_market', 0.7, relative_performance, 'BTC outperforming - Risk-off')
            else:
                return MarketRegimeSignal('sideways', 0.4, relative_performance, 'Balanced sector performance')
                
        except Exception as e:
            logger.warning(f"Sector calculation failed: {e}")
            return MarketRegimeSignal('sideways', 0.3, 0, 'Default sector signal')
    
    async def _calculate_structure_signal(self, all_tickers: Dict) -> MarketRegimeSignal:
        """Calculate market structure signal from overall market movement."""
        try:
            # Sample top coins for market breadth
            major_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            positive_count = 0
            total_count = 0
            
            for symbol in major_coins:
                ticker = all_tickers.get(symbol, {})
                change = ticker.get('percentage', 0)
                
                if change != 0:  # Valid data
                    total_count += 1
                    if change > 0:
                        positive_count += 1
            
            if total_count > 0:
                positive_ratio = positive_count / total_count
                
                if positive_ratio > 0.7:
                    return MarketRegimeSignal('bull_market', 0.6, positive_ratio * 100, 'Strong market breadth')
                elif positive_ratio < 0.3:
                    return MarketRegimeSignal('bear_market', 0.6, positive_ratio * 100, 'Weak market breadth')
                else:
                    return MarketRegimeSignal('sideways', 0.4, positive_ratio * 100, 'Mixed market breadth')
            else:
                return MarketRegimeSignal('sideways', 0.2, 50, 'No market breadth data')
                
        except Exception as e:
            logger.warning(f"Structure calculation failed: {e}")
            return MarketRegimeSignal('sideways', 0.2, 50, 'Default market structure')
    
    def _determine_regime(self, signals: Dict[str, MarketRegimeSignal]) -> tuple[str, float]:
        """Determine overall market regime from individual signals."""
        try:
            bull_score = 0
            bear_score = 0
            sideways_score = 0
            total_weight = 0
            
            for signal in signals.values():
                weight = signal.strength
                total_weight += weight
                
                if signal.signal == 'bull_market':
                    bull_score += weight
                elif signal.signal == 'bear_market':
                    bear_score += weight
                else:
                    sideways_score += weight
            
            if total_weight > 0:
                bull_ratio = bull_score / total_weight
                bear_ratio = bear_score / total_weight
                sideways_ratio = sideways_score / total_weight
                
                max_ratio = max(bull_ratio, bear_ratio, sideways_ratio)
                
                if bull_ratio == max_ratio and bull_ratio > 0.4:
                    return 'bull_market', bull_ratio
                elif bear_ratio == max_ratio and bear_ratio > 0.4:
                    return 'bear_market', bear_ratio
                else:
                    return 'sideways', max(sideways_ratio, 0.5)
            else:
                return 'sideways', 0.5
                
        except Exception as e:
            logger.warning(f"Regime determination failed: {e}")
            return 'sideways', 0.5

# Global instance
_regime_service: Optional[OptimizedMarketRegimeService] = None

def get_optimized_regime_service() -> OptimizedMarketRegimeService:
    """Get or create optimized market regime service."""
    global _regime_service
    if not _regime_service:
        _regime_service = OptimizedMarketRegimeService()
    return _regime_service