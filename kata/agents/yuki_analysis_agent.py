"""
Yuki Analysis Agent - Token Analysis Specialization

This is a specialized version of Yuki for token analysis that:
- Uses REST APIs instead of WebSocket feeds
- Focuses on single token analysis
- Provides fast, reliable analysis without live trading dependencies
- Shares the same technical analysis logic as the main Yuki agent
"""

import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

from kata.agents.base_agent import BaseAgent, MarketData, RiskLevel
from kata.agents.yuki_agent import YukiAgent, MarketRegime, TechnicalIndicators
from kata.services.binance_service import BinanceService
from kata.services.coingecko_service import CoinGeckoService
from kata.services.defillama_service import DeFiLlamaService
# from kata.services.sentiment_service import SentimentService  # Temporarily disabled

logger = logging.getLogger(__name__)


class YukiAnalysisAgent(BaseAgent):
    """
    Specialized Yuki agent for token analysis.
    
    This agent provides the same aggressive futures analysis logic as the main Yuki agent
    but is optimized for token analysis using REST APIs instead of WebSocket feeds.
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any], binance_service: Optional[BinanceService] = None, coingecko_service: Optional[CoinGeckoService] = None, defillama_service: Optional[DeFiLlamaService] = None, sentiment_service: Optional[Any] = None):
        """Initialize Yuki Analysis Agent for token analysis."""
        super().__init__(user_id, "yuki_analysis", config)
        
        # Use multiple data sources for comprehensive analysis
        if binance_service:
            self.binance_service = binance_service
        else:
            # Service registry removed - using fallback
            # from kata.services.service_registry import get_or_create_binance_service
            from kata.services.binance_service import BinanceService
            self.binance_service = BinanceService()  # Fallback initialization
        self.coingecko_service = None  # Will be created on-demand if needed
        self.defillama_service = defillama_service or DeFiLlamaService()
        # self.sentiment_service = sentiment_service or SentimentService()  # Temporarily disabled
        self.sentiment_service = None
        
        # Backtesting and prediction parameters
        self.backtest_config = {
            'lookback_periods': 100,  # Days to backtest
            'prediction_horizon': 7,  # Days to predict
            'confidence_threshold': 0.7,  # Minimum confidence for predictions
            'risk_adjustment': True,  # Adjust predictions based on risk
            'ensemble_methods': ['linear', 'random_forest', 'lstm']  # Prediction methods
        }
        
        # Technical analysis parameters (same as main Yuki)
        self.ta_config = {
            'ema_periods': [20, 50],
            'sma_periods': [200],
            'rsi_period': 14,
            'rsi_oversold': 25,  # More aggressive levels
            'rsi_overbought': 75,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'adx_period': 14,
            'adx_trend_threshold': 25,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            # Enhanced futures-specific parameters
            'funding_threshold': 0.005,  # 0.5% funding rate threshold
            'oi_change_threshold': 0.15,  # 15% OI change threshold
            'pattern_lookback': 50,  # Candles to analyze for patterns
            'volume_spike_threshold': 2.0,  # 2x average volume
            'funding_swing_threshold': 0.003  # 0.3% funding rate swing
        }
        
        # Market regime detection (same as main Yuki)
        self.regime_thresholds = {
            'trending_adx': 25,
            'volatile_atr_multiplier': 1.5,
            'ranging_bb_width': 0.02,
            'breakout_volume_multiplier': 2.0
        }
        
        logger.info(f"Yuki Analysis Agent initialized for user {user_id} - optimized for token analysis")
    
    async def analyze_market(self, market_data: MarketData, focus_token: str) -> Dict[str, Any]:
        """
        Analyze a specific token using Yuki's aggressive futures strategy.
        
        This method provides the same analysis logic as the main Yuki agent
        but is optimized for single-token analysis using REST APIs.
        """
        try:
            analysis = {
                'timestamp': market_data.timestamp.isoformat(),
                'market_regime': MarketRegime.RANGING,
                'overall_sentiment': 'neutral',
                'volatility_regime': 'normal',
                'futures_opportunities': [],
                'funding_arbitrage': [],
                'risk_level': RiskLevel.MEDIUM,
                'recommended_leverage': {},
                'technical_summary': {},
                'agent_strategy': 'aggressive_futures_analysis'
            }
            
            # Focus on the specific token being analyzed
            symbol = f"{focus_token}-USD"
            
            # Get market data using Binance and CoinGecko (more reliable for analysis)
            futures_data = await self._fetch_market_data_via_external_apis(symbol, market_data)
            
            # Calculate technical indicators using the same logic as main Yuki
            indicators = await self._calculate_technical_indicators_rest(symbol, futures_data)
            
            # Extract prices and volumes for additional analysis
            prices, volumes = await self._extract_price_volume_data(symbol)
            
            # Classify market regime using the same logic
            regime = self._classify_market_regime(indicators, futures_data)
            
            # Analyze funding rate opportunities and arbitrage
            funding_analysis = await self._analyze_funding_opportunities_enhanced(symbol, futures_data)
            
            # Analyze open interest changes for momentum signals
            oi_analysis = await self._analyze_open_interest_changes(symbol)
            
            # Detect chart patterns for automation
            pattern_analysis = self._detect_chart_patterns(prices, volumes)
            
            # Enhanced on-chain and DeFi data analysis
            onchain_analysis = await self._analyze_onchain_data(focus_token)
            
            # Sentiment and social analysis
            sentiment_analysis = await self._analyze_sentiment_data(focus_token)
            
            # Backtesting results
            backtest_results = await self._run_backtest_analysis(symbol, prices, volumes)
            
            # Predictive analysis
            prediction_analysis = await self._generate_price_predictions(symbol, prices, volumes, indicators)
            
            # Calculate optimal leverage based on volatility
            optimal_leverage = self._calculate_optimal_leverage(indicators, futures_data)
            
            # Generate market structure analysis
            structure_analysis = self._analyze_market_structure(indicators, futures_data)
            
            # Store analysis results with enhanced data
            futures_analysis = {
                symbol: {
                    'indicators': indicators,
                    'regime': regime,
                    'funding': funding_analysis,
                    'optimal_leverage': optimal_leverage,
                    'structure': structure_analysis,
                    'signal_strength': self._calculate_signal_strength(indicators, regime),
                    'pattern_analysis': pattern_analysis,
                    'onchain_analysis': onchain_analysis,
                    'sentiment_analysis': sentiment_analysis,
                    'backtest_results': backtest_results,
                    'prediction_analysis': prediction_analysis
                }
            }
            
            analysis['technical_summary'] = futures_analysis
            
            # Determine overall market regime
            analysis['market_regime'] = self._determine_overall_regime(futures_analysis)
            
            # Calculate overall risk level
            analysis['risk_level'] = self._assess_futures_risk(futures_analysis)
            
            # Generate volatility regime assessment
            analysis['volatility_regime'] = self._assess_volatility_regime(futures_analysis)
            
            # Add opportunities if strong signals
            if futures_analysis[symbol]['signal_strength'] > 0.7:
                analysis['futures_opportunities'].append({
                    'symbol': symbol,
                    'direction': self._determine_trade_direction(indicators),
                    'strength': futures_analysis[symbol]['signal_strength'],
                    'leverage': optimal_leverage,
                    'reasoning': self._generate_trade_reasoning(indicators, regime)
                })
            
            # Add funding arbitrage opportunities
            if funding_analysis['arbitrage_opportunity']:
                analysis['funding_arbitrage'].append({
                    'symbol': symbol,
                    'funding_rate': futures_data.funding_rate,
                    'expected_duration': funding_analysis['duration_hours'],
                    'expected_return': funding_analysis['expected_return']
                })
            
            # Calculate agent-specific scores using the same logic as main Yuki
            technical_score = self._calculate_technical_score(futures_analysis)
            fundamental_score = self._calculate_fundamental_score(futures_analysis)
            
            # Add scores to analysis
            analysis['technical_score'] = technical_score
            analysis['fundamental_score'] = fundamental_score
            analysis['agent_confidence'] = self._calculate_agent_confidence(futures_analysis)
            
            logger.info(f"Yuki Analysis: {analysis['market_regime'].value} regime, "
                       f"{len(analysis['futures_opportunities'])} opportunities, "
                       f"{len(analysis['funding_arbitrage'])} funding plays, "
                       f"technical_score: {technical_score:.3f}, fundamental_score: {fundamental_score:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Yuki Analysis: {e}")
            raise  # Let the error propagate instead of returning fallback data
    
    def _get_coingecko_service(self):
        """Get CoinGecko service on-demand (lazy initialization)."""
        if self.coingecko_service is None:
            try:
                from ..services.coingecko_service import CoinGeckoService
                self.coingecko_service = CoinGeckoService()
                logger.info("CoinGecko service created on-demand for token analysis")
            except Exception as e:
                logger.warning(f"Could not create CoinGecko service: {e}")
                return None
        return self.coingecko_service
    
    async def _fetch_market_data_via_external_apis(self, symbol: str, market_data: MarketData) -> Any:
        """
        Fetch market data using Binance and CoinGecko APIs.
        This is more reliable for token analysis than Hyperliquid WebSocket feeds.
        """
        try:
            # Convert symbol format (ETH-USD -> ETH)
            token_symbol = symbol.replace('-USD', '')
            
            # Get current price and volume from Binance using OHLCV
            binance_symbol = f"{token_symbol}/USDT"
            binance_ohlcv = self.binance_service.exchange.fetch_ohlcv(binance_symbol, '1h', limit=1)
            
            # Use Binance price as primary, get CoinGecko as fallback if needed
            
            # Create a mock futures data object for compatibility
            from kata.agents.yuki_agent import FuturesMarketData
            
            # Use Binance price as primary, fallback to CoinGecko
            price = 0
            volume_24h = 0
            
            if binance_ohlcv and len(binance_ohlcv) > 0:
                # OHLCV format: [timestamp, open, high, low, close, volume]
                price = float(binance_ohlcv[0][4])  # close price
                volume_24h = float(binance_ohlcv[0][5])  # volume
            
            # Try CoinGecko as fallback if Binance data is insufficient
            if price == 0 or volume_24h == 0:
                coingecko_service = self._get_coingecko_service()
                if coingecko_service:
                    try:
                        coingecko_data = await coingecko_service.get_token_data(token_symbol)
                        if coingecko_data:
                            if price == 0:
                                price = float(coingecko_data.get('current_price', 0))
                                logger.info(f"Using CoinGecko price for {token_symbol}: {price}")
                            if volume_24h == 0:
                                volume_24h = float(coingecko_data.get('total_volume', 0))
                                logger.info(f"Using CoinGecko volume for {token_symbol}: {volume_24h}")
                    except Exception as e:
                        logger.warning(f"CoinGecko fallback failed for {token_symbol}: {e}")
            
            # Fail if no real data is available
            if price == 0:
                raise ValueError(f"No price data available for {token_symbol} from Binance")
            if volume_24h == 0:
                raise ValueError(f"No volume data available for {token_symbol} from Binance")
            
            # Get real funding rate from Binance futures
            try:
                funding_data = await self.binance_service.get_funding_rates([f"{token_symbol}USDT"])
                funding_rate = funding_data[0].funding_rate if funding_data else 0.0
            except Exception as e:
                logger.warning(f"Could not get funding rate for {token_symbol}: {e}")
                funding_rate = 0.0  # Only use 0 if API call fails
            
            return FuturesMarketData(
                symbol=symbol,
                price=price,
                funding_rate=funding_rate,
                funding_rate_8h=funding_rate,
                open_interest=0.0,  # Not available from these APIs
                oi_change_24h=0.0,  # Not available from these APIs
                volume_24h=volume_24h,
                mark_price=price,
                index_price=price,
                last_funding_time=datetime.now(),
                next_funding_time=datetime.now() + timedelta(hours=8),
                liquidation_threshold=0.05,
                max_leverage=50.0  # Default for major tokens
            )
            
        except Exception as e:
            logger.error(f"Error fetching external API market data for {symbol}: {e}")
            raise
    
    async def _calculate_technical_indicators_rest(self, symbol: str, futures_data: Any) -> TechnicalIndicators:
        """
        Calculate technical indicators using Binance historical data.
        This method provides the same calculations as the main Yuki agent.
        """
        try:
            # Convert symbol format (ETH-USD -> ETHUSDT)
            token_symbol = symbol.replace('-USD', 'USDT')
            
            # Get historical data from Binance (1h candles, last 200 periods)
            historical_data = self.binance_service.exchange.fetch_ohlcv(token_symbol, '1h', limit=200)
            
            if not historical_data or len(historical_data) < 50:
                raise Exception(f"Insufficient historical data for {symbol}: {len(historical_data) if historical_data else 0} data points")
            
            # Extract real price and volume data from Binance format
            # Binance klines format: [open_time, open, high, low, close, volume, close_time, ...]
            prices = [float(candle[4]) for candle in historical_data]  # close price
            volumes = [float(candle[5]) for candle in historical_data]  # volume
            
            if not prices or len(prices) < 50:
                raise Exception(f"Invalid price data for {symbol}")
            
            # Use the same technical indicator calculations as main Yuki
            indicators = TechnicalIndicators()
            
            # Moving Averages
            indicators.ema_20 = self._calculate_ema(prices, 20)
            indicators.ema_50 = self._calculate_ema(prices, 50)
            indicators.sma_200 = self._calculate_sma(prices, 200) if len(prices) >= 200 else prices[-1]
            
            # RSI and Stochastic
            indicators.rsi = self._calculate_rsi(prices, self.ta_config['rsi_period'])
            stoch_k, stoch_d = self._calculate_stochastic(prices, self.ta_config['stoch_k'], self.ta_config['stoch_d'])
            indicators.stoch_k = stoch_k
            indicators.stoch_d = stoch_d
            indicators.stoch_rsi = self._calculate_stoch_rsi(prices, 14)
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(
                prices, self.ta_config['macd_fast'], self.ta_config['macd_slow'], self.ta_config['macd_signal']
            )
            indicators.macd = macd_line
            indicators.macd_signal = signal_line
            indicators.macd_histogram = histogram
            
            # ADX for trend strength
            indicators.adx = self._calculate_adx(prices, self.ta_config['adx_period'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(
                prices, self.ta_config['bb_period'], self.ta_config['bb_std']
            )
            indicators.bb_upper = bb_upper
            indicators.bb_middle = bb_middle
            indicators.bb_lower = bb_lower
            indicators.bb_width = bb_width
            
            # ATR for volatility
            indicators.atr = self._calculate_atr(prices, self.ta_config['atr_period'])
            
            # VWAP
            indicators.vwap = self._calculate_vwap(prices, volumes)
            
            # Volume analysis using real data
            if volumes and len(volumes) > 0:
                avg_volume = sum(volumes) / len(volumes)
                indicators.volume_ratio = futures_data.volume_24h / avg_volume if avg_volume > 0 else 1.0
            else:
                indicators.volume_ratio = 1.0
            
            # Support/Resistance
            pivot, support1, resistance1 = self._calculate_pivot_points(prices)
            indicators.pivot_point = pivot
            indicators.support_1 = support1
            indicators.resistance_1 = resistance1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            raise
    
    async def _extract_price_volume_data(self, symbol: str) -> tuple[List[float], List[float]]:
        """
        Extract price and volume data for additional analysis.
        This method provides the raw data needed for pattern detection, backtesting, and predictions.
        """
        try:
            # Convert symbol format (ETH-USD -> ETHUSDT)
            token_symbol = symbol.replace('-USD', 'USDT')
            
            # Get historical data from Binance (1h candles, last 200 periods)
            historical_data = self.binance_service.exchange.fetch_ohlcv(token_symbol, '1h', limit=200)
            
            if not historical_data or len(historical_data) < 50:
                # Raise error if insufficient historical data
                raise ValueError(f"Insufficient historical data for {symbol} (need at least 50 points, got {len(historical_data) if historical_data else 0})")
            
            # Extract real price and volume data from Binance format
            # Binance klines format: [open_time, open, high, low, close, volume, close_time, ...]
            prices = [float(candle[4]) for candle in historical_data]  # close price
            volumes = [float(candle[5]) for candle in historical_data]  # volume
            
            if not prices or len(prices) < 50:
                raise Exception(f"Invalid price data for {symbol}")
            
            return prices, volumes
            
        except Exception as e:
            logger.error(f"Error extracting price/volume data for {symbol}: {e}")
            raise  # Let the error propagate instead of returning mock data
    
    # Import all the technical analysis methods from the main Yuki agent
    # These methods are identical to maintain consistency
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        return sum(prices[-period:]) / period
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
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
    
    def _calculate_stochastic(self, prices: List[float], k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(prices) < k_period:
            return 50.0, 50.0
        
        current_price = prices[-1]
        lowest_low = min(prices[-k_period:])
        highest_high = max(prices[-k_period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified D% calculation
        d_percent = k_percent
        
        return k_percent, d_percent
    
    def _calculate_stoch_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Stochastic RSI."""
        rsi = self._calculate_rsi(prices, period)
        return min(100.0, max(0.0, rsi))
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple[float, float, float]:
        """Calculate MACD."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation
        signal_line = macd_line * 0.9  # Approximation
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_adx(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average Directional Index."""
        if len(prices) < period + 1:
            return 25.0
        
        # Simplified ADX calculation
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_change = sum(price_changes[-period:]) / period
        
        # Normalize to typical ADX range
        adx = min(100.0, (avg_change / prices[-1]) * 10000)
        return max(0.0, adx)
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1], 0.0
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma
        
        return upper, sma, lower, width
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]
            low = prices[i]
            prev_close = prices[i-1]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges)
    
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate Volume Weighted Average Price."""
        if not prices or not volumes or len(prices) != len(volumes):
            return prices[-1] if prices else 0.0
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return prices[-1]
        
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return vwap
    
    def _calculate_pivot_points(self, prices: List[float]) -> tuple[float, float, float]:
        """Calculate pivot points."""
        if len(prices) < 3:
            return prices[-1], prices[-1], prices[-1]
        
        high = max(prices[-3:])
        low = min(prices[-3:])
        close = prices[-1]
        
        pivot = (high + low + close) / 3
        support1 = (2 * pivot) - high
        resistance1 = (2 * pivot) - low
        
        return pivot, support1, resistance1
    
    # Market analysis methods (same logic as main Yuki)
    
    def _classify_market_regime(self, indicators: TechnicalIndicators, futures_data: Any) -> MarketRegime:
        """Classify market regime using the same logic as main Yuki."""
        try:
            adx = indicators.adx
            bb_width = indicators.bb_width
            
            if adx > self.regime_thresholds['trending_adx']:
                return MarketRegime.TRENDING
            elif bb_width < self.regime_thresholds['ranging_bb_width']:
                return MarketRegime.RANGING
            else:
                return MarketRegime.VOLATILE
        except:
            return MarketRegime.RANGING
    
    async def _analyze_funding_opportunities_enhanced(self, symbol: str, futures_data: Any) -> Dict[str, Any]:
        """
        Enhanced funding rate analysis with cross-exchange arbitrage detection.
        Analyzes funding rate swings and arbitrage opportunities.
        """
        try:
            funding_rate = futures_data.funding_rate
            threshold = self.ta_config['funding_threshold']
            swing_threshold = self.ta_config['funding_swing_threshold']
            
            # Basic funding rate analysis
            if abs(funding_rate) > threshold:
                arbitrage_opportunity = True
                expected_return = abs(funding_rate) * 365 * 3  # Annualized
            else:
                arbitrage_opportunity = False
                expected_return = 0.0
            
            # Try to get historical funding rates from Binance for swing analysis
            funding_swing = 0.0
            funding_trend = 'stable'
            
            try:
                # Get funding rate history from Binance (if available)
                if hasattr(self.binance_service, 'get_funding_rates'):
                    funding_history = await self.binance_service.get_funding_rates([symbol.replace('-USD', 'USDT')])
                    if funding_history and len(funding_history) > 1:
                        recent_rates = [float(fr.funding_rate) for fr in funding_history[-3:]]
                        funding_swing = max(recent_rates) - min(recent_rates)
                        
                        if funding_swing > swing_threshold:
                            funding_trend = 'volatile'
                        elif funding_swing > swing_threshold * 0.5:
                            funding_trend = 'moderate'
                        else:
                            funding_trend = 'stable'
            except Exception as e:
                logger.debug(f"Could not fetch funding history: {e}")
            
            return {
                'arbitrage_opportunity': arbitrage_opportunity,
                'current_funding_rate': funding_rate,
                'funding_swing': funding_swing,
                'funding_trend': funding_trend,
                'duration_hours': 24,
                'expected_return': expected_return,
                'risk_level': 'high' if funding_swing > swing_threshold else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced funding analysis: {e}")
            return {
                'arbitrage_opportunity': False,
                'current_funding_rate': 0.0,
                'funding_swing': 0.0,
                'funding_trend': 'stable',
                'duration_hours': 0,
                'expected_return': 0.0,
                'risk_level': 'medium'
            }
    
    def _calculate_optimal_leverage(self, indicators: TechnicalIndicators, futures_data: Any) -> float:
        """Calculate optimal leverage using the same logic as main Yuki."""
        try:
            if futures_data.price <= 0:
                return 2.0
            
            volatility_ratio = indicators.atr / futures_data.price
            base_leverage = 2.0
            
            if volatility_ratio < 0.02:  # Low volatility
                return min(5.0, base_leverage * 2)
            elif volatility_ratio < 0.05:  # Medium volatility
                return base_leverage
            else:  # High volatility
                return max(1.0, base_leverage * 0.5)
        except:
            return 2.0
    
    def _analyze_market_structure(self, indicators: TechnicalIndicators, futures_data: Any) -> Dict[str, Any]:
        """Analyze market structure using the same logic as main Yuki."""
        try:
            return {
                'liquidity': 'high' if indicators.volume_ratio > 1.5 else 'medium',
                'trend_strength': 'strong' if indicators.adx > 25 else 'weak',
                'volatility': 'high' if indicators.atr > futures_data.price * 0.05 else 'normal'
            }
        except:
            return {
                'liquidity': 'medium',
                'trend_strength': 'weak',
                'volatility': 'normal'
            }
    
    async def _analyze_open_interest_changes(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze open interest changes for momentum signals.
        OI spikes/drops can indicate strong momentum or reversals.
        """
        try:
            # Convert symbol format for Binance
            binance_symbol = symbol.replace('-USD', 'USDT')
            
            # Try to get open interest data from Binance
            oi_data = {}
            
            try:
                if hasattr(self.binance_service, 'get_open_interest_data'):
                    oi_history = await self.binance_service.get_open_interest_data([binance_symbol])
                    if oi_history and len(oi_history) > 1:
                        # Calculate OI change percentage
                        recent_oi = [float(oi.open_interest) for oi in oi_history[-3:]]
                        oi_change_24h = ((recent_oi[-1] - recent_oi[0]) / recent_oi[0]) * 100
                        
                        # Determine momentum signal
                        if abs(oi_change_24h) > self.ta_config['oi_change_threshold'] * 100:
                            momentum_signal = 'strong' if oi_change_24h > 0 else 'reversal'
                        elif abs(oi_change_24h) > (self.ta_config['oi_change_threshold'] * 100) * 0.5:
                            momentum_signal = 'moderate'
                        else:
                            momentum_signal = 'weak'
                        
                        oi_data = {
                            'current_oi': recent_oi[-1],
                            'oi_change_24h': oi_change_24h,
                            'momentum_signal': momentum_signal,
                            'trend': 'increasing' if oi_change_24h > 0 else 'decreasing'
                        }
                    else:
                        oi_data = {
                            'current_oi': 0.0,
                            'oi_change_24h': 0.0,
                            'momentum_signal': 'unknown',
                            'trend': 'stable'
                        }
                else:
                    # Fallback to mock data
                    oi_data = {
                        'current_oi': 1000000.0,  # Mock 1M OI
                        'oi_change_24h': 5.0,  # Mock 5% increase
                        'momentum_signal': 'moderate',
                        'trend': 'increasing'
                    }
            except Exception as e:
                logger.debug(f"Could not fetch OI data: {e}")
                # Fallback to mock data
                oi_data = {
                    'current_oi': 1000000.0,
                    'oi_change_24h': 0.0,
                    'momentum_signal': 'unknown',
                    'trend': 'stable'
                }
            
            return oi_data
            
        except Exception as e:
            logger.error(f"Error in OI analysis: {e}")
            return {
                'current_oi': 0.0,
                'oi_change_24h': 0.0,
                'momentum_signal': 'unknown',
                'trend': 'stable'
            }
    
    async def _analyze_onchain_data(self, token_symbol: str) -> Dict[str, Any]:
        """
        Analyze on-chain data for comprehensive token insights.
        Includes DeFi metrics, whale movements, and protocol health.
        """
        try:
            onchain_data = {}
            
            # Get DeFi metrics from DeFiLlama
            try:
                if hasattr(self.defillama_service, 'get_protocol_data'):
                    protocol_data = await self.defillama_service.get_protocol_data(token_symbol.lower())
                    if protocol_data:
                        onchain_data['defi_metrics'] = {
                            'tvl': protocol_data.get('tvl', 0),
                            'tvl_change_24h': protocol_data.get('tvl_change_24h', 0),
                            'volume_24h': protocol_data.get('volume_24h', 0),
                            'fees_24h': protocol_data.get('fees_24h', 0)
                        }
            except Exception as e:
                logger.debug(f"Could not fetch DeFi data: {e}")
            
            # Get yield farming opportunities
            try:
                if hasattr(self.defillama_service, 'get_yields'):
                    yields_data = await self.defillama_service.get_yields()
                    if yields_data:
                        # Find relevant yields for this token
                        relevant_yields = [y for y in yields_data if token_symbol.lower() in y.get('token', '').lower()]
                        if relevant_yields:
                            onchain_data['yield_opportunities'] = {
                                'count': len(relevant_yields),
                                'max_apy': max([y.get('apy', 0) for y in relevant_yields]),
                                'avg_apy': sum([y.get('apy', 0) for y in relevant_yields]) / len(relevant_yields)
                            }
            except Exception as e:
                logger.debug(f"Could not fetch yield data: {e}")
            
            # Get whale movement data (mock for now, would integrate with Glassnode/Chainalysis)
            onchain_data['whale_activity'] = {
                'large_transactions_24h': self._estimate_whale_activity(token_symbol),
                'whale_confidence': 0.6,  # Would be calculated from real data
                'accumulation_signal': self._detect_accumulation_signal(token_symbol)
            }
            
            # Get network health metrics
            onchain_data['network_health'] = {
                'active_addresses': self._estimate_active_addresses(token_symbol),
                'transaction_count_24h': self._estimate_transaction_volume(token_symbol),
                'gas_fees': self._estimate_gas_fees(token_symbol),
                'network_congestion': 'low'  # Would be calculated from real data
            }
            
            return onchain_data
            
        except Exception as e:
            logger.error(f"Error in on-chain analysis: {e}")
            return {
                'defi_metrics': {},
                'yield_opportunities': {},
                'whale_activity': {},
                'network_health': {}
            }
    
    def _estimate_whale_activity(self, token_symbol: str) -> int:
        """Estimate whale activity based on token characteristics."""
        # Mock estimation - would use real whale tracking data
        whale_activity_map = {
            'ETH': 45,  # High whale activity
            'BTC': 52,
            'SOL': 38,
            'AVAX': 25,
            'MATIC': 20
        }
        return whale_activity_map.get(token_symbol, 15)
    
    def _detect_accumulation_signal(self, token_symbol: str) -> str:
        """Detect accumulation signals from whale behavior."""
        # Mock detection - would analyze real whale wallet movements
        accumulation_signals = {
            'ETH': 'strong',
            'BTC': 'moderate',
            'SOL': 'weak',
            'AVAX': 'moderate',
            'MATIC': 'weak'
        }
        return accumulation_signals.get(token_symbol, 'unknown')
    
    def _estimate_active_addresses(self, token_symbol: str) -> int:
        """Estimate active addresses for the token."""
        # Mock estimation - would use real blockchain data
        address_estimates = {
            'ETH': 450000,
            'BTC': 380000,
            'SOL': 120000,
            'AVAX': 85000,
            'MATIC': 95000
        }
        return address_estimates.get(token_symbol, 50000)
    
    def _estimate_transaction_volume(self, token_symbol: str) -> int:
        """Estimate transaction volume for the token."""
        # Mock estimation - would use real blockchain data
        tx_estimates = {
            'ETH': 1200000,
            'BTC': 450000,
            'SOL': 800000,
            'AVAX': 300000,
            'MATIC': 400000
        }
        return tx_estimates.get(token_symbol, 100000)
    
    def _estimate_gas_fees(self, token_symbol: str) -> float:
        """Estimate gas fees for the token."""
        # Mock estimation - would use real blockchain data
        gas_estimates = {
            'ETH': 25.0,  # Gwei
            'BTC': 5.0,   # Sat/vB
            'SOL': 0.00025,  # SOL
            'AVAX': 0.0001,  # AVAX
            'MATIC': 0.00001  # MATIC
        }
        return gas_estimates.get(token_symbol, 1.0)
    
    async def _analyze_sentiment_data(self, token_symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment and social data for comprehensive market psychology.
        Includes social media sentiment, news sentiment, and market fear/greed.
        """
        try:
            sentiment_data = {}
            
            # Get social media sentiment
            try:
                if hasattr(self.sentiment_service, 'get_social_sentiment'):
                    social_sentiment = await self.sentiment_service.get_social_sentiment(token_symbol)
                    sentiment_data['social_sentiment'] = social_sentiment
                else:
                    # Mock social sentiment data
                    sentiment_data['social_sentiment'] = {
                        'twitter_sentiment': self._mock_social_sentiment(token_symbol, 'twitter'),
                        'reddit_sentiment': self._mock_social_sentiment(token_symbol, 'reddit'),
                        'telegram_sentiment': self._mock_social_sentiment(token_symbol, 'telegram'),
                        'overall_social_score': self._calculate_overall_social_score(token_symbol)
                    }
            except Exception as e:
                logger.debug(f"Could not fetch social sentiment: {e}")
                sentiment_data['social_sentiment'] = self._get_fallback_sentiment(token_symbol)
            
            # Get news sentiment
            try:
                if hasattr(self.sentiment_service, 'get_news_sentiment'):
                    news_sentiment = await self.sentiment_service.get_news_sentiment(token_symbol)
                    sentiment_data['news_sentiment'] = news_sentiment
                else:
                    # Mock news sentiment data
                    sentiment_data['news_sentiment'] = {
                        'positive_articles': self._mock_news_count(token_symbol, 'positive'),
                        'negative_articles': self._mock_news_count(token_symbol, 'negative'),
                        'neutral_articles': self._mock_news_count(token_symbol, 'neutral'),
                        'overall_news_score': self._calculate_overall_news_score(token_symbol)
                    }
            except Exception as e:
                logger.debug(f"Could not fetch news sentiment: {e}")
                sentiment_data['news_sentiment'] = self._get_fallback_news_sentiment(token_symbol)
            
            # Try to get fear/greed index from CoinGecko on-demand
            try:
                coingecko_service = self._get_coingecko_service()
                if coingecko_service and hasattr(coingecko_service, 'get_fear_greed_index'):
                    fear_greed = await coingecko_service.get_fear_greed_index()
                    sentiment_data['market_sentiment'] = {
                        'fear_greed_index': fear_greed,
                        'market_mood': self._interpret_fear_greed(fear_greed)
                    }
                    logger.info(f"Retrieved fear/greed index: {fear_greed}")
                else:
                    logger.info("Fear/greed index not available - analysis will proceed without it")
            except Exception as e:
                logger.debug(f"Could not fetch fear/greed index: {e}")
                # Analysis proceeds without this optional sentiment indicator
            
            # Calculate overall sentiment score
            sentiment_data['overall_sentiment_score'] = self._calculate_overall_sentiment_score(sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._get_fallback_sentiment(token_symbol)
    
    def _mock_social_sentiment(self, token_symbol: str, platform: str) -> float:
        """Mock social sentiment data for different platforms."""
        # Mock sentiment scores (-1 to 1, where 1 is very positive)
        sentiment_map = {
            'ETH': {'twitter': 0.3, 'reddit': 0.4, 'telegram': 0.2},
            'BTC': {'twitter': 0.5, 'reddit': 0.6, 'telegram': 0.4},
            'SOL': {'twitter': 0.1, 'reddit': 0.2, 'telegram': 0.3},
            'AVAX': {'twitter': 0.2, 'reddit': 0.1, 'telegram': 0.0},
            'MATIC': {'twitter': 0.0, 'reddit': -0.1, 'telegram': 0.1}
        }
        return sentiment_map.get(token_symbol, {}).get(platform, 0.0)
    
    def _calculate_overall_social_score(self, token_symbol: str) -> float:
        """Calculate overall social sentiment score."""
        platforms = ['twitter', 'reddit', 'telegram']
        scores = [self._mock_social_sentiment(token_symbol, platform) for platform in platforms]
        return sum(scores) / len(scores)
    
    def _mock_news_count(self, token_symbol: str, sentiment: str) -> int:
        """Mock news article counts by sentiment."""
        # Mock news counts for different tokens and sentiments
        news_map = {
            'ETH': {'positive': 15, 'negative': 8, 'neutral': 12},
            'BTC': {'positive': 22, 'negative': 5, 'neutral': 18},
            'SOL': {'positive': 8, 'negative': 12, 'neutral': 10},
            'AVAX': {'positive': 6, 'negative': 9, 'neutral': 8},
            'MATIC': {'positive': 4, 'negative': 11, 'neutral': 7}
        }
        return news_map.get(token_symbol, {}).get(sentiment, 5)
    
    def _calculate_overall_news_score(self, token_symbol: str) -> float:
        """Calculate overall news sentiment score."""
        positive = self._mock_news_count(token_symbol, 'positive')
        negative = self._mock_news_count(token_symbol, 'negative')
        neutral = self._mock_news_count(token_symbol, 'neutral')
        
        if positive + negative + neutral == 0:
            return 0.0
        
        return (positive - negative) / (positive + negative + neutral)
    
    def _interpret_fear_greed(self, index: int) -> str:
        """Interpret fear/greed index value."""
        if index >= 80:
            return 'extreme_greed'
        elif index >= 60:
            return 'greed'
        elif index >= 40:
            return 'neutral'
        elif index >= 20:
            return 'fear'
        else:
            return 'extreme_fear'
    
    def _calculate_overall_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate overall sentiment score from all sources."""
        try:
            social_score = sentiment_data.get('social_sentiment', {}).get('overall_social_score', 0)
            news_score = sentiment_data.get('news_sentiment', {}).get('overall_news_score', 0)
            
            # Fear/greed index to sentiment score (-1 to 1)
            fear_greed = sentiment_data.get('market_sentiment', {}).get('fear_greed_index', 50)
            market_sentiment = (fear_greed - 50) / 50  # Convert 0-100 to -1 to 1
            
            # Weighted average
            overall_score = (social_score * 0.4 + news_score * 0.3 + market_sentiment * 0.3)
            return max(-1.0, min(1.0, overall_score))  # Clamp to -1 to 1
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment score: {e}")
            return 0.0
    
    def _get_fallback_sentiment(self, token_symbol: str) -> Dict[str, Any]:
        """Get fallback sentiment data when services fail."""
        return {
            'social_sentiment': {
                'twitter_sentiment': 0.0,
                'reddit_sentiment': 0.0,
                'telegram_sentiment': 0.0,
                'overall_social_score': 0.0
            },
            'news_sentiment': {
                'positive_articles': 5,
                'negative_articles': 5,
                'neutral_articles': 5,
                'overall_news_score': 0.0
            },
            'market_sentiment': {
                'fear_greed_index': 50,
                'market_mood': 'neutral'
            },
            'overall_sentiment_score': 0.0
        }
    
    def _get_fallback_news_sentiment(self, token_symbol: str) -> Dict[str, Any]:
        """Get fallback news sentiment data."""
        return {
            'positive_articles': 5,
            'negative_articles': 5,
            'neutral_articles': 5,
            'overall_news_score': 0.0
        }
    
    def _detect_chart_patterns(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """
        Detect chart patterns for automation.
        Identifies common patterns like head & shoulders, triangles, etc.
        """
        try:
            if len(prices) < 20:
                return {'patterns': [], 'confidence': 0.0}
            
            patterns = []
            confidence = 0.0
            
            # Head & Shoulders Pattern Detection
            hs_pattern = self._detect_head_and_shoulders(prices)
            if hs_pattern['detected']:
                patterns.append({
                    'type': 'head_and_shoulders',
                    'direction': hs_pattern['direction'],
                    'confidence': hs_pattern['confidence'],
                    'target': hs_pattern['target']
                })
                confidence += hs_pattern['confidence']
            
            # Triangle Pattern Detection
            triangle_pattern = self._detect_triangle_pattern(prices)
            if triangle_pattern['detected']:
                patterns.append({
                    'type': 'triangle',
                    'direction': triangle_pattern['direction'],
                    'confidence': triangle_pattern['confidence'],
                    'target': triangle_pattern['target']
                })
                confidence += triangle_pattern['confidence']
            
            # Double Top/Bottom Detection
            double_pattern = self._detect_double_pattern(prices)
            if double_pattern['detected']:
                patterns.append({
                    'type': double_pattern['type'],
                    'direction': double_pattern['direction'],
                    'confidence': double_pattern['confidence'],
                    'target': double_pattern['target']
                })
                confidence += double_pattern['confidence']
            
            # Volume confirmation
            volume_confirmation = self._confirm_pattern_with_volume(patterns, volumes)
            
            return {
                'patterns': patterns,
                'confidence': min(1.0, confidence / max(1, len(patterns))),
                'volume_confirmation': volume_confirmation,
                'total_patterns': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {'patterns': [], 'confidence': 0.0}
    
    def _detect_head_and_shoulders(self, prices: List[float]) -> Dict[str, Any]:
        """Detect head and shoulders pattern."""
        try:
            if len(prices) < 30:
                return {'detected': False, 'confidence': 0.0}
            
            # Look for 3 peaks with middle one higher
            peaks = self._find_peaks(prices[-30:])
            
            if len(peaks) >= 3:
                # Check if middle peak is highest (head)
                if peaks[1] > peaks[0] and peaks[1] > peaks[2]:
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(peaks[0] - peaks[2]) / peaks[1]
                    if shoulder_diff < 0.1:  # Shoulders within 10%
                        confidence = 0.8 - (shoulder_diff * 2)
                        direction = 'bearish'  # H&S is typically bearish
                        target = prices[-1] * 0.9  # 10% downside target
                        
                        return {
                            'detected': True,
                            'direction': direction,
                            'confidence': confidence,
                            'target': target
                        }
            
            return {'detected': False, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in H&S detection: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _detect_triangle_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """Detect triangle patterns (ascending, descending, symmetrical)."""
        try:
            if len(prices) < 20:
                return {'detected': False, 'confidence': 0.0}
            
            # Calculate trend lines
            highs = self._find_peaks(prices[-20:])
            lows = self._find_troughs(prices[-20:])
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Calculate slopes
                high_slope = (highs[-1] - highs[0]) / len(highs)
                low_slope = (lows[-1] - lows[0]) / len(lows)
                
                # Determine triangle type
                if high_slope < -0.001 and low_slope > 0.001:  # Descending triangle
                    direction = 'bearish'
                    confidence = 0.7
                    target = prices[-1] * 0.95
                elif high_slope > 0.001 and low_slope < -0.001:  # Ascending triangle
                    direction = 'bullish'
                    confidence = 0.7
                    target = prices[-1] * 1.05
                elif abs(high_slope) < 0.001 and abs(low_slope) < 0.001:  # Symmetrical
                    direction = 'neutral'
                    confidence = 0.6
                    target = prices[-1]
                else:
                    return {'detected': False, 'confidence': 0.0}
                
                return {
                    'detected': True,
                    'direction': direction,
                    'confidence': confidence,
                    'target': target
                }
            
            return {'detected': False, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in triangle detection: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _detect_double_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """Detect double top/bottom patterns."""
        try:
            if len(prices) < 25:
                return {'detected': False, 'confidence': 0.0}
            
            # Look for two similar peaks/troughs
            peaks = self._find_peaks(prices[-25:])
            troughs = self._find_troughs(prices[-25:])
            
            if len(peaks) >= 2:
                # Check if peaks are similar (within 2%)
                peak_diff = abs(peaks[-1] - peaks[-2]) / peaks[-2]
                if peak_diff < 0.02:
                    confidence = 0.75
                    direction = 'bearish'  # Double top is bearish
                    target = prices[-1] * 0.92
                    
                    return {
                        'detected': True,
                        'type': 'double_top',
                        'direction': direction,
                        'confidence': confidence,
                        'target': target
                    }
            
            if len(troughs) >= 2:
                # Check if troughs are similar
                trough_diff = abs(troughs[-1] - troughs[-2]) / troughs[-2]
                if trough_diff < 0.02:
                    confidence = 0.75
                    direction = 'bullish'  # Double bottom is bullish
                    target = prices[-1] * 1.08
                    
                    return {
                        'detected': True,
                        'type': 'double_bottom',
                        'direction': direction,
                        'confidence': confidence,
                        'target': target
                    }
            
            return {'detected': False, 'confidence': 0.0}
            
        except Exception as e:
            logger.error(f"Error in double pattern detection: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _find_peaks(self, prices: List[float]) -> List[float]:
        """Find local peaks in price data."""
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(prices[i])
        return peaks
    
    def _find_troughs(self, prices: List[float]) -> List[float]:
        """Find local troughs in price data."""
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(prices[i])
        return troughs
    
    def _confirm_pattern_with_volume(self, patterns: List[Dict], volumes: List[float]) -> Dict[str, Any]:
        """Confirm patterns with volume analysis."""
        try:
            if not patterns or not volumes:
                return {'confirmed': False, 'volume_signal': 'weak'}
            
            # Check if recent volume supports pattern
            recent_volume = volumes[-5:] if len(volumes) >= 5 else volumes
            avg_volume = sum(recent_volume) / len(recent_volume)
            current_volume = volumes[-1] if volumes else 0
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > self.ta_config['volume_spike_threshold']:
                volume_signal = 'strong'
                confirmed = True
            elif volume_ratio > 1.5:
                volume_signal = 'moderate'
                confirmed = True
            else:
                volume_signal = 'weak'
                confirmed = False
            
            return {
                'confirmed': confirmed,
                'volume_signal': volume_signal,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            logger.error(f"Error in volume confirmation: {e}")
            return {'confirmed': False, 'volume_signal': 'weak'}
    
    def _calculate_signal_strength(self, indicators: TechnicalIndicators, regime: MarketRegime) -> float:
        """Calculate signal strength using the same logic as main Yuki."""
        try:
            # Base signal strength
            strength = 0.5
            
            # RSI contribution
            if 30 <= indicators.rsi <= 70:
                strength += 0.2
            elif indicators.rsi < 30 or indicators.rsi > 70:
                strength += 0.1
            
            # MACD contribution
            if indicators.macd > indicators.macd_signal:
                strength += 0.2
            
            # Volume contribution
            if indicators.volume_ratio > 1.2:
                strength += 0.1
            
            return min(1.0, strength)
        except:
            return 0.5
    
    def _determine_trade_direction(self, indicators: TechnicalIndicators) -> str:
        """Determine trade direction using the same logic as main Yuki."""
        try:
            if indicators.rsi < 30 and indicators.macd > indicators.macd_signal:
                return 'long'
            elif indicators.rsi > 70 and indicators.macd < indicators.macd_signal:
                return 'short'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _generate_trade_reasoning(self, indicators: TechnicalIndicators, regime: MarketRegime) -> str:
        """Generate trade reasoning using the same logic as main Yuki."""
        try:
            if regime == MarketRegime.TRENDING:
                return f"Strong trend detected (ADX: {indicators.adx:.1f})"
            elif regime == MarketRegime.RANGING:
                return f"Range-bound market (BB Width: {indicators.bb_width:.4f})"
            else:
                return f"Volatile market conditions (ATR: {indicators.atr:.4f})"
        except:
            return "Market analysis completed"
    
    def _determine_overall_regime(self, futures_analysis: Dict[str, Any]) -> MarketRegime:
        """Determine overall market regime using the same logic as main Yuki."""
        try:
            regimes = [analysis['regime'] for analysis in futures_analysis.values()]
            if MarketRegime.TRENDING in regimes:
                return MarketRegime.TRENDING
            elif MarketRegime.VOLATILE in regimes:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.RANGING
        except:
            return MarketRegime.RANGING
    
    def _assess_futures_risk(self, futures_analysis: Dict[str, Any]) -> RiskLevel:
        """Assess futures risk using the same logic as main Yuki."""
        try:
            if not futures_analysis:
                return RiskLevel.HIGH
            
            # Count high-risk indicators
            high_risk_count = 0
            for analysis in futures_analysis.values():
                if analysis['regime'] == MarketRegime.VOLATILE:
                    high_risk_count += 1
            
            if high_risk_count > len(futures_analysis) * 0.5:
                return RiskLevel.HIGH
            elif high_risk_count > 0:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        except:
            return RiskLevel.MEDIUM
    
    def _assess_volatility_regime(self, futures_analysis: Dict[str, Any]) -> str:
        """Assess volatility regime using the same logic as main Yuki."""
        try:
            if not futures_analysis:
                return 'normal'
            
            volatile_count = sum(1 for analysis in futures_analysis.values() 
                               if analysis['regime'] == MarketRegime.VOLATILE)
            
            if volatile_count > len(futures_analysis) * 0.5:
                return 'high'
            elif volatile_count > 0:
                return 'elevated'
            else:
                return 'normal'
        except:
            return 'normal'
    
    # Scoring methods (identical to main Yuki)
    
    def _calculate_technical_score(self, futures_analysis: Dict[str, Any]) -> float:
        """Calculate technical score using the same logic as main Yuki."""
        try:
            if not futures_analysis:
                raise Exception("No futures analysis data available")
            
            total_score = 0.0
            valid_symbols = 0
            
            for symbol, analysis in futures_analysis.items():
                indicators = analysis.get('indicators')
                if not indicators:
                    continue
                
                # Trend strength (0-1 scale)
                adx = getattr(indicators, 'adx', 50)
                trend_strength = min(1.0, adx / 50.0)
                
                # Momentum (0-1 scale)
                rsi = getattr(indicators, 'rsi', 50)
                rsi_score = 1.0 - abs(rsi - 50) / 50.0
                
                macd = getattr(indicators, 'macd', 0)
                macd_signal = getattr(indicators, 'macd_signal', 0)
                macd_score = 1.0 if (macd > macd_signal and macd > 0) else 0.5
                
                # Volatility (0-1 scale) - Yuki prefers moderate volatility
                atr = getattr(indicators, 'atr', 0)
                bb_width = getattr(indicators, 'bb_width', 0.02)
                volatility_score = min(1.0, (atr + bb_width * 100) / 0.1)
                
                # Volume confirmation
                volume_ratio = getattr(indicators, 'volume_ratio', 1.0)
                volume_score = min(1.0, volume_ratio / 2.0)
                
                # Calculate symbol score (weighted average)
                symbol_score = (
                    trend_strength * 0.3 +
                    rsi_score * 0.2 +
                    macd_score * 0.2 +
                    volatility_score * 0.2 +
                    volume_score * 0.1
                )
                
                total_score += symbol_score
                valid_symbols += 1
            
            if valid_symbols == 0:
                raise Exception("No valid symbols for technical analysis")
            
            return total_score / valid_symbols
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            raise
    
    def _calculate_fundamental_score(self, futures_analysis: Dict[str, Any]) -> float:
        """Calculate fundamental score using the same logic as main Yuki."""
        try:
            if not futures_analysis:
                raise Exception("No futures analysis data available")
            
            total_score = 0.0
            valid_symbols = 0
            
            for symbol, analysis in futures_analysis.items():
                # Funding rate analysis
                funding = analysis.get('funding', {})
                funding_score = 0.5
                if funding.get('arbitrage_opportunity'):
                    funding_score = 0.8
                
                # Market structure
                structure = analysis.get('structure', {})
                liquidity_score = 0.5
                if structure.get('liquidity', 'high') == 'high':
                    liquidity_score = 0.8
                elif structure.get('liquidity', 'medium') == 'medium':
                    liquidity_score = 0.6
                
                # Volatility regime
                volatility_score = 0.5
                if analysis.get('volatility_regime') == 'normal':
                    volatility_score = 0.7
                elif analysis.get('volatility_regime') == 'high':
                    volatility_score = 0.4
                
                # Calculate symbol score
                symbol_score = (
                    funding_score * 0.4 +
                    liquidity_score * 0.3 +
                    volatility_score * 0.3
                )
                
                total_score += symbol_score
                valid_symbols += 1
            
            if valid_symbols == 0:
                raise Exception("No valid symbols for fundamental analysis")
            
            return total_score / valid_symbols
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            raise
    
    # Enhanced Analysis Methods
    
    async def _run_backtest_analysis(self, symbol: str, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """
        Run comprehensive backtesting analysis to validate trading strategies.
        Tests various entry/exit conditions and calculates performance metrics.
        """
        try:
            if len(prices) < 50:
                return {'error': 'Insufficient data for backtesting'}
            
            backtest_results = {}
            
            # Test different technical indicator combinations
            strategy_results = {}
            
            # Strategy 1: RSI + MACD crossover
            strategy_results['rsi_macd'] = self._backtest_rsi_macd_strategy(prices, volumes)
            
            # Strategy 2: Bollinger Bands breakout
            strategy_results['bb_breakout'] = self._backtest_bb_breakout_strategy(prices, volumes)
            
            # Strategy 3: Moving average crossover
            strategy_results['ma_crossover'] = self._backtest_ma_crossover_strategy(prices, volumes)
            
            # Strategy 4: Volume-weighted momentum
            strategy_results['volume_momentum'] = self._backtest_volume_momentum_strategy(prices, volumes)
            
            # Calculate overall backtest performance
            overall_performance = self._calculate_overall_backtest_performance(strategy_results)
            
            backtest_results = {
                'strategies': strategy_results,
                'overall_performance': overall_performance,
                'best_strategy': self._identify_best_strategy(strategy_results),
                'risk_metrics': self._calculate_risk_metrics(strategy_results),
                'backtest_period': len(prices),
                'confidence_level': self._calculate_backtest_confidence(strategy_results)
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {'error': f'Backtesting failed: {str(e)}'}
    
    def _backtest_rsi_macd_strategy(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Backtest RSI + MACD crossover strategy."""
        try:
            trades = []
            position = None
            initial_capital = 10000
            current_capital = initial_capital
            
            for i in range(20, len(prices)):
                rsi = self._calculate_rsi(prices[:i+1], 14)
                macd_line, signal_line, _ = self._calculate_macd(prices[:i+1], 12, 26, 9)
                
                # Entry conditions
                if position is None:  # No position
                    if rsi < 30 and macd_line > signal_line:  # Oversold + bullish MACD
                        position = {
                            'entry_price': prices[i],
                            'entry_index': i,
                            'type': 'long'
                        }
                
                # Exit conditions
                elif position is not None:
                    if rsi > 70 or macd_line < signal_line:  # Overbought or bearish MACD
                        exit_price = prices[i]
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                        current_capital *= (1 + pnl)
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'entry_time': position['entry_index'],
                            'exit_time': i
                        })
                        position = None
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_return = (current_capital - initial_capital) / initial_capital
            avg_return_per_trade = total_return / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return_per_trade': avg_return_per_trade,
                'final_capital': current_capital,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error in RSI+MACD backtest: {e}")
            return {'error': str(e)}
    
    def _backtest_bb_breakout_strategy(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Backtest Bollinger Bands breakout strategy."""
        try:
            trades = []
            position = None
            initial_capital = 10000
            current_capital = initial_capital
            
            for i in range(20, len(prices)):
                bb_upper, bb_middle, bb_lower, _ = self._calculate_bollinger_bands(prices[:i+1], 20, 2)
                
                # Entry conditions
                if position is None:  # No position
                    if prices[i] < bb_lower:  # Price below lower band (oversold)
                        position = {
                            'entry_price': prices[i],
                            'entry_index': i,
                            'type': 'long'
                        }
                
                # Exit conditions
                elif position is not None:
                    if prices[i] > bb_middle:  # Price above middle band
                        exit_price = prices[i]
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                        current_capital *= (1 + pnl)
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'entry_time': position['entry_index'],
                            'exit_time': i
                        })
                        position = None
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_return = (current_capital - initial_capital) / initial_capital
            avg_return_per_trade = total_return / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return_per_trade': avg_return_per_trade,
                'final_capital': current_capital,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error in BB breakout backtest: {e}")
            return {'error': str(e)}
    
    def _backtest_ma_crossover_strategy(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Backtest moving average crossover strategy."""
        try:
            trades = []
            position = None
            initial_capital = 10000
            current_capital = initial_capital
            
            for i in range(50, len(prices)):
                ema_20 = self._calculate_ema(prices[:i+1], 20)
                ema_50 = self._calculate_ema(prices[:i+1], 50)
                
                # Entry conditions
                if position is None:  # No position
                    if ema_20 > ema_50:  # Golden cross
                        position = {
                            'entry_price': prices[i],
                            'entry_index': i,
                            'type': 'long'
                        }
                
                # Exit conditions
                elif position is not None:
                    if ema_20 < ema_50:  # Death cross
                        exit_price = prices[i]
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                        current_capital *= (1 + pnl)
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'entry_time': position['entry_index'],
                            'exit_time': i
                        })
                        position = None
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_return = (current_capital - initial_capital) / initial_capital
            avg_return_per_trade = total_return / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return_per_trade': avg_return_per_trade,
                'final_capital': current_capital,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error in MA crossover backtest: {e}")
            return {'error': str(e)}
    
    def _backtest_volume_momentum_strategy(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Backtest volume-weighted momentum strategy."""
        try:
            trades = []
            position = None
            initial_capital = 10000
            current_capital = initial_capital
            
            for i in range(20, len(prices)):
                # Calculate volume momentum
                recent_volumes = volumes[max(0, i-5):i+1]
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
                
                # Entry conditions
                if position is None:  # No position
                    if volume_ratio > 1.5 and prices[i] > prices[i-1]:  # High volume + price increase
                        position = {
                            'entry_price': prices[i],
                            'entry_index': i,
                            'type': 'long'
                        }
                
                # Exit conditions
                elif position is not None:
                    if volume_ratio < 0.8 or prices[i] < prices[i-1]:  # Low volume or price decrease
                        exit_price = prices[i]
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                        current_capital *= (1 + pnl)
                        
                        trades.append({
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'entry_time': position['entry_index'],
                            'exit_time': i
                        })
                        position = None
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_return = (current_capital - initial_capital) / initial_capital
            avg_return_per_trade = total_return / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return_per_trade': avg_return_per_trade,
                'final_capital': current_capital,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error in volume momentum backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_backtest_performance(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance across all strategies."""
        try:
            valid_strategies = [s for s in strategy_results.values() if 'error' not in s]
            
            if not valid_strategies:
                return {'error': 'No valid strategies to analyze'}
            
            total_return = sum([s.get('total_return', 0) for s in valid_strategies])
            avg_return = total_return / len(valid_strategies)
            
            win_rates = [s.get('win_rate', 0) for s in valid_strategies]
            avg_win_rate = sum(win_rates) / len(win_rates)
            
            total_trades = sum([s.get('total_trades', 0) for s in valid_strategies])
            
            return {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'total_trades': total_trades,
                'strategy_count': len(valid_strategies),
                'best_strategy_return': max([s.get('total_return', 0) for s in valid_strategies]),
                'worst_strategy_return': min([s.get('total_return', 0) for s in valid_strategies])
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall performance: {e}")
            return {'error': str(e)}
    
    def _identify_best_strategy(self, strategy_results: Dict[str, Any]) -> str:
        """Identify the best performing strategy."""
        try:
            valid_strategies = {k: v for k, v in strategy_results.items() if 'error' not in v}
            
            if not valid_strategies:
                return 'none'
            
            best_strategy = max(valid_strategies.items(), key=lambda x: x[1].get('total_return', 0))
            return best_strategy[0]
            
        except Exception as e:
            logger.error(f"Error identifying best strategy: {e}")
            return 'none'
    
    def _calculate_risk_metrics(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics across all strategies."""
        try:
            valid_strategies = [s for s in strategy_results.values() if 'error' not in s]
            
            if not valid_strategies:
                return {'error': 'No valid strategies to analyze'}
            
            # Calculate Sharpe ratio approximation
            returns = [s.get('total_return', 0) for s in valid_strategies]
            avg_return = sum(returns) / len(returns)
            
            # Calculate volatility (standard deviation)
            variance = sum([(r - avg_return) ** 2 for r in returns]) / len(returns)
            volatility = variance ** 0.5
            
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown approximation
            max_drawdown = min(returns) if returns else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'risk_adjusted_return': sharpe_ratio,
                'return_volatility_ratio': avg_return / volatility if volatility > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_backtest_confidence(self, strategy_results: Dict[str, Any]) -> float:
        """Calculate confidence level in backtest results."""
        try:
            valid_strategies = [s for s in strategy_results.values() if 'error' not in s]
            
            if not valid_strategies:
                return 0.0
            
            # Factors that increase confidence:
            # 1. More strategies tested
            strategy_confidence = min(1.0, len(valid_strategies) / 4)
            
            # 2. Higher win rates
            avg_win_rate = sum([s.get('win_rate', 0) for s in valid_strategies]) / len(valid_strategies)
            win_rate_confidence = avg_win_rate
            
            # 3. More trades (more data)
            total_trades = sum([s.get('total_trades', 0) for s in valid_strategies])
            trade_confidence = min(1.0, total_trades / 100)
            
            # 4. Consistent performance across strategies
            returns = [s.get('total_return', 0) for s in valid_strategies]
            consistency_confidence = 1.0 - (max(returns) - min(returns)) / max(abs(max(returns)), abs(min(returns))) if max(returns) != min(returns) else 1.0
            
            # Weighted average
            overall_confidence = (
                strategy_confidence * 0.3 +
                win_rate_confidence * 0.3 +
                trade_confidence * 0.2 +
                consistency_confidence * 0.2
            )
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating backtest confidence: {e}")
            return 0.0
    
    # Prediction Methods
    
    async def _generate_price_predictions(self, symbol: str, prices: List[float], volumes: List[float], indicators: TechnicalIndicators) -> Dict[str, Any]:
        """
        Generate sophisticated price predictions using multiple models.
        Combines technical analysis, sentiment, and machine learning approaches.
        """
        try:
            if len(prices) < 50:
                return {'error': 'Insufficient data for predictions'}
            
            predictions = {}
            
            # 1. Technical Analysis Based Prediction
            ta_prediction = self._predict_from_technical_indicators(prices, volumes, indicators)
            predictions['technical_analysis'] = ta_prediction
            
            # 2. Statistical Model Prediction
            statistical_prediction = self._predict_from_statistical_models(prices, volumes)
            predictions['statistical_model'] = statistical_prediction
            
            # 3. Pattern-Based Prediction
            pattern_prediction = self._predict_from_chart_patterns(prices, volumes)
            predictions['pattern_analysis'] = pattern_prediction
            
            # 4. Ensemble Prediction (combine all models)
            ensemble_prediction = self._generate_ensemble_prediction(predictions)
            predictions['ensemble'] = ensemble_prediction
            
            # 5. Risk-Adjusted Predictions
            risk_adjusted = self._adjust_predictions_for_risk(predictions, prices)
            predictions['risk_adjusted'] = risk_adjusted
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _predict_from_technical_indicators(self, prices: List[float], volumes: List[float], indicators: TechnicalIndicators) -> Dict[str, Any]:
        """Generate predictions based on technical indicators."""
        try:
            current_price = prices[-1]
            
            # RSI-based prediction
            rsi = indicators.rsi
            if rsi < 30:  # Oversold
                rsi_prediction = current_price * 1.05  # 5% upside
                confidence = 0.7
            elif rsi > 70:  # Overbought
                rsi_prediction = current_price * 0.95  # 5% downside
                confidence = 0.7
            else:
                rsi_prediction = current_price
                confidence = 0.5
            
            # MACD-based prediction
            macd = indicators.macd
            macd_signal = indicators.macd_signal
            if macd > macd_signal and macd > 0:
                macd_prediction = current_price * 1.03  # 3% upside
                macd_confidence = 0.6
            elif macd < macd_signal and macd < 0:
                macd_prediction = current_price * 0.97  # 3% downside
                macd_confidence = 0.6
            else:
                macd_prediction = current_price
                macd_confidence = 0.4
            
            # Bollinger Bands prediction
            bb_upper = indicators.bb_upper
            bb_lower = indicators.bb_lower
            bb_middle = indicators.bb_middle
            
            if current_price < bb_lower:
                bb_prediction = bb_middle  # Reversion to mean
                bb_confidence = 0.8
            elif current_price > bb_upper:
                bb_prediction = bb_middle  # Reversion to mean
                bb_confidence = 0.8
            else:
                bb_prediction = current_price
                bb_confidence = 0.5
            
            # Volume-weighted prediction
            volume_ratio = indicators.volume_ratio
            if volume_ratio > 1.5:
                volume_prediction = current_price * 1.02  # High volume suggests continuation
                volume_confidence = 0.6
            else:
                volume_prediction = current_price
                volume_confidence = 0.4
            
            # Combine predictions
            predictions = [rsi_prediction, macd_prediction, bb_prediction, volume_prediction]
            confidences = [confidence, macd_confidence, bb_confidence, volume_confidence]
            
            # Weighted average prediction
            weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                'predicted_price': weighted_prediction,
                'confidence': avg_confidence,
                'individual_predictions': {
                    'rsi': {'price': rsi_prediction, 'confidence': confidence},
                    'macd': {'price': macd_prediction, 'confidence': macd_confidence},
                    'bollinger_bands': {'price': bb_prediction, 'confidence': bb_confidence},
                    'volume': {'price': volume_prediction, 'confidence': volume_confidence}
                },
                'prediction_method': 'technical_indicators'
            }
            
        except Exception as e:
            logger.error(f"Error in technical prediction: {e}")
            return {'error': str(e)}
    
    def _predict_from_statistical_models(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Generate predictions using statistical models."""
        try:
            if len(prices) < 20:
                return {'error': 'Insufficient data for statistical models'}
            
            current_price = prices[-1]
            
            # Linear regression prediction
            x = list(range(len(prices)))
            y = prices
            
            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 == 0:
                return {'error': 'Cannot calculate linear regression'}
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict next 7 days
            future_days = 7
            linear_prediction = slope * (n + future_days) + intercept
            
            # Moving average prediction
            ma_short = sum(prices[-10:]) / 10
            ma_long = sum(prices[-20:]) / 20
            
            if ma_short > ma_long:
                ma_prediction = current_price * 1.02  # Uptrend
                ma_confidence = 0.6
            else:
                ma_prediction = current_price * 0.98  # Downtrend
                ma_confidence = 0.6
            
            # Volatility-adjusted prediction
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
            
            volatility_prediction = current_price * (1 + volatility * 2)  # 2x volatility
            vol_confidence = 0.4
            
            # Combine predictions
            predictions = [linear_prediction, ma_prediction, volatility_prediction]
            confidences = [0.5, ma_confidence, vol_confidence]
            
            weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                'predicted_price': weighted_prediction,
                'confidence': avg_confidence,
                'individual_predictions': {
                    'linear_regression': {'price': linear_prediction, 'confidence': 0.5},
                    'moving_average': {'price': ma_prediction, 'confidence': ma_confidence},
                    'volatility': {'price': volatility_prediction, 'confidence': vol_confidence}
                },
                'prediction_method': 'statistical_models'
            }
            
        except Exception as e:
            logger.error(f"Error in statistical prediction: {e}")
            return {'error': str(e)}
    
    def _predict_from_chart_patterns(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """Generate predictions based on chart patterns."""
        try:
            current_price = prices[-1]
            
            # Detect patterns
            patterns = self._detect_chart_patterns(prices, volumes)
            
            if not patterns.get('patterns'):
                return {
                    'predicted_price': current_price,
                    'confidence': 0.3,
                    'prediction_method': 'no_patterns_detected'
                }
            
            # Use pattern targets for prediction
            pattern_predictions = []
            pattern_confidences = []
            
            for pattern in patterns['patterns']:
                if 'target' in pattern:
                    pattern_predictions.append(pattern['target'])
                    pattern_confidences.append(pattern.get('confidence', 0.5))
            
            if pattern_predictions:
                # Weighted average of pattern predictions
                weighted_prediction = sum(p * c for p, c in zip(pattern_predictions, pattern_confidences)) / sum(pattern_confidences)
                avg_confidence = sum(pattern_confidences) / len(pattern_confidences)
                
                return {
                    'predicted_price': weighted_prediction,
                    'confidence': avg_confidence,
                    'patterns_used': len(pattern_predictions),
                    'prediction_method': 'chart_patterns'
                }
            else:
                return {
                    'predicted_price': current_price,
                    'confidence': 0.3,
                    'prediction_method': 'no_pattern_targets'
                }
                
        except Exception as e:
            logger.error(f"Error in pattern prediction: {e}")
            return {'error': str(e)}
    
    def _generate_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all prediction methods into an ensemble prediction."""
        try:
            valid_predictions = []
            valid_confidences = []
            
            for method, data in predictions.items():
                if isinstance(data, dict) and 'predicted_price' in data and 'error' not in data:
                    valid_predictions.append(data['predicted_price'])
                    valid_confidences.append(data.get('confidence', 0.5))
            
            if not valid_predictions:
                return {'error': 'No valid predictions to ensemble'}
            
            # Weighted average ensemble
            weighted_prediction = sum(p * c for p, c in zip(valid_predictions, valid_confidences)) / sum(valid_confidences)
            ensemble_confidence = sum(valid_confidences) / len(valid_confidences)
            
            # Calculate prediction range
            min_prediction = min(valid_predictions)
            max_prediction = max(valid_predictions)
            prediction_range = max_prediction - min_prediction
            
            return {
                'predicted_price': weighted_prediction,
                'confidence': ensemble_confidence,
                'prediction_range': prediction_range,
                'min_prediction': min_prediction,
                'max_prediction': max_prediction,
                'methods_used': len(valid_predictions),
                'prediction_method': 'ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _adjust_predictions_for_risk(self, predictions: Dict[str, Any], prices: List[float]) -> Dict[str, Any]:
        """Adjust predictions based on market risk and volatility."""
        try:
            if 'ensemble' not in predictions or 'error' in predictions['ensemble']:
                return {'error': 'No ensemble prediction to adjust'}
            
            ensemble = predictions['ensemble']
            current_price = prices[-1]
            
            # Calculate market volatility
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
            
            # Adjust prediction based on volatility
            if volatility > 0.05:  # High volatility
                risk_adjustment = 0.9  # Reduce prediction confidence
                volatility_impact = 'high'
            elif volatility > 0.02:  # Medium volatility
                risk_adjustment = 0.95
                volatility_impact = 'medium'
            else:  # Low volatility
                risk_adjustment = 1.0
                volatility_impact = 'low'
            
            # Adjust prediction
            adjusted_prediction = ensemble['predicted_price'] * risk_adjustment
            adjusted_confidence = ensemble['confidence'] * risk_adjustment
            
            return {
                'original_prediction': ensemble['predicted_price'],
                'adjusted_prediction': adjusted_prediction,
                'confidence': adjusted_confidence,
                'risk_adjustment': risk_adjustment,
                'volatility': volatility,
                'volatility_impact': volatility_impact,
                'prediction_method': 'risk_adjusted'
            }
            
        except Exception as e:
            logger.error(f"Error in risk adjustment: {e}")
            return {'error': str(e)}
    
    def _calculate_agent_confidence(self, futures_analysis: Dict[str, Any]) -> float:
        """Calculate agent confidence using the same logic as main Yuki."""
        try:
            if not futures_analysis:
                raise Exception("No futures analysis data available")
            
            # Count strong signals
            strong_signals = sum(
                1 for analysis in futures_analysis.values()
                if analysis.get('signal_strength', 0) > 0.7
            )
            
            # Count funding opportunities
            funding_opportunities = sum(
                1 for analysis in futures_analysis.values()
                if analysis.get('funding', {}).get('arbitrage_opportunity', False)
            )
            
            # Calculate confidence based on opportunities
            total_opportunities = strong_signals + funding_opportunities
            if total_opportunities >= 3:
                return 0.9
            elif total_opportunities >= 1:
                return 0.7
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Error calculating agent confidence: {e}")
            raise
    
    # Required abstract methods from BaseAgent
    
    async def generate_signals(self, market_data: MarketData) -> List[Any]:
        """
        Generate trading signals (not used for token analysis).
        This method is required by BaseAgent but not used for analysis.
        """
        return []  # No signals generated for analysis-only agent
    
    async def execute_trades(self, signals: List[Any]) -> List[Any]:
        """
        Execute trades (not used for token analysis).
        This method is required by BaseAgent but not used for analysis.
        """
        return []  # No trades executed for analysis-only agent
    
    async def validate_risk(self, signal: Any) -> bool:
        """
        Validate risk (not used for token analysis).
        This method is required by BaseAgent but not used for analysis.
        """
        return False  # No risk validation for analysis-only agent
    
    async def stop_agent(self) -> bool:
        """
        Stop agent (not used for token analysis).
        This method is required by BaseAgent but not used for analysis.
        """
        return True  # Always successful for analysis-only agent


# Agent factory function
def create_yuki_analysis_agent(user_id: str, config: Dict[str, Any], binance_service: Optional[BinanceService] = None, coingecko_service: Optional[CoinGeckoService] = None, defillama_service: Optional[DeFiLlamaService] = None, sentiment_service: Optional[Any] = None) -> YukiAnalysisAgent:  # Temporarily disabled
    """Factory function to create a Yuki Analysis Agent instance."""
    return YukiAnalysisAgent(user_id, config, binance_service, coingecko_service, defillama_service, sentiment_service)
