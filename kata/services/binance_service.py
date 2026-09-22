"""
Binance API Service for Yuki Agent

Provides advanced technical indicators and market data including:
- Real-time orderbook data and market depth
- Advanced technical indicators (RSI, MACD, Bollinger Bands)
- Funding rates and perpetual futures data  
- Volume profile and market microstructure
- Cross-exchange arbitrage opportunities
"""

import asyncio
import inspect
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import ccxt
import pandas as pd
import numpy as np
from ccxt.base.errors import NetworkError, ExchangeError
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class BinanceRateLimiter:
    """Advanced rate limiter for Binance API to prevent IP bans."""
    
    def __init__(self):
        # Binance rate limits (conservative estimates)
        self.limits = {
            'requests_per_minute': 1000,  # Conservative limit (actual is 1200)
            'requests_per_second': 16,    # Conservative limit (actual is ~20)
            'weight_per_minute': 5000     # For weighted endpoints
        }
        
        # Track request history
        self.request_times = deque()
        self.request_weights = deque()
        self.last_request_time = 0
        self.min_request_interval = 1.0 / self.limits['requests_per_second']
        
        # Circuit breaker for ban detection
        self.consecutive_errors = 0
        self.ban_detected_until = 0
        self.max_consecutive_errors = 3
        
        # Request queue for sequential processing
        self.request_queue = asyncio.Queue(maxsize=100)
        self.queue_processor_running = False
        
        logger.info("BinanceRateLimiter initialized with conservative limits")
    
    async def acquire(self, weight: int = 1) -> bool:
        """
        Acquire permission to make a request.
        Returns False if we should skip the request due to rate limits.
        """
        current_time = time.time()
        
        # Check if we're in a ban period
        if current_time < self.ban_detected_until:
            ban_remaining = int(self.ban_detected_until - current_time)
            if ban_remaining > 60:  # Only log if more than 1 minute left
                logger.warning(f"Still in ban period, {ban_remaining}s remaining")
            return False
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
            self.request_weights.popleft()
        
        # Check minute-based limits
        if len(self.request_times) >= self.limits['requests_per_minute']:
            logger.warning("Request per minute limit reached")
            return False
        
        if sum(self.request_weights) + weight > self.limits['weight_per_minute']:
            logger.warning("Weight per minute limit reached")
            return False
        
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
            current_time = time.time()
        
        # Record this request
        self.request_times.append(current_time)
        self.request_weights.append(weight)
        self.last_request_time = current_time
        
        return True
    
    def handle_error(self, error: Exception):
        """Handle API errors and detect bans."""
        error_msg = str(error).lower()
        
        # Detect IP ban
        if 'banned' in error_msg or '418' in error_msg or 'way too many requests' in error_msg:
            # Extract ban time if available
            if 'until' in error_msg:
                try:
                    # Extract timestamp from error message
                    import re
                    timestamp_match = re.search(r'until (\d+)', error_msg)
                    if timestamp_match:
                        ban_until_ms = int(timestamp_match.group(1))
                        self.ban_detected_until = ban_until_ms / 1000  # Convert to seconds
                        ban_duration = int(self.ban_detected_until - time.time())
                        logger.error(f"🚨 Binance IP ban detected! Ban expires in {ban_duration} seconds")
                        return
                except:
                    pass
            
            # Fallback: assume 10 minute ban
            self.ban_detected_until = time.time() + 600
            logger.error("🚨 Binance IP ban detected! Assuming 10 minute ban")
            
        elif 'rate limit' in error_msg or '429' in error_msg:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                # Temporary cool-down period
                self.ban_detected_until = time.time() + 60
                logger.warning("Multiple rate limit errors, applying 1 minute cool-down")
        else:
            # Reset error count on non-rate-limit errors
            self.consecutive_errors = 0
    
    def handle_success(self):
        """Handle successful request."""
        self.consecutive_errors = 0
    
    def is_banned(self) -> bool:
        """Check if we're currently in a ban period."""
        return time.time() < self.ban_detected_until
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        recent_requests = sum(1 for t in self.request_times if current_time - t < 60)
        recent_weight = sum(w for i, w in enumerate(self.request_weights) 
                          if current_time - self.request_times[i] < 60)
        
        return {
            'requests_last_minute': recent_requests,
            'weight_last_minute': recent_weight,
            'consecutive_errors': self.consecutive_errors,
            'ban_detected': self.is_banned(),
            'ban_remaining_seconds': max(0, int(self.ban_detected_until - current_time)),
            'queue_size': self.request_queue.qsize()
        }


@dataclass
class TechnicalIndicators:
    """Technical indicators data structure."""
    symbol: str
    rsi: float
    macd: Dict[str, float]  # {'macd': value, 'signal': value, 'histogram': value}
    bollinger_bands: Dict[str, float]  # {'upper': value, 'middle': value, 'lower': value}
    ema_20: float
    ema_50: float
    volume_sma: float
    atr: float  # Average True Range
    timestamp: datetime
    # Advanced AI signal metrics
    taker_buy_sell_ratio: Optional[float] = None  # >1.0 means more taker buys (momentum)
    liquidation_volume: Optional[float] = None    # Volume of liquidations (exhaustion)
    basis_premium: Optional[float] = None         # Spot vs Futures spread (sentiment)


@dataclass
class OrderBookData:
    """Order book data structure."""
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    bid_price: float
    ask_price: float
    spread: float
    spread_percentage: float
    timestamp: datetime


@dataclass
class FundingData:
    """Funding rate data structure."""
    symbol: str
    funding_rate: float
    predicted_funding_rate: float
    funding_time: datetime
    mark_price: float
    index_price: float
    timestamp: datetime


@dataclass
class LongShortData:
    """Long/short ratio data structure."""
    symbol: str
    global_long_ratio: float  # % of accounts that are long
    global_short_ratio: float  # % of accounts that are short
    top_trader_long_ratio: float  # % of top traders that are long
    top_trader_short_ratio: float  # % of top traders that are short
    top_position_long_ratio: float  # % of top trader positions that are long
    top_position_short_ratio: float  # % of top trader positions that are short
    timestamp: datetime


@dataclass  
class OpenInterestData:
    """Open interest data structure."""
    symbol: str
    open_interest: float
    open_interest_change_24h: float  # 24h change in OI
    open_interest_change_percentage: float  # 24h % change
    timestamp: datetime


@dataclass
class VolumeProfile:
    """Volume profile data structure."""
    symbol: str
    volume_by_price: Dict[float, float]  # price -> volume
    poc: float  # Point of Control (highest volume price)
    value_area_high: float
    value_area_low: float
    volume_profile_indicator: str  # 'bullish', 'bearish', 'neutral'
    timestamp: datetime


class BinanceService:
    """
    Binance API service for advanced technical analysis and market data.
    
    Uses CCXT library for reliable API access and includes comprehensive
    technical indicator calculations.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """Initialize Binance service."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize rate limiter
        self.rate_limiter = BinanceRateLimiter()
        
        # Initialize CCXT exchange
        self.exchange = None
        self._init_exchange()
        
        # Cache settings
        self.cache_duration = {
            'orderbook': 5,  # 5 seconds for orderbook
            'klines': 30,  # 30 seconds for klines
            'funding': 300,  # 5 minutes for funding rates
            'volume_profile': 600,  # 10 minutes for volume profile
            'ticker': 120,  # 2 minutes for ticker data (respects Binance rate limits)
            'daily_ohlc': 3600  # 1 hour for daily OHLC data (reduces API calls)
        }
        self.cache = {}
        
        logger.info(f"BinanceService initialized (testnet: {testnet})")
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics for monitoring."""
        return self.rate_limiter.get_stats()
    
    async def _make_rate_limited_request(self, request_func, weight: int = 1, *args, **kwargs):
        """
        Make a rate-limited API request with retry logic and ban detection.
        
        Args:
            request_func: The actual API function to call
            weight: Request weight for rate limiting (default 1)
            *args, **kwargs: Arguments for the request function
            
        Returns:
            Result from request_func or None if rate limited/banned
        """
        # Check if we can make the request
        if not await self.rate_limiter.acquire(weight):
            logger.debug("Request skipped due to rate limiting")
            return None
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Make the actual API request - check if it's async or sync
                if asyncio.iscoroutinefunction(request_func):
                    result = await request_func(*args, **kwargs)
                else:
                    result = request_func(*args, **kwargs)
                self.rate_limiter.handle_success()
                return result
                
            except ccxt.BadSymbol as e:
                # Permanent condition: the market simply isn't listed on Binance.
                # Retrying wastes rate-limit weight, and callers probe symbol
                # variants speculatively, so log quietly and give up immediately.
                logger.debug(f"Binance does not list this market, skipping retries: {e}")
                return None
            except Exception as e:
                last_error = e
                # Handle errors and update rate limiter state
                self.rate_limiter.handle_error(e)

                error_msg = str(e).lower()
                if 'banned' in error_msg or '418' in error_msg:
                    logger.error(f"Binance IP ban detected, aborting retries: {e}")
                    break
                elif 'does not have market symbol' in error_msg:
                    # Same permanent condition surfaced through a wrapper exception
                    logger.debug(f"Binance does not list this market, skipping retries: {e}")
                    return None
                elif 'rate limit' in error_msg or '429' in error_msg:
                    if attempt < max_retries - 1:
                        backoff_time = (2 ** attempt) * 0.5  # Exponential backoff
                        logger.warning(f"Rate limit hit, retry {attempt + 1}/{max_retries} after {backoff_time}s")
                        await asyncio.sleep(backoff_time)
                        continue
                else:
                    logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.1 * (attempt + 1))  # Small delay before retry
            
        # All retries failed
        logger.error(f"All API request attempts failed: {last_error}")
        return None
    
    def _init_exchange(self):
        """Initialize CCXT exchange instance."""
        try:
            config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Use futures for better data and higher rate limits
                }
            }
            
            if self.testnet:
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api'
                    }
                }
            
            self.exchange = ccxt.binance(config)
            logger.info("CCXT Binance exchange initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {e}")
            self.exchange = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the underlying exchange when the CCXT implementation supports it."""
        if not self.exchange:
            return

        close = getattr(self.exchange, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

        self.exchange = None
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_duration.get(cache_type, 60)
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_technical_indicators(self, symbol: str, timeframe: str = '1h', periods: int = 100) -> TechnicalIndicators:
        """
        Calculate comprehensive technical indicators for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis ('1m', '5m', '1h', '4h', '1d')
            periods: Number of periods to analyze
            
        Returns:
            TechnicalIndicators object
        """
        try:
            cache_key = f"indicators_{symbol}_{timeframe}_{periods}"
            
            if self._is_cache_valid(cache_key, 'klines'):
                return self.cache[cache_key]['data']
            
            if not self.exchange:
                self._init_exchange()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=periods)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            indicators.symbol = symbol
            indicators.timestamp = datetime.now()
            
            # Cache the result
            self._cache_data(cache_key, indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return TechnicalIndicators(
                symbol=symbol,
                rsi=50.0,
                macd={'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
                bollinger_bands={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
                ema_20=0.0,
                ema_50=0.0,
                volume_sma=0.0,
                atr=0.0,
                timestamp=datetime.now()
            )
    
    async def get_daily_ohlc(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get today's OHLC (Open, High, Low, Close) data for accurate signal detection.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with open, high, low, close prices for current day
        """
        try:
            cache_key = f"daily_ohlc_{symbol}"
            
            if self._is_cache_valid(cache_key, 'daily_ohlc'):
                return self.cache[cache_key]['data']
            
            if not self.exchange:
                self._init_exchange()
            
            # Fetch last 2 daily candles to get current day data
            ohlcv = await self._make_rate_limited_request(
                self.exchange.fetch_ohlcv, 1, symbol, '1d', limit=2
            )
            
            if not ohlcv or len(ohlcv) < 1:
                return None
            
            # Get the most recent (current) daily candle
            current_day = ohlcv[-1]  # [timestamp, open, high, low, close, volume]
            
            daily_data = {
                'open': float(current_day[1]),
                'high': float(current_day[2]), 
                'low': float(current_day[3]),
                'close': float(current_day[4]),  # Current price (last close)
                'timestamp': current_day[0]
            }
            
            # Cache with shorter duration for daily data (1 hour)
            self._cache_data(cache_key, daily_data)
            logger.debug(f"Daily OHLC for {symbol}: H:{daily_data['high']:.2f} L:{daily_data['low']:.2f} C:{daily_data['close']:.2f}")
            
            return daily_data
            
        except Exception as e:
            logger.warning(f"Error getting daily OHLC for {symbol}: {e}")
            return None

    async def get_historical_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical OHLC kline/candlestick data from Binance for institutional analysis.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Number of klines to retrieve (max 1000)

        Returns:
            List of OHLC candle dictionaries with timestamps, volumes
        """
        try:
            cache_key = f"klines_{symbol}_{interval}_{limit}"

            if self._is_cache_valid(cache_key, 'klines'):
                return self.cache[cache_key]['data']

            if not self.exchange:
                self._init_exchange()

            # Fetch OHLCV data using CCXT
            ohlcv_data = await self._make_rate_limited_request(
                self.exchange.fetch_ohlcv,
                2,  # Weight of 2 for klines endpoint
                symbol,
                interval,
                limit=min(limit, 1000)
            )

            if not ohlcv_data:
                # Callers probe symbol variants speculatively (spot/futures/1000x
                # prefixes), so an empty result per variant is expected noise;
                # genuine API failures are already logged by the retry wrapper.
                logger.debug(f"No kline data returned for {symbol}")
                return []

            # Convert CCXT format to our OHLC format
            ohlc_candles = []
            for candle in ohlcv_data:
                # CCXT format: [timestamp, open, high, low, close, volume]
                ohlc_candles.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),  # Convert ms to datetime
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]) if candle[5] else 0.0,
                    'interval': interval
                })

            # Cache the results
            self._cache_data(cache_key, ohlc_candles)

            logger.debug(f"✅ Retrieved {len(ohlc_candles)} {interval} OHLC candles for {symbol}")
            return ohlc_candles

        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} ({interval}): {e}")
            return []

    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate technical indicators from OHLCV data."""
        try:
            # RSI calculation
            rsi = self._calculate_rsi(df['close'].values, period=14)
            
            # MACD calculation
            macd_data = self._calculate_macd(df['close'].values)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['close'].values, period=20, std_dev=2)
            
            # EMAs
            ema_20 = self._calculate_ema(df['close'].values, period=20)
            ema_50 = self._calculate_ema(df['close'].values, period=50)
            
            # Volume SMA
            volume_sma = np.mean(df['volume'].values[-20:]) if len(df) >= 20 else 0.0
            
            # ATR (Average True Range)
            atr = self._calculate_atr(df[['high', 'low', 'close']].values, period=14)
            
            return TechnicalIndicators(
                symbol="",  # Will be set by caller
                rsi=float(rsi),
                macd=macd_data,
                bollinger_bands=bb_data,
                ema_20=float(ema_20),
                ema_50=float(ema_50),
                volume_sma=float(volume_sma),
                atr=float(atr),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in indicator calculations: {e}")
            raise
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_series = []
        for i in range(slow, len(prices)):
            ema_fast_i = self._calculate_ema(prices[:i+1], fast)
            ema_slow_i = self._calculate_ema(prices[:i+1], slow)
            macd_series.append(ema_fast_i - ema_slow_i)
        
        if len(macd_series) >= signal:
            signal_line = self._calculate_ema(np.array(macd_series), signal)
        else:
            signal_line = macd
        
        histogram = macd - signal_line
        
        return {
            'macd': float(macd),
            'signal': float(signal_line),
            'histogram': float(histogram)
        }
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            price = float(prices[-1]) if len(prices) > 0 else 0.0
            return {'upper': price, 'middle': price, 'lower': price}
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            'upper': float(upper),
            'middle': float(sma),
            'lower': float(lower)
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices)) if len(prices) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return float(ema)
    
    def _calculate_atr(self, hlc_data: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(hlc_data) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(hlc_data)):
            high, low, close = hlc_data[i]
            prev_close = hlc_data[i-1][2]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) >= period:
            return float(np.mean(true_ranges[-period:]))
        else:
            return float(np.mean(true_ranges)) if true_ranges else 0.0
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBookData:
        """
        Get real-time order book data.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to return
            
        Returns:
            OrderBookData object
        """
        try:
            cache_key = f"orderbook_{symbol}_{limit}"
            
            if self._is_cache_valid(cache_key, 'orderbook'):
                return self.cache[cache_key]['data']
            
            if not self.exchange:
                self._init_exchange()
            
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            
            bids = [(float(price), float(amount)) for price, amount in orderbook['bids'][:limit]]
            asks = [(float(price), float(amount)) for price, amount in orderbook['asks'][:limit]]
            
            bid_price = float(bids[0][0]) if bids else 0.0
            ask_price = float(asks[0][0]) if asks else 0.0
            spread = ask_price - bid_price
            spread_percentage = (spread / bid_price) * 100 if bid_price > 0 else 0.0
            
            orderbook_data = OrderBookData(
                symbol=symbol,
                bids=bids,
                asks=asks,
                bid_price=bid_price,
                ask_price=ask_price,
                spread=spread,
                spread_percentage=spread_percentage,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, orderbook_data)
            return orderbook_data
            
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return OrderBookData(
                symbol=symbol,
                bids=[],
                asks=[],
                bid_price=0.0,
                ask_price=0.0,
                spread=0.0,
                spread_percentage=0.0,
                timestamp=datetime.now()
            )
    
    async def get_funding_rates(self, symbols: List[str] = None) -> List[FundingData]:
        """
        Get funding rates using direct Binance Futures API for better reliability.
        
        Args:
            symbols: List of symbols to get funding rates for
            
        Returns:
            List of FundingData objects
        """
        try:
            import aiohttp
            
            cache_key = f"funding_rates_{symbols}"
            
            if self._is_cache_valid(cache_key, 'funding'):
                return self.cache[cache_key]['data']
            
            logger.info(f"Fetching funding rates for {len(symbols) if symbols else 0} symbols from Binance Futures API...")
            
            funding_rates = []
            
            if symbols:
                # Use direct Binance Futures API
                url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    for symbol in symbols:
                        try:
                            params = {'symbol': symbol}
                            async with session.get(url, params=params) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    funding_data = self._parse_funding_data_direct(symbol, data)
                                    funding_rates.append(funding_data)
                                else:
                                    logger.warning(f"Failed to get funding rate for {symbol}: HTTP {response.status}")
                        except Exception as e:
                            logger.warning(f"Failed to get funding rate for {symbol}: {e}")
                            continue
            
            self._cache_data(cache_key, funding_rates)
            logger.info(f"Retrieved funding rates for {len(funding_rates)} symbols")
            return funding_rates
            
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return []

    async def get_long_short_ratios(self, symbols: List[str] = None) -> List[LongShortData]:
        """
        Get long/short ratios from Binance Futures including global and top trader data.
        
        This provides crucial sentiment indicators:
        - Global account ratios: How retail traders are positioned
        - Top trader ratios: How experienced traders are positioned  
        - Position ratios: Actual size of positions (more accurate than account count)
        
        Args:
            symbols: List of symbols to get ratios for
            
        Returns:
            List of LongShortData objects
        """
        try:
            import aiohttp
            
            cache_key = f"long_short_ratios_{symbols}"
            
            if self._is_cache_valid(cache_key, 'funding'):  # Use same cache duration as funding
                return self.cache[cache_key]['data']
            
            logger.info(f"Fetching long/short ratios for {len(symbols) if symbols else 0} symbols from Binance...")
            
            long_short_data = []
            
            if symbols:
                # Setup SSL context
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    
                    for symbol in symbols:
                        try:
                            # Fetch all three ratio endpoints in parallel
                            global_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                            top_trader_url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio" 
                            top_position_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
                            
                            params = {
                                'symbol': symbol,
                                'period': '5m',  # 5 minute data for real-time analysis
                                'limit': 1  # Just get latest data point
                            }
                            
                            # Fetch all three metrics concurrently
                            global_task = session.get(global_url, params=params)
                            top_trader_task = session.get(top_trader_url, params=params)
                            top_position_task = session.get(top_position_url, params=params)
                            
                            global_resp, top_trader_resp, top_position_resp = await asyncio.gather(
                                global_task, top_trader_task, top_position_task, return_exceptions=True
                            )
                            
                            # Parse responses
                            global_data = None
                            top_trader_data = None
                            top_position_data = None
                            
                            if not isinstance(global_resp, Exception) and global_resp.status == 200:
                                global_json = await global_resp.json()
                                global_data = global_json[0] if global_json else None
                                
                            if not isinstance(top_trader_resp, Exception) and top_trader_resp.status == 200:
                                top_trader_json = await top_trader_resp.json()
                                top_trader_data = top_trader_json[0] if top_trader_json else None
                                
                            if not isinstance(top_position_resp, Exception) and top_position_resp.status == 200:
                                top_position_json = await top_position_resp.json()
                                top_position_data = top_position_json[0] if top_position_json else None
                            
                            # Create combined data structure
                            if global_data or top_trader_data or top_position_data:
                                long_short_data.append(LongShortData(
                                    symbol=symbol,
                                    global_long_ratio=float(global_data.get('longAccount', 0.5)) if global_data else 0.5,
                                    global_short_ratio=float(global_data.get('shortAccount', 0.5)) if global_data else 0.5,
                                    top_trader_long_ratio=float(top_trader_data.get('longAccount', 0.5)) if top_trader_data else 0.5,
                                    top_trader_short_ratio=float(top_trader_data.get('shortAccount', 0.5)) if top_trader_data else 0.5,
                                    top_position_long_ratio=float(top_position_data.get('longAccount', 0.5)) if top_position_data else 0.5,
                                    top_position_short_ratio=float(top_position_data.get('shortAccount', 0.5)) if top_position_data else 0.5,
                                    timestamp=datetime.now()
                                ))
                                
                        except Exception as e:
                            logger.warning(f"Failed to get long/short ratios for {symbol}: {e}")
                            continue
            
            self._cache_data(cache_key, long_short_data)
            logger.info(f"Retrieved long/short ratios for {len(long_short_data)} symbols")
            return long_short_data
            
        except Exception as e:
            logger.error(f"Failed to fetch long/short ratios: {e}")
            return []

    @staticmethod
    async def get_advanced_ai_metrics(symbol: str) -> Dict[str, Optional[float]]:
        """
        Fetch advanced metrics for AI signal generation.
        Includes Taker Buy/Sell Volume (momentum), Liquidations (exhaustion), and Basis (sentiment).
        """
        metrics = {
            'taker_buy_sell_ratio': None,
            'liquidation_volume': None,
            'basis_premium': None
        }
        
        try:
            from kata.services.session_manager import SessionManager
            import aiohttp
            
            session_manager = SessionManager()
            session = await session_manager.get_session("binance_advanced_metrics")
                
            taker_url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
            basis_url = "https://fapi.binance.com/futures/data/basis"
            
            params_taker = {'symbol': symbol, 'period': '5m', 'limit': 1}
            params_basis = {'pair': symbol, 'contractType': 'PERPETUAL', 'period': '5m', 'limit': 1}
            
            async def fetch_json(url, params):
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        return None
                except Exception as e:
                    logger.debug(f"Error fetching from {url}: {e}")
                    return None
            
            taker_json, basis_json = await asyncio.gather(
                fetch_json(taker_url, params_taker),
                fetch_json(basis_url, params_basis),
                return_exceptions=True
            )
            
            if not isinstance(taker_json, Exception) and taker_json and isinstance(taker_json, list) and len(taker_json) > 0:
                metrics['taker_buy_sell_ratio'] = float(taker_json[0].get('buySellRatio', 1.0))
                    
            if not isinstance(basis_json, Exception) and basis_json and isinstance(basis_json, list) and len(basis_json) > 0:
                metrics['basis_premium'] = float(basis_json[0].get('basis', 0.0))
                    
        except Exception as e:
            logger.warning(f"Failed to fetch advanced AI metrics for {symbol}: {e}")
            
        return metrics

    async def get_open_interest_data(self, symbols: List[str] = None) -> List[OpenInterestData]:
        """
        Get open interest data from Binance Futures.
        
        Open Interest is crucial for understanding:
        - Market participation and liquidity
        - Whether price moves are backed by new money (OI increases)
        - Potential for liquidations (high OI + price moves)
        - Market tops/bottoms (divergences between price and OI)
        
        Args:
            symbols: List of symbols to get open interest for
            
        Returns:
            List of OpenInterestData objects
        """
        try:
            import aiohttp
            
            cache_key = f"open_interest_{symbols}"
            
            if self._is_cache_valid(cache_key, 'funding'):
                return self.cache[cache_key]['data']
            
            logger.info(f"Fetching open interest for {len(symbols) if symbols else 0} symbols from Binance...")
            
            oi_data = []
            
            if symbols:
                # Setup SSL context
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    
                    for symbol in symbols:
                        try:
                            # Get current OI and historical OI for 24h change
                            current_oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
                            hist_oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
                            
                            current_params = {'symbol': symbol}
                            hist_params = {
                                'symbol': symbol,
                                'period': '1d',  # Daily data
                                'limit': 2  # Get current and previous day
                            }
                            
                            # Fetch both concurrently
                            current_task = session.get(current_oi_url, params=current_params)
                            hist_task = session.get(hist_oi_url, params=hist_params)
                            
                            current_resp, hist_resp = await asyncio.gather(
                                current_task, hist_task, return_exceptions=True
                            )
                            
                            current_oi = 0.0
                            oi_change_24h = 0.0
                            oi_change_percentage = 0.0
                            
                            # Parse current OI
                            if not isinstance(current_resp, Exception) and current_resp.status == 200:
                                current_data = await current_resp.json()
                                current_oi = float(current_data.get('openInterest', 0))
                            
                            # Parse historical OI for change calculation
                            if not isinstance(hist_resp, Exception) and hist_resp.status == 200:
                                hist_data = await hist_resp.json()
                                if len(hist_data) >= 2:
                                    today_oi = float(hist_data[0].get('sumOpenInterest', 0))
                                    yesterday_oi = float(hist_data[1].get('sumOpenInterest', 0))
                                    
                                    if yesterday_oi > 0:
                                        oi_change_24h = today_oi - yesterday_oi
                                        oi_change_percentage = (oi_change_24h / yesterday_oi) * 100
                            
                            oi_data.append(OpenInterestData(
                                symbol=symbol,
                                open_interest=current_oi,
                                open_interest_change_24h=oi_change_24h,
                                open_interest_change_percentage=oi_change_percentage,
                                timestamp=datetime.now()
                            ))
                                
                        except Exception as e:
                            logger.warning(f"Failed to get open interest for {symbol}: {e}")
                            continue
            
            self._cache_data(cache_key, oi_data)
            logger.info(f"Retrieved open interest data for {len(oi_data)} symbols")
            return oi_data
            
        except Exception as e:
            logger.error(f"Error fetching open interest data: {e}")
            return []
    
    def _parse_funding_data(self, symbol: str, funding_data: Dict[str, Any]) -> FundingData:
        """Parse funding rate data from exchange response."""
        return FundingData(
            symbol=symbol,
            funding_rate=float(funding_data.get('fundingRate', 0)),
            predicted_funding_rate=float(funding_data.get('predictedFundingRate', 0)),
            funding_time=datetime.fromtimestamp(funding_data.get('fundingDatetime', 0) / 1000) if funding_data.get('fundingDatetime') else datetime.now(),
            mark_price=float(funding_data.get('markPrice', 0)),
            index_price=float(funding_data.get('indexPrice', 0)),
            timestamp=datetime.now()
        )
    
    def _parse_funding_data_direct(self, symbol: str, funding_data: Dict[str, Any]) -> FundingData:
        """Parse funding rate data from direct Binance Futures API response."""
        return FundingData(
            symbol=symbol,
            funding_rate=float(funding_data.get('lastFundingRate', 0)),
            predicted_funding_rate=float(funding_data.get('lastFundingRate', 0)),  # Binance doesn't provide predicted rate in this endpoint
            funding_time=datetime.fromtimestamp(int(funding_data.get('nextFundingTime', 0)) / 1000) if funding_data.get('nextFundingTime') else datetime.now(),
            mark_price=float(funding_data.get('markPrice', 0)),
            index_price=float(funding_data.get('indexPrice', 0)),
            timestamp=datetime.now()
        )
    
    async def get_market_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get comprehensive market data for multiple symbols from Binance with rate limiting.
        This replaces CoinGecko for price data to avoid rate limits.
        
        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH', 'SOL'])
            
        Returns:
            List of market data dictionaries
        """
        try:
            if not self.exchange:
                self._init_exchange()
            
            # Check if we're banned
            if self.rate_limiter.is_banned():
                logger.warning("Binance service is currently banned, returning cached/empty data")
                return []
            
            market_data = []
            successful_requests = 0
            failed_requests = 0
            
            # Process symbols with rate limiting (sequential to avoid overwhelming API)
            for i, symbol in enumerate(symbols):
                try:
                    # Convert symbol to CCXT format if needed
                    if symbol.endswith('USDT') and '/' not in symbol:
                        # Symbol is in Binance format (e.g., BTCUSDT), convert to CCXT format
                        base_symbol = symbol[:-4]  # Remove 'USDT'
                        binance_symbol = f"{base_symbol}/USDT"
                    elif '/' in symbol:
                        # Symbol is already in CCXT format
                        binance_symbol = symbol
                    else:
                        # Symbol is base only (e.g., BTC), add /USDT
                        binance_symbol = f"{symbol}/USDT"
                    
                    # Rate limited ticker fetch (weight: 1)
                    ticker = await self._make_rate_limited_request(
                        lambda: self.exchange.fetch_ticker(binance_symbol), 
                        weight=1
                    )
                    
                    if ticker is None:
                        logger.debug(f"Ticker request skipped for {symbol} due to rate limiting")
                        failed_requests += 1
                        continue
                    
                    # Rate limited OHLCV fetch (weight: 1)
                    ohlcv = await self._make_rate_limited_request(
                        lambda: self.exchange.fetch_ohlcv(binance_symbol, '1d', limit=7),
                        weight=1
                    )
                    
                    # Calculate 7d change if we have OHLCV data
                    price_change_7d = 0.0
                    if ohlcv and len(ohlcv) >= 7:
                        current_price = float(ticker['close'])
                        price_7d_ago = float(ohlcv[-7][4])  # Close price 7 days ago
                        price_change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
                    
                    market_data.append({
                        'symbol': symbol,
                        'current_price': float(ticker['close']),
                        'volume_24h': float(ticker['baseVolume']),
                        'price_change_24h': float(ticker['change']) if ticker['change'] else 0.0,
                        'price_change_percentage_24h': float(ticker['percentage']) if ticker['percentage'] else 0.0,
                        'price_change_percentage_7d': price_change_7d,
                        'high_24h': float(ticker['high']),
                        'low_24h': float(ticker['low']),
                        'market_cap': 0,  # Binance doesn't provide market cap
                        'market_cap_rank': 0,  # Will need to estimate or use cached CoinGecko data
                        'timestamp': datetime.now()
                    })
                    
                    successful_requests += 1
                    
                    # Log progress every 10 symbols
                    if (i + 1) % 10 == 0:
                        stats = self.rate_limiter.get_stats()
                        logger.info(f"Market data progress: {i + 1}/{len(symbols)} symbols, "
                                  f"success: {successful_requests}, failed: {failed_requests}, "
                                  f"rate limit: {stats['requests_last_minute']}/min")
                    
                except Exception as e:
                    logger.warning(f"Failed to get market data for {symbol}: {e}")
                    failed_requests += 1
                    
                    # Check if this is a ban-related error
                    error_msg = str(e).lower()
                    if 'banned' in error_msg or '418' in error_msg:
                        logger.error("Binance ban detected during market data fetch, stopping")
                        break
                    
                    continue
            
            # Log final statistics
            logger.info(f"Market data fetch completed: {successful_requests} successful, "
                       f"{failed_requests} failed out of {len(symbols)} symbols")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data from Binance: {e}")
            return []
    
    async def get_trending_symbols(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending symbols based on 24h volume and price changes.
        This replaces CoinGecko trending data.
        
        Args:
            limit: Number of trending symbols to return
            
        Returns:
            List of trending symbol data
        """
        try:
            if not self.exchange:
                self._init_exchange()
            
            # Get all USDT pairs
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            # Get 24hr tickers for all USDT pairs
            tickers = self.exchange.fetch_tickers(usdt_pairs[:100])  # Limit to avoid rate limits
            
            # Sort by volume and price change
            trending = []
            for symbol, ticker in tickers.items():
                if ticker['baseVolume'] and ticker['percentage']:
                    base_symbol = symbol.split('/')[0]
                    trending.append({
                        'symbol': base_symbol,
                        'price_change_percentage_24h': float(ticker['percentage']),
                        'volume_24h': float(ticker['baseVolume']),
                        'current_price': float(ticker['close']),
                        'volume_rank': 0  # Will be calculated after sorting
                    })
            
            # Sort by volume (trending by activity)
            trending.sort(key=lambda x: x['volume_24h'], reverse=True)
            
            # Add volume ranks
            for i, item in enumerate(trending[:limit]):
                item['volume_rank'] = i + 1
            
            return trending[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching trending data from Binance: {e}")
            return []

    async def get_volume_profile(self, symbol: str, timeframe: str = '1h', periods: int = 24) -> VolumeProfile:
        """
        Calculate volume profile for price discovery analysis.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            periods: Number of periods to analyze
            
        Returns:
            VolumeProfile object
        """
        try:
            cache_key = f"volume_profile_{symbol}_{timeframe}_{periods}"
            
            if self._is_cache_valid(cache_key, 'volume_profile'):
                return self.cache[cache_key]['data']
            
            if not self.exchange:
                self._init_exchange()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=periods)
            
            # Calculate volume profile
            volume_by_price = {}
            total_volume = 0
            
            for candle in ohlcv:
                open_price, high, low, close, volume = candle[1], candle[2], candle[3], candle[4], candle[5]
                
                # Distribute volume across price range (simplified)
                price_range = high - low
                if price_range > 0:
                    num_levels = 10  # Divide each candle into 10 price levels
                    volume_per_level = volume / num_levels
                    
                    for i in range(num_levels):
                        price_level = low + (price_range * i / num_levels)
                        price_key = round(price_level, 2)
                        volume_by_price[price_key] = volume_by_price.get(price_key, 0) + volume_per_level
                
                total_volume += volume
            
            # Find Point of Control (POC) - price with highest volume
            poc = max(volume_by_price.keys(), key=lambda k: volume_by_price[k]) if volume_by_price else 0.0
            
            # Calculate Value Area (70% of volume)
            sorted_volumes = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
            value_area_volume = total_volume * 0.7
            current_volume = 0
            value_area_prices = []
            
            for price, vol in sorted_volumes:
                current_volume += vol
                value_area_prices.append(price)
                if current_volume >= value_area_volume:
                    break
            
            value_area_high = max(value_area_prices) if value_area_prices else poc
            value_area_low = min(value_area_prices) if value_area_prices else poc
            
            # Determine volume profile indicator
            recent_close = ohlcv[-1][4] if ohlcv else poc
            if recent_close > value_area_high:
                indicator = "bullish"
            elif recent_close < value_area_low:
                indicator = "bearish"
            else:
                indicator = "neutral"
            
            volume_profile = VolumeProfile(
                symbol=symbol,
                volume_by_price=volume_by_price,
                poc=float(poc),
                value_area_high=float(value_area_high),
                value_area_low=float(value_area_low),
                volume_profile_indicator=indicator,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, volume_profile)
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error calculating volume profile for {symbol}: {e}")
            return VolumeProfile(
                symbol=symbol,
                volume_by_price={},
                poc=0.0,
                value_area_high=0.0,
                value_area_low=0.0,
                volume_profile_indicator="neutral",
                timestamp=datetime.now()
            )

    async def get_24hr_ticker_stats(self) -> List[Dict[str, Any]]:
        """
        Get 24hr ticker statistics using direct Binance REST API.
        
        API Details:
        - Endpoint: /api/v3/ticker/24hr (weight: 40)
        - Rate limit: 1200 requests per minute (20 req/sec)
        - No API key required for this endpoint
        
        Returns:
            List of ticker data with volume, price changes, etc.
        """
        try:
            import aiohttp
            import asyncio
            
            # Check cache first (cache for 2 minutes to respect rate limits)
            cache_key = "24hr_ticker_stats"
            if self._is_cache_valid(cache_key, 'ticker'):
                logger.info("Using cached 24hr ticker stats")
                return self.cache[cache_key]['data']
            
            logger.info("Fetching 24hr ticker stats from Binance REST API...")
            
            # Use direct Binance API with timeout
            url = "https://api.binance.com/api/v3/ticker/24hr"
            
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        tickers = await response.json()
                        logger.info(f"Retrieved {len(tickers)} ticker stats from Binance API")
                        
                        # Cache the results for 2 minutes
                        self._cache_data(cache_key, tickers)
                        
                        return tickers
                    elif response.status == 429:
                        # Rate limit exceeded
                        raise Exception(f"Binance API rate limit exceeded (429). Please wait before retrying.")
                    elif response.status == 418:
                        # IP banned
                        raise Exception(f"IP has been auto-banned by Binance for continuing to send requests after 429")
                    else:
                        raise Exception(f"Binance API error: HTTP {response.status}")
            
        except asyncio.TimeoutError:
            logger.error("Timeout fetching 24hr ticker stats from Binance API")
            raise Exception("Binance API request timeout. Please try again later.")
        except Exception as e:
            logger.error(f"Error fetching 24hr ticker stats from Binance: {e}")
            raise

    # Enhanced Orderbook Analysis
    async def get_enhanced_orderbook_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get enhanced orderbook analysis including order flow, imbalance, and depth metrics.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing detailed orderbook analysis
        """
        try:
            # Get deeper orderbook data
            orderbook = await self.get_orderbook(symbol, limit=100)
            
            if not orderbook.bids or not orderbook.asks:
                return self._default_orderbook_analysis()
            
            # Calculate order flow metrics
            flow_analysis = self._calculate_order_flow(orderbook)
            
            # Calculate order book imbalance
            imbalance_analysis = self._calculate_orderbook_imbalance(orderbook)
            
            # Calculate market depth and liquidity
            depth_analysis = self._calculate_market_depth(orderbook)
            
            # Calculate order clustering
            clustering_analysis = self._calculate_order_clustering(orderbook)
            
            # Calculate support/resistance from orderbook
            sr_analysis = self._calculate_orderbook_support_resistance(orderbook)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'spread': orderbook.spread,
                'spread_percentage': orderbook.spread_percentage,
                **flow_analysis,
                **imbalance_analysis,
                **depth_analysis,
                **clustering_analysis,
                **sr_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced orderbook analysis for {symbol}: {e}")
            return self._default_orderbook_analysis()
    
    def _calculate_order_flow(self, orderbook: OrderBookData) -> Dict[str, Any]:
        """Calculate order flow pressure and momentum."""
        try:
            bids = orderbook.bids
            asks = orderbook.asks
            
            if not bids or not asks:
                return {'order_flow': 'NEUTRAL', 'flow_strength': 0.0}
            
            # Calculate total bid/ask volumes in different price ranges
            current_price = (orderbook.bid_price + orderbook.ask_price) / 2
            
            # Near market (within 1% of current price)
            near_range = current_price * 0.01
            near_bid_volume = sum(size for price, size in bids if current_price - price <= near_range)
            near_ask_volume = sum(size for price, size in asks if price - current_price <= near_range)
            
            # Mid market (1-2% from current price)
            mid_range = current_price * 0.02
            mid_bid_volume = sum(size for price, size in bids 
                               if near_range < current_price - price <= mid_range)
            mid_ask_volume = sum(size for price, size in asks 
                               if near_range < price - current_price <= mid_range)
            
            # Calculate flow metrics
            total_bid_volume = sum(size for _, size in bids)
            total_ask_volume = sum(size for _, size in asks)
            
            # Flow direction and strength
            if total_bid_volume == 0 and total_ask_volume == 0:
                return {'order_flow': 'NEUTRAL', 'flow_strength': 0.0}
            
            flow_ratio = total_bid_volume / (total_bid_volume + total_ask_volume)
            
            if flow_ratio > 0.6:
                flow_direction = 'BULLISH'
                flow_strength = (flow_ratio - 0.5) * 2
            elif flow_ratio < 0.4:
                flow_direction = 'BEARISH' 
                flow_strength = (0.5 - flow_ratio) * 2
            else:
                flow_direction = 'NEUTRAL'
                flow_strength = abs(flow_ratio - 0.5) * 2
            
            return {
                'order_flow': flow_direction,
                'flow_strength': round(flow_strength, 3),
                'total_bid_volume': round(total_bid_volume, 6),
                'total_ask_volume': round(total_ask_volume, 6),
                'near_bid_volume': round(near_bid_volume, 6),
                'near_ask_volume': round(near_ask_volume, 6),
                'mid_bid_volume': round(mid_bid_volume, 6),
                'mid_ask_volume': round(mid_ask_volume, 6),
                'flow_ratio': round(flow_ratio, 3)
            }
            
        except Exception as e:
            logger.error(f"Error calculating order flow: {e}")
            return {'order_flow': 'NEUTRAL', 'flow_strength': 0.0}
    
    def _calculate_orderbook_imbalance(self, orderbook: OrderBookData) -> Dict[str, Any]:
        """Calculate orderbook imbalance metrics."""
        try:
            bids = orderbook.bids[:10]  # Top 10 levels
            asks = orderbook.asks[:10]  # Top 10 levels
            
            if not bids or not asks:
                return {'imbalance': 0.0, 'imbalance_direction': 'NEUTRAL'}
            
            # Calculate volume at each level
            bid_volumes = [size for _, size in bids]
            ask_volumes = [size for _, size in asks]
            
            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)
            
            # Calculate weighted imbalance (closer levels have more weight)
            weighted_bid_volume = sum(size * (11 - i) for i, (_, size) in enumerate(bids, 1))
            weighted_ask_volume = sum(size * (11 - i) for i, (_, size) in enumerate(asks, 1))
            
            if total_bid_volume + total_ask_volume == 0:
                return {'imbalance': 0.0, 'imbalance_direction': 'NEUTRAL'}
            
            # Simple imbalance
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Weighted imbalance
            weighted_imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)
            
            # Determine direction
            if imbalance > 0.1:
                direction = 'BULLISH'
            elif imbalance < -0.1:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            return {
                'imbalance': round(imbalance, 3),
                'weighted_imbalance': round(weighted_imbalance, 3),
                'imbalance_direction': direction,
                'top_10_bid_volume': round(total_bid_volume, 6),
                'top_10_ask_volume': round(total_ask_volume, 6)
            }
            
        except Exception as e:
            logger.error(f"Error calculating orderbook imbalance: {e}")
            return {'imbalance': 0.0, 'imbalance_direction': 'NEUTRAL'}
    
    def _calculate_market_depth(self, orderbook: OrderBookData) -> Dict[str, Any]:
        """Calculate market depth and liquidity metrics."""
        try:
            current_price = (orderbook.bid_price + orderbook.ask_price) / 2
            
            # Calculate depth at different percentage levels
            depth_levels = [0.5, 1.0, 2.0, 5.0]  # Percentage levels
            depth_analysis = {}
            
            for level in depth_levels:
                price_range = current_price * (level / 100)
                
                # Calculate bid depth
                bid_depth = sum(size for price, size in orderbook.bids 
                              if current_price - price <= price_range)
                
                # Calculate ask depth  
                ask_depth = sum(size for price, size in orderbook.asks
                              if price - current_price <= price_range)
                
                depth_analysis[f'bid_depth_{level}%'] = round(bid_depth, 6)
                depth_analysis[f'ask_depth_{level}%'] = round(ask_depth, 6)
                depth_analysis[f'total_depth_{level}%'] = round(bid_depth + ask_depth, 6)
            
            # Calculate liquidity score (based on depth and spread)
            total_depth_1pct = depth_analysis['total_depth_1.0%']
            spread_pct = orderbook.spread_percentage
            
            if spread_pct > 0:
                liquidity_score = min(100, (total_depth_1pct * 1000) / spread_pct)
            else:
                liquidity_score = 100
            
            # Determine liquidity quality
            if liquidity_score > 80:
                liquidity_quality = 'EXCELLENT'
            elif liquidity_score > 60:
                liquidity_quality = 'GOOD'
            elif liquidity_score > 40:
                liquidity_quality = 'MODERATE'
            elif liquidity_score > 20:
                liquidity_quality = 'POOR'
            else:
                liquidity_quality = 'VERY_POOR'
            
            return {
                **depth_analysis,
                'liquidity_score': round(liquidity_score, 1),
                'liquidity_quality': liquidity_quality
            }
            
        except Exception as e:
            logger.error(f"Error calculating market depth: {e}")
            return {'liquidity_score': 50.0, 'liquidity_quality': 'MODERATE'}
    
    def _calculate_order_clustering(self, orderbook: OrderBookData) -> Dict[str, Any]:
        """Analyze order clustering at key price levels."""
        try:
            # Find price levels with unusually high order sizes
            bid_clusters = []
            ask_clusters = []
            
            if orderbook.bids:
                # Calculate average bid size
                bid_sizes = [size for _, size in orderbook.bids]
                avg_bid_size = sum(bid_sizes) / len(bid_sizes)
                
                # Find large orders (> 3x average)
                large_threshold = avg_bid_size * 3
                for price, size in orderbook.bids:
                    if size > large_threshold:
                        bid_clusters.append({'price': price, 'size': size})
            
            if orderbook.asks:
                # Calculate average ask size
                ask_sizes = [size for _, size in orderbook.asks]
                avg_ask_size = sum(ask_sizes) / len(ask_sizes)
                
                # Find large orders (> 3x average)
                large_threshold = avg_ask_size * 3
                for price, size in orderbook.asks:
                    if size > large_threshold:
                        ask_clusters.append({'price': price, 'size': size})
            
            # Analyze psychological levels (round numbers)
            current_price = (orderbook.bid_price + orderbook.ask_price) / 2
            psych_levels = self._find_psychological_levels_in_orderbook(current_price, orderbook)
            
            return {
                'large_bid_orders': len(bid_clusters),
                'large_ask_orders': len(ask_clusters),
                'bid_clusters': bid_clusters[:5],  # Top 5
                'ask_clusters': ask_clusters[:5],  # Top 5
                'psychological_level_strength': psych_levels
            }
            
        except Exception as e:
            logger.error(f"Error calculating order clustering: {e}")
            return {'large_bid_orders': 0, 'large_ask_orders': 0}
    
    def _calculate_orderbook_support_resistance(self, orderbook: OrderBookData) -> Dict[str, Any]:
        """Calculate support/resistance levels from orderbook data."""
        try:
            current_price = (orderbook.bid_price + orderbook.ask_price) / 2
            
            # Find significant price levels based on order sizes
            support_levels = []
            resistance_levels = []
            
            # Analyze bid levels (support)
            if orderbook.bids:
                bid_sizes = [size for _, size in orderbook.bids]
                avg_bid_size = sum(bid_sizes) / len(bid_sizes)
                significant_threshold = avg_bid_size * 2
                
                for price, size in orderbook.bids:
                    if size > significant_threshold:
                        support_levels.append({
                            'price': price,
                            'strength': round(size / avg_bid_size, 2),
                            'distance_pct': round(((current_price - price) / current_price) * 100, 2)
                        })
            
            # Analyze ask levels (resistance)
            if orderbook.asks:
                ask_sizes = [size for _, size in orderbook.asks]
                avg_ask_size = sum(ask_sizes) / len(ask_sizes)
                significant_threshold = avg_ask_size * 2
                
                for price, size in orderbook.asks:
                    if size > significant_threshold:
                        resistance_levels.append({
                            'price': price,
                            'strength': round(size / avg_ask_size, 2),
                            'distance_pct': round(((price - current_price) / current_price) * 100, 2)
                        })
            
            # Sort by proximity to current price
            support_levels.sort(key=lambda x: x['distance_pct'])
            resistance_levels.sort(key=lambda x: x['distance_pct'])
            
            return {
                'orderbook_support_levels': support_levels[:3],  # Top 3
                'orderbook_resistance_levels': resistance_levels[:3],  # Top 3
                'nearest_support': support_levels[0] if support_levels else None,
                'nearest_resistance': resistance_levels[0] if resistance_levels else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating orderbook S/R levels: {e}")
            return {'orderbook_support_levels': [], 'orderbook_resistance_levels': []}
    
    def _find_psychological_levels_in_orderbook(self, current_price: float, orderbook: OrderBookData) -> Dict[str, float]:
        """Find psychological price levels with significant order flow."""
        try:
            # Define psychological levels based on price
            if current_price >= 1000:
                round_factor = 100
            elif current_price >= 100:
                round_factor = 10
            elif current_price >= 10:
                round_factor = 1
            elif current_price >= 1:
                round_factor = 0.1
            else:
                round_factor = 0.01
            
            # Find orders at psychological levels
            psych_bid_volume = 0
            psych_ask_volume = 0
            
            for price, size in orderbook.bids:
                if abs(price - round(price / round_factor) * round_factor) < round_factor * 0.1:
                    psych_bid_volume += size
            
            for price, size in orderbook.asks:
                if abs(price - round(price / round_factor) * round_factor) < round_factor * 0.1:
                    psych_ask_volume += size
            
            total_bid_volume = sum(size for _, size in orderbook.bids)
            total_ask_volume = sum(size for _, size in orderbook.asks)
            
            psych_bid_ratio = psych_bid_volume / total_bid_volume if total_bid_volume > 0 else 0
            psych_ask_ratio = psych_ask_volume / total_ask_volume if total_ask_volume > 0 else 0
            
            return {
                'psychological_bid_ratio': round(psych_bid_ratio, 3),
                'psychological_ask_ratio': round(psych_ask_ratio, 3),
                'psychological_strength': round((psych_bid_ratio + psych_ask_ratio) / 2, 3)
            }
            
        except Exception as e:
            logger.error(f"Error finding psychological levels: {e}")
            return {'psychological_strength': 0.0}
    
    def _default_orderbook_analysis(self) -> Dict[str, Any]:
        """Return default orderbook analysis when data is unavailable."""
        return {
            'symbol': '',
            'timestamp': datetime.now().isoformat(),
            'spread': 0.0,
            'spread_percentage': 0.0,
            'order_flow': 'NEUTRAL',
            'flow_strength': 0.0,
            'imbalance': 0.0,
            'imbalance_direction': 'NEUTRAL',
            'liquidity_score': 50.0,
            'liquidity_quality': 'MODERATE',
            'large_bid_orders': 0,
            'large_ask_orders': 0,
            'orderbook_support_levels': [],
            'orderbook_resistance_levels': []
        }


# Factory function
def create_binance_service(api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False) -> BinanceService:
    """Create and return BinanceService instance."""
    return BinanceService(api_key, api_secret, testnet)
