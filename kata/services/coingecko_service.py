"""
CoinGecko API Service for Yuki Agent

Provides comprehensive cryptocurrency market data including:
- Real-time prices and market cap data
- Historical price data and charts
- Market trends and sentiment indicators
- Trending coins and market movers
- Global market statistics
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import ssl
import time

logger = logging.getLogger(__name__)


@dataclass
class CoinPrice:
    """Coin price data structure."""
    symbol: str
    name: str
    current_price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    price_change_percentage_7d: float
    market_cap_rank: int
    timestamp: datetime


@dataclass
class MarketTrend:
    """Market trend data structure."""
    trending_coins: List[str]
    trending_categories: List[str]
    market_sentiment: str
    fear_greed_index: Optional[int]
    global_market_cap: float
    total_volume: float
    btc_dominance: float
    timestamp: datetime


@dataclass
class HistoricalData:
    """Historical price data structure."""
    symbol: str
    prices: List[tuple]  # [(timestamp, price), ...]
    market_caps: List[tuple]
    volumes: List[tuple]
    timeframe: str
    timestamp: datetime


class CoinGeckoService:
    """
    CoinGecko API service with multiple API key rotation for comprehensive cryptocurrency market data.

    Supports multiple API keys for increased rate limits and improved reliability.
    """

    def __init__(self, api_keys: Optional[List[str]] = None):
        """Initialize CoinGecko service with multiple API keys."""
        # Support multiple API keys for rotation
        if api_keys is None:
            try:
                from kata.config.settings import settings
                # Try to get multiple API keys from settings
                api_keys_str = getattr(settings, 'COINGECKO_API_KEYS', None)
                if api_keys_str:
                    self.api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
                else:
                    self.api_keys = []
            except Exception:
                self.api_keys = []
        else:
            self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]

        self.current_key_index = 0  # Track which API key to use

        try:
            from kata.config.settings import settings
            free_base = getattr(settings, 'COINGECKO_BASE_URL_FREE', "https://api.coingecko.com/api/v3")
            self.base_url = free_base
        except Exception:
            self.base_url = "https://api.coingecko.com/api/v3"
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Rate limiting - adjusted for multiple keys
        try:
            from kata.config.settings import settings
            self.rate_limit_calls = int(getattr(settings, 'COINGECKO_FREE_RATE_CALLS_PER_MINUTE', 15))
            self.rate_limit_window = 60  # seconds
            self.min_delay_seconds = float(getattr(settings, 'COINGECKO_FREE_MIN_DELAY_SECONDS', 2.0))  # Reduced delay with multiple keys
            self.error_backoff_base = int(getattr(settings, 'COINGECKO_FREE_ERROR_BACKOFF_BASE', 30))  # Reduced backoff
            self.error_backoff_max = int(getattr(settings, 'COINGECKO_FREE_ERROR_BACKOFF_MAX', 300))   # Reduced max backoff
        except Exception:
            self.rate_limit_calls = 15
            self.rate_limit_window = 60
            self.min_delay_seconds = 2.0
            self.error_backoff_base = 30
            self.error_backoff_max = 300

        # Track call timestamps per API key
        self.call_timestamps = {i: [] for i in range(len(self.api_keys))} if self.api_keys else {0: []}
        self._rate_limit_locks = {
            key_index: asyncio.Lock() for key_index in self.call_timestamps
        }
        
        # Aggressive cache settings for performance optimization
        self.cache_duration = {
            'prices': 900,  # 15 minutes for market data (performance optimization)
            'market_data': 900,  # 15 minutes for market data
            'trending': 900,  # 15 minutes for trending data
            'historical': 3600,  # 1 hour for historical data
            'metadata': 86400,  # 24 hours for token metadata (name, image, etc.)
            'coin_list': 86400,  # 24 hours for coin list/search data
            'basic_info': 86400  # 24 hours for basic token info (stable data)
        }
        self.cache = {}
        
        logger.info(f"CoinGeckoService initialized with {len(self.api_keys)} API keys for rotation" if self.api_keys else "CoinGeckoService initialized with free tier (no API keys)")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session with SSL context."""
        if self.session and not self.session.closed:
            return
        async with self._session_lock:
            if self.session and not self.session.closed:
                return
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("CoinGecko HTTP session started with SSL bypass")
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("CoinGecko HTTP session closed")
    
    def _get_next_api_key(self) -> tuple[Optional[str], int]:
        """Get next available API key using round-robin rotation."""
        if not self.api_keys:
            return None, 0

        # Find the API key with the least recent usage
        now = time.time()
        best_key_index = 0
        best_available_calls = 0

        for i, key in enumerate(self.api_keys):
            # Clean old timestamps for this key
            if i in self.call_timestamps:
                self.call_timestamps[i] = [ts for ts in self.call_timestamps[i] if now - ts < self.rate_limit_window]
                available_calls = self.rate_limit_calls - len(self.call_timestamps[i])
            else:
                self.call_timestamps[i] = []
                available_calls = self.rate_limit_calls

            # Check if this key has more available calls
            if available_calls > best_available_calls:
                best_available_calls = available_calls
                best_key_index = i

        return self.api_keys[best_key_index], best_key_index

    async def _check_rate_limit(self, key_index: int = 0):
        """Check and enforce rate limiting for specific API key."""
        if key_index not in self.call_timestamps:
            self.call_timestamps[key_index] = []
        lock = self._rate_limit_locks.setdefault(key_index, asyncio.Lock())
        async with lock:
            now = time.time()
            timestamps = [
                ts for ts in self.call_timestamps[key_index]
                if now - ts < self.rate_limit_window
            ]
            if len(timestamps) >= self.rate_limit_calls:
                sleep_time = self.rate_limit_window - (now - timestamps[0])
                if sleep_time > 0:
                    logger.warning(
                        "Rate limit reached for API key %s. Waiting asynchronously for %.2f seconds",
                        key_index + 1,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)

            now = time.time()
            timestamps = [ts for ts in timestamps if now - ts < self.rate_limit_window]
            if timestamps:
                time_since_last = now - timestamps[-1]
                min_delay = getattr(self, 'min_delay_seconds', 2.0)
                if time_since_last < min_delay:
                    await asyncio.sleep(min_delay - time_since_last)

            timestamps.append(time.time())
            self.call_timestamps[key_index] = timestamps
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{param_str}"
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = time.time() - cached_data['timestamp']
        return cache_age < self.cache_duration.get(cache_type, 60)
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None, max_retries: int = 3) -> Dict[str, Any]:
        """Make HTTP request to CoinGecko API with API key rotation, rate limiting, caching, and exponential backoff."""
        if not self.session:
            await self.start_session()

        # Check cache first
        cache_key = self._get_cache_key(endpoint, params or {})
        cache_type = endpoint.split('/')[0]  # Use first part of endpoint for cache type

        if self._is_cache_valid(cache_key, cache_type):
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]['data']

        # Get next available API key
        current_api_key, key_index = self._get_next_api_key()

        # Apply rate limiting for this specific API key
        await self._check_rate_limit(key_index)

        # Prepare request
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'accept': 'application/json',
            'User-Agent': 'FlowAI-TradingPlatform/1.0'
        }

        # Add API key if available - use URL parameter for demo keys
        if current_api_key:
            if params is None:
                params = {}
            params['x_cg_demo_api_key'] = current_api_key
            logger.debug(f"Using demo API key {key_index + 1} for request to {endpoint}")
        
        for attempt in range(max_retries + 1):
            try:
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:  # Rate limited
                        # Try next API key if available
                        if len(self.api_keys) > 1 and attempt < max_retries:
                            logger.warning(f"Rate limited with API key {key_index + 1}, trying next key...")
                            current_api_key, key_index = self._get_next_api_key()
                            if current_api_key:
                                params['x_cg_demo_api_key'] = current_api_key
                                logger.info(f"Switched to API key {key_index + 1}")
                                continue

                        wait_time = min(getattr(self, 'error_backoff_base', 30) * (2 ** attempt), getattr(self, 'error_backoff_max', 300))
                        logger.warning(f"Rate limited by CoinGecko API (attempt {attempt + 1}), waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    elif response.status == 403:  # Forbidden - likely rate limited or blocked
                        # Try next API key if available
                        if len(self.api_keys) > 1 and attempt < max_retries:
                            logger.warning(f"Forbidden with API key {key_index + 1}, trying next key...")
                            current_api_key, key_index = self._get_next_api_key()
                            if current_api_key:
                                params['x_cg_demo_api_key'] = current_api_key
                                logger.info(f"Switched to API key {key_index + 1}")
                                continue

                        wait_time = min(getattr(self, 'error_backoff_base', 30) * (2 ** attempt), getattr(self, 'error_backoff_max', 300))
                        logger.warning(f"CoinGecko API forbidden (attempt {attempt + 1}), waiting {wait_time}s")
                        if attempt < max_retries:
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error("Max retries reached for 403 error")
                            return None
                    
                    elif response.status >= 500:  # Server error
                        wait_time = min(30 * (2 ** attempt), 180)  # Shorter wait for server errors
                        logger.warning(f"CoinGecko API server error {response.status} (attempt {attempt + 1}), waiting {wait_time}s")
                        if attempt < max_retries:
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Max retries reached for server error {response.status}")
                            return None
                    
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Cache the response
                    self.cache[cache_key] = {
                        'data': data,
                        'timestamp': time.time()
                    }
                    
                    return data
                    
            except asyncio.TimeoutError:
                wait_time = min(30 * (2 ** attempt), 120)
                logger.warning(f"CoinGecko API timeout (attempt {attempt + 1}), waiting {wait_time}s")
                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached for timeout")
                    return None
                    
            except Exception as e:
                if attempt < max_retries:
                    wait_time = min(30 * (2 ** attempt), 120)
                    logger.warning(f"CoinGecko API request failed (attempt {attempt + 1}): {e}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"CoinGecko API request failed after {max_retries} retries: {e}")
                    return None
        
        return None
    
    async def get_coin_prices(self, symbols: List[str], vs_currency: str = "usd") -> List[CoinPrice]:
        """
        Get current prices for multiple coins.
        
        Args:
            symbols: List of coin symbols (e.g., ['bitcoin', 'ethereum'])
            vs_currency: Currency to price against (default: 'usd')
            
        Returns:
            List of CoinPrice objects
        """
        try:
            # Convert symbols to CoinGecko IDs if needed
            coin_ids = await self._get_coin_ids(symbols)
            
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': vs_currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_7d_change': 'true',
                'include_market_cap_rank': 'true'
            }
            
            data = await self._make_request('simple/price', params)
            
            prices = []
            for coin_id, coin_data in data.items():
                if coin_data:
                    prices.append(CoinPrice(
                        symbol=coin_id,
                        name=coin_id.replace('-', ' ').title(),
                        current_price=float(coin_data.get(vs_currency, 0)),
                        market_cap=float(coin_data.get(f'{vs_currency}_market_cap', 0)),
                        volume_24h=float(coin_data.get(f'{vs_currency}_24h_vol', 0)),
                        price_change_24h=float(coin_data.get(f'{vs_currency}_24h_change', 0)),
                        price_change_percentage_24h=float(coin_data.get(f'{vs_currency}_24h_change', 0)),
                        price_change_percentage_7d=float(coin_data.get(f'{vs_currency}_7d_change', 0) or 0),
                        market_cap_rank=int(coin_data.get('market_cap_rank', 0) or 0),
                        timestamp=datetime.now()
                    ))
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching coin prices: {e}")
            return []
    
    async def get_coin_prices_batched(self, symbols: List[str], batch_size: int = 5, vs_currency: str = 'usd') -> List[CoinPrice]:
        """
        Get coin prices in batches to avoid rate limits.
        
        Args:
            symbols: List of coin symbols
            batch_size: Number of symbols per batch
            vs_currency: Currency to price against
            
        Returns:
            List of CoinPrice objects
        """
        all_prices = []
        
        # Process symbols in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                logger.info(f"Fetching batch {i//batch_size + 1}: {batch}")
                batch_prices = await self.get_coin_prices(batch, vs_currency)
                all_prices.extend(batch_prices)
                
                # Add delay between batches to respect rate limits
                if i + batch_size < len(symbols):
                    await asyncio.sleep(2.0)  # 2 second delay between batches
                    
            except Exception as e:
                logger.warning(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                # Continue with next batch even if one fails
                continue
        
        logger.info(f"Fetched {len(all_prices)} coin prices from {len(symbols)} requested symbols")
        return all_prices
    
    async def get_trending_data(self) -> MarketTrend:
        """
        Get trending coins and market sentiment data.
        
        Returns:
            MarketTrend object with trending information
        """
        try:
            # Get trending coins
            trending_data = await self._make_request('search/trending')
            trending_coins = [coin['item']['symbol'].upper() for coin in trending_data.get('coins', [])[:10]]
            trending_categories = [cat['name'] for cat in trending_data.get('categories', [])[:5]]
            
            # Get global market data
            global_data = await self._make_request('global')
            global_info = global_data.get('data', {})
            
            # Get fear & greed index (if available)
            fear_greed = None
            try:
                fg_data = await self._make_request('indexes')
                fear_greed = fg_data[0].get('value') if fg_data else None
            except:
                pass  # Fear & greed index not always available
            
            return MarketTrend(
                trending_coins=trending_coins,
                trending_categories=trending_categories,
                market_sentiment="neutral",  # Basic sentiment - can be enhanced
                fear_greed_index=fear_greed,
                global_market_cap=float(global_info.get('total_market_cap', {}).get('usd', 0)),
                total_volume=float(global_info.get('total_volume', {}).get('usd', 0)),
                btc_dominance=float(global_info.get('market_cap_percentage', {}).get('btc', 0)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching trending data: {e}")
            return MarketTrend(
                trending_coins=[],
                trending_categories=[],
                market_sentiment="unknown",
                fear_greed_index=None,
                global_market_cap=0.0,
                total_volume=0.0,
                btc_dominance=0.0,
                timestamp=datetime.now()
            )
    
    async def get_historical_data(self, symbol: str, days: int = 30, vs_currency: str = "usd") -> HistoricalData:
        """
        Get historical price data for a coin.
        
        Args:
            symbol: Coin symbol (e.g., 'bitcoin')
            days: Number of days of data (1-max)
            vs_currency: Currency to price against
            
        Returns:
            HistoricalData object
        """
        try:
            coin_id = await self._get_coin_id(symbol)
            
            params = {
                'vs_currency': vs_currency,
                'days': str(days),
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            
            data = await self._make_request(f'coins/{coin_id}/market_chart', params)
            
            return HistoricalData(
                symbol=symbol,
                prices=data.get('prices', []),
                market_caps=data.get('market_caps', []),
                volumes=data.get('total_volumes', []),
                timeframe=f"{days}d",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return HistoricalData(
                symbol=symbol,
                prices=[],
                market_caps=[],
                volumes=[],
                timeframe=f"{days}d",
                timestamp=datetime.now()
            )
    
    async def get_market_movers(self, limit: int = 10) -> Dict[str, List[CoinPrice]]:
        """
        Get top gainers and losers in the market.
        
        Args:
            limit: Number of coins to return for each category
            
        Returns:
            Dict with 'gainers' and 'losers' lists
        """
        try:
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': str(limit * 3),  # Get more to filter
                'page': '1',
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }
            
            data = await self._make_request('coins/markets', params)
            
            # Convert to CoinPrice objects
            coins = []
            for coin_data in data:
                coins.append(CoinPrice(
                    symbol=coin_data['symbol'].upper(),
                    name=coin_data['name'],
                    current_price=float(coin_data['current_price'] or 0),
                    market_cap=float(coin_data['market_cap'] or 0),
                    volume_24h=float(coin_data['total_volume'] or 0),
                    price_change_24h=float(coin_data['price_change_24h'] or 0),
                    price_change_percentage_24h=float(coin_data['price_change_percentage_24h'] or 0),
                    price_change_percentage_7d=float(coin_data.get('price_change_percentage_7d_in_currency', 0) or 0),
                    market_cap_rank=int(coin_data['market_cap_rank'] or 0),
                    timestamp=datetime.now()
                ))
            
            # Sort by 24h change percentage
            sorted_coins = sorted(coins, key=lambda x: x.price_change_percentage_24h, reverse=True)
            
            return {
                'gainers': sorted_coins[:limit],
                'losers': sorted_coins[-limit:]
            }
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
            return {'gainers': [], 'losers': []}
    
    async def _get_coin_ids(self, symbols: List[str]) -> List[str]:
        """Convert symbols to CoinGecko coin IDs using search API."""
        results = []
        
        for symbol in symbols:
            try:
                coin_id = await self._search_coin_id(symbol)
                results.append(coin_id)
            except Exception as e:
                logger.warning(f"Failed to find coin ID for {symbol}: {e}")
                # Fallback to lowercase symbol
                results.append(symbol.lower())
        
        return results
    
    async def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID."""
        try:
            return await self._search_coin_id(symbol)
        except Exception as e:
            logger.warning(f"Failed to find coin ID for {symbol}: {e}")
            return symbol.lower()

    async def _search_coin_id(self, symbol: str) -> str:
        """Search for coin ID using CoinGecko search API with aggressive caching."""
        try:
            # Hardcoded overrides for problematic symbols
            overrides = {
                'H': 'humanity-protocol',
            }
            if symbol.upper() in overrides:
                return overrides[symbol.upper()]

            # Check cache first (24 hour cache for coin ID lookups)
            cache_key = f"coin_id_{symbol.upper()}"
            if self._is_cache_valid(cache_key, 'basic_info'):
                cached_result = self.cache[cache_key]['data']
                logger.debug(f"Using cached coin ID for {symbol}: {cached_result}")
                return cached_result

            identity = await self.resolve_coin_identity(symbol)
            coin_id = identity['coingecko_id']

            # Cache the result for 24 hours to avoid repeated API calls
            self.cache[cache_key] = {
                'data': coin_id,
                'timestamp': time.time()
            }
            return coin_id

        except Exception as e:
            logger.error(f"Error searching for {symbol}: {e}")
            # Fallback to symbol
            return symbol.lower()

    async def resolve_coin_identity(self, symbol: str, coingecko_id: Optional[str] = None) -> Dict[str, Any]:
        """Resolve a ticker deterministically and expose ambiguity instead of hiding it."""
        normalized_symbol = str(symbol or '').upper().strip()
        if coingecko_id:
            return {
                'coingecko_id': str(coingecko_id).strip().lower(),
                'symbol': normalized_symbol,
                'resolution_method': 'explicit_coingecko_id',
                'ambiguous_matches': 0,
                'verified': True,
            }

        overrides = {'H': 'humanity-protocol'}
        if normalized_symbol in overrides:
            return {
                'coingecko_id': overrides[normalized_symbol],
                'symbol': normalized_symbol,
                'resolution_method': 'verified_override',
                'ambiguous_matches': 0,
                'verified': True,
            }

        data = await self._make_request('search', {'query': normalized_symbol}) or {}
        coins = data.get('coins') or []
        exact_matches = [
            coin for coin in coins
            if str(coin.get('symbol') or '').upper() == normalized_symbol and coin.get('id')
        ]

        def rank_key(coin: Dict[str, Any]):
            rank = coin.get('market_cap_rank')
            try:
                rank_value = int(rank)
            except (TypeError, ValueError):
                rank_value = 10**9
            return (rank_value <= 0, rank_value, str(coin.get('id') or ''))

        ranked = sorted(exact_matches, key=rank_key)
        if ranked:
            selected = ranked[0]
            return {
                'coingecko_id': selected['id'],
                'symbol': normalized_symbol,
                'name': selected.get('name'),
                'market_cap_rank': selected.get('market_cap_rank'),
                'resolution_method': 'unique_symbol' if len(ranked) == 1 else 'ranked_symbol',
                'ambiguous_matches': max(0, len(ranked) - 1),
                'verified': len(ranked) == 1,
            }

        if coins and coins[0].get('id'):
            selected = coins[0]
            return {
                'coingecko_id': selected['id'],
                'symbol': normalized_symbol,
                'name': selected.get('name'),
                'market_cap_rank': selected.get('market_cap_rank'),
                'resolution_method': 'search_fallback',
                'ambiguous_matches': len(coins),
                'verified': False,
            }

        return {
            'coingecko_id': normalized_symbol.lower(),
            'symbol': normalized_symbol,
            'resolution_method': 'unverified_fallback',
            'ambiguous_matches': 0,
            'verified': False,
        }

    @staticmethod
    def _asset_platform_for_chain(chain_id: Optional[int]) -> Optional[str]:
        return {
            1: 'ethereum',
            10: 'optimistic-ethereum',
            56: 'binance-smart-chain',
            101: 'solana',
            137: 'polygon-pos',
            250: 'fantom',
            324: 'zksync',
            8453: 'base',
            42161: 'arbitrum-one',
            43114: 'avalanche',
            59144: 'linea',
            534352: 'scroll',
        }.get(chain_id)

    async def get_token_data(
        self,
        symbol: str,
        *,
        coingecko_id: Optional[str] = None,
        contract_address: Optional[str] = None,
        chain_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive data for a single token with aggressive caching.

        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')

        Returns:
            Dict with token data or None if not found
        """
        try:
            # Check cache first (15 minutes for market data, 24h for metadata)
            identity_key = coingecko_id or (f"{chain_id}:{contract_address}" if contract_address else symbol.upper())
            cache_key = f"token_data_{identity_key}"
            if self._is_cache_valid(cache_key, 'market_data'):
                cached_result = self.cache[cache_key]['data']
                logger.debug(f"Using cached token data for {symbol}")
                return cached_result

            # Get detailed coin data
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }

            identity: Dict[str, Any]
            platform = self._asset_platform_for_chain(chain_id)
            if contract_address and not platform:
                logger.warning("Unsupported chain identity for %s: chain_id=%s", symbol, chain_id)
                return None
            if contract_address and platform:
                data = await self._make_request(
                    f"coins/{platform}/contract/{contract_address}",
                    params,
                )
                identity = {
                    'coingecko_id': (data or {}).get('id'),
                    'symbol': str((data or {}).get('symbol') or symbol).upper(),
                    'requested_symbol': symbol.upper(),
                    'contract_address': contract_address,
                    'chain_id': chain_id,
                    'resolution_method': 'contract_address',
                    'ambiguous_matches': 0,
                    'verified': bool(data),
                }
            else:
                identity = await self.resolve_coin_identity(symbol, coingecko_id=coingecko_id)
                data = await self._make_request(f"coins/{identity['coingecko_id']}", params)
            
            if not data:
                return None
            identity['resolved_symbol'] = str(data.get('symbol') or '').upper()
            identity['requested_symbol'] = symbol.upper()
            
            # Extract relevant data
            market_data = data.get('market_data', {})

            # Extract image data
            image_data = data.get('image', {})

            result = {
                'id': data.get('id'),
                'symbol': data.get('symbol', '').upper(),
                'name': data.get('name'),
                'description': (data.get('description') or {}).get('en', ''),
                'categories': [
                    str(category)
                    for category in (data.get('categories') or [])
                    if category
                ][:8],
                'image': {
                    'thumb': image_data.get('thumb'),
                    'small': image_data.get('small'),
                    'large': image_data.get('large')
                },
                'current_price': float(market_data.get('current_price', {}).get('usd', 0)),
                'market_cap': float(market_data.get('market_cap', {}).get('usd', 0)),
                'market_cap_rank': int(market_data.get('market_cap_rank', 0)),
                'total_volume': float(market_data.get('total_volume', {}).get('usd', 0)),
                'price_change_percentage_24h': float(market_data.get('price_change_percentage_24h', 0)),
                'price_change_percentage_7d': float(market_data.get('price_change_percentage_7d', 0)),
                'circulating_supply': float(market_data.get('circulating_supply', 0)),
                'total_supply': float(market_data.get('total_supply', 0)),
                'ath': float(market_data.get('ath', {}).get('usd', 0)),
                'atl': float(market_data.get('atl', {}).get('usd', 0)),
                'ath_change_percentage': float(market_data.get('ath_change_percentage', {}).get('usd', 0)),
                'atl_change_percentage': float(market_data.get('atl_change_percentage', {}).get('usd', 0)),
                'last_updated': data.get('last_updated'),
                # Contract addresses from platforms
                'platforms': data.get('platforms', {}),
                'contract_address': data.get('platforms', {}).get('ethereum', ''),
                'resolution': identity,
            }

            # Cache the result for 15 minutes to dramatically reduce API calls
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            return result

        except Exception as e:
            logger.error(f"Error fetching token data for {symbol}: {e}")
            return None


# Factory function
def create_coingecko_service(api_keys: Optional[List[str]] = None) -> CoinGeckoService:
    """Create and return CoinGeckoService instance with multiple API keys."""
    return CoinGeckoService(api_keys)
