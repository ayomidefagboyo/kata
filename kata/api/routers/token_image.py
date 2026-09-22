from fastapi import APIRouter
from fastapi.responses import JSONResponse
import httpx
import asyncio
from typing import Optional
import logging
import time
from kata.config.settings import settings
from kata.services.token_logo_service import get_token_logo_resolver, normalize_market_symbol

logger = logging.getLogger(__name__)

router = APIRouter()

# Environment configuration
COINGECKO_API_KEY = settings.COINGECKO_API_KEY
# TEMPORARILY DISABLE Pro API for token images due to 400 errors - use free tier as fallback only
COINGECKO_PRO_API = False  # Disabled to prevent unnecessary paid API calls

# Optimized rate limiting based on API tier
if COINGECKO_PRO_API:
    API_CALL_LIMIT_PER_MINUTE = 500  # Pro tier: 500 calls/minute
    RATE_LIMIT_WINDOW = 60  # 1 minute
else:
    API_CALL_LIMIT_PER_MINUTE = 30   # Free tier: 30 calls/minute  
    RATE_LIMIT_WINDOW = 60  # 1 minute

# Rate limiting tracking
api_calls_in_window = []
last_cleanup_time = time.time()

class TokenImageCache:
    """Simple in-memory cache for token images"""
    def __init__(self):
        self.cache = {}
        self.cache_duration = 7 * 24 * 60 * 60  # 7 days for successful images
        self.negative_cache_duration = 60 * 60  # retry missing images after 1 hour
    
    def get(self, symbol: str) -> Optional[dict]:
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            # A database fallback is safe to display, but it is not considered a
            # fresh provider resolution. Retry it on the shorter cadence so an
            # old logo can self-heal without a process restart.
            is_fresh_resolution = data.get('imageUrl') and data.get('source') != 'stored_signal'
            cache_duration = self.cache_duration if is_fresh_resolution else self.negative_cache_duration
            if asyncio.get_event_loop().time() - timestamp < cache_duration:
                return data
            else:
                del self.cache[symbol]
        return None
    
    def set(self, symbol: str, data: dict):
        self.cache[symbol] = (data, asyncio.get_event_loop().time())

# Global cache instance
image_cache = TokenImageCache()

def can_make_api_call() -> bool:
    """Check if we can make an API call within rate limits"""
    global api_calls_in_window, last_cleanup_time
    now = time.time()
    
    # Clean up old API calls outside the current window
    if now - last_cleanup_time > 10:  # Cleanup every 10 seconds
        api_calls_in_window = [call_time for call_time in api_calls_in_window 
                              if now - call_time < RATE_LIMIT_WINDOW]
        last_cleanup_time = now
    
    return len(api_calls_in_window) < API_CALL_LIMIT_PER_MINUTE

def record_api_call():
    """Record an API call for rate limiting"""
    global api_calls_in_window
    api_calls_in_window.append(time.time())


@router.get("/token-image/{symbol}")
async def get_token_image(symbol: str):
    """
    Resolve one stable token image and persist it back to platform signals.

    Market identity is preserved for namespaced HIP-3 assets, so an overlapping
    crypto ticker cannot replace a stock/perp logo. All consumers subsequently
    receive the same database-backed result.
    """
    try:
        logger.debug(f"[TOKEN_IMAGE] Request for symbol: {symbol}")
        requested_symbol = normalize_market_symbol(symbol)

        # Check cache first
        cached_data = image_cache.get(requested_symbol)
        if cached_data:
            logger.debug(f"[TOKEN_IMAGE] Cache hit for {requested_symbol}")
            return JSONResponse(
                content=cached_data,
                headers={"Cache-Control": "public, max-age=604800, s-maxage=604800, stale-while-revalidate=86400"}
            )

        platform_service = None
        matching_signals = []
        try:
            from kata.services.platform_signal_service import get_platform_signal_service
            platform_service = get_platform_signal_service()
            matching_signals = await platform_service.get_signals_by_symbol(requested_symbol, limit=25)
        except Exception as db_error:
            logger.warning(f"[TOKEN_IMAGE] Database lookup failed for {requested_symbol}: {db_error}")

        identity_symbol = matching_signals[0].token_symbol if matching_signals else requested_symbol
        stored_logo = next(
            (
                signal.logo_url.strip()
                for signal in matching_signals
                if isinstance(signal.logo_url, str) and signal.logo_url.strip()
            ),
            None,
        )
        resolved = await get_token_logo_resolver().resolve(
            identity_symbol,
            stored_logo_url=stored_logo,
        )

        if resolved:
            token_data = {
                'imageUrl': resolved.url,
                'name': identity_symbol,
                'symbol': requested_symbol,
                'source': resolved.source,
            }
            image_cache.set(requested_symbol, token_data)

            if platform_service and resolved.url != stored_logo:
                try:
                    updated = await platform_service.update_signal_logos_by_symbol(identity_symbol, resolved.url)
                    logger.info(
                        "[TOKEN_IMAGE] Persisted resolved %s logo to %s signal row(s)",
                        identity_symbol,
                        updated,
                    )
                except Exception as persist_error:
                    logger.warning(
                        "[TOKEN_IMAGE] Could not persist resolved logo for %s: %s",
                        identity_symbol,
                        persist_error,
                    )
            return JSONResponse(content=token_data)

        # Cache negative result to avoid repeated failed requests
        negative_result = {'imageUrl': None, 'name': identity_symbol, 'symbol': requested_symbol}
        image_cache.set(requested_symbol, negative_result)

        logger.warning(f"[TOKEN_IMAGE] No identity-safe image found for {identity_symbol}")
        return JSONResponse(content=negative_result)

    except Exception as e:
        logger.error(f"[TOKEN_IMAGE] Error for {symbol}: {e}")
        negative_result = {'imageUrl': None, 'name': symbol, 'symbol': symbol}
        return JSONResponse(content=negative_result, status_code=200)

@router.get("/token-image-stats")
async def get_token_image_stats():
    """Get statistics about the identity-aware token image service."""
    return JSONResponse(content={
        'primary_source': 'Persisted market identity',
        'identity_sources': ['Platform signals', 'Hyperliquid perp annotations'],
        'logo_sources': ['CoinGecko exact-symbol search', 'Binance CDN for default markets'],
        'coingecko_api_configured': bool(COINGECKO_API_KEY),
        'api_calls_in_window': len(api_calls_in_window),
        'api_limit_per_minute': API_CALL_LIMIT_PER_MINUTE,
        'rate_limit_window': RATE_LIMIT_WINDOW,
        'coingecko_pro_api': COINGECKO_PRO_API,
        'cache_size': len(image_cache.cache),
        'can_make_api_call': can_make_api_call()
    })

@router.get("/coingecko-markets")
async def get_coingecko_markets(
    vs_currency: str = "usd",
    order: str = "market_cap_desc", 
    per_page: int = 100,
    page: int = 1,
    sparkline: bool = False,
    locale: str = "en"
):
    """
    Proxy CoinGecko markets endpoint to avoid CORS and control rate limiting
    Used by token discovery service
    """
    try:
        logger.info(f"Fetching CoinGecko markets page {page}")
        
        # Check rate limits
        if not can_make_api_call():
            logger.warning(f"Rate limit reached for markets page {page}")
            return JSONResponse(content=[], status_code=429)
        
        # Prepare headers with API key if available
        headers = {'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)'}
        api_base = "https://api.coingecko.com/api/v3"
        
        if COINGECKO_PRO_API:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
            api_base = "https://api.coingecko.com/api/v3"
        
        # Fetch from CoinGecko API
        async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=True) as client:
            response = await client.get(
                f"{api_base}/coins/markets",
                params={
                    "vs_currency": vs_currency,
                    "order": order,
                    "per_page": per_page,
                    "page": page,
                    "sparkline": sparkline,
                    "locale": locale
                },
                headers=headers
            )
            
            # Record API call for rate limiting
            record_api_call()
            
            if response.status_code != 200:
                logger.error(f"CoinGecko markets API error: {response.status_code}")
                return JSONResponse(content=[], status_code=response.status_code)
            
            data = response.json()
            logger.info(f"CoinGecko markets page {page} fetched successfully ({len(data)} tokens)")
            
            return JSONResponse(content=data)
            
    except Exception as e:
        logger.error(f"Error fetching CoinGecko markets page {page}: {e}")
        return JSONResponse(content=[], status_code=500)

@router.get("/coingecko-simple-price")
async def get_coingecko_simple_price(
    ids: str,
    vs_currencies: str = "usd",
    include_24hr_change: bool = True,
    include_24hr_vol: bool = True,
    include_market_cap: bool = True
):
    """
    Proxy CoinGecko simple price endpoint to avoid CORS and control rate limiting
    """
    try:
        logger.info(f"Fetching CoinGecko simple price for {ids}")
        
        # Check rate limits
        if not can_make_api_call():
            logger.warning(f"Rate limit reached for simple price {ids}")
            return JSONResponse(content={}, status_code=429)
        
        # Prepare headers with API key if available
        headers = {'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)'}
        api_base = "https://api.coingecko.com/api/v3"
        
        if COINGECKO_PRO_API:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
            api_base = "https://api.coingecko.com/api/v3"
        
        # Fetch from CoinGecko API
        async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=True) as client:
            response = await client.get(
                f"{api_base}/simple/price",
                params={
                    "ids": ids,
                    "vs_currencies": vs_currencies,
                    "include_24hr_change": include_24hr_change,
                    "include_24hr_vol": include_24hr_vol,
                    "include_market_cap": include_market_cap
                },
                headers=headers
            )
            
            # Record API call for rate limiting
            record_api_call()
            
            if response.status_code != 200:
                logger.error(f"CoinGecko simple price API error: {response.status_code}")
                return JSONResponse(content={}, status_code=response.status_code)
            
            data = response.json()
            logger.info(f"CoinGecko simple price for {ids} fetched successfully")
            
            return JSONResponse(content=data)
            
    except Exception as e:
        logger.error(f"Error fetching CoinGecko simple price for {ids}: {e}")
        return JSONResponse(content={}, status_code=500)

@router.get("/coingecko-coin-details/{coingecko_id}")
async def get_coingecko_coin_details(coingecko_id: str):
    """
    Proxy CoinGecko coin details endpoint to avoid CORS and control rate limiting
    Used for blockchain detection and detailed token information
    """
    try:
        logger.info(f"Fetching CoinGecko coin details for {coingecko_id}")
        
        # Check cache first
        cache_key = f"coin_details_{coingecko_id}"
        cached_data = image_cache.get(cache_key)
        if cached_data:
            logger.debug(f"Returning cached coin details for {coingecko_id}")
            return JSONResponse(content=cached_data)
        
        # Check rate limits
        if not can_make_api_call():
            logger.warning(f"Rate limit reached for coin details {coingecko_id}")
            return JSONResponse(content={}, status_code=429)
        
        # Prepare headers with API key if available
        headers = {'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)'}
        api_base = "https://api.coingecko.com/api/v3"
        
        if COINGECKO_PRO_API:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
            api_base = "https://api.coingecko.com/api/v3"
        
        # Fetch from CoinGecko API
        async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=True) as client:
            response = await client.get(f"{api_base}/coins/{coingecko_id}", headers=headers)
            
            # Record API call for rate limiting
            record_api_call()
            
            if response.status_code != 200:
                logger.error(f"CoinGecko coin details API error: {response.status_code}")
                return JSONResponse(content={}, status_code=response.status_code)
            
            data = response.json()
            
            # Cache the result
            image_cache.set(cache_key, data)
            logger.info(f"CoinGecko coin details for {coingecko_id} fetched successfully")
            
            return JSONResponse(content=data)
            
    except Exception as e:
        logger.error(f"Error fetching CoinGecko coin details for {coingecko_id}: {e}")
        return JSONResponse(content={}, status_code=500)

@router.get("/coingecko-market-chart/{coingecko_id}")
async def get_coingecko_market_chart(
    coingecko_id: str,
    vs_currency: str = "usd",
    days: int = 30,
    interval: str = "daily"
):
    """
    Proxy CoinGecko market chart endpoint to avoid CORS and control rate limiting
    Used for historical price data analysis
    """
    try:
        logger.info(f"Fetching CoinGecko market chart for {coingecko_id}")
        
        # Check cache first
        cache_key = f"market_chart_{coingecko_id}_{days}_{interval}"
        cached_data = image_cache.get(cache_key)
        if cached_data:
            logger.debug(f"Returning cached market chart for {coingecko_id}")
            return JSONResponse(content=cached_data)
        
        # Check rate limits
        if not can_make_api_call():
            logger.warning(f"Rate limit reached for market chart {coingecko_id}")
            return JSONResponse(content={"prices": [], "total_volumes": []}, status_code=429)
        
        # Prepare headers with API key if available
        headers = {'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)'}
        api_base = "https://api.coingecko.com/api/v3"
        
        if COINGECKO_PRO_API:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
            api_base = "https://api.coingecko.com/api/v3"
        
        # Fetch from CoinGecko API
        async with httpx.AsyncClient(timeout=15.0, verify=False, follow_redirects=True) as client:
            response = await client.get(
                f"{api_base}/coins/{coingecko_id}/market_chart",
                params={
                    "vs_currency": vs_currency,
                    "days": days,
                    "interval": interval
                },
                headers=headers
            )
            
            # Record API call for rate limiting
            record_api_call()
            
            if response.status_code != 200:
                logger.error(f"CoinGecko market chart API error: {response.status_code}")
                return JSONResponse(content={"prices": [], "total_volumes": []}, status_code=response.status_code)
            
            data = response.json()
            
            # Cache the result (shorter cache for market data)
            image_cache.set(cache_key, data)
            logger.info(f"CoinGecko market chart for {coingecko_id} fetched successfully")
            
            return JSONResponse(content=data)
            
    except Exception as e:
        logger.error(f"Error fetching CoinGecko market chart for {coingecko_id}: {e}")
        return JSONResponse(content={"prices": [], "total_volumes": []}, status_code=500)


@router.get("/token-contract/{coingecko_id}")
async def get_token_contract_address(coingecko_id: str):
    """
    Get token contract address and blockchain info from CoinGecko
    Used by token discovery service to avoid CORS issues
    """
    try:
        logger.info(f"Fetching contract address for {coingecko_id}")
        
        # Check cache first
        cache_key = f"contract_{coingecko_id}"
        cached_data = image_cache.get(cache_key)
        if cached_data:
            logger.debug(f"Returning cached contract data for {coingecko_id}")
            return JSONResponse(content=cached_data)
        
        # Check rate limits
        if not can_make_api_call():
            logger.warning(f"Rate limit reached for contract address {coingecko_id}")
            return JSONResponse(content={
                'address': '',
                'blockchain': 'ethereum',
                'chainId': 1,
                'error': 'Rate limit exceeded'
            })
        
        # Prepare headers with API key if available
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)'
        }
        api_base = "https://api.coingecko.com/api/v3"
        
        if COINGECKO_PRO_API:
            headers['x-cg-pro-api-key'] = COINGECKO_API_KEY
            api_base = "https://api.coingecko.com/api/v3"
        
        # Fetch from CoinGecko API
        async with httpx.AsyncClient(timeout=10.0, verify=False, follow_redirects=True) as client:
            response = await client.get(f"{api_base}/coins/{coingecko_id}", headers=headers)
            
            # Record API call for rate limiting
            record_api_call()
            
            if response.status_code != 200:
                return JSONResponse(content={
                    'address': '',
                    'blockchain': 'ethereum', 
                    'chainId': 1,
                    'error': f'API error: {response.status_code}'
                })
            
            data = response.json()
            platforms = data.get('platforms', {})
            
            # Platform priority order (prefer more popular chains)
            platform_priority = [
                {'id': 'ethereum', 'name': 'ethereum', 'chainId': 1},
                {'id': 'base', 'name': 'base', 'chainId': 8453},
                {'id': 'arbitrum-one', 'name': 'arbitrum', 'chainId': 42161},
                {'id': 'optimistic-ethereum', 'name': 'optimism', 'chainId': 10},
                {'id': 'polygon-pos', 'name': 'polygon', 'chainId': 137},
                {'id': 'binance-smart-chain', 'name': 'bsc', 'chainId': 56},
                {'id': 'avalanche', 'name': 'avalanche', 'chainId': 43114},
                {'id': 'fantom', 'name': 'fantom', 'chainId': 250}
            ]
            
            # Find first available platform with contract address
            for platform in platform_priority:
                address = platforms.get(platform['id'])
                if address and address != '':
                    result = {
                        'address': address,
                        'blockchain': platform['name'],
                        'chainId': platform['chainId']
                    }
                    
                    # Cache the result
                    image_cache.set(cache_key, result)
                    logger.info(f"Found contract address for {coingecko_id} on {platform['name']}")
                    
                    return JSONResponse(content=result)
            
            # No contract address found
            result = {
                'address': '',
                'blockchain': 'ethereum',
                'chainId': 1
            }
            
            # Cache the negative result too
            image_cache.set(cache_key, result)
            logger.info(f"No contract address found for {coingecko_id}")
            
            return JSONResponse(content=result)
            
    except Exception as e:
        logger.error(f"Error fetching contract address for {coingecko_id}: {e}")
        return JSONResponse(content={
            'address': '',
            'blockchain': 'ethereum',
            'chainId': 1,
            'error': str(e)
        })
