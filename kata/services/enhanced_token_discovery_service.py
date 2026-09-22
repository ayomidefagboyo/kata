"""
Enhanced Token Discovery Service

Discovers and filters tokens using CoinGecko API with comprehensive analysis.
Implements real-time filtering, technical analysis, and security assessment.
"""

import asyncio
import logging
import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredToken:
    """Enhanced token data structure with contract addresses for LiFi integration."""
    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    fully_diluted_valuation: Optional[float]
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    price_change_percentage_7d: float
    price_change_percentage_30d: float
    price_change_percentage_1y: Optional[float]
    market_cap_change_24h: float
    market_cap_change_percentage_24h: float
    circulating_supply: float
    total_supply: Optional[float]
    max_supply: Optional[float]
    ath: float
    ath_change_percentage: float
    ath_date: str
    atl: float
    atl_change_percentage: float
    atl_date: str
    last_updated: str
    # Contract address fields for LiFi integration
    contract_addresses: Dict[str, str] = field(default_factory=dict)  # Chain name -> contract address
    primary_chain_id: Optional[int] = None     # Primary chain for trading
    primary_address: Optional[str] = None      # Primary contract address
    # Derived fields
    trading_pairs_count: int = 0
    liquidity_score: float = 0.0
    volume_to_market_cap_ratio: float = 0.0


@dataclass
class FilterCriteria:
    """Token filtering criteria."""
    min_market_cap: float = 1_000_000  # $1M
    max_market_cap: float = 500_000_000  # $500M
    min_daily_volume: float = 200_000  # $200K
    min_trading_pairs: int = 3
    exclude_stablecoins: bool = True
    start_page: int = 13  # Start from page 13 as requested


class TokenDiscoveryService:
    """
    Enhanced Token Discovery Service using CoinGecko API.
    
    Discovers tokens starting from page 13 with real-time filtering
    and comprehensive market cap analysis.
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_delay = 1.2  # Rate limiting: ~50 requests per minute
        self.last_request_time = 0
        logger.info("Enhanced Token Discovery Service initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'FlowAI-TradingPlatform/1.0',
                    'Accept': 'application/json'
                }
            )
        return self.session
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _make_request(self, url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make rate-limited HTTP request to CoinGecko API."""
        await self._rate_limit()
        
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limit hit, waiting longer...")
                    await asyncio.sleep(60)  # Wait 1 minute on rate limit
                    return await self._make_request(url, params)
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    async def discoverTokens(self, criteria: FilterCriteria = None) -> List[DiscoveredToken]:
        """
        Discover tokens starting from page 13 with comprehensive filtering.
        
        Args:
            criteria: Filtering criteria (uses defaults if None)
            
        Returns:
            List of discovered tokens meeting all criteria
        """
        if criteria is None:
            criteria = FilterCriteria()
        
        logger.info(f"Starting token discovery from page {criteria.start_page}")
        logger.info(f"Filters: MarketCap ${criteria.min_market_cap:,.0f}-${criteria.max_market_cap:,.0f}, Volume >${criteria.min_daily_volume:,.0f}")
        
        discovered_tokens = []
        current_page = criteria.start_page
        max_pages = criteria.start_page + 20  # Limit to 20 pages to avoid excessive API calls
        
        try:
            while current_page <= max_pages and len(discovered_tokens) < 100:
                logger.info(f"Fetching page {current_page}...")
                
                # Fetch page data from CoinGecko
                page_tokens = await self._fetch_page_tokens(current_page)
                if not page_tokens:
                    logger.warning(f"No tokens returned from page {current_page}")
                    break
                
                # Apply filters to page tokens
                filtered_tokens = await self._apply_filters(page_tokens, criteria)
                
                if filtered_tokens:
                    logger.info(f"Page {current_page}: {len(filtered_tokens)} tokens passed filters")
                    discovered_tokens.extend(filtered_tokens)
                else:
                    logger.info(f"Page {current_page}: No tokens passed filters")
                
                current_page += 1
                
                # Break if we're past the market cap range (optimization)
                if page_tokens and all(token.get('market_cap', 0) > criteria.max_market_cap for token in page_tokens):
                    logger.info("All tokens above max market cap, stopping discovery")
                    break
        
        except Exception as e:
            logger.error(f"Error during token discovery: {e}")
        
        # Sort by market cap descending
        discovered_tokens.sort(key=lambda t: t.market_cap, reverse=True)
        
        logger.info(f"Discovery complete: Found {len(discovered_tokens)} tokens meeting criteria")
        return discovered_tokens
    
    async def _fetch_page_tokens(self, page: int) -> List[Dict[str, Any]]:
        """Fetch tokens from a specific CoinGecko page."""
        url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 250,  # Maximum per page
            'page': page,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d,30d,1y'
        }
        
        data = await self._make_request(url, params)
        return data if data else []
    
    async def _apply_filters(self, tokens: List[Dict[str, Any]], criteria: FilterCriteria) -> List[DiscoveredToken]:
        """Apply comprehensive filtering criteria to tokens."""
        filtered_tokens = []
        
        for token_data in tokens:
            try:
                # Extract basic data
                market_cap = token_data.get('market_cap', 0)
                total_volume = token_data.get('total_volume', 0)
                symbol = token_data.get('symbol', '').upper()
                
                # Apply market cap filter
                if not (criteria.min_market_cap <= market_cap <= criteria.max_market_cap):
                    continue
                
                # Apply volume filter
                if total_volume < criteria.min_daily_volume:
                    continue
                
                # Exclude stablecoins
                if criteria.exclude_stablecoins and self._is_stablecoin(symbol, token_data.get('name', '')):
                    continue
                
                # Create DiscoveredToken object
                discovered_token = self._create_discovered_token(token_data)
                
                # Calculate additional metrics
                await self._enrich_token_data(discovered_token)
                
                # Check trading pairs requirement
                if discovered_token.trading_pairs_count < criteria.min_trading_pairs:
                    continue
                
                filtered_tokens.append(discovered_token)
                
            except Exception as e:
                logger.warning(f"Error processing token {token_data.get('symbol', 'UNKNOWN')}: {e}")
                continue
        
        return filtered_tokens
    
    def _create_discovered_token(self, token_data: Dict[str, Any]) -> DiscoveredToken:
        """Create DiscoveredToken from CoinGecko API response."""
        return DiscoveredToken(
            id=token_data.get('id', ''),
            symbol=token_data.get('symbol', '').upper(),
            name=token_data.get('name', ''),
            current_price=float(token_data.get('current_price', 0)),
            market_cap=float(token_data.get('market_cap', 0)),
            market_cap_rank=int(token_data.get('market_cap_rank', 999999)),
            fully_diluted_valuation=token_data.get('fully_diluted_valuation'),
            total_volume=float(token_data.get('total_volume', 0)),
            high_24h=float(token_data.get('high_24h', 0)),
            low_24h=float(token_data.get('low_24h', 0)),
            price_change_24h=float(token_data.get('price_change_24h', 0)),
            price_change_percentage_24h=float(token_data.get('price_change_percentage_24h', 0)),
            price_change_percentage_7d=float(token_data.get('price_change_percentage_7d_in_currency', 0)),
            price_change_percentage_30d=float(token_data.get('price_change_percentage_30d_in_currency', 0)),
            price_change_percentage_1y=token_data.get('price_change_percentage_1y_in_currency'),
            market_cap_change_24h=float(token_data.get('market_cap_change_24h', 0)),
            market_cap_change_percentage_24h=float(token_data.get('market_cap_change_percentage_24h', 0)),
            circulating_supply=float(token_data.get('circulating_supply', 0)),
            total_supply=token_data.get('total_supply'),
            max_supply=token_data.get('max_supply'),
            ath=float(token_data.get('ath', 0)),
            ath_change_percentage=float(token_data.get('ath_change_percentage', 0)),
            ath_date=token_data.get('ath_date', ''),
            atl=float(token_data.get('atl', 0)),
            atl_change_percentage=float(token_data.get('atl_change_percentage', 0)),
            atl_date=token_data.get('atl_date', ''),
            last_updated=token_data.get('last_updated', ''),
        )
    
    async def _enrich_token_data(self, token: DiscoveredToken):
        """Enrich token with additional calculated metrics and contract addresses."""
        # Calculate volume to market cap ratio
        if token.market_cap > 0:
            token.volume_to_market_cap_ratio = token.total_volume / token.market_cap

        # Get trading pairs count (simplified for now)
        token.trading_pairs_count = await self._get_trading_pairs_count(token.id)

        # Calculate liquidity score based on volume/market cap ratio
        token.liquidity_score = min(100, token.volume_to_market_cap_ratio * 1000)

        # Fetch contract addresses for LiFi integration
        await self._fetch_contract_addresses(token)
    
    async def _get_trading_pairs_count(self, coin_id: str) -> int:
        """Get trading pairs count for a token."""
        try:
            url = f"{self.base_url}/coins/{coin_id}/tickers"
            params = {'depth': 'true'}

            data = await self._make_request(url, params)
            if data and 'tickers' in data:
                # Count unique trading pairs
                pairs = set()
                for ticker in data['tickers']:
                    if ticker.get('trust_score') in ['green', 'yellow']:  # Only trusted exchanges
                        pairs.add(f"{ticker.get('base', '')}/{ticker.get('target', '')}")
                return len(pairs)

            return 0
        except Exception as e:
            logger.warning(f"Error getting trading pairs for {coin_id}: {e}")
            return 0

    async def _fetch_contract_addresses(self, token: DiscoveredToken):
        """Fetch contract addresses from CoinGecko for LiFi integration."""
        try:
            url = f"{self.base_url}/coins/{token.id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'false',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }

            data = await self._make_request(url, params)
            if not data or 'platforms' not in data:
                logger.warning(f"No platform data for {token.symbol}")
                return

            platforms = data['platforms']

            # Chain mapping for LiFi compatibility
            chain_mapping = {
                'ethereum': {'chain_id': 1, 'name': 'ethereum'},
                'binance-smart-chain': {'chain_id': 56, 'name': 'bsc'},
                'polygon-pos': {'chain_id': 137, 'name': 'polygon'},
                'base': {'chain_id': 8453, 'name': 'base'},
                'arbitrum-one': {'chain_id': 42161, 'name': 'arbitrum'},
                'optimistic-ethereum': {'chain_id': 10, 'name': 'optimism'},
                'avalanche': {'chain_id': 43114, 'name': 'avalanche'},
                'fantom': {'chain_id': 250, 'name': 'fantom'},
                'cronos': {'chain_id': 25, 'name': 'cronos'}
            }

            # Extract contract addresses
            for platform_name, address in platforms.items():
                if address and address.strip() and platform_name in chain_mapping:
                    chain_info = chain_mapping[platform_name]
                    token.contract_addresses[chain_info['name']] = address.strip()

            # Set primary chain and address with improved priority order
            # Priority: Ethereum > Solana > Base > Arbitrum > Polygon > BSC > Others
            priority_chains = ['ethereum', 'solana', 'base', 'arbitrum', 'polygon', 'bsc']

            primary_set = False
            for priority_chain in priority_chains:
                if priority_chain in token.contract_addresses:
                    for platform_name, chain_info in chain_mapping.items():
                        if chain_info['name'] == priority_chain:
                            token.primary_chain_id = chain_info['chain_id']
                            token.primary_address = token.contract_addresses[priority_chain]
                            primary_set = True
                            break
                    if primary_set:
                        break

            # If no priority chain found, use first available
            if not primary_set and token.contract_addresses:
                first_chain = next(iter(token.contract_addresses.keys()))
                for platform_name, chain_info in chain_mapping.items():
                    if chain_info['name'] == first_chain:
                        token.primary_chain_id = chain_info['chain_id']
                        token.primary_address = token.contract_addresses[first_chain]
                        break

            if token.contract_addresses:
                logger.debug(f"Found {len(token.contract_addresses)} contract addresses for {token.symbol}")
            else:
                logger.warning(f"No contract addresses found for {token.symbol}")

        except Exception as e:
            logger.error(f"Error fetching contract addresses for {token.symbol}: {e}")

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def _is_stablecoin(self, symbol: str, name: str) -> bool:
        """Check if a token is a stablecoin."""
        stablecoin_keywords = [
            'USD', 'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAXG', 'USDN',
            'FRAX', 'LUSD', 'USDP', 'USTC', 'VAI', 'SUSD', 'DUSD'
        ]
        
        stablecoin_name_keywords = [
            'dollar', 'usd', 'stable', 'peg', 'backed'
        ]
        
        symbol_upper = symbol.upper()
        name_lower = name.lower()
        
        # Check symbol
        if any(keyword in symbol_upper for keyword in stablecoin_keywords):
            return True
        
        # Check name
        if any(keyword in name_lower for keyword in stablecoin_name_keywords):
            return True
        
        return False
    
    def filterByMarketCap(self, tokens: List[DiscoveredToken], min_cap: float, max_cap: float) -> List[DiscoveredToken]:
        """Filter tokens by market cap range."""
        return [token for token in tokens if min_cap <= token.market_cap <= max_cap]
    
    def filterByRisk(self, tokens: List[DiscoveredToken], max_risk_score: float) -> List[DiscoveredToken]:
        """Filter tokens by risk score (to be implemented with RugCheckService)."""
        # Placeholder - will be implemented with RugCheckService
        return tokens
    
    def filterByQualityScore(self, tokens: List[DiscoveredToken], min_quality_score: float) -> List[DiscoveredToken]:
        """Filter tokens by quality score (to be implemented with AnalyticsAssessmentService).""" 
        # Placeholder - will be implemented with AnalyticsAssessmentService
        return tokens
    
    def autoSort(self, tokens: List[DiscoveredToken]) -> List[DiscoveredToken]:
        """Auto-sort tokens by combined quality, risk, and market metrics."""
        def sort_key(token):
            # Multi-criteria sorting
            # 1. Liquidity score (higher is better)
            # 2. Volume to market cap ratio (higher is better) 
            # 3. Market cap (moderate preference for mid-cap)
            # 4. Price momentum (24h change, but not too extreme)
            
            liquidity_weight = token.liquidity_score
            volume_ratio_weight = min(50, token.volume_to_market_cap_ratio * 1000)  # Cap at 50
            
            # Prefer mid-cap tokens (sweet spot scoring)
            market_cap_weight = 50
            if 50_000_000 <= token.market_cap <= 200_000_000:  # $50M-$200M sweet spot
                market_cap_weight = 100
            elif 10_000_000 <= token.market_cap <= 50_000_000:   # $10M-$50M good
                market_cap_weight = 75
            elif token.market_cap >= 200_000_000:  # Above $200M
                market_cap_weight = 25
            
            # Momentum score (prefer positive but not excessive)
            momentum_score = 50  # Neutral
            if 0 < token.price_change_percentage_24h <= 20:  # Positive but reasonable
                momentum_score = 75
            elif token.price_change_percentage_24h > 20:  # Too high, might be pump
                momentum_score = 25
            elif token.price_change_percentage_24h < -10:  # Too negative
                momentum_score = 25
            
            # Combined score
            total_score = (
                liquidity_weight * 0.3 +
                volume_ratio_weight * 0.3 +
                market_cap_weight * 0.2 +
                momentum_score * 0.2
            )
            
            return -total_score  # Negative for descending sort
        
        sorted_tokens = sorted(tokens, key=sort_key)
        logger.info(f"Auto-sorted {len(sorted_tokens)} tokens by quality metrics")
        return sorted_tokens
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Service instance
_token_discovery_service = None

def get_enhanced_token_discovery_service() -> TokenDiscoveryService:
    """Get singleton instance of enhanced token discovery service."""
    global _token_discovery_service
    if _token_discovery_service is None:
        _token_discovery_service = TokenDiscoveryService()
    return _token_discovery_service