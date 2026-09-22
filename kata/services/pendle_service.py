"""
Pendle Finance Integration Service for Flow AI Trading Platform.

Integrates with Pendle's hosted SDK and API to provide fixed yield opportunities
for the Sakura agent's conservative trading strategy.
"""

import aiohttp
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
import json

logger = logging.getLogger(__name__)


@dataclass
class PendleMarket:
    """Represents a Pendle market with PT/YT pair."""
    market_address: str
    pt_address: str
    yt_address: str
    sy_address: str
    underlying_asset: str
    underlying_symbol: str
    maturity: datetime
    implied_apy: float
    pt_price: float
    yt_price: float
    liquidity_usd: float
    chain_id: int

    # Additional metadata
    protocol_name: str = ""
    is_active: bool = True
    min_trade_size: float = 100.0


@dataclass
class PendleYieldOpportunity:
    """Represents a yield opportunity suitable for Sakura agent."""
    market: PendleMarket
    strategy_type: str  # "fixed_yield", "yield_trading"
    expected_apy: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    time_to_maturity: int  # days
    entry_amount_usd: float
    sakura_score: float  # Compatibility score (0-1)

    # Pricing info
    current_pt_price: float
    discount_to_maturity: float
    break_even_days: int

    # Risk metrics
    liquidity_score: float
    volatility_score: float
    protocol_risk_score: float


@dataclass
class PendleSwapCalldata:
    """Calldata for Pendle swap operations."""
    to: str
    data: str
    value: str
    gas_estimate: int
    route_summary: Dict[str, Any]


class PendleService:
    """Service for interacting with Pendle Finance via their API and SDK."""

    def __init__(self):
        """Initialize Pendle service with API configuration."""
        self.base_url = "https://api-v2.pendle.finance/core"
        self.chain_id = 8453  # Base mainnet
        self.session = None

        # Sakura-specific configuration
        self.min_liquidity_usd = 1_000_000  # $1M minimum liquidity
        self.max_maturity_days = 365  # 1 year maximum
        self.min_maturity_days = 30   # 1 month minimum
        self.preferred_assets = ['USDC', 'DAI', 'WETH', 'ETH', 'USDT']

        # Rate limiting
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0

        logger.info("PendleService initialized for Base chain")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()

    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            # Create SSL context that's more permissive for development
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def _close_session(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def _rate_limited_request(self, method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make rate-limited request to Pendle API."""
        try:
            # Simple rate limiting
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)

            await self._ensure_session()

            async with self.session.request(method, url, **kwargs) as response:
                self.last_request_time = asyncio.get_event_loop().time()

                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    logger.warning("Rate limited by Pendle API, waiting 1 second")
                    await asyncio.sleep(1)
                    return await self._rate_limited_request(method, url, **kwargs)
                else:
                    logger.error(f"Pendle API error: {response.status} - {await response.text()}")
                    return None

        except Exception as e:
            logger.error(f"Error making request to {url}: {e}")
            return None

    async def get_all_markets(self, limit: int = 100) -> List[PendleMarket]:
        """Get all available Pendle markets on Base chain."""
        try:
            url = f"{self.base_url}/v1/{self.chain_id}/markets"
            params = {
                "limit": limit,
                "order_by": "totalLiquidityUsd:-1",  # Sort by liquidity descending
                "is_active": "true"
            }

            data = await self._rate_limited_request("GET", url, params=params)
            if not data or 'results' not in data:
                logger.warning("No market data received from Pendle API")
                return []

            markets = []
            for market_data in data['results']:
                try:
                    market = self._parse_market(market_data)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.error(f"Error parsing market data: {e}")
                    continue

            logger.info(f"Retrieved {len(markets)} Pendle markets from API")
            return markets

        except Exception as e:
            logger.error(f"Error fetching Pendle markets: {e}")
            return []

    async def get_market_details(self, market_address: str) -> Optional[Dict[str, Any]]:
        """Get detailed market data for a specific market."""
        try:
            url = f"{self.base_url}/v1/{self.chain_id}/markets/{market_address}"
            return await self._rate_limited_request("GET", url)
        except Exception as e:
            logger.error(f"Error fetching market details for {market_address}: {e}")
            return None

    async def get_asset_prices(self, asset_addresses: List[str]) -> Dict[str, float]:
        """Get current prices for multiple assets."""
        try:
            url = f"{self.base_url}/v1/{self.chain_id}/assets"
            params = {"addresses": ",".join(asset_addresses)}

            data = await self._rate_limited_request("GET", url, params=params)
            if not data or 'results' not in data:
                return {}

            prices = {}
            for asset in data['results']:
                address = asset.get('address', '').lower()
                price = asset.get('price', 0)
                if address and price:
                    prices[address] = float(price)

            return prices

        except Exception as e:
            logger.error(f"Error fetching asset prices: {e}")
            return {}

    async def calculate_pt_yield(self, market: PendleMarket) -> Tuple[float, int]:
        """
        Calculate fixed yield for PT purchase.

        Returns:
            Tuple of (annual_yield_percentage, break_even_days)
        """
        try:
            # Handle timezone-aware datetime comparison
            now = datetime.now()
            if market.maturity.tzinfo is not None and now.tzinfo is None:
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            elif market.maturity.tzinfo is None and now.tzinfo is not None:
                from datetime import timezone
                market_maturity = market.maturity.replace(tzinfo=timezone.utc)
            else:
                market_maturity = market.maturity

            days_to_maturity = (market_maturity - now).days
            if days_to_maturity <= 0:
                return 0.0, 0

            # PT yield calculation: (1 - pt_price) gives discount
            # Annual yield = discount / (days_to_maturity / 365)
            discount = 1.0 - market.pt_price
            if discount <= 0:  # PT trading at premium
                return 0.0, 0

            annual_yield = (discount / (days_to_maturity / 365)) * 100

            # Break-even calculation (simplified)
            # Assuming linear yield accrual
            break_even_days = int(days_to_maturity * 0.1)  # 10% safety buffer

            return annual_yield, break_even_days

        except Exception as e:
            logger.error(f"Error calculating PT yield: {e}")
            return 0.0, 0

    async def get_sakura_opportunities(self, max_opportunities: int = 10) -> List[PendleYieldOpportunity]:
        """
        Get Pendle opportunities suitable for Sakura's conservative strategy.

        Filters:
        - Preferred assets (USDC, DAI, WETH, etc.)
        - Minimum liquidity threshold
        - Appropriate maturity range
        - Risk assessment
        """
        try:
            logger.info("Analyzing Pendle opportunities for Sakura agent")

            # Get all markets
            all_markets = await self.get_all_markets()
            if not all_markets:
                logger.warning("No markets available for analysis")
                return []

            opportunities = []

            for market in all_markets:
                # Filter by Sakura's criteria
                if not self._meets_sakura_criteria(market):
                    continue

                # Calculate yield and scores
                annual_yield, break_even_days = await self.calculate_pt_yield(market)
                if annual_yield <= 0:
                    continue

                sakura_score = await self._calculate_sakura_score(market, annual_yield)
                if sakura_score < 0.5:  # Minimum score threshold
                    continue

                # Create opportunity
                opportunity = PendleYieldOpportunity(
                    market=market,
                    strategy_type="fixed_yield",
                    expected_apy=annual_yield,
                    risk_level=self._assess_risk_level(market),
                    time_to_maturity=(market.maturity - datetime.now()).days,
                    entry_amount_usd=1000.0,  # Default entry size
                    sakura_score=sakura_score,
                    current_pt_price=market.pt_price,
                    discount_to_maturity=(1.0 - market.pt_price) * 100,
                    break_even_days=break_even_days,
                    liquidity_score=self._calculate_liquidity_score(market),
                    volatility_score=0.8,  # Placeholder - would need historical data
                    protocol_risk_score=0.9   # Pendle is well-established
                )

                opportunities.append(opportunity)

            # Sort by Sakura score (best opportunities first)
            opportunities.sort(key=lambda x: x.sakura_score, reverse=True)

            logger.info(f"Found {len(opportunities)} suitable Pendle opportunities for Sakura")
            return opportunities[:max_opportunities]

        except Exception as e:
            logger.error(f"Error analyzing Sakura opportunities: {e}")
            return []

    async def generate_swap_calldata(
        self,
        token_in_address: str,
        amount_in: str,
        pt_address: str,
        receiver_address: str,
        slippage: float = 0.01
    ) -> Optional[PendleSwapCalldata]:
        """Generate calldata for swapping tokens to PT using Pendle SDK."""
        try:
            url = f"{self.base_url}/v2/sdk/{self.chain_id}/convert"

            payload = {
                "tokensIn": token_in_address,
                "amountsIn": amount_in,
                "tokensOut": pt_address,
                "enableAggregator": True,
                "receiver": receiver_address,
                "slippage": slippage
            }

            data = await self._rate_limited_request("POST", url, json=payload)
            if not data:
                return None

            return PendleSwapCalldata(
                to=data.get('to', ''),
                data=data.get('data', ''),
                value=data.get('value', '0'),
                gas_estimate=data.get('gasEstimate', 0),
                route_summary=data.get('routeSummary', {})
            )

        except Exception as e:
            logger.error(f"Error generating swap calldata: {e}")
            return None

    def _parse_market(self, market_data: Dict[str, Any]) -> Optional[PendleMarket]:
        """Parse market data from Pendle API response."""
        try:
            # Extract expiry and convert to datetime
            expiry_timestamp = market_data.get('expiry')
            if not expiry_timestamp:
                return None

            # Handle different timestamp formats
            if isinstance(expiry_timestamp, str):
                maturity = datetime.fromisoformat(expiry_timestamp.replace('Z', '+00:00'))
            else:
                maturity = datetime.fromtimestamp(expiry_timestamp)

            # Extract underlying asset info
            underlying_asset = market_data.get('underlyingAsset', {})
            if isinstance(underlying_asset, str):
                underlying_symbol = underlying_asset
                underlying_name = underlying_asset
            else:
                underlying_symbol = underlying_asset.get('symbol', '')
                underlying_name = underlying_asset.get('name', underlying_symbol)

            return PendleMarket(
                market_address=market_data.get('address', ''),
                pt_address=market_data.get('pt', ''),
                yt_address=market_data.get('yt', ''),
                sy_address=market_data.get('sy', ''),
                underlying_asset=underlying_name,
                underlying_symbol=underlying_symbol,
                maturity=maturity,
                implied_apy=float(market_data.get('impliedApy', 0)) * 100,
                pt_price=float(market_data.get('ptPrice', 0)),
                yt_price=float(market_data.get('ytPrice', 0)),
                liquidity_usd=float(market_data.get('totalLiquidityUsd', 0)),
                chain_id=self.chain_id,
                protocol_name=market_data.get('protocol', ''),
                is_active=market_data.get('isActive', True),
                min_trade_size=float(market_data.get('minTradeSize', 100))
            )

        except Exception as e:
            logger.error(f"Error parsing market data: {e}")
            return None

    def _meets_sakura_criteria(self, market: PendleMarket) -> bool:
        """Check if market meets Sakura's conservative criteria."""
        try:
            # Check maturity range - ensure timezone-aware comparison
            now = datetime.now()
            if market.maturity.tzinfo is not None and now.tzinfo is None:
                # Make now timezone-aware for comparison
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            elif market.maturity.tzinfo is None and now.tzinfo is not None:
                # Make maturity timezone-aware for comparison
                from datetime import timezone
                market_maturity = market.maturity.replace(tzinfo=timezone.utc)
            else:
                market_maturity = market.maturity

            days_to_maturity = (market_maturity - now).days
            if days_to_maturity < self.min_maturity_days or days_to_maturity > self.max_maturity_days:
                return False

            # Check liquidity
            if market.liquidity_usd < self.min_liquidity_usd:
                return False

            # Check preferred assets
            if not any(asset.upper() in market.underlying_symbol.upper() for asset in self.preferred_assets):
                return False

            # Check PT price (must be at discount)
            if market.pt_price >= 1.0:
                return False

            # Check if market is active
            if not market.is_active:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking Sakura criteria: {e}")
            return False

    async def _calculate_sakura_score(self, market: PendleMarket, annual_yield: float) -> float:
        """Calculate compatibility score with Sakura's preferences."""
        try:
            score = 0.0

            # Liquidity score (30% weight)
            liquidity_score = self._calculate_liquidity_score(market)
            score += liquidity_score * 0.3

            # Time to maturity score (25% weight)
            days_to_maturity = (market.maturity - datetime.now()).days
            if 90 <= days_to_maturity <= 180:  # 3-6 months ideal
                maturity_score = 1.0
            elif 60 <= days_to_maturity < 90 or 180 < days_to_maturity <= 270:
                maturity_score = 0.8
            elif 30 <= days_to_maturity < 60 or 270 < days_to_maturity <= 365:
                maturity_score = 0.6
            else:
                maturity_score = 0.3
            score += maturity_score * 0.25

            # Yield score (25% weight) - Sakura prefers 8-20% range
            if 8 <= annual_yield <= 20:
                yield_score = 1.0
            elif 5 <= annual_yield < 8 or 20 < annual_yield <= 30:
                yield_score = 0.8
            elif 3 <= annual_yield < 5 or 30 < annual_yield <= 50:
                yield_score = 0.6
            else:
                yield_score = 0.3
            score += yield_score * 0.25

            # Asset preference score (20% weight)
            asset_preference = {
                'USDC': 1.0, 'DAI': 1.0, 'USDT': 0.9,
                'WETH': 0.8, 'ETH': 0.8,
            }

            asset_score = 0.5  # Default
            for asset, preference in asset_preference.items():
                if asset.upper() in market.underlying_symbol.upper():
                    asset_score = preference
                    break
            score += asset_score * 0.2

            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.error(f"Error calculating Sakura score: {e}")
            return 0.0

    def _calculate_liquidity_score(self, market: PendleMarket) -> float:
        """Calculate liquidity score (0-1) based on USD liquidity."""
        try:
            if market.liquidity_usd >= 50_000_000:  # $50M+
                return 1.0
            elif market.liquidity_usd >= 20_000_000:  # $20M+
                return 0.9
            elif market.liquidity_usd >= 10_000_000:  # $10M+
                return 0.8
            elif market.liquidity_usd >= 5_000_000:   # $5M+
                return 0.7
            elif market.liquidity_usd >= 2_000_000:   # $2M+
                return 0.6
            elif market.liquidity_usd >= 1_000_000:   # $1M+
                return 0.5
            else:
                return 0.3
        except:
            return 0.3

    def _assess_risk_level(self, market: PendleMarket) -> str:
        """Assess overall risk level for the market."""
        try:
            days_to_maturity = (market.maturity - datetime.now()).days
            liquidity_score = self._calculate_liquidity_score(market)

            # Conservative assessment
            if (liquidity_score >= 0.8 and
                days_to_maturity >= 90 and
                market.underlying_symbol.upper() in ['USDC', 'DAI', 'USDT']):
                return "LOW"
            elif (liquidity_score >= 0.6 and
                  days_to_maturity >= 60):
                return "MEDIUM"
            else:
                return "HIGH"

        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return "HIGH"


# Global service instance
_pendle_service: Optional[PendleService] = None


async def get_pendle_service() -> PendleService:
    """Get or create Pendle service instance."""
    global _pendle_service

    if _pendle_service is None:
        _pendle_service = PendleService()

    return _pendle_service


# Cleanup function for graceful shutdown
async def cleanup_pendle_service():
    """Cleanup Pendle service resources."""
    global _pendle_service

    if _pendle_service:
        await _pendle_service._close_session()
        _pendle_service = None