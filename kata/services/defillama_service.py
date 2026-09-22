"""
DeFiLlama API Service for Yuki Agent

Provides comprehensive DeFi and protocol data including:
- Total Value Locked (TVL) across protocols and chains
- Protocol revenue and fee analysis
- Yield farming opportunities and APY data
- Cross-chain liquidity and bridge data
- DeFi market trends and protocol comparisons
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import time

logger = logging.getLogger(__name__)


@dataclass
class ProtocolData:
    """Protocol data structure."""
    name: str
    slug: str
    symbol: str
    tvl: float
    tvl_change_24h: float
    tvl_change_7d: float
    category: str
    chain: str
    chains: List[str]
    logo: str
    url: str
    timestamp: datetime


@dataclass
class ChainData:
    """Blockchain data structure."""
    name: str
    tvl: float
    tvl_change_24h: float
    tvl_change_7d: float
    protocols: int
    symbol: str
    chain_id: Optional[int]
    timestamp: datetime


@dataclass
class YieldData:
    """Yield farming data structure."""
    pool: str
    project: str
    symbol: str
    chain: str
    apy: float
    apy_base: float
    apy_reward: float
    tvl: float
    rewards: List[str]
    exposure: str  # 'single', 'multi', 'stable'
    risk_level: str  # 'low', 'medium', 'high'
    timestamp: datetime


@dataclass
class DeFiMetrics:
    """DeFi market metrics structure."""
    total_tvl: float
    total_tvl_change_24h: float
    total_tvl_change_7d: float
    dominant_chain: str
    dominant_chain_percentage: float
    top_protocols: List[str]
    trending_protocols: List[str]
    defi_dominance: float  # % of total crypto market
    timestamp: datetime


@dataclass
class BridgeData:
    """Cross-chain bridge data structure."""
    name: str
    volume_24h: float
    volume_7d: float
    chains: List[str]
    largest_transaction: float
    total_addresses: int
    timestamp: datetime


class DeFiLlamaService:
    """
    DeFiLlama API service for comprehensive DeFi market data.
    
    Provides free access to extensive DeFi protocol and TVL data
    without requiring API keys.
    """
    
    def __init__(self):
        """Initialize DeFiLlama service."""
        self.base_url = "https://api.llama.fi"
        self.yields_url = "https://yields.llama.fi"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache settings
        self.cache_duration = {
            'protocols': 300,  # 5 minutes for protocol data
            'chains': 300,  # 5 minutes for chain data
            'yields': 600,  # 10 minutes for yield data
            'tvl': 180,  # 3 minutes for TVL data
            'bridges': 300  # 5 minutes for bridge data
        }
        self.cache = {}
        
        logger.info("DeFiLlamaService initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("DeFiLlama HTTP session started")
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("DeFiLlama HTTP session closed")
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_duration.get(cache_type, 300)
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def _make_request(self, base_url: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to DeFiLlama API."""
        if not self.session:
            await self.start_session()
        
        url = f"{base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logger.error(f"DeFiLlama API request failed: {e}")
            raise
    
    async def get_protocols(self, limit: int = 100) -> List[ProtocolData]:
        """
        Get all DeFi protocols with TVL data.
        
        Args:
            limit: Maximum number of protocols to return
            
        Returns:
            List of ProtocolData objects
        """
        try:
            cache_key = f"protocols_{limit}"
            
            if self._is_cache_valid(cache_key, 'protocols'):
                return self.cache[cache_key]['data']
            
            data = await self._make_request(self.base_url, "protocols")
            
            protocols = []
            for protocol_info in data[:limit]:
                protocols.append(ProtocolData(
                    name=protocol_info.get('name', ''),
                    slug=protocol_info.get('slug', ''),
                    symbol=protocol_info.get('symbol', ''),
                    tvl=float(protocol_info.get('tvl', 0)),
                    tvl_change_24h=float(protocol_info.get('change_1d', 0)),
                    tvl_change_7d=float(protocol_info.get('change_7d', 0)),
                    category=protocol_info.get('category', ''),
                    chain=protocol_info.get('chain', ''),
                    chains=protocol_info.get('chains', []),
                    logo=protocol_info.get('logo', ''),
                    url=protocol_info.get('url', ''),
                    timestamp=datetime.now()
                ))
            
            self._cache_data(cache_key, protocols)
            return protocols
            
        except Exception as e:
            logger.error(f"Error fetching protocols: {e}")
            return []
    
    async def get_chains(self) -> List[ChainData]:
        """
        Get all blockchain chains with TVL data.
        
        Returns:
            List of ChainData objects
        """
        try:
            cache_key = "chains_all"
            
            if self._is_cache_valid(cache_key, 'chains'):
                return self.cache[cache_key]['data']
            
            data = await self._make_request(self.base_url, "chains")
            
            chains = []
            for chain_info in data:
                chains.append(ChainData(
                    name=chain_info.get('name', ''),
                    tvl=float(chain_info.get('tvl', 0)),
                    tvl_change_24h=float(chain_info.get('change_1d', 0)),
                    tvl_change_7d=float(chain_info.get('change_7d', 0)),
                    protocols=int(chain_info.get('protocols', 0)),
                    symbol=chain_info.get('tokenSymbol', ''),
                    chain_id=chain_info.get('chainId'),
                    timestamp=datetime.now()
                ))
            
            # Sort by TVL descending
            chains.sort(key=lambda x: x.tvl, reverse=True)
            
            self._cache_data(cache_key, chains)
            return chains
            
        except Exception as e:
            logger.error(f"Error fetching chains: {e}")
            return []
    
    async def get_yields(self, min_tvl: float = 1000000, min_apy: float = 5.0, limit: int = 50) -> List[YieldData]:
        """
        Get yield farming opportunities.
        
        Args:
            min_tvl: Minimum TVL required for pools
            min_apy: Minimum APY required
            limit: Maximum number of pools to return
            
        Returns:
            List of YieldData objects
        """
        try:
            cache_key = f"yields_{min_tvl}_{min_apy}_{limit}"
            
            if self._is_cache_valid(cache_key, 'yields'):
                return self.cache[cache_key]['data']
            
            data = await self._make_request(self.yields_url, "pools")
            
            yields = []
            for pool_info in data['data']:
                apy = float(pool_info.get('apy', 0) or 0)
                tvl = float(pool_info.get('tvlUsd', 0) or 0)
                
                # Filter by minimum requirements
                if apy >= min_apy and tvl >= min_tvl:
                    # Determine risk level based on various factors
                    risk_level = self._assess_yield_risk(pool_info)
                    
                    yields.append(YieldData(
                        pool=pool_info.get('pool', ''),
                        project=pool_info.get('project', ''),
                        symbol=pool_info.get('symbol', ''),
                        chain=pool_info.get('chain', ''),
                        apy=apy,
                        apy_base=float(pool_info.get('apyBase', 0) or 0),
                        apy_reward=float(pool_info.get('apyReward', 0) or 0),
                        tvl=tvl,
                        rewards=pool_info.get('rewardTokens', []),
                        exposure=pool_info.get('exposure', 'multi'),
                        risk_level=risk_level,
                        timestamp=datetime.now()
                    ))
            
            # Sort by APY descending and limit results
            yields.sort(key=lambda x: x.apy, reverse=True)
            yields = yields[:limit]
            
            self._cache_data(cache_key, yields)
            return yields
            
        except Exception as e:
            logger.error(f"Error fetching yields: {e}")
            return []
    
    def _assess_yield_risk(self, pool_info: Dict[str, Any]) -> str:
        """Assess risk level of a yield farming pool."""
        try:
            # Factors that increase risk
            risk_score = 0
            
            # Low TVL increases risk
            tvl = float(pool_info.get('tvlUsd', 0) or 0)
            if tvl < 5000000:  # < $5M
                risk_score += 2
            elif tvl < 50000000:  # < $50M
                risk_score += 1
            
            # High APY can indicate higher risk
            apy = float(pool_info.get('apy', 0) or 0)
            if apy > 100:
                risk_score += 3
            elif apy > 50:
                risk_score += 2
            elif apy > 20:
                risk_score += 1
            
            # Multiple tokens (impermanent loss risk)
            exposure = pool_info.get('exposure', 'multi')
            if exposure == 'multi':
                risk_score += 1
            elif exposure == 'single':
                risk_score -= 1
            
            # New protocols have higher risk
            # (This would require additional data about protocol age)
            
            # Determine risk level
            if risk_score >= 5:
                return 'high'
            elif risk_score >= 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.warning(f"Error assessing yield risk: {e}")
            return 'medium'
    
    async def get_defi_metrics(self) -> DeFiMetrics:
        """
        Get overall DeFi market metrics.
        
        Returns:
            DeFiMetrics object with market overview
        """
        try:
            cache_key = "defi_metrics"
            
            if self._is_cache_valid(cache_key, 'tvl'):
                return self.cache[cache_key]['data']
            
            # Get total TVL
            tvl_data = await self._make_request(self.base_url, "tvl")
            
            # Get chains data for dominance
            chains = await self.get_chains()
            
            # Get protocols data for trending
            protocols = await self.get_protocols(50)
            
            # Calculate metrics
            total_tvl = float(tvl_data[0]['totalLiquidityUSD']) if tvl_data else 0.0
            
            # Find dominant chain
            dominant_chain = chains[0].name if chains else "Unknown"
            dominant_percentage = (chains[0].tvl / total_tvl) * 100 if chains and total_tvl > 0 else 0.0
            
            # Get top protocols by TVL
            top_protocols = [p.name for p in protocols[:10]]
            
            # Get trending protocols (positive 24h change)
            trending = [p.name for p in protocols if p.tvl_change_24h > 10][:5]
            
            # Calculate 24h and 7d changes (simplified)
            tvl_change_24h = sum(p.tvl_change_24h for p in protocols[:20]) / 20 if protocols else 0.0
            tvl_change_7d = sum(p.tvl_change_7d for p in protocols[:20]) / 20 if protocols else 0.0
            
            metrics = DeFiMetrics(
                total_tvl=total_tvl,
                total_tvl_change_24h=tvl_change_24h,
                total_tvl_change_7d=tvl_change_7d,
                dominant_chain=dominant_chain,
                dominant_chain_percentage=dominant_percentage,
                top_protocols=top_protocols,
                trending_protocols=trending,
                defi_dominance=5.0,  # Approximate DeFi dominance of crypto market
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching DeFi metrics: {e}")
            return DeFiMetrics(
                total_tvl=0.0,
                total_tvl_change_24h=0.0,
                total_tvl_change_7d=0.0,
                dominant_chain="Unknown",
                dominant_chain_percentage=0.0,
                top_protocols=[],
                trending_protocols=[],
                defi_dominance=0.0,
                timestamp=datetime.now()
            )
    
    async def get_protocol_tvl_history(self, protocol_slug: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """
        Get historical TVL data for a specific protocol.
        
        Args:
            protocol_slug: Protocol slug (e.g., 'uniswap')
            days: Number of days of historical data
            
        Returns:
            List of (timestamp, tvl) tuples
        """
        try:
            cache_key = f"protocol_history_{protocol_slug}_{days}"
            
            if self._is_cache_valid(cache_key, 'tvl'):
                return self.cache[cache_key]['data']
            
            data = await self._make_request(self.base_url, f"protocol/{protocol_slug}")
            
            # Extract TVL history
            tvl_history = []
            if 'tvl' in data:
                for entry in data['tvl']:
                    timestamp = datetime.fromtimestamp(entry['date'])
                    tvl = float(entry['totalLiquidityUSD'])
                    tvl_history.append((timestamp, tvl))
            
            # Filter to requested days
            cutoff_date = datetime.now() - timedelta(days=days)
            tvl_history = [(ts, tvl) for ts, tvl in tvl_history if ts >= cutoff_date]
            
            self._cache_data(cache_key, tvl_history)
            return tvl_history
            
        except Exception as e:
            logger.error(f"Error fetching TVL history for {protocol_slug}: {e}")
            return []
    
    async def get_bridges_data(self) -> List[BridgeData]:
        """
        Get cross-chain bridge volume data.
        
        Returns:
            List of BridgeData objects
        """
        try:
            cache_key = "bridges_data"
            
            if self._is_cache_valid(cache_key, 'bridges'):
                return self.cache[cache_key]['data']
            
            data = await self._make_request("https://bridges.llama.fi", "bridges")
            
            bridges = []
            for bridge_info in data['bridges']:
                bridges.append(BridgeData(
                    name=bridge_info.get('name', ''),
                    volume_24h=float(bridge_info.get('volume24h', 0)),
                    volume_7d=float(bridge_info.get('volume7d', 0)),
                    chains=bridge_info.get('chains', []),
                    largest_transaction=float(bridge_info.get('largestTx', 0)),
                    total_addresses=int(bridge_info.get('totalAddresses', 0)),
                    timestamp=datetime.now()
                ))
            
            # Sort by 24h volume
            bridges.sort(key=lambda x: x.volume_24h, reverse=True)
            
            self._cache_data(cache_key, bridges)
            return bridges
            
        except Exception as e:
            logger.error(f"Error fetching bridges data: {e}")
            return []
    
    async def get_trending_protocols(self, timeframe: str = '24h', limit: int = 10) -> List[ProtocolData]:
        """
        Get trending protocols based on TVL changes.
        
        Args:
            timeframe: '24h' or '7d'
            limit: Number of protocols to return
            
        Returns:
            List of trending ProtocolData objects
        """
        try:
            protocols = await self.get_protocols(100)
            
            # Sort by change percentage
            if timeframe == '7d':
                trending = sorted(protocols, key=lambda x: x.tvl_change_7d, reverse=True)
            else:
                trending = sorted(protocols, key=lambda x: x.tvl_change_24h, reverse=True)
            
            # Filter out protocols with very low TVL
            trending = [p for p in trending if p.tvl > 1000000]  # > $1M TVL
            
            return trending[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching trending protocols: {e}")
            return []


# Factory function
def create_defillama_service() -> DeFiLlamaService:
    """Create and return DeFiLlamaService instance."""
    return DeFiLlamaService()