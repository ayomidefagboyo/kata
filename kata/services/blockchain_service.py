"""
Blockchain Service for Multi-Chain Portfolio Data

Provides token balance and native balance fetching across multiple EVM chains
using web3.py and Alchemy API for comprehensive portfolio data.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp
import ssl
from web3 import Web3
import os
import time
import atexit

logger = logging.getLogger(__name__)

# Multi-chain configuration - Only include networks that your Alchemy API key supports
CHAINS = {
    1: {
        'name': 'Ethereum',
        'rpc_url': 'https://eth-mainnet.g.alchemy.com/v2/',
        'explorer': 'https://etherscan.io',
        'native_symbol': 'ETH'
    },
    8453: {
        'name': 'Base',
        'rpc_url': 'https://base-mainnet.g.alchemy.com/v2/',
        'explorer': 'https://basescan.org',
        'native_symbol': 'ETH'
    },
    42161: {
        'name': 'Arbitrum',
        'rpc_url': 'https://arb-mainnet.g.alchemy.com/v2/',
        'explorer': 'https://arbiscan.io',
        'native_symbol': 'ETH'
    }
    # Removed Polygon (137) and Optimism (10) as they're causing 403 errors
    # Add them back when you have proper API key permissions for these networks
}

# Token addresses across chains
CHAIN_TOKENS = {
    1: {
        'USDC': '0xA0b86a33E6441b8C4C8C8C8C8C8C8C8C8C8C8C8',  # Placeholder
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    },
    137: {
        'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        'WETH': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
        'DAI': '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063',
        'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
    },
    8453: {
        'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
        'WETH': '0x4200000000000000000000000000000000000006',
        'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0b',
        'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',
        'cbETH': '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
    },
    10: {
        'USDC': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
        'WETH': '0x4200000000000000000000000000000000000006',
        'DAI': '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',
        'USDT': '0x94b008aA00579c1307B0EF2c499aD98a8ce58e58',
    },
    42161: {
        'USDC': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',
        'WETH': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
        'DAI': '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',
        'USDT': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
    }
}

# Per-(kind, wallet, chain) balance cache. Module-level because BlockchainService is
# instantiated per request. Entries are fresh for BALANCE_CACHE_FRESH_SECONDS and
# retained up to BALANCE_CACHE_STALE_SECONDS as a fallback when Alchemy is rate limited.
_BALANCE_CACHE: Dict[tuple, tuple] = {}
BALANCE_CACHE_FRESH_SECONDS = 60
BALANCE_CACHE_STALE_SECONDS = 600


def _balance_cache_get(key: tuple, allow_stale: bool = False) -> Optional[List[Dict[str, Any]]]:
    entry = _BALANCE_CACHE.get(key)
    if not entry:
        return None
    value, cached_at = entry
    age = time.time() - cached_at
    if age < BALANCE_CACHE_FRESH_SECONDS or (allow_stale and age < BALANCE_CACHE_STALE_SECONDS):
        return value
    return None


def _balance_cache_put(key: tuple, value: List[Dict[str, Any]]):
    now = time.time()
    _BALANCE_CACHE[key] = (value, now)
    expired = [k for k, (_, cached_at) in _BALANCE_CACHE.items()
               if now - cached_at >= BALANCE_CACHE_STALE_SECONDS]
    for k in expired:
        _BALANCE_CACHE.pop(k, None)


# ERC20 ABI for token balance checking
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]


class BlockchainService:
    """
    Blockchain service for fetching token balances across multiple EVM chains.
    """
    
    def __init__(self):
        """Initialize blockchain service."""
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY', '')
        self.session: Optional[aiohttp.ClientSession] = None
        # Simple in-memory cache: key = (chain_id, contract_address.lower()) -> {symbol,name,decimals}
        self._metadata_cache: Dict[tuple, Dict[str, Any]] = {}
        logger.info("BlockchainService initialized")
        if not self.alchemy_api_key:
            logger.warning("ALCHEMY_API_KEY is not set. Chain calls will fail with 401/403.")

        # Ensure session is closed on process exit
        atexit.register(self._ensure_session_closed_sync)

    def _ensure_session_closed_sync(self):
        """Best-effort session close at process exit."""
        try:
            if self.session and not self.session.closed:
                try:
                    # Try to close using a running loop if available
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.close_session())
                    else:
                        loop.run_until_complete(self.close_session())
                except RuntimeError:
                    # Fallback if no loop
                    asyncio.run(self.close_session())
        except Exception:
            pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session with SSL context."""
        if not self.session:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            logger.info("Blockchain HTTP session started with SSL bypass")

    async def _post_with_retry(self, url: str, json: Dict[str, Any], max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """POST with minimal retry/backoff for 429 and log/skip on 403."""
        if not self.session:
            await self.start_session()

        backoff_seconds = 1.0
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            try:
                async with self.session.post(url, json=json) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 403:
                        logger.warning(f"Forbidden (403) calling {url}. Check API key/permissions. Skipping.")
                        return None
                    if response.status == 429:
                        logger.warning(f"Rate limited (429) calling {url}. Backing off {backoff_seconds:.1f}s (attempt {attempt}/{max_retries+1})")
                        await asyncio.sleep(backoff_seconds)
                        backoff_seconds *= 2
                        continue
                    logger.warning(f"HTTP {response.status} calling {url}. No retry.")
                    return None
            except Exception as e:
                logger.warning(f"POST error calling {url}: {e}. Attempt {attempt}/{max_retries+1}")
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2
        return None
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Blockchain HTTP session closed")
    
    async def get_token_balances(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        Get token balances for a wallet address across all chains.
        
        Args:
            wallet_address: The wallet address to fetch balances for
            
        Returns:
            List of token balance dictionaries
        """
        try:
            if not self.session:
                await self.start_session()
            
            all_token_balances = []
            
            # Fetch balances from all chains with rate limiting
            for chain_id, chain_config in CHAINS.items():
                try:
                    cache_key = ('tokens', wallet_address.lower(), chain_id)
                    cached = _balance_cache_get(cache_key)
                    if cached is not None:
                        all_token_balances.extend(cached)
                        continue

                    # Add delay between API calls to prevent rate limiting
                    await asyncio.sleep(0.5)  # 500ms delay between chain requests
                    logger.debug(f"Fetching token balances from {chain_config['name']}")

                    # Use Alchemy API for token balances
                    url = f"{chain_config['rpc_url']}{self.alchemy_api_key}"

                    # Get token balances using Alchemy API
                    params = {
                        'jsonrpc': '2.0',
                        'method': 'alchemy_getTokenBalances',
                        'params': [wallet_address],
                        'id': 1
                    }

                    data = await self._post_with_retry(url, params)
                    if not data:
                        stale = _balance_cache_get(cache_key, allow_stale=True)
                        if stale is not None:
                            logger.warning(f"Serving stale token balances for {chain_config['name']}")
                            all_token_balances.extend(stale)
                        continue

                    chain_token_balances = []
                    token_balances = data.get('result', {}).get('tokenBalances', [])

                    # De-duplicate contracts to avoid repeated metadata fetches
                    seen_contracts: set = set()

                    for balance in token_balances:
                        raw_hex = balance.get('tokenBalance')
                        if not raw_hex:
                            continue

                        # Parse hex balance to integer
                        try:
                            balance_int = int(raw_hex, 16)
                        except Exception:
                            logger.debug(f"Skipping non-hex balance value for {balance.get('contractAddress')}: {raw_hex}")
                            continue

                        if balance_int <= 0:
                            continue

                        contract_addr = balance['contractAddress']
                        cache_key = (chain_id, contract_addr.lower())

                        if cache_key in self._metadata_cache:
                            metadata = self._metadata_cache[cache_key]
                        else:
                            # Avoid fetching the same contract more than once in this pass
                            if cache_key in seen_contracts:
                                continue
                            seen_contracts.add(cache_key)

                            # Get token metadata (decimals/symbol)
                            metadata = await self._get_token_metadata(
                                contract_addr,
                                chain_id,
                                url
                            )
                            if metadata:
                                self._metadata_cache[cache_key] = metadata

                        if metadata:
                            decimals = int(metadata.get('decimals', 18) or 18)
                            balance_human = balance_int / (10 ** decimals)

                            token_balance = {
                                'contract_address': balance['contractAddress'],
                                'symbol': metadata.get('symbol', 'UNKNOWN'),
                                'name': metadata.get('name', 'Unknown Token'),
                                'decimals': decimals,
                                'balance': balance_human,
                                'balance_raw': raw_hex,
                                'balance_usd': 0,  # Will be calculated later
                                'price': 0,  # Will be fetched from CoinGecko
                                'change_24h': 0,  # Will be fetched from CoinGecko
                                'chain_id': chain_id,
                                'chain_name': chain_config['name'],
                            }
                            chain_token_balances.append(token_balance)

                    _balance_cache_put(cache_key, chain_token_balances)
                    all_token_balances.extend(chain_token_balances)

                except Exception as e:
                    logger.warning(f"Error fetching balances from {chain_config['name']}: {e}")
            
            logger.debug(f"Fetched {len(all_token_balances)} token balances")
            return all_token_balances
            
        except Exception as e:
            logger.error(f"Error fetching token balances: {e}")
            return []
    
    async def get_native_balances(self, wallet_address: str) -> List[Dict[str, Any]]:
        """
        Get native token balances for a wallet address across all chains.
        
        Args:
            wallet_address: The wallet address to fetch balances for
            
        Returns:
            List of native balance dictionaries
        """
        try:
            if not self.session:
                await self.start_session()
            
            native_balances = []
            
            for chain_id, chain_config in CHAINS.items():
                try:
                    cache_key = ('native', wallet_address.lower(), chain_id)
                    cached = _balance_cache_get(cache_key)
                    if cached is not None:
                        native_balances.extend(cached)
                        continue

                    logger.info(f"Fetching native balance from {chain_config['name']}")

                    url = f"{chain_config['rpc_url']}{self.alchemy_api_key}"

                    # Get native balance using Alchemy API
                    params = {
                        'jsonrpc': '2.0',
                        'method': 'eth_getBalance',
                        'params': [wallet_address, 'latest'],
                        'id': 1
                    }

                    data = await self._post_with_retry(url, params)
                    if not data:
                        stale = _balance_cache_get(cache_key, allow_stale=True)
                        if stale is not None:
                            logger.warning(f"Serving stale native balance for {chain_config['name']}")
                            native_balances.extend(stale)
                        continue
                    balance_hex = data.get('result', '0x0')
                    try:
                        balance_wei = int(balance_hex, 16)
                    except Exception:
                        logger.debug(f"Skipping non-hex native balance for {chain_config['name']}: {balance_hex}")
                        continue
                    balance_eth = balance_wei / (10 ** 18)

                    chain_native_balances = []
                    if balance_eth > 0:
                        chain_native_balances.append({
                            'chain_id': chain_id,
                            'chain_name': chain_config['name'],
                            'balance': balance_eth,
                            'symbol': chain_config['native_symbol'],
                        })

                    _balance_cache_put(cache_key, chain_native_balances)
                    native_balances.extend(chain_native_balances)

                except Exception as e:
                    logger.warning(f"Error fetching native balance from {chain_config['name']}: {e}")
            
            logger.debug(f"Fetched {len(native_balances)} native balances")
            return native_balances
            
        except Exception as e:
            logger.error(f"Error fetching native balances: {e}")
            return []
    
    async def _get_token_metadata(self, contract_address: str, chain_id: int, rpc_url: str) -> Optional[Dict[str, Any]]:
        """
        Get token metadata using Alchemy API.
        
        Args:
            contract_address: Token contract address
            chain_id: Chain ID
            rpc_url: RPC URL for the chain
            
        Returns:
            Token metadata dictionary
        """
        try:
            # Check cache first
            cache_key = (chain_id, contract_address.lower())
            if cache_key in self._metadata_cache:
                return self._metadata_cache[cache_key]

            params = {
                'jsonrpc': '2.0',
                'method': 'alchemy_getTokenMetadata',
                'params': [contract_address],
                'id': 1
            }
            
            data = await self._post_with_retry(rpc_url, params)
            if not data:
                return None
            result = data.get('result', {})

            metadata = {
                'symbol': result.get('symbol', 'UNKNOWN'),
                'name': result.get('name', 'Unknown Token'),
                'decimals': result.get('decimals', 18),
            }
            self._metadata_cache[cache_key] = metadata
            return metadata
                    
        except Exception as e:
            logger.warning(f"Error getting metadata for {contract_address}: {e}")
            return None


# Factory function
def create_blockchain_service() -> BlockchainService:
    """Create and return BlockchainService instance."""
    return BlockchainService() 