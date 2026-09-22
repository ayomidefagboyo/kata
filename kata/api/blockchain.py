"""
Blockchain API Proxy - Secure server-side blockchain data access
Replaces direct frontend Alchemy API calls for security
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import os
import logging
from datetime import datetime, timedelta
from kata.config.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

# Alchemy API key - SECURE server-side only
ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', '')

if not ALCHEMY_API_KEY:
    logger.warning("ALCHEMY_API_KEY not set - blockchain API will be limited")

# Cache for blockchain data. Entries are served as fresh for CACHE_FRESH_SECONDS,
# then retained up to CACHE_STALE_SECONDS so we can fall back to stale data when
# Alchemy is rate limiting instead of failing the request.
_blockchain_cache = {}
CACHE_FRESH_SECONDS = 60
CACHE_STALE_SECONDS = 600

def _get_cache_key(endpoint: str, address: str, chain_id: int) -> str:
    """Generate cache key for blockchain data."""
    return f"{endpoint}_{address}_{chain_id}"

def _get_cached_data(cache_key: str, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
    """Get cached data if fresh, or any retained entry when allow_stale is set."""
    if cache_key in _blockchain_cache:
        cached_data, cached_time = _blockchain_cache[cache_key]
        age = (datetime.now() - cached_time).total_seconds()
        if age < CACHE_FRESH_SECONDS or (allow_stale and age < CACHE_STALE_SECONDS):
            logger.debug(f"Using cached blockchain data for {cache_key}")
            return cached_data
    return None

def _cache_data(cache_key: str, data: Dict[str, Any]):
    """Cache blockchain data with timestamp."""
    _blockchain_cache[cache_key] = (data, datetime.now())
    _cleanup_cache()

def _cleanup_cache():
    """Remove entries too old even for stale fallback."""
    now = datetime.now()
    expired_keys = [
        key for key, (_, cached_time) in _blockchain_cache.items()
        if (now - cached_time).total_seconds() >= CACHE_STALE_SECONDS
    ]
    for key in expired_keys:
        del _blockchain_cache[key]

ALCHEMY_MAX_RETRIES = 3

async def _alchemy_post(client: httpx.AsyncClient, rpc_url: str, payload: Dict[str, Any]) -> httpx.Response:
    """POST to Alchemy, retrying with exponential backoff on 429."""
    backoff = 0.5
    response = None
    for attempt in range(ALCHEMY_MAX_RETRIES + 1):
        response = await client.post(rpc_url, json=payload)
        if response.status_code != 429 or attempt == ALCHEMY_MAX_RETRIES:
            return response
        delay = backoff
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                delay = max(float(retry_after), backoff)
            except ValueError:
                pass
        logger.warning(
            f"Alchemy rate limited (429) on {payload.get('method')}, "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{ALCHEMY_MAX_RETRIES})"
        )
        await asyncio.sleep(delay)
        backoff *= 2
    return response

# Chain configurations
CHAIN_CONFIGS = {
    1: {
        'name': 'Ethereum',
        'rpc_url': f'https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}',
        'explorer': 'https://etherscan.io'
    },
    8453: {
        'name': 'Base',
        'rpc_url': f'https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}',
        'explorer': 'https://basescan.org'
    },
    42161: {
        'name': 'Arbitrum',
        'rpc_url': f'https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}',
        'explorer': 'https://arbiscan.io'
    }
}

# The user-visible Floww Balance is native USDC on Base. Yuki routes assigned
# collateral to Hyperliquid internally, but that is not the deposit balance.
PORTFOLIO_CHAIN_IDS = [8453]

class TokenBalance(BaseModel):
    contractAddress: str
    symbol: str
    name: str
    decimals: int
    balance: str
    balanceUsd: float
    price: float
    change24h: float
    chainId: int
    chainName: str

class PortfolioResponse(BaseModel):
    totalBalance: float
    change24h: float
    change24hPercent: float
    totalReturn: float
    totalReturnPercent: float
    activePositions: int
    tokens: List[TokenBalance]
    chains: Dict[int, Dict[str, Any]]

@router.get("/balance/{address}")
async def get_wallet_balance(address: str, chain_id: int = 8453):
    """Get wallet balance for specific address and chain with 5-minute caching."""
    try:
        if not ALCHEMY_API_KEY:
            raise HTTPException(status_code=503, detail="Blockchain service unavailable")

        chain_config = CHAIN_CONFIGS.get(chain_id)
        if not chain_config:
            raise HTTPException(status_code=400, detail=f"Unsupported chain ID: {chain_id}")

        # Check cache first
        cache_key = _get_cache_key("balance", address, chain_id)
        cached_result = _get_cached_data(cache_key)
        if cached_result:
            return cached_result

        # Call Alchemy API securely from backend
        async with httpx.AsyncClient() as client:
            response = await _alchemy_post(
                client,
                chain_config['rpc_url'],
                {
                    "jsonrpc": "2.0",
                    "method": "eth_getBalance",
                    "params": [address, "latest"],
                    "id": 1
                }
            )

            if response.status_code != 200:
                stale = _get_cached_data(cache_key, allow_stale=True)
                if stale is not None:
                    logger.warning(
                        f"Serving stale balance for {address} on chain {chain_id} "
                        f"(Alchemy HTTP {response.status_code})"
                    )
                    return stale
                raise HTTPException(status_code=500, detail="Failed to fetch balance")

            data = response.json()
            if 'error' in data:
                raise HTTPException(status_code=500, detail=data['error']['message'])

            # Convert hex balance to decimal
            balance_wei = int(data['result'], 16)
            balance_eth = balance_wei / 10**18

            result = {
                "address": address,
                "chainId": chain_id,
                "chainName": chain_config['name'],
                "balance": str(balance_eth),
                "balanceWei": str(balance_wei),
                "currency": "ETH"
            }

            # Cache the result
            _cache_data(cache_key, result)
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching balance for {address}: {e}")
        stale = _get_cached_data(_get_cache_key("balance", address, chain_id), allow_stale=True)
        if stale is not None:
            logger.warning(f"Serving stale balance for {address} on chain {chain_id} after error")
            return stale
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/tokens/{address}")
async def get_token_balances(address: str, chain_id: int = 8453):
    """Get all token balances for an address with 5-minute caching."""
    try:
        if not ALCHEMY_API_KEY:
            raise HTTPException(status_code=503, detail="Blockchain service unavailable")

        chain_config = CHAIN_CONFIGS.get(chain_id)
        if not chain_config:
            raise HTTPException(status_code=400, detail=f"Unsupported chain ID: {chain_id}")

        # Check cache first
        cache_key = _get_cache_key("tokens", address, chain_id)
        cached_result = _get_cached_data(cache_key)
        if cached_result:
            return cached_result

        # Use Alchemy's getTokenBalances method
        async with httpx.AsyncClient() as client:
            response = await _alchemy_post(
                client,
                chain_config['rpc_url'],
                {
                    "jsonrpc": "2.0",
                    "method": "alchemy_getTokenBalances",
                    "params": [address],
                    "id": 1
                }
            )

            if response.status_code != 200:
                stale = _get_cached_data(cache_key, allow_stale=True)
                if stale is not None:
                    logger.warning(
                        f"Serving stale token balances for {address} on chain {chain_id} "
                        f"(Alchemy HTTP {response.status_code})"
                    )
                    return stale
                raise HTTPException(status_code=500, detail="Failed to fetch token balances")

            data = response.json()
            if 'error' in data:
                raise HTTPException(status_code=500, detail=data['error']['message'])

            # Process token balances
            tokens = []
            for token in data['result']['tokenBalances']:
                if int(token['tokenBalance'], 16) > 0:  # Only include non-zero balances
                    tokens.append({
                        "contractAddress": token['contractAddress'],
                        "balance": token['tokenBalance'],
                        "chainId": chain_id,
                        "chainName": chain_config['name']
                    })

            result = {
                "address": address,
                "chainId": chain_id,
                "chainName": chain_config['name'],
                "tokens": tokens
            }

            # Cache the result
            _cache_data(cache_key, result)
            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching token balances for {address}: {e}")
        stale = _get_cached_data(_get_cache_key("tokens", address, chain_id), allow_stale=True)
        if stale is not None:
            logger.warning(f"Serving stale token balances for {address} on chain {chain_id} after error")
            return stale
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/portfolio/{user_id}")
async def get_portfolio_data(user_id: str, wallet_address: str):
    """Get comprehensive portfolio data across all supported chains."""
    try:
        if not ALCHEMY_API_KEY:
            # Return mock data for development
            return PortfolioResponse(
                totalBalance=0.0,
                change24h=0.0,
                change24hPercent=0.0,
                totalReturn=0.0,
                totalReturnPercent=0.0,
                activePositions=0,
                tokens=[],
                chains={}
            )

        portfolio_data = {
            "totalBalance": 0.0,
            "change24h": 0.0,
            "change24hPercent": 0.0,
            "totalReturn": 0.0,
            "totalReturnPercent": 0.0,
            "activePositions": 0,
            "tokens": [],
            "chains": {}
        }

        # Fetch data from the chains the platform actually uses
        for chain_id in PORTFOLIO_CHAIN_IDS:
            chain_config = CHAIN_CONFIGS[chain_id]
            try:
                # Get ETH balance
                balance_data = await get_wallet_balance(wallet_address, chain_id)

                # Get token balances
                token_data = await get_token_balances(wallet_address, chain_id)

                # Process and add to portfolio
                chain_balance = float(balance_data['balance'])
                portfolio_data["totalBalance"] += chain_balance

                if chain_balance > 0 or token_data['tokens']:
                    portfolio_data["chains"][chain_id] = {
                        "name": chain_config['name'],
                        "balance": chain_balance,
                        "tokenCount": len(token_data['tokens'])
                    }

            except Exception as e:
                logger.warning(f"Failed to fetch data for chain {chain_id}: {e}")
                continue

        return PortfolioResponse(**portfolio_data)

    except Exception as e:
        logger.error(f"Error fetching portfolio for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio data")

@router.get("/transaction/{tx_hash}")
async def get_transaction_details(tx_hash: str, chain_id: int = 8453):
    """Get transaction details."""
    try:
        if not ALCHEMY_API_KEY:
            raise HTTPException(status_code=503, detail="Blockchain service unavailable")

        chain_config = CHAIN_CONFIGS.get(chain_id)
        if not chain_config:
            raise HTTPException(status_code=400, detail=f"Unsupported chain ID: {chain_id}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                chain_config['rpc_url'],
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_getTransactionByHash",
                    "params": [tx_hash],
                    "id": 1
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch transaction")

            data = response.json()
            if 'error' in data:
                raise HTTPException(status_code=500, detail=data['error']['message'])

            return {
                "transactionHash": tx_hash,
                "chainId": chain_id,
                "chainName": chain_config['name'],
                "explorerUrl": f"{chain_config['explorer']}/tx/{tx_hash}",
                "transaction": data['result']
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching transaction {tx_hash}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/chains")
async def get_supported_chains():
    """Get list of supported blockchain chains."""
    return {
        "chains": [
            {
                "chainId": chain_id,
                "name": config['name'],
                "explorer": config['explorer'],
                "hasAlchemySupport": True
            }
            for chain_id, config in CHAIN_CONFIGS.items()
        ]
    }
