"""
Portfolio router for portfolio data and holdings management.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from supabase import Client
from pydantic import BaseModel
from datetime import datetime

from kata.config.database import get_db_client
from kata.models.portfolio import (
    Portfolio, PortfolioSummary, Holding, HoldingCreate, 
    HoldingUpdate, PortfolioMetrics
)
from kata.services.coingecko_service import CoinGeckoService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


# Dependency functions
def get_coingecko_service() -> CoinGeckoService:
    """Get CoinGecko service instance."""
    return CoinGeckoService()


@router.get("/{user_id}", response_model=PortfolioSummary)
async def get_portfolio(
    user_id: str,
    db: Client = Depends(get_db_client)
) -> PortfolioSummary:
    """Get complete portfolio data for a user."""
    try:
        # Get portfolio summary from view
        portfolio_response = db.from_("user_portfolio_summary").select("*").eq("user_id", user_id).execute()
        
        if not portfolio_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )
        
        portfolio_data = portfolio_response.data[0]
        
        # Get holdings
        holdings_response = db.from_("portfolios").select("id").eq("user_id", user_id).execute()
        
        holdings = []
        if holdings_response.data:
            portfolio_id = holdings_response.data[0]["id"]
            holdings_data = db.from_("holdings").select("*").eq("portfolio_id", portfolio_id).order("value_usd", desc=True).execute()
            
            if holdings_data.data:
                holdings = [Holding(**holding) for holding in holdings_data.data]
        
        # Build portfolio summary
        portfolio_summary = PortfolioSummary(
            id=portfolio_data.get("portfolio_id", ""),
            user_id=user_id,
            total_value_usd=portfolio_data.get("total_value_usd", 0),
            total_pnl_usd=portfolio_data.get("total_pnl_usd", 0),
            total_pnl_percent=portfolio_data.get("total_pnl_percent", 0),
            last_updated=portfolio_data.get("last_updated", ""),
            holdings=holdings,
            total_holdings=len(holdings),
            active_agent_type=portfolio_data.get("active_agent_type"),
            agent_is_active=portfolio_data.get("agent_is_active")
        )
        
        logger.info(f"Retrieved portfolio for user {user_id} with {len(holdings)} holdings")
        return portfolio_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio: {str(e)}"
        )


@router.get("/{user_id}/holdings", response_model=List[Holding])
async def get_holdings(
    user_id: str,
    protocol: str = Query(None, description="Filter by protocol"),
    position_type: str = Query(None, description="Filter by position type"),
    db: Client = Depends(get_db_client)
) -> List[Holding]:
    """Get user's holdings with optional filters."""
    try:
        # Get user's portfolio ID
        portfolio_response = db.from_("portfolios").select("id").eq("user_id", user_id).execute()
        
        if not portfolio_response.data:
            return []
        
        portfolio_id = portfolio_response.data[0]["id"]
        
        # Build query with filters
        query = db.from_("holdings").select("*").eq("portfolio_id", portfolio_id)
        
        if protocol:
            query = query.eq("protocol", protocol)
        
        if position_type:
            query = query.eq("position_type", position_type)
        
        holdings_response = query.order("value_usd", desc=True).execute()
        
        holdings = [Holding(**holding) for holding in holdings_response.data] if holdings_response.data else []
        
        logger.info(f"Retrieved {len(holdings)} holdings for user {user_id}")
        return holdings
        
    except Exception as e:
        logger.error(f"Error getting holdings for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get holdings: {str(e)}"
        )


@router.post("/{user_id}/holdings", response_model=Holding)
async def create_holding(
    user_id: str,
    holding_data: HoldingCreate,
    db: Client = Depends(get_db_client)
) -> Holding:
    """Create a new holding in user's portfolio."""
    try:
        # Verify portfolio exists
        portfolio_response = db.from_("portfolios").select("id").eq("user_id", user_id).execute()
        
        if not portfolio_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )
        
        portfolio_id = portfolio_response.data[0]["id"]
        
        # Update holding data with correct portfolio ID
        holding_data.portfolio_id = portfolio_id
        
        # Create holding
        response = db.from_("holdings").insert(holding_data.model_dump()).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create holding"
            )
        
        holding = Holding(**response.data[0])
        logger.info(f"Created holding {holding.token_symbol} for user {user_id}")
        
        return holding
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating holding for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create holding: {str(e)}"
        )


@router.put("/holdings/{holding_id}", response_model=Holding)
async def update_holding(
    holding_id: str,
    holding_update: HoldingUpdate,
    db: Client = Depends(get_db_client)
) -> Holding:
    """Update an existing holding."""
    try:
        # Update only provided fields
        update_data = {k: v for k, v in holding_update.model_dump().items() if v is not None}
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        response = db.from_("holdings").update(update_data).eq("id", holding_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Holding not found"
            )
        
        holding = Holding(**response.data[0])
        logger.info(f"Updated holding {holding_id}")
        
        return holding
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating holding {holding_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update holding: {str(e)}"
        )


@router.delete("/holdings/{holding_id}")
async def delete_holding(
    holding_id: str,
    db: Client = Depends(get_db_client)
) -> Dict[str, Any]:
    """Delete a holding."""
    try:
        response = db.from_("holdings").delete().eq("id", holding_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Holding not found"
            )
        
        logger.info(f"Deleted holding {holding_id}")
        
        return {
            "success": True,
            "message": "Holding deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting holding {holding_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete holding: {str(e)}"
        )


@router.get("/{user_id}/metrics", response_model=PortfolioMetrics)
async def get_portfolio_metrics(
    user_id: str,
    db: Client = Depends(get_db_client)
) -> PortfolioMetrics:
    """Get portfolio performance metrics."""
    try:
        # Get recent performance metrics
        metrics_response = db.from_("performance_metrics").select("*").eq("user_id", user_id).order("metric_date", desc=True).limit(30).execute()
        
        if not metrics_response.data:
            # Return default metrics if no data
            return PortfolioMetrics()
        
        metrics_data = metrics_response.data
        
        # Calculate metrics from historical data
        latest_value = metrics_data[0]["portfolio_value_usd"]
        
        # Daily change (comparing with yesterday)
        daily_change_usd = 0
        daily_change_percent = 0
        if len(metrics_data) > 1:
            previous_value = metrics_data[1]["portfolio_value_usd"]
            daily_change_usd = latest_value - previous_value
            daily_change_percent = (daily_change_usd / previous_value * 100) if previous_value > 0 else 0
        
        # Weekly change (comparing with 7 days ago)
        weekly_change_usd = 0
        weekly_change_percent = 0
        if len(metrics_data) > 7:
            week_ago_value = metrics_data[7]["portfolio_value_usd"]
            weekly_change_usd = latest_value - week_ago_value
            weekly_change_percent = (weekly_change_usd / week_ago_value * 100) if week_ago_value > 0 else 0
        
        # Monthly change (comparing with 30 days ago)
        monthly_change_usd = 0
        monthly_change_percent = 0
        if len(metrics_data) == 30:
            month_ago_value = metrics_data[-1]["portfolio_value_usd"]
            monthly_change_usd = latest_value - month_ago_value
            monthly_change_percent = (monthly_change_usd / month_ago_value * 100) if month_ago_value > 0 else 0
        
        # All-time high and low
        all_values = [m["portfolio_value_usd"] for m in metrics_data]
        all_time_high = max(all_values) if all_values else 0
        all_time_low = min(all_values) if all_values else 0
        
        portfolio_metrics = PortfolioMetrics(
            daily_change_usd=daily_change_usd,
            daily_change_percent=daily_change_percent,
            weekly_change_usd=weekly_change_usd,
            weekly_change_percent=weekly_change_percent,
            monthly_change_usd=monthly_change_usd,
            monthly_change_percent=monthly_change_percent,
            all_time_high=all_time_high,
            all_time_low=all_time_low
        )
        
        logger.info(f"Retrieved portfolio metrics for user {user_id}")
        return portfolio_metrics
        
    except Exception as e:
        logger.error(f"Error getting portfolio metrics for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get portfolio metrics: {str(e)}"
        )


# Blockchain Portfolio Data Models
class BlockchainTokenBalance(BaseModel):
    """Token balance data structure for blockchain portfolio."""
    contract_address: str
    symbol: str
    name: str
    decimals: int
    balance: str
    balance_usd: float
    price: float
    change_24h: float
    chain_id: int
    chain_name: str


class BlockchainPortfolioData(BaseModel):
    """Blockchain portfolio data structure."""
    total_balance: float
    change_24h: float
    change_24h_percent: float
    total_return: float
    total_return_percent: float
    active_positions: int
    tokens: List[BlockchainTokenBalance]
    chains: Dict[str, Dict[str, Any]]
    last_updated: str


# Simple in-memory cache to prevent rate limiting
_portfolio_cache = {}
_cache_ttl = 300  # 5 minutes

@router.get("/{user_id}/blockchain", response_model=BlockchainPortfolioData)
async def get_blockchain_portfolio(
    user_id: str,
    wallet_address: str = Query(..., description="Wallet address to fetch data for"),
    coingecko_service: CoinGeckoService = Depends(get_coingecko_service)
) -> BlockchainPortfolioData:
    """
    Get real-time blockchain portfolio data using CoinGecko prices.

    This endpoint fetches token balances from multiple EVM chains and
    enriches them with real-time market data from CoinGecko.
    """
    try:
        # Check cache first to prevent rate limiting
        cache_key = f"{user_id}_{wallet_address}"
        current_time = datetime.now().timestamp()

        if cache_key in _portfolio_cache:
            cached_data, cache_time = _portfolio_cache[cache_key]
            if current_time - cache_time < _cache_ttl:
                logger.info(f"Returning cached blockchain portfolio for user {user_id}")
                return cached_data

        logger.info(f"Fetching blockchain portfolio for user {user_id}, wallet: {wallet_address}")
        
        # Import blockchain service here to avoid circular imports
        from kata.services.blockchain_service import BlockchainService
        
        # Initialize blockchain service
        blockchain_service = BlockchainService()
        
        # Get token balances from all chains
        token_balances = await blockchain_service.get_token_balances(wallet_address)
        
        # Get native balances from all chains
        native_balances = await blockchain_service.get_native_balances(wallet_address)
        
        # Collect all symbols for batch price lookup
        symbols_to_fetch = []
        
        # Add native token symbols
        for native in native_balances:
            symbols_to_fetch.append(native['symbol'])
        
        # Add token symbols
        for token in token_balances:
            if token['symbol'] not in symbols_to_fetch:
                symbols_to_fetch.append(token['symbol'])
        
        # Get real prices from CoinGecko using existing service (with error handling)
        try:
            if coingecko_service and symbols_to_fetch:
                logger.info(f"Fetching prices for {len(symbols_to_fetch)} symbols from CoinGecko")
                async with coingecko_service:
                    price_data = await coingecko_service.get_coin_prices_batched(symbols_to_fetch)
                    price_map = {price.symbol: price for price in price_data}
                    logger.info(f"Successfully fetched {len(price_map)} prices from CoinGecko")
            else:
                logger.warning("CoinGecko service unavailable or no symbols to fetch, using fallback values")
                price_map = {}  # Empty price map - will use fallback values
        except Exception as e:
            logger.warning(f"CoinGecko price fetch failed: {e}, using fallback values")
            price_map = {}  # Empty price map - will use fallback values
        
        # Add fallback prices for common tokens if CoinGecko fails
        if not price_map:
            logger.info("Using fallback prices for common tokens")
            fallback_prices = {
                'ETH': {'current_price': 3200.0, 'price_change_percentage_24h': 2.5},
                'USDC': {'current_price': 1.0, 'price_change_percentage_24h': 0.0},
                'USDT': {'current_price': 1.0, 'price_change_percentage_24h': 0.0},
                'DAI': {'current_price': 1.0, 'price_change_percentage_24h': 0.1},
                'WETH': {'current_price': 3200.0, 'price_change_percentage_24h': 2.5},
                'USDbC': {'current_price': 1.0, 'price_change_percentage_24h': 0.0},
                'cbETH': {'current_price': 3180.0, 'price_change_percentage_24h': 2.3},
            }
            price_map = fallback_prices
        
        # Process native tokens
        for native in native_balances:
            try:
                price_data = price_map.get(native['symbol'])
                if price_data:
                    # Handle both CoinGecko response format and fallback format
                    if hasattr(price_data, 'current_price'):
                        price = price_data.current_price
                        change_24h = price_data.price_change_percentage_24h
                    else:
                        price = price_data.get('current_price', 0)
                        change_24h = price_data.get('price_change_percentage_24h', 0)
                else:
                    price = 0
                    change_24h = 0
                
                balance_usd = native['balance'] * price
                
                token_balances.insert(0, {
                    'contract_address': '0x0000000000000000000000000000000000000000',
                    'symbol': native['symbol'],
                    'name': native['chain_name'],
                    'decimals': 18,
                    'balance': str(native['balance'] * (10 ** 18)),
                    'balance_usd': balance_usd,
                    'price': price,
                    'change_24h': change_24h,
                    'chain_id': native['chain_id'],
                    'chain_name': native['chain_name'],
                })
            except Exception as e:
                logger.warning(f"Error processing native token {native['symbol']}: {e}")
        
        # Update token prices with real data from CoinGecko
        for token in token_balances:
            try:
                price_data = price_map.get(token['symbol'])
                if price_data:
                    # Handle both CoinGecko response format and fallback format
                    if hasattr(price_data, 'current_price'):
                        price = price_data.current_price
                        change_24h = price_data.price_change_percentage_24h
                    else:
                        price = price_data.get('current_price', 0)
                        change_24h = price_data.get('price_change_percentage_24h', 0)
                    
                    token['price'] = price
                    token['change_24h'] = change_24h
                    token['balance_usd'] = (float(token['balance']) / (10 ** token['decimals'])) * price
            except Exception as e:
                logger.warning(f"Error updating price for {token['symbol']}: {e}")
        
        # Calculate portfolio totals
        total_balance = sum(token['balance_usd'] for token in token_balances)
        
        # Calculate weighted 24h change
        total_change_24h = 0
        total_change_percent = 0
        
        if total_balance > 0:
            weighted_changes = []
            for token in token_balances:
                weight = token['balance_usd'] / total_balance
                weighted_change = token['change_24h'] * weight
                weighted_changes.append(weighted_change)
            
            total_change_24h = sum(weighted_changes)
            total_change_percent = sum(weighted_changes)
        
        # Group by chain
        chains = {}
        for token in token_balances:
            chain_id = str(token['chain_id'])
            if chain_id not in chains:
                chains[chain_id] = {
                    'name': token['chain_name'],
                    'balance': 0,
                    'token_count': 0,
                }
            chains[chain_id]['balance'] += token['balance_usd']
            chains[chain_id]['token_count'] += 1
        
        # Convert to response models
        blockchain_tokens = [
            BlockchainTokenBalance(
                contract_address=token['contract_address'],
                symbol=token['symbol'],
                name=token['name'],
                decimals=token['decimals'],
                balance=token['balance'],
                balance_usd=token['balance_usd'],
                price=token['price'],
                change_24h=token['change_24h'],
                chain_id=token['chain_id'],
                chain_name=token['chain_name']
            )
            for token in token_balances
        ]
        
        portfolio_data = BlockchainPortfolioData(
            total_balance=total_balance,
            change_24h=total_change_24h,
            change_24h_percent=total_change_percent,
            total_return=total_change_24h,
            total_return_percent=total_change_percent,
            active_positions=len(token_balances),
            tokens=blockchain_tokens,
            chains=chains,
            last_updated=datetime.now().isoformat()
        )
        
        logger.info(f"Blockchain portfolio data fetched successfully: {len(token_balances)} tokens, ${total_balance:.2f} total")

        # Cache the result to prevent rate limiting
        _portfolio_cache[cache_key] = (portfolio_data, current_time)

        return portfolio_data
        
    except Exception as e:
        logger.error(f"Error fetching blockchain portfolio for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch blockchain portfolio: {str(e)}"
        )