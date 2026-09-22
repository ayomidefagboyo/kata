"""
Hyperliquid Markets API

This module provides endpoints to view all available perpetual futures
contracts on Hyperliquid, their specifications, and current market data.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hyperliquid.info import Info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hyperliquid", tags=["Hyperliquid Markets"])


class MarketInfo(BaseModel):
    """Market information model."""
    symbol: str
    name: str
    max_leverage: int
    tick_size: str
    lot_size: str
    min_size: str
    only_isolated: bool
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


class MarketSummary(BaseModel):
    """Market summary model."""
    total_markets: int
    active_markets: int
    total_volume_24h: float
    average_funding_rate: float
    top_gainers: List[MarketInfo]
    top_losers: List[MarketInfo]
    highest_volume: List[MarketInfo]
    last_updated: str


# Cache for market data
markets_cache = {
    "data": None,
    "last_update": None,
    "cache_duration": 300  # 5 minutes
}


@router.get("/markets")
async def get_all_markets(
    testnet: bool = Query(default=False, description="Use testnet or mainnet"),
    include_prices: bool = Query(default=True, description="Include current prices"),
    include_funding: bool = Query(default=False, description="Include funding rates (slower)")
) -> Dict[str, Any]:
    """
    Get all available perpetual futures markets on Hyperliquid.
    
    This endpoint fetches all available markets with their specifications
    and optionally includes current market data like prices and funding rates.
    """
    try:
        # Check cache first
        now = datetime.now()
        cache_key = f"markets_{testnet}_{include_prices}_{include_funding}"
        
        if (markets_cache["data"] and 
            markets_cache["last_update"] and 
            (now - markets_cache["last_update"]).seconds < markets_cache["cache_duration"]):
            logger.debug("Returning cached markets data")
            return markets_cache["data"]
        
        # Initialize Hyperliquid client
        base_url = "https://api.hyperliquid-testnet.xyz" if testnet else None
        from kata.services.hyperliquid_client_factory import create_info_client
        info_client = create_info_client(base_url=base_url)
        
        # Get market metadata
        meta = info_client.meta()
        universe = meta.get('universe', [])
        
        if not universe:
            return {
                "markets": [],
                "total_markets": 0,
                "error": "No markets found",
                "testnet": testnet,
                "timestamp": now.isoformat()
            }
        
        logger.info(f"Found {len(universe)} markets on Hyperliquid ({'testnet' if testnet else 'mainnet'})")
        
        markets = []
        
        # Get current prices if requested
        all_mids = None
        if include_prices:
            try:
                all_mids = info_client.all_mids()
                logger.debug(f"Fetched prices for {len(all_mids)} markets")
            except Exception as e:
                logger.warning(f"Failed to fetch prices: {e}")
                all_mids = None
        
        # Process each market
        for i, market in enumerate(universe):
            try:
                symbol = market.get('name', f'UNKNOWN_{i}')
                
                # Basic market info
                market_info = MarketInfo(
                    symbol=symbol,
                    name=market.get('name', symbol),
                    max_leverage=int(market.get('maxLeverage', 1)),
                    tick_size=str(market.get('szDecimals', 8)),
                    lot_size=str(market.get('szDecimals', 8)),
                    min_size=str(float(10 ** -int(market.get('szDecimals', 8)))),
                    only_isolated=market.get('onlyIsolated', False)
                )
                
                # Add current price if available
                if all_mids and i < len(all_mids):
                    try:
                        # all_mids is a dict, not a list - use symbol as key
                        current_price = float(all_mids.get(symbol, 0))
                        if current_price == 0 and i < len(all_mids):
                            # Fallback to index if symbol key doesn't work
                            mid_values = list(all_mids.values()) if isinstance(all_mids, dict) else all_mids
                            current_price = float(mid_values[i]) if i < len(mid_values) else 0
                        market_info.current_price = current_price
                        
                        # Get 24h price change
                        if include_prices:
                            try:
                                # Get candles with proper timestamp
                                import time
                                end_time = int(time.time() * 1000)  # Current time in milliseconds
                                start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago
                                
                                candles = info_client.candles_snapshot(symbol, "1d", start_time, end_time)
                                if len(candles) >= 1:
                                    # Use the latest candle for volume and calculate change from open
                                    latest_candle = candles[-1]
                                    candle_open = float(latest_candle['o'])
                                    if candle_open > 0:
                                        change_24h = ((current_price - candle_open) / candle_open) * 100
                                        market_info.price_change_24h = change_24h
                                    
                                    # 24h volume
                                    volume_24h = float(latest_candle['v']) * current_price
                                    market_info.volume_24h = volume_24h
                            except Exception as e:
                                logger.debug(f"Failed to get 24h data for {symbol}: {e}")
                    
                    except Exception as e:
                        logger.debug(f"Failed to process price for {symbol}: {e}")
                
                # Get funding rate if requested
                if include_funding:
                    try:
                        funding_history = info_client.funding_history(symbol, startTime=0, endTime=None)
                        if funding_history:
                            current_funding = float(funding_history[-1].get('fundingRate', 0))
                            market_info.funding_rate = current_funding
                    except Exception as e:
                        logger.debug(f"Failed to get funding for {symbol}: {e}")
                
                markets.append(market_info)
                
            except Exception as e:
                logger.warning(f"Error processing market {i}: {e}")
                continue
        
        # Sort markets by volume (if available)
        markets_with_volume = [m for m in markets if m.volume_24h is not None]
        markets_without_volume = [m for m in markets if m.volume_24h is None]
        
        if markets_with_volume:
            markets_with_volume.sort(key=lambda x: x.volume_24h, reverse=True)
            markets = markets_with_volume + markets_without_volume
        
        result = {
            "markets": [market.dict() for market in markets],
            "total_markets": len(markets),
            "active_markets": len([m for m in markets if m.current_price is not None]),
            "testnet": testnet,
            "timestamp": now.isoformat(),
            "data_included": {
                "prices": include_prices,
                "funding_rates": include_funding,
                "24h_changes": include_prices,
                "volumes": include_prices
            }
        }
        
        # Cache the result
        markets_cache["data"] = result
        markets_cache["last_update"] = now
        
        logger.info(f"Processed {len(markets)} markets successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching Hyperliquid markets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch markets: {e}")


@router.get("/markets/summary")
async def get_markets_summary(
    testnet: bool = Query(default=False, description="Use testnet or mainnet")
) -> MarketSummary:
    """
    Get a summary of all Hyperliquid markets with key statistics.
    
    This endpoint provides an overview including top gainers, losers,
    and highest volume markets.
    """
    try:
        # Get all markets with prices
        markets_data = await get_all_markets(testnet=testnet, include_prices=True, include_funding=True)
        markets = [MarketInfo(**m) for m in markets_data["markets"]]
        
        # Filter markets with price data
        active_markets = [m for m in markets if m.current_price is not None]
        
        # Calculate statistics
        total_volume = sum(m.volume_24h for m in active_markets if m.volume_24h is not None)
        
        funding_rates = [m.funding_rate for m in active_markets if m.funding_rate is not None]
        avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0
        
        # Top gainers (24h price change)
        gainers = sorted([m for m in active_markets if m.price_change_24h is not None], 
                        key=lambda x: x.price_change_24h, reverse=True)[:5]
        
        # Top losers (24h price change)
        losers = sorted([m for m in active_markets if m.price_change_24h is not None], 
                       key=lambda x: x.price_change_24h)[:5]
        
        # Highest volume
        high_volume = sorted([m for m in active_markets if m.volume_24h is not None], 
                           key=lambda x: x.volume_24h, reverse=True)[:5]
        
        return MarketSummary(
            total_markets=len(markets),
            active_markets=len(active_markets),
            total_volume_24h=total_volume,
            average_funding_rate=avg_funding,
            top_gainers=gainers,
            top_losers=losers,
            highest_volume=high_volume,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating markets summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@router.get("/markets/{symbol}")
async def get_market_details(
    symbol: str,
    testnet: bool = Query(default=False, description="Use testnet or mainnet")
) -> Dict[str, Any]:
    """
    Get detailed information for a specific market symbol.
    
    This endpoint provides comprehensive data for a single market including
    current price, funding history, and trading specifications.
    """
    try:
        # Initialize Hyperliquid client
        base_url = "https://api.hyperliquid-testnet.xyz" if testnet else None
        from kata.services.hyperliquid_client_factory import create_info_client
        info_client = create_info_client(base_url=base_url)
        
        # Get market metadata
        meta = info_client.meta()
        universe = meta.get('universe', [])
        
        # Find the symbol
        market_meta = None
        symbol_index = None
        for i, market in enumerate(universe):
            if market.get('name') == symbol:
                market_meta = market
                symbol_index = i
                break
        
        if not market_meta:
            raise HTTPException(status_code=404, detail=f"Market {symbol} not found")
        
        # Get current price
        all_mids = info_client.all_mids()
        current_price = float(all_mids[symbol_index]) if symbol_index < len(all_mids) else None
        
        # Get 24h candle data
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 60 * 60 * 1000)
        candles_24h = info_client.candles_snapshot(symbol, "1d", start_time, end_time)
        
        # Get funding history
        funding_history = info_client.funding_history(symbol, startTime=0, endTime=None)
        
        # Get recent trades (if available)
        try:
            recent_trades = info_client.recent_trades(symbol)
        except:
            recent_trades = []
        
        # Process candle data
        candle_data = {}
        if candles_24h:
            candle = candles_24h[0]
            candle_data = {
                "open": float(candle['o']),
                "high": float(candle['h']),
                "low": float(candle['l']),
                "close": float(candle['c']),
                "volume": float(candle['v']),
                "timestamp": candle['t']
            }
        
        # Process funding data
        funding_data = {
            "current_rate": float(funding_history[-1]['fundingRate']) if funding_history else 0,
            "next_funding": funding_history[-1]['time'] if funding_history else None,
            "history": [
                {
                    "rate": float(f['fundingRate']),
                    "time": f['time']
                } for f in funding_history[-10:]  # Last 10 funding periods
            ] if funding_history else []
        }
        
        # Calculate additional metrics
        price_change_24h = 0
        if candle_data and current_price:
            price_change_24h = ((current_price - candle_data["open"]) / candle_data["open"]) * 100
        
        return {
            "symbol": symbol,
            "market_info": {
                "name": market_meta.get('name'),
                "max_leverage": int(market_meta.get('maxLeverage', 1)),
                "sz_decimals": int(market_meta.get('szDecimals', 8)),
                "only_isolated": market_meta.get('onlyIsolated', False)
            },
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "candle_24h": candle_data,
            "funding": funding_data,
            "recent_trades_count": len(recent_trades),
            "testnet": testnet,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market details for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market details: {e}")


@router.get("/markets/search")
async def search_markets(
    query: str = Query(..., description="Search term (symbol or name)"),
    testnet: bool = Query(default=False, description="Use testnet or mainnet"),
    limit: int = Query(default=10, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search for markets by symbol or name.
    
    This endpoint allows searching through available markets to find
    specific contracts or tokens.
    """
    try:
        # Get all markets
        markets_data = await get_all_markets(testnet=testnet, include_prices=True)
        markets = markets_data["markets"]
        
        # Search through markets
        query_lower = query.lower()
        matching_markets = []
        
        for market in markets:
            symbol = market.get("symbol", "").lower()
            name = market.get("name", "").lower()
            
            if query_lower in symbol or query_lower in name:
                matching_markets.append(market)
        
        # Sort by relevance (exact matches first)
        def relevance_score(market):
            symbol = market.get("symbol", "").lower()
            name = market.get("name", "").lower()
            
            if symbol == query_lower or name == query_lower:
                return 3  # Exact match
            elif symbol.startswith(query_lower) or name.startswith(query_lower):
                return 2  # Starts with query
            else:
                return 1  # Contains query
        
        matching_markets.sort(key=relevance_score, reverse=True)
        
        # Limit results
        limited_results = matching_markets[:limit]
        
        return {
            "query": query,
            "results": limited_results,
            "total_matches": len(matching_markets),
            "returned_count": len(limited_results),
            "testnet": testnet,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching markets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search markets: {e}")