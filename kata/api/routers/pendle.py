"""
Pendle Finance API endpoints for Flow AI Trading Platform.

Provides endpoints for Pendle yield opportunities, market data, and position management.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field

from kata.services.pendle_service import (
    PendleService,
    PendleYieldOpportunity,
    get_pendle_service
)
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/pendle", tags=["Pendle Finance"])


# Request/Response models
class PendleMarketResponse(BaseModel):
    """Response model for Pendle market data."""
    market_address: str
    pt_address: str
    yt_address: str
    underlying_symbol: str
    maturity: str
    implied_apy: float
    pt_price: float
    liquidity_usd: float
    time_to_maturity_days: int
    is_active: bool


class PendleOpportunityResponse(BaseModel):
    """Response model for yield opportunities."""
    market_address: str
    underlying_symbol: str
    strategy_type: str
    expected_apy: float
    risk_level: str
    time_to_maturity: int
    sakura_score: float
    current_pt_price: float
    discount_to_maturity: float
    break_even_days: int
    liquidity_score: float
    recommended_entry_usd: float


class SwapRequest(BaseModel):
    """Request model for generating swap calldata."""
    token_in_address: str = Field(..., description="Address of input token")
    amount_in: str = Field(..., description="Amount to swap (in wei/smallest unit)")
    pt_address: str = Field(..., description="Address of PT to receive")
    receiver_address: str = Field(..., description="Address to receive tokens")
    slippage: float = Field(0.01, description="Slippage tolerance (0.01 = 1%)")


class SwapResponse(BaseModel):
    """Response model for swap calldata."""
    to: str
    data: str
    value: str
    gas_estimate: int
    route_summary: Dict[str, Any]


class PositionRequest(BaseModel):
    """Request model for creating a position."""
    market_address: str
    position_type: str  # 'PT', 'YT'
    amount_usd: float
    expected_yield: float
    strategy_type: str = "fixed_yield"


# API Endpoints

@router.get("/markets", response_model=List[PendleMarketResponse])
async def get_pendle_markets(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of markets to return"),
    min_liquidity: float = Query(1000000, description="Minimum liquidity in USD"),
    active_only: bool = Query(True, description="Return only active markets"),
    pendle_service: PendleService = Depends(get_pendle_service)
):
    """
    Get available Pendle markets on Base chain.

    Returns markets with PT/YT pairs for yield strategies.
    """
    try:
        logger.info(f"Fetching Pendle markets: limit={limit}, min_liquidity={min_liquidity}")

        # Get markets from Pendle service
        markets = await pendle_service.get_all_markets(limit=limit)

        if not markets:
            logger.warning("No markets received from Pendle service")
            return []

        # Filter and format response
        filtered_markets = []
        for market in markets:
            # Apply filters
            if market.liquidity_usd < min_liquidity:
                continue
            if active_only and not market.is_active:
                continue

            # Calculate time to maturity
            time_to_maturity = (market.maturity - datetime.now()).days
            if time_to_maturity <= 0:
                continue

            market_response = PendleMarketResponse(
                market_address=market.market_address,
                pt_address=market.pt_address,
                yt_address=market.yt_address,
                underlying_symbol=market.underlying_symbol,
                maturity=market.maturity.isoformat(),
                implied_apy=market.implied_apy,
                pt_price=market.pt_price,
                liquidity_usd=market.liquidity_usd,
                time_to_maturity_days=time_to_maturity,
                is_active=market.is_active
            )
            filtered_markets.append(market_response)

        logger.info(f"Returning {len(filtered_markets)} filtered Pendle markets")
        return filtered_markets

    except Exception as e:
        logger.error(f"Error fetching Pendle markets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Pendle markets: {str(e)}")


@router.get("/opportunities/sakura", response_model=List[PendleOpportunityResponse])
async def get_sakura_opportunities(
    max_opportunities: int = Query(10, ge=1, le=20, description="Maximum opportunities to return"),
    min_score: float = Query(0.5, ge=0, le=1, description="Minimum Sakura compatibility score"),
    pendle_service: PendleService = Depends(get_pendle_service)
):
    """
    Get Pendle yield opportunities suitable for Sakura's conservative strategy.

    Filters opportunities based on Sakura's risk criteria and preferences.
    """
    try:
        logger.info(f"Analyzing Pendle opportunities for Sakura: max={max_opportunities}, min_score={min_score}")

        # Get opportunities from Pendle service
        opportunities = await pendle_service.get_sakura_opportunities(max_opportunities=max_opportunities)

        if not opportunities:
            logger.info("No Pendle opportunities found for Sakura")
            return []

        # Filter by minimum score and format response
        filtered_opportunities = []
        for opp in opportunities:
            if opp.sakura_score < min_score:
                continue

            opp_response = PendleOpportunityResponse(
                market_address=opp.market.market_address,
                underlying_symbol=opp.market.underlying_symbol,
                strategy_type=opp.strategy_type,
                expected_apy=opp.expected_apy,
                risk_level=opp.risk_level,
                time_to_maturity=opp.time_to_maturity,
                sakura_score=opp.sakura_score,
                current_pt_price=opp.current_pt_price,
                discount_to_maturity=opp.discount_to_maturity,
                break_even_days=opp.break_even_days,
                liquidity_score=opp.liquidity_score,
                recommended_entry_usd=opp.entry_amount_usd
            )
            filtered_opportunities.append(opp_response)

        logger.info(f"Returning {len(filtered_opportunities)} Sakura-compatible opportunities")
        return filtered_opportunities

    except Exception as e:
        logger.error(f"Error analyzing Sakura opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze opportunities: {str(e)}")


@router.get("/market/{market_address}")
async def get_market_details(
    market_address: str = Path(..., description="Pendle market address"),
    pendle_service: PendleService = Depends(get_pendle_service)
):
    """
    Get detailed information for a specific Pendle market.
    """
    try:
        logger.info(f"Fetching details for Pendle market: {market_address}")

        market_data = await pendle_service.get_market_details(market_address)

        if not market_data:
            raise HTTPException(status_code=404, detail=f"Market not found: {market_address}")

        return market_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market details: {str(e)}")


@router.post("/swap/calldata", response_model=SwapResponse)
async def generate_swap_calldata(
    swap_request: SwapRequest,
    pendle_service: PendleService = Depends(get_pendle_service)
):
    """
    Generate calldata for swapping tokens to PT using Pendle SDK.

    Returns transaction data that can be executed on-chain.
    """
    try:
        logger.info(f"Generating swap calldata: {swap_request.token_in_address} -> {swap_request.pt_address}")

        calldata = await pendle_service.generate_swap_calldata(
            token_in_address=swap_request.token_in_address,
            amount_in=swap_request.amount_in,
            pt_address=swap_request.pt_address,
            receiver_address=swap_request.receiver_address,
            slippage=swap_request.slippage
        )

        if not calldata:
            raise HTTPException(status_code=400, detail="Failed to generate swap calldata")

        return SwapResponse(
            to=calldata.to,
            data=calldata.data,
            value=calldata.value,
            gas_estimate=calldata.gas_estimate,
            route_summary=calldata.route_summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating swap calldata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate calldata: {str(e)}")


@router.get("/yield/calculate/{market_address}")
async def calculate_pt_yield(
    market_address: str = Path(..., description="Pendle market address"),
    pendle_service: PendleService = Depends(get_pendle_service)
):
    """
    Calculate fixed yield for PT purchase in a specific market.

    Returns annual yield percentage and break-even analysis.
    """
    try:
        logger.info(f"Calculating PT yield for market: {market_address}")

        # Get market data first
        all_markets = await pendle_service.get_all_markets()
        target_market = None

        for market in all_markets:
            if market.market_address.lower() == market_address.lower():
                target_market = market
                break

        if not target_market:
            raise HTTPException(status_code=404, detail=f"Market not found: {market_address}")

        # Calculate yield
        annual_yield, break_even_days = await pendle_service.calculate_pt_yield(target_market)

        if annual_yield <= 0:
            raise HTTPException(status_code=400, detail="PT trading at premium - no fixed yield available")

        response = {
            "market_address": market_address,
            "underlying_symbol": target_market.underlying_symbol,
            "annual_yield_percent": annual_yield,
            "break_even_days": break_even_days,
            "time_to_maturity_days": (target_market.maturity - datetime.now()).days,
            "current_pt_price": target_market.pt_price,
            "discount_percent": (1.0 - target_market.pt_price) * 100,
            "maturity_date": target_market.maturity.isoformat()
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating PT yield: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate yield: {str(e)}")


@router.get("/positions/{user_id}")
async def get_user_positions(
    user_id: str = Path(..., description="User ID"),
    active_only: bool = Query(True, description="Return only active positions")
):
    """
    Get user's Pendle positions.

    Returns all PT/YT positions for the specified user.
    """
    try:
        logger.info(f"Fetching Pendle positions for user: {user_id}")

        # TODO: Implement position retrieval from database
        # This would query the PendlePosition model

        # Placeholder response
        return {
            "user_id": user_id,
            "positions": [],
            "total_value_usd": 0.0,
            "total_unrealized_pnl": 0.0,
            "message": "Position tracking not yet implemented"
        }

    except Exception as e:
        logger.error(f"Error fetching user positions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch positions: {str(e)}")


@router.post("/positions")
async def create_position(
    position_request: PositionRequest,
    user_id: str = Query(..., description="User ID")
):
    """
    Create a new Pendle position record.

    Records position details for tracking and analysis.
    """
    try:
        logger.info(f"Creating Pendle position for user {user_id}: {position_request.market_address}")

        # TODO: Implement position creation in database
        # This would create a new PendlePosition record

        # Placeholder response
        return {
            "message": "Position creation not yet implemented",
            "position_request": position_request.dict(),
            "user_id": user_id
        }

    except Exception as e:
        logger.error(f"Error creating position: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create position: {str(e)}")


@router.get("/analytics/performance")
async def get_pendle_performance(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    user_id: Optional[str] = Query(None, description="User ID for user-specific analytics")
):
    """
    Get Pendle strategy performance analytics.

    Returns performance metrics for Pendle yield strategies.
    """
    try:
        logger.info(f"Calculating Pendle performance analytics: {days} days, user={user_id}")

        # TODO: Implement performance analytics
        # This would analyze PendlePosition and PendleTransaction records

        # Placeholder response
        return {
            "period_days": days,
            "user_id": user_id,
            "total_positions": 0,
            "active_positions": 0,
            "total_invested_usd": 0.0,
            "current_value_usd": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "average_apy": 0.0,
            "best_position_apy": 0.0,
            "win_rate": 0.0,
            "message": "Performance analytics not yet implemented"
        }

    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance: {str(e)}")


@router.get("/health")
async def pendle_health_check():
    """Health check for Pendle integration."""
    try:
        # Test connection to Pendle API
        pendle_service = await get_pendle_service()
        markets = await pendle_service.get_all_markets(limit=1)

        api_status = "healthy" if markets else "degraded"

        return {
            "status": "healthy",
            "service": "pendle_integration",
            "api_status": api_status,
            "chain_id": 8453,
            "features": [
                "Market data retrieval",
                "Yield opportunity analysis",
                "Swap calldata generation",
                "Sakura compatibility scoring",
                "Position tracking (planned)",
                "Performance analytics (planned)"
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Pendle health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "pendle_integration",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }