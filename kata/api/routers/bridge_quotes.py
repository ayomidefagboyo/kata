"""
Bridge Quotes API Router

Provides real-time bridge quotes using LiFi for cross-chain transfers.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from kata.services.lifi_bridge_service import get_lifi_service

logger = logging.getLogger(__name__)
router = APIRouter()


class BridgeQuoteRequest(BaseModel):
    """Request model for bridge quote."""
    from_chain: str
    to_chain: str
    from_token: str
    to_token: str
    amount: float
    wallet_address: str


class BridgeQuoteResponse(BaseModel):
    """Response model for bridge quote."""
    success: bool
    quote: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/quote")
async def get_bridge_quote(
    from_chain: str = Query(..., description="Source chain (e.g., 'base')"),
    to_chain: str = Query(..., description="Destination chain (e.g., 'arbitrum')"),
    from_token: str = Query(..., description="Source token (e.g., 'ETH')"),
    to_token: str = Query(..., description="Destination token (e.g., 'USDC')"),
    amount: float = Query(..., description="Amount to bridge"),
    wallet_address: str = Query(..., description="User's wallet address")
):
    """
    Get real-time bridge quote from LiFi.

    Returns detailed pricing information including:
    - Input and output amounts
    - Bridge fees and gas costs
    - Estimated completion time
    - Bridge provider information
    """
    try:
        lifi_service = get_lifi_service()

        quote = await lifi_service.get_bridge_quote(
            from_chain=from_chain,
            to_chain=to_chain,
            from_token=from_token,
            to_token=to_token,
            amount=amount,
            wallet_address=wallet_address
        )

        if not quote:
            return BridgeQuoteResponse(
                success=False,
                error="Failed to get bridge quote"
            )

        # Format quote response
        quote_data = {
            "input_amount": quote.input_amount,
            "output_amount": quote.output_amount,
            "total_fees": quote.total_fees,
            "bridge_fee": quote.bridge_fee,
            "gas_fee": quote.gas_fee,
            "estimated_time_seconds": quote.estimated_time,
            "estimated_time_minutes": round(quote.estimated_time / 60, 1),
            "bridge_provider": quote.tool_name,
            "slippage": quote.slippage,
            "quote_id": quote.quote_id,
            "valid_until": quote.valid_until.isoformat(),
            "fee_breakdown": {
                "bridge_fee_usd": round(quote.bridge_fee, 2),
                "gas_fee_usd": round(quote.gas_fee, 2),
                "total_fees_usd": round(quote.total_fees, 2),
                "fee_percentage": round((quote.total_fees / (quote.input_amount * 3000)) * 100, 2) if from_token == "ETH" else round((quote.total_fees / quote.input_amount) * 100, 2)
            },
            "route_info": {
                "from_chain": from_chain,
                "to_chain": to_chain,
                "from_token": from_token,
                "to_token": to_token,
                "steps": [
                    {
                        "step": 1,
                        "action": f"Bridge {from_token} from {from_chain.title()} to {to_chain.title()}",
                        "provider": quote.tool_name,
                        "estimated_time": f"{round(quote.estimated_time / 60, 1)} minutes"
                    }
                ]
            }
        }

        return BridgeQuoteResponse(
            success=True,
            quote=quote_data
        )

    except Exception as e:
        logger.error(f"Error getting bridge quote: {e}")
        return BridgeQuoteResponse(
            success=False,
            error=str(e)
        )


@router.post("/quote")
async def post_bridge_quote(request: BridgeQuoteRequest):
    """
    Get bridge quote via POST request (alternative to GET).
    """
    try:
        lifi_service = get_lifi_service()

        quote = await lifi_service.get_bridge_quote(
            from_chain=request.from_chain,
            to_chain=request.to_chain,
            from_token=request.from_token,
            to_token=request.to_token,
            amount=request.amount,
            wallet_address=request.wallet_address
        )

        if not quote:
            return BridgeQuoteResponse(
                success=False,
                error="Failed to get bridge quote"
            )

        quote_data = {
            "input_amount": quote.input_amount,
            "output_amount": quote.output_amount,
            "total_fees": quote.total_fees,
            "bridge_fee": quote.bridge_fee,
            "gas_fee": quote.gas_fee,
            "estimated_time_seconds": quote.estimated_time,
            "estimated_time_minutes": round(quote.estimated_time / 60, 1),
            "bridge_provider": quote.tool_name,
            "slippage": quote.slippage,
            "quote_id": quote.quote_id,
            "valid_until": quote.valid_until.isoformat()
        }

        return BridgeQuoteResponse(
            success=True,
            quote=quote_data
        )

    except Exception as e:
        logger.error(f"Error getting bridge quote: {e}")
        return BridgeQuoteResponse(
            success=False,
            error=str(e)
        )


@router.get("/supported-chains")
async def get_supported_chains():
    """Get list of supported chains for bridging."""
    try:
        lifi_service = get_lifi_service()
        chains = await lifi_service.get_supported_chains()

        return {
            "success": True,
            "chains": chains
        }

    except Exception as e:
        logger.error(f"Error getting supported chains: {e}")
        return {
            "success": False,
            "error": str(e),
            "chains": []
        }


@router.get("/supported-tools")
async def get_supported_tools():
    """Get list of supported bridge providers."""
    try:
        lifi_service = get_lifi_service()
        tools = await lifi_service.get_supported_tools()

        return {
            "success": True,
            "tools": tools
        }

    except Exception as e:
        logger.error(f"Error getting supported tools: {e}")
        return {
            "success": False,
            "error": str(e),
            "tools": []
        }


@router.get("/health")
async def bridge_health_check():
    """Health check for bridge quote service."""
    try:
        lifi_service = get_lifi_service()

        # Test with a small quote to verify service is working
        test_quote = await lifi_service.get_bridge_quote(
            from_chain="base",
            to_chain="arbitrum",
            from_token="ETH",
            to_token="USDC",
            amount=0.001,
            wallet_address="0x0000000000000000000000000000000000000000"
        )

        return {
            "success": True,
            "service": "LiFi Bridge Service",
            "status": "healthy" if test_quote else "degraded",
            "timestamp": logger.name
        }

    except Exception as e:
        logger.error(f"Bridge health check failed: {e}")
        return {
            "success": False,
            "service": "LiFi Bridge Service",
            "status": "unhealthy",
            "error": str(e)
        }