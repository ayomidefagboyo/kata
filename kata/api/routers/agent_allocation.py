"""
Agent Allocation API endpoints for Flow AI Trading Platform.

Provides endpoints for users to allocate funds to trading agents and manage automated trading.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kata.services.agent_allocation_service import (
    get_agent_allocation_service,
    AgentType,
    AllocationStatus
)
from kata.services.hyperliquid_service import create_hyperliquid_service
# binance_service imported lazily inside the function that needs it (avoids loading numpy+pandas at startup)
from kata.services.lifi_bridge_service import get_lifi_service
from kata.services.privy_auth_service import get_privy_auth_service, PrivyAuthService
from kata.config.settings import settings

logger = logging.getLogger(__name__)

_POSITION_CHART_CACHE_TTL_SECONDS = 30
_POSITION_CHART_CACHE_MAX_ENTRIES = 64
_position_chart_cache: Dict[str, tuple[datetime, Dict[str, Any]]] = {}


def _is_material_yuki_position(position_value: Any) -> bool:
    """Hide venue dust that is too small for Yuki to manage as a trade."""
    try:
        return abs(float(position_value or 0)) >= float(settings.YUKI_MIN_TRADE_USDC)
    except (TypeError, ValueError):
        return False


def _position_chart_candle_point(
    timestamp_ms: float,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    entry_price: float,
    direction: str,
    leverage: float,
) -> Optional[Dict[str, Any]]:
    """Normalize exchange OHLC data into the chart response shape."""
    if close_price <= 0 or timestamp_ms <= 0:
        return None
    raw_pnl = (
        (close_price / entry_price - 1.0)
        if direction == "LONG"
        else (1.0 - close_price / entry_price)
    ) if entry_price > 0 else 0.0
    return {
        "timestamp": datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(),
        "price": close_price,
        "open": open_price or close_price,
        "high": high_price or close_price,
        "low": low_price or close_price,
        "close": close_price,
        "volume": volume,
        "pnl_pct": raw_pnl * leverage * 100,
        "kind": "candle",
    }


def _cache_position_chart(cache_key: str, payload: Dict[str, Any]) -> None:
    if len(_position_chart_cache) >= _POSITION_CHART_CACHE_MAX_ENTRIES:
        oldest_key = min(_position_chart_cache, key=lambda key: _position_chart_cache[key][0])
        _position_chart_cache.pop(oldest_key, None)
    _position_chart_cache[cache_key] = (datetime.now(timezone.utc), payload)

# Router instance
router = APIRouter(prefix="/agent-allocation", tags=["Agent Allocation"])


# Request/Response models
class TradingConfigRequest(BaseModel):
    """User-configurable trading parameters."""
    max_position_size_percent: Optional[float] = Field(default=None, ge=1.0, le=50.0, description="% of allocation per trade (1-50%)")
    max_leverage: Optional[float] = Field(default=None, ge=1.0, le=40.0, description="Maximum leverage (1-40x)")
    stop_loss_percent: Optional[float] = Field(default=None, ge=1.0, le=20.0, description="Stop loss percentage (1-20%)")
    take_profit_percent: Optional[float] = Field(default=None, ge=5.0, le=100.0, description="Take profit percentage (5-100%)")
    max_daily_loss_percent: Optional[float] = Field(default=None, ge=1.0, le=20.0, description="Daily loss limit (1-20%)")
    min_confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0, description="Minimum signal confidence (0.1-1.0)")
    max_trades_per_day: Optional[int] = Field(default=None, ge=1, le=50, description="Maximum trades per day (1-50)")
    trading_symbols: Optional[List[str]] = Field(default=None, description="Symbols to trade (e.g., ['BTC', 'ETH', 'SOL'])")
    signal_timeframe: Optional[str] = Field(default=None, description="Signal timeframe (1h, 4h, 1d)")
    risk_tolerance: Optional[str] = Field(default=None, description="Risk tolerance (low, medium, high)")


class AllocateFundsRequest(BaseModel):
    """Request to allocate funds to an agent."""
    agent_type: str = Field(..., description="Type of agent (yuki, sakura, ryu)")
    amount: float = Field(..., gt=0, description="USDC amount to reserve for the selected agent")
    auto_trading: bool = Field(default=True, description="Enable automatic trading")
    trading_config: Optional[TradingConfigRequest] = Field(default=None, description="Custom trading configuration")
    solana_wallet_address: Optional[str] = Field(
        default=None,
        description="Linked Privy Solana wallet used by Ryu",
    )
    funding_tx_hash: Optional[str] = Field(
        default=None,
        description="Legacy Base-to-Solana Ryu funding transaction",
    )
    funding_quote_id: Optional[str] = Field(
        default=None,
        description="LI.FI quote identifier shown during Ryu funding",
    )
    funding_input_usdc: Optional[float] = Field(
        default=None,
        gt=0,
        description="Base USDC sent into the Ryu route",
    )
    base_gas_tx_hash: Optional[str] = Field(
        default=None,
        description="Optional Base transaction used to create the user's Ryu gas reserve",
    )


class AddFundsRequest(BaseModel):
    """Request to add funds to an existing agent allocation."""
    amount: float = Field(..., gt=0, description="Additional USDC amount to add from the agent's unallocated wallet pool")
    solana_wallet_address: Optional[str] = None
    funding_tx_hash: Optional[str] = None
    funding_quote_id: Optional[str] = None
    funding_input_usdc: Optional[float] = Field(default=None, gt=0)
    base_gas_tx_hash: Optional[str] = None


class WithdrawAllocationFundsRequest(BaseModel):
    """Request to return unused allocation capital to the shared pool."""
    amount: float = Field(..., gt=0, description="USDC amount to return to the unallocated pool")


class CloseYukiPositionRequest(BaseModel):
    """Confirm the direction shown to the user before closing a live Yuki position."""
    expected_side: Literal["long", "short"]


class AllocationStatusResponse(BaseModel):
    """Response for allocation status."""
    allocation_id: str
    agent_type: str
    allocated_amount: float
    remaining_amount: float
    used_amount: float
    status: str
    total_trades: int
    realized_pnl: float
    unrealized_pnl: float
    created_at: str
    last_trade_at: Optional[str]
    performance_metrics: Dict[str, Any]


class AllocateFundsResponse(BaseModel):
    """Response for fund allocation."""
    success: bool
    allocation_id: Optional[str] = None
    agent_type: Optional[str] = None
    allocated_amount: Optional[float] = None
    added_amount: Optional[float] = None
    status: Optional[str] = None
    auto_trading_enabled: Optional[bool] = None
    platform_wallet_address: Optional[str] = None
    requires_live_trading_enablement: Optional[bool] = None
    readiness: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None


class YukiEthFundingPlanRequest(BaseModel):
    """Preview ETH funding for a Yuki Hyperliquid USDC allocation."""
    eth_amount: float = Field(..., gt=0, description="Net ETH amount to convert for Yuki (gas reserve is kept on top, not subtracted)")
    gas_reserve_eth: float = Field(
        default=0.002,
        ge=0.0001,
        le=0.05,
        description="ETH to keep in the Kata wallet for Arbitrum gas"
    )
    source_chain: str = Field(default="arbitrum", description="Source chain for ETH funding")
    wallet_address: Optional[str] = Field(default=None, description="Kata platform wallet address for quote routing")


class YukiEthFundingPlanResponse(BaseModel):
    """ETH funding preview for Yuki."""
    success: bool
    source_chain: str
    destination_chain: str
    funding_asset: str
    eth_amount: float
    gas_reserve_eth: float
    eth_to_convert: float
    estimated_usdc: float
    min_yuki_allocation_usdc: float
    min_hyperliquid_deposit_usdc: float
    allocation_ready_after_deposit: bool
    estimated_time_minutes: Optional[float] = None
    provider: Optional[str] = None
    quote_id: Optional[str] = None
    wallet_address: Optional[str] = None
    steps: List[Dict[str, Any]]
    warning: str
    error: Optional[str] = None


class YukiEthFundingExecuteRequest(BaseModel):
    """Execute the previewed ETH funding route for Yuki."""
    eth_amount: float = Field(..., gt=0, description="Net ETH amount to convert into Arbitrum USDC (gas reserve is kept on top)")
    gas_reserve_eth: float = Field(
        default=0.002,
        ge=0.0001,
        le=0.05,
        description="ETH to keep in the Kata wallet for source-chain gas"
    )
    source_chain: str = Field(default="arbitrum", description="Source chain holding the funded ETH")
    wallet_address: Optional[str] = Field(default=None, description="Kata platform wallet address")


class YukiWithdrawRequest(BaseModel):
    """Withdraw USDC from Hyperliquid back to the user's Kata wallet."""
    amount: float = Field(..., gt=0, description="USDC amount to withdraw (Hyperliquid deducts a $1 fee)")


class YukiWithdrawResponse(BaseModel):
    """Result of a Hyperliquid withdrawal."""
    success: bool
    amount_usdc: Optional[float] = None
    fee_usdc: Optional[float] = None
    net_usdc: Optional[float] = None
    destination: Optional[str] = None
    estimated_arrival_minutes: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


class YukiFundingExecutionResponse(BaseModel):
    """Status of a Yuki funding execution."""
    success: bool
    funding_id: Optional[str] = None
    status: Optional[str] = None
    current_step: Optional[str] = None
    source_chain: Optional[str] = None
    destination_chain: Optional[str] = None
    eth_amount: Optional[float] = None
    gas_reserve_eth: Optional[float] = None
    eth_to_convert: Optional[float] = None
    estimated_usdc: Optional[float] = None
    arrived_usdc: Optional[float] = None
    deposited_usdc: Optional[float] = None
    wallet_address: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# Dependency to get authenticated user info
async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current user from authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header required")

    access_token = authorization.replace("Bearer ", "")

    # Verify token with Privy
    privy_auth = get_privy_auth_service()
    user_info = await privy_auth.verify_privy_token(access_token)

    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid access token")

    return {
        "user_id": user_info.user_id,
        "wallet_address": user_info.wallet_address,
        "access_token": access_token  # Include access token for Privy-Hyperliquid service
    }


# Initialize services
def get_allocation_service():
    """Get agent allocation service with Privy-Hyperliquid integration."""
    return get_agent_allocation_service()


async def _resolve_platform_wallet_address(user: Dict[str, Any]) -> str:
    """
    Resolve the user's embedded (platform) wallet address.

    The auth token's wallet_address is the first linked wallet, which for
    external-wallet logins is the login wallet - not where Hyperliquid funds
    live. The delegation record stores the verified embedded wallet, so prefer
    it whenever it exists.
    """
    try:
        from kata.services.delegation_service import get_delegation_service

        delegation = await get_delegation_service().get_user_delegation(user["user_id"])
        if delegation and delegation.wallet_address:
            return delegation.wallet_address
    except Exception as e:
        logger.warning(f"Could not resolve delegated wallet for {user.get('user_id')}: {e}")
    return user.get("wallet_address") or ""


def _format_chain_name(chain: str) -> str:
    """Return a user-facing chain label."""
    chain_key = chain.lower()
    if chain_key == "ethereum":
        return "Ethereum"
    if chain_key == "arbitrum":
        return "Arbitrum"
    if chain_key == "base":
        return "Base"
    if chain_key == "optimism":
        return "Optimism"
    return chain.title()


# API Endpoints

@router.post("/authenticate")
async def authenticate_with_hyperliquid(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Authenticate user with Privy-Hyperliquid integration.

    This endpoint sets up the user's Hyperliquid trading account using their Privy wallet.
    Must be called before fund allocation or trading.
    """
    try:
        allocation_service = get_allocation_service()
        result = await allocation_service.get_yuki_trading_readiness(
            user_id=user["user_id"],
            wallet_address=await _resolve_platform_wallet_address(user)
        )

        if result.get("success"):
            return {
                "success": True,
                **result,
                "message": "Yuki trading readiness checked"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Readiness check failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error authenticating with Hyperliquid: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/funding-plan/yuki/eth", response_model=YukiEthFundingPlanResponse)
async def get_yuki_eth_funding_plan(
    request: YukiEthFundingPlanRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Preview the recommended ETH funding path for Yuki.

    This endpoint does not execute swaps, bridges, or Hyperliquid deposits. It gives
    the UI a safe, user-facing plan: reserve ETH for gas, convert the rest to USDC,
    then deposit USDC to Hyperliquid before allocation.
    """
    lifi_service = get_lifi_service()
    source_chain = lifi_service.normalize_chain(request.source_chain)
    supported_chains = {
        chain["slug"]
        for chain in lifi_service.get_configured_chains()
        if chain.get("supports_eth_funding")
    }
    if source_chain not in supported_chains:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source_chain. Supported: {sorted(supported_chains)}"
        )

    wallet_address = request.wallet_address or user.get("wallet_address")
    if not wallet_address:
        raise HTTPException(status_code=400, detail="Kata wallet address is required for a funding quote")

    gas_reserve_eth = request.gas_reserve_eth if source_chain == "arbitrum" else 0.0
    # eth_amount is the NET amount to convert for Yuki. The gas reserve is kept
    # in the wallet on top of it, never subtracted from it - the funding UI
    # already excludes the reserve when suggesting amounts (Max/percent).
    eth_to_convert = request.eth_amount
    source_label = _format_chain_name(source_chain)

    # This plan is built from ETH already sitting in the user's Kata Wallet
    # (the same balance the funding UI reads), not funds still to be sent.
    base_steps = [
        {
            "step": 1,
            "title": f"Use ETH from your Kata Wallet on {source_label}",
            "description": f"{eth_to_convert:.6f} ETH from your Kata Wallet goes to Yuki.",
            "asset": "ETH",
            "amount": eth_to_convert,
        },
    ]
    if source_chain == "arbitrum":
        base_steps.append({
            "step": 2,
            "title": "Reserve ETH for Arbitrum gas",
            "description": (
                f"Kata keeps {gas_reserve_eth:.6f} ETH available for Arbitrum gas "
                "needed to swap and deposit USDC."
            ),
            "asset": "ETH",
            "amount": gas_reserve_eth,
        })
    else:
        base_steps.append({
            "step": 2,
            "title": f"Reserve {source_label} ETH for gas",
            "description": (
                f"Kata keeps enough ETH on {source_label} to pay network gas, then "
                "routes the rest to Arbitrum USDC for Yuki."
            ),
            "asset": "ETH",
        })

    try:
        quote = await get_lifi_service().get_bridge_quote(
            from_chain=source_chain,
            to_chain="arbitrum",
            from_token="ETH",
            to_token="USDC",
            amount=eth_to_convert,
            wallet_address=wallet_address,
        )

        if not quote:
            raise HTTPException(status_code=502, detail="Could not get ETH to USDC quote")

        estimated_usdc = max(float(quote.output_amount or 0.0), 0.0)
        is_same_chain = source_chain == "arbitrum"
        convert_title = "Swap ETH to USDC on Arbitrum" if is_same_chain else "Bridge and swap ETH to Arbitrum USDC"

        steps = [
            *base_steps,
            {
                "step": 3,
                "title": convert_title,
                "description": (
                    f"Convert {eth_to_convert:.6f} ETH into approximately "
                    f"{estimated_usdc:.2f} USDC using {quote.tool_name}."
                ),
                "asset": "ETH",
                "amount": eth_to_convert,
            },
            {
                "step": 4,
                "title": "Deposit USDC to Hyperliquid",
                "description": (
                    "Only Arbitrum USDC is deposited to Hyperliquid. After the deposit lands, "
                    "Kata checks the Hyperliquid balance before activating Yuki."
                ),
                "asset": "USDC",
                "amount": estimated_usdc,
            },
        ]

        return YukiEthFundingPlanResponse(
            success=True,
            source_chain=source_chain,
            destination_chain="arbitrum",
            funding_asset="ETH",
            eth_amount=request.eth_amount,
            gas_reserve_eth=gas_reserve_eth,
            eth_to_convert=eth_to_convert,
            estimated_usdc=estimated_usdc,
            min_yuki_allocation_usdc=settings.YUKI_MIN_ALLOCATION_USDC,
            min_hyperliquid_deposit_usdc=5.0,
            allocation_ready_after_deposit=estimated_usdc >= settings.YUKI_MIN_ALLOCATION_USDC,
            estimated_time_minutes=round((quote.estimated_time or 0) / 60, 1),
            provider=quote.tool_name,
            quote_id=quote.quote_id,
            wallet_address=wallet_address,
            steps=steps,
            warning=(
                "Do not send ETH directly to Hyperliquid's Arbitrum USDC deposit path. "
                "Yuki trades with USDC collateral; Kata must route ETH into Arbitrum USDC first."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Yuki ETH funding plan: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


def _funding_response(result: Dict[str, Any], success: bool = True) -> "YukiFundingExecutionResponse":
    """Map a funding-service result onto the API response model."""
    allowed = set(YukiFundingExecutionResponse.model_fields.keys())
    payload = {key: value for key, value in result.items() if key in allowed}
    payload["success"] = success and result.get("success", True) is not False
    return YukiFundingExecutionResponse(**payload)


@router.post("/funding/yuki/eth/execute", response_model=YukiFundingExecutionResponse)
async def execute_yuki_eth_funding(
    request: YukiEthFundingExecuteRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute the previewed ETH funding route for Yuki.

    Runs the full pipeline from the user's delegated Kata wallet: convert ETH to
    Arbitrum USDC via the live LiFi route, then deposit the USDC to Hyperliquid.
    Progress is tracked per step and can be polled via the executions endpoint.
    """
    from kata.services.yuki_funding_service import get_yuki_funding_service

    wallet_address = request.wallet_address or user.get("wallet_address")
    if not wallet_address:
        raise HTTPException(status_code=400, detail="Kata wallet address is required to execute funding")

    try:
        result = await get_yuki_funding_service().start_funding(
            user_id=user["user_id"],
            wallet_address=wallet_address,
            source_chain=request.source_chain,
            eth_amount=request.eth_amount,
            gas_reserve_eth=request.gas_reserve_eth,
        )
    except Exception as e:
        logger.error(f"Error starting Yuki funding execution: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    if not result.get("success"):
        return _funding_response(result, success=False)
    return _funding_response(result)


@router.get("/readiness/yuki")
async def get_yuki_readiness(
    wallet_address: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Return the user's real Yuki readiness, including Hyperliquid USDC balance."""
    wallet = wallet_address or user.get("wallet_address")
    if not wallet:
        raise HTTPException(status_code=400, detail="Wallet address is required")

    try:
        allocation_service = get_allocation_service()
        return await allocation_service.get_yuki_trading_readiness(
            user_id=user["user_id"],
            wallet_address=wallet,
        )
    except Exception as e:
        logger.error(f"Error getting Yuki readiness: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/readiness/ryu")
async def get_ryu_readiness(
    solana_wallet_address: str,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Return Ryu's Base-home balance, fee reserve, and wallet readiness."""
    try:
        ethereum_wallet = await _resolve_platform_wallet_address(user)
        return await get_allocation_service().ryu_spot_trading_service.get_trading_readiness(
            user_id=user["user_id"],
            ethereum_wallet_address=ethereum_wallet,
            solana_wallet_address=solana_wallet_address,
        )
    except Exception as e:
        logger.error(f"Error getting Ryu readiness: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/funding/yuki/executions/{funding_id}", response_model=YukiFundingExecutionResponse)
async def get_yuki_funding_execution(
    funding_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Poll the status of a Yuki funding execution."""
    from kata.services.yuki_funding_service import get_yuki_funding_service

    status = get_yuki_funding_service().get_status(funding_id, user["user_id"])
    if not status:
        raise HTTPException(status_code=404, detail="Funding execution not found")
    return _funding_response(status, success=status.get("status") != "failed")


@router.post("/withdraw", response_model=YukiWithdrawResponse)
async def withdraw_from_hyperliquid(
    request: YukiWithdrawRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Withdraw USDC from the user's Hyperliquid account to their Kata wallet.

    Signed with the delegated wallet via Hyperliquid's native withdrawal;
    Arbitrum USDC lands in the Kata wallet in about 5 minutes ($1 fee).
    Allocation records are reduced to match the remaining balance.
    """
    try:
        allocation_service = get_allocation_service()
        result = await allocation_service.withdraw_from_hyperliquid(
            user_id=user["user_id"],
            amount=request.amount,
            access_token=user["access_token"],
            wallet_address=await _resolve_platform_wallet_address(user),
        )
        if result.get("success"):
            return YukiWithdrawResponse(**result)
        raise HTTPException(status_code=400, detail=result.get("error", "Withdrawal failed"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error withdrawing from Hyperliquid: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/allocate", response_model=AllocateFundsResponse)
async def allocate_funds_to_agent(
    request: AllocateFundsRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Allocate funds from platform wallet to a trading agent.

    This endpoint:
    1. Allocates USDC from user's platform wallet to the specified agent
    2. Sets up automated trading if enabled
    3. Returns allocation details and status
    """
    try:
        # Yuki allocation never executes synthetic funding. ETH funding must happen
        # as real on-chain transactions before this endpoint is called.
        if not request.amount:
            raise HTTPException(
                status_code=400,
                detail="USDC amount is required. Fund Hyperliquid with real USDC before allocating to Yuki."
            )

        # Validate agent type
        try:
            agent_type = AgentType(request.agent_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Supported: {[t.value for t in AgentType]}"
            )

        # Get allocation service
        allocation_service = get_allocation_service()

        # Convert trading config to dict if provided
        trading_config_dict = None
        if request.trading_config:
            trading_config_dict = request.trading_config.model_dump(exclude_none=True)

        # Allocate funds to agent, using the embedded wallet that actually
        # holds the Hyperliquid balance (not the login wallet from the token).
        result = await allocation_service.allocate_funds_to_agent(
            user_id=user["user_id"],
            agent_type=agent_type,
            amount=request.amount,
            auto_trading=request.auto_trading,
            trading_config=trading_config_dict,
            access_token=user["access_token"],  # Pass access token for Privy authentication
            wallet_address=await _resolve_platform_wallet_address(user),
            solana_wallet_address=request.solana_wallet_address,
            funding_tx_hash=request.funding_tx_hash,
            funding_quote_id=request.funding_quote_id,
            funding_input_usdc=request.funding_input_usdc,
            base_gas_tx_hash=request.base_gas_tx_hash,
        )

        if result.get("success"):
            return AllocateFundsResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Allocation failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating funds to agent: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/allocations/{allocation_id}/add-funds", response_model=AllocateFundsResponse)
async def add_funds_to_yuki_allocation(
    allocation_id: str,
    request: AddFundsRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add unallocated wallet USDC to an existing agent allocation.

    This keeps one allocation row and increases its available trading
    budget instead of creating a second agent session.
    """
    try:
        allocation_service = get_allocation_service()
        result = await allocation_service.add_funds_to_allocation(
            user_id=user["user_id"],
            allocation_id=allocation_id,
            amount=request.amount,
            wallet_address=await _resolve_platform_wallet_address(user),
            solana_wallet_address=request.solana_wallet_address,
            funding_tx_hash=request.funding_tx_hash,
            funding_quote_id=request.funding_quote_id,
            funding_input_usdc=request.funding_input_usdc,
            base_gas_tx_hash=request.base_gas_tx_hash,
        )

        if result.get("success"):
            return AllocateFundsResponse(**result)

        raise HTTPException(status_code=400, detail=result.get("error", "Could not add funds to Yuki"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding funds to agent allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/allocations/{allocation_id}/withdraw")
async def withdraw_from_agent_allocation(
    allocation_id: str,
    request: WithdrawAllocationFundsRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Return unused capital from one agent to its unallocated wallet pool."""
    try:
        result = await get_allocation_service().withdraw_funds_from_allocation(
            user_id=user["user_id"],
            allocation_id=allocation_id,
            amount=request.amount,
        )
        if result.get("success"):
            return result
        raise HTTPException(status_code=400, detail=result.get("error", "Withdrawal failed"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error withdrawing from agent allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/status")
async def get_allocation_status(
    allocation_id: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get allocation status for user.

    If allocation_id is provided, returns specific allocation.
    Otherwise, returns all allocations for the user.
    """
    try:
        allocation_service = get_allocation_service()

        result = await allocation_service.get_allocation_status(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting allocation status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/pause/{allocation_id}")
async def pause_agent_trading(
    allocation_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Pause automated trading for an agent allocation.

    The agent will stop executing new trades but existing positions remain open.
    """
    try:
        allocation_service = get_allocation_service()

        result = await allocation_service.pause_agent_trading(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )

        if result.get("success"):
            return {"message": "Agent trading paused successfully", "status": result["status"]}
        else:
            error = result.get("error", "Failed to pause agent trading")
            status_code = 403 if error == "Unauthorized" else 404 if error == "Allocation not found" else 400
            raise HTTPException(status_code=status_code, detail=error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing agent trading: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/resume/{allocation_id}")
async def resume_agent_trading(
    allocation_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Resume automated trading for an agent allocation.

    The agent will start executing trades again based on signals.
    """
    try:
        allocation_service = get_allocation_service()

        result = await allocation_service.resume_agent_trading(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )

        if result.get("success"):
            return {"message": "Agent trading resumed successfully", "status": result["status"]}
        else:
            error = result.get("error", "Failed to resume agent trading")
            status_code = 403 if error == "Unauthorized" else 404 if error == "Allocation not found" else 400
            raise HTTPException(status_code=status_code, detail=error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming agent trading: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/stop/{allocation_id}")
async def stop_agent_allocation(
    allocation_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Stop one agent allocation and return its capital to the unallocated pool."""
    try:
        result = await get_allocation_service().stop_agent_trading(
            user_id=user["user_id"],
            allocation_id=allocation_id,
        )
        if result.get("success"):
            return {"message": "Agent stopped successfully", **result}
        error = result.get("error", "Failed to stop agent")
        status_code = 403 if error == "Unauthorized" else 404 if error == "Allocation not found" else 400
        raise HTTPException(status_code=status_code, detail=error)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping agent allocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/agents")
async def get_available_agents():
    """
    Get list of available trading agents and their descriptions.
    """
    try:
        agents = {
            "yuki": {
                "name": "Yuki",
                "description": "Futures trading specialist",
                "specialization": "Hyperliquid perpetual futures",
                "risk_level": "High",
                "target_return": "30-100% annually",
                "supported_assets": ["BTC", "ETH", "SOL", "AVAX", "MATIC"],
                "features": [
                    "Advanced technical analysis",
                    "Dynamic leverage management (2x-10x)",
                    "Funding rate arbitrage",
                    "Real-time market data integration"
                ],
                "minimum_allocation": 5.0  # $5 minimum for testing
            },
            "sakura": {
                "name": "Sakura",
                "description": "Yield farming specialist",
                "specialization": "Pendle fixed yield strategies",
                "risk_level": "Low",
                "target_return": "5-15% annually",
                "supported_assets": ["USDC", "ETH", "cbETH", "stETH"],
                "features": [
                    "Fixed yield strategies",
                    "Pendle protocol integration",
                    "Automated position management",
                    "Low-risk yield optimization"
                ],
                "minimum_allocation": 10.0  # $10 minimum for testing
            },
            "ryu": {
                "name": "Ryu",
                "description": "Spot trading specialist",
                "specialization": "DeFi spot trading strategies",
                "risk_level": "Medium",
                "target_return": "15-30% annually",
                "supported_assets": ["ETH", "USDC", "major DeFi tokens"],
                "features": [
                    "Spot trading strategies",
                    "DCA and swing trading",
                    "Portfolio rebalancing",
                    "Medium-risk growth strategies"
                ],
                "minimum_allocation": 7.5  # $7.50 minimum for testing
            }
        }

        return {
            "agents": agents,
            "total_agents": len(agents),
            "currently_supported": ["yuki"],  # Only Yuki is fully implemented
            "coming_soon": ["sakura", "ryu"]
        }

    except Exception as e:
        logger.error(f"Error getting available agents: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/default-config/{agent_type}")
async def get_default_trading_config(agent_type: str):
    """
    Get default trading configuration for an agent type.

    This endpoint returns the platform's recommended default settings
    that users can customize when allocating funds.
    """
    try:
        # Validate agent type
        try:
            agent_enum = AgentType(agent_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Supported: {[t.value for t in AgentType]}"
            )

        allocation_service = get_allocation_service()

        # Get default config from platform signals service
        try:
            default_config = await allocation_service.platform_signals_service.get_agent_default_config(agent_type.lower())
        except Exception as e:
            logger.warning(f"Could not get platform default config: {e}")
            # Return hardcoded defaults as fallback
            default_config = {
                "max_position_size_percent": 15.0,
                "max_leverage": 5.0,
                "stop_loss_percent": 5.0,
                "take_profit_percent": 20.0,
                "max_daily_loss_percent": 5.0,
                "min_confidence_threshold": 0.7,
                "max_trades_per_day": 10,
                "trading_symbols": ["BTC", "ETH", "SOL"],
                "signal_timeframe": "1h",
                "risk_tolerance": "medium"
            }

        return {
            "agent_type": agent_type.lower(),
            "default_config": default_config,
            "description": "Platform recommended default settings",
            "customizable": True,
            "parameter_limits": {
                "max_position_size_percent": {"min": 1.0, "max": 50.0, "description": "Percentage of allocation per trade"},
                "max_leverage": {"min": 1.0, "max": 40.0, "description": "Maximum leverage multiplier"},
                "stop_loss_percent": {"min": 1.0, "max": 20.0, "description": "Stop loss percentage"},
                "take_profit_percent": {"min": 5.0, "max": 100.0, "description": "Take profit percentage"},
                "max_daily_loss_percent": {"min": 1.0, "max": 20.0, "description": "Daily loss limit percentage"},
                "min_confidence_threshold": {"min": 0.1, "max": 1.0, "description": "Minimum signal confidence (0.1=10%, 1.0=100%)"},
                "max_trades_per_day": {"min": 1, "max": 50, "description": "Maximum trades per day"},
                "signal_timeframe": {"options": ["1h", "4h", "1d"], "description": "Signal analysis timeframe"},
                "risk_tolerance": {"options": ["low", "medium", "high"], "description": "Overall risk tolerance"}
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting default trading config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/trades/{allocation_id}")
async def get_allocation_trades(
    allocation_id: str,
    limit: int = 100,
    offset: int = 0,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get trade history for an agent allocation.
    
    Args:
        allocation_id: Allocation ID
        limit: Maximum number of trades to return (default: 100)
        offset: Number of trades to skip (default: 0)
        
    Returns:
        List of trades with detailed information
    """
    try:
        allocation_service = get_allocation_service()
        
        # Verify user owns this allocation
        allocation_status = await allocation_service.get_allocation_status(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )
        
        if "error" in allocation_status:
            raise HTTPException(status_code=404, detail=allocation_status["error"])
        
        # Get trades from database
        from kata.services.agent_database_service import get_agent_db_service
        db_service = get_agent_db_service()
        
        trades = await db_service.get_agent_trades(allocation_id, limit, offset)
        
        return {
            "allocation_id": allocation_id,
            "trades": trades,
            "total_count": len(trades),
            "limit": limit,
            "offset": offset
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting allocation trades: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/positions/{allocation_id}")
async def get_allocation_positions(
    allocation_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get active positions for an agent allocation.
    
    Args:
        allocation_id: Allocation ID
        
    Returns:
        List of active positions with real-time PnL
    """
    try:
        allocation_service = get_allocation_service()
        
        # Verify user owns this allocation
        allocation_status = await allocation_service.get_allocation_status(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )
        
        if "error" in allocation_status:
            raise HTTPException(status_code=404, detail=allocation_status["error"])
        
        # Database rows carry the signal metadata, while Hyperliquid is the source
        # of truth for net size, mark, margin and live PnL. The exchange nets all
        # fills for one coin, so this also prevents duplicate reconciliation rows
        # from appearing as separate active positions in the dashboard.
        from kata.services.agent_database_service import get_agent_db_service
        db_service = get_agent_db_service()
        db_positions = await db_service.get_agent_positions(allocation_id)
        trades = await db_service.get_agent_trades(allocation_id, limit=200)

        wallet_address = None
        cached_allocation = allocation_service.active_allocations.get(allocation_id)
        if cached_allocation is not None:
            wallet_address = cached_allocation.platform_wallet_address
        if not wallet_address:
            allocation_rows = (
                allocation_service.supabase.table("agent_allocations")
                .select("platform_wallet_address")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user["user_id"])
                .limit(1)
                .execute()
            ).data or []
            wallet_address = allocation_rows[0].get("platform_wallet_address") if allocation_rows else None

        live_positions = None
        if wallet_address and allocation_service.hyperliquid_service and allocation_status.get("agent_type") == "yuki":
            live_positions = await allocation_service.hyperliquid_service.get_positions_for_wallet(wallet_address)

        now = datetime.now().isoformat()
        positions = [
            position for position in db_positions
            if _is_material_yuki_position(position.get("position_value"))
        ]
        data_source = "database_cache"
        if live_positions is not None:
            data_source = "hyperliquid_live"
            positions = []
            for coin, live_position in live_positions.items():
                if not _is_material_yuki_position(live_position.position_value):
                    continue
                normalized_coin = allocation_service._quote_stripped_symbol(coin)
                matching_rows = [
                    row for row in db_positions
                    if allocation_service._quote_stripped_symbol(row.get("symbol")) == normalized_coin
                    and str(row.get("side") or "").lower() == str(live_position.side).lower()
                ]
                db_row = matching_rows[0] if matching_rows else None

                trade = None
                if db_row and db_row.get("trade_id"):
                    trade = next((row for row in trades if row.get("id") == db_row.get("trade_id")), None)
                if trade is None:
                    expected_trade_side = "buy" if str(live_position.side).lower() == "long" else "sell"
                    trade = next((
                        row for row in trades
                        if allocation_service._quote_stripped_symbol(row.get("symbol")) == normalized_coin
                        and str(row.get("side") or "").lower() == expected_trade_side
                        and row.get("status") in ("filled", "partially_filled")
                    ), None)

                metadata = dict((db_row or {}).get("position_metadata") or {})
                trade_metadata = (trade or {}).get("trade_metadata") or {}
                for key in ("stop_loss", "target_1", "target_2", "protective_orders", "logo_url"):
                    if metadata.get(key) is None and trade_metadata.get(key) is not None:
                        metadata[key] = trade_metadata.get(key)
                signal_id = trade_metadata.get("signal_id")
                if signal_id:
                    metadata["signal_id"] = signal_id

                margin_used = float(live_position.margin_used or 0)
                unrealized_pnl = float(live_position.unrealized_pnl or 0)
                positions.append({
                    "id": (db_row or {}).get("id") or f"live-{allocation_id}-{normalized_coin}",
                    "user_id": user["user_id"],
                    "allocation_id": allocation_id,
                    "trade_id": (trade or {}).get("id") or (db_row or {}).get("trade_id"),
                    "agent_type": "yuki",
                    "symbol": normalized_coin,
                    "side": live_position.side,
                    "size": float(live_position.size or 0),
                    "entry_price": float(live_position.entry_price or 0),
                    "current_price": float(live_position.mark_price or live_position.entry_price or 0),
                    "leverage": float(live_position.leverage or 1),
                    "position_value": float(live_position.position_value or 0),
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": (unrealized_pnl / margin_used * 100) if margin_used > 0 else 0.0,
                    "margin_used": margin_used,
                    "liquidation_price": live_position.liquidation_price,
                    "margin_ratio": (margin_used / float(live_position.position_value)) if live_position.position_value else None,
                    "is_active": True,
                    "created_at": (db_row or {}).get("created_at") or (trade or {}).get("filled_at") or now,
                    "updated_at": now,
                    "hyperliquid_position_id": (db_row or {}).get("hyperliquid_position_id") or normalized_coin,
                    "position_metadata": metadata,
                    "signal_id": signal_id,
                    "signal_reasoning": (trade or {}).get("signal_reasoning"),
                    "data_source": data_source,
                })

        positions.sort(key=lambda p: (str(p.get("created_at") or ""), str(p.get("symbol") or "")), reverse=True)

        for position in positions:
            position.setdefault("position_metadata", position.get("metadata") or {})
            position.setdefault("data_source", data_source)

        # Calculate total unrealized PnL
        total_unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)
        total_position_value = sum(pos.get("position_value", 0) for pos in positions)
        
        return {
            "allocation_id": allocation_id,
            "positions": positions,
            "total_positions": len(positions),
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_position_value": total_position_value,
            "data_source": data_source,
            "last_updated": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting allocation positions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get("/positions/{allocation_id}/stream")
async def stream_allocation_positions(
    allocation_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """
    SSE stream of live Hyperliquid positions and order updates for an allocation.

    Sends ``text/event-stream`` events of two types:
    * ``event: positions`` – ``clearinghouseState`` parsed into the same shape
      as ``GET /positions/{allocation_id}``.
    * ``event: orders`` – list of live resting orders from Hyperliquid.

    Falls back gracefully: if the WS is unavailable the client will receive an
    ``event: error`` frame and should switch to REST polling.
    """
    from kata.services.hyperliquid_stream_manager import get_stream_manager
    from kata.services.agent_database_service import get_agent_db_service

    allocation_service = get_allocation_service()

    # -----------------------------------------------------------------
    # 1. Resolve wallet address for this allocation
    # -----------------------------------------------------------------
    wallet_address: Optional[str] = None
    cached_allocation = allocation_service.active_allocations.get(allocation_id)
    if cached_allocation:
        wallet_address = cached_allocation.platform_wallet_address
    if not wallet_address:
        try:
            rows = (
                allocation_service.supabase.table("agent_allocations")
                .select("platform_wallet_address")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user["user_id"])
                .limit(1)
                .execute()
            ).data or []
            wallet_address = rows[0].get("platform_wallet_address") if rows else None
        except Exception as _wex:
            logger.warning(f"[stream] Could not resolve wallet for {allocation_id}: {_wex}")

    if not wallet_address:
        raise HTTPException(status_code=404, detail="Allocation not found or no wallet")

    # -----------------------------------------------------------------
    # 2. Build snapshot payload (reuse existing polling logic)
    # -----------------------------------------------------------------
    async def _build_snapshot() -> Optional[Dict[str, Any]]:
        try:
            db_service = get_agent_db_service()
            db_positions = await db_service.get_agent_positions(allocation_id)
            trades = await db_service.get_agent_trades(allocation_id, limit=200)

            live_positions = None
            if allocation_service.hyperliquid_service:
                live_positions = await allocation_service.hyperliquid_service.get_positions_for_wallet(
                    wallet_address
                )

            positions = [
                position for position in db_positions
                if _is_material_yuki_position(position.get("position_value"))
            ]
            data_source = "database_cache"
            if live_positions is not None:
                data_source = "hyperliquid_live"
                now = datetime.now().isoformat()
                positions = []
                for coin, live_pos in live_positions.items():
                    if not _is_material_yuki_position(live_pos.position_value):
                        continue
                    norm = allocation_service._quote_stripped_symbol(coin)
                    matching = [
                        r for r in db_positions
                        if allocation_service._quote_stripped_symbol(r.get("symbol")) == norm
                        and str(r.get("side") or "").lower() == str(live_pos.side).lower()
                    ]
                    db_row = matching[0] if matching else None
                    trade = next(
                        (t for t in trades if t.get("id") == (db_row or {}).get("trade_id")),
                        None,
                    )
                    margin_used = float(live_pos.margin_used or 0)
                    unrealized_pnl = float(live_pos.unrealized_pnl or 0)
                    positions.append({
                        "id": (db_row or {}).get("id") or f"live-{allocation_id}-{norm}",
                        "user_id": user["user_id"],
                        "allocation_id": allocation_id,
                        "trade_id": (trade or {}).get("id") or (db_row or {}).get("trade_id"),
                        "agent_type": "yuki",
                        "symbol": norm,
                        "side": live_pos.side,
                        "size": float(live_pos.size or 0),
                        "entry_price": float(live_pos.entry_price or 0),
                        "current_price": float(live_pos.mark_price or live_pos.entry_price or 0),
                        "leverage": float(live_pos.leverage or 1),
                        "position_value": float(live_pos.position_value or 0),
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_percent": (unrealized_pnl / margin_used * 100) if margin_used > 0 else 0.0,
                        "margin_used": margin_used,
                        "liquidation_price": live_pos.liquidation_price,
                        "is_active": True,
                        "created_at": (db_row or {}).get("created_at") or now,
                        "updated_at": now,
                        "data_source": data_source,
                    })

            total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            total_val = sum(p.get("position_value", 0) for p in positions)
            return {
                "allocation_id": allocation_id,
                "positions": positions,
                "total_positions": len(positions),
                "total_unrealized_pnl": total_pnl,
                "total_position_value": total_val,
                "data_source": data_source,
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as snap_err:
            logger.warning(f"[stream] Snapshot failed for {allocation_id}: {snap_err}")
            return None

    # -----------------------------------------------------------------
    # 3. SSE generator
    # -----------------------------------------------------------------
    async def event_generator():
        stream_mgr = get_stream_manager()
        q: Optional[asyncio.Queue] = None
        try:
            # Send immediate snapshot so the UI doesn't wait for first WS tick
            snapshot = await _build_snapshot()
            if snapshot:
                yield f"event: positions\ndata: {json.dumps(snapshot)}\n\n"

            # Register with the stream manager
            q = await stream_mgr.subscribe(wallet_address, allocation_id)

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(q.get(), timeout=25)
                except asyncio.TimeoutError:
                    # Send a keepalive comment so proxies don't drop the connection
                    yield ": keepalive\n\n"
                    continue

                if event.event_type == "positions":
                    # Parse clearinghouseState into position list
                    try:
                        hl_service = allocation_service.hyperliquid_service
                        clearing = event.payload.get("clearinghouseState") or {}
                        if hl_service and clearing:
                            live_positions = hl_service.parse_positions_payload(clearing)
                            # Update positions cache so REST polling stays consistent
                            hl_service._positions_cache[wallet_address] = {
                                "positions": dict(live_positions),
                                "fetched_at": datetime.now(),
                            }
                            # Build a lightweight positions payload
                            now = datetime.now().isoformat()
                            pos_list = []
                            for coin, lp in live_positions.items():
                                if not _is_material_yuki_position(lp.position_value):
                                    continue
                                norm = allocation_service._quote_stripped_symbol(coin)
                                margin_used = float(lp.margin_used or 0)
                                unrealized_pnl = float(lp.unrealized_pnl or 0)
                                pos_list.append({
                                    "id": f"live-{allocation_id}-{norm}",
                                    "allocation_id": allocation_id,
                                    "symbol": norm,
                                    "side": lp.side,
                                    "size": float(lp.size or 0),
                                    "entry_price": float(lp.entry_price or 0),
                                    "current_price": float(lp.mark_price or lp.entry_price or 0),
                                    "leverage": float(lp.leverage or 1),
                                    "position_value": float(lp.position_value or 0),
                                    "unrealized_pnl": unrealized_pnl,
                                    "unrealized_pnl_percent": (unrealized_pnl / margin_used * 100) if margin_used > 0 else 0.0,
                                    "margin_used": margin_used,
                                    "liquidation_price": lp.liquidation_price,
                                    "is_active": True,
                                    "updated_at": now,
                                    "data_source": "hyperliquid_ws",
                                })
                            payload = {
                                "allocation_id": allocation_id,
                                "positions": pos_list,
                                "total_positions": len(pos_list),
                                "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in pos_list),
                                "total_position_value": sum(p["position_value"] for p in pos_list),
                                "data_source": "hyperliquid_ws",
                                "last_updated": now,
                            }
                            yield f"event: positions\ndata: {json.dumps(payload)}\n\n"
                    except Exception as parse_err:
                        logger.debug(f"[stream] Position parse error: {parse_err}")

                elif event.event_type == "orders":
                    try:
                        orders = event.payload.get("orders") or []
                        yield f"event: orders\ndata: {json.dumps({'allocation_id': allocation_id, 'orders': orders, 'last_updated': event.ts})}\n\n"
                    except Exception as order_err:
                        logger.debug(f"[stream] Order event error: {order_err}")

        except asyncio.CancelledError:
            pass
        except Exception as gen_err:
            logger.error(f"[stream] Generator error for {allocation_id}: {gen_err}")
            try:
                yield f"event: error\ndata: {json.dumps({'message': str(gen_err)})}\n\n"
            except Exception:
                pass
        finally:
            if q is not None:
                await stream_mgr.unsubscribe(allocation_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.post("/positions/{allocation_id}/{position_id}/close")
async def manually_close_yuki_position(
    allocation_id: str,
    position_id: str,
    request: CloseYukiPositionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Close one authenticated user's full Yuki position with a reduce-only market order."""
    try:
        result = await get_allocation_service().manually_close_yuki_position(
            user_id=user["user_id"],
            allocation_id=allocation_id,
            position_id=position_id,
            expected_side=request.expected_side,
        )
        if result.get("success"):
            return result
        raise HTTPException(
            status_code=int(result.get("status_code") or 400),
            detail=result.get("error") or "Could not close this position",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error manually closing Yuki position {position_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not close this position")


_POSITION_CHART_INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
# Candles of pre-entry context so the chart shows the setup, not just the trade.
_POSITION_CHART_CONTEXT_BARS = 150


@router.get("/positions/{allocation_id}/{symbol}/chart")
async def get_allocation_position_chart(
    allocation_id: str,
    symbol: str,
    interval: Optional[str] = Query(None, description="Candle interval override (1m/5m/15m/1h/4h/1d)"),
    user: Dict[str, Any] = Depends(get_current_user),
):
    """Return exchange candles and trade levels for one active Yuki position."""
    try:
        allocation_service = get_allocation_service()
        allocation_status = await allocation_service.get_allocation_status(
            user_id=user["user_id"],
            allocation_id=allocation_id,
        )
        if "error" in allocation_status:
            raise HTTPException(status_code=404, detail=allocation_status["error"])

        requested_interval = str(interval or "").lower().strip()
        if requested_interval and requested_interval not in _POSITION_CHART_INTERVAL_SECONDS:
            requested_interval = ""

        normalized_symbol = allocation_service._quote_stripped_symbol(symbol)
        cache_key = f"{allocation_id}:{normalized_symbol}:{requested_interval or 'auto'}"
        cached = _position_chart_cache.get(cache_key)
        if cached and (datetime.now(timezone.utc) - cached[0]).total_seconds() < _POSITION_CHART_CACHE_TTL_SECONDS:
            return cached[1]

        from kata.services.agent_database_service import get_agent_db_service
        db_service = get_agent_db_service()
        db_positions = await db_service.get_agent_positions(allocation_id)
        db_position = next((
            row for row in db_positions
            if allocation_service._quote_stripped_symbol(row.get("symbol")) == normalized_symbol
        ), None)
        if not db_position:
            raise HTTPException(status_code=404, detail="Active position not found")

        trades = await db_service.get_agent_trades(allocation_id, limit=200)
        trade = next((row for row in trades if row.get("id") == db_position.get("trade_id")), None)
        if trade is None:
            expected_side = "buy" if str(db_position.get("side")).lower() == "long" else "sell"
            trade = next((
                row for row in trades
                if allocation_service._quote_stripped_symbol(row.get("symbol")) == normalized_symbol
                and str(row.get("side") or "").lower() == expected_side
                and row.get("status") in ("filled", "partially_filled")
            ), None)

        def parse_time(value: Any) -> Optional[datetime]:
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                return None

        now = datetime.now(timezone.utc)
        start_at = (
            parse_time((trade or {}).get("filled_at"))
            or parse_time((trade or {}).get("created_at"))
            or parse_time(db_position.get("created_at"))
            or now
        )
        if start_at >= now:
            start_at = now - timedelta(minutes=5)
        age_seconds = max((now - start_at).total_seconds(), 60)
        if requested_interval:
            chart_interval = requested_interval
        else:
            chart_interval = "1h"
        interval_seconds = _POSITION_CHART_INTERVAL_SECONDS[chart_interval]
        # Pre-entry context, capped so position age + context stays within the
        # venue's per-request candle limits.
        position_bars = int(age_seconds / interval_seconds) + 2
        context_bars = max(0, min(_POSITION_CHART_CONTEXT_BARS, 1000 - position_bars - 5))
        fetch_start_at = start_at - timedelta(seconds=context_bars * interval_seconds)

        entry_price = float(db_position.get("entry_price") or (trade or {}).get("entry_price") or 0)
        leverage = float(db_position.get("leverage") or (trade or {}).get("leverage") or 1)
        direction = str(db_position.get("side") or "long").upper()
        points = [{
            "timestamp": start_at.isoformat(),
            "price": entry_price,
            "pnl_pct": 0.0,
            "kind": "entry",
        }] if entry_price > 0 else []
        market_source = "hyperliquid_candles"
        source_warning = None
        hyperliquid_candles: List[Dict[str, Any]] = []
        info_client = getattr(allocation_service.hyperliquid_service, "info_client", None)
        if info_client:
            try:
                hyperliquid_candles = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: info_client.candles_snapshot(
                            normalized_symbol,
                            chart_interval,
                            int(fetch_start_at.timestamp() * 1000),
                            int(now.timestamp() * 1000),
                        ),
                    ),
                    timeout=1.0,
                ) or []
            except Exception as hyperliquid_error:
                source_warning = f"Hyperliquid candles unavailable: {hyperliquid_error}"
                logger.warning(
                    "Hyperliquid candles failed/timed out for %s/%s; using fast Binance signal engine: %s",
                    allocation_id,
                    normalized_symbol,
                    hyperliquid_error,
                )
        else:
            source_warning = "Hyperliquid candle client is unavailable"

        for candle in hyperliquid_candles:
            close_price = float(candle.get("c") or 0)
            point = _position_chart_candle_point(
                float(candle.get("t") or 0),
                float(candle.get("o") or close_price),
                float(candle.get("h") or close_price),
                float(candle.get("l") or close_price),
                close_price,
                float(candle.get("v") or 0),
                entry_price,
                direction,
                leverage,
            )
            if point:
                points.append(point)

        # Hyperliquid occasionally rate-limits candle snapshots. Keep the chart
        # useful by switching to Binance futures OHLC while retaining the Yuki
        # entry, stop and target levels from Hyperliquid/our execution ledger.
        if len(points) < 3:
            source_warning = source_warning or "Hyperliquid returned too few candles"
            binance = None
            try:
                from kata.services.binance_service import create_binance_service
                binance = create_binance_service()
                limit = min(max(position_bars + context_bars + 12, 50), 1000)
                binance_candles = None
                for candidate in (
                    f"{normalized_symbol}/USDT:USDT",
                    f"{normalized_symbol}/USDT",
                    f"{normalized_symbol}USDT",
                ):
                    binance_candles = await binance._make_rate_limited_request(
                        binance.exchange.fetch_ohlcv,
                        2,
                        candidate,
                        chart_interval,
                        since=int(fetch_start_at.timestamp() * 1000),
                        limit=limit,
                    )
                    if binance_candles and len(binance_candles) >= 2:
                        break

                if binance_candles and len(binance_candles) >= 2:
                    points = points[:1]
                    for candle in binance_candles:
                        point = _position_chart_candle_point(
                            float(candle[0]),
                            float(candle[1]),
                            float(candle[2]),
                            float(candle[3]),
                            float(candle[4]),
                            float(candle[5] or 0),
                            entry_price,
                            direction,
                            leverage,
                        )
                        if point:
                            points.append(point)
                    market_source = "binance_candles_fallback"
            except Exception as binance_error:
                logger.warning(
                    "Binance candle fallback failed for %s/%s: %s",
                    allocation_id,
                    normalized_symbol,
                    binance_error,
                )
                source_warning = f"{source_warning or 'Primary candles unavailable'}; Binance fallback unavailable: {binance_error}"
            finally:
                if binance:
                    await binance.close()

        position_metadata = db_position.get("position_metadata") or {}
        trade_metadata = (trade or {}).get("trade_metadata") or {}
        protective_orders = position_metadata.get("protective_orders") or trade_metadata.get("protective_orders") or {}
        stop_order = protective_orders.get("stop_loss") or {}

        stop_loss = (
            position_metadata.get("stop_loss")
            or trade_metadata.get("stop_loss")
            or stop_order.get("trigger_px")
            or stop_order.get("limit_px")
        )
        if not stop_loss and entry_price > 0:
            stop_loss = entry_price * 0.95 if direction.lower() == "long" else entry_price * 1.05

        target_1 = (
            position_metadata.get("target_1")
            or trade_metadata.get("target_1")
            or trade_metadata.get("take_profit")
        )
        target_2 = (
            position_metadata.get("target_2")
            or trade_metadata.get("target_2")
        )

        signal_reasoning = (trade or {}).get("signal_reasoning") or (db_position or {}).get("signal_reasoning")
        try:
            db_client = db_service.get_client()
            signal_res = db_client.from_("platform_signals") \
                .select("signal_id, target_1, target_2, stop_loss, ai_confidence_breakdown, summary") \
                .ilike("token_symbol", normalized_symbol) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            if signal_res.data:
                matched_signal = signal_res.data[0]
                if not target_1:
                    target_1 = matched_signal.get("target_1")
                if not target_2:
                    target_2 = matched_signal.get("target_2")
                if not stop_loss:
                    stop_loss = matched_signal.get("stop_loss")
                if not signal_reasoning:
                    signal_reasoning = matched_signal.get("summary")
        except Exception as signal_lookup_err:
            logger.debug("Signal level enrichment skipped for %s: %s", normalized_symbol, signal_lookup_err)

        levels = {
            "entry": entry_price,
            "stop_loss": float(stop_loss) if stop_loss and float(stop_loss) > 0 else None,
            "target_1": float(target_1) if target_1 and float(target_1) > 0 else None,
            "target_2": float(target_2) if target_2 and float(target_2) > 0 else None,
        }

        response = {
            "success": True,
            "allocation_id": allocation_id,
            "symbol": normalized_symbol,
            "direction": direction,
            "interval": chart_interval,
            "available_intervals": list(_POSITION_CHART_INTERVAL_SECONDS.keys()),
            "source": market_source,
            "source_warning": source_warning if len(points) < 3 else None,
            "levels": levels,
            "points": points,
            "signal_id": trade_metadata.get("signal_id"),
            "signal_reasoning": signal_reasoning,
            "timestamp": now.isoformat(),
        }
        _cache_position_chart(cache_key, response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position chart for {allocation_id}/{symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load position chart: {e}")


@router.get("/performance/{allocation_id}")
async def get_allocation_performance(
    allocation_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get detailed performance metrics for an agent allocation.
    """
    try:
        allocation_service = get_allocation_service()

        # Verify user owns this allocation
        allocation_status = await allocation_service.get_allocation_status(
            user_id=user["user_id"],
            allocation_id=allocation_id
        )

        if "error" in allocation_status:
            raise HTTPException(status_code=404, detail=allocation_status["error"])

        # Get detailed performance metrics
        from kata.services.yuki_position_monitor import create_yuki_position_monitor
        from kata.services.hyperliquid_service import create_hyperliquid_service
        from kata.config.settings import settings
        
        # Create position monitor for performance calculation
        hyperliquid_service = create_hyperliquid_service(
            privy_app_id=settings.PRIVY_APP_ID,
            privy_app_secret=settings.PRIVY_APP_SECRET,
            testnet=settings.HYPERLIQUID_TESTNET
        )
        
        position_monitor = create_yuki_position_monitor(hyperliquid_service)
        performance_metrics = await position_monitor.get_allocation_performance(allocation_id)

        return {
            "allocation_id": allocation_id,
            "basic_metrics": allocation_status,
            "detailed_metrics": performance_metrics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting allocation performance: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
