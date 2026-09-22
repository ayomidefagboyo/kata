"""
Wallet API Router for withdrawal and platform wallet operations
"""

import asyncio
import logging
import httpx
import json
import base64
import os
from typing import Dict, Any, Optional, Union, List, Literal
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from privy.lib.authorization_signatures import get_authorization_signature

from ...services.privy_auth_service import get_privy_auth_service
from ...services.delegation_service import get_delegation_service, DelegationPolicy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wallet", tags=["wallet"])

class WithdrawalRequest(BaseModel):
    """Withdrawal request model"""
    user_id: str = Field(..., description="User ID from Privy")
    token_symbol: str = Field(..., description="Token symbol (e.g., USDC)")
    amount: float = Field(..., gt=0, description="Amount to withdraw")
    destination_address: str = Field(..., description="Destination wallet address")
    source: str = Field(..., description="Source of funds (platform or blockchain)")
    token_address: Optional[str] = Field(None, description="Token contract address (for blockchain tokens)")
    prepare_transaction: Optional[bool] = Field(True, description="Return transaction data for the user-paid Base USDC flow")

class WithdrawalResponse(BaseModel):
    """Withdrawal response model"""
    status: str
    transaction_hash: Optional[str] = None
    message: str
    amount: float
    token_symbol: str
    destination_address: str
    timestamp: datetime
    estimated_confirmation_time: Optional[str] = None

class TransactionPrepareResponse(BaseModel):
    """Transaction preparation response model"""
    to: str = Field(..., description="Contract address to send transaction to")
    data: str = Field(..., description="Encoded transaction data")
    value: str = Field(default="0", description="ETH value to send (usually 0 for token transfers)")
    gas_estimate: Optional[str] = Field(None, description="Estimated gas limit")
    message: str = Field(..., description="Human readable description")
    chain_id: int = Field(default=8453, description="Base mainnet chain ID")
    network: str = Field(default="Base", description="Canonical Floww balance network")
    fee_asset: str = Field(default="USDC", description="Asset used to pay the network fee")
    fee_payer: str = Field(default="user", description="Floww does not sponsor withdrawal fees")

class DelegationPolicyRequest(BaseModel):
    """User-defined trading policy limits"""
    max_daily_volume_usd: float = Field(default=1000.0, ge=0, description="Maximum daily trading volume in USD")
    max_position_size_percent: float = Field(default=10.0, ge=0, le=100, description="Maximum position size as percentage of portfolio")
    max_leverage: int = Field(default=5, ge=1, le=40, description="Maximum leverage allowed")
    allowed_chains: List[str] = Field(default=['base', 'arbitrum'], description="List of allowed blockchain networks")
    require_stop_loss: bool = Field(default=True, description="Whether stop loss is required for all trades")
    max_trades_per_day: int = Field(default=50, ge=1, description="Maximum number of trades per day")

class DelegationRequest(BaseModel):
    """Wallet delegation request model"""
    user_id: str = Field(..., description="User ID from Privy")
    wallet_id: str = Field(..., description="Embedded wallet ID to delegate")
    wallet_address: str = Field(..., description="Embedded wallet address")
    chain_type: Literal["ethereum", "solana"] = Field(
        default="ethereum",
        description="Privy wallet chain family",
    )
    policy_limits: DelegationPolicyRequest = Field(..., description="User-defined trading policy limits")

class DelegationResponse(BaseModel):
    """Wallet delegation response model"""
    status: str
    message: str
    wallet_id: str
    delegated: bool
    timestamp: datetime


def _delegation_failure_status(error: str) -> int:
    """Map known delegation failures to client-actionable status codes."""
    error_lower = error.lower()

    if "missing privy credentials" in error_lower:
        return 503
    if "signer approval" in error_lower or "signer was not found" in error_lower:
        return 409
    if "address does not match" in error_lower or "wallet not found" in error_lower:
        return 409
    if "privy wallet verification failed" in error_lower:
        return 502

    return 500


def _is_delegation_schema_error(error: str) -> bool:
    """Detect likely Supabase/PostgREST schema drift for delegation tables."""
    error_lower = error.lower()
    schema_markers = (
        "user_wallet_delegations",
        "delegation_status",
        "delegated_wallet_address",
        "delegation_last_checked_at",
    )

    return (
        any(marker in error_lower for marker in schema_markers)
        and any(
            phrase in error_lower
            for phrase in (
                "does not exist",
                "schema cache",
                "could not find",
                "relation",
                "column",
            )
        )
    )

async def _require_authenticated_user(authorization: Optional[str]) -> Dict[str, Any]:
    """Verify the Privy access token and return the authenticated user data."""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header. Please include 'Bearer <access_token>'"
        )

    user_access_token = authorization.replace('Bearer ', '')
    if not user_access_token:
        raise HTTPException(
            status_code=401,
            detail="Missing access token in authorization header"
        )

    user_data = await _verify_user_access_token(user_access_token)
    if not user_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token"
        )

    return {
        **user_data,
        "access_token": user_access_token
    }

async def _get_delegation_status_payload(
    user_id: str,
    account_wallet_address: Optional[str] = None,
    chain_type: str = "ethereum",
) -> Dict[str, Any]:
    delegation_service = get_delegation_service()
    delegation = await delegation_service.get_user_delegation_and_refresh_summary(
        user_id,
        account_wallet_address=account_wallet_address,
        chain_type=chain_type,
    )
    server_configured = delegation_service.is_configured_for_live_delegation()

    if delegation:
        return {
            "has_delegation": True,
            "is_active": delegation.is_active,
            "wallet_id": delegation.wallet_id,
            "wallet_address": delegation.wallet_address,
            "chain_type": delegation.chain_type,
            "policy_limits": delegation.policy_limits.to_dict(),
            "delegated_at": delegation.delegated_at.isoformat(),
            "revoked_at": delegation.revoked_at.isoformat() if delegation.revoked_at else None,
            "server_configured_for_delegation": server_configured
        }

    return {
        "has_delegation": False,
        "is_active": False,
        "chain_type": chain_type,
        "server_configured_for_delegation": server_configured
    }

@router.post("/delegate", response_model=DelegationResponse)
async def delegate_wallet(
    request: DelegationRequest,
    authorization: Optional[str] = Header(None)
) -> DelegationResponse:
    """
    Delegate user's embedded wallet for server-side transaction signing.

    This endpoint implements Privy's "Authorization Key as Signer" pattern:
    - User remains the owner of their wallet
    - Our authorization key is added as a signer with limited permissions
    - User can revoke delegation anytime
    - Once delegated, wallet can be used by any AI agent on the platform
    - Policy limits apply to all agent trading activities
    """
    try:
        logger.info(f"Processing delegation request for wallet: {request.wallet_id}")

        user_data = await _require_authenticated_user(authorization)
        authenticated_user_id = user_data["user_id"]

        if request.user_id != authenticated_user_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot delegate wallet for another user"
            )

        logger.info(f"Access token validated for user {authenticated_user_id}")

        # Verify the wallet belongs to the user
        user_wallet_accounts = await _get_user_wallet_accounts(authenticated_user_id)
        user_wallet_ids = [account["id"] for account in user_wallet_accounts]
        logger.info(f"🔍 Wallet validation - User {authenticated_user_id}")
        logger.info(f"   Requested wallet_id: {request.wallet_id}")
        logger.info(f"   User's wallet_ids: {user_wallet_ids}")

        if request.wallet_id not in user_wallet_ids:
            logger.error(f"❌ Wallet validation failed: {request.wallet_id} not in {user_wallet_ids}")
            raise HTTPException(
                status_code=403,
                detail="Wallet does not belong to the authenticated user"
            )

        requested_account = next(
            account for account in user_wallet_accounts
            if account["id"] == request.wallet_id
        )
        if requested_account["chain_type"] != request.chain_type:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Wallet is a {requested_account['chain_type']} wallet, "
                    f"not a {request.chain_type} wallet"
                ),
            )

        logger.info(f"✅ Wallet validation passed: {request.wallet_id} belongs to user {authenticated_user_id}")

        # Get delegation service
        delegation_service = get_delegation_service()

        # Convert policy request to policy object
        policy_limits = DelegationPolicy(
            max_daily_volume_usd=request.policy_limits.max_daily_volume_usd,
            max_position_size_percent=request.policy_limits.max_position_size_percent,
            max_leverage=request.policy_limits.max_leverage,
            allowed_chains=request.policy_limits.allowed_chains,
            require_stop_loss=request.policy_limits.require_stop_loss,
            max_trades_per_day=request.policy_limits.max_trades_per_day
        )

        # Check if user already has delegation
        existing_delegation = await delegation_service.get_user_delegation_and_refresh_summary(
            authenticated_user_id,
            account_wallet_address=user_data.get("wallet_address"),
            chain_type=request.chain_type,
        )
        if existing_delegation and existing_delegation.is_active:
            same_wallet = (
                existing_delegation.wallet_id == request.wallet_id
                and existing_delegation.wallet_address.lower() == request.wallet_address.lower()
            )
            if same_wallet and existing_delegation.policy_limits.covers(policy_limits):
                logger.info(f"User {authenticated_user_id} already has active delegation")
                from ...services.agent_allocation_service import get_agent_allocation_service

                asyncio.create_task(
                    get_agent_allocation_service().trigger_ryu_signal_check_for_user(
                        authenticated_user_id,
                        reason=f"{request.chain_type}_delegation_confirmed",
                    )
                )
                return DelegationResponse(
                    status="success",
                    message="Wallet delegation already active",
                    wallet_id=request.wallet_id,
                    delegated=True,
                    timestamp=datetime.now()
                )

            if same_wallet:
                await delegation_service.store_delegation(
                    user_id=authenticated_user_id,
                    wallet_id=request.wallet_id,
                    wallet_address=request.wallet_address,
                    chain_type=request.chain_type,
                    policy_limits=policy_limits,
                    delegation_id=existing_delegation.delegation_id,
                    account_wallet_address=user_data.get("wallet_address"),
                )
                logger.info(
                    "Wallet delegation policy refreshed for user %s on %s wallet %s",
                    authenticated_user_id,
                    request.chain_type,
                    request.wallet_id,
                )
                from ...services.agent_allocation_service import get_agent_allocation_service

                asyncio.create_task(
                    get_agent_allocation_service().trigger_ryu_signal_check_for_user(
                        authenticated_user_id,
                        reason=f"{request.chain_type}_delegation_refreshed",
                    )
                )
                return DelegationResponse(
                    status="success",
                    message="Wallet delegation already active; trading limits refreshed",
                    wallet_id=request.wallet_id,
                    delegated=True,
                    timestamp=datetime.now()
                )

            logger.info(
                "User %s already has an active %s delegation on another wallet",
                authenticated_user_id,
                request.chain_type,
            )
            return DelegationResponse(
                status="success",
                message="Wallet delegation already active",
                wallet_id=existing_delegation.wallet_id,
                delegated=True,
                timestamp=datetime.now()
            )

        if not delegation_service.is_configured_for_live_delegation():
            logger.error("Wallet delegation requested but Privy delegation credentials are missing")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Wallet delegation is not configured on the backend. "
                    "Set PRIVY_AUTHORIZATION_KEY_ID and PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY."
                )
            )

        # The browser performs user consent via Privy's signer SDK. The backend
        # only verifies the signer exists before storing local trading policy.
        delegation_result = await _verify_authorization_key_as_signer(
            request.wallet_id,
            request.wallet_address,
            request.chain_type,
        )

        if delegation_result["success"]:
            # Store delegation in our database
            try:
                await delegation_service.store_delegation(
                    user_id=authenticated_user_id,
                    wallet_id=request.wallet_id,
                    wallet_address=request.wallet_address,
                    chain_type=request.chain_type,
                    policy_limits=policy_limits,
                    delegation_id=delegation_result.get("delegation_id"),
                    account_wallet_address=user_data.get("wallet_address"),
                )
            except Exception as db_error:
                error_text = str(db_error)
                logger.error(f"Delegation signer verified but database storage failed: {error_text}")
                if _is_delegation_schema_error(error_text):
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Delegation signer was approved, but the delegation database migration "
                            "is not applied. Run backend/migrations/add_user_delegations.sql and "
                            "backend/migrations/add_user_delegation_summary.sql, then try again."
                        )
                    )

                raise HTTPException(
                    status_code=500,
                    detail=f"Delegation signer was approved, but storing delegation failed: {error_text}"
                )

            from ...services.agent_allocation_service import get_agent_allocation_service

            asyncio.create_task(
                get_agent_allocation_service().trigger_ryu_signal_check_for_user(
                    authenticated_user_id,
                    reason=f"{request.chain_type}_delegation_created",
                )
            )

            logger.info(f"Wallet delegation successful for user {authenticated_user_id}")
            return DelegationResponse(
                status="success",
                message="Your wallet has been delegated for automated trading. You remain the owner and can revoke access anytime.",
                wallet_id=request.wallet_id,
                delegated=True,
                timestamp=datetime.now()
            )
        else:
            delegation_error = delegation_result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=_delegation_failure_status(delegation_error),
                detail=f"Delegation failed: {delegation_error}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delegation error for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Delegation failed: {str(e)}"
        )

@router.get("/delegation-status")
async def get_current_user_delegation_status(
    chain_type: Literal["ethereum", "solana"] = "ethereum",
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Get delegation status for the authenticated user.
    """
    try:
        user_data = await _require_authenticated_user(authorization)
        return await _get_delegation_status_payload(
            user_data["user_id"],
            account_wallet_address=user_data.get("wallet_address"),
            chain_type=chain_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user delegation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/delegation-status/{user_id}")
async def get_delegation_status(
    user_id: str,
    chain_type: Literal["ethereum", "solana"] = "ethereum",
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Get delegation status for a user.

    Kept for compatibility, but the path user_id must match the authenticated
    Privy user.
    """
    try:
        user_data = await _require_authenticated_user(authorization)
        authenticated_user_id = user_data["user_id"]
        if user_id != authenticated_user_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot query delegation status for another user"
            )

        return await _get_delegation_status_payload(
            authenticated_user_id,
            account_wallet_address=user_data.get("wallet_address"),
            chain_type=chain_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting delegation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/revoke-delegation")
async def revoke_delegation(
    user_id: str,
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Revoke user's wallet delegation.
    """
    try:
        user_data = await _require_authenticated_user(authorization)
        authenticated_user_id = user_data["user_id"]
        if user_id != authenticated_user_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot revoke delegation for another user"
            )

        # Get delegation service
        delegation_service = get_delegation_service()
        success = await delegation_service.revoke_delegation(authenticated_user_id)

        if success:
            return {
                "status": "success",
                "message": "Delegation revoked successfully",
                "revoked_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="No active delegation found to revoke")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking delegation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/withdraw")
async def withdraw_funds(
    request: WithdrawalRequest,
    authorization: Optional[str] = Header(None)
) -> Union[WithdrawalResponse, TransactionPrepareResponse]:
    """
    Withdraw funds from platform wallet using Privy's session signer approach

    This endpoint handles withdrawals from platform USDC balance using:
    1. Privy wallet_id (not wallet_address)
    2. Session signer with authorization key
    3. Server-side transaction delegation
    """
    try:
        logger.info(f"Processing withdrawal request: {request.user_id} - {request.amount} {request.token_symbol}")

        # Extract and validate user access token
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid authorization header. Please include 'Bearer <access_token>'"
            )

        user_access_token = authorization.replace('Bearer ', '')
        if not user_access_token:
            raise HTTPException(
                status_code=401,
                detail="Missing access token in authorization header"
            )

        # Validate destination address
        if not request.destination_address.startswith('0x') or len(request.destination_address) != 42:
            raise HTTPException(
                status_code=400,
                detail="Invalid destination address format"
            )

        # Verify user access token and get user data
        user_data = await _verify_user_access_token(user_access_token)
        if not user_data:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired access token"
            )

        if request.user_id != user_data.get("user_id"):
            raise HTTPException(
                status_code=403,
                detail="Cannot withdraw funds for another user"
            )

        logger.info(f"Access token validated for user {request.user_id}")

        if request.source == "platform":
            # Handle platform balance withdrawal using proper Privy approach
            return await _handle_platform_withdrawal(request, user_access_token)
        else:
            raise HTTPException(
                status_code=400,
                detail="Only platform withdrawals are supported. Use 'platform' as source."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Withdrawal error for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Withdrawal failed: {str(e)}"
        )

async def _handle_platform_withdrawal(request: WithdrawalRequest, user_access_token: str) -> Union[WithdrawalResponse, TransactionPrepareResponse]:
    """Handle withdrawal from platform USDC balance using correct Privy wallet_id approach"""

    logger.info(f"Processing platform withdrawal: {request.amount} {request.token_symbol} (prepare_transaction: {request.prepare_transaction})")

    # Validate token symbol for platform withdrawals
    if request.token_symbol != "USDC":
        raise HTTPException(
            status_code=400,
            detail="Platform withdrawals only support USDC"
        )

    try:
        # Step 1: Get user's embedded wallet_id (not wallet_address)
        user_wallet_ids = await _get_user_wallet_ids(request.user_id)
        if not user_wallet_ids:
            raise HTTPException(
                status_code=400,
                detail="No embedded wallet found for user. Please create an embedded wallet first."
            )
        wallet_id = user_wallet_ids[0]

        logger.info(f"Found embedded wallet_id: {wallet_id}")

        # Step 2: Prepare USDC transfer transaction
        usdc_contract = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base

        # ERC-20 transfer function: transfer(address to, uint256 amount)
        transfer_method_id = "0xa9059cbb"
        to_address_padded = request.destination_address[2:].rjust(64, '0')
        amount_wei = int(request.amount * 1e6)  # USDC has 6 decimals
        amount_hex = hex(amount_wei)[2:].rjust(64, '0')
        transaction_data = f"{transfer_method_id}{to_address_padded}{amount_hex}"

        # Step 3: Handle prepare vs execute based on request
        transaction_params = {
            "to": usdc_contract,
            "value": "0x0",
            "data": transaction_data
        }

        if request.prepare_transaction:
            # Return transaction data for user to sign directly
            logger.info("Returning transaction data for direct user signing")
            return TransactionPrepareResponse(
                to=usdc_contract,
                value="0",
                data=transaction_data,
                message=f"Transfer {request.amount} USDC to {request.destination_address}"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Server-executed withdrawals are disabled. Request prepare_transaction=true "
                    "and submit the Base transaction with the user-paid USDC fee flow."
                )
            )

    except Exception as e:
        logger.error(f"Platform withdrawal failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Withdrawal failed: {str(e)}"
        )

async def _get_user_wallet_id(user_id: str) -> Optional[str]:
    """
    Get the embedded wallet_id for a user using Privy API.
    This is the correct approach according to Privy team guidance.
    """
    try:
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')

        if not privy_app_id or not privy_app_secret:
            logger.error("Missing Privy app credentials")
            return None

        # Use Basic Auth for server-to-server API calls
        credentials = f"{privy_app_id}:{privy_app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'privy-app-id': privy_app_id,
            'Content-Type': 'application/json'
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.privy.io/v1/users/{user_id}",
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                user_data = response.json()
                linked_accounts = user_data.get("linked_accounts", [])

                # Find embedded wallet (can be delegated or not delegated yet)
                # This legacy helper is only used by Base/EVM flows. Once a
                # user also has Ryu's Solana wallet, never let account ordering
                # select that wallet for an Ethereum transaction.
                for account in linked_accounts:
                    if (account.get("type") == "wallet" and
                        account.get("imported") == False and
                        str(
                            account.get("chain_type")
                            or account.get("chainType")
                            or "ethereum"
                        ).lower() == "ethereum"):

                        wallet_id = account.get("id")  # This is the wallet_id we need
                        wallet_address = account.get("address")

                        logger.info(f"Found embedded wallet - ID: {wallet_id}, Address: {wallet_address}")
                        return wallet_id

                logger.warning(f"No embedded wallet found for user {user_id}")
                return None
            else:
                logger.error(f"Failed to get user data: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Error getting user wallet ID: {e}")
        return None

async def _get_user_wallet_ids(user_id: str) -> list[str]:
    """
    Get all embedded wallet_ids for a user using Privy API.
    Returns a list of wallet IDs that belong to the user.
    """
    try:
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')

        if not privy_app_id or not privy_app_secret:
            logger.error("Missing Privy app credentials")
            return []

        # Use Basic Auth for server-to-server API calls
        credentials = f"{privy_app_id}:{privy_app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'privy-app-id': privy_app_id,
            'Content-Type': 'application/json'
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.privy.io/v1/users/{user_id}",
                headers=headers,
                timeout=10.0
            )

            logger.info(f"🔍 Privy API response for user {user_id}: {response.status_code}")

            if response.status_code == 200:
                user_data = response.json()
                linked_accounts = user_data.get("linked_accounts", [])
                logger.info(f"🔍 Found {len(linked_accounts)} linked accounts")

                # This helper backs Base withdrawals, so only return embedded
                # Ethereum wallets even when the account also owns a Solana
                # wallet for Ryu.
                wallet_ids = []
                for i, account in enumerate(linked_accounts):
                    logger.info(f"🔍 Account {i}: type={account.get('type')}, imported={account.get('imported')}, id={account.get('id')}")

                    if (account.get("type") == "wallet" and
                        account.get("imported") == False and
                        str(
                            account.get("chain_type")
                            or account.get("chainType")
                            or "ethereum"
                        ).lower() == "ethereum"):

                        wallet_id = account.get("id")  # This is the wallet_id we need
                        wallet_address = account.get("address")

                        logger.info(f"✅ Found embedded wallet - ID: {wallet_id}, Address: {wallet_address}")
                        wallet_ids.append(wallet_id)

                logger.info(f"🔍 Final wallet_ids list: {wallet_ids}")
                return wallet_ids
            else:
                logger.error(f"❌ Failed to get user data: {response.status_code} - {response.text}")
                return []

    except Exception as e:
        logger.error(f"Error getting user wallet IDs: {e}")
        return []


async def _get_user_wallet_accounts(user_id: str) -> List[Dict[str, str]]:
    """Return the authenticated user's embedded Privy wallets with chain type."""
    try:
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')
        if not privy_app_id or not privy_app_secret:
            logger.error("Missing Privy app credentials")
            return []

        credentials = base64.b64encode(
            f"{privy_app_id}:{privy_app_secret}".encode()
        ).decode()
        headers = {
            'Authorization': f'Basic {credentials}',
            'privy-app-id': privy_app_id,
            'Content-Type': 'application/json',
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.privy.io/v1/users/{user_id}",
                headers=headers,
                timeout=10.0,
            )
        if response.status_code != 200:
            logger.error(
                "Failed to get Privy wallet accounts: %s - %s",
                response.status_code,
                response.text,
            )
            return []

        accounts: List[Dict[str, str]] = []
        for account in response.json().get("linked_accounts", []):
            if account.get("type") != "wallet" or account.get("imported") is not False:
                continue
            wallet_id = account.get("id")
            address = account.get("address")
            chain_type = str(
                account.get("chain_type")
                or account.get("chainType")
                or "ethereum"
            ).lower()
            if wallet_id and address and chain_type in {"ethereum", "solana"}:
                accounts.append({
                    "id": wallet_id,
                    "address": address,
                    "chain_type": chain_type,
                })
        return accounts
    except Exception as e:
        logger.error(f"Error getting user wallet accounts: {e}")
        return []

def _signer_matches_authorization_key(signer: Any, auth_key_id: str) -> bool:
    if isinstance(signer, str):
        return signer == auth_key_id

    if not isinstance(signer, dict):
        return False

    for key in (
        "signer_id",
        "signerId",
        "id",
        "key_quorum_id",
        "keyQuorumId",
        "authorization_key_id",
        "authorizationKeyId",
    ):
        value = signer.get(key)
        if value == auth_key_id:
            return True

    nested_signer = signer.get("signer")
    if isinstance(nested_signer, dict):
        return _signer_matches_authorization_key(nested_signer, auth_key_id)

    return False


async def _verify_authorization_key_as_signer(
    wallet_id: str,
    wallet_address: str,
    chain_type: str = "ethereum",
) -> Dict[str, Any]:
    """
    Verify Privy already has our authorization key on the user's wallet.

    User consent and signer attachment must happen in the browser through
    Privy's SDK. The backend then reads the wallet from Privy and records the
    delegation only if the signer is actually present.
    """
    try:
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')
        auth_key_id = os.getenv('PRIVY_AUTHORIZATION_KEY_ID')
        auth_key_private = os.getenv('PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY')

        if not all([privy_app_id, privy_app_secret, auth_key_id, auth_key_private]):
            logger.error("Missing required Privy credentials for delegation verification")
            return {"success": False, "error": "Missing Privy credentials"}

        basic_auth = base64.b64encode(f"{privy_app_id}:{privy_app_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {basic_auth}",
            "privy-app-id": privy_app_id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.privy.io/v1/wallets/{wallet_id}",
                headers=headers,
                timeout=30.0
            )

        if response.status_code != 200:
            logger.error(f"Privy wallet verification failed: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"Privy wallet verification failed: {response.status_code}"
            }

        wallet = response.json()
        privy_wallet_address = wallet.get("address") or ""
        addresses_match = (
            privy_wallet_address.lower() == wallet_address.lower()
            if chain_type == "ethereum"
            else privy_wallet_address == wallet_address
        )
        if wallet_address and not addresses_match:
            return {
                "success": False,
                "error": "Privy wallet address does not match requested wallet address"
            }
        privy_chain_type = str(
            wallet.get("chain_type")
            or wallet.get("chainType")
            or chain_type
        ).lower()
        if privy_chain_type != chain_type:
            return {
                "success": False,
                "error": f"Privy wallet chain type is {privy_chain_type}, expected {chain_type}",
            }

        additional_signers = wallet.get("additional_signers") or wallet.get("additionalSigners") or []
        matching_signer = next(
            (signer for signer in additional_signers if _signer_matches_authorization_key(signer, auth_key_id)),
            None
        )

        if not matching_signer:
            logger.warning(f"Authorization signer not found on wallet {wallet_id}")
            return {
                "success": False,
                "error": "Privy signer approval was not found on this wallet"
            }

        return {
            "success": True,
            "wallet_id": wallet_id,
            "signer_added": True,
            "delegation_id": (
                matching_signer.get("signer_id")
                if isinstance(matching_signer, dict)
                else auth_key_id
            ),
            "auth_key_id": auth_key_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error verifying authorization key signer: {e}")
        return {"success": False, "error": str(e)}

async def _execute_delegated_transaction(
    wallet_id: str,
    transaction_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Execute a delegated transaction using Privy's session signer approach.
    Uses wallet_id instead of wallet_address as specified by Privy team.
    """
    try:
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')
        auth_key_id = os.getenv('PRIVY_AUTHORIZATION_KEY_ID')
        auth_key_private = os.getenv('PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY')

        if not all([privy_app_id, privy_app_secret, auth_key_id, auth_key_private]):
            logger.error("Missing required Privy credentials for delegated transaction")
            return None

        # Prepare RPC request body
        request_body = {
            "method": "eth_sendTransaction",
            "caip2": "eip155:8453",  # Base network
            "sponsor": False,  # Fees must always be paid by the user.
            "params": {
                "transaction": transaction_params
            }
        }

        # Create authorization headers using session signer approach
        headers = await _create_authorization_headers(
            method="POST",
            url=f"https://api.privy.io/v1/wallets/{wallet_id}/rpc",  # Using wallet_id!
            body=request_body,
            auth_key_id=auth_key_id,
            auth_key_private=auth_key_private,
            privy_app_id=privy_app_id
        )

        logger.info(f"Executing delegated transaction for wallet_id: {wallet_id}")
        logger.info(f"Request body: {json.dumps(request_body, indent=2)}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.privy.io/v1/wallets/{wallet_id}/rpc",  # Using wallet_id!
                headers=headers,
                json=request_body,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                transaction_hash = result.get("result")

                if transaction_hash:
                    logger.info(f"Delegated transaction successful: {transaction_hash}")
                    return {
                        "transaction_hash": transaction_hash,
                        "wallet_id": wallet_id,
                        "delegated": True,
                        "network": "base",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.error(f"No transaction hash in response: {result}")
                    return None
            else:
                logger.error(f"Privy API error: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Error executing delegated transaction: {e}")
        return None

async def _create_authorization_headers(
    method: str,
    url: str,
    body: Dict[str, Any],
    auth_key_id: str,
    auth_key_private: str,
    privy_app_id: str
) -> Dict[str, str]:
    """
    Create proper authorization headers for Privy session signer approach.
    Uses authorization key to sign requests for delegated wallet transactions.
    """
    try:
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')

        # Create Basic Auth header
        credentials = f"{privy_app_id}:{privy_app_secret}"
        basic_auth = base64.b64encode(credentials.encode()).decode()

        signature = get_authorization_signature(
            url=url,
            body=body,
            method=method.upper(),
            app_id=privy_app_id,
            private_key=auth_key_private,
        )

        # Create headers for session signer
        headers = {
            'Authorization': f'Basic {basic_auth}',
            'privy-app-id': privy_app_id,
            'privy-authorization-signature': signature,
            'Content-Type': 'application/json'
        }

        logger.info("Created authorization headers for session signer")
        return headers

    except Exception as e:
        logger.error(f"Error creating authorization headers: {e}")
        raise

@router.get("/balance/{user_id}")
async def get_wallet_balance(user_id: str) -> Dict[str, Any]:
    """Get user's platform wallet balance"""
    try:
        # TODO: Implement actual platform balance lookup
        # This would query the database for user's platform USDC balance

        return {
            "user_id": user_id,
            "platform_balance": {
                "USDC": 150.0  # Mock balance
            },
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching balance for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch balance: {str(e)}"
        )

@router.get("/status/{transaction_hash}")
async def get_withdrawal_status(transaction_hash: str) -> Dict[str, Any]:
    """Get status of a withdrawal transaction"""
    try:
        # TODO: Implement transaction status checking
        # This would query blockchain for transaction status

        return {
            "transaction_hash": transaction_hash,
            "status": "confirmed",
            "confirmations": 12,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking transaction status {transaction_hash}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check transaction status: {str(e)}"
        )

async def _verify_user_access_token(access_token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a user's access token using Privy's auth service.
    """
    if not access_token or len(access_token) < 10:
        logger.error("Invalid access token format")
        return None

    try:
        # Use Privy auth service to verify the token
        privy_auth = get_privy_auth_service()
        user_data = await privy_auth.verify_privy_token(access_token)

        if user_data:
            logger.info(f"Access token verified for user: {user_data.user_id}")
            return {
                "status": "validated",
                "user_id": user_data.user_id,
                "wallet_address": user_data.wallet_address
            }
        else:
            logger.error("Token verification failed")
            return None

    except Exception as e:
        logger.error(f"Error verifying access token: {e}")
        return None
