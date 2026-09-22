"""
Credit Management API endpoints for Flow AI Trading Platform.

Provides REST API endpoints for managing user credits, transactions, and balance.
"""

import logging
import hmac
import hashlib
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from kata.services.credit_service import CreditService, get_credit_service
from kata.models.user import CreditTransaction, CreditBalance

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/credits", tags=["Credits"])


# Pydantic models for API requests/responses
class CreditUsageRequest(BaseModel):
    """Request to use credits for a service."""
    service: str = Field(..., description="Service name (trading_scanner, token_analysis)")
    description: str = Field(default="AI analysis", description="Description of credit usage")
    credits_needed: int = Field(default=1, ge=1, le=10, description="Number of credits needed")


class CreditPurchaseRequest(BaseModel):
    """Request to purchase credits."""
    amount: int = Field(..., ge=1, le=100, description="Number of credits to purchase")
    payment_method: str = Field(default="wallet", description="Payment method")


class CreditUsageResponse(BaseModel):
    """Response for credit usage."""
    success: bool
    credits_remaining: int
    message: str
    error: Optional[str] = None
    credits_needed: Optional[int] = None


class CreditBalanceResponse(BaseModel):
    """Response for credit balance."""
    user_id: str
    credits: int
    last_updated: datetime


class HelioWebhookPayload(BaseModel):
    """Helio webhook payload for successful payments."""
    id: str
    amount: float
    currency: str
    status: str
    customer_email: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: str


@router.get("/balance/{user_id}", response_model=CreditBalanceResponse)
async def get_credit_balance(
    user_id: str,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Get current credit balance for a user."""
    try:
        logger.info(f"Getting credit balance for user {user_id}")
        
        balance = await credit_service.get_balance(user_id)
        
        return CreditBalanceResponse(
            user_id=balance.user_id,
            credits=balance.credits,
            last_updated=balance.last_updated
        )
        
    except Exception as e:
        logger.error(f"Error getting credit balance for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get credit balance: {str(e)}"
        )


@router.post("/use/{user_id}", response_model=CreditUsageResponse)
async def use_credits(
    user_id: str,
    request: CreditUsageRequest,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Use credits for a service."""
    try:
        logger.info(f"User {user_id} attempting to use {request.credits_needed} credits for {request.service}")
        
        # Validate service
        valid_services = ["trading_scanner", "token_analysis"]
        if request.service not in valid_services:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid service. Must be one of: {valid_services}"
            )
        
        result = await credit_service.deduct_credits(
            user_id=user_id,
            amount=request.credits_needed,
            service_used=request.service,
            description=request.description
        )
        
        if result["success"]:
            return CreditUsageResponse(
                success=True,
                credits_remaining=result["credits_remaining"],
                message=f"Successfully used {request.credits_needed} credits for {request.service}"
            )
        else:
            return CreditUsageResponse(
                success=False,
                credits_remaining=result["credits_remaining"],
                message=result["message"],
                error=result["error"],
                credits_needed=result.get("credits_needed")
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error using credits for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to use credits: {str(e)}"
        )


@router.post("/purchase/{user_id}")
async def purchase_credits(
    user_id: str,
    request: CreditPurchaseRequest,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Purchase credits for a user."""
    try:
        logger.info(f"User {user_id} purchasing {request.amount} credits")
        
        # In a real app, you'd integrate with payment processing here
        # For now, we'll just add the credits
        
        result = await credit_service.add_credits(
            user_id=user_id,
            amount=request.amount,
            transaction_type="purchase",
            description=f"Purchased {request.amount} credits via {request.payment_method}"
        )
        
        return {
            "success": True,
            "message": f"Successfully purchased {request.amount} credits",
            "credits_added": result["credits_added"],
            "credits_total": result["credits_total"],
            "transaction_id": result["transaction_id"]
        }
        
    except Exception as e:
        logger.error(f"Error purchasing credits for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purchase credits: {str(e)}"
        )


@router.get("/transactions/{user_id}")
async def get_transaction_history(
    user_id: str,
    limit: int = 50,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Get credit transaction history for a user."""
    try:
        logger.info(f"Getting transaction history for user {user_id}")
        
        transactions = await credit_service.get_transaction_history(user_id, limit)
        
        # Convert to dict for JSON response
        transaction_list = []
        for t in transactions:
            transaction_list.append({
                "user_id": t.user_id,
                "amount": t.amount,
                "transaction_type": t.transaction_type,
                "description": t.description,
                "service_used": t.service_used,
                "created_at": t.created_at.isoformat()
            })
        
        return {
            "user_id": user_id,
            "transactions": transaction_list,
            "total": len(transaction_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting transaction history for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get transaction history: {str(e)}"
        )


@router.post("/check-affordability/{user_id}")
async def check_service_affordability(
    user_id: str,
    service: str,
    credits_needed: int = 1,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Check if user can afford a service."""
    try:
        can_afford = await credit_service.can_afford_service(user_id, credits_needed)
        balance = await credit_service.get_balance(user_id)
        
        return {
            "user_id": user_id,
            "service": service,
            "credits_needed": credits_needed,
            "credits_available": balance.credits,
            "can_afford": can_afford,
            "credits_short": max(0, credits_needed - balance.credits) if not can_afford else 0
        }
        
    except Exception as e:
        logger.error(f"Error checking affordability for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check affordability: {str(e)}"
        )


@router.post("/initialize/{user_id}")
async def initialize_user_credits(
    user_id: str,
    initial_credits: int = 3,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Initialize credits for a new user."""
    try:
        await credit_service.initialize_user_credits(user_id, initial_credits)
        balance = await credit_service.get_balance(user_id)
        
        return {
            "message": f"Initialized {initial_credits} credits for user {user_id}",
            "credits_total": balance.credits
        }
        
    except Exception as e:
        logger.error(f"Error initializing credits for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize credits: {str(e)}"
        )


def verify_helio_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    """Verify Helio webhook signature for security."""
    try:
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload_body,
            hashlib.sha256
        ).hexdigest()
        
        # Helio typically sends signature as "sha256=<hash>"
        if signature.startswith('sha256='):
            signature = signature[7:]
        
        return hmac.compare_digest(expected_signature, signature)
    except Exception as e:
        logger.error(f"Error verifying Helio signature: {e}")
        return False


def _parse_helio_json(value: Any) -> Dict[str, Any]:
    """Decode Helio's sometimes double-stringified additionalJSON field."""
    parsed = value
    for _ in range(3):
        if not isinstance(parsed, str):
            break
        try:
            parsed = json.loads(parsed)
        except (TypeError, json.JSONDecodeError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _helio_paylink_credit_mapping() -> Dict[str, int]:
    packages = (
        (50, ("HELIO_PAYLINK_STARTER", "HELIO_PAYLINK_KOHAI"), "689e0da46405f5151e3cf162"),
        (100, ("HELIO_PAYLINK_PRO", "HELIO_PAYLINK_SENPAI", "HELIO_PAYLINK_ID"), "689e10b26014cbbf11051a65"),
        (300, ("HELIO_PAYLINK_PREMIUM", "HELIO_PAYLINK_SENSEI"), "689e10bb16dbe6ba38435dd9"),
    )
    mapping: Dict[str, int] = {}
    for credits, env_names, default_id in packages:
        mapping[default_id] = credits
        for env_name in env_names:
            configured_id = os.getenv(env_name)
            if configured_id:
                mapping[configured_id] = credits
    return mapping


def normalize_helio_payment(payload_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize current MoonPay Commerce and legacy Helio webhook payloads."""
    event_type = str(payload_data.get("event") or payload_data.get("type") or "")
    transaction = payload_data.get("transactionObject")

    if not isinstance(transaction, dict):
        transaction = _parse_helio_json(payload_data.get("transaction"))

    if transaction:
        meta = transaction.get("meta") if isinstance(transaction.get("meta"), dict) else {}
        customer_details = meta.get("customerDetails") if isinstance(meta.get("customerDetails"), dict) else {}
        additional_json = _parse_helio_json(
            customer_details.get("additionalJSON")
            or customer_details.get("additionalJson")
            or transaction.get("additionalJSON")
        )
        paylink_id = str(
            transaction.get("paylinkId")
            or transaction.get("paymentRequestId")
            or ""
        )
        credits = _helio_paylink_credit_mapping().get(paylink_id)
        return {
            "event": event_type,
            "payment_id": str(transaction.get("id") or meta.get("id") or ""),
            "paylink_id": paylink_id,
            "status": str(meta.get("transactionStatus") or transaction.get("transactionStatus") or ""),
            "user_id": additional_json.get("user_id") or additional_json.get("wallet_address"),
            "credits": credits,
            "amount": transaction.get("totalAmountAsUSD") or meta.get("totalAmountAsUSD"),
            "currency": (meta.get("currency") or {}).get("symbol") if isinstance(meta.get("currency"), dict) else None,
            "format": "current",
        }

    payment_data = payload_data.get("data") if isinstance(payload_data.get("data"), dict) else {}
    metadata = payment_data.get("metadata") if isinstance(payment_data.get("metadata"), dict) else {}
    amount = float(payment_data.get("amount", 0) or 0)
    legacy_credit_mapping = {
        30.0: 50,
        50.0: 100,
        100.0: 300,
        10.0: 15,
        20.0: 30,
    }
    return {
        "event": event_type,
        "payment_id": str(payment_data.get("id") or payload_data.get("id") or ""),
        "paylink_id": str(payment_data.get("paylinkId") or ""),
        "status": str(payment_data.get("status") or ""),
        "user_id": metadata.get("user_id") or metadata.get("wallet_address"),
        "credits": legacy_credit_mapping.get(amount, int(amount * 3) if amount > 0 else None),
        "amount": amount,
        "currency": payment_data.get("currency", "USD"),
        "format": "legacy",
    }


@router.post("/helio-webhook")
async def helio_webhook(
    request: Request,
    credit_service: CreditService = Depends(get_credit_service)
):
    """
    Secure Helio webhook handler for payment processing.
    
    Endpoint: /api/credits/helio-webhook
    """
    try:
        # Get webhook secret from environment
        webhook_secret = os.getenv('HELIO_WEBHOOK_SECRET')
        if not webhook_secret:
            logger.error("HELIO_WEBHOOK_SECRET not configured")
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
        
        # Get raw request body and signature header
        payload_body = await request.body()
        signature = (
            request.headers.get('x-signature')
            or request.headers.get('x-helio-signature')
            or request.headers.get('helio-signature')
        )
        
        if not signature:
            logger.error("Missing Helio webhook signature")
            raise HTTPException(status_code=400, detail="Missing webhook signature")
        
        # Verify webhook signature for security
        if not verify_helio_signature(payload_body, signature, webhook_secret):
            logger.error("Invalid Helio webhook signature")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse JSON payload
        try:
            payload_data = json.loads(payload_body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        logger.info(f"Received verified Helio webhook: {payload_data.get('id', 'unknown')}")
        
        payment = normalize_helio_payment(payload_data)
        event_type = payment["event"]
        valid_events = {"CREATED", "payment.completed", "payment.success"}
        if event_type not in valid_events:
            logger.info("Ignoring Helio event type: %s", event_type)
            return {"status": "ignored", "reason": f"Event type {event_type} not processed"}

        payment_id = payment["payment_id"]
        if not payment_id:
            raise HTTPException(status_code=422, detail="No payment ID found in webhook")

        payment_status = str(payment["status"]).upper()
        if payment["format"] == "current" and payment_status != "SUCCESS":
            logger.info("Ignoring incomplete Helio payment %s with status %s", payment_id, payment_status)
            return {"status": "ignored", "reason": f"Payment status {payment_status} not processed"}

        user_id = payment["user_id"]
        if not user_id:
            logger.error("No user_id found in additionalJSON for payment %s", payment_id)
            raise HTTPException(status_code=422, detail="No user_id found in payment additionalJSON")

        credits_to_add = payment["credits"]
        if not credits_to_add:
            logger.error("Unknown Helio pay link %s for payment %s", payment["paylink_id"], payment_id)
            raise HTTPException(status_code=422, detail="Unknown credit package pay link")

        amount = payment["amount"]
        currency = payment["currency"] or "USD"
        
        # Add credits to user account
        result = await credit_service.add_credits(
            user_id=user_id,
            amount=credits_to_add,
            transaction_type="helio_purchase",
            description=f"Helio payment {payment_id} - {credits_to_add} credits",
            external_reference=f"helio:{payment_id}",
        )

        if not result.get("success"):
            raise HTTPException(status_code=500, detail="Failed to add purchased credits")
        
        credited_amount = int(result.get("credits_added", credits_to_add))
        if result.get("duplicate"):
            logger.info("Ignored duplicate Helio payment %s for user %s", payment_id, user_id)
        else:
            logger.info("Added %s credits to user %s from Helio payment %s", credited_amount, user_id, payment_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "credits_added": credited_amount,
            "duplicate": bool(result.get("duplicate")),
            "payment_id": payment_id,
            "amount": amount,
            "currency": currency,
            "transaction_id": result.get("transaction_id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing Helio webhook: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process webhook: {str(e)}"
        )


@router.get("/debug/users")
async def debug_users(
    credit_service: CreditService = Depends(get_credit_service)
):
    """Debug endpoint to see users with wallet addresses (temporary for debugging)."""
    try:
        # Get the database service directly for debugging
        from kata.services.credit_database_service import get_credit_db_service
        db_service = get_credit_db_service()
        
        # Query all users with external wallets
        response = db_service.db.from_("users").select("id, external_wallet, credits, updated_at").execute()
        
        return {
            "status": "success",
            "users": response.data,
            "total": len(response.data) if response.data else 0
        }
        
    except Exception as e:
        logger.error(f"Debug users failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Debug failed: {str(e)}"
        )


@router.post("/debug/fix-wallet-case/{wallet_address}")
async def fix_wallet_case(
    wallet_address: str,
    credit_service: CreditService = Depends(get_credit_service)
):
    """Fix wallet address case mismatch (temporary debug endpoint)."""
    try:
        from kata.services.credit_database_service import get_credit_db_service
        db_service = get_credit_db_service()
        
        # Find all users with this wallet address (case insensitive)
        all_users = db_service.db.from_("users").select("id, external_wallet, credits, updated_at").execute()
        
        matching_users = [
            user for user in all_users.data 
            if user.get('external_wallet', '').lower() == wallet_address.lower()
        ]
        
        if len(matching_users) > 1:
            # Find the user with the highest credits
            best_user = max(matching_users, key=lambda x: x.get('credits', 0))
            
            # Delete the others and update the best one to use the correct wallet address
            for user in matching_users:
                if user['id'] != best_user['id']:
                    # Delete duplicate
                    db_service.db.from_("users").delete().eq("id", user['id']).execute()
                    logger.info(f"Deleted duplicate user {user['id']} with {user['credits']} credits")
            
            # Update the best user's wallet address to the requested format
            db_service.db.from_("users").update({
                "external_wallet": wallet_address,
                "updated_at": datetime.now().isoformat()
            }).eq("id", best_user['id']).execute()
            
            return {
                "status": "success",
                "message": f"Fixed wallet case, kept user with {best_user['credits']} credits",
                "kept_user_id": best_user['id'],
                "deleted_users": len(matching_users) - 1
            }
        else:
            return {
                "status": "no_action",
                "message": "No duplicate users found"
            }
        
    except Exception as e:
        logger.error(f"Fix wallet case failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fix failed: {str(e)}"
        )


@router.post("/refund/{user_id}")
async def refund_credits(
    user_id: str,
    request: dict,
    credit_service: CreditService = Depends(get_credit_service)
):
    """
    Refund credits to a user when a service fails after credit deduction.
    """
    try:
        amount = request.get("credits_to_refund", 0)
        service = request.get("service", "unknown")
        description = request.get("description", f"Refund for failed {service}")

        if amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="Refund amount must be greater than 0"
            )

        logger.info(f"Processing credit refund for user {user_id}: {amount} credits for {service}")

        result = await credit_service.add_credits(
            user_id=user_id,
            amount=amount,
            transaction_type="refund",
            description=description
        )

        if result["success"]:
            logger.info(f"✅ Successfully refunded {amount} credits to user {user_id}")
            return {
                "success": True,
                "message": f"Refunded {amount} credits successfully",
                "credits_refunded": amount,
                "new_balance": result["credits_total"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.error(f"❌ Credit refund failed for user {user_id}: {result}")
            raise HTTPException(
                status_code=500,
                detail="Credit refund failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Credit refund error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Credit refund failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for credit service."""
    try:
        credit_service = get_credit_service()

        return {
            "status": "healthy",
            "service": "credits",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Credit service health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Credit service unavailable"
        )


# Initialize services on startup
def initialize_credit_services():
    """Initialize credit services."""
    try:
        get_credit_service()
        logger.info("Credit services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize credit services: {e}")
        raise
