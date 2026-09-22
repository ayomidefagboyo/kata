"""
Authentication router for user login and wallet-based authentication.
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from kata.config.database import get_db_client, get_service_client
from kata.models.user import UserLogin, UserLoginResponse, User, UserCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    login_data: UserLogin,
    db: Client = Depends(get_db_client)
) -> UserLoginResponse:
    """
    Authenticate user with wallet address.
    Creates new user if doesn't exist.
    """
    try:
        wallet_address = login_data.wallet_address.lower()
        
        # Try to get existing user
        response = db.from_("users").select("*").eq("external_wallet", wallet_address).execute()
        
        if response.data:
            # User exists
            user_data = response.data[0]
            user = User(**user_data)
            logger.info(f"Existing user logged in: {wallet_address}")
        else:
            # Create new user
            new_user_data = UserCreate(external_wallet=wallet_address)
            
            insert_response = db.from_("users").insert(new_user_data.model_dump()).execute()
            
            if not insert_response.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user"
                )
            
            user = User(**insert_response.data[0])
            logger.info(f"New user created and logged in: {wallet_address}")
        
        return UserLoginResponse(
            user=user,
            message="Login successful"
        )
        
    except Exception as e:
        logger.error(f"Login error for {login_data.wallet_address}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


@router.get("/user/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    db: Client = Depends(get_db_client)
) -> User:
    """Get user by ID."""
    try:
        response = db.from_("users").select("*").eq("id", user_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return User(**response.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


@router.get("/user/wallet/{wallet_address}", response_model=User)
async def get_user_by_wallet(
    wallet_address: str,
    db: Client = Depends(get_db_client)
) -> User:
    """Get user by wallet address."""
    try:
        wallet_address = wallet_address.lower()
        response = db.from_("users").select("*").eq("external_wallet", wallet_address).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return User(**response.data[0])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by wallet {wallet_address}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}"
        )


@router.put("/user/{user_id}/platform-wallet")
async def update_platform_wallet(
    user_id: str,
    platform_wallet: str,
    db: Client = Depends(get_db_client)
) -> Dict[str, Any]:
    """Update user's platform wallet address."""
    try:
        platform_wallet = platform_wallet.lower()
        
        # Validate wallet format
        if not platform_wallet.startswith('0x') or len(platform_wallet) != 42:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid wallet address format"
            )
        
        response = db.from_("users").update({
            "platform_wallet": platform_wallet
        }).eq("id", user_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Updated platform wallet for user {user_id}: {platform_wallet}")
        
        return {
            "success": True,
            "message": "Platform wallet updated successfully",
            "platform_wallet": platform_wallet
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating platform wallet for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update platform wallet: {str(e)}"
        )


@router.post("/logout")
async def logout_user() -> Dict[str, str]:
    """Logout user (placeholder for future session management)."""
    return {"message": "Logout successful"}