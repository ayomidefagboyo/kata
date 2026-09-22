"""
Credit Database Service for Flow AI Trading Platform.

Handles credit operations with Supabase database persistence.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from supabase import Client
from uuid import uuid4

from kata.config.database import get_service_client
from kata.models.user import CreditTransaction, CreditBalance

logger = logging.getLogger(__name__)


class CreditDatabaseService:
    """Service for managing credits with database persistence."""
    
    def __init__(self, db_client: Client = None):
        self.db = db_client or get_service_client()
        logger.info("Credit database service initialized")
    
    def _find_user_by_identifier(self, user_id: str, select_fields: str = "id, credits, updated_at"):
        """
        Helper method to find user by various identifiers.
        Supports external_wallet, platform_wallet, and UUID lookup.
        """
        if user_id.startswith('0x'):
            # It's a wallet address, try multiple lookup strategies
            logger.info(f"🔍 Searching by wallet address: {user_id}")
            
            # Try external_wallet first (normal wallet users)
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("external_wallet", user_id).execute()
            if response.data:
                return response, 'external_wallet'
            
            # Try lowercase external_wallet 
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("external_wallet", user_id.lower()).execute()
            if response.data:
                return response, 'external_wallet'
            
            # Try platform_wallet (email users with platform wallet as identifier)
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("platform_wallet", user_id).execute()
            if response.data:
                return response, 'platform_wallet'
            
            # Try lowercase platform_wallet
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("platform_wallet", user_id.lower()).execute()
            if response.data:
                return response, 'platform_wallet'
                
        elif user_id.startswith('did:privy:'):
            # Privy DID format - search in existing wallet columns where DID might be stored
            logger.info(f"🔍 Searching by Privy DID: {user_id}")
            
            # Search by external_wallet (DID might be stored there)
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("external_wallet", user_id).execute()
            if response.data:
                return response, 'external_wallet'
                
            # Search by platform_wallet (DID might be stored there) 
            response = self.db.from_("users").select(select_fields + ", external_wallet, platform_wallet").eq("platform_wallet", user_id).execute()
            if response.data:
                return response, 'platform_wallet'
                
        else:
            # Assume it's a UUID, search by id
            logger.info(f"🔍 Searching by user id (UUID): {user_id}")
            try:
                response = self.db.from_("users").select(select_fields).eq("id", user_id).execute()
                if response.data:
                    return response, 'id'
            except Exception as e:
                logger.warning(f"UUID search failed for {user_id}: {e}")
        
        return None, None
    
    def _update_user_credits(self, user_id: str, lookup_type: str, new_balance: int, user_data: dict = None):
        """
        Helper method to update user credits based on how they were found.
        """
        update_data = {
            "credits": new_balance,
            "updated_at": datetime.now().isoformat()
        }

        if user_data and user_data.get("id"):
            return self.db.from_("users").update(update_data).eq("id", user_data["id"]).execute()

        if lookup_type == 'external_wallet':
            return self.db.from_("users").update(update_data).eq("external_wallet", user_id).execute()
        elif lookup_type == 'platform_wallet':  
            return self.db.from_("users").update(update_data).eq("platform_wallet", user_id).execute()
        elif lookup_type == 'id':
            return self.db.from_("users").update(update_data).eq("id", user_id).execute()
        else:
            raise Exception(f"Unknown lookup type: {lookup_type}")
    
    def _get_user_uuid(self, user_id: str, lookup_type: str, user_data: dict = None) -> str:
        """
        Helper method to get the actual UUID for transaction recording.
        """
        if lookup_type == 'id':
            return user_id  # It's already a UUID
        elif user_data:
            return user_data.get('id')  # Use UUID from existing query
        else:
            # Fallback: query for UUID
            if lookup_type == 'external_wallet':
                response = self.db.from_("users").select("id").eq("external_wallet", user_id).execute()
            elif lookup_type == 'platform_wallet':
                response = self.db.from_("users").select("id").eq("platform_wallet", user_id).execute()
            else:
                raise Exception(f"Cannot get UUID for lookup type: {lookup_type}")
            
            if response.data:
                return response.data[0]["id"]
            else:
                raise Exception(f"User not found for {lookup_type}: {user_id}")
    
    async def get_balance(self, user_id: str) -> CreditBalance:
        """Get current credit balance for a user from database."""
        try:
            
            # Find user using helper method
            response, lookup_type = self._find_user_by_identifier(user_id)
            
            if not response or not response.data:
                # User doesn't exist - this is a truly new user
                await self.initialize_user_credits(user_id, 3)
                return CreditBalance(
                    user_id=user_id,
                    credits=3,
                    last_updated=datetime.now()
                )
            
            user_data = response.data[0]
            
            # Handle credits field
            credits = user_data.get("credits")
            
            if credits is None:
                logger.warning(f"User {user_id} exists but has no credits field")
                credits = 0
            
            # Handle Supabase timestamp format (can have variable microsecond precision)
            timestamp_str = user_data["updated_at"]
            # Remove Z and add timezone info
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            
            # Parse with proper handling of microseconds
            try:
                last_updated = datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback: parse manually if fromisoformat fails with microseconds
                import re
                # Extract the datetime part without microseconds and timezone
                match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?([\+\-]\d{2}:\d{2}|Z?)$', timestamp_str)
                if match:
                    base_time = match.group(1)
                    microseconds = match.group(2) or '0'
                    timezone = match.group(3) or '+00:00'
                    
                    # Normalize microseconds to 6 digits
                    microseconds = microseconds.ljust(6, '0')[:6]
                    
                    # Reconstruct the timestamp
                    normalized_timestamp = f"{base_time}.{microseconds}{timezone}"
                    last_updated = datetime.fromisoformat(normalized_timestamp)
                else:
                    # Ultimate fallback
                    logger.warning(f"Could not parse timestamp {timestamp_str}, using current time")
                    last_updated = datetime.now()
            
            credit_balance = CreditBalance(
                user_id=user_id,
                credits=credits,
                last_updated=last_updated
            )
            
            logger.debug(f"✅ Returning credit balance: {credit_balance.credits} credits for user {user_id}")
            return credit_balance
            
        except Exception as e:
            logger.error(f"Error getting credit balance for user {user_id}: {e}")
            # Log the full exception for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # For database connectivity issues, return safe fallback without resetting existing users
            # This ensures the app continues to work while preserving user data
            logger.warning(f"Database issue for user {user_id}, using fallback credits")
            return CreditBalance(
                user_id=user_id,
                credits=0,  # Use 0 instead of 3 to avoid false positives for existing users
                last_updated=datetime.now()
            )
    
    async def deduct_credits(
        self, 
        user_id: str, 
        amount: int, 
        service_used: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Deduct credits from user's balance in database.
        
        Returns success status and remaining balance.
        """
        try:
            # Get current balance
            balance = await self.get_balance(user_id)
            current_credits = balance.credits
            
            if current_credits < amount:
                return {
                    "success": False,
                    "error": "insufficient_credits",
                    "message": f"Insufficient credits. Need {amount}, have {current_credits}",
                    "credits_remaining": current_credits,
                    "credits_needed": amount - current_credits
                }
            
            # Calculate new balance
            new_balance = current_credits - amount
            
            # Find user again to get lookup type for proper updating
            response, lookup_type = self._find_user_by_identifier(user_id)
            if not response or not response.data:
                raise Exception("User not found during credit deduction")
            
            user_data = response.data[0]
            
            # Update user's credits using the appropriate field
            update_response = self._update_user_credits(user_id, lookup_type, new_balance, user_data)
            
            if not update_response.data:
                raise Exception("Failed to update user credits")
            
            # Get the actual UUID for transaction recording
            actual_user_id = self._get_user_uuid(user_id, lookup_type, user_data)
            
            transaction_data = {
                "id": str(uuid4()),
                "user_id": actual_user_id,
                "amount": -amount,
                "transaction_type": "usage",
                "description": description,
                "service_used": service_used,
                "created_at": datetime.now().isoformat()
            }
            
            # Insert transaction record
            self.db.from_("credit_transactions").insert(transaction_data).execute()
            
            logger.debug(f"Deducted {amount} credits from user {user_id}. New balance: {new_balance}")
            
            return {
                "success": True,
                "credits_deducted": amount,
                "credits_remaining": new_balance,
                "transaction_id": transaction_data["id"]
            }
            
        except Exception as e:
            logger.error(f"Error deducting credits for user {user_id}: {e}")
            return {
                "success": False,
                "error": "deduction_failed",
                "message": f"Failed to deduct credits: {str(e)}",
                "credits_remaining": 0,
                "credits_needed": 0
            }
    
    async def add_credits(
        self,
        user_id: str,
        amount: int,
        transaction_type: str = "purchase",
        description: str = "Credit purchase",
        external_reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add credits to user's balance in database."""
        try:
            if external_reference:
                existing = self.db.from_("credit_transactions").select("id").eq(
                    "transaction_type", transaction_type
                ).eq("service_used", external_reference).limit(1).execute()
                if existing.data:
                    logger.info("Ignoring duplicate credit addition for %s", external_reference)
                    balance = await self.get_balance(user_id)
                    return {
                        "success": True,
                        "duplicate": True,
                        "credits_added": 0,
                        "credits_total": balance.credits,
                        "transaction_id": existing.data[0]["id"],
                    }

            # Get current balance
            balance = await self.get_balance(user_id)
            current_credits = balance.credits
            new_balance = current_credits + amount

            response, lookup_type = self._find_user_by_identifier(user_id)
            if not response or not response.data:
                raise Exception("User not found during credit addition")

            user_data = response.data[0]
            update_response = self._update_user_credits(user_id, lookup_type, new_balance, user_data)
            
            if not update_response.data:
                raise Exception("Failed to update user credits")
            
            actual_user_id = self._get_user_uuid(user_id, lookup_type, user_data)
            
            transaction_data = {
                "id": str(uuid4()),
                "user_id": actual_user_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "description": description,
                "service_used": external_reference,
                "created_at": datetime.now().isoformat()
            }
            
            # Insert transaction record
            self.db.from_("credit_transactions").insert(transaction_data).execute()
            
            logger.info(f"Added {amount} credits to user {user_id}. New balance: {new_balance}")
            
            return {
                "success": True,
                "credits_added": amount,
                "credits_total": new_balance,
                "transaction_id": transaction_data["id"]
            }
            
        except Exception as e:
            logger.error(f"Error adding credits for user {user_id}: {e}")
            return {
                "success": False,
                "error": "addition_failed",
                "message": f"Failed to add credits: {str(e)}",
                "credits_total": 0
            }
    
    async def get_transaction_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get transaction history for a user from database."""
        try:
            response = self.db.from_("credit_transactions").select("*").eq(
                "user_id", user_id
            ).order("created_at", desc=True).limit(limit).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error getting transaction history for user {user_id}: {e}")
            return []
    
    async def can_afford_service(self, user_id: str, service_cost: int) -> bool:
        """Check if user has enough credits for a service."""
        try:
            balance = await self.get_balance(user_id)
            return balance.credits >= service_cost
        except Exception as e:
            logger.error(f"Error checking affordability for user {user_id}: {e}")
            return False
    
    async def initialize_user_credits(self, user_id: str, initial_credits: int = 3):
        """Initialize credits for a user (usually called during user creation)."""
        try:
            # Check if user exists using comprehensive lookup
            response, lookup_type = self._find_user_by_identifier(user_id)
            
            if response and response.data:
                # User exists, only update if they have null/zero credits
                user_data = response.data[0]
                current_credits = user_data.get("credits")
                actual_user_id = user_data["id"]  # Get the actual UUID for transactions
                
                if current_credits is None or current_credits == 0:
                    # Update user with initial credits using appropriate field
                    self._update_user_credits(user_id, lookup_type, initial_credits)
                    
                    # Record welcome bonus transaction using the actual UUID
                    transaction_data = {
                        "id": str(uuid4()),
                        "user_id": actual_user_id,
                        "amount": initial_credits,
                        "transaction_type": "bonus",
                        "description": "Welcome bonus credits",
                        "service_used": None,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    self.db.from_("credit_transactions").insert(transaction_data).execute()
                    
                    logger.info(f"Initialized {initial_credits} credits for user {user_id}")
            else:
                # User doesn't exist, create them with initial credits
                if user_id.startswith('0x'):
                    # Create new user with wallet address
                    new_user_data = {
                        "id": str(uuid4()),
                        "external_wallet": user_id,
                        "platform_wallet": user_id,  # Use same address for now
                        "selected_agent": "yuki",
                        "risk_tolerance": 5,
                        "is_active": True,
                        "credits": initial_credits,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Insert new user
                    insert_response = self.db.from_("users").insert(new_user_data).execute()
                    
                    if insert_response.data:
                        # Record welcome bonus transaction
                        transaction_data = {
                            "id": str(uuid4()),
                            "user_id": new_user_data["id"],
                            "amount": initial_credits,
                            "transaction_type": "bonus",
                            "description": "Welcome bonus credits",
                            "service_used": None,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        self.db.from_("credit_transactions").insert(transaction_data).execute()
                        
                        logger.info(f"Created new user {user_id} with {initial_credits} credits")
                    else:
                        logger.error(f"Failed to create new user {user_id}")
                elif user_id.startswith('did:privy:'):
                    # Create new user with Privy DID
                    new_user_data = {
                        "id": str(uuid4()),
                        "external_wallet": user_id,  # Store DID in external_wallet for now
                        "platform_wallet": None,
                        "selected_agent": "yuki",
                        "risk_tolerance": 5,
                        "is_active": True,
                        "credits": initial_credits,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Insert new user
                    insert_response = self.db.from_("users").insert(new_user_data).execute()
                    
                    if insert_response.data:
                        # Record welcome bonus transaction
                        transaction_data = {
                            "id": str(uuid4()),
                            "user_id": new_user_data["id"],
                            "amount": initial_credits,
                            "transaction_type": "bonus",
                            "description": "Welcome bonus credits",
                            "service_used": None,
                            "created_at": datetime.now().isoformat()
                        }
                        
                        self.db.from_("credit_transactions").insert(transaction_data).execute()
                        
                        logger.info(f"Created new Privy DID user {user_id} with {initial_credits} credits")
                    else:
                        logger.error(f"Failed to create new Privy DID user {user_id}")
                else:
                    logger.warning(f"User {user_id} not found and unknown format, cannot initialize credits")
                
        except Exception as e:
            logger.error(f"Error initializing credits for user {user_id}: {e}")


# Global service instance
_credit_db_service: Optional[CreditDatabaseService] = None


def get_credit_db_service() -> CreditDatabaseService:
    """Get or create credit database service instance."""
    global _credit_db_service
    
    if _credit_db_service is None:
        _credit_db_service = CreditDatabaseService()
    
    return _credit_db_service


def create_credit_db_service() -> CreditDatabaseService:
    """Create a new credit database service instance."""
    return CreditDatabaseService()
