"""
Credit Management Service for Flow AI Trading Platform.

Handles user credit balance, deduction, and transaction tracking with database persistence.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from kata.models.user import CreditTransaction, CreditBalance
from kata.services.credit_database_service import CreditDatabaseService, get_credit_db_service

logger = logging.getLogger(__name__)


class CreditService:
    """Service for managing user credits with database persistence and caching."""

    def __init__(self, db_service: CreditDatabaseService = None):
        self.db_service = db_service or get_credit_db_service()
        # Cache for credit balances (5 minute expiry for performance optimization)
        self._balance_cache = {}
        self._cache_expiry_minutes = 5
        logger.info("Credit service initialized with database persistence and caching")

    async def get_balance(self, user_id: str) -> CreditBalance:
        """Get current credit balance for a user with 5-minute caching."""
        # Check cache first
        cache_key = f"balance_{user_id}"
        now = datetime.now()

        if cache_key in self._balance_cache:
            cached_data, cached_time = self._balance_cache[cache_key]
            if now - cached_time < timedelta(minutes=self._cache_expiry_minutes):
                logger.debug(f"Using cached credit balance for {user_id}")
                return cached_data

        # Cache miss or expired, fetch from database
        balance = await self.db_service.get_balance(user_id)

        # Cache the result
        self._balance_cache[cache_key] = (balance, now)

        # Clean old cache entries (simple cleanup)
        self._cleanup_cache()

        return balance

    def _cleanup_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = []
        for key, (_, cached_time) in self._balance_cache.items():
            if now - cached_time >= timedelta(minutes=self._cache_expiry_minutes * 2):  # Double expiry for cleanup
                expired_keys.append(key)

        for key in expired_keys:
            del self._balance_cache[key]

    def invalidate_cache(self, user_id: str):
        """Invalidate cache for a specific user (call after credit changes)."""
        cache_key = f"balance_{user_id}"
        if cache_key in self._balance_cache:
            del self._balance_cache[cache_key]
    
    async def deduct_credits(
        self,
        user_id: str,
        amount: int,
        service_used: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Deduct credits from user's balance.

        Returns success status and remaining balance.
        """
        result = await self.db_service.deduct_credits(user_id, amount, service_used, description)
        # Invalidate cache after credit change
        self.invalidate_cache(user_id)
        return result
    
    async def add_credits(
        self,
        user_id: str,
        amount: int,
        transaction_type: str = "purchase",
        description: str = "Credit purchase",
        external_reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add credits to user's balance."""
        result = await self.db_service.add_credits(
            user_id,
            amount,
            transaction_type,
            description,
            external_reference=external_reference,
        )
        # Invalidate cache after credit change
        self.invalidate_cache(user_id)
        return result
    
    async def get_transaction_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> list[Dict[str, Any]]:
        """Get transaction history for a user."""
        return await self.db_service.get_transaction_history(user_id, limit)
    
    async def can_afford_service(self, user_id: str, service_cost: int) -> bool:
        """Check if user has enough credits for a service."""
        return await self.db_service.can_afford_service(user_id, service_cost)
    
    async def initialize_user_credits(self, user_id: str, initial_credits: int = 3):
        """Initialize credits for a new user."""
        return await self.db_service.initialize_user_credits(user_id, initial_credits)


# Global service instance
_credit_service: Optional[CreditService] = None


def get_credit_service() -> CreditService:
    """Get or create credit service instance."""
    global _credit_service
    
    if _credit_service is None:
        _credit_service = CreditService()
    
    return _credit_service


def create_credit_service() -> CreditService:
    """Create a new credit service instance."""
    return CreditService()
