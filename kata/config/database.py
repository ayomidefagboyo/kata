"""
Database configuration and Supabase client setup.
"""
import logging
from typing import Optional
from supabase import create_client, Client
from .settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages Supabase database connections."""
    
    def __init__(self):
        self._client: Optional[Client] = None
        self._service_client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get Supabase client with anon key (for user operations)."""
        if self._client is None:
            self._client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_ANON_KEY
            )
            logger.info("Supabase client initialized")
        return self._client
    
    @property
    def service_client(self) -> Client:
        """Get Supabase client with service key (for admin operations)."""
        if self._service_client is None:
            self._service_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY
            )
            logger.info("Supabase service client initialized")
        return self._service_client
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            # Simple query to test connection
            response = self.client.from_("users").select("count", count="exact").limit(0).execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def reset_connections(self):
        """Reset database connections."""
        self._client = None
        self._service_client = None
        logger.info("Database connections reset")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_client() -> Client:
    """Get database client for dependency injection."""
    return db_manager.client


def get_service_client() -> Client:
    """Get service database client for dependency injection."""
    return db_manager.service_client