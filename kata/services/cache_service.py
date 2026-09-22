"""
Simple In-Memory Cache Service for Token Discovery

Provides basic caching without Redis dependency.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class CacheService:
    """Simple in-memory cache service for token discovery."""
    
    def __init__(self):
        self.is_connected = True  # Always connected for in-memory cache
        
        # In-memory cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache TTLs (in seconds)
        self.TTL = {
            'trending_tokens': 300,      # 5 minutes
            'token_analysis': 1800,      # 30 minutes
            'market_data': 60,           # 1 minute
            'user_preferences': 3600,    # 1 hour
            'discovery_queue': 600,      # 10 minutes
        }
        
        # Cache keys
        self.KEYS = {
            'trending_tokens': 'trending_tokens',
            'token_analysis': 'token_analysis:{token_id}',
            'market_data': 'market_data:{token_id}',
            'user_preferences': 'user_prefs:{user_id}',
            'discovery_queue': 'discovery_queue',
            'last_update': 'last_update',
            'cache_stats': 'cache_stats',
        }
    
    async def connect(self):
        """Connect to cache (no-op for in-memory cache)."""
        self.is_connected = True
        logger.info("✅ In-memory cache initialized successfully")
    
    async def disconnect(self):
        """Disconnect from cache (no-op for in-memory cache)."""
        self.is_connected = False
        logger.info("In-memory cache disconnected")
    
    def _is_expired(self, cache_entry: Dict[str, Any], ttl_seconds: int) -> bool:
        """Check if a cache entry is expired."""
        if 'timestamp' not in cache_entry:
            return True
        
        entry_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - entry_time > timedelta(seconds=ttl_seconds)
    
    def _set_cache(self, key: str, value: Any, ttl_seconds: int):
        """Set a cache entry with timestamp."""
        self._cache[key] = {
            'data': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_cache(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """Get a cache entry if not expired."""
        if key not in self._cache:
            return None
        
        cache_entry = self._cache[key]
        if self._is_expired(cache_entry, ttl_seconds):
            del self._cache[key]
            return None
        
        return cache_entry['data']
    
    async def get_trending_tokens(self, limit: int = 50, blockchain: str = None) -> Optional[List[Dict]]:
        """Get cached trending tokens."""
        try:
            cache_key = f"{self.KEYS['trending_tokens']}:{limit}:{blockchain or 'all'}"
            cached_data = self._get_cache(cache_key, self.TTL['trending_tokens'])
            
            if cached_data:
                logger.info(f"📦 Cache hit for trending tokens (limit={limit}, blockchain={blockchain})")
                return cached_data
            
            logger.info(f"❌ Cache miss for trending tokens (limit={limit}, blockchain={blockchain})")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set_trending_tokens(self, tokens: List[Dict], limit: int = 50, blockchain: str = None):
        """Cache trending tokens."""
        try:
            cache_key = f"{self.KEYS['trending_tokens']}:{limit}:{blockchain or 'all'}"
            self._set_cache(cache_key, tokens, self.TTL['trending_tokens'])
            
            # Update last update timestamp
            self._set_cache(self.KEYS['last_update'], datetime.now().isoformat(), self.TTL['trending_tokens'])
            
            logger.info(f"💾 Cached trending tokens (limit={limit}, blockchain={blockchain})")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def get_token_analysis(self, token_id: str) -> Optional[Dict]:
        """Get cached token analysis."""
        try:
            cache_key = self.KEYS['token_analysis'].format(token_id=token_id)
            return self._get_cache(cache_key, self.TTL['token_analysis'])
            
        except Exception as e:
            logger.error(f"Cache get error for token analysis: {e}")
            return None
    
    async def set_token_analysis(self, token_id: str, analysis: Dict):
        """Cache token analysis."""
        try:
            cache_key = self.KEYS['token_analysis'].format(token_id=token_id)
            self._set_cache(cache_key, analysis, self.TTL['token_analysis'])
            
        except Exception as e:
            logger.error(f"Cache set error for token analysis: {e}")
    
    async def get_market_data(self, token_id: str) -> Optional[Dict]:
        """Get cached market data."""
        try:
            cache_key = self.KEYS['market_data'].format(token_id=token_id)
            return self._get_cache(cache_key, self.TTL['market_data'])
            
        except Exception as e:
            logger.error(f"Cache get error for market data: {e}")
            return None
    
    async def set_market_data(self, token_id: str, market_data: Dict):
        """Cache market data."""
        try:
            cache_key = self.KEYS['market_data'].format(token_id=token_id)
            self._set_cache(cache_key, market_data, self.TTL['market_data'])
            
        except Exception as e:
            logger.error(f"Cache set error for market data: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Clean up expired entries before calculating stats
            await self._cleanup_expired()
            
            return {
                "status": "connected",
                "connected": True,
                "cache_type": "in_memory",
                "total_keys": len(self._cache),
                "memory_usage": "N/A (in-memory)",
                "hit_rate": "N/A (no tracking)",
                "notes": "Simple in-memory cache without Redis"
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"status": "error", "connected": False, "error": str(e)}
    
    async def _cleanup_expired(self):
        """Clean up expired cache entries."""
        try:
            expired_keys = []
            now = datetime.now()
            
            for key, cache_entry in self._cache.items():
                if 'timestamp' in cache_entry:
                    entry_time = datetime.fromisoformat(cache_entry['timestamp'])
                    # Use a default TTL of 1 hour for cleanup
                    if now - entry_time > timedelta(seconds=3600):
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# Global cache service instance
cache_service = CacheService()

async def get_cache_service() -> CacheService:
    """Get cache service dependency."""
    if not cache_service.is_connected:
        await cache_service.connect()
    return cache_service