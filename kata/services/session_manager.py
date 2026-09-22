"""
Global HTTP Session Manager for Flow AI Trading Platform

Provides centralized HTTP session management with connection pooling,
resource cleanup, and proper lifecycle management for all services.
"""

import asyncio
import logging
import weakref
from typing import Dict, Optional, Set
import aiohttp
import atexit
import ssl
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Global HTTP session manager with automatic cleanup and resource management.
    
    Features:
    - Connection pooling and reuse
    - Automatic resource cleanup
    - Memory leak prevention
    - Graceful shutdown handling
    """
    
    _instance: Optional['SessionManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._session_refs: Set[weakref.ref] = set()
        self._closed = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Register cleanup on exit
        atexit.register(self._sync_cleanup)
        
        logger.info("🔧 SessionManager initialized with global resource management")
    
    async def get_session(
        self, 
        service_name: str, 
        timeout: Optional[aiohttp.ClientTimeout] = None,
        connector_kwargs: Optional[Dict] = None
    ) -> aiohttp.ClientSession:
        """
        Get or create an HTTP session for a service.
        
        Args:
            service_name: Unique identifier for the service
            timeout: Custom timeout configuration
            connector_kwargs: Additional connector configuration
            
        Returns:
            Configured aiohttp.ClientSession
        """
        if self._closed:
            raise RuntimeError("SessionManager has been closed")
            
        async with self._lock:
            if service_name in self._sessions:
                session = self._sessions[service_name]
                if not session.closed:
                    return session
                else:
                    # Remove closed session
                    del self._sessions[service_name]
            
            # Create new session
            session = await self._create_session(service_name, timeout, connector_kwargs)
            self._sessions[service_name] = session
            
            # Add weak reference for cleanup tracking
            ref = weakref.ref(session, self._session_cleanup_callback)
            self._session_refs.add(ref)
            
            return session
    
    async def _create_session(
        self, 
        service_name: str,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        connector_kwargs: Optional[Dict] = None
    ) -> aiohttp.ClientSession:
        """Create a new configured HTTP session."""
        
        # Default timeout
        if timeout is None:
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=20
            )
        
        # SSL configuration
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Default connector configuration (memory optimized for 512MB RAM constraints)
        default_connector_kwargs = {
            'ssl': ssl_context,
            'limit': 30,  # Reduced from 100 to save memory footprint
            'limit_per_host': 10,  # Reduced from 20 to save memory footprint
            'ttl_dns_cache': 120,  # Reduced DNS cache TTL
            'use_dns_cache': True,
            'keepalive_timeout': 15, # Shorter keepalive to drop idle connections
            'enable_cleanup_closed': True
        }
        
        if connector_kwargs:
            default_connector_kwargs.update(connector_kwargs)
        
        connector = aiohttp.TCPConnector(**default_connector_kwargs)
        
        session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            connector_owner=True
        )
        
        logger.info(f"🌐 Created HTTP session for {service_name} with connection pooling")
        return session
    
    def _session_cleanup_callback(self, ref: weakref.ref):
        """Callback for cleaning up session references."""
        self._session_refs.discard(ref)
    
    async def close_session(self, service_name: str):
        """Close a specific service session."""
        async with self._lock:
            if service_name in self._sessions:
                session = self._sessions[service_name]
                if not session.closed:
                    await session.close()
                    logger.info(f"🔒 Closed HTTP session for {service_name}")
                del self._sessions[service_name]
    
    async def close_all_sessions(self):
        """Close all active sessions."""
        if self._closed:
            return
            
        async with self._lock:
            self._closed = True
            
            for service_name, session in list(self._sessions.items()):
                if not session.closed:
                    try:
                        await session.close()
                        logger.info(f"🔒 Closed HTTP session for {service_name}")
                    except Exception as e:
                        logger.error(f"Error closing session for {service_name}: {e}")
            
            self._sessions.clear()
            self._session_refs.clear()
            
            logger.info("🔒 All HTTP sessions closed successfully")
    
    def _sync_cleanup(self):
        """Synchronous cleanup for atexit."""
        try:
            if not self._closed and self._sessions:
                logger.warning("🧹 Emergency session cleanup on exit")
                # Note: We can't run async code in atexit, so we just log
                logger.info(f"Sessions still open: {list(self._sessions.keys())}")
        except:
            pass  # Don't raise exceptions in atexit
    
    async def get_session_stats(self) -> Dict[str, any]:
        """Get statistics about active sessions."""
        async with self._lock:
            return {
                'total_sessions': len(self._sessions),
                'active_sessions': [name for name, session in self._sessions.items() if not session.closed],
                'closed': self._closed,
                'session_names': list(self._sessions.keys()),
                'timestamp': datetime.now().isoformat()
            }
    
    def start_cleanup_task(self):
        """Start periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up closed sessions."""
        while not self._closed:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                async with self._lock:
                    closed_sessions = []
                    for service_name, session in self._sessions.items():
                        if session.closed:
                            closed_sessions.append(service_name)
                    
                    for service_name in closed_sessions:
                        del self._sessions[service_name]
                        logger.debug(f"🧹 Cleaned up closed session for {service_name}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
        
        logger.info("🧹 Session cleanup task stopped")


# Global instance
_session_manager: Optional[SessionManager] = None

async def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        _session_manager.start_cleanup_task()
    return _session_manager

async def cleanup_all_sessions():
    """Cleanup all sessions on application shutdown."""
    global _session_manager
    if _session_manager is not None:
        await _session_manager.close_all_sessions()
        _session_manager = None


class SessionMixin:
    """
    Mixin class for services that need HTTP session management.
    
    Provides standardized session management with automatic cleanup.
    """
    
    def __init__(self, service_name: str, **session_kwargs):
        self.service_name = service_name
        self.session_kwargs = session_kwargs
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_manager: Optional[SessionManager] = None
    
    async def start_session(self):
        """Start HTTP session using global session manager."""
        if not self.session or self.session.closed:
            self._session_manager = await get_session_manager()
            self.session = await self._session_manager.get_session(
                self.service_name, 
                **self.session_kwargs
            )
            logger.info(f"🌐 Started session for {self.service_name}")
    
    async def close_session(self):
        """Close HTTP session."""
        if self._session_manager and self.service_name:
            await self._session_manager.close_session(self.service_name)
            self.session = None
            logger.info(f"🔒 Closed session for {self.service_name}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()