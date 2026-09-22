"""
Application Shutdown Handler for Flow AI Trading Platform

Ensures graceful shutdown and proper resource cleanup for all services.
"""

import asyncio
import logging
import signal
import sys
from typing import List, Callable, Awaitable

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    Handles graceful application shutdown with proper resource cleanup.
    
    Coordinates shutdown of all services and ensures no resources are leaked.
    """
    
    def __init__(self):
        self._shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_requested = False
        self._signals_registered = False
        
    def register_cleanup_callback(self, callback: Callable[[], Awaitable[None]]):
        """Register a cleanup callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)
        logger.info(f"Registered shutdown callback: {callback.__name__}")
    
    def register_signals(self):
        """Register signal handlers for graceful shutdown."""
        if self._signals_registered:
            return
            
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Docker/systemd stop
        
        self._signals_registered = True
        logger.info("Signal handlers registered for graceful shutdown")
    
    async def shutdown(self):
        """Execute graceful shutdown sequence."""
        if not self._shutdown_callbacks:
            logger.info("No shutdown callbacks registered")
            return
        
        logger.info("🛑 Starting graceful shutdown sequence...")
        
        failed_cleanups = []
        
        for i, callback in enumerate(self._shutdown_callbacks, 1):
            try:
                callback_name = getattr(callback, '__name__', f'callback_{i}')
                logger.info(f"🔄 Running cleanup {i}/{len(self._shutdown_callbacks)}: {callback_name}")
                
                # Set timeout for each cleanup operation
                await asyncio.wait_for(callback(), timeout=30.0)
                logger.info(f"✅ Cleanup completed: {callback_name}")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Cleanup timeout: {callback_name}")
                failed_cleanups.append(callback_name)
            except Exception as e:
                logger.error(f"❌ Cleanup failed: {callback_name} - {e}")
                failed_cleanups.append(callback_name)
        
        if failed_cleanups:
            logger.warning(f"⚠️ {len(failed_cleanups)} cleanups failed: {failed_cleanups}")
        else:
            logger.info("✅ All cleanup operations completed successfully")
        
        logger.info("🛑 Graceful shutdown sequence completed")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)


# Global shutdown handler
_shutdown_handler: ShutdownHandler = None

def get_shutdown_handler() -> ShutdownHandler:
    """Get the global shutdown handler instance."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = ShutdownHandler()
    return _shutdown_handler


async def register_service_cleanup():
    """Register cleanup for all HTTP sessions and services."""
    from .session_manager import cleanup_all_sessions
    
    handler = get_shutdown_handler()
    handler.register_cleanup_callback(cleanup_all_sessions)
    handler.register_signals()
    
    logger.info("🔧 Service cleanup callbacks registered")


# Convenience functions for common shutdown patterns
async def shutdown_with_cleanup():
    """Complete shutdown sequence with all registered cleanups."""
    handler = get_shutdown_handler()
    await handler.shutdown()


def setup_graceful_shutdown():
    """Set up graceful shutdown handling for the application."""
    handler = get_shutdown_handler()
    handler.register_signals()
    
    # Register session cleanup
    import asyncio
    
    async def setup_cleanup():
        await register_service_cleanup()
    
    # Schedule the setup for when event loop is running
    loop = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(setup_cleanup())
        else:
            loop.run_until_complete(setup_cleanup())
    except RuntimeError:
        # Event loop not running - will be set up when loop starts
        pass