"""
FastAPI main application for Kata Autonomous Perpetual Futures Agent backend.
"""
import logging
import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kata.config.settings import settings
from kata.config.database import db_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected users."""
        disconnected_users = []
        for user_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)


# Global connection manager
manager = ConnectionManager()


async def initialize_all_services():
    """Initialize ALL services in background to avoid blocking startup health checks."""
    try:
        logger.info("Starting comprehensive service initialization...")
        
        # Initialize core services (fast operations)
        try:
            # dashboard.initialize_dashboard_services()  # Dashboard module doesn't exist
            token_analysis.initialize_token_analysis_services()
            credits.initialize_credit_services()
            token_discovery.initialize_token_discovery_services()
            logger.info("Core services initialized")
        except Exception as e:
            logger.error(f"Core service initialization failed: {e}")
        
        # Platform signals are handled by the background worker (platform_signals_worker.py)
        logger.info("Platform signals will be handled by background worker")
        
        # Now initialize slower async services
        await initialize_async_services()
        
        # Start background tasks
        asyncio.create_task(background_task_runner())
        
        logger.info("All services initialization complete")
        
    except Exception as e:
        logger.error(f"Error in comprehensive service initialization: {e}")


async def initialize_async_services():
    """Initialize async services in background to avoid blocking startup."""
    try:
        logger.info("Starting async service initialization...")
        
        # Test database connection (non-blocking for startup)
        try:
            is_healthy = await db_manager.health_check()
            if not is_healthy:
                logger.error("Database health check failed!")
            else:
                logger.info("Database connection healthy")
        except Exception as e:
            logger.error(f"Database health check error: {e}")
        
        # Queue service removed - no longer needed
        logger.info("Queue service initialization skipped (service removed)")
        
        # Cache service removed - no longer needed
        logger.info("Cache service initialization skipped (service removed)")
        
        logger.info("Async services initialization complete")
        
    except Exception as e:
        logger.error(f"Error in async service initialization: {e}")


async def cleanup_all_services():
    """Cleanup all services and close HTTP sessions to prevent memory leaks."""
    logger.info("Cleaning up all services...")
    
    try:
        # Close global session manager first
        from kata.services.session_manager import cleanup_all_sessions
        await cleanup_all_sessions()
        logger.info("✅ Global session manager cleaned up")
        
        # Import modules with global service instances
        try:
            from .api import token_analysis, token_discovery
            # trading_scanner module no longer exists
            
            # Close token analysis services safely
            if hasattr(token_analysis, 'coingecko_service') and token_analysis.coingecko_service:
                if hasattr(token_analysis.coingecko_service, 'close_session'):
                    await token_analysis.coingecko_service.close_session()
                elif hasattr(token_analysis.coingecko_service, 'close'):
                    await token_analysis.coingecko_service.close()
                    
            if hasattr(token_analysis, 'dexscreener_service') and token_analysis.dexscreener_service:
                if hasattr(token_analysis.dexscreener_service, 'close'):
                    await token_analysis.dexscreener_service.close()
                    
            if hasattr(token_analysis, 'enhanced_market_service') and token_analysis.enhanced_market_service:
                if hasattr(token_analysis.enhanced_market_service, 'close'):
                    await token_analysis.enhanced_market_service.close()
                    
            # Close token discovery services safely
            if hasattr(token_discovery, 'enhanced_discovery_service') and token_discovery.enhanced_discovery_service:
                if hasattr(token_discovery.enhanced_discovery_service, 'close'):
                    await token_discovery.enhanced_discovery_service.close()
                    
            if hasattr(token_discovery, 'coingecko_service') and token_discovery.coingecko_service:
                if hasattr(token_discovery.coingecko_service, 'close_session'):
                    await token_discovery.coingecko_service.close_session()
                elif hasattr(token_discovery.coingecko_service, 'close'):
                    await token_discovery.coingecko_service.close()
                    
            # trading_scanner services no longer exist - module removed
                    
        except ImportError as e:
            logger.warning(f"Could not import service modules for cleanup: {e}")
        
        # platform_analysis_engine service no longer exists - module removed
        logger.info("Platform analysis engine cleanup skipped - module removed")
        
        logger.info("✅ All HTTP sessions closed successfully")
        
    except Exception as e:
        logger.error(f"Error during service cleanup: {e}")
        # Continue shutdown even if cleanup fails


def _register_routers():
    """
    Import and register all API routers.

    This runs inside the lifespan startup (after Uvicorn has already bound the
    port), so the platform health check can hit an already-listening
    server while the heavy imports (ccxt, pandas, privy, etc.) load in the
    background.  All routes are available before the first real user request
    because the lifespan completes before FastAPI begins serving user traffic.
    """
    import importlib

    # --- lightweight routers (fast imports) ---
    from kata.api.routers import auth, agents, trading, token_image, pendle, bridge_quotes
    from kata.api.routers import portfolio
    from kata.api import blockchain, learning_analytics
    from kata.api import hyperliquid_markets, token_analysis_card, trading_history, credits, platform_signals

    # --- heavy routers (deferred to here so startup is fast) ---
    from kata.api.routers import wallet           # privy.lib ~0.78s
    from kata.api.routers import agent_allocation # agent_allocation_service + hyperliquid ~1.29s
    from kata.api import token_discovery          # market data ~0.60s
    from kata.api import token_analysis           # llm_analysis + binance ~1.01s
    from kata.api import agent_learning_dashboard # learning services ~0.80s

    prefix = settings.API_V1_PREFIX
    app.include_router(auth.router, prefix=prefix)
    app.include_router(portfolio.router, prefix=prefix)
    app.include_router(agents.router, prefix=prefix)
    app.include_router(agent_allocation.router, prefix=prefix)
    app.include_router(trading.router, prefix=prefix)
    app.include_router(hyperliquid_markets.router, prefix=prefix)
    app.include_router(token_analysis_card.router, prefix=prefix)
    app.include_router(trading_history.router, prefix=prefix)
    app.include_router(credits.router, prefix=prefix)
    app.include_router(token_discovery.router, prefix=prefix)
    app.include_router(token_analysis.router, prefix=prefix)
    app.include_router(platform_signals.router, prefix=prefix)
    app.include_router(pendle.router, prefix=prefix)
    app.include_router(bridge_quotes.router, prefix=f"{prefix}/bridge", tags=["Bridge Quotes"])
    app.include_router(token_image.router, prefix=prefix)
    app.include_router(wallet.router)
    app.include_router(blockchain.router, prefix=f"{prefix}/blockchain", tags=["Blockchain Proxy"])
    app.include_router(agent_learning_dashboard.router)
    app.include_router(learning_analytics.router, prefix=prefix)

    logger.info("✅ All API routers registered")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Kata Autonomous Agent backend...")

    # Mount routers before serving traffic so a deployment cannot come up
    # "healthy" while missing the actual API surface.
    try:
        _register_routers()
    except Exception:
        logger.exception("Router registration failed during startup")
        raise

    logger.info("✅ Backend startup complete — API routers registered")

    # Also start the HyperliquidStreamManager lazily
    try:
        from kata.services.hyperliquid_stream_manager import get_stream_manager
        _stream_mgr = get_stream_manager()
        logger.info("HyperliquidStreamManager registered (WS connects on first SSE subscriber)")
    except Exception as _sm_err:
        logger.warning(f"Could not initialise HyperliquidStreamManager: {_sm_err}")

    yield
    
    # Shutdown
    logger.info("Shutting down Kata Autonomous Agent backend...")

    # Shut down Hyperliquid WebSocket stream manager
    try:
        from kata.services.hyperliquid_stream_manager import get_stream_manager
        await get_stream_manager().shutdown()
        logger.info("HyperliquidStreamManager shut down")
    except Exception as _sm_err:
        logger.warning(f"Error shutting down HyperliquidStreamManager: {_sm_err}")
    
    # Close all HTTP sessions to prevent memory leaks
    try:
        await cleanup_all_services()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    try:
        db_manager.reset_connections()
    except Exception as e:
        logger.error(f"Error resetting database connections: {e}")
    
    logger.info("Backend shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
cors_origins = settings.cors_origins
logger.info(f"Configuring CORS with origins: {cors_origins}")
logger.info(f"FRONTEND_URL environment variable: {settings.FRONTEND_URL}")
logger.info(f"Total CORS origins count: {len(cors_origins)}")

# Log each origin for debugging
for i, origin in enumerate(cors_origins):
    logger.info(f"CORS origin {i+1}: {origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporary: Allow all origins for debugging
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# NOTE: Routers are now registered inside _register_routers() called from
# the lifespan startup, so there are no include_router() calls here.



# Mount static files for the Kata Cockpit web client
client_dir = Path(__file__).resolve().parent.parent / "client"
if not client_dir.exists():
    client_dir = Path.cwd() / "client"

if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the Kata Autonomous Agent Cockpit web client."""
        return FileResponse(client_dir / "index.html")

    @app.get("/manifest.json", include_in_schema=False)
    async def serve_manifest():
        """Serve the PWA web app manifest."""
        manifest_path = client_dir / "manifest.json"
        if manifest_path.exists():
            return FileResponse(manifest_path, media_type="application/manifest+json")
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon():
        """Serve the brand favicon mark."""
        favicon_path = client_dir / "brand" / "kata-mark-alt.svg"
        if favicon_path.exists():
            return FileResponse(favicon_path, media_type="image/svg+xml")
        return JSONResponse(status_code=404, content={"detail": "Not found"})
else:
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "healthy",
            "default_venue": settings.DEFAULT_VENUE
        }

@app.get("/api")
async def api_info():
    """API metadata and health info."""
    return {
        "message": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "default_venue": settings.DEFAULT_VENUE,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "trading_history": f"{settings.API_V1_PREFIX}/trading-history",
            "learning_overview": f"{settings.API_V1_PREFIX}/learning/overview/yuki"
        }
    }


@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    """Handle CORS preflight requests."""
    origin = request.headers.get("origin")
    
    # Check if origin is in allowed origins
    if origin in cors_origins:
        return JSONResponse(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    else:
        return JSONResponse(
            status_code=403,
            content={"error": "Origin not allowed"}
        )

@app.get("/health")
async def health_check():
    """Fast health check endpoint for deployment - no database dependency."""
    return {
        "status": "healthy",
        "service": "backend",
        "version": settings.VERSION,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with database testing."""
    try:
        db_healthy = await db_manager.health_check()

        # Check AI provider status
        import os
        ai_provider = os.getenv('LLM_PROVIDER', 'deepseek')
        ai_status = "unknown"

        try:
            from kata.services.llm_analysis_service import create_llm_analysis_service
            llm_service = create_llm_analysis_service()
            ai_status = "healthy" if llm_service.client else "no_client"
        except Exception as e:
            ai_status = f"error: {str(e)[:50]}"

        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "healthy" if db_healthy else "unhealthy",
            "ai_provider": ai_provider,
            "ai_status": ai_status,
            "service": "backend",
            "version": settings.VERSION,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": settings.VERSION
            }
        )


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection",
            "status": "connected",
            "user_id": user_id,
            "timestamp": "2024-01-01T00:00:00Z"
        }, user_id)
        
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": "2024-01-01T00:00:00Z"
                }, user_id)
            
            elif data.get("type") == "subscribe":
                # Handle subscription to specific data feeds
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "feed": data.get("feed"),
                    "timestamp": "2024-01-01T00:00:00Z"
                }, user_id)
            
            elif data.get("type") == "request_update":
                # Handle manual update requests
                await send_portfolio_update(user_id)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)


async def send_portfolio_update(user_id: str):
    """Send portfolio update to a specific user."""
    try:
        # This would fetch real portfolio data
        portfolio_data = {
            "type": "portfolio_update",
            "user_id": user_id,
            "data": {
                "total_value": 2850.00,
                "change_24h": 125.50,
                "change_percent": 4.6,
                "last_updated": "2024-01-01T00:00:00Z"
            }
        }
        
        await manager.send_personal_message(portfolio_data, user_id)
        
    except Exception as e:
        logger.error(f"Error sending portfolio update to user {user_id}: {e}")


async def send_trade_notification(user_id: str, trade_data: Dict[str, Any]):
    """Send trade notification to a specific user."""
    try:
        notification = {
            "type": "trade_notification",
            "user_id": user_id,
            "data": trade_data,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        await manager.send_personal_message(notification, user_id)
        
    except Exception as e:
        logger.error(f"Error sending trade notification to user {user_id}: {e}")


async def send_agent_status_update(user_id: str, agent_data: Dict[str, Any]):
    """Send agent status update to a specific user."""
    try:
        update = {
            "type": "agent_status_update",
            "user_id": user_id,
            "data": agent_data,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        await manager.send_personal_message(update, user_id)
        
    except Exception as e:
        logger.error(f"Error sending agent status update to user {user_id}: {e}")


# Removed broken send_trading_scanner_update function
# Trading scanner analysis is user-triggered only via API endpoints


async def background_task_runner():
    """Background task runner for periodic updates and performance monitoring."""
    logger.info("Starting background task runner...")
    logger.info("Platform signals run in the platform worker; learning runs in discovery maintenance.")

    while True:
        try:
            # Run background tasks every 30 seconds
            await asyncio.sleep(30)

            # Send periodic updates to connected users
            if manager.active_connections:
                logger.debug(f"Sending periodic updates to {len(manager.active_connections)} connected users")

                for user_id in list(manager.active_connections.keys()):
                    # Portfolio updates every 30 seconds
                    await send_portfolio_update(user_id)

        except Exception as e:
            logger.error(f"Error in background task runner: {e}")
            await asyncio.sleep(60)  # Wait longer on error


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "kata.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
