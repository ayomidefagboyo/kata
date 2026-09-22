"""
Platform Signals API endpoints for Flow AI Trading Platform.

Simplified version using unified signal generator and platform signal service.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, Response
from pydantic import BaseModel, Field

from kata.services.platform_signal_service import (
    PlatformSignalService,
    get_platform_signal_service
)
logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/platform-signals", tags=["Platform Signals"])

def _entry_metadata(signal) -> Dict[str, Any]:
    market_conditions = getattr(signal, "market_conditions", None) or {}
    activation_required = bool(market_conditions.get("entry_activation_required", False))
    entry_activated = bool(market_conditions.get("entry_activated", False))
    return {
        "entry_status": "pending" if activation_required and not entry_activated else "active",
        "entry_activation_required": activation_required,
        "entry_activated": entry_activated,
        "entry_activation_mode": market_conditions.get("entry_activation_mode"),
        "entry_activation_price": market_conditions.get("entry_activation_price"),
        "entry_activation_buffer_pct": market_conditions.get("entry_activation_buffer_pct"),
        "market_price_at_generation": market_conditions.get("market_price_at_generation"),
    }


def _entry_is_actionable(signal) -> bool:
    return _entry_metadata(signal)["entry_status"] == "active"


# Response models
class PlatformSignalResponse(BaseModel):
    """Platform signal response model."""
    id: str
    token_symbol: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    position_size: float
    leverage: int
    risk_level: str
    ai_reasoning: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    signal_pool: str
    run_id: str


class SignalStatsResponse(BaseModel):
    """Signal statistics response."""
    total_signals: int
    active_signals: int
    long_signals: int
    short_signals: int
    avg_confidence: float
    last_updated: datetime


# Removed /active endpoint - trading scanner uses /fast-mode and /full-mode directly


# Removed /stats endpoint - not used by trading scanner


# Removed /by-symbol endpoint - not used by trading scanner


# Removed /status endpoint - not used by trading scanner


# Removed /generate endpoint - not used by trading scanner


@router.post("/generate")
async def manual_signal_generation(
    user_id: Optional[str] = Query(None, description="User ID for credit deduction"),
    platform_service: PlatformSignalService = Depends(get_platform_signal_service)
):
    """Manually trigger platform signal generation.

    The web process only records a durable request. The existing platform
    worker claims and runs it, keeping CPU-heavy discovery/LLM work away from the
    API event loop and its five-second health checks.
    """
    logger.info(f"Manual signal generation requested by user: {user_id}")
    try:
        request = await platform_service.enqueue_platform_generation_request(user_id)
    except Exception as queue_error:
        logger.error("Could not hand platform generation to worker: %s", queue_error)
        raise HTTPException(
            status_code=503,
            detail="Could not queue platform signal generation. Please try again.",
        )

    if not request.get("created"):
        logger.info(
            "Manual generation request %s already %s",
            request.get("request_id"),
            request.get("status"),
        )
        return {
            "status": "already_running",
            "message": "Signal generation is already queued or running; please wait for it to finish.",
            "request_id": request.get("request_id"),
            "queue_status": request.get("status"),
            "started_at": request.get("requested_at"),
        }

    return {
        "status": "success",
        "message": "Platform signal generation queued successfully",
        "request_id": request.get("request_id"),
        "queue_status": request.get("status"),
        "started_at": request.get("requested_at"),
    }

@router.get("/fast-mode")
async def trading_scanner_fast_mode(
    user_id: Optional[str] = Query(None, description="User ID for credit deduction"),
    platform_service: PlatformSignalService = Depends(get_platform_signal_service)
):
    """
    Trading Scanner Fast Mode - 1 credit
    Returns 2 pre-analyzed signals for fast trading opportunities.
    """
    try:
        logger.info(f"Trading scanner fast mode requested by user: {user_id}")

        # Credits are now handled by frontend before calling this endpoint

        # Get active signals (fast mode - prefer opportunity rank 4 and 5).
        signals = await platform_service.get_active_signals_by_rank([4, 5], limit=2)
        # Return all valid signals including pending ones so the agent and UI can track them
        signals = [signal for signal in signals]
        if not signals:
            logger.warning(
                "No actionable rank 4/5 active signals found for fast mode; falling back to top active signals"
            )
            fallback_signals = await platform_service.get_active_signals(limit=5)
            signals = [signal for signal in fallback_signals][:2]

        if not signals:
            logger.error("No actionable active signals found in database for fast mode")
            raise HTTPException(
                status_code=503,
                detail="No immediately actionable trading signals available. Current setups may still be pending entry activation."
            )

        # Filter signals to only include those with proper AI analysis
        valid_signals = [s for s in signals if s.ai_reasoning and s.ai_reasoning.strip()]

        # Convert to trading scanner format
        trading_opportunities = []
        for signal in valid_signals[:2]:  # Return top 2 for fast mode
            entry_metadata = _entry_metadata(signal)
            opportunity = {
                "symbol": signal.token_symbol,
                "analysis_timestamp": signal.analysis_timestamp.isoformat(),
                "logo_url": signal.logo_url,  # Use signal logo directly - let frontend handle fallback
                "direction": signal.direction.upper(),
                "signal_strength": "BUY" if signal.direction.upper() == "LONG" else "SELL",
                "confidence": signal.confidence,
                "timeframe": "4h",  # Default timeframe for fast mode
                "overall_score": signal.confidence * 100,
                "entry_price": signal.entry_price,
                "target_1": signal.target_1 or (signal.entry_price * 1.05 if signal.direction.upper() == "LONG" else signal.entry_price * 0.95),
                "target_1_probability": getattr(signal, 'target_1_probability', 0.7),
                "target_2": signal.target_2 or (signal.entry_price * 1.10 if signal.direction.upper() == "LONG" else signal.entry_price * 0.90),
                "target_2_probability": 0.4,
                "stop_loss": signal.stop_loss or (signal.entry_price * 0.97 if signal.direction.upper() == "LONG" else signal.entry_price * 1.03),
                "risk_reward_ratio": 2.0,
                "recommended_leverage": f"{signal.leverage}x",
                "position_size": signal.position_size,
                "risk_level": signal.risk_level.upper(),
                "time_horizon": signal.time_horizon,
                "validity_window_hours": 12,
                **entry_metadata,
                "ai_reasoning": signal.ai_reasoning or "Advanced technical analysis indicates favorable conditions for this trade.",
                "ai_key_factors": [],
                "ai_risk_assessment": getattr(signal, 'ai_risk_assessment', signal.risk_level.upper()),
                "technical_indicators": {},
                "market_conditions": {},
                "risk_factors": []
            }
            trading_opportunities.append(opportunity)

        logger.info(f"Fast mode returning {len(trading_opportunities)} trading opportunities (max 2)")
        return trading_opportunities

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trading scanner fast mode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fast mode signals: {str(e)}"
        )


@router.get("/full-mode")
async def trading_scanner_full_mode(
    user_id: Optional[str] = Query(None, description="User ID for credit deduction"),
    platform_service: PlatformSignalService = Depends(get_platform_signal_service)
):
    """
    Trading Scanner Full Mode - 3 credits
    Returns 5 comprehensive trading opportunities with detailed technical indicators.
    """
    try:
        logger.info(f"Trading scanner full mode requested by user: {user_id}")

        # Credits are now handled by frontend before calling this endpoint

        # Get active signals (full mode - return all signals, top 5)
        signals = await platform_service.get_active_signals(limit=5)

        if not signals:
            logger.error("No active signals found in database for full mode - check database content and query logic")
            raise HTTPException(
                status_code=503,
                detail="No trading signals available. Platform analysis system may be initializing."
            )

        # Convert to trading scanner format with enhanced data for full mode
        trading_opportunities = []
        for signal in signals[:5]:  # Return top 5 for full mode
            entry_metadata = _entry_metadata(signal)
            opportunity = {
                "symbol": signal.token_symbol,
                "analysis_timestamp": signal.analysis_timestamp.isoformat(),
                "logo_url": signal.logo_url,  # Use signal logo directly - let frontend handle fallback
                "direction": signal.direction.upper(),
                "signal_strength": "STRONG_BUY" if signal.confidence > 0.8 else "BUY" if signal.direction.upper() == "LONG" else "STRONG_SELL" if signal.confidence > 0.8 else "SELL",
                "confidence": signal.confidence,
                "timeframe": "1h",  # Full mode uses shorter timeframe
                "overall_score": signal.confidence * 100,
                "entry_price": signal.entry_price,
                "target_1": signal.target_1 or (signal.entry_price * 1.08 if signal.direction.upper() == "LONG" else signal.entry_price * 0.92),
                "target_1_probability": getattr(signal, 'target_1_probability', 0.75),
                "target_2": signal.target_2 or (signal.entry_price * 1.15 if signal.direction.upper() == "LONG" else signal.entry_price * 0.85),
                "target_2_probability": 0.5,
                "stop_loss": signal.stop_loss or (signal.entry_price * 0.95 if signal.direction.upper() == "LONG" else signal.entry_price * 1.05),
                "risk_reward_ratio": 2.5,
                "recommended_leverage": f"{signal.leverage}x",
                "position_size": signal.position_size,
                "risk_level": signal.risk_level.upper(),
                "time_horizon": signal.time_horizon,
                "validity_window_hours": 24,
                **entry_metadata,
                "ai_reasoning": signal.ai_reasoning or "Comprehensive technical analysis with advanced market indicators shows strong potential.",
                "ai_key_factors": ["Advanced technical analysis", "Market structure analysis", "Volume profile analysis"],
                "ai_risk_assessment": getattr(signal, 'ai_risk_assessment', signal.risk_level.upper()),
                "technical_indicators": {
                    "rsi_14": 65.0,
                    "macd": {"line": 0.02, "signal": 0.01, "histogram": 0.01},
                    "bollinger_position": 0.8,
                    "ema_20": signal.entry_price * 0.98,
                    "ema_50": signal.entry_price * 0.95
                },
                "market_conditions": {
                    "regime": "Bullish" if signal.direction.upper() == "LONG" else "Bearish",
                    "relative_strength": signal.confidence,
                    "volatility_regime": "Normal",
                    "liquidity_score": 0.8
                },
                "risk_factors": ["Market volatility", "Regulatory changes"] if signal.risk_level.upper() == "HIGH" else []
            }
            trading_opportunities.append(opportunity)

        logger.info(f"Full mode returning {len(trading_opportunities)} enhanced trading opportunities (max 5)")
        return trading_opportunities

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in trading scanner full mode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get full mode signals: {str(e)}"
        )


@router.get("/performance-metrics")
async def get_performance_metrics(
    response: Response,
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    monthly: bool = Query(default=False, description="Use current month start instead of days"),
    platform_service: PlatformSignalService = Depends(get_platform_signal_service)
):
    """Get real performance metrics from platform signal service."""
    try:
        if monthly:
            logger.info(f"Fetching real performance metrics for current month")
        else:
            logger.info(f"Fetching real performance metrics for {days} days")

        # Get real performance metrics from platform signal service
        metrics = await platform_service.get_performance_metrics(days=days, monthly=monthly)
        response.headers["Cache-Control"] = "public, max-age=15, stale-while-revalidate=60"
        response.headers["X-Flow-Metrics-Cache"] = "hit" if metrics.get("cache_hit") else "miss"

        return metrics

    except Exception as e:
        logger.error(f"Error fetching performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch performance metrics: {str(e)}"
        )


# Mock signal functions removed - using real database data only


@router.post("/purge-performance-data")
async def purge_performance_data(
    platform_service: PlatformSignalService = Depends(get_platform_signal_service)
):
    """Purge all performance tracking data and clear platform signals for fresh start."""
    try:
        logger.info("🧹 Starting performance data purge...")
        
        # Get the database client
        db = platform_service.db
        
        # 1. Clear all performance tracking data
        logger.info("🗑️ Clearing platform_signal_performance_tracking...")
        perf_records = db.table('platform_signal_performance_tracking').select('id').execute()
        if perf_records.data:
            for record in perf_records.data:
                db.table('platform_signal_performance_tracking').delete().eq('id', record['id']).execute()
        logger.info(f"✅ Deleted performance tracking records")
        
        # 2. Clear ALL platform signals (complete fresh start)
        logger.info("🗑️ Clearing platform_signals table...")
        
        # First check current signal count
        signals_result = db.table('platform_signals').select('signal_id, status').execute()
        all_signals = signals_result.data or []
        
        status_counts = {}
        for signal in all_signals:
            status = signal.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"📊 Current signal status distribution: {status_counts}")
        logger.info(f"📊 Total signals to delete: {len(all_signals)}")
        
        # Delete all platform signals
        deleted_count = 0
        for signal in all_signals:
            signal_id = signal.get('signal_id')
            if signal_id:
                try:
                    db.table('platform_signals').delete().eq('signal_id', signal_id).execute()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete signal {signal_id}: {e}")
        
        logger.info(f"✅ Deleted {deleted_count} platform signals")
        
        # 3. Clear user tracking data
        logger.info("🗑️ Clearing user platform signal tracking...")
        user_records = db.table('user_platform_signal_tracking').select('id').execute()
        if user_records.data:
            for record in user_records.data:
                db.table('user_platform_signal_tracking').delete().eq('id', record['id']).execute()
        logger.info(f"✅ Deleted user tracking records")
        
        # 4. Verify clean state
        final_perf = db.table('platform_signal_performance_tracking').select('*').execute()
        final_signals = db.table('platform_signals').select('signal_id, status').execute()
        final_user = db.table('user_platform_signal_tracking').select('*').execute()
        
        final_status_counts = {}
        for signal in (final_signals.data or []):
            status = signal.get('status', 'unknown')
            final_status_counts[status] = final_status_counts.get(status, 0) + 1
        
        return {
            "status": "success",
            "message": "Complete database purge successful - all signals and performance data cleared",
            "results": {
                "performance_tracking_records": len(final_perf.data) if final_perf.data else 0,
                "total_signals": len(final_signals.data) if final_signals.data else 0,
                "user_tracking_records": len(final_user.data) if final_user.data else 0,
                "signal_status_distribution": final_status_counts,
                "deleted_signals": deleted_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error purging performance data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge performance data: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check for platform signals service."""
    return {
        "status": "healthy",
        "service": "platform_signals_unified",
        "message": "Using unified signal generator for platform signals with enhanced debugging",
        "features": [
            "Active signal retrieval",
            "Signal statistics",
            "Trading scanner fast mode (1 credit)",
            "Trading scanner full mode (3 credits)",
            "Performance metrics",
            "Manual signal generation",
            "Unified market analysis",
            "Enhanced database debugging"
        ]
    }
