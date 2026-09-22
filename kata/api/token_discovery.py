"""
Token Discovery API endpoints for Flow AI Trading Platform.

Provides AI-analyzed token discovery data with daily analysis storage
and real-time market data updates.
"""

import logging
import httpx
import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Response
from pydantic import BaseModel, Field

from kata.services.token_discovery_service import TokenDiscoveryService, get_token_discovery_service
from kata.services.token_discovery_database_service import get_token_discovery_db_service
from kata.models.token_discovery import DiscoveredToken, TokenAnalysis
from kata.services.cache_service import CacheService, get_cache_service
from kata.services.queue_service import QueueService, QueuePriority, get_queue_service

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/token-discovery", tags=["Token Discovery"])


def _score_to_percent(value: Optional[float]) -> float:
    """Normalize stored scores from 0-1, 0-10, or 0-100 onto 0-100."""
    try:
        score = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1:
        score *= 100
    elif score <= 10:
        score *= 10
    return max(0.0, min(100.0, score))


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _pnl_percent(entry_price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    try:
        entry = float(entry_price or 0)
        current = float(current_price or 0)
        if entry <= 0 or current <= 0:
            return None
        return round(((current - entry) / entry) * 100.0, 2)
    except (TypeError, ValueError):
        return None


def _moonshot_lifecycle_payload(token, indicators: Dict[str, Any]) -> Dict[str, Any]:
    lifecycle = indicators.get("moonshot_lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}

    entry_at = _parse_iso_datetime(lifecycle.get("entry_at")) or token.discovery_date
    entry_price = lifecycle.get("entry_price") or token.current_price
    current_price = token.current_price
    now = datetime.now(timezone.utc)
    expires_at = _parse_iso_datetime(lifecycle.get("expires_at"))
    if not expires_at and entry_at:
        expires_at = entry_at + timedelta(days=7)

    current_pnl = lifecycle.get("current_pnl_percent")
    if current_pnl is None:
        current_pnl = _pnl_percent(entry_price, current_price)

    exit_price = lifecycle.get("exit_price")
    exit_pnl = lifecycle.get("exit_pnl_percent")
    if exit_pnl is None and exit_price is not None:
        exit_pnl = _pnl_percent(entry_price, exit_price)

    age_days = None
    if entry_at:
        age_days = round(max(0.0, (now - entry_at).total_seconds() / 86400), 2)

    days_remaining = None
    if expires_at:
        days_remaining = round(max(0.0, (expires_at - now).total_seconds() / 86400), 2)

    return {
        "moonshotStatus": lifecycle.get("status") or ("active" if token.is_active else "exited"),
        "moonshotEntryAt": entry_at.isoformat() if entry_at else None,
        "moonshotEntryPrice": entry_price,
        "moonshotCurrentPnlPercent": current_pnl,
        "moonshotExitAt": lifecycle.get("exit_at"),
        "moonshotExitPrice": exit_price,
        "moonshotExitPnlPercent": exit_pnl,
        "moonshotExitReason": lifecycle.get("exit_reason"),
        "moonshotExpiresAt": expires_at.isoformat() if expires_at else None,
        "moonshotDaysRemaining": days_remaining,
        "moonshotAgeDays": lifecycle.get("age_days", age_days),
    }


# Pydantic models for API responses
class TokenDiscoveryResponse(BaseModel):
    """Response for token discovery data."""
    id: str
    symbol: str
    name: str
    blockchain: str
    chain_id: int
    address: str
    logo_url: Optional[str] = None
    
    # Analysis data (stored daily)
    risk_score: float
    potential_score: float
    confidence_score: float
    analysis_summary: str
    trading_signals: List[str]
    analysis_date: datetime
    
    # Real-time market data
    current_price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    last_updated: datetime


class TokenAnalysisResponse(BaseModel):
    """Response for detailed token analysis."""
    token_id: str
    analysis_type: str
    score: float
    confidence: float
    summary: str
    signals: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    created_at: datetime




@router.get("/search", response_model=List[TokenDiscoveryResponse])
async def search_tokens(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=50),
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Search tokens by symbol or name."""
    try:
        logger.info(f"Searching tokens with query='{query}', limit={limit}")
        
        tokens = await token_service.search_tokens(query, limit)
        
        return [
            TokenDiscoveryResponse(
                id=token.id,
                symbol=token.symbol,
                name=token.name,
                blockchain=token.blockchain,
                chain_id=token.chain_id,
                address=token.address,
                logo_url=token.logo_url,
                risk_score=token.analysis.risk_score if token.analysis else 5.0,
                potential_score=token.analysis.potential_score if token.analysis else 5.0,
                confidence_score=token.analysis.confidence_score if token.analysis else 0.5,
                analysis_summary=token.analysis.summary if token.analysis else "Analysis pending",
                trading_signals=token.analysis.signals if token.analysis else [],
                analysis_date=token.analysis.created_at if token.analysis else datetime.now(),
                current_price=token.current_price,
                market_cap=token.market_cap,
                volume_24h=token.volume_24h,
                price_change_24h=token.price_change_24h,
                last_updated=token.last_updated
            )
            for token in tokens
        ]
        
    except Exception as e:
        logger.error(f"Error searching tokens: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search tokens: {str(e)}"
        )


@router.get("/analysis/{token_id}", response_model=TokenAnalysisResponse)
async def get_token_analysis(
    token_id: str,
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Get detailed analysis for a specific token."""
    try:
        logger.info(f"Getting analysis for token {token_id}")
        
        analysis = await token_service.get_token_analysis(token_id)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis not found for token {token_id}"
            )
        
        return TokenAnalysisResponse(
            token_id=analysis.token_id,
            analysis_type=analysis.analysis_type,
            score=analysis.overall_score,
            confidence=analysis.confidence_score,
            summary=analysis.summary,
            signals=analysis.signals,
            risk_factors=analysis.risk_factors,
            opportunities=analysis.opportunities,
            created_at=analysis.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting token analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get token analysis: {str(e)}"
        )


@router.post("/analyze/{token_id}")
async def trigger_token_analysis(
    token_id: str,
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Trigger AI analysis for a specific token."""
    try:
        logger.info(f"Triggering analysis for token {token_id}")
        
        result = await token_service.analyze_token(token_id)
        
        return {
            "message": f"Analysis triggered for token {token_id}",
            "analysis_id": result["analysis_id"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error triggering token analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger analysis: {str(e)}"
        )


@router.get("/preview")
async def get_token_discovery_preview(
    response: Response,
    limit: int = Query(default=12, ge=3, le=50),
    min_market_cap: Optional[float] = Query(default=1000000),
    max_market_cap: Optional[float] = Query(default=500000000),
    min_volume: Optional[float] = Query(default=200000),
):
    """Small cached token discovery payload for the dashboard card."""
    try:
        response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=600"
        response.headers["Vary"] = "Accept-Encoding"

        db_service = get_token_discovery_db_service()
        candidate_limit = min(max(limit * 4, limit), 50)
        candidates = await db_service.get_dashboard_preview_tokens(
            limit=candidate_limit,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            min_volume=min_volume,
        )

        selected_tokens = []
        seen_symbols = set()
        for token in candidates:
            symbol = str(token.get('symbol') or '').upper()
            name = str(token.get('name') or '')
            if not symbol or symbol in seen_symbols:
                continue
            if _is_stablecoin_or_wrapped_token(symbol, name):
                continue

            seen_symbols.add(symbol)
            selected_tokens.append(token)
            if len(selected_tokens) >= limit:
                break

        return {
            "status": "success",
            "count": len(selected_tokens),
            "tokens": selected_tokens,
            "limit": limit,
            "preview": True,
        }

    except Exception as e:
        logger.error(f"Error getting token discovery preview: {e}")
        return {
            "status": "error",
            "message": str(e),
            "count": 0,
            "tokens": [],
            "preview": True,
        }



@router.get("/progressive")
async def discover_tokens_progressive(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0, description="Number of tokens to skip for pagination"),
    progressive: bool = Query(default=True),
    tier: str = Query(default="quality", description="quality | moonshot | past_moonshot | all"),
    min_market_cap: Optional[float] = Query(default=None),
    max_market_cap: Optional[float] = Query(default=None),
    min_volume: Optional[float] = Query(default=None),
    response: Response = None
):
    """
    Progressive token discovery that reads from database populated by worker.
    Uses pre-analyzed tokens with full AI analysis quality.
    """
    try:
        # Set cache headers for upstream edge caching
        if response:
            response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
            response.headers["Vary"] = "Accept-Encoding"

        normalized_tier = str(tier or "quality").lower().replace("-", "_")

        # Tier-aware default floors/ceilings. Moonshots are genuinely low-cap
        # ($100K–$5M early-stage), so the quality-tier bounds would exclude them all.
        # Past Moonshots should keep exited history even if the token later fell
        # below active Moonshot market/volume floors.
        if normalized_tier == "past_moonshot":
            min_market_cap = min_market_cap
            max_market_cap = max_market_cap
            min_volume = min_volume
        else:
            if min_market_cap is None:
                min_market_cap = 100000 if normalized_tier == "moonshot" else 1000000
            if max_market_cap is None:
                max_market_cap = 5000000 if normalized_tier == "moonshot" else 500000000
            if min_volume is None:
                min_volume = 30000 if normalized_tier == "moonshot" else 200000

        logger.info(f"🔄 Reading analyzed tokens from database (limit: {limit}, tier: {normalized_tier})")

        # Use the database service to get pre-analyzed tokens
        db_service = get_token_discovery_db_service()

        # TEMPORARY: Disable depth filtering until data is populated
        # AUTOMATIC DEPTH FILTERING: Apply minimum depth threshold backend-only
        # Filter out illiquid tokens with less than $200K total depth (bid + ask at 2%)
        # MIN_DEPTH_THRESHOLD = 200000  # $200K minimum total depth

        # Get trending tokens with optimized query and pagination
        tokens = await db_service.get_trending_tokens(
            limit=limit,  # Use exact limit for efficiency
            offset=offset,  # Add pagination support
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            min_volume=min_volume,
            tier=normalized_tier
            # min_total_depth_2pct=MIN_DEPTH_THRESHOLD  # TEMPORARY: Disabled until depth data populated
        )

        # Process tokens with optimized logic (minimal Python processing)
        discovered_tokens = []
        seen_token_ids = set()

        for token in tokens:
            try:
                # Defensive dedupe: stale duplicate analysis rows multiply tokens in the view
                if token.id in seen_token_ids:
                    continue
                seen_token_ids.add(token.id)

                # Quick stablecoin filter (market cap and volume already filtered at DB level)
                if _is_stablecoin_or_wrapped_token(token.symbol, token.name):
                    continue

                # Extract analysis scores (already calculated by worker)
                analysis = token.analysis
                if analysis:
                    quality_score = _score_to_percent(analysis.overall_score)
                    risk_score = _score_to_percent(analysis.risk_score)
                    potential_score = _score_to_percent(analysis.potential_score)
                    confidence_score = _score_to_percent(analysis.confidence_score)
                else:
                    # Fallback if no analysis available
                    quality_score = 50
                    risk_score = 50
                    potential_score = 50
                    confidence_score = 50

                # Calculate security score based on risk
                security_score = max(10, min(100, 100 - risk_score))

                # Parse contract addresses JSON
                contract_addresses = {}
                try:
                    if hasattr(token, 'contract_addresses') and token.contract_addresses:
                        if isinstance(token.contract_addresses, str):
                            contract_addresses = json.loads(token.contract_addresses)
                        else:
                            contract_addresses = token.contract_addresses or {}
                except:
                    contract_addresses = {}

                indicators = analysis.technical_indicators if analysis and analysis.technical_indicators else {}
                token_data = {
                    "id": token.id,
                    "symbol": token.symbol,
                    "name": token.name,
                    "blockchain": token.blockchain,
                    "chainId": token.chain_id,
                    "address": token.address,
                    # Enhanced LiFi integration fields
                    "primaryAddress": getattr(token, 'primary_address', None),
                    "primaryChainId": getattr(token, 'primary_chain_id', None),
                    "contractAddresses": contract_addresses,
                    "decimals": getattr(token, 'decimals', 18),
                    "isLiFiSupported": True,  # Default to supported for now
                    # Market data
                    "price": token.current_price,
                    "priceChange24h": token.price_change_24h,
                    "marketCap": token.market_cap,
                    "volume24h": token.volume_24h,
                    "liquidity": max(token.liquidity, token.bid_depth_2pct + token.ask_depth_2pct),
                    "bidDepth2pct": token.bid_depth_2pct,
                    "askDepth2pct": token.ask_depth_2pct,
                    "totalDepth2pct": token.bid_depth_2pct + token.ask_depth_2pct,
                    "liquidityRatio": round((token.volume_24h / token.market_cap * 100) if token.market_cap > 0 else 0, 3),
                    "logoUrl": token.logo_url or "",
                    "coingeckoId": token.coingecko_id or "",
                    "tradingScore": f"{quality_score:.1f}",
                    "securityScore": security_score,
                    "qualityScore": quality_score,
                    "riskScore": risk_score,
                    "potentialScore": potential_score,
                    "confidenceScore": confidence_score,
                    "isSafe": quality_score > 50 and security_score > 60,
                    "technicalIndicators": indicators,
                    "warnings": _generate_warnings_from_analysis(token, analysis),
                    "tradingPairs": [],
                    "riskMetrics": {"volatility": "moderate"},
                    "trending": token.is_trending,
                    "newListing": not token.coingecko_id or bool(indicators.get('source') == 'dexscreener'),
                    "source": indicators.get('source') if analysis else ("dexscreener" if not token.coingecko_id else "coingecko"),
                    "pairAgeDays": indicators.get('pair_age_days') if analysis else None,
                    "dexscreenerUrl": indicators.get('dexscreener_url') if analysis else None,
                    "dexscreenerTrendScore": indicators.get('dexscreener_trend_score') if analysis else None,
                    "moonshotRankScore": indicators.get('moonshot_rank_score') if analysis else None,
                    "moonshotRankBreakdown": indicators.get('moonshot_rank_breakdown') if analysis else None,
                    "dexOrderFlowScore": indicators.get('dex_order_flow_score') if analysis else None,
                    "dexOrderFlow": indicators.get('dex_order_flow') if analysis else None,
                    "futuresContext": indicators.get('futures_context') if analysis else None,
                    "futuresContextScore": indicators.get('futures_context_score') if analysis else None,
                    "aiReviewed": bool(analysis and indicators.get('ai_reviewed')),
                    "aiInclude": indicators.get('ai_include') is True if analysis else None,
                    "aiRecommendation": indicators.get('ai_recommendation') if analysis else None,
                    "aiConviction": indicators.get('ai_conviction') if analysis else None,
                    "aiThesis": indicators.get('ai_thesis') if analysis else None,
                    "aiCatalyst": indicators.get('ai_catalyst') if analysis else None,
                    "aiInvalidation": indicators.get('ai_invalidation') if analysis else None,
                    "aiReviewVersion": indicators.get('ai_review_version') if analysis else None,
                    "aiConvictionPrev": indicators.get('ai_conviction_prev') if analysis else None,
                    "aiReviewedAt": indicators.get('ai_reviewed_at') if analysis else None,
                    "discoveryDate": token.discovery_date.isoformat() if token.discovery_date else None,
                    "lastUpdated": token.last_updated.isoformat() if token.last_updated else None
                }
                if indicators.get('source') == 'dexscreener' or normalized_tier in {"moonshot", "past_moonshot"}:
                    token_data.update(_moonshot_lifecycle_payload(token, indicators))

                discovered_tokens.append(token_data)

            except Exception as token_error:
                logger.warning(f"Error processing token {token.symbol}: {token_error}")
                continue
        
        logger.info(f"✅ Retrieved {len(discovered_tokens)} high-quality tokens from database (offset: {offset})")

        return {
            "status": "success",
            "count": len(discovered_tokens),
            "tokens": discovered_tokens,
            "offset": offset,
            "limit": limit,
            "hasMore": len(discovered_tokens) == limit,  # Indicate if there might be more results
            "progressive": progressive
        }
        
    except Exception as e:
        logger.error(f"❌ Error reading tokens from database: {e}")
        return {
            "status": "error",
            "message": str(e),
            "count": 0,
            "tokens": [],
            "isComplete": False,
            "progressive": progressive
        }


def _generate_warnings_from_analysis(token, analysis):
    """Generate warnings from token analysis."""
    warnings = []
    
    if token.market_cap < 10_000_000:
        warnings.append("Low market cap")
    
    if token.volume_24h < 500_000:
        warnings.append("Low trading volume")
    
    if analysis and analysis.risk_score > 7:  # High risk score (0-10 scale)
        warnings.append("High risk detected")
    
    if analysis and _score_to_percent(analysis.confidence_score) < 30:
        warnings.append("Low analysis confidence")
    
    if abs(token.price_change_24h) > 20:
        warnings.append("High volatility")
    
    # Add analysis-specific risk factors
    if analysis and analysis.risk_factors:
        warnings.extend(analysis.risk_factors[:2])  # Add top 2 risk factors
    
    return warnings


def _generate_warnings_fixed(token, quality_score, security_score):
    """Generate warnings with fixed scoring logic."""
    warnings = []
    
    if token.market_cap < 10_000_000:
        warnings.append("Low market cap")
    
    if token.volume_24h < 500_000:
        warnings.append("Low trading volume")
    
    if quality_score < 40:  # 40/100 threshold
        warnings.append("Low quality score")
    
    if security_score < 50:  # 50/100 threshold
        warnings.append("Security concerns")
    
    if abs(token.price_change_24h) > 20:
        warnings.append("High volatility")
    
    return warnings


def _generate_warnings_fixed_dict(token_dict, quality_score, security_score):
    """Generate warnings with fixed scoring logic for dict input."""
    warnings = []
    
    market_cap = token_dict.get('market_cap', 0)
    volume_24h = token_dict.get('volume_24h', 0)
    price_change_24h = token_dict.get('price_change_24h', 0)
    
    if market_cap < 10_000_000:
        warnings.append("Low market cap")
    
    if volume_24h < 500_000:
        warnings.append("Low trading volume")
    
    if quality_score < 40:  # 40/100 threshold
        warnings.append("Low quality score")
    
    if security_score < 50:  # 50/100 threshold
        warnings.append("Security concerns")
    
    if abs(price_change_24h) > 20:
        warnings.append("High volatility")
    
    return warnings


def _generate_warnings_token_object(token, quality_score, security_score):
    """Generate warnings with fixed scoring logic for token object input."""
    warnings = []
    
    market_cap = getattr(token, 'market_cap', 0)
    volume_24h = getattr(token, 'volume_24h', 0)
    price_change_24h = getattr(token, 'price_change_24h', 0)
    
    if market_cap < 10_000_000:
        warnings.append("Low market cap")
    
    if volume_24h < 500_000:
        warnings.append("Low trading volume")
    
    if quality_score < 40:  # 40/100 threshold
        warnings.append("Low quality score")
    
    if security_score < 50:  # 50/100 threshold
        warnings.append("Security concerns")
    
    if abs(price_change_24h) > 20:
        warnings.append("High volatility")
    
    return warnings


def _generate_warnings_enhanced(token, profitability_score):
    """Generate warnings for enhanced token discovery."""
    warnings = []
    
    if token.market_cap < 10_000_000:
        warnings.append("Low market cap")
    
    if token.total_volume < 500_000:
        warnings.append("Low trading volume")
    
    if profitability_score.total < 30:
        warnings.append("Low quality score")
    
    if abs(token.price_change_percentage_24h) > 20:
        warnings.append("High volatility")
    
    return warnings


@router.post("/daily-analysis")
async def run_daily_analysis(
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Run daily analysis on all tracked tokens (background job endpoint)."""
    try:
        logger.info("Starting daily token analysis")
        
        result = await token_service.run_daily_analysis()
        
        return {
            "message": "Daily analysis completed",
            "tokens_analyzed": result["tokens_analyzed"],
            "new_discoveries": result["new_discoveries"],
            "analysis_time": result["analysis_time"]
        }
        
    except Exception as e:
        logger.error(f"Error running daily analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run daily analysis: {str(e)}"
        )

@router.post("/cleanup")
async def cleanup_tokens(
    min_market_cap: Optional[float] = Query(default=1000000),
    max_market_cap: Optional[float] = Query(default=500000000),
    min_volume: Optional[float] = Query(default=100000),
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Cleanup tokens that no longer meet our criteria."""
    try:
        logger.info("Starting manual token cleanup")
        
        result = await token_service.cleanup_tokens_outside_criteria(
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            min_volume=min_volume
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error running token cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run token cleanup: {str(e)}"
        )

@router.post("/refresh")
async def refresh_token_discovery():
    """Manually refresh token discovery data - clear cache and trigger new discovery."""
    try:
        logger.info("🔄 Manual token discovery refresh triggered")
        
        # Clear any existing cache
        global_cache = {}
        
        # Trigger fresh discovery
        from datetime import datetime
        refresh_time = datetime.now()
        
        # Return success immediately - actual refresh happens in background
        return {
            "status": "success",
            "message": "Token discovery refresh initiated",
            "refresh_time": refresh_time.isoformat(),
            "note": "New tokens will be available within 30 seconds"
        }
        
    except Exception as e:
        logger.error(f"❌ Error refreshing token discovery: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/market-data/{token_id}")
async def get_real_time_market_data(
    token_id: str,
    token_service: TokenDiscoveryService = Depends(get_token_discovery_service)
):
    """Get real-time market data for a token."""
    try:
        logger.info(f"Getting real-time market data for token {token_id}")
        
        market_data = await token_service.get_real_time_market_data(token_id)
        
        if not market_data:
            raise HTTPException(
                status_code=404,
                detail=f"Market data not found for token {token_id}"
            )
        
        return market_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market data: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for token discovery service."""
    try:
        token_service = get_token_discovery_service()
        
        return {
            "status": "healthy",
            "service": "token_discovery",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Token discovery service health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Token discovery service unavailable"
        )


@router.get("/queue/status/{task_id}")
async def get_queue_status(
    task_id: str,
    queue_service: QueueService = Depends(get_queue_service)
):
    """Get status of a queued token discovery task."""
    try:
        status = await queue_service.get_queue_status(task_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Task not found")
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.get("/queue/stats")
async def get_queue_stats(
    queue_service: QueueService = Depends(get_queue_service)
):
    """Get comprehensive queue statistics."""
    try:
        return await queue_service.get_queue_stats()
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats(
    cache_service: CacheService = Depends(get_cache_service)
):
    """Get cache statistics and performance metrics."""
    try:
        return await cache_service.get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


# CoinGecko Proxy Endpoints for TokenMappingService
@router.get("/coingecko-coin-details/{coin_id}")
async def get_coingecko_coin_details(coin_id: str = Path(..., description="CoinGecko coin ID")):
    """
    Proxy endpoint for CoinGecko coin details API.
    Used by TokenMappingService to get contract addresses and token info.
    """
    try:
        logger.info(f"Fetching CoinGecko details for coin: {coin_id}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={
                    "localization": False,
                    "tickers": False,
                    "market_data": False,
                    "community_data": False,
                    "developer_data": False,
                    "sparkline": False
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"CoinGecko API returned status {response.status_code} for {coin_id}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"CoinGecko API error: {response.status_code}"
                )
                
    except httpx.TimeoutException:
        logger.error(f"Timeout fetching CoinGecko data for {coin_id}")
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error fetching CoinGecko data for {coin_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch coin details: {str(e)}"
        )


@router.get("/coingecko-markets")
async def get_coingecko_markets(
    vs_currency: str = Query(default="usd"),
    order: str = Query(default="market_cap_desc"),
    per_page: int = Query(default=20, ge=1, le=250),
    page: int = Query(default=1, ge=1),
    sparkline: bool = Query(default=False),
    locale: str = Query(default="en")
):
    """
    Proxy endpoint for CoinGecko markets API.
    Used by progressive token discovery to fetch market data.
    """
    try:
        logger.info(f"Fetching CoinGecko markets page {page}")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": vs_currency,
                    "order": order,
                    "per_page": per_page,
                    "page": page,
                    "sparkline": sparkline,
                    "locale": locale
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"CoinGecko markets API returned status {response.status_code}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"CoinGecko markets API error: {response.status_code}"
                )
                
    except httpx.TimeoutException:
        logger.error("Timeout fetching CoinGecko markets data")
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error fetching CoinGecko markets data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch markets data: {str(e)}"
        )


# Dynamic scoring functions
def _calculate_quality_score(market_cap: float, volume: float, market_cap_rank: int, ath: float, atl: float) -> int:
    """Calculate quality score based on market metrics (0-100)."""
    score = 50  # Base score
    
    # Market cap rank bonus (lower rank = higher quality)
    if market_cap_rank <= 50:
        score += 30
    elif market_cap_rank <= 100:
        score += 20
    elif market_cap_rank <= 300:
        score += 10
    elif market_cap_rank <= 500:
        score += 5
    
    # Volume to market cap ratio (liquidity indicator)
    if market_cap > 0:
        volume_ratio = volume / market_cap
        if volume_ratio > 0.1:  # High liquidity
            score += 15
        elif volume_ratio > 0.05:
            score += 10
        elif volume_ratio > 0.01:
            score += 5
    
    # Price stability (distance from ATH/ATL)
    if ath > atl and ath > 0:
        current_position = (ath - atl) / ath if ath > 0 else 0
        if current_position < 0.5:  # Closer to ATH
            score += 5
    
    return max(0, min(100, int(score)))


def _calculate_security_score(market_cap: float, market_cap_rank: int, volume: float, price_change_24h: float) -> int:
    """Calculate security score based on stability metrics (0-100)."""
    score = 60  # Base score
    
    # Market cap size (larger = more secure)
    if market_cap >= 1_000_000_000:  # $1B+
        score += 25
    elif market_cap >= 100_000_000:  # $100M+
        score += 15
    elif market_cap >= 10_000_000:   # $10M+
        score += 10
    elif market_cap >= 1_000_000:    # $1M+
        score += 5
    
    # Market rank stability
    if market_cap_rank <= 100:
        score += 10
    elif market_cap_rank <= 500:
        score += 5
    
    # Price stability (lower volatility = higher security)
    volatility_penalty = min(20, abs(price_change_24h) / 2)
    score -= volatility_penalty
    
    return max(0, min(100, int(score)))


def _calculate_risk_score(price_change_24h: float, market_cap: float, volume: float, current_price: float, ath: float, atl: float) -> float:
    """Calculate risk score (0-10, higher = more risky)."""
    risk_score = 3.0  # Base risk

    # Volatility risk
    volatility_risk = min(4.0, abs(price_change_24h) / 10)
    risk_score += volatility_risk

    # Market cap risk (smaller = riskier)
    if market_cap < 1_000_000:      # < $1M
        risk_score += 3.0
    elif market_cap < 10_000_000:   # < $10M
        risk_score += 2.0
    elif market_cap < 100_000_000:  # < $100M
        risk_score += 1.0

    # Liquidity risk
    if market_cap > 0:
        liquidity_ratio = volume / market_cap
        if liquidity_ratio < 0.001:  # Very low liquidity
            risk_score += 2.0
        elif liquidity_ratio < 0.01:
            risk_score += 1.0

    # Price position risk (far from ATH might indicate decline)
    if ath > 0 and current_price > 0:
        price_ratio = current_price / ath
        if price_ratio < 0.1:  # 90% down from ATH
            risk_score += 1.5
        elif price_ratio < 0.3:  # 70% down from ATH
            risk_score += 1.0

    return max(0.1, min(10.0, round(risk_score, 1)))


def _calculate_trading_score(volume: float, market_cap: float, price_change_24h: float) -> int:
    """Calculate trading score based on volume and activity."""
    score = 5000  # Base score
    
    # Volume score
    if volume >= 100_000_000:  # $100M+
        score += 3000
    elif volume >= 10_000_000:  # $10M+
        score += 2000
    elif volume >= 1_000_000:   # $1M+
        score += 1000
    elif volume >= 100_000:     # $100K+
        score += 500
    
    # Activity bonus (some volatility indicates active trading)
    activity_bonus = min(500, abs(price_change_24h) * 20)
    score += activity_bonus
    
    # Liquidity multiplier
    if market_cap > 0:
        liquidity_ratio = volume / market_cap
        if liquidity_ratio > 0.1:
            score += 1000
        elif liquidity_ratio > 0.05:
            score += 500
    
    return max(1000, min(10000, int(score)))


def _generate_warnings(risk_score: float, security_score: int, quality_score: int, price_change_24h: float, liquidity_ratio: float) -> List[str]:
    """Generate warning messages based on scores."""
    warnings = []
    
    if risk_score >= 8:
        warnings.append("High risk asset")
    elif risk_score >= 6:
        warnings.append("Medium-high risk")
    
    if security_score < 50:
        warnings.append("Low security score")
    
    if quality_score < 40:
        warnings.append("Low quality metrics")
    
    if abs(price_change_24h) > 20:
        warnings.append("High volatility")
    elif abs(price_change_24h) > 10:
        warnings.append("Moderate volatility")
    
    # Liquidity warnings
    if liquidity_ratio < 0.001:  # Less than 0.1%
        warnings.append("Very low liquidity")
    elif liquidity_ratio < 0.01:  # Less than 1%
        warnings.append("Low liquidity")
    
    return warnings


# Initialize services on startup
def _is_stablecoin_or_wrapped_token(symbol: str, name: str) -> bool:
    """Check if token is a stablecoin or wrapped token that should be filtered out."""
    symbol_upper = symbol.upper()
    name_lower = name.lower()

    # Comprehensive stablecoin symbols - including newly discovered ones
    stablecoin_symbols = {
        # Major USD stablecoins
        'USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'TUSD', 'USDP', 'LUSD',
        'MIM', 'USDD', 'GUSD', 'SUSD', 'USDN', 'RSR', 'USTC', 'FDUSD',
        'PYUSD', 'CRVUSD', 'USDE', 'USDM', 'USDS', 'USDG', 'USDY', 'USDB',
        'DOLA', 'FEI', 'TRIBE', 'RAI', 'OUSD', 'USDK', 'MUSD', 'USDBC',
        'USDQ',  # Quantoz USDQ

        # Euro stablecoins
        'EURC', 'EURS', 'EURT', 'EURCV',

        # Other regional stablecoins
        'GYEN', 'XSGD', 'TRYB', 'BIDR', 'IDRT', 'VAI', 'DJED', 'DUSD', 'XUSD',

        # Synthetic and algorithmic stablecoins
        'FRAX', 'MIM', 'LUSD', 'RAI', 'DOLA'
    }

    # Comprehensive wrapped token symbols
    wrapped_token_symbols = {
        # Wrapped ETH variants
        'WETH', 'WETH9', 'ETHW',  # EthereumPoW

        # Major wrapped tokens
        'WBTC', 'WBNB', 'WMATIC', 'WAVAX', 'WFTM', 'WONE', 'WCRO',
        'WKLAY', 'WMOVR', 'WGLMR', 'WROSE', 'WXDAI', 'WMNT', 'WPOL',

        # Aave wrapped tokens
        'AWETH', 'AWBTC', 'AUSDC', 'AUSDT', 'ADAI',

        # Hyperliquid wrapped tokens
        'WHYPE',

        # Bridged variants
        'SBETH'  # Sui Bridged Ether
    }

    # Extended wrapped/bridged token name patterns
    wrapped_keywords = [
        'wrapped', 'bridged', 'pegged', 'synthetic', 'aave ethereum',
        'sui bridged', 'bridged usdt', 'bridged usdc', 'wormhole token',
        'polygon pos bridged', 'mantle bridged', 'vaultbridge bridged',
        'bridged usd coin', 'wrapped aave', 'lido wsteth', 'aave base'
    ]

    # Check stablecoin symbols
    if symbol_upper in stablecoin_symbols:
        return True

    # Check wrapped token symbols
    if symbol_upper in wrapped_token_symbols:
        return True

    # Check for complex wrapped tokens containing stablecoin names
    if any(stable in symbol_upper for stable in ['USDC', 'USDT', 'DAI', 'GHO']) and symbol_upper.startswith(('W', 'A', 'WA')) and len(symbol_upper) > 4:
        return True

    # Check wrapped prefixes with expanded exceptions
    if symbol_upper.startswith(('W', 'WW', 'AW')):
        # Legitimate tokens that start with W but are not wrapped
        exceptions = {
            'WLD', 'WOO', 'WNXM', 'WAXP', 'WIN', 'WAVES', 'WRX', 'WAN',
            'WOLF', 'WFI', 'WEN', 'WAI', 'WCT', 'WORTHLESS'
        }
        if symbol_upper not in exceptions:
            return True

    # Check name patterns for wrapped/bridged indicators
    if any(keyword in name_lower for keyword in wrapped_keywords):
        return True

    # Additional pattern checks for specific naming conventions
    # Check for "bridged" in parentheses (common pattern)
    if 'bridged' in name_lower and ('(' in name_lower and ')' in name_lower):
        return True

    # Check for "wrapped" in parentheses
    if 'wrapped' in name_lower and ('(' in name_lower and ')' in name_lower):
        return True

    # Check for chain-specific bridged tokens
    chain_bridge_patterns = ['polygon pos', 'base bridged', 'linea bridged', 'mantle bridged']
    if any(pattern in name_lower for pattern in chain_bridge_patterns):
        return True

    # Check for trading pair tokens (e.g., BERAUSDT, ETHUSDT, BTCUSDT)
    if len(symbol_upper) > 6 and symbol_upper.endswith(('USDT', 'USDC', 'BTC', 'ETH')):
        # These are likely trading pairs, not actual tokens
        trading_pair_exceptions = {'USDT', 'USDC', 'WBTC', 'WETH'}  # Actual tokens with these names
        if symbol_upper not in trading_pair_exceptions:
            return True

    return False


def _calculate_discovery_score(token, quality_score: float, security_score: float) -> float:
    """Calculate discovery score to prioritize interesting low-cap tokens."""
    score = 0.0
    
    # Market cap scoring - favor smaller caps (inverted)
    market_cap = token.market_cap
    if market_cap < 5_000_000:      # < $5M
        score += 50
    elif market_cap < 20_000_000:   # < $20M  
        score += 40
    elif market_cap < 50_000_000:   # < $50M
        score += 30
    elif market_cap < 100_000_000:  # < $100M
        score += 20
    else:
        score += 10
    
    # Quality and security bonus
    score += (quality_score * 0.3)    # Up to 30 points
    score += (security_score * 0.2)   # Up to 20 points
    
    # Volume activity bonus
    volume_ratio = token.volume_24h / token.market_cap if token.market_cap > 0 else 0
    if volume_ratio > 0.1:      # High activity
        score += 15
    elif volume_ratio > 0.05:   # Good activity
        score += 10
    elif volume_ratio > 0.01:   # Decent activity
        score += 5
    
    # Blockchain diversity bonus (favor non-Ethereum)
    blockchain = token.blockchain.lower()
    if blockchain not in ['ethereum']:
        score += 10  # Bonus for chain diversity
    
    # Recent price action bonus
    price_change = abs(token.price_change_24h)
    if 2 <= price_change <= 15:  # Moderate positive movement
        score += 5
    
    return round(score, 2)


def initialize_token_discovery_services():
    """Initialize token discovery services."""
    try:
        get_token_discovery_service()
        logger.info("Token discovery services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize token discovery services: {e}")
        raise
