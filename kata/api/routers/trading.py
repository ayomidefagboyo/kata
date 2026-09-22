"""
Trading router for trade history and trading operations.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from supabase import Client

from kata.config.database import get_db_client
from kata.models.trade import (
    Trade, TradeCreate, TradeHistory, TradeStats, 
    AgentDecision, AgentDecisionCreate
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading", tags=["Trading"])


@router.get("/history/{user_id}", response_model=TradeHistory)
async def get_trading_history(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    trade_type: Optional[str] = Query(None, description="Filter by trade type"),
    token_symbol: Optional[str] = Query(None, description="Filter by token symbol"),
    protocol: Optional[str] = Query(None, description="Filter by protocol"),
    days: Optional[int] = Query(None, ge=1, le=365, description="Filter by days back"),
    db: Client = Depends(get_db_client)
) -> TradeHistory:
    """Get trading history for a user with filters and pagination."""
    try:
        # Build base query
        query = db.from_("trades").select("*", count="exact").eq("user_id", user_id)
        
        # Apply filters
        if agent_type:
            query = query.eq("agent_type", agent_type)
        
        if trade_type:
            query = query.eq("trade_type", trade_type)
        
        if token_symbol:
            query = query.eq("token_symbol", token_symbol.upper())
        
        if protocol:
            query = query.eq("protocol", protocol)
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = query.gte("created_at", cutoff_date.isoformat())
        
        # Get total count for pagination
        count_response = query.execute()
        total_count = count_response.count if hasattr(count_response, 'count') else 0
        
        # Apply pagination and ordering
        offset = (page - 1) * page_size
        trades_response = query.order("created_at", desc=True).range(offset, offset + page_size - 1).execute()
        
        trades = [Trade(**trade) for trade in trades_response.data] if trades_response.data else []
        
        total_pages = (total_count + page_size - 1) // page_size
        
        trade_history = TradeHistory(
            trades=trades,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
        logger.info(f"Retrieved {len(trades)} trades for user {user_id} (page {page}/{total_pages})")
        return trade_history
        
    except Exception as e:
        logger.error(f"Error getting trading history for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trading history: {str(e)}"
        )


@router.get("/stats/{user_id}", response_model=TradeStats)
async def get_trading_stats(
    user_id: str,
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    days: Optional[int] = Query(30, ge=1, le=365, description="Period in days"),
    db: Client = Depends(get_db_client)
) -> TradeStats:
    """Get trading statistics for a user."""
    try:
        # Build base query
        query = db.from_("trades").select("*").eq("user_id", user_id)
        
        if agent_type:
            query = query.eq("agent_type", agent_type)
        
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = query.gte("created_at", cutoff_date.isoformat())
        
        trades_response = query.execute()
        trades_data = trades_response.data if trades_response.data else []
        
        if not trades_data:
            return TradeStats()
        
        # Calculate statistics
        total_trades = len(trades_data)
        total_volume_usd = sum(float(trade["value_usd"]) for trade in trades_data)
        total_buy_volume = sum(float(trade["value_usd"]) for trade in trades_data if trade["trade_type"] in ["buy", "open"])
        total_sell_volume = sum(float(trade["value_usd"]) for trade in trades_data if trade["trade_type"] in ["sell", "close"])
        avg_trade_size = total_volume_usd / total_trades if total_trades > 0 else 0
        
        # Calculate confidence stats
        confidence_scores = [float(trade["confidence_score"]) for trade in trades_data if trade["confidence_score"] is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
        
        # For win rate calculation, we'd need to track P&L per trade
        # For now, using a placeholder calculation
        win_rate = 0.0  # No real data available yet
        
        # For best/worst trades, we'd need P&L data
        best_trade = 0  # Placeholder
        worst_trade = 0  # Placeholder
        total_pnl = 0  # Placeholder
        
        trade_stats = TradeStats(
            total_trades=total_trades,
            total_volume_usd=total_volume_usd,
            total_buy_volume=total_buy_volume,
            total_sell_volume=total_sell_volume,
            avg_trade_size=avg_trade_size,
            win_rate=win_rate,
            total_pnl=total_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_confidence=avg_confidence
        )
        
        logger.info(f"Retrieved trading stats for user {user_id}: {total_trades} trades, ${total_volume_usd:.2f} volume")
        return trade_stats
        
    except Exception as e:
        logger.error(f"Error getting trading stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trading stats: {str(e)}"
        )


@router.post("/trades", response_model=Trade)
async def create_trade(
    trade_data: TradeCreate,
    db: Client = Depends(get_db_client)
) -> Trade:
    """Create a new trade record."""
    try:
        response = db.from_("trades").insert(trade_data.model_dump()).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create trade"
            )
        
        trade = Trade(**response.data[0])
        logger.info(f"Created trade {trade.id} for user {trade.user_id}: {trade.trade_type} {trade.token_symbol}")
        
        return trade
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create trade: {str(e)}"
        )


@router.get("/decisions/{user_id}", response_model=List[AgentDecision])
async def get_agent_decisions(
    user_id: str,
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    decision_type: Optional[str] = Query(None, description="Filter by decision type"),
    limit: int = Query(50, ge=1, le=100, description="Number of decisions to retrieve"),
    db: Client = Depends(get_db_client)
) -> List[AgentDecision]:
    """Get agent decision history for a user."""
    try:
        query = db.from_("agent_decisions").select("*").eq("user_id", user_id)
        
        if agent_type:
            query = query.eq("agent_type", agent_type)
        
        if decision_type:
            query = query.eq("decision_type", decision_type)
        
        decisions_response = query.order("created_at", desc=True).limit(limit).execute()
        
        decisions = [AgentDecision(**decision) for decision in decisions_response.data] if decisions_response.data else []
        
        logger.info(f"Retrieved {len(decisions)} agent decisions for user {user_id}")
        return decisions
        
    except Exception as e:
        logger.error(f"Error getting agent decisions for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent decisions: {str(e)}"
        )


@router.post("/decisions", response_model=AgentDecision)
async def create_agent_decision(
    decision_data: AgentDecisionCreate,
    db: Client = Depends(get_db_client)
) -> AgentDecision:
    """Create a new agent decision record."""
    try:
        response = db.from_("agent_decisions").insert(decision_data.model_dump()).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create agent decision"
            )
        
        decision = AgentDecision(**response.data[0])
        logger.info(f"Created agent decision {decision.id} for user {decision.user_id}: {decision.decision_type}")
        
        return decision
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent decision: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent decision: {str(e)}"
        )


@router.get("/performance/{user_id}")
async def get_performance_summary(
    user_id: str,
    period: str = Query("30d", description="Period: 1d, 7d, 30d, 90d, 1y"),
    db: Client = Depends(get_db_client)
) -> Dict[str, Any]:
    """Get performance summary for a user."""
    try:
        # Map period to days
        period_days = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365
        }
        
        days = period_days.get(period, 30)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get performance metrics
        metrics_response = db.from_("performance_metrics").select("*").eq("user_id", user_id).gte("metric_date", cutoff_date.date().isoformat()).order("metric_date", desc=True).execute()
        
        metrics_data = metrics_response.data if metrics_response.data else []
        
        if not metrics_data:
            return {
                "period": period,
                "total_value": 0,
                "total_change": 0,
                "total_change_percent": 0,
                "best_day": 0,
                "worst_day": 0,
                "total_trades": 0,
                "avg_daily_volume": 0,
                "data_points": []
            }
        
        # Calculate summary metrics
        latest_value = float(metrics_data[0]["portfolio_value_usd"])
        oldest_value = float(metrics_data[-1]["portfolio_value_usd"])
        
        total_change = latest_value - oldest_value
        total_change_percent = (total_change / oldest_value * 100) if oldest_value > 0 else 0
        
        daily_changes = []
        for i in range(len(metrics_data) - 1):
            current = float(metrics_data[i]["portfolio_value_usd"])
            previous = float(metrics_data[i + 1]["portfolio_value_usd"])
            daily_change = current - previous
            daily_changes.append(daily_change)
        
        best_day = max(daily_changes) if daily_changes else 0
        worst_day = min(daily_changes) if daily_changes else 0
        
        total_trades = sum(int(metric["trades_count"]) for metric in metrics_data)
        avg_daily_volume = total_trades / len(metrics_data) if metrics_data else 0
        
        # Prepare chart data
        data_points = [
            {
                "date": metric["metric_date"],
                "value": float(metric["portfolio_value_usd"]),
                "daily_pnl": float(metric["daily_pnl_usd"]),
                "trades_count": int(metric["trades_count"])
            }
            for metric in reversed(metrics_data)  # Reverse to get chronological order
        ]
        
        performance_summary = {
            "period": period,
            "total_value": latest_value,
            "total_change": total_change,
            "total_change_percent": total_change_percent,
            "best_day": best_day,
            "worst_day": worst_day,
            "total_trades": total_trades,
            "avg_daily_volume": avg_daily_volume,
            "data_points": data_points
        }
        
        logger.info(f"Retrieved performance summary for user {user_id} ({period})")
        return performance_summary
        
    except Exception as e:
        logger.error(f"Error getting performance summary for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance summary: {str(e)}"
        )