"""
Trading History API Endpoints

Provides endpoints for agent trading history, performance metrics,
and trading activity tracking.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal

from ..services.hyperliquid_service import HyperliquidService
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trading-history", tags=["trading-history"])

# Global service instances
hyperliquid_service: Optional[HyperliquidService] = None

class TradeRecord(BaseModel):
    """Individual trade record."""
    id: str
    agent_type: str  # yuki, sakura, ryu
    type: str  # buy, sell
    token: str
    amount: float
    price: float
    value: float
    timestamp: datetime
    reasoning: str
    status: str  # completed, pending, failed
    pnl: Optional[float] = None
    fees: Optional[float] = None

class TradingStats(BaseModel):
    """Trading performance statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_volume: float
    average_trade_size: float
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    sharpe_ratio: Optional[float] = None

class AgentPerformance(BaseModel):
    """Agent performance metrics."""
    agent_type: str
    stats: TradingStats
    recent_trades: List[TradeRecord]
    daily_pnl: List[Dict[str, Any]]
    active_positions: int

class TradingHistoryResponse(BaseModel):
    """Trading history response."""
    trades: List[TradeRecord]
    stats: TradingStats
    agent_performance: Dict[str, AgentPerformance]

def get_hyperliquid_service() -> HyperliquidService:
    """Get Hyperliquid service dependency."""
    global hyperliquid_service
    if not hyperliquid_service:
        from ..services.hyperliquid_service import HyperliquidService
        hyperliquid_service = HyperliquidService()
    return hyperliquid_service

@router.get("/recent-trades", summary="Get recent trading activity")
async def get_recent_trades(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    limit: int = Query(10, ge=1, le=100, description="Number of trades to return"),
    days: int = Query(7, ge=1, le=30, description="Number of days to look back"),
    hl_service: HyperliquidService = Depends(get_hyperliquid_service)
) -> List[TradeRecord]:
    """
    Get recent trading activity for agents.
    
    Args:
        agent_type: Optional agent filter (yuki, sakura, ryu)
        limit: Number of trades to return
        days: Number of days to look back
        
    Returns:
        List of recent trade records
    """
    try:
        # For now, generate realistic mock data based on agent performance
        # In production, this would fetch from Hyperliquid/database
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trades = await _generate_realistic_trades(agent_type, limit, start_date, end_date)
        
        return trades
        
    except Exception as e:
        logger.error(f"Error fetching recent trades: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trades: {str(e)}")

@router.get("/stats", summary="Get trading statistics")
async def get_trading_stats(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    days: int = Query(30, ge=1, le=365, description="Number of days to calculate stats"),
    hl_service: HyperliquidService = Depends(get_hyperliquid_service)
) -> TradingStats:
    """
    Get trading performance statistics.
    
    Args:
        agent_type: Optional agent filter
        days: Number of days to calculate stats
        
    Returns:
        Trading performance statistics
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate realistic stats based on agent type
        stats = await _calculate_agent_stats(agent_type, start_date, end_date)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating trading stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate stats: {str(e)}")

@router.get("/agent-performance", summary="Get agent performance comparison")
async def get_agent_performance(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    hl_service: HyperliquidService = Depends(get_hyperliquid_service)
) -> Dict[str, AgentPerformance]:
    """
    Get performance comparison for all agents.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Performance data for each agent
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        agent_performance = {}
        agents = ['yuki', 'sakura', 'ryu']
        
        for agent in agents:
            # Get trades and stats for each agent
            trades = await _generate_realistic_trades(agent, 50, start_date, end_date)
            stats = await _calculate_agent_stats(agent, start_date, end_date)
            daily_pnl = await _calculate_daily_pnl(agent, start_date, end_date)
            
            agent_performance[agent] = AgentPerformance(
                agent_type=agent,
                stats=stats,
                recent_trades=trades[:10],  # Last 10 trades
                daily_pnl=daily_pnl,
                active_positions=_get_active_positions(agent)
            )
        
        return agent_performance
        
    except Exception as e:
        logger.error(f"Error fetching agent performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch performance: {str(e)}")

@router.get("/daily-pnl/{agent_type}", summary="Get daily P&L for agent")
async def get_daily_pnl(
    agent_type: str,
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    hl_service: HyperliquidService = Depends(get_hyperliquid_service)
) -> List[Dict[str, Any]]:
    """
    Get daily P&L data for specific agent.
    
    Args:
        agent_type: Agent type (yuki, sakura, ryu)
        days: Number of days
        
    Returns:
        Daily P&L data
    """
    try:
        if agent_type not in ['yuki', 'sakura', 'ryu']:
            raise HTTPException(status_code=400, detail="Invalid agent type")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_pnl = await _calculate_daily_pnl(agent_type, start_date, end_date)
        
        return daily_pnl
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching daily P&L for {agent_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch daily P&L: {str(e)}")

# Helper functions to generate realistic trading data
async def _generate_realistic_trades(
    agent_type: Optional[str],
    limit: int,
    start_date: datetime,
    end_date: datetime
) -> List[TradeRecord]:
    """Generate realistic trading data based on agent characteristics."""
    
    import random
    import uuid
    
    trades = []
    agents_to_process = [agent_type] if agent_type else ['yuki', 'sakura', 'ryu']
    
    # Agent-specific characteristics
    agent_configs = {
        'yuki': {
            'trade_frequency': 0.8,  # High frequency
            'win_rate': 0.65,
            'avg_trade_size': 150,
            'volatility': 0.15,
            'tokens': ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'DOGE', 'SHIB'],
            'reasoning_templates': [
                'RSI oversold, momentum building',
                'Breakout above resistance confirmed',
                'Volume spike with positive sentiment',
                'Technical pattern completion',
                'Support level bounce detected'
            ]
        },
        'sakura': {
            'trade_frequency': 0.3,  # Low frequency, conservative
            'win_rate': 0.78,
            'avg_trade_size': 200,
            'volatility': 0.08,
            'tokens': ['BTC', 'ETH', 'USDC', 'DAI', 'AAVE', 'COMP'],
            'reasoning_templates': [
                'Strong fundamental support',
                'Conservative entry at support',
                'Risk-adjusted position sizing',
                'Long-term value opportunity',
                'Defensive positioning'
            ]
        },
        'ryu': {
            'trade_frequency': 0.5,  # Balanced
            'win_rate': 0.72,
            'avg_trade_size': 125,
            'volatility': 0.12,
            'tokens': ['BTC', 'ETH', 'SOL', 'UNI', 'LINK', 'DOT', 'ADA'],
            'reasoning_templates': [
                'Balanced risk-reward setup',
                'Technical confluence confirmed',
                'Market structure favorable',
                'Moderate momentum entry',
                'Strategic position adjustment'
            ]
        }
    }
    
    total_trades_needed = limit
    trades_per_agent = total_trades_needed // len(agents_to_process)
    
    for agent in agents_to_process:
        config = agent_configs[agent]
        agent_trades = min(trades_per_agent, int(config['trade_frequency'] * limit))
        
        for i in range(agent_trades):
            # Generate random timestamp within range
            time_range = end_date - start_date
            random_offset = random.random() * time_range.total_seconds()
            trade_time = start_date + timedelta(seconds=random_offset)
            
            # Select random token
            token = random.choice(config['tokens'])
            
            # Determine trade type (slight bias towards buys in bull market)
            trade_type = 'buy' if random.random() < 0.55 else 'sell'
            
            # Generate trade size around average with some variance
            size_multiplier = random.uniform(0.5, 2.0)
            trade_value = config['avg_trade_size'] * size_multiplier
            
            # Generate realistic prices (mock data)
            token_prices = {
                'BTC': 65000, 'ETH': 3200, 'SOL': 140, 'AVAX': 35,
                'MATIC': 0.85, 'DOGE': 0.08, 'SHIB': 0.000025,
                'USDC': 1.0, 'DAI': 1.0, 'AAVE': 95, 'COMP': 45,
                'UNI': 8.2, 'LINK': 14.5, 'DOT': 6.5, 'ADA': 0.45
            }
            
            base_price = token_prices.get(token, 100)
            price_variance = random.uniform(0.95, 1.05)
            price = base_price * price_variance
            amount = trade_value / price
            
            # Generate P&L based on win rate
            is_winning_trade = random.random() < config['win_rate']
            if is_winning_trade:
                pnl_percent = random.uniform(0.02, 0.12)  # 2-12% gain
            else:
                pnl_percent = random.uniform(-0.08, -0.02)  # 2-8% loss
            
            pnl = trade_value * pnl_percent
            fees = trade_value * 0.001  # 0.1% trading fee
            
            trade = TradeRecord(
                id=str(uuid.uuid4()),
                agent_type=agent,
                type=trade_type,
                token=token,
                amount=round(amount, 6),
                price=round(price, 6),
                value=round(trade_value, 2),
                timestamp=trade_time,
                reasoning=random.choice(config['reasoning_templates']),
                status='completed',
                pnl=round(pnl, 2),
                fees=round(fees, 2)
            )
            
            trades.append(trade)
    
    # Sort by timestamp (most recent first)
    trades.sort(key=lambda x: x.timestamp, reverse=True)
    
    return trades[:limit]

async def _calculate_agent_stats(
    agent_type: Optional[str],
    start_date: datetime,
    end_date: datetime
) -> TradingStats:
    """Calculate trading statistics for agent(s)."""
    
    # Generate trades for calculation
    trades = await _generate_realistic_trades(agent_type, 100, start_date, end_date)
    
    if not trades:
        return TradingStats(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_volume=0.0,
            average_trade_size=0.0
        )
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    total_volume = sum(t.value for t in trades)
    average_trade_size = total_volume / total_trades if total_trades > 0 else 0.0
    
    pnl_values = [t.pnl for t in trades if t.pnl]
    best_trade = max(pnl_values) if pnl_values else None
    worst_trade = min(pnl_values) if pnl_values else None
    
    # Simple Sharpe ratio calculation (daily returns / std deviation)
    daily_returns = [t.pnl / t.value for t in trades if t.pnl and t.value > 0]
    if len(daily_returns) > 1:
        import statistics
        avg_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns)
        sharpe_ratio = (avg_return / std_return) * (365 ** 0.5) if std_return > 0 else None
    else:
        sharpe_ratio = None
    
    return TradingStats(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate, 3),
        total_pnl=round(total_pnl, 2),
        total_volume=round(total_volume, 2),
        average_trade_size=round(average_trade_size, 2),
        best_trade=round(best_trade, 2) if best_trade else None,
        worst_trade=round(worst_trade, 2) if worst_trade else None,
        sharpe_ratio=round(sharpe_ratio, 3) if sharpe_ratio else None
    )

async def _calculate_daily_pnl(
    agent_type: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """Calculate daily P&L for agent."""
    
    daily_pnl = []
    current_date = start_date
    
    while current_date <= end_date:
        # Generate daily P&L based on agent characteristics
        agent_volatility = {
            'yuki': 0.15,    # High volatility
            'sakura': 0.05,  # Low volatility
            'ryu': 0.10      # Medium volatility
        }
        
        import random
        base_daily_return = random.uniform(-0.02, 0.03)  # -2% to 3% daily
        volatility = agent_volatility.get(agent_type, 0.10)
        daily_return = base_daily_return + random.gauss(0, volatility)
        
        # Simulate portfolio value (starting at $1000)
        base_value = 1000
        daily_pnl_value = base_value * daily_return
        
        daily_pnl.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'pnl': round(daily_pnl_value, 2),
            'cumulative_pnl': round(sum(d.get('pnl', 0) for d in daily_pnl), 2),
            'trades_count': random.randint(0, 5) if agent_type == 'yuki' else random.randint(0, 2)
        })
        
        current_date += timedelta(days=1)
    
    return daily_pnl

def _get_active_positions(agent_type: str) -> int:
    """Get number of active positions for agent."""
    # Mock active positions based on agent type
    position_ranges = {
        'yuki': (3, 8),    # Active trader
        'sakura': (1, 3),  # Conservative
        'ryu': (2, 5)      # Balanced
    }
    
    import random
    min_pos, max_pos = position_ranges.get(agent_type, (1, 3))
    return random.randint(min_pos, max_pos)

