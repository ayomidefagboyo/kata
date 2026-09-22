"""
Trading service for business logic related to trading operations and AI agents.
"""
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from supabase import Client

from kata.models.trade import Trade, TradeCreate, AgentDecision, AgentDecisionCreate
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


class TradingService:
    """Service class for trading operations."""
    
    def __init__(self):
        self.db: Client = get_service_client()
    
    async def execute_agent_decision(
        self, 
        user_id: str, 
        agent_type: str, 
        decision_data: Dict[str, Any]
    ) -> Optional[Trade]:
        """
        Execute a trading decision made by an AI agent.
        
        Args:
            user_id: User ID
            agent_type: Type of AI agent making the decision
            decision_data: Decision data containing trade parameters
            
        Returns:
            Created Trade object if executed, None if no trade was made
        """
        try:
            # Log the agent decision first
            agent_decision = AgentDecisionCreate(
                user_id=user_id,
                agent_type=agent_type,
                decision_type=decision_data.get("decision_type", "trade_signal"),
                market_data=decision_data.get("market_data", {}),
                decision_data=decision_data.get("decision_data", {}),
                action_taken=decision_data.get("action_taken"),
                confidence_score=decision_data.get("confidence_score")
            )
            
            decision_response = self.db.from_("agent_decisions").insert(agent_decision.model_dump()).execute()
            
            if not decision_response.data:
                logger.error(f"Failed to log agent decision for user {user_id}")
                return None
            
            # Check if decision resulted in a trade
            if decision_data.get("should_trade", False):
                trade_data = TradeCreate(
                    user_id=user_id,
                    agent_type=agent_type,
                    trade_type=decision_data.get("trade_type", "buy"),
                    token_symbol=decision_data.get("token_symbol", ""),
                    amount=Decimal(str(decision_data.get("amount", 0))),
                    price_usd=Decimal(str(decision_data.get("price_usd", 0))),
                    value_usd=Decimal(str(decision_data.get("value_usd", 0))),
                    protocol=decision_data.get("protocol", "base"),
                    tx_hash=decision_data.get("tx_hash"),
                    reasoning=decision_data.get("reasoning"),
                    confidence_score=decision_data.get("confidence_score")
                )
                
                trade_response = self.db.from_("trades").insert(trade_data.model_dump()).execute()
                
                if trade_response.data:
                    trade = Trade(**trade_response.data[0])
                    logger.info(f"Executed agent trade: {trade.trade_type} {trade.token_symbol} for user {user_id}")
                    return trade
                else:
                    logger.error(f"Failed to create trade for user {user_id}")
                    return None
            
            logger.info(f"Agent decision logged (no trade executed) for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error executing agent decision for user {user_id}: {e}")
            return None
    
    async def check_stop_loss_triggers(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Check for stop-loss triggers based on current portfolio positions.
        
        Args:
            user_id: User ID
            
        Returns:
            List of triggered stop-loss signals
        """
        try:
            # Get active agent configuration
            agent_response = self.db.from_("agent_allocations").select("*").eq("user_id", user_id).eq("status", "active").execute()
            
            if not agent_response.data:
                return []
            
            agent_config = agent_response.data[0]
            trading_config = agent_config.get("trading_config", {})
            stop_loss_percent = float(trading_config.get("stop_loss_percent", 5.0))
            
            # Get current holdings
            portfolio_response = self.db.from_("portfolios").select("id").eq("user_id", user_id).execute()
            
            if not portfolio_response.data:
                return []
            
            portfolio_id = portfolio_response.data[0]["id"]
            holdings_response = self.db.from_("holdings").select("*").eq("portfolio_id", portfolio_id).execute()
            
            triggered_signals = []
            
            for holding in holdings_response.data or []:
                pnl_percent = float(holding["pnl_usd"]) / float(holding["value_usd"]) * 100 if float(holding["value_usd"]) > 0 else 0
                
                if pnl_percent <= -stop_loss_percent:
                    signal = {
                        "holding_id": holding["id"],
                        "token_symbol": holding["token_symbol"],
                        "current_pnl_percent": pnl_percent,
                        "stop_loss_percent": stop_loss_percent,
                        "recommended_action": "sell",
                        "urgency": "high" if pnl_percent <= -stop_loss_percent * 1.5 else "medium"
                    }
                    triggered_signals.append(signal)
            
            if triggered_signals:
                logger.info(f"Found {len(triggered_signals)} stop-loss triggers for user {user_id}")
            
            return triggered_signals
            
        except Exception as e:
            logger.error(f"Error checking stop-loss triggers for user {user_id}: {e}")
            return []
    
    async def check_take_profit_triggers(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Check for take-profit triggers based on current portfolio positions.
        
        Args:
            user_id: User ID
            
        Returns:
            List of triggered take-profit signals
        """
        try:
            # Get active agent configuration
            agent_response = self.db.from_("agent_allocations").select("*").eq("user_id", user_id).eq("status", "active").execute()
            
            if not agent_response.data:
                return []
            
            agent_config = agent_response.data[0]
            trading_config = agent_config.get("trading_config", {})
            take_profit_percent = float(trading_config.get("take_profit_percent", 20.0))
            
            # Get current holdings
            portfolio_response = self.db.from_("portfolios").select("id").eq("user_id", user_id).execute()
            
            if not portfolio_response.data:
                return []
            
            portfolio_id = portfolio_response.data[0]["id"]
            holdings_response = self.db.from_("holdings").select("*").eq("portfolio_id", portfolio_id).execute()
            
            triggered_signals = []
            
            for holding in holdings_response.data or []:
                pnl_percent = float(holding["pnl_usd"]) / float(holding["value_usd"]) * 100 if float(holding["value_usd"]) > 0 else 0
                
                if pnl_percent >= take_profit_percent:
                    signal = {
                        "holding_id": holding["id"],
                        "token_symbol": holding["token_symbol"],
                        "current_pnl_percent": pnl_percent,
                        "take_profit_percent": take_profit_percent,
                        "recommended_action": "sell",
                        "urgency": "medium"
                    }
                    triggered_signals.append(signal)
            
            if triggered_signals:
                logger.info(f"Found {len(triggered_signals)} take-profit triggers for user {user_id}")
            
            return triggered_signals
            
        except Exception as e:
            logger.error(f"Error checking take-profit triggers for user {user_id}: {e}")
            return []
    
    async def get_agent_performance_summary(self, user_id: str, agent_type: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific agent.
        
        Args:
            user_id: User ID
            agent_type: Agent type
            
        Returns:
            Performance summary dictionary
        """
        try:
            # Get agent trades
            trades_response = self.db.from_("trades").select("*").eq("user_id", user_id).eq("agent_type", agent_type).order("created_at", desc=True).execute()
            
            trades = trades_response.data or []
            
            if not trades:
                return self._default_performance_summary(agent_type)
            
            # Calculate performance metrics
            total_trades = len(trades)
            total_volume = sum(float(trade["value_usd"]) for trade in trades)
            
            # Group trades by day for daily metrics
            daily_volume = {}
            for trade in trades:
                trade_date = trade["created_at"][:10]  # Extract date part
                if trade_date not in daily_volume:
                    daily_volume[trade_date] = 0
                daily_volume[trade_date] += float(trade["value_usd"])
            
            avg_daily_volume = sum(daily_volume.values()) / len(daily_volume) if daily_volume else 0
            
            # Calculate confidence metrics
            confidence_scores = [float(trade["confidence_score"]) for trade in trades if trade["confidence_score"] is not None]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
            
            # Get recent decisions
            decisions_response = self.db.from_("agent_decisions").select("*").eq("user_id", user_id).eq("agent_type", agent_type).order("created_at", desc=True).limit(10).execute()
            
            recent_decisions = decisions_response.data or []
            
            # Calculate win rate (simplified - would need actual P&L tracking)
            win_rate = 0.0  # No real data available yet
            
            performance_summary = {
                "agent_type": agent_type,
                "total_trades": total_trades,
                "total_volume_usd": total_volume,
                "avg_daily_volume": avg_daily_volume,
                "win_rate": win_rate,
                "avg_confidence": avg_confidence,
                "recent_decisions_count": len(recent_decisions),
                "last_trade_date": trades[0]["created_at"] if trades else None,
                "performance_trend": "positive",  # Would calculate from historical data
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Generated performance summary for agent {agent_type} (user {user_id})")
            return performance_summary
            
        except Exception as e:
            logger.error(f"Error getting agent performance summary for user {user_id}, agent {agent_type}: {e}")
            return self._default_performance_summary(agent_type)
    
    async def simulate_market_analysis(self, user_id: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate market analysis and generate trading signals.
        This would be replaced by actual AI/ML models in production.
        
        Args:
            user_id: User ID
            market_data: Current market data
            
        Returns:
            Analysis results and trading signals
        """
        try:
            # Get user's agent configuration
            agent_response = self.db.from_("agent_allocations").select("*").eq("user_id", user_id).eq("status", "active").execute()
            
            if not agent_response.data:
                return {"error": "No active agent found"}
            
            agent_config = agent_response.data[0]
            agent_type = agent_config["agent_type"]
            
            # Simulate different agent behaviors
            analysis = self._simulate_agent_analysis(agent_type, market_data, agent_config)
            
            logger.info(f"Generated market analysis for user {user_id} using agent {agent_type}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error simulating market analysis for user {user_id}: {e}")
            return {"error": str(e)}
    
    def _default_performance_summary(self, agent_type: str) -> Dict[str, Any]:
        """Return default performance summary when no data is available."""
        return {
            "agent_type": agent_type,
            "total_trades": 0,
            "total_volume_usd": 0.0,
            "avg_daily_volume": 0.0,
            "win_rate": 0.0,
            "avg_confidence": None,
            "recent_decisions_count": 0,
            "last_trade_date": None,
            "performance_trend": "neutral",
            "last_updated": datetime.now().isoformat()
        }
    
    def _simulate_agent_analysis(self, agent_type: str, market_data: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent-specific market analysis."""
        
        # Base analysis structure
        analysis = {
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": "neutral",
            "confidence": 0.7,
            "recommended_action": "hold",
            "risk_assessment": "medium",
            "signals": []
        }
        
        # Agent-specific behavior simulation
        if agent_type == "sakura":
            # Conservative agent - prefers stable coins and low-risk trades
            analysis.update({
                "market_sentiment": "cautious",
                "confidence": 0.6,
                "recommended_action": "buy" if market_data.get("volatility", 0) < 0.02 else "hold",
                "risk_assessment": "low",
                "signals": [
                    {"type": "stability_check", "status": "passed"},
                    {"type": "risk_analysis", "level": "low"}
                ]
            })
        
        elif agent_type == "ryu":
            # Balanced agent - moderate risk tolerance
            analysis.update({
                "market_sentiment": "balanced",
                "confidence": 0.75,
                "recommended_action": "buy" if market_data.get("trend", "neutral") == "bullish" else "hold",
                "risk_assessment": "medium",
                "signals": [
                    {"type": "trend_analysis", "direction": market_data.get("trend", "neutral")},
                    {"type": "volume_analysis", "status": "normal"}
                ]
            })
        
        elif agent_type == "yuki":
            # Aggressive agent - high risk, high reward
            analysis.update({
                "market_sentiment": "aggressive",
                "confidence": 0.8,
                "recommended_action": "buy" if market_data.get("momentum", 0) > 0.05 else "sell",
                "risk_assessment": "high",
                "signals": [
                    {"type": "momentum_analysis", "strength": market_data.get("momentum", 0)},
                    {"type": "opportunity_scanner", "count": 3}
                ]
            })
        
        return analysis


# Global service instance
trading_service = TradingService()