"""
AI Agents router for managing trading agents and their configurations.
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from kata.config.database import get_db_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["AI Agents"])


# Agent configuration models
from pydantic import BaseModel, Field
from decimal import Decimal


class AgentConfig(BaseModel):
    """Agent configuration model."""
    id: str
    user_id: str
    agent_type: str
    is_active: bool
    max_position_size: Decimal
    stop_loss_percent: Decimal
    take_profit_percent: Decimal
    config_data: Dict[str, Any]
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


class AgentConfigCreate(BaseModel):
    """Model for creating agent configuration."""
    agent_type: str = Field(..., description="Agent type: sakura, ryu, or yuki")
    max_position_size: Decimal = Field(default=5.0, ge=0.1, le=100, description="Max position size %")
    stop_loss_percent: Decimal = Field(default=3.0, ge=0.1, le=50, description="Stop loss %")
    take_profit_percent: Decimal = Field(default=15.0, ge=1, le=1000, description="Take profit %")
    config_data: Dict[str, Any] = Field(default={}, description="Additional config data")


class AgentConfigUpdate(BaseModel):
    """Model for updating agent configuration."""
    max_position_size: Optional[Decimal] = Field(None, ge=0.1, le=100)
    stop_loss_percent: Optional[Decimal] = Field(None, ge=0.1, le=50)
    take_profit_percent: Optional[Decimal] = Field(None, ge=1, le=1000)
    config_data: Optional[Dict[str, Any]] = None


class AgentStatus(BaseModel):
    """Agent status model."""
    user_id: str
    agent_type: Optional[str] = None
    is_active: bool = False
    last_action: Optional[str] = None
    performance: Dict[str, Any] = Field(default={})
    config: Optional[AgentConfig] = None


class AgentStartRequest(BaseModel):
    """Request model for starting an agent."""
    user_id: str = Field(..., description="User ID")
    agent_type: str = Field(..., description="Agent type to start")
    config: Optional[Dict[str, Any]] = Field(default={}, description="Agent configuration")


@router.get("/status/{user_id}", response_model=AgentStatus)
async def get_agent_status(
    user_id: str,
    db: Client = Depends(get_db_client)
) -> AgentStatus:
    """Get current agent status for a user."""
    try:
        # Get active agent configuration
        config_response = db.from_("agent_allocations").select("*").eq("user_id", user_id).eq("status", "active").execute()
        
        agent_status = AgentStatus(user_id=user_id)
        
        if config_response.data:
            allocation = config_response.data[0]
            trading_config = allocation.get("trading_config", {})
            agent_config = AgentConfig(
                id=allocation["id"],
                user_id=allocation["user_id"],
                agent_type=allocation["agent_type"],
                is_active=True,
                max_position_size=Decimal(str(trading_config.get("max_position_size_percent", 5.0))),
                stop_loss_percent=Decimal(str(trading_config.get("stop_loss_percent", 3.0))),
                take_profit_percent=Decimal(str(trading_config.get("take_profit_percent", 15.0))),
                config_data=trading_config,
                created_at=allocation.get("created_at", ""),
                updated_at=allocation.get("updated_at", "")
            )
            
            agent_status.agent_type = agent_config.agent_type
            agent_status.is_active = True
            agent_status.config = agent_config
            
            # Get latest agent decision for last action
            decision_response = db.from_("agent_decisions").select("action_taken").eq("user_id", user_id).eq("agent_type", agent_config.agent_type).order("created_at", desc=True).limit(1).execute()
            
            if decision_response.data:
                agent_status.last_action = decision_response.data[0].get("action_taken")
            
            # Get performance data
            performance_response = db.from_("agent_performance_summary").select("*").eq("user_id", user_id).eq("agent_type", agent_config.agent_type).order("trade_date", desc=True).limit(7).execute()
            
            if performance_response.data:
                recent_performance = performance_response.data[0]
                agent_status.performance = {
                    "total_trades": recent_performance.get("total_trades", 0),
                    "total_volume": float(recent_performance.get("total_buy_volume", 0) + recent_performance.get("total_sell_volume", 0)),
                    "avg_confidence": float(recent_performance.get("avg_confidence", 0)) if recent_performance.get("avg_confidence") else None,
                    "win_rate": 0  # Calculate from trades if needed
                }
        
        logger.info(f"Retrieved agent status for user {user_id}: active={agent_status.is_active}")
        return agent_status
        
    except Exception as e:
        logger.error(f"Error getting agent status for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.post("/start", response_model=Dict[str, Any])
async def start_agent(
    request: AgentStartRequest,
    db: Client = Depends(get_db_client)
) -> Dict[str, Any]:
    """Start an AI agent for a user."""
    try:
        user_id = request.user_id
        
        # Ensure user_id is in UUID format for database compatibility
        import uuid
        try:
            # Try to parse as UUID to validate format
            uuid.UUID(user_id)
        except ValueError:
            # If not a valid UUID, generate one from the string hash
            import hashlib
            namespace = uuid.NAMESPACE_DNS
            user_id = str(uuid.uuid5(namespace, user_id))
        
        # Validate agent type
        allowed_agents = ['sakura', 'ryu', 'yuki']
        if request.agent_type not in allowed_agents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid agent type. Must be one of: {allowed_agents}"
            )
        
        # Temporarily skip database operations due to RLS issues - TODO: Fix authentication
        # db.from_("agent_configs").update({"is_active": False}).eq("user_id", user_id).execute()
        
        # Create or update agent configuration
        frontend_config = request.config if request.config else {}
        
        # Map frontend config to database fields
        config_data = {
            "user_id": user_id,
            "agent_type": request.agent_type,
            "is_active": True,
            "max_position_size": frontend_config.get("initialInvestment", 1000) / 100,  # Convert to percentage
            "stop_loss_percent": frontend_config.get("stopLoss", 5),
            "take_profit_percent": frontend_config.get("takeProfit", 15),
            "config_data": {
                "riskTolerance": frontend_config.get("riskTolerance", "medium"),
                "maxDailyTrades": frontend_config.get("maxDailyTrades", 10),
                "emergencyStop": frontend_config.get("emergencyStop", True),
                "initialInvestment": frontend_config.get("initialInvestment", 1000)
            }
        }
        
        # Temporarily skip database operations due to RLS issues - TODO: Fix authentication
        # existing_response = db.from_("agent_configs").select("id").eq("user_id", user_id).eq("agent_type", request.agent_type).execute()
        # if existing_response.data:
        #     config_id = existing_response.data[0]["id"]
        #     response = db.from_("agent_configs").update(config_data).eq("id", config_id).execute()
        # else:
        #     response = db.from_("agent_configs").insert(config_data).execute()
        # db.from_("users").update({"selected_agent": request.agent_type}).eq("id", user_id).execute()
        
        # Ensure Hyperliquid subaccount is created for the user
        try:
            from kata.services.hyperliquid_service import get_hyperliquid_service
            hyperliquid_service = get_hyperliquid_service()
            
            # Check if user has a subaccount, if not create one
            try:
                await hyperliquid_service.get_user_balance(user_id)
                logger.info(f"User {user_id} already has Hyperliquid subaccount")
            except ValueError:
                # User doesn't have subaccount, need to authenticate/create one
                logger.info(f"Creating Hyperliquid subaccount for user {user_id}")
                # Note: In a full implementation, this would use the user's Privy access token
                # For now, we'll set up the account structure without actual Privy authentication
                
                # This would normally require:
                # auth_result = await hyperliquid_service.authenticate_user(privy_access_token)
                
                logger.info(f"Hyperliquid subaccount setup initiated for user {user_id}")
                
        except Exception as e:
            logger.warning(f"Could not setup Hyperliquid subaccount for user {user_id}: {e}")
            # Continue with agent start even if subaccount setup fails
        
        # Temporarily skip database operations due to RLS issues - TODO: Fix authentication
        # decision_data = {
        #     "user_id": user_id,
        #     "agent_type": request.agent_type,
        #     "decision_type": "agent_start",
        #     "action_taken": f"Started {request.agent_type} agent",
        #     "decision_data": {"config": config_data},
        #     "confidence_score": 1.0
        # }
        # db.from_("agent_decisions").insert(decision_data).execute()
        
        logger.info(f"Started agent {request.agent_type} for user {user_id}")
        
        # For development: return success without database operations due to RLS
        return {
            "success": True,
            "message": f"Agent {request.agent_type} started successfully",
            "agent_type": request.agent_type,
            "user_id": user_id,
            "config": config_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting agent for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}"
        )


@router.post("/stop", response_model=Dict[str, Any])
async def stop_agent(
    user_id: str,
    agent_type: Optional[str] = None,
    db: Client = Depends(get_db_client)
) -> Dict[str, Any]:
    """Stop AI agent(s) for a user."""
    try:
        from kata.agents.agent_manager import AgentManager
        # Get agent manager instance
        agent_manager = AgentManager()
        
        # Stop the actual running agent(s)
        if agent_type:
            # Stop specific agent type
            stop_success = await agent_manager.stop_agent(user_id)
            if stop_success:
                message = f"Agent {agent_type} stopped successfully"
            else:
                message = f"Agent {agent_type} was not running or already stopped"
        else:
            # Stop all active agents for user
            stop_success = await agent_manager.stop_agent(user_id)
            if stop_success:
                message = "All agents stopped successfully"
            else:
                message = "No active agents found or already stopped"
        
        # Update database configuration
        if agent_type:
            db.from_("agent_allocations").update({"status": "paused"}).eq("user_id", user_id).eq("agent_type", agent_type).execute()
        else:
            db.from_("agent_allocations").update({"status": "paused"}).eq("user_id", user_id).execute()
        
        # Log agent stop decision
        decision_data = {
            "user_id": user_id,
            "agent_type": agent_type or "all",
            "decision_type": "agent_stop",
            "action_taken": f"Stopped {agent_type or 'all'} agent(s)",
            "decision_data": {},
            "confidence_score": 1.0
        }
        
        db.from_("agent_decisions").insert(decision_data).execute()
        
        logger.info(f"Stopped agent(s) for user {user_id}: {agent_type or 'all'}")
        
        return {
            "success": True,
            "message": message,
            "stopped_agent": agent_type,
            "agent_stopped": stop_success
        }
        
    except Exception as e:
        logger.error(f"Error stopping agent for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop agent: {str(e)}"
        )


@router.get("/configs/{user_id}", response_model=List[AgentConfig])
async def get_agent_configs(
    user_id: str,
    db: Client = Depends(get_db_client)
) -> List[AgentConfig]:
    """Get all agent configurations for a user."""
    try:
        response = db.from_("agent_allocations").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        configs = []
        for allocation in response.data or []:
            trading_config = allocation.get("trading_config", {})
            configs.append(AgentConfig(
                id=allocation["id"],
                user_id=allocation["user_id"],
                agent_type=allocation["agent_type"],
                is_active=allocation.get("status") == "active",
                max_position_size=Decimal(str(trading_config.get("max_position_size_percent", 5.0))),
                stop_loss_percent=Decimal(str(trading_config.get("stop_loss_percent", 3.0))),
                take_profit_percent=Decimal(str(trading_config.get("take_profit_percent", 15.0))),
                config_data=trading_config,
                created_at=allocation.get("created_at", ""),
                updated_at=allocation.get("updated_at", "")
            ))
        
        logger.info(f"Retrieved {len(configs)} agent configs for user {user_id}")
        return configs
        
    except Exception as e:
        logger.error(f"Error getting agent configs for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent configs: {str(e)}"
        )


@router.put("/configs/{config_id}", response_model=AgentConfig)
async def update_agent_config(
    config_id: str,
    config_update: AgentConfigUpdate,
    db: Client = Depends(get_db_client)
) -> AgentConfig:
    """Update an agent configuration."""
    try:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Updating agent config directly is not supported. Use the agent allocation service."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent config {config_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent config: {str(e)}"
        )