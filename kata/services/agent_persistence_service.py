"""
Agent Persistence Service

This service ensures that trading agents continue running on the backend
even when the frontend is reloaded or disconnected.
"""

import asyncio
import logging
import weakref
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"


@dataclass
class AgentInstance:
    """Persistent agent instance."""
    user_id: str
    agent_type: str
    agent_instance: Any
    status: AgentStatus
    started_at: datetime
    last_activity: datetime
    error_count: int = 0
    max_errors: int = 5


class AgentPersistenceService:
    """
    Service to manage persistent agent execution.
    
    This ensures agents continue running on the backend regardless of
    frontend connection status.
    """
    
    def __init__(self):
        self.active_agents: Dict[str, AgentInstance] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.cleanup_interval = 300  # 5 minutes
        
    async def start_persistence_service(self):
        """Start the persistence service."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("🚀 Agent persistence service started")
        
        # Start background cleanup task
        asyncio.create_task(self._cleanup_inactive_agents())
        
    async def stop_persistence_service(self):
        """Stop the persistence service."""
        self.is_running = False
        
        # Stop all active agents
        for agent_id in list(self.active_agents.keys()):
            await self.stop_agent(agent_id)
            
        logger.info("🛑 Agent persistence service stopped")
    
    async def start_agent(self, user_id: str, agent_type: str, agent_instance: Any) -> bool:
        """
        Start a persistent agent.
        
        Args:
            user_id: User ID
            agent_type: Type of agent (yuki, sakura, ryu)
            agent_instance: The agent instance to run
            
        Returns:
            True if agent started successfully
        """
        try:
            agent_id = f"{agent_type}_{user_id}"
            
            # Check if agent is already running
            if agent_id in self.active_agents:
                logger.info(f"Agent {agent_id} is already running")
                return True
            
            # Create agent instance record
            agent_record = AgentInstance(
                user_id=user_id,
                agent_type=agent_type,
                agent_instance=agent_instance,
                status=AgentStatus.STARTING,
                started_at=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.active_agents[agent_id] = agent_record
            
            # Start agent execution task
            task = asyncio.create_task(self._run_agent_persistently(agent_id))
            self.agent_tasks[agent_id] = task
            
            logger.info(f"🚀 Started persistent agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting persistent agent {agent_type} for user {user_id}: {e}")
            return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a persistent agent.
        
        Args:
            agent_id: Agent ID to stop
            
        Returns:
            True if agent stopped successfully
        """
        try:
            if agent_id not in self.active_agents:
                return True
            
            # Cancel the agent task
            if agent_id in self.agent_tasks:
                task = self.agent_tasks[agent_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.agent_tasks[agent_id]
            
            # Remove agent record
            del self.active_agents[agent_id]
            
            logger.info(f"🛑 Stopped persistent agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agent {agent_id}: {e}")
            return False
    
    async def stop_user_agents(self, user_id: str) -> bool:
        """
        Stop all agents for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if all agents stopped successfully
        """
        try:
            agents_to_stop = [
                agent_id for agent_id in self.active_agents.keys()
                if agent_id.endswith(f"_{user_id}")
            ]
            
            for agent_id in agents_to_stop:
                await self.stop_agent(agent_id)
            
            logger.info(f"Stopped {len(agents_to_stop)} agents for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agents for user {user_id}: {e}")
            return False
    
    async def get_agent_status(self, user_id: str, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific agent.
        
        Args:
            user_id: User ID
            agent_type: Agent type
            
        Returns:
            Agent status information
        """
        agent_id = f"{agent_type}_{user_id}"
        
        if agent_id not in self.active_agents:
            return None
        
        agent_record = self.active_agents[agent_id]
        
        return {
            "agent_id": agent_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "status": agent_record.status.value,
            "started_at": agent_record.started_at.isoformat(),
            "last_activity": agent_record.last_activity.isoformat(),
            "error_count": agent_record.error_count,
            "uptime_seconds": int((datetime.now() - agent_record.started_at).total_seconds())
        }
    
    async def get_all_agent_statuses(self) -> Dict[str, Any]:
        """
        Get status of all active agents.
        
        Returns:
            Dictionary of all agent statuses
        """
        statuses = {}
        
        for agent_id, agent_record in self.active_agents.items():
            statuses[agent_id] = {
                "user_id": agent_record.user_id,
                "agent_type": agent_record.agent_type,
                "status": agent_record.status.value,
                "started_at": agent_record.started_at.isoformat(),
                "last_activity": agent_record.last_activity.isoformat(),
                "error_count": agent_record.error_count,
                "uptime_seconds": int((datetime.now() - agent_record.started_at).total_seconds())
            }
        
        return statuses
    
    async def _run_agent_persistently(self, agent_id: str):
        """
        Run an agent persistently in the background.
        
        This method ensures the agent continues running even if the
        frontend disconnects or the page is reloaded.
        """
        try:
            agent_record = self.active_agents[agent_id]
            agent_record.status = AgentStatus.RUNNING
            
            logger.info(f"🔄 Starting persistent execution for agent {agent_id}")
            
            # Run the agent's main loop
            if hasattr(agent_record.agent_instance, 'run'):
                await agent_record.agent_instance.run()
            elif hasattr(agent_record.agent_instance, 'run_trading_cycle'):
                # Fallback to trading cycle if no run method
                while self.is_running and agent_id in self.active_agents:
                    try:
                        await agent_record.agent_instance.run_trading_cycle()
                        agent_record.last_activity = datetime.now()
                        await asyncio.sleep(60)  # Wait 1 minute between cycles
                    except Exception as e:
                        agent_record.error_count += 1
                        agent_record.last_activity = datetime.now()
                        logger.error(f"Error in trading cycle for {agent_id}: {e}")
                        
                        if agent_record.error_count >= agent_record.max_errors:
                            logger.error(f"Agent {agent_id} exceeded max errors, stopping")
                            break
                        
                        await asyncio.sleep(120)  # Wait 2 minutes on error
            else:
                logger.error(f"Agent {agent_id} has no run method")
                agent_record.status = AgentStatus.ERROR
                return
            
        except asyncio.CancelledError:
            logger.info(f"Agent {agent_id} execution cancelled")
        except Exception as e:
            logger.error(f"Fatal error in agent {agent_id}: {e}")
            if agent_id in self.active_agents:
                self.active_agents[agent_id].status = AgentStatus.ERROR
        finally:
            # Clean up
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            if agent_id in self.agent_tasks:
                del self.agent_tasks[agent_id]
    
    async def _cleanup_inactive_agents(self):
        """Background task to clean up inactive agents."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                agents_to_remove = []
                
                for agent_id, agent_record in self.active_agents.items():
                    # Remove agents that haven't had activity in 1 hour
                    if (current_time - agent_record.last_activity) > timedelta(hours=1):
                        agents_to_remove.append(agent_id)
                    # Remove agents with too many errors
                    elif agent_record.error_count >= agent_record.max_errors:
                        agents_to_remove.append(agent_id)
                
                for agent_id in agents_to_remove:
                    logger.info(f"Cleaning up inactive agent {agent_id}")
                    await self.stop_agent(agent_id)
                    
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)


# Global instance
_agent_persistence_service: Optional[AgentPersistenceService] = None


def get_agent_persistence_service() -> AgentPersistenceService:
    """Get the global agent persistence service instance."""
    global _agent_persistence_service
    
    if _agent_persistence_service is None:
        _agent_persistence_service = AgentPersistenceService()
    
    return _agent_persistence_service


async def start_agent_persistence_service():
    """Start the global agent persistence service."""
    service = get_agent_persistence_service()
    await service.start_persistence_service()


async def stop_agent_persistence_service():
    """Stop the global agent persistence service."""
    service = get_agent_persistence_service()
    await service.stop_persistence_service()

