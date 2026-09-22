"""
Agent Manager for Flow AI Trading Platform

Manages the lifecycle and coordination of all trading agents.
Provides centralized control for starting, stopping, and monitoring agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from supabase import Client
from kata.config.database import get_service_client
from kata.agents.base_agent import BaseAgent
from kata.agents.sakura_agent import create_sakura_agent
from kata.agents.ryu_agent import create_ryu_agent
from kata.agents.yuki_agent import create_yuki_agent
from kata.services.hyperliquid_service import HyperliquidService

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Centralized manager for all trading agents.
    
    Responsibilities:
    - Agent lifecycle management (start/stop/restart)
    - Agent configuration management
    - Monitoring and health checks
    - Performance tracking
    - Error handling and recovery
    """
    
    def __init__(self):
        """Initialize the agent manager."""
        self.active_agents: Dict[str, BaseAgent] = {}  # user_id -> agent
        self.agent_configs: Dict[str, Dict[str, Any]] = {}  # user_id -> config
        self.db_client: Client = get_service_client()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Agent factory mapping
        self.agent_factories = {
            'sakura': create_sakura_agent,
            'ryu': create_ryu_agent,
            'yuki': create_yuki_agent
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.last_health_check = datetime.now()
        
        logger.info("Agent Manager initialized")
    
    async def start_manager(self) -> bool:
        """
        Start the agent manager and its monitoring systems.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Agent Manager is already running")
                return False
            
            self.is_running = True
            
            # Load existing agent configurations from database
            await self._load_agent_configurations()
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Agent Manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Agent Manager: {e}")
            self.is_running = False
            return False
    
    async def stop_manager(self) -> bool:
        """
        Stop the agent manager and all active agents.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                logger.warning("Agent Manager is not running")
                return False
            
            self.is_running = False
            
            # Stop monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all active agents
            stop_tasks = []
            for user_id, agent in self.active_agents.items():
                stop_tasks.append(self._stop_agent_safe(user_id, agent))
            
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Agent Manager stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Agent Manager: {e}")
            return False
    
    async def start_agent(self, user_id: str, agent_type: str, config: Dict[str, Any]) -> bool:
        """
        Start a trading agent for a user.
        
        Args:
            user_id: User ID
            agent_type: Type of agent (sakura, ryu, yuki)
            config: Agent configuration
            
        Returns:
            True if agent started successfully, False otherwise
        """
        try:
            # Stop existing agent if running
            if user_id in self.active_agents:
                await self.stop_agent(user_id)
            
            # Validate agent type
            if agent_type not in self.agent_factories:
                logger.error(f"Invalid agent type: {agent_type}")
                return False
            
            # Create agent instance
            agent_factory = self.agent_factories[agent_type]
            agent = agent_factory(user_id, config)
            
            # Start the agent
            if await agent.start_agent():
                self.active_agents[user_id] = agent
                self.agent_configs[user_id] = config
                
                # Update database
                await self._update_agent_config_db(user_id, agent_type, config, True)
                
                # Start agent trading cycle in background
                asyncio.create_task(self._run_agent_cycles(user_id, agent))
                
                logger.info(f"Started {agent_type} agent for user {user_id}")
                return True
            else:
                logger.error(f"Failed to start {agent_type} agent for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting agent for user {user_id}: {e}")
            return False
    
    async def stop_agent(self, user_id: str) -> bool:
        """
        Stop a trading agent for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if agent stopped successfully, False otherwise
        """
        try:
            if user_id not in self.active_agents:
                logger.warning(f"No active agent found for user {user_id}")
                return False
            
            agent = self.active_agents[user_id]
            
            # Stop the agent
            if await agent.stop_agent():
                del self.active_agents[user_id]
                
                # Update database
                await self._deactivate_agent_config_db(user_id)
                
                logger.info(f"Stopped agent for user {user_id}")
                return True
            else:
                logger.error(f"Failed to stop agent for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping agent for user {user_id}: {e}")
            return False
    
    async def restart_agent(self, user_id: str) -> bool:
        """
        Restart a trading agent for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if agent restarted successfully, False otherwise
        """
        try:
            if user_id not in self.active_agents:
                logger.warning(f"No active agent found for user {user_id}")
                return False
            
            agent = self.active_agents[user_id]
            config = self.agent_configs.get(user_id, {})
            agent_type = agent.agent_type
            
            # Stop and start the agent
            await self.stop_agent(user_id)
            return await self.start_agent(user_id, agent_type, config)
            
        except Exception as e:
            logger.error(f"Error restarting agent for user {user_id}: {e}")
            return False
    
    async def get_agent_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get status information for a user's agent.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary containing agent status information
        """
        try:
            if user_id not in self.active_agents:
                return {
                    'user_id': user_id,
                    'is_active': False,
                    'agent_type': None,
                    'status': 'inactive'
                }
            
            agent = self.active_agents[user_id]
            performance = await agent.get_performance_summary()
            
            return {
                'user_id': user_id,
                'is_active': True,
                'agent_type': agent.agent_type,
                'status': 'active',
                'performance': performance,
                'last_execution': agent.last_execution_time.isoformat() if agent.last_execution_time else None,
                'daily_trades': agent.daily_trades_count,
                'daily_pnl': float(agent.daily_pnl),
                'total_trades': agent.total_trades,
                'success_rate': (agent.successful_trades / max(agent.total_trades, 1)) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting agent status for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'is_active': False,
                'status': 'error',
                'error': str(e)
            }
    
    async def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """
        Get status information for all active agents.
        
        Returns:
            List of agent status dictionaries
        """
        try:
            status_list = []
            
            for user_id in self.active_agents.keys():
                status = await self.get_agent_status(user_id)
                status_list.append(status)
            
            return status_list
            
        except Exception as e:
            logger.error(f"Error getting all agents status: {e}")
            return []
    
    async def update_agent_config(self, user_id: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for an active agent.
        
        Args:
            user_id: User ID
            config: New configuration
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if user_id not in self.active_agents:
                logger.warning(f"No active agent found for user {user_id}")
                return False
            
            agent = self.active_agents[user_id]
            
            # Update agent configuration
            agent.config.update(config)
            self.agent_configs[user_id] = agent.config
            
            # Update risk parameters if provided
            if 'risk_params' in config:
                agent.risk_params.__dict__.update(config['risk_params'])
            
            # Update database
            await self._update_agent_config_db(user_id, agent.agent_type, agent.config, True)
            
            logger.info(f"Updated configuration for agent {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent config for user {user_id}: {e}")
            return False
    
    async def execute_manual_cycle(self, user_id: str) -> Dict[str, Any]:
        """
        Manually trigger a trading cycle for an agent.
        
        Args:
            user_id: User ID
            
        Returns:
            Trading cycle results
        """
        try:
            if user_id not in self.active_agents:
                return {
                    'success': False,
                    'error': 'No active agent found'
                }
            
            agent = self.active_agents[user_id]
            results = await agent.run_trading_cycle()
            
            logger.info(f"Manual trading cycle executed for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error executing manual cycle for user {user_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get overall manager statistics.
        
        Returns:
            Dictionary containing manager statistics
        """
        try:
            active_count = len(self.active_agents)
            agent_types = {}
            
            for agent in self.active_agents.values():
                agent_type = agent.agent_type
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            
            return {
                'is_running': self.is_running,
                'active_agents_count': active_count,
                'agent_types_distribution': agent_types,
                'last_health_check': self.last_health_check.isoformat(),
                'total_performance_entries': len(self.performance_metrics),
                'uptime_hours': (datetime.now() - self.last_health_check).total_seconds() / 3600
            }
            
        except Exception as e:
            logger.error(f"Error getting manager statistics: {e}")
            return {'error': str(e)}
    
    # Private methods
    
    async def _load_agent_configurations(self):
        """Load existing agent configurations from database."""
        try:
            response = self.db_client.from_("agent_allocations").select("*").eq("status", "active").execute()
            
            for config_data in response.data or []:
                user_id = config_data['user_id']
                agent_type = config_data['agent_type']
                trading_config = config_data.get('trading_config', {})
                config = trading_config.copy()
                
                # Merge with risk parameters from database
                risk_params = {
                    'max_position_size_percent': trading_config.get('max_position_size_percent', 15.0),
                    'stop_loss_percent': trading_config.get('stop_loss_percent', 5.0),
                    'take_profit_percent': trading_config.get('take_profit_percent', 20.0)
                }
                config['risk_params'] = risk_params
                
                self.agent_configs[user_id] = config
                
                logger.info(f"Loaded configuration for {agent_type} agent (user: {user_id})")
            
        except Exception as e:
            logger.error(f"Error loading agent configurations: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for agent health and performance."""
        try:
            while self.is_running:
                await asyncio.sleep(60)  # Check every minute
                
                # Health check for all active agents
                for user_id, agent in list(self.active_agents.items()):
                    try:
                        # Check if agent is still healthy
                        if not agent.is_running:
                            logger.warning(f"Agent for user {user_id} is not running, removing from active list")
                            del self.active_agents[user_id]
                            continue
                        
                        # Check for excessive errors
                        if len(agent.execution_errors) > 10:
                            logger.warning(f"Agent for user {user_id} has too many errors, considering restart")
                            # Could implement automatic restart logic here
                        
                        # Update performance metrics
                        performance = await agent.get_performance_summary()
                        self.performance_metrics[user_id] = {
                            'timestamp': datetime.now(),
                            'performance': performance
                        }
                        
                    except Exception as e:
                        logger.error(f"Error monitoring agent for user {user_id}: {e}")
                
                self.last_health_check = datetime.now()
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _run_agent_cycles(self, user_id: str, agent: BaseAgent):
        """Run trading cycles for an agent in background."""
        try:
            while user_id in self.active_agents and agent.is_running:
                try:
                    # Run trading cycle
                    results = await agent.run_trading_cycle()
                    
                    if not results['success']:
                        logger.warning(f"Trading cycle failed for user {user_id}: {results.get('errors', [])}")
                    
                    # Wait before next cycle (configurable interval)
                    cycle_interval = agent.config.get('cycle_interval_minutes', 15)
                    await asyncio.sleep(cycle_interval * 60)
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle for user {user_id}: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
                    
        except asyncio.CancelledError:
            logger.info(f"Agent cycles cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"Error running agent cycles for user {user_id}: {e}")
    
    async def _stop_agent_safe(self, user_id: str, agent: BaseAgent):
        """Safely stop an agent with error handling."""
        try:
            await agent.stop_agent()
        except Exception as e:
            logger.error(f"Error stopping agent for user {user_id}: {e}")
    
    async def _update_agent_config_db(
        self, 
        user_id: str, 
        agent_type: str, 
        config: Dict[str, Any], 
        is_active: bool
    ):
        """Update agent configuration in database."""
        try:
            status = 'active' if is_active else 'paused'
            
            # Check if config exists
            existing_response = self.db_client.from_("agent_allocations").select("id").eq("user_id", user_id).eq("agent_type", agent_type).execute()
            
            if existing_response.data:
                # Update existing
                config_id = existing_response.data[0]['id']
                self.db_client.from_("agent_allocations").update({'status': status, 'trading_config': config}).eq("id", config_id).execute()
            else:
                # Insert new (rare because allocations are created in the allocation service, but we handle it just in case)
                config_data = {
                    'user_id': user_id,
                    'agent_type': agent_type,
                    'status': status,
                    'trading_config': config,
                    'allocated_amount': 0.0,
                    'remaining_amount': 0.0
                }
                self.db_client.from_("agent_allocations").insert(config_data).execute()
                
        except Exception as e:
            logger.error(f"Error updating agent config in database: {e}")
    
    async def _deactivate_agent_config_db(self, user_id: str):
        """Deactivate agent configuration in database."""
        try:
            self.db_client.from_("agent_allocations").update({"status": "paused"}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"Error deactivating agent config in database: {e}")


# Global agent manager instance
agent_manager = AgentManager()