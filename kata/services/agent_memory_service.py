"""
Agent Memory Service for Flow AI Platform

Manages active goals, episodic working memory, and self-reflection insights
for Yuki (and other agents), enabling persistent context across decision cycles.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


@dataclass
class AgentGoal:
    """Active trading objective for an agent."""
    id: Optional[str]
    agent_id: str
    goal_statement: str
    status: str = "active"  # 'active', 'achieved', 'abandoned'
    target_metric: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class AgentEpisode:
    """Episodic memory item capturing a single decision step."""
    id: Optional[str]
    agent_id: str
    symbol: Optional[str]
    market_regime: Optional[str]
    thought_chain: str
    tool_calls: List[Dict[str, Any]]
    action_taken: str
    confidence: float
    observation: Optional[str] = None
    trade_id: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class AgentReflection:
    """Distilled lesson learned from post-trade review or weekly review."""
    id: Optional[str]
    agent_id: str
    review_type: str  # 'trade_postmortem', 'weekly_review'
    symbol: Optional[str]
    lesson_learned: str
    proposed_rule_changes: Dict[str, Any]
    applied: bool = False
    created_at: Optional[datetime] = None


class AgentMemoryService:
    """
    Persistent and working memory manager for trading agents.
    Backs memory up in Supabase with in-memory caching.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.db = get_service_client()
        
        # Local in-memory caches as fallback/speedups
        self._in_memory_episodes: List[AgentEpisode] = []
        self._in_memory_goals: List[AgentGoal] = [
            AgentGoal(
                id="default_goal",
                agent_id=agent_id,
                goal_statement="Maximize risk-adjusted returns on Hyperliquid perps while strictly managing liquidation buffers and downside drawdown.",
                status="active"
            )
        ]
        self._in_memory_reflections: List[AgentReflection] = []

    async def get_active_goals(self) -> List[AgentGoal]:
        """Fetch all currently active operational goals for this agent."""
        try:
            res = self.db.table('agent_goals').select('*').eq('agent_id', self.agent_id).eq('status', 'active').order('created_at', desc=True).execute()
            if res.data:
                return [
                    AgentGoal(
                        id=row.get('id'),
                        agent_id=row.get('agent_id'),
                        goal_statement=row.get('goal_statement'),
                        status=row.get('status', 'active'),
                        target_metric=row.get('target_metric'),
                        target_value=float(row.get('target_value')) if row.get('target_value') is not None else None,
                        current_value=float(row.get('current_value')) if row.get('current_value') is not None else None,
                        created_at=row.get('created_at')
                    )
                    for row in res.data
                ]
        except Exception as e:
            logger.debug(f"Could not load goals from DB (falling back to memory): {e}")
            
        return [g for g in self._in_memory_goals if g.status == "active"]

    async def set_goal(self, goal_statement: str, target_metric: Optional[str] = None, target_value: Optional[float] = None) -> AgentGoal:
        """Set a new operational goal for the agent."""
        goal = AgentGoal(
            id=None,
            agent_id=self.agent_id,
            goal_statement=goal_statement,
            status="active",
            target_metric=target_metric,
            target_value=target_value,
            current_value=0.0,
            created_at=datetime.now()
        )
        
        try:
            res = self.db.table('agent_goals').insert({
                'agent_id': self.agent_id,
                'goal_statement': goal_statement,
                'status': 'active',
                'target_metric': target_metric,
                'target_value': target_value,
                'current_value': 0.0
            }).execute()
            if res.data:
                goal.id = res.data[0].get('id')
        except Exception as e:
            logger.debug(f"Could not save goal to DB: {e}")
            
        self._in_memory_goals.append(goal)
        return goal

    async def append_episode(
        self,
        symbol: str,
        thought_chain: str,
        tool_calls: List[Dict[str, Any]],
        action_taken: str,
        confidence: float,
        market_regime: str = "normal",
        observation: str = "",
        trade_id: Optional[str] = None
    ) -> AgentEpisode:
        """Append a decision step / episode to working memory."""
        episode = AgentEpisode(
            id=None,
            agent_id=self.agent_id,
            symbol=symbol,
            market_regime=market_regime,
            thought_chain=thought_chain,
            tool_calls=tool_calls,
            action_taken=action_taken,
            confidence=confidence,
            observation=observation,
            trade_id=trade_id,
            created_at=datetime.now()
        )

        try:
            payload = {
                'agent_id': self.agent_id,
                'symbol': symbol,
                'market_regime': market_regime,
                'thought_chain': thought_chain,
                'tool_calls': tool_calls,
                'action_taken': action_taken,
                'confidence': confidence,
                'observation': observation,
                'trade_id': trade_id
            }
            res = self.db.table('agent_episodes').insert(payload).execute()
            if res.data:
                episode.id = res.data[0].get('id')
        except Exception as e:
            logger.debug(f"Could not save episode to DB: {e}")

        self._in_memory_episodes.append(episode)
        # Keep last 50 episodes in memory buffer
        if len(self._in_memory_episodes) > 50:
            self._in_memory_episodes = self._in_memory_episodes[-50:]
            
        return episode

    async def get_recent_episodes(self, symbol: Optional[str] = None, limit: int = 10) -> List[AgentEpisode]:
        """Fetch recent episodic memory entries."""
        try:
            query = self.db.table('agent_episodes').select('*').eq('agent_id', self.agent_id)
            if symbol:
                query = query.eq('symbol', symbol)
            res = query.order('created_at', desc=True).limit(limit).execute()
            if res.data:
                return [
                    AgentEpisode(
                        id=row.get('id'),
                        agent_id=row.get('agent_id'),
                        symbol=row.get('symbol'),
                        market_regime=row.get('market_regime'),
                        thought_chain=row.get('thought_chain', ''),
                        tool_calls=row.get('tool_calls', []),
                        action_taken=row.get('action_taken', ''),
                        confidence=float(row.get('confidence', 0.5)),
                        observation=row.get('observation', ''),
                        trade_id=row.get('trade_id'),
                        created_at=row.get('created_at')
                    )
                    for row in res.data
                ]
        except Exception as e:
            logger.debug(f"Could not fetch episodes from DB: {e}")

        episodes = self._in_memory_episodes
        if symbol:
            episodes = [e for e in episodes if e.symbol == symbol]
        return episodes[-limit:]

    async def save_reflection(
        self,
        lesson_learned: str,
        proposed_rule_changes: Optional[Dict[str, Any]] = None,
        review_type: str = "trade_postmortem",
        symbol: Optional[str] = None
    ) -> AgentReflection:
        """Save a self-improvement reflection / lesson learned."""
        rule_changes = proposed_rule_changes or {}
        reflection = AgentReflection(
            id=None,
            agent_id=self.agent_id,
            review_type=review_type,
            symbol=symbol,
            lesson_learned=lesson_learned,
            proposed_rule_changes=rule_changes,
            applied=True,
            created_at=datetime.now()
        )

        try:
            payload = {
                'agent_id': self.agent_id,
                'review_type': review_type,
                'symbol': symbol,
                'lesson_learned': lesson_learned,
                'proposed_rule_changes': rule_changes,
                'applied': True
            }
            res = self.db.table('agent_reflections').insert(payload).execute()
            if res.data:
                reflection.id = res.data[0].get('id')
        except Exception as e:
            logger.debug(f"Could not save reflection to DB: {e}")

        self._in_memory_reflections.append(reflection)
        return reflection

    async def get_active_lessons(self, symbol: Optional[str] = None, limit: int = 5) -> List[str]:
        """Fetch active lessons learned to include in ReAct reasoning context."""
        try:
            query = self.db.table('agent_reflections').select('*').eq('agent_id', self.agent_id)
            res = query.order('created_at', desc=True).limit(limit).execute()
            if res.data:
                return [row.get('lesson_learned') for row in res.data if row.get('lesson_learned')]
        except Exception as e:
            logger.debug(f"Could not fetch reflections from DB: {e}")

        return [r.lesson_learned for r in self._in_memory_reflections[-limit:]]


def get_agent_memory_service(agent_id: str = "yuki") -> AgentMemoryService:
    """Factory helper to get agent memory service."""
    return AgentMemoryService(agent_id)
