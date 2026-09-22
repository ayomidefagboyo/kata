"""
Multi-Level Learning Worker - Advanced Background Learning Pipeline

This worker orchestrates the 3-level learning architecture:
- Level 1: Real-Time Learning (every 30 seconds)
- Level 2: Pattern Learning (every 15 minutes)
- Level 3: Meta-Learning (daily)

Designed to dramatically improve agent performance through sophisticated learning loops.
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from kata.config.database import get_service_client
from kata.services.multi_level_learning_service import get_multi_level_learning_service, LearningLevel
from kata.services.platform_signal_service import get_platform_signal_service, PlatformSignal
from kata.services.agent_learning_service import get_agent_learning_service

logger = logging.getLogger(__name__)


class MultiLevelLearningWorker:
    """Advanced multi-level learning orchestrator."""

    def __init__(self):
        """Initialize the multi-level learning worker."""
        self.db = get_service_client()
        self.platform_service = get_platform_signal_service()

        # Multi-level learning services for each agent
        self.ml_services = {
            'yuki': get_multi_level_learning_service('yuki'),
            'ryu': get_multi_level_learning_service('ryu'),
            'sakura': get_multi_level_learning_service('sakura')
        }

        # Legacy learning services (for compatibility)
        self.legacy_services = {
            'yuki': get_agent_learning_service('yuki'),
            'ryu': get_agent_learning_service('ryu'),
            'sakura': get_agent_learning_service('sakura')
        }

        # Worker state
        self.is_running = False
        self.last_level1_update = {}
        self.last_level2_update = {}
        self.last_level3_update = {}
        self.maintenance_lock = None

        # Configuration
        self.config = {
            'level1_interval': 30,      # seconds - Real-time learning
            'level2_interval': 900,     # seconds - Pattern learning (15 min)
            'level3_interval': 86400,   # seconds - Meta-learning (daily)
            'max_concurrent_tasks': 10,
            'error_retry_delay': 60,
            'health_check_interval': 300
        }

        logger.info("🧠 Multi-Level Learning Worker initialized")

    def set_maintenance_lock(self, maintenance_lock):
        """Serialize heavy learning passes with discovery market maintenance."""
        self.maintenance_lock = maintenance_lock

    async def start(self):
        """Start the multi-level learning worker."""
        if self.is_running:
            logger.warning("⚠️ Worker already running")
            return

        self.is_running = True
        logger.info("🚀 Starting Multi-Level Learning Worker...")

        try:
            # Initialize all services
            await self._initialize_services()

            # Start learning loops
            await asyncio.gather(
                self._level1_learning_loop(),
                self._level2_learning_loop(),
                self._level3_learning_loop(),
                self._health_monitor_loop(),
                return_exceptions=True
            )

        except Exception as e:
            logger.error(f"❌ Worker startup failed: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the multi-level learning worker."""
        logger.info("🛑 Stopping Multi-Level Learning Worker...")
        self.is_running = False

        # Wait for current tasks to complete
        await asyncio.sleep(2)
        logger.info("✅ Multi-Level Learning Worker stopped")

    async def _initialize_services(self):
        """Initialize all learning services."""
        try:
            for agent_type in self.ml_services:
                # Initialize service state
                self.last_level1_update[agent_type] = datetime.now()
                self.last_level2_update[agent_type] = datetime.now() - timedelta(minutes=10)  # Trigger initial run
                self.last_level3_update[agent_type] = datetime.now() - timedelta(hours=20)    # Trigger initial run

                # Reconcile stale tracking rows left by older worker versions.
                reconcile_summary = await self.ml_services[agent_type].reconcile_tracking_records()
                if reconcile_summary.get('completed_rows', 0) or reconcile_summary.get('stopped_rows', 0):
                    logger.info(
                        f"🧹 Reconciled tracking rows for {agent_type}: "
                        f"{reconcile_summary.get('completed_rows', 0)} completed, "
                        f"{reconcile_summary.get('stopped_rows', 0)} stopped"
                    )

                logger.info(f"✅ {agent_type} multi-level learning service initialized")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    # ===== LEVEL 1: REAL-TIME LEARNING LOOP =====

    async def _level1_learning_loop(self):
        """Real-time learning loop (every 30 seconds)."""
        logger.info("🔄 Starting Level 1 Real-Time Learning Loop")

        while self.is_running:
            try:
                await asyncio.sleep(self.config['level1_interval'])

                # Process real-time learning for all agents
                tasks = []
                for agent_type, ml_service in self.ml_services.items():
                    if self._should_run_level1_update(agent_type):
                        task = asyncio.create_task(
                            self._run_level1_update(agent_type, ml_service)
                        )
                        tasks.append(task)

                # Execute with concurrency limit
                if tasks:
                    await self._execute_with_limit(tasks, self.config['max_concurrent_tasks'])

            except Exception as e:
                logger.error(f"❌ Level 1 learning loop error: {e}")
                await asyncio.sleep(self.config['error_retry_delay'])

    async def _run_level1_update(self, agent_type: str, ml_service):
        """Run Level 1 real-time learning update."""
        try:
            # Get active signals for this agent
            active_signals = await self._get_active_signals(agent_type)

            for signal in active_signals:
                # Start or update real-time tracking
                if signal.signal_id not in ml_service.active_signals:
                    await ml_service.start_real_time_tracking(signal)
                    logger.debug(f"📊 Started real-time tracking for {signal.token_symbol}")

            self.last_level1_update[agent_type] = datetime.now()

        except Exception as e:
            logger.error(f"❌ Level 1 update failed for {agent_type}: {e}")

    def _should_run_level1_update(self, agent_type: str) -> bool:
        """Check if Level 1 update should run for agent."""
        last_update = self.last_level1_update.get(agent_type)
        if not last_update:
            return True

        elapsed = (datetime.now() - last_update).total_seconds()
        return elapsed >= self.config['level1_interval']

    # ===== LEVEL 2: PATTERN LEARNING LOOP =====

    async def _level2_learning_loop(self):
        """Pattern learning loop (every 15 minutes)."""
        logger.info("🔄 Starting Level 2 Pattern Learning Loop")

        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Process pattern learning for all agents
                tasks = []
                for agent_type, ml_service in self.ml_services.items():
                    if self._should_run_level2_update(agent_type):
                        task = asyncio.create_task(
                            self._run_level2_update(agent_type, ml_service)
                        )
                        tasks.append(task)

                # Execute with concurrency limit
                if tasks:
                    await self._execute_with_limit(tasks, self.config['max_concurrent_tasks'])

            except Exception as e:
                logger.error(f"❌ Level 2 learning loop error: {e}")
                await asyncio.sleep(self.config['error_retry_delay'])

    async def _run_level2_update(self, agent_type: str, ml_service):
        """Run Level 2 pattern learning update."""
        try:
            if self.maintenance_lock is not None:
                async with self.maintenance_lock:
                    await self._perform_level2_update(agent_type, ml_service)
            else:
                await self._perform_level2_update(agent_type, ml_service)
        except Exception as e:
            logger.error(f"❌ Level 2 update failed for {agent_type}: {e}")

    async def _perform_level2_update(self, agent_type: str, ml_service):
        logger.info(f"🔄 Running Level 2 Pattern Learning for {agent_type}")

        # Update enhanced pattern learning
        await ml_service.update_pattern_learning()

        # Also update legacy learning system for compatibility
        legacy_service = self.legacy_services[agent_type]
        await self._update_legacy_patterns(legacy_service)

        self.last_level2_update[agent_type] = datetime.now()
        logger.info(f"✅ Level 2 Pattern Learning completed for {agent_type}")

    def _should_run_level2_update(self, agent_type: str) -> bool:
        """Check if Level 2 update should run for agent."""
        last_update = self.last_level2_update.get(agent_type)
        if not last_update:
            return True

        elapsed = (datetime.now() - last_update).total_seconds()
        return elapsed >= self.config['level2_interval']

    # ===== LEVEL 3: META-LEARNING LOOP =====

    async def _level3_learning_loop(self):
        """Meta-learning loop (daily)."""
        logger.info("🔄 Starting Level 3 Meta-Learning Loop")

        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Process meta-learning for all agents
                tasks = []
                for agent_type, ml_service in self.ml_services.items():
                    if self._should_run_level3_update(agent_type):
                        task = asyncio.create_task(
                            self._run_level3_update(agent_type, ml_service)
                        )
                        tasks.append(task)

                # Execute with concurrency limit
                if tasks:
                    await self._execute_with_limit(tasks, 1)  # Meta-learning runs sequentially

            except Exception as e:
                logger.error(f"❌ Level 3 learning loop error: {e}")
                await asyncio.sleep(self.config['error_retry_delay'])

    async def _run_level3_update(self, agent_type: str, ml_service):
        """Run Level 3 meta-learning update."""
        try:
            if self.maintenance_lock is not None:
                async with self.maintenance_lock:
                    await self._perform_level3_update(agent_type, ml_service)
            else:
                await self._perform_level3_update(agent_type, ml_service)
        except Exception as e:
            logger.error(f"❌ Level 3 update failed for {agent_type}: {e}")

    async def _perform_level3_update(self, agent_type: str, ml_service):
        logger.info(f"🧠 Running Level 3 Meta-Learning for {agent_type}")

        # Update meta-learning insights
        insights = await ml_service.update_meta_learning()

        if insights:
            # Log key insights
            logger.info(f"📊 Meta-Learning Insights for {agent_type}:")
            logger.info(f"   Strategy Effectiveness: {insights.strategy_effectiveness}")
            logger.info(f"   Learning Convergence: {insights.learning_convergence_rate:.3f}")
            logger.info(f"   Adaptation Recommendations: {len(insights.adaptation_recommendations)}")

            # Store insights for dashboard
            await self._store_meta_insights(agent_type, insights)

        self.last_level3_update[agent_type] = datetime.now()
        logger.info(f"✅ Level 3 Meta-Learning completed for {agent_type}")

    def _should_run_level3_update(self, agent_type: str) -> bool:
        """Check if Level 3 update should run for agent."""
        last_update = self.last_level3_update.get(agent_type)
        if not last_update:
            return True

        elapsed = (datetime.now() - last_update).total_seconds()
        return elapsed >= self.config['level3_interval']

    # ===== HEALTH MONITORING =====

    async def _health_monitor_loop(self):
        """Monitor worker health and performance."""
        logger.info("💚 Starting Health Monitor Loop")

        while self.is_running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])

                # Check service health
                health_status = await self._check_health()

                if not health_status['healthy']:
                    logger.warning(f"⚠️ Health check failed: {health_status['issues']}")
                    # Attempt recovery
                    await self._attempt_recovery(health_status['issues'])

                # Log performance metrics
                await self._log_performance_metrics()

            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(60)

    async def _check_health(self) -> Dict[str, Any]:
        """Check overall worker health."""
        issues = []
        healthy = True

        try:
            # Check if loops are running
            for agent_type in self.ml_services:
                # Level 1 health
                level1_elapsed = (datetime.now() - self.last_level1_update.get(agent_type, datetime.now())).total_seconds()
                if level1_elapsed > self.config['level1_interval'] * 3:  # 3x expected interval
                    issues.append(f"{agent_type} Level 1 not updating (last: {level1_elapsed:.0f}s ago)")
                    healthy = False

                # Level 2 health
                level2_elapsed = (datetime.now() - self.last_level2_update.get(agent_type, datetime.now())).total_seconds()
                if level2_elapsed > self.config['level2_interval'] * 2:  # 2x expected interval
                    issues.append(f"{agent_type} Level 2 not updating (last: {level2_elapsed:.0f}s ago)")

            # Check database connectivity
            try:
                self.db.from_('platform_signals').select('id').limit(1).execute()
            except Exception as e:
                issues.append(f"Database connectivity issue: {e}")
                healthy = False

        except Exception as e:
            issues.append(f"Health check error: {e}")
            healthy = False

        return {
            'healthy': healthy,
            'issues': issues,
            'timestamp': datetime.now()
        }

    async def _attempt_recovery(self, issues: List[str]):
        """Attempt to recover from health issues."""
        logger.info(f"🔧 Attempting recovery for issues: {issues}")

        try:
            # Reinitialize services if needed
            await self._initialize_services()
            logger.info("✅ Recovery attempt completed")
        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")

    # ===== UTILITY METHODS =====

    async def _get_active_signals(self, agent_type: str) -> List[PlatformSignal]:
        """Get active signals for an agent type."""
        try:
            active_signals = await self.platform_service.get_active_signals(limit=100)

            # Most platform-generated signals are tagged as yuki/platform.
            if agent_type == 'yuki':
                allowed_agents = {None, 'yuki', 'platform'}
                return [s for s in active_signals if s.generated_by_agent in allowed_agents]

            return [s for s in active_signals if s.generated_by_agent == agent_type]
        except Exception as e:
            logger.error(f"❌ Failed to get active signals: {e}")
            return []

    async def _update_legacy_patterns(self, legacy_service):
        """Update legacy learning patterns for compatibility."""
        try:
            # This maintains compatibility with existing learning system
            # while transitioning to multi-level architecture
            pass  # Legacy service updates handled in existing worker
        except Exception as e:
            logger.error(f"❌ Legacy pattern update failed: {e}")

    async def _store_meta_insights(self, agent_type: str, insights):
        """Store meta-learning insights in database."""
        try:
            insight_data = {
                'agent_type': agent_type,
                'insights': insights.__dict__ if hasattr(insights, '__dict__') else str(insights),
                'timestamp': datetime.now().isoformat(),
                'learning_level': 'meta',
                'learning_type': 'meta_learning_cycle',
                'confidence_score': min(max(float(getattr(insights, 'learning_convergence_rate', 0.0) or 0.0), 0.0), 1.0)
            }

            self.db.from_('agent_learning_insights').insert(insight_data).execute()

        except Exception as e:
            logger.error(f"❌ Failed to store meta insights: {e}")

    async def _execute_with_limit(self, tasks: List, limit: int):
        """Execute tasks with concurrency limit."""
        if len(tasks) <= limit:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute in batches
            for i in range(0, len(tasks), limit):
                batch = tasks[i:i + limit]
                await asyncio.gather(*batch, return_exceptions=True)

    async def _log_performance_metrics(self):
        """Log worker performance metrics."""
        try:
            metrics = {
                'active_signals_tracked': sum(len(service.active_signals) for service in self.ml_services.values()),
                'level1_updates': len(self.last_level1_update),
                'level2_updates': len(self.last_level2_update),
                'level3_updates': len(self.last_level3_update),
                'timestamp': datetime.now()
            }

            logger.debug(f"📊 Performance Metrics: {metrics}")

        except Exception as e:
            logger.error(f"❌ Performance logging failed: {e}")


# Global worker instance
_worker_instance = None

def get_multi_level_learning_worker() -> MultiLevelLearningWorker:
    """Get or create multi-level learning worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = MultiLevelLearningWorker()
    return _worker_instance

async def start_multi_level_learning():
    """Start the multi-level learning worker."""
    worker = get_multi_level_learning_worker()
    await worker.start()

async def stop_multi_level_learning():
    """Stop the multi-level learning worker."""
    global _worker_instance
    if _worker_instance:
        await _worker_instance.stop()
        _worker_instance = None
