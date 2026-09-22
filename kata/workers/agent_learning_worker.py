"""
Agent Learning Worker - Background Learning Pipeline

This worker service runs automatic learning updates for trading agents:
- Processes completed signals to extract learning patterns
- Updates pattern recognition models
- Monitors learning system performance
- Handles automatic outcome recording
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

from kata.config.database import get_service_client
from kata.services.agent_learning_service import get_agent_learning_service
from kata.services.platform_signal_service import get_platform_signal_service, PlatformSignal
from kata.services.lightweight_performance_service import get_lightweight_performance_service

logger = logging.getLogger(__name__)


class AgentLearningWorker:
    """Background worker for agent learning system."""

    def __init__(self):
        """Initialize learning worker."""
        self.db = get_service_client()
        self.platform_service = get_platform_signal_service()
        self.performance_service = get_lightweight_performance_service()

        # Learning services for each agent type
        self.learning_services = {
            'yuki': get_agent_learning_service('yuki'),
            'sakura': get_agent_learning_service('sakura'),
            'ryu': get_agent_learning_service('ryu')
        }

        # Worker configuration
        self.config = {
            'check_interval_minutes': 15,  # Check for new outcomes every 15 minutes
            'batch_size': 50,              # Process up to 50 signals per batch
            'pattern_update_interval_hours': 4,  # Update patterns every 4 hours
            'performance_tracking_days': 30,     # Track performance over 30 days
            'min_pattern_confidence': 0.6        # Minimum confidence for pattern recognition
        }

        # Worker state
        self.is_running = False
        self.last_pattern_update = {}
        self.processing_stats = {
            'signals_processed': 0,
            'patterns_updated': 0,
            'errors': 0,
            'last_run': None
        }

        logger.info("Agent Learning Worker initialized")

    async def start_background_learning(self):
        """Start the background learning process."""
        if self.is_running:
            logger.warning("Learning worker is already running")
            return

        self.is_running = True
        logger.info("🎓 Starting agent learning worker")

        try:
            while self.is_running:
                try:
                    # Main learning cycle
                    await self._run_learning_cycle()

                    # Wait for next cycle
                    await asyncio.sleep(self.config['check_interval_minutes'] * 60)

                except Exception as e:
                    logger.error(f"Error in learning cycle: {e}")
                    self.processing_stats['errors'] += 1
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        except asyncio.CancelledError:
            logger.info("Learning worker cancelled")
        finally:
            self.is_running = False
            logger.info("Agent learning worker stopped")

    async def stop_background_learning(self):
        """Stop the background learning process."""
        logger.info("Stopping agent learning worker")
        self.is_running = False

    async def _run_learning_cycle(self):
        """Run one complete learning cycle."""
        try:
            logger.debug("Running learning cycle")

            # 1. Process new signal outcomes
            await self._process_new_signal_outcomes()

            # 2. Update learning patterns (less frequently)
            await self._update_learning_patterns_if_needed()

            # 3. Update performance tracking
            await self._update_performance_tracking()

            # 4. Cleanup old data
            await self._cleanup_old_data()

            self.processing_stats['last_run'] = datetime.now()
            logger.debug("Learning cycle completed successfully")

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            self.processing_stats['errors'] += 1

    async def _process_new_signal_outcomes(self):
        """Process newly completed signals for learning."""
        try:
            # Get recently completed signals that haven't been processed for learning
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Query completed signals
            completed_signals_result = self.db.table('platform_signals').select('''
                signal_id, token_symbol, direction, confidence, entry_price,
                target_1, target_2, stop_loss, leverage, position_size,
                market_conditions, technical_indicators, sentiment_data,
                analysis_timestamp, status, updated_at
            ''').in_('status', [
                'hit_target_1', 'hit_target_2', 'hit_stop_loss', 'expired'
            ]).gte('updated_at', cutoff_time.isoformat()).execute()

            completed_signals = completed_signals_result.data or []

            if not completed_signals:
                logger.debug("No new completed signals to process")
                return

            logger.info(f"Processing {len(completed_signals)} completed signals for learning")

            # Process each completed signal
            for signal_data in completed_signals:
                try:
                    await self._process_single_signal_outcome(signal_data)
                    self.processing_stats['signals_processed'] += 1

                except Exception as e:
                    logger.error(f"Error processing signal {signal_data.get('signal_id')}: {e}")
                    continue

            logger.info(f"Processed {len(completed_signals)} signals for learning")

        except Exception as e:
            logger.error(f"Error processing signal outcomes: {e}")

    async def _process_single_signal_outcome(self, signal_data: Dict[str, Any]):
        """Process a single signal outcome for learning."""
        try:
            signal_id = signal_data['signal_id']
            status = signal_data['status']

            # Check if already processed for learning
            existing_result = self.db.table('agent_learning_memory').select('id').eq(
                'signal_id', signal_id
            ).execute()

            if existing_result.data:
                return  # Already processed

            # Determine outcome and PnL
            outcome, pnl_percentage, exit_reason = self._calculate_signal_outcome(signal_data)

            # Get performance tracking data for max profit/loss
            perf_result = self.db.table('platform_signal_performance_tracking').select(
                'max_profit_reached, max_loss_reached'
            ).eq('signal_id', signal_id).execute()

            max_profit = 0
            max_loss = 0
            if perf_result.data:
                max_profit = float(perf_result.data[0].get('max_profit_reached', 0))
                max_loss = float(perf_result.data[0].get('max_loss_reached', 0))

            # Create PlatformSignal object for learning service
            platform_signal = self._create_platform_signal_from_data(signal_data)

            # Determine which agent generated this signal
            # For now, most platform signals are from the unified generator ('yuki')
            # In the future, we can add agent attribution based on signal characteristics
            agent_type = self._determine_signal_agent_type(signal_data)
            learning_service = self.learning_services.get(agent_type, self.learning_services['yuki'])

            success = await learning_service.record_trade_outcome(
                signal=platform_signal,
                outcome=outcome,
                pnl_percentage=pnl_percentage,
                exit_reason=exit_reason,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss
            )

            if success:
                logger.debug(f"Recorded learning outcome for {signal_id}: {outcome} ({pnl_percentage:+.2f}%)")
            else:
                logger.warning(f"Failed to record learning outcome for {signal_id}")

        except Exception as e:
            logger.error(f"Error processing single signal outcome: {e}")

    def _determine_signal_agent_type(self, signal_data: Dict[str, Any]) -> str:
        """
        Determine which agent type generated the signal based on its characteristics.

        Args:
            signal_data: Platform signal data from database

        Returns:
            Agent type string ('yuki', 'sakura', 'ryu')
        """
        try:
            # For now, most signals come from the unified generator (yuki)
            # But we can add logic to distinguish based on signal characteristics

            # Check signal characteristics
            signal_strength = signal_data.get('signal_strength', '').lower()
            time_horizon = signal_data.get('time_horizon', '').lower()
            leverage = signal_data.get('leverage', 5)

            # Sakura agent typically focuses on DeFi/yield signals
            if 'defi' in signal_data.get('analysis_notes', '').lower():
                return 'sakura'

            # Ryu agent typically focuses on futures/high leverage signals
            if leverage >= 10 or 'futures' in signal_data.get('analysis_notes', '').lower():
                return 'ryu'

            # Default to yuki (unified signal generator)
            return 'yuki'

        except Exception as e:
            logger.debug(f"Error determining agent type: {e}")
            return 'yuki'  # Safe default

    def _calculate_signal_outcome(self, signal_data: Dict[str, Any]) -> tuple[str, float, str]:
        """Calculate outcome, PnL, and exit reason from signal data."""
        status = signal_data['status']
        direction = signal_data['direction']
        entry_price = float(signal_data['entry_price'])
        target_1 = float(signal_data['target_1'])
        target_2 = float(signal_data['target_2'])
        stop_loss = float(signal_data['stop_loss'])

        # Determine exit price and reason based on status
        if status == 'hit_target_1':
            exit_price = target_1
            exit_reason = 'TARGET_1'
            outcome = 'win'
        elif status == 'hit_target_2':
            exit_price = target_2
            exit_reason = 'TARGET_2'
            outcome = 'win'
        elif status == 'hit_stop_loss':
            exit_price = stop_loss
            exit_reason = 'STOP_LOSS'
            outcome = 'loss'
        else:  # expired
            # For expired signals, assume exit at entry price (scratch)
            exit_price = entry_price
            exit_reason = 'EXPIRED'
            outcome = 'scratch'

        # Calculate PnL percentage
        if direction.upper() == 'LONG':
            pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
        else:  # SHORT
            pnl_percentage = ((entry_price - exit_price) / entry_price) * 100

        return outcome, pnl_percentage, exit_reason

    def _create_platform_signal_from_data(self, signal_data: Dict[str, Any]) -> PlatformSignal:
        """Create PlatformSignal object from database data."""
        from kata.services.platform_signal_service import SignalPool

        return PlatformSignal(
            signal_id=signal_data['signal_id'],
            token_symbol=signal_data['token_symbol'],
            direction=signal_data['direction'],
            timeframe='1h',  # Default
            confidence=float(signal_data['confidence']),
            overall_score=float(signal_data['confidence']) * 100,
            signal_strength='medium',  # Default
            time_horizon='short',  # Default
            entry_price=float(signal_data['entry_price']),
            target_1=float(signal_data['target_1']),
            target_1_probability=0.6,  # Default
            target_2=float(signal_data['target_2']),
            target_2_probability=0.3,  # Default
            stop_loss=float(signal_data['stop_loss']),
            risk_reward_ratio=2.0,  # Default
            market_conditions=signal_data.get('market_conditions', {}),
            technical_indicators=signal_data.get('technical_indicators', {}),
            sentiment_data=signal_data.get('sentiment_data', {}),
            risk_factors=[],
            signal_pool=SignalPool.FAST_MODE,  # Default
            opportunity_rank=3,  # Default
            analysis_timestamp=datetime.fromisoformat(signal_data['analysis_timestamp']),
            expires_at=datetime.fromisoformat(signal_data['analysis_timestamp']) + timedelta(hours=24),
            validity_window_hours=24,
            leverage=signal_data.get('leverage', 5),
            position_size=signal_data.get('position_size', 10.0),
            risk_level='MEDIUM'
        )

    async def _update_learning_patterns_if_needed(self):
        """Update learning patterns if enough time has passed."""
        try:
            current_time = datetime.now()

            for agent_type in self.learning_services.keys():
                last_update = self.last_pattern_update.get(agent_type, datetime.min)

                if (current_time - last_update).total_seconds() > self.config['pattern_update_interval_hours'] * 3600:
                    await self._update_patterns_for_agent(agent_type)
                    self.last_pattern_update[agent_type] = current_time
                    self.processing_stats['patterns_updated'] += 1

        except Exception as e:
            logger.error(f"Error updating learning patterns: {e}")

    async def _update_patterns_for_agent(self, agent_type: str):
        """Update learning patterns for a specific agent type."""
        try:
            logger.info(f"Updating learning patterns for {agent_type} agent")

            # This is handled automatically by database triggers when new learning memory is inserted
            # But we can also run manual pattern analysis here if needed

            # Get pattern statistics
            pattern_result = self.db.table('agent_learning_patterns').select(
                'pattern_name, success_rate, total_trades'
            ).eq('agent_type', agent_type).execute()

            patterns = pattern_result.data or []

            # Log pattern summary
            if patterns:
                high_performance = [p for p in patterns if p['success_rate'] > 0.7 and p['total_trades'] >= 5]
                poor_performance = [p for p in patterns if p['success_rate'] < 0.4 and p['total_trades'] >= 5]

                logger.info(f"{agent_type}: {len(patterns)} patterns, {len(high_performance)} high-performing, {len(poor_performance)} poor-performing")
            else:
                logger.info(f"{agent_type}: No patterns found yet")

        except Exception as e:
            logger.error(f"Error updating patterns for {agent_type}: {e}")

    async def _update_performance_tracking(self):
        """Update learning system performance tracking."""
        try:
            current_date = datetime.now().date()

            for agent_type in self.learning_services.keys():
                # Get daily performance metrics
                daily_stats = await self._calculate_daily_learning_stats(agent_type, current_date)

                # Update or insert performance tracking record
                await self._upsert_performance_tracking(agent_type, current_date, daily_stats)

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    async def _calculate_daily_learning_stats(self, agent_type: str, date: datetime.date) -> Dict[str, Any]:
        """Calculate daily learning statistics for an agent."""
        try:
            date_str = date.isoformat()
            next_date_str = (date + timedelta(days=1)).isoformat()

            # Get learning memory records for the day
            memory_result = self.db.table('agent_learning_memory').select(
                'outcome, pnl_percentage'
            ).eq('agent_type', agent_type).gte(
                'created_at', date_str
            ).lt('created_at', next_date_str).execute()

            records = memory_result.data or []

            if not records:
                return {
                    'total_signals': 0,
                    'signals_with_learning': 0,
                    'learning_win_rate': 0,
                    'learning_avg_pnl': 0,
                    'baseline_win_rate': 0,
                    'baseline_avg_pnl': 0
                }

            # Calculate basic metrics
            total_signals = len(records)
            wins = len([r for r in records if r['outcome'] == 'win'])
            win_rate = wins / total_signals if total_signals > 0 else 0
            avg_pnl = sum(r['pnl_percentage'] for r in records) / total_signals if total_signals > 0 else 0

            return {
                'total_signals': total_signals,
                'signals_with_learning': total_signals,  # All new signals use learning
                'learning_win_rate': win_rate,
                'learning_avg_pnl': avg_pnl,
                'baseline_win_rate': win_rate,  # For now, same as learning (need historical baseline)
                'baseline_avg_pnl': avg_pnl
            }

        except Exception as e:
            logger.error(f"Error calculating daily stats for {agent_type}: {e}")
            return {}

    async def _upsert_performance_tracking(self, agent_type: str, date: datetime.date, stats: Dict[str, Any]):
        """Insert or update performance tracking record."""
        try:
            tracking_data = {
                'agent_type': agent_type,
                'measurement_date': date.isoformat(),
                'total_signals': stats.get('total_signals', 0),
                'signals_with_learning': stats.get('signals_with_learning', 0),
                'learning_win_rate': stats.get('learning_win_rate', 0),
                'learning_avg_pnl': stats.get('learning_avg_pnl', 0),
                'baseline_win_rate': stats.get('baseline_win_rate', 0),
                'baseline_avg_pnl': stats.get('baseline_avg_pnl', 0),
                'win_rate_improvement': stats.get('learning_win_rate', 0) - stats.get('baseline_win_rate', 0),
                'pnl_improvement': stats.get('learning_avg_pnl', 0) - stats.get('baseline_avg_pnl', 0)
            }

            # Upsert the record
            self.db.table('agent_learning_performance').upsert(
                tracking_data,
                on_conflict='agent_type,measurement_date'
            ).execute()

            logger.debug(f"Updated performance tracking for {agent_type} on {date}")

        except Exception as e:
            logger.error(f"Error upserting performance tracking: {e}")

    async def _cleanup_old_data(self):
        """Clean up old learning data to manage database size."""
        try:
            # Keep learning memory for 90 days
            cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()

            delete_result = self.db.table('agent_learning_memory').delete().lt(
                'created_at', cutoff_date
            ).execute()

            if delete_result.data:
                logger.info(f"Cleaned up {len(delete_result.data)} old learning memory records")

            # Keep performance tracking for 1 year
            perf_cutoff = (datetime.now() - timedelta(days=365)).date().isoformat()

            perf_delete = self.db.table('agent_learning_performance').delete().lt(
                'measurement_date', perf_cutoff
            ).execute()

            if perf_delete.data:
                logger.info(f"Cleaned up {len(perf_delete.data)} old performance tracking records")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    async def force_learning_update(self, agent_type: Optional[str] = None):
        """Force an immediate learning update for specified agent or all agents."""
        try:
            logger.info(f"Forcing learning update for {agent_type or 'all agents'}")

            if agent_type:
                agents_to_update = [agent_type] if agent_type in self.learning_services else []
            else:
                agents_to_update = list(self.learning_services.keys())

            for agent in agents_to_update:
                await self._update_patterns_for_agent(agent)
                self.last_pattern_update[agent] = datetime.now()
                logger.info(f"Completed forced update for {agent}")

            return {'updated_agents': agents_to_update, 'timestamp': datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Error in forced learning update: {e}")
            return {'error': str(e)}

    def get_worker_status(self) -> Dict[str, Any]:
        """Get current worker status and statistics."""
        return {
            'is_running': self.is_running,
            'configuration': self.config,
            'processing_stats': self.processing_stats.copy(),
            'last_pattern_updates': {
                agent: timestamp.isoformat() if timestamp != datetime.min else None
                for agent, timestamp in self.last_pattern_update.items()
            },
            'supported_agents': list(self.learning_services.keys()),
            'status_timestamp': datetime.now().isoformat()
        }


# Global worker instance
_learning_worker: Optional[AgentLearningWorker] = None


def get_agent_learning_worker() -> AgentLearningWorker:
    """Get or create the global learning worker instance."""
    global _learning_worker
    if _learning_worker is None:
        _learning_worker = AgentLearningWorker()
    return _learning_worker


async def start_learning_worker():
    """Start the background learning worker."""
    worker = get_agent_learning_worker()
    await worker.start_background_learning()


async def stop_learning_worker():
    """Stop the background learning worker."""
    worker = get_agent_learning_worker()
    await worker.stop_background_learning()