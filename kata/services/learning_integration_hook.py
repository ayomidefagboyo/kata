"""
Learning Integration Hook for Platform Signals Worker

This service automatically integrates learning capabilities with the existing
platform signals worker without requiring major changes to the worker code.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from kata.config.database import get_service_client
from kata.services.agent_learning_service import get_agent_learning_service
from kata.services.platform_signal_service import get_platform_signal_service, PlatformSignal

logger = logging.getLogger(__name__)


class LearningIntegrationHook:
    """Hooks into the existing platform signals system for automatic learning."""

    def __init__(self):
        """Initialize learning integration hook."""
        self.db = get_service_client()
        self.platform_service = get_platform_signal_service()

        # Learning services for each agent type
        self.learning_services = {
            'yuki': get_agent_learning_service('yuki'),
            'sakura': get_agent_learning_service('sakura'),
            'ryu': get_agent_learning_service('ryu')
        }

        # Track processed signals to avoid duplicates
        self.processed_signals = set()

        logger.info("Learning integration hook initialized")

    async def hook_into_platform_worker(self):
        """
        Hook into platform worker to automatically record outcomes.

        This runs as a background task alongside the platform signals worker.
        """
        logger.info("🎓 Starting learning integration hook")

        while True:
            try:
                await self._process_completed_signals()
                await self._process_ryu_recommendation_and_trade_outcomes()
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in learning hook: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _process_completed_signals(self):
        """Process recently completed signals for learning."""
        try:
            # Get recently completed signals (last 2 hours)
            cutoff_time = (datetime.now() - timedelta(hours=2)).isoformat()

            completed_result = self.db.table('platform_signals').select('''
                signal_id, token_symbol, direction, confidence, entry_price,
                target_1, target_2, stop_loss, leverage, position_size,
                market_conditions, technical_indicators, sentiment_data,
                analysis_timestamp, status, updated_at, signal_pool
            ''').in_('status', [
                'hit_target_1', 'hit_target_2', 'hit_stop_loss', 'expired'
            ]).gte('updated_at', cutoff_time).execute()

            completed_signals = completed_result.data or []

            new_signals = 0
            for signal_data in completed_signals:
                signal_id = signal_data['signal_id']

                # Skip if already processed
                if signal_id in self.processed_signals:
                    continue

                # Check if already in learning memory
                existing = self.db.table('agent_learning_memory').select('id').eq(
                    'signal_id', signal_id
                ).execute()

                if existing.data:
                    self.processed_signals.add(signal_id)
                    continue

                # Process this signal for learning
                await self._record_signal_for_learning(signal_data)
                self.processed_signals.add(signal_id)
                new_signals += 1

            if new_signals > 0:
                logger.info(f"🎓 Recorded {new_signals} new signal outcomes for learning")

        except Exception as e:
            logger.error(f"Error processing completed signals: {e}")

    async def _record_signal_for_learning(self, signal_data: Dict[str, Any]):
        """Record a single completed signal for learning."""
        try:
            # Determine which agent generated this signal
            agent_type = (signal_data.get('generated_by_agent') or 'yuki').lower()
            if agent_type not in self.learning_services:
                agent_type = 'yuki'

            # Get performance data
            perf_result = self.db.table('platform_signal_performance_tracking').select(
                'max_profit_reached, max_loss_reached'
            ).eq('signal_id', signal_data['signal_id']).execute()

            max_profit = 0
            max_loss = 0
            if perf_result.data:
                max_profit = float(perf_result.data[0].get('max_profit_reached', 0))
                max_loss = float(perf_result.data[0].get('max_loss_reached', 0))

            # Calculate outcome and PnL
            outcome, pnl_percentage, exit_reason = self._calculate_outcome(signal_data)

            # Create PlatformSignal object
            platform_signal = self._create_platform_signal_from_data(signal_data)

            # Record with learning service
            learning_service = self.learning_services[agent_type]

            success = await learning_service.record_trade_outcome(
                signal=platform_signal,
                outcome=outcome,
                pnl_percentage=pnl_percentage,
                exit_reason=exit_reason,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss
            )

            if success:
                logger.debug(f"✅ Recorded learning outcome: {signal_data['signal_id']} -> {outcome} ({pnl_percentage:+.2f}%)")
            else:
                logger.warning(f"❌ Failed to record learning outcome for {signal_data['signal_id']}")

        except Exception as e:
            logger.error(f"Error recording signal {signal_data.get('signal_id')} for learning: {e}")

    def _calculate_outcome(self, signal_data: Dict[str, Any]) -> tuple[str, float, str]:
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

        # Parse signal pool
        pool_value = signal_data.get('signal_pool', 'fast_mode')
        if pool_value == 'fast_mode':
            signal_pool = SignalPool.FAST_MODE
        elif pool_value == 'full_mode':
            signal_pool = SignalPool.FULL_MODE
        else:
            signal_pool = SignalPool.PREMIUM

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
            signal_pool=signal_pool,
            opportunity_rank=3,  # Default
            analysis_timestamp=datetime.fromisoformat(signal_data['analysis_timestamp']),
            expires_at=datetime.fromisoformat(signal_data['analysis_timestamp']) + timedelta(hours=24),
            validity_window_hours=24,
            leverage=signal_data.get('leverage', 5),
            position_size=signal_data.get('position_size', 10.0),
            risk_level='MEDIUM'
        )

    async def enhance_signal_generation(self, generated_signals: List[PlatformSignal]) -> List[PlatformSignal]:
        """
        Enhance newly generated signals with confidence feedback only.

        Position size and leverage are finalized during generation; this optional
        post-save hook must not apply a second sizing adjustment.
        """
        try:
            enhanced_signals = []

            for signal in generated_signals:
                try:
                    # Apply learning enhancement (assuming signals are from Yuki for now)
                    learning_service = self.learning_services['yuki']
                    learning_adj = await learning_service.get_learning_adjustment(signal)

                    # Apply confidence adjustment
                    enhanced_confidence = max(0.0, min(1.0,
                        signal.confidence + learning_adj.confidence_adjustment
                    ))

                    # Create enhanced signal
                    enhanced_signal = PlatformSignal(
                        signal_id=signal.signal_id,
                        token_symbol=signal.token_symbol,
                        direction=signal.direction,
                        timeframe=signal.timeframe,
                        confidence=enhanced_confidence,
                        overall_score=enhanced_confidence * 100,
                        signal_strength=signal.signal_strength,
                        time_horizon=signal.time_horizon,
                        entry_price=signal.entry_price,
                        target_1=signal.target_1,
                        target_1_probability=signal.target_1_probability,
                        target_2=signal.target_2,
                        target_2_probability=signal.target_2_probability,
                        stop_loss=signal.stop_loss,
                        risk_reward_ratio=signal.risk_reward_ratio,
                        market_conditions=signal.market_conditions,
                        technical_indicators=signal.technical_indicators,
                        sentiment_data=signal.sentiment_data,
                        risk_factors=signal.risk_factors,
                        signal_pool=signal.signal_pool,
                        opportunity_rank=signal.opportunity_rank,
                        analysis_timestamp=signal.analysis_timestamp,
                        expires_at=signal.expires_at,
                        validity_window_hours=signal.validity_window_hours,
                        status=signal.status,
                        run_id=signal.run_id,
                        analysis_notes=signal.analysis_notes,
                        ai_reasoning=f"{signal.ai_reasoning}\n\nLearning adjustment: {learning_adj.reasoning}",
                        ai_key_factors=signal.ai_key_factors,
                        ai_risk_assessment=signal.ai_risk_assessment,
                        ai_confidence_breakdown=signal.ai_confidence_breakdown,
                        logo_url=signal.logo_url,
                        leverage=signal.leverage,
                        position_size=signal.position_size,
                        risk_level=signal.risk_level
                    )

                    enhanced_signals.append(enhanced_signal)

                    if abs(learning_adj.confidence_adjustment) > 0.05:
                        logger.info(f"🎓 Applied learning to {signal.token_symbol}: confidence {signal.confidence:.3f} -> {enhanced_confidence:.3f} ({learning_adj.reasoning})")

                except Exception as e:
                    logger.warning(f"Error enhancing signal {signal.signal_id}: {e}")
                    enhanced_signals.append(signal)  # Use original signal if enhancement fails

            return enhanced_signals

        except Exception as e:
            logger.error(f"Error in signal enhancement: {e}")
            return generated_signals  # Return original signals if enhancement fails

    async def _process_ryu_recommendation_and_trade_outcomes(self):
        """Process recently resolved Ryu recommendations and spot trades for learning."""
        try:
            resolved_recs = self.db.table('agent_recommendation_tracking').select(
                'id, token_symbol, recommendation, direction, confidence, entry_price, '
                'target_1, target_2, stop_loss, horizon_7d_status, horizon_7d_pnl_pct, created_at'
            ).eq('agent_type', 'ryu').in_('horizon_7d_status', ['win', 'loss']).limit(100).execute().data or []

            new_ryu_records = 0
            for rec in resolved_recs:
                rec_id = f"ryu_rec_{rec['id']}"
                if rec_id in self.processed_signals:
                    continue

                existing = self.db.table('agent_learning_memory').select('id').eq(
                    'signal_id', rec_id
                ).execute().data
                if existing:
                    self.processed_signals.add(rec_id)
                    continue

                direction = (rec.get('direction') or 'LONG').upper()
                if direction not in ('LONG', 'SHORT'):
                    direction = 'LONG'

                entry = float(rec.get('entry_price') or 0.0)
                t1 = float(rec.get('target_1') or 0.0)
                sl = float(rec.get('stop_loss') or 0.0)

                from kata.services.platform_signal_service import PlatformSignal, SignalPool
                sig = PlatformSignal(
                    signal_id=rec_id,
                    token_symbol=rec.get('token_symbol') or 'UNKNOWN',
                    direction=direction,
                    timeframe='1d',
                    confidence=float(rec.get('confidence') or 0.6),
                    overall_score=float(rec.get('confidence') or 0.6) * 100,
                    signal_strength='medium',
                    time_horizon='medium',
                    entry_price=entry,
                    target_1=t1,
                    target_1_probability=0.6,
                    target_2=t1 * 1.1 if t1 else (entry * 1.2 if entry else 1.0),
                    target_2_probability=0.3,
                    stop_loss=sl,
                    risk_reward_ratio=2.0,
                    market_conditions={},
                    technical_indicators={},
                    sentiment_data={},
                    risk_factors=[],
                    signal_pool=SignalPool.FAST_MODE,
                    opportunity_rank=1,
                    analysis_timestamp=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7),
                    validity_window_hours=168,
                    leverage=1,
                    position_size=10.0,
                )

                outcome = 'win' if rec['horizon_7d_status'] == 'win' else 'loss'
                pnl_pct = float(rec.get('horizon_7d_pnl_pct') or 0.0)

                await self.learning_services['ryu'].record_trade_outcome(
                    signal=sig,
                    outcome=outcome,
                    pnl_percentage=pnl_pct,
                    exit_reason='7D_HORIZON',
                    max_profit_reached=max(0.0, pnl_pct),
                    max_loss_reached=min(0.0, pnl_pct)
                )
                self.processed_signals.add(rec_id)
                new_ryu_records += 1

            if new_ryu_records > 0:
                logger.info(f"🐉 Recorded {new_ryu_records} new Ryu recommendation outcomes for learning")

        except Exception as e:
            logger.error(f"Error processing Ryu recommendation learning outcomes: {e}")

    def get_hook_status(self) -> Dict[str, Any]:
        """Get status of the learning integration hook."""
        return {
            'processed_signals_count': len(self.processed_signals),
            'learning_services_active': list(self.learning_services.keys()),
            'last_check': datetime.now().isoformat()
        }


# Global hook instance
_learning_hook: Optional[LearningIntegrationHook] = None


def get_learning_integration_hook() -> LearningIntegrationHook:
    """Get or create the global learning integration hook."""
    global _learning_hook
    if _learning_hook is None:
        _learning_hook = LearningIntegrationHook()
    return _learning_hook


async def start_learning_integration():
    """Start the learning integration hook as a background task."""
    hook = get_learning_integration_hook()
    await hook.hook_into_platform_worker()
