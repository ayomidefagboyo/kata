"""
Multi-Level Learning Service for Advanced Trading Agents

This service implements a sophisticated 3-level learning architecture:
- Level 1: Real-Time Learning (0-5 minutes) - Immediate feedback loops
- Level 2: Pattern Learning (15 minutes) - Enhanced pattern recognition
- Level 3: Meta-Learning (Daily) - Strategy optimization and prompt improvement

Created to dramatically improve agent performance beyond basic pattern matching.
"""

import asyncio
import logging
import json
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
from collections import defaultdict, deque
from enum import Enum

from kata.config.database import get_service_client
from kata.config.settings import settings
from kata.services.platform_signal_service import (
    persist_entry_activation_if_current,
    persist_market_conditions_if_current,
    PlatformSignal,
    SignalPool,
)
from kata.services.agent_learning_service import LearningAdjustment
from kata.services.lightweight_performance_service import get_lightweight_performance_service

logger = logging.getLogger(__name__)


class LearningLevel(Enum):
    """Learning level classifications."""
    REAL_TIME = "real_time"      # 0-5 minutes
    PATTERN = "pattern"          # 15 minutes
    META = "meta"               # Daily


@dataclass
class MarketRegime:
    """Current market regime classification."""
    regime_type: str  # TRENDING_BULL, TRENDING_BEAR, RANGE_BOUND, HIGH_VOLATILITY, BREAKOUT
    strength: float   # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    duration_minutes: int
    supporting_factors: List[str]
    detected_at: datetime


@dataclass
class RealTimeFeedback:
    """Real-time learning feedback from active signals."""
    signal_id: str
    current_pnl: float
    max_favorable: float
    max_adverse: float
    time_elapsed_minutes: int
    regime_change_detected: bool
    volatility_spike: bool
    volume_confirmation: bool
    support_resistance_test: bool
    feedback_confidence: float
    adjustment_suggested: Dict[str, float]


@dataclass
class PatternPerformance:
    """Enhanced pattern performance tracking."""
    pattern_name: str
    pattern_features: Dict[str, Any]
    success_rate: float
    sample_size: int
    avg_pnl: float
    avg_time_to_target: float
    best_market_conditions: Dict[str, Any]
    risk_adjusted_return: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_correlation: float
    last_updated: datetime


@dataclass
class MetaLearningInsights:
    """Meta-level learning insights for strategy optimization."""
    strategy_effectiveness: Dict[str, float]
    optimal_market_conditions: Dict[str, Any]
    cross_agent_performance: Dict[str, Dict[str, float]]
    prompt_optimization_suggestions: List[str]
    risk_parameter_adjustments: Dict[str, float]
    temporal_performance_patterns: Dict[str, float]
    learning_convergence_rate: float
    adaptation_recommendations: List[str]


class MultiLevelLearningService:
    """Advanced multi-level learning service."""

    TERMINAL_SIGNAL_STATUSES = {
        'hit_target_1',
        'hit_target_2',
        'hit_stop_loss',
        'expired',
        'cancelled',
        'invalidated',
        # Backward compatibility with older status values.
        'completed',
        'failed',
    }

    # Statuses that represent a real, learnable win/loss outcome. Distinct from
    # TERMINAL_SIGNAL_STATUSES, which also includes expired/cancelled/invalidated —
    # those resolved the lifecycle but carry no directional outcome to learn from.
    RESOLVED_OUTCOME_STATUSES = {
        'hit_target_1',
        'hit_target_2',
        'hit_stop_loss',
        'completed',
        'failed',
    }

    def __init__(self, agent_type: str):
        """Initialize multi-level learning service."""
        self.agent_type = agent_type
        self.db = get_service_client()

        # Level 1: Real-time learning state
        self.active_signals = {}  # signal_id -> real-time tracking
        self.market_regime = None
        self.real_time_adjustments = deque(maxlen=100)
        self.volatility_tracker = deque(maxlen=50)

        # Level 2: Pattern learning state
        self.pattern_cache = {}
        self.pattern_performance = {}
        self.recent_patterns = deque(maxlen=200)

        # Level 3: Meta-learning state
        self.strategy_performance = defaultdict(list)
        self.cross_agent_metrics = {}
        self.prompt_performance_history = deque(maxlen=500)

        # Learning configuration
        self.config = {
            'real_time_update_interval': 30,  # seconds
            'pattern_update_interval': 900,   # 15 minutes
            'meta_update_interval': 86400,    # daily
            'min_data_points': 10,
            # Lookback for pattern learning. The old 24h window starved the updater at
            # current volume (<2 signals/day) so pattern_performance_tracking went stale.
            # Widened to a config-driven window (default 180d) so enough RESOLVED signals
            # accumulate; an all-history fallback kicks in when the table goes stale.
            'pattern_lookback_hours': int(getattr(settings, 'LEARNING_PATTERN_LOOKBACK_HOURS', 24 * 180)),
            'min_resolved_for_pattern': int(getattr(settings, 'LEARNING_MIN_PATTERN_SAMPLE', 8)),
            'staleness_hours': float(getattr(settings, 'LEARNING_STALENESS_HOURS', 36.0)),
            'confidence_decay_rate': 0.95,
            'learning_rate': 0.1,
            'volatility_threshold': 0.02,     # 2% price move
            'regime_change_threshold': 0.3
        }

        # Use the same Binance-backed performance service as the platform signals worker
        self.performance_service = get_lightweight_performance_service()

        logger.info(f"🧠 Multi-Level Learning Service initialized for {agent_type}")

    # ===== LEVEL 1: REAL-TIME LEARNING (0-5 minutes) =====

    def _parse_datetime(self, value: Any, default: Optional[datetime] = None) -> datetime:
        """Parse timestamp-like values into datetime."""
        if default is None:
            default = datetime.now()

        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00')).replace(tzinfo=None)
            except ValueError:
                return default
        return default

    def _coerce_signal(self, signal: Any) -> Optional[PlatformSignal]:
        """Normalize signal inputs to PlatformSignal objects."""
        if isinstance(signal, PlatformSignal):
            return signal

        if not isinstance(signal, dict):
            return None

        signal_id = str(signal.get('signal_id') or '').strip()
        token_symbol = str(signal.get('token_symbol') or '').strip()
        if not signal_id or not token_symbol:
            return None

        def _to_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        pool_raw = signal.get('signal_pool', SignalPool.FULL_MODE.value)
        if isinstance(pool_raw, SignalPool):
            signal_pool = pool_raw
        else:
            try:
                signal_pool = SignalPool(str(pool_raw))
            except ValueError:
                signal_pool = SignalPool.FULL_MODE

        entry_price = _to_float(signal.get('entry_price'), 0.0)
        target_1 = _to_float(signal.get('target_1'), entry_price)
        target_2 = _to_float(signal.get('target_2'), target_1)
        stop_loss = _to_float(signal.get('stop_loss'), entry_price)
        confidence = _to_float(signal.get('confidence'), 0.5)

        return PlatformSignal(
            signal_id=signal_id,
            token_symbol=token_symbol,
            direction=str(signal.get('direction') or 'LONG'),
            timeframe=str(signal.get('timeframe') or '4h'),
            confidence=confidence,
            overall_score=_to_float(signal.get('overall_score'), confidence),
            signal_strength=str(signal.get('signal_strength') or 'medium'),
            time_horizon=str(signal.get('time_horizon') or 'short'),
            entry_price=entry_price,
            target_1=target_1,
            target_1_probability=_to_float(signal.get('target_1_probability'), 0.5),
            target_2=target_2,
            target_2_probability=_to_float(signal.get('target_2_probability'), 0.3),
            stop_loss=stop_loss,
            risk_reward_ratio=_to_float(signal.get('risk_reward_ratio'), 1.0),
            market_conditions=signal.get('market_conditions') or {},
            technical_indicators=signal.get('technical_indicators') or {},
            sentiment_data=signal.get('sentiment_data') or {},
            risk_factors=signal.get('risk_factors') or [],
            signal_pool=signal_pool,
            opportunity_rank=_to_int(signal.get('opportunity_rank'), 1),
            analysis_timestamp=self._parse_datetime(signal.get('analysis_timestamp')),
            expires_at=self._parse_datetime(signal.get('expires_at'), datetime.now() + timedelta(hours=24)),
            validity_window_hours=_to_int(signal.get('validity_window_hours'), 24),
            status=str(signal.get('status') or 'active'),
            run_id=signal.get('run_id'),
            analysis_notes=signal.get('analysis_notes'),
            ai_reasoning=signal.get('ai_reasoning'),
            ai_key_factors=signal.get('ai_key_factors') if isinstance(signal.get('ai_key_factors'), list) else [],
            ai_risk_assessment=signal.get('ai_risk_assessment'),
            ai_confidence_breakdown=signal.get('ai_confidence_breakdown'),
            logo_url=signal.get('logo_url'),
            leverage=_to_int(signal.get('leverage'), 5),
            position_size=_to_float(signal.get('position_size'), 10.0),
            risk_level=str(signal.get('risk_level') or 'MEDIUM'),
            learning_tracked=bool(signal.get('learning_tracked', False)),
            learning_started_at=self._parse_datetime(signal.get('learning_started_at')) if signal.get('learning_started_at') else None,
            learning_completed_at=self._parse_datetime(signal.get('learning_completed_at')) if signal.get('learning_completed_at') else None,
            market_regime=signal.get('market_regime'),
            regime_confidence=_to_float(signal.get('regime_confidence'), 0.0) if signal.get('regime_confidence') is not None else None,
            volatility_environment=signal.get('volatility_environment'),
            pattern_classification=signal.get('pattern_classification'),
            pattern_confidence=_to_float(signal.get('pattern_confidence'), 0.0) if signal.get('pattern_confidence') is not None else None,
            historical_pattern_success_rate=_to_float(signal.get('historical_pattern_success_rate'), 0.0) if signal.get('historical_pattern_success_rate') is not None else None,
            learning_version=_to_int(signal.get('learning_version'), 1),
            prompt_version=signal.get('prompt_version'),
            generated_by_agent=signal.get('generated_by_agent'),
            predicted_success_probability=_to_float(signal.get('predicted_success_probability'), 0.0) if signal.get('predicted_success_probability') is not None else None,
            predicted_time_to_target=_to_int(signal.get('predicted_time_to_target'), 0) if signal.get('predicted_time_to_target') is not None else None,
            risk_adjusted_confidence=_to_float(signal.get('risk_adjusted_confidence'), 0.0) if signal.get('risk_adjusted_confidence') is not None else None,
            learning_insights_id=signal.get('learning_insights_id'),
            pattern_performance_id=signal.get('pattern_performance_id')
        )

    async def start_real_time_tracking(self, signal: Any) -> None:
        """Start real-time tracking for a new signal."""
        normalized_signal = self._coerce_signal(signal)
        if not normalized_signal:
            logger.warning("⚠️ start_real_time_tracking received invalid signal payload")
            return

        try:
            # Safety check: verify database tables exist
            try:
                self.db.from_('agent_learning_adjustments').select('id').limit(1).execute()
            except Exception as db_check:
                logger.warning(f"⚠️ Multi-level learning tables not available: {db_check}")
                return
            self.active_signals[normalized_signal.signal_id] = {
                'signal': normalized_signal,
                'start_time': datetime.now(),
                'entry_price': normalized_signal.entry_price,
                'initial_regime': self.market_regime,
                'price_history': deque(maxlen=100),
                'pnl_history': deque(maxlen=100),
                # Bounded: long-lived signals emit feedback every interval and an
                # unbounded list grows for days on a 512MB instance
                'feedback_events': deque(maxlen=200)
            }

            logger.info(f"📊 Real-time tracking started for {normalized_signal.token_symbol} ({normalized_signal.signal_id})")

            # Start background monitoring
            asyncio.create_task(self._monitor_signal_realtime(normalized_signal.signal_id))

        except Exception as e:
            logger.error(f"❌ Failed to start real-time tracking for {normalized_signal.signal_id}: {e}")

    async def _monitor_signal_realtime(self, signal_id: str) -> None:
        """Monitor a signal in real-time for learning feedback."""
        try:
            while signal_id in self.active_signals:
                # Wait for update interval
                await asyncio.sleep(self.config['real_time_update_interval'])

                tracking_data = self.active_signals[signal_id]
                signal = tracking_data['signal']

                # Always check completion first, even if price feed is unavailable.
                # Otherwise stale trackers can loop forever when price lookup fails.
                if await self._is_signal_completed(signal):
                    await self.stop_real_time_tracking(signal_id)
                    break

                # Get current price — stale-aware: refetch via REST if WS cache is >30s old.
                current_price = None
                try:
                    if hasattr(self.performance_service, "get_fresh_price"):
                        current_price = await self.performance_service.get_fresh_price(
                            signal.token_symbol, max_age_seconds=30.0
                        )
                except Exception as exc:
                    logger.debug(f"get_fresh_price failed for {signal.token_symbol}: {exc}")
                if not current_price:
                    current_price = await self._get_current_price(signal.token_symbol)
                if not current_price:
                    tracking_data['price_miss_count'] = tracking_data.get('price_miss_count', 0) + 1
                    # Avoid log spam on symbols that are not currently priced by the WS cache.
                    if tracking_data['price_miss_count'] % 30 == 1:
                        logger.warning(
                            f"⚠️ Could not get price for {signal.token_symbol} "
                            f"(misses: {tracking_data['price_miss_count']}, continuing retries)"
                        )
                    continue
                tracking_data['price_miss_count'] = 0

                # Respect pending-entry signals: only learn/score after entry activation.
                if not await self._ensure_entry_activated(signal_id, signal, current_price):
                    continue

                # Stale-price warning: if WS cache hasn't ticked in >60s for an active signal,
                # the WS connection may be degraded — log so monitoring can surface it.
                try:
                    if hasattr(self.performance_service, "get_price_age_seconds"):
                        age = self.performance_service.get_price_age_seconds(signal.token_symbol)
                        if age is not None and age > 60.0:
                            tracking_data['stale_price_warns'] = tracking_data.get('stale_price_warns', 0) + 1
                            if tracking_data['stale_price_warns'] % 10 == 1:
                                logger.warning(
                                    f"⚠️ Stale WS price for {signal.token_symbol}: age={age:.1f}s "
                                    f"(REST refetch in use)"
                                )
                        else:
                            tracking_data['stale_price_warns'] = 0
                except Exception:
                    pass

                # Calculate real-time metrics
                entry_price = tracking_data['entry_price']
                current_pnl = self._calculate_pnl(
                    entry_price, current_price, signal.direction, signal.leverage
                )

                # Update tracking data in memory
                tracking_data['price_history'].append((datetime.now(), current_price))
                tracking_data['pnl_history'].append((datetime.now(), current_pnl))

                # Update platform_signals market_conditions with current price
                try:
                    saved, latest_mc, _ = persist_market_conditions_if_current(
                        self.db,
                        signal_id,
                        lambda current: {
                            **current,
                            'current_price': current_price,
                            'last_updated_price': datetime.now().isoformat(),
                        },
                    )
                    if saved and latest_mc:
                        signal.market_conditions = latest_mc
                except Exception as db_err:
                    logger.debug(f"market_conditions update skipped for {signal_id}: {db_err}")

                # Generate real-time feedback
                feedback = await self._generate_realtime_feedback(signal_id, current_price, current_pnl)

                if feedback:
                    tracking_data['feedback_events'].append(feedback)
                    await self._apply_realtime_learning(feedback)

        except Exception as e:
            logger.error(f"❌ Real-time monitoring error for {signal_id}: {e}")

    async def _generate_realtime_feedback(self, signal_id: str, current_price: float, current_pnl: float) -> Optional[RealTimeFeedback]:
        """Generate real-time learning feedback."""
        try:
            tracking_data = self.active_signals[signal_id]
            signal = tracking_data['signal']

            # Calculate metrics
            time_elapsed = (datetime.now() - tracking_data['start_time']).total_seconds() / 60
            price_history = tracking_data['price_history']

            if len(price_history) < 5:  # Need minimum data points
                return None

            # Calculate max favorable/adverse moves
            prices = [p[1] for p in price_history]
            entry_price = tracking_data['entry_price']

            if signal.direction == 'LONG':
                max_favorable = max(prices) - entry_price
                max_adverse = entry_price - min(prices)
            else:
                max_favorable = entry_price - min(prices)
                max_adverse = max(prices) - entry_price

            # Detect regime changes and volatility spikes
            regime_change = await self._detect_regime_change()
            volatility_spike = self._detect_volatility_spike(prices)

            # Volume confirmation (simplified - you can enhance this)
            volume_confirmation = True  # Placeholder

            # Support/resistance test
            support_resistance_test = self._test_support_resistance(signal, current_price)

            # Generate adjustment suggestions
            adjustment_suggested = await self._suggest_realtime_adjustments(
                signal, current_pnl, max_favorable, max_adverse, time_elapsed
            )

            # Calculate feedback confidence
            feedback_confidence = self._calculate_feedback_confidence(
                time_elapsed, len(price_history), regime_change, volatility_spike
            )

            return RealTimeFeedback(
                signal_id=signal_id,
                current_pnl=current_pnl,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
                time_elapsed_minutes=int(time_elapsed),
                regime_change_detected=regime_change,
                volatility_spike=volatility_spike,
                volume_confirmation=volume_confirmation,
                support_resistance_test=support_resistance_test,
                feedback_confidence=feedback_confidence,
                adjustment_suggested=adjustment_suggested
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate real-time feedback: {e}")
            return None

    async def _apply_realtime_learning(self, feedback: RealTimeFeedback) -> None:
        """Apply real-time learning adjustments."""
        try:
            # Store feedback for pattern learning
            self.real_time_adjustments.append(feedback)

            # Log learning insights
            if feedback.regime_change_detected:
                logger.info(f"🔄 Regime change detected for {feedback.signal_id} - updating learning model")

            if feedback.volatility_spike:
                logger.info(f"📈 Volatility spike detected for {feedback.signal_id} - adjusting risk parameters")

            # Update immediate learning parameters (for next signals)
            if feedback.feedback_confidence > 0.7:
                await self._update_immediate_parameters(feedback)

        except Exception as e:
            logger.error(f"❌ Failed to apply real-time learning: {e}")

    async def stop_real_time_tracking(self, signal_id: str) -> None:
        """Stop real-time tracking and finalize learning."""
        try:
            if signal_id in self.active_signals:
                tracking_data = self.active_signals[signal_id]

                # Finalize real-time learning
                await self._finalize_realtime_learning(signal_id, tracking_data)

                # Mark tracking record as completed/stopped in DB.
                final_status = 'stopped'
                try:
                    status_res = self.db.from_('platform_signals').select('status').eq('signal_id', signal_id).limit(1).execute()
                    if status_res.data:
                        signal_status = str(status_res.data[0].get('status') or '').lower()
                        if signal_status in self.TERMINAL_SIGNAL_STATUSES:
                            final_status = 'completed'
                except Exception:
                    pass

                now_iso = datetime.now().isoformat()
                try:
                    self.db.from_('platform_signals').update({
                        'learning_tracked': True,
                        'learning_completed_at': now_iso,
                    }).eq('signal_id', signal_id).execute()
                except Exception as sync_error:
                    logger.debug(f"learning_completed_at sync skipped for {signal_id}: {sync_error}")

                try:
                    self.db.from_('platform_signal_performance_tracking').upsert({
                        'signal_id': signal_id,
                        'last_updated': now_iso,
                    }).execute()
                except Exception as perf_sync_error:
                    logger.debug(f"performance tracking touch skipped for {signal_id}: {perf_sync_error}")

                # Clean up
                del self.active_signals[signal_id]
                logger.info(f"✅ Real-time tracking stopped for {signal_id} ({final_status})")

        except Exception as e:
            logger.error(f"❌ Failed to stop real-time tracking for {signal_id}: {e}")

    async def reconcile_tracking_records(self) -> Dict[str, int]:
        """
        Reconcile in-memory active trackers with platform signal status.
        Marks trackers completed when signal is terminal, or stops tracking when signal is missing.
        """
        summary = {'active_rows': 0, 'completed_rows': 0, 'stopped_rows': 0}
        try:
            active_signal_ids = list(self.active_signals.keys())
            summary['active_rows'] = len(active_signal_ids)
            if not active_signal_ids:
                return summary

            signal_status_map: Dict[str, str] = {}

            if active_signal_ids:
                signal_rows = self.db.from_('platform_signals').select(
                    'signal_id,status'
                ).in_('signal_id', active_signal_ids).execute().data or []
                signal_status_map = {
                    str(row.get('signal_id')): str(row.get('status') or '').lower()
                    for row in signal_rows
                    if row.get('signal_id')
                }

            for sid in active_signal_ids:
                status = signal_status_map.get(sid)
                if status in self.TERMINAL_SIGNAL_STATUSES:
                    await self.stop_real_time_tracking(sid)
                    summary['completed_rows'] += 1
                elif not status:
                    self.active_signals.pop(sid, None)
                    summary['stopped_rows'] += 1

            return summary
        except Exception as e:
            logger.warning(f"⚠️ Failed to reconcile tracking records for {self.agent_type}: {e}")
            return summary

    # ===== LEVEL 2: ENHANCED PATTERN LEARNING (15 minutes) =====

    async def update_pattern_learning(self) -> None:
        """Update Level 2 pattern learning with enhanced features."""
        try:
            logger.info("🔄 Running Level 2 Pattern Learning update...")

            def _resolved(signals):
                return [
                    s for s in signals
                    if str(s.get('status') or '').lower() in self.RESOLVED_OUTCOME_STATUSES
                    and s.get('pnl_percentage') is not None
                ]

            # Get recent signals over a rolling window long enough to accumulate
            # resolved outcomes (24h was too short at current volume → stale table).
            lookback_hours = int(self.config.get('pattern_lookback_hours', 24 * 180))
            recent_signals = await self._get_recent_signals(hours=lookback_hours)
            resolved = _resolved(recent_signals)

            # Staleness self-heal: if the rolling window is still starved AND the table has
            # gone stale, fall back to an all-history backfill instead of bailing. This is
            # what breaks the doom loop (tight gates → low volume → never enough samples →
            # frozen table → no learning → bias never corrected).
            if len(resolved) < self.config['min_data_points'] and self._pattern_table_is_stale():
                logger.warning(
                    f"⚠️ Window starved ({len(resolved)} resolved in {lookback_hours}h) and "
                    f"table is stale — falling back to all-history backfill."
                )
                recent_signals = await self._get_recent_signals(hours=24 * 3650)  # ~10y = all
                resolved = _resolved(recent_signals)

            if len(resolved) < self.config['min_data_points']:
                logger.warning(
                    f"⚠️ Insufficient resolved data for pattern learning: "
                    f"{len(resolved)} resolved of {len(recent_signals)} signals"
                )
                return

            # Enhanced pattern extraction
            patterns = await self._extract_enhanced_patterns(recent_signals)

            # Update pattern performance metrics
            for pattern in patterns:
                await self._update_pattern_performance(pattern)

            # Risk-adjusted optimization
            await self._optimize_risk_adjusted_parameters(patterns)

            # Time-based performance analysis
            await self._analyze_temporal_patterns(recent_signals)

            logger.info(f"✅ Pattern learning updated with {len(patterns)} patterns")

        except Exception as e:
            logger.error(f"❌ Pattern learning update failed: {e}")

    def _pattern_table_is_stale(self) -> bool:
        """True if pattern_performance_tracking hasn't updated within the staleness window.

        Used to trigger the all-history backfill fallback so the table can never silently
        freeze for weeks again (the failure we observed Jan→May and again May→June).
        """
        try:
            staleness_hours = float(self.config.get('staleness_hours', 36.0))
            resp = (
                self.db.from_('pattern_performance_tracking')
                .select('last_updated')
                .eq('agent_type', self.agent_type)
                .order('last_updated', desc=True)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows or not rows[0].get('last_updated'):
                return True  # empty table counts as stale
            last = self._parse_datetime(rows[0]['last_updated'])
            age_hours = (datetime.now(last.tzinfo) - last).total_seconds() / 3600.0
            return age_hours > staleness_hours
        except Exception as e:
            logger.warning(f"⚠️ Could not check pattern table staleness: {e}")
            return True  # fail open → prefer to refresh

    async def _extract_enhanced_patterns(self, signals: List[Dict]) -> List[PatternPerformance]:
        """Extract enhanced patterns with advanced features.

        Statistical rigor: requires a configurable minimum of resolved trades per pattern
        (default 8) to avoid learning from pure noise. Small-N patterns are additionally
        DOWN-WEIGHTED (not trusted) downstream by the generator's sample-confidence scaling,
        so a modest floor is safe and avoids starving the table at low volume.
        """
        MIN_PATTERN_SAMPLE = int(self.config.get('min_resolved_for_pattern', 8))
        patterns = []

        try:
            # Group signals by similar market conditions
            pattern_groups = self._group_signals_by_conditions(signals)

            for pattern_name, all_group_signals in pattern_groups.items():
                # Only resolved signals carry a learnable win/loss outcome; expired
                # signals have a near-zero drift PnL that would otherwise be miscounted
                # as wins and inflate the success rate.
                group_signals = [
                    s for s in all_group_signals
                    if str(s.get('status') or '').lower() in self.RESOLVED_OUTCOME_STATUSES
                ]
                if len(group_signals) < MIN_PATTERN_SAMPLE:
                    continue

                # Calculate performance metrics
                pnl_values = [s.get('pnl_percentage', 0) for s in group_signals if s.get('pnl_percentage') is not None]
                if not pnl_values:
                    continue

                wins = [p for p in pnl_values if p > 0]
                losses = [p for p in pnl_values if p <= 0]
                win_rate = len(wins) / len(pnl_values)
                avg_pnl = statistics.mean(pnl_values)

                # Expected Value: E[PnL] = P(win)*avg_win - P(loss)*avg_loss
                avg_win = statistics.mean(wins) if wins else 0.0
                avg_loss = abs(statistics.mean(losses)) if losses else 0.0
                expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

                # Calculate advanced metrics — Sortino replaces Sharpe
                sortino_ratio = self._calculate_sortino_ratio(pnl_values)
                max_drawdown = self._calculate_max_drawdown(pnl_values)

                # Extract common features
                pattern_features = self._extract_pattern_features(group_signals)
                pattern_features['expected_value'] = round(expected_value, 4)

                pattern = PatternPerformance(
                    pattern_name=pattern_name,
                    pattern_features=pattern_features,
                    success_rate=win_rate,
                    sample_size=len(group_signals),
                    avg_pnl=avg_pnl,
                    avg_time_to_target=self._calculate_avg_time_to_target(group_signals),
                    best_market_conditions=self._identify_best_conditions(group_signals),
                    risk_adjusted_return=avg_pnl / max(abs(max_drawdown), 0.01),
                    sharpe_ratio=sortino_ratio,  # field reuse: stores Sortino now
                    max_drawdown=max_drawdown,
                    confidence_correlation=self._calculate_confidence_correlation(group_signals),
                    last_updated=datetime.now()
                )

                patterns.append(pattern)

        except Exception as e:
            logger.error(f"❌ Failed to extract enhanced patterns: {e}")

        return patterns

    # ===== LEVEL 3: META-LEARNING (Daily) =====

    async def update_meta_learning(self) -> MetaLearningInsights:
        """Update Level 3 meta-learning with strategy optimization."""
        try:
            logger.info("🧠 Running Level 3 Meta-Learning update...")

            # Strategy effectiveness evaluation
            strategy_effectiveness = await self._evaluate_strategy_effectiveness()

            # Cross-agent performance comparison
            cross_agent_performance = await self._compare_agent_performance()

            # Prompt optimization analysis
            prompt_suggestions = await self._analyze_prompt_effectiveness()

            # Market condition adaptation
            optimal_conditions = await self._identify_optimal_conditions()

            # Risk parameter optimization
            risk_adjustments = await self._optimize_risk_parameters()

            # Temporal pattern analysis
            temporal_patterns = await self._analyze_temporal_performance()

            # Learning convergence analysis
            convergence_rate = await self._calculate_learning_convergence()

            # Generate adaptation recommendations
            recommendations = await self._generate_adaptation_recommendations(
                strategy_effectiveness, cross_agent_performance, temporal_patterns
            )

            insights = MetaLearningInsights(
                strategy_effectiveness=strategy_effectiveness,
                optimal_market_conditions=optimal_conditions,
                cross_agent_performance=cross_agent_performance,
                prompt_optimization_suggestions=prompt_suggestions,
                risk_parameter_adjustments=risk_adjustments,
                temporal_performance_patterns=temporal_patterns,
                learning_convergence_rate=convergence_rate,
                adaptation_recommendations=recommendations
            )

            # Apply meta-learning improvements
            await self._apply_meta_learning_improvements(insights)

            logger.info("✅ Meta-learning update completed")
            return insights

        except Exception as e:
            logger.error(f"❌ Meta-learning update failed: {e}")
            return None

    # ===== UTILITY METHODS =====

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol using Binance (no Hyperliquid dependency)."""
        try:
            if not symbol:
                return None

            # Normalize symbol: worker typically stores spot/futures as e.g. CLOUSDT, FHEUSDT, AVNTUSDT
            symbols_to_try: List[str] = []

            # If already base (no USDT suffix), try as-is
            if not symbol.endswith("USDT"):
                symbols_to_try.append(symbol)
                symbols_to_try.append(symbol.upper())
                # Also try with explicit USDT suffix
                symbols_to_try.append(f"{symbol.upper()}USDT")
            else:
                # With USDT suffix: try base symbol and original
                base = symbol[:-4]
                symbols_to_try.append(base)
                symbols_to_try.append(base.upper())
                symbols_to_try.append(symbol)
                symbols_to_try.append(symbol.upper())

            # Deduplicate while preserving order
            seen = set()
            normalized_symbols = []
            for s in symbols_to_try:
                if s not in seen:
                    seen.add(s)
                    normalized_symbols.append(s)

            # Ask the lightweight performance service (Binance-backed) for prices
            prices = await self.performance_service.get_current_prices(normalized_symbols)
            if not prices:
                logger.debug(f"No Binance prices returned for symbols: {normalized_symbols}")
                return None

            # Prefer exact/closest match in order
            for s in normalized_symbols:
                price = prices.get(s)
                if price and price > 0:
                    logger.debug(f"Using Binance price for {symbol} via key {s}: {price}")
                    return float(price)

            logger.debug(f"Price not found for {symbol} via Binance (symbols tried: {normalized_symbols})")
            return None
        except Exception as e:
            logger.error(f"Error fetching Binance price for {symbol}: {e}")
            return None

    def _calculate_pnl(self, entry_price: float, current_price: float, direction: str, leverage: int) -> float:
        """Calculate current PnL percentage."""
        price_change = (current_price - entry_price) / entry_price
        if direction == 'SHORT':
            price_change = -price_change
        return price_change * leverage * 100

    def _entry_triggered(self, signal: PlatformSignal, current_price: float) -> bool:
        """Check whether pending entry criteria are satisfied for this signal."""
        mc = signal.market_conditions or {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True

        mode = str(mc.get('entry_activation_mode') or '').strip().lower()
        entry = float(signal.entry_price or 0.0)
        direction = str(signal.direction or '').upper()
        if entry <= 0 or direction not in {'LONG', 'SHORT'}:
            return True

        # Activation must match a real fill: price has to actually reach the entry
        # level. The previous `entry * (1 ± buffer)` activated ~0.15% early on the
        # favorable side, so a resting limit that never filled (e.g. UNI long
        # @ 3.57, price only dipped to ~3.575) was marked activated and later
        # scored as a target "hit" — inflating win rate with entries no real
        # position ever took. A genuine gap-through still satisfies these.
        if mode == 'limit_retest':
            return current_price <= entry if direction == 'LONG' else current_price >= entry
        if mode == 'breakout_stop':
            return current_price >= entry if direction == 'LONG' else current_price <= entry
        return True

    async def _ensure_entry_activated(self, signal_id: str, signal: PlatformSignal, current_price: float) -> bool:
        """Persist entry activation once touched and block learning updates until then."""
        mc = signal.market_conditions or {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True
        if not self._entry_triggered(signal, current_price):
            return False

        try:
            activated, latest_mc, result = persist_entry_activation_if_current(
                self.db,
                signal_id,
                current_price,
            )
            if latest_mc:
                signal.market_conditions = latest_mc
            if result == "activated":
                logger.info(f"✅ Entry activated for learning: {signal.token_symbol} {signal.direction} @ {current_price}")
            return activated
        except Exception as exc:
            logger.warning(f"Failed to persist entry activation for learning {signal_id}: {exc}")
            return False

    async def _detect_regime_change(self) -> bool:
        """Detect if market regime has changed."""
        # Placeholder - implement regime detection logic
        return False

    def _detect_volatility_spike(self, prices: List[float]) -> bool:
        """Detect volatility spike in price series."""
        if len(prices) < 10:
            return False

        recent_volatility = np.std(prices[-10:])
        baseline_volatility = np.std(prices[:-10]) if len(prices) > 10 else recent_volatility

        return bool(recent_volatility > baseline_volatility * 2)

    async def _is_signal_completed(self, signal: PlatformSignal) -> bool:
        """Check if signal has been completed."""
        try:
            normalized_signal = self._coerce_signal(signal)
            if not normalized_signal:
                return False

            # Check signal status in database
            response = self.db.from_('platform_signals').select('status').eq('signal_id', normalized_signal.signal_id).execute()
            if not response.data:
                return False

            current_status = response.data[0].get('status')
            return current_status in self.TERMINAL_SIGNAL_STATUSES
        except Exception as e:
            logger.error(f"❌ Error checking signal completion: {e}")
            return False

    def _test_support_resistance(self, signal, current_price: float) -> bool:
        """Test if current price is near support/resistance levels."""
        entry_price = signal.entry_price
        price_diff = abs(current_price - entry_price) / entry_price

        # Consider significant if price moved more than 1%
        return price_diff > 0.01

    async def _suggest_realtime_adjustments(self, signal, current_pnl: float, max_favorable: float, max_adverse: float, time_elapsed: float) -> Dict[str, float]:
        """Suggest real-time adjustments based on current performance."""
        adjustments = {}

        try:
            # Stop-loss adjustment suggestions
            if max_adverse > 0.02:  # 2% adverse move
                adjustments['stop_loss_tightening'] = 0.5  # Suggest tightening stop

            # Take-profit adjustment suggestions
            if max_favorable > 0.05 and current_pnl < max_favorable * 0.5:  # Gave back 50% of gains
                adjustments['take_profit_lowering'] = 0.3

            # Position size adjustments for future signals
            if current_pnl < -0.03:  # 3% loss
                adjustments['position_size_reduction'] = 0.2

        except Exception as e:
            logger.error(f"❌ Error suggesting adjustments: {e}")

        return adjustments

    def _calculate_feedback_confidence(self, time_elapsed: float, data_points: int, regime_change: bool, volatility_spike: bool) -> float:
        """Calculate confidence in real-time feedback."""
        confidence = 0.5  # Base confidence

        # More data points = higher confidence
        confidence += min(data_points / 50.0, 0.3)

        # Longer time = slightly higher confidence
        confidence += min(time_elapsed / 60.0, 0.1)

        # Regime changes reduce confidence
        if regime_change:
            confidence -= 0.2

        # Volatility spikes reduce confidence
        if volatility_spike:
            confidence -= 0.15

        return max(0.1, min(1.0, confidence))

    async def _update_immediate_parameters(self, feedback: RealTimeFeedback) -> None:
        """Update immediate learning parameters based on feedback."""
        try:
            # Store immediate adjustments for pattern learning
            adjustment_data = {
                'signal_id': feedback.signal_id,
                'agent_type': self.agent_type,
                'adjustment_type': 'real_time',
                'adjustments': feedback.adjustment_suggested,
                'confidence': feedback.feedback_confidence,
                'timestamp': datetime.now().isoformat()
            }

            # Store in learning adjustments table
            self.db.from_('agent_learning_adjustments').insert(adjustment_data).execute()

        except Exception as e:
            logger.error(f"❌ Error updating immediate parameters: {e}")

    async def _finalize_realtime_learning(self, signal_id: str, tracking_data: Dict) -> None:
        """Finalize real-time learning when signal completes."""
        try:
            # Extract learning insights from tracking data
            feedback_events = tracking_data.get('feedback_events', [])

            if feedback_events:
                # Calculate aggregate learning metrics
                avg_confidence = statistics.mean([f.feedback_confidence for f in feedback_events])
                regime_changes = len([f for f in feedback_events if f.regime_change_detected])
                volatility_spikes = len([f for f in feedback_events if f.volatility_spike])

                # Store final learning summary
                learning_summary = {
                    'signal_id': signal_id,
                    'agent_type': self.agent_type,
                    'learning_level': 'real_time',
                    'learning_type': 'real_time_complete',
                    'confidence_score': avg_confidence,
                    'insights': {
                        'regime_changes_detected': regime_changes,
                        'volatility_spikes_detected': volatility_spikes,
                        'total_feedback_events': len(feedback_events),
                        'avg_feedback_confidence': avg_confidence
                    }
                }

                self.db.from_('agent_learning_insights').insert(learning_summary).execute()

        except Exception as e:
            logger.error(f"❌ Error finalizing real-time learning: {e}")

    async def _get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """Get recent signals for pattern analysis, enriched with realized PnL.

        Bug fix: realized PnL is stored in `platform_signal_performance_tracking`,
        NOT on `platform_signals`. Pattern extraction reads `pnl_percentage`, so
        without this join every pattern's PnL list was empty and patterns were
        silently skipped — a primary reason pattern_performance_tracking froze.
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            response = (
                self.db.from_('platform_signals')
                .select('*')
                .gte('created_at', cutoff_time.isoformat())
                .execute()
            )
            signals = response.data or []
            if not signals:
                return []

            # Pull realized PnL/outcome for these signals and merge it in.
            try:
                signal_ids = [s.get('signal_id') for s in signals if s.get('signal_id')]
                perf_by_id: Dict[str, Dict] = {}
                # Chunk to stay within IN-clause limits.
                for i in range(0, len(signal_ids), 200):
                    chunk = signal_ids[i:i + 200]
                    perf = (
                        self.db.from_('platform_signal_performance_tracking')
                        .select('signal_id,outcome,actual_pnl_percent,leveraged_pnl_percent')
                        .in_('signal_id', chunk)
                        .execute()
                    )
                    for row in (perf.data or []):
                        perf_by_id[row['signal_id']] = row

                for s in signals:
                    perf = perf_by_id.get(s.get('signal_id'))
                    if not perf:
                        continue
                    # Prefer leveraged PnL (what the user actually experiences),
                    # fall back to unleveraged.
                    pnl = perf.get('leveraged_pnl_percent')
                    if pnl is None:
                        pnl = perf.get('actual_pnl_percent')
                    if pnl is not None and s.get('pnl_percentage') is None:
                        try:
                            s['pnl_percentage'] = float(pnl)
                        except (TypeError, ValueError):
                            pass
                    if perf.get('outcome') and not s.get('outcome'):
                        s['outcome'] = perf.get('outcome')
            except Exception as enrich_error:
                logger.warning(f"⚠️ Could not enrich signals with realized PnL: {enrich_error}")

            return signals
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []

    def _group_signals_by_conditions(self, signals: List[Dict]) -> Dict[str, List[Dict]]:
        """Group signals by market conditions for pattern analysis."""
        groups = defaultdict(list)

        for signal in signals:
            # Create condition key based on signal characteristics
            direction = signal.get('direction', 'UNKNOWN')
            confidence_range = self._discretize_confidence(signal.get('confidence', 0.5))
            leverage_range = self._discretize_leverage(signal.get('leverage', 1))

            condition_key = f"{direction}_{confidence_range}_{leverage_range}"
            groups[condition_key].append(signal)

        return dict(groups)

    def _discretize_confidence(self, confidence: float) -> str:
        """Discretize confidence into ranges."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.65:
            return "medium_high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"

    def _discretize_leverage(self, leverage: int) -> str:
        """Discretize leverage into ranges."""
        if leverage >= 10:
            return "very_high"
        elif leverage >= 5:
            return "high"
        elif leverage >= 3:
            return "medium"
        else:
            return "low"

    def _signal_matches_agent(self, signal: Dict[str, Any]) -> bool:
        """Check if a database signal row belongs to this service's agent."""
        generated_by = signal.get('generated_by_agent')
        if self.agent_type == 'yuki':
            return generated_by in {None, 'yuki', 'platform'}
        return generated_by == self.agent_type

    def _signal_outcome_value(self, signal: Dict[str, Any]) -> float:
        """Convert signal outcome into a numeric value."""
        pnl_value = signal.get('pnl_percentage')
        if pnl_value is not None:
            try:
                return float(pnl_value)
            except (TypeError, ValueError):
                pass

        status = str(signal.get('status') or '').lower()
        if status in {'hit_target_1', 'hit_target_2'}:
            return 1.0
        if status == 'hit_stop_loss':
            return -1.0
        return 0.0

    async def _update_pattern_performance(self, pattern: PatternPerformance) -> None:
        """Persist and cache pattern-level performance metrics."""
        try:
            self.pattern_performance[pattern.pattern_name] = pattern
            self.pattern_cache[pattern.pattern_name] = pattern

            timeframe_distribution = pattern.pattern_features.get('timeframe_distribution', {})
            optimal_timeframes = []
            if isinstance(timeframe_distribution, dict):
                optimal_timeframes = [
                    str(timeframe)
                    for timeframe, _ in sorted(
                        timeframe_distribution.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )[:3]
                ]

            record = {
                'pattern_name': pattern.pattern_name,
                'agent_type': self.agent_type,
                'pattern_features': pattern.pattern_features,
                'market_conditions': pattern.best_market_conditions,
                'sample_size': pattern.sample_size,
                'success_rate': pattern.success_rate,
                'avg_pnl': pattern.avg_pnl,
                'avg_time_to_target': pattern.avg_time_to_target,
                'risk_adjusted_return': pattern.risk_adjusted_return,
                'sharpe_ratio': pattern.sharpe_ratio,
                'max_drawdown': pattern.max_drawdown,
                'confidence_correlation': pattern.confidence_correlation,
                'best_market_conditions': pattern.best_market_conditions,
                'optimal_timeframes': optimal_timeframes,
                'last_updated': pattern.last_updated.isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            self.db.from_('pattern_performance_tracking').upsert(
                record,
                on_conflict='pattern_name,agent_type'
            ).execute()
        except Exception as e:
            logger.error(f"❌ Failed to update pattern performance for {pattern.pattern_name}: {e}")

    async def _optimize_risk_adjusted_parameters(self, patterns: List[PatternPerformance]) -> None:
        """Derive risk adjustments from recent pattern performance.

        Estimation-error safeguard: adjustments are scaled by
        min(1.0, sqrt(n / 30)) so small samples produce proportionally
        smaller changes. This prevents the system from overreacting to
        patterns that haven't yet proven statistically meaningful.
        """
        if not patterns:
            return

        try:
            MIN_SAMPLE = 15  # Raised from 3 — same threshold as pattern extraction
            active_patterns = [p for p in patterns if p.sample_size >= MIN_SAMPLE]
            if not active_patterns:
                return

            avg_success = statistics.mean([p.success_rate for p in active_patterns])
            avg_drawdown = statistics.mean([p.max_drawdown for p in active_patterns])
            avg_sortino = statistics.mean([p.sharpe_ratio for p in active_patterns])  # now Sortino

            # Sample-size dampening factor: sqrt(n / 30) capped at 1.0
            total_samples = sum(p.sample_size for p in active_patterns)
            avg_sample = total_samples / len(active_patterns)
            dampening = min(1.0, (avg_sample / 30) ** 0.5)

            # Raw adjustments
            raw_leverage = (avg_sortino - 1.0) * 0.08
            raw_position = -avg_drawdown * 0.6
            raw_confidence = (avg_success - 0.5) * 0.2

            # Dampen by sample size — small samples → small adjustments
            leverage_multiplier = max(0.7, min(1.3, 1.0 + raw_leverage * dampening))
            position_multiplier = max(0.6, min(1.2, 1.0 + raw_position * dampening))
            confidence_adjustment = max(-0.08, min(0.08, raw_confidence * dampening))

            adjustment_payload = {
                'confidence_adjustment': round(confidence_adjustment, 4),
                'leverage_multiplier': round(leverage_multiplier, 4),
                'position_size_multiplier': round(position_multiplier, 4),
                'pattern_count': len(active_patterns),
                'avg_sample_size': round(avg_sample, 1),
                'dampening_factor': round(dampening, 3),
            }

            self.db.from_('agent_learning_adjustments').insert({
                'signal_id': f"pattern_batch_{int(datetime.now().timestamp())}",
                'agent_type': self.agent_type,
                'adjustment_type': 'pattern_risk_optimization',
                'adjustments': adjustment_payload,
                'confidence': min(max(avg_success, 0.0), 1.0),
                'timestamp': datetime.now().isoformat(),
            }).execute()
        except Exception as e:
            logger.error(f"❌ Risk-adjusted optimization failed: {e}")

    async def _analyze_temporal_patterns(self, signals: List[Dict]) -> Dict[str, float]:
        """Analyze hourly performance patterns."""
        temporal_buckets = defaultdict(list)

        try:
            for signal in signals:
                if not self._signal_matches_agent(signal):
                    continue

                analysis_time = self._parse_datetime(signal.get('analysis_timestamp') or signal.get('created_at'))
                temporal_buckets[str(analysis_time.hour)].append(self._signal_outcome_value(signal))

            temporal_performance = {
                hour: round(statistics.mean(values), 4)
                for hour, values in temporal_buckets.items()
                if values
            }

            if temporal_performance:
                self.db.from_('agent_learning_insights').insert({
                    'agent_type': self.agent_type,
                    'learning_level': 'pattern',
                    'learning_type': 'temporal_pattern_analysis',
                    'insights': {'hourly_performance': temporal_performance},
                    'confidence_score': 0.7,
                    'timestamp': datetime.now().isoformat(),
                }).execute()

            return temporal_performance
        except Exception as e:
            logger.error(f"❌ Temporal pattern analysis failed: {e}")
            return {}

    def _calculate_sharpe_ratio(self, pnl_values: List[float]) -> float:
        """Calculate simplified Sharpe ratio (kept for backward compat)."""
        if len(pnl_values) < 2:
            return 0.0

        std_dev = float(np.std(pnl_values))
        if std_dev <= 1e-9:
            return 0.0
        return float(np.mean(pnl_values) / std_dev)

    def _calculate_sortino_ratio(self, pnl_values: List[float]) -> float:
        """Calculate Sortino ratio — uses downside deviation only.

        More appropriate than Sharpe for crypto, whose returns are
        asymmetric (fat left tail). Sortino penalizes only harmful
        volatility (losses), not beneficial volatility (big winners).
        """
        if len(pnl_values) < 2:
            return 0.0

        mean_pnl = float(np.mean(pnl_values))
        # Downside deviation: std of values below zero only
        downside = [min(p, 0.0) for p in pnl_values]
        downside_dev = float(np.std(downside))
        if downside_dev <= 1e-9:
            # No downside: either all-positive or identical values
            return 2.0 if mean_pnl > 0 else 0.0
        return float(mean_pnl / downside_dev)

    def _calculate_max_drawdown(self, pnl_values: List[float]) -> float:
        """Calculate max drawdown from sequential pnl values."""
        if not pnl_values:
            return 0.0

        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0

        for pnl in pnl_values:
            equity *= max(0.0001, 1.0 + (pnl / 100.0))
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        return float(max_drawdown)

    def _extract_pattern_features(self, signals: List[Dict]) -> Dict[str, Any]:
        """Extract aggregate pattern features from grouped signals."""
        if not signals:
            return {}

        direction_distribution = defaultdict(int)
        timeframe_distribution = defaultdict(int)
        confidence_values = []
        leverage_values = []
        symbols = set()
        volatility_values = []

        for signal in signals:
            direction_distribution[str(signal.get('direction') or 'UNKNOWN')] += 1
            timeframe_distribution[str(signal.get('timeframe') or 'unknown')] += 1
            symbols.add(str(signal.get('token_symbol') or signal.get('symbol') or 'unknown'))

            try:
                confidence_values.append(float(signal.get('confidence', 0.5)))
            except (TypeError, ValueError):
                pass

            try:
                leverage_values.append(float(signal.get('leverage', 1)))
            except (TypeError, ValueError):
                pass

            market_conditions = signal.get('market_conditions') or {}
            try:
                volatility_values.append(float(market_conditions.get('volatility_24h')))
            except (TypeError, ValueError):
                pass

        return {
            'direction_distribution': dict(direction_distribution),
            'timeframe_distribution': dict(timeframe_distribution),
            'symbol_diversity': len(symbols),
            'avg_confidence': statistics.mean(confidence_values) if confidence_values else 0.0,
            'avg_leverage': statistics.mean(leverage_values) if leverage_values else 1.0,
            'avg_volatility_24h': statistics.mean(volatility_values) if volatility_values else 0.0,
        }

    def _calculate_avg_time_to_target(self, signals: List[Dict]) -> float:
        """Calculate average completion time in minutes for terminal signals."""
        completion_minutes = []

        for signal in signals:
            status = str(signal.get('status') or '').lower()
            if status not in self.TERMINAL_SIGNAL_STATUSES:
                continue

            start_time = self._parse_datetime(signal.get('analysis_timestamp') or signal.get('created_at'))
            end_time = self._parse_datetime(
                signal.get('updated_at') or signal.get('completed_at') or signal.get('learning_completed_at'),
                default=start_time
            )
            elapsed_minutes = max(0.0, (end_time - start_time).total_seconds() / 60.0)
            completion_minutes.append(elapsed_minutes)

        if not completion_minutes:
            return 0.0
        return float(statistics.mean(completion_minutes))

    def _identify_best_conditions(self, signals: List[Dict]) -> Dict[str, Any]:
        """Identify conditions associated with strongest signal outcomes."""
        if not signals:
            return {}

        winning_signals = [
            signal for signal in signals
            if self._signal_outcome_value(signal) > 0
        ]
        reference_set = winning_signals if winning_signals else signals

        direction_counts = defaultdict(int)
        timeframe_counts = defaultdict(int)
        confidence_values = []
        leverage_values = []

        for signal in reference_set:
            direction_counts[str(signal.get('direction') or 'UNKNOWN')] += 1
            timeframe_counts[str(signal.get('timeframe') or 'unknown')] += 1
            try:
                confidence_values.append(float(signal.get('confidence', 0.5)))
            except (TypeError, ValueError):
                pass
            try:
                leverage_values.append(float(signal.get('leverage', 1)))
            except (TypeError, ValueError):
                pass

        best_direction = max(direction_counts.items(), key=lambda item: item[1])[0] if direction_counts else 'UNKNOWN'
        best_timeframe = max(timeframe_counts.items(), key=lambda item: item[1])[0] if timeframe_counts else 'unknown'

        return {
            'best_direction': best_direction,
            'best_timeframe': best_timeframe,
            'median_confidence': float(statistics.median(confidence_values)) if confidence_values else 0.5,
            'median_leverage': float(statistics.median(leverage_values)) if leverage_values else 1.0,
            'win_sample_size': len(winning_signals),
        }

    def _calculate_confidence_correlation(self, signals: List[Dict]) -> float:
        """Calculate correlation between model confidence and realized outcome.

        Bug fix: previously this included unresolved/expired signals, all of which
        map to outcome 0.0. With most signals expired, the outcome series had near-zero
        variance and the function returned a misleading 0.0 — making the dashboard claim
        confidence carried no signal, which the resolved-only data contradicts.
        We now correlate ONLY resolved signals, using realized PnL as a continuous
        outcome when available (richer than a ±1 win/loss flag).
        """
        confidence_values = []
        outcome_values = []

        for signal in signals:
            status = str(signal.get('status') or '').lower()
            if status not in self.RESOLVED_OUTCOME_STATUSES:
                continue  # only resolved signals carry a real outcome
            try:
                conf = float(signal.get('confidence', 0.5))
            except (TypeError, ValueError):
                continue
            pnl = signal.get('pnl_percentage')
            if pnl is not None:
                try:
                    outcome = float(pnl)
                except (TypeError, ValueError):
                    outcome = float(self._signal_outcome_value(signal))
            else:
                outcome = float(self._signal_outcome_value(signal))
            confidence_values.append(conf)
            outcome_values.append(outcome)

        if len(confidence_values) < 3:
            return 0.0
        if np.std(confidence_values) <= 1e-9 or np.std(outcome_values) <= 1e-9:
            return 0.0

        correlation = float(np.corrcoef(confidence_values, outcome_values)[0, 1])
        if np.isnan(correlation):
            return 0.0
        return max(-1.0, min(1.0, correlation))

    async def _evaluate_strategy_effectiveness(self) -> Dict[str, float]:
        """Evaluate recent strategy effectiveness for this agent."""
        try:
            recent_signals = await self._get_recent_signals(hours=24 * 7)
            agent_signals = [s for s in recent_signals if self._signal_matches_agent(s)]
            if not agent_signals:
                return {
                    'sample_size': 0,
                    'win_rate': 0.0,
                    'avg_outcome': 0.0,
                    'active_signal_ratio': 0.0,
                }

            wins = sum(1 for s in agent_signals if str(s.get('status') or '').lower() in {'hit_target_1', 'hit_target_2'})
            terminal = [
                s for s in agent_signals
                if str(s.get('status') or '').lower() in self.TERMINAL_SIGNAL_STATUSES
            ]
            win_rate = (wins / len(terminal) * 100) if terminal else 0.0
            avg_outcome = statistics.mean([self._signal_outcome_value(s) for s in terminal]) if terminal else 0.0
            active_ratio = len([s for s in agent_signals if s.get('status') == 'active']) / len(agent_signals)

            return {
                'sample_size': float(len(agent_signals)),
                'win_rate': round(win_rate, 2),
                'avg_outcome': round(avg_outcome, 4),
                'active_signal_ratio': round(active_ratio, 4),
            }
        except Exception as e:
            logger.error(f"❌ Strategy effectiveness evaluation failed: {e}")
            return {'sample_size': 0.0, 'win_rate': 0.0, 'avg_outcome': 0.0, 'active_signal_ratio': 0.0}

    async def _compare_agent_performance(self) -> Dict[str, Dict[str, float]]:
        """Compare recent performance across agents."""
        try:
            recent_signals = await self._get_recent_signals(hours=24 * 7)
            grouped_signals = defaultdict(list)

            for signal in recent_signals:
                agent = signal.get('generated_by_agent') or 'platform'
                grouped_signals[agent].append(signal)

            comparison = {}
            for agent, rows in grouped_signals.items():
                terminal = [
                    row for row in rows
                    if str(row.get('status') or '').lower() in self.TERMINAL_SIGNAL_STATUSES
                ]
                wins = len([row for row in terminal if str(row.get('status') or '').lower() in {'hit_target_1', 'hit_target_2'}])
                win_rate = (wins / len(terminal) * 100) if terminal else 0.0
                avg_outcome = statistics.mean([self._signal_outcome_value(row) for row in terminal]) if terminal else 0.0

                comparison[agent] = {
                    'sample_size': float(len(rows)),
                    'terminal_sample_size': float(len(terminal)),
                    'win_rate': round(win_rate, 2),
                    'avg_outcome': round(avg_outcome, 4),
                }

            if self.agent_type not in comparison:
                comparison[self.agent_type] = {
                    'sample_size': 0.0,
                    'terminal_sample_size': 0.0,
                    'win_rate': 0.0,
                    'avg_outcome': 0.0,
                }

            return comparison
        except Exception as e:
            logger.error(f"❌ Cross-agent performance comparison failed: {e}")
            return {}

    async def _analyze_prompt_effectiveness(self) -> List[str]:
        """Generate prompt optimization suggestions from recent outcomes."""
        strategy_effectiveness = await self._evaluate_strategy_effectiveness()
        recommendations: List[str] = []

        sample_size = strategy_effectiveness.get('sample_size', 0)
        win_rate = strategy_effectiveness.get('win_rate', 0)
        avg_outcome = strategy_effectiveness.get('avg_outcome', 0)

        if sample_size < self.config['min_data_points']:
            recommendations.append("Increase signal volume before major prompt changes to avoid overfitting.")
        if win_rate < 45:
            recommendations.append("Raise minimum confidence threshold in prompts during weak market conditions.")
        if avg_outcome < 0:
            recommendations.append("Bias prompts toward tighter downside risk framing and earlier invalidation criteria.")
        if not recommendations:
            recommendations.append("Current prompt quality is stable; continue incremental tuning with weekly reviews.")

        return recommendations

    async def _identify_optimal_conditions(self) -> Dict[str, Any]:
        """Identify optimal market conditions for this agent."""
        try:
            recent_signals = await self._get_recent_signals(hours=24 * 14)
            agent_winners = [
                signal for signal in recent_signals
                if self._signal_matches_agent(signal)
                and str(signal.get('status') or '').lower() in {'hit_target_1', 'hit_target_2'}
            ]
            if not agent_winners:
                return {}

            best_conditions = self._identify_best_conditions(agent_winners)
            confidence_values = [
                float(signal.get('confidence'))
                for signal in agent_winners
                if signal.get('confidence') is not None
            ]
            if confidence_values:
                best_conditions['recommended_confidence_floor'] = round(float(np.percentile(confidence_values, 25)), 3)

            return best_conditions
        except Exception as e:
            logger.error(f"❌ Failed to identify optimal conditions: {e}")
            return {}

    async def _optimize_risk_parameters(self) -> Dict[str, float]:
        """Calculate risk parameter multipliers based on recent agent performance."""
        try:
            recent_signals = await self._get_recent_signals(hours=24 * 7)
            agent_terminal = [
                signal for signal in recent_signals
                if self._signal_matches_agent(signal)
                and str(signal.get('status') or '').lower() in self.TERMINAL_SIGNAL_STATUSES
            ]
            if not agent_terminal:
                return {
                    'leverage_multiplier': 1.0,
                    'position_size_multiplier': 1.0,
                    'confidence_threshold_adjustment': 0.0,
                }

            wins = len([s for s in agent_terminal if str(s.get('status') or '').lower() in {'hit_target_1', 'hit_target_2'}])
            stop_losses = len([s for s in agent_terminal if str(s.get('status') or '').lower() == 'hit_stop_loss'])
            win_rate = wins / len(agent_terminal)
            stop_rate = stop_losses / len(agent_terminal)

            leverage_multiplier = max(0.7, min(1.3, 1.0 + ((win_rate - stop_rate) * 0.35)))
            position_size_multiplier = max(0.65, min(1.2, 1.0 + ((win_rate - 0.5) * 0.3)))
            confidence_adjustment = max(-0.08, min(0.08, (stop_rate - 0.3) * 0.15))

            return {
                'leverage_multiplier': round(leverage_multiplier, 4),
                'position_size_multiplier': round(position_size_multiplier, 4),
                'confidence_threshold_adjustment': round(confidence_adjustment, 4),
            }
        except Exception as e:
            logger.error(f"❌ Risk parameter optimization failed: {e}")
            return {
                'leverage_multiplier': 1.0,
                'position_size_multiplier': 1.0,
                'confidence_threshold_adjustment': 0.0,
            }

    async def _analyze_temporal_performance(self) -> Dict[str, float]:
        """Return top performing hours for this agent."""
        try:
            recent_signals = await self._get_recent_signals(hours=24 * 14)
            hourly_values = defaultdict(list)

            for signal in recent_signals:
                if not self._signal_matches_agent(signal):
                    continue
                analysis_time = self._parse_datetime(signal.get('analysis_timestamp') or signal.get('created_at'))
                hourly_values[str(analysis_time.hour)].append(self._signal_outcome_value(signal))

            hourly_scores = {
                hour: statistics.mean(values)
                for hour, values in hourly_values.items()
                if values
            }

            top_hours = sorted(hourly_scores.items(), key=lambda item: item[1], reverse=True)[:6]
            return {hour: round(score, 4) for hour, score in top_hours}
        except Exception as e:
            logger.error(f"❌ Temporal performance analysis failed: {e}")
            return {}

    async def _calculate_learning_convergence(self) -> float:
        """Estimate convergence of pattern learning updates."""
        try:
            result = self.db.from_('pattern_performance_tracking').select(
                'success_rate,last_updated'
            ).eq('agent_type', self.agent_type).order('last_updated', desc=True).limit(30).execute()

            rows = result.data or []
            success_values = [
                float(row['success_rate'])
                for row in rows
                if row.get('success_rate') is not None
            ]
            if len(success_values) < 3:
                return 0.0

            recent_std = float(np.std(success_values[:10]))
            baseline_std = float(np.std(success_values))
            variance_ratio = recent_std / baseline_std if baseline_std > 1e-9 else 0.0

            convergence = 1.0 - min(1.0, variance_ratio)
            return round(max(0.0, min(1.0, convergence)), 4)
        except Exception as e:
            logger.error(f"❌ Learning convergence calculation failed: {e}")
            return 0.0

    async def _generate_adaptation_recommendations(
        self,
        strategy_effectiveness: Dict[str, float],
        cross_agent_performance: Dict[str, Dict[str, float]],
        temporal_patterns: Dict[str, float]
    ) -> List[str]:
        """Generate actionable adaptation recommendations from meta metrics."""
        recommendations: List[str] = []

        win_rate = float(strategy_effectiveness.get('win_rate', 0.0))
        if win_rate < 45.0:
            recommendations.append("Reduce low-confidence setups and prioritize higher-conviction entries.")

        avg_outcome = float(strategy_effectiveness.get('avg_outcome', 0.0))
        if avg_outcome < 0:
            recommendations.append("Tighten stop placement logic and shorten invalidation windows.")

        current_agent = cross_agent_performance.get(self.agent_type, {})
        current_win_rate = float(current_agent.get('win_rate', 0.0))
        for agent, metrics in cross_agent_performance.items():
            if agent == self.agent_type:
                continue
            peer_win_rate = float(metrics.get('win_rate', 0.0))
            if peer_win_rate > current_win_rate + 8:
                recommendations.append(f"Review {agent} signal heuristics and reuse its strongest filters.")
                break

        if temporal_patterns:
            best_hour = max(temporal_patterns.items(), key=lambda item: item[1])[0]
            recommendations.append(f"Bias signal generation toward higher-performing hour bucket: {best_hour}:00 UTC.")

        if not recommendations:
            recommendations.append("No urgent adaptations required; continue current learning cadence.")

        return recommendations[:5]

    async def _apply_meta_learning_improvements(self, insights: MetaLearningInsights) -> None:
        """Persist meta-learning outputs for downstream optimization."""
        try:
            analysis_date = datetime.now().date().isoformat()

            meta_record = {
                'analysis_date': analysis_date,
                'agent_type': self.agent_type,
                'time_period_hours': 24,
                'strategy_effectiveness': insights.strategy_effectiveness,
                'cross_agent_performance': insights.cross_agent_performance,
                'optimal_market_conditions': insights.optimal_market_conditions,
                'learning_convergence_rate': insights.learning_convergence_rate,
                'adaptation_recommendations': insights.adaptation_recommendations,
                'risk_parameter_adjustments': insights.risk_parameter_adjustments,
                'temporal_performance_patterns': insights.temporal_performance_patterns,
                'analysis_timestamp': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            existing = self.db.from_('meta_learning_performance').select('id').eq('analysis_date', analysis_date).eq('agent_type', self.agent_type).execute()
            if existing.data:
                self.db.from_('meta_learning_performance').update(meta_record).eq('id', existing.data[0]['id']).execute()
            else:
                self.db.from_('meta_learning_performance').insert(meta_record).execute()

            self.db.from_('agent_learning_insights').insert({
                'agent_type': self.agent_type,
                'learning_level': 'meta',
                'learning_type': 'meta_learning_cycle',
                'insights': asdict(insights),
                'confidence_score': min(max(insights.learning_convergence_rate, 0.0), 1.0),
                'timestamp': datetime.now().isoformat(),
            }).execute()
        except Exception as e:
            logger.error(f"❌ Failed to apply meta-learning improvements: {e}")


# Global service registry
_multi_level_services = {}

def get_multi_level_learning_service(agent_type: str) -> MultiLevelLearningService:
    """Get or create multi-level learning service for agent type."""
    if agent_type not in _multi_level_services:
        _multi_level_services[agent_type] = MultiLevelLearningService(agent_type)
    return _multi_level_services[agent_type]
