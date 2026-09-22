"""
Agent Learning Service for Flow AI Trading Platform

This service enables trading agents to learn from their past performance,
recognize patterns, and adjust their confidence and risk management accordingly.
"""

import asyncio
import hashlib
import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

from kata.config.database import get_service_client
from kata.services.platform_signal_service import PlatformSignal

logger = logging.getLogger(__name__)


@dataclass
class PatternFeatures:
    """Structured pattern features for machine learning."""
    # Market Context
    market_regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile', 'breakout'
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    volume_profile: str  # 'low', 'normal', 'high', 'spike'

    # Technical Indicators
    rsi_range: str  # 'oversold', 'low', 'neutral', 'high', 'overbought'
    trend_strength: str  # 'weak', 'moderate', 'strong'
    momentum_state: str  # 'bearish', 'neutral', 'bullish'

    # Timing Context
    time_category: str  # 'asia', 'europe', 'us', 'overlap'
    day_type: str  # 'weekday', 'weekend'
    timeframe_bucket: str  # 'scalp', 'intraday', 'swing_short', 'swing', 'position'

    # Signal Context
    direction: str  # 'LONG', 'SHORT'
    setup_type: str  # 'bullish_breakout', 'mean_reversion_long', etc.
    leverage_category: str  # 'low', 'medium', 'high'
    confidence_level: str  # 'low', 'medium', 'high'


@dataclass
class LearningPattern:
    """Recognized trading pattern with performance statistics."""
    pattern_name: str
    pattern_hash: str
    conditions: Dict[str, Any]
    success_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_pnl: float
    best_pnl: float
    worst_pnl: float
    confidence_boost: float
    suggested_leverage_multiplier: float
    suggested_position_size_multiplier: float
    last_updated: datetime


@dataclass
class LearningAdjustment:
    """Adjustment recommendations from learning system."""
    confidence_adjustment: float
    leverage_multiplier: float
    position_size_multiplier: float
    reasoning: str
    pattern_matches: List[str]
    confidence_level: str  # 'low', 'medium', 'high'


@dataclass
class ScopedLearningEvidence:
    """Hierarchical evidence slice used to blend token, cluster, and global learning."""
    scope: str
    label: str
    success_rate: float
    total_trades: int
    avg_net_pnl: float
    expected_value: float
    avg_win: float
    avg_loss: float
    confidence_adjustment: float
    leverage_multiplier: float
    position_size_multiplier: float
    weight: float
    confidence_level: str
    pattern_matches: List[str]


class AgentLearningService:
    """Core learning service for trading agents."""

    def __init__(self, agent_type: str):
        """
        Initialize learning service for specific agent type.

        Args:
            agent_type: Type of agent ('yuki', 'sakura', 'ryu')
        """
        self.agent_type = agent_type
        self.db = get_service_client()
        self.pattern_cache = {}
        self.cache_expiry = datetime.now()
        self.cache_duration = timedelta(minutes=30)
        # Same conservative execution-cost model used by the signal generator:
        # 0.04% taker fee + 0.05% assumed slippage per side, round trip.
        self.learning_round_trip_cost_pct = 0.18
        self.learning_expiry_penalty_pct = 0.25
        logger.info(f"Initialized learning service for {agent_type} agent")

    async def record_trade_outcome(
        self,
        signal: PlatformSignal,
        outcome: str,
        pnl_percentage: float,
        exit_reason: Optional[str] = None,
        max_profit_reached: float = 0,
        max_loss_reached: float = 0
    ) -> bool:
        """
        Record trade outcome for learning purposes.

        Args:
            signal: Original platform signal
            outcome: Trade outcome ('win', 'loss', 'scratch')
            pnl_percentage: Final P&L percentage
            exit_reason: Reason for exit ('TARGET_1', 'TARGET_2', 'STOP_LOSS', 'EXPIRED')
            max_profit_reached: Maximum profit reached during trade
            max_loss_reached: Maximum loss reached during trade

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            # Extract pattern features
            pattern_features = self._extract_pattern_features(signal)

            # Generate lesson learned
            lesson = self._generate_lesson(signal, outcome, pnl_percentage, exit_reason)

            # Prepare learning memory record
            learning_data = {
                'agent_type': self.agent_type,
                'signal_id': signal.signal_id,
                'token_symbol': signal.token_symbol,
                'direction': signal.direction,

                # Market context
                'market_conditions': signal.market_conditions,
                'technical_indicators': signal.technical_indicators,
                'sentiment_data': signal.sentiment_data,

                # Signal properties
                'signal_confidence': float(signal.confidence),
                'leverage_used': signal.leverage,
                'entry_price': float(signal.entry_price),
                'target_1': float(signal.target_1),
                'target_2': float(signal.target_2),
                'stop_loss': float(signal.stop_loss),

                # Trade outcome
                'outcome': outcome,
                'exit_reason': exit_reason,
                'pnl_percentage': pnl_percentage,
                'max_profit_reached': max_profit_reached,
                'max_loss_reached': max_loss_reached,

                # Learning features
                'pattern_features': pattern_features.__dict__,
                'lesson_learned': lesson,

                # Timestamps
                'analysis_timestamp': signal.analysis_timestamp.isoformat(),
                'exit_timestamp': datetime.now().isoformat() if outcome != 'active' else None
            }

            # Insert into database
            result = self.db.table('agent_learning_memory').insert(learning_data).execute()

            if result.data:
                logger.info(f"📚 Recorded learning outcome for {signal.token_symbol}: {outcome} ({pnl_percentage:+.2f}%)")

                # Clear pattern cache to force refresh
                self._clear_pattern_cache()
                return True
            else:
                logger.error(f"Failed to record learning outcome for {signal.signal_id}")
                return False

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
            return False

    async def get_learning_adjustment(self, signal: PlatformSignal) -> LearningAdjustment:
        """
        Get learning-based adjustments for signal confidence and risk parameters.

        Args:
            signal: Platform signal to analyze

        Returns:
            LearningAdjustment with recommended changes
        """
        try:
            # Extract pattern features
            features = self._extract_pattern_features(signal)

            # Find hierarchical evidence in priority order:
            # token+timeframe memory, token memory, token-cluster memory, global prior.
            scoped_evidence = await self._find_hierarchical_learning_evidence(signal, features)

            if not scoped_evidence:
                return LearningAdjustment(
                    confidence_adjustment=0.0,
                    leverage_multiplier=1.0,
                    position_size_multiplier=1.0,
                    reasoning="No token, cluster, or global historical pattern data available",
                    pattern_matches=[],
                    confidence_level='medium'
                )

            # Calculate weighted adjustments
            total_weight = sum(max(0.0, e.weight) for e in scoped_evidence)

            if total_weight > 0:
                final_confidence_adj = sum(
                    e.confidence_adjustment * e.weight for e in scoped_evidence
                ) / total_weight
                final_leverage_mult = sum(
                    e.leverage_multiplier * e.weight for e in scoped_evidence
                ) / total_weight
                final_position_mult = sum(
                    e.position_size_multiplier * e.weight for e in scoped_evidence
                ) / total_weight
            else:
                final_confidence_adj = 0
                final_leverage_mult = 1.0
                final_position_mult = 1.0

            # Determine confidence level
            confidence_level = self._determine_scoped_confidence_level(scoped_evidence, total_weight)

            # Generate reasoning
            reasoning = self._generate_scoped_adjustment_reasoning(
                signal,
                features,
                scoped_evidence,
                final_confidence_adj
            )
            pattern_matches = [
                match
                for evidence in scoped_evidence
                for match in evidence.pattern_matches
            ]

            return LearningAdjustment(
                confidence_adjustment=final_confidence_adj,
                leverage_multiplier=final_leverage_mult,
                position_size_multiplier=final_position_mult,
                reasoning=reasoning,
                pattern_matches=pattern_matches,
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.error(f"Error getting learning adjustment: {e}")
            return LearningAdjustment(
                confidence_adjustment=0.0,
                leverage_multiplier=1.0,
                position_size_multiplier=1.0,
                reasoning=f"Error in learning system: {str(e)}",
                pattern_matches=[],
                confidence_level='medium'
            )

    async def get_pre_generation_context(self, signal: PlatformSignal) -> Dict[str, Any]:
        """
        Return scoped learning context before the LLM drafts a signal.

        The context is intentionally conservative:
        - Exact token+timeframe and token memory can veto when net EV is clearly poor.
        - Cluster evidence can veto only with larger samples and very poor net EV.
        - Global cross-token prior never hard-vetoes by itself.
        """
        try:
            features = self._extract_pattern_features(signal)
            evidence = await self._find_hierarchical_learning_evidence(signal, features)
            if not evidence:
                return {
                    "has_evidence": False,
                    "hard_veto": False,
                    "prompt_text": "- No scoped token/timeframe learning evidence available.",
                    "reason": "No token, cluster, or global prior available",
                    "evidence": [],
                }

            veto_evidence = self._select_pre_generation_veto(evidence)
            prompt_text = self._generate_pre_generation_prompt_text(signal, features, evidence)
            return {
                "has_evidence": True,
                "hard_veto": veto_evidence is not None,
                "veto_scope": veto_evidence.scope if veto_evidence else None,
                "veto_reason": (
                    f"{veto_evidence.label} net_EV={veto_evidence.expected_value:+.2f}% "
                    f"WR={veto_evidence.success_rate:.1%} n={veto_evidence.total_trades}"
                    if veto_evidence
                    else None
                ),
                "prompt_text": prompt_text,
                "reason": self._summarize_scoped_evidence(evidence),
                "evidence": [
                    {
                        "scope": item.scope,
                        "label": item.label,
                        "success_rate": round(float(item.success_rate), 4),
                        "total_trades": int(item.total_trades),
                        "expected_value": round(float(item.expected_value), 4),
                        "avg_net_pnl": round(float(item.avg_net_pnl), 4),
                        "weight": round(float(item.weight), 4),
                        "confidence_level": item.confidence_level,
                    }
                    for item in evidence
                ],
            }
        except Exception as e:
            logger.debug(f"Pre-generation learning context skipped: {e}")
            return {
                "has_evidence": False,
                "hard_veto": False,
                "prompt_text": "- Scoped learning context unavailable.",
                "reason": f"learning_context_error: {e}",
                "evidence": [],
            }

    async def evaluate_pre_entry_gate(self, signal: Any) -> Dict[str, Any]:
        """
        Evaluate a learned pass / defer / invalidate gate for a prospective entry.

        The gate is intentionally conservative:
        - outcome-backed hard vetoes invalidate the setup,
        - uncertain or weakly supported setups defer for re-analysis,
        - clearly supportive setups pass through.
        """
        signal_like = self._coerce_learning_signal(signal)

        try:
            context = await self.get_pre_generation_context(signal_like)
            adjustment = await self.get_learning_adjustment(signal_like)

            signal_confidence = max(0.0, min(1.0, self._safe_float(getattr(signal_like, 'confidence', None), 0.0)))
            effective_confidence = max(
                0.0,
                min(1.0, signal_confidence + adjustment.confidence_adjustment),
            )
            evidence = context.get("evidence") or []
            weighted_expected_value = self._gate_weighted_expected_value(evidence)
            strongest_evidence = self._strongest_gate_evidence(evidence)
            strongest_scope = strongest_evidence.get("scope") if strongest_evidence else None
            strongest_expected_value = self._safe_float(
                strongest_evidence.get("expected_value") if strongest_evidence else None,
                0.0,
            )
            evidence_summary = context.get("reason") or self._summarize_scoped_evidence(
                [
                    ScopedLearningEvidence(
                        scope=str(item.get("scope") or ""),
                        label=str(item.get("label") or ""),
                        success_rate=self._safe_float(item.get("success_rate"), 0.0),
                        total_trades=int(item.get("total_trades") or 0),
                        avg_net_pnl=self._safe_float(item.get("avg_net_pnl"), 0.0),
                        expected_value=self._safe_float(item.get("expected_value"), 0.0),
                        avg_win=0.0,
                        avg_loss=0.0,
                        confidence_adjustment=0.0,
                        leverage_multiplier=1.0,
                        position_size_multiplier=1.0,
                        weight=self._safe_float(item.get("weight"), 0.0),
                        confidence_level=str(item.get("confidence_level") or "low"),
                        pattern_matches=[],
                    )
                    for item in evidence[:4]
                ]
            ) if evidence else "No scoped learning evidence available"

            if context.get("hard_veto"):
                decision = "invalidate"
                reason = context.get("veto_reason") or context.get("reason") or "Outcome-backed hard veto"
            else:
                has_evidence = bool(context.get("has_evidence"))
                strong_positive = (
                    effective_confidence >= 0.80
                    and (
                        weighted_expected_value >= 0.0
                        or strongest_expected_value >= -0.05
                    )
                )
                weak_or_uncertain = (
                    not has_evidence
                    or adjustment.confidence_level == "low"
                    or effective_confidence < 0.70
                    or weighted_expected_value < -0.15
                )

                if not has_evidence and effective_confidence >= 0.82:
                    decision = "pass"
                    reason = "No scoped learning evidence yet, but the setup remains strong"
                elif strong_positive and not weak_or_uncertain:
                    decision = "pass"
                    reason = "Scoped learning evidence supports the entry"
                elif weighted_expected_value <= -0.25 or strongest_expected_value <= -0.35:
                    decision = "defer"
                    reason = "Historical learning is negative enough to wait for a cleaner retest"
                else:
                    decision = "defer"
                    reason = "Scoped learning evidence is mixed or incomplete; holding for re-analysis"

            gate_reason = (
                f"{reason} | {adjustment.reasoning}"
                if adjustment.reasoning
                else reason
            )
            gate_record = {
                "decision": decision,
                "reason": gate_reason,
                "signal_confidence": round(signal_confidence, 4),
                "effective_confidence": round(effective_confidence, 4),
                "learning_confidence_level": adjustment.confidence_level,
                "learning_adjustment": {
                    "confidence_adjustment": round(float(adjustment.confidence_adjustment), 4),
                    "leverage_multiplier": round(float(adjustment.leverage_multiplier), 4),
                    "position_size_multiplier": round(float(adjustment.position_size_multiplier), 4),
                    "reasoning": adjustment.reasoning,
                    "pattern_matches": adjustment.pattern_matches,
                    "confidence_level": adjustment.confidence_level,
                },
                "pre_generation_context": context,
                "weighted_expected_value": round(weighted_expected_value, 4),
                "strongest_scope": strongest_scope,
                "strongest_expected_value": round(strongest_expected_value, 4),
                "evidence_summary": evidence_summary,
                "applied": decision != "pass",
                "hard_veto": bool(context.get("hard_veto")),
            }
            await self._persist_pre_entry_gate_decision(signal_like, gate_record)
            return gate_record
        except Exception as e:
            logger.debug(f"Pre-entry gate evaluation skipped: {e}")
            gate_record = {
                "decision": "defer",
                "reason": f"pre_entry_gate_error: {e}",
                "signal_confidence": round(
                    max(0.0, min(1.0, self._safe_float(getattr(signal_like, 'confidence', None), 0.0))),
                    4,
                ),
                "effective_confidence": round(
                    max(0.0, min(1.0, self._safe_float(getattr(signal_like, 'confidence', None), 0.0))),
                    4,
                ),
                "learning_confidence_level": "low",
                "learning_adjustment": {
                    "confidence_adjustment": 0.0,
                    "leverage_multiplier": 1.0,
                    "position_size_multiplier": 1.0,
                    "reasoning": f"Error in learning system: {e}",
                    "pattern_matches": [],
                    "confidence_level": "low",
                },
                "pre_generation_context": {
                    "has_evidence": False,
                    "hard_veto": False,
                    "reason": f"learning_context_error: {e}",
                    "evidence": [],
                },
                "weighted_expected_value": 0.0,
                "strongest_scope": None,
                "strongest_expected_value": 0.0,
                "evidence_summary": f"pre_entry_gate_error: {e}",
                "applied": True,
                "hard_veto": False,
            }
            await self._persist_pre_entry_gate_decision(signal_like, gate_record)
            return gate_record

    def _coerce_learning_signal(self, signal: Any) -> Any:
        """Convert dict-based live execution payloads into a signal-like object."""
        if not isinstance(signal, dict):
            return signal

        raw_direction = str(signal.get("direction") or signal.get("signal") or "").upper().strip()
        if raw_direction in {"BUY", "STRONG_BUY", "LONG"}:
            direction = "LONG"
        elif raw_direction in {"SELL", "STRONG_SELL", "SHORT"}:
            direction = "SHORT"
        else:
            direction = raw_direction or "LONG"

        analysis_timestamp = signal.get("analysis_timestamp") or signal.get("timestamp") or datetime.now()
        if isinstance(analysis_timestamp, str):
            try:
                analysis_timestamp = datetime.fromisoformat(analysis_timestamp.replace("Z", "+00:00"))
            except Exception:
                analysis_timestamp = datetime.now()
        elif not isinstance(analysis_timestamp, datetime):
            analysis_timestamp = datetime.now()

        timeframe = str(signal.get("timeframe") or signal.get("time_horizon") or "4h")

        return SimpleNamespace(
            signal_id=str(signal.get("signal_id") or signal.get("id") or ""),
            token_symbol=str(signal.get("token_symbol") or signal.get("symbol") or ""),
            direction=direction,
            timeframe=timeframe,
            confidence=self._safe_float(signal.get("confidence"), 0.0),
            market_conditions=signal.get("market_conditions") or {},
            technical_indicators=signal.get("technical_indicators") or {},
            sentiment_data=signal.get("sentiment_data") or {},
            risk_factors=signal.get("risk_factors") or [],
            signal_pool=signal.get("signal_pool"),
            opportunity_rank=signal.get("opportunity_rank", 0),
            analysis_timestamp=analysis_timestamp,
            expires_at=signal.get("expires_at") or analysis_timestamp,
            validity_window_hours=signal.get("validity_window_hours") or 12,
            leverage=self._safe_float(signal.get("leverage"), 1.0),
            position_size=self._safe_float(signal.get("position_size"), 0.0),
            time_horizon=signal.get("time_horizon") or timeframe,
            pattern_classification=signal.get("pattern_classification") or signal.get("setup_type"),
        )

    def _gate_weighted_expected_value(self, evidence: List[Dict[str, Any]]) -> float:
        """Compute the EV-weighted center of the learned evidence."""
        total_weight = 0.0
        weighted_sum = 0.0
        for item in evidence or []:
            weight = max(0.0, self._safe_float(item.get("weight"), 0.0))
            if weight <= 0:
                continue
            weighted_sum += self._safe_float(item.get("expected_value"), 0.0) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _strongest_gate_evidence(self, evidence: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the most informative evidence slice for gate logging."""
        if not evidence:
            return None

        scope_rank = {
            'token_timeframe': 0,
            'token_memory': 1,
            'token_cluster': 2,
            'global_prior': 3,
        }

        def score(item: Dict[str, Any]) -> Tuple[float, int, int]:
            expected_value = self._safe_float(item.get("expected_value"), 0.0)
            weight = max(0.0, self._safe_float(item.get("weight"), 0.0))
            total_trades = max(0, int(item.get("total_trades") or 0))
            scope = str(item.get("scope") or "")
            return (
                abs(expected_value) * max(0.1, weight),
                total_trades,
                -scope_rank.get(scope, 99),
            )

        return max(evidence, key=score)

    async def _persist_pre_entry_gate_decision(self, signal: Any, gate_record: Dict[str, Any]) -> None:
        """Persist gate decisions so later outcomes can be compared against them."""
        try:
            signal_id = str(getattr(signal, "signal_id", "") or gate_record.get("signal_id") or "")
            if not signal_id:
                return

            record = {
                "signal_id": signal_id,
                "agent_type": self.agent_type,
                "adjustment_type": "pre_entry_gate",
                "adjustments": {
                    "decision": gate_record.get("decision"),
                    "reason": gate_record.get("reason"),
                    "signal_confidence": gate_record.get("signal_confidence"),
                    "effective_confidence": gate_record.get("effective_confidence"),
                    "weighted_expected_value": gate_record.get("weighted_expected_value"),
                    "strongest_scope": gate_record.get("strongest_scope"),
                    "strongest_expected_value": gate_record.get("strongest_expected_value"),
                    "evidence_summary": gate_record.get("evidence_summary"),
                    "hard_veto": gate_record.get("hard_veto"),
                    "pre_generation_context": gate_record.get("pre_generation_context"),
                    "learning_adjustment": gate_record.get("learning_adjustment"),
                },
                "confidence": max(0.0, min(1.0, self._safe_float(gate_record.get("effective_confidence"), 0.0))),
                "applied": bool(gate_record.get("applied")),
                "applied_at": datetime.now().isoformat() if gate_record.get("applied") else None,
                "timestamp": datetime.now().isoformat(),
            }
            self.db.table('agent_learning_adjustments').insert(record).execute()
        except Exception as e:
            logger.debug(f"Failed to persist pre-entry gate decision: {e}")

    async def _find_similar_patterns(self, features: PatternFeatures) -> List[LearningPattern]:
        """Find patterns similar to current market conditions."""
        try:
            # Check cache first
            if datetime.now() < self.cache_expiry and self.pattern_cache:
                cached_patterns = self.pattern_cache.get(self.agent_type, [])
            else:
                # Fetch patterns from database
                result = self.db.table('agent_learning_patterns').select('*').eq(
                    'agent_type', self.agent_type
                ).gte('total_trades', 3).execute()  # Minimum 3 trades for reliability

                cached_patterns = []
                for record in result.data or []:
                    pattern = LearningPattern(
                        pattern_name=record['pattern_name'],
                        pattern_hash=record['pattern_hash'],
                        conditions=record['conditions'],
                        success_rate=float(record['success_rate']),
                        total_trades=record['total_trades'],
                        winning_trades=record['winning_trades'],
                        losing_trades=record['losing_trades'],
                        avg_pnl=float(record['avg_pnl']),
                        best_pnl=float(record['best_pnl']),
                        worst_pnl=float(record['worst_pnl']),
                        confidence_boost=float(record['confidence_boost']),
                        suggested_leverage_multiplier=float(record['suggested_leverage_multiplier']),
                        suggested_position_size_multiplier=float(record['suggested_position_size_multiplier']),
                        last_updated=self._parse_datetime_safe(record['last_updated'])
                    )
                    cached_patterns.append(pattern)

                # Update cache
                self.pattern_cache[self.agent_type] = cached_patterns
                self.cache_expiry = datetime.now() + self.cache_duration

            # Filter similar patterns
            similar_patterns = []
            for pattern in cached_patterns:
                similarity = self._calculate_pattern_similarity(features, pattern)
                if similarity >= 0.6:  # 60% similarity threshold
                    similar_patterns.append(pattern)

            # Sort by similarity and sample size
            similar_patterns.sort(
                key=lambda p: (
                    self._calculate_pattern_similarity(features, p) * 0.7 +
                    min(1.0, p.total_trades / 20.0) * 0.3
                ),
                reverse=True
            )

            return similar_patterns[:5]  # Top 5 most similar patterns

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    async def _find_hierarchical_learning_evidence(
        self,
        signal: PlatformSignal,
        features: PatternFeatures
    ) -> List[ScopedLearningEvidence]:
        """
        Return partial-pooling evidence in descending specificity:
        exact token+timeframe, token memory, token-cluster, then global prior.
        """
        evidence: List[ScopedLearningEvidence] = []
        symbol = self._normalize_token_symbol(signal.token_symbol)
        cluster = self._classify_token_cluster(symbol, features.volatility_level)

        recent_records = await self._fetch_recent_learning_memory(days=120, limit=1500)
        token_records = [
            r for r in recent_records
            if self._normalize_token_symbol(r.get('token_symbol')) == symbol
        ]

        exact_token_evidence = self._build_memory_scope_evidence(
            scope='token_timeframe',
            label=f"token+timeframe {symbol} {features.timeframe_bucket}",
            records=token_records,
            features=features,
            min_trades=8,
            scope_weight=1.0,
            full_at=20,
            require_timeframe=True,
        )
        if exact_token_evidence:
            evidence.append(exact_token_evidence)
        else:
            token_memory_evidence = self._build_memory_scope_evidence(
                scope='token_memory',
                label=f"token memory {symbol}",
                records=token_records,
                features=features,
                min_trades=8,
                scope_weight=0.75,
                full_at=30,
                require_timeframe=False,
            )
            if token_memory_evidence:
                evidence.append(token_memory_evidence)

        cluster_records = [
            r for r in recent_records
            if self._normalize_token_symbol(r.get('token_symbol')) != symbol
            and self._classify_token_cluster(
                self._normalize_token_symbol(r.get('token_symbol')),
                self._coerce_feature_dict(r.get('pattern_features')).get('volatility_level')
            ) == cluster
        ]
        cluster_evidence = self._build_memory_scope_evidence(
            scope='token_cluster',
            label=f"{cluster} cluster",
            records=cluster_records,
            features=features,
            min_trades=20,
            scope_weight=0.45,
            full_at=60,
            require_timeframe=False,
        )
        if cluster_evidence:
            evidence.append(cluster_evidence)

        global_prior = self._build_global_prior_evidence(
            await self._find_similar_patterns(features),
            features
        )
        if global_prior:
            evidence.append(global_prior)

        scope_order = {
            'token_timeframe': 0,
            'token_memory': 1,
            'token_cluster': 2,
            'global_prior': 3,
        }
        evidence.sort(key=lambda item: scope_order.get(item.scope, 99))
        return evidence

    async def _fetch_recent_learning_memory(self, days: int = 120, limit: int = 1500) -> List[Dict[str, Any]]:
        """Fetch recent resolved learning records for runtime scoped evidence."""
        try:
            result = (
                self.db.table('agent_learning_memory')
                .select('token_symbol,outcome,exit_reason,pnl_percentage,leverage_used,pattern_features,created_at')
                .eq('agent_type', self.agent_type)
                .gte('created_at', (datetime.now() - timedelta(days=days)).isoformat())
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.debug(f"Recent learning memory fetch skipped: {e}")
            return []

    def _build_memory_scope_evidence(
        self,
        scope: str,
        label: str,
        records: List[Dict[str, Any]],
        features: PatternFeatures,
        min_trades: int,
        scope_weight: float,
        full_at: int,
        require_timeframe: bool,
    ) -> Optional[ScopedLearningEvidence]:
        """Build EV-ranked evidence from raw learning memory rows."""
        matched: List[Tuple[Dict[str, Any], float]] = []
        for record in records:
            record_features = self._coerce_feature_dict(record.get('pattern_features'))
            if require_timeframe and record_features.get('timeframe_bucket') != features.timeframe_bucket:
                continue

            similarity = self._calculate_feature_similarity(features, record_features)
            if similarity >= 0.62:
                matched.append((record, similarity))

        if len(matched) < min_trades:
            return None

        matched.sort(key=lambda item: item[1], reverse=True)
        matched = matched[:200]
        net_pnls = [self._net_learning_pnl(record) for record, _ in matched]
        stats = self._calculate_net_ev_stats(net_pnls)
        avg_similarity = sum(sim for _, sim in matched) / len(matched)
        sample_confidence = self._sample_confidence(len(matched), full_at=full_at)
        weight = max(0.0, scope_weight * sample_confidence * avg_similarity)
        leverage_mult, position_mult = self._ev_based_risk_multipliers(
            stats['expected_value'],
            len(matched),
            scope
        )
        confidence_adj = self._ev_based_confidence_adjustment(
            stats['expected_value'],
            stats['success_rate'],
            len(matched),
            scope
        )

        summary = (
            f"{label}: WR={stats['success_rate']:.1%}, "
            f"net_EV={stats['expected_value']:+.2f}%, n={len(matched)}"
        )

        return ScopedLearningEvidence(
            scope=scope,
            label=label,
            success_rate=stats['success_rate'],
            total_trades=len(matched),
            avg_net_pnl=stats['avg_net_pnl'],
            expected_value=stats['expected_value'],
            avg_win=stats['avg_win'],
            avg_loss=stats['avg_loss'],
            confidence_adjustment=confidence_adj,
            leverage_multiplier=leverage_mult,
            position_size_multiplier=position_mult,
            weight=weight,
            confidence_level=self._evidence_confidence_level(scope, len(matched), weight),
            pattern_matches=[summary],
        )

    def _build_global_prior_evidence(
        self,
        patterns: List[LearningPattern],
        features: PatternFeatures
    ) -> Optional[ScopedLearningEvidence]:
        """Convert legacy cross-token pattern table matches into the weakest prior."""
        if not patterns:
            return None

        total_trades = sum(max(0, int(p.total_trades or 0)) for p in patterns)
        if total_trades < 20:
            return None

        weighted_success = sum(p.success_rate * p.total_trades for p in patterns) / total_trades
        weighted_avg_pnl = sum(p.avg_pnl * p.total_trades for p in patterns) / total_trades
        avg_similarity = sum(
            self._calculate_pattern_similarity(features, p) * p.total_trades
            for p in patterns
        ) / total_trades
        net_avg_pnl = float(weighted_avg_pnl) - self.learning_round_trip_cost_pct
        expected_value = net_avg_pnl
        sample_confidence = self._sample_confidence(total_trades, full_at=120)
        weight = max(0.0, 0.25 * sample_confidence * avg_similarity)
        leverage_mult, position_mult = self._ev_based_risk_multipliers(
            expected_value,
            total_trades,
            'global_prior'
        )
        confidence_adj = self._ev_based_confidence_adjustment(
            expected_value,
            weighted_success,
            total_trades,
            'global_prior'
        )
        pattern_names = ', '.join(p.pattern_name for p in patterns[:3])
        summary = (
            f"global agent prior (cross-token): WR={weighted_success:.1%}, "
            f"net_EV={expected_value:+.2f}%, n={total_trades}"
        )
        if pattern_names:
            summary = f"{summary}, patterns={pattern_names}"

        return ScopedLearningEvidence(
            scope='global_prior',
            label='global agent prior (cross-token)',
            success_rate=weighted_success,
            total_trades=total_trades,
            avg_net_pnl=net_avg_pnl,
            expected_value=expected_value,
            avg_win=0.0,
            avg_loss=0.0,
            confidence_adjustment=confidence_adj,
            leverage_multiplier=leverage_mult,
            position_size_multiplier=position_mult,
            weight=weight,
            confidence_level=self._evidence_confidence_level('global_prior', total_trades, weight),
            pattern_matches=[summary],
        )

    def _calculate_net_ev_stats(self, net_pnls: List[float]) -> Dict[str, float]:
        """Calculate win rate and expected value from net PnL observations."""
        if not net_pnls:
            return {
                'success_rate': 0.0,
                'avg_net_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'expected_value': 0.0,
            }

        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        success_rate = len(wins) / len(net_pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expected_value = success_rate * avg_win - (1.0 - success_rate) * avg_loss

        return {
            'success_rate': success_rate,
            'avg_net_pnl': sum(net_pnls) / len(net_pnls),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expected_value': expected_value,
        }

    def _net_learning_pnl(self, record: Dict[str, Any]) -> float:
        """Apply conservative fees/slippage/funding/expiry penalties to learned PnL."""
        pnl = self._safe_float(record.get('pnl_percentage'), 0.0)
        leverage = max(1.0, self._safe_float(record.get('leverage_used'), 1.0))
        net_pnl = pnl - (self.learning_round_trip_cost_pct * leverage)

        exit_reason = str(record.get('exit_reason') or '').upper()
        outcome = str(record.get('outcome') or '').lower()
        if exit_reason == 'EXPIRED':
            net_pnl -= self.learning_expiry_penalty_pct
        elif outcome == 'scratch':
            net_pnl -= self.learning_expiry_penalty_pct * 0.5

        return float(net_pnl)

    def _ev_based_confidence_adjustment(
        self,
        expected_value: float,
        success_rate: float,
        sample_size: int,
        scope: str,
    ) -> float:
        """Use expected value first; allow positive boosts only when evidence is strong."""
        full_at = {
            'token_timeframe': 20,
            'token_memory': 30,
            'token_cluster': 60,
            'global_prior': 120,
        }.get(scope, 60)
        sample_scale = self._sample_confidence(sample_size, full_at=full_at)

        if expected_value < 0:
            base = max(-0.18, expected_value / 10.0)
            if success_rate < 0.35:
                base = min(base, -0.05 - ((0.35 - success_rate) * 0.15))
            return base * sample_scale

        proven_positive = {
            'token_timeframe': sample_size >= 20 and expected_value > 0.35 and success_rate >= 0.48,
            'token_memory': sample_size >= 30 and expected_value > 0.45 and success_rate >= 0.50,
            'token_cluster': sample_size >= 50 and expected_value > 0.60 and success_rate >= 0.52,
            'global_prior': sample_size >= 100 and expected_value > 0.75 and success_rate >= 0.55,
        }.get(scope, False)

        if not proven_positive:
            return 0.0

        cap = 0.10 if scope == 'token_timeframe' else 0.06 if scope == 'token_memory' else 0.04
        base = min(cap, (expected_value / 25.0) + max(0.0, success_rate - 0.50) * 0.06)
        return base * sample_scale

    def _ev_based_risk_multipliers(
        self,
        expected_value: float,
        sample_size: int,
        scope: str,
    ) -> Tuple[float, float]:
        """Downside-only sizing: learning can reduce exposure, never increase it."""
        if expected_value >= 0:
            return 1.0, 1.0

        full_at = {
            'token_timeframe': 20,
            'token_memory': 30,
            'token_cluster': 60,
            'global_prior': 120,
        }.get(scope, 60)
        sample_scale = self._sample_confidence(sample_size, full_at=full_at)
        weakness = min(1.0, abs(expected_value) / 5.0)
        leverage_reduction = min(0.35, 0.30 * weakness * sample_scale)
        position_reduction = min(0.45, 0.40 * weakness * sample_scale)
        return max(0.55, 1.0 - leverage_reduction), max(0.45, 1.0 - position_reduction)

    def _sample_confidence(self, sample_size: int, full_at: int = 80) -> float:
        """Map sample size to a bounded confidence scale."""
        try:
            n = max(0, int(sample_size or 0))
            target = max(1, int(full_at or 80))
            return max(0.0, min(1.0, n / target))
        except Exception:
            return 0.0

    def _evidence_confidence_level(self, scope: str, sample_size: int, weight: float) -> str:
        """Confidence label for a scoped evidence slice."""
        if scope == 'token_timeframe' and sample_size >= 20 and weight >= 0.45:
            return 'high'
        if scope == 'token_memory' and sample_size >= 30 and weight >= 0.35:
            return 'medium'
        if scope == 'token_cluster' and sample_size >= 50 and weight >= 0.25:
            return 'medium'
        if scope == 'global_prior' and sample_size >= 100 and weight >= 0.15:
            return 'medium'
        return 'low'

    def _select_pre_generation_veto(
        self,
        evidence: List[ScopedLearningEvidence]
    ) -> Optional[ScopedLearningEvidence]:
        """Pick a strong negative scoped evidence slice that is allowed to veto pre-LLM."""
        veto_candidates: List[ScopedLearningEvidence] = []
        for item in evidence:
            if item.scope == 'token_timeframe':
                if item.total_trades >= 12 and item.expected_value <= -0.35 and item.success_rate <= 0.45:
                    veto_candidates.append(item)
            elif item.scope == 'token_memory':
                if item.total_trades >= 20 and item.expected_value <= -0.55 and item.success_rate <= 0.42:
                    veto_candidates.append(item)
            elif item.scope == 'token_cluster':
                if item.total_trades >= 60 and item.expected_value <= -0.90 and item.success_rate <= 0.38:
                    veto_candidates.append(item)

        if not veto_candidates:
            return None
        return sorted(veto_candidates, key=lambda item: (item.scope != 'token_timeframe', item.expected_value))[0]

    def _summarize_scoped_evidence(self, evidence: List[ScopedLearningEvidence]) -> str:
        """Compact text summary for logs/rejection reasons."""
        return "; ".join(
            f"{item.label}: EV={item.expected_value:+.2f}% WR={item.success_rate:.1%} n={item.total_trades}"
            for item in evidence[:4]
        )

    def _generate_pre_generation_prompt_text(
        self,
        signal: PlatformSignal,
        features: PatternFeatures,
        evidence: List[ScopedLearningEvidence]
    ) -> str:
        """Prompt-ready scoped learning prior for the LLM."""
        if not evidence:
            return "- No scoped token/timeframe learning evidence available."

        lines = [
            (
                f"- Setup key: {self._normalize_token_symbol(signal.token_symbol)} "
                f"{features.timeframe_bucket} {features.setup_type}/{features.market_regime}"
            )
        ]
        for item in evidence[:4]:
            scope_note = {
                'token_timeframe': 'highest authority',
                'token_memory': 'token fallback',
                'token_cluster': 'partial-pooling cluster',
                'global_prior': 'weak cross-token prior',
            }.get(item.scope, 'learning prior')
            lines.append(
                f"- {item.label} ({scope_note}): WR={item.success_rate:.1%}, "
                f"net_EV={item.expected_value:+.2f}%, avg_net={item.avg_net_pnl:+.2f}%, "
                f"n={item.total_trades}, confidence={item.confidence_level}"
            )

        lines.extend([
            "- Use net_EV before raw win rate; low WR can still be acceptable only if payoff is strong.",
            "- Exact token+timeframe evidence should strongly influence HOLD/size/leverage.",
            "- Cluster/global evidence is a prior only; do not let global history alone override current chart strength.",
            "- If scoped evidence is negative but not veto-level, lower confidence, choose smaller risk, or HOLD.",
        ])
        return "\n".join(lines)

    def _coerce_feature_dict(self, raw_features: Any) -> Dict[str, Any]:
        """Normalize JSON/string feature payloads from old and new learning rows."""
        if isinstance(raw_features, dict):
            return raw_features
        if isinstance(raw_features, str):
            try:
                parsed = json.loads(raw_features)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Best-effort numeric conversion."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_token_symbol(self, symbol: Any) -> str:
        """Normalize spot/futures symbols to a base token key."""
        text = str(symbol or '').upper().strip()
        if '/' in text:
            text = text.split('/')[0]
        if ':' in text:
            text = text.split(':')[0]
        for suffix in ('-PERP', 'PERP'):
            if text.endswith(suffix):
                text = text[:-len(suffix)]
        for suffix in ('USDT', 'USDC', 'USD'):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[:-len(suffix)]
                break
        return text.strip()

    def _classify_token_cluster(self, symbol: str, volatility_level: Optional[str] = None) -> str:
        """Coarse asset families for partial pooling when exact token history is sparse."""
        base = self._normalize_token_symbol(symbol)
        display_base = base[4:] if base.startswith('1000') and len(base) > 4 else base

        majors = {'BTC', 'ETH'}
        large_cap_l1 = {
            'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC', 'BCH',
            'TRX', 'TON', 'SUI', 'NEAR', 'APT', 'ARB', 'OP', 'MATIC', 'POL',
            'ATOM', 'INJ', 'SEI', 'TIA',
        }
        memes = {
            'DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'FLOKI', 'MEME', 'BOME',
            'TURBO', 'PNUT', 'BRETT', 'MOG', 'NEIRO', 'POPCAT', 'MEW',
        }
        ai_tokens = {
            'FET', 'ASI', 'RNDR', 'RENDER', 'TAO', 'WLD', 'AKT', 'AI', 'ARKM',
            'AGIX', 'OCEAN', 'NMR', 'GRT',
        }
        defi_tokens = {
            'UNI', 'AAVE', 'MKR', 'COMP', 'CRV', 'SNX', 'LDO', 'PENDLE',
            'ENA', 'RUNE', 'SUSHI', 'YFI', 'GMX', 'DYDX', 'CAKE',
        }

        if display_base in majors:
            return 'majors'
        if display_base in memes:
            return 'memes'
        if display_base in ai_tokens:
            return 'ai_tokens'
        if display_base in defi_tokens:
            return 'defi_tokens'
        if display_base in large_cap_l1:
            return 'large_cap_alts'
        if str(volatility_level or '').lower() in {'high', 'extreme'}:
            return 'high_vol_alts'
        return 'alts'

    def _extract_pattern_features(self, signal: PlatformSignal) -> PatternFeatures:
        """Extract standardized pattern features from signal."""
        try:
            # Market context
            market_conditions = signal.market_conditions or {}
            technical_indicators = signal.technical_indicators or {}
            market_regime = self._categorize_market_regime(market_conditions)
            volatility_level = self._categorize_volatility(market_conditions)
            volume_profile = self._categorize_volume(market_conditions)
            rsi_value = technical_indicators.get('rsi_14')
            if rsi_value is None:
                rsi_value = technical_indicators.get('rsi')
            trend_strength = self._categorize_trend_strength(
                technical_indicators.get('adx'),
                technical_indicators.get('strength_score')
            )
            momentum_state = self._categorize_momentum(technical_indicators)
            direction = str(signal.direction or '').upper()
            timeframe_bucket = self._categorize_timeframe(
                getattr(signal, 'timeframe', None),
                getattr(signal, 'time_horizon', None)
            )

            # Extract and categorize features
            features = PatternFeatures(
                # Market regime
                market_regime=market_regime,
                volatility_level=volatility_level,
                volume_profile=volume_profile,

                # Technical indicators
                rsi_range=self._categorize_rsi(rsi_value),
                trend_strength=trend_strength,
                momentum_state=momentum_state,

                # Timing
                time_category=self._categorize_time_of_day(signal.analysis_timestamp),
                day_type=self._categorize_day_type(signal.analysis_timestamp),
                timeframe_bucket=timeframe_bucket,

                # Signal context
                direction=direction,
                setup_type=self._categorize_setup(
                    signal,
                    market_conditions,
                    technical_indicators,
                    market_regime,
                    momentum_state,
                    volume_profile,
                    direction
                ),
                leverage_category=self._categorize_leverage(signal.leverage),
                confidence_level=self._categorize_confidence(signal.confidence)
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            # Return default features
            return PatternFeatures(
                market_regime='unknown',
                volatility_level='medium',
                volume_profile='normal',
                rsi_range='neutral',
                trend_strength='moderate',
                momentum_state='neutral',
                time_category='us',
                day_type='weekday',
                timeframe_bucket=self._categorize_timeframe(
                    getattr(signal, 'timeframe', None),
                    getattr(signal, 'time_horizon', None)
                ),
                direction=str(signal.direction or '').upper(),
                setup_type='unknown',
                leverage_category='medium',
                confidence_level='medium'
            )

    def _categorize_market_regime(self, market_conditions: Dict[str, Any]) -> str:
        """Categorize market regime from conditions."""
        # First try the 'regime' field if available
        regime = market_conditions.get('regime')
        if regime in ['trending_up', 'trending_down', 'ranging', 'volatile', 'breakout']:
            return regime

        # If no regime field, derive from trend_direction and volatility
        trend_direction = market_conditions.get('trend_direction', '').lower()
        volatility_24h = market_conditions.get('volatility_24h', 0)
        price_change_24h = abs(market_conditions.get('price_change_24h', 0))

        # High volatility conditions
        if volatility_24h > 15 or price_change_24h > 10:
            return 'volatile'

        # Trend classification
        if trend_direction == 'bullish':
            return 'trending_up'
        elif trend_direction == 'bearish':
            return 'trending_down'
        elif trend_direction == 'sideways':
            return 'ranging'

        # Fallback based on price movement
        if price_change_24h > 5:
            return 'volatile'
        elif price_change_24h > 2:
            return 'trending_up' if market_conditions.get('price_change_24h', 0) > 0 else 'trending_down'
        else:
            return 'ranging'

    def _categorize_volatility(self, market_conditions: Dict[str, Any]) -> str:
        """Categorize volatility level."""
        # First try percentile if available
        volatility = market_conditions.get('volatility_percentile')
        if volatility is not None:
            if volatility < 25:
                return 'low'
            elif volatility < 50:
                return 'medium'
            elif volatility < 75:
                return 'high'
            else:
                return 'extreme'

        # Use volatility_24h field as available in actual data
        volatility_24h = market_conditions.get('volatility_24h', 0)
        if volatility_24h < 5:
            return 'low'
        elif volatility_24h < 10:
            return 'medium'
        elif volatility_24h < 20:
            return 'high'
        else:
            return 'extreme'

    def _categorize_volume(self, market_conditions: Dict[str, Any]) -> str:
        """Categorize volume profile."""
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:
            return 'low'
        elif volume_ratio < 1.5:
            return 'normal'
        elif volume_ratio < 3.0:
            return 'high'
        else:
            return 'spike'

    def _categorize_rsi(self, rsi: float) -> str:
        """Categorize RSI into ranges."""
        if rsi is None:
            return None
        if rsi < 30:
            return 'oversold'
        elif rsi < 45:
            return 'low'
        elif rsi < 55:
            return 'neutral'
        elif rsi < 70:
            return 'high'
        else:
            return 'overbought'

    def _categorize_trend_strength(self, adx: float, strength_score: float = None) -> str:
        """Categorize trend strength from ADX or derive from available indicators."""
        if adx is not None:
            if adx < 20:
                return 'weak'
            elif adx < 40:
                return 'moderate'
            else:
                return 'strong'

        # Use strength_score as fallback (0-1 scale)
        if strength_score is not None:
            if strength_score < 0.4:
                return 'weak'
            elif strength_score < 0.7:
                return 'moderate'
            else:
                return 'strong'

        # Safe fallback when neither available
        return 'moderate'

    def _categorize_momentum(self, technical_indicators: Dict[str, Any]) -> str:
        """Categorize momentum state."""
        # Try MACD histogram first (available in actual data)
        macd_histogram = technical_indicators.get('macd_histogram', 0)

        if macd_histogram > 0.001:  # Positive and significant
            return 'bullish'
        elif macd_histogram < -0.001:  # Negative and significant
            return 'bearish'

        # Fallback to momentum_score if available
        momentum_score = technical_indicators.get('momentum_score', 0.5)
        if momentum_score > 0.6:
            return 'bullish'
        elif momentum_score < 0.4:
            return 'bearish'
        else:
            return 'neutral'

    def _categorize_time_of_day(self, timestamp: datetime) -> str:
        """Categorize time of day for trading session."""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'europe'
        elif 16 <= hour < 24:
            return 'us'
        else:
            return 'overlap'

    def _categorize_day_type(self, timestamp: datetime) -> str:
        """Categorize day type."""
        return 'weekend' if timestamp.weekday() >= 5 else 'weekday'

    def _categorize_timeframe(self, timeframe: Optional[str], time_horizon: Optional[str] = None) -> str:
        """Bucket chart timeframes so exact learning is specific but not over-fragmented."""
        raw = str(timeframe or '').lower().strip().replace(' ', '')
        horizon = str(time_horizon or '').lower()

        if raw in {'1m', '3m', '5m', '15m'}:
            return 'scalp'
        if raw in {'30m', '45m', '1h'}:
            return 'intraday'
        if raw in {'2h', '3h', '4h', '6h', '8h', '12h'}:
            return 'swing_short'
        if raw in {'1d', '2d', '3d'}:
            return 'swing'
        if raw in {'1w', '2w', '1mo'}:
            return 'position'

        if any(token in horizon for token in ('minutes', 'scalp', '15m', '30m')):
            return 'scalp'
        if any(token in horizon for token in ('4h', '12h', '24h', 'intraday')):
            return 'intraday'
        if any(token in horizon for token in ('1-3', '3-7', 'days', 'swing')):
            return 'swing_short'
        if any(token in horizon for token in ('week', 'position')):
            return 'position'

        return 'intraday'

    def _categorize_setup(
        self,
        signal: PlatformSignal,
        market_conditions: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        market_regime: str,
        momentum_state: str,
        volume_profile: str,
        direction: str,
    ) -> str:
        """Classify the setup used for hierarchical pattern matching."""
        explicit_setup = (
            getattr(signal, 'pattern_classification', None)
            or market_conditions.get('setup')
            or market_conditions.get('setup_type')
            or market_conditions.get('pattern')
            or market_conditions.get('pattern_classification')
            or technical_indicators.get('setup')
            or technical_indicators.get('pattern')
        )
        if explicit_setup:
            return self._normalize_label(explicit_setup)

        direction = (direction or '').upper()
        if direction == 'LONG':
            if market_regime in {'breakout', 'trending_up'} and momentum_state == 'bullish':
                return 'bullish_breakout' if volume_profile in {'high', 'spike'} else 'bullish_trend_follow'
            if market_regime == 'ranging':
                return 'range_long'
            if momentum_state == 'bullish':
                return 'bullish_momentum'
        elif direction == 'SHORT':
            if market_regime in {'breakout', 'trending_down'} and momentum_state == 'bearish':
                return 'bearish_breakdown' if volume_profile in {'high', 'spike'} else 'bearish_trend_follow'
            if market_regime == 'ranging':
                return 'range_short'
            if momentum_state == 'bearish':
                return 'bearish_momentum'

        if market_regime == 'volatile':
            return 'volatility_continuation'
        return 'mixed_setup'

    def _normalize_label(self, value: Any) -> str:
        """Normalize human/model labels into stable feature buckets."""
        text = str(value or 'unknown').strip().lower()
        normalized = ''.join(ch if ch.isalnum() else '_' for ch in text)
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        return normalized.strip('_') or 'unknown'

    def _categorize_leverage(self, leverage: int) -> str:
        """Categorize leverage level."""
        if leverage <= 2:
            return 'low'
        elif leverage <= 5:
            return 'medium'
        else:
            return 'high'

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level."""
        if confidence < 0.6:
            return 'low'
        elif confidence < 0.8:
            return 'medium'
        else:
            return 'high'

    def _calculate_pattern_similarity(self, features: PatternFeatures, pattern: LearningPattern) -> float:
        """Calculate similarity between current features and stored pattern."""
        try:
            return self._calculate_feature_similarity(features, pattern.conditions)

        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0

    def _similarity_weights(self) -> Dict[str, float]:
        """Feature weights for pattern similarity across old and new memories."""
        return {
            'market_regime': 0.20,
            'setup_type': 0.16,
            'direction': 0.16,
            'timeframe_bucket': 0.10,
            'volatility_level': 0.12,
            'trend_strength': 0.09,
            'momentum_state': 0.08,
            'rsi_range': 0.04,
            'volume_profile': 0.03,
            'leverage_category': 0.01,
            'confidence_level': 0.01,
            'time_category': 0.01,
            'day_type': 0.01,
        }

    def _calculate_feature_similarity(self, features: PatternFeatures, conditions: Dict[str, Any]) -> float:
        """Calculate weighted similarity against a JSON feature dictionary."""
        if not isinstance(conditions, dict):
            return 0.0

        similarity_score = 0.0
        total_weight = 0.0
        for feature, weight in self._similarity_weights().items():
            current_value = getattr(features, feature, None)
            pattern_value = conditions.get(feature)

            if current_value is not None and pattern_value is not None:
                if str(current_value).lower() == str(pattern_value).lower():
                    similarity_score += weight
                total_weight += weight

        return similarity_score / total_weight if total_weight > 0 else 0.0

    def _determine_confidence_level(self, patterns: List[LearningPattern], total_weight: float) -> str:
        """Determine confidence level in learning adjustment."""
        if not patterns or total_weight < 0.3:
            return 'low'

        avg_sample_size = sum(p.total_trades for p in patterns) / len(patterns)
        if avg_sample_size >= 10 and total_weight >= 0.7:
            return 'high'
        elif avg_sample_size >= 5 and total_weight >= 0.5:
            return 'medium'
        else:
            return 'low'

    def _determine_scoped_confidence_level(
        self,
        evidence: List[ScopedLearningEvidence],
        total_weight: float
    ) -> str:
        """Determine confidence in the partial-pooling adjustment."""
        if not evidence or total_weight < 0.15:
            return 'low'

        if any(e.scope == 'token_timeframe' and e.confidence_level == 'high' for e in evidence):
            return 'high'
        if any(e.confidence_level in {'high', 'medium'} for e in evidence) and total_weight >= 0.30:
            return 'medium'
        return 'low'

    def _generate_scoped_adjustment_reasoning(
        self,
        signal: PlatformSignal,
        features: PatternFeatures,
        evidence: List[ScopedLearningEvidence],
        confidence_adj: float
    ) -> str:
        """Generate honest reasoning that names token/cluster/global scope."""
        if not evidence:
            return "No historical token, cluster, or global pattern evidence found"

        summaries = []
        for item in evidence[:4]:
            summaries.append(
                f"{item.label}: WR={item.success_rate:.1%}, "
                f"net_EV={item.expected_value:+.2f}%, n={item.total_trades}"
            )

        scopes = {item.scope for item in evidence}
        missing_context = []
        if 'token_timeframe' not in scopes:
            missing_context.append("no exact token+timeframe sample")
        if 'token_cluster' not in scopes:
            missing_context.append("no usable cluster sample")

        if confidence_adj > 0.03:
            action = "cautious confidence boost"
        elif confidence_adj < -0.03:
            action = "reducing confidence/risk"
        else:
            action = "no material confidence shift"

        setup_context = (
            f"{self._normalize_token_symbol(signal.token_symbol)} "
            f"{features.timeframe_bucket} {features.setup_type}/{features.market_regime}"
        )
        suffix = f" ({'; '.join(missing_context)})" if missing_context else ""
        return (
            f"Hierarchical learning for {setup_context}: "
            f"{'; '.join(summaries)} — EV-weighted partial pooling, {action}{suffix}."
        )

    def _generate_adjustment_reasoning(self, patterns: List[LearningPattern], confidence_adj: float) -> str:
        """Generate human-readable reasoning for adjustments."""
        if not patterns:
            return "No historical patterns found for similar market conditions"

        avg_success_rate = sum(p.success_rate for p in patterns) / len(patterns)
        total_trades = sum(p.total_trades for p in patterns)

        if confidence_adj > 0.05:
            return f"Similar conditions succeeded {avg_success_rate:.1%} of the time in {total_trades} past trades - boosting confidence"
        elif confidence_adj < -0.05:
            return f"Similar conditions only succeeded {avg_success_rate:.1%} of the time in {total_trades} past trades - reducing confidence"
        else:
            return f"Mixed results in similar conditions ({avg_success_rate:.1%} success rate, {total_trades} trades) - no adjustment"

    def _generate_lesson(self, signal: PlatformSignal, outcome: str, pnl: float, exit_reason: Optional[str]) -> str:
        """Generate lesson learned from trade outcome."""
        try:
            market_regime = signal.market_conditions.get('regime', 'unknown')
            rsi = signal.technical_indicators.get('rsi', 50)

            if outcome == 'win':
                if exit_reason in ['TARGET_1', 'TARGET_2']:
                    return f"Successful {signal.direction} in {market_regime} market with RSI {rsi:.1f} - targets were well-calculated"
                else:
                    return f"Profitable {signal.direction} exit in {market_regime} conditions"
            elif outcome == 'loss':
                if exit_reason == 'STOP_LOSS':
                    return f"Stop loss hit on {signal.direction} in {market_regime} market - consider tighter risk management"
                else:
                    return f"Loss on {signal.direction} trade in {market_regime} conditions - review entry criteria"
            else:
                return f"Neutral outcome on {signal.direction} in {market_regime} market"

        except Exception as e:
            return f"Trade completed with {outcome} outcome"

    def _clear_pattern_cache(self):
        """Clear the pattern cache to force refresh."""
        self.pattern_cache.clear()
        self.cache_expiry = datetime.now()

    def _parse_datetime_safe(self, datetime_str: str) -> datetime:
        """Safely parse datetime string from database."""
        try:
            # Handle various timestamp formats from database
            cleaned = datetime_str.strip()

            # Handle microseconds precision issue - truncate to 6 digits
            if '.' in cleaned:
                parts = cleaned.split('.')
                if len(parts) == 2:
                    base_time, microseconds_part = parts
                    # Extract just the digits from microseconds part (remove timezone, etc.)
                    microseconds = ''.join(c for c in microseconds_part if c.isdigit())[:6].ljust(6, '0')
                    cleaned = f"{base_time}.{microseconds}"

            return datetime.fromisoformat(cleaned)
        except Exception as e:
            logger.debug(f"Failed to parse datetime '{datetime_str}': {e}. Using current time.")
            return datetime.now()

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics for monitoring."""
        try:
            # Get learning memory stats
            memory_result = self.db.table('agent_learning_memory').select(
                'outcome', 'pnl_percentage', 'created_at'
            ).eq('agent_type', self.agent_type).gte(
                'created_at', (datetime.now() - timedelta(days=30)).isoformat()
            ).execute()

            memory_data = memory_result.data or []

            # Get pattern stats
            pattern_result = self.db.table('agent_learning_patterns').select(
                'success_rate', 'total_trades', 'confidence_boost'
            ).eq('agent_type', self.agent_type).execute()

            pattern_data = pattern_result.data or []

            # Calculate statistics
            total_trades = len(memory_data)
            wins = len([d for d in memory_data if d['outcome'] == 'win'])
            losses = len([d for d in memory_data if d['outcome'] == 'loss'])

            if total_trades > 0:
                win_rate = wins / total_trades
                avg_pnl = sum(d['pnl_percentage'] for d in memory_data) / total_trades
            else:
                win_rate = 0
                avg_pnl = 0

            return {
                'agent_type': self.agent_type,
                'total_trades_recorded': total_trades,
                'win_rate': win_rate,
                'average_pnl': avg_pnl,
                'total_patterns': len(pattern_data),
                'active_patterns': len([p for p in pattern_data if p['total_trades'] >= 3]),
                'patterns_with_boost': len([p for p in pattern_data if p['confidence_boost'] > 0.05]),
                'patterns_with_reduction': len([p for p in pattern_data if p['confidence_boost'] < -0.05]),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {
                'agent_type': self.agent_type,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }


# Global instances
_learning_services: Dict[str, AgentLearningService] = {}


def get_agent_learning_service(agent_type: str) -> AgentLearningService:
    """Get or create learning service for specific agent type."""
    global _learning_services

    if agent_type not in _learning_services:
        _learning_services[agent_type] = AgentLearningService(agent_type)

    return _learning_services[agent_type]
