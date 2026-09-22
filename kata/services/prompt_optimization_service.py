"""
Prompt Optimization Service - AI-Driven Prompt Enhancement

This service analyzes signal outcomes to optimize trading prompts:
- Tracks which prompt components lead to better signals
- Identifies failure patterns in AI reasoning
- Automatically suggests prompt improvements
- A/B tests prompt variations for performance

Part of Level 3 Meta-Learning architecture.
"""

import asyncio
import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


@dataclass
class PromptComponent:
    """Represents a component of the trading prompt."""
    component_type: str  # 'instruction', 'context', 'constraint', 'example'
    content: str
    importance_weight: float
    performance_impact: float
    usage_frequency: int
    success_correlation: float


@dataclass
class PromptPerformance:
    """Performance metrics for a prompt variation."""
    prompt_version: str
    prompt_hash: str
    total_signals: int
    successful_signals: int
    avg_confidence: float
    avg_pnl: float
    success_rate: float
    avg_execution_time: float
    reasoning_quality_score: float
    consistency_score: float
    created_at: datetime
    last_updated: datetime


@dataclass
class PromptOptimization:
    """Optimization recommendation for prompts."""
    optimization_type: str  # 'add', 'remove', 'modify', 'reorder'
    component: str
    old_content: Optional[str]
    new_content: str
    expected_improvement: float
    confidence: float
    reasoning: str
    supporting_evidence: List[str]


class PromptOptimizationService:
    """Service for optimizing trading prompts based on performance."""

    def __init__(self, agent_type: str):
        """Initialize prompt optimization service."""
        self.agent_type = agent_type
        self.db = get_service_client()

        # Prompt analysis state
        self.current_prompt_components = {}
        self.prompt_performance_history = []
        self.successful_reasoning_patterns = []
        self.failed_reasoning_patterns = []

        # Optimization configuration
        self.config = {
            'min_signals_for_analysis': 20,
            'performance_window_days': 7,
            'min_improvement_threshold': 0.05,  # 5% improvement minimum
            'max_prompt_length': 2000,          # Character limit
            'ab_test_duration_days': 3,
            'confidence_threshold': 0.7,
            'generation_context_cache_seconds': 1800,
            'max_generation_directives': 4,
            'auto_optimization_cooldown_hours': 12,
        }

        self._generation_context_cache: Optional[Dict[str, Any]] = None
        self._generation_context_cached_at: Optional[datetime] = None

        logger.info(f"🎯 Prompt Optimization Service initialized for {agent_type}")

    def _signal_matches_agent(self, signal: Dict[str, Any]) -> bool:
        """Check if a platform signal belongs to this optimization service's agent."""
        generated_by = signal.get('generated_by_agent')
        if self.agent_type == 'yuki':
            return generated_by in {None, 'yuki', 'platform'}
        return generated_by == self.agent_type

    def _normalize_list(self, value: Any) -> List[str]:
        """Normalize list-like values from Postgres JSON/array fields."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith('[') and stripped.endswith(']'):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            return [stripped]
        return [str(value).strip()]

    def _build_guidance_block(
        self,
        improvement_suggestions: List[str],
        success_patterns: Optional[List[str]] = None,
        failure_patterns: Optional[List[str]] = None,
    ) -> str:
        """Build compact prompt directives block to append to generation prompt."""
        directives = []
        for suggestion in improvement_suggestions:
            cleaned = suggestion.strip().rstrip('.')
            if cleaned:
                directives.append(f"- {cleaned}.")
        directives = directives[:self.config['max_generation_directives']]

        blocks: List[str] = []
        if directives:
            blocks.append("**LEARNING-OPTIMIZED DIRECTIVES (MANDATORY):**")
            blocks.extend(directives)

        success_patterns = (success_patterns or [])[:2]
        failure_patterns = (failure_patterns or [])[:2]

        if success_patterns:
            blocks.append("**SUCCESS PATTERNS TO PRESERVE:**")
            blocks.extend([f"- {pattern}" for pattern in success_patterns if pattern])
        if failure_patterns:
            blocks.append("**FAILURE PATTERNS TO AVOID:**")
            blocks.extend([f"- {pattern}" for pattern in failure_patterns if pattern])

        return "\n".join(blocks).strip()

    async def _get_latest_prompt_record(self) -> Optional[Dict[str, Any]]:
        """Fetch latest signal-generation optimization record for this agent."""
        try:
            response = (
                self.db.from_('prompt_optimization_history')
                .select('*')
                .eq('agent_type', self.agent_type)
                .eq('prompt_type', 'signal_generation')
                .order('optimization_version', desc=True)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch latest prompt optimization history: {e}")
        return None

    async def _compute_recent_signal_metrics(self) -> Dict[str, Any]:
        """Compute lightweight outcome metrics used for prompt directives."""
        metrics: Dict[str, Any] = {
            'sample_size': 0,
            'expiry_rate': 0.0,
            'resolved_win_rate': 0.0,
            'avg_confidence': 0.0,
            'confidence_std': 0.0,
            'median_t1_move_pct': 0.0,
            'high_t1_move_ratio': 0.0,
            'validity_12h_ratio': 0.0,
        }

        try:
            rows_result = (
                self.db.from_('platform_signals')
                .select(
                    'signal_id,status,direction,confidence,entry_price,target_1,'
                    'validity_window_hours,time_horizon,generated_by_agent,created_at'
                )
                .order('created_at', desc=True)
                .limit(300)
                .execute()
            )
            rows = [row for row in (rows_result.data or []) if self._signal_matches_agent(row)]
            if not rows:
                return metrics

            metrics['sample_size'] = len(rows)
            expired = [r for r in rows if str(r.get('status') or '').lower() == 'expired']
            wins = [r for r in rows if str(r.get('status') or '').lower() in {'hit_target_1', 'hit_target_2'}]
            stops = [r for r in rows if str(r.get('status') or '').lower() == 'hit_stop_loss']
            resolved = len(wins) + len(stops)

            metrics['expiry_rate'] = len(expired) / len(rows)
            metrics['resolved_win_rate'] = (len(wins) / resolved) if resolved > 0 else 0.0

            confidence_values = []
            t1_moves = []
            validity_12h = 0
            for row in rows:
                try:
                    confidence_values.append(float(row.get('confidence') or 0))
                except (TypeError, ValueError):
                    pass

                validity_hours = row.get('validity_window_hours')
                try:
                    if int(validity_hours) <= 12:
                        validity_12h += 1
                except (TypeError, ValueError):
                    pass

                try:
                    entry = float(row.get('entry_price') or 0)
                    target_1 = float(row.get('target_1') or 0)
                    if entry <= 0:
                        continue
                    is_long = str(row.get('direction') or '').upper() == 'LONG'
                    move_pct = ((target_1 - entry) / entry * 100.0) if is_long else ((entry - target_1) / entry * 100.0)
                    if move_pct > 0:
                        t1_moves.append(move_pct)
                except (TypeError, ValueError):
                    continue

            if confidence_values:
                metrics['avg_confidence'] = statistics.mean(confidence_values)
                metrics['confidence_std'] = statistics.pstdev(confidence_values) if len(confidence_values) > 1 else 0.0

            if t1_moves:
                metrics['median_t1_move_pct'] = statistics.median(t1_moves)
                metrics['high_t1_move_ratio'] = len([move for move in t1_moves if move > 8.0]) / len(t1_moves)

            metrics['validity_12h_ratio'] = validity_12h / len(rows)
            return metrics
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute recent prompt metrics: {e}")
            return metrics

    def _derive_generation_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        """Create actionable prompt suggestions from outcome metrics."""
        suggestions: List[str] = []

        sample_size = int(metrics.get('sample_size') or 0)
        if sample_size < self.config['min_signals_for_analysis']:
            return [
                "Keep risk framing explicit and prioritize confluence over trade frequency",
                "Only issue directional calls when setup quality is clear; otherwise HOLD",
            ]

        expiry_rate = float(metrics.get('expiry_rate') or 0.0)
        resolved_win_rate = float(metrics.get('resolved_win_rate') or 0.0)
        median_t1_move_pct = float(metrics.get('median_t1_move_pct') or 0.0)
        high_t1_move_ratio = float(metrics.get('high_t1_move_ratio') or 0.0)
        validity_12h_ratio = float(metrics.get('validity_12h_ratio') or 0.0)
        confidence_std = float(metrics.get('confidence_std') or 0.0)

        if expiry_rate > 0.55:
            suggestions.append(
                "Constrain target_1 distance for short horizons; keep target_1 typically within 3-7% unless conviction and volatility are exceptional"
            )
        if high_t1_move_ratio > 0.35 and validity_12h_ratio > 0.6:
            suggestions.append(
                "When target_1 requires >8% move, shift time_horizon to 4h-24h or 1-3 days instead of forcing 4h-12h"
            )
        if resolved_win_rate > 0.45 and expiry_rate > 0.5:
            suggestions.append(
                "Prioritize higher probability first targets and allow secondary targets only after locking realistic target_1 levels"
            )
        if median_t1_move_pct > 7.0:
            suggestions.append(
                "Reduce aggressive target placement; tie target distance to ATR and regime, not idealized trend continuation"
            )
        if confidence_std < 0.04:
            suggestions.append(
                "Widen confidence dispersion by penalizing mixed setups and rewarding multi-factor confluence only"
            )

        if not suggestions:
            suggestions.append(
                "Keep current prompt structure; continue emphasizing risk-adjusted entries and clear invalidation logic"
            )

        return suggestions[:self.config['max_generation_directives']]

    async def _store_auto_optimization(
        self,
        base_prompt: str,
        suggestions: List[str],
        metrics: Dict[str, Any],
        latest_record: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Persist auto-generated prompt optimization snapshots with cooldown."""
        try:
            cooldown_hours = self.config['auto_optimization_cooldown_hours']
            if latest_record and latest_record.get('optimization_timestamp'):
                try:
                    last_ts = datetime.fromisoformat(str(latest_record['optimization_timestamp']).replace('Z', '+00:00')).replace(tzinfo=None)
                    if (datetime.now() - last_ts).total_seconds() < cooldown_hours * 3600:
                        return int(latest_record.get('optimization_version', 1))
                except Exception:
                    pass

            next_version = int(latest_record.get('optimization_version', 0)) + 1 if latest_record else 1
            original_prompt = base_prompt[:4000]
            guidance_text = self._build_guidance_block(suggestions)
            optimized_prompt = f"{original_prompt}\n\n{guidance_text}"[:6000]

            payload = {
                'agent_type': self.agent_type,
                'prompt_type': 'signal_generation',
                'optimization_version': next_version,
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'optimization_rationale': (
                    f"Auto optimization from recent outcomes: sample={metrics.get('sample_size', 0)}, "
                    f"expiry_rate={metrics.get('expiry_rate', 0):.2%}, "
                    f"resolved_win_rate={metrics.get('resolved_win_rate', 0):.2%}"
                ),
                'success_patterns': [],
                'failure_patterns': [],
                'improvement_suggestions': suggestions,
                'baseline_performance': metrics,
                'optimized_performance': {},
                'performance_delta': {},
                'optimization_timestamp': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
            }

            self.db.from_('prompt_optimization_history').insert(payload).execute()
            return next_version
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist auto prompt optimization: {e}")
            return None

    async def get_signal_generation_context(self, base_prompt: str) -> Dict[str, Any]:
        """
        Get optimization directives to inject into live signal generation prompt.
        """
        now = datetime.now()
        if (
            self._generation_context_cache
            and self._generation_context_cached_at
            and (now - self._generation_context_cached_at).total_seconds() < self.config['generation_context_cache_seconds']
        ):
            return self._generation_context_cache

        latest_record = await self._get_latest_prompt_record()

        improvement_suggestions: List[str] = []
        success_patterns: List[str] = []
        failure_patterns: List[str] = []
        prompt_version = f"{self.agent_type}_v0"
        source = 'fallback'

        if latest_record:
            improvement_suggestions = self._normalize_list(latest_record.get('improvement_suggestions'))
            success_patterns = self._normalize_list(latest_record.get('success_patterns'))
            failure_patterns = self._normalize_list(latest_record.get('failure_patterns'))
            prompt_version = f"{self.agent_type}_v{int(latest_record.get('optimization_version', 0))}"
            source = 'history'

        if not improvement_suggestions:
            metrics = await self._compute_recent_signal_metrics()
            improvement_suggestions = self._derive_generation_suggestions(metrics)
            stored_version = await self._store_auto_optimization(base_prompt, improvement_suggestions, metrics, latest_record)
            if stored_version:
                prompt_version = f"{self.agent_type}_v{stored_version}"
            source = 'auto'

        guidance_text = self._build_guidance_block(
            improvement_suggestions=improvement_suggestions,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
        )

        context = {
            'prompt_version': prompt_version,
            'guidance_text': guidance_text,
            'improvement_suggestions': improvement_suggestions,
            'source': source,
            'generated_at': now.isoformat(),
        }
        self._generation_context_cache = context
        self._generation_context_cached_at = now
        return context

    async def analyze_prompt_performance(self) -> PromptPerformance:
        """Analyze current prompt performance."""
        try:
            # Get recent signals with AI reasoning
            recent_signals = await self._get_recent_signals_with_reasoning()

            if len(recent_signals) < self.config['min_signals_for_analysis']:
                logger.warning(f"⚠️ Insufficient signals for prompt analysis: {len(recent_signals)}")
                return None

            # Extract current prompt version
            current_prompt = await self._get_current_prompt()
            prompt_hash = self._hash_prompt(current_prompt)

            # Calculate performance metrics
            total_signals = len(recent_signals)
            successful_signals = len([s for s in recent_signals if self._is_signal_successful(s)])

            success_rate = successful_signals / total_signals if total_signals > 0 else 0
            avg_confidence = statistics.mean([s.get('confidence', 0) for s in recent_signals])

            # Calculate average PnL (only for completed signals)
            completed_signals = [s for s in recent_signals if s.get('pnl_percentage') is not None]
            avg_pnl = statistics.mean([s['pnl_percentage'] for s in completed_signals]) if completed_signals else 0

            # Reasoning quality analysis
            reasoning_quality = await self._analyze_reasoning_quality(recent_signals)

            # Consistency analysis
            consistency_score = self._calculate_consistency_score(recent_signals)

            performance = PromptPerformance(
                prompt_version=f"{self.agent_type}_v{prompt_hash[:8]}",
                prompt_hash=prompt_hash,
                total_signals=total_signals,
                successful_signals=successful_signals,
                avg_confidence=avg_confidence,
                avg_pnl=avg_pnl,
                success_rate=success_rate,
                avg_execution_time=30.0,  # Placeholder
                reasoning_quality_score=reasoning_quality,
                consistency_score=consistency_score,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )

            # Store performance record
            await self._store_prompt_performance(performance)

            return performance

        except Exception as e:
            logger.error(f"❌ Prompt performance analysis failed: {e}")
            return None

    async def generate_optimization_suggestions(self) -> List[PromptOptimization]:
        """Generate prompt optimization suggestions based on performance analysis."""
        try:
            logger.info("🔍 Analyzing prompt components for optimization opportunities...")

            # Analyze successful vs failed reasoning patterns
            await self._analyze_reasoning_patterns()

            # Extract current prompt components
            current_components = await self._extract_prompt_components()

            # Generate optimization suggestions
            optimizations = []

            # 1. Analyze successful reasoning patterns
            successful_patterns = await self._identify_successful_patterns()
            for pattern in successful_patterns:
                if pattern['frequency'] >= 5 and pattern['success_rate'] > 0.8:
                    optimization = self._create_enhancement_optimization(pattern)
                    if optimization:
                        optimizations.append(optimization)

            # 2. Analyze failure patterns
            failure_patterns = await self._identify_failure_patterns()
            for pattern in failure_patterns:
                if pattern['frequency'] >= 3 and pattern['failure_rate'] > 0.7:
                    optimization = self._create_correction_optimization(pattern)
                    if optimization:
                        optimizations.append(optimization)

            # 3. Analyze constraint effectiveness
            constraint_analysis = await self._analyze_constraint_effectiveness()
            constraint_optimizations = self._generate_constraint_optimizations(constraint_analysis)
            optimizations.extend(constraint_optimizations)

            # 4. Context optimization
            context_optimization = await self._optimize_context_components()
            if context_optimization:
                optimizations.append(context_optimization)

            # Sort by expected improvement
            optimizations.sort(key=lambda x: x.expected_improvement, reverse=True)

            logger.info(f"✅ Generated {len(optimizations)} optimization suggestions")
            return optimizations[:5]  # Return top 5

        except Exception as e:
            logger.error(f"❌ Optimization generation failed: {e}")
            return []

    async def _analyze_reasoning_patterns(self):
        """Analyze patterns in successful vs failed reasoning."""
        try:
            recent_signals = await self._get_recent_signals_with_reasoning()

            successful_reasoning = []
            failed_reasoning = []

            for signal in recent_signals:
                reasoning = signal.get('ai_reasoning', '')
                is_success = self._is_signal_successful(signal)

                if reasoning:
                    if is_success:
                        successful_reasoning.append(reasoning)
                    else:
                        failed_reasoning.append(reasoning)

            # Extract patterns from successful reasoning
            self.successful_reasoning_patterns = self._extract_reasoning_patterns(successful_reasoning)

            # Extract patterns from failed reasoning
            self.failed_reasoning_patterns = self._extract_reasoning_patterns(failed_reasoning)

            logger.info(f"📊 Analyzed reasoning: {len(successful_reasoning)} successful, {len(failed_reasoning)} failed")

        except Exception as e:
            logger.error(f"❌ Reasoning pattern analysis failed: {e}")

    def _extract_reasoning_patterns(self, reasoning_texts: List[str]) -> List[Dict]:
        """Extract common patterns from reasoning texts."""
        patterns = []

        try:
            # Common phrase analysis
            phrase_counter = Counter()
            decision_patterns = Counter()
            confidence_expressions = Counter()

            for reasoning in reasoning_texts:
                # Extract key phrases
                phrases = self._extract_key_phrases(reasoning)
                phrase_counter.update(phrases)

                # Extract decision patterns
                decisions = self._extract_decision_patterns(reasoning)
                decision_patterns.update(decisions)

                # Extract confidence expressions
                confidence_exprs = self._extract_confidence_expressions(reasoning)
                confidence_expressions.update(confidence_exprs)

            # Build pattern objects
            for phrase, count in phrase_counter.most_common(20):
                if count >= 3:  # Minimum frequency
                    patterns.append({
                        'type': 'phrase',
                        'content': phrase,
                        'frequency': count,
                        'category': 'common_phrase'
                    })

            for decision, count in decision_patterns.most_common(10):
                if count >= 2:
                    patterns.append({
                        'type': 'decision_pattern',
                        'content': decision,
                        'frequency': count,
                        'category': 'decision_logic'
                    })

        except Exception as e:
            logger.error(f"❌ Pattern extraction failed: {e}")

        return patterns

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from reasoning text."""
        phrases = []

        try:
            # Look for specific trading-related phrases
            trading_phrases = [
                r'strong momentum',
                r'volume confirmation',
                r'support level',
                r'resistance break',
                r'trend reversal',
                r'consolidation pattern',
                r'oversold condition',
                r'overbought territory',
                r'bullish sentiment',
                r'bearish pressure',
                r'risk management',
                r'entry opportunity'
            ]

            for phrase_pattern in trading_phrases:
                matches = re.findall(phrase_pattern, text.lower())
                phrases.extend(matches)

        except Exception as e:
            logger.error(f"❌ Key phrase extraction failed: {e}")

        return phrases

    def _extract_decision_patterns(self, text: str) -> List[str]:
        """Extract decision-making patterns from reasoning."""
        patterns = []

        try:
            # Look for decision structures
            decision_patterns = [
                r'based on.*I (recommend|suggest)',
                r'given.*the.*I believe',
                r'considering.*factors.*decision is',
                r'analysis shows.*therefore',
                r'technical indicators.*suggest'
            ]

            for pattern in decision_patterns:
                matches = re.findall(pattern, text.lower())
                patterns.extend(matches)

        except Exception as e:
            logger.error(f"❌ Decision pattern extraction failed: {e}")

        return patterns

    def _extract_confidence_expressions(self, text: str) -> List[str]:
        """Extract confidence expression patterns."""
        expressions = []

        try:
            # Look for confidence expressions
            confidence_patterns = [
                r'high confidence',
                r'moderate confidence',
                r'low confidence',
                r'very confident',
                r'somewhat confident',
                r'uncertain about',
                r'clear signal',
                r'strong indication'
            ]

            for pattern in confidence_patterns:
                matches = re.findall(pattern, text.lower())
                expressions.extend(matches)

        except Exception as e:
            logger.error(f"❌ Confidence expression extraction failed: {e}")

        return expressions

    async def _identify_successful_patterns(self) -> List[Dict]:
        """Identify patterns that correlate with successful signals."""
        successful_patterns = []

        try:
            for pattern in self.successful_reasoning_patterns:
                # Calculate success rate for this pattern
                success_rate = await self._calculate_pattern_success_rate(pattern['content'])

                if success_rate > 0.7:  # 70% success rate threshold
                    pattern['success_rate'] = success_rate
                    successful_patterns.append(pattern)

        except Exception as e:
            logger.error(f"❌ Successful pattern identification failed: {e}")

        return successful_patterns

    async def _identify_failure_patterns(self) -> List[Dict]:
        """Identify patterns that correlate with failed signals."""
        failure_patterns = []

        try:
            for pattern in self.failed_reasoning_patterns:
                # Calculate failure rate for this pattern
                failure_rate = await self._calculate_pattern_failure_rate(pattern['content'])

                if failure_rate > 0.6:  # 60% failure rate threshold
                    pattern['failure_rate'] = failure_rate
                    failure_patterns.append(pattern)

        except Exception as e:
            logger.error(f"❌ Failure pattern identification failed: {e}")

        return failure_patterns

    def _create_enhancement_optimization(self, successful_pattern: Dict) -> Optional[PromptOptimization]:
        """Create optimization to enhance successful patterns."""
        try:
            if successful_pattern['type'] == 'phrase':
                return PromptOptimization(
                    optimization_type='add',
                    component='instruction',
                    old_content=None,
                    new_content=f"Emphasize {successful_pattern['content']} in analysis",
                    expected_improvement=0.05 * successful_pattern['success_rate'],
                    confidence=0.7,
                    reasoning=f"Pattern '{successful_pattern['content']}' shows {successful_pattern['success_rate']:.1%} success rate",
                    supporting_evidence=[f"Found in {successful_pattern['frequency']} successful signals"]
                )

            return None

        except Exception as e:
            logger.error(f"❌ Enhancement optimization creation failed: {e}")
            return None

    def _create_correction_optimization(self, failure_pattern: Dict) -> Optional[PromptOptimization]:
        """Create optimization to correct failure patterns."""
        try:
            if failure_pattern['type'] == 'phrase':
                return PromptOptimization(
                    optimization_type='add',
                    component='constraint',
                    old_content=None,
                    new_content=f"Avoid overreliance on {failure_pattern['content']} without confirmation",
                    expected_improvement=0.03 * failure_pattern['failure_rate'],
                    confidence=0.6,
                    reasoning=f"Pattern '{failure_pattern['content']}' shows {failure_pattern['failure_rate']:.1%} failure rate",
                    supporting_evidence=[f"Found in {failure_pattern['frequency']} failed signals"]
                )

            return None

        except Exception as e:
            logger.error(f"❌ Correction optimization creation failed: {e}")
            return None

    # ===== UTILITY METHODS =====

    async def _get_recent_signals_with_reasoning(self) -> List[Dict]:
        """Get recent signals that have AI reasoning data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['performance_window_days'])

            response = self.db.from_('platform_signals').select(
                'signal_id, token_symbol, direction, confidence, ai_reasoning, status, pnl_percentage, created_at'
            ).gte(
                'created_at', cutoff_date.isoformat()
            ).not_.is_(
                'ai_reasoning', 'null'
            ).execute()

            return response.data or []

        except Exception as e:
            logger.error(f"❌ Failed to get recent signals: {e}")
            return []

    async def _get_current_prompt(self) -> str:
        """Get the current trading prompt for the agent."""
        # This would return the current prompt template
        # For now, return a placeholder
        return f"Current trading prompt template for {self.agent_type}"

    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt versioning."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()

    def _is_signal_successful(self, signal: Dict) -> bool:
        """Determine if a signal was successful."""
        pnl = signal.get('pnl_percentage')
        if pnl is not None:
            return pnl > 0

        # If no PnL data, use confidence and status
        confidence = signal.get('confidence', 0)
        status = signal.get('status', 'unknown')

        return confidence > 0.7 and status in ['hit_target_1', 'hit_target_2']

    async def _store_prompt_performance(self, performance: PromptPerformance):
        """Store prompt performance record."""
        try:
            performance_data = {
                'agent_type': self.agent_type,
                'prompt_version': performance.prompt_version,
                'prompt_hash': performance.prompt_hash,
                'total_signals': performance.total_signals,
                'successful_signals': performance.successful_signals,
                'success_rate': performance.success_rate,
                'avg_confidence': performance.avg_confidence,
                'avg_pnl': performance.avg_pnl,
                'reasoning_quality_score': performance.reasoning_quality_score,
                'consistency_score': performance.consistency_score,
                'created_at': performance.created_at.isoformat(),
                'last_updated': performance.last_updated.isoformat()
            }

            self.db.from_('prompt_performance_history').insert(performance_data).execute()

        except Exception as e:
            logger.error(f"❌ Failed to store prompt performance: {e}")

    async def _analyze_reasoning_quality(self, signals: List[Dict]) -> float:
        """Analyze the quality of AI reasoning."""
        try:
            if not signals:
                return 0.5

            quality_scores = []

            for signal in signals:
                reasoning = signal.get('ai_reasoning', '')
                if not reasoning:
                    continue

                # Score based on reasoning characteristics
                score = 0.0

                # Length (not too short, not too long)
                length_score = min(len(reasoning) / 500, 1.0) if len(reasoning) < 500 else max(1.0 - (len(reasoning) - 500) / 1000, 0.5)
                score += length_score * 0.2

                # Specific trading terms
                trading_terms = ['support', 'resistance', 'volume', 'momentum', 'trend', 'volatility']
                term_count = sum(1 for term in trading_terms if term.lower() in reasoning.lower())
                term_score = min(term_count / len(trading_terms), 1.0)
                score += term_score * 0.3

                # Decision structure
                has_decision_structure = any(phrase in reasoning.lower() for phrase in ['based on', 'therefore', 'conclude', 'recommend'])
                score += 0.3 if has_decision_structure else 0.0

                # Confidence expression
                has_confidence = any(phrase in reasoning.lower() for phrase in ['confident', 'confidence', 'certain', 'likely'])
                score += 0.2 if has_confidence else 0.0

                quality_scores.append(score)

            return statistics.mean(quality_scores) if quality_scores else 0.5

        except Exception as e:
            logger.error(f"❌ Reasoning quality analysis failed: {e}")
            return 0.5

    def _calculate_consistency_score(self, signals: List[Dict]) -> float:
        """Calculate consistency score for signals."""
        try:
            if len(signals) < 5:
                return 0.5

            # Check consistency in confidence levels
            confidences = [s.get('confidence', 0.5) for s in signals]
            confidence_std = statistics.stdev(confidences)
            confidence_consistency = max(0, 1.0 - confidence_std)

            # Check reasoning length consistency
            reasoning_lengths = [len(s.get('ai_reasoning', '')) for s in signals]
            length_std = statistics.stdev(reasoning_lengths) if len(reasoning_lengths) > 1 else 0
            length_consistency = max(0, 1.0 - length_std / 500)  # Normalize by 500 chars

            # Overall consistency
            consistency = (confidence_consistency * 0.6 + length_consistency * 0.4)
            return max(0.0, min(1.0, consistency))

        except Exception as e:
            logger.error(f"❌ Consistency calculation failed: {e}")
            return 0.5

    # Placeholder methods for additional functionality
    async def _extract_prompt_components(self) -> Dict:
        """Extract current prompt components."""
        return {}

    async def _analyze_constraint_effectiveness(self) -> Dict:
        """Analyze effectiveness of prompt constraints."""
        return {}

    def _generate_constraint_optimizations(self, analysis: Dict) -> List[PromptOptimization]:
        """Generate constraint-based optimizations."""
        return []

    async def _optimize_context_components(self) -> Optional[PromptOptimization]:
        """Optimize context components of the prompt."""
        return None

    async def _calculate_pattern_success_rate(self, pattern: str) -> float:
        """Calculate success rate for a specific pattern."""
        return 0.75  # Placeholder

    async def _calculate_pattern_failure_rate(self, pattern: str) -> float:
        """Calculate failure rate for a specific pattern."""
        return 0.25  # Placeholder


# Global service registry
_prompt_optimization_services = {}

def get_prompt_optimization_service(agent_type: str) -> PromptOptimizationService:
    """Get or create prompt optimization service for agent type."""
    if agent_type not in _prompt_optimization_services:
        _prompt_optimization_services[agent_type] = PromptOptimizationService(agent_type)
    return _prompt_optimization_services[agent_type]
