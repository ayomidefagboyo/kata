"""
Advanced Agent Learning Features

This module provides advanced learning capabilities including:
- Ensemble learning combining multiple pattern models
- Dynamic risk adjustment based on market conditions
- Cross-agent learning and pattern sharing
- Advanced statistical analysis and confidence intervals
- Market regime-specific learning models
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import asyncio

from kata.config.database import get_service_client
from kata.services.agent_learning_service import AgentLearningService, PatternFeatures, LearningAdjustment
from kata.services.platform_signal_service import PlatformSignal

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of learning models."""
    PATTERN_MATCHING = "pattern_matching"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"
    REGIME_SPECIFIC = "regime_specific"


class MarketRegimeType(Enum):
    """Market regime types for specialized learning."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class EnsembleModel:
    """Ensemble learning model combining multiple approaches."""
    pattern_weight: float = 0.4
    statistical_weight: float = 0.3
    regime_weight: float = 0.2
    momentum_weight: float = 0.1
    confidence_threshold: float = 0.6


@dataclass
class RiskAdjustment:
    """Dynamic risk adjustment recommendations."""
    position_size_multiplier: float
    leverage_multiplier: float
    stop_loss_adjustment: float  # Percentage adjustment to stop loss
    take_profit_adjustment: float  # Percentage adjustment to targets
    confidence_penalty: float
    reasoning: str


@dataclass
class CrossAgentInsight:
    """Insights learned from other agents."""
    source_agent: str
    pattern_similarity: float
    success_rate: float
    sample_size: int
    market_conditions: Dict[str, Any]
    recommendation: str


class AdvancedAgentLearning:
    """Advanced learning system with ensemble methods and cross-agent insights."""

    def __init__(self, agent_type: str):
        """Initialize advanced learning system."""
        self.agent_type = agent_type
        self.db = get_service_client()
        self.base_learning = AgentLearningService(agent_type)

        # Ensemble model configuration
        self.ensemble_model = EnsembleModel()

        # Market regime models
        self.regime_models = {}
        self._initialize_regime_models()

        # Statistical models
        self.statistical_cache = {}
        self.cache_expiry = datetime.now()

        logger.info(f"Advanced learning system initialized for {agent_type}")

    def _initialize_regime_models(self):
        """Initialize specialized models for different market regimes."""
        for regime in MarketRegimeType:
            self.regime_models[regime.value] = {
                'min_samples': 5,
                'confidence_multiplier': 1.0,
                'volatility_adjustment': 0.0,
                'trend_bias': 0.0
            }

        # Configure regime-specific parameters
        self.regime_models['bull_trending']['trend_bias'] = 0.1
        self.regime_models['bear_trending']['trend_bias'] = -0.1
        self.regime_models['high_volatility']['volatility_adjustment'] = -0.15
        self.regime_models['low_volatility']['volatility_adjustment'] = 0.05

    async def get_ensemble_learning_adjustment(self, signal: PlatformSignal) -> LearningAdjustment:
        """Get learning adjustment using ensemble of multiple models."""
        try:
            # Get base pattern matching adjustment
            pattern_adjustment = await self.base_learning.get_learning_adjustment(signal)

            # Get statistical model adjustment
            statistical_adjustment = await self._get_statistical_adjustment(signal)

            # Get regime-specific adjustment
            regime_adjustment = await self._get_regime_specific_adjustment(signal)

            # Get momentum-based adjustment
            momentum_adjustment = await self._get_momentum_adjustment(signal)

            # Combine using ensemble weights
            final_confidence_adj = (
                pattern_adjustment.confidence_adjustment * self.ensemble_model.pattern_weight +
                statistical_adjustment.confidence_adjustment * self.ensemble_model.statistical_weight +
                regime_adjustment.confidence_adjustment * self.ensemble_model.regime_weight +
                momentum_adjustment.confidence_adjustment * self.ensemble_model.momentum_weight
            )

            # Combine leverage multipliers
            final_leverage_mult = (
                pattern_adjustment.leverage_multiplier * self.ensemble_model.pattern_weight +
                statistical_adjustment.leverage_multiplier * self.ensemble_model.statistical_weight +
                regime_adjustment.leverage_multiplier * self.ensemble_model.regime_weight +
                momentum_adjustment.leverage_multiplier * self.ensemble_model.momentum_weight
            )

            # Combine position size multipliers
            final_position_mult = (
                pattern_adjustment.position_size_multiplier * self.ensemble_model.pattern_weight +
                statistical_adjustment.position_size_multiplier * self.ensemble_model.statistical_weight +
                regime_adjustment.position_size_multiplier * self.ensemble_model.regime_weight +
                momentum_adjustment.position_size_multiplier * self.ensemble_model.momentum_weight
            )

            # Determine overall confidence
            model_confidences = [
                pattern_adjustment.confidence_level,
                statistical_adjustment.confidence_level,
                regime_adjustment.confidence_level,
                momentum_adjustment.confidence_level
            ]

            high_confidence_count = sum(1 for c in model_confidences if c == 'high')
            if high_confidence_count >= 3:
                ensemble_confidence = 'high'
            elif high_confidence_count >= 2:
                ensemble_confidence = 'medium'
            else:
                ensemble_confidence = 'low'

            # Generate ensemble reasoning
            reasoning = self._generate_ensemble_reasoning(
                pattern_adjustment, statistical_adjustment, regime_adjustment, momentum_adjustment
            )

            # Combine pattern matches
            all_patterns = pattern_adjustment.pattern_matches + statistical_adjustment.pattern_matches

            return LearningAdjustment(
                confidence_adjustment=final_confidence_adj,
                leverage_multiplier=final_leverage_mult,
                position_size_multiplier=final_position_mult,
                reasoning=reasoning,
                pattern_matches=all_patterns,
                confidence_level=ensemble_confidence
            )

        except Exception as e:
            logger.error(f"Error in ensemble learning adjustment: {e}")
            # Fallback to base learning
            return await self.base_learning.get_learning_adjustment(signal)

    async def _get_statistical_adjustment(self, signal: PlatformSignal) -> LearningAdjustment:
        """Get adjustment based on statistical analysis of historical data.

        Statistical rigor improvements:
        - Minimum 20 trades (up from 5) to avoid learning from noise
        - Computes E[PnL] = P(win)*avg_win - P(loss)*avg_loss
        - Only applies adjustments when the 95% CI excludes zero
        - Scales adjustment magnitude by sqrt(n/30) dampening factor
        """
        MIN_RECORDS = 20  # Raised from 5 — estimation-error safeguard

        try:
            # Get historical performance for similar confidence levels
            confidence_range = (signal.confidence - 0.1, signal.confidence + 0.1)

            # Query historical performance in confidence range
            historical_result = self.db.table('agent_learning_memory').select(
                'outcome, pnl_percentage, signal_confidence'
            ).eq('agent_type', self.agent_type).eq(
                'token_symbol', signal.token_symbol
            ).gte('signal_confidence', confidence_range[0]).lte(
                'signal_confidence', confidence_range[1]
            ).gte('created_at', (datetime.now() - timedelta(days=60)).isoformat()).execute()

            records = historical_result.data or []

            if len(records) < MIN_RECORDS:
                return LearningAdjustment(
                    confidence_adjustment=0.0,
                    leverage_multiplier=1.0,
                    position_size_multiplier=1.0,
                    reasoning=f"Insufficient statistical data ({len(records)}/{MIN_RECORDS} trades)",
                    pattern_matches=[],
                    confidence_level='low'
                )

            # Calculate statistical metrics
            pnl_values = [r['pnl_percentage'] for r in records]
            wins = [r for r in records if r['outcome'] == 'win']
            win_rate = len(wins) / len(records)
            avg_pnl = sum(pnl_values) / len(pnl_values)
            std_pnl = np.std(pnl_values) if len(pnl_values) > 1 else 0

            # Expected Value: E[PnL] = P(win)*avg_win - P(loss)*avg_loss
            win_pnls = [r['pnl_percentage'] for r in records if r['outcome'] == 'win']
            loss_pnls = [abs(r['pnl_percentage']) for r in records if r['outcome'] != 'win']
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
            avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
            expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

            # 95% Confidence interval for the mean PnL
            confidence_interval = 1.96 * (std_pnl / np.sqrt(len(pnl_values))) if std_pnl > 0 else 0
            ci_lower = avg_pnl - confidence_interval
            ci_upper = avg_pnl + confidence_interval

            # Sample-size dampening: sqrt(n / 30), maxes at 1.0
            dampening = min(1.0, (len(records) / 30) ** 0.5)

            # Only apply adjustments if the CI EXCLUDES zero
            # (i.e., we have statistically meaningful evidence of direction)
            ci_excludes_zero = ci_lower > 0 or ci_upper < 0

            if not ci_excludes_zero:
                return LearningAdjustment(
                    confidence_adjustment=0.0,
                    leverage_multiplier=1.0,
                    position_size_multiplier=1.0,
                    reasoning=(
                        f"CI includes zero ({ci_lower:+.2f} to {ci_upper:+.2f}); "
                        f"EV={expected_value:+.2f}%, WR={win_rate:.1%} over {len(records)} trades — "
                        f"not yet statistically significant"
                    ),
                    pattern_matches=[f"Confidence range {confidence_range[0]:.2f}-{confidence_range[1]:.2f}"],
                    confidence_level='medium'
                )

            # Statistical adjustment based on performance — dampened by sample size
            if win_rate > 0.60 and expected_value > 1.0:
                stat_confidence_adj = min(0.15, (win_rate - 0.5) * 0.3) * dampening
                stat_leverage_mult = min(1.3, 1.0 + (expected_value / 15.0) * dampening)
            elif win_rate < 0.40 or expected_value < -0.5:
                stat_confidence_adj = max(-0.15, (win_rate - 0.5) * 0.3) * dampening
                stat_leverage_mult = max(0.7, 1.0 + (expected_value / 25.0) * dampening)
            else:
                stat_confidence_adj = 0.0
                stat_leverage_mult = 1.0

            # Adjust for volatility (higher volatility = lower position)
            volatility_factor = min(1.2, max(0.8, 1.0 - (std_pnl / 50.0)))

            reasoning = (
                f"Stats: WR={win_rate:.1%}, avg={avg_pnl:+.1f}%, EV={expected_value:+.2f}%, "
                f"CI=[{ci_lower:+.2f}, {ci_upper:+.2f}], n={len(records)}, "
                f"dampening={dampening:.2f}"
            )

            return LearningAdjustment(
                confidence_adjustment=stat_confidence_adj,
                leverage_multiplier=stat_leverage_mult,
                position_size_multiplier=volatility_factor,
                reasoning=reasoning,
                pattern_matches=[f"Confidence range {confidence_range[0]:.2f}-{confidence_range[1]:.2f}"],
                confidence_level='high' if len(records) >= 30 else 'medium'
            )

        except Exception as e:
            logger.error(f"Error in statistical adjustment: {e}")
            return LearningAdjustment(
                confidence_adjustment=0.0,
                leverage_multiplier=1.0,
                position_size_multiplier=1.0,
                reasoning=f"Statistical analysis error: {str(e)}",
                pattern_matches=[],
                confidence_level='low'
            )

    async def _get_regime_specific_adjustment(self, signal: PlatformSignal) -> LearningAdjustment:
        """Get adjustment based on current market regime."""
        try:
            # Determine current market regime
            market_regime = signal.market_conditions.get('regime', 'unknown')
            volatility = signal.market_conditions.get('volatility_percentile', 50)

            # Map to our regime types
            if market_regime == 'trending_up':
                regime_type = MarketRegimeType.BULL_TRENDING
            elif market_regime == 'trending_down':
                regime_type = MarketRegimeType.BEAR_TRENDING
            elif volatility > 75:
                regime_type = MarketRegimeType.HIGH_VOLATILITY
            elif volatility < 25:
                regime_type = MarketRegimeType.LOW_VOLATILITY
            else:
                regime_type = MarketRegimeType.SIDEWAYS

            # Get regime-specific model
            regime_model = self.regime_models.get(regime_type.value, {})

            # Query historical performance in this regime
            regime_result = self.db.table('agent_learning_memory').select(
                'outcome, pnl_percentage'
            ).eq('agent_type', self.agent_type).contains(
                'pattern_features', {'market_regime': market_regime}
            ).gte('created_at', (datetime.now() - timedelta(days=45)).isoformat()).execute()

            regime_records = regime_result.data or []

            if len(regime_records) < regime_model.get('min_samples', 5):
                return LearningAdjustment(
                    confidence_adjustment=0.0,
                    leverage_multiplier=1.0,
                    position_size_multiplier=1.0,
                    reasoning=f"Insufficient data for {regime_type.value} regime",
                    pattern_matches=[],
                    confidence_level='low'
                )

            # Calculate regime performance
            regime_win_rate = len([r for r in regime_records if r['outcome'] == 'win']) / len(regime_records)
            regime_avg_pnl = sum(r['pnl_percentage'] for r in regime_records) / len(regime_records)

            # Apply regime-specific adjustments
            regime_confidence_mult = regime_model.get('confidence_multiplier', 1.0)
            regime_bias = regime_model.get('trend_bias', 0.0)
            volatility_adj = regime_model.get('volatility_adjustment', 0.0)

            # Calculate final adjustments
            regime_confidence_adj = (regime_win_rate - 0.5) * 0.2 * regime_confidence_mult + regime_bias

            # Adjust for signal direction vs regime bias
            if signal.direction == 'LONG' and regime_bias > 0:
                regime_confidence_adj += 0.05
            elif signal.direction == 'SHORT' and regime_bias < 0:
                regime_confidence_adj += 0.05

            regime_leverage_mult = 1.0 + volatility_adj
            regime_position_mult = 1.0 + volatility_adj

            reasoning = f"{regime_type.value} regime: {regime_win_rate:.1%} win rate over {len(regime_records)} trades"

            return LearningAdjustment(
                confidence_adjustment=regime_confidence_adj,
                leverage_multiplier=regime_leverage_mult,
                position_size_multiplier=regime_position_mult,
                reasoning=reasoning,
                pattern_matches=[f"{regime_type.value} regime"],
                confidence_level='medium' if len(regime_records) >= 10 else 'low'
            )

        except Exception as e:
            logger.error(f"Error in regime-specific adjustment: {e}")
            return LearningAdjustment(
                confidence_adjustment=0.0,
                leverage_multiplier=1.0,
                position_size_multiplier=1.0,
                reasoning=f"Regime analysis error: {str(e)}",
                pattern_matches=[],
                confidence_level='low'
            )

    async def _get_momentum_adjustment(self, signal: PlatformSignal) -> LearningAdjustment:
        """Get adjustment based on recent trading momentum."""
        try:
            # Get recent trading performance (last 7 days)
            recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()

            recent_result = self.db.table('agent_learning_memory').select(
                'outcome, pnl_percentage, created_at'
            ).eq('agent_type', self.agent_type).gte('created_at', recent_cutoff).execute()

            recent_records = recent_result.data or []

            if len(recent_records) < 3:
                return LearningAdjustment(
                    confidence_adjustment=0.0,
                    leverage_multiplier=1.0,
                    position_size_multiplier=1.0,
                    reasoning="Insufficient recent trading data",
                    pattern_matches=[],
                    confidence_level='low'
                )

            # Calculate momentum metrics
            recent_wins = len([r for r in recent_records if r['outcome'] == 'win'])
            recent_win_rate = recent_wins / len(recent_records)
            recent_avg_pnl = sum(r['pnl_percentage'] for r in recent_records) / len(recent_records)

            # Calculate streak (last 5 trades)
            last_trades = sorted(recent_records, key=lambda x: x['created_at'], reverse=True)[:5]
            current_streak = 0
            for trade in last_trades:
                if trade['outcome'] == 'win':
                    current_streak += 1
                else:
                    break

            # Momentum adjustments
            if recent_win_rate > 0.7 and recent_avg_pnl > 2.0:
                # Hot streak - slight confidence boost but reduced position size (avoid overconfidence)
                momentum_confidence_adj = 0.1
                momentum_leverage_mult = 1.1
                momentum_position_mult = 0.9  # Reduce position to manage risk
                momentum_reason = f"Hot streak: {recent_win_rate:.1%} recent win rate"
            elif recent_win_rate < 0.3 or recent_avg_pnl < -2.0:
                # Cold streak - reduce confidence and position
                momentum_confidence_adj = -0.1
                momentum_leverage_mult = 0.8
                momentum_position_mult = 0.7
                momentum_reason = f"Cold streak: {recent_win_rate:.1%} recent win rate"
            elif current_streak >= 3:
                # Long winning streak - be cautious
                momentum_confidence_adj = 0.05
                momentum_leverage_mult = 0.9
                momentum_position_mult = 0.8
                momentum_reason = f"Winning streak of {current_streak} trades - exercising caution"
            else:
                # Normal momentum
                momentum_confidence_adj = 0.0
                momentum_leverage_mult = 1.0
                momentum_position_mult = 1.0
                momentum_reason = "Normal trading momentum"

            return LearningAdjustment(
                confidence_adjustment=momentum_confidence_adj,
                leverage_multiplier=momentum_leverage_mult,
                position_size_multiplier=momentum_position_mult,
                reasoning=momentum_reason,
                pattern_matches=[f"Recent {len(recent_records)} trades"],
                confidence_level='medium'
            )

        except Exception as e:
            logger.error(f"Error in momentum adjustment: {e}")
            return LearningAdjustment(
                confidence_adjustment=0.0,
                leverage_multiplier=1.0,
                position_size_multiplier=1.0,
                reasoning=f"Momentum analysis error: {str(e)}",
                pattern_matches=[],
                confidence_level='low'
            )

    def _generate_ensemble_reasoning(
        self,
        pattern: LearningAdjustment,
        statistical: LearningAdjustment,
        regime: LearningAdjustment,
        momentum: LearningAdjustment
    ) -> str:
        """Generate combined reasoning for ensemble adjustment."""
        parts = []

        if abs(pattern.confidence_adjustment) > 0.05:
            parts.append(f"Pattern: {pattern.reasoning}")

        if abs(statistical.confidence_adjustment) > 0.05:
            parts.append(f"Stats: {statistical.reasoning}")

        if abs(regime.confidence_adjustment) > 0.05:
            parts.append(f"Regime: {regime.reasoning}")

        if abs(momentum.confidence_adjustment) > 0.05:
            parts.append(f"Momentum: {momentum.reasoning}")

        if not parts:
            return "Ensemble analysis shows neutral conditions"

        return "Ensemble: " + "; ".join(parts)

    async def get_dynamic_risk_adjustment(self, signal: PlatformSignal) -> RiskAdjustment:
        """Get dynamic risk adjustment based on advanced learning models."""
        try:
            # Get ensemble learning adjustment
            learning_adj = await self.get_ensemble_learning_adjustment(signal)

            # Calculate base risk adjustments
            base_position_mult = learning_adj.position_size_multiplier
            base_leverage_mult = learning_adj.leverage_multiplier

            # Market condition adjustments
            volatility = signal.market_conditions.get('volatility_percentile', 50)
            market_regime = signal.market_conditions.get('regime', 'ranging')

            # Volatility-based adjustments
            if volatility > 80:  # Extreme volatility
                volatility_position_mult = 0.6
                volatility_leverage_mult = 0.7
                stop_loss_tightening = 0.8  # Tighter stop loss
                take_profit_adjustment = 1.2  # Wider targets
            elif volatility > 60:  # High volatility
                volatility_position_mult = 0.8
                volatility_leverage_mult = 0.9
                stop_loss_tightening = 0.9
                take_profit_adjustment = 1.1
            elif volatility < 20:  # Low volatility
                volatility_position_mult = 1.2
                volatility_leverage_mult = 1.1
                stop_loss_tightening = 1.1  # Looser stop loss
                take_profit_adjustment = 0.9  # Tighter targets
            else:  # Normal volatility
                volatility_position_mult = 1.0
                volatility_leverage_mult = 1.0
                stop_loss_tightening = 1.0
                take_profit_adjustment = 1.0

            # Confidence-based penalty
            if signal.confidence < 0.4:
                confidence_penalty = 0.5  # Severe reduction
            elif signal.confidence < 0.6:
                confidence_penalty = 0.8  # Moderate reduction
            else:
                confidence_penalty = 1.0  # No penalty

            # Combine all adjustments
            final_position_mult = base_position_mult * volatility_position_mult * confidence_penalty
            final_leverage_mult = base_leverage_mult * volatility_leverage_mult * confidence_penalty

            # Generate reasoning
            risk_reasoning_parts = []
            if volatility > 60:
                risk_reasoning_parts.append(f"High volatility ({volatility}%) - reduced position")
            if signal.confidence < 0.6:
                risk_reasoning_parts.append(f"Low confidence ({signal.confidence:.1%}) - conservative sizing")
            if learning_adj.confidence_adjustment < -0.05:
                risk_reasoning_parts.append("Learning suggests caution")

            risk_reasoning = "; ".join(risk_reasoning_parts) if risk_reasoning_parts else "Standard risk parameters"

            return RiskAdjustment(
                position_size_multiplier=final_position_mult,
                leverage_multiplier=final_leverage_mult,
                stop_loss_adjustment=stop_loss_tightening,
                take_profit_adjustment=take_profit_adjustment,
                confidence_penalty=confidence_penalty,
                reasoning=risk_reasoning
            )

        except Exception as e:
            logger.error(f"Error in dynamic risk adjustment: {e}")
            return RiskAdjustment(
                position_size_multiplier=0.5,  # Conservative default
                leverage_multiplier=0.5,
                stop_loss_adjustment=0.8,
                take_profit_adjustment=1.2,
                confidence_penalty=0.5,
                reasoning=f"Error in risk calculation - using conservative defaults: {str(e)}"
            )

    async def get_cross_agent_insights(self, signal: PlatformSignal) -> List[CrossAgentInsight]:
        """Get insights from other agents' performance on similar patterns."""
        try:
            insights = []

            # Get pattern features for current signal
            features = self.base_learning._extract_pattern_features(signal)

            # Check other agents' performance on similar patterns
            other_agents = ['yuki', 'sakura', 'ryu']
            other_agents.remove(self.agent_type)  # Remove current agent

            for other_agent in other_agents:
                # Get similar patterns from other agent
                similar_result = self.db.table('agent_learning_memory').select(
                    'outcome, pnl_percentage, pattern_features'
                ).eq('agent_type', other_agent).eq(
                    'token_symbol', signal.token_symbol
                ).gte('created_at', (datetime.now() - timedelta(days=30)).isoformat()).execute()

                similar_records = similar_result.data or []

                if len(similar_records) < 3:
                    continue

                # Calculate similarity and performance
                relevant_records = []
                for record in similar_records:
                    try:
                        # Simple similarity check based on key features
                        other_features = record.get('pattern_features', {})
                        if (other_features.get('direction') == features.direction and
                            other_features.get('market_regime') == features.market_regime):
                            relevant_records.append(record)
                    except:
                        continue

                if len(relevant_records) >= 3:
                    wins = len([r for r in relevant_records if r['outcome'] == 'win'])
                    success_rate = wins / len(relevant_records)
                    avg_pnl = sum(r['pnl_percentage'] for r in relevant_records) / len(relevant_records)

                    # Generate recommendation
                    if success_rate > 0.7 and avg_pnl > 2.0:
                        recommendation = f"{other_agent} shows strong performance ({success_rate:.1%}) - consider similar approach"
                    elif success_rate < 0.3:
                        recommendation = f"{other_agent} struggled ({success_rate:.1%}) - exercise caution"
                    else:
                        recommendation = f"{other_agent} shows mixed results ({success_rate:.1%})"

                    insight = CrossAgentInsight(
                        source_agent=other_agent,
                        pattern_similarity=0.8,  # Simplified for now
                        success_rate=success_rate,
                        sample_size=len(relevant_records),
                        market_conditions=signal.market_conditions,
                        recommendation=recommendation
                    )
                    insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Error getting cross-agent insights: {e}")
            return []


# Global instances
_advanced_learning_services: Dict[str, AdvancedAgentLearning] = {}


def get_advanced_agent_learning(agent_type: str) -> AdvancedAgentLearning:
    """Get or create advanced learning service for agent type."""
    global _advanced_learning_services

    if agent_type not in _advanced_learning_services:
        _advanced_learning_services[agent_type] = AdvancedAgentLearning(agent_type)

    return _advanced_learning_services[agent_type]