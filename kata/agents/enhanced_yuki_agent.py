"""
Enhanced Yuki Agent with Machine Learning Integration

This enhanced version of the Yuki agent includes:
- Learning from past trading outcomes
- Pattern recognition and confidence adjustment
- Dynamic risk management based on historical performance
- Intelligent leverage and position sizing recommendations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

from kata.agents.yuki_agent import YukiAgent, FuturesMarketData, TechnicalIndicators, MarketRegime
from kata.agents.base_agent import MarketData, TradingSignal, SignalType, RiskLevel
from kata.services.agent_learning_service import (
    get_agent_learning_service,
    AgentLearningService,
    LearningAdjustment,
    PatternFeatures
)
from kata.services.platform_signal_service import PlatformSignal
from kata.services.lightweight_performance_service import get_lightweight_performance_service

logger = logging.getLogger(__name__)


class EnhancedYukiAgent(YukiAgent):
    """
    Enhanced Yuki Agent with machine learning capabilities.

    Learns from trading outcomes to improve signal quality, confidence scoring,
    and risk management over time.
    """

    def __init__(self, user_id: str, config: Dict[str, Any]):
        """Initialize enhanced Yuki agent with learning capabilities."""
        super().__init__(user_id, config)

        # Initialize learning service
        self.learning_service: AgentLearningService = get_agent_learning_service('yuki')
        self.performance_service = get_lightweight_performance_service()

        # Learning configuration
        self.learning_config = {
            'min_confidence_adjustment': -0.25,  # Maximum confidence reduction
            'max_confidence_adjustment': 0.25,   # Maximum confidence boost
            'min_pattern_matches': 3,            # Minimum trades to trust pattern
            'similarity_threshold': 0.6,         # Minimum pattern similarity
            'learning_enabled': config.get('learning_enabled', True),
            'auto_record_outcomes': config.get('auto_record_outcomes', True)
        }

        # Enhanced risk parameters with learning
        self.enhanced_risk_params = {
            'base_position_size': 0.20,          # 20% base position size
            'max_position_size': 1.00,           # 100% maximum position size
            'confidence_position_multiplier': 2.0, # Multiply position by confidence
            'learning_position_multiplier': 1.5,   # Additional multiplier from learning
            'max_leverage': 40.0,                # 40x maximum leverage
            'leverage_learning_threshold': 0.7,  # Require 70% win rate to increase leverage
            'max_leverage_learning_boost': 2.0,    # Max leverage boost from learning
            'risk_reduction_threshold': 0.3        # Reduce risk if confidence < 30%
        }

        # Learning statistics
        self.learning_stats = {
            'patterns_applied': 0,
            'confidence_adjustments': 0,
            'learning_signals_generated': 0,
            'last_learning_update': None
        }

        logger.info(f"Enhanced Yuki agent initialized with learning capabilities for user {user_id}")

    async def analyze_market_with_learning(
        self,
        market_data: MarketData,
        focus_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced market analysis with learning integration.

        Args:
            market_data: Market data for analysis
            focus_token: Optional token to focus analysis on

        Returns:
            Analysis results with learning enhancements
        """
        try:
            # Perform base market analysis
            base_analysis = await super().analyze_market(market_data, focus_token)

            if not self.learning_config['learning_enabled']:
                return base_analysis

            # Apply learning enhancements to any identified opportunities
            enhanced_opportunities = []

            for opportunity in base_analysis.get('futures_opportunities', []):
                try:
                    # Create a mock PlatformSignal for learning analysis
                    mock_signal = self._create_mock_signal_from_opportunity(
                        opportunity, market_data, base_analysis
                    )

                    if mock_signal:
                        # Get learning adjustment
                        learning_adj = await self.learning_service.get_learning_adjustment(mock_signal)

                        # Apply learning enhancements
                        enhanced_opportunity = await self._apply_learning_to_opportunity(
                            opportunity, learning_adj, base_analysis
                        )

                        enhanced_opportunities.append(enhanced_opportunity)
                        self.learning_stats['patterns_applied'] += 1

                        if abs(learning_adj.confidence_adjustment) > 0.05:
                            self.learning_stats['confidence_adjustments'] += 1
                    else:
                        enhanced_opportunities.append(opportunity)

                except Exception as e:
                    logger.warning(f"Error applying learning to opportunity: {e}")
                    enhanced_opportunities.append(opportunity)

            # Update analysis with enhanced opportunities
            base_analysis['futures_opportunities'] = enhanced_opportunities

            # Add learning metadata
            base_analysis['learning_metadata'] = {
                'learning_enabled': True,
                'patterns_applied': self.learning_stats['patterns_applied'],
                'confidence_adjustments': self.learning_stats['confidence_adjustments'],
                'last_update': datetime.now().isoformat()
            }

            # Update learning statistics
            learning_stats = await self.learning_service.get_learning_statistics()
            base_analysis['learning_stats'] = learning_stats

            return base_analysis

        except Exception as e:
            logger.error(f"Error in enhanced market analysis: {e}")
            # Fallback to base analysis
            return await super().analyze_market(market_data, focus_token)

    async def record_signal_outcome(
        self,
        signal: PlatformSignal,
        outcome: str,
        pnl_percentage: float,
        exit_reason: Optional[str] = None,
        max_profit: float = 0,
        max_loss: float = 0
    ) -> bool:
        """
        Record trading signal outcome for learning.

        Args:
            signal: The original platform signal
            outcome: Trade outcome ('win', 'loss', 'scratch')
            pnl_percentage: Final P&L percentage
            exit_reason: Reason for exit
            max_profit: Maximum profit reached
            max_loss: Maximum loss reached

        Returns:
            True if recorded successfully
        """
        try:
            if not self.learning_config['auto_record_outcomes']:
                return True

            success = await self.learning_service.record_trade_outcome(
                signal=signal,
                outcome=outcome,
                pnl_percentage=pnl_percentage,
                exit_reason=exit_reason,
                max_profit_reached=max_profit,
                max_loss_reached=max_loss
            )

            if success:
                logger.info(f"📚 Recorded learning outcome for Yuki: {signal.token_symbol} {outcome} ({pnl_percentage:+.2f}%)")
                self.learning_stats['last_learning_update'] = datetime.now()

            return success

        except Exception as e:
            logger.error(f"Error recording signal outcome: {e}")
            return False

    async def get_enhanced_signal_confidence(self, signal: PlatformSignal) -> Tuple[float, Dict[str, Any]]:
        """
        Get enhanced confidence score using learning data.

        Args:
            signal: Platform signal to analyze

        Returns:
            Tuple of (enhanced_confidence, learning_metadata)
        """
        try:
            # Get learning adjustment
            learning_adj = await self.learning_service.get_learning_adjustment(signal)

            # Apply confidence adjustment
            original_confidence = signal.confidence
            enhanced_confidence = max(0.0, min(1.0,
                original_confidence + learning_adj.confidence_adjustment
            ))

            # Create learning metadata
            learning_metadata = {
                'original_confidence': original_confidence,
                'learning_adjustment': learning_adj.confidence_adjustment,
                'enhanced_confidence': enhanced_confidence,
                'adjustment_reasoning': learning_adj.reasoning,
                'pattern_matches': learning_adj.pattern_matches,
                'learning_confidence_level': learning_adj.confidence_level,
                'leverage_multiplier': learning_adj.leverage_multiplier,
                'position_size_multiplier': learning_adj.position_size_multiplier
            }

            return enhanced_confidence, learning_metadata

        except Exception as e:
            logger.error(f"Error getting enhanced confidence: {e}")
            return signal.confidence, {'error': str(e)}

    async def get_smart_position_sizing(
        self,
        signal: PlatformSignal,
        portfolio_value: float,
        learning_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate intelligent position sizing based on learning data.

        Args:
            signal: Platform signal
            portfolio_value: Current portfolio value
            learning_metadata: Learning adjustment data

        Returns:
            Position sizing recommendations
        """
        try:
            # Base position size from the LLM signal
            llm_position_size = getattr(signal, 'position_size', self.enhanced_risk_params['base_position_size'] * 100)
            base_size = llm_position_size / 100.0

            # Confidence multiplier
            confidence = learning_metadata.get('enhanced_confidence', signal.confidence) if learning_metadata else signal.confidence
            confidence_multiplier = 1.0 + (confidence - 0.5) * self.enhanced_risk_params['confidence_position_multiplier']

            # Learning multiplier
            learning_multiplier = 1.0
            if learning_metadata:
                learning_pos_mult = learning_metadata.get('position_size_multiplier', 1.0)
                learning_multiplier = min(
                    self.enhanced_risk_params['learning_position_multiplier'],
                    learning_pos_mult
                )

            # Calculate final position size
            position_size_percent = min(
                self.enhanced_risk_params['max_position_size'],
                base_size * confidence_multiplier * learning_multiplier
            )

            # Apply risk reduction for low confidence
            if confidence < self.enhanced_risk_params['risk_reduction_threshold']:
                position_size_percent *= 0.5

            # Calculate dollar amounts
            position_value = portfolio_value * position_size_percent

            # Smart leverage calculation
            base_leverage = signal.leverage
            learning_leverage_mult = learning_metadata.get('leverage_multiplier', 1.0) if learning_metadata else 1.0

            smart_leverage = min(
                40,  # Maximum leverage cap
                max(1, base_leverage * learning_leverage_mult)
            )

            return {
                'position_size_percent': position_size_percent,
                'position_value_usd': position_value,
                'recommended_leverage': smart_leverage,
                'confidence_multiplier': confidence_multiplier,
                'learning_multiplier': learning_multiplier,
                'risk_adjusted': confidence < self.enhanced_risk_params['risk_reduction_threshold'],
                'reasoning': self._generate_position_reasoning(
                    confidence, learning_multiplier, position_size_percent
                )
            }

        except Exception as e:
            logger.error(f"Error calculating smart position sizing: {e}")
            # Return LLM defaults
            fallback_pos = getattr(signal, 'position_size', 2.0) / 100.0
            fallback_lev = getattr(signal, 'leverage', 2)
            
            return {
                'position_size_percent': fallback_pos,
                'position_value_usd': portfolio_value * fallback_pos if portfolio_value else 100.0,
                'recommended_leverage': fallback_lev,
                'confidence_multiplier': 1.0,
                'learning_multiplier': 1.0,
                'risk_adjusted': True,
                'reasoning': f"Error in calculation, using conservative defaults: {str(e)}"
            }

    def _create_mock_signal_from_opportunity(
        self,
        opportunity: Dict[str, Any],
        market_data: MarketData,
        analysis: Dict[str, Any]
    ) -> Optional[PlatformSignal]:
        """Create a mock PlatformSignal from trading opportunity for learning analysis."""
        try:
            # Extract opportunity data
            symbol = opportunity.get('symbol', 'ETH')
            direction = opportunity.get('signal', 'LONG').upper()
            confidence = opportunity.get('confidence', 0.5)

            # Get current price
            current_price = market_data.token_prices.get(symbol, Decimal('0'))
            if current_price == 0:
                return None

            price_float = float(current_price)

            # Calculate targets and stop loss based on opportunity
            if direction == 'LONG':
                target_1 = price_float * (1 + opportunity.get('target_1_percent', 5) / 100)
                target_2 = price_float * (1 + opportunity.get('target_2_percent', 10) / 100)
                stop_loss = price_float * (1 - opportunity.get('stop_loss_percent', 3) / 100)
            else:
                target_1 = price_float * (1 - opportunity.get('target_1_percent', 5) / 100)
                target_2 = price_float * (1 - opportunity.get('target_2_percent', 10) / 100)
                stop_loss = price_float * (1 + opportunity.get('stop_loss_percent', 3) / 100)

            # Create mock signal
            mock_signal = PlatformSignal(
                signal_id=f"mock_{symbol}_{direction}_{int(datetime.now().timestamp())}",
                token_symbol=symbol,
                direction=direction,
                timeframe='1h',
                confidence=confidence,
                overall_score=confidence * 100,
                signal_strength='medium',
                time_horizon='short',
                entry_price=price_float,
                target_1=target_1,
                target_1_probability=0.6,
                target_2=target_2,
                target_2_probability=0.3,
                stop_loss=stop_loss,
                risk_reward_ratio=2.0,
                market_conditions=analysis.get('market_conditions', {}),
                technical_indicators=analysis.get('technical_summary', {}),
                sentiment_data={},
                risk_factors=[],
                signal_pool='fast_mode',
                opportunity_rank=opportunity.get('rank', 3),
                analysis_timestamp=market_data.timestamp,
                expires_at=market_data.timestamp + timedelta(hours=24),
                validity_window_hours=24,
                leverage=opportunity.get('leverage', 5),
                position_size=10.0,
                risk_level='MEDIUM'
            )

            return mock_signal

        except Exception as e:
            logger.error(f"Error creating mock signal: {e}")
            return None

    async def _apply_learning_to_opportunity(
        self,
        opportunity: Dict[str, Any],
        learning_adj: LearningAdjustment,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply learning adjustments to trading opportunity."""
        try:
            enhanced_opportunity = opportunity.copy()

            # Apply confidence adjustment
            original_confidence = opportunity.get('confidence', 0.5)
            enhanced_confidence = max(0.0, min(1.0,
                original_confidence + learning_adj.confidence_adjustment
            ))

            enhanced_opportunity['confidence'] = enhanced_confidence
            enhanced_opportunity['original_confidence'] = original_confidence

            # Apply leverage adjustment
            original_leverage = opportunity.get('leverage', 5)
            enhanced_leverage = max(1, min(10,
                original_leverage * learning_adj.leverage_multiplier
            ))
            enhanced_opportunity['leverage'] = enhanced_leverage

            # Apply position size adjustment
            enhanced_opportunity['position_size_multiplier'] = learning_adj.position_size_multiplier

            # Add learning metadata
            enhanced_opportunity['learning_data'] = {
                'confidence_adjustment': learning_adj.confidence_adjustment,
                'leverage_multiplier': learning_adj.leverage_multiplier,
                'position_size_multiplier': learning_adj.position_size_multiplier,
                'reasoning': learning_adj.reasoning,
                'pattern_matches': learning_adj.pattern_matches,
                'confidence_level': learning_adj.confidence_level,
                'learning_applied': True
            }

            # Update opportunity scoring
            if enhanced_confidence > original_confidence + 0.1:
                enhanced_opportunity['learning_boost'] = 'high'
            elif enhanced_confidence > original_confidence + 0.05:
                enhanced_opportunity['learning_boost'] = 'medium'
            elif enhanced_confidence < original_confidence - 0.05:
                enhanced_opportunity['learning_boost'] = 'negative'
            else:
                enhanced_opportunity['learning_boost'] = 'none'

            return enhanced_opportunity

        except Exception as e:
            logger.error(f"Error applying learning to opportunity: {e}")
            return opportunity

    def _generate_position_reasoning(
        self,
        confidence: float,
        learning_multiplier: float,
        position_size: float
    ) -> str:
        """Generate human-readable reasoning for position sizing."""
        reasons = []

        if confidence > 0.8:
            reasons.append("High confidence signal")
        elif confidence < 0.4:
            reasons.append("Low confidence - reduced position")

        if learning_multiplier > 1.1:
            reasons.append("Learning data suggests larger position")
        elif learning_multiplier < 0.9:
            reasons.append("Learning data suggests smaller position")

        if position_size > 0.1:
            reasons.append("Large position due to strong conviction")
        elif position_size < 0.03:
            reasons.append("Conservative position due to uncertainty")

        return "; ".join(reasons) if reasons else "Standard position sizing"

    async def get_learning_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive learning performance report."""
        try:
            # Get learning statistics
            learning_stats = await self.learning_service.get_learning_statistics()

            # Get recent performance from platform signals
            recent_performance = await self._get_recent_performance_metrics()

            # Compile report
            report = {
                'agent_type': 'yuki',
                'learning_enabled': self.learning_config['learning_enabled'],
                'learning_statistics': learning_stats,
                'recent_performance': recent_performance,
                'session_stats': self.learning_stats.copy(),
                'configuration': {
                    'min_pattern_matches': self.learning_config['min_pattern_matches'],
                    'similarity_threshold': self.learning_config['similarity_threshold'],
                    'max_confidence_adjustment': self.learning_config['max_confidence_adjustment']
                },
                'generated_at': datetime.now().isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"Error generating learning performance report: {e}")
            return {
                'agent_type': 'yuki',
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    async def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics for learning evaluation."""
        try:
            # This would typically get data from the performance service
            # For now, return placeholder data
            return {
                'signals_last_30_days': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'learning_impact_estimated': 0.0,
                'note': 'Metrics will be available after first trades are recorded'
            }

        except Exception as e:
            logger.error(f"Error getting recent performance metrics: {e}")
            return {'error': str(e)}

    async def reflect_on_trade_outcome(
        self,
        symbol: str,
        pnl_percent: float,
        exit_reason: str,
        trade_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a postmortem self-reflection on a closed trade using DeepSeek.
        Extracts actionable lessons learned and saves them to AgentMemoryService.
        """
        try:
            logger.info(f"🤔 [Self-Reflection] Running trade postmortem for {symbol} (PnL: {pnl_percent:+.2f}%, Exit: {exit_reason})")
            
            prompt = f"""
You are Yuki, analyzing your recent trade outcome to self-improve your future trading strategy.

CLOSED TRADE DETAILS:
- Symbol: {symbol}
- PnL %: {pnl_percent:+.2f}%
- Exit Reason: {exit_reason}

TASK:
Write a concise, 1-2 sentence lesson learned from this trade outcome. Focus on what caused the outcome (e.g. stop-loss hit in high volatility, profit target reached on funding divergence, leverage was too high) and what rule adjustment should be applied next time.

FORMAT:
LESSON: [1-2 sentence actionable rule]
"""
            lesson_text = f"When trading {symbol}, strictly observe liquidation distance and monitor volatility before adjusting leverage."
            
            if self.llm_service and self.llm_service.client:
                llm_res, _ = await self.llm_service._query_llm(prompt)
                if "LESSON:" in llm_res:
                    lesson_text = llm_res.split("LESSON:")[1].strip().split("\n")[0]

            # Save lesson to memory
            reflection = await self.memory_service.save_reflection(
                lesson_learned=lesson_text,
                review_type="trade_postmortem",
                symbol=symbol,
                proposed_rule_changes={'last_pnl_percent': pnl_percent, 'exit_reason': exit_reason}
            )

            logger.info(f"💡 [Self-Reflection Saved] {symbol}: '{lesson_text}'")
            return {
                'status': 'success',
                'symbol': symbol,
                'lesson': lesson_text,
                'reflection_id': reflection.id if reflection else None
            }

        except Exception as e:
            logger.error(f"Error running trade postmortem reflection: {e}")
            return {'status': 'error', 'error': str(e)}

    async def run_weekly_self_review(self) -> Dict[str, Any]:
        """
        Run a weekly self-improvement review across all recent trades and active goals.
        Synthesizes lessons and updates operational goals.
        """
        try:
            logger.info("📅 [Weekly Review] Running agent self-improvement review...")
            
            recent_lessons = await self.memory_service.get_active_lessons(limit=10)
            lessons_summary = "\n".join([f"- {l}" for l in recent_lessons]) if recent_lessons else "No recent trade lessons."

            prompt = f"""
You are Yuki performing a weekly strategy review to refine your trading philosophy.

RECENT LESSONS LEARNED:
{lessons_summary}

PROPOSE AN UPDATED OPERATIONAL GOAL for the next week:
Write a 1-sentence strategic goal statement for Yuki to focus on.

FORMAT:
GOAL: [1-sentence goal]
"""
            new_goal_statement = "Maximize win rate on high-liquidity perpetuals while maintaining strict 30% margin buffers."
            
            if self.llm_service and self.llm_service.client:
                llm_res, _ = await self.llm_service._query_llm(prompt)
                if "GOAL:" in llm_res:
                    new_goal_statement = llm_res.split("GOAL:")[1].strip().split("\n")[0]

            # Update goal in memory service
            new_goal = await self.memory_service.set_goal(goal_statement=new_goal_statement)

            logger.info(f"🎯 [New Weekly Goal Set] '{new_goal_statement}'")
            return {
                'status': 'success',
                'new_goal': new_goal_statement,
                'goal_id': new_goal.id if new_goal else None
            }

        except Exception as e:
            logger.error(f"Error running weekly self review: {e}")
            return {'status': 'error', 'error': str(e)}


# Factory function for creating enhanced Yuki agent
def create_enhanced_yuki_agent(user_id: str, config: Dict[str, Any]) -> EnhancedYukiAgent:
    """Create an enhanced Yuki agent with learning capabilities."""
    return EnhancedYukiAgent(user_id, config)
