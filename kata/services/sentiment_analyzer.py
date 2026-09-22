"""
Advanced Sentiment Scoring and Analysis Algorithms for Yuki Agent

Provides sophisticated sentiment analysis algorithms including:
- Multi-dimensional sentiment scoring
- Temporal sentiment analysis and trend detection
- Market regime classification
- Risk sentiment modeling
- Cross-asset sentiment correlation
- Sentiment-based trading signals generation
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import math

logger = logging.getLogger(__name__)


@dataclass
class SentimentDimensions:
    """Multi-dimensional sentiment breakdown."""
    bullish_bearish: float  # -1 to 1 (traditional sentiment)
    fear_greed: float  # -1 to 1 (fear vs greed)
    confidence_uncertainty: float  # -1 to 1 (uncertainty vs confidence)
    euphoria_depression: float  # -1 to 1 (extreme emotional states)
    fomo_fud: float  # -1 to 1 (FOMO vs FUD)
    accumulation_distribution: float  # -1 to 1 (smart money vs retail)


@dataclass
class TemporalSentiment:
    """Time-based sentiment analysis."""
    current_sentiment: float
    momentum: float  # Rate of sentiment change
    acceleration: float  # Rate of momentum change
    volatility: float  # Sentiment volatility
    mean_reversion_signal: float  # Mean reversion tendency
    breakout_signal: float  # Trend continuation signal
    cycle_position: str  # 'early', 'mid', 'late' cycle
    time_decay_factor: float  # How quickly sentiment decays


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # 'bull_market', 'bear_market', 'sideways', 'crisis', 'euphoria'
    confidence: float  # Confidence in regime classification
    regime_strength: float  # How strong the regime is
    regime_duration: int  # How long regime has been active (hours)
    transition_probability: float  # Probability of regime change
    next_likely_regime: str  # Most likely next regime
    regime_characteristics: Dict[str, float]  # Key regime metrics


@dataclass
class RiskSentiment:
    """Risk-based sentiment analysis."""
    risk_appetite: float  # -1 (risk off) to 1 (risk on)
    systemic_risk: float  # 0 to 1 (low to high systemic risk)
    tail_risk: float  # 0 to 1 (low to high tail risk probability)
    correlation_risk: float  # 0 to 1 (asset correlation risk)
    liquidity_risk: float  # 0 to 1 (market liquidity risk)
    volatility_regime: str  # 'low', 'medium', 'high', 'extreme'
    stress_level: str  # 'calm', 'elevated', 'stressed', 'crisis'


@dataclass
class TradingSignals:
    """Sentiment-based trading signals."""
    primary_signal: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    signal_strength: float  # 0 to 1
    time_horizon: str  # 'short', 'medium', 'long'
    confidence: float  # 0 to 1
    risk_reward_ratio: float  # Expected risk/reward
    position_size_recommendation: str  # 'aggressive', 'moderate', 'conservative', 'minimal'
    entry_conditions: List[str]  # Conditions for entry
    exit_conditions: List[str]  # Conditions for exit
    stop_loss_level: Optional[float]  # Suggested stop loss (if applicable)
    take_profit_levels: List[float]  # Suggested take profit levels


@dataclass
class AdvancedSentimentAnalysis:
    """Comprehensive advanced sentiment analysis."""
    overall_score: float  # -1 to 1
    dimensions: SentimentDimensions
    temporal: TemporalSentiment
    regime: MarketRegime
    risk: RiskSentiment
    signals: TradingSignals
    anomalies: List[str]  # Detected sentiment anomalies
    correlations: Dict[str, float]  # Cross-asset correlations
    contrarian_indicators: List[str]  # Contrarian signals
    consensus_breakdown: Dict[str, float]  # Sentiment consensus analysis
    timestamp: datetime


class SentimentAnalyzer:
    """
    Advanced sentiment analysis engine with sophisticated algorithms
    for multi-dimensional sentiment scoring and trading signal generation.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.sentiment_history = deque(maxlen=1000)  # Store last 1000 sentiment readings
        self.regime_history = deque(maxlen=100)  # Store last 100 regime classifications
        self.signal_history = deque(maxlen=50)  # Store last 50 trading signals
        
        # Algorithm parameters
        self.momentum_window = 12  # Hours for momentum calculation
        self.volatility_window = 24  # Hours for volatility calculation
        self.mean_reversion_threshold = 0.7  # Threshold for mean reversion signals
        self.breakout_threshold = 0.8  # Threshold for breakout signals
        self.regime_transition_threshold = 0.6  # Threshold for regime transitions
        
        # Risk parameters
        self.risk_free_rate = 0.05  # Annual risk-free rate
        self.volatility_multiplier = 2.0  # Volatility multiplier for risk calculations
        
        logger.info("SentimentAnalyzer initialized")
    
    def analyze_sentiment(self, unified_sentiment: Any, market_data: Optional[Dict[str, Any]] = None) -> AdvancedSentimentAnalysis:
        """
        Perform comprehensive advanced sentiment analysis.
        
        Args:
            unified_sentiment: UnifiedSentiment object from sentiment_service
            market_data: Optional market data for correlation analysis
            
        Returns:
            AdvancedSentimentAnalysis object
        """
        try:
            # Store current sentiment in history
            self._update_sentiment_history(unified_sentiment)
            
            # Multi-dimensional sentiment analysis
            dimensions = self._analyze_sentiment_dimensions(unified_sentiment)
            
            # Temporal sentiment analysis
            temporal = self._analyze_temporal_sentiment(unified_sentiment)
            
            # Market regime classification
            regime = self._classify_market_regime(unified_sentiment, temporal)
            
            # Risk sentiment analysis
            risk = self._analyze_risk_sentiment(unified_sentiment, temporal, regime)
            
            # Generate trading signals
            signals = self._generate_trading_signals(unified_sentiment, dimensions, temporal, regime, risk)
            
            # Detect anomalies
            anomalies = self._detect_sentiment_anomalies(unified_sentiment, temporal)
            
            # Calculate correlations
            correlations = self._calculate_cross_asset_correlations(market_data) if market_data else {}
            
            # Generate contrarian indicators
            contrarian_indicators = self._generate_contrarian_indicators(unified_sentiment, dimensions, temporal)
            
            # Analyze consensus
            consensus_breakdown = self._analyze_consensus_breakdown(unified_sentiment)
            
            return AdvancedSentimentAnalysis(
                overall_score=unified_sentiment.overall_sentiment,
                dimensions=dimensions,
                temporal=temporal,
                regime=regime,
                risk=risk,
                signals=signals,
                anomalies=anomalies,
                correlations=correlations,
                contrarian_indicators=contrarian_indicators,
                consensus_breakdown=consensus_breakdown,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in advanced sentiment analysis: {e}")
            return self._empty_analysis()
    
    def _update_sentiment_history(self, unified_sentiment: Any):
        """Update sentiment history for temporal analysis."""
        self.sentiment_history.append({
            'timestamp': unified_sentiment.timestamp,
            'sentiment': unified_sentiment.overall_sentiment,
            'confidence': unified_sentiment.confidence,
            'fear_greed': unified_sentiment.fear_greed_index,
            'market_stress': unified_sentiment.market_stress,
            'source_agreement': unified_sentiment.source_agreement
        })
    
    def _analyze_sentiment_dimensions(self, unified_sentiment: Any) -> SentimentDimensions:
        """Analyze multi-dimensional sentiment components."""
        try:
            # Extract base sentiment
            base_sentiment = unified_sentiment.overall_sentiment
            
            # Bullish/Bearish dimension (primary sentiment)
            bullish_bearish = base_sentiment
            
            # Fear/Greed dimension (from fear_greed_index)
            fear_greed = (unified_sentiment.fear_greed_index - 50) / 50  # Convert 0-100 to -1,1
            
            # Confidence/Uncertainty dimension (inverse of market stress)
            confidence_uncertainty = 1 - (unified_sentiment.market_stress * 2)  # Convert 0-1 to 1,-1
            confidence_uncertainty = max(-1, min(1, confidence_uncertainty))
            
            # Euphoria/Depression dimension (extreme sentiment detection)
            euphoria_depression = self._calculate_euphoria_depression(unified_sentiment)
            
            # FOMO/FUD dimension (social media sentiment intensity)
            fomo_fud = self._calculate_fomo_fud(unified_sentiment)
            
            # Accumulation/Distribution dimension (smart money vs retail)
            accumulation_distribution = self._calculate_accumulation_distribution(unified_sentiment)
            
            return SentimentDimensions(
                bullish_bearish=bullish_bearish,
                fear_greed=fear_greed,
                confidence_uncertainty=confidence_uncertainty,
                euphoria_depression=euphoria_depression,
                fomo_fud=fomo_fud,
                accumulation_distribution=accumulation_distribution
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment dimensions: {e}")
            return SentimentDimensions(0, 0, 0, 0, 0, 0)
    
    def _analyze_temporal_sentiment(self, unified_sentiment: Any) -> TemporalSentiment:
        """Analyze temporal aspects of sentiment."""
        try:
            current_sentiment = unified_sentiment.overall_sentiment
            
            if len(self.sentiment_history) < 3:
                return TemporalSentiment(
                    current_sentiment=current_sentiment,
                    momentum=0.0,
                    acceleration=0.0,
                    volatility=0.0,
                    mean_reversion_signal=0.0,
                    breakout_signal=0.0,
                    cycle_position='early',
                    time_decay_factor=1.0
                )
            
            # Calculate momentum (rate of change)
            momentum = self._calculate_sentiment_momentum()
            
            # Calculate acceleration (rate of momentum change)
            acceleration = self._calculate_sentiment_acceleration()
            
            # Calculate volatility
            volatility = self._calculate_sentiment_volatility()
            
            # Generate mean reversion signal
            mean_reversion_signal = self._calculate_mean_reversion_signal(current_sentiment)
            
            # Generate breakout signal
            breakout_signal = self._calculate_breakout_signal(current_sentiment, momentum)
            
            # Determine cycle position
            cycle_position = self._determine_cycle_position(current_sentiment, momentum, volatility)
            
            # Calculate time decay factor
            time_decay_factor = self._calculate_time_decay_factor()
            
            return TemporalSentiment(
                current_sentiment=current_sentiment,
                momentum=momentum,
                acceleration=acceleration,
                volatility=volatility,
                mean_reversion_signal=mean_reversion_signal,
                breakout_signal=breakout_signal,
                cycle_position=cycle_position,
                time_decay_factor=time_decay_factor
            )
            
        except Exception as e:
            logger.error(f"Error in temporal sentiment analysis: {e}")
            return TemporalSentiment(0, 0, 0, 0, 0, 0, 'early', 1.0)
    
    def _classify_market_regime(self, unified_sentiment: Any, temporal: TemporalSentiment) -> MarketRegime:
        """Classify current market regime based on sentiment patterns."""
        try:
            sentiment = unified_sentiment.overall_sentiment
            momentum = temporal.momentum
            volatility = temporal.volatility
            stress = unified_sentiment.market_stress
            
            # Regime classification logic
            if sentiment > 0.5 and momentum > 0.1 and stress < 0.3:
                regime = 'euphoria'
                characteristics = {'sentiment': sentiment, 'momentum': momentum, 'low_stress': 1-stress}
            elif sentiment > 0.2 and momentum > 0.05 and volatility < 0.3:
                regime = 'bull_market'
                characteristics = {'bullish_sentiment': sentiment, 'positive_momentum': momentum}
            elif sentiment < -0.5 and stress > 0.7:
                regime = 'crisis'
                characteristics = {'negative_sentiment': abs(sentiment), 'high_stress': stress}
            elif sentiment < -0.2 and momentum < -0.05:
                regime = 'bear_market'
                characteristics = {'bearish_sentiment': abs(sentiment), 'negative_momentum': abs(momentum)}
            else:
                regime = 'sideways'
                characteristics = {'neutral_sentiment': 1-abs(sentiment), 'low_momentum': 1-abs(momentum)}
            
            # Calculate regime confidence
            confidence = self._calculate_regime_confidence(regime, sentiment, momentum, volatility, stress)
            
            # Calculate regime strength
            regime_strength = self._calculate_regime_strength(regime, characteristics)
            
            # Estimate regime duration
            regime_duration = self._estimate_regime_duration(regime)
            
            # Calculate transition probability
            transition_probability = self._calculate_transition_probability(regime, temporal)
            
            # Predict next likely regime
            next_likely_regime = self._predict_next_regime(regime, temporal)
            
            # Store regime in history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': confidence
            })
            
            return MarketRegime(
                regime=regime,
                confidence=confidence,
                regime_strength=regime_strength,
                regime_duration=regime_duration,
                transition_probability=transition_probability,
                next_likely_regime=next_likely_regime,
                regime_characteristics=characteristics
            )
            
        except Exception as e:
            logger.error(f"Error in market regime classification: {e}")
            return MarketRegime('sideways', 0.5, 0.5, 0, 0.5, 'sideways', {})
    
    def _analyze_risk_sentiment(self, unified_sentiment: Any, temporal: TemporalSentiment, regime: MarketRegime) -> RiskSentiment:
        """Analyze risk-based sentiment components."""
        try:
            # Risk appetite (inverse of market stress)
            risk_appetite = 1 - (unified_sentiment.market_stress * 2)
            risk_appetite = max(-1, min(1, risk_appetite))
            
            # Systemic risk (based on regime and stress)
            systemic_risk = unified_sentiment.market_stress
            if regime.regime in ['crisis', 'bear_market']:
                systemic_risk = min(systemic_risk + 0.3, 1.0)
            
            # Tail risk (extreme negative scenarios)
            tail_risk = self._calculate_tail_risk(unified_sentiment, temporal, regime)
            
            # Correlation risk (when sources disagree)
            correlation_risk = 1 - unified_sentiment.source_agreement
            
            # Liquidity risk (based on volatility and stress)
            liquidity_risk = min(temporal.volatility + unified_sentiment.market_stress, 1.0)
            
            # Volatility regime
            if temporal.volatility > 0.7:
                volatility_regime = 'extreme'
            elif temporal.volatility > 0.5:
                volatility_regime = 'high'
            elif temporal.volatility > 0.3:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'low'
            
            # Stress level
            if unified_sentiment.market_stress > 0.8:
                stress_level = 'crisis'
            elif unified_sentiment.market_stress > 0.6:
                stress_level = 'stressed'
            elif unified_sentiment.market_stress > 0.4:
                stress_level = 'elevated'
            else:
                stress_level = 'calm'
            
            return RiskSentiment(
                risk_appetite=risk_appetite,
                systemic_risk=systemic_risk,
                tail_risk=tail_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                volatility_regime=volatility_regime,
                stress_level=stress_level
            )
            
        except Exception as e:
            logger.error(f"Error in risk sentiment analysis: {e}")
            return RiskSentiment(0, 0.5, 0.5, 0.5, 0.5, 'medium', 'elevated')
    
    def _generate_trading_signals(self, unified_sentiment: Any, dimensions: SentimentDimensions, 
                                temporal: TemporalSentiment, regime: MarketRegime, risk: RiskSentiment) -> TradingSignals:
        """Generate sophisticated trading signals from sentiment analysis."""
        try:
            # Base signal from overall sentiment
            base_sentiment = unified_sentiment.overall_sentiment
            confidence = unified_sentiment.confidence
            
            # Adjust signal based on momentum and regime
            momentum_adjusted_sentiment = base_sentiment + (temporal.momentum * 0.3)
            
            # Generate primary signal
            if momentum_adjusted_sentiment > 0.4 and confidence > 0.7 and risk.risk_appetite > 0.2:
                primary_signal = 'strong_buy'
                signal_strength = min(abs(momentum_adjusted_sentiment) + confidence, 1.0)
            elif momentum_adjusted_sentiment > 0.1 and confidence > 0.5:
                primary_signal = 'buy'
                signal_strength = (abs(momentum_adjusted_sentiment) + confidence) / 2
            elif momentum_adjusted_sentiment < -0.4 and confidence > 0.7:
                primary_signal = 'strong_sell'
                signal_strength = min(abs(momentum_adjusted_sentiment) + confidence, 1.0)
            elif momentum_adjusted_sentiment < -0.1 and confidence > 0.5:
                primary_signal = 'sell'
                signal_strength = (abs(momentum_adjusted_sentiment) + confidence) / 2
            else:
                primary_signal = 'hold'
                signal_strength = confidence
            
            # Determine time horizon
            if temporal.momentum > 0.2 or temporal.volatility > 0.6:
                time_horizon = 'short'
            elif abs(temporal.momentum) < 0.1 and temporal.volatility < 0.3:
                time_horizon = 'long'
            else:
                time_horizon = 'medium'
            
            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                momentum_adjusted_sentiment, temporal.volatility, risk.systemic_risk
            )
            
            # Position sizing recommendation
            if signal_strength > 0.8 and risk.stress_level == 'calm' and confidence > 0.8:
                position_size = 'aggressive'
            elif signal_strength > 0.6 and risk.stress_level in ['calm', 'elevated']:
                position_size = 'moderate'
            elif signal_strength > 0.4:
                position_size = 'conservative'
            else:
                position_size = 'minimal'
            
            # Generate entry conditions
            entry_conditions = self._generate_entry_conditions(
                primary_signal, unified_sentiment, temporal, regime, risk
            )
            
            # Generate exit conditions
            exit_conditions = self._generate_exit_conditions(
                primary_signal, temporal, risk
            )
            
            # Calculate stop loss and take profit levels
            stop_loss_level = self._calculate_stop_loss(momentum_adjusted_sentiment, temporal.volatility)
            take_profit_levels = self._calculate_take_profit_levels(
                momentum_adjusted_sentiment, temporal.volatility, risk_reward_ratio
            )
            
            return TradingSignals(
                primary_signal=primary_signal,
                signal_strength=signal_strength,
                time_horizon=time_horizon,
                confidence=confidence,
                risk_reward_ratio=risk_reward_ratio,
                position_size_recommendation=position_size,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                stop_loss_level=stop_loss_level,
                take_profit_levels=take_profit_levels
            )
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return TradingSignals('hold', 0, 'medium', 0, 1, 'minimal', [], [], None, [])
    
    def _calculate_euphoria_depression(self, unified_sentiment: Any) -> float:
        """Calculate euphoria/depression dimension from extreme sentiment indicators."""
        # Check for extreme positive sentiment (euphoria)
        if unified_sentiment.fear_greed_index > 80 and unified_sentiment.overall_sentiment > 0.4:
            return min(1.0, (unified_sentiment.fear_greed_index - 50) / 30)
        
        # Check for extreme negative sentiment (depression)
        if unified_sentiment.fear_greed_index < 20 and unified_sentiment.overall_sentiment < -0.4:
            return max(-1.0, (unified_sentiment.fear_greed_index - 50) / 30)
        
        return 0.0
    
    def _calculate_fomo_fud(self, unified_sentiment: Any) -> float:
        """Calculate FOMO/FUD dimension from social sentiment intensity."""
        social_sentiment = unified_sentiment.social_sentiment.sentiment
        
        # FOMO indicators (positive social sentiment with high engagement)
        if social_sentiment > 0.3:
            return min(1.0, social_sentiment * 1.5)
        
        # FUD indicators (negative social sentiment with high spread)
        if social_sentiment < -0.3:
            return max(-1.0, social_sentiment * 1.5)
        
        return social_sentiment
    
    def _calculate_accumulation_distribution(self, unified_sentiment: Any) -> float:
        """Calculate smart money vs retail sentiment."""
        # Simplified: use macro sentiment as proxy for smart money
        macro_sentiment = unified_sentiment.macro_sentiment.sentiment
        social_sentiment = unified_sentiment.social_sentiment.sentiment
        
        # When macro and social disagree, assume smart money (macro) is right
        if abs(macro_sentiment - social_sentiment) > 0.3:
            return macro_sentiment  # Smart money accumulation/distribution
        
        return 0.0  # No clear accumulation/distribution signal
    
    def _calculate_sentiment_momentum(self) -> float:
        """Calculate sentiment momentum from recent history."""
        if len(self.sentiment_history) < 6:
            return 0.0
        
        recent_sentiments = [entry['sentiment'] for entry in list(self.sentiment_history)[-6:]]
        
        # Calculate linear trend
        x = np.arange(len(recent_sentiments))
        momentum = np.polyfit(x, recent_sentiments, 1)[0]  # Slope of linear trend
        
        return max(-1, min(1, momentum * 10))  # Scale and clamp
    
    def _calculate_sentiment_acceleration(self) -> float:
        """Calculate sentiment acceleration (rate of momentum change)."""
        if len(self.sentiment_history) < 12:
            return 0.0
        
        # Calculate momentum for two different periods
        recent_momentum = self._calculate_momentum_for_period(-6, None)
        older_momentum = self._calculate_momentum_for_period(-12, -6)
        
        acceleration = recent_momentum - older_momentum
        return max(-1, min(1, acceleration * 20))  # Scale and clamp
    
    def _calculate_momentum_for_period(self, start: int, end: Optional[int]) -> float:
        """Calculate momentum for a specific period."""
        sentiments = [entry['sentiment'] for entry in list(self.sentiment_history)[start:end]]
        if len(sentiments) < 3:
            return 0.0
        
        x = np.arange(len(sentiments))
        return np.polyfit(x, sentiments, 1)[0]
    
    def _calculate_sentiment_volatility(self) -> float:
        """Calculate sentiment volatility."""
        if len(self.sentiment_history) < 5:
            return 0.0
        
        recent_sentiments = [entry['sentiment'] for entry in list(self.sentiment_history)[-24:]]
        volatility = np.std(recent_sentiments)
        
        return min(1.0, volatility * 2)  # Scale to 0-1 range
    
    def _calculate_mean_reversion_signal(self, current_sentiment: float) -> float:
        """Calculate mean reversion signal strength."""
        if len(self.sentiment_history) < 10:
            return 0.0
        
        # Calculate long-term mean
        long_term_sentiments = [entry['sentiment'] for entry in self.sentiment_history]
        long_term_mean = np.mean(long_term_sentiments)
        
        # Distance from mean
        distance_from_mean = abs(current_sentiment - long_term_mean)
        
        # Mean reversion signal increases with distance from mean
        if distance_from_mean > self.mean_reversion_threshold:
            # Return signal pointing back to mean
            return -np.sign(current_sentiment - long_term_mean) * min(1.0, distance_from_mean / 0.5)
        
        return 0.0
    
    def _calculate_breakout_signal(self, current_sentiment: float, momentum: float) -> float:
        """Calculate breakout signal strength."""
        # Breakout when sentiment and momentum align and are strong
        if abs(current_sentiment) > self.breakout_threshold and abs(momentum) > 0.3:
            if np.sign(current_sentiment) == np.sign(momentum):
                return np.sign(current_sentiment) * min(1.0, abs(current_sentiment) + abs(momentum))
        
        return 0.0
    
    def _determine_cycle_position(self, sentiment: float, momentum: float, volatility: float) -> str:
        """Determine position in sentiment cycle."""
        if abs(sentiment) < 0.2 and abs(momentum) < 0.1:
            return 'early'  # Early in cycle, low sentiment and momentum
        elif abs(momentum) > 0.2:
            return 'mid'  # Middle of cycle, high momentum
        elif abs(sentiment) > 0.6 or volatility > 0.7:
            return 'late'  # Late in cycle, extreme sentiment or high volatility
        else:
            return 'mid'
    
    def _calculate_time_decay_factor(self) -> float:
        """Calculate how quickly sentiment signals decay over time."""
        if not self.sentiment_history:
            return 1.0
        
        # More recent and volatile sentiment decays faster
        latest_entry = self.sentiment_history[-1]
        time_since_update = (datetime.now() - latest_entry['timestamp']).total_seconds() / 3600  # Hours
        
        # Decay factor: 0.95^hours (5% decay per hour)
        decay_factor = 0.95 ** time_since_update
        return max(0.1, decay_factor)  # Minimum 10% weight
    
    def _calculate_regime_confidence(self, regime: str, sentiment: float, momentum: float, volatility: float, stress: float) -> float:
        """Calculate confidence in regime classification."""
        # Base confidence from clear signals
        confidence = 0.5
        
        # Boost confidence for clear regimes
        if regime == 'euphoria' and sentiment > 0.6 and stress < 0.2:
            confidence += 0.4
        elif regime == 'crisis' and sentiment < -0.6 and stress > 0.8:
            confidence += 0.4
        elif regime in ['bull_market', 'bear_market'] and abs(momentum) > 0.2:
            confidence += 0.3
        
        # Reduce confidence for mixed signals
        if abs(sentiment) < 0.1 and abs(momentum) < 0.05:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_regime_strength(self, regime: str, characteristics: Dict[str, float]) -> float:
        """Calculate how strong the current regime is."""
        if not characteristics:
            return 0.5
        
        # Average of regime characteristics
        strength = np.mean(list(characteristics.values()))
        return max(0.0, min(1.0, strength))
    
    def _estimate_regime_duration(self, regime: str) -> int:
        """Estimate how long the current regime has been active."""
        if len(self.regime_history) < 2:
            return 1
        
        # Count consecutive periods with same regime
        duration = 1
        for entry in reversed(list(self.regime_history)[:-1]):
            if entry['regime'] == regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_transition_probability(self, regime: str, temporal: TemporalSentiment) -> float:
        """Calculate probability of regime transition."""
        base_probability = 0.1  # Base 10% chance of transition
        
        # Increase probability with high volatility or momentum changes
        if temporal.volatility > 0.6:
            base_probability += 0.3
        
        if abs(temporal.acceleration) > 0.3:
            base_probability += 0.2
        
        # Regime-specific adjustments
        if regime in ['euphoria', 'crisis']:
            base_probability += 0.2  # Extreme regimes are less stable
        
        return min(1.0, base_probability)
    
    def _predict_next_regime(self, current_regime: str, temporal: TemporalSentiment) -> str:
        """Predict the most likely next regime."""
        # Regime transition logic
        if current_regime == 'euphoria':
            if temporal.momentum < 0:
                return 'sideways'
            else:
                return 'bull_market'
        elif current_regime == 'crisis':
            if temporal.momentum > 0:
                return 'bear_market'
            else:
                return 'sideways'
        elif current_regime == 'bull_market':
            if temporal.acceleration < -0.2:
                return 'sideways'
            elif temporal.momentum > 0.3:
                return 'euphoria'
            else:
                return 'bull_market'
        elif current_regime == 'bear_market':
            if temporal.acceleration > 0.2:
                return 'sideways'
            elif temporal.momentum < -0.3:
                return 'crisis'
            else:
                return 'bear_market'
        else:  # sideways
            if temporal.momentum > 0.2:
                return 'bull_market'
            elif temporal.momentum < -0.2:
                return 'bear_market'
            else:
                return 'sideways'
    
    def _calculate_tail_risk(self, unified_sentiment: Any, temporal: TemporalSentiment, regime: MarketRegime) -> float:
        """Calculate tail risk probability."""
        tail_risk = 0.1  # Base tail risk
        
        # Increase with extreme sentiment
        if abs(unified_sentiment.overall_sentiment) > 0.7:
            tail_risk += 0.3
        
        # Increase with high volatility
        if temporal.volatility > 0.7:
            tail_risk += 0.2
        
        # Increase in crisis or euphoria regimes
        if regime.regime in ['crisis', 'euphoria']:
            tail_risk += 0.3
        
        # Increase with low source agreement
        if unified_sentiment.source_agreement < 0.4:
            tail_risk += 0.2
        
        return min(1.0, tail_risk)
    
    def _calculate_risk_reward_ratio(self, sentiment: float, volatility: float, systemic_risk: float) -> float:
        """Calculate expected risk-reward ratio."""
        # Expected return based on sentiment
        expected_return = abs(sentiment) * 0.1  # 10% max expected return
        
        # Risk based on volatility and systemic risk
        risk = volatility * 0.05 + systemic_risk * 0.1  # Combined risk measure
        
        if risk == 0:
            return 10.0  # Very high ratio when risk is near zero
        
        ratio = expected_return / risk
        return min(10.0, max(0.1, ratio))  # Clamp between 0.1 and 10
    
    def _generate_entry_conditions(self, signal: str, unified_sentiment: Any, temporal: TemporalSentiment, 
                                 regime: MarketRegime, risk: RiskSentiment) -> List[str]:
        """Generate specific entry conditions for the trading signal."""
        conditions = []
        
        if signal in ['buy', 'strong_buy']:
            conditions.append(f"Positive sentiment confirmed ({unified_sentiment.overall_sentiment:.2f})")
            if temporal.momentum > 0.1:
                conditions.append("Positive momentum trend")
            if risk.risk_appetite > 0.2:
                conditions.append("Risk-on environment")
            if regime.regime in ['bull_market', 'euphoria']:
                conditions.append(f"Bullish regime ({regime.regime})")
        
        elif signal in ['sell', 'strong_sell']:
            conditions.append(f"Negative sentiment confirmed ({unified_sentiment.overall_sentiment:.2f})")
            if temporal.momentum < -0.1:
                conditions.append("Negative momentum trend")
            if risk.stress_level in ['stressed', 'crisis']:
                conditions.append(f"High stress environment ({risk.stress_level})")
            if regime.regime in ['bear_market', 'crisis']:
                conditions.append(f"Bearish regime ({regime.regime})")
        
        # Common conditions
        conditions.append(f"Confidence above threshold ({unified_sentiment.confidence:.2f})")
        
        if unified_sentiment.source_agreement > 0.6:
            conditions.append("High source agreement")
        
        return conditions
    
    def _generate_exit_conditions(self, signal: str, temporal: TemporalSentiment, risk: RiskSentiment) -> List[str]:
        """Generate specific exit conditions for the trading signal."""
        conditions = []
        
        # Common exit conditions
        conditions.append("Sentiment reversal signal")
        conditions.append("Stop loss hit")
        conditions.append("Take profit target reached")
        
        # Signal-specific conditions
        if signal in ['buy', 'strong_buy']:
            conditions.append("Negative momentum divergence")
            if risk.stress_level in ['stressed', 'crisis']:
                conditions.append("Risk-off environment emerges")
        
        elif signal in ['sell', 'strong_sell']:
            conditions.append("Positive momentum divergence")
            conditions.append("Stress levels normalize")
        
        # Time-based exits
        if temporal.volatility > 0.6:
            conditions.append("High volatility exit signal")
        
        return conditions
    
    def _calculate_stop_loss(self, sentiment: float, volatility: float) -> Optional[float]:
        """Calculate suggested stop loss level."""
        if abs(sentiment) < 0.1:
            return None  # No clear directional bias
        
        # Stop loss based on volatility (larger stops for higher volatility)
        base_stop = 0.05  # 5% base stop
        volatility_adjustment = volatility * 0.1  # Up to 10% additional for high volatility
        
        total_stop = base_stop + volatility_adjustment
        return min(0.2, total_stop)  # Cap at 20%
    
    def _calculate_take_profit_levels(self, sentiment: float, volatility: float, risk_reward_ratio: float) -> List[float]:
        """Calculate suggested take profit levels."""
        if abs(sentiment) < 0.1:
            return []
        
        base_target = abs(sentiment) * 0.15  # Base target from sentiment strength
        
        # Multiple take profit levels
        levels = [
            base_target * 0.5,  # Conservative target
            base_target,        # Base target
            base_target * 1.5   # Aggressive target
        ]
        
        # Adjust for risk-reward ratio
        levels = [level * min(risk_reward_ratio / 2, 2.0) for level in levels]
        
        return [min(0.5, level) for level in levels if level > 0.02]  # Filter out tiny targets
    
    def _detect_sentiment_anomalies(self, unified_sentiment: Any, temporal: TemporalSentiment) -> List[str]:
        """Detect unusual patterns in sentiment data."""
        anomalies = []
        
        # Extreme sentiment readings
        if abs(unified_sentiment.overall_sentiment) > 0.8:
            anomalies.append(f"Extreme sentiment reading: {unified_sentiment.overall_sentiment:.2f}")
        
        # High volatility
        if temporal.volatility > 0.8:
            anomalies.append(f"Extremely high sentiment volatility: {temporal.volatility:.2f}")
        
        # Sentiment-momentum divergence
        if abs(unified_sentiment.overall_sentiment - temporal.momentum) > 0.5:
            anomalies.append("Sentiment-momentum divergence detected")
        
        # Low source agreement
        if unified_sentiment.source_agreement < 0.3:
            anomalies.append(f"Very low source agreement: {unified_sentiment.source_agreement:.2f}")
        
        # Rapid sentiment changes
        if abs(temporal.acceleration) > 0.5:
            anomalies.append("Rapid sentiment acceleration detected")
        
        return anomalies
    
    def _calculate_cross_asset_correlations(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sentiment correlations with other assets."""
        # Placeholder - would need actual market data
        correlations = {
            'btc_correlation': 0.6,
            'eth_correlation': 0.7,
            'gold_correlation': -0.2,
            'spy_correlation': 0.4,
            'vix_correlation': -0.5
        }
        return correlations
    
    def _generate_contrarian_indicators(self, unified_sentiment: Any, dimensions: SentimentDimensions, 
                                      temporal: TemporalSentiment) -> List[str]:
        """Generate contrarian trading indicators."""
        indicators = []
        
        # Extreme sentiment contrarian signals
        if unified_sentiment.fear_greed_index > 85:
            indicators.append("Extreme greed - contrarian sell signal")
        elif unified_sentiment.fear_greed_index < 15:
            indicators.append("Extreme fear - contrarian buy signal")
        
        # Euphoria/depression contrarian signals
        if dimensions.euphoria_depression > 0.7:
            indicators.append("Market euphoria - potential top signal")
        elif dimensions.euphoria_depression < -0.7:
            indicators.append("Market depression - potential bottom signal")
        
        # Social sentiment contrarian signals
        if dimensions.fomo_fud > 0.8:
            indicators.append("Extreme FOMO - fade the crowd signal")
        elif dimensions.fomo_fud < -0.8:
            indicators.append("Extreme FUD - contrarian buying opportunity")
        
        # Momentum exhaustion
        if abs(temporal.momentum) > 0.6 and abs(temporal.acceleration) < 0.1:
            indicators.append("Momentum exhaustion - potential reversal")
        
        return indicators
    
    def _analyze_consensus_breakdown(self, unified_sentiment: Any) -> Dict[str, float]:
        """Analyze sentiment consensus across different sources."""
        consensus = {}
        
        # Source sentiment breakdown
        consensus['news_weight'] = unified_sentiment.news_sentiment.weight
        consensus['reddit_weight'] = unified_sentiment.reddit_sentiment.weight
        consensus['twitter_weight'] = unified_sentiment.twitter_sentiment.weight
        consensus['macro_weight'] = unified_sentiment.macro_sentiment.weight
        
        # Agreement levels
        consensus['source_agreement'] = unified_sentiment.source_agreement
        consensus['confidence_weighted_agreement'] = self._calculate_confidence_weighted_agreement(unified_sentiment)
        
        # Divergence measures
        consensus['max_divergence'] = self._calculate_max_divergence(unified_sentiment)
        consensus['consensus_strength'] = self._calculate_consensus_strength(unified_sentiment)
        
        return consensus
    
    def _calculate_confidence_weighted_agreement(self, unified_sentiment: Any) -> float:
        """Calculate agreement weighted by confidence levels."""
        sentiments = [
            (unified_sentiment.news_sentiment.sentiment, unified_sentiment.news_sentiment.confidence),
            (unified_sentiment.reddit_sentiment.sentiment, unified_sentiment.reddit_sentiment.confidence),
            (unified_sentiment.twitter_sentiment.sentiment, unified_sentiment.twitter_sentiment.confidence),
            (unified_sentiment.macro_sentiment.sentiment, unified_sentiment.macro_sentiment.confidence)
        ]
        
        # Remove zero-confidence sources
        valid_sentiments = [(s, c) for s, c in sentiments if c > 0]
        
        if len(valid_sentiments) < 2:
            return 1.0
        
        # Calculate weighted average
        total_weight = sum(c for _, c in valid_sentiments)
        weighted_avg = sum(s * c for s, c in valid_sentiments) / total_weight
        
        # Calculate agreement as inverse of weighted variance
        weighted_variance = sum(c * (s - weighted_avg) ** 2 for s, c in valid_sentiments) / total_weight
        agreement = max(0, 1 - weighted_variance)
        
        return agreement
    
    def _calculate_max_divergence(self, unified_sentiment: Any) -> float:
        """Calculate maximum divergence between any two sources."""
        sentiments = [
            unified_sentiment.news_sentiment.sentiment,
            unified_sentiment.reddit_sentiment.sentiment,
            unified_sentiment.twitter_sentiment.sentiment,
            unified_sentiment.macro_sentiment.sentiment
        ]
        
        # Filter out zero sentiments (missing data)
        valid_sentiments = [s for s in sentiments if s != 0]
        
        if len(valid_sentiments) < 2:
            return 0.0
        
        max_divergence = 0.0
        for i in range(len(valid_sentiments)):
            for j in range(i + 1, len(valid_sentiments)):
                divergence = abs(valid_sentiments[i] - valid_sentiments[j])
                max_divergence = max(max_divergence, divergence)
        
        return max_divergence
    
    def _calculate_consensus_strength(self, unified_sentiment: Any) -> float:
        """Calculate overall consensus strength."""
        agreement = unified_sentiment.source_agreement
        confidence = unified_sentiment.confidence
        
        # Consensus strength combines agreement and confidence
        consensus_strength = (agreement * 0.6 + confidence * 0.4)
        
        return consensus_strength
    
    def _empty_analysis(self) -> AdvancedSentimentAnalysis:
        """Return empty analysis object."""
        return AdvancedSentimentAnalysis(
            overall_score=0.0,
            dimensions=SentimentDimensions(0, 0, 0, 0, 0, 0),
            temporal=TemporalSentiment(0, 0, 0, 0, 0, 0, 'early', 1.0),
            regime=MarketRegime('sideways', 0.5, 0.5, 0, 0.5, 'sideways', {}),
            risk=RiskSentiment(0, 0.5, 0.5, 0.5, 0.5, 'medium', 'elevated'),
            signals=TradingSignals('hold', 0, 'medium', 0, 1, 'minimal', [], [], None, []),
            anomalies=[],
            correlations={},
            contrarian_indicators=[],
            consensus_breakdown={},
            timestamp=datetime.now()
        )


# Factory function
def create_sentiment_analyzer() -> SentimentAnalyzer:
    """Create and return SentimentAnalyzer instance."""
    return SentimentAnalyzer()