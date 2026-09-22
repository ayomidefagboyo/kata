"""
Ryu Agent - Balanced Trading Strategy

Ryu is the balanced agent that focuses on:
- Moderate risk-reward ratios
- Technical analysis-based decisions
- Diversified portfolio approach
- Adaptive strategy based on market conditions
"""

import logging
from typing import Dict, Any, List
from decimal import Decimal
import statistics

from kata.agents.base_agent import BaseAgent, MarketData, TradingSignal, SignalType, RiskLevel

logger = logging.getLogger(__name__)


class RyuAgent(BaseAgent):
    """
    Balanced trading agent focused on technical analysis and adaptive strategies.
    
    Ryu's trading philosophy:
    - Balance risk and reward
    - Use technical indicators for entry/exit
    - Adapt to market conditions
    - Maintain diversified portfolio
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        """Initialize Ryu agent with balanced defaults."""
        
        # Set balanced risk parameters
        balanced_risk_params = {
            'max_position_size_percent': 5.0,  # Standard positions
            'stop_loss_percent': 3.0,  # Moderate stop losses
            'take_profit_percent': 15.0,  # Balanced profit targets
            'max_daily_loss_percent': 8.0,  # Moderate daily loss limit
            'max_portfolio_concentration': 25.0,  # Balanced diversification
            'min_confidence_threshold': 0.65,  # Moderate confidence required
            'max_trades_per_day': 6,  # Moderate trading frequency
            'cooldown_minutes': 10  # Moderate cooldown
        }
        
        # Merge with user config
        config_risk_params = config.get('risk_params', {})
        merged_risk_params = {**balanced_risk_params, **config_risk_params}
        config['risk_params'] = merged_risk_params
        
        super().__init__(user_id, "ryu", config)
        
        # Ryu-specific parameters
        self.technical_indicators = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'trend_threshold': 0.02,  # 2% for trend confirmation
            'volume_spike_threshold': 1.5  # 50% above average
        }
        
        self.market_cap_tiers = {
            'large_cap': Decimal('10000000000'),  # 10B+
            'mid_cap': Decimal('1000000000'),     # 1B+
            'small_cap': Decimal('100000000')     # 100M+
        }
        
        self.analysis_weights = {
            'technical': 0.4,
            'momentum': 0.3,
            'volume': 0.2,
            'market_structure': 0.1
        }
        
        logger.info(f"Ryu agent initialized for user {user_id} with balanced settings")
    
    async def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Analyze market conditions using technical analysis and momentum indicators.
        
        Ryu's analysis includes:
        - Technical indicator analysis
        - Momentum and trend detection
        - Volume analysis
        - Market structure assessment
        """
        try:
            analysis = {
                'timestamp': market_data.timestamp.isoformat(),
                'market_sentiment': 'neutral',
                'trend_strength': 0.0,
                'momentum_score': 0.0,
                'risk_assessment': RiskLevel.MEDIUM,
                'technical_analysis': {},
                'trading_opportunities': []
            }
            
            # Overall market trend analysis
            trend_analysis = self._analyze_market_trend(market_data)
            analysis['trend_strength'] = trend_analysis['strength']
            analysis['market_sentiment'] = trend_analysis['direction']
            
            # Volume analysis
            volume_analysis = self._analyze_volume_patterns(market_data)
            analysis['volume_patterns'] = volume_analysis
            
            # Technical analysis for each requested token. Token Analysis accepts a
            # broad market universe, so silently dropping assets that are not in a
            # legacy Base allow-list (including BTC and SOL) produces incomplete
            # analysis and must not happen here.
            technical_scores = {}
            for token in market_data.token_prices.keys():
                tech_score = self._calculate_token_technical_score(token, market_data)
                technical_scores[token] = tech_score

                # Identify sufficiently directional long and short opportunities.
                if tech_score['confidence'] >= self.risk_params.min_confidence_threshold:
                    opportunity = {
                        'token': token,
                        'score': tech_score['overall_score'],
                        'signal_type': tech_score['recommended_action'],
                        'reasoning': tech_score['reasoning'],
                        'confidence': tech_score['confidence']
                    }
                    analysis['trading_opportunities'].append(opportunity)
            
            analysis['technical_analysis'] = technical_scores
            
            # Calculate overall momentum score
            momentum_scores = [score['momentum'] for score in technical_scores.values()]
            analysis['momentum_score'] = statistics.mean(momentum_scores) if momentum_scores else 0.5
            
            # Risk assessment based on market conditions
            analysis['risk_assessment'] = self._assess_market_risk(
                trend_analysis, volume_analysis, analysis['momentum_score']
            )
            
            # Sort opportunities by score
            analysis['trading_opportunities'].sort(
                key=lambda x: x['score'], reverse=True
            )
            
            # Calculate agent-specific scores for token analysis
            technical_score = self._calculate_technical_score(market_data)
            fundamental_score = self._calculate_fundamental_score(market_data)
            
            # Add scores to analysis
            analysis['technical_score'] = technical_score
            analysis['fundamental_score'] = fundamental_score
            analysis['agent_strategy'] = 'balanced_technical'
            
            logger.info(f"Ryu market analysis completed: {analysis['market_sentiment']} trend, "
                       f"momentum {analysis['momentum_score']:.2f}, "
                       f"{len(analysis['trading_opportunities'])} opportunities, "
                       f"technical_score: {technical_score:.3f}, fundamental_score: {fundamental_score:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Ryu market analysis: {e}")
            raise  # Let the error propagate instead of returning fallback data
    
    async def generate_signals(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Generate balanced trading signals based on technical analysis.
        
        Ryu's signal generation combines:
        - Technical indicator signals
        - Momentum-based entries
        - Volume confirmation
        - Risk-adjusted position sizing
        """
        signals = []
        
        try:
            # Get market analysis
            analysis = await self.analyze_market(market_data)
            
            if analysis.get('error'):
                logger.error(f"Cannot generate signals due to analysis error: {analysis['error']}")
                return signals
            
            # Get current portfolio positions
            portfolio_positions = await self._get_portfolio_positions()
            current_holdings = {pos.token_symbol: pos for pos in portfolio_positions}
            
            # Process trading opportunities
            opportunities = analysis.get('trading_opportunities', [])
            for opportunity in opportunities[:4]:  # Limit to top 4 opportunities
                token_symbol = opportunity['token']
                signal_type = opportunity['signal_type']
                
                # Generate buy signals
                if signal_type == 'buy':
                    # Check if we already have a position
                    if token_symbol in current_holdings:
                        current_allocation = current_holdings[token_symbol].allocation_percent
                        if current_allocation >= 20:  # Max 20% per token for Ryu
                            continue
                    
                    # Create buy signal
                    signal = TradingSignal(
                        signal_type=SignalType.BUY,
                        token_symbol=token_symbol,
                        confidence=opportunity['confidence'],
                        reasoning=opportunity['reasoning'],
                        risk_level=self._determine_signal_risk_level(opportunity['score']),
                        metadata={
                            'technical_score': opportunity['score'],
                            'analysis_type': 'technical_momentum',
                            'market_sentiment': analysis['market_sentiment'],
                            'trend_strength': analysis['trend_strength']
                        }
                    )
                    signals.append(signal)
                
                # Generate sell signals for existing positions
                elif signal_type == 'sell' and token_symbol in current_holdings:
                    signal = TradingSignal(
                        signal_type=SignalType.SELL,
                        token_symbol=token_symbol,
                        confidence=opportunity['confidence'],
                        reasoning=f"Technical exit signal: {opportunity['reasoning']}",
                        risk_level=RiskLevel.MEDIUM,
                        metadata={
                            'technical_score': opportunity['score'],
                            'current_pnl': current_holdings[token_symbol].pnl_percent
                        }
                    )
                    signals.append(signal)
            
            # Generate stop-loss and take-profit signals for existing positions
            for position in portfolio_positions:
                exit_signals = await self._generate_technical_exit_signals(
                    position, market_data, analysis
                )
                signals.extend(exit_signals)
            
            # Filter signals based on market conditions
            filtered_signals = self._filter_signals_by_market_conditions(
                signals, analysis
            )
            
            logger.info(f"Ryu generated {len(filtered_signals)} balanced trading signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating Ryu signals: {e}")
            return []
    
    def _analyze_market_trend(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze overall market trend and strength."""
        try:
            # Calculate average price change
            price_changes = list(market_data.price_changes.values())
            avg_change = statistics.mean([float(change) for change in price_changes])
            
            # Determine trend direction
            # price_changes are percentage points (for example 2.5 means +2.5%),
            # while trend_threshold is stored as a fraction. Normalize once here.
            trend_threshold_pct = self.technical_indicators['trend_threshold'] * 100
            if avg_change > trend_threshold_pct:
                direction = 'bullish'
                strength = min(1.0, avg_change / 10.0)
            elif avg_change < -trend_threshold_pct:
                direction = 'bearish'
                strength = min(1.0, abs(avg_change) / 10.0)
            else:
                direction = 'neutral'
                strength = 0.5
            
            return {
                'direction': direction,
                'strength': strength,
                'avg_change': avg_change,
                'trend_confirmation': self._confirm_trend(market_data, direction)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trend: {e}")
            return {'direction': 'neutral', 'strength': 0.5}
    
    def _analyze_volume_patterns(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze volume patterns across tokens."""
        try:
            volumes = list(market_data.volume_24h.values())
            avg_volume = statistics.mean([float(vol) for vol in volumes])
            
            volume_spikes = []
            for token, volume in market_data.volume_24h.items():
                if float(volume) > avg_volume * self.technical_indicators['volume_spike_threshold']:
                    volume_spikes.append({
                        'token': token,
                        'volume': float(volume),
                        'spike_ratio': float(volume) / avg_volume
                    })
            
            return {
                'avg_volume': avg_volume,
                'volume_spikes': volume_spikes,
                'high_volume_tokens': [spike['token'] for spike in volume_spikes]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {e}")
            return {'avg_volume': 0, 'volume_spikes': []}
    
    def _calculate_token_technical_score(self, token: str, market_data: MarketData) -> Dict[str, Any]:
        """Calculate technical analysis score for a token."""
        try:
            score_components = self._calculate_directional_components(token, market_data)
            overall_score = score_components['overall']
            price_change = float(market_data.price_changes.get(token, Decimal('0')))
            
            # Determine recommended action
            if overall_score >= 0.60:
                recommended_action = 'buy'
                confidence = min(0.90, 0.5 + abs(overall_score - 0.5))
                reasoning = (
                    f"Bullish convergence: momentum {score_components['momentum']:.2f}, "
                    f"trend {score_components['trend']:.2f}, RSI {score_components['rsi']:.2f}, "
                    f"MACD {score_components['macd']:.2f}"
                )
            elif overall_score <= 0.40:
                recommended_action = 'sell'
                confidence = min(0.90, 0.5 + abs(overall_score - 0.5))
                reasoning = (
                    f"Bearish convergence: momentum {score_components['momentum']:.2f}, "
                    f"trend {score_components['trend']:.2f}, RSI {score_components['rsi']:.2f}, "
                    f"MACD {score_components['macd']:.2f}"
                )
            else:
                recommended_action = 'hold'
                confidence = max(0.35, 0.6 - abs(overall_score - 0.5))
                reasoning = f"Mixed technical signals around neutral; 24h move is {price_change:+.2f}%"
            
            return {
                'overall_score': overall_score,
                'components': score_components,
                'recommended_action': recommended_action,
                'confidence': confidence,
                'reasoning': reasoning,
                'momentum': score_components['momentum']
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical score for {token}: {e}")
            return {
                'overall_score': 0.5,
                'recommended_action': 'hold',
                'confidence': 0.5,
                'reasoning': 'Error in analysis',
                'momentum': 0.5
            }

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _calculate_directional_components(self, token: str, market_data: MarketData) -> Dict[str, float]:
        """Build a directional 0..1 score from the real indicators already fetched."""
        indicators = market_data.trend_indicators or {}
        price_change_pct = float(market_data.price_changes.get(token, Decimal('0')))
        rsi = float(indicators.get('rsi') or 50.0)
        ema_20 = float(indicators.get('ema_20') or 0.0)
        ema_50 = float(indicators.get('ema_50') or 0.0)
        macd_data = indicators.get('macd') or {}
        bollinger = indicators.get('bollinger_bands') or {}

        momentum_score = self._clamp_score(0.5 + price_change_pct / 20.0)
        rsi_score = self._clamp_score(1.0 - rsi / 100.0)

        if ema_20 > 0 and ema_50 > 0:
            ema_spread = (ema_20 / ema_50) - 1.0
            trend_score = self._clamp_score(0.5 + ema_spread * 10.0)
        else:
            trend_score = momentum_score

        if isinstance(macd_data, dict):
            macd_line = float(macd_data.get('macd') or 0.0)
            signal_line = float(macd_data.get('signal') or 0.0)
            histogram = float(macd_data.get('histogram') or (macd_line - signal_line))
        else:
            histogram = float(macd_data or 0.0)
        macd_score = 0.65 if histogram > 0 else 0.35 if histogram < 0 else 0.5

        current_price = float(market_data.token_prices.get(token, Decimal('0')))
        bb_upper = float(bollinger.get('upper') or 0.0) if isinstance(bollinger, dict) else 0.0
        bb_lower = float(bollinger.get('lower') or 0.0) if isinstance(bollinger, dict) else 0.0
        if current_price > 0 and bb_upper > bb_lower:
            bb_position = self._clamp_score((current_price - bb_lower) / (bb_upper - bb_lower))
            bollinger_score = self._clamp_score(1.0 - bb_position)
        else:
            bollinger_score = 0.5

        directional_core = (
            momentum_score * 0.30
            + trend_score * 0.25
            + rsi_score * 0.20
            + macd_score * 0.15
            + bollinger_score * 0.10
        )

        # Liquidity/volume confirms conviction without manufacturing direction.
        volume = float(market_data.volume_24h.get(token, Decimal('0')))
        volume_confirmation = 1.0 if volume > 0 else 0.6
        overall = 0.5 + (directional_core - 0.5) * volume_confirmation
        return {
            'overall': self._clamp_score(overall),
            'momentum': momentum_score,
            'trend': trend_score,
            'rsi': rsi_score,
            'macd': macd_score,
            'bollinger': bollinger_score,
            'volume_confirmation': volume_confirmation,
        }
    
    def _confirm_trend(self, market_data: MarketData, direction: str) -> bool:
        """Confirm trend using multiple indicators."""
        try:
            # Count tokens following the trend
            following_trend = 0
            total_tokens = 0
            
            for token, change in market_data.price_changes.items():
                total_tokens += 1
                price_change = float(change)
                
                if direction == 'bullish' and price_change > 0:
                    following_trend += 1
                elif direction == 'bearish' and price_change < 0:
                    following_trend += 1
                elif direction == 'neutral' and abs(price_change) < 1:
                    following_trend += 1
            
            # Need at least 60% confirmation
            confirmation_ratio = following_trend / total_tokens if total_tokens > 0 else 0
            return confirmation_ratio >= 0.6
            
        except Exception as e:
            logger.error(f"Error confirming trend: {e}")
            return False
    
    def _assess_market_risk(
        self, 
        trend_analysis: Dict[str, Any], 
        volume_analysis: Dict[str, Any], 
        momentum_score: float
    ) -> RiskLevel:
        """Assess overall market risk level."""
        try:
            risk_factors = []
            
            # Trend risk
            if not trend_analysis.get('trend_confirmation', False):
                risk_factors.append('unconfirmed_trend')
            
            if trend_analysis.get('strength', 0) < 0.3:
                risk_factors.append('weak_trend')
            
            # Volume risk
            if len(volume_analysis.get('volume_spikes', [])) > 3:
                risk_factors.append('excessive_volume_spikes')
            
            # Momentum risk
            if momentum_score < 0.3 or momentum_score > 0.8:
                risk_factors.append('extreme_momentum')
            
            # Determine risk level
            if len(risk_factors) >= 3:
                return RiskLevel.HIGH
            elif len(risk_factors) >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return RiskLevel.HIGH
    
    def _determine_signal_risk_level(self, technical_score: float) -> RiskLevel:
        """Determine risk level for a trading signal."""
        if technical_score >= 0.8:
            return RiskLevel.LOW
        elif technical_score >= 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def _get_tradeable_tokens(self) -> List[str]:
        """Legacy compatibility list; detailed analysis no longer filters by it."""
        return ['BTC', 'ETH', 'SOL', 'USDC', 'DAI', 'WETH', 'LINK', 'UNI', 'AAVE', 'CRV']
    
    async def _generate_technical_exit_signals(
        self, 
        position, 
        market_data: MarketData, 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Generate exit signals based on technical analysis."""
        signals = []
        
        try:
            token_symbol = position.token_symbol
            
            # Get technical analysis for this token
            tech_analysis = analysis.get('technical_analysis', {}).get(token_symbol, {})
            overall_score = tech_analysis.get('overall_score', 0.5)
            
            # Take profit based on technical signals
            if (position.pnl_percent >= self.risk_params.take_profit_percent * 0.7 and  # 70% of target
                tech_analysis.get('recommended_action') == 'sell'):
                
                signal = TradingSignal(
                    signal_type=SignalType.TAKE_PROFIT,
                    token_symbol=token_symbol,
                    confidence=0.8,
                    reasoning=f"Technical take-profit: {position.pnl_percent:.1f}% gain with sell signal",
                    risk_level=RiskLevel.LOW,
                    metadata={
                        'current_pnl': position.pnl_percent,
                        'technical_score': overall_score
                    }
                )
                signals.append(signal)
            
            # Stop loss with technical confirmation
            elif position.pnl_percent <= -self.risk_params.stop_loss_percent:
                signal = TradingSignal(
                    signal_type=SignalType.STOP_LOSS,
                    token_symbol=token_symbol,
                    confidence=1.0,
                    reasoning=f"Technical stop-loss at {position.pnl_percent:.1f}% loss",
                    risk_level=RiskLevel.LOW,
                    metadata={'current_pnl': position.pnl_percent}
                )
                signals.append(signal)
            
            # Exit on very negative technical signals
            elif (overall_score < 0.3 and position.pnl_percent < 0):
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    token_symbol=token_symbol,
                    confidence=0.7,
                    reasoning=f"Poor technical signals (score: {overall_score:.2f}) with unrealized loss",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={
                        'technical_score': overall_score,
                        'current_pnl': position.pnl_percent
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical exit signals for {position.token_symbol}: {e}")
            return []
    
    def _filter_signals_by_market_conditions(
        self, 
        signals: List[TradingSignal], 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Filter signals based on current market conditions."""
        try:
            risk_assessment = analysis.get('risk_assessment', RiskLevel.MEDIUM)
            market_sentiment = analysis.get('market_sentiment', 'neutral')
            
            filtered_signals = []
            
            for signal in signals:
                should_include = True
                
                # In high-risk markets, only take exit signals and high-confidence entries
                if risk_assessment == RiskLevel.HIGH:
                    if signal.signal_type == SignalType.BUY and signal.confidence < 0.8:
                        should_include = False
                
                # In bearish markets, be more selective with buy signals
                if (market_sentiment == 'bearish' and 
                    signal.signal_type == SignalType.BUY and 
                    signal.confidence < 0.75):
                    should_include = False
                
                # Limit number of buy signals in uncertain conditions
                if (market_sentiment == 'neutral' and 
                    signal.signal_type == SignalType.BUY and 
                    len([s for s in filtered_signals if s.signal_type == SignalType.BUY]) >= 2):
                    should_include = False
                
                if should_include:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals

    def _calculate_technical_score(self, market_data: MarketData) -> float:
        """
        Calculate technical analysis score based on Ryu's balanced strategy.
        
        Ryu focuses on:
        - Technical indicator convergence
        - Momentum and trend strength
        - Volume confirmation
        - Market structure analysis
        """
        try:
            if not market_data:
                return 0.5
            
            scores = [
                self._calculate_directional_components(token, market_data)['overall']
                for token in market_data.token_prices.keys()
            ]
            return statistics.mean(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    def _calculate_fundamental_score(self, market_data: MarketData) -> float:
        """
        Calculate fundamental score based on Ryu's balanced strategy.
        
        Ryu considers:
        - Market cap diversity
        - Token utility and adoption
        - Risk-adjusted returns
        - Market sentiment balance
        """
        try:
            if not market_data:
                return 0.5
            
            # Calculate fundamental factors
            market_cap_scores = []
            utility_scores = []
            sentiment_scores = []
            
            for token in market_data.token_prices.keys():
                # Market cap score (balanced approach)
                market_cap = market_data.market_cap.get(token, 0)
                if market_cap > 10000000000:  # 10B+
                    market_cap_scores.append(0.8)
                elif market_cap > 1000000000:  # 1B+
                    market_cap_scores.append(0.7)
                elif market_cap > 100000000:  # 100M+
                    market_cap_scores.append(0.6)
                else:
                    market_cap_scores.append(0.4)
                
                # Utility score (balanced approach)
                if token in ['ETH', 'WETH', 'USDC', 'DAI']:
                    utility_scores.append(0.8)
                elif token in ['BTC', 'WBTC']:
                    utility_scores.append(0.7)
                else:
                    utility_scores.append(0.5)
                
                # Sentiment score (balanced approach)
                sentiment = market_data.trend_indicators.get('overall_sentiment', 'neutral')
                if sentiment == 'neutral':
                    sentiment_scores.append(0.7)  # Neutral is good for balanced strategy
                elif sentiment in ['stable', 'moderate']:
                    sentiment_scores.append(0.8)
                else:
                    sentiment_scores.append(0.5)
            
            # Calculate weighted average
            if market_cap_scores:
                avg_market_cap_score = sum(market_cap_scores) / len(market_cap_scores)
            else:
                avg_market_cap_score = 0.5
            
            if utility_scores:
                avg_utility_score = sum(utility_scores) / len(utility_scores)
            else:
                avg_utility_score = 0.5
            
            if sentiment_scores:
                avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            else:
                avg_sentiment_score = 0.5
            
            # Weighted fundamental score (Ryu balances all factors)
            fundamental_score = (
                avg_market_cap_score * 0.35 +  # Market cap
                avg_utility_score * 0.35 +     # Utility/adoption
                avg_sentiment_score * 0.3      # Sentiment balance
            )
            
            return fundamental_score
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    async def execute_trades(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """
        Execute trades using Ryu's balanced trading service with TokenAnalysisCard methodology.

        Override the base agent's execute_trades method to use the specialized
        RyuTradingService that implements TokenAnalysisCard analysis techniques.

        Args:
            signals: List of validated trading signals

        Returns:
            List of executed trade results
        """
        try:
            # Import here to avoid circular imports
            from kata.services.ryu_trading_service import get_ryu_trading_service

            trading_service = get_ryu_trading_service()
            user_wallet = await self.get_user_wallet_address()

            if not user_wallet:
                logger.error("No delegated wallet found for Ryu agent")
                return []

            executed_trades = []

            for signal in signals:
                try:
                    # Execute signal using Ryu trading service with TokenAnalysisCard methodology
                    execution_result = await trading_service.execute_ryu_signal(
                        agent=self,
                        signal=signal,
                        user_wallet_address=user_wallet
                    )

                    if execution_result.success:
                        # Convert to expected format
                        trade_result = {
                            'success': True,
                            'trade_id': execution_result.signal_id,
                            'executed_at': execution_result.executed_at,
                            'transaction_hash': execution_result.transaction_hash,
                            'trade_data': {
                                'trade_type': execution_result.trade_type,
                                'token_symbol': execution_result.token_symbol,
                                'amount': float(execution_result.amount),
                                'price': float(execution_result.execution_price),
                                'total_cost': float(execution_result.total_cost),
                                'gas_used': execution_result.gas_used,
                                'analysis_confidence': execution_result.analysis_confidence
                            }
                        }

                        executed_trades.append(trade_result)
                        self.successful_trades += 1

                        # Update daily P&L tracking
                        if signal.signal_type in [SignalType.SELL, SignalType.TAKE_PROFIT]:
                            estimated_pnl = execution_result.total_cost * Decimal('0.03')  # 3% profit assumption for balanced strategy
                            self.daily_pnl += estimated_pnl

                        logger.info(f"Ryu trade executed: {execution_result.trade_type} {execution_result.token_symbol} "
                                   f"(analysis confidence: {execution_result.analysis_confidence:.3f}, "
                                   f"tx: {execution_result.transaction_hash[:10]}...)")

                    else:
                        logger.error(f"Ryu trade execution failed: {execution_result.error_message}")

                except Exception as e:
                    logger.error(f"Error executing Ryu signal {signal.token_symbol}: {e}")
                    continue

            logger.info(f"Ryu agent executed {len(executed_trades)} trades successfully using TokenAnalysisCard methodology")
            return executed_trades

        except Exception as e:
            logger.error(f"Error in Ryu execute_trades: {e}")
            return []


# Agent factory function
def create_ryu_agent(user_id: str, config: Dict[str, Any]) -> RyuAgent:
    """Factory function to create a Ryu agent instance."""
    return RyuAgent(user_id, config)
