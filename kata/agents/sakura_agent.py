"""
Sakura Agent - Conservative Trading Strategy

Sakura is the conservative agent that focuses on:
- Low-risk, stable investments
- Gradual portfolio growth
- Strong risk management
- Preference for stablecoins and established tokens
- Fixed yield opportunities via Pendle Finance
"""

import logging
from typing import Dict, Any, List
from decimal import Decimal

from kata.agents.base_agent import BaseAgent, MarketData, TradingSignal, SignalType, RiskLevel
from kata.services.pendle_service import get_pendle_service, PendleYieldOpportunity

logger = logging.getLogger(__name__)


class SakuraAgent(BaseAgent):
    """
    Conservative trading agent focused on stability and risk management.
    
    Sakura's trading philosophy:
    - Prioritize capital preservation over aggressive gains
    - Focus on low-volatility assets
    - Implement strict risk controls
    - Gradual position building and taking
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        """Initialize Sakura agent with conservative defaults."""
        
        # Set conservative risk parameters
        conservative_risk_params = {
            'max_position_size_percent': 3.0,  # Smaller positions
            'stop_loss_percent': 2.0,  # Tighter stop losses
            'take_profit_percent': 8.0,  # Lower profit targets
            'max_daily_loss_percent': 5.0,  # Lower daily loss limit
            'max_portfolio_concentration': 20.0,  # Better diversification
            'min_confidence_threshold': 0.8,  # Higher confidence required
            'max_trades_per_day': 3,  # Fewer trades
            'cooldown_minutes': 15  # Longer cooldown
        }
        
        # Merge with user config, keeping conservative defaults
        config_risk_params = config.get('risk_params', {})
        merged_risk_params = {**conservative_risk_params, **config_risk_params}
        config['risk_params'] = merged_risk_params
        
        super().__init__(user_id, "sakura", config)

        # Sakura-specific parameters
        self.volatility_threshold = 0.05  # Only trade low volatility assets
        self.min_market_cap = Decimal('1000000000')  # 1B min market cap
        self.preferred_tokens = ['USDC', 'DAI', 'ETH', 'WETH']  # Safe assets
        self.stability_weights = {
            'volatility': 0.4,
            'market_cap': 0.3,
            'volume': 0.2,
            'trend_strength': 0.1
        }

        # Pendle/yield strategy parameters - 100% specialization
        self.max_pendle_allocation = 100.0  # Max 100% in Pendle strategies (specialist)
        self.min_yield_apy = 5.0  # Minimum acceptable yield (5%)
        self.max_yield_apy = 25.0  # Maximum yield to avoid high-risk assets
        self.preferred_yield_timeframe = (90, 365)  # 3-12 months preferred

        logger.info(f"Sakura agent initialized for user {user_id} with conservative settings and Pendle integration")
    
    async def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Analyze market conditions with focus on stability and risk.
        
        Sakura's analysis prioritizes:
        - Market stability indicators
        - Low volatility opportunities
        - Risk-adjusted returns
        - Trend confirmation across multiple timeframes
        """
        try:
            analysis = {
                'timestamp': market_data.timestamp.isoformat(),
                'market_sentiment': 'neutral',
                'stability_score': 0.0,
                'risk_assessment': RiskLevel.MEDIUM,
                'recommended_allocation': {},
                'market_conditions': {}
            }
            
            # Calculate overall market stability
            avg_volatility = sum(market_data.volatility.values()) / len(market_data.volatility)
            stability_score = max(0, 1 - (avg_volatility / 0.1))  # Higher score = more stable
            analysis['stability_score'] = stability_score
            
            # Assess market sentiment based on stability
            if stability_score > 0.8:
                analysis['market_sentiment'] = 'stable'
                analysis['risk_assessment'] = RiskLevel.LOW
            elif stability_score > 0.6:
                analysis['market_sentiment'] = 'cautious'
                analysis['risk_assessment'] = RiskLevel.MEDIUM
            else:
                analysis['market_sentiment'] = 'volatile'
                analysis['risk_assessment'] = RiskLevel.HIGH
            
            # Analyze individual tokens for stability
            stable_tokens = []
            for token, volatility in market_data.volatility.items():
                if (volatility < self.volatility_threshold and 
                    token in self.preferred_tokens):
                    
                    token_stability = self._calculate_token_stability(token, market_data)
                    if token_stability > 0.7:
                        stable_tokens.append({
                            'symbol': token,
                            'stability_score': token_stability,
                            'current_price': float(market_data.token_prices.get(token, 0)),
                            'volatility': volatility
                        })
            
            analysis['stable_tokens'] = stable_tokens
            
            # Calculate recommended allocation (conservative)
            total_allocation = 0
            for token_info in stable_tokens:
                if total_allocation < 70:  # Max 70% allocated
                    allocation = min(15, (token_info['stability_score'] * 20))
                    analysis['recommended_allocation'][token_info['symbol']] = allocation
                    total_allocation += allocation
            
            # Reserve cash for stability
            analysis['recommended_allocation']['CASH'] = 100 - total_allocation
            
            # Market conditions assessment
            analysis['market_conditions'] = {
                'avg_volatility': avg_volatility,
                'high_volume_tokens': self._identify_high_volume_tokens(market_data),
                'trend_direction': market_data.trend_indicators.get('market_trend', 'neutral'),
                'fear_greed_index': market_data.trend_indicators.get('fear_greed_index', 50)
            }
            
            # Calculate agent-specific scores for token analysis
            technical_score = self._calculate_technical_score(market_data)
            fundamental_score = self._calculate_fundamental_score(market_data)
            
            # Add scores to analysis
            analysis['technical_score'] = technical_score
            analysis['fundamental_score'] = fundamental_score
            analysis['agent_strategy'] = 'conservative_stable'

            # Add Pendle yield opportunities analysis
            try:
                pendle_analysis = await self.analyze_pendle_opportunities()
                analysis['pendle_opportunities'] = pendle_analysis
                analysis['yield_strategies_available'] = len(pendle_analysis.get('top_opportunities', []))
            except Exception as e:
                logger.warning(f"Could not analyze Pendle opportunities: {e}")
                analysis['pendle_opportunities'] = {}
                analysis['yield_strategies_available'] = 0

            logger.info(f"Sakura market analysis completed: {analysis['market_sentiment']} market, "
                       f"{len(stable_tokens)} stable tokens identified, "
                       f"technical_score: {technical_score:.3f}, fundamental_score: {fundamental_score:.3f}, "
                       f"yield_strategies: {analysis['yield_strategies_available']}")

            return analysis
            
        except Exception as e:
            logger.error(f"Error in Sakura market analysis: {e}")
            raise  # Let the error propagate instead of returning fallback data
    
    async def generate_signals(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Generate conservative trading signals based on stability analysis.
        
        Sakura's signal generation focuses on:
        - High-confidence, low-risk opportunities
        - Gradual position building
        - Strong trend confirmation
        - Risk-adjusted entry and exit points
        """
        signals = []
        
        try:
            # Get market analysis
            analysis = await self.analyze_market(market_data)
            
            if analysis.get('error'):
                logger.error(f"Cannot generate signals due to analysis error: {analysis['error']}")
                return signals
            
            # Only generate signals in stable or cautious markets
            if analysis['risk_assessment'] == RiskLevel.HIGH:
                logger.info("Sakura: High risk market detected, no signals generated")
                return signals
            
            # Get current portfolio positions
            portfolio_positions = await self._get_portfolio_positions()
            current_holdings = {pos.token_symbol: pos for pos in portfolio_positions}
            
            # Generate buy signals for stable tokens
            stable_tokens = analysis.get('stable_tokens', [])
            for token_info in stable_tokens:
                token_symbol = token_info['symbol']
                
                # Skip if already have significant position
                if token_symbol in current_holdings:
                    current_allocation = current_holdings[token_symbol].allocation_percent
                    if current_allocation >= 15:  # Max 15% per token for Sakura
                        continue
                
                # Generate buy signal with conservative parameters
                confidence = self._calculate_buy_confidence(token_info, analysis, market_data)
                
                if confidence >= self.risk_params.min_confidence_threshold:
                    signal = TradingSignal(
                        signal_type=SignalType.BUY,
                        token_symbol=token_symbol,
                        confidence=confidence,
                        reasoning=f"Stable token with {token_info['stability_score']:.2f} stability score, "
                                f"low volatility ({token_info['volatility']:.3f}), conservative entry",
                        risk_level=RiskLevel.LOW,
                        metadata={
                            'stability_score': token_info['stability_score'],
                            'volatility': token_info['volatility'],
                            'analysis_type': 'stability_based',
                            'market_sentiment': analysis['market_sentiment']
                        }
                    )
                    signals.append(signal)
            
            # Generate exit signals for existing positions
            for position in portfolio_positions:
                exit_signals = await self._generate_exit_signals(position, market_data, analysis)
                signals.extend(exit_signals)

            # Generate Pendle yield signals
            try:
                pendle_signals = await self.generate_pendle_signals()
                signals.extend(pendle_signals)
                logger.info(f"Added {len(pendle_signals)} Pendle yield signals")
            except Exception as e:
                logger.warning(f"Could not generate Pendle signals: {e}")

            # Limit signals based on conservative approach
            max_signals = 3  # Increased to accommodate yield strategies
            if len(signals) > max_signals:
                # Sort by confidence and take top signals
                signals.sort(key=lambda s: s.confidence, reverse=True)
                signals = signals[:max_signals]

            logger.info(f"Sakura generated {len(signals)} conservative trading signals (including yield opportunities)")

            return signals
            
        except Exception as e:
            logger.error(f"Error generating Sakura signals: {e}")
            return []
    
    def _calculate_token_stability(self, token: str, market_data: MarketData) -> float:
        """Calculate stability score for a token."""
        try:
            score = 0.0
            
            # Volatility component (lower is better)
            volatility = market_data.volatility.get(token, 1.0)
            volatility_score = max(0, 1 - (volatility / self.volatility_threshold))
            score += volatility_score * self.stability_weights['volatility']
            
            # Market cap component (higher is better)
            market_cap = market_data.market_cap.get(token, Decimal('0'))
            market_cap_score = min(1.0, float(market_cap / self.min_market_cap))
            score += market_cap_score * self.stability_weights['market_cap']
            
            # Volume component (consistent volume is good)
            volume_24h = market_data.volume_24h.get(token, Decimal('0'))
            # Normalize volume score (simplified)
            volume_score = min(1.0, float(volume_24h / Decimal('100000000')))  # 100M baseline
            score += volume_score * self.stability_weights['volume']
            
            # Trend strength component (moderate trends preferred)
            price_change = market_data.price_changes.get(token, Decimal('0'))
            trend_score = 1.0 - min(1.0, abs(float(price_change)) / 10.0)  # Penalize large moves
            score += trend_score * self.stability_weights['trend_strength']
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating stability for {token}: {e}")
            return 0.0
    
    def _identify_high_volume_tokens(self, market_data: MarketData) -> List[str]:
        """Identify tokens with high trading volume."""
        try:
            volume_threshold = Decimal('50000000')  # 50M USD threshold
            high_volume_tokens = []
            
            for token, volume in market_data.volume_24h.items():
                if volume >= volume_threshold:
                    high_volume_tokens.append(token)
            
            return high_volume_tokens
            
        except Exception as e:
            logger.error(f"Error identifying high volume tokens: {e}")
            return []
    
    def _calculate_buy_confidence(
        self, 
        token_info: Dict[str, Any], 
        analysis: Dict[str, Any], 
        market_data: MarketData
    ) -> float:
        """Calculate confidence score for a buy signal."""
        try:
            base_confidence = token_info['stability_score']
            
            # Boost confidence for preferred tokens
            if token_info['symbol'] in ['USDC', 'DAI']:
                base_confidence += 0.1  # Stablecoin bonus
            elif token_info['symbol'] in ['ETH', 'WETH']:
                base_confidence += 0.05  # Blue chip bonus
            
            # Adjust based on market sentiment
            sentiment_multiplier = {
                'stable': 1.1,
                'cautious': 1.0,
                'volatile': 0.8
            }
            
            market_sentiment = analysis.get('market_sentiment', 'cautious')
            confidence = base_confidence * sentiment_multiplier.get(market_sentiment, 1.0)
            
            # Ensure within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating buy confidence: {e}")
            return 0.0
    
    async def _generate_exit_signals(
        self, 
        position, 
        market_data: MarketData, 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Generate exit signals for existing positions."""
        signals = []
        
        try:
            token_symbol = position.token_symbol
            
            # Conservative take-profit (lower targets)
            if position.pnl_percent >= (self.risk_params.take_profit_percent * 0.8):  # 80% of target
                signal = TradingSignal(
                    signal_type=SignalType.TAKE_PROFIT,
                    token_symbol=token_symbol,
                    confidence=0.9,
                    reasoning=f"Conservative profit taking at {position.pnl_percent:.1f}% gain",
                    risk_level=RiskLevel.LOW,
                    metadata={'current_pnl': position.pnl_percent}
                )
                signals.append(signal)
            
            # Strict stop-loss
            elif position.pnl_percent <= -self.risk_params.stop_loss_percent:
                signal = TradingSignal(
                    signal_type=SignalType.STOP_LOSS,
                    token_symbol=token_symbol,
                    confidence=1.0,
                    reasoning=f"Stop-loss triggered at {position.pnl_percent:.1f}% loss",
                    risk_level=RiskLevel.LOW,
                    metadata={'current_pnl': position.pnl_percent}
                )
                signals.append(signal)
            
            # Exit if token becomes too volatile
            token_volatility = market_data.volatility.get(token_symbol, 0)
            if token_volatility > self.volatility_threshold * 2:  # Double the threshold
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    token_symbol=token_symbol,
                    confidence=0.8,
                    reasoning=f"Exiting due to increased volatility ({token_volatility:.3f})",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={'volatility': token_volatility}
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating exit signals for {position.token_symbol}: {e}")
            return []

    def _calculate_technical_score(self, market_data: MarketData) -> float:
        """
        Calculate technical analysis score based on Sakura's conservative strategy.
        
        Sakura focuses on:
        - Stability indicators
        - Low volatility patterns
        - Trend confirmation
        - Volume consistency
        """
        try:
            if not market_data:
                return 0.5
            
            # Calculate stability-based technical score
            volatility_scores = []
            trend_scores = []
            volume_scores = []
            
            for token, volatility in market_data.volatility.items():
                # Volatility score (lower is better for Sakura)
                vol_score = max(0, 1 - (volatility / 0.1))  # 0.1 = 10% volatility threshold
                volatility_scores.append(vol_score)
                
                # Trend score (stable trends are better)
                trend_indicator = market_data.trend_indicators.get(f'{token}_trend', 'neutral')
                if trend_indicator == 'stable':
                    trend_scores.append(0.8)
                elif trend_indicator == 'neutral':
                    trend_scores.append(0.6)
                else:
                    trend_scores.append(0.4)
                
                # Volume score (consistent volume is better)
                volume = market_data.volume_24h.get(token, 0)
                if volume > 0:
                    volume_scores.append(0.7)  # Consistent volume
                else:
                    volume_scores.append(0.3)  # Low volume
            
            # Calculate weighted average
            if volatility_scores:
                avg_volatility_score = sum(volatility_scores) / len(volatility_scores)
            else:
                avg_volatility_score = 0.5
            
            if trend_scores:
                avg_trend_score = sum(trend_scores) / len(trend_scores)
            else:
                avg_trend_score = 0.5
            
            if volume_scores:
                avg_volume_score = sum(volume_scores) / len(volume_scores)
            else:
                avg_volume_score = 0.5
            
            # Weighted technical score (Sakura prioritizes stability)
            technical_score = (
                avg_volatility_score * 0.5 +  # Stability is most important
                avg_trend_score * 0.3 +       # Trend confirmation
                avg_volume_score * 0.2        # Volume consistency
            )
            
            return technical_score
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    def _calculate_fundamental_score(self, market_data: MarketData) -> float:
        """
        Calculate fundamental score based on Sakura's conservative strategy.
        
        Sakura considers:
        - Market cap stability
        - Token utility and adoption
        - Risk-adjusted returns
        - Market sentiment stability
        """
        try:
            if not market_data:
                return 0.5
            
            # Calculate fundamental factors
            market_cap_scores = []
            utility_scores = []
            sentiment_scores = []
            
            for token in market_data.token_prices.keys():
                # Market cap score (higher is better for stability)
                market_cap = market_data.market_cap.get(token, 0)
                if market_cap > 10000000000:  # 10B+
                    market_cap_scores.append(0.9)
                elif market_cap > 1000000000:  # 1B+
                    market_cap_scores.append(0.7)
                elif market_cap > 100000000:  # 100M+
                    market_cap_scores.append(0.5)
                else:
                    market_cap_scores.append(0.3)
                
                # Utility score (preferred tokens get higher scores)
                if token in self.preferred_tokens:
                    utility_scores.append(0.8)
                else:
                    utility_scores.append(0.4)
                
                # Sentiment score (stable sentiment is better)
                sentiment = market_data.trend_indicators.get('overall_sentiment', 'neutral')
                if sentiment == 'stable':
                    sentiment_scores.append(0.8)
                elif sentiment == 'neutral':
                    sentiment_scores.append(0.6)
                else:
                    sentiment_scores.append(0.4)
            
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
            
            # Weighted fundamental score
            fundamental_score = (
                avg_market_cap_score * 0.4 +  # Market cap is important
                avg_utility_score * 0.4 +     # Utility/adoption
                avg_sentiment_score * 0.2     # Sentiment stability
            )
            
            return fundamental_score
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    async def analyze_pendle_opportunities(self) -> Dict[str, Any]:
        """
        Analyze Pendle yield opportunities for Sakura's conservative strategy.

        Returns:
            Dictionary containing top opportunities, risk assessment, and recommendations
        """
        try:
            pendle_service = await get_pendle_service()

            # Get Sakura-suitable opportunities
            opportunities = await pendle_service.get_sakura_opportunities(max_opportunities=5)

            if not opportunities:
                return {
                    'top_opportunities': [],
                    'total_opportunities': 0,
                    'average_apy': 0.0,
                    'recommended_allocation': 0.0,
                    'risk_assessment': 'No suitable opportunities'
                }

            # Filter by Sakura's yield criteria
            suitable_opportunities = []
            for opp in opportunities:
                if (self.min_yield_apy <= opp.expected_apy <= self.max_yield_apy and
                    opp.risk_level in ['LOW', 'MEDIUM'] and
                    opp.sakura_score >= 0.7):
                    suitable_opportunities.append(opp)

            if not suitable_opportunities:
                return {
                    'top_opportunities': [],
                    'total_opportunities': len(opportunities),
                    'average_apy': 0.0,
                    'recommended_allocation': 0.0,
                    'risk_assessment': 'No opportunities meet Sakura criteria'
                }

            # Calculate metrics
            avg_apy = sum(opp.expected_apy for opp in suitable_opportunities) / len(suitable_opportunities)
            avg_score = sum(opp.sakura_score for opp in suitable_opportunities) / len(suitable_opportunities)

            # Recommended allocation based on opportunity quality
            if avg_score >= 0.9 and avg_apy >= 10.0:
                recommended_allocation = min(25.0, self.max_pendle_allocation)
            elif avg_score >= 0.8 and avg_apy >= 8.0:
                recommended_allocation = min(15.0, self.max_pendle_allocation)
            else:
                recommended_allocation = min(10.0, self.max_pendle_allocation)

            return {
                'top_opportunities': [
                    {
                        'market_address': opp.market.market_address,
                        'underlying_symbol': opp.market.underlying_symbol,
                        'expected_apy': opp.expected_apy,
                        'risk_level': opp.risk_level,
                        'time_to_maturity': opp.time_to_maturity,
                        'sakura_score': opp.sakura_score,
                        'liquidity_score': opp.liquidity_score,
                        'discount_percent': opp.discount_to_maturity
                    }
                    for opp in suitable_opportunities[:3]  # Top 3
                ],
                'total_opportunities': len(opportunities),
                'suitable_opportunities': len(suitable_opportunities),
                'average_apy': avg_apy,
                'average_sakura_score': avg_score,
                'recommended_allocation': recommended_allocation,
                'risk_assessment': self._assess_pendle_portfolio_risk(suitable_opportunities)
            }

        except Exception as e:
            logger.error(f"Error analyzing Pendle opportunities: {e}")
            return {
                'top_opportunities': [],
                'total_opportunities': 0,
                'error': str(e)
            }

    async def generate_pendle_signals(self) -> List[TradingSignal]:
        """
        Generate trading signals for Pendle yield opportunities.

        Returns:
            List of trading signals for PT purchases
        """
        signals = []

        try:
            pendle_service = await get_pendle_service()
            opportunities = await pendle_service.get_sakura_opportunities(max_opportunities=3)

            for opp in opportunities:
                # Apply Sakura's strict criteria
                if (opp.sakura_score < 0.8 or
                    opp.expected_apy < self.min_yield_apy or
                    opp.expected_apy > self.max_yield_apy or
                    opp.risk_level == 'HIGH'):
                    continue

                # Create yield signal
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    token_symbol=f"PT-{opp.market.underlying_symbol}",
                    confidence=min(0.95, opp.sakura_score),  # Cap at 95% for conservative approach
                    reasoning=f"Pendle fixed yield: {opp.expected_apy:.1f}% APY, "
                             f"{opp.time_to_maturity}d to maturity, "
                             f"risk: {opp.risk_level}, "
                             f"discount: {opp.discount_to_maturity:.1f}%",
                    risk_level=RiskLevel.LOW if opp.risk_level == "LOW" else RiskLevel.MEDIUM,
                    metadata={
                        'strategy_type': 'pendle_fixed_yield',
                        'market_address': opp.market.market_address,
                        'pt_address': opp.market.pt_address,
                        'expected_apy': opp.expected_apy,
                        'time_to_maturity': opp.time_to_maturity,
                        'sakura_score': opp.sakura_score,
                        'underlying_asset': opp.market.underlying_symbol,
                        'current_pt_price': opp.current_pt_price,
                        'break_even_days': opp.break_even_days,
                        'recommended_allocation': min(5.0, self.max_pendle_allocation / 3)  # Max 5% per position
                    }
                )
                signals.append(signal)

            logger.info(f"Generated {len(signals)} Pendle yield signals for Sakura")
            return signals

        except Exception as e:
            logger.error(f"Error generating Pendle signals: {e}")
            return []

    def _assess_pendle_portfolio_risk(self, opportunities: List[PendleYieldOpportunity]) -> str:
        """Assess overall risk of Pendle portfolio for Sakura."""
        if not opportunities:
            return "No risk - no positions"

        # Calculate risk factors
        avg_liquidity = sum(opp.liquidity_score for opp in opportunities) / len(opportunities)
        max_time_to_maturity = max(opp.time_to_maturity for opp in opportunities)
        avg_yield = sum(opp.expected_apy for opp in opportunities) / len(opportunities)

        # Risk assessment logic
        if (avg_liquidity >= 0.8 and
            max_time_to_maturity <= 180 and
            avg_yield <= 15.0):
            return "LOW - Conservative yield strategy"
        elif (avg_liquidity >= 0.6 and
              max_time_to_maturity <= 270 and
              avg_yield <= 20.0):
            return "MEDIUM - Balanced yield approach"
        else:
            return "HIGH - Consider reducing exposure"

    async def execute_trades(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """
        Execute trades using Sakura's live trading service.

        Override the base agent's execute_trades method to use the specialized
        SakuraTradingService for live Pendle and spot trading execution.

        Args:
            signals: List of validated trading signals

        Returns:
            List of executed trade results
        """
        try:
            # Import here to avoid circular imports
            from kata.services.sakura_trading_service import get_sakura_trading_service

            trading_service = get_sakura_trading_service()
            user_wallet = await self.get_user_wallet_address()

            if not user_wallet:
                logger.error("No delegated wallet found for Sakura agent")
                return []

            executed_trades = []

            for signal in signals:
                try:
                    # Execute signal using Sakura trading service
                    execution_result = await trading_service.execute_sakura_signal(
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
                                'gas_used': execution_result.gas_used
                            }
                        }

                        executed_trades.append(trade_result)
                        self.successful_trades += 1

                        # Update daily P&L tracking
                        if signal.signal_type in [SignalType.SELL, SignalType.TAKE_PROFIT]:
                            estimated_pnl = execution_result.total_cost * Decimal('0.02')
                            self.daily_pnl += estimated_pnl

                        logger.info(f"Sakura trade executed: {execution_result.trade_type} {execution_result.token_symbol} "
                                   f"(tx: {execution_result.transaction_hash[:10]}...)")

                    else:
                        logger.error(f"Sakura trade execution failed: {execution_result.error_message}")

                except Exception as e:
                    logger.error(f"Error executing Sakura signal {signal.token_symbol}: {e}")
                    continue

            logger.info(f"Sakura agent executed {len(executed_trades)} trades successfully")
            return executed_trades

        except Exception as e:
            logger.error(f"Error in Sakura execute_trades: {e}")
            return []


# Agent factory function
def create_sakura_agent(user_id: str, config: Dict[str, Any]) -> SakuraAgent:
    """Factory function to create a Sakura agent instance."""
    return SakuraAgent(user_id, config)