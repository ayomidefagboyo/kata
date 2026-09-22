"""
Ryu Agent Trading Service for Flow AI Trading Platform.

This service handles Ryu agent's balanced spot trading strategy using
TokenAnalysisCard methodology for comprehensive token analysis and execution.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from kata.agents.base_agent import TradingSignal, SignalType, MarketData
from kata.agents.ryu_agent import RyuAgent
from kata.services.privy_auth_service import PrivyAuthService
from kata.services.platform_wallet_service import get_platform_wallet_service
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


@dataclass
class RyuTradeExecution:
    """Trade execution result for Ryu agent."""
    signal_id: str
    user_id: str
    trade_type: str
    token_symbol: str
    amount: Decimal
    execution_price: Decimal
    transaction_hash: str
    gas_used: int
    total_cost: Decimal
    analysis_confidence: float
    success: bool
    error_message: Optional[str] = None
    executed_at: datetime = None

    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.now()


@dataclass
class TokenAnalysisResult:
    """Token analysis result using TokenAnalysisCard methodology."""
    token_symbol: str
    current_price: Decimal
    technical_score: float
    fundamental_score: float
    momentum_score: float
    sentiment_score: float
    overall_score: float
    recommendation: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    risk_level: str
    entry_price_target: Optional[Decimal] = None
    exit_price_target: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None


class RyuTradingService:
    """
    Service for executing Ryu agent's balanced spot trading strategy.

    This service integrates TokenAnalysisCard methodology to provide:
    - Comprehensive token analysis (technical, fundamental, sentiment)
    - Balanced risk-reward spot trading
    - Multi-factor decision making
    - Adaptive strategy based on market conditions
    """

    def __init__(self):
        """Initialize Ryu trading service."""
        self.db = get_service_client()
        self.privy_service: Optional[PrivyAuthService] = None
        self.platform_wallet_service = get_platform_wallet_service()

        # Ryu-specific trading parameters
        self.max_slippage = 0.01  # 1% max slippage (more flexible than Sakura)
        self.gas_price_multiplier = 1.15  # 15% gas buffer
        self.min_trade_value = 50.0  # $50 minimum trade
        self.max_trade_value = 25000.0  # $25k maximum trade for balanced approach

        # TokenAnalysisCard parameters
        self.analysis_weights = {
            'technical': 0.35,
            'fundamental': 0.25,
            'momentum': 0.25,
            'sentiment': 0.15
        }

        logger.info("RyuTradingService initialized with TokenAnalysisCard methodology")

    async def execute_ryu_signal(
        self,
        agent: RyuAgent,
        signal: TradingSignal,
        user_wallet_address: str
    ) -> RyuTradeExecution:
        """
        Execute a Ryu trading signal using TokenAnalysisCard methodology.

        Args:
            agent: Ryu agent instance
            signal: Trading signal to execute
            user_wallet_address: User's delegated wallet address

        Returns:
            RyuTradeExecution result
        """
        try:
            logger.info(f"Executing Ryu signal: {signal.signal_type.value} {signal.token_symbol} (confidence: {signal.confidence:.3f})")

            # Validate delegation authority
            if not await self._validate_wallet_delegation(agent.user_id, user_wallet_address):
                return RyuTradeExecution(
                    signal_id=f"ryu_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    analysis_confidence=signal.confidence,
                    success=False,
                    error_message="Wallet delegation validation failed"
                )

            # Perform comprehensive token analysis using TokenAnalysisCard methodology
            analysis_result = await self._perform_token_analysis(signal.token_symbol)

            if not analysis_result:
                return RyuTradeExecution(
                    signal_id=f"ryu_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    analysis_confidence=signal.confidence,
                    success=False,
                    error_message="Token analysis failed"
                )

            # Validate analysis recommendation against signal
            if not self._validate_analysis_signal_alignment(analysis_result, signal):
                return RyuTradeExecution(
                    signal_id=f"ryu_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    analysis_confidence=analysis_result.confidence,
                    success=False,
                    error_message=f"Analysis recommends {analysis_result.recommendation}, signal is {signal.signal_type.value}"
                )

            # Execute the trade with TokenAnalysisCard insights
            return await self._execute_analyzed_trade(agent, signal, analysis_result, user_wallet_address)

        except Exception as e:
            logger.error(f"Error executing Ryu signal: {e}")
            return RyuTradeExecution(
                signal_id=f"ryu_{int(datetime.now().timestamp())}",
                user_id=agent.user_id,
                trade_type=signal.signal_type.value,
                token_symbol=signal.token_symbol,
                amount=Decimal('0'),
                execution_price=Decimal('0'),
                transaction_hash="",
                gas_used=0,
                total_cost=Decimal('0'),
                analysis_confidence=signal.confidence,
                success=False,
                error_message=str(e)
            )

    async def _perform_token_analysis(self, token_symbol: str) -> Optional[TokenAnalysisResult]:
        """
        Perform comprehensive token analysis using TokenAnalysisCard methodology.

        This replicates the analysis logic from the TokenAnalysisCard to ensure
        consistency with the manual analysis tool.
        """
        try:
            # Get current market data
            current_market_data = await self._get_current_market_data()
            if not current_market_data:
                return None

            current_price = current_market_data.token_prices.get(token_symbol)
            if not current_price:
                return None

            # Technical Analysis (35% weight)
            technical_score = await self._calculate_technical_score(token_symbol, current_market_data)

            # Fundamental Analysis (25% weight)
            fundamental_score = await self._calculate_fundamental_score(token_symbol, current_market_data)

            # Momentum Analysis (25% weight)
            momentum_score = await self._calculate_momentum_score(token_symbol, current_market_data)

            # Sentiment Analysis (15% weight)
            sentiment_score = await self._calculate_sentiment_score(token_symbol, current_market_data)

            # Calculate overall score
            overall_score = (
                technical_score * self.analysis_weights['technical'] +
                fundamental_score * self.analysis_weights['fundamental'] +
                momentum_score * self.analysis_weights['momentum'] +
                sentiment_score * self.analysis_weights['sentiment']
            )

            # Determine recommendation and confidence
            if overall_score >= 0.7:
                recommendation = "BUY"
                confidence = min(0.95, overall_score)
            elif overall_score <= 0.3:
                recommendation = "SELL"
                confidence = min(0.95, 1.0 - overall_score)
            else:
                recommendation = "HOLD"
                confidence = 0.5

            # Determine risk level
            volatility = current_market_data.volatility.get(token_symbol, 0.05)
            if volatility < 0.03:
                risk_level = "LOW"
            elif volatility < 0.08:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            # Generate reasoning
            reasoning = self._generate_analysis_reasoning(
                token_symbol, technical_score, fundamental_score, momentum_score, sentiment_score
            )

            # Calculate price targets
            entry_target, exit_target, stop_loss = self._calculate_price_targets(
                current_price, recommendation, volatility, overall_score
            )

            return TokenAnalysisResult(
                token_symbol=token_symbol,
                current_price=current_price,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                momentum_score=momentum_score,
                sentiment_score=sentiment_score,
                overall_score=overall_score,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                risk_level=risk_level,
                entry_price_target=entry_target,
                exit_price_target=exit_target,
                stop_loss_price=stop_loss
            )

        except Exception as e:
            logger.error(f"Error performing token analysis for {token_symbol}: {e}")
            return None

    async def _calculate_technical_score(self, token_symbol: str, market_data: MarketData) -> float:
        """Calculate technical analysis score (RSI, MACD, moving averages, etc.)."""
        try:
            # Price change trend
            price_change = float(market_data.price_changes.get(token_symbol, Decimal('0')))
            trend_score = min(1.0, max(0.0, (price_change + 10) / 20))  # Normalize -10% to +10%

            # Volume analysis
            volume = market_data.volume_24h.get(token_symbol, Decimal('0'))
            avg_volume = sum(market_data.volume_24h.values()) / len(market_data.volume_24h)
            volume_score = min(1.0, float(volume / avg_volume))

            # Volatility analysis (lower volatility = higher score for Ryu's balanced approach)
            volatility = market_data.volatility.get(token_symbol, 0.05)
            volatility_score = max(0.0, 1.0 - (volatility / 0.15))  # Normalize against 15% volatility

            # Combine technical indicators
            technical_score = (trend_score * 0.4 + volume_score * 0.3 + volatility_score * 0.3)

            return technical_score

        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5

    async def _calculate_fundamental_score(self, token_symbol: str, market_data: MarketData) -> float:
        """Calculate fundamental analysis score (market cap, utility, adoption)."""
        try:
            # Market cap analysis
            market_cap = market_data.market_cap.get(token_symbol, Decimal('0'))
            if market_cap > Decimal('50000000000'):  # 50B+
                mcap_score = 0.9
            elif market_cap > Decimal('10000000000'):  # 10B+
                mcap_score = 0.8
            elif market_cap > Decimal('1000000000'):  # 1B+
                mcap_score = 0.6
            elif market_cap > Decimal('100000000'):  # 100M+
                mcap_score = 0.4
            else:
                mcap_score = 0.2

            # Token utility (based on Ryu's preferred tokens)
            utility_tokens = ['ETH', 'WETH', 'USDC', 'WBTC', 'DAI', 'USDT']
            utility_score = 0.8 if token_symbol in utility_tokens else 0.4

            # Combine fundamental factors
            fundamental_score = (mcap_score * 0.6 + utility_score * 0.4)

            return fundamental_score

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 0.5

    async def _calculate_momentum_score(self, token_symbol: str, market_data: MarketData) -> float:
        """Calculate momentum analysis score (price trends, acceleration)."""
        try:
            # Recent price momentum
            price_change = float(market_data.price_changes.get(token_symbol, Decimal('0')))

            # Positive momentum scoring
            if price_change > 5:
                momentum_score = 0.9
            elif price_change > 2:
                momentum_score = 0.7
            elif price_change > 0:
                momentum_score = 0.6
            elif price_change > -2:
                momentum_score = 0.4
            elif price_change > -5:
                momentum_score = 0.3
            else:
                momentum_score = 0.1

            return momentum_score

        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.5

    async def _calculate_sentiment_score(self, token_symbol: str, market_data: MarketData) -> float:
        """Calculate market sentiment score."""
        try:
            # Overall market sentiment
            sentiment = market_data.trend_indicators.get('overall_sentiment', 'neutral')
            fear_greed = market_data.trend_indicators.get('fear_greed_index', 50)

            # Sentiment scoring
            if sentiment == 'bullish' and fear_greed > 60:
                sentiment_score = 0.8
            elif sentiment == 'bullish' or fear_greed > 55:
                sentiment_score = 0.7
            elif sentiment == 'neutral' and 45 <= fear_greed <= 55:
                sentiment_score = 0.6
            elif sentiment == 'bearish' or fear_greed < 45:
                sentiment_score = 0.4
            else:
                sentiment_score = 0.5

            return sentiment_score

        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return 0.5

    def _generate_analysis_reasoning(
        self,
        token_symbol: str,
        technical: float,
        fundamental: float,
        momentum: float,
        sentiment: float
    ) -> str:
        """Generate human-readable reasoning for the analysis."""

        reasoning_parts = []

        # Technical analysis
        if technical > 0.7:
            reasoning_parts.append(f"Strong technical indicators (score: {technical:.2f})")
        elif technical < 0.4:
            reasoning_parts.append(f"Weak technical setup (score: {technical:.2f})")

        # Fundamental analysis
        if fundamental > 0.7:
            reasoning_parts.append(f"solid fundamentals (score: {fundamental:.2f})")
        elif fundamental < 0.4:
            reasoning_parts.append(f"concerning fundamentals (score: {fundamental:.2f})")

        # Momentum
        if momentum > 0.7:
            reasoning_parts.append(f"positive momentum (score: {momentum:.2f})")
        elif momentum < 0.4:
            reasoning_parts.append(f"negative momentum (score: {momentum:.2f})")

        # Sentiment
        if sentiment > 0.6:
            reasoning_parts.append(f"favorable market sentiment (score: {sentiment:.2f})")
        elif sentiment < 0.4:
            reasoning_parts.append(f"unfavorable sentiment (score: {sentiment:.2f})")

        if reasoning_parts:
            return f"{token_symbol} analysis shows " + ", ".join(reasoning_parts) + ". Balanced risk-reward assessment."
        else:
            return f"{token_symbol} shows mixed signals across all analysis factors. Neutral stance recommended."

    def _calculate_price_targets(
        self,
        current_price: Decimal,
        recommendation: str,
        volatility: float,
        overall_score: float
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, exit, and stop-loss price targets."""

        if recommendation == "BUY":
            # Buy targets
            entry_target = current_price * Decimal('0.98')  # 2% below current
            exit_target = current_price * Decimal(str(1.0 + (overall_score * 0.15)))  # Up to 15% gain
            stop_loss = current_price * Decimal('0.95')  # 5% stop loss

        elif recommendation == "SELL":
            # Sell targets
            entry_target = current_price * Decimal('1.02')  # 2% above current
            exit_target = current_price * Decimal(str(1.0 - (overall_score * 0.10)))  # Up to 10% drop
            stop_loss = current_price * Decimal('1.05')  # 5% stop loss

        else:  # HOLD
            entry_target = None
            exit_target = None
            stop_loss = current_price * Decimal('0.90')  # 10% emergency stop

        return entry_target, exit_target, stop_loss

    def _validate_analysis_signal_alignment(
        self,
        analysis: TokenAnalysisResult,
        signal: TradingSignal
    ) -> bool:
        """Validate that TokenAnalysisCard recommendation aligns with the signal."""

        signal_type = signal.signal_type.value.upper()
        analysis_rec = analysis.recommendation.upper()

        # Direct alignment
        if signal_type == analysis_rec:
            return True

        # Allow some flexibility for borderline cases
        if analysis.overall_score > 0.6 and signal_type == "BUY" and analysis_rec == "HOLD":
            return True

        if analysis.overall_score < 0.4 and signal_type == "SELL" and analysis_rec == "HOLD":
            return True

        return False

    async def _execute_analyzed_trade(
        self,
        agent: RyuAgent,
        signal: TradingSignal,
        analysis: TokenAnalysisResult,
        user_wallet_address: str
    ) -> RyuTradeExecution:
        """Execute trade using TokenAnalysisCard insights."""
        try:
            # Calculate trade amount based on risk parameters and analysis confidence
            user_balance = 5000.0  # Mock balance for now
            base_position_size = agent.risk_params.max_position_size_percent / 100

            # Adjust position size based on analysis confidence and risk level
            confidence_multiplier = analysis.confidence
            risk_multiplier = 1.0 if analysis.risk_level == "LOW" else 0.8 if analysis.risk_level == "MEDIUM" else 0.6

            adjusted_position_size = base_position_size * confidence_multiplier * risk_multiplier
            trade_amount_usd = min(
                user_balance * adjusted_position_size,
                self.max_trade_value
            )

            if trade_amount_usd < self.min_trade_value:
                raise ValueError(f"Trade amount {trade_amount_usd} below minimum {self.min_trade_value}")

            # Calculate token amount
            token_amount = trade_amount_usd / float(analysis.current_price)

            # Prepare transaction data
            if signal.signal_type == SignalType.BUY:
                transaction_data = await self._prepare_buy_transaction(
                    signal.token_symbol,
                    token_amount,
                    analysis.current_price,
                    user_wallet_address
                )
            elif signal.signal_type == SignalType.SELL:
                transaction_data = await self._prepare_sell_transaction(
                    signal.token_symbol,
                    token_amount,
                    analysis.current_price,
                    user_wallet_address
                )
            else:
                raise ValueError(f"Unsupported signal type: {signal.signal_type}")

            # Execute transaction
            tx_result = await self._execute_delegated_transaction(
                agent.user_id,
                user_wallet_address,
                transaction_data
            )

            if tx_result and tx_result.get('transaction_hash'):
                execution = RyuTradeExecution(
                    signal_id=f"ryu_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal(str(token_amount)),
                    execution_price=analysis.current_price,
                    transaction_hash=tx_result['transaction_hash'],
                    gas_used=tx_result.get('gas_used', 0),
                    total_cost=Decimal(str(trade_amount_usd)),
                    analysis_confidence=analysis.confidence,
                    success=True
                )

                logger.info(f"Successfully executed Ryu trade: {signal.signal_type.value} {token_amount:.6f} {signal.token_symbol} "
                           f"(analysis score: {analysis.overall_score:.3f})")
                return execution

            else:
                raise ValueError("Transaction execution failed")

        except Exception as e:
            logger.error(f"Error executing analyzed trade: {e}")
            raise

    async def _validate_wallet_delegation(self, user_id: str, wallet_address: str) -> bool:
        """Validate wallet delegation for Ryu agent."""
        # Mock validation for now
        logger.info("Validating wallet delegation for Ryu agent")
        return True

    async def _execute_delegated_transaction(
        self,
        user_id: str,
        wallet_address: str,
        transaction_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute transaction using delegated wallet authority."""
        # Mock execution for now
        return {
            'transaction_hash': f"0x{''.join(['b' for _ in range(64)])}",
            'gas_used': transaction_data.get('gas_limit', 21000),
            'status': 'success'
        }

    async def _get_current_market_data(self) -> Optional[MarketData]:
        """Get current market data for analysis."""
        # Return mock data for now - in production this would fetch real data
        return MarketData(
            timestamp=datetime.now(),
            token_prices={
                'ETH': Decimal('3200.00'),
                'USDC': Decimal('1.00'),
                'DAI': Decimal('1.00'),
                'WETH': Decimal('3200.00'),
                'WBTC': Decimal('67000.00'),
                'USDT': Decimal('1.00')
            },
            volume_24h={
                'ETH': Decimal('2000000000'),
                'USDC': Decimal('1500000000'),
                'DAI': Decimal('800000000'),
                'WETH': Decimal('1200000000'),
                'WBTC': Decimal('800000000'),
                'USDT': Decimal('2500000000')
            },
            price_changes={
                'ETH': Decimal('2.1'),
                'USDC': Decimal('0.0'),
                'DAI': Decimal('0.1'),
                'WETH': Decimal('2.0'),
                'WBTC': Decimal('1.8'),
                'USDT': Decimal('0.05')
            },
            market_cap={
                'ETH': Decimal('400000000000'),
                'USDC': Decimal('150000000000'),
                'DAI': Decimal('5000000000'),
                'WETH': Decimal('50000000000'),
                'WBTC': Decimal('80000000000'),
                'USDT': Decimal('120000000000')
            },
            volatility={
                'ETH': 0.05,
                'USDC': 0.001,
                'DAI': 0.002,
                'WETH': 0.055,
                'WBTC': 0.06,
                'USDT': 0.001
            },
            trend_indicators={
                'overall_sentiment': 'bullish',
                'fear_greed_index': 65,
                'market_trend': 'upward'
            }
        )

    async def _prepare_buy_transaction(
        self,
        token_symbol: str,
        amount: float,
        price: Decimal,
        user_wallet: str
    ) -> Dict[str, Any]:
        """Prepare buy transaction data."""
        return {
            'to': '0x' + '3' * 40,  # Mock DEX address
            'value': '0',
            'data': f"0xbuy{token_symbol}",
            'gas_limit': 250000,
            'type': 'buy'
        }

    async def _prepare_sell_transaction(
        self,
        token_symbol: str,
        amount: float,
        price: Decimal,
        user_wallet: str
    ) -> Dict[str, Any]:
        """Prepare sell transaction data."""
        return {
            'to': '0x' + '4' * 40,  # Mock DEX address
            'value': '0',
            'data': f"0xsell{token_symbol}",
            'gas_limit': 220000,
            'type': 'sell'
        }


# Global service instance
_ryu_trading_service: Optional[RyuTradingService] = None

def get_ryu_trading_service() -> RyuTradingService:
    """Get or create Ryu trading service instance."""
    global _ryu_trading_service
    if not _ryu_trading_service:
        _ryu_trading_service = RyuTradingService()
    return _ryu_trading_service