"""
Base agent class for Flow AI Trading Platform.

This module provides the foundational BaseAgent class that all specific trading agents
(Sakura, Ryu, Yuki) will inherit from. It includes core functionality for market analysis,
risk management, trade execution, and decision logging.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from supabase import Client
from kata.config.database import get_service_client
from kata.models.trade import TradeCreate, AgentDecisionCreate

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    STRONG_BUY = "strong_buy"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    HOLD = "hold"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    signal_type: SignalType
    token_symbol: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_amount: Optional[Decimal] = None
    suggested_price: Optional[Decimal] = None
    risk_level: RiskLevel = RiskLevel.MEDIUM
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketData:
    """Market data structure."""
    timestamp: datetime
    token_prices: Dict[str, Decimal]
    volume_24h: Dict[str, Decimal]
    price_changes: Dict[str, Decimal]
    market_cap: Dict[str, Decimal]
    volatility: Dict[str, float]
    trend_indicators: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_size_percent: float = 30.0  # Max % of portfolio per position
    stop_loss_percent: float = 30.0  # Stop loss threshold
    take_profit_percent: float = 40.0  # Take profit threshold
    max_daily_loss_percent: float = 10.0  # Max daily loss
    max_portfolio_concentration: float = 30.0  # Max % in single asset
    min_confidence_threshold: float = 0.6  # Min confidence to execute
    max_trades_per_day: int = 10  # Max trades per day
    cooldown_minutes: int = 5  # Cooldown between trades


@dataclass
class PortfolioPosition:
    """Portfolio position data."""
    token_symbol: str
    amount: Decimal
    value_usd: Decimal
    entry_price: Decimal
    current_price: Decimal
    pnl_usd: Decimal
    pnl_percent: float
    allocation_percent: float


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Provides core functionality for market analysis, risk management, 
    trade execution, and decision logging that all specific agents inherit.
    """
    
    def __init__(
        self, 
        user_id: str, 
        agent_type: str,
        config: Dict[str, Any]
    ):
        """
        Initialize the base agent.
        
        Args:
            user_id: User ID this agent belongs to
            agent_type: Type of agent (sakura, ryu, yuki)
            config: Agent configuration dictionary
        """
        self.user_id = user_id
        self.agent_type = agent_type
        self.config = config
        self.db_client: Client = get_service_client()
        
        # Initialize risk parameters from config
        self.risk_params = RiskParameters(**config.get('risk_params', {}))
        
        # Runtime state
        self.is_running = False
        self.last_execution_time: Optional[datetime] = None
        self.daily_trades_count = 0
        self.daily_pnl = Decimal('0')
        self.execution_errors = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.total_trades = 0
        self.successful_trades = 0
        
        logger.info(f"Initialized {agent_type} agent for user {user_id}")
    
    @abstractmethod
    async def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Analyze market conditions and generate insights.
        
        This method must be implemented by each specific agent to provide
        their unique market analysis approach.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary containing market analysis results
        """
        pass
    
    @abstractmethod
    async def generate_signals(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Generate trading signals based on market analysis.
        
        This method must be implemented by each specific agent to provide
        their unique signal generation logic.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        pass
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """
        Execute the main trading cycle.
        
        This is the primary method that orchestrates the entire trading process:
        1. Fetch market data
        2. Analyze market conditions
        3. Generate trading signals
        4. Validate signals against risk parameters
        5. Execute approved trades
        6. Log decisions and results
        
        Returns:
            Dictionary containing cycle execution results
        """
        cycle_start_time = time.time()
        cycle_results = {
            'success': False,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': [],
            'execution_time_ms': 0
        }
        
        try:
            logger.info(f"Starting trading cycle for {self.agent_type} agent (user: {self.user_id})")
            
            # Check if agent should run (cooldown, daily limits, etc.)
            if not await self._can_execute_cycle():
                cycle_results['errors'].append("Cycle execution prevented by constraints")
                return cycle_results
            
            # Step 1: Fetch market data
            market_data = await self._fetch_market_data()
            if not market_data:
                cycle_results['errors'].append("Failed to fetch market data")
                return cycle_results
            
            # Step 2: Analyze market conditions
            analysis_results = await self.analyze_market(market_data)
            
            # Step 3: Generate trading signals
            signals = await self.generate_signals(market_data)
            cycle_results['signals_generated'] = len(signals)
            
            # Step 4: Validate signals against risk parameters
            validated_signals = []
            for signal in signals:
                if await self.validate_risk(signal):
                    validated_signals.append(signal)
                else:
                    await self._log_rejected_signal(signal, "Risk validation failed")
            
            # Step 5: Execute approved trades
            if validated_signals:
                executed_trades = await self.execute_trades(validated_signals)
                cycle_results['trades_executed'] = len(executed_trades)
                self.total_trades += len(executed_trades)
                self.daily_trades_count += len(executed_trades)
            
            # Step 6: Log cycle completion
            await self.log_agent_decision(
                decision_type="trading_cycle_complete",
                market_data=market_data.__dict__,
                decision_data={
                    'analysis_results': analysis_results,
                    'signals_generated': len(signals),
                    'signals_validated': len(validated_signals),
                    'trades_executed': cycle_results['trades_executed']
                },
                action_taken=f"Completed trading cycle with {cycle_results['trades_executed']} trades",
                confidence_score=1.0
            )
            
            self.last_execution_time = datetime.now()
            cycle_results['success'] = True
            
            logger.info(f"Trading cycle completed successfully for {self.agent_type} agent")
            
        except Exception as e:
            error_msg = f"Trading cycle failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            cycle_results['errors'].append(error_msg)
            self.execution_errors.append({
                'timestamp': datetime.now(),
                'error': error_msg,
                'cycle_step': 'unknown'
            })
            
            # Log the error
            await self.log_agent_decision(
                decision_type="trading_cycle_error",
                market_data={},
                decision_data={'error': error_msg},
                action_taken="Trading cycle aborted due to error",
                confidence_score=0.0
            )
        
        finally:
            cycle_results['execution_time_ms'] = int((time.time() - cycle_start_time) * 1000)
        
        return cycle_results
    
    async def execute_trades(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """
        Execute trades based on validated signals.

        This method should be overridden by specific agents to use their
        specialized trading services (e.g., SakuraTradingService, YukiTradingService).

        Args:
            signals: List of validated trading signals

        Returns:
            List of executed trade results
        """
        executed_trades = []

        for signal in signals:
            try:
                # Default simulation for base agent
                trade_params = await self._calculate_trade_parameters(signal)

                if not trade_params:
                    continue

                # Create trade record
                trade_data = TradeCreate(
                    user_id=self.user_id,
                    agent_type=self.agent_type,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=trade_params['amount'],
                    price_usd=trade_params['price'],
                    value_usd=trade_params['value'],
                    protocol="base",
                    reasoning=signal.reasoning,
                    confidence_score=signal.confidence
                )

                # Execute trade (simulated in base class)
                trade_result = await self._simulate_trade_execution(trade_data)

                if trade_result['success']:
                    executed_trades.append(trade_result)
                    self.successful_trades += 1

                    # Update daily P&L tracking
                    if signal.signal_type in [SignalType.SELL, SignalType.TAKE_PROFIT]:
                        estimated_pnl = trade_params['value'] * Decimal('0.02')
                        self.daily_pnl += estimated_pnl

                    logger.info(f"Trade simulated successfully: {signal.signal_type.value} {signal.token_symbol}")
                else:
                    logger.error(f"Trade simulation failed: {trade_result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error executing trade for signal {signal.token_symbol}: {e}")
                continue

        return executed_trades

    async def get_user_wallet_address(self) -> Optional[str]:
        """
        Get the user's delegated wallet address for trading.

        This method should be implemented to retrieve the user's wallet
        address that has been delegated for agent trading.

        Returns:
            Wallet address if available, None otherwise
        """
        try:
            # In production, this would query the user's delegated wallet
            # For now, return a mock address
            return f"0x{self.user_id[:8]}{'0' * 32}"

        except Exception as e:
            logger.error(f"Error getting user wallet address: {e}")
            return None
    
    async def validate_risk(self, signal: TradingSignal) -> bool:
        """
        Validate a trading signal against risk parameters.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal passes risk validation, False otherwise
        """
        try:
            # Check confidence threshold
            if signal.confidence < self.risk_params.min_confidence_threshold:
                logger.debug(f"Signal rejected: confidence {signal.confidence} below threshold {self.risk_params.min_confidence_threshold}")
                return False
            
            # Check daily trade limit
            if self.daily_trades_count >= self.risk_params.max_trades_per_day:
                logger.debug(f"Signal rejected: daily trade limit reached ({self.daily_trades_count})")
                return False
            
            # Check cooldown period
            if self.last_execution_time:
                time_since_last = datetime.now() - self.last_execution_time
                if time_since_last.total_seconds() < (self.risk_params.cooldown_minutes * 60):
                    logger.debug(f"Signal rejected: cooldown period active")
                    return False
            
            # Check daily loss limit
            if self.daily_pnl < -Decimal(str(self.risk_params.max_daily_loss_percent)) * Decimal('100'):
                logger.debug(f"Signal rejected: daily loss limit reached")
                return False
            
            # Get current portfolio for position-specific checks
            portfolio_positions = await self._get_portfolio_positions()
            
            # Check position size limits for buy signals
            if signal.signal_type == SignalType.BUY:
                if not await self._validate_position_size(signal, portfolio_positions):
                    return False
                
                if not await self._validate_portfolio_concentration(signal, portfolio_positions):
                    return False
            
            # Check stop-loss and take-profit thresholds for existing positions
            if signal.signal_type in [SignalType.STOP_LOSS, SignalType.TAKE_PROFIT]:
                if not await self._validate_exit_thresholds(signal, portfolio_positions):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk for signal {signal.token_symbol}: {e}")
            return False
    
    async def log_agent_decision(
        self,
        decision_type: str,
        market_data: Dict[str, Any],
        decision_data: Dict[str, Any],
        action_taken: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> bool:
        """
        Log an agent decision to the database.
        
        Args:
            decision_type: Type of decision made
            market_data: Market data at time of decision
            decision_data: Decision details and reasoning
            action_taken: Action taken based on decision
            confidence_score: Confidence in the decision (0.0-1.0)
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            decision = AgentDecisionCreate(
                user_id=self.user_id,
                agent_type=self.agent_type,
                decision_type=decision_type,
                market_data=market_data,
                decision_data=decision_data,
                action_taken=action_taken,
                confidence_score=confidence_score
            )
            
            response = self.db_client.from_("agent_decisions").insert(decision.model_dump()).execute()
            
            if response.data:
                logger.debug(f"Agent decision logged: {decision_type}")
                return True
            else:
                logger.error(f"Failed to log agent decision: {decision_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging agent decision: {e}")
            return False
    
    async def start_agent(self) -> bool:
        """
        Start the agent's trading operations.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning(f"Agent {self.agent_type} is already running")
                return False
            
            self.is_running = True
            self.start_time = datetime.now()
            self._reset_daily_counters()
            
            await self.log_agent_decision(
                decision_type="agent_start",
                market_data={},
                decision_data={'config': self.config},
                action_taken=f"Started {self.agent_type} agent",
                confidence_score=1.0
            )
            
            logger.info(f"Agent {self.agent_type} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting agent {self.agent_type}: {e}")
            return False
    
    async def stop_agent(self) -> bool:
        """
        Stop the agent's trading operations.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                logger.warning(f"Agent {self.agent_type} is not running")
                return False
            
            self.is_running = False
            
            # Calculate session statistics
            session_duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
            success_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
            
            await self.log_agent_decision(
                decision_type="agent_stop",
                market_data={},
                decision_data={
                    'session_duration_minutes': session_duration.total_seconds() / 60,
                    'total_trades': self.total_trades,
                    'successful_trades': self.successful_trades,
                    'success_rate': success_rate,
                    'daily_pnl': float(self.daily_pnl)
                },
                action_taken=f"Stopped {self.agent_type} agent",
                confidence_score=1.0
            )
            
            logger.info(f"Agent {self.agent_type} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_type}: {e}")
            return False
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get agent performance summary.
        
        Returns:
            Dictionary containing performance metrics
        """
        try:
            session_duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
            success_rate = (self.successful_trades / max(self.total_trades, 1)) * 100
            
            return {
                'agent_type': self.agent_type,
                'is_running': self.is_running,
                'session_duration_minutes': session_duration.total_seconds() / 60,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'success_rate': success_rate,
                'daily_trades_count': self.daily_trades_count,
                'daily_pnl': float(self.daily_pnl),
                'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None,
                'error_count': len(self.execution_errors),
                'risk_parameters': self.risk_params.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    # Private helper methods
    
    async def _can_execute_cycle(self) -> bool:
        """Check if trading cycle can be executed."""
        # Check if agent is running
        if not self.is_running:
            return False
        
        # Check daily trade limit
        if self.daily_trades_count >= self.risk_params.max_trades_per_day:
            return False
        
        # Check cooldown period
        if self.last_execution_time:
            time_since_last = datetime.now() - self.last_execution_time
            if time_since_last.total_seconds() < (self.risk_params.cooldown_minutes * 60):
                return False
        
        # Check daily loss limit
        if self.daily_pnl < -Decimal(str(self.risk_params.max_daily_loss_percent)) * Decimal('100'):
            return False
        
        return True
    
    async def _fetch_market_data(self) -> Optional[MarketData]:
        """Fetch current market data."""
        try:
            # In production, this would fetch real market data from APIs
            # For now, return simulated data
            return MarketData(
                timestamp=datetime.now(),
                token_prices={
                    'ETH': Decimal('3200.00'),
                    'USDC': Decimal('1.00'),
                    'DAI': Decimal('1.00')
                },
                volume_24h={
                    'ETH': Decimal('1000000000'),
                    'USDC': Decimal('500000000'),
                    'DAI': Decimal('300000000')
                },
                price_changes={
                    'ETH': Decimal('2.5'),
                    'USDC': Decimal('0.0'),
                    'DAI': Decimal('0.1')
                },
                market_cap={
                    'ETH': Decimal('400000000000'),
                    'USDC': Decimal('150000000000'),
                    'DAI': Decimal('5000000000')
                },
                volatility={
                    'ETH': 0.15,
                    'USDC': 0.001,
                    'DAI': 0.002
                },
                trend_indicators={
                    'overall_sentiment': 'bullish',
                    'fear_greed_index': 65,
                    'market_trend': 'upward'
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    async def _get_portfolio_positions(self) -> List[PortfolioPosition]:
        """Get current portfolio positions."""
        try:
            # Get user's portfolio
            portfolio_response = self.db_client.from_("portfolios").select("id, total_value_usd").eq("user_id", self.user_id).execute()
            
            if not portfolio_response.data:
                return []
            
            portfolio = portfolio_response.data[0]
            portfolio_id = portfolio['id']
            total_portfolio_value = Decimal(str(portfolio['total_value_usd']))
            
            # Get holdings
            holdings_response = self.db_client.from_("holdings").select("*").eq("portfolio_id", portfolio_id).execute()
            
            positions = []
            for holding in holdings_response.data or []:
                value_usd = Decimal(str(holding['value_usd']))
                pnl_usd = Decimal(str(holding['pnl_usd']))
                allocation_percent = float(value_usd / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                
                position = PortfolioPosition(
                    token_symbol=holding['token_symbol'],
                    amount=Decimal(str(holding['amount'])),
                    value_usd=value_usd,
                    entry_price=Decimal('0'),  # Would need to track this separately
                    current_price=Decimal('0'),  # Would need current price data
                    pnl_usd=pnl_usd,
                    pnl_percent=float(pnl_usd / value_usd * 100) if value_usd > 0 else 0,
                    allocation_percent=allocation_percent
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return []
    
    async def _validate_position_size(self, signal: TradingSignal, positions: List[PortfolioPosition]) -> bool:
        """Validate position size against limits."""
        # This would calculate if the new position would exceed max position size
        # For now, return True (implement actual validation logic)
        return True
    
    async def _validate_portfolio_concentration(self, signal: TradingSignal, positions: List[PortfolioPosition]) -> bool:
        """Validate portfolio concentration limits."""
        # Check if adding this position would exceed concentration limits
        for position in positions:
            if position.token_symbol == signal.token_symbol:
                if position.allocation_percent > self.risk_params.max_portfolio_concentration:
                    return False
        return True
    
    async def _validate_exit_thresholds(self, signal: TradingSignal, positions: List[PortfolioPosition]) -> bool:
        """Validate exit signal thresholds."""
        for position in positions:
            if position.token_symbol == signal.token_symbol:
                if signal.signal_type == SignalType.STOP_LOSS:
                    return position.pnl_percent <= -self.risk_params.stop_loss_percent
                elif signal.signal_type == SignalType.TAKE_PROFIT:
                    return position.pnl_percent >= self.risk_params.take_profit_percent
        return False
    
    async def _calculate_trade_parameters(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Calculate trade parameters for a signal."""
        try:
            # Get current market price (simplified)
            market_data = await self._fetch_market_data()
            if not market_data or signal.token_symbol not in market_data.token_prices:
                return None
            
            current_price = market_data.token_prices[signal.token_symbol]
            
            # Calculate trade amount based on position sizing
            # For now, use a simple percentage of portfolio
            portfolio_response = self.db_client.from_("portfolios").select("total_value_usd").eq("user_id", self.user_id).execute()
            
            if not portfolio_response.data:
                return None
            
            total_portfolio_value = Decimal(str(portfolio_response.data[0]['total_value_usd']))
            position_size_usd = total_portfolio_value * Decimal(str(self.risk_params.max_position_size_percent / 100))
            
            amount = position_size_usd / current_price
            
            return {
                'amount': amount.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
                'price': current_price,
                'value': position_size_usd.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade parameters: {e}")
            return None
    
    async def _simulate_trade_execution(self, trade_data: TradeCreate) -> Dict[str, Any]:
        """Simulate trade execution (replace with real execution in production)."""
        try:
            # Create trade record in database
            response = self.db_client.from_("trades").insert(trade_data.model_dump()).execute()
            
            if response.data:
                return {
                    'success': True,
                    'trade_id': response.data[0]['id'],
                    'executed_at': datetime.now(),
                    'trade_data': trade_data.model_dump()
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create trade record'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _log_rejected_signal(self, signal: TradingSignal, reason: str):
        """Log a rejected trading signal."""
        await self.log_agent_decision(
            decision_type="signal_rejected",
            market_data={},
            decision_data={
                'signal': signal.__dict__,
                'rejection_reason': reason
            },
            action_taken=f"Rejected {signal.signal_type.value} signal for {signal.token_symbol}",
            confidence_score=signal.confidence
        )
    
    def _reset_daily_counters(self):
        """Reset daily counters (called at start of new day)."""
        self.daily_trades_count = 0
        self.daily_pnl = Decimal('0')
        self.execution_errors = []
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_type.title()}Agent(user_id={self.user_id}, running={self.is_running})"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"{self.__class__.__name__}(user_id='{self.user_id}', "
                f"agent_type='{self.agent_type}', is_running={self.is_running}, "
                f"total_trades={self.total_trades})")
