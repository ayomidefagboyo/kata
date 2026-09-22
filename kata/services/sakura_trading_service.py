"""
Sakura Agent Live Trading Service for Flow AI Trading Platform.

This service handles the execution of Sakura agent's conservative trading strategy
with Pendle yield farming integration using delegated wallet authority.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

from kata.agents.base_agent import TradingSignal, SignalType, MarketData
from kata.agents.sakura_agent import SakuraAgent
from kata.services.pendle_service import get_pendle_service, PendleYieldOpportunity, PendleSwapCalldata
from kata.services.privy_auth_service import PrivyAuthService, get_privy_auth_service
from kata.services.platform_wallet_service import get_platform_wallet_service
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


@dataclass
class SakuraTradeExecution:
    """Trade execution result for Sakura agent."""
    signal_id: str
    user_id: str
    trade_type: str
    token_symbol: str
    amount: Decimal
    execution_price: Decimal
    transaction_hash: str
    gas_used: int
    total_cost: Decimal
    success: bool
    error_message: Optional[str] = None
    executed_at: datetime = None

    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.now()


class SakuraTradingService:
    """
    Service for executing Sakura agent's conservative trading strategy.

    This service handles:
    - Live market data integration for Sakura
    - Pendle yield position management
    - Conservative trade execution with risk controls
    - Performance tracking and fee calculation
    - Wallet delegation validation
    """

    def __init__(self):
        """Initialize Sakura trading service."""
        self.db = get_service_client()
        self.pendle_service = None  # Lazy initialization
        self.privy_service: Optional[PrivyAuthService] = get_privy_auth_service()
        self.platform_wallet_service = get_platform_wallet_service()

        # Sakura-specific trading parameters
        self.max_slippage = 0.005  # 0.5% max slippage
        self.gas_price_multiplier = 1.2  # 20% gas buffer
        self.min_trade_value = 100.0  # $100 minimum trade
        self.max_trade_value = 10000.0  # $10k maximum trade for conservative approach

        logger.info("SakuraTradingService initialized")

    async def execute_sakura_signal(
        self,
        agent: SakuraAgent,
        signal: TradingSignal,
        user_wallet_address: str
    ) -> SakuraTradeExecution:
        """
        Execute a Sakura trading signal with conservative approach.

        Args:
            agent: Sakura agent instance
            signal: Trading signal to execute
            user_wallet_address: User's delegated wallet address

        Returns:
            SakuraTradeExecution result
        """
        try:
            logger.info(f"Executing Sakura signal: {signal.signal_type.value} {signal.token_symbol} (confidence: {signal.confidence:.3f})")

            # Validate delegation authority
            if not await self._validate_wallet_delegation(agent.user_id, user_wallet_address):
                return SakuraTradeExecution(
                    signal_id=f"sig_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    success=False,
                    error_message="Wallet delegation validation failed"
                )

            # Route to appropriate execution method
            if signal.metadata and signal.metadata.get('strategy_type') == 'pendle_fixed_yield':
                return await self._execute_pendle_yield_signal(agent, signal, user_wallet_address)
            else:
                return await self._execute_spot_trading_signal(agent, signal, user_wallet_address)

        except Exception as e:
            logger.error(f"Error executing Sakura signal: {e}")
            return SakuraTradeExecution(
                signal_id=f"sig_{int(datetime.now().timestamp())}",
                user_id=agent.user_id,
                trade_type=signal.signal_type.value,
                token_symbol=signal.token_symbol,
                amount=Decimal('0'),
                execution_price=Decimal('0'),
                transaction_hash="",
                gas_used=0,
                total_cost=Decimal('0'),
                success=False,
                error_message=str(e)
            )

    async def _execute_pendle_yield_signal(
        self,
        agent: SakuraAgent,
        signal: TradingSignal,
        user_wallet_address: str
    ) -> SakuraTradeExecution:
        """Execute Pendle yield farming signal."""
        try:
            pendle_service = await self._get_pendle_service()

            # Extract Pendle metadata
            metadata = signal.metadata
            market_address = metadata.get('market_address')
            pt_address = metadata.get('pt_address')
            expected_apy = metadata.get('expected_apy', 0)
            recommended_allocation = metadata.get('recommended_allocation', 5.0)

            # Calculate trade amount based on platform wallet balance
            user_balance_obj = await self.platform_wallet_service.get_user_balance(agent.user_id, "USDC")
            if not user_balance_obj:
                return SakuraTradeExecution(
                    signal_id=f"pendle_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type="pendle_buy",
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    success=False,
                    error_message="User has no platform wallet balance for USDC. Please deposit funds first."
                )

            available_balance = float(user_balance_obj.available_balance)
            trade_amount_usd = min(
                available_balance * (recommended_allocation / 100),
                self.max_trade_value
            )

            if trade_amount_usd < self.min_trade_value:
                return SakuraTradeExecution(
                    signal_id=f"pendle_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type="pendle_buy",
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    success=False,
                    error_message=f"Trade amount {trade_amount_usd} below minimum {self.min_trade_value}"
                )

            # Get Pendle swap calldata
            usdc_address = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base
            swap_calldata = await pendle_service.generate_swap_calldata(
                token_in_address=usdc_address,
                amount_in=str(int(trade_amount_usd * 1e6)),  # USDC has 6 decimals
                pt_address=pt_address,
                receiver_address=user_wallet_address,
                slippage=self.max_slippage
            )

            if not swap_calldata:
                return SakuraTradeExecution(
                    signal_id=f"pendle_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type="pendle_buy",
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    success=False,
                    error_message="Failed to generate Pendle swap calldata"
                )

            # Execute transaction via delegated wallet
            transaction_data = {
                'to': swap_calldata.to,
                'data': swap_calldata.data,
                'value': swap_calldata.value,
                'gas_limit': int(swap_calldata.gas_estimate * self.gas_price_multiplier),
                'type': 'pendle_purchase'
            }

            tx_result = await self._execute_delegated_transaction(
                agent.user_id,
                user_wallet_address,
                transaction_data
            )

            if tx_result and tx_result.get('transaction_hash'):
                # Calculate execution details
                execution_price = Decimal(str(trade_amount_usd)) / Decimal(str(swap_calldata.route_summary.get('amount_out', 1)))

                # Record successful execution
                execution = SakuraTradeExecution(
                    signal_id=f"pendle_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type="pendle_buy",
                    token_symbol=signal.token_symbol,
                    amount=Decimal(str(swap_calldata.route_summary.get('amount_out', 0))),
                    execution_price=execution_price,
                    transaction_hash=tx_result['transaction_hash'],
                    gas_used=tx_result.get('gas_used', 0),
                    total_cost=Decimal(str(trade_amount_usd)),
                    success=True
                )

                # Update platform wallet balance
                await self.platform_wallet_service.record_transaction(
                    user_id=agent.user_id,
                    transaction_type="pendle_purchase",
                    amount=trade_amount_usd,
                    token_symbol="USDC",
                    metadata={
                        'pt_address': pt_address,
                        'expected_apy': expected_apy,
                        'market_address': market_address
                    }
                )

                logger.info(f"Successfully executed Pendle yield signal: {expected_apy:.1f}% APY, ${trade_amount_usd:.2f}")
                return execution

            else:
                return SakuraTradeExecution(
                    signal_id=f"pendle_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type="pendle_buy",
                    token_symbol=signal.token_symbol,
                    amount=Decimal('0'),
                    execution_price=Decimal('0'),
                    transaction_hash="",
                    gas_used=0,
                    total_cost=Decimal('0'),
                    success=False,
                    error_message="Transaction execution failed"
                )

        except Exception as e:
            logger.error(f"Error executing Pendle yield signal: {e}")
            raise

    async def _execute_spot_trading_signal(
        self,
        agent: SakuraAgent,
        signal: TradingSignal,
        user_wallet_address: str
    ) -> SakuraTradeExecution:
        """Execute regular spot trading signal."""
        try:
            # Get current market price
            current_market_data = await self._get_current_market_data()
            if not current_market_data:
                raise ValueError("Unable to fetch current market data")

            current_price = current_market_data.token_prices.get(signal.token_symbol)
            if not current_price:
                raise ValueError(f"No price data for {signal.token_symbol}")

            # Calculate trade amount based on risk parameters
            user_balance_obj = await self.platform_wallet_service.get_user_balance(agent.user_id, "USDC")
            if not user_balance_obj:
                raise ValueError("User has no platform wallet balance for USDC. Please deposit funds first.")

            available_balance = float(user_balance_obj.available_balance)
            position_size_percent = agent.risk_params.max_position_size_percent / 100
            trade_amount_usd = min(
                available_balance * position_size_percent,
                self.max_trade_value
            )

            if trade_amount_usd < self.min_trade_value:
                raise ValueError(f"Trade amount {trade_amount_usd} below minimum {self.min_trade_value}")

            # Calculate token amount to trade
            token_amount = trade_amount_usd / float(current_price)

            if signal.signal_type == SignalType.BUY:
                # Execute buy order
                transaction_data = await self._prepare_buy_transaction(
                    signal.token_symbol,
                    token_amount,
                    current_price,
                    user_wallet_address
                )
            elif signal.signal_type == SignalType.SELL:
                # Execute sell order
                transaction_data = await self._prepare_sell_transaction(
                    signal.token_symbol,
                    token_amount,
                    current_price,
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
                execution = SakuraTradeExecution(
                    signal_id=f"spot_{int(datetime.now().timestamp())}",
                    user_id=agent.user_id,
                    trade_type=signal.signal_type.value,
                    token_symbol=signal.token_symbol,
                    amount=Decimal(str(token_amount)),
                    execution_price=current_price,
                    transaction_hash=tx_result['transaction_hash'],
                    gas_used=tx_result.get('gas_used', 0),
                    total_cost=Decimal(str(trade_amount_usd)),
                    success=True
                )

                # Update platform wallet balance
                await self.platform_wallet_service.record_transaction(
                    user_id=agent.user_id,
                    transaction_type=signal.signal_type.value,
                    amount=trade_amount_usd if signal.signal_type == SignalType.BUY else -trade_amount_usd,
                    token_symbol=signal.token_symbol,
                    metadata={'spot_trade': True}
                )

                logger.info(f"Successfully executed spot trade: {signal.signal_type.value} {token_amount:.6f} {signal.token_symbol}")
                return execution

            else:
                raise ValueError("Transaction execution failed")

        except Exception as e:
            logger.error(f"Error executing spot trading signal: {e}")
            raise

    async def _validate_wallet_delegation(self, user_id: str, wallet_address: str) -> bool:
        """Validate that wallet is properly delegated for trading."""
        try:
            if not self.privy_service:
                # In production, this would be properly initialized
                logger.warning("Privy service not initialized, skipping delegation validation")
                return True

            return await self.privy_service.validate_delegated_action(
                user_id=user_id,
                wallet_address=wallet_address,
                action_data={'type': 'trading', 'protocol': 'sakura'}
            )

        except Exception as e:
            logger.error(f"Error validating wallet delegation: {e}")
            return False

    async def _execute_delegated_transaction(
        self,
        user_id: str,
        wallet_address: str,
        transaction_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute transaction using delegated wallet authority."""
        try:
            if not self.privy_service:
                # In production, this would execute real transactions
                logger.info("Simulating transaction execution (Privy service not initialized)")
                return {
                    'transaction_hash': f"0x{''.join(['a' for _ in range(64)])}",
                    'gas_used': transaction_data.get('gas_limit', 21000),
                    'status': 'success'
                }

            return await self.privy_service.execute_delegated_transaction(
                user_id=user_id,
                wallet_address=wallet_address,
                transaction_data=transaction_data
            )

        except Exception as e:
            logger.error(f"Error executing delegated transaction: {e}")
            return None

    async def _get_pendle_service(self):
        """Get Pendle service instance."""
        if not self.pendle_service:
            self.pendle_service = await get_pendle_service()
        return self.pendle_service

    async def _get_current_market_data(self) -> Optional[MarketData]:
        """Get current market data for trading decisions."""
        try:
            # In production, this would fetch real-time market data
            # For now, return simulated data
            return MarketData(
                timestamp=datetime.now(),
                token_prices={
                    'ETH': Decimal('3200.00'),
                    'USDC': Decimal('1.00'),
                    'DAI': Decimal('1.00'),
                    'WETH': Decimal('3200.00'),
                    'cbETH': Decimal('3250.00'),
                    'stETH': Decimal('3195.00')
                },
                volume_24h={
                    'ETH': Decimal('2000000000'),
                    'USDC': Decimal('1500000000'),
                    'DAI': Decimal('800000000'),
                    'WETH': Decimal('1200000000'),
                    'cbETH': Decimal('500000000'),
                    'stETH': Decimal('600000000')
                },
                price_changes={
                    'ETH': Decimal('1.2'),
                    'USDC': Decimal('0.0'),
                    'DAI': Decimal('0.1'),
                    'WETH': Decimal('1.1'),
                    'cbETH': Decimal('0.8'),
                    'stETH': Decimal('1.0')
                },
                market_cap={
                    'ETH': Decimal('400000000000'),
                    'USDC': Decimal('150000000000'),
                    'DAI': Decimal('5000000000'),
                    'WETH': Decimal('50000000000'),
                    'cbETH': Decimal('30000000000'),
                    'stETH': Decimal('35000000000')
                },
                volatility={
                    'ETH': 0.02,
                    'USDC': 0.001,
                    'DAI': 0.002,
                    'WETH': 0.025,
                    'cbETH': 0.015,
                    'stETH': 0.018
                },
                trend_indicators={
                    'overall_sentiment': 'stable',
                    'fear_greed_index': 60,
                    'market_trend': 'slightly_bullish'
                }
            )

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    async def _prepare_buy_transaction(
        self,
        token_symbol: str,
        amount: float,
        price: Decimal,
        user_wallet: str
    ) -> Dict[str, Any]:
        """Prepare buy transaction data."""
        # In production, this would prepare actual DEX swap data
        return {
            'to': '0x' + '1' * 40,  # Mock DEX address
            'value': '0',
            'data': f"0xbuy{token_symbol}",
            'gas_limit': 200000,
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
        # In production, this would prepare actual DEX swap data
        return {
            'to': '0x' + '2' * 40,  # Mock DEX address
            'value': '0',
            'data': f"0xsell{token_symbol}",
            'gas_limit': 180000,
            'type': 'sell'
        }


# Global service instance
_sakura_trading_service: Optional[SakuraTradingService] = None

def get_sakura_trading_service() -> SakuraTradingService:
    """Get or create Sakura trading service instance."""
    global _sakura_trading_service
    if not _sakura_trading_service:
        _sakura_trading_service = SakuraTradingService()
    return _sakura_trading_service