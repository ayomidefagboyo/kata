"""
Platform Wallet Service for Flow AI Trading Platform.

Manages user funds, allocations, and strategy execution across all agents
with integrated Pendle support using the existing platform wallet system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from kata.config.database import get_service_client
from kata.models.platform_wallet import (
    PendleStrategyAllocation,
    PendlePositionAllocation,
    DepositTransaction,
    AGENT_STRATEGY_CONFIGS,
    PENDLE_SUPPORTED_TOKENS,
    StrategyType,
    RiskLevel
)
from kata.services.pendle_service import PendleService, get_pendle_service

logger = logging.getLogger(__name__)


@dataclass
class AllocationRequest:
    """Request to allocate funds to a strategy."""
    user_id: str
    agent_name: str
    strategy_type: str
    token_symbol: str
    amount: float
    risk_level: str = "MEDIUM"
    min_yield_target: float = 5.0
    max_maturity_days: int = 90
    auto_reinvest: bool = True


@dataclass
class PortfolioSummary:
    """Summary of user's portfolio across all strategies."""
    total_balance: float
    available_balance: float
    allocated_balance: float
    pendle_allocation: float
    trading_allocation: float
    lending_allocation: float
    total_pnl: float
    total_yield: float
    active_strategies: int


@dataclass
class PlatformWalletBalanceRecord:
    """Supabase row used by wallet and allocation services.

    The platform-wallet tables are read through Supabase rather than a
    SQLAlchemy session. Keeping the response as a small data record avoids
    initializing unrelated legacy Pendle ORM relationships while handling an
    agent allocation request.
    """

    id: Optional[int]
    user_id: str
    platform_wallet_address: str
    token_symbol: str
    token_address: str
    chain_id: int
    total_balance: Decimal
    available_balance: Decimal
    allocated_balance: Decimal
    pendle_allocation: Decimal
    trading_allocation: Decimal
    lending_allocation: Decimal
    last_updated: Optional[Any] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PlatformWalletBalanceRecord":
        return cls(
            id=row.get("id"),
            user_id=str(row.get("user_id") or ""),
            platform_wallet_address=str(row.get("platform_wallet_address") or ""),
            token_symbol=str(row.get("token_symbol") or ""),
            token_address=str(row.get("token_address") or ""),
            chain_id=int(row.get("chain_id") or 8453),
            total_balance=Decimal(str(row.get("total_balance") or 0)),
            available_balance=Decimal(str(row.get("available_balance") or 0)),
            allocated_balance=Decimal(str(row.get("allocated_balance") or 0)),
            pendle_allocation=Decimal(str(row.get("pendle_allocation") or 0)),
            trading_allocation=Decimal(str(row.get("trading_allocation") or 0)),
            lending_allocation=Decimal(str(row.get("lending_allocation") or 0)),
            last_updated=row.get("last_updated"),
        )

    def get_allocation_percentage(self, strategy: str) -> float:
        total = float(self.total_balance)
        if total <= 0:
            return 0.0
        allocation_map = {
            "pendle": float(self.pendle_allocation),
            "trading": float(self.trading_allocation),
            "lending": float(self.lending_allocation),
        }
        return allocation_map.get(strategy, 0.0) / total * 100

    def can_allocate(self, amount: float, strategy: str = None) -> bool:
        return float(self.available_balance) >= amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "platform_wallet_address": self.platform_wallet_address,
            "token_symbol": self.token_symbol,
            "token_address": self.token_address,
            "chain_id": self.chain_id,
            "total_balance": float(self.total_balance),
            "available_balance": float(self.available_balance),
            "allocated_balance": float(self.allocated_balance),
            "pendle_allocation": float(self.pendle_allocation),
            "trading_allocation": float(self.trading_allocation),
            "lending_allocation": float(self.lending_allocation),
            "last_updated": self.last_updated,
        }


class PlatformWalletService:
    """
    Service for managing platform wallet funds and strategy allocations.

    Integrates with existing platform wallet system to provide unified
    fund management across all agents (Sakura, Yuki, Ryu).
    """

    def __init__(self):
        self.pendle_service: Optional[PendleService] = None

    async def _get_pendle_service(self) -> PendleService:
        """Get Pendle service instance."""
        if not self.pendle_service:
            self.pendle_service = await get_pendle_service()
        return self.pendle_service

    async def get_user_balance(
        self,
        user_id: str,
        token_symbol: str = "USDC",
    ) -> Optional[PlatformWalletBalanceRecord]:
        """Get user's platform wallet balance for a specific token."""
        try:
            client = get_service_client()
            response = await asyncio.to_thread(
                lambda: (
                    client.table('platform_wallet_balances')
                    .select('*')
                    .eq('user_id', user_id)
                    .eq('token_symbol', token_symbol)
                    .single()
                    .execute()
                )
            )

            if response.data:
                return PlatformWalletBalanceRecord.from_row(response.data)
            return None

        except Exception as e:
            logger.error(f"Error fetching balance for user {user_id}: {e}")
            return None

    async def get_user_portfolio_summary(self, user_id: str) -> PortfolioSummary:
        """Get comprehensive portfolio summary for user."""
        try:
            # Get all balances
            async with get_service_client() as client:
                balances_response = await client.table('platform_wallet_balances').select('*').eq('user_id', user_id).execute()

                strategies_response = await client.table('pendle_strategy_allocations').select('*').eq('user_id', user_id).eq('is_active', True).execute()

            total_balance = 0.0
            available_balance = 0.0
            allocated_balance = 0.0
            pendle_allocation = 0.0
            trading_allocation = 0.0
            lending_allocation = 0.0

            # Sum up all token balances (convert to USD equivalent)
            for balance in balances_response.data:
                total_balance += float(balance['total_balance'])
                available_balance += float(balance['available_balance'])
                allocated_balance += float(balance['allocated_balance'])
                pendle_allocation += float(balance['pendle_allocation'])
                trading_allocation += float(balance['trading_allocation'])
                lending_allocation += float(balance['lending_allocation'])

            # Calculate total PnL and yield from strategies
            total_pnl = 0.0
            total_yield = 0.0
            active_strategies = len(strategies_response.data)

            for strategy in strategies_response.data:
                total_pnl += float(strategy['unrealized_pnl']) + float(strategy['realized_pnl'])
                total_yield += float(strategy['total_yield_earned'])

            return PortfolioSummary(
                total_balance=total_balance,
                available_balance=available_balance,
                allocated_balance=allocated_balance,
                pendle_allocation=pendle_allocation,
                trading_allocation=trading_allocation,
                lending_allocation=lending_allocation,
                total_pnl=total_pnl,
                total_yield=total_yield,
                active_strategies=active_strategies
            )

        except Exception as e:
            logger.error(f"Error getting portfolio summary for user {user_id}: {e}")
            return PortfolioSummary(0, 0, 0, 0, 0, 0, 0, 0, 0)

    async def allocate_to_pendle_strategy(self, request: AllocationRequest) -> Dict[str, Any]:
        """
        Allocate funds to a Pendle strategy for a specific agent.

        This is the core function that bridges platform wallet funds with Pendle strategies.
        """
        try:
            logger.info(f"Allocating {request.amount} {request.token_symbol} to {request.agent_name} {request.strategy_type}")

            # Validate agent and strategy compatibility
            agent_config = AGENT_STRATEGY_CONFIGS.get(request.agent_name)
            if not agent_config:
                raise ValueError(f"Unknown agent: {request.agent_name}")

            if request.strategy_type not in agent_config['preferred_strategies']:
                logger.warning(f"Strategy {request.strategy_type} not preferred for agent {request.agent_name}")

            # Validate token support
            if request.token_symbol not in PENDLE_SUPPORTED_TOKENS:
                raise ValueError(f"Token {request.token_symbol} not supported for Pendle strategies")

            # Check user balance
            balance = await self.get_user_balance(request.user_id, request.token_symbol)
            if not balance or not balance.can_allocate(request.amount):
                raise ValueError(f"Insufficient {request.token_symbol} balance for allocation")

            # Check agent-specific allocation limits
            max_pendle_pct = agent_config['max_pendle_allocation_pct']
            current_pendle_pct = balance.get_allocation_percentage('pendle')
            new_pendle_pct = ((float(balance.pendle_allocation) + request.amount) / float(balance.total_balance)) * 100

            if new_pendle_pct > max_pendle_pct:
                raise ValueError(f"Allocation would exceed {request.agent_name}'s max Pendle allocation of {max_pendle_pct}%")

            # Create strategy allocation record
            strategy_allocation = {
                'user_id': request.user_id,
                'platform_wallet_address': balance.platform_wallet_address,
                'agent_name': request.agent_name,
                'strategy_type': request.strategy_type,
                'base_token_symbol': request.token_symbol,
                'base_token_address': PENDLE_SUPPORTED_TOKENS[request.token_symbol]['address'],
                'allocated_amount': str(request.amount),
                'risk_level': request.risk_level,
                'min_yield_target': str(request.min_yield_target),
                'max_maturity_days': request.max_maturity_days,
                'auto_reinvest': request.auto_reinvest,
                'is_active': True
            }

            # Update balance allocation
            new_available = float(balance.available_balance) - request.amount
            new_allocated = float(balance.allocated_balance) + request.amount
            new_pendle_allocation = float(balance.pendle_allocation) + request.amount

            async with get_service_client() as client:
                # Insert strategy allocation
                strategy_response = await client.table('pendle_strategy_allocations').insert(strategy_allocation).execute()

                # Update balance
                await client.table('platform_wallet_balances').update({
                    'available_balance': str(new_available),
                    'allocated_balance': str(new_allocated),
                    'pendle_allocation': str(new_pendle_allocation),
                    'last_updated': datetime.utcnow().isoformat()
                }).eq('user_id', request.user_id).eq('token_symbol', request.token_symbol).execute()

            strategy_id = strategy_response.data[0]['id']

            # Execute the Pendle strategy
            await self._execute_pendle_strategy(strategy_id, request)

            return {
                'status': 'success',
                'strategy_id': strategy_id,
                'allocated_amount': request.amount,
                'token_symbol': request.token_symbol,
                'agent_name': request.agent_name,
                'strategy_type': request.strategy_type,
                'message': f'Successfully allocated {request.amount} {request.token_symbol} to {request.agent_name} {request.strategy_type} strategy'
            }

        except Exception as e:
            logger.error(f"Error allocating to Pendle strategy: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to allocate funds to Pendle strategy'
            }

    async def _execute_pendle_strategy(self, strategy_id: int, request: AllocationRequest):
        """Execute the actual Pendle strategy based on type."""
        try:
            pendle_service = await self._get_pendle_service()

            if request.strategy_type == StrategyType.FIXED_YIELD:
                await self._execute_fixed_yield_strategy(strategy_id, request, pendle_service)
            elif request.strategy_type == StrategyType.LIQUIDITY_PROVIDING:
                await self._execute_liquidity_strategy(strategy_id, request, pendle_service)
            elif request.strategy_type == StrategyType.YIELD_TRADING:
                await self._execute_yield_trading_strategy(strategy_id, request, pendle_service)
            else:
                raise ValueError(f"Unknown strategy type: {request.strategy_type}")

        except Exception as e:
            logger.error(f"Error executing Pendle strategy {strategy_id}: {e}")
            raise

    async def _execute_fixed_yield_strategy(self, strategy_id: int, request: AllocationRequest, pendle_service: PendleService):
        """Execute fixed yield strategy by purchasing PT tokens."""
        try:
            # Get suitable opportunities based on agent criteria
            opportunities = await pendle_service.get_sakura_opportunities(max_opportunities=10)

            if not opportunities:
                logger.warning("No suitable Pendle opportunities found for fixed yield strategy")
                return

            # Filter opportunities based on request criteria
            suitable_opportunities = [
                opp for opp in opportunities
                if (opp.expected_apy >= request.min_yield_target and
                    opp.time_to_maturity <= request.max_maturity_days and
                    opp.risk_level == request.risk_level)
            ]

            if not suitable_opportunities:
                logger.warning(f"No opportunities matching criteria: min_yield={request.min_yield_target}%, max_maturity={request.max_maturity_days} days")
                return

            # Sort by Sakura score (or yield for other agents)
            suitable_opportunities.sort(key=lambda x: x.sakura_score, reverse=True)

            # Diversify across multiple positions (max 15% each for Sakura)
            max_position_size = request.amount * 0.15  # 15% max per position for Sakura
            remaining_amount = request.amount
            position_allocations = []

            for opportunity in suitable_opportunities[:5]:  # Max 5 positions
                if remaining_amount <= 0:
                    break

                position_amount = min(max_position_size, remaining_amount)

                # Create position allocation record
                position_allocation = {
                    'strategy_allocation_id': strategy_id,
                    'user_id': request.user_id,
                    'market_address': opportunity.market.market_address,
                    'position_type': 'PT',
                    'allocated_amount': str(position_amount),
                    'entry_price': str(opportunity.current_pt_price),
                    'current_price': str(opportunity.current_pt_price),
                    'is_active': True
                }

                position_allocations.append(position_allocation)
                remaining_amount -= position_amount

            # Insert all position allocations
            if position_allocations:
                async with get_service_client() as client:
                    await client.table('pendle_position_allocations').insert(position_allocations).execute()

                logger.info(f"Created {len(position_allocations)} PT positions for strategy {strategy_id}")

        except Exception as e:
            logger.error(f"Error executing fixed yield strategy: {e}")
            raise

    async def _execute_liquidity_strategy(self, strategy_id: int, request: AllocationRequest, pendle_service: PendleService):
        """Execute liquidity providing strategy."""
        # TODO: Implement LP strategy execution
        logger.info(f"Liquidity strategy execution for strategy {strategy_id} - Implementation pending")

    async def _execute_yield_trading_strategy(self, strategy_id: int, request: AllocationRequest, pendle_service: PendleService):
        """Execute yield trading strategy by purchasing YT tokens."""
        # TODO: Implement YT strategy execution
        logger.info(f"Yield trading strategy execution for strategy {strategy_id} - Implementation pending")

    async def get_user_strategies(self, user_id: str, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active strategies for a user."""
        try:
            async with get_service_client() as client:
                query = client.table('pendle_strategy_allocations').select('*').eq('user_id', user_id).eq('is_active', True)

                if agent_name:
                    query = query.eq('agent_name', agent_name)

                response = await query.execute()

            strategies = []
            for strategy_data in response.data:
                strategy = PendleStrategyAllocation(**strategy_data)
                strategies.append({
                    **strategy.to_dict(),
                    'performance_metrics': strategy.get_performance_metrics()
                })

            return strategies

        except Exception as e:
            logger.error(f"Error fetching strategies for user {user_id}: {e}")
            return []

    async def rebalance_strategies(self, user_id: str, agent_name: Optional[str] = None):
        """Rebalance user's strategies based on performance and market conditions."""
        try:
            logger.info(f"Rebalancing strategies for user {user_id}, agent: {agent_name}")

            strategies = await self.get_user_strategies(user_id, agent_name)
            pendle_service = await self._get_pendle_service()

            for strategy in strategies:
                # Check if rebalance is needed (daily for active strategies)
                last_rebalance = datetime.fromisoformat(strategy['last_rebalance'])
                if datetime.utcnow() - last_rebalance < timedelta(hours=24):
                    continue

                # Get current market conditions
                opportunities = await pendle_service.get_sakura_opportunities(max_opportunities=5)

                # TODO: Implement rebalancing logic based on:
                # 1. Performance vs target yield
                # 2. Time to maturity
                # 3. New better opportunities
                # 4. Risk management

                logger.info(f"Rebalancing strategy {strategy['id']} - Implementation pending")

        except Exception as e:
            logger.error(f"Error rebalancing strategies: {e}")

    async def update_strategy_performance(self, strategy_id: int):
        """Update strategy performance based on current Pendle market prices."""
        try:
            # Get strategy and its positions
            async with get_service_client() as client:
                strategy_response = await client.table('pendle_strategy_allocations').select('*').eq('id', strategy_id).single().execute()

                positions_response = await client.table('pendle_position_allocations').select('*').eq('strategy_allocation_id', strategy_id).eq('is_active', True).execute()

            if not strategy_response.data:
                return

            strategy = strategy_response.data
            positions = positions_response.data

            total_current_value = 0.0
            total_unrealized_pnl = 0.0
            total_yield_earned = 0.0

            pendle_service = await self._get_pendle_service()

            # Update each position's current value
            for position in positions:
                market_address = position['market_address']

                # Get current market data
                markets = await pendle_service.get_all_markets()
                current_market = next((m for m in markets if m.market_address == market_address), None)

                if current_market:
                    # Update position current value based on position type
                    if position['position_type'] == 'PT':
                        current_price = current_market.pt_price
                    elif position['position_type'] == 'YT':
                        current_price = current_market.yt_price
                    else:
                        current_price = float(position['entry_price'])

                    allocated_amount = float(position['allocated_amount'])
                    entry_price = float(position['entry_price'])

                    current_value = allocated_amount * (current_price / entry_price)
                    unrealized_pnl = current_value - allocated_amount

                    # Calculate yield earned (for PT positions approaching maturity)
                    yield_earned = 0.0
                    if position['position_type'] == 'PT':
                        days_held = (datetime.utcnow() - datetime.fromisoformat(position['entry_timestamp'])).days
                        time_to_maturity = (current_market.maturity - datetime.utcnow()).days

                        if time_to_maturity <= 30:  # Near maturity
                            yield_earned = allocated_amount * (1 - entry_price) * (days_held / (days_held + time_to_maturity))

                    # Update position
                    await client.table('pendle_position_allocations').update({
                        'current_price': str(current_price),
                        'current_value': str(current_value),
                        'unrealized_pnl': str(unrealized_pnl),
                        'yield_earned': str(yield_earned)
                    }).eq('id', position['id']).execute()

                    total_current_value += current_value
                    total_unrealized_pnl += unrealized_pnl
                    total_yield_earned += yield_earned

            # Update strategy totals
            await client.table('pendle_strategy_allocations').update({
                'current_value': str(total_current_value),
                'unrealized_pnl': str(total_unrealized_pnl),
                'total_yield_earned': str(total_yield_earned),
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', strategy_id).execute()

            logger.info(f"Updated strategy {strategy_id} performance: value={total_current_value:.2f}, pnl={total_unrealized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error updating strategy performance {strategy_id}: {e}")


# Global service instance
_platform_wallet_service = None


def get_platform_wallet_service() -> PlatformWalletService:
    """Get or create the global platform wallet service instance."""
    global _platform_wallet_service
    if _platform_wallet_service is None:
        _platform_wallet_service = PlatformWalletService()
    return _platform_wallet_service
