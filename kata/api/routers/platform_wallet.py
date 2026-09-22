"""
Platform Wallet API endpoints for Flow AI Trading Platform.

Provides unified wallet management with integrated Pendle strategies
for all agents (Sakura, Yuki, Ryu).
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field

from kata.services.platform_wallet_service import (
    PlatformWalletService,
    AllocationRequest,
    get_platform_wallet_service
)
from kata.models.platform_wallet import AGENT_STRATEGY_CONFIGS, PENDLE_SUPPORTED_TOKENS

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/platform-wallet", tags=["Platform Wallet"])


# Request/Response models

class BalanceResponse(BaseModel):
    """Response model for wallet balance."""
    user_id: str
    platform_wallet_address: str
    token_symbol: str
    total_balance: float
    available_balance: float
    allocated_balance: float
    pendle_allocation: float
    trading_allocation: float
    lending_allocation: float
    allocation_percentages: Dict[str, float]


class PortfolioSummaryResponse(BaseModel):
    """Response model for portfolio summary."""
    total_balance: float
    available_balance: float
    allocated_balance: float
    pendle_allocation: float
    trading_allocation: float
    lending_allocation: float
    total_pnl: float
    total_yield: float
    active_strategies: int
    allocation_breakdown: Dict[str, float]
    performance_metrics: Dict[str, float]


class AllocateToStrategyRequest(BaseModel):
    """Request model for allocating funds to a strategy."""
    agent_name: str = Field(..., description="Agent name (sakura, yuki, ryu)")
    strategy_type: str = Field(..., description="Strategy type (fixed_yield, liquidity_providing, yield_trading)")
    token_symbol: str = Field(..., description="Token to allocate (USDC, ETH, cbETH, stETH)")
    amount: float = Field(..., gt=0, description="Amount to allocate")
    risk_level: str = Field("MEDIUM", description="Risk level (LOW, MEDIUM, HIGH)")
    min_yield_target: float = Field(5.0, ge=0, le=50, description="Minimum acceptable yield %")
    max_maturity_days: int = Field(90, ge=1, le=365, description="Maximum days to maturity")
    auto_reinvest: bool = Field(True, description="Enable automatic reinvestment")


class StrategyResponse(BaseModel):
    """Response model for strategy information."""
    id: int
    user_id: str
    agent_name: str
    strategy_type: str
    base_token_symbol: str
    allocated_amount: float
    risk_level: str
    min_yield_target: float
    max_maturity_days: int
    total_invested: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_yield_earned: float
    performance_metrics: Dict[str, float]
    is_active: bool
    auto_reinvest: bool
    created_at: str
    last_rebalance: str


class RebalanceRequest(BaseModel):
    """Request model for strategy rebalancing."""
    agent_name: Optional[str] = Field(None, description="Specific agent to rebalance (optional)")
    force_rebalance: bool = Field(False, description="Force rebalance even if recent")


# API Endpoints

@router.get("/balance/{user_id}", response_model=BalanceResponse)
async def get_wallet_balance(
    user_id: str = Path(..., description="User ID"),
    token_symbol: str = Query("USDC", description="Token symbol"),
    wallet_service: PlatformWalletService = Depends(get_platform_wallet_service)
):
    """
    Get user's platform wallet balance for a specific token.

    Returns detailed balance information including allocations.
    """
    try:
        logger.info(f"Fetching wallet balance for user {user_id}, token: {token_symbol}")

        balance = await wallet_service.get_user_balance(user_id, token_symbol)

        if not balance:
            raise HTTPException(status_code=404, detail=f"No {token_symbol} balance found for user")

        # Calculate allocation percentages
        allocation_percentages = {
            'pendle': balance.get_allocation_percentage('pendle'),
            'trading': balance.get_allocation_percentage('trading'),
            'lending': balance.get_allocation_percentage('lending'),
            'available': (float(balance.available_balance) / float(balance.total_balance)) * 100 if float(balance.total_balance) > 0 else 0
        }

        return BalanceResponse(
            user_id=balance.user_id,
            platform_wallet_address=balance.platform_wallet_address,
            token_symbol=balance.token_symbol,
            total_balance=float(balance.total_balance),
            available_balance=float(balance.available_balance),
            allocated_balance=float(balance.allocated_balance),
            pendle_allocation=float(balance.pendle_allocation),
            trading_allocation=float(balance.trading_allocation),
            lending_allocation=float(balance.lending_allocation),
            allocation_percentages=allocation_percentages
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching wallet balance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch wallet balance: {str(e)}")


@router.get("/portfolio/{user_id}", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    user_id: str = Path(..., description="User ID"),
    wallet_service: PlatformWalletService = Depends(get_platform_wallet_service)
):
    """
    Get comprehensive portfolio summary across all tokens and strategies.

    Returns total balances, allocations, and performance metrics.
    """
    try:
        logger.info(f"Fetching portfolio summary for user {user_id}")

        summary = await wallet_service.get_user_portfolio_summary(user_id)

        # Calculate allocation breakdown percentages
        total_balance = summary.total_balance
        allocation_breakdown = {}
        performance_metrics = {}

        if total_balance > 0:
            allocation_breakdown = {
                'pendle': (summary.pendle_allocation / total_balance) * 100,
                'trading': (summary.trading_allocation / total_balance) * 100,
                'lending': (summary.lending_allocation / total_balance) * 100,
                'available': (summary.available_balance / total_balance) * 100
            }

            performance_metrics = {
                'total_return_pct': (summary.total_pnl / total_balance) * 100 if total_balance > 0 else 0,
                'yield_return_pct': (summary.total_yield / total_balance) * 100 if total_balance > 0 else 0,
                'allocation_efficiency': (summary.allocated_balance / total_balance) * 100 if total_balance > 0 else 0
            }

        return PortfolioSummaryResponse(
            total_balance=summary.total_balance,
            available_balance=summary.available_balance,
            allocated_balance=summary.allocated_balance,
            pendle_allocation=summary.pendle_allocation,
            trading_allocation=summary.trading_allocation,
            lending_allocation=summary.lending_allocation,
            total_pnl=summary.total_pnl,
            total_yield=summary.total_yield,
            active_strategies=summary.active_strategies,
            allocation_breakdown=allocation_breakdown,
            performance_metrics=performance_metrics
        )

    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio summary: {str(e)}")


@router.post("/allocate/{user_id}")
async def allocate_to_strategy(
    user_id: str = Path(..., description="User ID"),
    request: AllocateToStrategyRequest = ...,
    wallet_service: PlatformWalletService = Depends(get_platform_wallet_service)
):
    """
    Allocate funds to a Pendle strategy for a specific agent.

    Creates strategy allocation and executes initial positions.
    """
    try:
        logger.info(f"Allocating funds for user {user_id}: {request.amount} {request.token_symbol} to {request.agent_name} {request.strategy_type}")

        # Validate agent
        if request.agent_name not in AGENT_STRATEGY_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid agent: {request.agent_name}")

        # Validate strategy for agent
        agent_config = AGENT_STRATEGY_CONFIGS[request.agent_name]
        if request.strategy_type not in agent_config['preferred_strategies']:
            logger.warning(f"Strategy {request.strategy_type} not preferred for {request.agent_name}")

        # Validate token
        if request.token_symbol not in PENDLE_SUPPORTED_TOKENS:
            raise HTTPException(status_code=400, detail=f"Token {request.token_symbol} not supported for Pendle strategies")

        # Create allocation request
        allocation_request = AllocationRequest(
            user_id=user_id,
            agent_name=request.agent_name,
            strategy_type=request.strategy_type,
            token_symbol=request.token_symbol,
            amount=request.amount,
            risk_level=request.risk_level,
            min_yield_target=request.min_yield_target,
            max_maturity_days=request.max_maturity_days,
            auto_reinvest=request.auto_reinvest
        )

        # Execute allocation
        result = await wallet_service.allocate_to_pendle_strategy(allocation_request)

        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['error'])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating to strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to allocate to strategy: {str(e)}")


@router.get("/strategies/{user_id}", response_model=List[StrategyResponse])
async def get_user_strategies(
    user_id: str = Path(..., description="User ID"),
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    wallet_service: PlatformWalletService = Depends(get_platform_wallet_service)
):
    """
    Get all active strategies for a user.

    Optionally filter by specific agent.
    """
    try:
        logger.info(f"Fetching strategies for user {user_id}, agent: {agent_name}")

        strategies = await wallet_service.get_user_strategies(user_id, agent_name)

        strategy_responses = []
        for strategy in strategies:
            strategy_responses.append(StrategyResponse(
                id=strategy['id'],
                user_id=strategy['user_id'],
                agent_name=strategy['agent_name'],
                strategy_type=strategy['strategy_type'],
                base_token_symbol=strategy['base_token_symbol'],
                allocated_amount=strategy['allocated_amount'],
                risk_level=strategy['risk_level'],
                min_yield_target=strategy['min_yield_target'],
                max_maturity_days=strategy['max_maturity_days'],
                total_invested=strategy['total_invested'],
                current_value=strategy['current_value'],
                unrealized_pnl=strategy['unrealized_pnl'],
                realized_pnl=strategy['realized_pnl'],
                total_yield_earned=strategy['total_yield_earned'],
                performance_metrics=strategy['performance_metrics'],
                is_active=strategy['is_active'],
                auto_reinvest=strategy['auto_reinvest'],
                created_at=strategy['created_at'],
                last_rebalance=strategy['last_rebalance']
            ))

        return strategy_responses

    except Exception as e:
        logger.error(f"Error fetching user strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategies: {str(e)}")


@router.post("/rebalance/{user_id}")
async def rebalance_strategies(
    user_id: str = Path(..., description="User ID"),
    request: RebalanceRequest = ...,
    wallet_service: PlatformWalletService = Depends(get_platform_wallet_service)
):
    """
    Rebalance user's strategies based on performance and market conditions.

    Optimizes allocations across different Pendle opportunities.
    """
    try:
        logger.info(f"Rebalancing strategies for user {user_id}, agent: {request.agent_name}")

        await wallet_service.rebalance_strategies(user_id, request.agent_name)

        return {
            'status': 'success',
            'message': f'Successfully initiated rebalancing for user {user_id}',
            'agent_name': request.agent_name,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error rebalancing strategies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebalance strategies: {str(e)}")


@router.get("/config")
async def get_platform_wallet_config():
    """
    Get platform wallet configuration including supported tokens and agent settings.

    Returns configuration data for frontend integration.
    """
    try:
        return {
            'supported_tokens': PENDLE_SUPPORTED_TOKENS,
            'agent_configs': AGENT_STRATEGY_CONFIGS,
            'strategy_types': {
                'fixed_yield': {
                    'name': 'Fixed Yield',
                    'description': 'Purchase PT tokens for guaranteed yield',
                    'risk_level': 'LOW',
                    'expected_apy_range': [5, 15],
                    'suitable_for': ['sakura', 'yuki', 'ryu']
                },
                'liquidity_providing': {
                    'name': 'Liquidity Providing',
                    'description': 'Provide liquidity to earn trading fees + yield',
                    'risk_level': 'MEDIUM',
                    'expected_apy_range': [8, 20],
                    'suitable_for': ['sakura', 'yuki', 'ryu']
                },
                'yield_trading': {
                    'name': 'Yield Trading',
                    'description': 'Trade YT tokens to bet on yield changes',
                    'risk_level': 'HIGH',
                    'expected_apy_range': [12, 35],
                    'suitable_for': ['yuki', 'ryu']
                }
            },
            'base_chain_id': 8453,
            'minimum_allocations': {
                'USDC': 50,
                'ETH': 0.02,
                'cbETH': 0.02,
                'stETH': 0.02
            }
        }

    except Exception as e:
        logger.error(f"Error fetching platform wallet config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch configuration: {str(e)}")


@router.get("/health")
async def platform_wallet_health_check():
    """Health check for platform wallet integration."""
    try:
        return {
            'status': 'healthy',
            'service': 'platform_wallet_integration',
            'features': [
                'Multi-agent fund management',
                'Pendle strategy allocation',
                'Real-time performance tracking',
                'Automated rebalancing',
                'Risk management',
                'Cross-token support'
            ],
            'supported_agents': list(AGENT_STRATEGY_CONFIGS.keys()),
            'supported_tokens': list(PENDLE_SUPPORTED_TOKENS.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Platform wallet health check failed: {e}")
        return {
            'status': 'unhealthy',
            'service': 'platform_wallet_integration',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }