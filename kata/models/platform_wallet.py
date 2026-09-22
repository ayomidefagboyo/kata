"""
Platform Wallet database models for Flow AI Trading Platform.

Models for managing user funds, allocations, and strategy configurations
across all agents (Sakura, Yuki, Ryu) with integrated Pendle support.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey, DECIMAL
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

Base = declarative_base()


class StrategyType(str, Enum):
    """Available strategy types across all agents."""
    FIXED_YIELD = "fixed_yield"
    LIQUIDITY_PROVIDING = "liquidity_providing"
    YIELD_TRADING = "yield_trading"
    SPOT_TRADING = "spot_trading"
    ARBITRAGE = "arbitrage"
    LENDING = "lending"


class RiskLevel(str, Enum):
    """Risk levels for strategies."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PlatformWalletBalance(Base):
    """
    Tracks user funds in platform wallets with strategy allocations.

    This is the core balance tracking for the unified wallet system.
    """
    __tablename__ = 'platform_wallet_balances'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # User identification
    user_id = Column(String(255), nullable=False, index=True)
    platform_wallet_address = Column(String(42), nullable=False, index=True)

    # Asset tracking
    token_symbol = Column(String(20), nullable=False, index=True)
    token_address = Column(String(42), nullable=False)
    chain_id = Column(Integer, default=8453)  # Base

    # Balance tracking (using DECIMAL for precision)
    total_balance = Column(DECIMAL(28, 18), nullable=False, default=0)
    available_balance = Column(DECIMAL(28, 18), nullable=False, default=0)
    allocated_balance = Column(DECIMAL(28, 18), nullable=False, default=0)

    # Strategy allocations
    pendle_allocation = Column(DECIMAL(28, 18), default=0)
    trading_allocation = Column(DECIMAL(28, 18), default=0)
    lending_allocation = Column(DECIMAL(28, 18), default=0)

    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<PlatformWalletBalance({self.user_id}, {self.token_symbol}: {float(self.total_balance):.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'platform_wallet_address': self.platform_wallet_address,
            'token_symbol': self.token_symbol,
            'token_address': self.token_address,
            'chain_id': self.chain_id,
            'total_balance': float(self.total_balance),
            'available_balance': float(self.available_balance),
            'allocated_balance': float(self.allocated_balance),
            'pendle_allocation': float(self.pendle_allocation),
            'trading_allocation': float(self.trading_allocation),
            'lending_allocation': float(self.lending_allocation),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

    def get_allocation_percentage(self, strategy: str) -> float:
        """Get percentage of total balance allocated to strategy."""
        if float(self.total_balance) == 0:
            return 0.0

        allocation_map = {
            'pendle': float(self.pendle_allocation),
            'trading': float(self.trading_allocation),
            'lending': float(self.lending_allocation)
        }

        return (allocation_map.get(strategy, 0) / float(self.total_balance)) * 100

    def can_allocate(self, amount: float, strategy: str = None) -> bool:
        """Check if amount can be allocated."""
        return float(self.available_balance) >= amount


class PendleStrategyAllocation(Base):
    """
    Manages agent-specific Pendle strategy configurations and performance.

    Each agent (Sakura, Yuki, Ryu) can have different Pendle strategies.
    """
    __tablename__ = 'pendle_strategy_allocations'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # User and wallet info
    user_id = Column(String(255), nullable=False, index=True)
    platform_wallet_address = Column(String(42), nullable=False)

    # Strategy configuration
    agent_name = Column(String(50), nullable=False, index=True)  # sakura, yuki, ryu
    strategy_type = Column(String(50), nullable=False)  # fixed_yield, liquidity_providing, yield_trading

    # Asset allocation
    base_token_symbol = Column(String(20), nullable=False)
    base_token_address = Column(String(42), nullable=False)
    allocated_amount = Column(DECIMAL(28, 18), nullable=False)

    # Strategy parameters
    risk_level = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    min_yield_target = Column(DECIMAL(8, 4), nullable=False)  # Minimum acceptable yield
    max_maturity_days = Column(Integer, nullable=False)  # Maximum days to maturity
    max_single_position_pct = Column(DECIMAL(5, 2), default=15.00)  # Max % in single position

    # Performance tracking
    total_invested = Column(DECIMAL(28, 18), default=0)
    current_value = Column(DECIMAL(28, 18), default=0)
    unrealized_pnl = Column(DECIMAL(28, 18), default=0)
    realized_pnl = Column(DECIMAL(28, 18), default=0)
    total_yield_earned = Column(DECIMAL(28, 18), default=0)

    # Status
    is_active = Column(Boolean, default=True, index=True)
    auto_reinvest = Column(Boolean, default=True)

    # Timestamps
    last_rebalance = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PendleStrategyAllocation({self.agent_name}, {self.strategy_type}, {float(self.allocated_amount):.2f} {self.base_token_symbol})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'platform_wallet_address': self.platform_wallet_address,
            'agent_name': self.agent_name,
            'strategy_type': self.strategy_type,
            'base_token_symbol': self.base_token_symbol,
            'base_token_address': self.base_token_address,
            'allocated_amount': float(self.allocated_amount),
            'risk_level': self.risk_level,
            'min_yield_target': float(self.min_yield_target),
            'max_maturity_days': self.max_maturity_days,
            'max_single_position_pct': float(self.max_single_position_pct),
            'total_invested': float(self.total_invested),
            'current_value': float(self.current_value),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'total_yield_earned': float(self.total_yield_earned),
            'is_active': self.is_active,
            'auto_reinvest': self.auto_reinvest,
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_invested = float(self.total_invested)
        current_value = float(self.current_value)

        if total_invested == 0:
            return {
                'total_return_pct': 0.0,
                'unrealized_return_pct': 0.0,
                'realized_return_pct': 0.0,
                'total_yield_pct': 0.0
            }

        return {
            'total_return_pct': ((current_value + float(self.realized_pnl)) / total_invested - 1) * 100,
            'unrealized_return_pct': (float(self.unrealized_pnl) / total_invested) * 100,
            'realized_return_pct': (float(self.realized_pnl) / total_invested) * 100,
            'total_yield_pct': (float(self.total_yield_earned) / total_invested) * 100
        }


class PendlePositionAllocation(Base):
    """
    Links individual Pendle positions to strategy allocations.

    Tracks how platform wallet funds are allocated to specific Pendle positions.
    """
    __tablename__ = 'pendle_position_allocations'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Link to strategy
    strategy_allocation_id = Column(Integer, ForeignKey('pendle_strategy_allocations.id'), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)

    # Position details
    pendle_position_id = Column(Integer, ForeignKey('pendle_positions.id'), nullable=True)
    market_address = Column(String(42), ForeignKey('pendle_markets.market_address'), nullable=False)
    position_type = Column(String(20), nullable=False)  # PT, YT, LP

    # Allocation tracking
    allocated_amount = Column(DECIMAL(28, 18), nullable=False)
    current_value = Column(DECIMAL(28, 18), default=0)
    entry_price = Column(DECIMAL(18, 8), nullable=False)
    current_price = Column(DECIMAL(18, 8), default=0)

    # Performance
    unrealized_pnl = Column(DECIMAL(28, 18), default=0)
    yield_earned = Column(DECIMAL(28, 18), default=0)

    # Status
    is_active = Column(Boolean, default=True, index=True)
    entry_timestamp = Column(DateTime, default=datetime.utcnow)
    exit_timestamp = Column(DateTime, nullable=True)

    # Relationships
    strategy = relationship("PendleStrategyAllocation", backref="position_allocations")
    pendle_position = relationship("PendlePosition", backref="allocations")
    market = relationship("PendleMarket", backref="allocations")

    def __repr__(self):
        return f"<PendlePositionAllocation({self.position_type}, {float(self.allocated_amount):.2f}, PnL: {float(self.unrealized_pnl):.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'strategy_allocation_id': self.strategy_allocation_id,
            'user_id': self.user_id,
            'pendle_position_id': self.pendle_position_id,
            'market_address': self.market_address,
            'position_type': self.position_type,
            'allocated_amount': float(self.allocated_amount),
            'current_value': float(self.current_value),
            'entry_price': float(self.entry_price),
            'current_price': float(self.current_price),
            'unrealized_pnl': float(self.unrealized_pnl),
            'yield_earned': float(self.yield_earned),
            'is_active': self.is_active,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None
        }


class DepositTransaction(Base):
    """
    Tracks all deposits to platform wallets with allocation intent.

    Records user deposits and their intended use across different agents/strategies.
    """
    __tablename__ = 'deposit_transactions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # User info
    user_id = Column(String(255), nullable=False, index=True)
    external_wallet = Column(String(42), nullable=False)  # User's connected wallet
    platform_wallet = Column(String(42), nullable=False, index=True)  # Platform wallet address

    # Transaction details
    transaction_hash = Column(String(66), unique=True, nullable=False, index=True)
    token_symbol = Column(String(20), nullable=False)
    token_address = Column(String(42), nullable=False)
    amount = Column(DECIMAL(28, 18), nullable=False)
    amount_usd = Column(DECIMAL(20, 8), default=0)

    # Status tracking
    status = Column(String(20), default='pending', index=True)  # pending, confirmed, failed
    block_number = Column(Integer, default=0)
    confirmations = Column(Integer, default=0)

    # Agent allocation intent
    intended_agent = Column(String(50), nullable=True)  # Which agent this deposit is for
    intended_strategy = Column(String(50), nullable=True)  # Which strategy to allocate to
    auto_allocate = Column(Boolean, default=False)  # Auto-allocate to strategies

    # Timestamps
    deposit_timestamp = Column(DateTime, default=datetime.utcnow)
    confirmed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<DepositTransaction({self.user_id}, {float(self.amount):.2f} {self.token_symbol}, {self.status})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'external_wallet': self.external_wallet,
            'platform_wallet': self.platform_wallet,
            'transaction_hash': self.transaction_hash,
            'token_symbol': self.token_symbol,
            'token_address': self.token_address,
            'amount': float(self.amount),
            'amount_usd': float(self.amount_usd),
            'status': self.status,
            'block_number': self.block_number,
            'confirmations': self.confirmations,
            'intended_agent': self.intended_agent,
            'intended_strategy': self.intended_strategy,
            'auto_allocate': self.auto_allocate,
            'deposit_timestamp': self.deposit_timestamp.isoformat() if self.deposit_timestamp else None,
            'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None
        }


# Agent-specific strategy configurations with 100% specialization
AGENT_STRATEGY_CONFIGS = {
    'sakura': {
        'max_pendle_allocation_pct': 100.0,  # 100% - Pendle fixed yield specialist
        'max_hyperliquid_allocation_pct': 0.0,
        'max_spot_allocation_pct': 0.0,
        'preferred_strategies': ['fixed_yield', 'liquidity_providing'],
        'risk_tolerance': 'LOW',
        'min_yield_target': 5.0,
        'max_maturity_days': 180,
        'auto_reinvest': True
    },
    'yuki': {
        'max_pendle_allocation_pct': 0.0,  # 0% - Hyperliquid futures specialist
        'max_hyperliquid_allocation_pct': 100.0,  # 100% futures allocation
        'max_spot_allocation_pct': 0.0,
        'preferred_strategies': ['futures_trading', 'perpetual_swaps', 'arbitrage'],
        'risk_tolerance': 'HIGH',
        'min_yield_target': 15.0,
        'max_leverage': 10,
        'auto_reinvest': True
    },
    'ryu': {
        'max_pendle_allocation_pct': 0.0,  # 0% - Spot trading specialist
        'max_hyperliquid_allocation_pct': 0.0,
        'max_spot_allocation_pct': 100.0,  # 100% spot trading allocation
        'preferred_strategies': ['spot_trading', 'swing_trading', 'dca_strategy'],
        'risk_tolerance': 'MEDIUM',
        'min_yield_target': 8.0,
        'max_drawdown': 15,
        'auto_reinvest': False  # Manual control for spot strategies
    }
}

# Supported tokens for Pendle strategies on Base
PENDLE_SUPPORTED_TOKENS = {
    'USDC': {
        'address': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
        'decimals': 6,
        'strategies': ['fixed_yield', 'liquidity_providing', 'yield_trading']
    },
    'ETH': {
        'address': '0x0000000000000000000000000000000000000000',
        'decimals': 18,
        'strategies': ['fixed_yield', 'liquidity_providing']
    },
    'cbETH': {
        'address': '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',
        'decimals': 18,
        'strategies': ['fixed_yield', 'liquidity_providing', 'yield_trading']
    },
    'stETH': {
        'address': '0xc1CBa3fCea344f92D9239c08C0568f6F2F0ee452',
        'decimals': 18,
        'strategies': ['fixed_yield', 'liquidity_providing', 'yield_trading']
    }
}