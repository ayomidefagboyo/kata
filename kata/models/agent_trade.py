"""
Agent Trade Models

Database models for storing agent trading data including trades, positions, and performance metrics.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
from enum import Enum

Base = declarative_base()


class TradeStatus(Enum):
    """Trade status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CLOSED = "closed"


class TradeType(Enum):
    """Trade type enumeration."""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    CLOSE = "close"


class AgentType(Enum):
    """Agent type enumeration."""
    YUKI = "yuki"
    SAKURA = "sakura"
    RYU = "ryu"


class AgentTrade(Base):
    """Store agent trades with comprehensive tracking."""
    __tablename__ = "agent_trades"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    user_id = Column(String(255), nullable=False, index=True)
    allocation_id = Column(String(255), nullable=False, index=True)
    
    # Agent information
    agent_type = Column(SQLEnum(AgentType), nullable=False, index=True)
    
    # Trade details
    trade_type = Column(SQLEnum(TradeType), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    
    # Price and size
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    position_size = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False, default=1.0)
    
    # Financial tracking
    trade_amount = Column(Float, nullable=False)  # USD amount used
    realized_pnl = Column(Float, nullable=True)  # PnL when closed
    unrealized_pnl = Column(Float, nullable=True)  # Current PnL if open
    fees = Column(Float, nullable=True, default=0.0)
    
    # Status and timing
    status = Column(SQLEnum(TradeStatus), nullable=False, default=TradeStatus.PENDING)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    # External references
    hyperliquid_order_id = Column(String(100), nullable=True, index=True)
    tx_hash = Column(String(66), nullable=True, index=True)
    
    # Metadata
    signal_confidence = Column(Float, nullable=True)
    signal_reasoning = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    trade_metadata = Column(JSONB, nullable=True)  # Additional trade data
    
    # Performance tracking
    max_profit_reached = Column(Float, nullable=True)
    max_loss_reached = Column(Float, nullable=True)
    time_in_position_minutes = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<AgentTrade({self.id}, {self.agent_type.value}, {self.symbol}, {self.side})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "allocation_id": self.allocation_id,
            "agent_type": self.agent_type.value,
            "trade_type": self.trade_type.value,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "trade_amount": self.trade_amount,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees": self.fees,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "hyperliquid_order_id": self.hyperliquid_order_id,
            "tx_hash": self.tx_hash,
            "signal_confidence": self.signal_confidence,
            "signal_reasoning": self.signal_reasoning,
            "error_message": self.error_message,
            "trade_metadata": self.trade_metadata,
            "max_profit_reached": self.max_profit_reached,
            "max_loss_reached": self.max_loss_reached,
            "time_in_position_minutes": self.time_in_position_minutes
        }


class AgentPosition(Base):
    """Store current agent positions for real-time tracking."""
    __tablename__ = "agent_positions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    user_id = Column(String(255), nullable=False, index=True)
    allocation_id = Column(String(255), nullable=False, index=True)
    trade_id = Column(UUID(as_uuid=True), ForeignKey('agent_trades.id'), nullable=False)
    
    # Position details
    agent_type = Column(SQLEnum(AgentType), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    
    # Position metrics
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False)
    
    # Financial tracking
    position_value = Column(Float, nullable=False)  # Current value
    unrealized_pnl = Column(Float, nullable=False)  # Current PnL
    unrealized_pnl_percent = Column(Float, nullable=False)  # PnL percentage
    margin_used = Column(Float, nullable=False)  # Margin required
    
    # Risk metrics
    liquidation_price = Column(Float, nullable=True)
    margin_ratio = Column(Float, nullable=True)  # Margin ratio for liquidation risk
    
    # Status and timing
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # External references
    hyperliquid_position_id = Column(String(100), nullable=True)
    
    # Metadata
    position_metadata = Column(JSONB, nullable=True)
    
    # Relationships
    trade = relationship("AgentTrade", backref="positions")
    
    def __repr__(self):
        return f"<AgentPosition({self.id}, {self.agent_type.value}, {self.symbol}, {self.side})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "allocation_id": self.allocation_id,
            "trade_id": str(self.trade_id),
            "agent_type": self.agent_type.value,
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "leverage": self.leverage,
            "position_value": self.position_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "margin_used": self.margin_used,
            "liquidation_price": self.liquidation_price,
            "margin_ratio": self.margin_ratio,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "hyperliquid_position_id": self.hyperliquid_position_id,
            "position_metadata": self.position_metadata
        }


class AgentPerformance(Base):
    """Store agent performance metrics and statistics."""
    __tablename__ = "agent_performance"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    user_id = Column(String(255), nullable=False, index=True)
    allocation_id = Column(String(255), nullable=False, index=True)
    
    # Performance period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly'
    
    # Financial metrics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=False, default=0.0)
    
    total_pnl = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    
    total_volume = Column(Float, nullable=False, default=0.0)
    average_trade_size = Column(Float, nullable=False, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    
    # Agent-specific metrics
    agent_type = Column(SQLEnum(AgentType), nullable=False, index=True)
    max_leverage_used = Column(Float, nullable=False, default=1.0)
    average_leverage = Column(Float, nullable=False, default=1.0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    position_metadata = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<AgentPerformance({self.id}, {self.agent_type.value}, {self.period_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "allocation_id": self.allocation_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_type": self.period_type,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_volume": self.total_volume,
            "average_trade_size": self.average_trade_size,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "agent_type": self.agent_type.value,
            "max_leverage_used": self.max_leverage_used,
            "average_leverage": self.average_leverage,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "position_metadata": self.position_metadata
        }
