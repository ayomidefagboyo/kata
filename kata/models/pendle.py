"""
Pendle Finance database models for Flow AI Trading Platform.

Models for storing Pendle market data, positions, and yield opportunities.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from typing import Optional, Dict, Any

from kata.models.base import BaseModel

Base = declarative_base()


class PendleMarket(BaseModel):
    """
    Pendle market data storage.

    Stores information about PT/YT markets for yield analysis.
    """
    __tablename__ = 'pendle_markets'

    # Market identifiers
    market_address = Column(String(42), unique=True, nullable=False, index=True)
    pt_address = Column(String(42), nullable=False)
    yt_address = Column(String(42), nullable=False)
    sy_address = Column(String(42), nullable=False)

    # Asset information
    underlying_asset = Column(String(100), nullable=False)
    underlying_symbol = Column(String(20), nullable=False, index=True)
    protocol_name = Column(String(50), default='')

    # Market data
    maturity = Column(DateTime, nullable=False, index=True)
    implied_apy = Column(Float, default=0.0)
    pt_price = Column(Float, default=0.0)
    yt_price = Column(Float, default=0.0)
    liquidity_usd = Column(Float, default=0.0, index=True)

    # Configuration
    chain_id = Column(Integer, default=8453)  # Base
    is_active = Column(Boolean, default=True, index=True)
    min_trade_size = Column(Float, default=100.0)

    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data_source = Column(String(50), default='pendle_api')

    def __repr__(self):
        return f"<PendleMarket({self.underlying_symbol}, maturity={self.maturity}, apy={self.implied_apy:.2f}%)>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'market_address': self.market_address,
            'pt_address': self.pt_address,
            'yt_address': self.yt_address,
            'sy_address': self.sy_address,
            'underlying_asset': self.underlying_asset,
            'underlying_symbol': self.underlying_symbol,
            'protocol_name': self.protocol_name,
            'maturity': self.maturity.isoformat() if self.maturity else None,
            'implied_apy': self.implied_apy,
            'pt_price': self.pt_price,
            'yt_price': self.yt_price,
            'liquidity_usd': self.liquidity_usd,
            'chain_id': self.chain_id,
            'is_active': self.is_active,
            'min_trade_size': self.min_trade_size,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class PendlePosition(BaseModel):
    """
    User positions in Pendle markets.

    Tracks PT/YT holdings and performance.
    """
    __tablename__ = 'pendle_positions'

    # Position identifiers
    user_id = Column(String(255), nullable=False, index=True)
    market_address = Column(String(42), ForeignKey('pendle_markets.market_address'), nullable=False)
    position_type = Column(String(20), nullable=False)  # 'PT', 'YT', 'LP'

    # Position details
    token_address = Column(String(42), nullable=False)  # PT or YT address
    amount = Column(Float, nullable=False)  # Token amount
    entry_price = Column(Float, nullable=False)  # Price when position opened
    entry_value_usd = Column(Float, nullable=False)  # USD value at entry

    # Performance tracking
    current_price = Column(Float, default=0.0)
    current_value_usd = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_percent = Column(Float, default=0.0)

    # Yield tracking (for PT positions)
    expected_yield = Column(Float, default=0.0)  # Expected APY at entry
    accrued_yield = Column(Float, default=0.0)   # Yield earned so far
    days_held = Column(Integer, default=0)

    # Strategy metadata
    strategy_type = Column(String(50), default='fixed_yield')  # Strategy used
    agent_name = Column(String(50), default='sakura')         # Agent that created position
    confidence_score = Column(Float, default=0.0)             # Confidence at entry

    # Position management
    is_active = Column(Boolean, default=True, index=True)
    entry_timestamp = Column(DateTime, default=datetime.utcnow)
    exit_timestamp = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_value_usd = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)

    # Relationships
    market = relationship("PendleMarket", backref="positions")

    def __repr__(self):
        return f"<PendlePosition({self.user_id}, {self.position_type}, {self.amount:.2f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'market_address': self.market_address,
            'position_type': self.position_type,
            'token_address': self.token_address,
            'amount': self.amount,
            'entry_price': self.entry_price,
            'entry_value_usd': self.entry_value_usd,
            'current_price': self.current_price,
            'current_value_usd': self.current_value_usd,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_percent': self.unrealized_pnl_percent,
            'expected_yield': self.expected_yield,
            'accrued_yield': self.accrued_yield,
            'days_held': self.days_held,
            'strategy_type': self.strategy_type,
            'agent_name': self.agent_name,
            'confidence_score': self.confidence_score,
            'is_active': self.is_active,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'exit_price': self.exit_price,
            'exit_value_usd': self.exit_value_usd,
            'realized_pnl': self.realized_pnl
        }


class PendleYieldAnalysis(BaseModel):
    """
    Historical yield analysis for Pendle opportunities.

    Stores Sakura agent's analysis of yield opportunities.
    """
    __tablename__ = 'pendle_yield_analysis'

    # Analysis identifiers
    market_address = Column(String(42), ForeignKey('pendle_markets.market_address'), nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    agent_name = Column(String(50), default='sakura')

    # Opportunity assessment
    strategy_type = Column(String(50), default='fixed_yield')
    expected_apy = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    time_to_maturity = Column(Integer, nullable=False)  # days
    sakura_score = Column(Float, nullable=False)  # 0-1 compatibility score

    # Pricing analysis
    pt_price_at_analysis = Column(Float, nullable=False)
    discount_to_maturity = Column(Float, nullable=False)
    break_even_days = Column(Integer, nullable=False)

    # Risk metrics
    liquidity_score = Column(Float, default=0.0)
    volatility_score = Column(Float, default=0.0)
    protocol_risk_score = Column(Float, default=0.0)

    # Market conditions
    market_conditions = Column(JSON, default=dict)  # Store market context
    technical_indicators = Column(JSON, default=dict)  # Technical analysis

    # Analysis metadata
    confidence_level = Column(Float, default=0.0)
    recommended_allocation = Column(Float, default=0.0)  # Percentage of portfolio
    notes = Column(Text, default='')

    # Relationships
    market = relationship("PendleMarket", backref="yield_analyses")

    def __repr__(self):
        return f"<PendleYieldAnalysis({self.market_address}, {self.expected_apy:.2f}% APY, score={self.sakura_score:.3f})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'market_address': self.market_address,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'agent_name': self.agent_name,
            'strategy_type': self.strategy_type,
            'expected_apy': self.expected_apy,
            'risk_level': self.risk_level,
            'time_to_maturity': self.time_to_maturity,
            'sakura_score': self.sakura_score,
            'pt_price_at_analysis': self.pt_price_at_analysis,
            'discount_to_maturity': self.discount_to_maturity,
            'break_even_days': self.break_even_days,
            'liquidity_score': self.liquidity_score,
            'volatility_score': self.volatility_score,
            'protocol_risk_score': self.protocol_risk_score,
            'market_conditions': self.market_conditions,
            'technical_indicators': self.technical_indicators,
            'confidence_level': self.confidence_level,
            'recommended_allocation': self.recommended_allocation,
            'notes': self.notes
        }


class PendleTransaction(BaseModel):
    """
    Pendle transaction history.

    Records all Pendle-related transactions for audit and analysis.
    """
    __tablename__ = 'pendle_transactions'

    # Transaction identifiers
    user_id = Column(String(255), nullable=False, index=True)
    transaction_hash = Column(String(66), unique=True, nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('pendle_positions.id'), nullable=True)

    # Transaction details
    transaction_type = Column(String(50), nullable=False)  # 'buy_pt', 'sell_pt', 'claim_yield', etc.
    market_address = Column(String(42), nullable=False)
    token_in_address = Column(String(42), nullable=False)
    token_out_address = Column(String(42), nullable=False)

    # Amounts
    amount_in = Column(Float, nullable=False)
    amount_out = Column(Float, nullable=False)
    amount_in_usd = Column(Float, default=0.0)
    amount_out_usd = Column(Float, default=0.0)

    # Transaction metadata
    gas_used = Column(Integer, default=0)
    gas_price = Column(Float, default=0.0)
    transaction_fee_usd = Column(Float, default=0.0)
    block_number = Column(Integer, default=0)

    # Status
    status = Column(String(20), default='pending')  # pending, confirmed, failed
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confirmed_at = Column(DateTime, nullable=True)

    # Strategy context
    agent_name = Column(String(50), default='sakura')
    strategy_type = Column(String(50), default='fixed_yield')

    # Relationships
    position = relationship("PendlePosition", backref="transactions")

    def __repr__(self):
        return f"<PendleTransaction({self.transaction_type}, {self.amount_in_usd:.2f} USD)>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'transaction_hash': self.transaction_hash,
            'position_id': self.position_id,
            'transaction_type': self.transaction_type,
            'market_address': self.market_address,
            'token_in_address': self.token_in_address,
            'token_out_address': self.token_out_address,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'amount_in_usd': self.amount_in_usd,
            'amount_out_usd': self.amount_out_usd,
            'gas_used': self.gas_used,
            'gas_price': self.gas_price,
            'transaction_fee_usd': self.transaction_fee_usd,
            'block_number': self.block_number,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
            'agent_name': self.agent_name,
            'strategy_type': self.strategy_type
        }