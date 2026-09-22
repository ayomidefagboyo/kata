"""
Trade-related Pydantic models for data validation and serialization.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from pydantic import BaseModel, Field, validator


class TradeBase(BaseModel):
    """Base trade model."""
    agent_type: Optional[str] = Field(None, description="AI agent that made the trade")
    trade_type: str = Field(..., description="Trade type: buy, sell, open, close")
    token_symbol: str = Field(..., description="Token symbol")
    amount: Decimal = Field(..., gt=0, description="Trade amount")
    price_usd: Decimal = Field(..., gt=0, description="Price in USD")
    value_usd: Decimal = Field(..., gt=0, description="Total value in USD")
    protocol: str = Field(default="base", description="Protocol used")
    tx_hash: Optional[str] = Field(None, description="Transaction hash")
    reasoning: Optional[str] = Field(None, description="Trade reasoning")
    confidence_score: Optional[Decimal] = Field(None, ge=0, le=1, description="AI confidence score")
    
    @validator('trade_type')
    def validate_trade_type(cls, v):
        allowed_types = ['buy', 'sell', 'open', 'close']
        if v not in allowed_types:
            raise ValueError(f'Trade type must be one of: {allowed_types}')
        return v
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v is not None:
            allowed_agents = ['sakura', 'ryu', 'yuki', 'manual']
            if v not in allowed_agents:
                raise ValueError(f'Agent type must be one of: {allowed_agents}')
        return v


class Trade(TradeBase):
    """Complete trade model with database fields."""
    id: str = Field(..., description="Trade ID")
    user_id: str = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Trade timestamp")
    
    class Config:
        from_attributes = True


class TradeCreate(TradeBase):
    """Model for creating a new trade."""
    user_id: str = Field(..., description="User ID")


class TradeHistory(BaseModel):
    """Trade history response model."""
    trades: list[Trade] = Field(default=[], description="List of trades")
    total_count: int = Field(default=0, description="Total number of trades")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Page size")
    total_pages: int = Field(default=0, description="Total pages")


class TradeStats(BaseModel):
    """Trade statistics model."""
    total_trades: int = Field(default=0, description="Total number of trades")
    total_volume_usd: Decimal = Field(default=0, description="Total trading volume in USD")
    total_buy_volume: Decimal = Field(default=0, description="Total buy volume")
    total_sell_volume: Decimal = Field(default=0, description="Total sell volume")
    avg_trade_size: Decimal = Field(default=0, description="Average trade size")
    win_rate: Decimal = Field(default=0, description="Win rate percentage")
    total_pnl: Decimal = Field(default=0, description="Total profit/loss")
    best_trade: Decimal = Field(default=0, description="Best single trade P&L")
    worst_trade: Decimal = Field(default=0, description="Worst single trade P&L")
    avg_confidence: Optional[Decimal] = Field(None, description="Average AI confidence score")


class AgentDecisionBase(BaseModel):
    """Base agent decision model."""
    agent_type: str = Field(..., description="AI agent type")
    decision_type: str = Field(..., description="Decision type")
    market_data: Dict[str, Any] = Field(default={}, description="Market data at decision time")
    decision_data: Dict[str, Any] = Field(default={}, description="Decision data and reasoning")
    action_taken: Optional[str] = Field(None, description="Action taken based on decision")
    confidence_score: Optional[Decimal] = Field(None, ge=0, le=1, description="Decision confidence")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        allowed_agents = ['sakura', 'ryu', 'yuki']
        if v not in allowed_agents:
            raise ValueError(f'Agent type must be one of: {allowed_agents}')
        return v
    
    @validator('decision_type')
    def validate_decision_type(cls, v):
        allowed_types = [
            'buy_signal', 'sell_signal', 'hold_signal', 'risk_assessment',
            'portfolio_rebalance', 'stop_loss_trigger', 'take_profit_trigger'
        ]
        if v not in allowed_types:
            raise ValueError(f'Decision type must be one of: {allowed_types}')
        return v


class AgentDecision(AgentDecisionBase):
    """Complete agent decision model with database fields."""
    id: str = Field(..., description="Decision ID")
    user_id: str = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Decision timestamp")
    
    class Config:
        from_attributes = True


class AgentDecisionCreate(AgentDecisionBase):
    """Model for creating a new agent decision."""
    user_id: str = Field(..., description="User ID")