"""
Portfolio-related Pydantic models for data validation and serialization.
"""
from datetime import datetime
from typing import List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field


class HoldingBase(BaseModel):
    """Base holding model."""
    token_address: str = Field(..., description="Token contract address")
    token_symbol: str = Field(..., description="Token symbol")
    amount: Decimal = Field(..., ge=0, description="Token amount")
    value_usd: Decimal = Field(..., ge=0, description="Value in USD")
    pnl_usd: Decimal = Field(default=0, description="Profit/Loss in USD")
    protocol: str = Field(default="base", description="Protocol name")
    position_type: str = Field(default="spot", description="Position type")


class Holding(HoldingBase):
    """Complete holding model with database fields."""
    id: str = Field(..., description="Holding ID")
    portfolio_id: str = Field(..., description="Portfolio ID")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class HoldingCreate(HoldingBase):
    """Model for creating a new holding."""
    portfolio_id: str = Field(..., description="Portfolio ID")


class HoldingUpdate(BaseModel):
    """Model for updating holding data."""
    amount: Optional[Decimal] = Field(None, ge=0)
    value_usd: Optional[Decimal] = Field(None, ge=0)
    pnl_usd: Optional[Decimal] = None


class PortfolioBase(BaseModel):
    """Base portfolio model."""
    total_value_usd: Decimal = Field(default=0, ge=0, description="Total portfolio value in USD")
    total_pnl_usd: Decimal = Field(default=0, description="Total P&L in USD")
    total_pnl_percent: Decimal = Field(default=0, description="Total P&L percentage")


class Portfolio(PortfolioBase):
    """Complete portfolio model with database fields."""
    id: str = Field(..., description="Portfolio ID")
    user_id: str = Field(..., description="User ID")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class PortfolioSummary(Portfolio):
    """Portfolio summary with holdings and additional data."""
    holdings: List[Holding] = Field(default=[], description="Portfolio holdings")
    total_holdings: int = Field(default=0, description="Number of holdings")
    active_agent_type: Optional[str] = None
    agent_is_active: Optional[bool] = None


class PortfolioUpdate(BaseModel):
    """Model for updating portfolio data."""
    total_value_usd: Optional[Decimal] = Field(None, ge=0)
    total_pnl_usd: Optional[Decimal] = None
    total_pnl_percent: Optional[Decimal] = None


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics."""
    daily_change_usd: Decimal = Field(default=0, description="Daily change in USD")
    daily_change_percent: Decimal = Field(default=0, description="Daily change percentage")
    weekly_change_usd: Decimal = Field(default=0, description="Weekly change in USD")
    weekly_change_percent: Decimal = Field(default=0, description="Weekly change percentage")
    monthly_change_usd: Decimal = Field(default=0, description="Monthly change in USD")
    monthly_change_percent: Decimal = Field(default=0, description="Monthly change percentage")
    all_time_high: Decimal = Field(default=0, description="All-time high value")
    all_time_low: Decimal = Field(default=0, description="All-time low value")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[Decimal] = Field(None, description="Maximum drawdown percentage")