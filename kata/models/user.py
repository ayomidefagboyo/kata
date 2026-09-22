"""
User-related Pydantic models for data validation and serialization.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class UserBase(BaseModel):
    """Base user model with common fields."""
    external_wallet: str = Field(..., min_length=42, max_length=42, description="External wallet address")
    platform_wallet: Optional[str] = Field(None, min_length=42, max_length=42, description="Platform wallet address")
    selected_agent: str = Field(default="sakura", description="Selected AI agent")
    risk_tolerance: int = Field(default=5, ge=1, le=10, description="Risk tolerance level 1-10")
    is_active: bool = Field(default=True, description="User active status")
    credits: int = Field(default=3, ge=0, description="AI analysis credits available")
    delegation_status: str = Field(default="not_delegated", description="Cached wallet delegation status")
    delegated_wallet_address: Optional[str] = Field(None, min_length=42, max_length=42, description="Cached delegated wallet address")
    delegation_last_checked_at: Optional[datetime] = Field(None, description="Last delegation cache refresh")
    
    @validator('external_wallet', 'platform_wallet', 'delegated_wallet_address')
    def validate_wallet_address(cls, v):
        """Validate wallet address format."""
        if v and not v.startswith('0x'):
            raise ValueError('Wallet address must start with 0x')
        return v.lower() if v else v
    
    @validator('selected_agent')
    def validate_agent_type(cls, v):
        """Validate agent type."""
        allowed_agents = ['sakura', 'ryu', 'yuki']
        if v not in allowed_agents:
            raise ValueError(f'Agent must be one of: {allowed_agents}')
        return v

    @validator('delegation_status')
    def validate_delegation_status(cls, v):
        """Validate cached delegation status."""
        allowed_statuses = ['not_delegated', 'delegated', 'revoked', 'unknown']
        if v not in allowed_statuses:
            raise ValueError(f'Delegation status must be one of: {allowed_statuses}')
        return v


class UserCreate(UserBase):
    """Model for creating a new user."""
    pass


class UserUpdate(BaseModel):
    """Model for updating user data."""
    platform_wallet: Optional[str] = Field(None, min_length=42, max_length=42)
    selected_agent: Optional[str] = None
    risk_tolerance: Optional[int] = Field(None, ge=1, le=10)
    is_active: Optional[bool] = None
    credits: Optional[int] = Field(None, ge=0)
    delegation_status: Optional[str] = None
    delegated_wallet_address: Optional[str] = Field(None, min_length=42, max_length=42)
    delegation_last_checked_at: Optional[datetime] = None
    
    @validator('platform_wallet', 'delegated_wallet_address')
    def validate_wallet_address(cls, v):
        if v and not v.startswith('0x'):
            raise ValueError('Wallet address must start with 0x')
        return v.lower() if v else v

    @validator('delegation_status')
    def validate_delegation_status(cls, v):
        if v is not None:
            allowed_statuses = ['not_delegated', 'delegated', 'revoked', 'unknown']
            if v not in allowed_statuses:
                raise ValueError(f'Delegation status must be one of: {allowed_statuses}')
        return v
    
    @validator('selected_agent')
    def validate_agent_type(cls, v):
        if v is not None:
            allowed_agents = ['sakura', 'ryu', 'yuki']
            if v not in allowed_agents:
                raise ValueError(f'Agent must be one of: {allowed_agents}')
        return v


class User(UserBase):
    """Complete user model with database fields."""
    id: str = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Model for user login request."""
    wallet_address: str = Field(..., min_length=42, max_length=42, description="Wallet address for login")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        if not v.startswith('0x'):
            raise ValueError('Wallet address must start with 0x')
        return v.lower()


class UserLoginResponse(BaseModel):
    """Model for user login response."""
    user: User
    access_token: Optional[str] = None
    token_type: str = "bearer"
    message: str = "Login successful"


class CreditTransaction(BaseModel):
    """Model for credit transactions."""
    user_id: str
    amount: int = Field(..., description="Credits amount (positive for add, negative for deduct)")
    transaction_type: str = Field(..., description="Type: purchase, usage, bonus, refund")
    description: str = Field(..., description="Transaction description")
    service_used: Optional[str] = Field(None, description="Service that used credits (trading_scanner, token_analysis)")
    created_at: datetime = Field(default_factory=datetime.now)


class CreditBalance(BaseModel):
    """Model for credit balance response."""
    user_id: str
    credits: int = Field(..., ge=0, description="Current credit balance")
    last_updated: datetime
