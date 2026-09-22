"""
Token Discovery Data Models for Flow AI Trading Platform.

Defines the data structures for storing analyzed tokens and their analysis results.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DiscoveredToken(BaseModel):
    """Model for discovered tokens with analysis and market data."""
    
    # Basic token info
    id: str
    symbol: str
    name: str
    blockchain: str
    chain_id: int
    address: str
    logo_url: Optional[str] = None
    coingecko_id: Optional[str] = None
    
    # Analysis data (updated daily)
    analysis: Optional['TokenAnalysis'] = None
    discovery_date: datetime
    
    # Real-time market data (updated frequently)
    current_price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    liquidity: float
    bid_depth_2pct: float = 0.0
    ask_depth_2pct: float = 0.0
    total_depth_2pct: float = 0.0
    liquidity_quality: str = 'UNKNOWN'
    liquidity_score: float = 0.0
    last_updated: datetime
    
    # Status flags
    is_trending: bool = False
    is_active: bool = True
    
    class Config:
        from_attributes = True


class TokenAnalysis(BaseModel):
    """Model for AI analysis results of tokens."""
    
    id: str
    token_id: str
    analysis_type: str = "daily_discovery"  # daily_discovery, on_demand, etc.
    
    # Analysis scores (0-10 scale)
    risk_score: float = Field(..., ge=0, le=10)
    potential_score: float = Field(..., ge=0, le=10)
    confidence_score: float = Field(..., ge=0, le=1)
    overall_score: float = Field(..., ge=0, le=10)
    
    # Analysis details
    summary: str
    signals: List[str] = []
    risk_factors: List[str] = []
    opportunities: List[str] = []
    
    # Technical analysis
    technical_indicators: dict = {}
    sentiment_score: float = Field(default=0.5, ge=0, le=1)
    social_metrics: dict = {}
    
    # Metadata
    analysis_version: str = "1.0"
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TokenMarketData(BaseModel):
    """Model for real-time market data."""
    
    token_id: str
    source: str  # coingecko, dexscreener, etc.
    
    # Price data
    current_price: float
    price_change_1h: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_7d: Optional[float] = None
    
    # Volume and liquidity
    volume_24h: float
    market_cap: float
    liquidity: float
    bid_depth_2pct: float = 0.0
    ask_depth_2pct: float = 0.0
    total_depth_2pct: float = 0.0
    liquidity_quality: str = 'UNKNOWN'
    liquidity_score: float = 0.0
    
    # Trading data
    trading_pairs_count: int = 0
    top_exchanges: List[str] = []
    
    # Timestamps
    fetched_at: datetime
    
    class Config:
        from_attributes = True


class TokenDiscoveryFilters(BaseModel):
    """Model for token discovery filters."""
    
    blockchain: Optional[str] = None
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    min_volume: Optional[float] = None
    min_liquidity: Optional[float] = None
    min_bid_depth_2pct: Optional[float] = None
    min_ask_depth_2pct: Optional[float] = None
    min_total_depth_2pct: Optional[float] = None
    min_liquidity_score: Optional[float] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    min_potential_score: Optional[float] = None
    trending_only: bool = False
    analyzed_only: bool = False
    
    class Config:
        from_attributes = True


class DailyAnalysisResult(BaseModel):
    """Model for daily analysis batch results."""
    
    id: str
    analysis_date: datetime
    tokens_analyzed: int
    new_discoveries: int
    analysis_duration: float  # seconds
    
    # Summary statistics
    avg_risk_score: float
    avg_potential_score: float
    top_discoveries: List[str] = []  # token IDs
    
    # Status
    status: str  # completed, failed, in_progress
    error_message: Optional[str] = None
    
    created_at: datetime
    
    class Config:
        from_attributes = True


# Database table models (SQLAlchemy equivalents would be created separately)
class TokenDiscoveryTable:
    """Database table structure for discovered tokens."""
    
    # Table: discovered_tokens
    columns = {
        'id': 'VARCHAR(36) PRIMARY KEY',
        'symbol': 'VARCHAR(20) NOT NULL',
        'name': 'VARCHAR(255) NOT NULL',
        'blockchain': 'VARCHAR(50) NOT NULL',
        'chain_id': 'INTEGER NOT NULL',
        'address': 'VARCHAR(255) NOT NULL UNIQUE',
        'logo_url': 'TEXT',
        'coingecko_id': 'VARCHAR(255)',
        'discovery_date': 'TIMESTAMP NOT NULL',
        'current_price': 'DECIMAL(20,8) DEFAULT 0',
        'market_cap': 'DECIMAL(20,2) DEFAULT 0',
        'volume_24h': 'DECIMAL(20,2) DEFAULT 0',
        'price_change_24h': 'DECIMAL(10,4) DEFAULT 0',
        'liquidity': 'DECIMAL(20,2) DEFAULT 0',
        'last_updated': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'is_trending': 'BOOLEAN DEFAULT FALSE',
        'is_active': 'BOOLEAN DEFAULT TRUE',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'
    }


class TokenAnalysisTable:
    """Database table structure for token analysis."""
    
    # Table: token_analysis
    columns = {
        'id': 'VARCHAR(36) PRIMARY KEY',
        'token_id': 'VARCHAR(36) NOT NULL',
        'analysis_type': 'VARCHAR(50) NOT NULL',
        'risk_score': 'DECIMAL(3,2) NOT NULL',
        'potential_score': 'DECIMAL(3,2) NOT NULL',
        'confidence_score': 'DECIMAL(3,2) NOT NULL',
        'overall_score': 'DECIMAL(3,2) NOT NULL',
        'summary': 'TEXT NOT NULL',
        'signals': 'JSON',
        'risk_factors': 'JSON',
        'opportunities': 'JSON',
        'technical_indicators': 'JSON',
        'sentiment_score': 'DECIMAL(3,2) DEFAULT 0.5',
        'social_metrics': 'JSON',
        'analysis_version': 'VARCHAR(10) DEFAULT "1.0"',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP',
        'FOREIGN KEY (token_id)': 'REFERENCES discovered_tokens(id) ON DELETE CASCADE'
    }


class TokenMarketDataTable:
    """Database table structure for market data history."""
    
    # Table: token_market_data
    columns = {
        'id': 'VARCHAR(36) PRIMARY KEY',
        'token_id': 'VARCHAR(36) NOT NULL',
        'source': 'VARCHAR(50) NOT NULL',
        'current_price': 'DECIMAL(20,8) NOT NULL',
        'price_change_1h': 'DECIMAL(10,4)',
        'price_change_24h': 'DECIMAL(10,4)',
        'price_change_7d': 'DECIMAL(10,4)',
        'volume_24h': 'DECIMAL(20,2) NOT NULL',
        'market_cap': 'DECIMAL(20,2) NOT NULL',
        'liquidity': 'DECIMAL(20,2) NOT NULL',
        'trading_pairs_count': 'INTEGER DEFAULT 0',
        'top_exchanges': 'JSON',
        'fetched_at': 'TIMESTAMP NOT NULL',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'FOREIGN KEY (token_id)': 'REFERENCES discovered_tokens(id) ON DELETE CASCADE',
        'INDEX idx_token_fetched': '(token_id, fetched_at)'
    }