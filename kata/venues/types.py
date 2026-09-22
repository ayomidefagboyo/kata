"""Venue-neutral trading types.

These mirror the shapes the agents already consume so that a venue adapter is a
thin translation layer rather than a rewrite of agent logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(Enum):
    GTC = "gtc"
    IOC = "ioc"
    ALO = "alo"


@dataclass
class SymbolSpec:
    """Venue-specific trading constraints for one market."""

    symbol: str
    size_decimals: int
    price_decimals: int
    max_leverage: float
    min_size: float = 0.0
    supported_leverages: Optional[List[float]] = None


@dataclass
class Quote:
    symbol: str
    price: float
    bid: float
    ask: float
    mark_price: float
    timestamp: datetime
    mid_price: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0

    def __post_init__(self) -> None:
        if not self.mid_price and self.bid and self.ask:
            self.mid_price = (self.bid + self.ask) / 2


@dataclass
class OrderBook:
    symbol: str
    bids: List[tuple]
    asks: List[tuple]
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin_used: float
    leverage: float
    timestamp: datetime
    position_value: float = 0.0
    liquidation_price: Optional[float] = None


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    size: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    leverage: Optional[float] = None
    reduce_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    success: bool
    venue: str
    order_id: Optional[str] = None
    client_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    filled_size: Optional[float] = None
    average_price: Optional[float] = None
    fee: Optional[float] = None
    status: Optional[str] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class VenueError(Exception):
    """Raised when a venue rejects a request or is unreachable."""

    def __init__(self, venue: str, message: str, retryable: bool = False):
        super().__init__(f"[{venue}] {message}")
        self.venue = venue
        self.retryable = retryable
