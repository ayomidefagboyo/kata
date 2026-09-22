"""Venue abstraction - agents route orders through `PerpVenue`, never a concrete venue."""

from kata.venues.base import PerpVenue
from kata.venues.registry import available, connect_all, get_venue, register
from kata.venues.types import (
    OrderBook,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    Quote,
    SymbolSpec,
    TimeInForce,
    VenueError,
)

__all__ = [
    "PerpVenue",
    "OrderBook",
    "OrderRequest",
    "OrderResult",
    "OrderSide",
    "OrderType",
    "Position",
    "Quote",
    "SymbolSpec",
    "TimeInForce",
    "VenueError",
    "available",
    "connect_all",
    "get_venue",
    "register",
]
