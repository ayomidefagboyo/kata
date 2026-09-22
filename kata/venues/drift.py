"""Drift (Solana) adapter - Week 2 target.

Scaffold only. Every method raises until the Drift client is wired; the class
exists now so the venue contract can be exercised and the agent can be routed
without further refactoring.

Integration path:
  - `driftpy` (Python SDK) keeps execution in-process with the agent stack.
  - Solana keypair / delegated authority mirrors the delegated-trading model
    already used for Hyperliquid in `services/delegation_service.py`.
  - Drift markets are indexed by `market_index`, not symbol, so
    `_resolve_market_index` is the main translation point to build out.
"""

import logging
from typing import Dict, List, Optional

from kata.venues.base import PerpVenue
from kata.venues.types import (
    OrderBook,
    OrderRequest,
    OrderResult,
    Position,
    Quote,
    SymbolSpec,
)

logger = logging.getLogger(__name__)

_NOT_WIRED = "Drift adapter is not wired yet"


class DriftVenue(PerpVenue):
    name = "drift"
    chain = "solana"

    def __init__(self, rpc_url: Optional[str] = None, authority: Optional[str] = None):
        self.rpc_url = rpc_url
        self.authority = authority
        self._client = None

    async def connect(self) -> bool:
        raise NotImplementedError(_NOT_WIRED)

    async def disconnect(self) -> None:
        self._client = None

    async def list_symbols(self, force_refresh: bool = False) -> List[str]:
        raise NotImplementedError(_NOT_WIRED)

    async def get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        raise NotImplementedError(_NOT_WIRED)

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        raise NotImplementedError(_NOT_WIRED)

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        raise NotImplementedError(_NOT_WIRED)

    async def place_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError(_NOT_WIRED)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        raise NotImplementedError(_NOT_WIRED)

    async def get_positions(self, account: Optional[str] = None) -> Dict[str, Position]:
        raise NotImplementedError(_NOT_WIRED)

    async def close_position(
        self, symbol: str, percentage: float = 100.0, client_id: Optional[str] = None
    ) -> OrderResult:
        raise NotImplementedError(_NOT_WIRED)

    async def get_account_value(self, account: Optional[str] = None) -> float:
        raise NotImplementedError(_NOT_WIRED)

    def _resolve_market_index(self, symbol: str) -> int:
        raise NotImplementedError(_NOT_WIRED)
