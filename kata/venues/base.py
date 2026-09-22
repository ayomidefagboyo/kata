"""The perp venue contract.

Every execution venue Kata trades on implements `PerpVenue`. Agent and
learning code depends only on this interface, never on a concrete venue, so the
same signal pipeline and learning loop run unchanged across venues.

Idempotency is part of the contract: callers pass `client_id` on every order and
adapters must surface it back on `OrderResult` so execution and accounting can
be replayed safely after a crash or reconnect.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from kata.venues.types import (
    OrderBook,
    OrderRequest,
    OrderResult,
    Position,
    Quote,
    SymbolSpec,
)


class PerpVenue(ABC):
    """A perpetual futures venue Kata can route orders to."""

    name: str
    chain: str

    @abstractmethod
    async def connect(self) -> bool:
        """Establish clients/streams. Safe to call repeatedly."""

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def list_symbols(self, force_refresh: bool = False) -> List[str]:
        ...

    @abstractmethod
    async def get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """Tick/lot sizing and leverage bounds, or None if unlisted."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        ...

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_positions(self, account: Optional[str] = None) -> Dict[str, Position]:
        """Open positions keyed by symbol."""

    @abstractmethod
    async def close_position(
        self, symbol: str, percentage: float = 100.0, client_id: Optional[str] = None
    ) -> OrderResult:
        ...

    @abstractmethod
    async def get_account_value(self, account: Optional[str] = None) -> float:
        """Total account equity in USD."""

    async def supports_symbol(self, symbol: str) -> bool:
        return await self.get_symbol_spec(symbol) is not None

    async def quantize_size(self, symbol: str, size: float) -> float:
        """Round a size down to the venue's lot precision."""
        spec = await self.get_symbol_spec(symbol)
        if spec is None:
            return size
        return round(size, spec.size_decimals)

    async def clamp_leverage(self, symbol: str, leverage: float) -> float:
        """Clamp requested leverage to the nearest venue-supported value."""
        spec = await self.get_symbol_spec(symbol)
        if spec is None:
            return leverage
        if spec.supported_leverages:
            return min(
                spec.supported_leverages, key=lambda candidate: abs(candidate - leverage)
            )
        return max(1.0, min(leverage, spec.max_leverage))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} chain={self.chain}>"
