"""Hyperliquid adapter.

Wraps the existing HyperliquidService (carried over from Flow) behind the
`PerpVenue` contract. Translation only - all execution logic stays in the
underlying service.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from kata.services.hyperliquid_service import (
    HyperliquidService,
    OrderSide as HLOrderSide,
    OrderType as HLOrderType,
)
from kata.venues.base import PerpVenue
from kata.venues.types import (
    OrderBook,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    Quote,
    SymbolSpec,
    VenueError,
)

logger = logging.getLogger(__name__)

_SIDE = {OrderSide.BUY: HLOrderSide.BUY, OrderSide.SELL: HLOrderSide.SELL}
_TYPE = {OrderType.MARKET: HLOrderType.MARKET, OrderType.LIMIT: HLOrderType.LIMIT}


class HyperliquidVenue(PerpVenue):
    name = "hyperliquid"
    chain = "hyperliquid-l1"

    def __init__(self, service: Optional[HyperliquidService] = None):
        self._service = service or HyperliquidService()
        self._connected = False

    async def connect(self) -> bool:
        if self._connected:
            return True
        self._connected = await self._service.initialize_market_data_client()
        return self._connected

    async def disconnect(self) -> None:
        self._connected = False

    async def list_symbols(self, force_refresh: bool = False) -> List[str]:
        universe = await self._service.get_perp_universe(force_refresh=force_refresh)
        return sorted(universe.keys())

    async def get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        if not await self._service.is_perp_listed(symbol):
            return None
        size_decimals = await self._service.get_perp_sz_decimals(symbol)
        max_leverage = await self._service.get_perp_max_leverage(symbol)
        return SymbolSpec(
            symbol=symbol,
            size_decimals=size_decimals if size_decimals is not None else 4,
            price_decimals=6,
            max_leverage=max_leverage if max_leverage is not None else 1.0,
        )

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        data = await self._service.get_live_market_data(symbol)
        if data is None:
            return None
        return Quote(
            symbol=data.symbol,
            price=data.price,
            bid=data.bid,
            ask=data.ask,
            mid_price=data.mid_price,
            mark_price=data.mark_price,
            funding_rate=data.funding_rate,
            open_interest=data.open_interest,
            volume_24h=data.volume_24h,
            price_change_24h=data.price_change_24h,
            timestamp=data.timestamp,
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> Optional[OrderBook]:
        book = await self._service.get_orderbook(symbol, limit=depth)
        if book is None:
            return None
        return OrderBook(
            symbol=symbol,
            bids=list(getattr(book, "bids", []) or []),
            asks=list(getattr(book, "asks", []) or []),
            timestamp=getattr(book, "timestamp", datetime.utcnow()),
        )

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if request.leverage is not None:
            request.leverage = await self.clamp_leverage(request.symbol, request.leverage)
        size = await self.quantize_size(request.symbol, request.size)

        try:
            result = await self._service.place_order(
                symbol=request.symbol,
                side=_SIDE[request.side],
                size=size,
                order_type=_TYPE[request.order_type],
                price=request.price,
                reduce_only=request.reduce_only,
            )
        except Exception as exc:
            logger.exception("Hyperliquid order failed for %s", request.symbol)
            raise VenueError(self.name, str(exc), retryable=True) from exc

        return self._to_order_result(result, client_id=request.client_id)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return await self._service.cancel_order(symbol, order_id)

    async def get_positions(self, account: Optional[str] = None) -> Dict[str, Position]:
        raw = await self._service.get_positions(wallet_address=account)
        return {
            symbol: Position(
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                mark_price=position.mark_price,
                unrealized_pnl=position.unrealized_pnl,
                margin_used=position.margin_used,
                leverage=position.leverage,
                position_value=position.position_value,
                liquidation_price=position.liquidation_price,
                timestamp=position.timestamp,
            )
            for symbol, position in (raw or {}).items()
        }

    async def close_position(
        self, symbol: str, percentage: float = 100.0, client_id: Optional[str] = None
    ) -> OrderResult:
        result = await self._service.close_position(symbol, percentage=percentage)
        return self._to_order_result(result, client_id=client_id)

    async def get_account_value(self, account: Optional[str] = None) -> float:
        positions = await self.get_positions(account)
        return sum(position.position_value for position in positions.values())

    def _to_order_result(self, result, client_id: Optional[str]) -> OrderResult:
        return OrderResult(
            success=result.success,
            venue=self.name,
            order_id=str(result.order_id) if result.order_id is not None else None,
            client_id=client_id,
            symbol=result.symbol,
            side=result.side,
            size=result.size,
            price=result.price,
            filled_size=result.filled_size,
            average_price=result.average_price,
            fee=result.fee,
            status=result.status,
            error=result.error,
            timestamp=result.timestamp or datetime.utcnow(),
        )
