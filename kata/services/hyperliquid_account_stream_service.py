"""
Hyperliquid Account State Stream Service

Replaces repeated REST clearinghouseState polling (which Hyperliquid rate-limits
under concurrent load) with one persistent WebSocket connection per wallet,
subscribed to all-DEX account state plus main-DEX open orders. Each connection is
scoped to exactly one wallet address, so pushed data cannot cross accounts.

Callers should treat get_cached_state() as best-effort: a wallet with no active
stream yet (first check, or a connection that failed) returns None and the
caller should fall back to a REST call.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from kata.config.settings import settings
from kata.services.hyperliquid_websocket import HyperliquidWebSocketClient

logger = logging.getLogger(__name__)

# webData2 pushes on essentially every account-affecting update (fills, funding,
# liquidations); anything older than this suggests the stream stalled silently.
_MAX_STATE_AGE_SECONDS = 20.0


class HyperliquidAccountStreamService:
    """Owns one webData2 WebSocket connection per wallet address."""

    def __init__(self):
        self._clients: Dict[str, HyperliquidWebSocketClient] = {}
        self._connecting: set = set()

    async def ensure_stream(
        self,
        wallet_address: str,
        wait_for_data_seconds: float = 5.0,
        account_mode: Optional[str] = None,
    ) -> None:
        """
        Start (or verify) a webData2 stream for this wallet. Best-effort.

        After subscribing for the first time, waits up to wait_for_data_seconds
        for the initial push to land - otherwise every caller's first check
        would find an empty cache and fall through to REST, and a restart would
        burst REST calls for all wallets at once (the exact moment Hyperliquid's
        rate limit bites hardest).
        """
        if not wallet_address:
            return
        key = wallet_address.lower()

        # Guard against exceeding Hyperliquid's 15-connection IP limit
        if len(self._clients) >= 8 and key not in self._clients:
            logger.warning(f"Skipping WS stream for {wallet_address[:10]}…: max pool size (8) reached")
            return

        existing = self._clients.get(key)
        if key in self._connecting:
            return

        self._connecting.add(key)
        try:
            client = existing or HyperliquidWebSocketClient(
                testnet=settings.HYPERLIQUID_TESTNET,
                user_address=wallet_address,
            )
            self._clients[key] = client

            if not client.has_active_listener():
                if not client.is_connected:
                    connected = await client.connect()
                    if not connected:
                        logger.debug(
                            "Could not open Hyperliquid account stream for %s; retrying later",
                            wallet_address[:10],
                        )
                        return
                # connect() only opens the socket and starts the queue processor;
                # start_listening() is the actual recv() loop that feeds it. Use
                # ensure_listening() so only one recv loop ever runs per client.
                client.ensure_listening()

            all_dex_subscription = f"allDexsClearinghouseState_{wallet_address}"
            if all_dex_subscription not in client.subscriptions:
                await client.subscribe_to_all_dexs_clearinghouse_state()
            if "openOrders|" not in client.subscriptions:
                await client.subscribe_to_open_orders()
            portfolio_margin = account_mode == "portfolioMargin"
            spot_subscription = f"spotState|{int(portfolio_margin)}"
            if spot_subscription not in client.subscriptions:
                await client.subscribe_to_spot_state(portfolio_margin)

            # Only block when no push has EVER arrived (fresh subscription).
            # A stream that has data but went stale shouldn't add latency here;
            # the caller falls back to REST while the reconnect loop recovers.
            uses_spot_collateral = account_mode in {"unifiedAccount", "portfolioMargin"}

            def initial_state_ready() -> bool:
                if uses_spot_collateral:
                    return client.latest_spot_state is not None
                if account_mode is None:
                    return (
                        client.latest_spot_state is not None
                        and (
                            client.latest_account_state is not None
                            or client.latest_all_dex_states is not None
                        )
                    )
                return client.latest_account_state is not None or client.latest_all_dex_states is not None

            if not initial_state_ready():
                deadline = asyncio.get_event_loop().time() + wait_for_data_seconds
                while (
                    not initial_state_ready()
                    and asyncio.get_event_loop().time() < deadline
                ):
                    await asyncio.sleep(0.2)
                if not initial_state_ready():
                    logger.warning(
                        f"Hyperliquid account stream for {wallet_address} produced no data within "
                        f"{wait_for_data_seconds}s; caller will fall back to REST"
                    )
        except Exception as e:
            logger.warning(f"Error starting webData2 stream for {wallet_address}: {e}")
        finally:
            self._connecting.discard(key)

    def get_cached_state(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        Return {balance_usdc, withdrawable_usdc, fetched_at} from the live push
        cache, or None if there's no fresh data (caller should fall back to REST).
        """
        client = self._clients.get((wallet_address or "").lower())
        if client is None:
            return None

        state = None
        fetched_at = None
        if client.latest_all_dex_states and client.latest_all_dex_states_at:
            age = (datetime.now() - client.latest_all_dex_states_at).total_seconds()
            if age <= _MAX_STATE_AGE_SECONDS:
                state = client.latest_all_dex_states.get("")
                fetched_at = client.latest_all_dex_states_at
        if state is None:
            fresh_client = self._fresh_client(wallet_address)
            if fresh_client is None:
                return None
            state = fresh_client.latest_account_state
            fetched_at = fresh_client.latest_account_state_at
        try:
            cross_margin_summary = state.get("crossMarginSummary", {}) or {}
            balance_usdc = float(
                cross_margin_summary.get("accountValue")
                or state.get("withdrawable")
                or 0
            )
            withdrawable_usdc = float(state.get("withdrawable") or 0)
        except (TypeError, ValueError):
            return None

        return {
            "balance_usdc": balance_usdc,
            "withdrawable_usdc": withdrawable_usdc,
            "fetched_at": fetched_at,
        }

    def get_cached_spot_balance(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """Fresh pushed USDC balance for unified or portfolio-margin accounts."""
        client = self._clients.get((wallet_address or "").lower())
        if client is None or client.latest_spot_state_at is None:
            return None
        age = (datetime.now() - client.latest_spot_state_at).total_seconds()
        if age > _MAX_STATE_AGE_SECONDS:
            return None

        balances = (client.latest_spot_state or {}).get("balances", [])
        usdc = next(
            (
                balance for balance in balances
                if str(balance.get("coin") or "").upper() == "USDC"
                or balance.get("token") == 0
            ),
            None,
        )
        if not usdc:
            return None
        try:
            total = float(usdc.get("total") or 0)
            held = float(usdc.get("hold") or 0)
        except (TypeError, ValueError):
            return None
        return {
            "balance_usdc": total,
            "withdrawable_usdc": max(0.0, total - held),
            "fetched_at": client.latest_spot_state_at,
        }

    def get_cached_positions(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        Live positions from the push stream as {coin: Position}, or None when
        there's no fresh data (caller should fall back to REST). None vs {} is
        meaningful: {} is a confirmed flat account, None is "don't know".
        """
        # Lazy import: hyperliquid_service imports the websocket module, so
        # importing it at module load would create a cycle.
        from kata.services.hyperliquid_service import HyperliquidService

        key = (wallet_address or "").lower()
        client = self._clients.get(key)
        if client and client.latest_all_dex_states and client.latest_all_dex_states_at:
            age = (datetime.now() - client.latest_all_dex_states_at).total_seconds()
            if age <= _MAX_STATE_AGE_SECONDS:
                positions: Dict[str, Any] = {}
                for state in client.latest_all_dex_states.values():
                    positions.update(HyperliquidService.parse_positions_payload(state))
                return positions

        client = self._fresh_client(wallet_address)
        if client is None:
            return None
        return HyperliquidService.parse_positions_payload(client.latest_account_state)

    def get_cached_open_order_ids(self, wallet_address: str) -> Optional[set]:
        """Open order ids from the push stream, or None when there's no fresh data."""
        orders = self.get_cached_open_orders(wallet_address)
        if orders is None:
            return None
        return {str(order.get("oid")) for order in orders if order.get("oid") is not None}

    def get_cached_open_orders(self, wallet_address: str) -> Optional[list]:
        """Raw open-order objects (frontendOpenOrders shape: oid, coin, isTrigger,
        reduceOnly, sz, timestamp, ...) from the push stream, or None when there's
        no fresh data."""
        client = self._clients.get((wallet_address or "").lower())
        if (
            client is None
            or client.latest_open_orders is None
            or client.latest_open_orders_at is None
        ):
            return None
        age = (datetime.now() - client.latest_open_orders_at).total_seconds()
        if age > _MAX_STATE_AGE_SECONDS:
            return None
        return list(client.latest_open_orders)

    def _fresh_client(self, wallet_address: str):
        """The wallet's client if its pushed state is fresh enough to trust."""
        if not wallet_address:
            return None
        client = self._clients.get(wallet_address.lower())
        if not client or not client.latest_account_state or not client.latest_account_state_at:
            return None
        age = (datetime.now() - client.latest_account_state_at).total_seconds()
        if age > _MAX_STATE_AGE_SECONDS:
            return None
        return client

    async def stop_stream(self, wallet_address: str) -> None:
        """Tear down a wallet's stream (e.g. when its allocation stops)."""
        key = (wallet_address or "").lower()
        client = self._clients.pop(key, None)
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning(f"Error stopping webData2 stream for {wallet_address}: {e}")


_account_stream_service: Optional[HyperliquidAccountStreamService] = None


def get_hyperliquid_account_stream_service() -> HyperliquidAccountStreamService:
    """Get singleton instance of the account state stream service."""
    global _account_stream_service
    if _account_stream_service is None:
        _account_stream_service = HyperliquidAccountStreamService()
    return _account_stream_service
