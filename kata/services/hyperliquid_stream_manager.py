"""
HyperliquidStreamManager
========================
Maintains a single Hyperliquid WebSocket connection for the whole backend
process and fans real-time position + order events to per-allocation
``asyncio.Queue`` subscribers consumed by the SSE endpoint.

Subscription map
----------------
* **webData2:{wallet}**  → ``clearinghouseState`` payload (positions, margin,
  mark prices) pushed on every mark-price tick (~1-2 s).
* **orderUpdates**       → fired whenever a resting order is placed, cancelled,
  or filled (same data as ``get_open_orders``).

The SDK's ``WebsocketManager`` is thread-based (uses ``websocket-client``).  We
bridge it to the asyncio event loop via ``asyncio.Queue`` objects so the SSE
generator can ``await queue.get()`` without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal data class
# ---------------------------------------------------------------------------

class StreamEvent:
    """A parsed position or order event ready to send to a subscriber."""

    __slots__ = ("event_type", "wallet", "payload", "ts")

    def __init__(self, event_type: str, wallet: str, payload: Any) -> None:
        self.event_type = event_type   # "positions" | "orders"
        self.wallet = wallet.lower()
        self.payload = payload
        self.ts = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

class HyperliquidStreamManager:
    """
    Singleton that owns one Hyperliquid WebSocket thread and fans events out
    to per-(allocation_id) asyncio queues.

    Usage::

        mgr = get_stream_manager()
        q = await mgr.subscribe(wallet_address, allocation_id)
        try:
            event = await asyncio.wait_for(q.get(), timeout=30)
        finally:
            await mgr.unsubscribe(allocation_id)
    """

    _instance: Optional["HyperliquidStreamManager"] = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "HyperliquidStreamManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # wallet (lower) → set of allocation IDs watching it
        self._wallet_allocations: Dict[str, Set[str]] = defaultdict(set)
        # allocation_id → asyncio.Queue
        self._queues: Dict[str, asyncio.Queue] = {}
        # allocation_id → wallet
        self._allocation_wallets: Dict[str, str] = {}

        # The asyncio event loop running the SSE generators (set on first use)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Hyperliquid SDK WebsocketManager (started lazily)
        self._ws_manager: Any = None
        self._ws_lock: asyncio.Lock = asyncio.Lock()
        self._ws_ready: asyncio.Event = asyncio.Event()
        self._started: bool = False

        # Track which wallets we've already subscribed webData2 for
        self._subscribed_wallets: Set[str] = set()
        # True once orderUpdates is subscribed (it's user-agnostic on HL)
        self._order_updates_subscribed: bool = False

        self._internal_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe(
        self, wallet_address: str, allocation_id: str, maxsize: int = 20
    ) -> asyncio.Queue:
        """Register an SSE listener and return its event queue."""
        self._loop = asyncio.get_event_loop()
        wallet = wallet_address.lower()

        with self._internal_lock:
            q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
            self._queues[allocation_id] = q
            self._allocation_wallets[allocation_id] = wallet
            self._wallet_allocations[wallet].add(allocation_id)

        # Ensure WS is running
        await self._ensure_ws_started()

        # Subscribe to webData2 for this wallet if not already done
        await self._subscribe_wallet(wallet)

        logger.info(f"[stream] subscribed allocation={allocation_id} wallet={wallet[:10]}…")
        return q

    async def unsubscribe(self, allocation_id: str) -> None:
        """Remove a listener. Cleans up wallet subscription if no more listeners."""
        wallet_to_unsubscribe: Optional[str] = None
        with self._internal_lock:
            wallet = self._allocation_wallets.pop(allocation_id, None)
            self._queues.pop(allocation_id, None)
            if wallet:
                self._wallet_allocations[wallet].discard(allocation_id)
                if not self._wallet_allocations[wallet] and self._ws_manager:
                    self._wallet_allocations.pop(wallet, None)
                    wallet_to_unsubscribe = wallet

        # Do not call this while holding _internal_lock. The synchronous helper
        # updates subscription state under the same lock, so calling it inside
        # the critical section deadlocks the event loop when an SSE client
        # disconnects. Keep the WebSocket send off the event loop as well.
        if wallet_to_unsubscribe:
            await asyncio.to_thread(self._unsubscribe_wallet_sync, wallet_to_unsubscribe)

        logger.info(f"[stream] unsubscribed allocation={allocation_id}")

    async def shutdown(self) -> None:
        """Stop the WebSocket thread cleanly."""
        if self._ws_manager:
            try:
                self._ws_manager.ws.close()
                logger.info("[stream] WebSocket closed on shutdown")
            except Exception as exc:
                logger.warning(f"[stream] Error closing WS: {exc}")
        self._ws_manager = None
        self._started = False
        self._ws_ready.clear()

    # ------------------------------------------------------------------
    # Internal – WS lifecycle
    # ------------------------------------------------------------------

    async def _ensure_ws_started(self) -> None:
        async with self._ws_lock:
            if self._started:
                return
            await asyncio.get_event_loop().run_in_executor(None, self._start_ws_thread)
            self._started = True
            # Give the thread up to 10 s to connect
            try:
                await asyncio.wait_for(self._ws_ready.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("[stream] WebSocket did not become ready within 10 s")

    def _start_ws_thread(self) -> None:
        """Create and start the SDK WebsocketManager in a daemon thread."""
        try:
            from hyperliquid.websocket_manager import WebsocketManager

            base_url = "https://api.hyperliquid.xyz"
            self._ws_manager = WebsocketManager(base_url)
            self._ws_manager.daemon = True

            # Patch on_open BEFORE calling start() so the handshake event is never missed
            original_on_open = getattr(self._ws_manager, 'on_open', None)

            def patched_on_open(ws: Any) -> None:
                if original_on_open:
                    try:
                        original_on_open(ws)
                    except Exception:
                        pass
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._ws_ready.set)

            self._ws_manager.on_open = patched_on_open
            self._ws_manager.start()

            # Signal ready immediately so callers never block unnecessarily
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._ws_ready.set)

            logger.info("[stream] Hyperliquid WebSocket thread started")
        except Exception as exc:
            logger.error(f"[stream] Failed to start WebSocket thread: {exc}")
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._ws_ready.set)

    # ------------------------------------------------------------------
    # Internal – subscriptions
    # ------------------------------------------------------------------

    async def _subscribe_wallet(self, wallet: str) -> None:
        """Subscribe to webData2 for a wallet (idempotent)."""
        with self._internal_lock:
            if wallet in self._subscribed_wallets:
                return
            self._subscribed_wallets.add(wallet)

        if not self._ws_manager:
            return

        def _do_subscribe() -> None:
            try:
                subscription: Dict[str, Any] = {"type": "webData2", "user": wallet}
                self._ws_manager.subscribe(subscription, self._make_position_callback(wallet))
                logger.info(f"[stream] subscribed webData2 for {wallet[:10]}…")
            except Exception as exc:
                logger.error(f"[stream] Error subscribing webData2 for {wallet[:10]}: {exc}")

        await asyncio.get_event_loop().run_in_executor(None, _do_subscribe)

        # Subscribe orderUpdates once (wallet-agnostic on Hyperliquid)
        await self._subscribe_order_updates()

    async def _subscribe_order_updates(self) -> None:
        """Subscribe to orderUpdates (idempotent)."""
        with self._internal_lock:
            if self._order_updates_subscribed:
                return
            self._order_updates_subscribed = True

        if not self._ws_manager:
            return

        def _do_subscribe() -> None:
            try:
                subscription: Dict[str, Any] = {"type": "orderUpdates"}
                self._ws_manager.subscribe(subscription, self._order_updates_callback)
                logger.info("[stream] subscribed orderUpdates")
            except Exception as exc:
                logger.error(f"[stream] Error subscribing orderUpdates: {exc}")

        await asyncio.get_event_loop().run_in_executor(None, _do_subscribe)

    def _unsubscribe_wallet_sync(self, wallet: str) -> None:
        """Unsubscribe webData2 (called from thread-safe context)."""
        if not self._ws_manager:
            return
        try:
            subscription: Dict[str, Any] = {"type": "webData2", "user": wallet}
            # SDK subscribe_id tracking not directly exposed; send unsub manually
            self._ws_manager.ws.send(
                json.dumps({"method": "unsubscribe", "subscription": subscription})
            )
            with self._internal_lock:
                self._subscribed_wallets.discard(wallet)
            logger.info(f"[stream] unsubscribed webData2 for {wallet[:10]}…")
        except Exception as exc:
            logger.warning(f"[stream] Error unsubscribing {wallet[:10]}: {exc}")

    # ------------------------------------------------------------------
    # Internal – callbacks (called from WS thread)
    # ------------------------------------------------------------------

    def _make_position_callback(self, wallet: str) -> Callable[[Any], None]:
        """Return a callback that fans clearinghouseState events to queues."""

        def callback(msg: Any) -> None:
            try:
                # msg is the raw WS message dict: {"channel": "webData2", "data": {...}}
                data = msg if isinstance(msg, dict) else {}
                ch = data.get("channel") or ""
                payload_data = data.get("data") or {}

                # webData2 bundles clearinghouseState + open orders
                clearing = payload_data.get("clearinghouseState") or {}
                if not clearing:
                    return

                event = StreamEvent(
                    event_type="positions",
                    wallet=wallet,
                    payload={
                        "clearinghouseState": clearing,
                        "wallet": wallet,
                    },
                )
                self._fan_out(wallet, event)
            except Exception as exc:
                logger.debug(f"[stream] Error in position callback: {exc}")

        return callback

    def _order_updates_callback(self, msg: Any) -> None:
        """Fan order-update events to all currently subscribed queues."""
        try:
            data = msg if isinstance(msg, dict) else {}
            orders = data.get("data") or []
            if not orders:
                return

            # Find which wallets are affected by these orders
            for allocation_id, wallet in list(self._allocation_wallets.items()):
                event = StreamEvent(
                    event_type="orders",
                    wallet=wallet,
                    payload={"orders": orders},
                )
                self._enqueue(allocation_id, event)
        except Exception as exc:
            logger.debug(f"[stream] Error in order_updates callback: {exc}")

    # ------------------------------------------------------------------
    # Internal – fan-out helpers
    # ------------------------------------------------------------------

    def _fan_out(self, wallet: str, event: StreamEvent) -> None:
        """Push an event to all queues watching `wallet`."""
        allocation_ids = list(self._wallet_allocations.get(wallet, set()))
        for allocation_id in allocation_ids:
            self._enqueue(allocation_id, event)

    def _enqueue(self, allocation_id: str, event: StreamEvent) -> None:
        """Thread-safe enqueue via the asyncio loop."""
        q = self._queues.get(allocation_id)
        if q is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._put_nowait, q, event)
        except RuntimeError:
            pass  # loop closed

    @staticmethod
    def _put_nowait(q: asyncio.Queue, event: StreamEvent) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest item and retry
            try:
                q.get_nowait()
                q.put_nowait(event)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------

def get_stream_manager() -> HyperliquidStreamManager:
    return HyperliquidStreamManager.get_instance()
