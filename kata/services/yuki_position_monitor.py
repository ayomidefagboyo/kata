"""
Yuki Position Monitor Service

Real-time position monitoring for Yuki agent with PnL tracking, risk management,
and automated position management including stop-loss and take-profit execution.
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

from kata.services.hyperliquid_service import HyperliquidService, OrderSide, OrderType
from kata.services.agent_database_service import get_agent_db_service
from kata.config.settings import settings

logger = logging.getLogger(__name__)

SIGNAL_ENTRY_TERMINAL_STATUSES = {"invalidated", "expired", "hit_stop_loss"}


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionRiskMetrics:
    """Position risk metrics."""
    liquidation_risk: float
    margin_ratio: float
    unrealized_pnl_percent: float
    time_in_position_hours: float
    risk_level: RiskLevel
    should_close: bool
    reason: str


class YukiPositionMonitor:
    """
    Real-time position monitoring for Yuki agent.
    
    Features:
    - Real-time PnL tracking
    - Risk management and liquidation prevention
    - Stop-loss and take-profit execution
    - Position health monitoring
    - Automated position management
    """
    
    def __init__(
        self,
        hyperliquid_service: HyperliquidService,
        react_agent_resolver: Optional[Any] = None,
    ):
        """Initialize position monitor."""
        self.hyperliquid_service = hyperliquid_service
        self.db_service = get_agent_db_service()
        
        # Monitoring state
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.is_monitoring = False
        self._pending_entry_locks: Dict[str, asyncio.Lock] = {}
        # Consecutive "missing from book" counts per pending trade id (debounce
        # before finalizing a resting entry as cancelled)
        self._entry_missing_counts: Dict[str, int] = {}
        # Last successful exact-duplicate protective-order sweep per allocation.
        self._order_safety_last_runs: Dict[str, float] = {}
        # Small bounded cache: ATR only changes once per candle and recalculating it
        # every 30-second monitor cycle wastes requests and invites 429 responses.
        self._atr_cache: Dict[str, Tuple[float, float]] = {}
        self._target_candle_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
        self.react_agent_resolver = react_agent_resolver
        self._react_agents: Dict[str, Any] = {}
        
        logger.info("YukiPositionMonitor initialized")
    
    async def start_monitoring(self, user_id: str, allocation_id: str) -> bool:
        """
        Start position monitoring for a user's allocation.
        
        Args:
            user_id: User ID
            allocation_id: Allocation ID
            
        Returns:
            True if monitoring started successfully
        """
        try:
            existing = self.monitoring_tasks.get(allocation_id)
            if existing is not None:
                if not existing.done():
                    logger.debug(f"Monitoring already active for allocation {allocation_id}")
                    return True
                # Task crashed or was cancelled; fall through and restart it
                logger.warning(f"Monitoring task for allocation {allocation_id} had stopped; restarting")
                del self.monitoring_tasks[allocation_id]

            # Start monitoring task
            task = asyncio.create_task(
                self._monitor_allocation_positions(user_id, allocation_id)
            )
            self.monitoring_tasks[allocation_id] = task
            
            logger.info(f"Started position monitoring for allocation {allocation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting position monitoring: {e}")
            return False
    
    async def stop_monitoring(self, allocation_id: str) -> bool:
        """
        Stop position monitoring for an allocation.
        
        Args:
            allocation_id: Allocation ID
            
        Returns:
            True if monitoring stopped successfully
        """
        try:
            if allocation_id in self.monitoring_tasks:
                task = self.monitoring_tasks[allocation_id]
                task.cancel()
                del self.monitoring_tasks[allocation_id]
                self._pending_entry_locks.pop(allocation_id, None)
                
                logger.info(f"Stopped position monitoring for allocation {allocation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error stopping position monitoring: {e}")
            return False
    
    async def _monitor_allocation_positions(self, user_id: str, allocation_id: str):
        """
        Monitor positions for a specific allocation.
        
        Args:
            user_id: User ID
            allocation_id: Allocation ID
        """
        try:
            logger.info(f"Starting position monitoring for allocation {allocation_id}")

            wallet_address: Optional[str] = None

            while True:
                try:
                    # Resolve the allocation's platform wallet - that's where Yuki's
                    # Hyperliquid positions live. Trading methods use the wallet-scoped
                    # delegated client bound to this monitor task.
                    if not wallet_address:
                        wallet_address = await self._resolve_allocation_wallet(allocation_id)
                        if not wallet_address:
                            logger.warning(f"No platform wallet found for allocation {allocation_id}; retrying in 60s")
                            await asyncio.sleep(60)
                            continue

                    # Prefer the webData2 push stream (not subject to Hyperliquid's
                    # REST rate limit); fall back to REST until the stream is live.
                    # None = no data either way, which must not be mistaken for
                    # "all positions closed".
                    from kata.services.hyperliquid_account_stream_service import get_hyperliquid_account_stream_service

                    stream_service = get_hyperliquid_account_stream_service()
                    await stream_service.ensure_stream(wallet_address)
                    hyperliquid_positions = stream_service.get_cached_positions(wallet_address)
                    if hyperliquid_positions is None:
                        hyperliquid_positions = await self.hyperliquid_service.get_positions_for_wallet(wallet_address)
                    if hyperliquid_positions is None:
                        logger.debug(
                            "Hyperliquid positions unavailable for allocation %s; safety cycle skipped",
                            allocation_id,
                        )
                        await asyncio.sleep(30)
                        continue

                    # Bind the delegated trading client for THIS allocation's
                    # wallet before any management action. Cancels, protective SL/TP,
                    # and risk closes all need an authenticated delegated client, and
                    # the worker never authenticates on its own — so without this the
                    # monitor's cancel/close/SL-TP calls raise "no trading client
                    # available", leaving orphan orders live and positions unprotected.
                    trading_ready = await self._ensure_trading_client(user_id, wallet_address)

                    safety_now = asyncio.get_running_loop().time()
                    last_safety_run = self._order_safety_last_runs.get(allocation_id, 0.0)
                    if trading_ready and safety_now - last_safety_run >= 300.0:
                        safety_ok = await self._cancel_duplicate_protective_orders(wallet_address)
                        if safety_ok:
                            self._order_safety_last_runs[allocation_id] = safety_now

                    # Manage resting entry orders (fill detection, expiry cancellation)
                    await self._manage_pending_entries(user_id, allocation_id, hyperliquid_positions, wallet_address)

                    # Get stored positions from database
                    db_positions = await self.db_service.get_agent_positions(allocation_id)

                    # Update existing positions
                    await self._update_existing_positions(
                        user_id, allocation_id, hyperliquid_positions, db_positions
                    )
                    
                    # Check for new positions
                    await self._check_new_positions(
                        user_id, allocation_id, hyperliquid_positions, db_positions
                    )

                    db_positions = await self.db_service.get_agent_positions(allocation_id)
                    
                    # Check risk limits
                    await self._check_risk_limits(user_id, allocation_id, db_positions)
                    
                    # Wait before next check (30 seconds)
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    logger.info(f"Position monitoring cancelled for allocation {allocation_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in position monitoring loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in position monitoring: {e}")

    async def _resolve_allocation_wallet(self, allocation_id: str) -> Optional[str]:
        """Look up the platform wallet address that holds this allocation's positions."""
        try:
            from kata.config.database import get_service_client

            rows = (
                get_service_client()
                .table("agent_allocations")
                .select("platform_wallet_address")
                .eq("allocation_id", allocation_id)
                .limit(1)
                .execute()
            ).data or []
            wallet = rows[0].get("platform_wallet_address") if rows else None
            return str(wallet) if wallet else None
        except Exception as e:
            logger.error(f"Error resolving wallet for allocation {allocation_id}: {e}")
            return None

    @classmethod
    def _protective_order_fingerprint(cls, order: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
        """Identify only exact duplicate reduce-only protection orders."""
        if not order.get("reduceOnly"):
            return None
        return (
            cls._normalize_symbol(order.get("coin")),
            str(order.get("side") or "").upper(),
            str(bool(order.get("isTrigger"))),
            str(order.get("triggerPx") or ""),
            str(order.get("limitPx") or ""),
            str(order.get("sz") or ""),
            str(order.get("orderType") or ""),
        )

    async def _cancel_duplicate_protective_orders(
        self,
        wallet_address: str,
    ) -> bool:
        """Keep one copy of each exact protective order and cancel extras."""
        open_orders = await self.hyperliquid_service.get_open_orders(wallet_address)
        if open_orders is None:
            return False

        grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
        for order in open_orders:
            fingerprint = self._protective_order_fingerprint(order)
            if fingerprint is not None:
                grouped.setdefault(fingerprint, []).append(order)

        all_cancelled = True
        for duplicates in grouped.values():
            if len(duplicates) < 2:
                continue
            ordered = sorted(duplicates, key=lambda item: int(item.get("oid") or 0))
            for duplicate in ordered[1:]:
                symbol = str(duplicate.get("coin") or "")
                order_id = duplicate.get("oid")
                if not symbol or order_id is None:
                    all_cancelled = False
                    continue
                cancelled = await self.hyperliquid_service.cancel_order(symbol, order_id)
                all_cancelled = all_cancelled and cancelled
                if cancelled:
                    logger.warning(
                        "Cancelled duplicate Yuki protective order %s for %s",
                        order_id,
                        symbol,
                    )
        return all_cancelled

    async def _ensure_trading_client(self, user_id: Optional[str], wallet_address: Optional[str]) -> bool:
        """Bind this monitor task to the Hyperliquid client for its wallet.

        Idempotent and cached. The binding is task-local, so another allocation's
        monitor cannot replace this task's signer. Returns False when no delegation
        is available — callers still run read-only monitoring, and exchange-side
        SL/TP placed at entry remain the hard backstop.
        """
        try:
            if not (user_id and wallet_address and self.hyperliquid_service):
                return False
            ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                user_id=user_id, wallet_address=wallet_address
            )
            if not ready:
                logger.warning(
                    f"Trading client not authenticated for allocation wallet {wallet_address}; "
                    "cancels / protective SL-TP / risk closes are skipped this cycle "
                    "(exchange-side SL/TP from entry remain in force)"
                )
            return ready
        except Exception as e:
            logger.error(
                "Error ensuring trading client for %s: %s",
                HyperliquidService._masked_wallet(wallet_address),
                e,
            )
            return False

    @staticmethod
    def _normalize_symbol(symbol: Optional[str]) -> str:
        """Normalize symbol formats so DB rows (\"NEAR-PERP\") match Hyperliquid coins (\"NEAR\").

        Strips:
        - Quote suffixes   : -PERP, /USD, -USD, USDT
        - HIP-3 DEX prefix : "XYZ:" / "xyz:" etc.  (e.g. XYZ:NATGAS -> NATGAS)

        This ensures a DB trade stored as \"NATGAS\" matches a Hyperliquid live position
        returned as \"XYZ:NATGAS\", preventing the duplicate pending+active display bug.
        """
        base = str(symbol or "").upper().strip()
        for suffix in ("-PERP", "/USD", "-USD", "USDT"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        # Strip HIP-3 venue prefix (e.g. "XYZ:NATGAS" -> "NATGAS")
        if ":" in base:
            base = base.split(":", 1)[1]
        return base

    @classmethod
    def _summarize_trade_fills(
        cls,
        fills: List[Dict[str, Any]],
        symbol: str,
        side: str,
        entry_order_id: Optional[str] = None,
        window_start_ms: Optional[int] = None,
        expected_size: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate every fill in one Hyperliquid position lifecycle.

        Hyperliquid emits one fill for each partial TP and another for the final
        stop/close.  The dashboard stores one row per Yuki thesis, so realized
        P&L must be the sum of all close fills, net of both entry and exit fees.
        """
        target_symbol = cls._normalize_symbol(symbol)
        normalized_side = str(side or "").lower()
        entry_direction = "open long" if normalized_side == "long" else "open short"
        close_direction = "close long" if normalized_side == "long" else "close short"
        expected_entry_oid = str(entry_order_id) if entry_order_id is not None else None

        matching = sorted(
            (
                fill for fill in fills or []
                if cls._normalize_symbol(fill.get("coin")) == target_symbol
                and int(fill.get("time") or 0) >= int(window_start_ms or 0)
            ),
            key=lambda fill: int(fill.get("time") or 0),
        )
        if not matching:
            return None

        anchor_entry_fills = [
            fill for fill in matching
            if str(fill.get("dir") or "").lower() == entry_direction
            and (
                expected_entry_oid is None
                or str(fill.get("oid")) == expected_entry_oid
            )
        ]
        if not anchor_entry_fills and expected_entry_oid is not None:
            # Synthetic/fallback DB order ids do not always equal the venue oid.
            anchor_entry_fills = [
                fill for fill in matching
                if str(fill.get("dir") or "").lower() == entry_direction
            ]

        lifecycle_start = min(
            (int(fill.get("time") or 0) for fill in anchor_entry_fills),
            default=int(window_start_ms or 0),
        )
        # Include every same-side opening fill in this lifecycle. Yuki may add
        # once to a confirmed winner, and its extra entry fee/size belongs to
        # the same dashboard trade rather than a fabricated second thesis.
        entry_fills = [
            fill for fill in matching
            if int(fill.get("time") or 0) >= lifecycle_start
            and str(fill.get("dir") or "").lower() == entry_direction
        ]
        close_fills = [
            fill for fill in matching
            if int(fill.get("time") or 0) >= lifecycle_start
            and str(fill.get("dir") or "").lower() == close_direction
        ]
        if not close_fills:
            return None

        closed_size = sum(max(0.0, float(fill.get("sz") or 0)) for fill in close_fills)
        if closed_size <= 0:
            return None
        if expected_size:
            size_tolerance = max(float(expected_size) * 0.005, 1e-9)
            if abs(closed_size - float(expected_size)) > size_tolerance:
                # An incomplete window or overlapping legacy position is worse
                # than the safe mark-price fallback: do not silently publish a
                # partial or cross-trade P&L total.
                return None

        gross_realized_pnl = sum(float(fill.get("closedPnl") or 0) for fill in close_fills)
        entry_fees = sum(max(0.0, float(fill.get("fee") or 0)) for fill in entry_fills)
        exit_fees = sum(max(0.0, float(fill.get("fee") or 0)) for fill in close_fills)
        total_fees = entry_fees + exit_fees
        opened_size = sum(max(0.0, float(fill.get("sz") or 0)) for fill in entry_fills)
        weighted_entry_notional = sum(
            float(fill.get("px") or 0) * max(0.0, float(fill.get("sz") or 0))
            for fill in entry_fills
        )
        average_entry_price = (
            weighted_entry_notional / opened_size if opened_size > 0 else None
        )
        weighted_exit_notional = sum(
            float(fill.get("px") or 0) * max(0.0, float(fill.get("sz") or 0))
            for fill in close_fills
        )
        average_exit_price = weighted_exit_notional / closed_size

        return {
            "gross_realized_pnl": gross_realized_pnl,
            "entry_fees": entry_fees,
            "exit_fees": exit_fees,
            "fees": total_fees,
            "funding_pnl": 0.0,
            "net_realized_pnl": gross_realized_pnl - total_fees,
            "average_entry_price": average_entry_price,
            "average_exit_price": average_exit_price,
            "opened_size": opened_size,
            "closed_size": closed_size,
            "entry_fill_count": len(entry_fills),
            "exit_fill_count": len(close_fills),
            "entry_order_ids": list(dict.fromkeys(str(fill.get("oid")) for fill in entry_fills)),
            "exit_order_ids": list(dict.fromkeys(str(fill.get("oid")) for fill in close_fills)),
            "first_entry_at_ms": min(
                (int(fill.get("time") or 0) for fill in entry_fills),
                default=None,
            ),
            "last_exit_at_ms": max(int(fill.get("time") or 0) for fill in close_fills),
        }

    async def _fetch_closed_trade_fill_summary(
        self,
        db_position: Dict[str, Any],
        require_full_close: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and aggregate exchange fills for a position lifecycle.

        Closed-position reconciliation requires the full expected size. Live
        positions use the same venue ledger with ``require_full_close=False``
        so partial target profits reach allocation P&L immediately.
        """
        trade_id = db_position.get("trade_id")
        allocation_id = db_position.get("allocation_id")
        info_client = getattr(self.hyperliquid_service, "info_client", None)
        if not (trade_id and allocation_id and info_client):
            return None

        try:
            trade_rows = (
                self.db_service.db.table("agent_trades")
                .select("id,symbol,position_size,created_at,filled_at,hyperliquid_order_id,trade_metadata")
                .eq("id", trade_id)
                .limit(1)
                .execute()
            ).data or []
            if not trade_rows:
                return None
            trade = trade_rows[0]
            wallet_address = await self._resolve_allocation_wallet(str(allocation_id))
            if not wallet_address:
                return None

            start_value = trade.get("created_at") or trade.get("filled_at") or db_position.get("created_at")
            start_at = self._parse_datetime(start_value) or datetime.now(timezone.utc) - timedelta(days=7)
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            start_ms = int((start_at - timedelta(minutes=5)).timestamp() * 1000)
            end_ms = int((datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp() * 1000)
            fills = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: info_client.user_fills_by_time(wallet_address, start_ms, end_ms),
            )
            metadata = trade.get("trade_metadata") or {}
            entry_order_id = (
                metadata.get("entry_order_id")
                or trade.get("hyperliquid_order_id")
            )
            summary = self._summarize_trade_fills(
                fills=fills or [],
                symbol=str(trade.get("symbol") or db_position.get("symbol") or ""),
                side=str(db_position.get("side") or ""),
                entry_order_id=str(entry_order_id) if entry_order_id is not None else None,
                window_start_ms=start_ms,
                expected_size=(
                    float(trade.get("position_size") or 0) or None
                    if require_full_close
                    else None
                ),
            )
            if summary:
                funding_pnl = 0.0
                funding_available = False
                try:
                    funding_rows = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: info_client.post(
                            "/info",
                            {
                                "type": "userFunding",
                                "user": wallet_address,
                                "startTime": start_ms,
                                "endTime": end_ms,
                            },
                        ),
                    )
                    target_coin = self._normalize_symbol(
                        str(trade.get("symbol") or db_position.get("symbol") or "")
                    )
                    for funding_row in funding_rows or []:
                        delta = funding_row.get("delta") or {}
                        if self._normalize_symbol(delta.get("coin")) != target_coin:
                            continue
                        funding_pnl += float(delta.get("usdc") or 0.0)
                    funding_available = True
                except Exception as funding_exc:
                    logger.debug(
                        "Funding history unavailable for trade %s: %s",
                        trade_id,
                        funding_exc,
                    )

                summary["funding_pnl"] = funding_pnl
                summary["funding_data_available"] = funding_available
                summary["net_realized_pnl"] = (
                    float(summary.get("gross_realized_pnl") or 0.0)
                    - float(summary.get("fees") or 0.0)
                    + funding_pnl
                )

                signal_entry_price = self._positive_float(
                    metadata.get("signal_entry_price")
                )
                actual_entry_price = self._positive_float(
                    summary.get("average_entry_price")
                )
                if signal_entry_price and actual_entry_price:
                    normalized_side = str(db_position.get("side") or "").lower()
                    adverse_move = (
                        actual_entry_price - signal_entry_price
                        if normalized_side == "long"
                        else signal_entry_price - actual_entry_price
                    )
                    summary["entry_slippage_pct"] = (
                        adverse_move / signal_entry_price * 100.0
                    )
                    summary["entry_slippage_usd"] = adverse_move * float(
                        summary.get("opened_size") or 0.0
                    )

                signal_generated_at = self._parse_datetime(
                    metadata.get("signal_generated_at")
                )
                first_entry_at_ms = summary.get("first_entry_at_ms")
                if signal_generated_at and first_entry_at_ms:
                    if signal_generated_at.tzinfo is None:
                        signal_generated_at = signal_generated_at.replace(tzinfo=timezone.utc)
                    filled_at = datetime.fromtimestamp(
                        int(first_entry_at_ms) / 1000.0,
                        tz=timezone.utc,
                    )
                    summary["entry_age_hours_at_fill"] = max(
                        0.0,
                        (filled_at - signal_generated_at).total_seconds() / 3600.0,
                    )
                summary["source"] = "hyperliquid_user_fills"
                summary["reconciled_at"] = datetime.now(timezone.utc).isoformat()
            return summary
        except Exception as exc:
            logger.warning(
                "Could not aggregate Hyperliquid fills for position %s: %s",
                db_position.get("id"), exc,
            )
            return None

    async def _reconcile_partial_trade_fill_summary(
        self,
        db_position: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Persist realized P&L for target fills while a runner remains open."""
        trade_id = db_position.get("trade_id")
        if not trade_id:
            return

        summary = await self._fetch_closed_trade_fill_summary(
            db_position,
            require_full_close=False,
        )
        if not summary:
            return

        fingerprint = ":".join((
            str(summary.get("exit_fill_count") or 0),
            str(round(float(summary.get("closed_size") or 0), 12)),
            str(summary.get("last_exit_at_ms") or 0),
        ))
        if metadata.get("partial_fill_reconciliation_fingerprint") == fingerprint:
            return

        now = datetime.now(timezone.utc).isoformat()
        summary["source"] = "hyperliquid_user_fills_partial_reconciliation"
        summary["reconciled_at"] = now
        trade_rows = (
            self.db_service.db.table("agent_trades")
            .select("trade_metadata")
            .eq("id", trade_id)
            .limit(1)
            .execute()
        ).data or []
        trade_metadata = dict((trade_rows[0] if trade_rows else {}).get("trade_metadata") or {})
        trade_metadata["partial_fill_reconciliation"] = summary
        updated = (
            self.db_service.db.table("agent_trades")
            .update({
                "status": "partially_filled",
                "realized_pnl": float(summary.get("net_realized_pnl") or 0.0),
                "fees": float(summary.get("fees") or 0.0),
                "trade_metadata": trade_metadata,
            })
            .eq("id", trade_id)
            .in_("status", ["filled", "partially_filled"])
            .execute()
        ).data or []
        if not updated:
            return

        metadata["partial_fill_reconciliation_fingerprint"] = fingerprint
        metadata["partial_fill_reconciliation"] = summary
        await self.db_service.reconcile_agent_allocation_ledger(
            db_position.get("allocation_id")
        )
        logger.info(
            "Reconciled %s partial exit fill(s) for %s: net realized PnL $%.2f",
            summary.get("exit_fill_count"),
            db_position.get("symbol"),
            float(summary.get("net_realized_pnl") or 0.0),
        )

    @classmethod
    def _find_hl_position(cls, hyperliquid_positions: Dict[str, Any], symbol: Optional[str]) -> Optional[Any]:
        """Find a Hyperliquid position for a DB symbol, tolerating format differences."""
        if symbol in hyperliquid_positions:
            return hyperliquid_positions[symbol]
        target = cls._normalize_symbol(symbol)
        for coin, position in hyperliquid_positions.items():
            if cls._normalize_symbol(coin) == target:
                return position
        return None

    @classmethod
    def _pending_entry_fills(
        cls,
        fills: List[Dict[str, Any]],
        row: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return exact exchange fills for one pending entry order."""
        order_id = row.get("hyperliquid_order_id")
        if order_id is None:
            return []
        expected_direction = (
            "open long"
            if str(row.get("side") or "").lower() in {"buy", "long"}
            else "open short"
        )
        target_symbol = cls._normalize_symbol(row.get("symbol"))
        return sorted(
            [
                fill for fill in fills or []
                if str(fill.get("oid")) == str(order_id)
                and cls._normalize_symbol(fill.get("coin")) == target_symbol
                and str(fill.get("dir") or "").lower() == expected_direction
            ],
            key=lambda fill: int(fill.get("time") or 0),
        )

    async def _fetch_pending_entry_fills(
        self,
        wallet_address: Optional[str],
        rows: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch one fill-history window for all entries missing from the book."""
        info_client = getattr(self.hyperliquid_service, "info_client", None)
        if not (wallet_address and rows and info_client):
            return None
        try:
            starts = [
                self._parse_datetime(row.get("created_at"))
                for row in rows
            ]
            starts = [value for value in starts if value is not None]
            start_at = min(starts) if starts else datetime.now(timezone.utc) - timedelta(days=7)
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            start_ms = int((start_at - timedelta(minutes=5)).timestamp() * 1000)
            end_ms = int((datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp() * 1000)
            fills = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: info_client.user_fills_by_time(
                    wallet_address,
                    start_ms,
                    end_ms,
                ),
            )
            return list(fills or [])
        except Exception as exc:
            logger.warning(
                "Could not verify pending-entry fills for wallet %s: %s",
                str(wallet_address)[:10],
                exc,
            )
            return None

    async def _reconcile_missing_pending_entry_fill(
        self,
        db,
        row: Dict[str, Any],
        fills: List[Dict[str, Any]],
    ) -> bool:
        """Preserve an executed entry when no current position snapshot remains.

        A resting order can fill and close between 30-second monitor snapshots.
        In that case, absence from both the order book and current positions is
        not cancellation evidence; the immutable fill ledger is authoritative.
        """
        entry_fills = self._pending_entry_fills(fills, row)
        if not entry_fills:
            return False

        position_side = (
            "long"
            if str(row.get("side") or "").lower() in {"buy", "long"}
            else "short"
        )
        summary = self._summarize_trade_fills(
            fills=fills,
            symbol=str(row.get("symbol") or ""),
            side=position_side,
            entry_order_id=str(row.get("hyperliquid_order_id")),
            window_start_ms=min(int(fill.get("time") or 0) for fill in entry_fills),
            expected_size=float(row.get("position_size") or 0) or None,
        )
        metadata = dict(row.get("trade_metadata") or {})
        now = datetime.now(timezone.utc).isoformat()

        if summary:
            summary["source"] = "hyperliquid_user_fills_pending_reconciliation"
            summary["reconciled_at"] = now
            metadata["fill_reconciliation"] = summary
            first_entry_ms = summary.get("first_entry_at_ms")
            last_exit_ms = summary.get("last_exit_at_ms")
            payload = {
                "status": "closed",
                "entry_price": summary.get("average_entry_price") or row.get("entry_price"),
                "exit_price": summary["average_exit_price"],
                "position_size": summary.get("opened_size") or row.get("position_size"),
                "realized_pnl": summary["net_realized_pnl"],
                "fees": summary["fees"],
                "filled_at": (
                    datetime.fromtimestamp(int(first_entry_ms) / 1000.0, tz=timezone.utc).isoformat()
                    if first_entry_ms else now
                ),
                "closed_at": (
                    datetime.fromtimestamp(int(last_exit_ms) / 1000.0, tz=timezone.utc).isoformat()
                    if last_exit_ms else now
                ),
                "trade_metadata": metadata,
            }
            updated = (
                db.table("agent_trades").update(payload)
                .eq("id", row["id"])
                .eq("status", "pending")
                .execute()
            ).data or []
            if updated:
                await self.db_service.reconcile_agent_allocation_ledger(row.get("allocation_id"))
                logger.info(
                    "Recovered filled-and-closed entry %s for %s from Hyperliquid fills",
                    row.get("hyperliquid_order_id"),
                    row.get("symbol"),
                )
            return True

        # Fill evidence exists but the current fill window does not prove a
        # complete close. Keep the reservation pending and retry rather than
        # publishing a false cancellation or a phantom closed trade.
        opened_size = sum(float(fill.get("sz") or 0) for fill in entry_fills)
        weighted_notional = sum(
            float(fill.get("px") or 0) * float(fill.get("sz") or 0)
            for fill in entry_fills
        )
        metadata["pending_entry_fill_evidence"] = {
            "order_id": str(row.get("hyperliquid_order_id")),
            "fill_count": len(entry_fills),
            "filled_size": opened_size,
            "average_entry_price": (
                weighted_notional / opened_size if opened_size > 0 else None
            ),
            "first_fill_at_ms": min(int(fill.get("time") or 0) for fill in entry_fills),
            "verified_at": now,
        }
        (
            db.table("agent_trades").update({"trade_metadata": metadata})
            .eq("id", row["id"])
            .eq("status", "pending")
            .execute()
        )
        logger.warning(
            "Entry %s for %s has fill evidence but no complete live/closed lifecycle yet; preserving it",
            row.get("hyperliquid_order_id"),
            row.get("symbol"),
        )
        return True

    async def reconcile_pending_entries_for_allocation(
        self,
        user_id: str,
        allocation_id: str,
        wallet_address: Optional[str],
    ) -> bool:
        """Release terminal or stale entry reservations before a signal sweep."""
        if not wallet_address:
            return False

        hyperliquid_positions = await self.hyperliquid_service.get_positions_for_wallet(
            wallet_address
        )
        if hyperliquid_positions is None:
            logger.warning(
                "Could not verify positions before pending-entry reconciliation for allocation %s",
                allocation_id,
            )
            return False

        await self._manage_pending_entries(
            user_id,
            allocation_id,
            hyperliquid_positions,
            wallet_address,
        )
        return True

    async def _manage_pending_entries(
        self,
        user_id: str,
        allocation_id: str,
        hyperliquid_positions: Dict[str, Any],
        wallet_address: Optional[str] = None,
    ):
        """Serialize monitor and pre-sweep reconciliation for one allocation."""
        lock = self._pending_entry_locks.setdefault(allocation_id, asyncio.Lock())
        async with lock:
            await self._manage_pending_entries_locked(
                user_id,
                allocation_id,
                hyperliquid_positions,
                wallet_address,
            )

    async def _manage_pending_entries_locked(
        self,
        user_id: str,
        allocation_id: str,
        hyperliquid_positions: Dict[str, Any],
        wallet_address: Optional[str] = None,
    ):
        """
        Manage resting GTC entry orders placed at the signal's entry price:
        - order gone from the book + position exists -> mark filled, place protective orders
        - order gone + no position -> cancelled externally, credit budget back
        - signal expired while still resting -> cancel the order, credit budget back
        """
        try:
            from kata.config.database import get_service_client

            db = get_service_client()
            pending = (
                db.table("agent_trades")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("agent_type", "yuki")
                .eq("status", "pending")
                .not_.is_("hyperliquid_order_id", "null")
                .execute()
            ).data or []
            if not pending:
                return

            # Ensure the trading client targets this wallet before any cancel /
            # fill-driven protective-order placement below — otherwise a stale
            # or wrong-wallet client leaves orphan orders live and fills
            # unprotected (the DOGE incident).
            await self._ensure_trading_client(user_id, wallet_address)

            # Prefer the pushed main-DEX order snapshot for main-DEX entries.
            # HIP-3 open orders are DEX-scoped, so fetch
            # the aggregated REST book whenever any pending entry is HIP-3.
            open_orders = None
            has_hip3_pending = any(":" in str(row.get("symbol") or "") for row in pending)
            if wallet_address and not has_hip3_pending:
                from kata.services.hyperliquid_account_stream_service import get_hyperliquid_account_stream_service

                open_orders = get_hyperliquid_account_stream_service().get_cached_open_orders(wallet_address)
            if open_orders is None:
                open_orders = await self.hyperliquid_service.get_open_orders(wallet_address)
            if open_orders is None:
                # Cannot see the book right now; try again next cycle.
                return
            open_order_ids = {
                str(order.get("oid")) for order in open_orders if order.get("oid") is not None
            }
            missing_from_book = [
                row for row in pending
                if str(row.get("hyperliquid_order_id")) not in open_order_ids
            ]
            missing_entry_fills = await self._fetch_pending_entry_fills(
                wallet_address,
                missing_from_book,
            ) if missing_from_book else []

            signal_ids = {
                str((row.get("trade_metadata") or {}).get("signal_id") or "")
                for row in pending
            }
            signal_ids.discard("")
            source_statuses: Dict[str, str] = {}
            source_entry_states: Dict[str, str] = {}
            if signal_ids:
                try:
                    source_rows = (
                        db.table("platform_signals")
                        .select("signal_id,status,market_conditions")
                        .in_("signal_id", list(signal_ids))
                        .execute()
                    ).data or []
                    source_statuses = {
                        str(source.get("signal_id")): str(source.get("status") or "").lower()
                        for source in source_rows
                    }
                    source_entry_states = {
                        str(source.get("signal_id")): str(
                            ((source.get("market_conditions") or {}).get("entry_order_state"))
                            or ""
                        ).lower()
                        for source in source_rows
                    }
                except Exception as source_error:
                    # A failed lifecycle lookup is not proof of invalidation.
                    logger.warning("Could not verify source status for pending Yuki entries: %s", source_error)

            now = datetime.now(timezone.utc)
            for row in pending:
                try:
                    oid = str(row.get("hyperliquid_order_id"))
                    symbol = row.get("symbol")
                    meta = row.get("trade_metadata") or {}
                    signal_id = str(meta.get("signal_id") or "")
                    source_status = source_statuses.get(signal_id)
                    source_entry_state = source_entry_states.get(signal_id)

                    source_entry_terminal = source_entry_state in {
                        "expired_pending_reanalysis",
                        "awaiting_confirmation",
                        "invalidated",
                        "missed_entry",
                        "thesis_expired",
                    }
                    if source_status in SIGNAL_ENTRY_TERMINAL_STATUSES or source_entry_terminal:
                        terminal_reason = (
                            source_entry_state
                            if source_entry_terminal
                            else source_status or "terminal"
                        )
                        if oid in open_order_ids:
                            cancelled = await self.hyperliquid_service.cancel_order(symbol, oid)
                            if not cancelled:
                                # The snapshot may have raced a fill. Keep the row
                                # reserved and let the next cycle confirm the
                                # position instead of falsely releasing capital.
                                logger.warning(
                                    "Pending %s entry %s could not be cancelled after source became %s; "
                                    "will confirm fill state next cycle",
                                    symbol, oid, terminal_reason,
                                )
                                continue
                            await self._finalize_pending_entry(
                                db,
                                row,
                                status="cancelled",
                                cancellation_reason=f"source_signal_{terminal_reason}",
                            )
                            logger.info(
                                "Cancelled resting entry %s for %s because source signal %s is %s",
                                oid, symbol, signal_id, terminal_reason,
                            )
                            continue

                        # If the order has already left the book, retain the normal
                        # fill-detection path below. It may have filled just before
                        # invalidation and must not be mislabeled as cancelled.
                        position = self._find_hl_position(hyperliquid_positions, symbol)
                        if not position or float(getattr(position, "size", 0) or 0) <= 0:
                            if missing_entry_fills is None:
                                # No immutable fill evidence is available right
                                # now, so absence alone cannot prove cancellation.
                                continue
                            if await self._reconcile_missing_pending_entry_fill(
                                db,
                                row,
                                missing_entry_fills,
                            ):
                                self._entry_missing_counts.pop(str(row.get("id")), None)
                                continue
                            await self._finalize_pending_entry(
                                db,
                                row,
                                status="cancelled",
                                cancellation_reason=f"source_signal_{terminal_reason}",
                            )
                            logger.info(
                                "Finalized absent entry %s for %s after source signal became %s",
                                oid, symbol, terminal_reason,
                            )
                            continue

                    if oid in open_order_ids:
                        expires_raw = meta.get("entry_order_expires_at") or meta.get("signal_expires_at")
                        expires_at = None
                        if expires_raw:
                            try:
                                expires_at = datetime.fromisoformat(str(expires_raw).replace("Z", "+00:00"))
                                if expires_at.tzinfo is None:
                                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                            except (TypeError, ValueError):
                                expires_at = None
                        if expires_at and now > expires_at:
                            cancelled = await self.hyperliquid_service.cancel_order(symbol, oid)
                            if cancelled:
                                await self._finalize_pending_entry(
                                    db,
                                    row,
                                    status="cancelled",
                                    cancellation_reason="entry_ttl_expired_reanalysis_required",
                                )
                                if signal_id:
                                    from kata.services.platform_signal_service import get_platform_signal_service

                                    await get_platform_signal_service().request_signal_reanalysis(
                                        signal_id,
                                        "entry TTL expired before fill",
                                        expire_entry_order=True,
                                        source="yuki_position_monitor",
                                    )
                                logger.info(
                                    "Cancelled stale resting entry %s for %s and queued fresh re-analysis",
                                    oid,
                                    symbol,
                                )
                        continue

                    # Order left the book: filled if a position now exists.
                    position = self._find_hl_position(hyperliquid_positions, symbol)
                    if position and float(getattr(position, "size", 0) or 0) > 0:
                        self._entry_missing_counts.pop(str(row.get("id")), None)
                        await self._activate_filled_entry(db, user_id, allocation_id, row, position)
                    else:
                        if missing_entry_fills is None:
                            # Fill-history lookup failed. Retrying is safer than
                            # converting a possibly executed order to cancelled.
                            continue
                        if await self._reconcile_missing_pending_entry_fill(
                            db,
                            row,
                            missing_entry_fills,
                        ):
                            self._entry_missing_counts.pop(str(row.get("id")), None)
                            continue
                        # Debounce: one glitchy/stale open-orders view must not kill a
                        # live entry. A false "cancelled" here previously left the order
                        # resting on-exchange while the budget was credited back and the
                        # signal became re-tradeable - producing duplicate live orders.
                        row_id = str(row.get("id"))
                        misses = self._entry_missing_counts.get(row_id, 0) + 1
                        self._entry_missing_counts[row_id] = misses
                        if misses < 2:
                            logger.info(f"Resting entry {oid} for {symbol} missing from book (check {misses}/2); confirming next cycle")
                            continue
                        self._entry_missing_counts.pop(row_id, None)

                        # Belt-and-braces: issue an exchange cancel before finalizing.
                        # If the order is truly gone this is a harmless error; if our
                        # book view was wrong, this removes the live order so it can't
                        # fill after we've released its budget.
                        try:
                            await self.hyperliquid_service.cancel_order(symbol, oid)
                        except Exception:
                            pass
                        await self._finalize_pending_entry(db, row, status="cancelled")
                        logger.info(f"Resting entry {oid} for {symbol} left the book with no position; released budget")
                except Exception as row_error:
                    logger.error(f"Error managing pending entry {row.get('id')}: {row_error}")

            await self._sweep_orphan_entry_orders(db, pending, wallet_address, open_orders=open_orders)
        except Exception as e:
            logger.error(f"Error managing pending entries for allocation {allocation_id}: {e}")

    async def _finalize_pending_entry(
        self,
        db,
        row: Dict[str, Any],
        status: str,
        cancellation_reason: Optional[str] = None,
    ):
        """Mark a pending entry terminal and credit the reserved budget back."""
        payload: Dict[str, Any] = {
            "status": status,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        if cancellation_reason:
            metadata = dict(row.get("trade_metadata") or {})
            metadata["entry_cancellation_reason"] = cancellation_reason
            metadata["entry_cancelled_at"] = datetime.now(timezone.utc).isoformat()
            payload["trade_metadata"] = metadata
        updated_rows = (
            db.table("agent_trades").update(payload)
            .eq("id", row["id"])
            .eq("status", "pending")
            .execute()
        ).data or []
        if not updated_rows:
            return

        allocation_id = row.get("allocation_id")
        await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
        if allocation_id:
            try:
                from kata.services.agent_allocation_service import get_agent_allocation_service

                asyncio.create_task(
                    get_agent_allocation_service().trigger_yuki_signal_check_for_allocation(
                        allocation_id
                    )
                )
            except Exception as trigger_error:
                logger.warning(
                    "Could not trigger Yuki after pending margin was released: %s",
                    trigger_error,
                )

    async def _sweep_orphan_entry_orders(
        self,
        db,
        pending_rows: List[Dict[str, Any]],
        wallet_address: Optional[str],
        open_orders: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Cancel entry orders resting on-exchange that no pending trade row claims.

        These orphans exist when a trade row was finalized as cancelled without the
        exchange order actually being cancelled (e.g. a wrong-wallet or stale book
        view). An orphan is invisible to trade management - its budget was already
        credited back and its signal may have been re-entered - so if it fills, the
        account carries an unmanaged duplicate position.
        """
        try:
            if not wallet_address:
                return
            if open_orders is None:
                from kata.services.hyperliquid_account_stream_service import get_hyperliquid_account_stream_service

                open_orders = get_hyperliquid_account_stream_service().get_cached_open_orders(wallet_address)
                if open_orders is None:
                    return  # no fresh stream view; sweep is opportunistic

            claimed_oids = {
                str(row.get("hyperliquid_order_id"))
                for row in pending_rows
                if row.get("hyperliquid_order_id") is not None
            }
            now_ms = datetime.now(timezone.utc).timestamp() * 1000

            for order in open_orders:
                oid = order.get("oid")
                if oid is None or str(oid) in claimed_oids:
                    continue
                # Only plain entry orders: never touch protective SL/TP triggers or
                # any reduce-only order, and give just-placed orders a 3-minute
                # grace window against the order-placed/row-inserted race.
                if order.get("isTrigger") or order.get("reduceOnly"):
                    continue
                placed_ms = float(order.get("timestamp") or 0)
                if placed_ms and (now_ms - placed_ms) < 3 * 60 * 1000:
                    continue

                coin = order.get("coin")
                cancelled = await self.hyperliquid_service.cancel_order(coin, oid)
                if cancelled:
                    logger.warning(
                        f"Cancelled ORPHAN entry order {oid} ({coin}, size {order.get('sz')}): "
                        "no pending trade row claims it - it would have filled as an unmanaged position"
                    )
        except Exception as e:
            logger.error(f"Error sweeping orphan entry orders: {e}")

    async def _activate_filled_entry(
        self,
        db,
        user_id: str,
        allocation_id: str,
        row: Dict[str, Any],
        position: Any,
    ):
        """A resting entry filled: record the fill and place protective orders."""
        # Cross-process claim: both the API service and worker can observe the
        # same venue fill. Only the process that atomically transitions the
        # pending row to filled may create position/protective records.
        filled_at = datetime.now(timezone.utc).isoformat()
        claimed_rows = (
            db.table("agent_trades")
            .update({
                "status": "filled",
                "entry_price": float(
                    getattr(position, "entry_price", 0)
                    or row.get("entry_price")
                    or 0
                ),
                "filled_at": filled_at,
            })
            .eq("id", row["id"])
            .eq("status", "pending")
            .execute()
        ).data or []
        if not claimed_rows:
            logger.info(
                "Fill activation for trade %s was already claimed by another process",
                row.get("id"),
            )
            return

        row = {**row, **claimed_rows[0]}
        meta = row.get("trade_metadata") or {}
        entry_price = float(getattr(position, "entry_price", 0) or row.get("entry_price") or 0)
        position_size = float(row.get("position_size") or getattr(position, "size", 0) or 0)
        entry_side = OrderSide.BUY if str(row.get("side")) == "buy" else OrderSide.SELL

        # Lazy import to avoid a circular module dependency.
        from kata.services.agent_allocation_service import get_agent_allocation_service

        allocation_service = get_agent_allocation_service()
        protective_orders = await allocation_service._place_yuki_protective_orders(
            symbol=row.get("symbol"),
            entry_side=entry_side,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=meta.get("stop_loss"),
            target_1=meta.get("target_1"),
            target_2=meta.get("target_2"),
        )

        meta["protective_orders"] = protective_orders
        db.table("agent_trades").update({
            "entry_price": entry_price,
            "trade_metadata": meta,
        }).eq("id", row["id"]).execute()

        allocation = allocation_service.active_allocations.get(allocation_id)
        if allocation is None:
            allocation_rows = (
                db.table("agent_allocations").select("*").eq("allocation_id", allocation_id).limit(1).execute()
            ).data or []
            allocation = allocation_service._allocation_from_row(allocation_rows[0]) if allocation_rows else None

        stored_position = None
        if allocation is not None:
            stored_position = await allocation_service._store_yuki_position_record(
                user_id=user_id,
                allocation=allocation,
                trade_id=row.get("id"),
                symbol=row.get("symbol"),
                side=entry_side,
                position_size=position_size,
                entry_price=entry_price,
                leverage=float(row.get("leverage") or 1),
                order_id=row.get("hyperliquid_order_id"),
                stop_loss=meta.get("stop_loss"),
                target_1=meta.get("target_1"),
                target_2=meta.get("target_2"),
                protective_orders=protective_orders,
                signal_expires_at=meta.get("signal_expires_at"),
                time_horizon=meta.get("time_horizon"),
                signal_id=meta.get("signal_id"),
            )

        logger.info(
            f"Resting entry filled: {row.get('side')} {position_size} {row.get('symbol')} @ ${entry_price}; "
            "protective orders placed"
        )

        db_service = getattr(self, "db_service", None)
        if db_service is not None:
            await db_service.reconcile_agent_allocation_ledger(allocation_id)

        await self._trigger_post_fill_review(
            user_id=user_id,
            allocation_id=allocation_id,
            position_row=stored_position,
            live_position=position,
            filled_trade=row,
            protective_orders=protective_orders,
        )

    async def _trigger_post_fill_review(
        self,
        user_id: str,
        allocation_id: str,
        position_row: Optional[Dict[str, Any]],
        live_position: Any,
        filled_trade: Optional[Dict[str, Any]] = None,
        protective_orders: Optional[Dict[str, Any]] = None,
        acquire_cycle_lock: bool = True,
    ) -> None:
        """Run one immediate ReAct review after a tracked entry becomes live."""
        if not position_row or not position_row.get("id"):
            return

        try:
            position = dict(position_row)
            side = str(position.get("side") or "").lower()
            entry_price = float(position.get("entry_price") or 0.0)
            leverage = float(position.get("leverage") or 1.0)
            stored_size = float(position.get("size") or 0.0)
            live_size = abs(float(getattr(live_position, "size", 0) or 0.0))
            size = live_size or stored_size
            current_price = float(getattr(live_position, "mark_price", 0) or 0.0) or entry_price
            position_value = size * current_price
            unrealized_pnl = float(getattr(live_position, "unrealized_pnl", 0) or 0.0)
            if unrealized_pnl == 0.0 and entry_price > 0.0 and size > 0.0:
                price_diff = (
                    current_price - entry_price
                    if side == "long"
                    else entry_price - current_price
                )
                unrealized_pnl = price_diff * size
            margin_used = position_value / leverage if leverage > 0 else position_value
            margin_ratio = (
                margin_used / (position_value + unrealized_pnl)
                if (position_value + unrealized_pnl) > 0
                else 1.0
            )
            unrealized_pnl_percent = (
                (unrealized_pnl / margin_used) * 100.0 if margin_used > 0 else 0.0
            )

            position.update({
                "size": size,
                "current_price": current_price,
                "position_value": position_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "margin_used": margin_used,
                "margin_ratio": margin_ratio,
            })

            risk_metrics = await self._calculate_position_risk(position)
            if not self._event_needs_review(position, risk_metrics, "entry_filled"):
                return
            if not self._position_review_due(position, "entry_filled"):
                return
            if risk_metrics.should_close:
                logger.info(
                    "Skipping immediate post-fill ReAct review for %s because emergency risk handling is required: %s",
                    position.get("symbol"),
                    risk_metrics.reason,
                )
                return

            stop_record = ((protective_orders or {}).get("stop_loss") or {})
            await self._review_position_event(
                user_id=user_id,
                allocation_id=allocation_id,
                position=position,
                risk_metrics=risk_metrics,
                event="entry_filled",
                trigger_context={
                    "filled_at": (filled_trade or {}).get("filled_at"),
                    "trade_id": position.get("trade_id"),
                    "signal_id": ((filled_trade or {}).get("trade_metadata") or {}).get("signal_id"),
                    "protective_stop_live": bool(stop_record.get("success") and stop_record.get("order_id")),
                    "protective_targets_live": bool((protective_orders or {}).get("take_profit")),
                },
                acquire_cycle_lock=acquire_cycle_lock,
            )
        except Exception as exc:
            logger.error(
                "Immediate post-fill ReAct review failed for %s: %s",
                (position_row or {}).get("symbol"),
                exc,
            )

    async def _update_existing_positions(
        self,
        user_id: str,
        allocation_id: str,
        hyperliquid_positions: Dict[str, Any],
        db_positions: List[Dict[str, Any]]
    ):
        """Update existing positions with current market data."""
        try:
            for db_position in db_positions:
                symbol = db_position["symbol"]
                hl_position = self._find_hl_position(hyperliquid_positions, symbol)

                if hl_position is None:
                    # Position no longer exists on Hyperliquid: an exchange-side
                    # SL/TP trigger filled or it was closed manually. Close the DB
                    # record so realized PnL lands in the trade ledger.
                    await self._close_missing_position(db_position)
                    continue

                # Calculate current metrics
                entry_price = float(db_position["entry_price"])
                stored_size = float(db_position["size"])
                live_size = abs(float(getattr(hl_position, "size", 0) or 0))
                size = live_size or stored_size
                side = db_position["side"]
                leverage = float(db_position["leverage"] or 1)

                current_price = float(getattr(hl_position, "mark_price", 0) or 0)
                if current_price <= 0:
                    current_price = float(db_position.get("current_price") or entry_price)

                position_value = size * current_price

                # USD PnL straight from Hyperliquid; fall back to notional price move
                unrealized_pnl = float(getattr(hl_position, "unrealized_pnl", 0) or 0)
                if unrealized_pnl == 0 and entry_price > 0:
                    price_diff = (current_price - entry_price) if side == "long" else (entry_price - current_price)
                    unrealized_pnl = price_diff * size

                # Percent return on margin (what the user actually put up)
                margin_used = position_value / leverage if leverage > 0 else position_value
                unrealized_pnl_percent = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0.0

                # Calculate margin ratio
                margin_ratio = margin_used / (position_value + unrealized_pnl) if (position_value + unrealized_pnl) > 0 else 1.0

                # Infer target fills both from price and from the reduced live size.
                # Size inference covers the important case where price touches a TP,
                # the exchange fills it, then price retreats before the next 30s poll.
                metadata = dict(db_position.get("position_metadata") or {})
                target_1_hit = metadata.get("target_1_hit", False)
                target_1 = self._positive_float(metadata.get("target_1"))
                target_2 = self._positive_float(metadata.get("target_2"))
                runner_enabled = bool(metadata.get("runner_enabled"))
                original_size = self._positive_float(metadata.get("original_size")) or stored_size
                protective_orders = metadata.get("protective_orders") or {}
                target_orders = protective_orders.get("take_profit") or []
                tp1_size = self._positive_float(target_orders[0].get("size")) if target_orders else None
                runner_size = self._positive_float(metadata.get("runner_size"))
                size_tolerance = max(original_size * 1e-6, 1e-12)
                tp1_filled_by_size = bool(
                    tp1_size and size <= original_size - tp1_size + size_tolerance
                )
                tp2_filled_by_size = bool(
                    runner_enabled and runner_size and size <= runner_size + size_tolerance
                )

                if size < original_size - size_tolerance:
                    await self._reconcile_partial_trade_fill_summary(
                        db_position,
                        metadata,
                    )

                if not target_1_hit and target_1:
                    if self._take_profit_hit(side, current_price, target_1) or tp1_filled_by_size:
                        target_1_hit = True
                        metadata["target_1_hit"] = True

                was_target_1_confirmed = bool(metadata.get("target_1_confirmed"))
                target_1_confirmed = was_target_1_confirmed
                if (
                    runner_enabled
                    and target_1_hit
                    and target_1
                    and not target_1_confirmed
                ):
                    target_1_confirmed = await self._target_confirmation_ready(
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        target=target_1,
                        stage="target_1",
                        metadata=metadata,
                    )
                if target_1_confirmed and not was_target_1_confirmed:
                    review_events = dict(metadata.get("react_review_events") or {})
                    review_events["target_1_confirmed"] = {
                        "status": "pending",
                        "requested_at": datetime.now(timezone.utc).isoformat(),
                        "target_price": target_1,
                        "remaining_size": size,
                    }
                    metadata["react_review_events"] = review_events

                target_2_hit = bool(metadata.get("target_2_hit"))
                if not target_2_hit and target_2:
                    if self._take_profit_hit(side, current_price, target_2) or tp2_filled_by_size:
                        target_2_hit = True
                        metadata["target_2_hit"] = True

                was_target_2_confirmed = bool(metadata.get("target_2_confirmed"))
                target_2_confirmed = was_target_2_confirmed
                if (
                    runner_enabled
                    and target_2_hit
                    and target_2
                    and not target_2_confirmed
                ):
                    target_2_confirmed = await self._target_confirmation_ready(
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        target=target_2,
                        stage="target_2",
                        metadata=metadata,
                    )
                if target_2_confirmed and not was_target_2_confirmed:
                    review_events = dict(metadata.get("react_review_events") or {})
                    review_events["target_2_confirmed"] = {
                        "status": "pending",
                        "requested_at": datetime.now(timezone.utc).isoformat(),
                        "target_price": target_2,
                        "remaining_size": size,
                    }
                    metadata["react_review_events"] = review_events

                desired_stop = None
                stop_stage = None
                # A confirmed target is a Yuki decision event. Do not
                # automatically ratchet or trail the runner; the ReAct cycle can
                # hold, add, reduce, adjust protection, or exit after reviewing
                # the complete trade memory and current evidence.
                if not self._positive_float(metadata.get("active_stop")):
                    desired_stop = self._positive_float(metadata.get("stop_loss"))
                    stop_stage = "initial_repair"

                if desired_stop and stop_stage:
                    await self._ratchet_exchange_stop(
                        user_id=user_id,
                        allocation_id=allocation_id,
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        remaining_size=size,
                        desired_stop=desired_stop,
                        stage=stop_stage,
                        metadata=metadata,
                    )

                # Repair take-profit legs that were rejected at entry (e.g. a 429
                # burst) so the 25/25/50 adaptive-runner plan stays intact.
                await self._repair_take_profit_orders(
                    user_id=user_id,
                    allocation_id=allocation_id,
                    symbol=symbol,
                    side=side,
                    current_price=current_price,
                    remaining_size=size,
                    metadata=metadata,
                )
                
                # Update position in database
                await self.db_service.update_agent_position(
                    position_id=db_position["id"],
                    current_price=current_price,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    position_value=position_value,
                    margin_ratio=margin_ratio,
                    position_metadata=metadata,
                    size=size,
                    margin_used=margin_used,
                )
                
                logger.debug(f"Updated position {symbol}: PnL {unrealized_pnl_percent:.2f}%, margin ratio {margin_ratio:.2f}")

        except Exception as e:
            logger.error(f"Error updating existing positions: {e}")

    @staticmethod
    def _tighter_stop(side: str, current_stop: Optional[float], candidate: float) -> float:
        """Return the tighter stop without ever loosening existing protection."""
        if not current_stop:
            return candidate
        return max(current_stop, candidate) if side == "long" else min(current_stop, candidate)

    @staticmethod
    def _stop_improvement(side: str, current_stop: Optional[float], candidate: float) -> float:
        if not current_stop:
            return float("inf")
        return candidate - current_stop if side == "long" else current_stop - candidate

    @classmethod
    def _winner_scale_in_plan(
        cls,
        position: Dict[str, Any],
        available_margin: float,
        protected_stop_override: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Return a strictly risk-capped add plan for a confirmed winner.

        Entry selection remains owned by Platform Signals. This plan is only
        available after entry, before either target is touched, and only when a
        stop above/below fee-adjusted breakeven can protect the combined trade.
        """
        if not settings.YUKI_WINNER_SCALE_IN_ENABLED:
            return None

        metadata = position.get("position_metadata") or {}
        if (
            str(position.get("_source_signal_status") or "").lower() != "active"
            and not position.get("_react_position_add_authorized")
            and not position.get("_react_horizon_add_authorized")
        ):
            return None
        if metadata.get("execution_policy") != "signal_led_adaptive":
            return None
        if (
            metadata.get("scale_in_completed_at")
            or metadata.get("scale_in_in_progress_at")
            or metadata.get("target_1_hit")
            or metadata.get("target_2_hit")
            or not metadata.get("runner_enabled")
        ):
            return None

        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return None
        entry_price = cls._positive_float(metadata.get("initial_entry_price"))
        stop_loss = cls._positive_float(metadata.get("initial_stop_loss"))
        initial_size = cls._positive_float(metadata.get("initial_size"))
        current_size = cls._positive_float(position.get("size"))
        current_price = cls._positive_float(position.get("current_price"))
        leverage = cls._positive_float(position.get("leverage")) or 1.0
        if not all((entry_price, stop_loss, initial_size, current_size, current_price)):
            return None
        if current_size < initial_size * 0.995 or current_size > initial_size * 1.005:
            return None

        risk_per_unit = (
            entry_price - stop_loss if side == "long" else stop_loss - entry_price
        )
        if risk_per_unit <= 0:
            return None

        trigger_r = max(0.25, float(settings.YUKI_WINNER_SCALE_IN_TRIGGER_R))
        trigger_price = (
            entry_price + risk_per_unit * trigger_r
            if side == "long"
            else entry_price - risk_per_unit * trigger_r
        )
        target_1 = cls._positive_float(metadata.get("target_1"))
        if target_1 and (
            (side == "long" and trigger_price >= target_1)
            or (side == "short" and trigger_price <= target_1)
        ):
            return None
        if not cls._price_beyond_target(side, current_price, trigger_price):
            return None

        fee_floor = max(0.0, float(settings.YUKI_RUNNER_PROFIT_FLOOR_PCT))
        policy_protected_stop = (
            entry_price * (1.0 + fee_floor)
            if side == "long"
            else entry_price * (1.0 - fee_floor)
        )
        protected_stop = policy_protected_stop
        override_stop = cls._positive_float(protected_stop_override)
        if override_stop:
            if side == "long":
                candidate_stop = max(stop_loss, min(policy_protected_stop, override_stop))
            else:
                candidate_stop = min(stop_loss, max(policy_protected_stop, override_stop))
            if not cls._stop_loss_hit(side, current_price, candidate_stop):
                protected_stop = candidate_stop
        if cls._stop_loss_hit(side, current_price, protected_stop):
            return None

        size_fraction = max(
            0.0,
            min(1.0, float(settings.YUKI_WINNER_SCALE_IN_MAX_ORIGINAL_SIZE_FRACTION)),
        )
        max_risk_fraction = max(
            0.0,
            min(1.0, float(settings.YUKI_WINNER_SCALE_IN_MAX_INITIAL_RISK_FRACTION)),
        )
        # Hyperliquid market orders are IOC limits padded by 1%. Size against
        # that adverse boundary so a normal fill cannot breach the risk cap.
        risk_price = current_price * (1.01 if side == "long" else 0.99)
        max_size_by_fraction = initial_size * size_fraction
        initial_dollar_risk = initial_size * risk_per_unit
        added_risk_per_unit = abs(risk_price - protected_stop)
        max_size_by_risk = (
            initial_dollar_risk * max_risk_fraction / added_risk_per_unit
            if added_risk_per_unit > 0
            else 0.0
        )
        add_size = min(max_size_by_fraction, max_size_by_risk)
        add_margin = add_size * risk_price / leverage
        if (
            add_size <= 0
            or add_margin < float(settings.YUKI_MIN_TRADE_USDC)
            or add_margin > max(0.0, float(available_margin)) + 1e-9
        ):
            return None

        return {
            "trigger_price": trigger_price,
            "protected_stop": protected_stop,
            "policy_protected_stop": policy_protected_stop,
            "risk_price": risk_price,
            "size": add_size,
            "margin": add_margin,
            "initial_dollar_risk": initial_dollar_risk,
            "added_risk_at_stop": add_size * added_risk_per_unit,
            "leverage": leverage,
        }

    @staticmethod
    def _price_beyond_target(side: str, price: float, target: float) -> bool:
        return price >= target if side == "long" else price <= target

    async def _target_confirmation_candles(
        self,
        symbol: str,
        side: str,
        target: float,
        reached_at: datetime,
        now: datetime,
    ) -> int:
        """Count consecutive closed 1m candles beyond a target after first touch."""
        client = getattr(self.hyperliquid_service, "info_client", None)
        if not client:
            return 0

        cache_key = f"{symbol}:target-confirmation:1m"
        now_ts = now.timestamp()
        cache = getattr(self, "_target_candle_cache", {})
        cached = cache.get(cache_key)
        candles: List[Dict[str, Any]]
        if cached and now_ts - cached[0] < 20:
            candles = cached[1]
        else:
            end_ms = int(now_ts * 1000)
            start_ms = end_ms - 10 * 60 * 1000
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol,
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
            try:
                candles = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: client.post("/info", payload)
                )
            except Exception as exc:
                logger.warning("Target confirmation candles unavailable for %s: %s", symbol, exc)
                return 0
            if not hasattr(self, "_target_candle_cache"):
                self._target_candle_cache = {}
            self._target_candle_cache[cache_key] = (now_ts, candles or [])

        reached_ms = int(reached_at.timestamp() * 1000)
        now_ms = int(now_ts * 1000)
        closed = []
        for candle in candles or []:
            open_ms = int(candle.get("t") or 0)
            close_ms = int(candle.get("T") or (open_ms + 60_000))
            if close_ms > now_ms or close_ms < reached_ms:
                continue
            close_price = self._positive_float(candle.get("c"))
            if close_price is not None:
                closed.append((close_ms, close_price))
        closed.sort(key=lambda value: value[0])

        consecutive = 0
        for _, close_price in reversed(closed):
            if self._price_beyond_target(side, close_price, target):
                consecutive += 1
            else:
                break
        return consecutive

    async def _target_confirmation_ready(
        self,
        symbol: str,
        side: str,
        current_price: float,
        target: float,
        stage: str,
        metadata: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> bool:
        """Confirm a target after two 1m closes or three continuous minutes."""
        now = now or datetime.now(timezone.utc)
        reached_key = f"{stage}_first_reached_at"
        above_since_key = f"{stage}_beyond_since"
        confirmed_key = f"{stage}_confirmed"
        confirmed_at_key = f"{stage}_confirmed_at"

        reached_at = self._parse_datetime(metadata.get(reached_key))
        if reached_at is None:
            reached_at = now
            metadata[reached_key] = now.isoformat()

        beyond_target = self._price_beyond_target(side, current_price, target)
        beyond_since = self._parse_datetime(metadata.get(above_since_key))
        if beyond_target:
            if beyond_since is None:
                beyond_since = now
                metadata[above_since_key] = now.isoformat()
        else:
            beyond_since = None
            metadata.pop(above_since_key, None)

        required_minutes = max(0.0, float(settings.YUKI_TARGET_CONFIRMATION_MINUTES))
        time_confirmed = bool(
            beyond_since
            and (now - beyond_since).total_seconds() >= required_minutes * 60
        )
        consecutive_candles = await self._target_confirmation_candles(
            symbol=symbol,
            side=side,
            target=target,
            reached_at=reached_at,
            now=now,
        )
        required_candles = max(1, int(settings.YUKI_TARGET_CONFIRMATION_CANDLES))
        candle_confirmed = consecutive_candles >= required_candles

        metadata[f"{stage}_confirmation_candles"] = consecutive_candles
        if not (time_confirmed or candle_confirmed):
            return False

        metadata[confirmed_key] = True
        metadata[confirmed_at_key] = now.isoformat()
        metadata[f"{stage}_confirmation_method"] = (
            "continuous_time" if time_confirmed else "closed_candles"
        )
        logger.info(
            "%s confirmed for %s after %s",
            stage.upper(), symbol, metadata[f"{stage}_confirmation_method"],
        )
        return True

    async def _buffered_target_stop(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        target: float,
        stage: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Place a target stop outside normal noise while retaining net profit."""
        agent_buffer_pct = metadata.get("post_tp1_stop_buffer_pct") or metadata.get("agent_post_tp1_stop_buffer_pct")
        
        if stage == "target_2":
            price_buffer_pct = max(0.0, float(agent_buffer_pct if agent_buffer_pct is not None else settings.YUKI_TARGET_2_STOP_BUFFER_PCT))
            atr_fraction = max(0.0, float(settings.YUKI_TARGET_2_STOP_ATR_FRACTION))
        else:
            price_buffer_pct = max(0.005, float(agent_buffer_pct if agent_buffer_pct is not None else settings.YUKI_TARGET_1_STOP_BUFFER_PCT))
            atr_fraction = max(0.0, float(settings.YUKI_TARGET_1_STOP_ATR_FRACTION))

        atr = await self._runner_atr(symbol)
        fee_floor_pct = max(0.0, float(settings.YUKI_RUNNER_PROFIT_FLOOR_PCT))

        if stage == "target_1":
            # Target 1 Stop: Protect entry (Breakeven + fee/ATR buffer), giving the runner room to breathe up to TP1
            target_gain = abs(target - entry_price)
            buffer = max(entry_price * fee_floor_pct, min(target_gain * 0.4, atr * 0.5 if atr > 0 else entry_price * 0.005))
            if side == "long":
                stop = entry_price + buffer
            else:
                stop = entry_price - buffer
            distance = abs(target - stop)
        else:
            distance = max(target * price_buffer_pct, atr * atr_fraction if atr > 0 else 0.0)
            if side == "long":
                buffered = target - distance
                profit_floor = entry_price * (1.0 + fee_floor_pct)
                stop = max(buffered, profit_floor)
            else:
                buffered = target + distance
                profit_floor = entry_price * (1.0 - fee_floor_pct)
                stop = min(buffered, profit_floor)

        metadata[f"{stage}_buffered_stop"] = stop
        metadata[f"{stage}_stop_buffer_distance"] = distance
        metadata[f"{stage}_stop_atr"] = atr
        return stop

    async def _ratchet_exchange_stop(
        self,
        user_id: str,
        allocation_id: str,
        symbol: str,
        side: str,
        current_price: float,
        remaining_size: float,
        desired_stop: float,
        stage: str,
        metadata: Dict[str, Any],
        force_resize: bool = False,
    ) -> bool:
        """
        Replace the stop without a protection gap: place the tighter exchange
        trigger first, then cancel the old trigger. If placement fails the original
        stop stays live and the tighter policy stop is retried next cycle.
        """
        current_active = self._positive_float(metadata.get("active_stop"))
        candidate = self._tighter_stop(side, current_active, float(desired_stop))
        improvement = self._stop_improvement(side, current_active, candidate)
        if improvement <= 1e-12 and not force_resize:
            return False

        if stage == "runner_trail" and current_active and not force_resize:
            min_step = max(0.0, current_price * float(settings.YUKI_RUNNER_TRAIL_MIN_STEP_PCT))
            if improvement < min_step:
                return False

        # Persist the intended stop even when it was crossed between monitor polls;
        # the software risk check will close immediately instead of placing an
        # already-triggered order that the venue may reject.
        metadata["policy_stop"] = candidate
        metadata["desired_stop_stage"] = stage
        if self._stop_loss_hit(side, current_price, candidate):
            metadata["stop_update_error"] = "Desired stop already crossed; software close queued"
            return False

        wallet_address = await self._resolve_allocation_wallet(allocation_id)
        if not await self._ensure_trading_client(user_id, wallet_address):
            metadata["stop_update_error"] = "Trading client unavailable; previous exchange stop retained"
            return False

        close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        new_stop = await self.hyperliquid_service.place_order_live(
            symbol=symbol,
            side=close_side,
            size=remaining_size,
            order_type=OrderType.MARKET,
            reduce_only=True,
            trigger_price=candidate,
            trigger_tpsl="sl",
            trigger_is_market=True,
        )
        if not new_stop.success:
            metadata["stop_update_error"] = (
                f"Exchange rejected {stage} stop; previous stop retained: {new_stop.error}"
            )
            logger.error(
                "Yuki could not ratchet %s stop for %s to %s: %s",
                stage,
                symbol,
                candidate,
                new_stop.error,
            )
            return False

        protective_orders = dict(metadata.get("protective_orders") or {})
        previous_order = dict(protective_orders.get("stop_loss") or {})
        previous_order_id = previous_order.get("order_id")
        old_cancelled = True
        if previous_order_id and str(previous_order_id) != str(new_stop.order_id):
            old_cancelled = await self.hyperliquid_service.cancel_order(symbol, previous_order_id)
            if not old_cancelled:
                # Two reduce-only stops are safer than cancelling first and risking
                # no stop. Whichever fires first cannot reverse the position.
                logger.warning(
                    "New tighter stop %s is live for %s, but old stop %s could not be cancelled",
                    new_stop.order_id,
                    symbol,
                    previous_order_id,
                )

        supplemental_cancelled = []
        for supplemental in protective_orders.get("supplemental_stops") or []:
            supplemental_id = supplemental.get("order_id")
            if not supplemental_id or str(supplemental_id) == str(new_stop.order_id):
                continue
            cancelled = await self.hyperliquid_service.cancel_order(symbol, supplemental_id)
            supplemental_cancelled.append({
                "order_id": supplemental_id,
                "cancelled": bool(cancelled),
            })
        protective_orders["supplemental_stops"] = [
            supplemental
            for supplemental in protective_orders.get("supplemental_stops") or []
            if not any(
                str(result["order_id"]) == str(supplemental.get("order_id"))
                and result["cancelled"]
                for result in supplemental_cancelled
            )
        ]

        history = list(metadata.get("stop_history") or [])[-19:]
        history.append({
            "stage": stage,
            "price": candidate,
            "size": remaining_size,
            "order_id": new_stop.order_id,
            "previous_order_id": previous_order_id,
            "previous_order_cancelled": old_cancelled,
            "supplemental_stops": supplemental_cancelled,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "resized": force_resize,
        })
        protective_orders["stop_loss"] = {
            "success": True,
            "order_id": new_stop.order_id,
            "price": candidate,
            "size": remaining_size,
            "stage": stage,
            "error": None,
        }
        metadata["protective_orders"] = protective_orders
        metadata["active_stop"] = candidate
        metadata["policy_stop"] = candidate
        metadata["stop_stage"] = stage
        metadata["stop_history"] = history
        metadata["stop_last_updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata.pop("stop_update_error", None)
        logger.info(
            "Yuki ratcheted %s %s stop to %s for remaining size %s",
            symbol,
            stage,
            candidate,
            remaining_size,
        )
        return True

    @staticmethod
    def _is_rate_limit_error(error: Any) -> bool:
        text = str(error or "").lower()
        return "429" in text or "rate limit" in text or "too many requests" in text

    async def _repair_take_profit_orders(
        self,
        user_id: str,
        allocation_id: str,
        symbol: str,
        side: str,
        current_price: float,
        remaining_size: float,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Reconcile exchange take-profit legs against the intended exit plan.

        The stop has a repair path (initial_repair ratchet) but TP legs
        historically did not: if the exchange rejected them at entry - typically
        during a rate-limit burst - they were never retried, and the adaptive
        runner exit plan never executed.
        Retries use exponential backoff persisted in position metadata so a
        rejecting venue is not hammered every 30-second cycle.
        """
        try:
            protective = dict(metadata.get("protective_orders") or {})
            tp_orders = list(protective.get("take_profit") or [])
            failed_indices = [
                index for index, order in enumerate(tp_orders)
                if not (order.get("success") and order.get("order_id"))
            ]
            if not failed_indices:
                return False

            # Never place a TP behind price: a crossed "tp" trigger executes
            # immediately, which is a market exit the risk logic should own.
            hit_flags = (bool(metadata.get("target_1_hit")), bool(metadata.get("target_2_hit")))
            repairable: List[int] = []
            for index in failed_indices:
                target_price = self._positive_float(tp_orders[index].get("price"))
                if not target_price:
                    continue
                if len(tp_orders) == 2 and index < len(hit_flags) and hit_flags[index]:
                    continue
                if self._take_profit_hit(side, current_price, target_price):
                    continue
                repairable.append(index)
            if not repairable:
                return False

            now = datetime.now(timezone.utc)
            repair_state = dict(metadata.get("tp_repair") or {})
            next_retry_at = self._parse_datetime(repair_state.get("next_retry_at"))
            if next_retry_at and now < next_retry_at:
                return False

            wallet_address = await self._resolve_allocation_wallet(allocation_id)
            if not await self._ensure_trading_client(user_id, wallet_address):
                return False

            close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
            live_tp_size = sum(
                self._positive_float(order.get("size")) or 0.0
                for order in tp_orders
                if order.get("success") and order.get("order_id")
            )
            placeable_budget = max(0.0, remaining_size - live_tp_size)

            any_failure = False
            any_success = False
            for index in repairable:
                order = dict(tp_orders[index])
                target_price = float(order.get("price"))
                target_size = min(
                    self._positive_float(order.get("size")) or placeable_budget,
                    placeable_budget,
                )
                if target_size <= 0:
                    continue
                result = await self.hyperliquid_service.place_order_live(
                    symbol=symbol,
                    side=close_side,
                    size=target_size,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                    trigger_price=target_price,
                    trigger_tpsl="tp",
                    trigger_is_market=True,
                )
                order["success"] = bool(result.success)
                order["order_id"] = result.order_id
                order["error"] = result.error
                if result.success:
                    any_success = True
                    placeable_budget = max(0.0, placeable_budget - target_size)
                    order["repaired_at"] = now.isoformat()
                    logger.info(
                        "Repaired missing %s take-profit @ %s (size %s) for allocation %s",
                        symbol, target_price, target_size, allocation_id,
                    )
                else:
                    any_failure = True
                    logger.warning(
                        "Take-profit repair failed for %s @ %s: %s",
                        symbol, target_price, result.error,
                    )
                tp_orders[index] = order

            protective["take_profit"] = tp_orders
            metadata["protective_orders"] = protective

            if any_failure:
                attempts = int(repair_state.get("attempts") or 0) + 1
                delay_seconds = min(60 * (2 ** (attempts - 1)), 1800)
                metadata["tp_repair"] = {
                    "attempts": attempts,
                    "next_retry_at": (now + timedelta(seconds=delay_seconds)).isoformat(),
                    "last_error_at": now.isoformat(),
                }
            else:
                metadata.pop("tp_repair", None)

            # Restore the runner once both TP legs are genuinely live.
            runner_size = self._positive_float(protective.get("runner_size"))
            if runner_size is None:
                original_size = self._positive_float(metadata.get("original_size")) or remaining_size
                planned = original_size - sum(
                    self._positive_float(order.get("size")) or 0.0 for order in tp_orders
                )
                runner_size = planned if planned > 0 else None
            if (
                any_success
                and len(tp_orders) == 2
                and all(order.get("success") and order.get("order_id") for order in tp_orders)
                and runner_size
            ):
                protective["runner_enabled"] = True
                protective["runner_size"] = runner_size
                metadata["runner_enabled"] = True
                metadata["runner_size"] = runner_size
                logger.info(
                    "Runner re-enabled for %s (size %s): both take-profit legs are live",
                    symbol, runner_size,
                )
            return any_success
        except Exception as e:
            logger.error(f"Error repairing take-profit orders for {symbol}: {e}")
            return False

    async def _dynamic_runner_stop(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        target_2: float,
        metadata: Dict[str, Any],
        target_floor: Optional[float] = None,
    ) -> float:
        """Blend an ATR trail with a maximum-profit-giveback trail after TP2."""
        prior_extreme = self._positive_float(metadata.get("runner_extreme_price"))
        if side == "long":
            extreme = max(prior_extreme or entry_price, current_price)
        else:
            extreme = min(prior_extreme or entry_price, current_price)
        metadata["runner_extreme_price"] = extreme

        giveback = max(0.0, min(0.95, float(settings.YUKI_RUNNER_TRAIL_GIVEBACK_FRACTION)))
        if side == "long":
            giveback_stop = entry_price + (extreme - entry_price) * (1.0 - giveback)
        else:
            giveback_stop = entry_price - (entry_price - extreme) * (1.0 - giveback)

        atr = await self._runner_atr(symbol)
        atr_multiplier = max(0.1, float(settings.YUKI_RUNNER_TRAIL_ATR_MULTIPLIER))
        atr_stop = None
        if atr > 0:
            atr_stop = extreme - atr * atr_multiplier if side == "long" else extreme + atr * atr_multiplier
            metadata["runner_atr"] = atr

        target_floor = float(target_floor if target_floor is not None else target_2)
        if side == "long":
            candidate = max(target_floor, giveback_stop, atr_stop or target_floor)
        else:
            candidate = min(target_floor, giveback_stop, atr_stop or target_floor)

        # Leave a small trigger gap. If price already retraced through the
        # buffered TP2 floor, return that floor so the software fallback exits.
        if self._stop_loss_hit(side, current_price, target_floor):
            return target_floor
        safe_gap = max(0.0005, float(settings.YUKI_TARGET_CONFIRMATION_BUFFER_PCT) / 2.0)
        if side == "long":
            return min(candidate, current_price * (1.0 - safe_gap))
        return max(candidate, current_price * (1.0 + safe_gap))

    async def _runner_atr(self, symbol: str) -> float:
        """Fetch a bounded, cached ATR snapshot for the runner trail."""
        interval = str(settings.YUKI_RUNNER_ATR_INTERVAL or "15m")
        cache_key = f"{symbol}:{interval}"
        now_ts = datetime.now(timezone.utc).timestamp()
        cached = self._atr_cache.get(cache_key)
        if cached and now_ts - cached[0] < max(30, int(settings.YUKI_RUNNER_ATR_CACHE_SECONDS)):
            return cached[1]

        client = getattr(self.hyperliquid_service, "info_client", None)
        if not client:
            return 0.0
        match = re.fullmatch(r"(\d+)([mhd])", interval.lower().strip())
        if not match:
            return 0.0
        multiplier = {"m": 60, "h": 3600, "d": 86400}[match.group(2)]
        interval_seconds = int(match.group(1)) * multiplier
        period = max(2, int(settings.YUKI_RUNNER_ATR_PERIOD))
        end_ms = int(now_ts * 1000)
        start_ms = int((now_ts - interval_seconds * (period + 3)) * 1000)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        try:
            candles = await asyncio.get_running_loop().run_in_executor(
                None, lambda: client.post("/info", payload)
            )
            ordered = sorted(candles or [], key=lambda candle: int(candle.get("t") or 0))
            true_ranges = []
            previous_close = None
            for candle in ordered:
                high = float(candle.get("h") or 0)
                low = float(candle.get("l") or 0)
                close = float(candle.get("c") or 0)
                if high <= 0 or low <= 0 or close <= 0:
                    continue
                true_range = high - low
                if previous_close:
                    true_range = max(true_range, abs(high - previous_close), abs(low - previous_close))
                true_ranges.append(true_range)
                previous_close = close
            atr = sum(true_ranges[-period:]) / min(period, len(true_ranges)) if true_ranges else 0.0
            if len(self._atr_cache) >= 128:
                oldest_key = min(self._atr_cache, key=lambda key: self._atr_cache[key][0])
                self._atr_cache.pop(oldest_key, None)
            self._atr_cache[cache_key] = (now_ts, atr)
            return atr
        except Exception as exc:
            logger.warning("Runner ATR unavailable for %s: %s", symbol, exc)
            return 0.0

    async def _close_missing_position(self, db_position: Dict[str, Any]):
        """
        Close a DB position whose Hyperliquid position has disappeared (exchange-side
        SL/TP filled or manual close). Without this, the row stays active forever with
        stale prices and the realized PnL never reaches the trade ledger.
        """
        try:
            # Grace period: skip rows created moments ago in case the exchange
            # position hasn't propagated to user_state yet.
            created_at = datetime.fromisoformat(str(db_position["created_at"]).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - created_at).total_seconds() < 180:
                return

            symbol = db_position["symbol"]
            side = db_position["side"]
            entry_price = float(db_position.get("entry_price") or 0)
            size = float(db_position.get("size") or 0)

            # Prefer venue fills over the last cached position size. Partial TP
            # fills reduce ``db_position.size`` before the final exit, so using
            # only that remainder is exactly how XRP's first 120-token profit
            # disappeared from the dashboard.
            fill_summary = await self._fetch_closed_trade_fill_summary(db_position)
            if fill_summary:
                await self._close_position(
                    db_position["id"],
                    float(fill_summary["average_exit_price"]),
                    float(fill_summary["net_realized_pnl"]),
                    fees=float(fill_summary["fees"]),
                    trade_metadata_updates={"fill_reconciliation": fill_summary},
                )
                logger.info(
                    "Position %s no longer on Hyperliquid; aggregated %d exit fill(s), "
                    "gross PnL $%.2f, fees $%.2f, net PnL $%.2f",
                    symbol,
                    int(fill_summary["exit_fill_count"]),
                    float(fill_summary["gross_realized_pnl"]),
                    float(fill_summary["fees"]),
                    float(fill_summary["net_realized_pnl"]),
                )
                return

            # Best-effort exit price: live mid, else last stored mark, else entry
            exit_price = 0.0
            try:
                coin = self._normalize_symbol(symbol)
                snapshot = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.hyperliquid_service._fetch_rest_market_snapshot(coin)
                )
                if snapshot and snapshot.mid_price > 0:
                    exit_price = float(snapshot.mid_price)
            except Exception:
                pass
            if exit_price <= 0:
                exit_price = float(db_position.get("current_price") or entry_price)

            if entry_price > 0 and exit_price > 0 and size > 0:
                realized_pnl = ((exit_price - entry_price) if side == "long" else (entry_price - exit_price)) * size
            else:
                realized_pnl = float(db_position.get("unrealized_pnl") or 0)

            await self._close_position(db_position["id"], exit_price, realized_pnl)
            logger.info(
                f"Position {symbol} no longer on Hyperliquid; closed DB record at ${exit_price} "
                f"with realized PnL ${realized_pnl:.2f}"
            )
        except Exception as e:
            logger.error(f"Error closing missing position {db_position.get('id')}: {e}")

    async def _check_new_positions(
        self,
        user_id: str,
        allocation_id: str,
        hyperliquid_positions: Dict[str, Any],
        db_positions: List[Dict[str, Any]]
    ):
        """Check for new positions that aren't in the database yet."""
        try:
            all_db_positions = await self.db_service.get_agent_positions(allocation_id, active_only=False)
            db_symbols = {self._normalize_symbol(pos["symbol"]) for pos in (all_db_positions or db_positions)}

            for symbol, hl_position in hyperliquid_positions.items():
                if self._normalize_symbol(symbol) not in db_symbols:
                    # New position found
                    await self._create_position_record(
                        user_id, allocation_id, symbol, hl_position
                    )
                    
        except Exception as e:
            logger.error(f"Error checking new positions: {e}")
    
    async def _create_position_record(
        self,
        user_id: str,
        allocation_id: str,
        symbol: str,
        hl_position: Any
    ):
        """Create a new position record in the database."""
        try:
            # Get the most recent trade for this symbol
            trades = await self.db_service.get_agent_trades(allocation_id)
            recent_trade = None
            
            for trade in trades:
                if self._normalize_symbol(trade["symbol"]) == self._normalize_symbol(symbol) and trade["status"] == "filled":
                    recent_trade = trade
                    break

            if not recent_trade:
                logger.debug(f"No recent filled trade found for external position {symbol}")
                return

            # Calculate position metrics
            entry_price = float(recent_trade["entry_price"])
            size = float(recent_trade["position_size"])
            side = "long" if recent_trade["side"] == "buy" else "short"
            leverage = float(recent_trade["leverage"] or 1)

            current_price = float(getattr(hl_position, "mark_price", 0) or 0) or entry_price
            position_value = size * current_price

            # USD PnL straight from Hyperliquid; fall back to notional price move
            unrealized_pnl = float(getattr(hl_position, "unrealized_pnl", 0) or 0)
            if unrealized_pnl == 0 and entry_price > 0:
                price_diff = (current_price - entry_price) if side == "long" else (entry_price - current_price)
                unrealized_pnl = price_diff * size

            margin_used = position_value / leverage if leverage > 0 else position_value
            unrealized_pnl_percent = (unrealized_pnl / margin_used) * 100 if margin_used > 0 else 0.0
            margin_ratio = margin_used / (position_value + unrealized_pnl) if (position_value + unrealized_pnl) > 0 else 1.0
            
            # Create position record
            trade_metadata = recent_trade.get("trade_metadata") or {}
            protective_orders = trade_metadata.get("protective_orders") or {}
            stop_record = protective_orders.get("stop_loss") or {}
            exchange_stop_live = bool(stop_record.get("success") and stop_record.get("order_id"))
            position_data = {
                "user_id": user_id,
                "allocation_id": allocation_id,
                "trade_id": recent_trade["id"],
                "agent_type": "yuki",
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "current_price": current_price,
                "leverage": leverage,
                "position_value": position_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "margin_used": margin_used,
                "margin_ratio": margin_ratio,
                "is_active": True,
                "hyperliquid_position_id": f"{symbol}_{int(datetime.now().timestamp())}",
                "metadata": {
                    "created_from_monitoring": True,
                    "monitoring_started": datetime.now().isoformat(),
                    "signal_id": trade_metadata.get("signal_id"),
                    "execution_policy": "signal_led_adaptive",
                    "stop_loss": trade_metadata.get("stop_loss"),
                    "active_stop": trade_metadata.get("stop_loss") if exchange_stop_live else None,
                    "policy_stop": trade_metadata.get("stop_loss"),
                    "stop_stage": "initial" if exchange_stop_live else "initial_repair_pending",
                    "target_1": trade_metadata.get("target_1"),
                    "target_2": trade_metadata.get("target_2"),
                    "protective_orders": protective_orders,
                    "runner_enabled": bool(protective_orders.get("runner_enabled")),
                    "runner_size": protective_orders.get("runner_size"),
                    "original_size": size,
                    "initial_size": trade_metadata.get("initial_size") or size,
                    "initial_entry_price": trade_metadata.get("signal_entry_price") or entry_price,
                    "initial_stop_loss": trade_metadata.get("stop_loss"),
                    "initial_margin_used": trade_metadata.get("requested_trade_amount") or margin_used,
                    "signal_expires_at": trade_metadata.get("signal_expires_at"),
                    "time_horizon": trade_metadata.get("time_horizon"),
                }
            }
            
            await self.db_service.store_agent_position(position_data)
            logger.info(f"Created position record for {symbol}: {side} {size} @ ${current_price}")
            
        except Exception as e:
            logger.error(f"Error creating position record: {e}")
    
    async def _check_risk_limits(
        self,
        user_id: str,
        allocation_id: str,
        positions: List[Dict[str, Any]]
    ):
        """Check risk limits and execute risk management actions."""
        try:
            source_signal_ids = {
                str((position.get("position_metadata") or {}).get("signal_id") or "")
                for position in positions
            }
            source_signal_ids.discard("")
            source_statuses: Dict[str, str] = {}
            if source_signal_ids:
                try:
                    source_rows = (
                        self.db_service.db.table("platform_signals")
                        .select("signal_id,status")
                        .in_("signal_id", list(source_signal_ids))
                        .execute()
                    ).data or []
                    source_statuses = {
                        str(row.get("signal_id")): str(row.get("status") or "").lower()
                        for row in source_rows
                    }
                except Exception as source_error:
                    # Missing state is not proof of invalidation. Existing
                    # exchange-side protection remains active and the next
                    # monitor cycle retries.
                    logger.warning(
                        "Could not verify Yuki source lifecycle for allocation %s: %s",
                        allocation_id,
                        source_error,
                    )

            for position in positions:
                metadata = position.get("position_metadata") or {}
                signal_id = str(metadata.get("signal_id") or "")
                if signal_id:
                    position["_source_signal_status"] = source_statuses.get(signal_id)
                risk_metrics = await self._calculate_position_risk(position)
                
                if risk_metrics.should_close:
                    logger.warning(f"Risk limit triggered for {position['symbol']}: {risk_metrics.reason}")
                    await self._execute_risk_management_action(
                        user_id, allocation_id, position, risk_metrics
                    )
                else:
                    review_event = self._pending_position_review_event(position, risk_metrics)
                    if review_event and self._position_review_due(position, review_event):
                        await self._review_position_event(
                            user_id=user_id,
                            allocation_id=allocation_id,
                            position=position,
                            risk_metrics=risk_metrics,
                            event=review_event,
                        )
                    elif not review_event:
                        await self._maybe_review_scale_in_opportunity(
                            user_id=user_id,
                            allocation_id=allocation_id,
                            position=position,
                            risk_metrics=risk_metrics,
                        )
                    
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")

    @classmethod
    def _position_horizon_elapsed(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
    ) -> bool:
        metadata = position.get("position_metadata") or {}
        return risk_metrics.time_in_position_hours >= cls._position_hold_limit_hours(metadata)

    @classmethod
    def _horizon_review_due(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
        now: Optional[datetime] = None,
    ) -> bool:
        """Debounce durable horizon reviews while still allowing periodic reassessment."""
        if not cls._position_horizon_elapsed(position, risk_metrics):
            return False
        return cls._position_review_due(
            position,
            "time_horizon_elapsed",
            now=now,
        )

    @classmethod
    def _position_review_due(
        cls,
        position: Dict[str, Any],
        event: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Reject overlapping/stale duplicate execution; this is not scheduling."""
        now = now or datetime.now(timezone.utc)
        metadata = position.get("position_metadata") or {}
        in_progress = metadata.get("react_review_in_progress_by_event") or {}
        in_progress_at = cls._parse_datetime(in_progress.get(event))
        if event == "time_horizon_elapsed" and not in_progress_at:
            in_progress_at = cls._parse_datetime(metadata.get("horizon_review_in_progress_at"))
        lock_minutes = max(
            1.0,
            float(getattr(settings, "YUKI_HORIZON_REVIEW_LOCK_MINUTES", 15.0)),
        )
        if in_progress_at and (now - in_progress_at).total_seconds() < lock_minutes * 60.0:
            return False
        return True

    @classmethod
    def _review_event_signature(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
        event: str,
    ) -> str:
        """Identify the concrete state change behind an event."""
        metadata = position.get("position_metadata") or {}
        if event == "source_signal_invalidated":
            return f"{metadata.get('signal_id') or 'unknown'}:invalidated"
        if event in {"target_1_confirmed", "target_2_confirmed"}:
            event_state = (metadata.get("react_review_events") or {}).get(event) or {}
            return str(
                metadata.get(f"{event}_at")
                or event_state.get("requested_at")
                or "confirmed"
            )
        if event == "time_horizon_elapsed":
            return (
                f"{position.get('created_at')}:{metadata.get('time_horizon')}:"
                f"{cls._position_hold_limit_hours(metadata):.6g}"
            )
        if event == "winner_scale_in_opportunity":
            return str(metadata.get("scale_in_candidate_signature") or "scale_in_opportunity")
        if event == "profit_without_targets":
            return "no_targets:profit_above_40"
        if event == "legacy_profit_protection_touched":
            return f"{metadata.get('target_1')}:{metadata.get('active_stop')}"
        if event == "post_target_breakeven_touched":
            return f"{position.get('entry_price')}:{metadata.get('target_1_confirmed_at')}"
        if event == "protective_order_issue":
            protective = metadata.get("protective_orders") or {}
            errors = [str(metadata.get("stop_update_error") or "")]
            for order in [protective.get("stop_loss") or {}, *(protective.get("take_profit") or [])]:
                if order.get("success") is False or order.get("error"):
                    errors.append(str(order.get("error") or "rejected"))
            return "|".join(value for value in errors if value)
        if event == "material_market_change":
            change = cls._material_market_change(position, risk_metrics) or {}
            price_trigger = max(
                0.1,
                float(getattr(settings, "YUKI_REACT_PRICE_MOVE_TRIGGER_PCT", 3.0)),
            )
            pnl_trigger = max(
                0.5,
                float(getattr(settings, "YUKI_REACT_PNL_MOVE_TRIGGER_POINTS", 10.0)),
            )
            price_move_pct = float(change.get("price_move_pct") or 0.0)
            pnl_move_points = float(change.get("pnl_move_points") or 0.0)
            price_bucket = int(abs(price_move_pct) / price_trigger) * (1 if price_move_pct >= 0 else -1)
            pnl_bucket = int(abs(pnl_move_points) / pnl_trigger) * (1 if pnl_move_points >= 0 else -1)
            return (
                f"price_bucket={price_bucket}:"
                f"pnl_bucket={pnl_bucket}:"
                f"size={float(position.get('size') or 0.0):.6g}"
            )
        return event

    @classmethod
    def _normalize_review_hash_value(
        cls,
        value: Any,
        *,
        depth: int = 0,
    ) -> Any:
        """Build a stable, compact representation for live-review dedupe hashes."""
        if depth >= 4:
            return "<trimmed>"
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 6)
        if isinstance(value, str):
            normalized = " ".join(value.split())
            return normalized[:160]
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for key in sorted(value.keys(), key=str)[:20]:
                compact[str(key)] = cls._normalize_review_hash_value(value.get(key), depth=depth + 1)
            if len(value) > 20:
                compact["_truncated_keys"] = len(value) - 20
            return compact
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            compact_items = [
                cls._normalize_review_hash_value(item, depth=depth + 1)
                for item in items[:10]
            ]
            if len(items) > 10:
                compact_items.append({"_truncated_items": len(items) - 10})
            return compact_items
        return cls._normalize_review_hash_value(str(value), depth=depth + 1)

    @classmethod
    def _position_review_state_hash(
        cls,
        position_context: Dict[str, Any],
        event_context: Dict[str, Any],
    ) -> str:
        """Hash the materially relevant review state for a live-position event."""
        metadata = position_context.get("position_metadata") or {}
        payload = {
            "symbol": position_context.get("symbol"),
            "side": position_context.get("side"),
            "event": event_context.get("event"),
            "source_signal_status": event_context.get("source_signal_status"),
            "declared_time_horizon": event_context.get("declared_time_horizon"),
            "elapsed_hours": round(float(event_context.get("elapsed_hours") or 0.0), 3),
            "review_threshold_hours": round(float(event_context.get("review_threshold_hours") or 0.0), 3),
            "entry_price": position_context.get("entry_price"),
            "current_price": position_context.get("current_price"),
            "size": position_context.get("size"),
            "unrealized_pnl_percent": position_context.get("unrealized_pnl_percent"),
            "margin_ratio": position_context.get("margin_ratio"),
            "active_stop": (
                metadata.get("active_stop")
                or metadata.get("policy_stop")
                or metadata.get("stop_loss")
            ),
            "target_1": metadata.get("target_1"),
            "target_2": metadata.get("target_2"),
            "event_details": cls._normalize_review_hash_value(event_context.get("event_details")),
            "policy_proposal": cls._normalize_review_hash_value(event_context.get("policy_proposal")),
        }
        serialized = json.dumps(
            cls._normalize_review_hash_value(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def _cached_review_is_usable(
        cls,
        review_record: Any,
        *,
        now: datetime,
    ) -> bool:
        """Allow cache hits only for recent, successfully executed reviews."""
        if not isinstance(review_record, dict):
            return False
        reviewed_at = cls._parse_datetime(review_record.get("reviewed_at"))
        if reviewed_at is None:
            return False
        ttl_minutes = max(
            1,
            int(getattr(settings, "YUKI_REACT_REVIEW_CACHE_MINUTES", 20) or 20),
        )
        if (now - reviewed_at).total_seconds() > ttl_minutes * 60.0:
            return False
        action = str(review_record.get("action") or "").strip().upper()
        execution_status = str(review_record.get("execution_status") or "").strip().lower()
        if not action:
            return False
        if execution_status.startswith("execution_error") or execution_status.startswith("error"):
            return False
        return True

    @classmethod
    def _event_needs_review(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
        event: str,
    ) -> bool:
        metadata = position.get("position_metadata") or {}
        reviewed = metadata.get("react_reviewed_event_signatures") or {}
        return reviewed.get(event) != cls._review_event_signature(position, risk_metrics, event)

    @classmethod
    def _material_market_change(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
    ) -> Optional[Dict[str, float]]:
        """Detect a meaningful move from entry or the most recent ReAct decision."""
        metadata = position.get("position_metadata") or {}
        snapshot = metadata.get("react_last_review_snapshot") or {}
        reference_price = cls._positive_float(snapshot.get("current_price")) or cls._positive_float(
            position.get("entry_price")
        )
        current_price = cls._positive_float(position.get("current_price"))
        reference_pnl = float(snapshot.get("unrealized_pnl_percent") or 0.0)
        current_pnl = float(risk_metrics.unrealized_pnl_percent)
        price_move_pct = (
            (current_price / reference_price - 1.0) * 100.0
            if current_price and reference_price
            else 0.0
        )
        pnl_move_points = current_pnl - reference_pnl
        price_trigger = max(
            0.1,
            float(getattr(settings, "YUKI_REACT_PRICE_MOVE_TRIGGER_PCT", 3.0)),
        )
        pnl_trigger = max(
            0.5,
            float(getattr(settings, "YUKI_REACT_PNL_MOVE_TRIGGER_POINTS", 10.0)),
        )
        if abs(price_move_pct) < price_trigger and abs(pnl_move_points) < pnl_trigger:
            return None
        return {
            "reference_price": float(reference_price or 0.0),
            "current_price": float(current_price or 0.0),
            "price_move_pct": price_move_pct,
            "reference_pnl_percent": reference_pnl,
            "current_pnl_percent": current_pnl,
            "pnl_move_points": pnl_move_points,
        }

    @staticmethod
    def _has_protective_order_issue(position: Dict[str, Any]) -> bool:
        metadata = position.get("position_metadata") or {}
        if metadata.get("stop_update_error"):
            return True
        protective = metadata.get("protective_orders") or {}
        orders = [protective.get("stop_loss") or {}, *(protective.get("take_profit") or [])]
        return any(order.get("success") is False or order.get("error") for order in orders)

    @classmethod
    def _pending_position_review_event(
        cls,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
    ) -> Optional[str]:
        """Return the highest-priority non-emergency event Yuki must decide."""
        metadata = position.get("position_metadata") or {}
        review_events = metadata.get("react_review_events") or {}
        entry_event = review_events.get("entry_filled") or {}
        if (
            str(entry_event.get("status") or "").lower() == "pending"
            and cls._event_needs_review(position, risk_metrics, "entry_filled")
        ):
            return "entry_filled"

        if str(position.get("_source_signal_status") or "").lower() == "invalidated":
            event = "source_signal_invalidated"
            if cls._event_needs_review(position, risk_metrics, event):
                return event

        for event in ("target_2_confirmed", "target_1_confirmed"):
            if event not in review_events:
                continue
            event_state = review_events.get(event) or {}
            if str(event_state.get("status") or "pending").lower() == "pending":
                return event

        # Existing positions may have reached a runner milestone before this
        # release. Give those milestones one ReAct pass instead of silently
        # inheriting the old deterministic trailing behavior.
        if metadata.get("runner_enabled"):
            for event in ("target_2_confirmed", "target_1_confirmed"):
                if metadata.get(event) and event not in review_events:
                    return event

        if cls._has_protective_order_issue(position):
            event = "protective_order_issue"
            if cls._event_needs_review(position, risk_metrics, event):
                return event

        target_1 = cls._positive_float(metadata.get("target_1"))
        target_2 = cls._positive_float(metadata.get("target_2"))
        side = str(position.get("side") or "").lower()
        current_price = float(position.get("current_price") or 0.0)
        entry_price = float(position.get("entry_price") or 0.0)
        if (
            metadata.get("buffer_hit")
            and target_1
            and cls._stop_loss_hit(side, current_price, target_1)
        ):
            event = "legacy_profit_protection_touched"
            if cls._event_needs_review(position, risk_metrics, event):
                return event
        if (
            settings.YUKI_MOVE_STOP_TO_BREAKEVEN_AFTER_TP1
            and metadata.get("target_1_hit")
            and not metadata.get("buffer_hit")
            and entry_price > 0
            and cls._stop_loss_hit(side, current_price, entry_price)
        ):
            event = "post_target_breakeven_touched"
            if cls._event_needs_review(position, risk_metrics, event):
                return event
        if not (target_1 or target_2) and risk_metrics.unrealized_pnl_percent > 40.0:
            event = "profit_without_targets"
            if cls._event_needs_review(position, risk_metrics, event):
                return event
        if cls._position_horizon_elapsed(position, risk_metrics):
            event = "time_horizon_elapsed"
            if cls._event_needs_review(position, risk_metrics, event):
                return event
        if cls._material_market_change(position, risk_metrics):
            return "material_market_change"
        return None

    def _persist_position_metadata(self, position_id: Any, metadata: Dict[str, Any]) -> bool:
        """Persist lifecycle review state without overwriting live PnL fields."""
        if not position_id:
            return False
        try:
            response = (
                self.db_service.db.table("agent_positions")
                .update({
                    "position_metadata": metadata,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .eq("id", position_id)
                .eq("is_active", True)
                .execute()
            )
            return bool(response.data)
        except Exception as exc:
            logger.warning("Could not persist Yuki horizon review metadata: %s", exc)
            return False

    async def _resolve_react_agent(self, user_id: str, allocation_id: str) -> Any:
        resolver = getattr(self, "react_agent_resolver", None)
        if resolver:
            resolved = resolver(user_id)
            if asyncio.iscoroutine(resolved):
                resolved = await resolved
            if resolved is not None:
                return resolved

        cache = getattr(self, "_react_agents", None)
        if cache is None:
            cache = {}
            self._react_agents = cache
        if user_id not in cache:
            from kata.agents.yuki_agent import YukiAgent

            cache[user_id] = YukiAgent(
                user_id=user_id,
                config={
                    "allocation_id": allocation_id,
                    "llm_provider": getattr(settings, "YUKI_REACT_LLM_PROVIDER", "deepseek"),
                    "llm_model": getattr(settings, "YUKI_REACT_LLM_MODEL", "deepseek-v4-flash"),
                    "llm_thinking_type": getattr(settings, "YUKI_REACT_THINKING_TYPE", "disabled"),
                    "llm_reasoning_effort": getattr(settings, "YUKI_REACT_REASONING_EFFORT", "low"),
                    "llm_strict_json": bool(getattr(settings, "YUKI_REACT_STRICT_JSON", True)),
                    "llm_max_output_tokens": int(
                        getattr(settings, "YUKI_REACT_MAX_OUTPUT_TOKENS", 1200) or 1200
                    ),
                    "risk_params": {},
                },
                hyperliquid_service=self.hyperliquid_service,
            )
        return cache[user_id]

    async def _review_elapsed_horizon(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
    ) -> Dict[str, Any]:
        """Backward-compatible horizon entry point for the generic ReAct review."""
        return await self._review_position_event(
            user_id=user_id,
            allocation_id=allocation_id,
            position=position,
            risk_metrics=risk_metrics,
            event="time_horizon_elapsed",
        )

    async def _review_position_event(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
        event: str,
        trigger_context: Optional[Dict[str, Any]] = None,
        acquire_cycle_lock: bool = True,
    ) -> Dict[str, Any]:
        """Ask Yuki to decide and immediately execute a live-position action."""
        now = datetime.now(timezone.utc)
        metadata = dict(position.get("position_metadata") or {})
        event_signature = self._review_event_signature(position, risk_metrics, event)
        review_state_hash = event_signature
        in_progress = dict(metadata.get("react_review_in_progress_by_event") or {})
        in_progress[event] = now.isoformat()
        metadata["react_review_in_progress_by_event"] = in_progress
        if event == "time_horizon_elapsed":
            metadata.setdefault("horizon_elapsed_at", now.isoformat())
            metadata["horizon_review_in_progress_at"] = now.isoformat()
        position["position_metadata"] = metadata
        self._persist_position_metadata(position.get("id"), metadata)

        try:
            react_agent = await self._resolve_react_agent(user_id, allocation_id)
            position_context = {
                key: position.get(key)
                for key in (
                    "id", "trade_id", "allocation_id", "symbol", "side", "size",
                    "entry_price", "current_price", "leverage", "margin_used",
                    "position_value", "unrealized_pnl", "unrealized_pnl_percent",
                    "margin_ratio", "created_at",
                )
            }
            position_context["position_metadata"] = metadata
            review_events = metadata.get("react_review_events") or {}
            event_details = review_events.get(event)
            if event == "material_market_change":
                event_details = self._material_market_change(position, risk_metrics)
            elif event == "protective_order_issue":
                event_details = {"signature": event_signature}
            event_context = dict(trigger_context or {})
            event_context.update({
                "event": event,
                "declared_time_horizon": metadata.get("time_horizon"),
                "elapsed_hours": risk_metrics.time_in_position_hours,
                "review_threshold_hours": self._position_hold_limit_hours(metadata),
                "source_signal_status": position.get("_source_signal_status"),
                "event_details": event_details,
            })
            review_state_hash = self._position_review_state_hash(position_context, event_context)
            review_state_hashes = dict(metadata.get("react_review_state_hashes") or {})
            prior_reviews = dict(metadata.get("react_review_records_by_event") or {})
            cached_review = prior_reviews.get(event) or {}
            if (
                review_state_hashes.get(event) == review_state_hash
                and self._cached_review_is_usable(cached_review, now=now)
            ):
                logger.info(
                    "Skipping duplicate Yuki %s review for %s; state hash unchanged",
                    event,
                    position.get("symbol"),
                )
                metadata["react_cached_review_hits"] = int(metadata.get("react_cached_review_hits") or 0) + 1
                metadata["react_last_cached_review_at"] = now.isoformat()
                reviewed_signatures = dict(metadata.get("react_reviewed_event_signatures") or {})
                reviewed_signatures[event] = event_signature
                metadata["react_reviewed_event_signatures"] = reviewed_signatures
                review_state_hashes[event] = review_state_hash
                metadata["react_review_state_hashes"] = review_state_hashes
                in_progress = dict(metadata.get("react_review_in_progress_by_event") or {})
                in_progress.pop(event, None)
                metadata["react_review_in_progress_by_event"] = in_progress
                if event == "time_horizon_elapsed":
                    metadata.pop("horizon_review_in_progress_at", None)
                review_events = dict(metadata.get("react_review_events") or {})
                if event in review_events or event in {"target_1_confirmed", "target_2_confirmed"}:
                    event_state = dict(review_events.get(event) or {})
                    event_state.update({
                        "status": "reviewed",
                        "reviewed_at": now.isoformat(),
                        "action": str(cached_review.get("action") or "HOLD").upper(),
                        "execution_status": cached_review.get("execution_status"),
                        "cached": True,
                    })
                    review_events[event] = event_state
                    metadata["react_review_events"] = review_events
                position["position_metadata"] = metadata
                self._persist_position_metadata(position.get("id"), metadata)
                cached_decision = {
                    "symbol": position.get("symbol"),
                    "action": str(cached_review.get("action") or "HOLD").upper(),
                    "confidence": cached_review.get("confidence"),
                    "market_regime": str(cached_review.get("market_regime") or "cached"),
                    "reasoning": str(cached_review.get("reasoning") or ""),
                    "evidence_used": list(cached_review.get("evidence_used") or []),
                    "next_review_conditions": list(cached_review.get("next_review_conditions") or []),
                    "episode_id": cached_review.get("episode_id"),
                    "llm_usage": cached_review.get("llm_usage"),
                    "llm_retry_used": bool(cached_review.get("llm_retry_used")),
                    "cached_review": True,
                    "trigger": event,
                }
                return {
                    "decision": cached_decision,
                    "execution_status": cached_review.get("execution_status"),
                    "cached": True,
                }
            decision = await react_agent.execute_position_management_react_cycle(
                position_context=position_context,
                trigger_context=event_context,
            )
        except Exception as exc:
            logger.error("Could not run Yuki %s review for %s: %s", event, position.get("symbol"), exc)
            decision = {
                "action": "HOLD",
                "confidence": 0.0,
                "next_review_conditions": ["A new material position event occurs"],
                "reasoning": f"ReAct review unavailable; existing position protection retained: {exc}",
                "evidence_used": [],
            }

        try:
            execution_status = await self._execute_horizon_decision(
                user_id=user_id,
                allocation_id=allocation_id,
                position=position,
                risk_metrics=risk_metrics,
                decision=decision,
                event=event,
                acquire_cycle_lock=acquire_cycle_lock,
            )
        except Exception as exc:
            logger.error(
                "Yuki %s decision could not execute for %s: %s",
                event,
                position.get("symbol"),
                exc,
            )
            execution_status = f"execution_error: {exc}"
        metadata = dict(position.get("position_metadata") or metadata)
        completed_at = datetime.now(timezone.utc)
        review_record = {
            "event": event,
            "reviewed_at": completed_at.isoformat(),
            "elapsed_hours": risk_metrics.time_in_position_hours,
            "threshold_hours": self._position_hold_limit_hours(metadata),
            "action": str(decision.get("action") or "HOLD").upper(),
            "confidence": decision.get("confidence"),
            "market_regime": decision.get("market_regime"),
            "reasoning": decision.get("reasoning"),
            "evidence_used": decision.get("evidence_used") or [],
            "episode_id": decision.get("episode_id"),
            "next_review_conditions": decision.get("next_review_conditions") or [],
            "execution_status": execution_status,
            "llm_usage": decision.get("llm_usage"),
            "llm_retry_used": bool(decision.get("llm_retry_used")),
            "review_state_hash": review_state_hash,
        }
        metadata["react_review_count"] = int(metadata.get("react_review_count") or 0) + 1
        metadata["react_last_review"] = review_record
        metadata["react_review_history"] = [
            *list(metadata.get("react_review_history") or []),
            review_record,
        ][-10:]
        review_records_by_event = dict(metadata.get("react_review_records_by_event") or {})
        review_records_by_event[event] = review_record
        metadata["react_review_records_by_event"] = review_records_by_event
        reviewed_signatures = dict(metadata.get("react_reviewed_event_signatures") or {})
        reviewed_signatures[event] = event_signature
        metadata["react_reviewed_event_signatures"] = reviewed_signatures
        review_state_hashes = dict(metadata.get("react_review_state_hashes") or {})
        review_state_hashes[event] = review_state_hash
        metadata["react_review_state_hashes"] = review_state_hashes
        metadata["react_last_review_snapshot"] = {
            "reviewed_at": completed_at.isoformat(),
            "event": event,
            "current_price": float(position.get("current_price") or 0.0),
            "unrealized_pnl_percent": float(risk_metrics.unrealized_pnl_percent),
            "size": float(position.get("size") or 0.0),
            "source_signal_status": position.get("_source_signal_status"),
        }
        # Remove scheduling fields left by the previous implementation. Future
        # reviews are caused only by a new event signature or material move.
        metadata.pop("react_review_next_at_by_event", None)
        metadata.pop("horizon_review_next_at", None)
        in_progress = dict(metadata.get("react_review_in_progress_by_event") or {})
        in_progress.pop(event, None)
        metadata["react_review_in_progress_by_event"] = in_progress

        review_events = dict(metadata.get("react_review_events") or {})
        if event in review_events or event in {"target_1_confirmed", "target_2_confirmed"}:
            event_state = dict(review_events.get(event) or {})
            event_state.update({
                "status": "reviewed",
                "reviewed_at": completed_at.isoformat(),
                "action": review_record["action"],
                "execution_status": execution_status,
            })
            review_events[event] = event_state
            metadata["react_review_events"] = review_events

        if event == "time_horizon_elapsed":
            metadata["horizon_review_count"] = int(metadata.get("horizon_review_count") or 0) + 1
            metadata["horizon_last_review"] = review_record
            metadata["horizon_review_history"] = [
                *list(metadata.get("horizon_review_history") or []),
                review_record,
            ][-10:]
            metadata.pop("horizon_review_in_progress_at", None)
        position["position_metadata"] = metadata
        self._persist_position_metadata(position.get("id"), metadata)
        return {"decision": decision, "execution_status": execution_status}

    async def _execute_horizon_decision(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
        decision: Dict[str, Any],
        event: str = "",
        acquire_cycle_lock: bool = True,
    ) -> str:
        """Execute only the lifecycle action Yuki selected, under existing safety limits."""
        action = str(decision.get("action") or "HOLD").strip().upper()

        # Breakeven lock guardrail: when a source signal is invalidated and the
        # position is already in profit, do NOT allow an immediate full exit.
        # Instead, ratchet the stop to the fee-adjusted break-even price so the
        # trade becomes risk-free while giving price time to continue or reverse
        # to entry. This prevents leaving good P&L on the table just because the
        # originating signal was superseded by a newer analysis pass.
        if (
            event == "source_signal_invalidated"
            and action == "FULL_EXIT"
            and float(risk_metrics.unrealized_pnl_percent) > 0.0
        ):
            entry_price = self._positive_float(position.get("entry_price")) or 0.0
            side = str(position.get("side") or "").lower()
            # Fee-adjusted breakeven: add ~0.1 % buffer above entry for longs
            # (below for shorts) so the stop covers the round-trip commission.
            fee_buffer_pct = 0.001
            if entry_price > 0:
                if side == "long":
                    breakeven_stop = round(entry_price * (1.0 - fee_buffer_pct), 6)
                else:
                    breakeven_stop = round(entry_price * (1.0 + fee_buffer_pct), 6)
                override_reasoning = (
                    f"Breakeven lock applied: signal invalidated but position is in profit "
                    f"({float(risk_metrics.unrealized_pnl_percent):.2f}%). "
                    f"Overriding FULL_EXIT to ADJUST_STOP at {breakeven_stop} (entry={entry_price}) "
                    f"to preserve risk-free exposure. Original ReAct reasoning: {decision.get('reasoning')}"
                )
                logger.info(
                    "🔒 [BreakevenLock] %s signal_invalidated + in profit → overriding FULL_EXIT "
                    "to ADJUST_STOP at %.6g (pnl=%.2f%%)",
                    position.get("symbol"), breakeven_stop, float(risk_metrics.unrealized_pnl_percent),
                )
                decision = {
                    **decision,
                    "action": "ADJUST_STOP",
                    "stop_price": breakeven_stop,
                    "reasoning": override_reasoning,
                }
                action = "ADJUST_STOP"

        if action == "FULL_EXIT":
            from kata.services.agent_allocation_service import get_agent_allocation_service

            allocation_service = get_agent_allocation_service()
            async def close_position() -> bool:
                return await self._execute_risk_management_action(
                    user_id,
                    allocation_id,
                    position,
                    PositionRiskMetrics(
                        liquidation_risk=risk_metrics.liquidation_risk,
                        margin_ratio=risk_metrics.margin_ratio,
                        unrealized_pnl_percent=risk_metrics.unrealized_pnl_percent,
                        time_in_position_hours=risk_metrics.time_in_position_hours,
                        risk_level=RiskLevel.MEDIUM,
                        should_close=True,
                        reason=(
                            "Yuki ReAct position review selected FULL_EXIT: "
                            f"{decision.get('reasoning') or 'thesis no longer supported'}"
                        ),
                    ),
                )

            if acquire_cycle_lock:
                async with allocation_service.yuki_signal_cycle_lock:
                    closed = await close_position()
            else:
                closed = await close_position()
            return "full_exit_filled" if closed else "full_exit_not_executed"

        if action == "PARTIAL_EXIT":
            try:
                fraction = max(0.10, min(0.90, float(decision.get("size_fraction") or 0.25)))
            except (TypeError, ValueError):
                fraction = 0.25
            from kata.services.agent_allocation_service import get_agent_allocation_service

            allocation_service = get_agent_allocation_service()
            async def partial_exit() -> bool:
                return await self._execute_partial_horizon_exit(
                    user_id=user_id,
                    allocation_id=allocation_id,
                    position=position,
                    fraction=fraction,
                    reasoning=str(decision.get("reasoning") or "Yuki position review"),
                )
            if acquire_cycle_lock:
                async with allocation_service.yuki_signal_cycle_lock:
                    executed = await partial_exit()
            else:
                executed = await partial_exit()
            return "partial_exit_filled" if executed else "partial_exit_not_executed"

        if action == "ADD":
            position["_react_position_add_authorized"] = True
            try:
                added = await self._maybe_scale_in_winner(
                    user_id=user_id,
                    allocation_id=allocation_id,
                    position=position,
                    decision=decision,
                    acquire_cycle_lock=acquire_cycle_lock,
                )
            finally:
                position.pop("_react_position_add_authorized", None)
            return "add_filled" if added else "add_blocked_by_risk_guardrails"

        if action == "ADJUST_STOP":
            try:
                requested_stop = float(decision.get("stop_price") or 0.0)
            except (TypeError, ValueError):
                requested_stop = 0.0
            metadata = dict(position.get("position_metadata") or {})
            side = str(position.get("side") or "").lower()
            current_price = float(position.get("current_price") or 0.0)
            existing_stop = (
                self._positive_float(metadata.get("active_stop"))
                or self._positive_float(metadata.get("policy_stop"))
                or self._positive_float(metadata.get("stop_loss"))
            )
            if requested_stop <= 0 or self._stop_loss_hit(side, current_price, requested_stop):
                return "stop_adjustment_rejected"
            requested_stop = self._tighter_stop(side, existing_stop, requested_stop)
            adjusted = await self._ratchet_exchange_stop(
                user_id=user_id,
                allocation_id=allocation_id,
                symbol=str(position.get("symbol") or ""),
                side=side,
                current_price=current_price,
                remaining_size=float(position.get("size") or 0.0),
                desired_stop=requested_stop,
                stage="react_position_review",
                metadata=metadata,
            )
            position["position_metadata"] = metadata
            return "stop_adjusted" if adjusted else "stop_adjustment_not_tighter_or_unavailable"

        return "held_existing_position"

    async def _execute_partial_horizon_exit(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        fraction: float,
        reasoning: str,
    ) -> bool:
        """Reduce a live position and resize its exchange protection to the remainder."""
        symbol = str(position.get("symbol") or "")
        side = str(position.get("side") or "").lower()
        current_size = float(position.get("size") or 0.0)
        if current_size <= 0:
            return False
        size_decimals = await self.hyperliquid_service.get_perp_sz_decimals(symbol)
        requested_size = self.hyperliquid_service._round_size_for_hl(
            current_size * fraction,
            size_decimals,
        )
        remaining_size = self.hyperliquid_service._round_size_for_hl(
            current_size - requested_size,
            size_decimals,
        )
        if requested_size <= 0 or remaining_size <= 0:
            return False

        wallet_address = await self._resolve_allocation_wallet(allocation_id)
        if not await self._ensure_trading_client(user_id, wallet_address):
            return False
        close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        result = await self.hyperliquid_service.place_order_live(
            symbol=symbol,
            side=close_side,
            size=requested_size,
            order_type=OrderType.MARKET,
            reduce_only=True,
        )
        if not result.success:
            logger.error("Yuki partial horizon exit failed for %s: %s", symbol, result.error)
            return False
        if result.status and str(result.status).lower() != "filled":
            logger.error(
                "Yuki partial horizon exit for %s was not confirmed filled (status %s)",
                symbol,
                result.status,
            )
            return False

        filled_size = float(result.filled_size or requested_size)
        remaining_size = max(0.0, current_size - filled_size)
        if remaining_size <= 0:
            return False
        metadata = dict(position.get("position_metadata") or {})
        partial_record = {
            "closed_fraction": filled_size / current_size,
            "closed_size": filled_size,
            "remaining_size": remaining_size,
            "fill_price": result.average_price or result.price or position.get("current_price"),
            "order_id": result.order_id,
            "reasoning": reasoning,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata["react_partial_exits"] = [
            *list(metadata.get("react_partial_exits") or []),
            partial_record,
        ][-10:]

        current_price = float(position.get("current_price") or position.get("entry_price") or 0.0)
        active_stop = (
            self._positive_float(metadata.get("active_stop"))
            or self._positive_float(metadata.get("policy_stop"))
            or self._positive_float(metadata.get("stop_loss"))
        )
        if active_stop:
            await self._ratchet_exchange_stop(
                user_id=user_id,
                allocation_id=allocation_id,
                symbol=symbol,
                side=side,
                current_price=current_price,
                remaining_size=remaining_size,
                desired_stop=active_stop,
                stage="react_partial_exit_resize",
                metadata=metadata,
                force_resize=True,
            )

        # Scale remaining TP legs to the new position. Successfully cancelled
        # legs are immediately eligible for the normal repair path; a failed
        # cancel stays recorded as live and reduce-only prevents a reversal.
        protective = dict(metadata.get("protective_orders") or {})
        resized_targets = []
        remaining_ratio = remaining_size / current_size
        for order in list(protective.get("take_profit") or []):
            resized = dict(order)
            order_id = resized.get("order_id")
            if resized.get("success") and order_id:
                cancelled = await self.hyperliquid_service.cancel_order(symbol, order_id)
                if cancelled:
                    resized.update({
                        "success": False,
                        "order_id": None,
                        "error": "resizing_after_react_partial_exit",
                        "size": (self._positive_float(resized.get("size")) or 0.0) * remaining_ratio,
                    })
            else:
                resized["size"] = (
                    self._positive_float(resized.get("size")) or 0.0
                ) * remaining_ratio
            resized_targets.append(resized)
        protective["take_profit"] = resized_targets
        metadata["protective_orders"] = protective
        await self._repair_take_profit_orders(
            user_id=user_id,
            allocation_id=allocation_id,
            symbol=symbol,
            side=side,
            current_price=current_price,
            remaining_size=remaining_size,
            metadata=metadata,
        )

        position["size"] = remaining_size
        position["position_metadata"] = metadata
        leverage = max(1.0, float(position.get("leverage") or 1.0))
        updated = await self.db_service.update_agent_position(
            position_id=position["id"],
            current_price=current_price,
            unrealized_pnl=float(position.get("unrealized_pnl") or 0.0) * remaining_ratio,
            unrealized_pnl_percent=float(position.get("unrealized_pnl_percent") or 0.0),
            position_value=remaining_size * current_price,
            margin_ratio=float(position.get("margin_ratio") or 0.0),
            position_metadata=metadata,
            size=remaining_size,
            margin_used=remaining_size * current_price / leverage,
        )
        return bool(updated)

    async def _maybe_scale_in_winner(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
        acquire_cycle_lock: bool = True,
    ) -> bool:
        """Add once to a confirmed winner without increasing initial thesis risk."""
        metadata = dict(position.get("position_metadata") or {})
        db = self.db_service.db
        try:
            allocation_rows = (
                db.table("agent_allocations")
                .select("remaining_amount,status")
                .eq("allocation_id", allocation_id)
                .limit(1)
                .execute()
            ).data or []
            if not allocation_rows or str(allocation_rows[0].get("status") or "").lower() != "active":
                return False

            available_margin = float(allocation_rows[0].get("remaining_amount") or 0.0)
            position["position_metadata"] = metadata
            requested_stop = self._positive_float((decision or {}).get("stop_price"))
            plan = self._winner_scale_in_plan(
                position,
                available_margin,
                protected_stop_override=requested_stop,
            )
            if not plan:
                return False

            confirmed = await self._target_confirmation_ready(
                symbol=str(position.get("symbol") or ""),
                side=str(position.get("side") or "").lower(),
                current_price=float(position.get("current_price") or 0),
                target=plan["trigger_price"],
                stage="scale_in",
                metadata=metadata,
            )
            if not confirmed:
                db.table("agent_positions").update({
                    "position_metadata": metadata,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", position["id"]).eq("is_active", True).execute()
                return False

            # Share the same lock as new platform-signal entries and manual
            # closes so collateral and the delegated exchange client cannot be
            # changed underneath this one-time management action.
            from kata.services.agent_allocation_service import get_agent_allocation_service

            allocation_service = get_agent_allocation_service()
            class _ExistingCycleLock:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, exc_type, exc, traceback):
                    return False

            cycle_lock = (
                allocation_service.yuki_signal_cycle_lock
                if acquire_cycle_lock
                else _ExistingCycleLock()
            )
            async with cycle_lock:
                refreshed_allocations = (
                    db.table("agent_allocations")
                    .select("remaining_amount,status")
                    .eq("allocation_id", allocation_id)
                    .limit(1)
                    .execute()
                ).data or []
                if (
                    not refreshed_allocations
                    or str(refreshed_allocations[0].get("status") or "").lower() != "active"
                ):
                    return False
                available_margin = float(
                    refreshed_allocations[0].get("remaining_amount") or 0.0
                )
                plan = self._winner_scale_in_plan(
                    position,
                    available_margin,
                    protected_stop_override=requested_stop,
                )
                if not plan:
                    return False

                symbol = str(position.get("symbol") or "")
                side = str(position.get("side") or "").lower()
                current_size = float(position.get("size") or 0.0)
                current_price = float(position.get("current_price") or 0.0)
                current_entry = float(position.get("entry_price") or current_price)
                leverage = float(plan["leverage"])
                protected_stop = float(plan["protected_stop"])
                active_stop = self._positive_float(metadata.get("active_stop"))
                stop_is_ready = bool(
                    active_stop
                    and (
                        (side == "long" and active_stop >= protected_stop)
                        or (side == "short" and active_stop <= protected_stop)
                    )
                )
                if not stop_is_ready:
                    stop_is_ready = await self._ratchet_exchange_stop(
                        user_id=user_id,
                        allocation_id=allocation_id,
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        remaining_size=current_size,
                        desired_stop=protected_stop,
                        stage="winner_scale_in_guard",
                        metadata=metadata,
                    )
                if not stop_is_ready:
                    logger.info(
                        "Yuki did not add to %s: a profit-protecting stop could not be confirmed",
                        symbol,
                    )
                    return False

                trade_rows = (
                    db.table("agent_trades")
                    .select("id,trade_amount,position_size,fees,trade_metadata")
                    .eq("id", position.get("trade_id"))
                    .in_("status", ["filled", "partially_filled"])
                    .limit(1)
                    .execute()
                ).data or []
                if not trade_rows:
                    logger.error(
                        "Yuki did not add to %s: its active aggregate trade row is missing",
                        symbol,
                    )
                    return False
                trade = trade_rows[0]

                size_decimals = await self.hyperliquid_service.get_perp_sz_decimals(symbol)
                requested_size = self.hyperliquid_service._round_size_for_hl(
                    float(plan["size"]), size_decimals
                )
                requested_margin = requested_size * float(plan["risk_price"]) / leverage
                if (
                    requested_size <= 0
                    or requested_margin < float(settings.YUKI_MIN_TRADE_USDC)
                    or requested_margin > available_margin + 1e-9
                ):
                    return False

                now = datetime.now(timezone.utc)
                metadata["scale_in_in_progress_at"] = now.isoformat()
                metadata["scale_in_plan"] = {
                    **plan,
                    "size": requested_size,
                    "margin": requested_margin,
                }
                db.table("agent_positions").update({
                    "position_metadata": metadata,
                    "updated_at": now.isoformat(),
                }).eq("id", position["id"]).eq("is_active", True).execute()

                wallet_address = await self._resolve_allocation_wallet(allocation_id)
                if not await self._ensure_trading_client(user_id, wallet_address):
                    metadata.pop("scale_in_in_progress_at", None)
                    metadata["scale_in_last_error"] = "Delegated trading client unavailable"
                    db.table("agent_positions").update({
                        "position_metadata": metadata,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", position["id"]).execute()
                    return False

                entry_side = OrderSide.BUY if side == "long" else OrderSide.SELL
                add_result = await self.hyperliquid_service.place_order_live(
                    symbol=symbol,
                    side=entry_side,
                    size=requested_size,
                    order_type=OrderType.MARKET,
                    reduce_only=False,
                    leverage=leverage,
                )
                if not add_result.success:
                    metadata.pop("scale_in_in_progress_at", None)
                    metadata["scale_in_last_error"] = add_result.error or "Scale-in order rejected"
                    metadata["scale_in_last_failed_at"] = datetime.now(timezone.utc).isoformat()
                    db.table("agent_positions").update({
                        "position_metadata": metadata,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", position["id"]).execute()
                    logger.warning("Yuki winner add rejected for %s: %s", symbol, add_result.error)
                    return False

                added_size = float(add_result.filled_size or add_result.size or requested_size)
                added_price = float(add_result.average_price or add_result.price or current_price)
                added_margin = added_size * added_price / leverage
                actual_added_risk = added_size * abs(added_price - protected_stop)
                allowed_added_risk = float(plan["initial_dollar_risk"]) * max(
                    0.0,
                    min(
                        1.0,
                        float(settings.YUKI_WINNER_SCALE_IN_MAX_INITIAL_RISK_FRACTION),
                    ),
                )
                if actual_added_risk > allowed_added_risk + 1e-9:
                    close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                    unwind = await self.hyperliquid_service.place_order_live(
                        symbol=symbol,
                        side=close_side,
                        size=added_size,
                        order_type=OrderType.MARKET,
                        reduce_only=True,
                    )
                    metadata.pop("scale_in_in_progress_at", None)
                    metadata["scale_in_last_error"] = "Actual fill exceeded the add-risk cap"
                    metadata["scale_in_last_failed_at"] = datetime.now(timezone.utc).isoformat()
                    db.table("agent_positions").update({
                        "position_metadata": metadata,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", position["id"]).execute()
                    logger.error(
                        "Yuki winner add risk cap exceeded for %s; unwind=%s",
                        symbol,
                        unwind.success,
                    )
                    return False
                total_size = current_size + added_size
                average_entry = (
                    current_entry * current_size + added_price * added_size
                ) / total_size

                # Resize the exchange stop to the combined position. If that is
                # rejected, protect the added leg with a separate stop before
                # continuing; the original tightened stop always remains live.
                full_stop_live = await self._ratchet_exchange_stop(
                    user_id=user_id,
                    allocation_id=allocation_id,
                    symbol=symbol,
                    side=side,
                    current_price=added_price,
                    remaining_size=total_size,
                    desired_stop=protected_stop,
                    stage="winner_scale_in_resize",
                    metadata=metadata,
                    force_resize=True,
                )
                supplemental_stop = None
                if not full_stop_live:
                    close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
                    supplemental = await allocation_service._place_protective_trigger_with_retry(
                        symbol=symbol,
                        close_side=close_side,
                        size=added_size,
                        trigger_price=protected_stop,
                        trigger_tpsl="sl",
                    )
                    if supplemental.success:
                        supplemental_stop = {
                            "success": True,
                            "order_id": supplemental.order_id,
                            "price": protected_stop,
                            "size": added_size,
                            "stage": "winner_scale_in_supplemental",
                        }
                        protective = dict(metadata.get("protective_orders") or {})
                        extra_stops = list(protective.get("supplemental_stops") or [])
                        extra_stops.append(supplemental_stop)
                        protective["supplemental_stops"] = extra_stops
                        metadata["protective_orders"] = protective
                    else:
                        unwind = await self.hyperliquid_service.place_order_live(
                            symbol=symbol,
                            side=close_side,
                            size=added_size,
                            order_type=OrderType.MARKET,
                            reduce_only=True,
                        )
                        metadata.pop("scale_in_in_progress_at", None)
                        metadata["scale_in_last_error"] = (
                            "Combined and supplemental stops failed; added size was unwound"
                            if unwind.success
                            else "CRITICAL: added size could not be protected or unwound"
                        )
                        metadata["scale_in_last_failed_at"] = datetime.now(timezone.utc).isoformat()
                        db.table("agent_positions").update({
                            "position_metadata": metadata,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", position["id"]).execute()
                        logger.error("Yuki could not protect winner add for %s; unwind=%s", symbol, unwind.success)
                        return False

                completed_at = datetime.now(timezone.utc).isoformat()
                scale_in_record = {
                    "order_id": add_result.order_id,
                    "size": added_size,
                    "price": added_price,
                    "margin": added_margin,
                    "leverage": leverage,
                    "trigger_price": plan["trigger_price"],
                    "protected_stop": protected_stop,
                    "initial_dollar_risk": plan["initial_dollar_risk"],
                    "added_risk_at_stop": actual_added_risk,
                    "supplemental_stop": supplemental_stop,
                    "executed_at": completed_at,
                }
                protective = dict(metadata.get("protective_orders") or {})
                protective["runner_enabled"] = True
                protective["runner_size"] = (
                    float(protective.get("runner_size") or metadata.get("runner_size") or 0.0)
                    + added_size
                )
                metadata["protective_orders"] = protective
                metadata["runner_enabled"] = True
                metadata["runner_size"] = protective["runner_size"]
                metadata["original_size"] = total_size
                metadata["scale_ins"] = [*list(metadata.get("scale_ins") or []), scale_in_record][-3:]
                metadata["scale_in_completed_at"] = completed_at
                metadata["scale_in_order_id"] = add_result.order_id
                metadata.pop("scale_in_in_progress_at", None)
                metadata.pop("scale_in_last_error", None)

                trade_metadata = dict(trade.get("trade_metadata") or {})
                trade_metadata["execution_policy"] = "signal_led_adaptive"
                trade_metadata["protective_orders"] = protective
                trade_metadata["scale_ins"] = [
                    *list(trade_metadata.get("scale_ins") or []),
                    scale_in_record,
                ][-3:]
                trade_update = (
                    db.table("agent_trades")
                    .update({
                        "entry_price": average_entry,
                        "position_size": total_size,
                        "trade_amount": float(trade.get("trade_amount") or 0.0) + added_margin,
                        "fees": float(trade.get("fees") or 0.0) + float(add_result.fee or 0.0),
                        "trade_metadata": trade_metadata,
                    })
                    .eq("id", trade["id"])
                    .in_("status", ["filled", "partially_filled"])
                    .execute()
                )
                if not trade_update.data:
                    # The monitor lock makes a concurrent lifecycle transition
                    # unlikely, but retry the durable write without the status
                    # predicate before declaring the protected exchange add lost.
                    trade_update = (
                        db.table("agent_trades")
                        .update({
                            "entry_price": average_entry,
                            "position_size": total_size,
                            "trade_amount": float(trade.get("trade_amount") or 0.0) + added_margin,
                            "fees": float(trade.get("fees") or 0.0) + float(add_result.fee or 0.0),
                            "trade_metadata": trade_metadata,
                        })
                        .eq("id", trade["id"])
                        .execute()
                    )
                    if not trade_update.data:
                        logger.critical(
                            "Protected Yuki winner add for %s could not be attached to its trade ledger",
                            symbol,
                        )
                        return False

                db.table("agent_positions").update({
                    "size": total_size,
                    "entry_price": average_entry,
                    "position_value": total_size * current_price,
                    "margin_used": total_size * current_price / leverage,
                    "position_metadata": metadata,
                    "updated_at": completed_at,
                }).eq("id", position["id"]).eq("is_active", True).execute()

                try:
                    db.table("agent_trade_executions").insert({
                        "execution_id": str(uuid.uuid4()),
                        "allocation_id": allocation_id,
                        "user_id": user_id,
                        "agent_type": "yuki",
                        "symbol": symbol,
                        "side": entry_side.value,
                        "size": added_size,
                        "price": added_price,
                        "amount_used": added_margin,
                        "leverage": leverage,
                        "executed_at": completed_at,
                        "hyperliquid_order_id": add_result.order_id,
                        "status": "filled",
                    }).execute()
                except Exception as audit_error:
                    logger.warning("Could not store Yuki scale-in execution audit: %s", audit_error)

                await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
                logger.info(
                    "Yuki added once to confirmed winner %s: +%s @ %s; stop %s; total size %s",
                    symbol,
                    added_size,
                    added_price,
                    protected_stop,
                    total_size,
                )
                return True
        except Exception as exc:
            metadata.pop("scale_in_in_progress_at", None)
            metadata["scale_in_last_error"] = str(exc)
            metadata["scale_in_last_failed_at"] = datetime.now(timezone.utc).isoformat()
            try:
                db.table("agent_positions").update({
                    "position_metadata": metadata,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", position.get("id")).execute()
            except Exception:
                pass
            logger.error("Yuki winner scale-in failed for %s: %s", position.get("symbol"), exc)
            return False

    async def _maybe_review_scale_in_opportunity(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics,
    ) -> bool:
        """Let ReAct decide whether a newly confirmed winner should add or wait."""
        metadata = dict(position.get("position_metadata") or {})
        db = self.db_service.db
        try:
            allocation_rows = (
                db.table("agent_allocations")
                .select("remaining_amount,status")
                .eq("allocation_id", allocation_id)
                .limit(1)
                .execute()
            ).data or []
            if not allocation_rows or str(allocation_rows[0].get("status") or "").lower() != "active":
                return False

            available_margin = float(allocation_rows[0].get("remaining_amount") or 0.0)
            position["position_metadata"] = metadata
            had_signature = "scale_in_candidate_signature" in metadata
            plan = self._winner_scale_in_plan(position, available_margin)
            if not plan:
                metadata.pop("scale_in_candidate_signature", None)
                if had_signature:
                    self._persist_position_metadata(position.get("id"), metadata)
                return False

            confirmed = await self._target_confirmation_ready(
                symbol=str(position.get("symbol") or ""),
                side=str(position.get("side") or "").lower(),
                current_price=float(position.get("current_price") or 0),
                target=plan["trigger_price"],
                stage="scale_in",
                metadata=metadata,
            )
            if not confirmed:
                db.table("agent_positions").update({
                    "position_metadata": metadata,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", position["id"]).eq("is_active", True).execute()
                return False

            signature = (
                f"{plan['trigger_price']:.5f}:{plan['policy_protected_stop']:.5f}:"
                f"{plan['size']:.4f}:{plan['margin']:.2f}"
            )
            metadata["scale_in_candidate_signature"] = signature
            position["position_metadata"] = metadata
            if not self._event_needs_review(position, risk_metrics, "winner_scale_in_opportunity"):
                self._persist_position_metadata(position.get("id"), metadata)
                return False
            self._persist_position_metadata(position.get("id"), metadata)
            await self._review_position_event(
                user_id=user_id,
                allocation_id=allocation_id,
                position=position,
                risk_metrics=risk_metrics,
                event="winner_scale_in_opportunity",
                trigger_context={
                    "policy_proposal": {
                        "trigger_price": plan["trigger_price"],
                        "protected_stop": plan["policy_protected_stop"],
                        "current_stop": self._positive_float(metadata.get("active_stop"))
                        or self._positive_float(metadata.get("policy_stop"))
                        or self._positive_float(metadata.get("stop_loss")),
                        "size": plan["size"],
                        "margin": plan["margin"],
                        "added_risk_at_stop": plan["added_risk_at_stop"],
                        "initial_dollar_risk": plan["initial_dollar_risk"],
                        "leverage": plan["leverage"],
                    },
                    "confirmation": "closed_candles",
                },
            )
            return True
        except Exception as exc:
            logger.error(
                "Yuki scale-in opportunity review failed for %s: %s",
                position.get("symbol"),
                exc,
            )
            return False
    
    async def _calculate_position_risk(self, position: Dict[str, Any]) -> PositionRiskMetrics:
        """Calculate risk metrics for a position."""
        try:
            # Extract position data
            symbol = position["symbol"]
            side = position["side"]
            entry_price = float(position["entry_price"])
            current_price = float(position["current_price"])
            leverage = float(position["leverage"])
            unrealized_pnl_percent = float(position["unrealized_pnl_percent"])
            margin_ratio = float(position["margin_ratio"])
            created_at = datetime.fromisoformat(str(position["created_at"]).replace("Z", "+00:00"))
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            time_in_position_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
            metadata = position.get("position_metadata") or {}
            stop_loss = (
                self._positive_float(metadata.get("policy_stop"))
                or self._positive_float(metadata.get("active_stop"))
                or self._positive_float(metadata.get("stop_loss"))
            )
            target_1 = self._positive_float(metadata.get("target_1"))
            target_2 = self._positive_float(metadata.get("target_2"))
            runner_enabled = bool(metadata.get("runner_enabled"))
            hold_limit_hours = self._position_hold_limit_hours(metadata)
            
            # Liquidation risk = fraction of posted margin eroded by unrealized
            # losses. unrealized_pnl_percent is return-on-margin and already
            # leverage-aware (a 1% adverse move at 10x is -10% on margin), so a
            # position that has lost most of its margin is genuinely near
            # liquidation, while a healthy/profitable position sits near zero.
            #
            # The previous formula was `1 - margin_ratio`, where margin_ratio was
            # ~1/leverage — so every healthy 10x position read as 0.90 "critical"
            # and got force-closed (the XRP incident), and every 2x position read
            # as 0.50 "medium", which (via the old elif chain) suppressed every
            # signal SL/TP check below.
            loss_fraction = max(0.0, -unrealized_pnl_percent / 100.0)
            liquidation_risk = min(loss_fraction, 1.0)

            # Determine risk level
            risk_level = RiskLevel.LOW
            should_close = False
            reason = ""

            if str(position.get("_source_signal_status") or "").lower() == "invalidated":
                risk_level = RiskLevel.MEDIUM
                reason = "Source signal invalidated; Yuki ReAct review required"

            # 1) Hard liquidation protection — only force-close when margin is
            #    actually eroded, never merely because leverage is high.
            if not should_close and liquidation_risk > 0.8:
                risk_level = RiskLevel.CRITICAL
                should_close = True
                reason = f"Critical liquidation risk: {liquidation_risk:.2f} (margin mostly eroded)"
            elif not should_close and liquidation_risk > 0.6:
                risk_level = RiskLevel.HIGH
                should_close = True
                reason = f"High liquidation risk: {liquidation_risk:.2f}"

            # 2) Signal SL/TP and fallbacks. Evaluated independently of the
            #    liquidation branch above so an informational "medium" reading can
            #    never short-circuit these checks (the previous 2x bug).
            if not should_close:
                if stop_loss and self._stop_loss_hit(side, current_price, stop_loss):
                    risk_level = RiskLevel.HIGH
                    should_close = True
                    reason = f"Signal stop-loss hit at {current_price:.8g} (stop {stop_loss:.8g})"

                elif not runner_enabled and self._take_profit_hit(side, current_price, target_2 or target_1):
                    risk_level = RiskLevel.LOW
                    should_close = True
                    take_profit = target_2 or target_1
                    reason = f"Signal take-profit hit at {current_price:.8g} (target {take_profit:.8g})"

                # Fallback stop-loss when no signal stop is available.
                elif not stop_loss and unrealized_pnl_percent < -8.0:
                    risk_level = RiskLevel.HIGH
                    should_close = True
                    reason = f"Stop-loss triggered: {unrealized_pnl_percent:.2f}% loss"

                # Fallback take-profit when no signal target is available.
                elif not (target_1 or target_2) and unrealized_pnl_percent > 40.0:
                    risk_level = RiskLevel.LOW
                    reason = (
                        f"Unplanned profit reached {unrealized_pnl_percent:.2f}%; "
                        "Yuki ReAct review required"
                    )

                # The declared horizon is a ReAct decision checkpoint, not a
                # time-based exit. The monitor loop sees this informational state
                # and asks Yuki to reassess the full trade with current evidence.
                elif time_in_position_hours >= hold_limit_hours:
                    risk_level = RiskLevel.MEDIUM
                    reason = (
                        f"Time horizon elapsed after {time_in_position_hours:.1f}h "
                        f"(review threshold {hold_limit_hours:.1f}h); Yuki ReAct review required"
                    )

                # Informational elevated-liquidation reading (no forced close).
                elif liquidation_risk > 0.4:
                    risk_level = RiskLevel.MEDIUM
                    reason = f"Elevated liquidation risk: {liquidation_risk:.2f}"
            
            return PositionRiskMetrics(
                liquidation_risk=liquidation_risk,
                margin_ratio=margin_ratio,
                unrealized_pnl_percent=unrealized_pnl_percent,
                time_in_position_hours=time_in_position_hours,
                risk_level=risk_level,
                should_close=should_close,
                reason=reason
            )
            
        except Exception as e:
            # Fail safe: a monitoring/software error must never liquidate a healthy
            # position. Exchange-side SL/TP trigger orders placed at entry remain
            # the hard protection; log loudly and retry next cycle.
            logger.error(f"Error calculating position risk (position kept open): {e}")
            return PositionRiskMetrics(
                liquidation_risk=0.0,
                margin_ratio=1.0,
                unrealized_pnl_percent=0.0,
                time_in_position_hours=0.0,
                risk_level=RiskLevel.HIGH,
                should_close=False,
                reason=f"Risk calculation error (monitoring only): {str(e)}"
            )

    @staticmethod
    def _positive_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _horizon_upper_hours(time_horizon: Any) -> Optional[float]:
        """Parse values such as 4h-12h, 1-3 days, or 1-2 weeks."""
        text = str(time_horizon or "").lower().strip().replace("—", "-").replace("–", "-")
        if not text:
            return None
        unit_pattern = r"h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks"
        range_match = re.search(
            rf"\d+(?:\.\d+)?\s*(?:{unit_pattern})?\s*-\s*"
            rf"(\d+(?:\.\d+)?)\s*({unit_pattern})",
            text,
        )
        single_match = re.search(rf"(\d+(?:\.\d+)?)\s*({unit_pattern})", text)
        match = range_match or single_match
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit.startswith("d"):
                return value * 24.0
            if unit.startswith("w"):
                return value * 168.0
            return value
        if "scalp" in text:
            return 4.0
        if "intraday" in text or "short" in text:
            return 24.0
        if "swing" in text or "medium" in text:
            return 72.0
        if "long" in text:
            return 168.0
        return None

    @classmethod
    def _position_hold_limit_hours(cls, metadata: Dict[str, Any]) -> float:
        parsed = cls._horizon_upper_hours(metadata.get("time_horizon"))
        fallback = max(1.0, float(settings.YUKI_DEFAULT_HOLD_HOURS))
        maximum = max(1.0, float(settings.YUKI_MAX_HOLD_HOURS))
        return min(parsed if parsed and parsed > 0 else fallback, maximum)

    @staticmethod
    def _stop_loss_hit(side: str, current_price: float, stop_loss: float) -> bool:
        if side == "long":
            return current_price <= stop_loss
        return current_price >= stop_loss

    @staticmethod
    def _take_profit_hit(side: str, current_price: float, take_profit: Optional[float]) -> bool:
        if not take_profit:
            return False
        if side == "long":
            return current_price >= take_profit
        return current_price <= take_profit
    
    async def _execute_risk_management_action(
        self,
        user_id: str,
        allocation_id: str,
        position: Dict[str, Any],
        risk_metrics: PositionRiskMetrics
    ) -> bool:
        """Execute risk management action (close position)."""
        try:
            symbol = position["symbol"]
            side = position["side"]
            size = position["size"]

            # Re-authenticate for this allocation's wallet immediately before the
            # close. The shared trading client is a singleton that a concurrent
            # allocation's monitor may have re-pointed at a different wallet since
            # the start of this cycle — closing against the wrong client fails or,
            # worse, could act on the wrong account.
            wallet_address = await self._resolve_allocation_wallet(allocation_id)
            if not await self._ensure_trading_client(user_id, wallet_address):
                logger.error(
                    f"Skipping risk close for {symbol}: trading client not authenticated "
                    f"for allocation {allocation_id}. Will retry next cycle."
                )
                return False

            # Determine close side
            close_side = OrderSide.SELL if side == "long" else OrderSide.BUY

            # Execute close order
            order_result = await self.hyperliquid_service.place_order_live(
                symbol=symbol,
                side=close_side,
                size=size,
                order_type=OrderType.MARKET,
                reduce_only=True
            )
            
            if order_result.success:
                # Record the actual fill, not the stale cached mark - closed-trade
                # PnL feeds the dashboard's cumulative P&L.
                exit_price = float(
                    order_result.average_price
                    or order_result.price
                    or position["current_price"]
                    or 0
                )
                entry_price = float(position.get("entry_price") or 0)
                size_float = float(size or 0)
                if entry_price > 0 and exit_price > 0 and size_float > 0:
                    realized_pnl = (
                        (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
                    ) * size_float
                    realized_pnl_percent = (
                        (exit_price / entry_price - 1.0) if side == "long" else (1.0 - exit_price / entry_price)
                    ) * 100.0 * float(position.get("leverage") or 1)
                else:
                    realized_pnl = float(position.get("unrealized_pnl") or 0)
                    realized_pnl_percent = float(position.get("unrealized_pnl_percent") or 0)

                fill_summary = None
                for delay in (0.0, 0.5, 1.0):
                    if delay:
                        await asyncio.sleep(delay)
                    fill_summary = await self._fetch_closed_trade_fill_summary(position)
                    if fill_summary:
                        exit_price = float(fill_summary["average_exit_price"])
                        realized_pnl = float(fill_summary["net_realized_pnl"])
                        break

                # Update position as closed
                await self.db_service.update_agent_position(
                    position_id=position["id"],
                    current_price=exit_price,
                    unrealized_pnl=realized_pnl,
                    unrealized_pnl_percent=realized_pnl_percent,
                    position_value=exit_price * size_float,
                    margin_ratio=position["margin_ratio"]
                )

                # Mark position as inactive
                await self._close_position(
                    position["id"],
                    exit_price,
                    realized_pnl,
                    fees=float(fill_summary["fees"]) if fill_summary else None,
                    trade_metadata_updates=(
                        {"fill_reconciliation": fill_summary} if fill_summary else None
                    ),
                )
                await self._cancel_remaining_protective_orders(position)

                logger.info(f"Risk management: Closed {symbol} position due to {risk_metrics.reason}")
                return True
            else:
                logger.error(f"Failed to close position {symbol}: {order_result.error}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing risk management action: {e}")
            return False

    async def _cancel_remaining_protective_orders(self, position: Dict[str, Any]):
        """Remove any fixed triggers left after a successful full position close."""
        metadata = position.get("position_metadata") or {}
        protective = metadata.get("protective_orders") or {}
        order_ids = []
        stop_order = protective.get("stop_loss") or {}
        if stop_order.get("order_id"):
            order_ids.append(stop_order["order_id"])
        for take_profit in protective.get("take_profit") or []:
            if take_profit.get("order_id"):
                order_ids.append(take_profit["order_id"])
        for supplemental in protective.get("supplemental_stops") or []:
            if supplemental.get("order_id"):
                order_ids.append(supplemental["order_id"])
        for order_id in dict.fromkeys(str(value) for value in order_ids):
            await self.hyperliquid_service.cancel_order(position["symbol"], order_id)

    def _platform_signal_exit_reason_from_close(
        self,
        metadata: Dict[str, Any],
        side: str,
        exit_price: float,
        closed_at: Optional[Any],
    ) -> str:
        """Classify a Yuki close into the platform signal lifecycle."""
        target_1 = self._positive_float(metadata.get("target_1"))
        target_2 = self._positive_float(metadata.get("target_2"))
        stop_loss = (
            self._positive_float(metadata.get("stop_loss"))
            or self._positive_float(metadata.get("active_stop"))
            or self._positive_float(metadata.get("policy_stop"))
            or self._positive_float(metadata.get("initial_stop_loss"))
        )
        runner_enabled = bool(metadata.get("runner_enabled"))
        target_1_hit = bool(metadata.get("target_1_hit"))
        target_2_hit = bool(metadata.get("target_2_hit") or metadata.get("target_2_confirmed"))
        closed_at_dt = self._parse_datetime(closed_at)
        expires_at = self._parse_datetime(metadata.get("signal_expires_at"))

        if target_2 and (target_2_hit or self._take_profit_hit(side, exit_price, target_2)):
            return "TARGET_2"
        if target_1 and (target_1_hit or self._take_profit_hit(side, exit_price, target_1)) and (not runner_enabled or not target_2):
            return "TARGET_1"
        if stop_loss and self._stop_loss_hit(side, exit_price, stop_loss):
            return "STOP_LOSS"
        if expires_at and ((closed_at_dt and closed_at_dt >= expires_at) or datetime.now(timezone.utc) >= expires_at):
            return "EXPIRED"
        return "MANUAL"

    async def _sync_linked_platform_signal_on_close(
        self,
        closed: Dict[str, Any],
        exit_price: float,
    ) -> None:
        """Reconcile the learning ledger from Yuki's actual executed close."""
        position = dict(closed.get("position") or {})
        trade = dict(closed.get("trade") or {})
        position_metadata = dict(position.get("position_metadata") or {})
        trade_metadata = dict(trade.get("trade_metadata") or {})
        metadata = dict(position_metadata)
        for key, value in trade_metadata.items():
            metadata.setdefault(key, value)

        signal_id = str(metadata.get("signal_id") or "").strip()
        if not signal_id:
            return

        signal_rows = (
            self.db_service.db.table("platform_signals")
            .select("status")
            .eq("signal_id", signal_id)
            .limit(1)
            .execute()
        ).data or []
        signal_status = str((signal_rows[0] if signal_rows else {}).get("status") or "").lower()
        if signal_status in {"hit_target_1", "hit_target_2", "hit_stop_loss", "expired", "cancelled", "invalidated"}:
            return

        side = str(position.get("side") or "").lower().strip()
        if side not in {"long", "short"}:
            direction = str(metadata.get("signal_direction") or "").upper().strip()
            if direction in {"LONG", "BUY"}:
                side = "long"
            elif direction in {"SHORT", "SELL"}:
                side = "short"
        if side not in {"long", "short"}:
            return

        from kata.services.platform_signal_service import get_platform_signal_service

        platform_signal_service = get_platform_signal_service()
        if bool(metadata.get("target_1_hit")):
            target_1_price = self._positive_float(metadata.get("target_1"))
            if target_1_price:
                await platform_signal_service.record_signal_target_1_milestone(
                    signal_id,
                    target_1_price,
                )

        signal_entry_price = (
            self._positive_float(metadata.get("signal_entry_price"))
            or self._positive_float(metadata.get("initial_entry_price"))
            or self._positive_float(metadata.get("entry_price"))
            or self._positive_float(position.get("entry_price"))
        )
        if signal_entry_price and exit_price > 0:
            raw_pnl = (
                ((exit_price - signal_entry_price) / signal_entry_price) * 100.0
                if side == "long"
                else ((signal_entry_price - exit_price) / signal_entry_price) * 100.0
            )
            self.db_service.db.rpc(
                "rpc_update_signal_performance",
                {
                    "p_signal_id": signal_id,
                    "p_pnl": float(raw_pnl),
                },
            ).execute()

        exit_reason = self._platform_signal_exit_reason_from_close(
            metadata=metadata,
            side=side,
            exit_price=float(exit_price),
            closed_at=trade.get("closed_at"),
        )
        await platform_signal_service.update_signal_exit(
            signal_id,
            float(exit_price),
            exit_reason,
        )
        logger.info(
            "Reconciled platform signal %s from Yuki close: %s @ %.8g",
            signal_id,
            exit_reason,
            exit_price,
        )
    
    async def _close_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        fees: Optional[float] = None,
        trade_metadata_updates: Optional[Dict[str, Any]] = None,
    ):
        """Close a position and update database records."""
        try:
            closed = await self.db_service.close_agent_position(
                position_id,
                exit_price,
                realized_pnl,
                fees=fees,
                trade_metadata_updates=trade_metadata_updates,
            )
            if closed:
                logger.info(f"Position {position_id} closed at ${exit_price} with PnL ${realized_pnl:.2f}")
                try:
                    await self._sync_linked_platform_signal_on_close(
                        closed,
                        exit_price,
                    )
                except Exception as sync_err:
                    logger.warning("Could not reconcile platform signal on Yuki close: %s", sync_err)
                try:
                    from kata.services.agent_allocation_service import get_agent_allocation_service
                    alloc_id = closed.get("allocation_id")
                    if alloc_id:
                        asyncio.create_task(
                            get_agent_allocation_service().trigger_yuki_signal_check_for_allocation(alloc_id)
                        )
                except Exception as trigger_err:
                    logger.warning("Could not trigger post-close Yuki signal check: %s", trigger_err)
            else:
                logger.warning(f"Position {position_id} close order filled, but database close update did not persist")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def get_allocation_performance(self, allocation_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for an allocation.
        
        Args:
            allocation_id: Allocation ID
            
        Returns:
            Performance metrics
        """
        try:
            # Get all trades for allocation
            trades = await self.db_service.get_agent_trades(allocation_id)
            
            # Get current positions
            positions = await self.db_service.get_agent_positions(allocation_id)
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get("realized_pnl", 0) > 0])
            losing_trades = len([t for t in trades if t.get("realized_pnl", 0) < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            realized_pnl = sum(t.get("realized_pnl", 0) for t in trades)
            unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
            total_pnl = realized_pnl + unrealized_pnl
            
            total_volume = sum(t.get("trade_amount", 0) for t in trades)
            average_trade_size = total_volume / total_trades if total_trades > 0 else 0
            
            return {
                "allocation_id": allocation_id,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "total_volume": total_volume,
                "average_trade_size": average_trade_size,
                "active_positions": len(positions),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting allocation performance: {e}")
            return {
                "allocation_id": allocation_id,
                "error": str(e)
            }
    
    async def stop_all_monitoring(self):
        """Stop all position monitoring tasks."""
        try:
            for allocation_id, task in self.monitoring_tasks.items():
                task.cancel()
                logger.info(f"Stopped monitoring for allocation {allocation_id}")
            
            self.monitoring_tasks.clear()
            self._pending_entry_locks.clear()
            logger.info("All position monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping all monitoring: {e}")


# Factory function
def create_yuki_position_monitor(
    hyperliquid_service: HyperliquidService,
    react_agent_resolver: Optional[Any] = None,
) -> YukiPositionMonitor:
    """Create a Yuki position monitor instance."""
    return YukiPositionMonitor(
        hyperliquid_service,
        react_agent_resolver=react_agent_resolver,
    )
