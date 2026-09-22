"""
Agent Database Service

Handles database operations for agent trades, decisions, and market analysis.
Ensures proper tracking of all agent activities for real-time updates.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from supabase import Client
from kata.config.database import get_service_client
from kata.models.trade import TradeCreate, Trade
from kata.models.trade import AgentDecisionCreate, AgentDecision

logger = logging.getLogger(__name__)


class AgentDatabaseService:
    """Service for managing agent data in the database."""
    
    def __init__(self):
        self.db: Client = get_service_client()
        logger.info("AgentDatabaseService initialized")

    @staticmethod
    def _allocation_ledger_totals(
        allocated_amount: float,
        trade_rows: List[Dict[str, Any]],
        principal_amount: Optional[float] = None,
    ) -> Dict[str, float]:
        """Derive compounded equity and committed margin from the trade ledger.

        ``allocated_amount`` was historically the fixed deposit.  Capital
        accounting v2 persists that deposit separately as ``principal_amount``
        and makes ``allocated_amount`` the live compounded equity shown to the
        user.  Keeping this helper backward-compatible lets legacy allocations
        bootstrap their principal on the first reconciliation.
        """
        executed_statuses = {"filled", "partially_filled", "closed"}
        realized_pnl = sum(
            float(row.get("realized_pnl") or 0)
            for row in trade_rows
            if str(row.get("status") or "").lower() in executed_statuses
        )
        committed_margin = sum(
            max(0.0, float(row.get("trade_amount") or 0))
            for row in trade_rows
            if str(row.get("status") or "").lower()
            in {"pending", "filled", "partially_filled"}
        )
        executed_trades = sum(
            1
            for row in trade_rows
            if str(row.get("status") or "").lower() in executed_statuses
        )
        principal = float(allocated_amount if principal_amount is None else principal_amount)
        compounded_equity = max(0.0, principal + realized_pnl)
        return {
            "principal_amount": principal,
            "allocated_amount": compounded_equity,
            "realized_pnl": realized_pnl,
            "committed_margin": committed_margin,
            "remaining_amount": max(0.0, compounded_equity - committed_margin),
            "total_trades": executed_trades,
        }

    async def reconcile_agent_allocation_ledger(
        self,
        allocation_id: Optional[str],
    ) -> Optional[Dict[str, float]]:
        """Self-heal allocation cash from terminal PnL and currently reserved trades."""
        if not allocation_id:
            return None
        try:
            allocation_rows = (
                self.db.table("agent_allocations")
                .select(
                    "allocated_amount,remaining_amount,realized_pnl,unrealized_pnl,"
                    "total_trades,performance_metrics"
                )
                .eq("allocation_id", allocation_id)
                .limit(1)
                .execute()
            ).data or []
            if not allocation_rows:
                return None

            trade_rows: List[Dict[str, Any]] = []
            offset = 0
            page_size = 1000
            while True:
                page = (
                    self.db.table("agent_trades")
                    .select("id,status,trade_amount,realized_pnl")
                    .eq("allocation_id", allocation_id)
                    .range(offset, offset + page_size - 1)
                    .execute()
                ).data or []
                trade_rows.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size

            current = allocation_rows[0]
            performance_metrics = dict(current.get("performance_metrics") or {})
            capital_metrics = dict(performance_metrics.get("capital") or {})
            accounting_version = int(capital_metrics.get("accounting_version") or 1)
            if accounting_version >= 2:
                principal_amount = float(
                    capital_metrics.get("principal_allocated_amount")
                    if capital_metrics.get("principal_allocated_amount") is not None
                    else current.get("allocated_amount") or 0
                )
            else:
                # Legacy rows stored the original deposit in allocated_amount.
                principal_amount = float(current.get("allocated_amount") or 0)

            totals = self._allocation_ledger_totals(
                float(current.get("allocated_amount") or 0),
                trade_rows,
                principal_amount=principal_amount,
            )

            # Position rows are a cache of the live venue state. Deduplicate by
            # trade id here as a second line of defence so a legacy duplicate
            # row can never double the P&L shown to the user.
            position_rows = (
                self.db.table("agent_positions")
                .select("id,trade_id,symbol,unrealized_pnl,updated_at")
                .eq("allocation_id", allocation_id)
                .eq("is_active", True)
                .order("updated_at", desc=True)
                .execute()
            ).data or []
            unique_positions: Dict[str, Dict[str, Any]] = {}
            for position in position_rows:
                key = str(
                    position.get("trade_id")
                    or position.get("symbol")
                    or position.get("id")
                )
                unique_positions.setdefault(key, position)
            totals["unrealized_pnl"] = sum(
                float(position.get("unrealized_pnl") or 0)
                for position in unique_positions.values()
            )
            capital_metrics.update({
                "accounting_version": 2,
                "principal_allocated_amount": totals["principal_amount"],
                "compounded_equity": totals["allocated_amount"],
                "last_reconciled_at": datetime.now().isoformat(),
            })
            performance_metrics["capital"] = capital_metrics

            allocated_drift = totals["allocated_amount"] - float(current.get("allocated_amount") or 0)
            remaining_drift = totals["remaining_amount"] - float(current.get("remaining_amount") or 0)
            pnl_drift = totals["realized_pnl"] - float(current.get("realized_pnl") or 0)
            unrealized_drift = totals["unrealized_pnl"] - float(current.get("unrealized_pnl") or 0)
            trade_count_changed = totals["total_trades"] != int(current.get("total_trades") or 0)
            metadata_changed = accounting_version < 2
            if (
                abs(allocated_drift) >= 0.005
                or abs(remaining_drift) >= 0.005
                or abs(pnl_drift) >= 0.005
                or abs(unrealized_drift) >= 0.005
                or trade_count_changed
                or metadata_changed
            ):
                self.db.table("agent_allocations").update({
                    "allocated_amount": totals["allocated_amount"],
                    "remaining_amount": totals["remaining_amount"],
                    "realized_pnl": totals["realized_pnl"],
                    "unrealized_pnl": totals["unrealized_pnl"],
                    "total_trades": totals["total_trades"],
                    "performance_metrics": performance_metrics,
                    "updated_at": datetime.now().isoformat(),
                }).eq("allocation_id", allocation_id).execute()
                logger.warning(
                    "Reconciled Yuki allocation %s ledger drift: equity %+.2f, remaining %+.2f, "
                    "realized PnL %+.2f, unrealized PnL %+.2f",
                    allocation_id, allocated_drift, remaining_drift, pnl_drift, unrealized_drift,
                )

            try:
                from kata.services.agent_allocation_service import get_agent_allocation_service

                cached = get_agent_allocation_service().active_allocations.get(allocation_id)
                if cached is not None:
                    cached.allocated_amount = totals["allocated_amount"]
                    cached.remaining_amount = totals["remaining_amount"]
                    cached.realized_pnl = totals["realized_pnl"]
                    cached.unrealized_pnl = totals["unrealized_pnl"]
                    cached.total_trades = int(totals["total_trades"])
                    cached.performance_metrics = performance_metrics
            except Exception:
                pass
            return totals
        except Exception as exc:
            logger.error("Error reconciling allocation %s ledger: %s", allocation_id, exc)
            return None
    
    async def store_agent_trade(
        self, 
        user_id: str, 
        agent_type: str, 
        trade_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Store a trade executed by an agent in the database.
        
        Args:
            user_id: User ID (will be converted to UUID)
            agent_type: Agent type (yuki, sakura, ryu)
            trade_data: Trade execution data
            
        Returns:
            Stored trade data or None if failed
        """
        try:
            # Convert user_id to UUID format for database
            user_uuid = self._ensure_uuid(user_id)

            # Ensure user exists in users table to satisfy foreign key constraint
            wallet = trade_data.get("wallet_address") or user_id
            await self.ensure_user_exists(user_id, wallet)
            
            # Prepare trade record
            trade_record = {
                "id": str(uuid.uuid4()),
                "user_id": user_uuid,
                "agent_type": agent_type,
                "trade_type": trade_data.get("side", "buy").lower(),
                "token_symbol": trade_data.get("symbol", ""),
                "amount": float(trade_data.get("size", 0)),
                "price_usd": float(trade_data.get("price", 0)),
                "value_usd": float(trade_data.get("value", 0)),
                "protocol": "hyperliquid",
                "tx_hash": trade_data.get("order_id"),
                "reasoning": trade_data.get("reasoning", ""),
                "confidence_score": float(trade_data.get("confidence", 0.8)),
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into database
            response = self.db.from_("trades").insert(trade_record).execute()
            
            if response.data:
                stored_trade = response.data[0]
                logger.info(f"Stored agent trade: {agent_type} {trade_record['trade_type']} {trade_record['token_symbol']}")
                return stored_trade
            else:
                logger.error(f"Failed to store agent trade: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing agent trade: {e}")
            return None

    async def store_agent_trade_record(self, trade_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Store a Yuki allocation trade with execution metadata."""
        try:
            now = datetime.now().isoformat()
            trade_record = {
                "id": trade_data.get("id") or str(uuid.uuid4()),
                "user_id": trade_data.get("user_id"),
                "allocation_id": trade_data.get("allocation_id"),
                "agent_type": trade_data.get("agent_type", "yuki"),
                "trade_type": trade_data.get("trade_type"),
                "symbol": trade_data.get("symbol"),
                "side": trade_data.get("side"),
                "entry_price": float(trade_data.get("entry_price") or 0),
                "exit_price": trade_data.get("exit_price"),
                "position_size": float(trade_data.get("position_size") or 0),
                "leverage": float(trade_data.get("leverage") or 1),
                "trade_amount": float(trade_data.get("trade_amount") or 0),
                "realized_pnl": trade_data.get("realized_pnl"),
                "unrealized_pnl": trade_data.get("unrealized_pnl"),
                "fees": float(trade_data.get("fees") or 0),
                "status": trade_data.get("status", "pending"),
                "created_at": trade_data.get("created_at") or now,
                "filled_at": trade_data.get("filled_at"),
                "closed_at": trade_data.get("closed_at"),
                "hyperliquid_order_id": trade_data.get("hyperliquid_order_id"),
                "tx_hash": trade_data.get("tx_hash"),
                "signal_confidence": trade_data.get("signal_confidence"),
                "signal_reasoning": trade_data.get("signal_reasoning"),
                "error_message": trade_data.get("error_message"),
                "trade_metadata": trade_data.get("metadata") or trade_data.get("trade_metadata") or {},
                "max_profit_reached": trade_data.get("max_profit_reached"),
                "max_loss_reached": trade_data.get("max_loss_reached"),
                "time_in_position_minutes": trade_data.get("time_in_position_minutes"),
            }

            response = self.db.table("agent_trades").insert(trade_record).execute()
            if response.data:
                logger.info(f"Stored Yuki trade record {trade_record['id']} for allocation {trade_record['allocation_id']}")
                return response.data[0]

            logger.error(f"Failed to store Yuki trade record: {response}")
            return None
        except Exception as e:
            logger.error(f"Error storing Yuki trade record: {e}")
            return None

    async def get_agent_trades(
        self,
        allocation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get Yuki trade records for an allocation."""
        try:
            response = (
                self.db.table("agent_trades")
                .select("*")
                .eq("allocation_id", allocation_id)
                .order("created_at", desc=True)
                .range(offset, offset + max(limit, 1) - 1)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting agent trades for allocation {allocation_id}: {e}")
            return []

    async def store_agent_position(self, position_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Store an active Yuki position for live monitoring."""
        try:
            now = datetime.now().isoformat()
            position_record = {
                "user_id": position_data.get("user_id"),
                "allocation_id": position_data.get("allocation_id"),
                "trade_id": position_data.get("trade_id"),
                "agent_type": position_data.get("agent_type", "yuki"),
                "symbol": position_data.get("symbol"),
                "side": position_data.get("side"),
                "size": float(position_data.get("size") or 0),
                "entry_price": float(position_data.get("entry_price") or 0),
                "current_price": float(position_data.get("current_price") or 0),
                "leverage": float(position_data.get("leverage") or 1),
                "position_value": float(position_data.get("position_value") or 0),
                "unrealized_pnl": float(position_data.get("unrealized_pnl") or 0),
                "unrealized_pnl_percent": float(position_data.get("unrealized_pnl_percent") or 0),
                "margin_used": float(position_data.get("margin_used") or 0),
                "liquidation_price": position_data.get("liquidation_price"),
                "margin_ratio": position_data.get("margin_ratio"),
                "is_active": position_data.get("is_active", True),
                "created_at": position_data.get("created_at") or now,
                "updated_at": position_data.get("updated_at") or now,
                "hyperliquid_position_id": position_data.get("hyperliquid_position_id"),
                "position_metadata": position_data.get("metadata") or position_data.get("position_metadata") or {},
            }

            trade_id = position_record.get("trade_id")
            if trade_id:
                # A deterministic primary key makes creation idempotent even
                # before the companion unique-trade migration is applied.
                existing_rows = (
                    self.db.table("agent_positions")
                    .select("id")
                    .eq("trade_id", trade_id)
                    .limit(1)
                    .execute()
                ).data or []
                position_record["id"] = str(
                    existing_rows[0].get("id") if existing_rows else trade_id
                )
                response = (
                    self.db.table("agent_positions")
                    .upsert(position_record, on_conflict="id")
                    .execute()
                )
            else:
                position_record["id"] = position_data.get("id") or str(uuid.uuid4())
                response = self.db.table("agent_positions").insert(position_record).execute()
            if response.data:
                logger.info(
                    "Stored Yuki position %s for allocation %s",
                    response.data[0].get("id"),
                    position_record["allocation_id"],
                )
                return response.data[0]

            logger.error(f"Failed to store Yuki position: {response}")
            return None
        except Exception as e:
            logger.error(f"Error storing Yuki position: {e}")
            return None

    async def get_agent_positions(
        self,
        allocation_id: str,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get Yuki positions for an allocation."""
        try:
            query = self.db.table("agent_positions").select("*").eq("allocation_id", allocation_id)
            if active_only:
                query = query.eq("is_active", True)

            response = query.order("created_at", desc=True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting agent positions for allocation {allocation_id}: {e}")
            return []

    async def update_agent_position(
        self,
        position_id: str,
        current_price: float,
        unrealized_pnl: float,
        unrealized_pnl_percent: float,
        position_value: float,
        margin_ratio: float,
        position_metadata: Optional[Dict[str, Any]] = None,
        size: Optional[float] = None,
        margin_used: Optional[float] = None,
    ) -> bool:
        """Update live PnL and risk metrics for an active Yuki position."""
        try:
            update_data = {
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "position_value": position_value,
                "margin_ratio": margin_ratio,
                "updated_at": datetime.now().isoformat()
            }
            if position_metadata is not None:
                update_data["position_metadata"] = position_metadata
            if size is not None:
                update_data["size"] = float(size)
            if margin_used is not None:
                update_data["margin_used"] = float(margin_used)

            response = self.db.table("agent_positions").update(update_data).eq("id", position_id).execute()

            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating agent position {position_id}: {e}")
            return False

    async def close_agent_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        fees: Optional[float] = None,
        trade_metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark a Yuki position inactive and return the closed ledger rows."""
        try:
            now = datetime.now().isoformat()
            position_response = self.db.table("agent_positions").update({
                "is_active": False,
                "current_price": exit_price,
                "unrealized_pnl": 0,
                "unrealized_pnl_percent": 0,
                "updated_at": now
            }).eq("id", position_id).eq("is_active", True).execute()

            position = position_response.data[0] if position_response.data else None
            trade = None
            if position and position.get("trade_id"):
                trade_rows = (
                    self.db.table("agent_trades")
                    .select("trade_metadata")
                    .eq("id", position["trade_id"])
                    .limit(1)
                    .execute()
                ).data or []
                trade_metadata = dict((trade_rows[0] if trade_rows else {}).get("trade_metadata") or {})
                if trade_metadata_updates:
                    trade_metadata.update(trade_metadata_updates)

                trade_update: Dict[str, Any] = {
                    "status": "closed",
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "closed_at": now,
                    "trade_metadata": trade_metadata,
                }
                if fees is not None:
                    trade_update["fees"] = max(0.0, float(fees))

                trade_response = (
                    self.db.table("agent_trades")
                    .update(trade_update)
                    .eq("id", position["trade_id"])
                    .in_("status", ["pending", "filled", "partially_filled"])
                    .execute()
                )
                trade = trade_response.data[0] if trade_response.data else None
                if not trade:
                    trade_rows = (
                        self.db.table("agent_trades")
                        .select("*")
                        .eq("id", position["trade_id"])
                        .limit(1)
                        .execute()
                    ).data or []
                    trade = trade_rows[0] if trade_rows else None

            if position and trade:
                await self._credit_allocation_on_close(
                    allocation_id=position.get("allocation_id"),
                    trade_amount=float((trade or {}).get("trade_amount") or 0),
                    realized_pnl=realized_pnl,
                )

            if not position:
                return None

            return {
                "position": position,
                "trade": trade,
                "allocation_id": position.get("allocation_id"),
            }
        except Exception as e:
            logger.error(f"Error closing agent position {position_id}: {e}")
            return None

    @staticmethod
    def _normalize_agent_symbol(symbol: Any) -> str:
        normalized = str(symbol or "").upper().strip().replace("/", "-").replace("_", "-")
        for suffix in ("-PERP", "-USDT", "-USD", "USDT", "USD"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    async def close_agent_positions_for_symbol(
        self,
        allocation_id: str,
        symbol: str,
        exit_price: float,
        realized_pnl: float,
        fees: Optional[float] = None,
        trade_metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Close one net exchange position without double-crediting duplicate DB rows."""
        try:
            target_symbol = self._normalize_agent_symbol(symbol)
            active_rows = (
                self.db.table("agent_positions")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("is_active", True)
                .execute()
            ).data or []
            matching_rows = [
                row for row in active_rows
                if self._normalize_agent_symbol(row.get("symbol")) == target_symbol
            ]
            if not matching_rows:
                logger.warning(f"No active DB position rows found for reversal close {allocation_id}/{target_symbol}")
                return False

            now = datetime.now().isoformat()
            position_ids = [row["id"] for row in matching_rows if row.get("id")]
            closed_positions = (
                self.db.table("agent_positions")
                .update({
                    "is_active": False,
                    "current_price": exit_price,
                    "unrealized_pnl": 0,
                    "unrealized_pnl_percent": 0,
                    "updated_at": now,
                })
                .in_("id", position_ids)
                .eq("is_active", True)
                .execute()
            ).data or []
            if not closed_positions:
                return False

            # Several reconciliation rows may point at the same trade. Close and
            # credit each unique trade once so a duplicate position row cannot mint
            # extra allocation budget.
            trade_ids = list({
                str(row.get("trade_id"))
                for row in closed_positions
                if row.get("trade_id")
            })
            trades = []
            if trade_ids:
                trades = (
                    self.db.table("agent_trades")
                    .select("id, trade_amount, position_size")
                    .in_("id", trade_ids)
                    .execute()
                ).data or []

            total_weight = sum(max(float(row.get("position_size") or 0), 0.0) for row in trades)
            capital_to_release = 0.0
            for index, trade in enumerate(trades):
                weight = max(float(trade.get("position_size") or 0), 0.0)
                if total_weight > 0:
                    pnl_share = realized_pnl * (weight / total_weight)
                    fee_share = max(0.0, float(fees or 0)) * (weight / total_weight)
                else:
                    pnl_share = realized_pnl if index == 0 else 0.0
                    fee_share = max(0.0, float(fees or 0)) if index == 0 else 0.0
                capital_to_release += max(float(trade.get("trade_amount") or 0), 0.0)
                trade_update: Dict[str, Any] = {
                    "status": "closed",
                    "exit_price": exit_price,
                    "realized_pnl": pnl_share,
                    "closed_at": now,
                }
                if fees is not None:
                    trade_update["fees"] = fee_share
                if trade_metadata_updates:
                    metadata_rows = (
                        self.db.table("agent_trades")
                        .select("trade_metadata")
                        .eq("id", trade["id"])
                        .limit(1)
                        .execute()
                    ).data or []
                    metadata = dict((metadata_rows[0] if metadata_rows else {}).get("trade_metadata") or {})
                    metadata.update(trade_metadata_updates)
                    trade_update["trade_metadata"] = metadata
                self.db.table("agent_trades").update(trade_update).eq("id", trade["id"]).execute()

            await self._credit_allocation_on_close(
                allocation_id=allocation_id,
                trade_amount=capital_to_release,
                realized_pnl=realized_pnl,
            )
            logger.info(
                f"Closed {len(closed_positions)} DB position row(s) for {target_symbol}; "
                f"credited {len(trades)} unique trade(s) once"
            )
            try:
                from kata.services.agent_allocation_service import get_agent_allocation_service
                asyncio.create_task(
                    get_agent_allocation_service().trigger_yuki_signal_check_for_allocation(allocation_id)
                )
            except Exception as trigger_err:
                logger.warning("Could not trigger post-close Yuki signal check: %s", trigger_err)
            return True
        except Exception as e:
            logger.error(f"Error closing position rows for {allocation_id}/{symbol}: {e}")
            return False

    async def _credit_allocation_on_close(
        self,
        allocation_id: Optional[str],
        trade_amount: float,
        realized_pnl: float
    ):
        """
        Return the trade's capital plus its realized PnL to the allocation budget.

        This makes the allocation compound: a $100 profit on a $400 allocation
        leaves $500 available for the next trade, and losses shrink the budget.
        Without it, capital debited at trade-open is never returned and the
        budget only bleeds down.
        """
        totals = await self.reconcile_agent_allocation_ledger(allocation_id)
        if totals is not None:
            logger.info(
                "Settled allocation %s after close (released $%.2f capital, PnL %+.2f); remaining $%.2f",
                allocation_id, trade_amount, realized_pnl, totals["remaining_amount"],
            )
    
    async def store_agent_decision(
        self, 
        user_id: str, 
        agent_type: str, 
        decision_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Store an agent decision/analysis in the database.
        
        Args:
            user_id: User ID
            agent_type: Agent type
            decision_data: Decision/analysis data
            
        Returns:
            Stored decision data or None if failed
        """
        try:
            # Convert user_id to UUID format for database
            user_uuid = self._ensure_uuid(user_id)
            
            # Prepare decision record
            decision_record = {
                "id": str(uuid.uuid4()),
                "user_id": user_uuid,
                "agent_type": agent_type,
                "decision_type": decision_data.get("decision_type", "market_analysis"),
                "market_data": decision_data.get("market_data", {}),
                "decision_data": decision_data.get("decision_data", {}),
                "action_taken": decision_data.get("action_taken", ""),
                "confidence_score": float(decision_data.get("confidence_score", 0.8)),
                "created_at": datetime.now().isoformat()
            }
            
            # Insert into database
            response = self.db.from_("agent_decisions").insert(decision_record).execute()
            
            if response.data:
                stored_decision = response.data[0]
                logger.info(f"Stored agent decision: {agent_type} {decision_record['decision_type']}")
                return stored_decision
            else:
                logger.error(f"Failed to store agent decision: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing agent decision: {e}")
            return None
    
    async def get_recent_agent_activity(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent agent activity (trades + decisions) for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of activities to return
            
        Returns:
            List of recent activities sorted by timestamp
        """
        try:
            user_uuid = self._ensure_uuid(user_id)
            activities = []
            
            # Get recent trades
            trades_response = self.db.from_("trades").select("*").eq("user_id", user_uuid).order("created_at", desc=True).limit(limit).execute()
            
            if trades_response.data:
                for trade in trades_response.data:
                    activities.append({
                        "type": "trade",
                        "id": trade["id"],
                        "agent_type": trade["agent_type"],
                        "action": f"{trade['trade_type'].title()} {trade['token_symbol']}",
                        "details": f"${trade['value_usd']:.2f} at ${trade['price_usd']:.4f}",
                        "timestamp": trade["created_at"],
                        "confidence": trade.get("confidence_score"),
                        "reasoning": trade.get("reasoning", "")
                    })
            
            # Get recent decisions
            decisions_response = self.db.from_("agent_decisions").select("*").eq("user_id", user_uuid).order("created_at", desc=True).limit(limit).execute()
            
            if decisions_response.data:
                for decision in decisions_response.data:
                    action_taken = decision.get("action_taken", "")
                    if action_taken and "rejected" not in action_taken.lower():
                        activities.append({
                            "type": "decision",
                            "id": decision["id"],
                            "agent_type": decision["agent_type"],
                            "action": action_taken,
                            "details": decision["decision_type"].replace("_", " ").title(),
                            "timestamp": decision["created_at"],
                            "confidence": decision.get("confidence_score"),
                            "reasoning": decision.get("decision_data", {}).get("reasoning", "")
                        })
            
            # Sort by timestamp and return most recent
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            return activities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent agent activity: {e}")
            return []
    
    async def get_agent_last_action(self, user_id: str, agent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the last action performed by a specific agent.
        
        Args:
            user_id: User ID
            agent_type: Agent type (yuki, sakura, ryu)
            
        Returns:
            Last action data or None if no actions found
        """
        try:
            activities = await self.get_recent_agent_activity(user_id, limit=50)
            
            # Find the most recent action for the specified agent
            for activity in activities:
                if activity["agent_type"] == agent_type:
                    return {
                        "action": activity["action"],
                        "details": activity["details"],
                        "timestamp": activity["timestamp"],
                        "confidence": activity.get("confidence"),
                        "reasoning": activity.get("reasoning"),
                        "type": activity["type"]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting agent last action: {e}")
            return None
    
    async def get_agent_performance_stats(self, user_id: str, agent_type: str, days: int = 30) -> Dict[str, Any]:
        """
        Get performance statistics for an agent.
        
        Args:
            user_id: User ID
            agent_type: Agent type
            days: Number of days to analyze
            
        Returns:
            Performance statistics
        """
        try:
            user_uuid = self._ensure_uuid(user_id)
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get trades for the period
            trades_response = self.db.from_("trades").select("*").eq("user_id", user_uuid).eq("agent_type", agent_type).gte("created_at", start_date).execute()
            
            if not trades_response.data:
                return {
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "avg_trade_size": 0.0,
                    "performance_period_days": days
                }
            
            trades = trades_response.data
            total_volume = sum(float(trade["value_usd"]) for trade in trades)
            avg_trade_size = total_volume / len(trades) if trades else 0
            
            return {
                "total_trades": len(trades),
                "total_volume": total_volume,
                "avg_trade_size": avg_trade_size,
                "performance_period_days": days,
                "most_recent_trade": trades[0]["created_at"] if trades else None
            }
            
        except Exception as e:
            logger.error(f"Error getting agent performance stats: {e}")
            return {}
    
    def _ensure_uuid(self, user_id: str) -> str:
        """
        Ensure user_id is in UUID format for database operations.
        
        Args:
            user_id: User ID (string or UUID)
            
        Returns:
            UUID string
        """
        try:
            # Try to parse as UUID
            uuid.UUID(user_id)
            return user_id
        except ValueError:
            # Generate a deterministic UUID from the user_id string
            import hashlib
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
            return str(uuid.uuid5(namespace, user_id))
    
    async def ensure_user_exists(self, user_id: str, wallet_address: str) -> str:
        """
        Ensure user exists in database, create if not found.
        
        Args:
            user_id: User ID
            wallet_address: User's wallet address
            
        Returns:
            User UUID
        """
        try:
            user_uuid = self._ensure_uuid(user_id)
            
            # Check if user exists
            user_response = self.db.from_("users").select("id").eq("id", user_uuid).execute()
            
            if not user_response.data:
                # Create user
                user_record = {
                    "id": user_uuid,
                    "external_wallet": wallet_address,
                    "selected_agent": "yuki",
                    "risk_tolerance": 5,
                    "is_active": True,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                create_response = self.db.from_("users").insert(user_record).execute()
                
                if create_response.data:
                    logger.info(f"Created user {user_uuid} with wallet {wallet_address}")
                else:
                    logger.error(f"Failed to create user: {create_response}")
            
            return user_uuid
            
        except Exception as e:
            logger.error(f"Error ensuring user exists: {e}")
            return user_uuid


# Global instance
agent_db_service = AgentDatabaseService()


def get_agent_db_service() -> AgentDatabaseService:
    """Get the global agent database service instance."""
    return agent_db_service
