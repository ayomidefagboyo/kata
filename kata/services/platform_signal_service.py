"""
Platform Signal Service for Flow AI Trading Platform.

Manages shared trading signals generated through comprehensive platform-wide analysis.
Handles signal generation, storage, retrieval, and user interaction tracking.
Updated: Fixed timezone import issue - forced reload v2.
"""

import logging
import asyncio
import csv
import json
import uuid
from typing import Callable, Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import re

from kata.config.database import get_service_client
from kata.config.settings import settings
# Removed dependency on deleted market_data_service

logger = logging.getLogger(__name__)

THESIS_LIVE_STATUSES = ("active", "hit_target_1", "hit_target_2")
REANALYSIS_TERMINAL_STATUSES = (
    "hit_target_1",
    "hit_target_2",
    "hit_stop_loss",
    "expired",
    "cancelled",
)
ENTRY_REANALYSIS_INVALIDATION_REASONS = frozenset({
    "current_price_crossed_entry_stop_or_target_geometry",
})
ENTRY_TERMINAL_STATES = frozenset({
    "invalidated",
    "missed_entry",
    "skipped_negative_ev",
    "thesis_expired",
})
PLATFORM_GENERATION_REQUEST_TYPE = "platform_signal_generation_request"
PLATFORM_GENERATION_ACTIVE_STATUSES = frozenset({"pending", "running"})


def entry_execution_block_reason(conditions: Any) -> Optional[str]:
    """Explain why an entry cannot activate or be submitted right now."""
    if not isinstance(conditions, dict):
        return None

    if (
        bool(conditions.get("reanalysis_requested"))
        and str(conditions.get("reanalysis_request_status") or "pending").lower()
        == "pending"
    ):
        return "entry re-analysis is pending"

    state = str(conditions.get("entry_order_state") or "").strip().lower()
    if state in ENTRY_TERMINAL_STATES or state == "terminal":
        return f"entry path is terminal ({state})"
    if state in {"awaiting_confirmation", "expired_pending_reanalysis"}:
        return f"entry path is paused ({state})"
    if conditions.get("direction_reversal_pending"):
        return "direction reversal confirmation is pending"
    return None


def persist_entry_activation_if_current(
    db: Any,
    signal_id: str,
    activation_price: float,
    *,
    now: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Activate an entry using the latest database state and an optimistic lock.

    Price monitors keep signal objects in memory. A cancellation/re-analysis can
    update the same signal after that object was loaded, so writing the cached
    ``market_conditions`` object back wholesale can resurrect a terminal entry.
    Reading the latest row and matching ``updated_at`` on update prevents that
    lost-update race without requiring a database migration.
    """
    rows = (
        db.table("platform_signals")
        .select("signal_id,status,market_conditions,updated_at")
        .eq("signal_id", signal_id)
        .limit(1)
        .execute()
    ).data or []
    if not rows:
        return False, {}, "missing_signal"

    row = rows[0]
    conditions = dict(row.get("market_conditions") or {})
    status = str(row.get("status") or "").strip().lower()
    if status not in THESIS_LIVE_STATUSES:
        return False, conditions, f"signal_status_{status or 'unknown'}"
    if not bool(conditions.get("entry_activation_required", False)):
        return True, conditions, "activation_not_required"
    if bool(conditions.get("entry_activated", False)):
        return True, conditions, "already_activated"

    version = row.get("updated_at")
    if not version:
        # A versionless full-JSON write cannot be made race-safe. The normal
        # platform_signals schema always supplies updated_at, so retry rather
        # than risking stale state if a malformed legacy row is encountered.
        return False, conditions, "missing_updated_at_version"

    activated_at = now or datetime.now(timezone.utc)
    if activated_at.tzinfo is None:
        activated_at = activated_at.replace(tzinfo=timezone.utc)
    now_iso = activated_at.isoformat()
    updated_conditions = dict(conditions)
    updated_conditions.update({
        "entry_activated": True,
        "entry_activated_at": now_iso,
        "entry_activation_price": float(activation_price),
    })
    result = (
        db.table("platform_signals")
        .update({
            "market_conditions": updated_conditions,
            "updated_at": now_iso,
        })
        .eq("signal_id", signal_id)
        .eq("updated_at", version)
        .execute()
    )
    if not (result.data or []):
        return False, conditions, "concurrent_state_update"
    return True, updated_conditions, "activated"


def persist_market_conditions_if_current(
    db: Any,
    signal_id: str,
    transform: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    *,
    now: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    """Safely transform market conditions without overwriting concurrent state."""
    rows = (
        db.table("platform_signals")
        .select("signal_id,status,market_conditions,updated_at")
        .eq("signal_id", signal_id)
        .limit(1)
        .execute()
    ).data or []
    if not rows:
        return False, {}, "missing_signal"

    row = rows[0]
    conditions = dict(row.get("market_conditions") or {})
    status = str(row.get("status") or "").strip().lower()
    if status not in THESIS_LIVE_STATUSES:
        return False, conditions, f"signal_status_{status or 'unknown'}"

    updated_conditions = transform(dict(conditions))
    if updated_conditions is None or updated_conditions == conditions:
        return True, conditions, "unchanged"
    if not isinstance(updated_conditions, dict):
        raise TypeError("market_conditions transform must return a dict or None")

    version = row.get("updated_at")
    if not version:
        return False, conditions, "missing_updated_at_version"

    updated_at = now or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    now_iso = updated_at.isoformat()
    result = (
        db.table("platform_signals")
        .update({
            "market_conditions": updated_conditions,
            "updated_at": now_iso,
        })
        .eq("signal_id", signal_id)
        .eq("updated_at", version)
        .execute()
    )
    if not (result.data or []):
        return False, conditions, "concurrent_state_update"
    return True, updated_conditions, "updated"


def is_missed_entry_reason(reason: Any) -> bool:
    """Return whether an unfilled entry is stale rather than thesis-invalid."""
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in (
        "expired",
        "entry_ttl",
        "missed_entry",
        "stale",
        "target_1_touched",
        "target_2_touched",
        "current_price_crossed_entry_stop_or_target_geometry",
    ))


def terminalize_entry_conditions(
    conditions: Dict[str, Any],
    state: str,
    reason: Any,
    now_iso: str,
) -> Dict[str, Any]:
    """Close an execution path without misclassifying the platform thesis."""
    conditions.update({
        "entry_order_state": state,
        "entry_terminal_at": now_iso,
        "entry_terminal_reason": str(reason or state)[:500],
        "entry_revalidation_required_before_resubmit": False,
        "reanalysis_requested": False,
        "reanalysis_request_status": state,
        "reanalysis_processed_at": now_iso,
    })
    return conditions


def entry_order_deadline(
    signal_started_at: Any,
    signal_expires_at: Any,
    max_ttl_hours: float = 12.0,
    validity_fraction: float = 0.50,
    now: Optional[datetime] = None,
) -> datetime:
    """
    Return the deadline for an unfilled entry order.

    Entry freshness and thesis validity are deliberately separate:
    an entry may work for at most ``max_ttl_hours`` and never for more than
    ``validity_fraction`` of the original thesis window. The thesis can remain
    valid after this deadline, but it must be re-analysed before a new order is
    submitted.
    """
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    def _parse(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            raw = str(value).replace("Z", "+00:00")
            fractional = re.match(r"^(.*\.)(\d+)([+-]\d{2}:\d{2})$", raw)
            if fractional:
                raw = (
                    fractional.group(1)
                    + fractional.group(2)[:6].ljust(6, "0")
                    + fractional.group(3)
                )
            parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    try:
        started_at = _parse(signal_started_at) or current_time
    except (TypeError, ValueError):
        started_at = current_time
    try:
        thesis_expires_at = _parse(signal_expires_at)
    except (TypeError, ValueError):
        thesis_expires_at = None

    ttl_hours = max(0.25, float(max_ttl_hours or 0.0))
    fraction = max(0.01, min(1.0, float(validity_fraction or 0.50)))
    ttl_deadline = started_at + timedelta(hours=ttl_hours)

    if thesis_expires_at and thesis_expires_at > started_at:
        thesis_window = thesis_expires_at - started_at
        fractional_deadline = started_at + thesis_window * fraction
        return min(thesis_expires_at, ttl_deadline, fractional_deadline)

    return ttl_deadline


def _normalize_text_array(value: Any) -> List[str]:
    """Normalize AI/database list-like values for Postgres TEXT[] columns."""
    if value is None:
        return []

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        if raw.startswith("["):
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    return _normalize_text_array(decoded)
            except (TypeError, ValueError):
                pass

        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = next(csv.reader([raw[1:-1]], skipinitialspace=True))
                return [
                    item.strip()
                    for item in parsed
                    if item and item.strip() and item.strip().upper() != "NULL"
                ]
            except csv.Error:
                pass

        separators = [";", "\n"] if (";" in raw or "\n" in raw) else []
        if separators:
            parts = [raw]
            for separator in separators:
                parts = [
                    chunk
                    for part in parts
                    for chunk in part.split(separator)
                ]
            return [part.strip() for part in parts if part.strip()]

        return [raw]

    if isinstance(value, (list, tuple, set)):
        normalized: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple, set)):
                normalized.extend(_normalize_text_array(item))
                continue
            text = str(item).strip() if item is not None else ""
            if text:
                normalized.append(text)
        return normalized

    text = str(value).strip()
    return [text] if text else []


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Robust timestamp parsing that handles various formats from PostgreSQL.
    
    Args:
        timestamp_str: Timestamp string from database
        
    Returns:
        timezone-aware datetime in UTC
    """
    if not timestamp_str:
        return datetime.now(timezone.utc)

    try:
        # Remove timezone indicator and normalize
        cleaned = timestamp_str.replace('Z', '+00:00')
        
        # Handle microseconds precision issue - truncate to 6 digits
        if '.' in cleaned and '+' in cleaned:
            date_part, tz_part = cleaned.split('+')
            if '.' in date_part:
                base, microseconds = date_part.split('.')
                # Truncate microseconds to 6 digits
                microseconds = microseconds[:6].ljust(6, '0')
                cleaned = f"{base}.{microseconds}+{tz_part}"
        
        # Parse and normalize to UTC-aware datetime
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
        
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}. Using current time.")
        return datetime.now(timezone.utc)


def signal_thesis_is_live(
    signal_row: Dict[str, Any],
    now: Optional[datetime] = None,
) -> bool:
    """Return whether a signal still owns its token/direction lifecycle slot."""
    status = str(signal_row.get("status") or "").lower()
    if status not in THESIS_LIVE_STATUSES:
        return False

    expires_at = signal_row.get("expires_at")
    if not expires_at:
        # Missing horizon data must fail closed so a generator error cannot stack exposure.
        return True

    current_time = now or datetime.now(timezone.utc)
    return parse_timestamp(str(expires_at)) > current_time


class SignalPool(Enum):
    """Signal pool categories for different user modes."""
    FAST_MODE = "fast_mode"
    FULL_MODE = "full_mode"  
    PREMIUM = "premium"




@dataclass
class PlatformSignal:
    """Platform-generated trading signal."""
    # Required fields (no defaults)
    signal_id: str
    token_symbol: str
    direction: str  # LONG/SHORT
    timeframe: str
    
    # Signal Quality
    confidence: float
    overall_score: float
    signal_strength: str
    time_horizon: str
    
    # Price Targets
    entry_price: float
    target_1: float
    target_1_probability: float
    target_2: float
    target_2_probability: float
    stop_loss: float
    risk_reward_ratio: float
    
    # Analysis Data
    market_conditions: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    risk_factors: List[str]
    
    # Platform Metadata
    signal_pool: SignalPool
    opportunity_rank: int
    analysis_timestamp: datetime
    expires_at: datetime
    validity_window_hours: int
    
    # Optional fields (all with defaults must come last)
    status: str = "active"
    run_id: Optional[str] = None  # Analysis run identifier
    analysis_notes: Optional[str] = None
    ai_reasoning: Optional[str] = None
    ai_key_factors: Optional[List[str]] = None
    ai_risk_assessment: Optional[str] = None
    ai_confidence_breakdown: Optional[Dict[str, Any]] = None
    logo_url: Optional[str] = None  # Token logo URL
    target_1_hit: bool = False
    target_1_hit_price: Optional[float] = None
    target_1_hit_at: Optional[datetime] = None
    target_1_leveraged_pnl_percent: Optional[float] = None
    
    # Position Management (from AI analysis)
    leverage: int = 5
    position_size: float = 10.0
    risk_level: str = "MEDIUM"

    # Learning System Integration
    learning_tracked: bool = False
    learning_started_at: Optional[datetime] = None
    learning_completed_at: Optional[datetime] = None
    market_regime: Optional[Dict[str, Any]] = None
    regime_confidence: Optional[float] = None
    volatility_environment: Optional[str] = None
    pattern_classification: Optional[str] = None
    pattern_confidence: Optional[float] = None
    historical_pattern_success_rate: Optional[float] = None
    learning_version: int = 1
    prompt_version: Optional[str] = None
    generated_by_agent: Optional[str] = None
    predicted_success_probability: Optional[float] = None
    predicted_time_to_target: Optional[int] = None
    risk_adjusted_confidence: Optional[float] = None
    learning_insights_id: Optional[str] = None
    pattern_performance_id: Optional[str] = None

    def __post_init__(self):
        self.risk_factors = _normalize_text_array(self.risk_factors)
        if self.ai_key_factors is not None:
            self.ai_key_factors = _normalize_text_array(self.ai_key_factors)




@dataclass
class UserSignalTracking:
    """User interaction with platform signals."""
    user_id: str
    platform_signal_id: str
    signal_id: str
    viewed_at: datetime
    followed: bool = False
    followed_at: Optional[datetime] = None
    user_entry_price: Optional[float] = None
    user_exit_price: Optional[float] = None
    user_exit_timestamp: Optional[datetime] = None
    user_position_size: Optional[float] = None
    profit_loss_percentage: Optional[float] = None
    profit_loss_absolute: Optional[float] = None
    user_outcome: Optional[str] = None
    delivery_mode: str = "fast_mode"


class PlatformSignalService:
    """Core platform signal management service."""

    def __init__(self):
        """Initialize platform signal service."""
        self.db = get_service_client()
        self.market_service = None
        self._last_no_signals_log_at: Optional[datetime] = None
        self._performance_metrics_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self._performance_metrics_cache_ttl_seconds = 15
        self._summary_metrics_unavailable_logged_at: Optional[datetime] = None
        logger.info("Platform signal service initialized")
    
    def set_market_service(self, market_service):
        """Set market data service for price monitoring."""
        self.market_service = market_service
        logger.info("Market service connected to platform signal service")

    @staticmethod
    def _request_timestamp(value: Any) -> Optional[datetime]:
        """Parse queue timestamps stored inside JSON request metadata."""
        if not value:
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _recent_platform_generation_requests(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Load recent durable generation-control messages from the shared DB."""
        result = (
            self.db.from_("agent_learning_insights")
            .select("id,signal_id,insights,performance_impact,timestamp,updated_at")
            .eq("agent_type", "yuki")
            .eq("learning_type", PLATFORM_GENERATION_REQUEST_TYPE)
            .order("timestamp", desc=True)
            .limit(max(1, min(500, int(limit or 200))))
            .execute()
        )
        return result.data or []

    async def enqueue_platform_generation_request(
        self,
        requested_by: Optional[str] = None,
        *,
        trigger: str = "manual_override",
        scheduled_for_utc: Optional[Any] = None,
        configured_hours_utc: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Queue one full platform-generation request for the existing worker.

        The learning-insights table already acts as the durable worker control
        ledger for schedule tracking, so this handoff requires no third service
        or ephemeral in-process task. Repeated manual clicks reuse the active
        generation request; schedule-change messages remain one event per Apply.
        """
        now = datetime.now(timezone.utc)
        normalized_trigger = str(trigger or "manual_override").strip().lower()
        scheduled_for = self._request_timestamp(scheduled_for_utc)
        try:
            if normalized_trigger == "scheduled":
                if scheduled_for is None:
                    raise ValueError("scheduled_for_utc is required for a scheduled request")
                request_token = f"scheduled_{scheduled_for.strftime('%Y%m%dT%H%M%SZ')}"
                request_signal_id = f"platform_generation_request_{request_token}"
                existing = (
                    self.db.from_("agent_learning_insights")
                    .select("id,signal_id,insights,timestamp,updated_at")
                    .eq("agent_type", "yuki")
                    .eq("learning_type", PLATFORM_GENERATION_REQUEST_TYPE)
                    .eq("signal_id", request_signal_id)
                    .limit(1)
                    .execute()
                ).data or []
                if existing:
                    row = existing[0]
                    insights = row.get("insights") or {}
                    return {
                        "created": False,
                        "request_id": str(row.get("id") or request_token),
                        "status": str(insights.get("status") or "pending"),
                        "requested_at": insights.get("requested_at") or row.get("timestamp"),
                        "trigger": normalized_trigger,
                    }
            elif normalized_trigger == "schedule_change":
                request_token = f"schedule_change_{uuid.uuid4().hex}"
                request_signal_id = f"platform_generation_request_{request_token}"
            else:
                for row in self._recent_platform_generation_requests():
                    insights = row.get("insights") or {}
                    if not isinstance(insights, dict):
                        continue
                    status = str(insights.get("status") or "").lower()
                    if status not in PLATFORM_GENERATION_ACTIVE_STATUSES:
                        continue
                    if str(insights.get("trigger") or "").lower() == "schedule_change":
                        continue
                    return {
                        "created": False,
                        "request_id": str(row.get("id") or insights.get("request_id") or ""),
                        "status": status,
                        "requested_at": insights.get("requested_at") or row.get("timestamp"),
                        "trigger": str(insights.get("trigger") or normalized_trigger),
                    }

                request_token = uuid.uuid4().hex
                request_signal_id = f"platform_generation_request_{request_token}"

            requested_at = now.isoformat()
            payload = {
                "agent_type": "yuki",
                "signal_id": request_signal_id,
                "learning_level": "meta",
                "learning_type": PLATFORM_GENERATION_REQUEST_TYPE,
                "confidence_score": 1.0,
                "timestamp": requested_at,
                "insights": {
                    "request_id": request_token,
                    "status": "pending",
                    "trigger": normalized_trigger,
                    "requested_at": requested_at,
                    "requested_by": str(requested_by)[:255] if requested_by else None,
                    "scheduled_for_utc": scheduled_for.isoformat() if scheduled_for else None,
                    "slot_hour_utc": scheduled_for.hour if scheduled_for else None,
                    "configured_hours_utc": list(configured_hours_utc or []),
                    "attempt_count": 0,
                },
                "performance_impact": {},
                "risk_metrics": {},
            }
            result = self.db.from_("agent_learning_insights").insert(payload).execute()
            row = (result.data or [{}])[0]
            request_id = str(row.get("id") or request_token)
            logger.info(
                "Queued %s platform generation request %s",
                normalized_trigger,
                request_id,
            )
            return {
                "created": True,
                "request_id": request_id,
                "status": "pending",
                "requested_at": requested_at,
                "trigger": normalized_trigger,
            }
        except Exception as exc:
            if normalized_trigger == "scheduled" and scheduled_for is not None:
                # Concurrent deployment instances can enqueue the same deterministic
                # slot during a rolling deploy. Treat the winner's row as success.
                request_signal_id = (
                    "platform_generation_request_scheduled_"
                    f"{scheduled_for.strftime('%Y%m%dT%H%M%SZ')}"
                )
                try:
                    existing = (
                        self.db.from_("agent_learning_insights")
                        .select("id,insights,timestamp")
                        .eq("signal_id", request_signal_id)
                        .limit(1)
                        .execute()
                    ).data or []
                    if existing:
                        row = existing[0]
                        insights = row.get("insights") or {}
                        return {
                            "created": False,
                            "request_id": str(row.get("id") or ""),
                            "status": str(insights.get("status") or "pending"),
                            "requested_at": insights.get("requested_at") or row.get("timestamp"),
                            "trigger": normalized_trigger,
                        }
                except Exception:
                    pass
            logger.error("Could not queue platform generation request: %s", exc)
            raise

    async def claim_platform_generation_request(
        self,
        worker_id: str,
        stale_after_minutes: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Claim the oldest pending request, including abandoned stale claims."""
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(minutes=max(15, int(stale_after_minutes or 30)))
        try:
            candidates: List[Dict[str, Any]] = []
            for row in self._recent_platform_generation_requests():
                insights = row.get("insights") or {}
                if not isinstance(insights, dict):
                    continue
                status = str(insights.get("status") or "").lower()
                if status == "pending":
                    candidates.append(row)
                    continue
                if status != "running":
                    continue
                claimed_at = self._request_timestamp(
                    insights.get("claimed_at") or row.get("updated_at")
                )
                if claimed_at is None or claimed_at <= stale_before:
                    candidates.append(row)

            if not candidates:
                return None
            candidates.sort(
                key=lambda row: str(
                    (row.get("insights") or {}).get("requested_at")
                    or row.get("timestamp")
                    or ""
                )
            )
            request = candidates[0]
            insights = dict(request.get("insights") or {})
            insights.update({
                "status": "running",
                "claimed_at": now.isoformat(),
                "claimed_by": str(worker_id),
                "attempt_count": int(insights.get("attempt_count") or 0) + 1,
            })
            claim_query = (
                self.db.from_("agent_learning_insights")
                .update({"insights": insights, "updated_at": now.isoformat()})
                .eq("id", request["id"])
            )
            # Compare-and-set prevents old/new deployment instances from both
            # claiming the same pending slot during a rolling deployment.
            if request.get("updated_at") is not None:
                claim_query = claim_query.eq("updated_at", request["updated_at"])
            updated = claim_query.execute()
            if not updated.data:
                return None
            request["insights"] = insights
            logger.info(
                "Worker %s claimed platform generation request %s",
                worker_id,
                request.get("id"),
            )
            return request
        except Exception as exc:
            logger.warning("Could not claim platform generation request: %s", exc)
            return None

    async def finish_platform_generation_request(
        self,
        request_id: str,
        status: str,
        *,
        signals_generated: Optional[int] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Persist the terminal or retryable state of a generation request."""
        normalized_status = str(status or "failed").lower()
        if normalized_status not in {"pending", "completed", "failed"}:
            raise ValueError(f"Unsupported generation request status: {status}")
        try:
            rows = (
                self.db.from_("agent_learning_insights")
                .select("insights,performance_impact")
                .eq("id", request_id)
                .limit(1)
                .execute()
            ).data or []
            if not rows:
                return False
            now_iso = datetime.now(timezone.utc).isoformat()
            insights = dict(rows[0].get("insights") or {})
            insights["status"] = normalized_status
            if normalized_status == "pending":
                insights["released_at"] = now_iso
                insights.pop("claimed_by", None)
            else:
                insights["completed_at"] = now_iso
            if signals_generated is not None:
                insights["signals_generated"] = int(signals_generated)
            if error:
                insights["error"] = str(error)[:1000]
            else:
                insights.pop("error", None)

            performance_impact = dict(rows[0].get("performance_impact") or {})
            if signals_generated is not None:
                performance_impact["signals_generated"] = int(signals_generated)
            updated = (
                self.db.from_("agent_learning_insights")
                .update({
                    "insights": insights,
                    "performance_impact": performance_impact,
                    "updated_at": now_iso,
                })
                .eq("id", request_id)
                .execute()
            )
            return bool(updated.data)
        except Exception as exc:
            logger.warning(
                "Could not finish platform generation request %s: %s",
                request_id,
                exc,
            )
            return False

    async def request_signal_reanalysis(
        self,
        signal_id: str,
        reason: str,
        *,
        observed_price: Optional[float] = None,
        observed_volume: Optional[float] = None,
        invalidate_pending_entry: bool = False,
        expire_entry_order: bool = False,
        source: str = "entry_monitor",
    ) -> bool:
        """
        Queue an event-driven, full-stack signal re-evaluation.

        A stale live order is expired before it is queued, without making the
        source thesis terminal. Only an explicit structural caller may set
        ``invalidate_pending_entry``. A material price/volume event only queues
        the request; the accepted thesis remains authoritative until fresh
        direction, regime, levels and EV are ready.
        """
        try:
            rows = (
                self.db.table("platform_signals")
                .select("signal_id,status,market_conditions,analysis_notes")
                .eq("signal_id", signal_id)
                .limit(1)
                .execute()
            ).data or []
            if not rows:
                return False

            row = rows[0]
            now_iso = datetime.now(timezone.utc).isoformat()
            conditions = dict(row.get("market_conditions") or {})
            if str(conditions.get("entry_order_state") or "").lower() in ENTRY_TERMINAL_STATES:
                logger.debug("Entry path is already terminal for %s", signal_id)
                return True
            if expire_entry_order and is_missed_entry_reason(reason):
                terminalize_entry_conditions(
                    conditions,
                    "missed_entry",
                    reason,
                    now_iso,
                )
                existing_notes = str(row.get("analysis_notes") or "").strip()
                note = f"Entry closed as missed without invalidating thesis: {reason}"
                result = (
                    self.db.table("platform_signals")
                    .update({
                        "market_conditions": conditions,
                        "analysis_notes": f"{existing_notes}\n{note}".strip(),
                        "updated_at": now_iso,
                    })
                    .eq("signal_id", signal_id)
                    .execute()
                )
                if result.data:
                    logger.info("Closed stale entry path for %s as missed_entry (%s)", signal_id, reason)
                return bool(result.data)
            if (
                expire_entry_order
                and bool(conditions.get("reanalysis_requested"))
                and str(conditions.get("reanalysis_request_status") or "pending").lower()
                == "pending"
                and str(conditions.get("entry_order_state") or "").lower()
                == "expired_pending_reanalysis"
            ):
                logger.debug("Entry re-analysis already pending for %s", signal_id)
                return True
            conditions.update({
                "reanalysis_requested": True,
                "reanalysis_requested_at": now_iso,
                "reanalysis_reason": str(reason or "material_entry_event")[:500],
                "reanalysis_source": str(source or "entry_monitor")[:100],
                "reanalysis_request_status": "pending",
            })
            if observed_price is not None:
                conditions["reanalysis_observed_price"] = float(observed_price)
            if observed_volume is not None:
                conditions["reanalysis_observed_volume"] = float(observed_volume)
            if expire_entry_order:
                conditions.update({
                    "entry_order_state": "expired_pending_reanalysis",
                    "entry_order_expired_at": now_iso,
                    "entry_order_expiration_reason": str(reason or "entry_expired")[:500],
                    "entry_revalidation_required_before_resubmit": True,
                })

            payload: Dict[str, Any] = {
                "market_conditions": conditions,
                "updated_at": now_iso,
            }
            if invalidate_pending_entry:
                conditions.update({
                    "entry_order_state": "invalidated",
                    "invalidation_kind": "structural",
                    "invalidated_at": now_iso,
                    "invalidation_reason": str(reason or "structural_invalidation")[:500],
                })
                existing_notes = str(row.get("analysis_notes") or "").strip()
                note = f"Thesis structurally invalidated: {reason}"
                payload.update({
                    "status": "invalidated",
                    "analysis_notes": f"{existing_notes}\n{note}".strip(),
                })
            elif expire_entry_order:
                existing_notes = str(row.get("analysis_notes") or "").strip()
                note = f"Entry window expired pending fresh re-analysis: {reason}"
                payload["analysis_notes"] = f"{existing_notes}\n{note}".strip()

            result = (
                self.db.table("platform_signals")
                .update(payload)
                .eq("signal_id", signal_id)
                .execute()
            )
            if result.data:
                if invalidate_pending_entry:
                    try:
                        (
                            self.db.table("platform_signal_performance_tracking")
                            .update({
                                "outcome": "invalidated",
                                "exit_reason": "INVALIDATED",
                                "exit_timestamp": now_iso,
                                "last_updated": now_iso,
                            })
                            .eq("signal_id", signal_id)
                            .eq("outcome", "active")
                            .execute()
                        )
                    except Exception as perf_exc:
                        logger.warning(
                            "Could not synchronize structural invalidation for %s: %s",
                            signal_id,
                            perf_exc,
                        )
                logger.info(
                    "Queued signal %s for event re-analysis (%s, expire=%s, invalidate=%s)",
                    signal_id,
                    reason,
                    expire_entry_order,
                    invalidate_pending_entry,
                )
                return True
            return False
        except Exception as exc:
            logger.warning("Could not queue signal %s for re-analysis: %s", signal_id, exc)
            return False

    async def get_pending_reanalysis_requests(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return queued entry re-analysis requests, oldest first."""
        try:
            batch_size = max(25, int(limit or 20) * 5)
            rows = (
                self.db.table("platform_signals")
                .select(
                    "id,signal_id,token_symbol,direction,status,analysis_timestamp,expires_at,"
                    "entry_price,stop_loss,target_1,target_2,time_horizon,confidence,"
                    "market_conditions,analysis_notes,ai_reasoning,leverage,position_size"
                )
                # Query the pending queue directly. Scanning a generic active-signal
                # window can miss freshly queued rows whose updated_at moved them
                # past the first page of active signals.
                .in_("status", ["active", "invalidated"])
                .contains("market_conditions", {"reanalysis_requested": True})
                .eq("market_conditions->>reanalysis_request_status", "pending")
                .order("updated_at", desc=False)
                .limit(batch_size)
                .execute()
            ).data or []
            requests = []
            for row in rows:
                conditions = row.get("market_conditions") or {}
                if not isinstance(conditions, dict):
                    continue
                if not conditions.get("reanalysis_requested"):
                    continue
                if str(conditions.get("reanalysis_request_status") or "pending") != "pending":
                    continue
                requests.append(row)
            requests.sort(
                key=lambda row: (
                    0
                    if str(
                        ((row.get("market_conditions") or {}).get("reanalysis_source") or "")
                    ).lower() == "manual_operator"
                    else 1,
                    str(
                        (row.get("market_conditions") or {}).get("reanalysis_requested_at") or ""
                    ),
                )
            )
            return requests[: max(1, int(limit or 20))]
        except Exception as exc:
            logger.warning("Could not load event re-analysis requests: %s", exc)
            return []

    async def record_reanalysis_attempt(
        self,
        signal_id: str,
        state_hash: str,
    ) -> bool:
        """Persist the state hash for the latest actual re-analysis attempt."""
        if not signal_id or not state_hash:
            return False
        try:
            rows = (
                self.db.table("platform_signals")
                .select("market_conditions")
                .eq("signal_id", signal_id)
                .limit(1)
                .execute()
            ).data or []
            if not rows:
                return False
            conditions = dict(rows[0].get("market_conditions") or {})
            now_iso = datetime.now(timezone.utc).isoformat()
            conditions.update({
                "reanalysis_last_attempt_state_hash": state_hash,
                "reanalysis_last_attempted_at": now_iso,
            })
            updated = (
                self.db.table("platform_signals")
                .update({
                    "market_conditions": conditions,
                    "updated_at": now_iso,
                })
                .eq("signal_id", signal_id)
                .execute()
            )
            return bool(updated.data)
        except Exception as exc:
            logger.warning("Could not persist re-analysis attempt state for %s: %s", signal_id, exc)
            return False

    async def mark_reanalysis_request_coalesced(
        self,
        signal_id: str,
        state_hash: str,
    ) -> bool:
        """Record that a pending re-analysis was skipped because the state is unchanged."""
        if not signal_id or not state_hash:
            return False
        try:
            rows = (
                self.db.table("platform_signals")
                .select("market_conditions")
                .eq("signal_id", signal_id)
                .limit(1)
                .execute()
            ).data or []
            if not rows:
                return False
            conditions = dict(rows[0].get("market_conditions") or {})
            now_iso = datetime.now(timezone.utc).isoformat()
            conditions.update({
                "reanalysis_last_coalesced_state_hash": state_hash,
                "reanalysis_last_coalesced_at": now_iso,
            })
            updated = (
                self.db.table("platform_signals")
                .update({
                    "market_conditions": conditions,
                    "updated_at": now_iso,
                })
                .eq("signal_id", signal_id)
                .execute()
            )
            return bool(updated.data)
        except Exception as exc:
            logger.warning("Could not mark re-analysis %s as coalesced: %s", signal_id, exc)
            return False

    @staticmethod
    def _close_terminal_reanalysis_context(
        market_conditions: Any,
        terminal_at: Any,
    ) -> Tuple[Dict[str, Any], bool]:
        """Close queue metadata that can no longer be processed after termination."""
        conditions = dict(market_conditions or {})
        request_status = str(
            conditions.get("reanalysis_request_status") or ""
        ).lower()
        entry_state = str(conditions.get("entry_order_state") or "").lower()
        is_stale = (
            bool(conditions.get("reanalysis_requested"))
            or request_status == "pending"
            or entry_state in {
                "expired_pending_reanalysis",
                "awaiting_confirmation",
                "thesis_expired",
            }
        )
        if not is_stale:
            return conditions, False

        terminal_iso = (
            terminal_at.isoformat()
            if isinstance(terminal_at, datetime)
            else str(terminal_at)
        )
        conditions.update({
            "reanalysis_requested": False,
            "reanalysis_request_status": "terminal",
            "reanalysis_processed_at": terminal_iso,
            "entry_order_state": "terminal",
            "entry_revalidation_required_before_resubmit": False,
        })
        return conditions, True

    async def close_terminal_reanalysis_requests(self, limit: int = 100) -> int:
        """Repair stale queue flags on already-terminal signals in small batches."""
        try:
            rows = (
                self.db.table("platform_signals")
                .select("signal_id,status,market_conditions")
                .in_("status", list(REANALYSIS_TERMINAL_STATUSES))
                .contains("market_conditions", {"reanalysis_requested": True})
                .eq("market_conditions->>reanalysis_request_status", "pending")
                .limit(max(1, int(limit or 100)))
                .execute()
            ).data or []
            if not rows:
                return 0

            now_iso = datetime.now(timezone.utc).isoformat()
            closed = 0
            for row in rows:
                signal_id = str(row.get("signal_id") or "")
                if not signal_id:
                    continue
                conditions, changed = self._close_terminal_reanalysis_context(
                    row.get("market_conditions"),
                    now_iso,
                )
                if not changed:
                    continue
                result = (
                    self.db.table("platform_signals")
                    .update({
                        "market_conditions": conditions,
                        "updated_at": now_iso,
                    })
                    .eq("signal_id", signal_id)
                    .in_("status", list(REANALYSIS_TERMINAL_STATUSES))
                    .execute()
                )
                if result.data:
                    closed += 1

            if closed:
                logger.info(
                    "Closed %d stale terminal signal re-analysis request(s)",
                    closed,
                )
            return closed
        except Exception as exc:
            logger.warning("Could not close stale terminal re-analysis requests: %s", exc)
            return 0

    async def mark_reanalysis_request_processed(
        self,
        signal_id: str,
        result: str,
        replacement_signal_id: Optional[str] = None,
    ) -> bool:
        """Close a re-analysis queue item while preserving its audit trail."""
        try:
            rows = (
                self.db.table("platform_signals")
                .select(
                    "status,analysis_timestamp,created_at,expires_at,"
                    "market_conditions,analysis_notes"
                )
                .eq("signal_id", signal_id)
                .limit(1)
                .execute()
            ).data or []
            if not rows:
                return False
            row = rows[0]
            conditions = dict(row.get("market_conditions") or {})
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            conditions.update({
                "reanalysis_requested": False,
                "reanalysis_request_status": str(result or "processed"),
                "reanalysis_processed_at": now_iso,
            })
            if replacement_signal_id:
                conditions["reanalysis_replacement_signal_id"] = replacement_signal_id

            latest_revalidation = str(
                conditions.get("last_entry_revalidation_result") or ""
            ).lower()
            result_value = str(result or "processed")
            row_status = str(row.get("status") or "").lower()
            entry_order_state = str(conditions.get("entry_order_state") or "").lower()
            reanalysis_reason = str(conditions.get("reanalysis_reason") or "").lower()
            invalidate_stale_entry = (
                result_value == "fresh_stack_rejected"
                and row_status in THESIS_LIVE_STATUSES
                and entry_order_state == "expired_pending_reanalysis"
                and reanalysis_reason in ENTRY_REANALYSIS_INVALIDATION_REASONS
            )
            can_rearm = (
                result_value == "revalidated_without_replacement"
                and row_status in THESIS_LIVE_STATUSES
                and latest_revalidation in {"confirmed", "retained"}
            )
            payload: Dict[str, Any] = {
                "market_conditions": conditions,
                "updated_at": now_iso,
            }
            if result_value in {"missed_entry", "skipped_negative_ev"}:
                terminalize_entry_conditions(
                    conditions,
                    result_value,
                    conditions.get("last_entry_revalidation_reason") or result_value,
                    now_iso,
                )
            elif can_rearm:
                try:
                    current_revision = max(0, int(conditions.get("entry_revision") or 0))
                except (TypeError, ValueError):
                    current_revision = 0
                max_revisions = max(
                    0,
                    int(getattr(settings, "YUKI_MAX_ENTRY_REVISIONS", 2) or 2),
                )
                deadline = entry_order_deadline(
                    signal_started_at=row.get("analysis_timestamp") or row.get("created_at"),
                    signal_expires_at=row.get("expires_at"),
                    max_ttl_hours=float(settings.YUKI_ENTRY_ORDER_TTL_HOURS or 12.0),
                    validity_fraction=float(settings.YUKI_ENTRY_TTL_VALIDITY_FRACTION or 0.50),
                    now=now,
                )
                if deadline > now and current_revision < max_revisions:
                    revision = current_revision + 1
                    conditions.update({
                        "entry_order_state": "revalidated",
                        "entry_order_rearmed_at": now_iso,
                        "entry_order_expires_at": deadline.isoformat(),
                        "entry_revision": revision,
                        "entry_activated": False,
                        "entry_activated_at": None,
                        "entry_activation_price": None,
                        "entry_revalidation_required_before_resubmit": False,
                    })
                else:
                    terminal_reason = (
                        "maximum_entry_revisions_reached"
                        if current_revision >= max_revisions
                        else "original_entry_window_expired"
                    )
                    terminalize_entry_conditions(
                        conditions,
                        "missed_entry",
                        terminal_reason,
                        now_iso,
                    )
            elif invalidate_stale_entry:
                invalidation_reason = str(
                    conditions.get("reanalysis_reason")
                    or "entry_reanalysis_rejected"
                )[:500]
                terminalize_entry_conditions(
                    conditions,
                    "missed_entry",
                    invalidation_reason,
                    now_iso,
                )
                existing_notes = str(row.get("analysis_notes") or "").strip()
                lifecycle_note = (
                    "Pending entry closed as missed after re-analysis: "
                    f"{invalidation_reason}"
                )
                payload["analysis_notes"] = f"{existing_notes}\n{lifecycle_note}".strip()
            elif entry_order_state == "expired_pending_reanalysis":
                conditions.update({
                    "entry_order_state": "awaiting_confirmation",
                    "entry_revalidation_required_before_resubmit": True,
                })
            updated = (
                self.db.table("platform_signals")
                .update(payload)
                .eq("signal_id", signal_id)
                .execute()
            )
            return bool(updated.data)
        except Exception as exc:
            logger.warning("Could not finish re-analysis request %s: %s", signal_id, exc)
            return False

    def _is_missing_relation_error(self, error: Exception, relation_name: str) -> bool:
        """Detect missing relation errors from Supabase/Postgres."""
        message = str(error).lower()
        relation = relation_name.lower()
        return (
            relation in message
            and (
                'does not exist' in message
                or 'undefined table' in message
                or '42p01' in message
            )
        )

    def _run_active_signal_query(
        self,
        source_table: str,
        signal_pool: Optional[SignalPool] = None,
        opportunity_ranks: Optional[List[int]] = None,
        limit: int = 10,
        order_by_rank_first: bool = False,
    ):
        """Build and execute active-signal query from a source table/view."""
        query = self.db.table(source_table).select('*')

        if signal_pool:
            query = query.eq('signal_pool', signal_pool.value)

        if opportunity_ranks is not None:
            query = query.in_('opportunity_rank', opportunity_ranks)

        query = query.eq('status', 'active')

        current_time = datetime.now(timezone.utc).isoformat()
        query = query.gt('expires_at', current_time)

        if order_by_rank_first:
            query = query.order('opportunity_rank', desc=False).order('analysis_timestamp', desc=True)
        else:
            query = query.order('analysis_timestamp', desc=True).order('opportunity_rank', desc=False)

        return query.limit(limit).execute()

    def _convert_db_records_to_signals(self, records: List[Dict[str, Any]], context: str) -> List[PlatformSignal]:
        """Convert DB rows to PlatformSignal objects with safe logging."""
        signals = []
        for record in records:
            try:
                signals.append(self._db_record_to_platform_signal(record))
            except Exception as convert_error:
                logger.error(f"Error converting {context} record: {convert_error}")
                logger.error(f"Problem record: {record}")
        return signals

    def enable_learning_tracking(self, signal: PlatformSignal,
                                generated_by_agent: str = None,
                                market_regime: Dict[str, Any] = None,
                                pattern_classification: str = None,
                                pattern_confidence: float = None) -> PlatformSignal:
        """
        Enable learning tracking for a platform signal.

        Args:
            signal: PlatformSignal to enable learning for
            generated_by_agent: Agent that generated this signal ('yuki', 'ryu', 'sakura', 'platform')
            market_regime: Current market regime information
            pattern_classification: Identified pattern type
            pattern_confidence: Confidence in pattern classification

        Returns:
            Updated PlatformSignal with learning tracking enabled
        """
        from datetime import datetime

        # Enable learning tracking
        signal.learning_tracked = True
        signal.learning_started_at = datetime.now()

        # Set agent information
        if generated_by_agent:
            signal.generated_by_agent = generated_by_agent

        # Set market regime information
        if market_regime:
            signal.market_regime = market_regime
            signal.regime_confidence = market_regime.get('confidence', 0.5)
            signal.volatility_environment = market_regime.get('volatility', 'normal')

        # Set pattern information
        if pattern_classification:
            signal.pattern_classification = pattern_classification
            signal.pattern_confidence = pattern_confidence or 0.5

        # Set learning version
        signal.learning_version = 1

        logger.info(f"🧠 Learning tracking enabled for signal {signal.signal_id} by agent {generated_by_agent}")
        return signal
    
    async def get_active_signals(self, signal_pool: SignalPool = None, limit: int = 10) -> List[PlatformSignal]:
        """
        Get active platform signals from database.

        Args:
            signal_pool: Filter by signal pool (fast_mode/full_mode)
            limit: Maximum signals to return

        Returns:
            List of active platform signals
        """
        logger.debug(f"Fetching active signals: pool={signal_pool}, limit={limit}")
        try:
            for source_table in ('enhanced_platform_signals', 'platform_signals'):
                try:
                    if source_table == 'enhanced_platform_signals':
                        logger.debug("Querying enhanced view for active signals")
                    else:
                        logger.debug("Querying base platform_signals table as fallback")

                    result = self._run_active_signal_query(
                        source_table=source_table,
                        signal_pool=signal_pool,
                        limit=limit,
                        order_by_rank_first=False,
                    )

                    if not result.data:
                        now = datetime.now()
                        # Throttle empty-state log to reduce production noise in polling loops.
                        if (
                            self._last_no_signals_log_at is None
                            or (now - self._last_no_signals_log_at).total_seconds() >= 600
                        ):
                            logger.info(f"No active platform signals found for pool: {signal_pool}")
                            self._last_no_signals_log_at = now
                        return []

                    signals = self._convert_db_records_to_signals(result.data, source_table)
                    logger.debug(
                        "Converted %s active signals from %s",
                        len(signals),
                        source_table,
                    )
                    return signals
                except Exception as source_error:
                    if source_table == 'enhanced_platform_signals' and self._is_missing_relation_error(source_error, 'enhanced_platform_signals'):
                        logger.warning("enhanced_platform_signals view missing; falling back to platform_signals")
                        continue
                    raise

            return []

        except Exception as e:
            logger.error(f"Error retrieving active platform signals: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    async def get_closed_signals_in_window(self, limit: int = 50) -> List[PlatformSignal]:
        """
        Get TP2-closed signals whose trade window has not elapsed yet.

        Used by the WebSocket monitor to keep tracking the peak
        (max_profit_reached) after a trade closes at TP2.
        """
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            result = self.db.table('platform_signals').select('*').eq(
                'status', 'hit_target_2'
            ).gt('expires_at', now_iso).order('created_at', desc=True).limit(limit).execute()

            if not result.data:
                return []
            return self._convert_db_records_to_signals(result.data, 'closed_in_window')
        except Exception as e:
            logger.error(f"Error retrieving TP2-closed signals in window: {e}")
            return []

    async def get_active_signals_by_rank(self, opportunity_ranks: List[int], limit: int = 10) -> List[PlatformSignal]:
        """
        Get active platform signals filtered by opportunity ranks.

        Args:
            opportunity_ranks: List of opportunity ranks to filter by (e.g., [4, 5] for fast mode)
            limit: Maximum signals to return

        Returns:
            List of active platform signals with specified ranks
        """
        logger.info(f"🗃️  Fetching signals by rank: {opportunity_ranks}, limit={limit}")
        try:
            if not opportunity_ranks:
                return []

            for source_table in ('enhanced_platform_signals', 'platform_signals'):
                try:
                    if source_table == 'enhanced_platform_signals':
                        logger.info("Querying enhanced view for rank-filtered signals")
                    else:
                        logger.info("Querying base platform_signals table as fallback for rank filtering")

                    result = self._run_active_signal_query(
                        source_table=source_table,
                        opportunity_ranks=opportunity_ranks,
                        limit=limit,
                        order_by_rank_first=True,
                    )

                    if not result.data:
                        logger.warning(f"No active signals found with ranks: {opportunity_ranks}")
                        return []

                    signals = self._convert_db_records_to_signals(result.data, source_table)
                    logger.debug(
                        "Converted %s rank-filtered signals from %s",
                        len(signals),
                        source_table,
                    )
                    return signals
                except Exception as source_error:
                    if source_table == 'enhanced_platform_signals' and self._is_missing_relation_error(source_error, 'enhanced_platform_signals'):
                        logger.warning("enhanced_platform_signals view missing; falling back to platform_signals for rank filtering")
                        continue
                    raise

            return []

        except Exception as e:
            logger.error(f"Error retrieving signals by rank: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    async def get_signals_by_run(self, run_id: str, limit: int = 100) -> List[PlatformSignal]:
        """
        Get platform signals from a specific analysis run.
        
        Args:
            run_id: Analysis run identifier
            limit: Maximum signals to return
            
        Returns:
            List of platform signals from the specified run
        """
        try:
            query = self.db.table('platform_signals').select('*').eq('run_id', run_id).eq('status', 'active')
            result = query.order('created_at', desc=False).limit(limit).execute()
            
            if not result.data:
                logger.info(f"No platform signals found for run: {run_id}")
                return []
            
            signals = []
            for record in result.data:
                signal = self._db_record_to_platform_signal(record)
                signals.append(signal)
            
            logger.info(f"Retrieved {len(signals)} platform signals for run: {run_id}")
            return signals
            
        except Exception as e:
            logger.error(f"Error retrieving platform signals by run {run_id}: {e}")
            return []
    
    async def get_signals_for_user(self, user_mode: str) -> List[Dict[str, Any]]:
        """
        Get appropriate signals for user based on mode.
        
        Args:
            user_mode: 'fast' or 'full'
            
        Returns:
            List of signals formatted for API response
        """
        try:
            # Map user mode to signal pool
            signal_pool = SignalPool.FAST_MODE if user_mode == 'fast' else SignalPool.FULL_MODE
            limit = 2 if user_mode == 'fast' else 5
            
            # Get active signals
            platform_signals = await self.get_active_signals(signal_pool, limit)
            
            if not platform_signals:
                logger.warning(f"No platform signals available for {user_mode} mode")
                return []
            
            # Convert to API format
            signals = []
            for ps in platform_signals:
                signal_data = {
                    'signal_id': ps.signal_id,
                    'symbol': ps.token_symbol,
                    'direction': ps.direction,
                    'timeframe': ps.timeframe,
                    'confidence': ps.confidence,
                    'overall_score': ps.overall_score,
                    'signal_strength': ps.signal_strength,
                    'time_horizon': ps.time_horizon,
                    'entry_price': ps.entry_price,
                    'target_1': ps.target_1,
                    'target_1_probability': ps.target_1_probability,
                    'target_1_hit': ps.target_1_hit,
                    'target_1_hit_price': ps.target_1_hit_price,
                    'target_1_hit_at': ps.target_1_hit_at.isoformat() if ps.target_1_hit_at else None,
                    'target_1_leveraged_pnl_percent': ps.target_1_leveraged_pnl_percent,
                    'target_2': ps.target_2,
                    'target_2_probability': ps.target_2_probability,
                    'stop_loss': ps.stop_loss,
                    'risk_reward_ratio': ps.risk_reward_ratio,
                    'market_conditions': ps.market_conditions,
                    'technical_indicators': ps.technical_indicators,
                    'sentiment_data': ps.sentiment_data,
                    'risk_factors': ps.risk_factors,
                    'opportunity_rank': ps.opportunity_rank,
                    'analysis_timestamp': ps.analysis_timestamp.isoformat(),
                    'expires_at': ps.expires_at.isoformat(),
                    'validity_window_hours': ps.validity_window_hours,
                    'analysis_notes': ps.analysis_notes,
                    'platform_generated': True,
                    'last_updated': ps.analysis_timestamp.isoformat(),
                    
                    # AI Analysis Fields
                    'ai_reasoning': ps.ai_reasoning,
                    'ai_key_factors': ps.ai_key_factors,
                    'ai_risk_assessment': ps.ai_risk_assessment,
                    'ai_confidence_breakdown': ps.ai_confidence_breakdown,
                    
                    # Position Management Fields (from AI analysis)
                    'leverage': ps.leverage,
                    'recommended_leverage': f"{ps.leverage}x",
                    'position_size': ps.position_size,
                    'risk_level': ps.risk_level
                }
                signals.append(signal_data)
            
            logger.info(f"Returning {len(signals)} platform signals for {user_mode} mode")
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals for user mode {user_mode}: {e}")
            return []
    
    async def save_platform_signal(self, signal: PlatformSignal) -> bool:
        """
        Save platform signal to database.
        
        Args:
            signal: PlatformSignal to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            signal_data = {
                'signal_id': signal.signal_id,
                'token_symbol': signal.token_symbol,
                'direction': signal.direction,
                'timeframe': signal.timeframe,
                'logo_url': signal.logo_url,
                'confidence': float(signal.confidence),
                'overall_score': float(signal.overall_score),
                'signal_strength': signal.signal_strength,
                'time_horizon': signal.time_horizon,
                'entry_price': float(signal.entry_price),
                'target_1': float(signal.target_1),
                'target_1_probability': float(signal.target_1_probability),
                'target_2': float(signal.target_2),
                'target_2_probability': float(signal.target_2_probability),
                'stop_loss': float(signal.stop_loss),
                'risk_reward_ratio': float(signal.risk_reward_ratio),
                'market_conditions': signal.market_conditions,
                'technical_indicators': signal.technical_indicators,
                'sentiment_data': signal.sentiment_data,
                'risk_factors': _normalize_text_array(signal.risk_factors),
                'signal_pool': signal.signal_pool.value,
                'opportunity_rank': signal.opportunity_rank,
                'analysis_timestamp': signal.analysis_timestamp.isoformat(),
                'expires_at': signal.expires_at.isoformat(),
                'validity_window_hours': signal.validity_window_hours,
                'status': signal.status,
                'run_id': signal.run_id,  # Analysis run identifier
                'analysis_notes': signal.analysis_notes,
                
                # AI Analysis Fields
                'ai_reasoning': signal.ai_reasoning,
                'ai_key_factors': (
                    _normalize_text_array(signal.ai_key_factors)
                    if signal.ai_key_factors is not None
                    else None
                ),
                'ai_risk_assessment': signal.ai_risk_assessment,
                'ai_confidence_breakdown': signal.ai_confidence_breakdown,
                
                # Position Management Fields
                'leverage': signal.leverage,
                'recommended_leverage': f"{signal.leverage}x",  # Sync with actual leverage value
                'position_size': signal.position_size,
                'risk_level': signal.risk_level,

                # Learning System Integration Fields
                'learning_tracked': signal.learning_tracked,
                'learning_started_at': signal.learning_started_at.isoformat() if signal.learning_started_at else None,
                'market_regime': signal.market_regime or {},
                'regime_confidence': signal.regime_confidence,
                'volatility_environment': signal.volatility_environment,
                'pattern_classification': signal.pattern_classification,
                'pattern_confidence': signal.pattern_confidence,
                'learning_version': signal.learning_version,
                'prompt_version': signal.prompt_version,
                'generated_by_agent': signal.generated_by_agent,
                'risk_adjusted_confidence': signal.risk_adjusted_confidence,
                'learning_insights_id': signal.learning_insights_id,
                'pattern_performance_id': signal.pattern_performance_id,
            }
            
            # STRICT lifecycle prevention: TP milestones do not free the token slot.
            # A thesis remains live through its original horizon so runner tracking,
            # resting entries, and open positions cannot be stacked by a fresh row.

            existing_active = self.db.table('platform_signals').select(
                'signal_id, direction, entry_price, analysis_timestamp, expires_at, status, run_id, confidence'
            ).eq('token_symbol', signal.token_symbol).in_(
                'status', list(THESIS_LIVE_STATUSES)
            ).order('analysis_timestamp', desc=True).execute()

            if existing_active.data:
                live_theses = [
                    existing for existing in existing_active.data
                    if signal_thesis_is_live(existing)
                ]
                same_direction = next(
                    (
                        existing for existing in live_theses
                        if str(existing.get('direction') or '').upper() == signal.direction.upper()
                    ),
                    None,
                )
                if same_direction:
                    logger.warning(
                        f"🚫 Live {signal.direction} thesis {same_direction.get('signal_id')} already owns "
                        f"{signal.token_symbol} through {same_direction.get('expires_at')} - skipping duplicate"
                    )
                    return False

                for existing in live_theses:
                    logger.info(
                        f"🔄 Market direction changed for {signal.token_symbol}: "
                        f"{existing.get('direction')} → {signal.direction} - invalidating old thesis"
                    )
                    self.db.table('platform_signals').update({
                        'status': 'invalidated',
                        'analysis_notes': f'Replaced by opposite {signal.direction} thesis',
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }).eq('signal_id', existing['signal_id']).execute()

            # 2. Double-check: Exact duplicate within same run_id (shouldn't happen)
            if signal.run_id:
                same_run_check = self.db.table('platform_signals').select('signal_id').eq('token_symbol', signal.token_symbol).eq('direction', signal.direction).eq('run_id', signal.run_id).eq('status', 'active').execute()

                if same_run_check.data:
                    logger.warning(f"🚫 Exact duplicate detected in run {signal.run_id} for {signal.token_symbol} {signal.direction} - this shouldn't happen!")
                    return False
            
            # Upsert signal (update if exists, insert if new)
            result = self.db.table('platform_signals').upsert(signal_data, on_conflict='signal_id').execute()
            
            if result.data:
                logger.debug(f"💾 Saved platform signal {signal.token_symbol} {signal.direction} (confidence: {signal.confidence:.2f})")
                # Performance tracking record will be created automatically by database trigger
                return True
            else:
                logger.error(f"❌ Failed to save platform signal {signal.signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving platform signal {signal.signal_id}: {e}")
            return False

    
    async def track_user_signal_interaction(self, user_id: str, signal_id: str, delivery_mode: str, followed: bool = False) -> bool:
        """
        Track user interaction with platform signal.
        
        Args:
            user_id: User identifier (wallet address or UUID)
            signal_id: Platform signal ID
            delivery_mode: 'fast_mode' or 'full_mode'
            followed: Whether user followed the signal
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get platform signal
            signal_result = self.db.table('platform_signals').select('id').eq('signal_id', signal_id).single().execute()
            
            if not signal_result.data:
                logger.error(f"Platform signal not found: {signal_id}")
                return False
            
            platform_signal_id = signal_result.data['id']
            
            # Resolve user_id to UUID if possible; fall back to raw identifier.
            # user_platform_signal_tracking.user_id is TEXT after migration, so
            # wallet addresses (0x...) are accepted directly.
            resolved = await self._resolve_user_id_to_uuid(user_id)
            actual_user_id = resolved or user_id
            if not resolved:
                logger.debug(f"user_id not resolved to UUID, storing raw: {user_id}")
            
            # Create or update tracking record
            tracking_data = {
                'user_id': actual_user_id,
                'platform_signal_id': platform_signal_id,
                'signal_id': signal_id,
                'viewed_at': datetime.now().isoformat(),
                'followed': followed,
                'delivery_mode': delivery_mode
            }
            
            if followed:
                tracking_data['followed_at'] = datetime.now().isoformat()
            
            result = self.db.table('user_platform_signal_tracking').upsert(
                tracking_data, 
                on_conflict='user_id,platform_signal_id'
            ).execute()
            
            if result.data:
                logger.debug(f"Tracked user {actual_user_id} interaction with signal {signal_id}")
                return True
            else:
                logger.error(f"Failed to track user interaction: {actual_user_id}, {signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error tracking user signal interaction: {e}")
            return False
    
    async def _resolve_user_id_to_uuid(self, user_id: str) -> Optional[str]:
        """
        Resolve user identifier (wallet address or UUID) to actual UUID.
        
        Args:
            user_id: User identifier (can be wallet address or UUID)
            
        Returns:
            UUID string if found, None otherwise
        """
        try:
            # If it's already a UUID format, return as-is
            if not user_id.startswith('0x') and len(user_id) == 36 and user_id.count('-') == 4:
                return user_id
            
            # It's likely a wallet address, resolve to UUID
            if user_id.startswith('0x'):
                # Try external_wallet first
                response = self.db.table('users').select('id').eq('external_wallet', user_id).execute()
                if response.data:
                    return response.data[0]['id']
                
                # Try lowercase external_wallet
                response = self.db.table('users').select('id').eq('external_wallet', user_id.lower()).execute()
                if response.data:
                    return response.data[0]['id']
                
                # Try platform_wallet
                response = self.db.table('users').select('id').eq('platform_wallet', user_id).execute()
                if response.data:
                    return response.data[0]['id']
                
                # Try lowercase platform_wallet
                response = self.db.table('users').select('id').eq('platform_wallet', user_id.lower()).execute()
                if response.data:
                    return response.data[0]['id']
            
            logger.warning(f"Could not resolve user_id to UUID: {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving user_id to UUID: {e}")
            return None
    
    async def cleanup_duplicate_signals(self) -> int:
        """
        Clean up ALL duplicate signals enforcing ONE active signal per token.

        Strategy: For each token, keep only the most recent signal and expire all others.

        Returns:
            Number of duplicate signals expired
        """
        try:
            expired_count = 0

            # Get all active signals grouped by token
            result = self.db.table('platform_signals').select(
                'signal_id, token_symbol, direction, analysis_timestamp, run_id, confidence, market_conditions'
            ).eq('status', 'active').order('analysis_timestamp', desc=True).execute()

            if not result.data:
                return expired_count

            # Group signals by token symbol
            from collections import defaultdict
            signals_by_token = defaultdict(list)

            for record in result.data:
                signals_by_token[record['token_symbol']].append(record)

            # For each token with multiple active signals, keep newest and expire others
            for token, signals in signals_by_token.items():
                if len(signals) > 1:
                    # Keep the most recent (first in desc order), expire others
                    most_recent = signals[0]
                    duplicates = signals[1:]

                    logger.info(f"🧹 {token}: Found {len(signals)} active signals - keeping most recent ({most_recent['direction']}, {most_recent['analysis_timestamp'][:16]})")

                    for dup in duplicates:
                        # Expire duplicate signal
                        terminal_at = datetime.now(timezone.utc)
                        conditions, _ = self._close_terminal_reanalysis_context(
                            dup.get('market_conditions'),
                            terminal_at,
                        )
                        update_result = (
                            self.db.table('platform_signals')
                            .update({
                                'status': 'expired',
                                'market_conditions': conditions,
                                'updated_at': terminal_at.isoformat(),
                            })
                            .eq('signal_id', dup['signal_id'])
                            .eq('status', 'active')
                            .execute()
                        )

                        if update_result.data:
                            expired_count += 1
                            dup_time = parse_timestamp(dup['analysis_timestamp'])
                            age = terminal_at - dup_time
                            logger.info(f"   ⏰ Expired duplicate: {dup['direction']} signal (age: {age}, ID: {dup['signal_id'][:8]}...)")

            if expired_count > 0:
                logger.info(f"🧹 Duplicate cleanup complete: {expired_count} duplicate signals expired")
            else:
                logger.info("✅ No duplicate signals found")

            return expired_count

        except Exception as e:
            logger.error(f"Error cleaning up duplicate signals: {e}")
            return 0
    
    def _milestone_exit_reason(self, signal_id: str) -> Optional[str]:
        """Return 'TARGET_2'/'TARGET_1' if the signal reached that milestone, else None."""
        try:
            sig_result = self.db.table('platform_signals').select(
                'market_conditions'
            ).eq('signal_id', signal_id).limit(1).execute()
            mc = (sig_result.data[0].get('market_conditions') or {}) if sig_result.data else {}
            if mc.get('target_2_hit'):
                return 'TARGET_2'

            perf_result = self.db.table('platform_signal_performance_tracking').select(
                'target_1_hit'
            ).eq('signal_id', signal_id).limit(1).execute()
            if perf_result.data and perf_result.data[0].get('target_1_hit'):
                return 'TARGET_1'
            return None
        except Exception as e:
            logger.warning(f"Milestone lookup failed for {signal_id}: {e}")
            return None

    async def update_signal_exit(self, signal_id: str, exit_price: float, exit_reason: str) -> bool:
        """
        Update signal with exit information.
        
        Args:
            signal_id: Platform signal ID
            exit_price: Price at which signal was exited
            exit_reason: Reason for exit ('TARGET_1', 'TARGET_2', 'STOP_LOSS', 'EXPIRED', 'MANUAL')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # A later runner reversal/expiry must not replace an already-proven
            # target milestone with break-even or the original stop. The signal
            # learning ledger resolves at the highest protected target reached;
            # the database trigger then calculates PnL from that target price.
            if exit_reason in {'EXPIRED', 'STOP_LOSS'}:
                milestone_exit = self._milestone_exit_reason(signal_id)
                if milestone_exit:
                    logger.info(
                        f"🎯 Signal {signal_id} reached {milestone_exit} before {exit_reason} "
                        "— preserving the protected target outcome"
                    )
                    exit_reason = milestone_exit

            # Map exit reason to appropriate status
            status_mapping = {
                'TARGET_1': 'hit_target_1',
                'TARGET_2': 'hit_target_2',
                'STOP_LOSS': 'hit_stop_loss',
                'EXPIRED': 'expired',
                'MANUAL': 'cancelled'
            }
            
            # Get the appropriate status, default to 'expired' if unknown
            status = status_mapping.get(exit_reason, 'expired')

            # Validate status to prevent constraint violations
            valid_statuses = {'active', 'hit_target_1', 'hit_target_2', 'hit_stop_loss', 'expired', 'cancelled', 'invalidated'}
            if status not in valid_statuses:
                logger.warning(f"Invalid status '{status}' for exit reason '{exit_reason}', using 'expired' instead")
                status = 'expired'

            # Persist terminal timing next to the immutable generation context.
            terminal_at = datetime.now(timezone.utc)
            terminal_context: Dict[str, Any] = {}
            try:
                source_rows = (
                    self.db.table("platform_signals")
                    .select("analysis_timestamp,market_conditions")
                    .eq("signal_id", signal_id)
                    .limit(1)
                    .execute()
                ).data or []
                if source_rows and isinstance(source_rows[0], dict):
                    source = source_rows[0]
                    terminal_context = dict(source.get("market_conditions") or {})
                    analyzed_at = parse_timestamp(source.get("analysis_timestamp"))
                    terminal_context.update({
                        "terminal_event": exit_reason,
                        "terminal_price": float(exit_price or 0.0),
                        "terminal_at": terminal_at.isoformat(),
                        "time_to_terminal_minutes": max(
                            0.0,
                            (terminal_at - analyzed_at).total_seconds() / 60.0,
                        ),
                    })
                    if exit_reason in {"TARGET_1", "TARGET_2"}:
                        terminal_context["time_to_target_minutes"] = terminal_context[
                            "time_to_terminal_minutes"
                        ]
                    elif exit_reason == "STOP_LOSS":
                        terminal_context["time_to_stop_minutes"] = terminal_context[
                            "time_to_terminal_minutes"
                        ]
            except Exception as timing_exc:
                logger.debug("Could not calculate terminal timing for %s: %s", signal_id, timing_exc)

            terminal_context, _ = self._close_terminal_reanalysis_context(
                terminal_context,
                terminal_at,
            )

            # Update platform signal status
            signal_update = {
                'status': status,
                'updated_at': terminal_at.isoformat(),
            }
            if terminal_context:
                signal_update["market_conditions"] = terminal_context

            logger.info(f"🔄 Updating signal {signal_id} status to '{status}' for exit reason '{exit_reason}'")
            result = self.db.table('platform_signals').update(signal_update).eq('signal_id', signal_id).execute()
            
            if result.data:
                logger.info(f"✅ Updated signal {signal_id} exit: {exit_reason} at {exit_price}")
                self._performance_metrics_cache.clear()

                # User tracking will be updated automatically by database trigger

                return True
            else:
                logger.warning(f"No signal found to update: {signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating signal exit for {signal_id}: {e}")
            return False

    async def record_signal_target_1_milestone(
        self,
        signal_id: str,
        price: float,
        pnl_percentage: Optional[float] = None,
    ) -> bool:
        """
        Record that TP1 was reached without closing the signal.

        TP1 is a milestone for marketing/UX and optional partial execution. The
        signal remains active so monitoring can continue toward TP2, stop loss,
        or expiry. Closed-trade stats should not count this as a terminal win.
        """
        try:
            signal_result = self.db.table('platform_signals').select(
                'signal_id, direction, entry_price, leverage'
            ).eq('signal_id', signal_id).limit(1).execute()
            if not signal_result.data:
                logger.warning(f"Signal {signal_id} not found for TP1 milestone")
                return False

            signal = signal_result.data[0]
            entry_price = float(signal.get('entry_price') or 0)
            if entry_price <= 0:
                return False

            milestone_price = float(price or 0)
            if milestone_price <= 0:
                return False

            direction = str(signal.get('direction') or '').upper()
            leverage = max(float(signal.get('leverage') or 1), 1.0)
            if pnl_percentage is None:
                if direction == 'LONG':
                    pnl_percentage = ((milestone_price - entry_price) / entry_price) * 100
                else:
                    pnl_percentage = ((entry_price - milestone_price) / entry_price) * 100

            leveraged_pnl = float(pnl_percentage) * leverage

            existing_result = self.db.table('platform_signal_performance_tracking').select(
                'signal_id, target_1_hit, max_profit_reached, max_loss_reached'
            ).eq('signal_id', signal_id).limit(1).execute()
            existing = existing_result.data[0] if existing_result.data else None
            if existing and existing.get('target_1_hit'):
                return False

            now_iso = datetime.now(timezone.utc).isoformat()
            existing_max_profit = float((existing or {}).get('max_profit_reached') or 0)
            existing_max_loss = float((existing or {}).get('max_loss_reached') or 0)
            update_data = {
                'signal_id': signal_id,
                'outcome': 'active',
                'target_1_hit': True,
                'target_1_hit_price': milestone_price,
                'target_1_hit_timestamp': now_iso,
                'target_1_pnl_percent': round(float(pnl_percentage), 4),
                'target_1_leveraged_pnl_percent': round(leveraged_pnl, 4),
                'actual_pnl_percent': round(float(pnl_percentage), 4),
                'leveraged_pnl_percent': round(leveraged_pnl, 4),
                'max_profit_reached': max(existing_max_profit, round(leveraged_pnl, 4), 0),
                'max_loss_reached': min(existing_max_loss, round(leveraged_pnl, 4), 0),
                'last_updated': now_iso,
            }

            self.db.table('platform_signal_performance_tracking').upsert(
                update_data,
                on_conflict='signal_id',
            ).execute()

            self._performance_metrics_cache.clear()
            logger.info(
                "🎯 TP1 milestone recorded for %s at %s (%+.2f%% leveraged)",
                signal_id,
                milestone_price,
                leveraged_pnl,
            )
            return True
        except Exception as e:
            logger.error(f"Error recording TP1 milestone for {signal_id}: {e}")
            return False

    async def _update_user_tracking_on_exit(self, signal_id: str, exit_reason: str, exit_price: float):
        """Update user tracking records when a signal exits for performance calculation."""
        try:
            # Get the signal details for P&L calculation
            signal_result = self.db.table('platform_signals').select('*').eq('signal_id', signal_id).execute()
            if not signal_result.data:
                logger.warning(f"Signal {signal_id} not found for tracking update")
                return

            signal = signal_result.data[0]
            entry_price = float(signal['entry_price'])
            direction = signal['direction']

            # Calculate P&L percentage
            if direction.upper() == 'LONG':
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_percentage = ((entry_price - exit_price) / entry_price) * 100

            # Determine outcome
            if exit_reason in ['TARGET_1', 'TARGET_2']:
                user_outcome = 'win'
            elif exit_reason == 'STOP_LOSS':
                user_outcome = 'loss'
            else:  # EXPIRED, MANUAL, etc.
                user_outcome = 'scratch' if abs(pnl_percentage) < 1.0 else ('win' if pnl_percentage > 0 else 'loss')

            # Update all user tracking records for this signal
            tracking_update = {
                'user_outcome': user_outcome,
                'profit_loss_percentage': round(pnl_percentage, 4),
                'user_exit_timestamp': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            update_result = self.db.table('user_platform_signal_tracking').update(tracking_update).eq('signal_id', signal_id).execute()

            if update_result.data:
                logger.info(f"✅ Updated {len(update_result.data)} user tracking records for signal {signal_id}: {user_outcome} ({pnl_percentage:+.2f}%)")
            else:
                logger.debug(f"No user tracking records found for signal {signal_id}")

        except Exception as e:
            logger.error(f"Error updating user tracking for signal {signal_id}: {e}")

    def _metric_float(self, row: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Convert database numeric values to floats without leaking endpoint errors."""
        try:
            value = row.get(key)
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _metric_optional_float(self, row: Dict[str, Any], key: str) -> Optional[float]:
        try:
            value = row.get(key)
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _summary_row_to_metrics(self, row: Dict[str, Any], days_analyzed: int) -> Dict[str, Any]:
        """Map a platform_metrics_summary row to the public metrics payload."""
        return {
            'total_signals': int(self._metric_float(row, 'total_signals')),
            'win_rate': round(self._metric_float(row, 'win_rate'), 2),
            'average_pnl': round(self._metric_float(row, 'average_pnl'), 2),
            'total_return_percentage': round(self._metric_float(row, 'total_return_percentage'), 2),
            'profit_factor': round(self._metric_float(row, 'profit_factor'), 2),
            'active_signals': int(self._metric_float(row, 'active_signals')),
            'long_signals': int(self._metric_float(row, 'long_signals')),
            'short_signals': int(self._metric_float(row, 'short_signals')),
            'winning_signals': int(self._metric_float(row, 'winning_signals')),
            'losing_signals': int(self._metric_float(row, 'losing_signals')),
            'neutral_signals': int(self._metric_float(row, 'neutral_signals')),
            'informative_completed_trades': int(self._metric_float(row, 'informative_completed_trades')),
            'days_analyzed': int(self._metric_float(row, 'days_analyzed', days_analyzed)),
            'best_trade_percentage': self._metric_optional_float(row, 'best_trade_percentage'),
            'worst_trade_percentage': self._metric_optional_float(row, 'worst_trade_percentage'),
            'average_completion_time_hours': self._metric_float(row, 'average_completion_time_hours'),
            'min_completion_time_hours': self._metric_float(row, 'min_completion_time_hours'),
            'max_completion_time_hours': self._metric_float(row, 'max_completion_time_hours'),
            'total_completed_trades': int(self._metric_float(row, 'total_completed_trades')),
            'source': 'platform_metrics_summary',
            'period_key': row.get('period_key'),
            'summary_updated_at': row.get('updated_at'),
        }

    def _get_best_peak_trade_percentage(
        self,
        period_start: str,
        period_end: str,
    ) -> Optional[float]:
        """Return the largest completed winning signal peak for the period.

        The scanner defines a signal's Final PnL as its maximum favorable PnL,
        so the Best tile must use max_profit_reached as well.  The precomputed
        summary still stores realized/exit PnL and is retained for every other
        aggregate metric.
        """
        try:
            result = (
                self.db.table('platform_signal_performance_tracking')
                .select('max_profit_reached')
                .eq('outcome', 'win')
                .gte('created_at', period_start)
                .lt('created_at', period_end)
                .gt('max_profit_reached', 0)
                .order('max_profit_reached', desc=True)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            if not rows:
                return None
            peak = self._metric_optional_float(rows[0], 'max_profit_reached')
            return peak if peak is not None and peak > 0 else None
        except Exception as peak_error:
            logger.debug("Best peak PnL lookup unavailable: %s", peak_error)
            return None

    def _apply_peak_best_metric(
        self,
        metrics: Dict[str, Any],
        period_start: str,
        period_end: str,
    ) -> Dict[str, Any]:
        """Align Best with the peak-based Final PnL definition."""
        peak = self._get_best_peak_trade_percentage(period_start, period_end)
        if peak is None:
            return metrics

        response = dict(metrics)
        existing = self._metric_optional_float(response, 'best_trade_percentage')
        response['best_trade_percentage'] = round(
            max(peak, existing) if existing is not None else peak,
            4,
        )
        response['best_trade_source'] = 'max_profit_reached'
        return response

    def _log_summary_metrics_unavailable(self, error: Exception):
        now = datetime.now(timezone.utc)
        if (
            self._summary_metrics_unavailable_logged_at is None
            or (now - self._summary_metrics_unavailable_logged_at).total_seconds() > 300
        ):
            logger.info(
                "Platform metrics summary fast path unavailable; falling back to live aggregation: %s",
                error,
            )
            self._summary_metrics_unavailable_logged_at = now

    def _summary_row_is_fresh(self, row: Dict[str, Any], now: datetime, monthly: bool) -> bool:
        if monthly:
            return True

        updated_at = row.get('updated_at')
        if not updated_at:
            return False

        try:
            if isinstance(updated_at, datetime):
                updated_dt = updated_at
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            else:
                updated_dt = parse_timestamp(str(updated_at))
            return (now - updated_dt.astimezone(timezone.utc)).total_seconds() <= 60
        except Exception:
            return False

    def _get_precomputed_performance_metrics(
        self,
        *,
        now: datetime,
        monthly: bool,
        days: int,
        cutoff_date: str,
        days_analyzed: int,
    ) -> Optional[Dict[str, Any]]:
        """Read the precomputed metrics row, refreshing via DB RPC when needed."""
        if monthly:
            period_type = 'monthly'
            period_key = f"{now.year:04d}-{now.month:02d}"
        else:
            period_type = 'rolling_days'
            period_key = f"{days}:{cutoff_date[:10]}"

        try:
            summary_result = (
                self.db.table('platform_metrics_summary')
                .select('*')
                .eq('period_type', period_type)
                .eq('period_key', period_key)
                .limit(1)
                .execute()
            )
            summary_rows = summary_result.data or []
            if summary_rows and self._summary_row_is_fresh(summary_rows[0], now, monthly):
                metrics = self._summary_row_to_metrics(summary_rows[0], days_analyzed)
                return self._apply_peak_best_metric(
                    metrics,
                    cutoff_date,
                    now.isoformat(),
                )

            rpc_result = self.db.rpc(
                'refresh_platform_metrics_summary',
                {
                    'p_period_type': period_type,
                    'p_anchor': now.isoformat(),
                    'p_days': days,
                },
            ).execute()
            rpc_data = rpc_result.data or []
            if isinstance(rpc_data, list):
                row = rpc_data[0] if rpc_data else None
            else:
                row = rpc_data
            if row:
                metrics = self._summary_row_to_metrics(row, days_analyzed)
                return self._apply_peak_best_metric(
                    metrics,
                    cutoff_date,
                    now.isoformat(),
                )
        except Exception as summary_error:
            self._log_summary_metrics_unavailable(summary_error)

        return None

    async def get_performance_metrics(self, days: int = 30, monthly: bool = False) -> Dict[str, Any]:
        """
        Get platform-wide performance metrics for the specified time period.

        Args:
            days: Number of days to look back for performance metrics
            monthly: If True, use current month start instead of days parameter

        Returns:
            Dictionary with performance metrics for the specified time period
        """
        try:
            now = datetime.now(timezone.utc)

            if monthly:
                # Use current month start for monthly performance
                current_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
                cutoff_date = current_month_start.isoformat()
                period_description = f"current month ({now.month}/{now.year})"
                days_analyzed = (now - current_month_start).days + 1
            else:
                # Use days parameter for rolling time window
                cutoff_date = (now - timedelta(days=days)).isoformat()
                period_description = f"last {days} days"
                days_analyzed = days

            cache_key = f"monthly:{now.year}:{now.month}" if monthly else f"days:{days}:{cutoff_date[:10]}"
            cached = self._performance_metrics_cache.get(cache_key)
            if cached:
                cached_at, cached_metrics = cached
                if (now - cached_at).total_seconds() < self._performance_metrics_cache_ttl_seconds:
                    response = dict(cached_metrics)
                    response['cache_hit'] = True
                    return response

            def _cache_response(metrics: Dict[str, Any]) -> Dict[str, Any]:
                response = dict(metrics)
                response['cache_hit'] = False
                response['cache_ttl_seconds'] = self._performance_metrics_cache_ttl_seconds
                response['generated_at'] = now.isoformat()
                self._performance_metrics_cache[cache_key] = (now, dict(response))
                return response

            precomputed_metrics = self._get_precomputed_performance_metrics(
                now=now,
                monthly=monthly,
                days=days,
                cutoff_date=cutoff_date,
                days_analyzed=days_analyzed,
            )
            if precomputed_metrics is not None:
                return _cache_response(precomputed_metrics)

            # Get the signals and fields needed for metrics in one query.
            signals_result = self.db.table('platform_signals').select("""
                signal_id,
                analysis_timestamp,
                leverage,
                status,
                direction,
                entry_price,
                target_1,
                target_2,
                stop_loss,
                created_at
            """).gte('created_at', cutoff_date).execute()

            filtered_signals = signals_result.data or []
            all_signals = filtered_signals

            # Count active signals
            active_signals = len([s for s in all_signals if str(s.get('status') or '').lower() == 'active'])
            long_signals = len([s for s in all_signals if str(s.get('direction') or '').upper() == 'LONG'])
            short_signals = len([s for s in all_signals if str(s.get('direction') or '').upper() == 'SHORT'])

            # If no signals exist, return empty stats
            if not all_signals:
                return _cache_response({
                    'total_signals': 0,
                    'win_rate': 0.0,
                    'average_pnl': 0.0,
                    'total_return_percentage': 0.0,
                    'profit_factor': 0.0,
                    'active_signals': 0,
                    'long_signals': 0,
                    'short_signals': 0,
                    'winning_signals': 0,
                    'losing_signals': 0,
                    'neutral_signals': 0,
                    'informative_completed_trades': 0,
                    'days_analyzed': days_analyzed,
                    'best_trade_percentage': 0.0,
                    'worst_trade_percentage': 0.0,
                    'average_completion_time_hours': 0.0,
                    'min_completion_time_hours': 0.0,
                    'max_completion_time_hours': 0.0,
                    'total_completed_trades': 0
                })

            # Get performance data for these signals
            signal_ids = [s.get('signal_id') for s in filtered_signals if s.get('signal_id')]

            performance_data = {}
            if signal_ids:
                # Fetch in batches to avoid query limits
                perf_result = self.db.table('platform_signal_performance_tracking').select("""
                    signal_id,
                    outcome,
                    actual_pnl_percent,
                    leveraged_pnl_percent,
                    max_profit_reached,
                    max_loss_reached
                """).in_('signal_id', signal_ids).execute()

                performance_data = {p['signal_id']: p for p in (perf_result.data or [])}

            # Calculate performance metrics
            wins = 0
            losses = 0
            neutral = 0
            completed_signals = []
            pnl_values = []
            winning_peak_values = []
            terminal_statuses = {
                'hit_target_1',
                'hit_target_2',
                'hit_stop_loss',
                'expired',
            }

            def _to_float(value, default=0.0) -> float:
                try:
                    if value is None:
                        return default
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _fallback_pnl_from_status(signal: Dict[str, Any], status: str) -> float:
                entry = _to_float(signal.get('entry_price'))
                if entry <= 0:
                    return 0.0

                direction = str(signal.get('direction') or '').upper()
                leverage = max(_to_float(signal.get('leverage'), 1.0), 1.0)
                exit_price = None
                if status == 'hit_target_1':
                    exit_price = _to_float(signal.get('target_1'))
                elif status == 'hit_target_2':
                    exit_price = _to_float(signal.get('target_2') or signal.get('target_1'))
                elif status == 'hit_stop_loss':
                    exit_price = _to_float(signal.get('stop_loss'))

                if not exit_price or exit_price <= 0:
                    return 0.0

                if direction == 'SHORT':
                    return ((entry - exit_price) / entry) * 100.0 * leverage
                return ((exit_price - entry) / entry) * 100.0 * leverage

            for signal in filtered_signals:
                status = str(signal.get('status') or 'unknown').lower()
                signal_id = signal.get('signal_id')
                if not signal_id:
                    continue
                perf = performance_data.get(signal_id, {})
                perf_outcome = str(perf.get('outcome') or '').lower()

                if status not in terminal_statuses:
                    continue

                pnl_candidate = perf.get('leveraged_pnl_percent')
                if pnl_candidate is None:
                    pnl_candidate = perf.get('actual_pnl_percent')
                pnl_numeric = _to_float(pnl_candidate, None)
                if pnl_numeric is None:
                    pnl_numeric = _fallback_pnl_from_status(signal, status)

                if status in ['hit_target_1', 'hit_target_2']:
                    outcome = 'win'
                elif status == 'hit_stop_loss':
                    outcome = 'loss'
                elif perf_outcome in ['win', 'loss'] and abs(pnl_numeric) > 1e-9:
                    outcome = perf_outcome
                elif pnl_numeric > 0:
                    outcome = 'win'
                elif pnl_numeric < 0:
                    outcome = 'loss'
                else:
                    outcome = 'neutral'

                if outcome == 'loss' and pnl_numeric > 0:
                    pnl_numeric = -abs(pnl_numeric)
                elif outcome == 'win' and pnl_numeric < 0:
                    pnl_numeric = abs(pnl_numeric)

                if outcome == 'win':
                    wins += 1
                    peak_candidate = _to_float(perf.get('max_profit_reached'), None)
                    winning_peak_values.append(
                        max(
                            pnl_numeric,
                            peak_candidate if peak_candidate is not None and peak_candidate > 0 else pnl_numeric,
                        )
                    )
                elif outcome == 'loss':
                    losses += 1
                else:
                    neutral += 1

                completed_signals.append({**signal, 'outcome': outcome})
                pnl_values.append(pnl_numeric)

            # If no completed signals, calculate basic stats from all signals
            if not completed_signals:
                return _cache_response({
                    'total_signals': len(all_signals),
                    'win_rate': 0.0,
                    'average_pnl': 0.0,
                    'total_return_percentage': 0.0,
                    'profit_factor': 0.0,
                    'active_signals': active_signals,
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'winning_signals': 0,
                    'losing_signals': 0,
                    'neutral_signals': 0,
                    'informative_completed_trades': 0,
                    'days_analyzed': days_analyzed,
                    'best_trade_percentage': 0.0,
                    'worst_trade_percentage': 0.0,
                    'average_completion_time_hours': 0.0,
                    'min_completion_time_hours': 0.0,
                    'max_completion_time_hours': 0.0,
                    'total_completed_trades': 0
                })

            # Calculate metrics from completed signals
            informative_completed = wins + losses
            win_rate = (wins / informative_completed * 100) if informative_completed else 0

            average_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0

            # Profit factor
            total_wins = sum([pnl for pnl in pnl_values if pnl > 0])
            total_losses = abs(sum([pnl for pnl in pnl_values if pnl < 0]))
            profit_factor = total_wins / total_losses if total_losses > 0 else (total_wins if total_wins > 0 else 0)

            # Best and worst are directional display metrics, not just max/min
            # aliases. Avoid showing a positive value in the "Worst" slot when
            # the period has wins but no losses.
            losing_pnls = [pnl for pnl in pnl_values if pnl < 0]
            best_trade = max(winning_peak_values) if winning_peak_values else None
            worst_trade = min(losing_pnls) if losing_pnls else None

            # Calculate days analyzed
            if monthly:
                current_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
                days_analyzed = (now - current_month_start).days + 1
            else:
                days_analyzed = days

            return _cache_response({
                'total_signals': len(all_signals),  # Total platform signals generated
                'win_rate': round(win_rate, 2),
                'average_pnl': round(average_pnl, 2),
                'total_return_percentage': round(sum(pnl_values), 2) if pnl_values else 0.0,
                'profit_factor': round(profit_factor, 2),
                'active_signals': active_signals,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'winning_signals': wins,
                'losing_signals': losses,
                'neutral_signals': neutral,
                'informative_completed_trades': informative_completed,
                'days_analyzed': days_analyzed,
                'best_trade_percentage': round(best_trade, 2) if best_trade is not None else None,
                'worst_trade_percentage': round(worst_trade, 2) if worst_trade is not None else None,
                'average_completion_time_hours': 0.0,
                'min_completion_time_hours': 0.0,
                'max_completion_time_hours': 0.0,
                'total_completed_trades': len(completed_signals)
            })
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_signals': 0,
                'win_rate': 0,
                'average_pnl': 0,
                'total_return_percentage': 0.0,
                'profit_factor': 0,
                'active_signals': 0,
                'long_signals': 0,
                'short_signals': 0,
                'winning_signals': 0,
                'losing_signals': 0,
                'neutral_signals': 0,
                'informative_completed_trades': 0,
                'days_analyzed': days,
                'best_trade_percentage': 0.0,
                'worst_trade_percentage': 0.0,
                'average_completion_time_hours': 0.0,
                'min_completion_time_hours': 0.0,
                'max_completion_time_hours': 0.0,
                'total_completed_trades': 0,
                'error': str(e)
            }
    
    async def expire_old_signals(self) -> int:
        """
        Mark expired signals as expired.

        Signals that already reached TP1/TP2 stay active while monitoring runs
        out the trade window; at expiry they finalize as hit_target_1 /
        hit_target_2 wins (the exit trigger fills in exit price/PnL from the
        target price), never as neutral expiries.

        Returns:
            Number of signals expired
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()

            candidates = self.db.table('platform_signals').select(
                'signal_id, market_conditions'
            ).lt('expires_at', current_time).eq('status', 'active').execute()
            candidate_rows = candidates.data or []
            candidate_ids = [row['signal_id'] for row in candidate_rows]

            if candidate_ids:
                tp2_ids = [
                    row['signal_id'] for row in candidate_rows
                    if (row.get('market_conditions') or {}).get('target_2_hit')
                ]

                tp1_result = self.db.table('platform_signal_performance_tracking').select(
                    'signal_id'
                ).in_('signal_id', candidate_ids).eq('target_1_hit', True).execute()
                tp1_ids = [
                    row['signal_id'] for row in (tp1_result.data or [])
                    if row['signal_id'] not in set(tp2_ids)
                ]

                tp2_id_set = set(tp2_ids)
                tp1_id_set = set(tp1_ids)
                finalized_tp2 = 0
                finalized_tp1 = 0
                expired_count = 0

                for row in candidate_rows:
                    signal_id = row['signal_id']
                    if signal_id in tp2_id_set:
                        terminal_status = 'hit_target_2'
                    elif signal_id in tp1_id_set:
                        terminal_status = 'hit_target_1'
                    else:
                        terminal_status = 'expired'

                    conditions, _ = self._close_terminal_reanalysis_context(
                        row.get('market_conditions'),
                        current_time,
                    )
                    update_result = (
                        self.db.table('platform_signals')
                        .update({
                            'status': terminal_status,
                            'market_conditions': conditions,
                            'updated_at': current_time,
                        })
                        .eq('signal_id', signal_id)
                        .eq('status', 'active')
                        .execute()
                    )
                    if not update_result.data:
                        continue
                    if terminal_status == 'hit_target_2':
                        finalized_tp2 += 1
                    elif terminal_status == 'hit_target_1':
                        finalized_tp1 += 1
                    else:
                        expired_count += 1

                if finalized_tp2:
                    logger.info(
                        f"🏆 Finalized {finalized_tp2} TP2-hit signals as hit_target_2 at expiry"
                    )
                if finalized_tp1:
                    logger.info(
                        f"🎯 Finalized {finalized_tp1} TP1-hit signals as hit_target_1 at expiry"
                    )
                if finalized_tp2 or finalized_tp1 or expired_count:
                    self._performance_metrics_cache.clear()
            else:
                expired_count = 0

            if expired_count > 0:
                logger.info(f"Expired {expired_count} platform signals")

            return expired_count

        except Exception as e:
            logger.error(f"Error expiring old signals: {e}")
            return 0
    
    
    def _db_record_to_platform_signal(self, record: Dict[str, Any]) -> PlatformSignal:
        """Convert database record to PlatformSignal object."""
        return PlatformSignal(
            signal_id=record['signal_id'],
            token_symbol=record['token_symbol'],
            direction=record['direction'],
            timeframe=record['timeframe'],
            confidence=float(record['confidence']),
            overall_score=float(record['overall_score']),
            signal_strength=record['signal_strength'],
            time_horizon=record['time_horizon'],
            entry_price=float(record['entry_price']),
            target_1=float(record['target_1']),
            target_1_probability=float(record['target_1_probability']),
            target_2=float(record['target_2']),
            target_2_probability=float(record['target_2_probability']),
            stop_loss=float(record['stop_loss']),
            risk_reward_ratio=float(record['risk_reward_ratio']),
            market_conditions=record['market_conditions'] or {},
            technical_indicators=record['technical_indicators'] or {},
            sentiment_data=record['sentiment_data'] or {},
            risk_factors=_normalize_text_array(record.get('risk_factors')),
            signal_pool=SignalPool(record['signal_pool']),
            opportunity_rank=record['opportunity_rank'],
            analysis_timestamp=parse_timestamp(record['analysis_timestamp']),
            expires_at=parse_timestamp(record['expires_at']),
            validity_window_hours=record['validity_window_hours'],
            status=record['status'],
            run_id=record.get('run_id'),
            analysis_notes=record.get('analysis_notes'),
            
            # AI Analysis Fields
            ai_reasoning=record.get('ai_reasoning'),
            ai_key_factors=_normalize_text_array(record.get('ai_key_factors')),
            ai_risk_assessment=record.get('ai_risk_assessment'),
            ai_confidence_breakdown=record.get('ai_confidence_breakdown', {}),
            logo_url=record.get('logo_url'),
            target_1_hit=bool(record.get('target_1_hit', False)),
            target_1_hit_price=float(record['target_1_hit_price']) if record.get('target_1_hit_price') is not None else None,
            target_1_hit_at=parse_timestamp(record['target_1_hit_timestamp']) if record.get('target_1_hit_timestamp') else None,
            target_1_leveraged_pnl_percent=(
                float(record['target_1_leveraged_pnl_percent'])
                if record.get('target_1_leveraged_pnl_percent') is not None
                else None
            ),
            
            # Position Management Fields (from AI analysis with fallbacks)
            leverage=record.get('leverage', 5),
            position_size=record.get('position_size', 10.0),
            risk_level=record.get('risk_level', 'MEDIUM'),

            # Learning System Integration Fields
            learning_tracked=record.get('learning_tracked', False),
            learning_started_at=parse_timestamp(record['learning_started_at']) if record.get('learning_started_at') else None,
            learning_completed_at=parse_timestamp(record['learning_completed_at']) if record.get('learning_completed_at') else None,
            market_regime=record.get('market_regime', {}),
            regime_confidence=record.get('regime_confidence'),
            volatility_environment=record.get('volatility_environment'),
            pattern_classification=record.get('pattern_classification'),
            pattern_confidence=record.get('pattern_confidence'),
            historical_pattern_success_rate=record.get('historical_pattern_success_rate'),
            learning_version=record.get('learning_version', 1),
            prompt_version=record.get('prompt_version'),
            generated_by_agent=record.get('generated_by_agent'),
            predicted_success_probability=record.get('predicted_success_probability'),
            predicted_time_to_target=record.get('predicted_time_to_target'),
            risk_adjusted_confidence=record.get('risk_adjusted_confidence'),
            learning_insights_id=record.get('learning_insights_id'),
            pattern_performance_id=record.get('pattern_performance_id')
        )

    async def get_signals_by_symbol(self, symbol: str, limit: int = 10) -> List[PlatformSignal]:
        """Return recent signals matching the same normalized market identity.

        The query is case-insensitive but exact, so a namespaced HIP-3 market
        never crosses into an unnamespaced market with the same ticker.
        """
        try:
            from kata.services.token_logo_service import normalize_market_symbol

            market_symbol = normalize_market_symbol(symbol)
            if not market_symbol:
                return []

            response = (
                self.db.from_("platform_signals")
                .select("*")
                .ilike("token_symbol", market_symbol)
                .order("analysis_timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return [
                self._db_record_to_platform_signal(record)
                for record in (response.data or [])
            ]
        except Exception as e:
            logger.error(f"Error getting signals for token logo identity {symbol}: {e}")
            return []

    async def update_signal_logos_by_symbol(self, symbol: str, logo_url: str) -> int:
        """Persist one resolved logo across every row for the same asset."""
        try:
            from kata.services.token_logo_service import normalize_market_symbol

            market_symbol = normalize_market_symbol(symbol)
            if not market_symbol:
                return 0
            response = (
                self.db.from_("platform_signals")
                .update({
                    "logo_url": logo_url,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .ilike("token_symbol", market_symbol)
                .execute()
            )
            return len(response.data or [])
        except Exception as e:
            logger.error(f"Error persisting token logo identity {symbol}: {e}")
            return 0

    async def get_signals_without_logos(self, limit: int = 10) -> List[PlatformSignal]:
        """Get signals that don't have logo URLs."""
        try:
            # Query for signals without logos (handle both null and empty string cases)
            response = self.db.from_('platform_signals').select('*').or_('logo_url.is.null,logo_url.eq.').eq('status', 'active').order('created_at', desc=True).limit(limit).execute()
            
            signals = []
            for record in response.data:
                signal = self._db_record_to_platform_signal(record)
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals without logos: {e}")
            return []

    async def update_signal_logo(self, signal_id: str, logo_url: str) -> bool:
        """Update a signal's logo URL."""
        try:
            response = self.db.from_('platform_signals').update({
                'logo_url': logo_url,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('signal_id', signal_id).execute()
            
            if response.data:
                logger.debug(f"Updated logo for signal {signal_id}: {logo_url}")
                return True
            else:
                logger.warning(f"No signal found with ID {signal_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating signal logo for {signal_id}: {e}")
            return False

    async def update_signal_with_learning_enhancements(self, enhanced_signal: PlatformSignal) -> bool:
        """Update learning feedback while preserving finalized executable sizing."""
        try:
            update_data = {
                'confidence': enhanced_signal.confidence,
                'ai_reasoning': enhanced_signal.ai_reasoning,
                'updated_at': datetime.now().isoformat()
            }

            result = self.db.table('platform_signals').update(update_data).eq(
                'signal_id', enhanced_signal.signal_id
            ).execute()

            if result.data:
                logger.debug(f"🧠 Applied learning confidence enhancement to signal {enhanced_signal.signal_id}: confidence={enhanced_signal.confidence:.3f}")
                return True
            else:
                logger.warning(f"No signal found to update with learning: {enhanced_signal.signal_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating signal with learning enhancements: {e}")
            return False


# Global service instance
_platform_signal_service: Optional[PlatformSignalService] = None


def get_platform_signal_service() -> PlatformSignalService:
    """Get or create platform signal service instance."""
    global _platform_signal_service
    
    if _platform_signal_service is None:
        _platform_signal_service = PlatformSignalService()
        
        # Market service no longer needed - using unified signal generator
    
    return _platform_signal_service
