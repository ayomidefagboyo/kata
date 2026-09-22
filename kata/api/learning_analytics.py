"""
Learning Analytics API

Provides endpoints for accessing multi-level learning analytics data.
Secure endpoint with authentication for admin access only.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import HTMLResponse
from typing import Callable, Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

from kata.config.database import get_service_client
from kata.config.policy_versions import policy_bundle_version, policy_stamp
from kata.config.settings import settings
# binance_service imported lazily inside functions that need it (avoids loading numpy+pandas at startup)
from kata.services.signal_schedule_optimizer import (
    DEFAULT_GENERATION_HOURS_UTC,
    get_signal_schedule_optimizer,
    normalize_schedule_hours,
)
from kata.services.platform_signal_service import get_platform_signal_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Supabase exposes a synchronous client. Keep its calls off the FastAPI event
# loop, but deliberately serialize this analytics module's database work. The
# dashboard fans out many sub-calculations at once; allowing each one to create
# another default-executor thread exhausted the container's process resources and
# produced `[Errno 11] Resource temporarily unavailable` responses.
try:
    _requested_learning_db_workers = int(
        os.getenv("LEARNING_ANALYTICS_DB_WORKERS", "1") or "1"
    )
except ValueError:
    _requested_learning_db_workers = 1
_LEARNING_DB_MAX_WORKERS = max(1, min(4, _requested_learning_db_workers))
_LEARNING_DB_EXECUTOR = ThreadPoolExecutor(
    max_workers=_LEARNING_DB_MAX_WORKERS,
    thread_name_prefix="learning-db",
)


async def _execute_db_query(operation):
    """Run one synchronous Supabase operation on the bounded analytics pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_LEARNING_DB_EXECUTOR, operation)

_BINANCE_INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

_TERMINAL_SIGNAL_STATUSES = {
    "hit_target_1",
    "hit_target_2",
    "hit_stop_loss",
    "expired",
    "completed",
    "invalidated",
}

_VISIBLE_YUKI_TRADE_STATUSES = {"pending", "filled", "closed", "cancelled"}


def verify_admin_access(request: Request):
    """
    Admin verification disabled - dashboard is now publicly accessible.
    """
    # Admin verification disabled for public access
    pass


def _dashboard_date_range_label(
    date_filter_start: Optional[str],
    date_filter_end: Optional[str],
) -> str:
    if not date_filter_start:
        return "All Time"
    return f"{date_filter_start[:10]} to {date_filter_end[:10] if date_filter_end else 'now'}"


def _default_dashboard_summary(
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "total_tracked_signals": 0,
        "active_tracking_sessions": 0,
        "recent_insights_30d": 0,
        "learning_system_uptime": "Degraded",
        "rejection_validation": {
            "total_candidates": 0,
            "pending_candidates": 0,
            "resolved_candidates": 0,
            "good_rejects": 0,
            "bad_rejects": 0,
            "ambiguous_rejects": 0,
            "informative_resolved": 0,
            "reject_precision_percent": 0.0,
            "bad_reject_rate_percent": 0.0,
            "timeline": [],
            "source_mix": [],
            "rejection_code_mix": [],
            "resolved_status_mix": [],
            "recent_resolutions": [],
        },
        "confidence_gate_tuning": {
            "enabled": False,
            "current_adjustment": 0.0,
            "target_bad_reject_rate": 0.0,
            "sample_resolved": 0,
            "timeline": [],
        },
        "signal_schedule_optimization": {
            "available": False,
            "configured_schedule": _augment_schedule({"hours_utc": _default_generation_hours_utc()}),
            "active_schedule": _augment_schedule({"hours_utc": _default_generation_hours_utc()}),
            "recommended_schedule": _augment_schedule({}),
            "latest_run": None,
            "latest_auto_apply_decision": None,
        },
        "date_range": {
            "start": date_filter_start,
            "end": date_filter_end,
            "label": _dashboard_date_range_label(date_filter_start, date_filter_end),
        },
    }


def _default_model_performance() -> Dict[str, Any]:
    return {
        "models": [],
        "prompts": [],
        "total_signals": 0,
        "attribution": {
            "model_attributed_signals": 0,
            "model_unattributed_signals": 0,
            "prompt_attributed_signals": 0,
            "prompt_unattributed_signals": 0,
        },
        "cost_summary": {
            "range_primary_cost_usd": 0.0,
            "range_challenger_cost_usd": 0.0,
            "range_total_estimated_cost_usd": 0.0,
            "workflow_breakdown": [],
            "billing_breakdown": {
                "total_billed_requests_count": 0,
                "total_billed_credits": 0,
                "services": [],
            },
            "coverage": {
                "observed_calls_count": 0,
                "telemetry_tracked_calls_count": 0,
                "telemetry_coverage_percent": 0.0,
                "cached_calls_count": 0,
                "billed_token_analysis_requests_count": 0,
                "untracked_token_analysis_requests_count": 0,
            },
            "budget": {
                "budget_enabled": False,
                "monthly_budget_usd": 0.0,
                "spent_usd": 0.0,
                "remaining_usd": None,
                "usage_percent": 0.0,
                "blocked": False,
                "challenger_calls_count": 0,
            },
        },
    }


def _default_prompt_optimization_summary() -> Dict[str, Any]:
    return {
        "total_optimizations": 0,
        "latest_by_agent": {},
        "recent_optimizations": [],
    }


async def _run_dashboard_section(
    section: str,
    loader: Callable[[], Any],
    default_factory: Callable[[], Any],
    warnings: List[Dict[str, str]],
):
    try:
        result = await loader()
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))
        return result
    except Exception as exc:
        message = str(exc)
        logger.warning("Learning dashboard section '%s' failed: %s", section, message)
        warnings.append({"section": section, "message": message})
        return default_factory()


async def _filter_active_yuki_rows_without_trade_attempts(
    db,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return all platform generated signals for learning tracking display."""
    return rows


_DASHBOARD_CACHE: Dict[str, Any] = {}

@router.get("/analytics/dashboard")
async def get_learning_dashboard_data(
    date_range: Optional[str] = Query(None, description="Predefined date range: 24h, 7d, 30d, 90d"),
    start_date: Optional[str] = Query(None, description="Custom start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Custom end date (YYYY-MM-DD)"),
    horizon: Optional[str] = Query(None, description="Optional recommendation outcome horizon override: 24h or 7d")
) -> Dict[str, Any]:
    """Get comprehensive learning analytics dashboard data with parallel async queries and 15s TTL cache."""
    cache_key = f"{date_range}:{start_date}:{end_date}:{horizon}"
    now_ts = time.time()
    if cache_key in _DASHBOARD_CACHE:
        cached_res, cached_time = _DASHBOARD_CACHE[cache_key]
        if now_ts - cached_time < 15:
            return cached_res

    try:
        db = get_service_client()
        date_filter_start, date_filter_end = _calculate_date_filters(date_range, start_date, end_date)
        warnings: List[Dict[str, str]] = []

        async def fetch_performance():
            res = await _execute_db_query(lambda: db.from_('learning_performance_view').select('*').limit(50).execute())
            return res.data or []

        async def fetch_patterns():
            res = await _execute_db_query(lambda: db.from_('pattern_performance_summary').select('*').order('success_rate', desc=True).limit(20).execute())
            return res.data or []

        async def fetch_past_trades():
            query = db.from_('past_trades_learning_dashboard').select('*')
            if date_filter_start and date_filter_end:
                query = query.gte('signal_created_at', date_filter_start).lte('signal_created_at', date_filter_end)
            res = await _execute_db_query(lambda: query.order('signal_created_at', desc=True).limit(500).execute())
            rows = res.data or []
            rows = await _hydrate_current_pnl_from_performance_tracking(db, rows)
            return await _filter_active_yuki_rows_without_trade_attempts(db, rows)

        async def fetch_ryu_recommendation_history():
            return await _fetch_recommendation_history(
                db,
                agent_type='ryu',
                date_filter_start=date_filter_start,
                date_filter_end=date_filter_end,
                # The learning dashboard paginates client-side and the Ryu buy
                # monitor intentionally filters to BUY-like rows only. Keep a
                # wider window here so the history does not appear artificially
                # capped at ~10 records when many older buy evaluations exist.
                limit=1000,
                horizon=horizon or "auto",
            )

        async def fetch_agent_comparison():
            if date_filter_start and date_filter_end:
                trade_agents = await _calculate_agent_comparison_for_range(db, date_filter_start, date_filter_end)
            else:
                res = await _execute_db_query(lambda: db.from_('agent_performance_comparison').select('*').execute())
                trade_agents = res.data or []
            trade_agents = [{**a, 'metric_source': a.get('metric_source') or 'executed_trades'} for a in trade_agents]
            recommendation_agents = await _calculate_recommendation_agent_comparison(
                db, date_filter_start, date_filter_end, horizon=horizon or "auto"
            )
            seen_tuples = {
                (str(a.get('agent_type', '')).lower(), str(a.get('metric_source', '')).lower())
                for a in trade_agents
            }
            return list(trade_agents) + [
                a for a in recommendation_agents
                if (str(a.get('agent_type', '')).lower(), str(a.get('metric_source', '')).lower()) not in seen_tuples
            ]

        async def fetch_system_health():
            res = await _execute_db_query(lambda: db.from_('learning_system_health').select('*').execute())
            return res.data or []

        async def fetch_summary():
            return await _calculate_summary_stats(db, date_filter_start, date_filter_end)

        async def fetch_model_and_prompt_analytics():
            model_perf = await _calculate_model_performance(db, date_filter_start, date_filter_end)
            prompt_opt = await _get_prompt_optimization_summary(db, date_filter_start, date_filter_end)
            return model_perf, prompt_opt

        # Run all dashboard sub-queries concurrently in parallel
        (
            learning_perf,
            pattern_perf,
            real_time_tracking,
            ryu_recommendation_history,
            agent_comp,
            system_health,
            summary_stats,
            (model_perf, prompt_opt),
        ) = await asyncio.gather(
            _run_dashboard_section("learning_performance", fetch_performance, list, warnings),
            _run_dashboard_section("pattern_performance", fetch_patterns, list, warnings),
            _run_dashboard_section("real_time_tracking", fetch_past_trades, list, warnings),
            _run_dashboard_section("ryu_recommendation_history", fetch_ryu_recommendation_history, list, warnings),
            _run_dashboard_section("agent_comparison", fetch_agent_comparison, list, warnings),
            _run_dashboard_section("system_health", fetch_system_health, list, warnings),
            _run_dashboard_section(
                "summary",
                fetch_summary,
                lambda: _default_dashboard_summary(date_filter_start, date_filter_end),
                warnings,
            ),
            _run_dashboard_section(
                "model_and_prompt_analytics",
                fetch_model_and_prompt_analytics,
                lambda: (_default_model_performance(), _default_prompt_optimization_summary()),
                warnings,
            ),
        )

        if warnings and isinstance(summary_stats, dict):
            summary_stats["learning_system_uptime"] = "Degraded"
            summary_stats["dashboard_warning_count"] = len(warnings)

        dashboard_data = {
            'learning_performance': learning_perf,
            'pattern_performance': pattern_perf,
            'real_time_tracking': real_time_tracking,
            'ryu_recommendation_history': ryu_recommendation_history,
            'agent_comparison': agent_comp,
            'system_health': system_health,
            'summary': summary_stats,
            'model_performance': model_perf,
            'prompt_optimization': prompt_opt,
            'warnings': warnings,
        }

        if warnings:
            logger.info("⚡ Learning dashboard data retrieved with %s degraded section(s)", len(warnings))
        else:
            logger.info("⚡ Learning dashboard data retrieved in parallel")
        res_payload = {
            "success": True,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
        _DASHBOARD_CACHE[cache_key] = (res_payload, time.time())
        return res_payload

    except Exception as e:
        logger.error(f"❌ Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
    except (TypeError, ValueError):
        pass
    return default


def _parse_datetime(value: Any) -> Optional[datetime]:
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
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _default_generation_hours_utc() -> List[int]:
    """Product fallback used only until a dashboard schedule is saved."""
    return list(DEFAULT_GENERATION_HOURS_UTC)


def _to_lagos_hours(hours_utc: List[int]) -> List[int]:
    # Lagos is UTC+1 year-round.
    return [int((hour + 1) % 24) for hour in hours_utc]


def _hours_label(hours: List[int], suffix: str = "UTC") -> str:
    if not hours:
        return "N/A"
    return "/".join(f"{int(hour):02d}" for hour in hours) + f" {suffix}"


def _normalized_hours(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    hours: List[int] = []
    for item in value:
        try:
            hour = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23:
            hours.append(hour)
    return sorted(set(hours))


def _augment_schedule(schedule: Any, fallback_hours: Optional[List[int]] = None) -> Dict[str, Any]:
    payload = dict(schedule) if isinstance(schedule, dict) else {}
    hours_utc = _normalized_hours(payload.get("hours_utc")) or sorted(set(fallback_hours or []))
    lagos_hours = _to_lagos_hours(hours_utc)
    payload["hours_utc"] = hours_utc
    payload["hours_lagos"] = lagos_hours
    payload["label"] = payload.get("label") or _hours_label(hours_utc, "UTC")
    payload["label_lagos"] = _hours_label(lagos_hours, "Lagos")
    return payload


def _signal_symbol_candidates(token_symbol: str) -> List[str]:
    raw_symbol = str(token_symbol or "").strip().upper().replace(" ", "")
    if not raw_symbol:
        return []

    if "/" in raw_symbol:
        pair_symbol = raw_symbol.split(":")[0]
        base_symbol = pair_symbol.split("/")[0]
    elif raw_symbol.endswith("USDT"):
        base_symbol = raw_symbol[:-4]
    else:
        base_symbol = raw_symbol

    candidates = [
        f"{base_symbol}/USDT:USDT",
        f"{base_symbol}/USDT",
        f"{base_symbol}USDT",
    ]
    if "/" in raw_symbol:
        candidates.append(raw_symbol)

    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _pick_trade_chart_interval(start_at: datetime, end_at: datetime) -> str:
    duration = max((end_at - start_at).total_seconds(), 0)
    if duration <= 12 * 60 * 60:
        return "5m"
    if duration <= 3 * 24 * 60 * 60:
        return "15m"
    if duration <= 14 * 24 * 60 * 60:
        return "1h"
    if duration <= 90 * 24 * 60 * 60:
        return "4h"
    return "1d"


def _leveraged_pnl_from_price(direction: str, entry_price: float, price: float, leverage: float) -> float:
    if entry_price <= 0:
        return 0.0
    if str(direction or "").upper() == "SHORT":
        raw_pnl = (entry_price - price) / entry_price * 100
    else:
        raw_pnl = (price - entry_price) / entry_price * 100
    return raw_pnl * max(leverage, 1)


def _price_from_leveraged_pnl(direction: str, entry_price: float, pnl_pct: Optional[float], leverage: float) -> float:
    if entry_price <= 0 or pnl_pct is None:
        return entry_price
    move_fraction = pnl_pct / 100 / max(leverage, 1)
    if str(direction or "").upper() == "SHORT":
        return entry_price * (1 - move_fraction)
    return entry_price * (1 + move_fraction)


def _trade_chart_point(
    timestamp: datetime,
    price: float,
    entry_price: float,
    direction: str,
    leverage: float,
    kind: str = "candle",
    candle: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    point: Dict[str, Any] = {
        "timestamp": _iso_utc(timestamp),
        "price": float(price),
        "pnl_pct": float(_leveraged_pnl_from_price(direction, entry_price, price, leverage)),
        "kind": kind,
    }

    if candle and len(candle) >= 6:
        point.update(
            {
                "open": _safe_float(candle[1], 0.0),
                "high": _safe_float(candle[2], 0.0),
                "low": _safe_float(candle[3], 0.0),
                "close": _safe_float(candle[4], 0.0),
                "volume": _safe_float(candle[5], 0.0),
            }
        )
    return point


def _signal_entry_activation(signal: Dict[str, Any]) -> Optional[tuple[datetime, float]]:
    """Return the recorded entry activation event when the signal required one."""
    market_conditions = signal.get("market_conditions")
    if not isinstance(market_conditions, dict):
        return None

    activated_at = _parse_datetime(market_conditions.get("entry_activated_at"))
    activated_price = _safe_float(market_conditions.get("entry_activation_price"))
    if activated_at is None or activated_price is None or activated_price <= 0:
        return None
    return activated_at, activated_price


def _dedupe_chart_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_timestamp: Dict[str, Dict[str, Any]] = {}
    for point in points:
        timestamp = point.get("timestamp")
        if timestamp:
            by_timestamp[str(timestamp)] = point

    return sorted(
        by_timestamp.values(),
        key=lambda point: _parse_datetime(point.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
    )


def _fallback_trade_chart_points(
    signal: Dict[str, Any],
    perf: Dict[str, Any],
    start_at: datetime,
    end_at: datetime,
) -> List[Dict[str, Any]]:
    entry_price = _safe_float(signal.get("entry_price"), 0.0) or 0.0
    leverage = max(_safe_float(signal.get("leverage"), 1.0) or 1.0, 1.0)
    direction = str(signal.get("direction") or "LONG").upper()
    exit_price = _safe_float(perf.get("exit_price"))
    leveraged_pnl = _safe_float(perf.get("leveraged_pnl_percent"))
    if leveraged_pnl is None:
        actual_pnl = _safe_float(perf.get("actual_pnl_percent"))
        leveraged_pnl = actual_pnl * leverage if actual_pnl is not None else None

    mark_price = exit_price if exit_price and exit_price > 0 else _price_from_leveraged_pnl(
        direction,
        entry_price,
        leveraged_pnl,
        leverage,
    )

    if end_at <= start_at:
        end_at = start_at + timedelta(minutes=5)

    points = [
        _trade_chart_point(start_at, entry_price, entry_price, direction, leverage, "signal_generated"),
    ]
    entry_activation = _signal_entry_activation(signal)
    if entry_activation:
        activated_at, activated_price = entry_activation
        points.append(
            _trade_chart_point(
                activated_at,
                activated_price,
                entry_price,
                direction,
                leverage,
                "entry_activation",
            )
        )

    endpoint_kind = "exit" if perf.get("exit_timestamp") else "estimated"
    points.append(
        _trade_chart_point(end_at, mark_price, entry_price, direction, leverage, endpoint_kind)
    )
    return points


def _extract_signal_chart_overlays(signal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract stored signal overlays from strict market-structure context."""
    for container_key in ("market_conditions", "ai_confidence_breakdown"):
        container = signal.get(container_key)
        if not isinstance(container, dict):
            continue
        context = container.get("market_structure_context")
        if not isinstance(context, dict):
            continue
        overlays = context.get("chart_overlays")
        if isinstance(overlays, list):
            return [overlay for overlay in overlays if isinstance(overlay, dict)]
    return []


@router.get("/analytics/signal-chart/{signal_id}")
async def get_signal_chart_data(
    signal_id: str,
    interval: Optional[str] = Query(None, description="Candle interval override (1m/5m/15m/1h/4h/1d)"),
) -> Dict[str, Any]:
    """Return timestamped price and leveraged-PnL path for one tracked signal."""
    try:
        db = get_service_client()

        signal_query = (
            db.from_("platform_signals")
            .select(
                "signal_id, token_symbol, direction, timeframe, entry_price, target_1, target_2, "
                "stop_loss, leverage, analysis_timestamp, created_at, expires_at, updated_at, status, "
                "market_conditions, ai_confidence_breakdown"
            )
            .eq("signal_id", signal_id)
            .limit(1)
        )
        signal_result = await _execute_db_query(signal_query.execute)
        signal_rows = signal_result.data or []
        if not signal_rows:
            raise HTTPException(status_code=404, detail="Signal not found")

        signal = signal_rows[0]

        perf_query = (
            db.from_("platform_signal_performance_tracking")
            .select(
                "signal_id, outcome, exit_price, exit_timestamp, exit_reason, actual_pnl_percent, "
                "leveraged_pnl_percent, max_profit_reached, max_loss_reached, last_updated"
            )
            .eq("signal_id", signal_id)
            .limit(1)
        )
        perf_result = await _execute_db_query(perf_query.execute)
        perf = (perf_result.data or [{}])[0]

        entry_price = _safe_float(signal.get("entry_price"))
        if not entry_price or entry_price <= 0:
            raise HTTPException(status_code=422, detail="Signal has no valid entry price")

        now = datetime.now(timezone.utc)
        start_at = (
            _parse_datetime(signal.get("created_at"))
            or _parse_datetime(signal.get("analysis_timestamp"))
            or now - timedelta(hours=4)
        )
        status = str(signal.get("status") or "").lower()
        exit_at = _parse_datetime(perf.get("exit_timestamp"))
        updated_at = _parse_datetime(signal.get("updated_at")) or _parse_datetime(perf.get("last_updated"))
        expires_at = _parse_datetime(signal.get("expires_at"))

        if exit_at:
            end_at = exit_at
        elif status in _TERMINAL_SIGNAL_STATUSES:
            end_at = updated_at or expires_at or now
        else:
            end_at = now

        if end_at <= start_at:
            end_at = start_at + timedelta(hours=1)

        direction = str(signal.get("direction") or "LONG").upper()
        leverage = max(_safe_float(signal.get("leverage"), 1.0) or 1.0, 1.0)
        requested_interval = str(interval or "").lower().strip()
        chart_interval = (
            requested_interval
            if requested_interval in _BINANCE_INTERVAL_MS
            else _pick_trade_chart_interval(start_at, end_at)
        )
        interval_ms = _BINANCE_INTERVAL_MS[chart_interval]

        # Extend the window beyond the trade itself so the chart shows market
        # context: candles before entry (pan left to inspect the setup) and,
        # for closed signals, some continuation after the exit.
        window_candles = int(((end_at - start_at).total_seconds() * 1000) / interval_ms) + 2
        post_candles = 0
        if end_at < now:
            post_candles = min(60, max(0, int(((now - end_at).total_seconds() * 1000) / interval_ms)))
        context_candles = max(0, min(150, 1000 - window_candles - post_candles - 5))
        fetch_start_at = start_at - timedelta(milliseconds=context_candles * interval_ms)
        fetch_end_at = min(now, end_at + timedelta(milliseconds=post_candles * interval_ms))
        fetch_since_ms = int(fetch_start_at.timestamp() * 1000)
        limit = min(1000, max(5, window_candles + context_candles + post_candles))

        candles: List[List[Any]] = []
        used_symbol: Optional[str] = None
        candle_error: Optional[str] = None
        candidates = _signal_symbol_candidates(str(signal.get("token_symbol") or ""))

        try:
            from kata.services.binance_service import create_binance_service
            binance = create_binance_service()
            if not binance.exchange:
                binance._init_exchange()

            for candidate in candidates:
                fetched = await binance._make_rate_limited_request(
                    binance.exchange.fetch_ohlcv,
                    2,
                    candidate,
                    chart_interval,
                    since=fetch_since_ms,
                    limit=limit,
                )
                if fetched:
                    candles = fetched
                    used_symbol = candidate
                    break
        except Exception as market_err:
            candle_error = str(market_err)
            logger.warning(f"Signal chart candle fetch failed for {signal_id}: {market_err}")

        points: List[Dict[str, Any]] = [
            _trade_chart_point(
                start_at,
                entry_price,
                entry_price,
                direction,
                leverage,
                "signal_generated",
            )
        ]

        entry_activation = _signal_entry_activation(signal)
        if entry_activation:
            activated_at, activated_price = entry_activation
            points.append(
                _trade_chart_point(
                    activated_at,
                    activated_price,
                    entry_price,
                    direction,
                    leverage,
                    "entry_activation",
                )
            )

        fetch_end_ms = int(fetch_end_at.timestamp() * 1000)
        for candle in candles:
            if not candle or len(candle) < 6:
                continue
            timestamp_ms = int(candle[0])
            if timestamp_ms < fetch_since_ms or timestamp_ms > fetch_end_ms:
                continue

            close_price = _safe_float(candle[4])
            if close_price is None or close_price <= 0:
                continue

            candle_time = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
            points.append(
                _trade_chart_point(
                    candle_time,
                    close_price,
                    entry_price,
                    direction,
                    leverage,
                    "candle",
                    candle,
                )
            )

        exit_price = _safe_float(perf.get("exit_price"))
        final_leveraged_pnl = _safe_float(perf.get("leveraged_pnl_percent"))
        if final_leveraged_pnl is None:
            actual_pnl = _safe_float(perf.get("actual_pnl_percent"))
            final_leveraged_pnl = actual_pnl * leverage if actual_pnl is not None else None

        terminal_price = exit_price if exit_price and exit_price > 0 else None
        if terminal_price is None:
            if status == "hit_target_2":
                terminal_price = _safe_float(signal.get("target_2")) or _safe_float(signal.get("target_1"))
            elif status == "hit_target_1":
                terminal_price = _safe_float(signal.get("target_1"))
            elif status == "hit_stop_loss":
                terminal_price = _safe_float(signal.get("stop_loss"))
            else:
                terminal_price = _price_from_leveraged_pnl(
                    direction,
                    entry_price,
                    final_leveraged_pnl,
                    leverage,
                )

        if terminal_price and terminal_price > 0:
            endpoint_kind = "exit" if status in _TERMINAL_SIGNAL_STATUSES else "mark"
            exit_point = _trade_chart_point(end_at, terminal_price, entry_price, direction, leverage, endpoint_kind)
            if final_leveraged_pnl is not None:
                exit_point["pnl_pct"] = final_leveraged_pnl
            points.append(exit_point)

        points = _dedupe_chart_points(points)
        source = "candles" if len(points) >= 2 and any(point.get("kind") == "candle" for point in points) else "estimated"
        if source == "estimated":
            points = _fallback_trade_chart_points(signal, perf, start_at, end_at)

        return {
            "success": True,
            "signal_id": signal_id,
            "token_symbol": signal.get("token_symbol"),
            "market_symbol": used_symbol,
            "candidate_symbols": candidates,
            "source": source,
            "interval": chart_interval,
            "available_intervals": list(_BINANCE_INTERVAL_MS.keys()),
            "window": {
                "start": _iso_utc(start_at),
                "end": _iso_utc(end_at),
            },
            "levels": {
                "entry": entry_price,
                "target_1": _safe_float(signal.get("target_1")),
                "target_2": _safe_float(signal.get("target_2")),
                "stop_loss": _safe_float(signal.get("stop_loss")),
            },
            "overlays": _extract_signal_chart_overlays(signal),
            "points": points,
            "warning": candle_error,
            "timestamp": _iso_utc(datetime.now(timezone.utc)),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Signal chart retrieval failed for {signal_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signal chart: {str(e)}")


async def _hydrate_current_pnl_from_performance_tracking(db, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Backfill current_pnl (and exit_pnl), then enrich rows with canonical signal
    fields used by the frontend drilldown chart.

    Priority order:
    1) leveraged_pnl_percent  — populated by the exit trigger for completed/expired signals
    2) actual_pnl_percent     — populated by the WebSocket RPC for live active signals
    """
    if not rows:
        return rows

    signal_ids = [str(r["signal_id"]) for r in rows if r.get("signal_id")]
    if not signal_ids:
        return rows

    perf_by_id: Dict[str, Dict[str, Any]] = {}
    signal_by_id: Dict[str, Dict[str, Any]] = {}

    chunk_size = 300
    for i in range(0, len(signal_ids), chunk_size):
        batch = signal_ids[i:i + chunk_size]
        perf_query = (
            db.from_("platform_signal_performance_tracking")
            .select(
                "signal_id, leveraged_pnl_percent, actual_pnl_percent, "
                "exit_price, exit_timestamp, max_profit_reached, max_loss_reached, "
                "outcome, exit_reason, target_1_hit, target_1_hit_price, "
                "target_1_hit_timestamp, target_1_leveraged_pnl_percent"
            )
            .in_("signal_id", batch)
        )
        signal_query = (
            db.from_("platform_signals")
            .select(
                "signal_id, entry_price, target_1, target_2, stop_loss, "
                "leverage, expires_at, status"
            )
            .in_("signal_id", batch)
        )

        try:
            perf_res, signal_res = await asyncio.gather(
                _execute_db_query(perf_query.execute),
                _execute_db_query(signal_query.execute),
            )
            for perf_row in (perf_res.data or []):
                sid = perf_row.get("signal_id")
                if sid:
                    perf_by_id[str(sid)] = perf_row
            for signal_row in (signal_res.data or []):
                sid = signal_row.get("signal_id")
                if sid:
                    signal_by_id[str(sid)] = signal_row
        except Exception as hydration_err:
            logger.debug(f"current_pnl hydration batch failed: {hydration_err}")

    hydrated = []
    for row in rows:
        sid = str(row.get("signal_id") or "")
        perf = perf_by_id.get(sid) or {}
        signal = signal_by_id.get(sid) or {}

        for field in ("entry_price", "target_1", "target_2", "stop_loss", "leverage", "status"):
            if signal.get(field) is not None:
                row[field] = signal.get(field)
        if signal.get("expires_at") is not None:
            row["signal_expires_at"] = signal.get("expires_at")
        if perf.get("exit_price") is not None:
            row["exit_price"] = perf.get("exit_price")
        if perf.get("exit_timestamp") is not None:
            row["exit_timestamp"] = perf.get("exit_timestamp")
        if perf.get("max_profit_reached") is not None:
            row["max_profit_reached"] = perf.get("max_profit_reached")
        if perf.get("max_loss_reached") is not None:
            row["max_loss_reached"] = perf.get("max_loss_reached")
        for field in (
            "target_1_hit",
            "target_1_hit_price",
            "target_1_hit_timestamp",
            "target_1_leveraged_pnl_percent",
        ):
            if perf.get(field) is not None:
                row[field] = perf.get(field)

        signal_status = str(signal.get("status") or row.get("status") or "").lower()
        perf_outcome = str(perf.get("outcome") or "").lower()
        exit_reason = str(perf.get("exit_reason") or "").lower()
        target_1_hit = bool(perf.get("target_1_hit") or row.get("target_1_hit"))
        expired_signal = (
            signal_status == "expired"
            or perf_outcome == "expired"
            or exit_reason == "expired"
        )

        # Expired signals are neutral missed/expired outcomes, even when the
        # mark price at expiry was favorable. Keep the favorable move available
        # as max_profit_reached, but never promote it to win/final PnL.
        # Exception: a signal that hit TP1 before expiry is a realized TP1 win —
        # legacy rows were stamped 'expired' before expire_old_signals learned
        # to finalize them as hit_target_1.
        if expired_signal:
            if target_1_hit:
                tp1_pnl = (
                    perf.get("target_1_leveraged_pnl_percent")
                    or perf.get("leveraged_pnl_percent")
                    or perf.get("actual_pnl_percent")
                )
                row["status"] = "hit_target_1"
                row["tracking_status"] = "completed"
                row["performance_status"] = "profit"
                if tp1_pnl is not None:
                    try:
                        pnl_val = abs(float(tp1_pnl))
                        row["current_pnl"] = pnl_val
                        row["exit_pnl"] = pnl_val
                    except (TypeError, ValueError):
                        pass
                hydrated.append(row)
                continue
            row["status"] = "expired"
            row["tracking_status"] = "expired"
            row["performance_status"] = "neutral"
            row["current_pnl"] = None
            row["exit_pnl"] = None
            hydrated.append(row)
            continue

        # Invalidated = the thesis was reversed/replaced before resolving. It is
        # a first-class lifecycle outcome, not a neutral leftover: surface it as
        # such and keep the PnL at invalidation time when tracking recorded one.
        invalidated_signal = (
            signal_status == "invalidated"
            or perf_outcome == "invalidated"
            or exit_reason == "invalidated"
        )
        if invalidated_signal:
            row["status"] = "invalidated"
            row["tracking_status"] = "completed"
            row["performance_status"] = "invalidated"
            candidate = perf.get("leveraged_pnl_percent")
            if candidate is None:
                candidate = perf.get("actual_pnl_percent")
            try:
                pnl_val = float(candidate) if candidate is not None else None
            except (TypeError, ValueError):
                pnl_val = None
            row["current_pnl"] = pnl_val
            row["exit_pnl"] = pnl_val
            hydrated.append(row)
            continue

        if signal_status in ("hit_target_1", "hit_target_2"):
            row["performance_status"] = "profit"
        elif signal_status == "hit_stop_loss":
            row["performance_status"] = "loss"
        elif target_1_hit and signal_status == "active":
            row["performance_status"] = "target_1_hit"

        # For terminal wins/losses, platform_signal_performance_tracking is the
        # canonical realized result. This also corrects legacy view rows where
        # hit_target_2 current_pnl was still calculated from target_1.
        candidate = perf.get("leveraged_pnl_percent")
        if candidate is None:
            candidate = perf.get("actual_pnl_percent")

        if signal_status in ("hit_target_1", "hit_target_2", "hit_stop_loss") and candidate is not None:
            try:
                pnl_val = float(candidate)
                if signal_status == "hit_stop_loss" and pnl_val > 0:
                    pnl_val = -abs(pnl_val)
                elif signal_status in ("hit_target_1", "hit_target_2") and pnl_val < 0:
                    pnl_val = abs(pnl_val)
                row["current_pnl"] = pnl_val
                row["exit_pnl"] = pnl_val
                hydrated.append(row)
                continue
            except (TypeError, ValueError):
                pass

        if signal_status == "active" and candidate is not None:
            try:
                row["current_pnl"] = float(candidate)
                hydrated.append(row)
                continue
            except (TypeError, ValueError):
                pass

        if row.get("current_pnl") is not None:
            # View already has a value — trust it, while keeping enriched fields.
            hydrated.append(row)
            continue

        # leveraged_pnl_percent is set by the exit trigger (completed/expired signals).
        # actual_pnl_percent is set by the WebSocket RPC (live active signals).
        if candidate is not None:
            try:
                pnl_val = float(candidate)
                row["current_pnl"] = pnl_val
                # Also backfill exit_pnl for completed/expired rows if still missing.
                tracking_status = str(row.get("tracking_status") or "").lower()
                if tracking_status in ("completed",) and row.get("exit_pnl") is None:
                    row["exit_pnl"] = pnl_val
            except (TypeError, ValueError):
                pass

        hydrated.append(row)

    return hydrated


@router.get("/analytics/performance/{agent_type}")
async def get_agent_performance(agent_type: str) -> Dict[str, Any]:
    """Get detailed performance data for a specific agent."""
    try:
        db = get_service_client()

        # Agent-specific performance data
        performance_data = {}

        # Recent signals performance
        signals_query = db.from_('platform_signals').select('''
            signal_id, token_symbol, direction, confidence,
            created_at, learning_tracked
        ''').eq('generated_by_agent', agent_type).order('created_at', desc=True).limit(100)
        signals_result = await _execute_db_query(signals_query.execute)

        performance_data['recent_signals'] = signals_result.data or []

        # Learning insights for this agent
        insights_query = db.from_('agent_learning_insights').select('*').eq('agent_type', agent_type).order('timestamp', desc=True).limit(50)
        insights_result = await _execute_db_query(insights_query.execute)

        performance_data['learning_insights'] = insights_result.data or []

        # Performance tracking data (real_time_signal_tracking removed 2026-05-05)
        tracking_query = (
            db.from_('platform_signal_performance_tracking')
            .select('signal_id, outcome, exit_reason, actual_pnl_percent, leveraged_pnl_percent, last_updated')
            .in_('signal_id',
                 [r['signal_id'] for r in (performance_data.get('recent_signals') or [])])
            .order('last_updated', desc=True)
            .limit(30)
        )
        tracking_result = await _execute_db_query(tracking_query.execute)

        performance_data['real_time_tracking'] = tracking_result.data or []

        # Performance calculations
        performance_data['statistics'] = await _calculate_agent_stats(db, agent_type)

        return {
            "success": True,
            "agent_type": agent_type,
            "data": performance_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Agent performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent performance: {str(e)}")


@router.get("/analytics/patterns")
async def get_pattern_analysis() -> Dict[str, Any]:
    """Get detailed pattern performance analysis."""
    try:
        db = get_service_client()

        # Get pattern performance tracking
        patterns_query = db.from_('pattern_performance_tracking').select('*').order('success_rate', desc=True)
        patterns_result = await _execute_db_query(patterns_query.execute)

        patterns_data = {
            "patterns": patterns_result.data or [],
            "pattern_insights": await _analyze_patterns(db, patterns_result.data or [])
        }

        return {
            "success": True,
            "data": patterns_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pattern analysis: {str(e)}")


@router.get("/analytics/health")
async def get_system_health() -> Dict[str, Any]:
    """Get learning system health status."""
    try:
        db = get_service_client()

        # System health from view
        health_query = db.from_('learning_system_health').select('*')
        health_result = await _execute_db_query(health_query.execute)
        health_data = health_result.data or []

        # Additional health checks
        additional_health = {
            "database_connectivity": True,
            "recent_signal_activity": await _check_recent_activity(db),
            "learning_system_status": await _check_learning_status(db)
        }

        return {
            "success": True,
            "health_components": health_data,
            "system_status": additional_health,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _calculate_summary_stats(db, date_filter_start: Optional[str] = None, date_filter_end: Optional[str] = None) -> Dict[str, Any]:
    """Calculate summary statistics for the dashboard using parallel async tasks."""
    try:
        signals_query = db.from_('platform_signals').select('signal_id', count='exact').eq('learning_tracked', True)
        if date_filter_start and date_filter_end:
            signals_query = signals_query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)

        active_tracking_query = db.from_('platform_signals').select('signal_id', count='exact').eq('status', 'active').eq('learning_tracked', True)
        if date_filter_start and date_filter_end:
            active_tracking_query = active_tracking_query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)

        insights_start_date = (
            date_filter_start
            if date_filter_start
            else (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        )
        insights_end_date = date_filter_end if date_filter_end else datetime.now(timezone.utc).isoformat()
        insights_query = db.from_('agent_learning_insights').select('id', count='exact').gte('timestamp', insights_start_date).lte('timestamp', insights_end_date)

        # Run summary count queries and sub-status calculations in parallel
        (
            signals_count_res,
            active_tracking_res,
            recent_insights_res,
            rejection_validation,
            confidence_gate_tuning,
            signal_schedule_optimization,
        ) = await asyncio.gather(
            _execute_db_query(signals_query.execute),
            _execute_db_query(active_tracking_query.execute),
            _execute_db_query(insights_query.execute),
            _calculate_rejection_validation_stats(db, date_filter_start, date_filter_end),
            _get_confidence_gate_tuning_status(db, date_filter_start, date_filter_end),
            _get_signal_schedule_optimization_status(db),
        )

        insights_label = "recent_insights_30d" if not date_filter_start else f"insights_{date_filter_start[:10]}_to_{date_filter_end[:10] if date_filter_end else 'now'}"

        return {
            "total_tracked_signals": signals_count_res.count or 0,
            "active_tracking_sessions": active_tracking_res.count or 0,
            "recent_insights_30d": recent_insights_res.count or 0,
            insights_label: recent_insights_res.count or 0,
            "learning_system_uptime": "Active",
            "rejection_validation": rejection_validation,
            "confidence_gate_tuning": confidence_gate_tuning,
            "signal_schedule_optimization": signal_schedule_optimization,
            "date_range": {
                "start": date_filter_start,
                "end": date_filter_end,
                "label": "All Time" if not date_filter_start else f"{date_filter_start[:10]} to {date_filter_end[:10] if date_filter_end else 'now'}"
            }
        }
    except Exception as e:
        logger.error(f"Summary stats calculation failed: {e}")
        return {"error": str(e)}


async def _get_signal_schedule_optimization_status(db) -> Dict[str, Any]:
    """Return the latest timing optimizer summary for analytics display."""
    configured_hours = _default_generation_hours_utc()
    auto_apply_enabled = _env_bool("SIGNAL_SCHEDULE_AUTO_OPTIMIZE", False)

    try:
        optimizer = get_signal_schedule_optimizer()
        active_hours = await _execute_db_query(
            lambda: optimizer.load_active_schedule(configured_hours)
        )
        latest_snapshot_query = (
            db.from_("agent_learning_insights")
            .select("id, timestamp, insights, confidence_score, performance_impact")
            .eq("agent_type", "yuki")
            .eq("learning_type", "signal_schedule_optimization")
            .order("timestamp", desc=True)
            .limit(1)
        )
        latest_snapshot_result = await _execute_db_query(latest_snapshot_query.execute)
        latest_snapshot = (latest_snapshot_result.data or [None])[0]

        latest_run_query = (
            db.from_("agent_learning_insights")
            .select("id, timestamp, insights, confidence_score, performance_impact")
            .eq("agent_type", "yuki")
            .eq("learning_type", "signal_generation_schedule_run")
            .order("timestamp", desc=True)
            .limit(1)
        )
        latest_run_result = await _execute_db_query(latest_run_query.execute)
        latest_run = (latest_run_result.data or [None])[0]

        latest_req_query = (
            db.from_("agent_learning_insights")
            .select("id, timestamp, insights, confidence_score, performance_impact")
            .eq("agent_type", "yuki")
            .eq("learning_type", "platform_signal_generation_request")
            .order("timestamp", desc=True)
            .limit(1)
        )
        latest_req_result = await _execute_db_query(latest_req_query.execute)
        latest_req = (latest_req_result.data or [None])[0]

        latest_apply_query = (
            db.from_("agent_learning_insights")
            .select("id, timestamp, insights, confidence_score, performance_impact")
            .eq("agent_type", "yuki")
            .eq("learning_type", "signal_schedule_auto_apply")
            .order("timestamp", desc=True)
            .limit(1)
        )
        latest_apply_result = await _execute_db_query(latest_apply_query.execute)
        latest_apply = (latest_apply_result.data or [None])[0]

        source = "latest_snapshot"
        snapshot_payload = latest_snapshot.get("insights") if isinstance(latest_snapshot, dict) else None
        should_live_calculate = _env_bool("ANALYTICS_LIVE_SCHEDULE_OPTIMIZER", False) or not isinstance(snapshot_payload, dict)
        if should_live_calculate:
            lookback_days = max(14, int(os.getenv("SIGNAL_SCHEDULE_LOOKBACK_DAYS", "365") or "365"))
            min_hour_samples = max(3, int(os.getenv("SIGNAL_SCHEDULE_MIN_HOUR_SAMPLES", "8") or "8"))
            min_total_samples = max(20, int(os.getenv("SIGNAL_SCHEDULE_MIN_TOTAL_SAMPLES", "120") or "120"))
            snapshot_payload = optimizer.calculate(
                configured_hours_utc=active_hours,
                lookback_days=lookback_days,
                min_hour_samples=min_hour_samples,
                min_total_samples=min_total_samples,
            )
            source = "live_calculation"

        summary = dict(snapshot_payload or {})

        recommended_schedule = _augment_schedule(summary.get("recommended_schedule"))
        configured_schedule = _augment_schedule({"hours_utc": configured_hours})
        active_schedule = _augment_schedule({"hours_utc": active_hours})

        latest_run_payload = None
        run_ts = None
        if isinstance(latest_run, dict):
            run_insights = latest_run.get("insights") if isinstance(latest_run.get("insights"), dict) else {}
            slot_hour = run_insights.get("slot_hour_utc")
            try:
                slot_hour = int(slot_hour) if slot_hour is not None else None
            except (TypeError, ValueError):
                slot_hour = None
            run_ts = latest_run.get("timestamp") or run_insights.get("actual_run_at_utc")
            latest_run_payload = {
                "timestamp": latest_run.get("timestamp"),
                "run_id": run_insights.get("run_id"),
                "signals_generated": int(run_insights.get("signals_generated") or 0),
                "trigger": run_insights.get("trigger"),
                "scheduled_for_utc": run_insights.get("scheduled_for_utc"),
                "actual_run_at_utc": run_insights.get("actual_run_at_utc") or latest_run.get("timestamp"),
                "slot_hour_utc": slot_hour,
                "slot_hour_lagos": ((slot_hour + 1) % 24) if slot_hour is not None else None,
            }

        # Check if there is a newer platform generation request (running or completed)
        if isinstance(latest_req, dict):
            req_insights = latest_req.get("insights") if isinstance(latest_req.get("insights"), dict) else {}
            req_ts = latest_req.get("timestamp") or req_insights.get("requested_at") or req_insights.get("claimed_at")
            if req_ts and (not run_ts or str(req_ts) > str(run_ts)):
                slot_hour = req_insights.get("slot_hour_utc")
                try:
                    slot_hour = int(slot_hour) if slot_hour is not None else None
                except (TypeError, ValueError):
                    slot_hour = None

                signals_gen = req_insights.get("signals_generated")
                if signals_gen is None:
                    req_time_str = req_insights.get("claimed_at") or req_insights.get("requested_at") or req_ts
                    try:
                        sig_count_query = (
                            db.from_("platform_signals")
                            .select("id", count="exact")
                            .gte("created_at", str(req_time_str))
                        )
                        sig_res = await _execute_db_query(sig_count_query.execute)
                        signals_gen = sig_res.count if sig_res and sig_res.count is not None else 0
                    except Exception:
                        signals_gen = 0
                else:
                    signals_gen = int(signals_gen or 0)

                actual_run = req_insights.get("completed_at") or req_insights.get("claimed_at") or req_insights.get("actual_run_at_utc") or req_ts
                latest_run_payload = {
                    "timestamp": req_ts,
                    "run_id": req_insights.get("request_id") or latest_req.get("id"),
                    "signals_generated": signals_gen,
                    "trigger": req_insights.get("trigger", "scheduled"),
                    "scheduled_for_utc": req_insights.get("scheduled_for_utc"),
                    "actual_run_at_utc": actual_run,
                    "slot_hour_utc": slot_hour,
                    "slot_hour_lagos": ((slot_hour + 1) % 24) if slot_hour is not None else None,
                }

        latest_apply_payload = None
        if isinstance(latest_apply, dict):
            apply_insights = latest_apply.get("insights") if isinstance(latest_apply.get("insights"), dict) else {}
            latest_apply_payload = {
                "timestamp": latest_apply.get("timestamp"),
                "applied": bool(apply_insights.get("applied", False)),
                "reason": apply_insights.get("reason"),
                "previous_schedule": _augment_schedule({"hours_utc": _normalized_hours(apply_insights.get("previous_hours_utc"))}),
                "active_schedule": _augment_schedule({"hours_utc": _normalized_hours(apply_insights.get("active_hours_utc"))}),
                "recommended_schedule": _augment_schedule({"hours_utc": _normalized_hours(apply_insights.get("recommended_hours_utc"))}),
            }

        summary.update(
            {
                "available": True,
                "source": source,
                "schedule_source": "learning_dashboard",
                "auto_apply_enabled": auto_apply_enabled,
                "auto_min_confidence": _safe_float(os.getenv("SIGNAL_SCHEDULE_AUTO_MIN_CONFIDENCE"), 0.70),
                "auto_min_score_delta": _safe_float(os.getenv("SIGNAL_SCHEDULE_AUTO_MIN_SCORE_DELTA"), 1.0),
                "configured_schedule": configured_schedule,
                "active_schedule": active_schedule,
                "recommended_schedule": recommended_schedule,
                "latest_run": latest_run_payload,
                "latest_auto_apply_decision": latest_apply_payload,
                "latest_snapshot_at": latest_snapshot.get("timestamp") if isinstance(latest_snapshot, dict) else None,
            }
        )
        return summary
    except Exception as e:
        logger.warning(f"Signal schedule optimization summary failed: {e}")
        return {
            "available": False,
            "error": str(e),
            "auto_apply_enabled": auto_apply_enabled,
            "configured_schedule": _augment_schedule({"hours_utc": configured_hours}),
            "active_schedule": _augment_schedule({"hours_utc": configured_hours}),
            "recommended_schedule": _augment_schedule({}),
            "latest_run": None,
            "latest_auto_apply_decision": None,
        }


@router.post("/analytics/schedule/apply")
async def apply_signal_schedule(request: Request) -> Dict[str, Any]:
    """Make a dashboard-selected schedule authoritative for the live worker."""
    try:
        data = await request.json()
        schedule = data.get("schedule", {}) if isinstance(data, dict) else {}
        hours = normalize_schedule_hours(schedule.get("hours_utc"))
        if not hours:
            raise HTTPException(
                status_code=400,
                detail="Choose at least one valid UTC schedule hour.",
            )

        db = get_service_client()
        optimizer = get_signal_schedule_optimizer()
        applied = await _execute_db_query(
            lambda: optimizer.apply_active_schedule(
                hours,
                source="learning_dashboard",
                applied_by="dashboard_user",
            )
        )

        insights_payload = {
            "applied": True,
            "reason": "Manually applied from Learning Dashboard",
            "previous_hours_utc": applied.get("previous_hours_utc", []),
            "active_hours_utc": hours,
            "recommended_hours_utc": hours,
            "source": "learning_dashboard",
        }
        insert_query = db.table("agent_learning_insights").insert({
            "agent_type": "yuki",
            "signal_id": f"schedule_dashboard_apply_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "learning_level": "meta",
            "learning_type": "signal_schedule_auto_apply",
            "confidence_score": 1.0,
            "insights": insights_payload,
            "performance_impact": {
                "previous_hours_utc": applied.get("previous_hours_utc", []),
                "active_hours_utc": hours,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await _execute_db_query(insert_query.execute)
        worker_update = await get_platform_signal_service().enqueue_platform_generation_request(
            requested_by="learning_dashboard",
            trigger="schedule_change",
            configured_hours_utc=hours,
        )
        logger.info("Learning Dashboard applied signal schedule: %s UTC", hours)
        return {
            "success": True,
            "message": "Schedule applied and sent to the signal worker.",
            "active_schedule": _augment_schedule({"hours_utc": hours}),
            "previous_schedule": _augment_schedule({
                "hours_utc": applied.get("previous_hours_utc", []),
            }),
            "schedule_source": "learning_dashboard",
            "worker_update": {
                "request_id": worker_update.get("request_id"),
                "status": worker_update.get("status"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _calculate_agent_comparison_for_range(
    db,
    date_filter_start: Optional[str],
    date_filter_end: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Calculate agent performance metrics for the selected date range
    using the past_trades_learning_dashboard view.

    This keeps the frontend Agent Performance cards in sync with the
    date range selector on the dashboard.
    """
    try:
        if not date_filter_start or not date_filter_end:
            return []

        # Pull completed/learning-tracked trades in the date window
        query = (
            db.from_('past_trades_learning_dashboard')
            .select('generated_by_agent, exit_pnl, performance_status')
            .gte('signal_created_at', date_filter_start)
            .lte('signal_created_at', date_filter_end)
        )
        result = await _execute_db_query(query.execute)

        rows = result.data or []
        if not rows:
            return []

        # Aggregate by agent_type
        by_agent: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            agent = row.get('generated_by_agent') or 'unknown'
            if agent not in by_agent:
                by_agent[agent] = {
                    'agent_type': agent,
                    'total_signals': 0,
                    'wins': 0,
                    'losses': 0,
                    'tracked_signals': 0,
                    'sum_pnl': 0.0,
                }

            bucket = by_agent[agent]
            bucket['total_signals'] += 1
            bucket['tracked_signals'] += 1  # past_trades_learning_dashboard already filters to learning_tracked=TRUE

            status = row.get('performance_status')
            if status == 'profit':
                bucket['wins'] += 1
            elif status == 'loss':
                bucket['losses'] += 1

            exit_pnl = row.get('exit_pnl')
            if exit_pnl is None:
                exit_pnl = row.get('final_pnl') or row.get('current_pnl') or 0
            bucket['sum_pnl'] += float(exit_pnl or 0)

        # Finalize metrics per agent
        agents: List[Dict[str, Any]] = []
        for agent, bucket in by_agent.items():
            wins = bucket['wins']
            losses = bucket['losses']
            total = bucket['total_signals']
            decided = wins + losses
            pnl_avg = bucket['sum_pnl'] / total if total > 0 else 0.0
            win_rate = wins / decided if decided > 0 else 0.0

            agents.append(
                {
                    'agent_type': agent,
                    'total_signals': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    # Win rate uses decided outcomes only; expiries/neutral and
                    # invalidated signals are excluded. Surface both counts so a
                    # rate built on a handful of trades cannot read as robust.
                    'win_rate_sample_size': decided,
                    'excluded_from_win_rate': total - decided,
                    'avg_pnl': pnl_avg,
                    'tracked_signals': bucket['tracked_signals'],
                }
            )

        # Sort similar to the view: by total_signals desc, then win_rate desc
        agents.sort(key=lambda a: (-a['total_signals'], -a['win_rate']))
        return agents
    except Exception as e:
        logger.error(f"Agent comparison calculation for date range failed: {e}")
        return []


# Analysis agents whose win rate comes from resolved recommendations, not tracked trades.
# Yuki is intentionally excluded here: its card is built from real trade outcomes.
_RECOMMENDATION_AGENTS = ("ryu", "sakura")


def _normalize_recommendation_horizon(horizon: Optional[str]) -> str:
    normalized = str(horizon or "auto").strip().lower()
    return normalized if normalized in {"24h", "7d"} else "auto"


def _select_recommendation_outcome(
    row: Dict[str, Any],
    horizon_mode: str,
) -> Optional[tuple[str, Any, str]]:
    if horizon_mode == "24h":
        status = row.get("horizon_24h_status")
        pnl = row.get("horizon_24h_pnl_pct")
        used_horizon = "24h"
    elif horizon_mode == "7d":
        status = row.get("horizon_7d_status")
        pnl = row.get("horizon_7d_pnl_pct")
        used_horizon = "7d"
    else:
        status = row.get("horizon_7d_status")
        pnl = row.get("horizon_7d_pnl_pct")
        used_horizon = "7d"
        if not str(status or "").strip():
            status = row.get("horizon_24h_status")
            pnl = row.get("horizon_24h_pnl_pct")
            used_horizon = "24h"

    normalized_status = str(status or "").strip().lower()
    if not normalized_status:
        return None
    return normalized_status, pnl, used_horizon


async def _calculate_recommendation_agent_comparison(
    db,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
    horizon: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Build Agent Performance cards for token-analysis agents (Ryu / Sakura).

    By default, this follows the dashboard date selector and uses the best resolved
    outcome per recommendation: 7d when it has matured, otherwise 24h. A specific
    horizon can still be supplied by diagnostics or older clients.

    Win rate is directional only: HOLD recommendations ('neutral') are tracked in the
    totals but excluded from wins/losses and from average PnL.
    """
    try:
        horizon_mode = _normalize_recommendation_horizon(horizon)

        query = (
            db.from_('agent_recommendation_tracking')
            .select(
                'id,agent_type,token_symbol,recommendation,direction,confidence,'
                f'entry_price,target_1,stop_loss,metadata,created_at,'
                'horizon_24h_status,horizon_24h_pnl_pct,'
                'horizon_7d_status,horizon_7d_pnl_pct'
            )
            .in_('agent_type', list(_RECOMMENDATION_AGENTS))
        )
        if date_filter_start and date_filter_end:
            query = query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)

        raw_rows = (await _execute_db_query(query.execute)).data or []
        rows: List[Dict[str, Any]] = []
        for row in raw_rows:
            selected = _select_recommendation_outcome(row, horizon_mode)
            if not selected:
                continue
            status, pnl, used_horizon = selected
            rows.append({
                **row,
                "_outcome_status": status,
                "_outcome_pnl_pct": pnl,
                "_outcome_horizon": used_horizon,
            })

        if not rows:
            return []

        # One user can repeat the same card request. Collapse identical plans so a
        # refresh or double-submit cannot masquerade as an independent trial.
        unique_rows: List[Dict[str, Any]] = []
        seen_setups = set()
        for row in rows:
            setup_key = (
                str(row.get('agent_type') or '').lower(),
                str(row.get('token_symbol') or '').upper(),
                str(row.get('recommendation') or '').upper(),
                str(row.get('direction') or '').lower(),
                str(row.get('entry_price')),
                str(row.get('target_1')),
                str(row.get('stop_loss')),
            )
            if setup_key in seen_setups:
                continue
            seen_setups.add(setup_key)
            unique_rows.append(row)

        raw_counts: Dict[str, int] = {}
        for row in rows:
            agent = str(row.get('agent_type') or 'unknown').lower()
            raw_counts[agent] = raw_counts.get(agent, 0) + 1

        by_agent: Dict[str, Dict[str, Any]] = {}
        for row in unique_rows:
            agent = (row.get('agent_type') or 'unknown').lower()
            bucket = by_agent.setdefault(agent, {
                'agent_type': agent,
                'metric_source': 'recommendation_tracking',
                'horizon': horizon_mode if horizon_mode != "auto" else "selected_range",
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'neutral': 0,
                'non_directional': 0,
                'sum_pnl': 0.0,
                'directional': 0,
                'invalid_plans': 0,
                'sum_confidence': 0.0,
                'resolved_24h': 0,
                'resolved_7d': 0,
            })

            bucket['total_signals'] += 1
            used_horizon = row.get("_outcome_horizon")
            if used_horizon == "24h":
                bucket['resolved_24h'] += 1
            elif used_horizon == "7d":
                bucket['resolved_7d'] += 1

            status = str(row.get("_outcome_status") or '').lower()
            direction = str(row.get('direction') or 'neutral').lower()
            try:
                entry = float(row.get('entry_price'))
                target = float(row.get('target_1'))
                stop = float(row.get('stop_loss'))
            except (TypeError, ValueError):
                entry = target = stop = 0.0
            valid_plan = (
                (direction == 'long' and target > entry > stop > 0)
                or (direction == 'short' and 0 < target < entry < stop)
            )

            if direction in {'long', 'short'} and not valid_plan:
                bucket['invalid_plans'] += 1
                continue
            if status == 'win':
                bucket['wins'] += 1
            elif status == 'loss':
                bucket['losses'] += 1
            else:  # 'neutral' (HOLD) — tracked but excluded from win rate
                bucket['neutral'] += 1
                bucket['non_directional'] += 1
                continue

            pnl = _safe_float(row.get("_outcome_pnl_pct"), 0.0) or 0.0
            bucket['sum_pnl'] += pnl
            bucket['directional'] += 1
            bucket['sum_confidence'] += _safe_float(row.get('confidence'), 0.0) or 0.0

        agents: List[Dict[str, Any]] = []
        for agent, bucket in by_agent.items():
            wins = bucket['wins']
            losses = bucket['losses']
            directional = bucket['directional']
            decided = wins + losses
            win_rate = wins / decided if decided > 0 else 0.0
            if decided > 0:
                z = 1.96
                denominator = 1 + z * z / decided
                center = (win_rate + z * z / (2 * decided)) / denominator
                margin = z * math.sqrt(
                    (win_rate * (1 - win_rate) / decided) + (z * z / (4 * decided * decided))
                ) / denominator
                ci_low = max(0.0, center - margin)
                ci_high = min(1.0, center + margin)
            else:
                ci_low = ci_high = 0.0
            agents.append({
                'agent_type': agent,
                'metric_source': 'recommendation_tracking',
                'horizon': bucket['horizon'],
                'outcome_horizon_counts': {
                    '24h': bucket['resolved_24h'],
                    '7d': bucket['resolved_7d'],
                },
                'total_signals': bucket['total_signals'],
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'win_rate_sample_size': decided,
                'excluded_from_win_rate': bucket['total_signals'] - decided,
                'avg_pnl': bucket['sum_pnl'] / directional if directional > 0 else 0.0,
                'tracked_signals': raw_counts.get(agent, bucket['total_signals']),
                'unique_setups': decided,
                'unique_recommendations': bucket['total_signals'],
                'hold_rate': bucket['non_directional'] / bucket['total_signals'] if bucket['total_signals'] else 0.0,
                'invalid_plans_excluded': bucket['invalid_plans'],
                'confidence_interval_low': ci_low,
                'confidence_interval_high': ci_high,
                'avg_confidence': bucket['sum_confidence'] / directional if directional else 0.0,
                'sample_status': 'established' if decided >= 30 else 'early' if decided >= 10 else 'insufficient',
            })

        agents.sort(key=lambda a: (-a['total_signals'], -a['win_rate']))
        return agents
    except Exception as e:
        logger.error(f"Recommendation agent comparison calculation failed: {e}")
        return []


async def _fetch_recommendation_history(
    db,
    agent_type: str,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
    limit: int = 80,
    horizon: str = "auto",
) -> List[Dict[str, Any]]:
    """Return recent persisted recommendation snapshots for one analysis agent."""
    try:
        horizon_mode = _normalize_recommendation_horizon(horizon)
        query = (
            db.from_('agent_recommendation_tracking')
            .select(
                'id,agent_type,token_symbol,token_name,recommendation,direction,confidence,'
                'entry_price,target_1,target_2,stop_loss,metadata,created_at,'
                'horizon_24h_status,horizon_24h_pnl_pct,'
                'horizon_7d_status,horizon_7d_pnl_pct'
            )
            .eq('agent_type', str(agent_type or '').lower())
        )
        if date_filter_start and date_filter_end:
            query = query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)

        res = await _execute_db_query(lambda: query.order('created_at', desc=True).limit(limit).execute())
        rows = res.data or []
        history: List[Dict[str, Any]] = []
        for row in rows:
            metadata = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
            selected = _select_recommendation_outcome(row, horizon_mode)
            history.append({
                **row,
                'trade_intent': metadata.get('trade_intent'),
                'valid_directional_plan': metadata.get('valid_directional_plan'),
                'performance_exclusion_reason': metadata.get('performance_exclusion_reason'),
                'selected_outcome_status': selected[0] if selected else None,
                'selected_outcome_pnl_pct': selected[1] if selected else None,
                'selected_outcome_horizon': selected[2] if selected else None,
            })
        return history
    except Exception as e:
        logger.error(f"Recommendation history fetch failed for {agent_type}: {e}")
        return []


def _counterfactual_min_bad_pnl_pct() -> float:
    return max(0.0, float(getattr(settings, "REJECT_COUNTERFACTUAL_MIN_BAD_PNL_PCT", 1.0) or 1.0))


def _counterfactual_min_good_loss_pct() -> float:
    return max(0.0, float(getattr(settings, "REJECT_COUNTERFACTUAL_MIN_GOOD_LOSS_PCT", 0.5) or 0.5))


def _classify_reject_quality_from_counterfactual(terminal_status: str, pnl_pct: float) -> str:
    status = str(terminal_status or "").lower()
    pnl = float(pnl_pct or 0.0)
    min_bad_pnl = _counterfactual_min_bad_pnl_pct()
    min_good_loss = _counterfactual_min_good_loss_pct()
    if status == "hit_stop_loss":
        return "good_reject" if pnl <= -min_good_loss else "ambiguous_reject"
    if status in {"hit_target_1", "hit_target_2"}:
        return "bad_reject" if pnl >= min_bad_pnl else "ambiguous_reject"
    if status == "expired":
        return "good_reject" if pnl <= -min_good_loss else "ambiguous_reject"
    return "unknown"


def _normalize_reject_quality(insights: Dict[str, Any]) -> str:
    terminal_status = str(insights.get("counterfactual_status") or "").lower()
    if terminal_status in {"hit_stop_loss", "hit_target_1", "hit_target_2", "expired"}:
        try:
            pnl_pct = float(insights.get("counterfactual_pnl_pct") or 0.0)
        except (TypeError, ValueError):
            pnl_pct = 0.0
        return _classify_reject_quality_from_counterfactual(terminal_status, pnl_pct)

    reject_quality = str(insights.get('reject_quality') or '').lower()
    if reject_quality in {"good_reject", "bad_reject", "ambiguous_reject"}:
        return reject_quality
    return "unknown"


def _counterfactual_display_quality(
    candidate_source: str,
    reject_quality: str,
    terminal_status: str,
    pnl_pct: float,
) -> Dict[str, str]:
    source = str(candidate_source or "").lower()
    status = str(terminal_status or "").lower()
    quality = str(reject_quality or "unknown").lower()
    if source == "rule_counterfactual":
        if status in {"hit_target_1", "hit_target_2"} and quality == "bad_reject":
            return {"label": "rule beat AI", "tone": "risk", "context": "rule_vs_ai"}
        if quality == "good_reject":
            return {"label": "AI veto right", "tone": "edge", "context": "rule_vs_ai"}
        if quality == "ambiguous_reject":
            return {"label": "unclear rule veto", "tone": "caution", "context": "rule_vs_ai"}
        return {"label": "unknown", "tone": "neutral", "context": "rule_vs_ai"}

    if quality == "good_reject":
        return {"label": "good reject", "tone": "edge", "context": "ai_gate"}
    if quality == "bad_reject":
        return {"label": "bad reject", "tone": "risk", "context": "ai_gate"}
    if quality == "ambiguous_reject":
        return {"label": "ambiguous", "tone": "caution", "context": "ai_gate"}
    return {"label": "unknown", "tone": "neutral", "context": "ai_gate"}


async def _calculate_rejection_validation_stats(
    db,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize counterfactual outcomes for rejected-signal tracking."""
    try:
        # Read recent matching rows (up to 2,000 candidates for fast analytics rendering)
        batch_size = 1000
        offset = 0
        rows: List[Dict[str, Any]] = []
        total_candidates = 0
        max_batches = 2  # Capped at 2 batches (2,000 rows max) for fast response times

        for batch_idx in range(max_batches):
            query = (
                db.from_('agent_learning_insights')
                .select('id, insights, timestamp', count='exact' if batch_idx == 0 else None)
                .eq('learning_type', 'signal_rejected_candidate')
            )
            if date_filter_start and date_filter_end:
                query = query.gte('timestamp', date_filter_start).lte('timestamp', date_filter_end)

            batch_query = query.order('timestamp', desc=True).range(offset, offset + batch_size - 1)
            batch_result = await _execute_db_query(batch_query.execute)
            batch_rows = batch_result.data or []

            if batch_idx == 0:
                total_candidates = batch_result.count or 0

            if not batch_rows:
                break

            rows.extend(batch_rows)
            if len(batch_rows) < batch_size:
                break
            offset += batch_size

        if total_candidates == 0:
            total_candidates = len(rows)
        pending_candidates = 0
        resolved_candidates = 0
        good_rejects = 0
        bad_rejects = 0
        ambiguous_rejects = 0
        unknown_resolutions = 0
        daily_buckets: Dict[str, Dict[str, int]] = {}
        source_distribution: Dict[str, int] = {}
        rejection_code_distribution: Dict[str, int] = {}
        resolved_status_distribution: Dict[str, int] = {}
        recent_resolutions: List[Dict[str, Any]] = []

        for row in rows:
            insights = row.get('insights')
            if not isinstance(insights, dict):
                continue

            day_key = str(row.get('timestamp') or '')[:10]
            if day_key:
                daily_buckets.setdefault(day_key, {
                    "date": day_key,
                    "total": 0,
                    "resolved": 0,
                    "good_rejects": 0,
                    "bad_rejects": 0,
                    "ambiguous_rejects": 0,
                    "unknown": 0,
                    "pending": 0,
                })
                daily_buckets[day_key]["total"] += 1

            candidate_source = str(insights.get('candidate_source') or 'unknown').lower()
            source_distribution[candidate_source] = source_distribution.get(candidate_source, 0) + 1

            rejection_code = str(insights.get('rejection_code') or 'unknown').lower()
            rejection_code_distribution[rejection_code] = rejection_code_distribution.get(rejection_code, 0) + 1

            tracking_status = str(insights.get('tracking_status') or '').lower()
            if tracking_status == 'pending':
                pending_candidates += 1
                if day_key and day_key in daily_buckets:
                    daily_buckets[day_key]["pending"] += 1
                continue
            if tracking_status != 'resolved':
                continue

            resolved_candidates += 1
            if day_key and day_key in daily_buckets:
                daily_buckets[day_key]["resolved"] += 1
            terminal_status = str(insights.get("counterfactual_status") or "unknown").lower()
            resolved_status_distribution[terminal_status] = resolved_status_distribution.get(terminal_status, 0) + 1
            reject_quality = _normalize_reject_quality(insights)
            try:
                counterfactual_pnl = float(insights.get("counterfactual_pnl_pct") or 0.0)
            except (TypeError, ValueError):
                counterfactual_pnl = 0.0
            display_quality = _counterfactual_display_quality(
                candidate_source,
                reject_quality,
                terminal_status,
                counterfactual_pnl,
            )

            # ---------------------------------------------------------------
            # Gate quality pool: ai_directional ONLY.
            # rule_counterfactual entries are tracked separately below and
            # must NOT influence reject_precision_percent, which measures
            # whether our risk gates block good AI-agreed trades.
            # ---------------------------------------------------------------
            if candidate_source == "ai_directional":
                if reject_quality == 'good_reject':
                    good_rejects += 1
                    if day_key and day_key in daily_buckets:
                        daily_buckets[day_key]["good_rejects"] += 1
                elif reject_quality == 'bad_reject':
                    bad_rejects += 1
                    if day_key and day_key in daily_buckets:
                        daily_buckets[day_key]["bad_rejects"] += 1
                elif reject_quality == 'ambiguous_reject':
                    ambiguous_rejects += 1
                    if day_key and day_key in daily_buckets:
                        daily_buckets[day_key]["ambiguous_rejects"] += 1
                else:
                    unknown_resolutions += 1
                    if day_key and day_key in daily_buckets:
                        daily_buckets[day_key]["unknown"] += 1

            if len(recent_resolutions) < 25:
                recent_resolutions.append({
                    "timestamp": row.get("timestamp"),
                    "symbol": insights.get("symbol"),
                    "direction": insights.get("direction"),
                    "reject_quality": reject_quality or "unknown",
                    "decision_quality_label": display_quality["label"],
                    "decision_quality_tone": display_quality["tone"],
                    "decision_quality_context": display_quality["context"],
                    "counterfactual_status": terminal_status,
                    "counterfactual_pnl_pct": insights.get("counterfactual_pnl_pct"),
                    "counterfactual_pnl_pct_leveraged": insights.get("counterfactual_pnl_pct_leveraged"),
                    "candidate_source": insights.get("candidate_source"),
                    "rejection_code": insights.get("rejection_code"),
                })

        precision_denom = good_rejects + bad_rejects
        reject_precision_percent = round((good_rejects / precision_denom) * 100, 2) if precision_denom > 0 else 0.0
        bad_reject_rate_percent = round((bad_rejects / precision_denom) * 100, 2) if precision_denom > 0 else 0.0
        timeline = sorted(daily_buckets.values(), key=lambda d: d["date"])
        for bucket in timeline:
            denom = bucket["good_rejects"] + bucket["bad_rejects"]
            bucket["bad_reject_rate_percent"] = round((bucket["bad_rejects"] / denom) * 100, 2) if denom > 0 else 0.0
            bucket["reject_precision_percent"] = round((bucket["good_rejects"] / denom) * 100, 2) if denom > 0 else 0.0

        source_mix = [
            {"source": source, "count": count}
            for source, count in sorted(source_distribution.items(), key=lambda item: item[1], reverse=True)
        ]
        rejection_code_mix = [
            {"rejection_code": code, "count": count}
            for code, count in sorted(rejection_code_distribution.items(), key=lambda item: item[1], reverse=True)[:8]
        ]
        resolved_status_mix = [
            {"status": status, "count": count}
            for status, count in sorted(resolved_status_distribution.items(), key=lambda item: item[1], reverse=True)
        ]

        # ---------------------------------------------------------------
        # AI-vs-rule accuracy: rule_counterfactual entries ONLY.
        # Answers: "When the rule engine said LONG/SHORT and the AI said
        # HOLD, who was right?" — grouped by market regime so the bot can
        # learn when to trust the AI's HOLD decision and when to push back.
        # rule_was_right  → market moved in rule_direction (hit target)
        # ai_was_right    → market went against rule_direction (hit SL /
        #                   expired flat or negative)
        # ---------------------------------------------------------------
        ai_vs_rule: Dict[str, Any] = {}
        for row in rows:
            insights = row.get('insights')
            if not isinstance(insights, dict):
                continue
            if str(insights.get('candidate_source') or '').lower() != 'rule_counterfactual':
                continue
            if str(insights.get('tracking_status') or '').lower() != 'resolved':
                continue

            regime = str(insights.get('regime_type') or 'unknown').strip() or 'unknown'
            terminal_status = str(insights.get('counterfactual_status') or '').lower()
            try:
                pnl = float(insights.get('counterfactual_pnl_pct') or 0.0)
            except (TypeError, ValueError):
                pnl = 0.0

            # rule_was_right: the market did what the rule engine expected
            if terminal_status in {'hit_target_1', 'hit_target_2'}:
                verdict = 'rule_was_right'
            elif terminal_status == 'hit_stop_loss':
                verdict = 'ai_was_right'
            elif terminal_status == 'expired':
                verdict = 'ai_was_right' if pnl <= 0 else 'rule_was_right'
            else:
                verdict = 'unknown'

            if verdict == 'unknown':
                continue

            bucket = ai_vs_rule.setdefault(regime, {
                'regime': regime,
                'total': 0,
                'rule_was_right': 0,
                'ai_was_right': 0,
                'rule_accuracy_pct': None,
                'ai_accuracy_pct': None,
            })
            bucket['total'] += 1
            if verdict == 'rule_was_right':
                bucket['rule_was_right'] += 1
            else:
                bucket['ai_was_right'] += 1

        # Compute accuracy percentages per regime
        ai_vs_rule_list: List[Dict[str, Any]] = []
        total_rule_right = 0
        total_ai_right = 0
        for regime_key, bucket in ai_vs_rule.items():
            n = bucket['total']
            rr = bucket['rule_was_right']
            ar = bucket['ai_was_right']
            bucket['rule_accuracy_pct'] = round(rr / n * 100, 2) if n > 0 else None
            bucket['ai_accuracy_pct'] = round(ar / n * 100, 2) if n > 0 else None
            total_rule_right += rr
            total_ai_right += ar
            ai_vs_rule_list.append(bucket)

        ai_vs_rule_list.sort(key=lambda b: b['total'], reverse=True)
        total_ai_rule_resolved = total_rule_right + total_ai_right
        ai_vs_rule_summary = {
            'total_resolved': total_ai_rule_resolved,
            'rule_was_right': total_rule_right,
            'ai_was_right': total_ai_right,
            'rule_accuracy_pct': round(total_rule_right / total_ai_rule_resolved * 100, 2) if total_ai_rule_resolved > 0 else None,
            'ai_accuracy_pct': round(total_ai_right / total_ai_rule_resolved * 100, 2) if total_ai_rule_resolved > 0 else None,
            'note': (
                'When rule engine says LONG/SHORT but AI says HOLD: '
                'rule_was_right means the market moved in the rule direction (missed opportunity); '
                'ai_was_right means AI was correct to veto.'
            ),
            'by_regime': ai_vs_rule_list,
        }

        return {
            "total_candidates": total_candidates,
            "pending_candidates": pending_candidates,
            "resolved_candidates": resolved_candidates,
            # Gate quality — ai_directional entries only
            "good_rejects": good_rejects,
            "bad_rejects": bad_rejects,
            "ambiguous_rejects": ambiguous_rejects,
            "unknown_resolutions": unknown_resolutions,
            "informative_resolved": precision_denom,
            "reject_precision_percent": reject_precision_percent,
            "bad_reject_rate_percent": bad_reject_rate_percent,
            "gate_quality_note": "reject_precision_percent counts ai_directional candidates only",
            "timeline": timeline,
            "source_mix": source_mix,
            "rejection_code_mix": rejection_code_mix,
            "resolved_status_mix": resolved_status_mix,
            "recent_resolutions": recent_resolutions,
            # AI calibration — rule_counterfactual entries only
            "ai_vs_rule_accuracy": ai_vs_rule_summary,
        }
    except Exception as e:
        logger.warning(f"Rejected-signal validation summary failed: {e}")
        return {
            "total_candidates": 0,
            "pending_candidates": 0,
            "resolved_candidates": 0,
            "good_rejects": 0,
            "bad_rejects": 0,
            "ambiguous_rejects": 0,
            "unknown_resolutions": 0,
            "informative_resolved": 0,
            "reject_precision_percent": 0.0,
            "bad_reject_rate_percent": 0.0,
            "gate_quality_note": "reject_precision_percent counts ai_directional candidates only",
            "timeline": [],
            "source_mix": [],
            "rejection_code_mix": [],
            "resolved_status_mix": [],
            "recent_resolutions": [],
            "ai_vs_rule_accuracy": {},
            "error": str(e),
        }


async def _get_confidence_gate_tuning_status(
    db,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize confidence gate auto-tuning snapshots for dashboard charts."""
    try:
        query = (
            db.from_('agent_learning_insights')
            .select('id, agent_type, timestamp, insights, performance_impact')
            .eq('learning_type', 'threshold_gate_adjustment')
            .eq('agent_type', 'yuki')
        )
        if date_filter_start and date_filter_end:
            query = query.gte('timestamp', date_filter_start).lte('timestamp', date_filter_end)

        result_query = query.order('timestamp', desc=True).limit(250)
        result = await _execute_db_query(result_query.execute)
        rows = result.data or []
        if not rows:
            return {
                "enabled": bool(getattr(settings, "REJECT_TUNING_ENABLED", False)),
                "current_adjustment": 0.0,
                "target_bad_reject_rate": float(getattr(settings, "REJECT_TUNING_TARGET_BAD_RATE", 0.30)),
                "latest_bad_reject_rate": None,
                "sample_resolved": 0,
                "latest_updated_at": None,
                "timeline": [],
            }

        timeline: List[Dict[str, Any]] = []
        for row in reversed(rows):
            insights = row.get('insights') if isinstance(row.get('insights'), dict) else {}
            performance_impact = row.get('performance_impact') if isinstance(row.get('performance_impact'), dict) else {}

            adjustment = performance_impact.get('confidence_adjustment', insights.get('adjustment', 0.0))
            bad_rate = performance_impact.get('bad_reject_rate', insights.get('bad_reject_rate'))
            target_rate = performance_impact.get('target_bad_reject_rate', insights.get('target_bad_reject_rate'))
            avg_reject_pnl = insights.get('avg_reject_pnl_pct_leveraged')
            target_reject_pnl = insights.get('target_reject_pnl_pct_leveraged')
            raw_bad_rate_adjustment = insights.get('raw_bad_rate_adjustment')
            raw_pnl_adjustment = insights.get('raw_pnl_adjustment')
            raw_adjustment = insights.get('raw_adjustment')

            try:
                adjustment = float(adjustment or 0.0)
            except (TypeError, ValueError):
                adjustment = 0.0
            try:
                bad_rate = float(bad_rate) if bad_rate is not None else None
            except (TypeError, ValueError):
                bad_rate = None
            try:
                target_rate = float(target_rate) if target_rate is not None else None
            except (TypeError, ValueError):
                target_rate = None
            try:
                avg_reject_pnl = float(avg_reject_pnl) if avg_reject_pnl is not None else None
            except (TypeError, ValueError):
                avg_reject_pnl = None
            try:
                target_reject_pnl = float(target_reject_pnl) if target_reject_pnl is not None else None
            except (TypeError, ValueError):
                target_reject_pnl = None
            try:
                raw_bad_rate_adjustment = float(raw_bad_rate_adjustment) if raw_bad_rate_adjustment is not None else None
            except (TypeError, ValueError):
                raw_bad_rate_adjustment = None
            try:
                raw_pnl_adjustment = float(raw_pnl_adjustment) if raw_pnl_adjustment is not None else None
            except (TypeError, ValueError):
                raw_pnl_adjustment = None
            try:
                raw_adjustment = float(raw_adjustment) if raw_adjustment is not None else None
            except (TypeError, ValueError):
                raw_adjustment = None

            timeline.append({
                "timestamp": row.get('timestamp'),
                "date": str(row.get('timestamp') or '')[:10],
                "confidence_adjustment": round(adjustment, 5),
                "bad_reject_rate_percent": round((bad_rate or 0.0) * 100, 2) if bad_rate is not None else None,
                "target_bad_reject_rate_percent": round((target_rate or 0.0) * 100, 2) if target_rate is not None else None,
                "avg_reject_pnl_pct_leveraged": round(avg_reject_pnl, 4) if avg_reject_pnl is not None else None,
                "target_reject_pnl_pct_leveraged": round(target_reject_pnl, 4) if target_reject_pnl is not None else None,
                "raw_bad_rate_adjustment": round(raw_bad_rate_adjustment, 5) if raw_bad_rate_adjustment is not None else None,
                "raw_pnl_adjustment": round(raw_pnl_adjustment, 5) if raw_pnl_adjustment is not None else None,
                "raw_adjustment": round(raw_adjustment, 5) if raw_adjustment is not None else None,
                "resolved_candidates": int(insights.get('resolved_candidates') or 0),
                "informative_resolved": int(insights.get('informative_resolved') or 0),
                "applied": bool(insights.get('applied', False)),
            })

        latest = timeline[-1] if timeline else {}
        return {
            "enabled": bool(getattr(settings, "REJECT_TUNING_ENABLED", False)),
            "current_adjustment": latest.get("confidence_adjustment", 0.0),
            "target_bad_reject_rate": float(getattr(settings, "REJECT_TUNING_TARGET_BAD_RATE", 0.30)),
            "latest_bad_reject_rate": latest.get("bad_reject_rate_percent"),
            "sample_resolved": latest.get("informative_resolved", 0),
            "latest_updated_at": latest.get("timestamp"),
            "timeline": timeline[-60:],
        }
    except Exception as e:
        logger.warning(f"Confidence gate tuning summary failed: {e}")
        return {
            "enabled": bool(getattr(settings, "REJECT_TUNING_ENABLED", False)),
            "current_adjustment": 0.0,
            "target_bad_reject_rate": float(getattr(settings, "REJECT_TUNING_TARGET_BAD_RATE", 0.30)),
            "latest_bad_reject_rate": None,
            "sample_resolved": 0,
            "latest_updated_at": None,
            "timeline": [],
            "error": str(e),
        }


async def _calculate_agent_stats(db, agent_type: str) -> Dict[str, Any]:
    """Calculate performance statistics for a specific agent."""
    try:
        # Get recent signals (last 30 days)
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        # Signals with tracking data
        signals_query = db.from_('platform_signals').select('''
            id, confidence, learning_tracked
        ''').eq('generated_by_agent', agent_type).gte('created_at', thirty_days_ago)
        signals_result = await _execute_db_query(signals_query.execute)

        signals = signals_result.data or []

        # User tracking results
        signal_ids = [s['id'] for s in signals]
        if signal_ids:
            tracking_query = db.from_('user_platform_signal_tracking').select('''
                platform_signal_id, profit_loss_percentage, user_outcome
            ''').in_('platform_signal_id', signal_ids)
            tracking_result = await _execute_db_query(tracking_query.execute)

            tracking_data = tracking_result.data or []

            # Calculate stats
            total_signals = len(signals)
            tracked_signals = len([s for s in signals if s.get('learning_tracked')])

            wins = len([t for t in tracking_data if t.get('user_outcome') == 'win'])
            losses = len([t for t in tracking_data if t.get('user_outcome') == 'loss'])

            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            avg_confidence = sum(s.get('confidence', 0) for s in signals) / len(signals) if signals else 0
            avg_pnl = sum(t.get('profit_loss_percentage', 0) for t in tracking_data) / len(tracking_data) if tracking_data else 0

            return {
                "total_signals_30d": total_signals,
                "tracked_signals_30d": tracked_signals,
                "win_rate_30d": round(win_rate, 4),
                "avg_confidence_30d": round(avg_confidence, 4),
                "avg_pnl_30d": round(avg_pnl, 4),
                "wins": wins,
                "losses": losses
            }
        else:
            return {"no_data": "No signals found for this agent in the last 30 days"}

    except Exception as e:
        logger.error(f"Agent stats calculation failed: {e}")
        return {"error": str(e)}


async def _analyze_patterns(db, patterns: List[Dict]) -> Dict[str, Any]:
    """Analyze pattern performance data."""
    try:
        if not patterns:
            return {"no_patterns": True}

        # Best performing patterns
        best_patterns = sorted(patterns, key=lambda p: p.get('success_rate', 0), reverse=True)[:5]

        # Patterns needing improvement
        poor_patterns = [p for p in patterns if p.get('success_rate', 0) < 0.5]

        # Average metrics
        avg_success_rate = sum(p.get('success_rate', 0) for p in patterns) / len(patterns)
        valid_sharpes = []
        for pattern in patterns:
            sharpe = pattern.get('sharpe_ratio')
            if sharpe is None:
                continue
            try:
                valid_sharpes.append(float(sharpe))
            except (TypeError, ValueError):
                continue
        avg_sharpe = (sum(valid_sharpes) / len(valid_sharpes)) if valid_sharpes else 0.0

        return {
            "best_patterns": best_patterns,
            "poor_patterns": poor_patterns,
            "avg_success_rate": round(avg_success_rate, 4),
            "avg_sharpe_ratio": round(avg_sharpe, 4),
            "total_patterns": len(patterns)
        }

    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        return {"error": str(e)}


async def _check_recent_activity(db) -> Dict[str, Any]:
    """Check for recent signal and learning activity."""
    try:
        # Recent signals (last hour)
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        recent_signals_query = db.from_('platform_signals').select('*', count='exact').gte('created_at', one_hour_ago)
        recent_signals = await _execute_db_query(recent_signals_query.execute)

        # Recent learning insights (last hour)
        recent_insights_query = db.from_('agent_learning_insights').select('*', count='exact').gte('timestamp', one_hour_ago)
        recent_insights = await _execute_db_query(recent_insights_query.execute)

        return {
            "signals_last_hour": recent_signals.count or 0,
            "insights_last_hour": recent_insights.count or 0,
            "activity_status": "active" if (recent_signals.count or 0) > 0 else "quiet"
        }

    except Exception as e:
        return {"error": str(e)}


def _humanize_insight_type(learning_type: str) -> str:
    """Turn a raw learning_type slug into a readable title."""
    labels = {
        'threshold_gate_adjustment': 'Confidence gate tuning',
        'signal_schedule_auto_apply': 'Schedule auto-apply',
        'signal_generation_active_schedule': 'Active signal schedule',
        'signal_schedule_optimization': 'Schedule optimization',
        'signal_generation_schedule_run': 'Signal run',
        'signal_rejected_candidate': 'Signal rejected',
        'temporal_pattern_analysis': 'Hourly timing pattern',
    }
    return labels.get(learning_type, str(learning_type).replace('_', ' ').title())


def _format_hour_utc(hour: Any) -> str:
    try:
        return f"{int(hour) % 24:02d}:00 UTC"
    except (TypeError, ValueError):
        return f"{hour}:00 UTC"


def _format_signed_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _summarize_insight(learning_type: str, data: Any) -> str:
    """Render a learning insight's JSONB payload as one plain-English sentence.

    The raw payloads are large nested stats dicts; dumping them verbatim is
    unreadable. Each known type gets a hand-written summary of the few fields
    that actually matter; unknown types fall back to a short key/value preview.
    """
    if not isinstance(data, dict):
        return str(data)

    def num(key, default=0.0):
        try:
            return float(data.get(key, default) or 0)
        except (TypeError, ValueError):
            return float(default)

    try:
        if learning_type == 'threshold_gate_adjustment':
            adj = num('adjustment')
            direction = 'raised' if adj > 0.0001 else 'lowered' if adj < -0.0001 else 'held'
            verb = 'Adjusted' if data.get('applied') else 'Proposed adjusting'
            bad_rate = num('bad_reject_rate') * 100
            target = num('target_bad_reject_rate') * 100
            days = int(num('lookback_days', 30))
            return (
                f"{verb} the confidence gate — {direction} it by {adj:+.4f}. "
                f"Bad-reject rate is {bad_rate:.1f}% against a {target:.0f}% target over the last {days} days "
                f"({int(num('bad_rejects'))} bad vs {int(num('good_rejects'))} good rejects)."
            )

        if learning_type == 'signal_schedule_auto_apply':
            conf = num('confidence_score') * 100
            label = data.get('recommended_label', 'the recommended schedule')
            if data.get('applied'):
                return f"Applied a new signal schedule: {label} ({conf:.0f}% confidence)."
            reason = str(data.get('reason', '')).replace('_', ' ') or 'no change needed'
            return f"Kept the current schedule ({label}) — {reason} ({conf:.0f}% confidence)."

        if learning_type == 'signal_generation_active_schedule':
            hours = data.get('active_hours_utc') or []
            label = "/".join(_format_hour_utc(hour) for hour in hours) or "no hours"
            source = str(data.get('source') or 'Learning Dashboard').replace('_', ' ')
            return f"Active signal generation schedule: {label}. Source: {source}."

        if learning_type == 'signal_schedule_optimization':
            overall = data.get('overall', {}) if isinstance(data.get('overall'), dict) else {}
            rec = data.get('recommended_schedule', {}) if isinstance(data.get('recommended_schedule'), dict) else {}
            n = int(overall.get('n', 0) or 0)
            wr = overall.get('win_rate', 0)
            pf = overall.get('profit_factor', 0)
            label = rec.get('label', 'n/a')
            return (
                f"Reviewed {n} resolved trades (win rate {wr}%, profit factor {pf}). "
                f"Best trading hours: {label}."
            )

        if learning_type == 'signal_generation_schedule_run':
            hour = data.get('slot_hour_utc')
            hour_str = f"{int(hour):02d}:00 UTC" if hour is not None else "a scheduled"
            return f"{hour_str} run generated {int(num('signals_generated'))} signal(s)."

        if learning_type == 'signal_rejected_candidate':
            sym = data.get('symbol', 'a token')
            dirn = data.get('direction', '')
            conf = num('ai_confidence') * 100
            quality = str(data.get('reject_quality', '')).replace('_', ' ')
            cf = data.get('counterfactual_pnl_pct')
            verdict = ''
            if cf is not None:
                try:
                    move = float(cf)
                    if 'good' in quality:
                        verdict = f" Good call — it would have moved {move:+.1f}%."
                    elif 'bad' in quality:
                        verdict = f" Missed opportunity — it would have moved {move:+.1f}%."
                except (TypeError, ValueError):
                    pass
            return f"Rejected {dirn} {sym} — AI confidence {conf:.0f}% was below the threshold.{verdict}"

        if learning_type == 'temporal_pattern_analysis':
            hourly = data.get('hourly_performance') if isinstance(data.get('hourly_performance'), dict) else {}
            parsed_hours = []
            for hour, value in hourly.items():
                try:
                    parsed_hours.append((str(hour), float(value)))
                except (TypeError, ValueError):
                    continue

            if not parsed_hours:
                return "Reviewed hourly timing, but there was not enough resolved signal data to rank the hours yet."

            ranked = sorted(parsed_hours, key=lambda item: item[1], reverse=True)
            strongest = ranked[:3]
            weakest = sorted(parsed_hours, key=lambda item: item[1])[:3]

            strongest_text = ", ".join(
                f"{_format_hour_utc(hour)} ({_format_signed_pct(score)} avg)"
                for hour, score in strongest
            )
            weak_hours = [item for item in weakest if item[1] < 0]
            weak_text = ", ".join(
                f"{_format_hour_utc(hour)} ({_format_signed_pct(score)} avg)"
                for hour, score in weak_hours[:3]
            )

            if weak_text:
                return (
                    f"Signals performed best around {strongest_text}. "
                    f"Weakest hours were {weak_text}; use stricter confirmation during those windows."
                )

            return (
                f"Signals performed best around {strongest_text}. "
                "No negative hourly bucket stood out in this sample."
            )
    except Exception:
        pass

    # Fallback: short preview of the first few fields only
    preview = list(data.items())[:4]
    return ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in preview) or "No details available"


@router.get("/analytics/recent-insights")
async def get_recent_insights(days: int = Query(30, ge=1, le=90)) -> Dict[str, Any]:
    """Get recent learning insights with their content.

    Defaults to a 30-day window to match the "30d insights" dashboard stat;
    a 24h window was showing an empty modal even when the stat counted 1500+.
    """
    try:
        db = get_service_client()
        window_start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get recent insights with details. Note: this table has no `metadata`
        # column - the real JSONB extras live in `performance_impact`.
        insights_query = db.from_('agent_learning_insights').select('''
            id, agent_type, learning_level, learning_type, insights,
            confidence_score, timestamp, performance_impact
        ''').gte('timestamp', window_start).order('timestamp', desc=True).limit(50)
        insights_result = await _execute_db_query(insights_query.execute)

        insights_data = insights_result.data or []

        # Format insights for frontend display
        formatted_insights = []
        for insight in insights_data:
            learning_type = insight.get('learning_type', 'General')
            raw_content = insight.get('insights', {})
            content_str = _summarize_insight(learning_type, raw_content) or "No details available"

            formatted_insights.append({
                "id": insight.get('id'),
                "agent": insight.get('agent_type', 'Unknown').capitalize(),
                "level": insight.get('learning_level', 'N/A'),
                "type": _humanize_insight_type(learning_type),
                "content": content_str,
                "confidence": float(insight.get('confidence_score', 0) or 0) * 100,
                "timestamp": insight.get('timestamp'),
                "metadata": insight.get('performance_impact', {})
            })

        return {
            "success": True,
            "insights": formatted_insights,
            "total_count": len(formatted_insights),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch recent insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch insights: {str(e)}")


@router.get("/analytics/model-performance")
async def get_model_performance(
    date_range: Optional[str] = Query(None, description="Predefined date range: 24h, 7d, 30d, 90d"),
    start_date: Optional[str] = Query(None, description="Custom start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Custom end date (YYYY-MM-DD)")
) -> Dict[str, Any]:
    """Get LLM model performance and prompt version metrics."""
    try:
        db = get_service_client()
        date_filter_start, date_filter_end = _calculate_date_filters(date_range, start_date, end_date)
        data = await _calculate_model_performance(db, date_filter_start, date_filter_end)

        return {
            "success": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Model performance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model performance: {str(e)}")


@router.get("/analytics/prompt-optimization")
async def get_prompt_optimization_analytics(
    date_range: Optional[str] = Query(None, description="Predefined date range: 24h, 7d, 30d, 90d"),
    start_date: Optional[str] = Query(None, description="Custom start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Custom end date (YYYY-MM-DD)")
) -> Dict[str, Any]:
    """Get prompt optimization history and impact summary."""
    try:
        db = get_service_client()
        date_filter_start, date_filter_end = _calculate_date_filters(date_range, start_date, end_date)
        data = await _get_prompt_optimization_summary(db, date_filter_start, date_filter_end)

        return {
            "success": True,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Prompt optimization analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prompt optimization analytics: {str(e)}")


async def _check_learning_status(db) -> Dict[str, Any]:
    """Check learning system operational status."""
    try:
        # Check if learning tables exist and have data
        tables_status = {}

        learning_tables = [
            'agent_learning_insights',
            'platform_signal_performance_tracking',
            'signal_policy_bundles',
        ]

        for table in learning_tables:
            try:
                query = db.from_(table).select('*', count='exact').limit(1)
                result = await _execute_db_query(query.execute)
                tables_status[table] = {
                    "accessible": True,
                    "record_count": result.count or 0
                }
            except Exception as e:
                tables_status[table] = {
                    "accessible": False,
                    "error": str(e)
                }

        return {
            "learning_tables": tables_status,
            "overall_status": "operational" if all(t["accessible"] for t in tables_status.values()) else "degraded"
        }

    except Exception as e:
        return {"error": str(e)}


async def _calculate_model_performance(
    db,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate per-model and per-prompt-version performance and cost metrics."""
    try:
        query = db.from_('platform_signals').select(
            'signal_id,token_symbol,status,confidence,prompt_version,ai_confidence_breakdown,generated_by_agent,created_at,'
            'direction,entry_price,target_1,target_2,stop_loss'
        )
        if date_filter_start and date_filter_end:
            query = query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)

        rows_query = query.order('created_at', desc=True).limit(2500)
        rows_result = await _execute_db_query(rows_query.execute)
        rows = rows_result.data or []

        # Union in Ryu / Sakura recommendation tracking LLM runs
        try:
            rec_query = db.from_('agent_recommendation_tracking').select(
                'id,agent_type,token_symbol,created_at,confidence,direction,entry_price,target_1,target_2,stop_loss,metadata,horizon_7d_status,horizon_7d_pnl_pct'
            )
            if date_filter_start and date_filter_end:
                rec_query = rec_query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)
            rec_rows_query = rec_query.order('created_at', desc=True).limit(2500)
            rec_rows = (await _execute_db_query(rec_rows_query.execute)).data or []

            for rec in rec_rows:
                meta = rec.get('metadata') if isinstance(rec.get('metadata'), dict) else {}
                status_7d = (rec.get('horizon_7d_status') or '').lower()
                mapped_status = 'hit_target_1' if status_7d == 'win' else ('hit_stop_loss' if status_7d == 'loss' else 'expired')
                rec_breakdown = meta.get('ai_confidence_breakdown') if isinstance(meta.get('ai_confidence_breakdown'), dict) else {}
                rows.append({
                    'signal_id': f"rec_{rec['id']}",
                    'token_symbol': rec.get('token_symbol'),
                    'status': mapped_status,
                    'confidence': rec.get('confidence'),
                    'prompt_version': meta.get('prompt_version'),
                    'ai_confidence_breakdown': rec_breakdown or {
                        'llm_provider': meta.get('llm_provider'),
                        'llm_model': meta.get('llm_model'),
                    },
                    'generated_by_agent': rec.get('agent_type', 'ryu'),
                    'created_at': rec.get('created_at'),
                    'direction': rec.get('direction'),
                    'entry_price': rec.get('entry_price'),
                    'target_1': rec.get('target_1'),
                    'target_2': rec.get('target_2'),
                    'stop_loss': rec.get('stop_loss'),
                    'workflow_source': 'token_analysis',
                })
        except Exception as rec_err:
            logger.warning(f"Failed to union recommendation tracking rows into model performance: {rec_err}")
        if not rows:
            return _default_model_performance()

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {'1', 'true', 'yes', 'y'}
            if isinstance(value, (int, float)):
                return value != 0
            return False

        def _to_int(value: Any, default: int = 0) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

        def _extract_breakdown(raw: Any) -> Dict[str, Any]:
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return {}
            return {}

        def _extract_token_analysis_symbol(description: Any) -> Optional[str]:
            text = str(description or '').strip()
            marker = 'AI token analysis for '
            if not text.startswith(marker):
                return None
            symbol = text[len(marker):].strip().upper()
            return symbol[:32] if symbol else None

        def _billing_service_meta(service_used: Any) -> tuple[str, str, str]:
            key = str(service_used or '').strip().lower() or 'unknown'
            if key == 'token_analysis':
                return key, 'Token analysis', 'Most requested tokens'
            if key == 'trading_scanner':
                return key, 'Trading scanner', 'Most used modes'
            return key, key.replace('_', ' ').title(), 'Top requests'

        def _extract_billing_driver(service_used: str, description: Any) -> Optional[str]:
            text = str(description or '').strip()
            if service_used == 'token_analysis':
                return _extract_token_analysis_symbol(text)
            if service_used == 'trading_scanner':
                lowered = text.lower()
                if 'full mode' in lowered:
                    return 'Full Mode'
                if 'fast mode' in lowered:
                    return 'Fast Mode'
                if text:
                    return text[:48]
                return 'Unknown mode'
            return text[:48] if text else None

        def _billing_bucket(service_key: str, label: str, driver_label: str) -> Dict[str, Any]:
            return {
                'key': service_key,
                'label': label,
                'driver_label': driver_label,
                'billed_requests_count': 0,
                'billed_credits': 0,
                'latest_billed_at': None,
                'driver_counts': {},
                'driver_credits': {},
            }

        def _workflow_bucket(key: str, label: str) -> Dict[str, Any]:
            return {
                'key': key,
                'label': label,
                'observed_calls_count': 0,
                'telemetry_tracked_calls_count': 0,
                'billed_requests_count': 0,
                'cached_calls_count': 0,
                'challenger_calls_count': 0,
                'primary_prompt_tokens': 0,
                'primary_completion_tokens': 0,
                'challenger_prompt_tokens': 0,
                'challenger_completion_tokens': 0,
                'primary_cost_usd': 0.0,
                'challenger_cost_usd': 0.0,
                'total_estimated_cost_usd': 0.0,
                'models': {},
                'agent_counts': {},
                'symbol_counts': {},
            }

        def _estimated_terminal_return_pct(row: Dict[str, Any], status: str) -> Optional[float]:
            entry = _to_float(row.get('entry_price'))
            if entry <= 0:
                return None

            direction = str(row.get('direction') or '').upper()
            if direction not in {'LONG', 'SHORT'}:
                return None

            exit_price = None
            if status == 'hit_target_1':
                exit_price = _to_float(row.get('target_1'))
            elif status == 'hit_target_2':
                exit_price = _to_float(row.get('target_2'))
            elif status == 'hit_stop_loss':
                exit_price = _to_float(row.get('stop_loss'))
            elif status in {'expired', 'cancelled', 'invalidated'}:
                # Treat non-hit terminal exits as neutral for model-level attribution.
                return 0.0

            if not exit_price or exit_price <= 0:
                return None

            if direction == 'LONG':
                return ((exit_price - entry) / entry) * 100.0
            return ((entry - exit_price) / entry) * 100.0

        model_buckets: Dict[str, Dict[str, Any]] = {}
        prompt_buckets: Dict[str, Dict[str, Any]] = {}
        workflow_buckets: Dict[str, Dict[str, Any]] = {
            'platform_signals': _workflow_bucket('platform_signals', 'Platform signals'),
            'token_analysis': _workflow_bucket('token_analysis', 'Token analysis'),
        }
        billing_buckets: Dict[str, Dict[str, Any]] = {}
        terminal_statuses = {'hit_target_1', 'hit_target_2', 'hit_stop_loss', 'expired', 'cancelled', 'invalidated'}
        range_primary_cost_total = 0.0
        range_challenger_cost_total = 0.0
        model_attributed_signals = 0
        model_unattributed_signals = 0
        prompt_attributed_signals = 0
        prompt_unattributed_signals = 0

        for row in rows:
            status = str(row.get('status') or '').lower()
            confidence = _to_float(row.get('confidence'))
            breakdown = _extract_breakdown(row.get('ai_confidence_breakdown'))
            workflow_key = str(row.get('workflow_source') or 'platform_signals').strip() or 'platform_signals'
            workflow = workflow_buckets.setdefault(
                workflow_key,
                _workflow_bucket(workflow_key, workflow_key.replace('_', ' ').title()),
            )

            llm_provider_raw = breakdown.get('llm_provider')
            llm_model_raw = breakdown.get('llm_model')
            prompt_version_raw = row.get('prompt_version') or breakdown.get('prompt_version')

            llm_provider = str(llm_provider_raw).strip() if llm_provider_raw is not None else ''
            llm_model = str(llm_model_raw).strip() if llm_model_raw is not None else ''
            prompt_version = str(prompt_version_raw).strip() if prompt_version_raw is not None else ''
            has_model_attribution = bool(llm_provider and llm_model)
            has_prompt_attribution = bool(prompt_version)
            p_tokens = max(_to_int(breakdown.get('llm_prompt_tokens')), 0)
            c_tokens = max(_to_int(breakdown.get('llm_completion_tokens')), 0)
            cp_tokens = max(_to_int(breakdown.get('challenger_prompt_tokens')), 0)
            cc_tokens = max(_to_int(breakdown.get('challenger_completion_tokens')), 0)
            llm_cache_hit = _to_bool(breakdown.get('llm_cache_hit'))

            if has_model_attribution:
                model_attributed_signals += 1
            else:
                model_unattributed_signals += 1
            if has_prompt_attribution:
                prompt_attributed_signals += 1
            else:
                prompt_unattributed_signals += 1

            model_key = f"{llm_provider}:{llm_model}" if has_model_attribution else None

            primary_cost = max(0.0, _to_float(breakdown.get('llm_estimated_cost_usd')))
            if primary_cost <= 0.0:
                if p_tokens > 0 or c_tokens > 0:
                    primary_cost = (
                        (p_tokens / 1_000_000.0) * getattr(settings, "PRIMARY_LLM_INPUT_COST_PER_1M", 0.435)
                        + (c_tokens / 1_000_000.0) * getattr(settings, "PRIMARY_LLM_OUTPUT_COST_PER_1M", 0.87)
                    )

            challenger_cost = max(0.0, _to_float(breakdown.get('challenger_estimated_cost_usd')))
            if challenger_cost <= 0.0:
                if cp_tokens > 0 or cc_tokens > 0:
                    challenger_cost = (
                        (cp_tokens / 1_000_000.0) * getattr(settings, "CHALLENGER_LLM_INPUT_COST_PER_1M", 0.15)
                        + (cc_tokens / 1_000_000.0) * getattr(settings, "CHALLENGER_LLM_OUTPUT_COST_PER_1M", 0.60)
                    )

            total_cost = primary_cost + challenger_cost
            challenger_applied = _to_bool(breakdown.get('challenger_applied'))
            has_cost_telemetry = any(
                field in breakdown
                for field in (
                    'llm_prompt_tokens',
                    'llm_completion_tokens',
                    'llm_estimated_cost_usd',
                    'challenger_prompt_tokens',
                    'challenger_completion_tokens',
                    'challenger_estimated_cost_usd',
                    'llm_cache_hit',
                )
            )
            range_primary_cost_total += primary_cost
            range_challenger_cost_total += challenger_cost

            workflow['observed_calls_count'] += 1
            if has_cost_telemetry:
                workflow['telemetry_tracked_calls_count'] += 1
            if llm_cache_hit:
                workflow['cached_calls_count'] += 1
            if challenger_applied:
                workflow['challenger_calls_count'] += 1
            workflow['primary_prompt_tokens'] += p_tokens
            workflow['primary_completion_tokens'] += c_tokens
            workflow['challenger_prompt_tokens'] += cp_tokens
            workflow['challenger_completion_tokens'] += cc_tokens
            workflow['primary_cost_usd'] += primary_cost
            workflow['challenger_cost_usd'] += challenger_cost
            workflow['total_estimated_cost_usd'] += total_cost

            agent_key = str(row.get('generated_by_agent') or 'unknown').lower()
            workflow['agent_counts'][agent_key] = int(workflow['agent_counts'].get(agent_key, 0) or 0) + 1
            symbol = str(row.get('token_symbol') or '').upper().strip()
            if symbol:
                workflow['symbol_counts'][symbol] = int(workflow['symbol_counts'].get(symbol, 0) or 0) + 1

            if has_model_attribution:
                workflow_model = workflow['models'].setdefault(model_key, {
                    'model_key': model_key,
                    'llm_provider': llm_provider,
                    'llm_model': llm_model,
                    'calls_count': 0,
                    'total_estimated_cost_usd': 0.0,
                })
                workflow_model['calls_count'] += 1
                workflow_model['total_estimated_cost_usd'] += total_cost

            target_buckets: List[Dict[str, Any]] = []

            if has_model_attribution:
                if model_key not in model_buckets:
                    model_buckets[model_key] = {
                        'model_key': model_key,
                        'llm_provider': llm_provider,
                        'llm_model': llm_model,
                        'total_signals': 0,
                        'terminal_signals': 0,
                        'wins': 0,
                        'losses': 0,
                        'expired': 0,
                        'sum_confidence': 0.0,
                        'total_estimated_cost_usd': 0.0,
                        'challenger_signals': 0,
                        'cached_calls_count': 0,
                        'primary_prompt_tokens': 0,
                        'primary_completion_tokens': 0,
                        'challenger_prompt_tokens': 0,
                        'challenger_completion_tokens': 0,
                        'primary_cost_usd': 0.0,
                        'challenger_cost_usd': 0.0,
                        'returns': [],
                    }
                target_buckets.append(model_buckets[model_key])

            if has_prompt_attribution:
                if prompt_version not in prompt_buckets:
                    prompt_buckets[prompt_version] = {
                        'prompt_version': prompt_version,
                        'total_signals': 0,
                        'terminal_signals': 0,
                        'wins': 0,
                        'losses': 0,
                        'expired': 0,
                        'sum_confidence': 0.0,
                        'total_estimated_cost_usd': 0.0,
                        'challenger_signals': 0,
                        'cached_calls_count': 0,
                        'primary_prompt_tokens': 0,
                        'primary_completion_tokens': 0,
                        'challenger_prompt_tokens': 0,
                        'challenger_completion_tokens': 0,
                        'primary_cost_usd': 0.0,
                        'challenger_cost_usd': 0.0,
                        'returns': [],
                    }
                target_buckets.append(prompt_buckets[prompt_version])

            est_return = _estimated_terminal_return_pct(row, status)

            for bucket in target_buckets:
                bucket['total_signals'] += 1
                bucket['sum_confidence'] += confidence
                bucket['total_estimated_cost_usd'] += total_cost
                bucket['primary_prompt_tokens'] += p_tokens
                bucket['primary_completion_tokens'] += c_tokens
                bucket['challenger_prompt_tokens'] += cp_tokens
                bucket['challenger_completion_tokens'] += cc_tokens
                bucket['primary_cost_usd'] += primary_cost
                bucket['challenger_cost_usd'] += challenger_cost
                if llm_cache_hit:
                    bucket['cached_calls_count'] += 1
                if challenger_applied:
                    bucket['challenger_signals'] += 1

                if status in terminal_statuses:
                    bucket['terminal_signals'] += 1
                    if est_return is not None:
                        bucket['returns'].append(est_return)

                if status in {'hit_target_1', 'hit_target_2'}:
                    bucket['wins'] += 1
                elif status == 'hit_stop_loss':
                    bucket['losses'] += 1
                elif status == 'expired':
                    bucket['expired'] += 1

        def _finalize(bucket: Dict[str, Any]) -> Dict[str, Any]:
            total = bucket['total_signals']
            terminal = bucket['terminal_signals']
            wins = bucket['wins']
            losses = bucket['losses']
            expired = bucket['expired']
            resolved = wins + losses
            returns = bucket.get('returns', [])
            avg_return = (sum(returns) / len(returns)) if returns else 0.0

            if len(returns) > 1:
                variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
                std_dev = math.sqrt(max(variance, 0.0))
            else:
                std_dev = 0.0

            sharpe_like = (avg_return / std_dev) * math.sqrt(len(returns)) if std_dev > 1e-9 else 0.0
            total_cost = max(0.0, bucket.get('total_estimated_cost_usd', 0.0))

            return {
                key: value for key, value in {
                    **bucket,
                    'avg_confidence': round(bucket['sum_confidence'] / total, 4) if total > 0 else 0.0,
                    'win_rate_terminal': round((wins / terminal) * 100, 2) if terminal > 0 else 0.0,
                    'win_rate_resolved': round((wins / resolved) * 100, 2) if resolved > 0 else 0.0,
                    'expiry_rate': round((expired / total) * 100, 2) if total > 0 else 0.0,
                    'avg_return_pct': round(avg_return, 4),
                    'sharpe_like': round(sharpe_like, 4),
                    'total_estimated_cost_usd': round(total_cost, 6),
                    'primary_cost_usd': round(max(bucket.get('primary_cost_usd', 0.0), 0.0), 6),
                    'challenger_cost_usd': round(max(bucket.get('challenger_cost_usd', 0.0), 0.0), 6),
                    'avg_estimated_cost_usd': round(total_cost / total, 8) if total > 0 else 0.0,
                    'cost_per_win_usd': round(total_cost / wins, 6) if wins > 0 else 0.0,
                    'challenger_usage_rate': round((bucket['challenger_signals'] / total) * 100, 2) if total > 0 else 0.0,
                    'cached_calls_rate': round((bucket['cached_calls_count'] / total) * 100, 2) if total > 0 else 0.0,
                    'primary_prompt_tokens': int(bucket.get('primary_prompt_tokens', 0) or 0),
                    'primary_completion_tokens': int(bucket.get('primary_completion_tokens', 0) or 0),
                    'challenger_prompt_tokens': int(bucket.get('challenger_prompt_tokens', 0) or 0),
                    'challenger_completion_tokens': int(bucket.get('challenger_completion_tokens', 0) or 0),
                    'total_prompt_tokens': int(bucket.get('primary_prompt_tokens', 0) or 0) + int(bucket.get('challenger_prompt_tokens', 0) or 0),
                    'total_completion_tokens': int(bucket.get('primary_completion_tokens', 0) or 0) + int(bucket.get('challenger_completion_tokens', 0) or 0),
                    'total_tokens': (
                        int(bucket.get('primary_prompt_tokens', 0) or 0)
                        + int(bucket.get('primary_completion_tokens', 0) or 0)
                        + int(bucket.get('challenger_prompt_tokens', 0) or 0)
                        + int(bucket.get('challenger_completion_tokens', 0) or 0)
                    ),
                }.items() if key not in {'sum_confidence', 'returns'}
            }

        try:
            credit_query = (
                db.from_('credit_transactions')
                .select('created_at,description,transaction_type,amount,service_used')
                .eq('transaction_type', 'usage')
                .lt('amount', 0)
                .in_('service_used', ['token_analysis', 'trading_scanner'])
            )
            if date_filter_start and date_filter_end:
                credit_query = credit_query.gte('created_at', date_filter_start).lte('created_at', date_filter_end)
            credit_rows = (
                await _execute_db_query(
                    credit_query.order('created_at', desc=True).limit(5000).execute
                )
            ).data or []
            token_analysis_workflow = workflow_buckets['token_analysis']
            for credit_row in credit_rows:
                service_key, service_label, driver_label = _billing_service_meta(credit_row.get('service_used'))
                billing_bucket = billing_buckets.setdefault(
                    service_key,
                    _billing_bucket(service_key, service_label, driver_label),
                )
                billed_credits = max(abs(_to_int(credit_row.get('amount'))), 0)
                billing_bucket['billed_requests_count'] += 1
                billing_bucket['billed_credits'] += billed_credits
                latest_billed_at = str(credit_row.get('created_at') or '').strip() or None
                if latest_billed_at and (
                    not billing_bucket['latest_billed_at']
                    or latest_billed_at > billing_bucket['latest_billed_at']
                ):
                    billing_bucket['latest_billed_at'] = latest_billed_at

                driver = _extract_billing_driver(service_key, credit_row.get('description'))
                if driver:
                    billing_bucket['driver_counts'][driver] = int(
                        billing_bucket['driver_counts'].get(driver, 0) or 0
                    ) + 1
                    billing_bucket['driver_credits'][driver] = int(
                        billing_bucket['driver_credits'].get(driver, 0) or 0
                    ) + billed_credits

                if service_key == 'token_analysis':
                    token_analysis_workflow['billed_requests_count'] += 1
                    billed_symbol = _extract_token_analysis_symbol(credit_row.get('description'))
                    if billed_symbol:
                        token_analysis_workflow['symbol_counts'][billed_symbol] = int(
                            token_analysis_workflow['symbol_counts'].get(billed_symbol, 0) or 0
                        ) + 1
        except Exception as credit_err:
            logger.warning(f"Token analysis credit usage aggregation failed: {credit_err}")

        def _finalize_billing(bucket: Dict[str, Any]) -> Dict[str, Any]:
            billed_requests = int(bucket.get('billed_requests_count', 0) or 0)
            billed_credits = int(bucket.get('billed_credits', 0) or 0)
            top_drivers = sorted(
                [
                    {
                        'label': label,
                        'count': int(count or 0),
                        'billed_credits': int((bucket.get('driver_credits') or {}).get(label, 0) or 0),
                    }
                    for label, count in (bucket.get('driver_counts') or {}).items()
                ],
                key=lambda item: (-item['billed_credits'], -item['count'], item['label']),
            )[:5]
            return {
                'key': bucket.get('key'),
                'label': bucket.get('label'),
                'driver_label': bucket.get('driver_label'),
                'billed_requests_count': billed_requests,
                'billed_credits': billed_credits,
                'avg_credits_per_request': round(billed_credits / billed_requests, 4) if billed_requests > 0 else 0.0,
                'latest_billed_at': bucket.get('latest_billed_at'),
                'top_drivers': top_drivers,
            }

        def _finalize_workflow(bucket: Dict[str, Any]) -> Dict[str, Any]:
            observed_calls = int(bucket.get('observed_calls_count', 0) or 0)
            billed_requests = int(bucket.get('billed_requests_count', 0) or 0)
            tracked_calls = int(bucket.get('telemetry_tracked_calls_count', 0) or 0)
            cached_calls = int(bucket.get('cached_calls_count', 0) or 0)
            total_cost = round(max(float(bucket.get('total_estimated_cost_usd', 0.0) or 0.0), 0.0), 6)
            denominator = max(observed_calls, billed_requests)
            top_models = sorted(
                [
                    {
                        'model_key': model_key,
                        'llm_provider': value.get('llm_provider'),
                        'llm_model': value.get('llm_model'),
                        'calls_count': int(value.get('calls_count', 0) or 0),
                        'total_estimated_cost_usd': round(max(float(value.get('total_estimated_cost_usd', 0.0) or 0.0), 0.0), 6),
                    }
                    for model_key, value in (bucket.get('models') or {}).items()
                ],
                key=lambda item: (-item['total_estimated_cost_usd'], -item['calls_count']),
            )[:3]
            top_agents = sorted(
                [
                    {'agent': agent, 'calls_count': int(count or 0)}
                    for agent, count in (bucket.get('agent_counts') or {}).items()
                ],
                key=lambda item: (-item['calls_count'], item['agent']),
            )[:3]
            top_symbols = sorted(
                [
                    {'symbol': symbol, 'count': int(count or 0)}
                    for symbol, count in (bucket.get('symbol_counts') or {}).items()
                ],
                key=lambda item: (-item['count'], item['symbol']),
            )[:5]

            return {
                'key': bucket.get('key'),
                'label': bucket.get('label'),
                'observed_calls_count': observed_calls,
                'billed_requests_count': billed_requests,
                'telemetry_tracked_calls_count': tracked_calls,
                'telemetry_coverage_percent': round((tracked_calls / denominator) * 100, 2) if denominator > 0 else 0.0,
                'untracked_calls_count': max(denominator - tracked_calls, 0),
                'cached_calls_count': cached_calls,
                'challenger_calls_count': int(bucket.get('challenger_calls_count', 0) or 0),
                'primary_prompt_tokens': int(bucket.get('primary_prompt_tokens', 0) or 0),
                'primary_completion_tokens': int(bucket.get('primary_completion_tokens', 0) or 0),
                'challenger_prompt_tokens': int(bucket.get('challenger_prompt_tokens', 0) or 0),
                'challenger_completion_tokens': int(bucket.get('challenger_completion_tokens', 0) or 0),
                'total_tokens': (
                    int(bucket.get('primary_prompt_tokens', 0) or 0)
                    + int(bucket.get('primary_completion_tokens', 0) or 0)
                    + int(bucket.get('challenger_prompt_tokens', 0) or 0)
                    + int(bucket.get('challenger_completion_tokens', 0) or 0)
                ),
                'primary_cost_usd': round(max(float(bucket.get('primary_cost_usd', 0.0) or 0.0), 0.0), 6),
                'challenger_cost_usd': round(max(float(bucket.get('challenger_cost_usd', 0.0) or 0.0), 0.0), 6),
                'total_estimated_cost_usd': total_cost,
                'avg_cost_per_observed_call_usd': round(total_cost / observed_calls, 8) if observed_calls > 0 else 0.0,
                'avg_cost_per_tracked_call_usd': round(total_cost / tracked_calls, 8) if tracked_calls > 0 else 0.0,
                'top_models': top_models,
                'top_agents': top_agents,
                'top_symbols': top_symbols,
            }

        models = sorted(
            [_finalize(b) for b in model_buckets.values()],
            key=lambda item: (-item['total_signals'], -item['win_rate_terminal'])
        )
        prompts = sorted(
            [_finalize(b) for b in prompt_buckets.values()],
            key=lambda item: (-item['total_signals'], -item['win_rate_terminal'])
        )
        workflow_breakdown = [
            _finalize_workflow(workflow_buckets['platform_signals']),
            _finalize_workflow(workflow_buckets['token_analysis']),
        ]
        billing_breakdown_services = sorted(
            [_finalize_billing(bucket) for bucket in billing_buckets.values()],
            key=lambda item: (-item['billed_credits'], -item['billed_requests_count'], item['label']),
        )
        observed_calls_total = sum(item['observed_calls_count'] for item in workflow_breakdown)
        tracked_calls_total = sum(item['telemetry_tracked_calls_count'] for item in workflow_breakdown)
        cached_calls_total = sum(item['cached_calls_count'] for item in workflow_breakdown)
        billed_token_analysis_requests = workflow_breakdown[1]['billed_requests_count']
        budget_status = await _get_challenger_budget_status(db)

        return {
            "models": models,
            "prompts": prompts,
            "total_signals": len(rows),
            # Live policy configuration; new signals are stamped with these, so
            # the dashboard can show which era current outcomes belong to.
            "active_policy_versions": policy_stamp(),
            "active_policy_bundle": policy_bundle_version(),
            "attribution": {
                "model_attributed_signals": model_attributed_signals,
                "model_unattributed_signals": model_unattributed_signals,
                "prompt_attributed_signals": prompt_attributed_signals,
                "prompt_unattributed_signals": prompt_unattributed_signals,
            },
            "cost_summary": {
                "range_primary_cost_usd": round(max(range_primary_cost_total, 0.0), 6),
                "range_challenger_cost_usd": round(max(range_challenger_cost_total, 0.0), 6),
                "range_total_estimated_cost_usd": round(max(range_primary_cost_total + range_challenger_cost_total, 0.0), 6),
                "workflow_breakdown": workflow_breakdown,
                "billing_breakdown": {
                    "total_billed_requests_count": sum(item['billed_requests_count'] for item in billing_breakdown_services),
                    "total_billed_credits": sum(item['billed_credits'] for item in billing_breakdown_services),
                    "services": billing_breakdown_services,
                },
                "coverage": {
                    "observed_calls_count": observed_calls_total,
                    "telemetry_tracked_calls_count": tracked_calls_total,
                    "telemetry_coverage_percent": round((tracked_calls_total / observed_calls_total) * 100, 2) if observed_calls_total > 0 else 0.0,
                    "cached_calls_count": cached_calls_total,
                    "billed_token_analysis_requests_count": billed_token_analysis_requests,
                    "untracked_token_analysis_requests_count": max(billed_token_analysis_requests - workflow_breakdown[1]['telemetry_tracked_calls_count'], 0),
                },
                "budget": budget_status,
            },
        }
    except Exception as e:
        logger.error(f"Model performance calculation failed: {e}")
        return {
            "error": str(e),
            "models": [],
            "prompts": [],
            "total_signals": 0,
            "attribution": {
                "model_attributed_signals": 0,
                "model_unattributed_signals": 0,
                "prompt_attributed_signals": 0,
                "prompt_unattributed_signals": 0,
            },
            "cost_summary": {
                "range_primary_cost_usd": 0.0,
                "range_challenger_cost_usd": 0.0,
                "range_total_estimated_cost_usd": 0.0,
                "workflow_breakdown": [],
                "billing_breakdown": {
                    "total_billed_requests_count": 0,
                    "total_billed_credits": 0,
                    "services": [],
                },
                "coverage": {
                    "observed_calls_count": 0,
                    "telemetry_tracked_calls_count": 0,
                    "telemetry_coverage_percent": 0.0,
                    "cached_calls_count": 0,
                    "billed_token_analysis_requests_count": 0,
                    "untracked_token_analysis_requests_count": 0,
                },
                "budget": {
                    "budget_enabled": False,
                    "monthly_budget_usd": 0.0,
                    "spent_usd": 0.0,
                    "remaining_usd": None,
                    "usage_percent": 0.0,
                    "blocked": False,
                    "challenger_calls_count": 0,
                },
            },
        }


async def _get_challenger_budget_status(db) -> Dict[str, Any]:
    """Calculate challenger monthly budget utilization for dashboard and guardrail visibility."""
    try:
        monthly_budget = max(float(getattr(settings, 'CHALLENGER_MONTHLY_BUDGET_USD', 0.0) or 0.0), 0.0)
        now = datetime.now(timezone.utc)
        month_start_dt = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        month_start = month_start_dt.isoformat()

        query = (
            db.from_('platform_signals')
            .select('ai_confidence_breakdown')
            .gte('created_at', month_start)
            .order('created_at', desc=True)
            .limit(5000)
        )
        response = await _execute_db_query(query.execute)
        rows = response.data or []

        challenger_spend = 0.0
        challenger_calls = 0
        for row in rows:
            raw = row.get('ai_confidence_breakdown')
            breakdown: Dict[str, Any]
            if isinstance(raw, dict):
                breakdown = raw
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    breakdown = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    breakdown = {}
            else:
                breakdown = {}

            try:
                call_cost = max(0.0, float(breakdown.get('challenger_estimated_cost_usd') or 0.0))
            except (TypeError, ValueError):
                call_cost = 0.0

            challenger_spend += call_cost
            if breakdown.get('challenger_applied') in {True, 'true', 'True', 1, '1'}:
                challenger_calls += 1

        challenger_spend = round(max(challenger_spend, 0.0), 6)
        remaining = round(max(monthly_budget - challenger_spend, 0.0), 6) if monthly_budget > 0 else None
        usage_pct = round((challenger_spend / monthly_budget) * 100, 2) if monthly_budget > 0 else 0.0
        blocked = bool(monthly_budget > 0 and challenger_spend >= monthly_budget)

        return {
            "month_start": month_start_dt.date().isoformat(),
            "budget_enabled": monthly_budget > 0,
            "monthly_budget_usd": round(monthly_budget, 2),
            "spent_usd": challenger_spend,
            "remaining_usd": remaining,
            "usage_percent": usage_pct,
            "blocked": blocked,
            "challenger_calls_count": challenger_calls,
        }
    except Exception as budget_error:
        logger.warning(f"Challenger budget status calculation failed: {budget_error}")
        return {
            "month_start": datetime.now(timezone.utc).date().isoformat(),
            "budget_enabled": False,
            "monthly_budget_usd": 0.0,
            "spent_usd": 0.0,
            "remaining_usd": None,
            "usage_percent": 0.0,
            "blocked": False,
            "challenger_calls_count": 0,
            "error": str(budget_error),
        }


async def _get_prompt_optimization_summary(
    db,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize prompt optimization history for dashboard consumption."""
    try:
        query = db.from_('prompt_optimization_history').select('*')
        if date_filter_start and date_filter_end:
            query = query.gte('optimization_timestamp', date_filter_start).lte('optimization_timestamp', date_filter_end)

        result_query = query.order('optimization_version', desc=True).limit(100)
        result = await _execute_db_query(result_query.execute)
        rows = result.data or []
        if not rows:
            return {
                "total_optimizations": 0,
                "latest_by_agent": {},
                "recent_optimizations": [],
            }

        latest_by_agent: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            agent = row.get('agent_type') or 'unknown'
            if agent not in latest_by_agent:
                suggestions = row.get('improvement_suggestions')
                if not isinstance(suggestions, list):
                    suggestions = []
                latest_by_agent[agent] = {
                    "agent_type": agent,
                    "prompt_type": row.get('prompt_type'),
                    "optimization_version": row.get('optimization_version'),
                    "optimization_timestamp": row.get('optimization_timestamp'),
                    "optimization_rationale": row.get('optimization_rationale'),
                    "improvement_suggestions": suggestions[:5],
                }

        recent = []
        for row in rows[:20]:
            suggestions = row.get('improvement_suggestions')
            if not isinstance(suggestions, list):
                suggestions = []
            recent.append({
                "agent_type": row.get('agent_type'),
                "prompt_type": row.get('prompt_type'),
                "optimization_version": row.get('optimization_version'),
                "optimization_timestamp": row.get('optimization_timestamp'),
                "optimization_rationale": row.get('optimization_rationale'),
                "improvement_suggestions": suggestions[:5],
            })

        return {
            "total_optimizations": len(rows),
            "latest_by_agent": latest_by_agent,
            "recent_optimizations": recent,
        }
    except Exception as e:
        logger.error(f"Prompt optimization summary failed: {e}")
        return {"error": str(e), "total_optimizations": 0, "latest_by_agent": {}, "recent_optimizations": []}


def _calculate_date_filters(date_range: Optional[str], start_date: Optional[str], end_date: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Calculate start and end date filters based on parameters."""
    if start_date and end_date:
        # Custom date range (inclusive full-day UTC window).
        try:
            start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(microseconds=1)
            return start.isoformat(), end.isoformat()
        except ValueError:
            logger.warning(f"Invalid custom date range values: start={start_date}, end={end_date}")
            return start_date, end_date
    elif date_range:
        # Predefined date ranges
        now = datetime.now(timezone.utc)
        if date_range == '24h':
            start = now - timedelta(hours=24)
            return start.isoformat(), now.isoformat()
        elif date_range == '7d':
            start = now - timedelta(days=7)
            return start.isoformat(), now.isoformat()
        elif date_range == '30d':
            start = now - timedelta(days=30)
            return start.isoformat(), now.isoformat()
        elif date_range == '90d':
            start = now - timedelta(days=90)
            return start.isoformat(), now.isoformat()
        elif date_range == 'this_month':
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return start.isoformat(), now.isoformat()
        elif date_range == 'last_month':
            first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = first_day_this_month - timedelta(microseconds=1)
            start = end.replace(day=1)
            return start.isoformat(), end.isoformat()

    # No date filtering
    return None, None
