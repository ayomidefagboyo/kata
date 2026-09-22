"""
Recommendation Resolver

Scores pending token-analysis recommendations (Ryu / Sakura / Yuki) stored in
`agent_recommendation_tracking` against real price action, so analysis agents get a
measurable win rate.

Each recommendation is judged over two horizons (24h and 7d). Only explicit
LONG_ENTRY / SHORT_ENTRY intents with a directionally valid trade plan are scored:

    win  -> target_1 is reached before stop_loss within the horizon
    loss -> stop_loss is reached before target_1
    neither hit by the horizon -> resolved on close-vs-entry in the recommended
            direction (favorable = win, adverse = loss)

HOLD recommendations are stored but marked 'neutral' and excluded from win rate
downstream. Same-candle ambiguity (both target and stop inside one candle) is resolved
conservatively as a stop hit.

This module only *resolves* rows; it never creates them (that happens at analysis time
in token_analysis.py).
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from kata.config.database import get_service_client
from kata.services.market_data_service import create_market_data_service
from kata.api.token_analysis import _token_market_symbol_candidates

logger = logging.getLogger(__name__)

# horizon_key -> (candle interval, seconds per candle)
_HORIZON_INTERVALS: Dict[str, Tuple[str, int]] = {
    "24h": ("15m", 15 * 60),
    "7d": ("1h", 60 * 60),
}

# Statuses that count toward win rate
_MIN_CANDLES = 2
_MAX_CANDLE_LIMIT = 1000


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse a timestamp into a tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _evaluate_outcome(
    direction: str,
    entry: float,
    target_1: Optional[float],
    stop_loss: Optional[float],
    candles: List[Dict[str, Any]],
) -> Optional[Tuple[str, str, float]]:
    """
    Return (status, exit_reason, pnl_pct) or None if it cannot be evaluated.

    status: 'win' | 'loss' | 'neutral'
    """
    if entry <= 0 or not candles:
        return None

    # HOLD: tracked but not scored as directional win/loss
    if direction == "neutral":
        last_close = _float_or_none(candles[-1].get("close")) or entry
        pnl = (last_close / entry - 1.0) * 100.0
        return ("neutral", "HOLD", round(pnl, 4))

    is_long = direction == "long"
    have_target = target_1 is not None and target_1 > 0
    have_stop = stop_loss is not None and stop_loss > 0

    valid_plan = (
        have_target
        and have_stop
        and (
            (is_long and target_1 > entry > stop_loss)
            or (direction == "short" and target_1 < entry < stop_loss)
        )
    )
    if direction not in {"long", "short"} or not valid_plan:
        last_close = _float_or_none(candles[-1].get("close")) or entry
        pnl = (last_close / entry - 1.0) * 100.0
        return ("neutral", "INVALID_TRADE_PLAN", round(pnl, 4))

    for candle in candles:
        high = _float_or_none(candle.get("high"))
        low = _float_or_none(candle.get("low"))
        if high is None or low is None:
            continue

        target_hit = have_target and (high >= target_1 if is_long else low <= target_1)
        stop_hit = have_stop and (low <= stop_loss if is_long else high >= stop_loss)

        # Conservative: if a single candle spans both, assume the stop triggered first
        if stop_hit:
            pnl = (stop_loss / entry - 1.0) * 100.0 if is_long else (1.0 - stop_loss / entry) * 100.0
            return ("loss", "STOP_LOSS", round(pnl, 4))
        if target_hit:
            pnl = (target_1 / entry - 1.0) * 100.0 if is_long else (1.0 - target_1 / entry) * 100.0
            return ("win", "TARGET_1", round(pnl, 4))

    # Neither level hit within the horizon -> judge on close vs entry, in-direction
    last_close = _float_or_none(candles[-1].get("close"))
    if last_close is None:
        return None
    directional = (last_close / entry - 1.0) * 100.0 if is_long else (1.0 - last_close / entry) * 100.0
    if directional >= 0:
        return ("win", "EXPIRED_FAVORABLE", round(directional, 4))
    return ("loss", "EXPIRED_ADVERSE", round(directional, 4))


class RecommendationResolver:
    """Resolves due, unscored recommendations in agent_recommendation_tracking."""

    def __init__(self):
        self.db = get_service_client()
        self._market_service = None  # created lazily inside an async context

    async def _get_candles_for_window(
        self,
        token_symbol: str,
        interval: str,
        seconds_per_candle: int,
        created_at: datetime,
        due_at: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch candles covering [created_at, due_at] and return them in time order."""
        binance = getattr(self._market_service, "binance", None)
        if not binance:
            return []

        now = datetime.now(timezone.utc)
        span_seconds = max(0.0, (now - created_at).total_seconds())
        limit = min(_MAX_CANDLE_LIMIT, int(span_seconds / seconds_per_candle) + 8)
        limit = max(limit, 10)

        for market_symbol in _token_market_symbol_candidates(token_symbol):
            try:
                candles = await binance.get_historical_klines(market_symbol, interval=interval, limit=limit)
            except Exception as exc:
                logger.debug("Klines unavailable for %s via %s: %s", token_symbol, market_symbol, exc)
                continue

            windowed = []
            for candle in candles or []:
                ts = _parse_dt(candle.get("timestamp"))
                if ts is None:
                    continue
                if created_at <= ts <= due_at:
                    windowed.append(candle)
            if len(windowed) >= _MIN_CANDLES:
                windowed.sort(key=lambda c: _parse_dt(c.get("timestamp")) or created_at)
                return windowed
        return []

    async def _resolve_horizon(self, horizon: str, batch_size: int) -> Dict[str, int]:
        """Resolve all due, unscored rows for a single horizon."""
        interval, seconds_per_candle = _HORIZON_INTERVALS[horizon]
        status_col = f"horizon_{horizon}_status"
        due_col = f"horizon_{horizon}_due_at"
        pnl_col = f"horizon_{horizon}_pnl_pct"
        reason_col = f"horizon_{horizon}_exit_reason"
        resolved_col = f"horizon_{horizon}_resolved_at"

        now_iso = datetime.now(timezone.utc).isoformat()
        result = (
            self.db.table("agent_recommendation_tracking")
            .select("*")
            .is_(status_col, "null")
            .lte(due_col, now_iso)
            .order(due_col, desc=False)
            .limit(batch_size)
            .execute()
        )
        rows = result.data or []
        stats = {"scanned": len(rows), "resolved": 0, "skipped": 0}

        for row in rows:
            created_at = _parse_dt(row.get("created_at"))
            due_at = _parse_dt(row.get(due_col))
            entry = _float_or_none(row.get("entry_price"))
            direction = str(row.get("direction") or "neutral").lower()

            if created_at is None or due_at is None or entry is None or entry <= 0:
                stats["skipped"] += 1
                continue

            candles = await self._get_candles_for_window(
                row.get("token_symbol") or "", interval, seconds_per_candle, created_at, due_at
            )
            outcome = _evaluate_outcome(
                direction,
                entry,
                _float_or_none(row.get("target_1")),
                _float_or_none(row.get("stop_loss")),
                candles,
            )
            if outcome is None:
                # Could not fetch price data yet; leave pending for a later pass.
                stats["skipped"] += 1
                continue

            status, exit_reason, pnl_pct = outcome
            try:
                self.db.table("agent_recommendation_tracking").update({
                    status_col: status,
                    pnl_col: pnl_pct,
                    reason_col: exit_reason,
                    resolved_col: datetime.now(timezone.utc).isoformat(),
                }).eq("id", row["id"]).execute()
                stats["resolved"] += 1
            except Exception as exc:
                logger.warning("Failed to update recommendation %s (%s): %s", row.get("id"), horizon, exc)
                stats["skipped"] += 1

        return stats

    async def resolve_pending(self, batch_size: int = 200) -> Dict[str, Any]:
        """Resolve all due recommendations across both horizons. Returns a summary."""
        summary: Dict[str, Any] = {}
        market_service = create_market_data_service()
        try:
            async with market_service:
                self._market_service = market_service
                for horizon in _HORIZON_INTERVALS:
                    try:
                        summary[horizon] = await self._resolve_horizon(horizon, batch_size)
                    except Exception as exc:
                        logger.error("Recommendation resolution failed for horizon %s: %s", horizon, exc)
                        summary[horizon] = {"error": str(exc)}
        finally:
            self._market_service = None

        total_resolved = sum(
            h.get("resolved", 0) for h in summary.values() if isinstance(h, dict)
        )
        if total_resolved:
            logger.info("🧮 Recommendation resolver scored %d outcomes: %s", total_resolved, summary)
        return summary


_resolver_singleton: Optional[RecommendationResolver] = None


def get_recommendation_resolver() -> RecommendationResolver:
    global _resolver_singleton
    if _resolver_singleton is None:
        _resolver_singleton = RecommendationResolver()
    return _resolver_singleton
