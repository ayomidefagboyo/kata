"""
Signal generation schedule optimizer.

Computes the best UTC generation slots from accepted signal outcomes and
rejected-candidate counterfactual learning. The worker uses this for tracking
and recommendations; schedule auto-application is intentionally opt-in.
"""

import logging
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from kata.config.database import get_service_client

logger = logging.getLogger(__name__)

TERMINAL_WIN_STATUSES = {"hit_target_1", "hit_target_2"}
TERMINAL_LOSS_STATUSES = {"hit_stop_loss"}
DEFAULT_GENERATION_HOURS_UTC = [2, 8, 14, 20]
ACTIVE_SCHEDULE_LEARNING_TYPE = "signal_generation_active_schedule"
ACTIVE_SCHEDULE_SIGNAL_ID = "signal_generation_active_schedule"


def normalize_schedule_hours(hours: Any) -> List[int]:
    """Return a sorted, unique list of valid UTC hours."""
    if not isinstance(hours, (list, tuple, set)):
        return []

    normalized: List[int] = []
    for value in hours:
        try:
            hour = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= hour <= 23 and hour not in normalized:
            normalized.append(hour)
    return sorted(normalized)


class SignalScheduleOptimizer:
    """Analyze and persist best signal-generation timing windows."""

    def __init__(self):
        self.db = get_service_client()

    def load_active_schedule(
        self,
        default_hours_utc: Optional[List[int]] = None,
    ) -> List[int]:
        """Load the dashboard-applied schedule, falling back to the product default.

        The canonical row makes the database the only runtime source of truth.
        Older manually/automatically applied audit rows are supported as a
        one-time compatibility fallback for deployments created before the
        canonical row existed.
        """
        fallback = normalize_schedule_hours(default_hours_utc) or list(
            DEFAULT_GENERATION_HOURS_UTC
        )
        try:
            canonical = (
                self.db.from_("agent_learning_insights")
                .select("insights,timestamp,updated_at")
                .eq("agent_type", "yuki")
                .eq("learning_type", ACTIVE_SCHEDULE_LEARNING_TYPE)
                .eq("signal_id", ACTIVE_SCHEDULE_SIGNAL_ID)
                .limit(1)
                .execute()
            ).data or []
            if canonical:
                hours = normalize_schedule_hours(
                    (canonical[0].get("insights") or {}).get("active_hours_utc")
                )
                if hours:
                    return hours

            legacy_rows = (
                self.db.from_("agent_learning_insights")
                .select("insights,timestamp")
                .eq("agent_type", "yuki")
                .eq("learning_type", "signal_schedule_auto_apply")
                .order("timestamp", desc=True)
                .limit(100)
                .execute()
            ).data or []
            for row in legacy_rows:
                insights = row.get("insights") or {}
                if not insights.get("applied"):
                    continue
                hours = normalize_schedule_hours(insights.get("active_hours_utc"))
                if hours:
                    return hours
        except Exception as exc:
            logger.warning("Could not load active signal schedule; using fallback: %s", exc)
        return fallback

    def apply_active_schedule(
        self,
        hours_utc: List[int],
        *,
        source: str,
        applied_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist the active schedule selected by the dashboard or optimizer."""
        hours = normalize_schedule_hours(hours_utc)
        if not hours:
            raise ValueError("Schedule must contain at least one valid UTC hour")

        now = datetime.now(timezone.utc)
        existing = (
            self.db.from_("agent_learning_insights")
            .select("id,insights")
            .eq("agent_type", "yuki")
            .eq("learning_type", ACTIVE_SCHEDULE_LEARNING_TYPE)
            .eq("signal_id", ACTIVE_SCHEDULE_SIGNAL_ID)
            .limit(1)
            .execute()
        ).data or []
        previous_hours = normalize_schedule_hours(
            (existing[0].get("insights") or {}).get("active_hours_utc")
        ) if existing else []
        insights = {
            "active_hours_utc": hours,
            "previous_hours_utc": previous_hours,
            "source": str(source or "learning_dashboard")[:80],
            "applied_by": str(applied_by)[:255] if applied_by else None,
            "applied_at": now.isoformat(),
        }
        payload = {
            "agent_type": "yuki",
            "signal_id": ACTIVE_SCHEDULE_SIGNAL_ID,
            "learning_level": "meta",
            "learning_type": ACTIVE_SCHEDULE_LEARNING_TYPE,
            "confidence_score": 1.0,
            "timestamp": now.isoformat(),
            "updated_at": now.isoformat(),
            "insights": insights,
            "performance_impact": {
                "previous_hours_utc": previous_hours,
                "active_hours_utc": hours,
            },
        }
        if existing:
            result = (
                self.db.from_("agent_learning_insights")
                .update(payload)
                .eq("id", existing[0]["id"])
                .execute()
            )
        else:
            result = self.db.from_("agent_learning_insights").insert(payload).execute()

        if not result.data:
            raise RuntimeError("The active schedule could not be saved")
        return {
            "hours_utc": hours,
            "previous_hours_utc": previous_hours,
            "source": insights["source"],
            "applied_at": insights["applied_at"],
        }

    def _fetch_all(
        self,
        table: str,
        select: str,
        order_col: Optional[str] = None,
        desc: bool = True,
        filters: Optional[List[Tuple[str, Tuple[Any, ...]]]] = None,
        max_rows: int = 20_000,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        start = 0
        page_size = 1000

        while start < max_rows:
            query = self.db.from_(table).select(select)
            for method, args in filters or []:
                query = getattr(query, method)(*args)
            if order_col:
                query = query.order(order_col, desc=desc)

            result = query.range(start, start + page_size - 1).execute()
            batch = result.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size

        return rows

    @staticmethod
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

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, Any]:
        clean_values = [float(v) for v in values if math.isfinite(float(v))]
        wins = [v for v in clean_values if v > 0]
        losses = [v for v in clean_values if v < 0]
        total_losses = abs(sum(losses))

        return {
            "n": len(clean_values),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(clean_values) * 100.0, 2) if clean_values else None,
            "avg_pnl": round(sum(clean_values) / len(clean_values), 4) if clean_values else None,
            "median_pnl": round(statistics.median(clean_values), 4) if clean_values else None,
            "total_pnl": round(sum(clean_values), 4) if clean_values else None,
            "profit_factor": round(sum(wins) / total_losses, 4) if total_losses > 0 else (999.0 if wins else None),
            "best": round(max(clean_values), 4) if clean_values else None,
            "worst": round(min(clean_values), 4) if clean_values else None,
        }

    def _normalized_final_pnl(self, signal: Dict[str, Any], perf: Dict[str, Any]) -> Optional[float]:
        outcome = str(perf.get("outcome") or "").lower()
        status = str(signal.get("status") or "").lower()
        is_win = outcome == "win" or status in TERMINAL_WIN_STATUSES
        is_loss = outcome == "loss" or status in TERMINAL_LOSS_STATUSES
        if not is_win and not is_loss:
            return None

        leverage = self._safe_float(signal.get("leverage")) or 1.0
        pnl = self._safe_float(perf.get("leveraged_pnl_percent"))
        if pnl is None:
            actual = self._safe_float(perf.get("actual_pnl_percent"))
            pnl = actual * leverage if actual is not None else None
        if pnl is None:
            return None

        # Older rows have occasional sign drift; outcome/status is canonical.
        return abs(pnl) if is_win else -abs(pnl)

    @staticmethod
    def _schedule_label(hours: List[int]) -> str:
        return "/".join(f"{hour:02d}" for hour in hours) + " UTC"

    def _score_offset_schedule(self, summary: Dict[str, Any], hour_summaries: Dict[str, Dict[str, Any]], min_hour_samples: int) -> float:
        avg = float(summary.get("avg_pnl") or 0.0)
        win_rate = float(summary.get("win_rate") or 0.0)
        profit_factor = float(summary.get("profit_factor") or 0.0)
        weakest_avg = min(float(hour.get("avg_pnl") or 0.0) for hour in hour_summaries.values())
        min_hour_n = min(int(hour.get("n") or 0) for hour in hour_summaries.values())

        score = avg
        score += (win_rate - 50.0) * 0.12
        score += min(profit_factor, 5.0) * 1.5
        if weakest_avg < 0:
            score += weakest_avg * 0.4
        if min_hour_n < min_hour_samples:
            score -= 25.0
        return round(score, 4)

    def _normalize_reject_quality(self, insights: Dict[str, Any]) -> str:
        terminal_status = str(insights.get("counterfactual_status") or "").lower()
        if terminal_status == "hit_stop_loss":
            return "good_reject"
        if terminal_status in {"hit_target_1", "hit_target_2"}:
            return "bad_reject"
        if terminal_status == "expired":
            pnl = self._safe_float(insights.get("counterfactual_pnl_pct")) or 0.0
            return "good_reject" if pnl <= 0 else "ambiguous_reject"

        quality = str(insights.get("reject_quality") or "").lower()
        if quality in {"good_reject", "bad_reject", "ambiguous_reject"}:
            return quality
        return "unknown"

    def _analyze_reject_hours(self, lookback_start: str) -> Dict[str, Any]:
        try:
            rows = self._fetch_all(
                "agent_learning_insights",
                "id, insights, timestamp",
                order_col="timestamp",
                filters=[
                    ("eq", ("agent_type", "yuki")),
                    ("eq", ("learning_type", "signal_rejected_candidate")),
                    ("gte", ("timestamp", lookback_start)),
                ],
                max_rows=10_000,
            )
        except Exception as exc:
            logger.warning("Could not analyze rejected-candidate schedule hours: %s", exc)
            return {"available": False, "error": str(exc)}

        by_hour: Dict[int, List[Tuple[str, Optional[float]]]] = defaultdict(list)
        quality_counts: Dict[str, int] = defaultdict(int)
        # rule_counterfactual rows tracked separately — they answer a different question
        # (AI-vs-rule accuracy) and must NOT bias gate-quality timing analysis.
        ai_rule_by_hour: Dict[int, Dict[str, int]] = defaultdict(lambda: {"rule_was_right": 0, "ai_was_right": 0, "total": 0})

        for row in rows:
            insights = row.get("insights")
            if not isinstance(insights, dict):
                continue

            candidate_source = str(insights.get("candidate_source") or "").lower()
            timestamp = self._parse_datetime(row.get("timestamp"))
            if not timestamp:
                continue

            # ── ai_directional: gate quality timing ─────────────────────
            if candidate_source == "ai_directional":
                quality = self._normalize_reject_quality(insights)
                quality_counts[quality] += 1
                if quality not in {"good_reject", "bad_reject", "ambiguous_reject"}:
                    continue
                pnl = self._safe_float(insights.get("counterfactual_pnl_pct_leveraged"))
                by_hour[timestamp.hour].append((quality, pnl))

            # ── rule_counterfactual: AI-vs-rule timing ───────────────────
            elif candidate_source == "rule_counterfactual":
                tracking_status = str(insights.get("tracking_status") or "").lower()
                if tracking_status != "resolved":
                    continue
                terminal_status = str(insights.get("counterfactual_status") or "").lower()
                try:
                    pnl = float(insights.get("counterfactual_pnl_pct") or 0.0)
                except (TypeError, ValueError):
                    pnl = 0.0
                if terminal_status in {"hit_target_1", "hit_target_2"}:
                    verdict = "rule_was_right"
                elif terminal_status == "hit_stop_loss":
                    verdict = "ai_was_right"
                elif terminal_status == "expired":
                    verdict = "ai_was_right" if pnl <= 0 else "rule_was_right"
                else:
                    continue
                ai_rule_by_hour[timestamp.hour]["total"] += 1
                ai_rule_by_hour[timestamp.hour][verdict] += 1

        hour_summary: Dict[str, Dict[str, Any]] = {}
        for hour, values in by_hour.items():
            good = sum(1 for quality, _ in values if quality == "good_reject")
            bad = sum(1 for quality, _ in values if quality == "bad_reject")
            ambiguous = sum(1 for quality, _ in values if quality == "ambiguous_reject")
            informative = good + bad
            pnl_values = [pnl for _, pnl in values if pnl is not None]
            hour_summary[f"{hour:02d}:00 UTC"] = {
                "n": len(values),
                "good_rejects": good,
                "bad_rejects": bad,
                "ambiguous_rejects": ambiguous,
                "bad_reject_rate": round(bad / informative * 100.0, 2) if informative else None,
                "avg_counterfactual_pnl": round(sum(pnl_values) / len(pnl_values), 4) if pnl_values else None,
            }

        high_bad_reject_hours = dict(
            sorted(
                (
                    (hour, summary)
                    for hour, summary in hour_summary.items()
                    if int(summary.get("n") or 0) >= 10 and summary.get("bad_reject_rate") is not None
                ),
                key=lambda item: float(item[1].get("bad_reject_rate") or 0.0),
                reverse=True,
            )[:8]
        )

        # Build ai_vs_rule timing summary (rule_counterfactual entries by hour)
        ai_vs_rule_by_hour: Dict[str, Dict[str, Any]] = {}
        for hour, counts in ai_rule_by_hour.items():
            n = counts["total"]
            rr = counts["rule_was_right"]
            ar = counts["ai_was_right"]
            ai_vs_rule_by_hour[f"{hour:02d}:00 UTC"] = {
                "total": n,
                "rule_was_right": rr,
                "ai_was_right": ar,
                "rule_accuracy_pct": round(rr / n * 100.0, 2) if n > 0 else None,
                "ai_accuracy_pct": round(ar / n * 100.0, 2) if n > 0 else None,
            }

        return {
            "available": True,
            "total_candidates": len(rows),
            "quality_counts": dict(quality_counts),
            "high_bad_reject_hours": high_bad_reject_hours,
            "gate_quality_note": "high_bad_reject_hours uses ai_directional candidates only",
            "ai_vs_rule_by_hour": ai_vs_rule_by_hour,
        }

    def calculate(
        self,
        configured_hours_utc: Optional[List[int]] = None,
        lookback_days: int = 365,
        min_hour_samples: int = 8,
        min_total_samples: int = 120,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        lookback_start_dt = now - timedelta(days=max(lookback_days, 14))
        lookback_start = lookback_start_dt.isoformat()

        signals = self._fetch_all(
            "platform_signals",
            "signal_id, token_symbol, direction, generated_by_agent, leverage, analysis_timestamp, created_at, status",
            order_col="created_at",
            filters=[("gte", ("created_at", lookback_start))],
            max_rows=20_000,
        )
        performance_rows = self._fetch_all(
            "platform_signal_performance_tracking",
            "signal_id, outcome, actual_pnl_percent, leveraged_pnl_percent, exit_timestamp, last_updated",
            order_col="last_updated",
            max_rows=20_000,
        )
        perf_by_signal = {
            str(row.get("signal_id")): row
            for row in performance_rows
            if row.get("signal_id")
        }

        informative_rows: List[Dict[str, Any]] = []
        by_hour: Dict[int, List[float]] = defaultdict(list)
        by_window: Dict[str, List[float]] = defaultdict(list)
        by_day: Dict[str, List[float]] = defaultdict(list)

        for signal in signals:
            timestamp = self._parse_datetime(signal.get("analysis_timestamp") or signal.get("created_at"))
            if not timestamp:
                continue
            perf = perf_by_signal.get(str(signal.get("signal_id")), {})
            pnl = self._normalized_final_pnl(signal, perf)
            if pnl is None:
                continue

            hour = timestamp.hour
            window_start = (hour // 6) * 6
            window_label = f"{window_start:02d}:00-{window_start + 5:02d}:59 UTC"
            by_hour[hour].append(pnl)
            by_window[window_label].append(pnl)
            by_day[timestamp.strftime("%a")].append(pnl)
            informative_rows.append(
                {
                    "signal_id": signal.get("signal_id"),
                    "token_symbol": signal.get("token_symbol"),
                    "direction": signal.get("direction"),
                    "timestamp": timestamp.isoformat(),
                    "pnl": pnl,
                }
            )

        offset_schedules: Dict[str, Dict[str, Any]] = {}
        for offset in range(6):
            hours = [offset, offset + 6, offset + 12, offset + 18]
            hour_summaries = {
                f"{hour:02d}:00 UTC": self._summary(by_hour.get(hour, []))
                for hour in hours
            }
            values = [pnl for hour in hours for pnl in by_hour.get(hour, [])]
            summary = self._summary(values)
            schedule_label = self._schedule_label(hours)
            min_n = min(int(hour_summary.get("n") or 0) for hour_summary in hour_summaries.values())
            offset_schedules[schedule_label] = {
                "hours_utc": hours,
                "summary": summary,
                "hour_summaries": hour_summaries,
                "min_hour_samples": min_n,
                "sample_ok": min_n >= min_hour_samples,
                "balanced_score": self._score_offset_schedule(summary, hour_summaries, min_hour_samples),
            }

        ranked_schedules = dict(
            sorted(
                offset_schedules.items(),
                key=lambda item: float(item[1].get("balanced_score") or -999.0),
                reverse=True,
            )
        )
        configured_hours = sorted(set(configured_hours_utc or []))
        configured_summary = None
        if configured_hours:
            configured_values = [pnl for hour in configured_hours for pnl in by_hour.get(hour, [])]
            configured_summary = {
                "hours_utc": configured_hours,
                "summary": self._summary(configured_values),
                "hour_summaries": {
                    f"{hour:02d}:00 UTC": self._summary(by_hour.get(hour, []))
                    for hour in configured_hours
                },
            }

        raw_recommended_label, raw_recommended = next(iter(ranked_schedules.items())) if ranked_schedules else (None, {})
        recommended_label, recommended = raw_recommended_label, raw_recommended
        recommendation_reason = "ranked_best_schedule"

        if len(informative_rows) < min_total_samples and configured_hours:
            configured_label = self._schedule_label(configured_hours)
            recommended_label = configured_label
            recommended = ranked_schedules.get(configured_label) or {
                "hours_utc": configured_hours,
                "summary": (configured_summary or {}).get("summary"),
                "sample_ok": False,
                "min_hour_samples": 0,
                "balanced_score": None,
            }
            recommendation_reason = "insufficient_total_samples_keep_configured_schedule"

        exact_hours = dict(
            sorted(
                (
                    (f"{hour:02d}:00 UTC", self._summary(values))
                    for hour, values in by_hour.items()
                    if len(values) >= max(4, min_hour_samples)
                ),
                key=lambda item: float(item[1].get("avg_pnl") or -999.0),
                reverse=True,
            )[:12]
        )

        sample_count = len(informative_rows)
        confidence = min(0.95, max(0.35, sample_count / 350.0))
        if recommended and not recommended.get("sample_ok"):
            confidence = min(confidence, 0.55)

        return {
            "lookback_days": max(lookback_days, 14),
            "generated_at": now.isoformat(),
            "informative_final_outcomes": sample_count,
            "overall": self._summary([row["pnl"] for row in informative_rows]),
            "configured_schedule": configured_summary,
            "recommended_schedule": {
                "label": recommended_label,
                "hours_utc": recommended.get("hours_utc", []),
                "balanced_score": recommended.get("balanced_score"),
                "summary": recommended.get("summary"),
                "sample_ok": recommended.get("sample_ok"),
                "min_hour_samples": recommended.get("min_hour_samples"),
                "reason": recommendation_reason,
                "raw_ranked_best": {
                    "label": raw_recommended_label,
                    "hours_utc": raw_recommended.get("hours_utc", []),
                    "balanced_score": raw_recommended.get("balanced_score"),
                } if raw_recommended_label else None,
            },
            "ranked_6h_anchor_schedules": ranked_schedules,
            "six_hour_windows": dict(sorted((label, self._summary(values)) for label, values in by_window.items())),
            "top_exact_hours": exact_hours,
            "day_of_week": dict(sorted((day, self._summary(values)) for day, values in by_day.items())),
            "rejected_candidate_timing": self._analyze_reject_hours(lookback_start),
            "confidence_score": round(confidence, 4),
        }

    def persist_schedule_run(
        self,
        run_id: str,
        signals_generated: int,
        schedule_context: Dict[str, Any],
        configured_hours_utc: List[int],
    ) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "agent_type": "yuki",
            "signal_id": f"schedule_run_{run_id}_{uuid.uuid4().hex[:8]}",
            "learning_level": "meta",
            "learning_type": "signal_generation_schedule_run",
            "confidence_score": 0.8,
            "timestamp": now.isoformat(),
            "insights": {
                "run_id": run_id,
                "signals_generated": int(signals_generated or 0),
                "configured_hours_utc": configured_hours_utc,
                "trigger": schedule_context.get("trigger", "scheduled"),
                "scheduled_for_utc": schedule_context.get("scheduled_for_utc"),
                "actual_run_at_utc": schedule_context.get("actual_run_at_utc") or now.isoformat(),
                "slot_hour_utc": schedule_context.get("slot_hour_utc"),
                "optimization_tracking": "enabled",
            },
            "performance_impact": {
                "signals_generated": int(signals_generated or 0),
                "slot_hour_utc": schedule_context.get("slot_hour_utc"),
            },
        }
        self.db.from_("agent_learning_insights").insert(payload).execute()

    def has_completed_schedule_run(self, scheduled_for_utc: Any) -> bool:
        """Return whether the exact scheduled slot already reached run tracking."""
        scheduled_for = self._parse_datetime(scheduled_for_utc)
        if scheduled_for is None:
            return False
        try:
            rows = (
                self.db.from_("agent_learning_insights")
                .select("insights,timestamp")
                .eq("agent_type", "yuki")
                .eq("learning_type", "signal_generation_schedule_run")
                .order("timestamp", desc=True)
                .limit(200)
                .execute()
            ).data or []
        except Exception as exc:
            logger.warning("Could not verify completed signal schedule slot: %s", exc)
            return False

        for row in rows:
            insights = row.get("insights") or {}
            if str(insights.get("trigger") or "").lower() != "scheduled":
                continue
            completed_slot = self._parse_datetime(insights.get("scheduled_for_utc"))
            if completed_slot is None:
                continue
            if abs((completed_slot - scheduled_for).total_seconds()) < 1:
                return True
        return False

    def persist_optimization_snapshot(
        self,
        configured_hours_utc: Optional[List[int]],
        lookback_days: int = 365,
        min_hour_samples: int = 8,
        min_total_samples: int = 120,
        triggered_by: str = "scheduled_run",
        auto_apply_enabled: bool = False,
    ) -> Dict[str, Any]:
        result = self.calculate(
            configured_hours_utc=configured_hours_utc,
            lookback_days=lookback_days,
            min_hour_samples=min_hour_samples,
            min_total_samples=min_total_samples,
        )

        recommended = result.get("recommended_schedule") or {}
        configured = result.get("configured_schedule") or {}
        configured_hours = configured.get("hours_utc") or configured_hours_utc or []
        recommendation = (
            f"Use signal generation schedule {recommended.get('label')}"
            if recommended.get("label")
            else "Keep collecting timing samples before changing schedule."
        )
        if configured_hours and recommended.get("hours_utc") == configured_hours:
            recommendation = f"Current schedule matches optimizer recommendation: {recommended.get('label')}."

        now = datetime.now(timezone.utc)
        insight_payload = {
            "agent_type": "yuki",
            "signal_id": f"schedule_opt_{uuid.uuid4().hex[:12]}",
            "learning_level": "meta",
            "learning_type": "signal_schedule_optimization",
            "confidence_score": result.get("confidence_score", 0.5),
            "timestamp": now.isoformat(),
            "insights": {
                **result,
                "triggered_by": triggered_by,
                "recommendation": recommendation,
                "auto_apply_enabled": bool(auto_apply_enabled),
                "auto_apply": bool(auto_apply_enabled),
            },
            "performance_impact": {
                "recommended_hours_utc": recommended.get("hours_utc", []),
                "configured_hours_utc": configured_hours,
                "recommended_score": recommended.get("balanced_score"),
                "informative_final_outcomes": result.get("informative_final_outcomes"),
            },
        }
        self.db.from_("agent_learning_insights").insert(insight_payload).execute()

        try:
            analysis_date = now.date().isoformat()
            meta_record = {
                "analysis_date": analysis_date,
                "agent_type": "all",
                "time_period_hours": 24 * int(result.get("lookback_days") or lookback_days),
                "strategy_effectiveness": result.get("overall") or {},
                "cross_agent_performance": {},
                "optimal_market_conditions": {
                    "recommended_schedule": recommended,
                    "configured_schedule": configured,
                },
                "learning_convergence_rate": result.get("confidence_score", 0.5),
                "adaptation_recommendations": [recommendation],
                "risk_parameter_adjustments": {
                    "auto_apply": False,
                    "configured_hours_utc": configured_hours,
                    "recommended_hours_utc": recommended.get("hours_utc", []),
                },
                "temporal_performance_patterns": result,
                "analysis_timestamp": now.isoformat(),
                "updated_at": now.isoformat(),
            }

            existing = self.db.from_("meta_learning_performance").select("id").eq("analysis_date", analysis_date).eq("agent_type", "all").execute()
            if existing.data:
                self.db.from_("meta_learning_performance").update(meta_record).eq("id", existing.data[0]["id"]).execute()
            else:
                self.db.from_("meta_learning_performance").insert(meta_record).execute()
        except Exception as exc:
            logger.debug("Could not upsert meta schedule optimization snapshot: %s", exc)

        return result

    def persist_auto_apply_decision(
        self,
        optimization_result: Dict[str, Any],
        previous_hours_utc: List[int],
        active_hours_utc: List[int],
        applied: bool,
        reason: str,
    ) -> None:
        """Persist whether the worker accepted or rejected a schedule change."""
        recommended = optimization_result.get("recommended_schedule") or {}
        now = datetime.now(timezone.utc)
        self.db.from_("agent_learning_insights").insert(
            {
                "agent_type": "yuki",
                "signal_id": f"schedule_auto_apply_{uuid.uuid4().hex[:12]}",
                "learning_level": "meta",
                "learning_type": "signal_schedule_auto_apply",
                "confidence_score": optimization_result.get("confidence_score", 0.5),
                "timestamp": now.isoformat(),
                "insights": {
                    "applied": bool(applied),
                    "reason": reason,
                    "previous_hours_utc": previous_hours_utc,
                    "active_hours_utc": active_hours_utc,
                    "recommended_hours_utc": recommended.get("hours_utc", []),
                    "recommended_label": recommended.get("label"),
                    "recommended_score": recommended.get("balanced_score"),
                    "confidence_score": optimization_result.get("confidence_score"),
                    "informative_final_outcomes": optimization_result.get("informative_final_outcomes"),
                    "generated_at": optimization_result.get("generated_at"),
                },
                "performance_impact": {
                    "applied": bool(applied),
                    "previous_hours_utc": previous_hours_utc,
                    "active_hours_utc": active_hours_utc,
                    "recommended_hours_utc": recommended.get("hours_utc", []),
                    "recommended_score": recommended.get("balanced_score"),
                    "informative_final_outcomes": optimization_result.get("informative_final_outcomes"),
                },
            }
        ).execute()


_signal_schedule_optimizer: Optional[SignalScheduleOptimizer] = None


def get_signal_schedule_optimizer() -> SignalScheduleOptimizer:
    global _signal_schedule_optimizer
    if _signal_schedule_optimizer is None:
        _signal_schedule_optimizer = SignalScheduleOptimizer()
    return _signal_schedule_optimizer
