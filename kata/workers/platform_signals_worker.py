#!/usr/bin/env python3
"""
Platform Signals Background Worker for Flow AI Trading Platform.

This worker runs the platform analysis engine in the background to generate
signals on fixed UTC windows learned from signal performance. Optimized for
cloud deployment.
"""

import asyncio
import gc
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Add the backend directory to Python path for cloud runtime compatibility
current_file = os.path.abspath(__file__)
workers_dir = os.path.dirname(current_file)  # /path/to/backend/app/workers
app_dir = os.path.dirname(workers_dir)       # /path/to/backend/app
backend_dir = os.path.dirname(app_dir)       # /path/to/backend

sys.path.insert(0, backend_dir)

from kata.services.unified_signal_generator import UnifiedSignalGenerator
from kata.services.platform_signal_service import get_platform_signal_service, PlatformSignal, SignalPool
from kata.services.lightweight_performance_service import get_lightweight_performance_service
from kata.services.simple_websocket_performance_monitor import get_simple_websocket_monitor
from kata.services.learning_integration_hook import get_learning_integration_hook
from kata.services.signal_schedule_optimizer import (
    DEFAULT_GENERATION_HOURS_UTC,
    get_signal_schedule_optimizer,
    normalize_schedule_hours,
)
import uuid
from datetime import datetime, timedelta, timezone

# Configure logging for cloud deployment - cleaner worker logs
log_level = logging.INFO  # Always use INFO level for cleaner logs
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only console logging for cloud workers
    ]
)

# Reduce noise from HTTP libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING) 
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class PlatformSignalsWorker:
    """Background worker for platform signal generation using Platform Analysis Engine."""
    
    def __init__(self):
        """Initialize the worker."""
        self.analysis_engine = None
        self.platform_service = None
        self.performance_service = None
        self.simple_websocket_monitor = None
        self.learning_hook = get_learning_integration_hook()
        self.schedule_optimizer = None
        self.analysis_task = None
        self.generation_request_task = None
        self.performance_task = None
        self.websocket_task = None
        self.shutdown_requested = False
        self.worker_instance_id = uuid.uuid4().hex[:12]
        self._analysis_lock = asyncio.Lock()
        self._schedule_changed_event = asyncio.Event()
        self.last_reject_learning_refresh_at = None
        self.last_schedule_optimization_at = None
        self.event_reanalysis_in_progress = False
        self.analysis_in_progress = False
        # The Learning Dashboard persists the authoritative schedule. This
        # default is used only during the few seconds before services load.
        self.generation_hours_utc = list(DEFAULT_GENERATION_HOURS_UTC)
        logger.info("Platform Signals Worker initialized")

    @staticmethod
    def _current_rss_mb() -> float:
        """Current resident memory in MB (Linux /proc; ru_maxrss peak as fallback)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            pass
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KB, macOS reports bytes
            return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    def _log_memory(self, tag: str):
        rss = self._current_rss_mb()
        if rss <= 0:
            return 0.0
        if rss > 400:
            logger.warning(f"🧠 Memory [{tag}]: {rss:.0f}MB RSS - approaching 512MB limit")
        else:
            logger.info(f"🧠 Memory [{tag}]: {rss:.0f}MB RSS")
        return rss

    @staticmethod
    def _trim_allocator_memory() -> bool:
        """Return free Python/glibc arenas to Linux before container OOM pressure."""
        gc.collect()
        if not sys.platform.startswith("linux"):
            return False
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            return bool(malloc_trim(0))
        except (AttributeError, OSError):
            return False

    def _release_analysis_memory(self):
        """Drop per-run caches held between analysis cycles and force a GC pass."""
        generator = getattr(self, "signal_generator", None)
        if generator is not None:
            for cache_attr in (
                "_ohlcv_cache",
                "_opportunity_score_cache",
                "_orderbook_cache",
                "_liq_cluster_cache",
                "_positioning_cache",
                "_open_interest_cache",
                "_hyperliquid_snapshot_cache",
            ):
                cache = getattr(generator, cache_attr, None)
                if isinstance(cache, dict):
                    cache.clear()
        gc.collect()

    def _schedule_label(self) -> str:
        return ", ".join(f"{hour:02d}:00 UTC" for hour in self.generation_hours_utc)

    def _next_scheduled_analysis_time(self, now: datetime = None) -> datetime:
        """Return the next configured scheduled run time in UTC."""
        now = now or datetime.now(timezone.utc)
        now = now.astimezone(timezone.utc)

        for hour in self.generation_hours_utc:
            candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate

        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=self.generation_hours_utc[0], minute=0, second=0, microsecond=0)

    def _previous_scheduled_analysis_time(self, now: datetime = None) -> datetime:
        """Return the most recent configured slot at or before now."""
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        for hour in reversed(self.generation_hours_utc):
            candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate <= now:
                return candidate
        yesterday = now - timedelta(days=1)
        return yesterday.replace(
            hour=self.generation_hours_utc[-1],
            minute=0,
            second=0,
            microsecond=0,
        )

    async def _reload_applied_schedule(self) -> bool:
        """Hot-reload the dashboard schedule; return True when it changed."""
        if not self.schedule_optimizer:
            return False
        try:
            hours = await asyncio.to_thread(
                self.schedule_optimizer.load_active_schedule,
                DEFAULT_GENERATION_HOURS_UTC,
            )
        except Exception as exc:
            logger.warning("Could not reload dashboard signal schedule: %s", exc)
            return False

        return self._set_generation_hours(hours)

    def _set_generation_hours(self, hours) -> bool:
        """Use a validated dashboard schedule and report whether it changed."""
        normalized_hours = normalize_schedule_hours(hours)
        if not normalized_hours:
            raise ValueError("The dashboard schedule did not contain a valid UTC hour")
        if normalized_hours == self.generation_hours_utc:
            return False
        previous = list(self.generation_hours_utc)
        self.generation_hours_utc = normalized_hours
        logger.info(
            "🕒 Dashboard signal schedule reloaded: %s -> %s",
            self._schedule_label_for_hours(previous),
            self._schedule_label(),
        )
        return True

    async def _enqueue_scheduled_generation(
        self,
        scheduled_for: datetime,
        *,
        catch_up: bool = False,
    ) -> Dict[str, Any]:
        request = await self.platform_service.enqueue_platform_generation_request(
            requested_by="schedule_worker",
            trigger="scheduled",
            scheduled_for_utc=scheduled_for.isoformat(),
            configured_hours_utc=self.generation_hours_utc,
        )
        logger.info(
            "%s scheduled generation for %s (request=%s status=%s)",
            "♻️ Queued missed" if catch_up else "🗓️ Queued",
            scheduled_for.isoformat(),
            request.get("request_id"),
            request.get("status"),
        )
        return request

    async def _enqueue_missed_schedule_if_due(self) -> Optional[Dict[str, Any]]:
        """Recover a recent slot interrupted by a deploy or worker restart."""
        if not self.platform_service:
            return None
        now = datetime.now(timezone.utc)
        scheduled_for = self._previous_scheduled_analysis_time(now)
        catch_up_minutes = max(
            15,
            min(
                360,
                int(os.getenv("PLATFORM_SIGNAL_SCHEDULE_CATCHUP_MINUTES", "180") or "180"),
            ),
        )
        age_minutes = (now - scheduled_for).total_seconds() / 60.0
        if age_minutes > catch_up_minutes:
            return None
        if self.schedule_optimizer:
            completed = await asyncio.to_thread(
                self.schedule_optimizer.has_completed_schedule_run,
                scheduled_for,
            )
            if completed:
                logger.info(
                    "✅ Scheduled slot %s already completed; no catch-up needed",
                    scheduled_for.isoformat(),
                )
                return None
        return await self._enqueue_scheduled_generation(scheduled_for, catch_up=True)

    async def _sleep_until(self, target_time: datetime) -> bool:
        """Sleep until a slot, waking only for a queued schedule-change event."""
        wait_seconds = (target_time - datetime.now(timezone.utc)).total_seconds()
        if wait_seconds <= 0:
            return True
        try:
            await asyncio.wait_for(
                self._schedule_changed_event.wait(),
                timeout=wait_seconds,
            )
        except asyncio.TimeoutError:
            return True
        self._schedule_changed_event.clear()
        return False

    def _startup_generation_enabled(self) -> bool:
        return os.getenv("PLATFORM_SIGNAL_RUN_ON_STARTUP", "false").lower() in {"1", "true", "yes", "on"}

    def _schedule_auto_optimize_enabled(self) -> bool:
        return os.getenv("SIGNAL_SCHEDULE_AUTO_OPTIMIZE", "false").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _schedule_label_for_hours(hours):
        return "/".join(f"{int(hour):02d}" for hour in sorted(set(hours or []))) + " UTC"

    def _schedule_auto_apply_decision(self, optimization_result, min_total_samples):
        """Return whether a recommended timing schedule should replace the active one."""
        if not self._schedule_auto_optimize_enabled():
            return False, "auto_optimize_disabled"

        recommended = optimization_result.get("recommended_schedule") or {}
        recommended_hours = sorted({
            int(hour)
            for hour in recommended.get("hours_utc", [])
            if isinstance(hour, (int, float)) or str(hour).isdigit()
        })
        active_hours = sorted(set(self.generation_hours_utc or []))
        if not recommended_hours:
            return False, "missing_recommended_hours"
        if recommended_hours == active_hours:
            return False, "active_schedule_already_matches"
        if recommended.get("reason") != "ranked_best_schedule":
            return False, str(recommended.get("reason") or "recommendation_not_ranked_best")
        if not recommended.get("sample_ok"):
            return False, "recommended_hours_need_more_samples"

        informative_outcomes = int(optimization_result.get("informative_final_outcomes") or 0)
        if informative_outcomes < int(min_total_samples or 0):
            return False, "insufficient_total_outcomes"

        min_confidence = self._safe_float(os.getenv("SIGNAL_SCHEDULE_AUTO_MIN_CONFIDENCE", "0.70"), 0.70)
        confidence = self._safe_float(optimization_result.get("confidence_score"), 0.0)
        if confidence < min_confidence:
            return False, "confidence_below_auto_apply_threshold"

        ranked = optimization_result.get("ranked_6h_anchor_schedules") or {}
        active_label = self._schedule_label_for_hours(active_hours)
        active_score = self._safe_float((ranked.get(active_label) or {}).get("balanced_score"), 0.0)
        recommended_score = self._safe_float(recommended.get("balanced_score"), 0.0)
        min_score_improvement = self._safe_float(os.getenv("SIGNAL_SCHEDULE_AUTO_MIN_SCORE_DELTA", "1.00"), 1.0)
        if (recommended_score - active_score) < min_score_improvement:
            return False, "score_improvement_below_auto_apply_threshold"

        return True, "auto_applied_ranked_best_schedule"

    @staticmethod
    def _task_running(task) -> bool:
        return bool(task and not task.done())

    async def _ensure_background_tasks_healthy(self):
        """Restart failed background tasks to keep the worker self-healing."""
        if self.shutdown_requested:
            return

        if self.analysis_task and self.analysis_task.done():
            if self.analysis_task.cancelled():
                logger.warning("⚠️ Analysis task was cancelled unexpectedly; restarting...")
            elif self.analysis_task.exception():
                logger.error(f"❌ Analysis task crashed: {self.analysis_task.exception()}; restarting...")
            self.analysis_task = asyncio.create_task(self._run_analysis_cycle())

        if self.generation_request_task and self.generation_request_task.done():
            if self.generation_request_task.cancelled():
                logger.warning("⚠️ Generation request task was cancelled unexpectedly; restarting...")
            elif self.generation_request_task.exception():
                logger.error(
                    "❌ Generation request task crashed: %s; restarting...",
                    self.generation_request_task.exception(),
                )
            self.generation_request_task = asyncio.create_task(
                self._run_generation_request_cycle()
            )

        if self.performance_task and self.performance_task.done():
            if self.performance_task.cancelled():
                logger.warning("⚠️ Signal refresh task was cancelled unexpectedly; restarting...")
            elif self.performance_task.exception():
                logger.error(f"❌ Signal refresh task crashed: {self.performance_task.exception()}; restarting...")
            self.performance_task = asyncio.create_task(self._run_signal_refresh_cycle())

        websocket_running = (
            self.simple_websocket_monitor.get_status().get("is_running", False)
            if self.simple_websocket_monitor else False
        )
        if (self.websocket_task and self.websocket_task.done()) or (self.websocket_task and not websocket_running):
            if self.websocket_task and self.websocket_task.done():
                if self.websocket_task.cancelled():
                    logger.warning("⚠️ WebSocket task was cancelled unexpectedly; restarting...")
                elif self.websocket_task.exception():
                    logger.error(f"❌ WebSocket task crashed: {self.websocket_task.exception()}; restarting...")
            else:
                logger.warning("⚠️ WebSocket monitor reported not running; restarting task...")
            self.websocket_task = asyncio.create_task(self.simple_websocket_monitor.start_monitoring())

        # Pick up allocations created since boot (via the web API) so their position
        # monitors run here even if the web process restarts. Skips known allocations.
        try:
            from kata.services.agent_allocation_service import get_agent_allocation_service

            await get_agent_allocation_service()._rehydrate_active_yuki_allocations()
        except Exception as rehydrate_error:
            logger.error(f"⚠️ Periodic allocation rehydration failed: {rehydrate_error}")

    async def start(self):
        """Start the background worker using Platform Analysis Engine."""
        try:
            logger.info("🚀 Starting Platform Signals Worker with Platform Analysis Engine...")
            
            # Initialize services
            await self._initialize_services()
            await self._reload_applied_schedule()

            # Rehydrate active Yuki allocations immediately so position monitors
            # (fill detection for resting entries, protective orders, risk checks)
            # run from boot instead of waiting for the first generation cycle.
            try:
                from kata.services.agent_allocation_service import get_agent_allocation_service

                restored = await get_agent_allocation_service()._rehydrate_active_yuki_allocations()
                logger.info(f"✅ Startup allocation rehydration complete ({restored} restored); position monitors active")
            except Exception as rehydrate_error:
                logger.error(f"⚠️ Startup allocation rehydration failed (will retry on first Yuki cycle): {rehydrate_error}")

            await self._refresh_schedule_optimization_if_due(triggered_by="worker_startup")

            if self._startup_generation_enabled():
                logger.info("🎯 Startup signal generation override enabled; running initial analysis...")
                await self._run_single_analysis(schedule_context={
                    "trigger": "startup_override",
                    "actual_run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "slot_hour_utc": datetime.now(timezone.utc).hour,
                })
                logger.info("✅ Startup analysis completed")
            else:
                await self._enqueue_missed_schedule_if_due()
                next_run = self._next_scheduled_analysis_time()
                logger.info(
                    "⏭️ Skipping immediate startup analysis; next scheduled generation is %s",
                    next_run.isoformat(),
                )

            # Start the database-controlled schedule producer. Actual analysis
            # runs through the durable request consumer below.
            self.analysis_task = asyncio.create_task(self._run_analysis_cycle())
            logger.info("✅ Background analysis cycle started - dashboard schedule: %s", self._schedule_label())

            # Consume durable control requests written by the web API. The same
            # existing queue handles manual runs and one-off schedule changes.
            self.generation_request_task = asyncio.create_task(
                self._run_generation_request_cycle()
            )
            logger.info("✅ Worker control request consumer started")

            # Start simple real-time WebSocket monitoring (zero API calls)
            logger.info("🚀 Starting simple WebSocket performance monitoring...")
            self.websocket_task = asyncio.create_task(self.simple_websocket_monitor.start_monitoring())
            logger.info("✅ Simple WebSocket monitoring started - real-time exit detection")

            # Start periodic signal refresh task (every 10 minutes - much less frequent)
            self.performance_task = asyncio.create_task(self._run_signal_refresh_cycle())
            logger.info("✅ Signal refresh cycle started - runs every 10 minutes")

            # Keep the worker running with health checks
            logger.info("🔄 Worker is now running in background...")
            logger.info("💚 Worker health monitoring active")
            
            health_check_counter = 0
            
            # Wait for shutdown signal
            while not self.shutdown_requested:
                await asyncio.sleep(300)  # Check every 5 minutes
                rss = self._log_memory("health-check")
                if rss >= 400:
                    trimmed = self._trim_allocator_memory()
                    after_trim = self._current_rss_mb()
                    logger.warning(
                        "🧹 Memory guard: %.0fMB → %.0fMB RSS (allocator trim=%s)",
                        rss,
                        after_trim,
                        trimmed,
                    )
                await self._ensure_background_tasks_healthy()
                
                # Periodic health logging
                health_check_counter += 1
                if health_check_counter % 12 == 0:  # Every hour
                    analysis_running = self._task_running(self.analysis_task)
                    request_consumer_running = self._task_running(self.generation_request_task)
                    refresh_running = self._task_running(self.performance_task)
                    websocket_stats = self.simple_websocket_monitor.get_status() if self.simple_websocket_monitor else {}

                    logger.info(
                        f"💚 Health check #{health_check_counter}: "
                        f"Analysis cycle running={analysis_running}, "
                        f"Request consumer running={request_consumer_running}, "
                        f"Signal refresh running={refresh_running}"
                    )
                    logger.info(f"📡 Simple WebSocket: Running={websocket_stats.get('is_running', False)}, Signals={websocket_stats.get('signals_monitored', 0)}")

                    # Check if we have any signals in the database
                    try:
                        signals = await self.platform_service.get_active_signals(limit=50)  # Check all active signals
                        logger.info(f"📊 Current active signals: {len(signals)}")
                        if len(signals) > 0:
                            symbols = [s.token_symbol for s in signals[:5]]  # Show first 5 symbols
                            logger.info(f"📊 Active symbols: {', '.join(symbols)}{' (+more)' if len(signals) > 5 else ''}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not check signals: {e}")
                
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
            raise
    
    async def _initialize_services(self):
        """Initialize Unified Signal Generator and supporting services."""
        try:
            # Initialize the unified signal generation system
            # This includes market discovery, technical analysis, and AI decision making
            self.signal_generator = UnifiedSignalGenerator()
            await self.signal_generator._initialize_services()
            # Initialize platform signal service
            self.platform_service = get_platform_signal_service()

            # Initialize lightweight performance monitoring service (backup/emergency only)
            self.performance_service = get_lightweight_performance_service()

            # Initialize simple WebSocket monitor (real-time, zero API calls)
            self.simple_websocket_monitor = get_simple_websocket_monitor()

            # Initialize schedule optimizer/tracker
            self.schedule_optimizer = get_signal_schedule_optimizer()

            logger.info("✅ Platform Analysis Engine and Signal services initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            raise

    async def _run_analysis_cycle(self):
        """Queue durable analysis requests on the dashboard-controlled schedule."""
        logger.info("🔄 Starting scheduled analysis cycle: %s", self._schedule_label())
        
        while not self.shutdown_requested:
            try:
                scheduled_for = self._next_scheduled_analysis_time()
                wait_seconds = max(0, int((scheduled_for - datetime.now(timezone.utc)).total_seconds()))
                logger.info(
                    "⏰ Next scheduled signal generation: %s (%0.2f hours)",
                    scheduled_for.isoformat(),
                    wait_seconds / 3600.0,
                )

                reached_slot = await self._sleep_until(scheduled_for)
                if self.shutdown_requested:
                    return
                if not reached_slot:
                    continue

                await self._enqueue_scheduled_generation(scheduled_for)
                    
            except Exception as e:
                logger.error(f"❌ Analysis cycle error: {e}")
                await asyncio.sleep(30)

    async def _run_generation_request_cycle(self):
        """Claim durable generation and schedule-control requests."""
        poll_seconds = max(
            2,
            min(
                60,
                int(os.getenv("PLATFORM_SIGNAL_REQUEST_POLL_SECONDS", "5") or "5"),
            ),
        )
        logger.info(
            "🔔 Listening for worker control requests every %ss (worker=%s)",
            poll_seconds,
            self.worker_instance_id,
        )

        while not self.shutdown_requested:
            request = None
            try:
                if not self._analysis_lock.locked():
                    request = await self.platform_service.claim_platform_generation_request(
                        self.worker_instance_id
                    )
                if not request:
                    await asyncio.sleep(poll_seconds)
                    continue

                request_id = str(request.get("id") or "")
                insights = request.get("insights") or {}
                requested_at = insights.get("requested_at")
                trigger = str(insights.get("trigger") or "manual_override")
                scheduled_for = insights.get("scheduled_for_utc")
                slot_hour = insights.get("slot_hour_utc")
                if slot_hour is None:
                    slot_hour = datetime.now(timezone.utc).hour
                logger.info(
                    "▶️ Running queued %s platform generation request %s",
                    trigger,
                    request_id,
                )
                if trigger == "schedule_change":
                    changed = self._set_generation_hours(
                        insights.get("configured_hours_utc")
                    )
                    self._schedule_changed_event.set()
                    if changed:
                        try:
                            await self._enqueue_missed_schedule_if_due()
                        except Exception as catch_up_error:
                            logger.warning(
                                "Could not queue schedule catch-up after applying %s: %s",
                                self._schedule_label(),
                                catch_up_error,
                            )
                    await self.platform_service.finish_platform_generation_request(
                        request_id,
                        "completed",
                        signals_generated=0,
                    )
                    logger.info(
                        "✅ Applied queued dashboard schedule change %s (changed=%s)",
                        request_id,
                        changed,
                    )
                    continue
                signals_generated = await self._run_single_analysis(schedule_context={
                    "trigger": trigger,
                    "generation_request_id": request_id,
                    "requested_at_utc": requested_at,
                    "scheduled_for_utc": scheduled_for,
                    "actual_run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "slot_hour_utc": int(slot_hour),
                    "configured_hours_utc": insights.get("configured_hours_utc") or self.generation_hours_utc,
                })
                await self.platform_service.finish_platform_generation_request(
                    request_id,
                    "completed",
                    signals_generated=int(signals_generated or 0),
                )
                logger.info(
                    "✅ %s generation request %s completed (%s signals)",
                    trigger,
                    request_id,
                    int(signals_generated or 0),
                )
            except asyncio.CancelledError:
                if request and request.get("id"):
                    await self.platform_service.finish_platform_generation_request(
                        str(request["id"]),
                        "pending",
                        error="worker shutdown interrupted the request; queued for retry",
                    )
                raise
            except Exception as exc:
                request_id = str((request or {}).get("id") or "")
                logger.error(
                    "❌ Platform generation request %s failed: %s",
                    request_id or "unknown",
                    exc,
                )
                if request_id:
                    await self.platform_service.finish_platform_generation_request(
                        request_id,
                        "pending" if str((request or {}).get("insights", {}).get("trigger")) == "scheduled" else "failed",
                        error=str(exc),
                    )
                await asyncio.sleep(max(30, poll_seconds))

    async def _run_signal_refresh_cycle(self):
        """Run periodic signal refresh cycle every 10 minutes (WebSocket handles real-time monitoring)."""
        interval_minutes = 10
        logger.info(f"🔄 Starting signal refresh cycle - will run every {interval_minutes} minutes")

        while not self.shutdown_requested:
            try:
                # Position protection must run before logo retries and queued
                # LLM re-analysis, either of which can take several minutes.
                await self._run_ryu_position_risk_cycle(
                    triggered_by="ten_minute_refresh"
                )

                # Simple refresh - just retry logo fetching (WebSocket auto-refreshes signals)

                # Also retry logo fetching for signals missing logos (fast, filtered)
                try:
                    logger.debug("🖼️ Refresh cycle: retrying missing logos (fast pass)...")
                    await self._retry_logo_fetching()
                except Exception as logo_err:
                    logger.warning(f"Logo retry during refresh failed: {logo_err}")

                try:
                    await self._refresh_reject_learning_if_due()
                except Exception as learning_err:
                    logger.warning(f"Reject learning refresh during signal cycle failed: {learning_err}")

                await self._process_event_reanalysis_requests()
                await self._run_yuki_live_signal_cycle(triggered_by="ten_minute_refresh")
                # Wait for next cycle (10 minutes)
                wait_seconds = interval_minutes * 60
                logger.debug(f"⏰ Next refresh in {interval_minutes} minutes...")

                # Sleep in chunks to allow for shutdown
                for _ in range(wait_seconds // 60):  # Sleep in 1-minute chunks
                    if self.shutdown_requested:
                        return
                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"❌ Signal refresh cycle error: {e}")
                # Wait 2 minutes before retrying on error
                await asyncio.sleep(120)

    async def _run_yuki_live_signal_cycle(self, triggered_by: str) -> None:
        """Re-sweep active platform signals for live Yuki allocations."""
        try:
            from kata.services.agent_allocation_service import get_agent_allocation_service

            summary = await get_agent_allocation_service().run_yuki_signal_cycle_after_platform_generation(
                analysis_run_id=f"{triggered_by}_{datetime.now(timezone.utc).isoformat()}",
                signals_generated=0,
            )
            if summary.get("skipped"):
                logger.info(
                    "🤖 Yuki live signal refresh skipped (%s): %s",
                    triggered_by,
                    summary.get("reason"),
                )
            else:
                logger.info(
                    "🤖 Yuki live signal refresh complete (%s): allocations=%s attempted=%s executed=%s errors=%s",
                    triggered_by,
                    summary.get("allocations_checked", 0),
                    summary.get("trades_attempted", 0),
                    summary.get("trades_executed", 0),
                    len(summary.get("errors", [])),
                )
        except Exception as e:
            logger.error(f"❌ Yuki live signal refresh failed ({triggered_by}): {e}")

    async def _run_ryu_position_risk_cycle(self, triggered_by: str) -> None:
        """Mark and protect existing Ryu holdings without running discovery."""
        try:
            from kata.services.agent_allocation_service import get_agent_allocation_service

            summary = await get_agent_allocation_service().run_ryu_position_risk_cycle()
            if summary.get("skipped"):
                logger.info(
                    "🌸 Ryu risk refresh skipped (%s): %s",
                    triggered_by,
                    summary.get("reason"),
                )
            elif summary.get("success"):
                logger.info(
                    "🌸 Ryu risk refresh complete (%s): allocations=%s "
                    "positions=%s exits=%s partial=%s errors=%s",
                    triggered_by,
                    summary.get("allocations_checked", 0),
                    summary.get("positions_checked", 0),
                    summary.get("exits_executed", 0),
                    summary.get("partial_exits", 0),
                    summary.get("errors", 0),
                )
            else:
                logger.error(
                    "❌ Ryu risk refresh failed (%s): %s",
                    triggered_by,
                    summary.get("error"),
                )
        except Exception as e:
            logger.error(f"❌ Ryu risk refresh failed ({triggered_by}): {e}")

    async def _process_event_reanalysis_requests(self) -> None:
        """Regenerate queued stale/materially-changed entries between schedule slots."""
        close_terminal_requests = getattr(
            self.platform_service,
            "close_terminal_reanalysis_requests",
            None,
        )
        if callable(close_terminal_requests):
            await close_terminal_requests(limit=100)

        if self.event_reanalysis_in_progress:
            logger.debug("⏭️ Reanalysis already in progress — skipping this cycle")
            return
        if self.analysis_in_progress or self._analysis_lock.locked():
            logger.debug("⏭️ Full analysis in progress — deferring event reanalysis")
            return
        requests = await self.platform_service.get_pending_reanalysis_requests(limit=10)
        if not requests:
            return

        logger.info("🔁 Processing %d pending event re-analysis request(s)...", len(requests))
        self.event_reanalysis_in_progress = True
        try:
            processed = 0
            replacements = 0
            rearmed = 0
            completed_runs = []
            # Commit each request independently. A whole-batch timeout used to
            # cancel the final classification pass after several successful
            # re-tests, leaving every request (including SOXL) stuck pending.
            # Per-request isolation preserves completed work and lets a slow
            # symbol remain queued without blocking the others.
            for request in requests:
                signal_id = str(request.get("signal_id") or "unknown")
                token_symbol = str(request.get("token_symbol") or "unknown")
                try:
                    result = await asyncio.wait_for(
                        self.signal_generator.run_event_reanalysis_requests([request]),
                        timeout=300,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "❌ Event re-analysis timed out after 5 min for %s (%s); "
                        "request left pending for retry",
                        token_symbol,
                        signal_id,
                    )
                    continue
                except Exception as exc:
                    logger.error(
                        "❌ Event-driven signal re-analysis failed for %s (%s): %s",
                        token_symbol,
                        signal_id,
                        exc,
                    )
                    continue

                processed += int(result.get("processed") or 0)
                replacements += int(result.get("saved") or 0)
                rearmed += int(result.get("rearmed") or 0)
                if result.get("run_id"):
                    completed_runs.append(result["run_id"])
                actionable_signals = int(result.get("saved") or 0) + int(
                    result.get("rearmed") or 0
                )
                if actionable_signals:
                    await self._trigger_yuki_after_signal_generation(
                        result.get("run_id"),
                        actionable_signals,
                    )

            logger.info(
                "🔁 Event re-analysis complete: processed=%s replacements=%s "
                "rearmed=%s runs=%s",
                processed,
                replacements,
                rearmed,
                len(completed_runs),
            )
        except Exception as exc:
            logger.error("❌ Event-driven signal re-analysis failed: %s", exc)
        finally:
            self.event_reanalysis_in_progress = False

    async def _refresh_reject_learning_if_due(self):
        """Keep rejected-candidate learning fresh between full analysis runs."""
        interval_minutes = max(10, int(os.getenv("REJECT_LEARNING_REFRESH_MINUTES", "60") or "60"))
        now = datetime.now(timezone.utc)
        if (
            self.last_reject_learning_refresh_at is not None
            and (now - self.last_reject_learning_refresh_at) < timedelta(minutes=interval_minutes)
        ):
            return

        signal_generator = getattr(self, "signal_generator", None)
        if not signal_generator:
            return

        max_to_process = max(1, min(250, int(os.getenv("REJECT_LEARNING_MAX_PER_REFRESH", "80") or "80")))
        logger.info(
            "🧪 Refreshing reject counterfactual learning backlog (max=%d)...",
            max_to_process,
        )
        result = await signal_generator.refresh_reject_learning_backlog(max_to_process=max_to_process)
        self.last_reject_learning_refresh_at = now
        tuning_state = (result or {}).get("tuning_state") or {}
        logger.info(
            "🧪 Reject learning refresh complete: resolved=%d, adjustment=%s, sample_resolved=%s",
            int((result or {}).get("resolved", 0) or 0),
            tuning_state.get("adjustment"),
            tuning_state.get("sample_resolved"),
        )

    async def _run_single_monitoring(self):
        """Run a single performance monitoring cycle - lightweight and fast."""
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"📊 Starting monitoring run: {run_id}")
        
        try:
            # Monitor active signals for exits (lightweight - price data only)
            logger.info("📊 Monitoring active signals for exits...")
            performance_data = await self.performance_service.update_signal_performance_lightweight()
            
            if not performance_data or performance_data.get('monitored', 0) == 0:
                logger.info("📊 No active signals to monitor")
                return
            
            # Log monitoring results
            monitored = performance_data.get('monitored', 0)
            updated = performance_data.get('updated', 0)
            exits = performance_data.get('exits', 0)
            
            logger.info(f"📊 Monitoring run {run_id} complete: {monitored} signals monitored, {updated} updated, {exits} exits detected")
            
        except Exception as e:
            logger.error(f"❌ Monitoring run {run_id} failed: {e}")
            # Don't raise - monitoring failures shouldn't crash the worker
            logger.warning(f"⚠️ Monitoring will retry in 30 minutes")

    async def _run_single_analysis(self, schedule_context=None):
        """Run a single comprehensive analysis cycle with performance monitoring."""
        run_id = str(uuid.uuid4())[:8]
        schedule_context = schedule_context or {
            "trigger": "manual_or_internal",
            "actual_run_at_utc": datetime.now(timezone.utc).isoformat(),
            "slot_hour_utc": datetime.now(timezone.utc).hour,
        }
        await self._analysis_lock.acquire()
        self.analysis_in_progress = True

        try:
            logger.info(
                "🚀 Starting analysis run: %s | trigger=%s | slot=%s",
                run_id,
                schedule_context.get("trigger"),
                schedule_context.get("slot_hour_utc"),
            )
            analysis_run_id = None

            # 1. Run unified signal generation with AI analysis
            logger.info("🔍 Running unified market analysis with AI decision making...")
            try:
                analysis_run_id = await self.signal_generator.run_full_analysis_cycle(
                    max_signals=5  # Generate up to 5 high-quality signals
                )

                logger.info(f"📊 Unified Signal Generator completed analysis run: {analysis_run_id}")

                # Get generated signals from the analysis run
                generated_signals = await self.platform_service.get_signals_by_run(analysis_run_id)
                signals_generated = len(generated_signals)

                logger.info(f"✅ Retrieved {signals_generated} signals from analysis run {analysis_run_id}")

                # Learned context is already used by UnifiedSignalGenerator. This optional
                # experiment updates confidence/reasoning only, never final sizing again.
                if generated_signals and os.getenv("ENABLE_POST_SAVE_LEARNING_ENHANCEMENT", "false").lower() == "true":
                    logger.info("🧪 Applying post-save confidence learning enhancement (experimental override enabled)...")
                    try:
                        enhanced_signals = await self.learning_hook.enhance_signal_generation(generated_signals)

                        enhanced_count = 0
                        for enhanced_signal in enhanced_signals:
                            success = await self.platform_service.update_signal_with_learning_enhancements(enhanced_signal)
                            if success:
                                enhanced_count += 1

                        logger.info(f"✅ Applied post-save confidence enhancements to {enhanced_count}/{len(enhanced_signals)} signals")
                    except Exception as learning_error:
                        logger.warning(f"Post-save learning enhancement failed (signals still valid): {learning_error}")
                elif generated_signals:
                    logger.info("🧠 Skipping post-save confidence enhancement (learning context already used in generator)")

            except Exception as analysis_error:
                logger.error(f"❌ Unified signal generation failed: {analysis_error}")
                logger.error(f"Analysis error type: {type(analysis_error).__name__}")
                import traceback
                logger.error(f"Analysis traceback: {traceback.format_exc()}")

                # Continue with 0 signals instead of crashing
                signals_generated = 0
            
            # Log signal generation results
            if signals_generated == 0:
                logger.warning("⚠️ No new signals generated in this analysis cycle")
            else:
                logger.info(f"✅ Platform Analysis Engine generated {signals_generated} new high-quality signals")

            # Persist schedule tracking immediately so Learning Dashboard and Optimizer reflect run completion
            await self._persist_schedule_tracking(run_id, signals_generated, schedule_context)
            await self._refresh_schedule_optimization_if_due(triggered_by=schedule_context.get("trigger", "scheduled"))

            await self._trigger_yuki_after_signal_generation(analysis_run_id, signals_generated)
            await self._trigger_ryu_after_signal_generation(analysis_run_id)

            # 2. Retry logo fetching for signals without logos
            logger.info("🖼️ Retrying logo fetching for signals without logos...")
            await self._retry_logo_fetching()

            # 3. Monitor performance of ALL signals (existing + newly generated) with leverage
            logger.info("📊 Monitoring performance of all active signals with leverage...")
            await self._monitor_signal_performance(run_id)

            # 3b. Ensure a final lightweight performance update after generation (batch with leverage)
            try:
                logger.info("📊 Finalizing performance update after signal generation (batch sweep with leverage)...")
                await self.performance_service.update_signal_performance_lightweight_batch()
            except Exception as e:
                logger.warning(f"⚠️ Post-generation batch performance update failed: {e}")
            
            # 4. Clean up expired signals AFTER monitoring
            await self._cleanup_expired_signals()
            
            # 4. Show performance summary
            await self._log_performance_summary()
            
            logger.info(f"✅ Analysis run {run_id} complete: {signals_generated} platform signals generated")
            return signals_generated

        except Exception as e:
            logger.error(f"❌ Analysis run {run_id} failed: {e}")
            raise
        finally:
            self.analysis_in_progress = False
            if self._analysis_lock.locked():
                self._analysis_lock.release()
            # Candle/orderbook caches are only useful within a run; holding them
            # for the ~6h until the next cycle wastes RAM on a 512MB instance.
            self._release_analysis_memory()
            self._log_memory(f"post-analysis {run_id}")

    async def _trigger_yuki_after_signal_generation(self, analysis_run_id: str = None, signals_generated: int = 0):
        """Notify Yuki to evaluate platform signals once after generation finishes."""
        try:
            from kata.services.agent_allocation_service import get_agent_allocation_service

            allocation_service = get_agent_allocation_service()
            summary = await allocation_service.run_yuki_signal_cycle_after_platform_generation(
                analysis_run_id=analysis_run_id,
                signals_generated=signals_generated
            )
            if summary.get("skipped"):
                logger.info(f"🤖 Yuki post-generation cycle skipped: {summary.get('reason')}")
            else:
                logger.info(
                    "🤖 Yuki post-generation cycle complete: allocations=%s attempted=%s executed=%s errors=%s",
                    summary.get("allocations_checked", 0),
                    summary.get("trades_attempted", 0),
                    summary.get("trades_executed", 0),
                    len(summary.get("errors", [])),
                )
        except Exception as e:
            logger.error(f"❌ Yuki post-generation trigger failed: {e}")

    async def _trigger_ryu_after_signal_generation(self, analysis_run_id: str = None):
        """Notify Ryu to discover, analyze, and (if enabled) execute spot trades.

        Unlike Yuki, Ryu does not consume platform_signals -- it runs its own
        discovery + LLM-authoritative analysis (the same pipeline as the
        token-analysis card). This still runs once per worker cycle so
        discovery/LLM cost is shared across every funded user, not per-user.
        """
        try:
            from kata.services.agent_allocation_service import get_agent_allocation_service

            allocation_service = get_agent_allocation_service()
            summary = await allocation_service.run_ryu_signal_cycle_after_platform_generation(
                analysis_run_id=analysis_run_id,
            )
            if summary.get("skipped"):
                logger.info(f"🌸 Ryu post-generation cycle skipped: {summary.get('reason')}")
            else:
                logger.info(
                    "🌸 Ryu post-generation cycle complete: allocations=%s candidates=%s buys=%s exits=%s errors=%s",
                    summary.get("allocations_checked", 0),
                    summary.get("candidates_analyzed", 0),
                    summary.get("buys_executed", 0),
                    summary.get("exits_executed", 0),
                    summary.get("errors", 0),
                )
        except Exception as e:
            logger.error(f"❌ Ryu post-generation trigger failed: {e}")

    async def _persist_schedule_tracking(self, run_id: str, signals_generated: int, schedule_context: dict):
        """Persist a lightweight audit row for schedule/run attribution."""
        try:
            if not self.schedule_optimizer:
                return

            self.schedule_optimizer.persist_schedule_run(
                run_id=run_id,
                signals_generated=signals_generated,
                schedule_context=schedule_context,
                configured_hours_utc=self.generation_hours_utc,
            )
            logger.info(
                "🕒 Schedule tracking persisted: run=%s slot=%s generated=%s",
                run_id,
                schedule_context.get("slot_hour_utc"),
                signals_generated,
            )
        except Exception as e:
            logger.warning(f"⚠️ Could not persist schedule tracking snapshot: {e}")

    async def _refresh_schedule_optimization_if_due(self, triggered_by: str = "scheduled"):
        """Persist schedule optimizer output on a cooldown for ongoing timing learning."""
        try:
            if not self.schedule_optimizer:
                return

            interval_hours = max(1, int(os.getenv("SIGNAL_SCHEDULE_OPTIMIZATION_HOURS", "24") or "24"))
            now = datetime.now(timezone.utc)
            if (
                self.last_schedule_optimization_at is not None
                and (now - self.last_schedule_optimization_at) < timedelta(hours=interval_hours)
            ):
                return

            lookback_days = max(14, int(os.getenv("SIGNAL_SCHEDULE_LOOKBACK_DAYS", "365") or "365"))
            min_hour_samples = max(3, int(os.getenv("SIGNAL_SCHEDULE_MIN_HOUR_SAMPLES", "8") or "8"))
            min_total_samples = max(20, int(os.getenv("SIGNAL_SCHEDULE_MIN_TOTAL_SAMPLES", "120") or "120"))
            result = self.schedule_optimizer.persist_optimization_snapshot(
                configured_hours_utc=self.generation_hours_utc,
                lookback_days=lookback_days,
                min_hour_samples=min_hour_samples,
                min_total_samples=min_total_samples,
                triggered_by=triggered_by,
                auto_apply_enabled=self._schedule_auto_optimize_enabled(),
            )
            self.last_schedule_optimization_at = now

            recommended = result.get("recommended_schedule") or {}
            previous_hours = list(self.generation_hours_utc)
            should_apply, decision_reason = self._schedule_auto_apply_decision(result, min_total_samples)
            if should_apply:
                applied_schedule = await asyncio.to_thread(
                    self.schedule_optimizer.apply_active_schedule,
                    recommended.get("hours_utc", []),
                    source="schedule_optimizer_auto_apply",
                    applied_by=self.worker_instance_id,
                )
                self._set_generation_hours(applied_schedule["hours_utc"])
                self._schedule_changed_event.set()
                logger.info(
                    "🕒 Signal generation schedule auto-updated: %s -> %s",
                    self._schedule_label_for_hours(previous_hours),
                    self._schedule_label(),
                )

            try:
                self.schedule_optimizer.persist_auto_apply_decision(
                    optimization_result=result,
                    previous_hours_utc=previous_hours,
                    active_hours_utc=self.generation_hours_utc,
                    applied=should_apply,
                    reason=decision_reason,
                )
            except Exception as decision_err:
                logger.debug("Could not persist schedule auto-apply decision: %s", decision_err)

            logger.info(
                "🧠 Schedule optimization snapshot: recommended=%s confidence=%s samples=%s auto_apply=%s reason=%s",
                recommended.get("label"),
                result.get("confidence_score"),
                result.get("informative_final_outcomes"),
                should_apply,
                decision_reason,
            )
        except Exception as e:
            logger.warning(f"⚠️ Schedule optimization refresh failed: {e}")

    async def _retry_logo_fetching(self):
        """Retry logo fetching for signals that don't have logos."""
        try:
            # Get signals without logos (created in the last 24 hours to avoid retrying old signals)
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Get active signals without logos
            signals_without_logos = await self.platform_service.get_signals_without_logos(limit=10)
            
            if not signals_without_logos:
                logger.debug("🖼️ No signals found without logos")
                return
            
            logger.info(f"🖼️ Found {len(signals_without_logos)} signals without logos, retrying...")
            
            retry_count = 0
            for signal in signals_without_logos:
                try:
                    # Prepare symbol for logo fetching - handle symbols that already end with USDT
                    base_symbol = signal.token_symbol
                    if base_symbol.endswith('USDT'):
                        # Remove USDT suffix to get base token (e.g., TAOUSDT -> TAO)
                        base_symbol = base_symbol[:-4]
                    symbol_with_suffix = f"{base_symbol}/USDT"
                    
                    # Fetch logo with retry logic
                    logo_url = await self.signal_generator.fetch_token_logo_url(symbol_with_suffix)
                    
                    if logo_url:
                        # Update the signal with the new logo URL
                        await self.platform_service.update_signal_logo(signal.signal_id, logo_url)
                        retry_count += 1
                        logger.debug(f"✅ Updated logo for {signal.token_symbol}: {logo_url}")
                    else:
                        logger.warning(f"⚠️ Still no logo found for {signal.token_symbol}")
                    
                    # Add delay between retries to avoid rate limiting
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error retrying logo for {signal.token_symbol}: {e}")
                    continue
            
            if retry_count > 0:
                logger.info(f"✅ Successfully updated {retry_count} signal logos")
            else:
                logger.info("ℹ️ No logos were successfully fetched in this retry cycle")
                
        except Exception as e:
            logger.warning(f"⚠️ Error in logo retry process: {e}")

    async def _cleanup_expired_signals(self):
        """Remove expired signals from database."""
        try:
            expired_count = await self.platform_service.expire_old_signals()
            if expired_count > 0:
                logger.info(f"🧹 Cleaned up {expired_count} expired signals")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up expired signals: {e}")



    async def _monitor_signal_performance(self, run_id: str):
        """Monitor active signals and process any exits."""
        try:
            logger.info("📊 Monitoring active signals for exits...")
            websocket_stats = (
                self.simple_websocket_monitor.get_status()
                if self.simple_websocket_monitor
                else {}
            )
            websocket_running = bool(websocket_stats.get("is_running", False))
            websocket_monitored = int(websocket_stats.get("signals_monitored", 0) or 0)

            if websocket_running:
                logger.info(
                    f"📡 Monitoring mode=websocket_realtime | monitored={websocket_monitored} | batch_fallback=disabled"
                )
                return

            logger.warning(
                "⚠️ WebSocket monitor is not running; executing lightweight batch fallback monitoring"
            )
            performance_data = await self.performance_service.update_signal_performance_lightweight()
            monitored = int((performance_data or {}).get('monitored', 0) or 0)
            updated = int((performance_data or {}).get('updated', 0) or 0)
            exits = int((performance_data or {}).get('exits', 0) or 0)

            if monitored == 0:
                logger.info("📊 Batch fallback found no active signals to monitor")
                return

            logger.info(
                f"📊 Batch fallback monitoring: {monitored} signals monitored, {updated} updated, {exits} exits detected"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error monitoring signal performance: {e}")

    async def _log_performance_summary(self):
        """Log overall platform performance summary."""
        try:
            # Get metrics from platform signal service to avoid heavy performance service
            active_signals = await self.platform_service.get_active_signals(limit=100)
            total_signals = len(active_signals)
            
            # Simple metrics calculation without heavy performance service
            logger.info(f"📈 Platform Performance Summary:")
            logger.info(f"  • Active Signals: {total_signals}")
            logger.info(f"  • Platform Analysis: Fixed UTC schedule ({self._schedule_label()})")
            logger.info("  • Schedule Optimization: Tracking enabled, auto-apply disabled")
            websocket_stats = (
                self.simple_websocket_monitor.get_status()
                if self.simple_websocket_monitor
                else {}
            )
            logger.info(
                f"  • Performance Monitoring: mode={'websocket_realtime' if websocket_stats.get('is_running') else 'batch_fallback'} "
                f"(ws_signals={int(websocket_stats.get('signals_monitored', 0) or 0)})"
            )
            logger.info("  • Batch Fallback: Enabled only when WebSocket is unhealthy")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate performance summary: {e}")

    async def stop(self):
        """Stop the background worker."""
        try:
            logger.info("🛑 Stopping Platform Signals Worker...")
            
            self.shutdown_requested = True
            if getattr(self, "signal_generator", None):
                self.signal_generator.request_shutdown()
            
            # Stop analysis task
            if self.analysis_task and not self.analysis_task.done():
                self.analysis_task.cancel()
                try:
                    await self.analysis_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Background analysis stopped")

            # Stop accepting/processing durable manual requests before other
            # services are torn down. An interrupted request is released to
            # pending by the consumer for the replacement worker to retry.
            if self.generation_request_task and not self.generation_request_task.done():
                self.generation_request_task.cancel()
                try:
                    await self.generation_request_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Worker control request consumer stopped")
            
            # Stop simple WebSocket monitoring
            if self.simple_websocket_monitor:
                self.simple_websocket_monitor.stop_monitoring()
                logger.info("✅ Simple WebSocket monitoring stopped")

            # Stop websocket monitor task
            if self.websocket_task and not self.websocket_task.done():
                self.websocket_task.cancel()
                try:
                    await self.websocket_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ WebSocket monitor task stopped")

            # Stop signal refresh task
            if self.performance_task and not self.performance_task.done():
                self.performance_task.cancel()
                try:
                    await self.performance_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Signal refresh task stopped")

            logger.info("✅ Worker stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping worker: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        if getattr(self, "signal_generator", None):
            self.signal_generator.request_shutdown()


async def main():
    """Main worker function with automatic restart on failure."""
    max_restarts = int(os.getenv("WORKER_MAX_RESTARTS", "10"))
    restart_delay = int(os.getenv("WORKER_RESTART_DELAY_SECONDS", "60"))
    attempt = 0

    while attempt <= max_restarts:
        worker = PlatformSignalsWorker()

        # Set up signal handlers
        signal.signal(signal.SIGINT, worker.signal_handler)
        signal.signal(signal.SIGTERM, worker.signal_handler)

        try:
            await worker.start()
            # Clean exit (shutdown_requested)
            logger.info("Worker exited cleanly.")
            break
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt — shutting down.")
            break
        except Exception as e:
            attempt += 1
            logger.error(
                "❌ Worker crashed (attempt %d/%d): %s",
                attempt, max_restarts, e,
            )
            import traceback
            logger.error("Traceback:\n%s", traceback.format_exc())
            if worker.shutdown_requested:
                logger.info("Shutdown was requested — not restarting.")
                break
            if attempt > max_restarts:
                logger.error("Max restart attempts reached. Exiting.")
                return False
            backoff = min(restart_delay * attempt, 600)
            logger.info("Restarting worker in %ds...", backoff)
            await asyncio.sleep(backoff)
        finally:
            try:
                await worker.stop()
            except Exception:
                pass

    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FLOW AI TRADING PLATFORM - SIGNALS WORKER")
    logger.info("=" * 60)
    
    # Run the worker
    success = asyncio.run(main())
    
    if success:
        logger.info("Worker exited successfully")
        sys.exit(0)
    else:
        logger.error("Worker exited with errors")
        sys.exit(1)
