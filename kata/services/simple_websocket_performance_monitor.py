#!/usr/bin/env python3
"""
WebSocket Performance Monitor
- One purpose: real-time signal exit detection and live PnL tracking
- Connects to Binance USDM futures WebSocket (spot fallback)
- Exponential backoff on reconnect, symbol-change-triggered reconnect,
  silence detection, and no double-leverage in RPC calls
"""

import asyncio
import json
import logging
import re
import websockets
import aiohttp
from typing import Dict, Optional, Set, Any
from datetime import datetime, timedelta, timezone

from kata.config.settings import settings
from kata.services.platform_signal_service import (
    get_platform_signal_service,
    parse_timestamp,
    persist_entry_activation_if_current,
    persist_market_conditions_if_current,
)
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)

# Binance endpoints:
# - Futures "market" routed path is required for market streams (@markPrice/@aggTrade/etc)
#   per Binance WebSocket migration notice (legacy unrouted /stream no longer receives market feeds).
# - Prefer a combined stream containing only active signal symbols. The all-market
#   stream remains a compatibility fallback, but parsing every futures contract
#   produced ~3.1M row allocations/hour while only four symbols were relevant.
_FUTURES_WS_TARGETED = "wss://fstream.binance.com/market/stream?streams={streams}"
_SPOT_WS_TARGETED    = "wss://stream.binance.com:9443/stream?streams={streams}"
_FUTURES_WS_ALL_MARK = "wss://fstream.binance.com/market/ws/!markPrice@arr@1s"
_SPOT_WS_ALL_MARK    = "wss://stream.binance.com:9443/ws/!miniTicker@arr"

_RECONNECT_MIN_S   = 5
_RECONNECT_MAX_S   = 120
_STABLE_SECS       = 300   # connection older than this → reset backoff
_SILENCE_TIMEOUT_S = 60    # no message for this long → reconnect
_SIGNAL_RELOAD_INTERVAL_S = 30  # refresh active-signal set every 30s while stream is live
_EMPTY_POLL_S      = 300   # 5 min between polls when no active signals
class SimpleWebSocketPerformanceMonitor:
    """Reliable real-time performance monitor with reconnect and silence detection."""

    def __init__(self):
        self.is_running = False
        self.signals: Dict = {}           # signal_id → signal object
        self._last_prices: Dict = {}      # symbol (lowercase) → latest price
        self._last_quote_volumes: Dict = {}
        self._reanalysis_event_state: Dict[str, Dict[str, Any]] = {}
        self.price_updates_count = 0
        self.last_status_log = datetime.now()
        self._empty_poll_count = 0
        self._reconnect_count = 0
        self._reconnect_delay = _RECONNECT_MIN_S
        self._last_connected_at: Optional[datetime] = None
        self._use_futures = True          # try futures first; flip to spot on failure
        self._use_targeted_stream = True  # all-market is a no-data compatibility fallback

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    async def start_monitoring(self):
        """Main loop: load signals → connect → process ticks → reconnect."""
        self.is_running = True
        platform_service = get_platform_signal_service()
        db = get_service_client()
        logger.info("🚀 Starting WebSocket performance monitoring")

        while self.is_running:
            try:
                signals_list = await platform_service.get_active_signals(limit=100)
                self.signals = {s.signal_id: s for s in signals_list}

                # Restore milestone state (TP1/buffer) so break-even and trailing
                # protections survive restarts/reconnects — platform_signals has no
                # target_1_hit column, so the reload alone always yields False.
                self._hydrate_milestone_flags(db)

                # Re-attach TP2-closed trades still inside their window so peak
                # tracking survives restarts/reconnects.
                try:
                    closed_list = await platform_service.get_closed_signals_in_window(limit=50)
                    for s in closed_list:
                        if s.signal_id not in self.signals:
                            setattr(s, 'closed_tracking', True)
                            self.signals[s.signal_id] = s
                except Exception as e:
                    logger.warning(f"Failed to load TP2-closed signals for peak tracking: {e}")

                if not self.signals:
                    self._empty_poll_count += 1
                    if self._empty_poll_count == 1 or self._empty_poll_count % 12 == 0:
                        logger.info(
                            f"📊 No active signals — polling every {_EMPTY_POLL_S}s "
                            f"(poll #{self._empty_poll_count})"
                        )
                    await asyncio.sleep(_EMPTY_POLL_S)
                    continue
                self._empty_poll_count = 0

                current_symbols = self._build_symbol_set(self.signals.values())
                url = self._build_ws_url(current_symbols)
                feed_scope = 'targeted' if self._use_targeted_stream else 'all-market fallback'
                logger.info(
                    f"📡 Connecting to {'futures' if self._use_futures else 'spot'} "
                    f"WebSocket for {len(current_symbols)} symbols ({feed_scope})"
                )

                ticks_before_connect = self.price_updates_count
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as websocket:
                    self._on_connected()
                    expiration_task = asyncio.create_task(
                        self._periodic_expiration_check(platform_service)
                    )
                    reconnect = await self._process_stream(
                        websocket, current_symbols, platform_service, db
                    )
                    expiration_task.cancel()
                    try:
                        await expiration_task
                    except asyncio.CancelledError:
                        pass

                if self._use_targeted_stream and self.price_updates_count == ticks_before_connect:
                    self._use_targeted_stream = False
                    logger.warning(
                        "⚠️ Targeted Binance stream closed without a price update; "
                        "falling back to the all-market feed"
                    )
                    continue

                if reconnect:
                    # Symbol set changed — reconnect immediately with no penalty
                    logger.info("🔄 Symbol set changed — reconnecting WebSocket")
                    continue

                # Clean disconnect — reset backoff
                self._reconnect_delay = _RECONNECT_MIN_S

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"🔌 WebSocket closed ({e.code}), retry in {self._reconnect_delay}s")
                await self._backoff_sleep()
            except OSError as e:
                # Futures endpoint not reachable → switch to spot
                if self._use_futures:
                    logger.warning(f"⚠️ Futures WS failed ({e}), switching to spot endpoint")
                    self._use_futures = False
                else:
                    logger.error(f"❌ WebSocket OS error: {e}, retry in {self._reconnect_delay}s")
                    await self._backoff_sleep()
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}, retry in {self._reconnect_delay}s")
                await self._backoff_sleep()

    def stop_monitoring(self):
        self.is_running = False
        logger.info("🛑 WebSocket monitoring stopped")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "signals_monitored": len(self.signals),
            "symbols": [s.token_symbol for s in self.signals.values()],
            "price_updates": self.price_updates_count,
            "reconnect_count": self._reconnect_count,
            "last_connected_at": self._last_connected_at.isoformat() if self._last_connected_at else None,
            "using_futures_ws": self._use_futures,
            "feed_scope": "targeted" if self._use_targeted_stream else "all_market_fallback",
        }

    # ------------------------------------------------------------------ #
    # Inner stream processor
    # ------------------------------------------------------------------ #

    async def _process_stream(self, websocket, current_symbols: Set[str],
                              platform_service, db) -> bool:
        """
        Read messages until disconnect.
        Uses an all-market stream and filters active symbols locally.
        """
        last_message_at = asyncio.get_event_loop().time()
        last_reload_at = asyncio.get_event_loop().time()
        _error_count = 0

        async def _check_silence():
            """Background task: reconnect on stream silence."""
            while True:
                await asyncio.sleep(10)
                if asyncio.get_event_loop().time() - last_message_at > _SILENCE_TIMEOUT_S:
                    logger.warning(f"🔇 No messages for {_SILENCE_TIMEOUT_S}s — reconnecting")
                    await websocket.close()
                    return

        silence_task = asyncio.create_task(_check_silence())

        try:
            async for message in websocket:
                if not self.is_running:
                    break

                last_message_at = asyncio.get_event_loop().time()

                try:
                    data = json.loads(message)
                    payload = data.get('data', data) if isinstance(data, dict) else data
                    rows = payload if isinstance(payload, list) else [payload]

                    for inner in rows:
                        if not isinstance(inner, dict):
                            continue
                        symbol, price, quote_volume = self._parse_stream_tick(inner)
                        if not symbol or symbol not in current_symbols:
                            continue

                        if quote_volume is not None:
                            self._last_quote_volumes[symbol] = quote_volume

                        # Futures mini-ticker and mark-price events can disagree
                        # around an exact target. Use mini-ticker only for volume;
                        # all lifecycle/PnL decisions use the canonical mark price.
                        if price is None:
                            continue

                        self._last_prices[symbol] = price
                        self.price_updates_count += 1
                        await self._check_signals_for_exits(
                            symbol,
                            price,
                            platform_service,
                            db,
                            quote_volume=(
                                quote_volume
                                if quote_volume is not None
                                else self._last_quote_volumes.get(symbol)
                            ),
                        )

                    # Periodic: reload signal list and check for new symbols.
                    # Time-based reload is robust across sparse/dense stream types.
                    now_loop = asyncio.get_event_loop().time()
                    if (now_loop - last_reload_at) >= _SIGNAL_RELOAD_INTERVAL_S:
                        updated = await platform_service.get_active_signals(limit=100)
                        old_count = len(self.signals)
                        new_signals = {s.signal_id: s for s in updated}

                        # Carry in-memory state onto the freshly loaded objects so
                        # a reload does not disarm break-even/trailing protection
                        # or reset the peak cache mid-stream.
                        for sid, fresh in new_signals.items():
                            prior = self.signals.get(sid)
                            if prior is None:
                                continue
                            for flag in ('target_1_hit', 'buffer_hit', '_peak_cache'):
                                if getattr(prior, flag, None) and not getattr(fresh, flag, None):
                                    setattr(fresh, flag, getattr(prior, flag))

                        # Keep TP2-closed peak-tracking signals attached until their
                        # window elapses. The outer connect loop adds them, but this
                        # reload used to rebuild self.signals from active signals
                        # only — shrinking the symbol universe, forcing a reconnect,
                        # after which the outer loop re-added them and grew it again.
                        # That 5→3→5 oscillation reconnected the stream roughly
                        # every reload interval instead of holding one connection.
                        for sid, sig in self.signals.items():
                            if (
                                sid not in new_signals
                                and getattr(sig, 'closed_tracking', False)
                                and not self._is_signal_expired(sig)
                            ):
                                new_signals[sid] = sig

                        new_symbols = self._build_symbol_set(new_signals.values())
                        self.signals = new_signals
                        last_reload_at = now_loop

                        # Do not retain prices for symbols that are no longer
                        # tracked (especially after using the all-market fallback).
                        for stale_symbol in set(self._last_prices) - new_symbols:
                            self._last_prices.pop(stale_symbol, None)
                            self._last_quote_volumes.pop(stale_symbol, None)

                        if new_symbols != current_symbols:
                            logger.info(
                                f"🔄 Active symbol universe changed ({len(current_symbols)}→{len(new_symbols)})"
                            )
                            if self._use_targeted_stream:
                                return True
                            current_symbols = new_symbols

                        if len(self.signals) != old_count:
                            logger.info(f"🔄 Signals reloaded: {old_count} → {len(self.signals)}")

                    # Periodic status log
                    now = datetime.now()
                    if (now - self.last_status_log).seconds >= 1800:
                        logger.info(
                            f"📡 WS active: {self.price_updates_count} ticks, "
                            f"{len(self.signals)} signals, reconnects={self._reconnect_count}"
                        )
                        self.last_status_log = now
                        _error_count = 0  # reset periodic error count

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    _error_count += 1
                    if _error_count % 100 == 1:
                        logger.warning(f"Tick processing errors (×{_error_count}): {e}")

        finally:
            silence_task.cancel()
            try:
                await silence_task
            except asyncio.CancelledError:
                pass

        return False  # normal disconnect, no symbol change

    # ------------------------------------------------------------------ #
    # Signal exit handling
    # ------------------------------------------------------------------ #

    async def _check_signals_for_exits(
        self,
        symbol,
        price,
        platform_service,
        db,
        quote_volume: Optional[float] = None,
    ):
        for signal_id, signal in list(self.signals.items()):
            sig_sym = self._normalize_to_futures_symbol(signal.token_symbol).lower()
            if sig_sym != symbol:
                continue

            # Closed trades (TP2 already hit): the result is locked; only track
            # the peak until the trade window elapses, then drop the subscription.
            if getattr(signal, 'closed_tracking', False):
                if self._is_signal_expired(signal):
                    logger.info(f"⏱️ Peak-tracking window ended: {signal.token_symbol} {signal.direction}")
                    del self.signals[signal_id]
                else:
                    self._update_peak_only(signal_id, signal, price, db)
                continue

            if self._entry_reanalysis_pending(signal):
                # The source thesis is still live, but its old entry window is
                # deliberately paused until a fresh decision re-arms it.
                continue

            if self._entry_window_expired(signal):
                await platform_service.request_signal_reanalysis(
                    signal_id,
                    "platform entry TTL expired before activation",
                    observed_price=price,
                    observed_volume=quote_volume,
                    expire_entry_order=True,
                    source="platform_price_monitor",
                )
                del self.signals[signal_id]
                continue

            # Gate exits/PnL until entry is actually activated for non-market entry setups.
            if not await self._ensure_entry_activated(signal_id, signal, price, db):
                await self._maybe_request_entry_event_reanalysis(
                    signal_id,
                    signal,
                    price,
                    quote_volume,
                    platform_service,
                )
                continue

            self._update_excursion_metadata(signal_id, signal, price, db)
            exit_reason = self._should_exit(signal, price)
            if not exit_reason and self._is_signal_expired(signal):
                path_reason, path_price = await self._expiry_exit_from_venue_path(signal)
                exit_reason = path_reason or self._expiry_exit_reason(signal, price)
                if path_price is not None:
                    price = path_price
            if exit_reason:
                exit_type = exit_reason.split('(')[0].strip()
                logger.info(f"🎯 EXIT: {signal.token_symbol} {signal.direction} — {exit_reason} @ ${price}")
                if exit_type in {'TARGET_1', 'TARGET_2'} and not getattr(signal, 'target_1_hit', False):
                    recorded = await platform_service.record_signal_target_1_milestone(
                        signal_id,
                        signal.target_1,
                    )
                    if recorded:
                        setattr(signal, 'target_1_hit', True)
                await self._record_exit_pnl(signal_id, signal, price, db)
                await platform_service.update_signal_exit(signal_id, price, exit_type)
                if exit_type == 'TARGET_2':
                    # Trade closed at TP2, but keep the price subscription until
                    # the window elapses so Final PnL captures the peak past TP2.
                    setattr(signal, 'closed_tracking', True)
                    logger.info(
                        f"📈 {signal.token_symbol} closed at TP2 — peak tracking continues until expiry"
                    )
                else:
                    del self.signals[signal_id]
            else:
                if self._target_1_reached(signal, price) and not getattr(signal, 'target_1_hit', False):
                    recorded = await platform_service.record_signal_target_1_milestone(
                        signal_id,
                        signal.target_1,
                    )
                    if recorded:
                        setattr(signal, 'target_1_hit', True)
                        logger.info(
                            f"🎯 TP1 milestone: {signal.token_symbol} {signal.direction} "
                            f"({self._calculate_pnl(signal, signal.target_1):+.1f}%) @ ${signal.target_1}"
                        )

                # Check for buffer hit (halfway to TP2)
                if getattr(signal, 'target_1_hit', False) and not getattr(signal, 'buffer_hit', False):
                    t1 = signal.target_1
                    t2 = signal.target_2 if signal.target_2 else t1 + (t1 - signal.entry_price)
                    buffer_price = t1 + ((t2 - t1) / 2)
                    d = signal.direction.upper()
                    
                    if (d == 'LONG' and price >= buffer_price) or (d == 'SHORT' and price <= buffer_price):
                        saved, latest_mc, update_state = persist_market_conditions_if_current(
                            db,
                            signal_id,
                            lambda current: {**current, 'buffer_hit': True},
                        )
                        if latest_mc:
                            signal.market_conditions = latest_mc
                        if saved:
                            setattr(signal, 'buffer_hit', True)
                            if update_state == "updated":
                                logger.info(f"🛡️ Buffer reached for {signal.token_symbol} {signal.direction}, trailing stop to TP1")

                # Live PnL update — pass RAW (unleveraged) pnl; RPC applies leverage from DB
                raw_pnl = self._calculate_raw_pnl(signal, price)
                try:
                    db.rpc('rpc_update_signal_performance', {
                        'p_signal_id': str(signal_id),
                        'p_pnl': float(raw_pnl),
                    }).execute()
                except Exception as e:
                    if not hasattr(self, '_rpc_err'):
                        self._rpc_err = 0
                    self._rpc_err += 1
                    if self._rpc_err % 50 == 1:
                        logger.warning(f"RPC errors (×{self._rpc_err}) for {signal.token_symbol}: {e}")

    def _entry_window_expired(self, signal) -> bool:
        """Return True only for an unactivated entry whose short entry TTL elapsed."""
        mc = (signal.market_conditions or {}) if hasattr(signal, "market_conditions") else {}
        if not bool(mc.get("entry_activation_required", False)):
            return False
        if bool(mc.get("entry_activated", False)):
            return False
        raw = mc.get("entry_order_expires_at")
        if not raw:
            return False
        try:
            deadline = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= deadline
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _entry_reanalysis_pending(signal) -> bool:
        """Keep a stale entry inert while preserving its nonterminal thesis."""
        mc = (signal.market_conditions or {}) if hasattr(signal, "market_conditions") else {}
        request_pending = (
            bool(mc.get("reanalysis_requested"))
            and str(mc.get("reanalysis_request_status") or "pending").lower() == "pending"
        )
        state = str(mc.get("entry_order_state") or "").lower()
        return request_pending or state in {
            "expired_pending_reanalysis",
            "awaiting_confirmation",
            "thesis_expired",
        }

    def _update_excursion_metadata(self, signal_id: str, signal, price: float, db) -> None:
        """Persist ATR-normalized MFE/MAE only when a material new extreme occurs."""
        try:
            entry = float(getattr(signal, "entry_price", 0.0) or 0.0)
            if entry <= 0 or price <= 0:
                return
            direction = str(getattr(signal, "direction", "") or "").upper()
            cached_mc = dict(
                (signal.market_conditions or {})
                if hasattr(signal, "market_conditions")
                else {}
            )
            cached_atr = float(
                cached_mc.get("atr_14")
                or (getattr(signal, "technical_indicators", {}) or {}).get("atr_14")
                or 0.0
            )
            if cached_atr <= 0:
                return
            cached_signed_move = (
                price - entry if direction == "LONG" else entry - price
            )
            cached_mfe = max(0.0, cached_signed_move / cached_atr)
            cached_mae = max(0.0, -cached_signed_move / cached_atr)
            if (
                cached_mfe < float(cached_mc.get("mfe_atr") or 0.0) + 0.10
                and cached_mae < float(cached_mc.get("mae_atr") or 0.0) + 0.10
            ):
                return

            def update_excursions(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                atr = float(
                    current.get("atr_14")
                    or (getattr(signal, "technical_indicators", {}) or {}).get("atr_14")
                    or 0.0
                )
                if atr <= 0:
                    return None
                signed_move = price - entry if direction == "LONG" else entry - price
                mfe_atr = max(0.0, signed_move / atr)
                mae_atr = max(0.0, -signed_move / atr)
                improves_mfe = mfe_atr >= float(current.get("mfe_atr") or 0.0) + 0.10
                improves_mae = mae_atr >= float(current.get("mae_atr") or 0.0) + 0.10
                if not improves_mfe and not improves_mae:
                    return None

                now_iso = datetime.now(timezone.utc).isoformat()
                if improves_mfe:
                    current.update({
                        "mfe_atr": round(mfe_atr, 4),
                        "mfe_price": float(price),
                        "mfe_timestamp": now_iso,
                    })
                if improves_mae:
                    current.update({
                        "mae_atr": round(mae_atr, 4),
                        "mae_price": float(price),
                        "mae_timestamp": now_iso,
                    })
                return current

            saved, latest_mc, _ = persist_market_conditions_if_current(
                db,
                signal_id,
                update_excursions,
            )
            if saved and latest_mc:
                signal.market_conditions = latest_mc
        except Exception as exc:
            logger.debug("Excursion metadata update failed for %s: %s", signal_id, exc)

    async def _maybe_request_entry_event_reanalysis(
        self,
        signal_id: str,
        signal,
        price: float,
        quote_volume: Optional[float],
        platform_service,
    ) -> bool:
        """Queue fresh analysis after a material pending-entry price/volume event."""
        mc = (signal.market_conditions or {}) if hasattr(signal, "market_conditions") else {}
        now = datetime.now(timezone.utc)
        state = self._reanalysis_event_state.setdefault(signal_id, {
            "price": float(mc.get("market_price_at_generation") or price),
            "volume": quote_volume,
            "requested_at": None,
        })
        reference_price = float(state.get("price") or price)
        price_move = abs(price - reference_price) / reference_price if reference_price > 0 else 0.0
        volume_reference = state.get("volume")
        volume_change = 0.0
        if quote_volume is not None and volume_reference not in (None, 0):
            volume_change = max(0.0, (quote_volume - float(volume_reference)) / abs(float(volume_reference)))

        price_threshold = max(
            0.001,
            float(getattr(settings, "YUKI_ENTRY_REVALIDATION_PRICE_MOVE_PCT", 0.025) or 0.025),
        )
        volume_threshold = max(
            0.01,
            float(getattr(settings, "YUKI_ENTRY_REVALIDATION_VOLUME_CHANGE_PCT", 0.10) or 0.10),
        )
        if price_move < price_threshold and volume_change < volume_threshold:
            return False

        requested_at = state.get("requested_at")
        cooldown_minutes = max(
            1,
            int(getattr(settings, "YUKI_ENTRY_REVALIDATION_COOLDOWN_MINUTES", 30) or 30),
        )
        if isinstance(requested_at, datetime) and now - requested_at < timedelta(minutes=cooldown_minutes):
            return False

        reasons = []
        if price_move >= price_threshold:
            reasons.append(f"pending-entry price moved {price_move:.2%}")
        if volume_change >= volume_threshold:
            reasons.append(f"rolling quote volume increased {volume_change:.1%}")
        queued = await platform_service.request_signal_reanalysis(
            signal_id,
            "; ".join(reasons),
            observed_price=price,
            observed_volume=quote_volume,
            invalidate_pending_entry=False,
            source="platform_market_event",
        )
        if queued:
            state.update({
                "price": price,
                "volume": quote_volume,
                "requested_at": now,
            })
        return queued

    def _expiry_exit_reason(self, signal, price: float) -> str:
        """Resolve the highest target milestone before considering stop/expiry."""
        direction = str(getattr(signal, "direction", "") or "").upper()
        target_1 = float(getattr(signal, "target_1", 0.0) or 0.0)
        target_2 = float(getattr(signal, "target_2", 0.0) or 0.0)
        stop = float(getattr(signal, "stop_loss", 0.0) or 0.0)
        target_1_hit = bool(getattr(signal, "target_1_hit", False))
        if direction == "LONG":
            if target_2 and price >= target_2:
                return "TARGET_2"
            if target_1_hit or (target_1 and price >= target_1):
                return "TARGET_1"
            if stop and price <= stop:
                return "STOP_LOSS"
        elif direction == "SHORT":
            if target_2 and price <= target_2:
                return "TARGET_2"
            if target_1_hit or (target_1 and price <= target_1):
                return "TARGET_1"
            if stop and price >= stop:
                return "STOP_LOSS"
        return "EXPIRED"

    @staticmethod
    def _classify_terminal_candles(signal, candles) -> tuple:
        """Classify a venue path chronologically, with targets winning same-bar races."""
        direction = str(getattr(signal, "direction", "") or "").upper()
        target_1 = float(getattr(signal, "target_1", 0.0) or 0.0)
        target_2 = float(getattr(signal, "target_2", 0.0) or 0.0)
        stop = float(getattr(signal, "stop_loss", 0.0) or 0.0)
        for candle in sorted(candles or [], key=lambda row: int(row.get("t") or 0)):
            try:
                high = float(candle.get("h") or 0.0)
                low = float(candle.get("l") or 0.0)
            except (TypeError, ValueError):
                continue
            if direction == "LONG":
                if target_2 and high >= target_2:
                    return "TARGET_2", target_2
                if target_1 and high >= target_1:
                    return "TARGET_1", target_1
                if stop and low <= stop:
                    return "STOP_LOSS", stop
            elif direction == "SHORT":
                if target_2 and low <= target_2:
                    return "TARGET_2", target_2
                if target_1 and low <= target_1:
                    return "TARGET_1", target_1
                if stop and high >= stop:
                    return "STOP_LOSS", stop
        return None, None

    async def _expiry_exit_from_venue_path(self, signal) -> tuple:
        """Replay Hyperliquid candles through thesis expiry to recover missed milestones."""
        token = str(getattr(signal, "token_symbol", "") or "").upper().strip()
        if token.endswith("USDT"):
            token = token[:-4]
        if not token or ":" in token:
            return None, None

        mc = (
            (signal.market_conditions or {})
            if hasattr(signal, "market_conditions")
            else {}
        )
        started_raw = (
            mc.get("entry_activated_at")
            or getattr(signal, "analysis_timestamp", None)
        )
        expires_raw = getattr(signal, "expires_at", None)
        if not started_raw or not expires_raw:
            return None, None
        try:
            started_at = (
                started_raw
                if isinstance(started_raw, datetime)
                else parse_timestamp(str(started_raw))
            )
            expires_at = (
                expires_raw
                if isinstance(expires_raw, datetime)
                else parse_timestamp(str(expires_raw))
            )
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None, None

        hours = max(0.0, (expires_at - started_at).total_seconds() / 3600.0)
        interval = "5m" if hours <= 24 * 7 else "1h"
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": token,
                "interval": interval,
                "startTime": int(started_at.timestamp() * 1000),
                "endTime": int(expires_at.timestamp() * 1000),
            },
        }
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.hyperliquid.xyz/info",
                    json=payload,
                ) as response:
                    if response.status != 200:
                        return None, None
                    candles = await response.json()
            return self._classify_terminal_candles(signal, candles)
        except Exception as exc:
            logger.debug(
                "Hyperliquid expiry replay unavailable for %s: %s",
                token,
                exc,
            )
            return None, None

    async def _record_exit_pnl(self, signal_id, signal, exit_price, db):
        """Record final PnL on all exit paths via RPC (pass raw; RPC applies leverage)."""
        try:
            raw_pnl = self._calculate_raw_pnl(signal, exit_price)
            db.rpc('rpc_update_signal_performance', {
                'p_signal_id': str(signal_id),
                'p_pnl': float(raw_pnl),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to record exit PnL for {signal.token_symbol}: {e}")

    def _entry_triggered(self, signal, price: float) -> bool:
        """Return True if a pending-entry signal has now been activated by market price."""
        mc: Dict[str, Any] = (signal.market_conditions or {}) if hasattr(signal, 'market_conditions') else {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True

        mode = str(mc.get('entry_activation_mode') or '').strip().lower()
        tol = float(mc.get('entry_activation_buffer_pct', 0.0015) or 0.0015)
        entry = float(getattr(signal, 'entry_price', 0.0) or 0.0)
        direction = str(getattr(signal, 'direction', '') or '').upper()
        if entry <= 0 or direction not in {'LONG', 'SHORT'}:
            return True

        if mode == 'limit_retest':
            return price <= entry * (1 + tol) if direction == 'LONG' else price >= entry * (1 - tol)
        if mode == 'breakout_stop':
            return price >= entry * (1 - tol) if direction == 'LONG' else price <= entry * (1 + tol)
        return True

    async def _ensure_entry_activated(self, signal_id: str, signal, price: float, db) -> bool:
        """
        Ensure entry activation state is respected and persisted.
        Returns True when signal is active for exit/PnL checks.
        """
        mc: Dict[str, Any] = (signal.market_conditions or {}) if hasattr(signal, 'market_conditions') else {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True
        if not self._entry_triggered(signal, price):
            return False

        try:
            activated, latest_mc, result = persist_entry_activation_if_current(
                db,
                signal_id,
                price,
            )
            if latest_mc:
                signal.market_conditions = latest_mc
            if result == "activated":
                logger.info(f"✅ Entry activated: {signal.token_symbol} {signal.direction} @ {price}")
            elif not activated:
                logger.info(
                    "Entry activation skipped for %s: %s",
                    signal_id,
                    result,
                )
            return activated
        except Exception as e:
            logger.warning(f"Entry activation persistence failed for {signal_id}: {e}")
            return False

    async def _periodic_expiration_check(self, platform_service):
        """Minute-level expiry sweep for signals that get no price ticks."""
        while self.is_running:
            try:
                await asyncio.sleep(60)
                if not self.is_running:
                    break

                db = get_service_client()
                stale_entries = [
                    (sid, sig) for sid, sig in list(self.signals.items())
                    if not self._entry_reanalysis_pending(sig)
                    and self._entry_window_expired(sig)
                ]
                for signal_id, signal in stale_entries:
                    symbol_key = self._normalize_to_futures_symbol(signal.token_symbol).lower()
                    last_price = self._last_prices.get(symbol_key)
                    await platform_service.request_signal_reanalysis(
                        signal_id,
                        "platform entry TTL expired before activation",
                        observed_price=last_price,
                        observed_volume=self._last_quote_volumes.get(symbol_key),
                        expire_entry_order=True,
                        source="platform_expiration_sweep",
                    )
                    self.signals.pop(signal_id, None)

                expired = [
                    (sid, sig) for sid, sig in list(self.signals.items())
                    if self._is_signal_expired(sig)
                ]
                for signal_id, signal in expired:
                    if getattr(signal, 'closed_tracking', False):
                        # Trade already closed at TP2 — window over, just stop tracking
                        del self.signals[signal_id]
                        continue
                    symbol_key = self._normalize_to_futures_symbol(signal.token_symbol).lower()
                    last_price = self._last_prices.get(symbol_key, 0)
                    if last_price <= 0:
                        # No cached WebSocket price — try REST so the exit PnL isn't 0%
                        last_price = await self._fetch_rest_price(symbol_key.upper())
                    if last_price > 0:
                        path_reason, path_price = await self._expiry_exit_from_venue_path(signal)
                        exit_reason = path_reason or self._expiry_exit_reason(signal, last_price)
                        if path_price is not None:
                            last_price = path_price
                        if exit_reason in {"TARGET_1", "TARGET_2"} and not getattr(signal, "target_1_hit", False):
                            recorded = await platform_service.record_signal_target_1_milestone(
                                signal_id,
                                signal.target_1,
                            )
                            if recorded:
                                setattr(signal, "target_1_hit", True)
                        await self._record_exit_pnl(signal_id, signal, last_price, db)
                    else:
                        exit_reason = "EXPIRED"
                        logger.warning(f"⚠️ No price available for {signal.token_symbol} at expiry — PnL will be preserved from last live tick")
                    await platform_service.update_signal_exit(signal_id, last_price, exit_reason)
                    del self.signals[signal_id]

                if expired:
                    logger.info(f"🧹 Expired {len(expired)} signal(s) via periodic sweep")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic expiry error: {e}")

    # ------------------------------------------------------------------ #
    # Price / exit logic
    # ------------------------------------------------------------------ #

    def _parse_stream_tick(self, inner: Dict[str, Any]) -> tuple:
        """
        Return (symbol, canonical_price, quote_volume) for a Binance event.

        Futures subscriptions intentionally include mini-ticker for rolling
        volume, but only mark-price events expose ``p`` and may drive signal
        milestones or exits. Spot fallback uses mini-ticker close ``c``.
        """
        symbol = str(inner.get('s') or '').strip().lower()

        quote_volume = None
        for volume_key in ("q", "Q", "v"):
            if inner.get(volume_key) in (None, ""):
                continue
            try:
                quote_volume = float(inner[volume_key])
            except (TypeError, ValueError):
                quote_volume = None
            break

        price_raw = inner.get('p')
        if not self._use_futures and price_raw in (None, ''):
            price_raw = inner.get('c')

        if price_raw in (None, ''):
            return symbol, None, quote_volume

        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            price = None
        if price is not None and price <= 0:
            price = None
        return symbol, price, quote_volume

    def _should_exit(self, signal, price):
        try:
            d = signal.direction.upper()
            target_1_hit = bool(getattr(signal, 'target_1_hit', False))
            mc = (signal.market_conditions or {}) if hasattr(signal, 'market_conditions') else {}
            buffer_hit = bool(getattr(signal, 'buffer_hit', False) or mc.get('buffer_hit', False))

            current_pnl = self._calculate_pnl(signal, price)
            peak_pnl = max(float(getattr(signal, 'max_profit_reached', 0.0) or 0.0), float(mc.get('mfe_pct') or 0.0), current_pnl)
            if current_pnl > float(getattr(signal, 'max_profit_reached', 0.0) or 0.0):
                setattr(signal, 'max_profit_reached', current_pnl)

            if d == 'LONG':
                if signal.target_2 and price >= signal.target_2:
                    return f"TARGET_2 (+{current_pnl:.1f}%)"
                # TP1 is a milestone, not an immediate exit. Before the halfway
                # buffer to TP2, protect at entry; after the buffer, trail to TP1.
                if target_1_hit and buffer_hit and signal.target_1 and price <= signal.target_1:
                    return f"TARGET_1 (+{self._calculate_pnl(signal, signal.target_1):.1f}%)"
                if target_1_hit and not buffer_hit and price <= signal.entry_price:
                    return f"TARGET_1 (+{self._calculate_pnl(signal, signal.target_1):.1f}%)"
                # Profit Protection Guard: If peak profit reached +5.0%, lock in profit at +1.0% breakeven floor
                if peak_pnl >= 5.0 and current_pnl <= 1.0:
                    return f"PROFIT_PROTECTION (+{current_pnl:.1f}%)"
                if signal.stop_loss and price <= signal.stop_loss:
                    return f"STOP_LOSS ({current_pnl:.1f}%)"
            elif d == 'SHORT':
                if signal.target_2 and price <= signal.target_2:
                    return f"TARGET_2 (+{current_pnl:.1f}%)"
                if target_1_hit and buffer_hit and signal.target_1 and price >= signal.target_1:
                    return f"TARGET_1 (+{self._calculate_pnl(signal, signal.target_1):.1f}%)"
                if target_1_hit and not buffer_hit and price >= signal.entry_price:
                    return f"TARGET_1 (+{self._calculate_pnl(signal, signal.target_1):.1f}%)"
                # Profit Protection Guard: If peak profit reached +5.0%, lock in profit at +1.0% breakeven floor
                if peak_pnl >= 5.0 and current_pnl <= 1.0:
                    return f"PROFIT_PROTECTION (+{current_pnl:.1f}%)"
                if signal.stop_loss and price >= signal.stop_loss:
                    return f"STOP_LOSS ({current_pnl:.1f}%)"
            return None
        except Exception:
            return None

    def _target_1_reached(self, signal, price) -> bool:
        try:
            if not signal.target_1:
                return False
            direction = signal.direction.upper()
            if direction == 'LONG':
                return price >= signal.target_1
            if direction == 'SHORT':
                return price <= signal.target_1
            return False
        except Exception:
            return False

    def _hydrate_milestone_flags(self, db):
        """
        Restore target_1_hit (from performance tracking) and buffer_hit (from
        market_conditions) onto freshly loaded signals. Without this, a worker
        restart silently disarms the break-even and trailing-stop protections.
        """
        if not self.signals:
            return
        try:
            ids = list(self.signals.keys())
            tp1_ids = set()
            chunk = 200
            for i in range(0, len(ids), chunk):
                perf = db.table('platform_signal_performance_tracking').select(
                    'signal_id'
                ).in_('signal_id', ids[i:i + chunk]).eq('target_1_hit', True).execute()
                tp1_ids.update(r['signal_id'] for r in (perf.data or []))

            restored_tp1 = 0
            restored_buffer = 0
            for sid, sig in self.signals.items():
                if sid in tp1_ids and not getattr(sig, 'target_1_hit', False):
                    setattr(sig, 'target_1_hit', True)
                    restored_tp1 += 1
                mc = (sig.market_conditions or {}) if hasattr(sig, 'market_conditions') else {}
                if mc.get('buffer_hit') and not getattr(sig, 'buffer_hit', False):
                    setattr(sig, 'buffer_hit', True)
                    restored_buffer += 1

            if restored_tp1 or restored_buffer:
                logger.info(
                    f"🔄 Milestone state restored: {restored_tp1} TP1 flags, "
                    f"{restored_buffer} buffer flags — break-even/trailing protections re-armed"
                )
        except Exception as e:
            logger.warning(f"Milestone flag hydration failed: {e}")

    def _update_peak_only(self, signal_id, signal, price, db):
        """
        Update max_profit_reached for a closed trade without touching the
        realized exit PnL (the full RPC would overwrite leveraged_pnl_percent,
        which is the canonical TP2 result).

        Writes only on a new peak, so this is cheap on regular ticks.
        """
        try:
            leverage = max(float(getattr(signal, 'leverage', 1) or 1), 1.0)
            lev_pnl = self._calculate_raw_pnl(signal, price) * leverage

            peak = getattr(signal, '_peak_cache', None)
            if peak is None:
                row = db.table('platform_signal_performance_tracking').select(
                    'max_profit_reached'
                ).eq('signal_id', signal_id).limit(1).execute()
                peak = float(row.data[0].get('max_profit_reached') or 0) if row.data else 0.0
                setattr(signal, '_peak_cache', peak)

            if lev_pnl > peak:
                setattr(signal, '_peak_cache', lev_pnl)
                db.table('platform_signal_performance_tracking').update({
                    'max_profit_reached': round(lev_pnl, 4),
                    'last_updated': datetime.now(timezone.utc).isoformat(),
                }).eq('signal_id', signal_id).execute()
        except Exception as e:
            logger.debug(f"Peak-only update failed for {signal.token_symbol}: {e}")

    def _is_signal_expired(self, signal) -> bool:
        try:
            now = datetime.now(timezone.utc)
            exp = signal.expires_at
            if isinstance(exp, str):
                exp = datetime.fromisoformat(exp.replace('Z', '+00:00'))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return now > exp
        except Exception:
            return False

    def _calculate_raw_pnl(self, signal, price) -> float:
        """Raw % price change with NO leverage — RPC applies leverage from DB."""
        try:
            ep = signal.entry_price
            if signal.direction.upper() == 'LONG':
                return ((price - ep) / ep) * 100
            return ((ep - price) / ep) * 100
        except Exception:
            return 0.0

    async def _fetch_rest_price(self, symbol: str) -> float:
        """
        Last-resort REST price fetch for expiry when no WebSocket tick was cached.
        Tries Binance futures first, then spot.  Returns 0.0 on any failure.
        """
        urls = [
            f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}",
            f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
        ]
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                for url in urls:
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                price = float(data.get("price", 0))
                                if price > 0:
                                    return price
                    except Exception:
                        continue
        except Exception:
            pass
        return 0.0

    def _calculate_pnl(self, signal, price) -> float:
        """Leveraged PnL for display/logging only."""
        try:
            raw = self._calculate_raw_pnl(signal, price)
            lev = getattr(signal, 'leverage', 1) or 1
            return raw * lev
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_symbol_set(self, signals) -> Set[str]:
        out = set()
        for s in signals:
            sym = self._normalize_to_futures_symbol(s.token_symbol).lower()
            if sym:
                out.add(sym)
        return out

    def _build_ws_url(self, symbols: Set[str]) -> str:
        if self._use_targeted_stream and symbols:
            if self._use_futures:
                stream_names = ("markPrice@1s", "miniTicker")
            else:
                stream_names = ("miniTicker",)
            streams = "/".join(
                f"{symbol}@{stream_name}"
                for symbol in sorted(symbols)
                for stream_name in stream_names
            )
            template = _FUTURES_WS_TARGETED if self._use_futures else _SPOT_WS_TARGETED
            return template.format(streams=streams)
        return _FUTURES_WS_ALL_MARK if self._use_futures else _SPOT_WS_ALL_MARK

    def _normalize_to_futures_symbol(self, symbol: str) -> str:
        """
        Normalize symbol variants to Binance futures market id format (e.g. BTCUSDT).
        Handles forms like BTC, BTCUSDT, BTC/USDT, BTC/USDT:USDT, BTC-USDT.
        """
        if not symbol:
            return ""

        s = str(symbol).upper().strip()
        s = s.replace("/USDT:USDT", "").replace("/USDT", "")
        s = s.replace(":USDT", "").replace("-USDT", "").replace("_USDT", "")
        s = re.sub(r"[^A-Z0-9]", "", s)

        if not s:
            return ""

        if s.endswith("USDT"):
            base = s[:-4]
        else:
            base = s

        base = re.sub(r"[^A-Z0-9]", "", base)
        if not base:
            return ""
        return f"{base}USDT"

    def _on_connected(self):
        self._last_connected_at = datetime.now(timezone.utc)
        self._reconnect_count += 1
        self._reconnect_delay = _RECONNECT_MIN_S  # reset on successful connect
        logger.info(f"✅ WebSocket connected (attempt #{self._reconnect_count}, "
                    f"{'futures' if self._use_futures else 'spot'})")

    async def _backoff_sleep(self):
        import random
        jitter = random.uniform(0, self._reconnect_delay * 0.2)
        delay = self._reconnect_delay + jitter
        logger.info(f"⏳ Reconnecting in {delay:.1f}s (backoff)")
        await asyncio.sleep(delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, _RECONNECT_MAX_S)


# Global instance
_simple_monitor = SimpleWebSocketPerformanceMonitor()

def get_simple_websocket_monitor():
    return _simple_monitor
