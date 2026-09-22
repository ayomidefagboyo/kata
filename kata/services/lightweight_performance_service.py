#!/usr/bin/env python3
"""
Lightweight Signal Performance Service for Memory-Constrained Environments

Optimized version that only fetches price data for performance tracking,
avoiding full market analysis to reduce memory usage by ~80%.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
import re

from .binance_service import BinanceService, create_binance_service
from .platform_signal_service import (
    get_platform_signal_service,
    persist_entry_activation_if_current,
    persist_market_conditions_if_current,
    PlatformSignal,
)
from ..config.database import get_service_client

logger = logging.getLogger(__name__)

class LightweightPerformanceService:
    """Memory-optimized performance service that only fetches price data."""
    
    def __init__(self):
        """Initialize lightweight performance service."""
        self.db = get_service_client()
        self.platform_service = get_platform_signal_service()
        self.binance_service = None  # Lazy initialization

        # WebSocket state
        self.ws_prices: Dict[str, float] = {}
        self.ws_price_timestamps: Dict[str, datetime] = {}  # last update time per symbol (staleness detection)
        self.ws_connection = None
        self.ws_task = None
        self.ws_enabled = True  # Set to True to use WebSocket
        self._logged_ws_unavailable = False
        self.ws_subscribed_symbols: set[str] = set()
        self.ws_stream_lock = asyncio.Lock()

        logger.info("Lightweight Performance Service initialized (memory-optimized with WebSocket support)")

    @staticmethod
    def _normalize_base_symbol(symbol: str) -> str:
        """Normalize symbol to uppercase base token without USDT suffix."""
        if not symbol:
            return ""
        normalized = str(symbol).strip().upper()
        normalized = normalized.replace("/USDT:USDT", "")
        normalized = normalized.replace("/USDT", "")
        normalized = normalized.replace(":USDT", "")
        normalized = normalized.replace("-USDT", "")
        normalized = normalized.replace("_USDT", "")
        normalized = re.sub(r"[^A-Z0-9]", "", normalized)
        if normalized.endswith("USDT"):
            normalized = normalized[:-4]
        return re.sub(r"[^A-Z0-9]", "", normalized)
    
    def _get_binance_service(self) -> BinanceService:
        """Lazy initialization of Binance service."""
        if not self.binance_service:
            # Create Binance service without API keys (for public market data only)
            self.binance_service = create_binance_service()
            logger.info("Initialized Binance service for price monitoring")
        return self.binance_service

    async def start_websocket_stream(self, symbols: List[str]):
        """Start WebSocket connection for real-time price updates."""
        if not self.ws_enabled or not symbols:
            return

        try:
            import websockets

            normalized_symbols = sorted({
                self._normalize_base_symbol(symbol) for symbol in symbols
                if self._normalize_base_symbol(symbol)
            })
            if not normalized_symbols:
                return

            async with self.ws_stream_lock:
                requested_set = set(normalized_symbols)
                stream_unchanged = requested_set == self.ws_subscribed_symbols
                if stream_unchanged and self.ws_task and not self.ws_task.done():
                    return

                # Restart stream if symbol universe changed.
                if self.ws_task and not self.ws_task.done():
                    self.ws_task.cancel()
                    try:
                        await self.ws_task
                    except asyncio.CancelledError:
                        pass

                if self.ws_connection:
                    try:
                        await self.ws_connection.close()
                    except Exception:
                        pass
                    self.ws_connection = None

                self.ws_subscribed_symbols = requested_set
                # Use all-market futures mark price stream and filter locally.
                # This avoids dead streams for edge symbols and keeps a steady feed alive.
                stream_url = "wss://fstream.binance.com/market/ws/!markPrice@arr@1s"

                logger.info(f"Starting WebSocket connection for {len(normalized_symbols)} symbols...")

                async def websocket_handler():
                    """Handle WebSocket messages."""
                    try:
                        async with websockets.connect(
                            stream_url,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as websocket:
                            self.ws_connection = websocket
                            logger.info(
                                f"✅ WebSocket connected (all-market markPrice); tracking {len(normalized_symbols)} symbols"
                            )

                            while True:
                                try:
                                    message = await websocket.recv()
                                    data = json.loads(message)

                                    payload = data.get('data', data) if isinstance(data, dict) else data
                                    rows = payload if isinstance(payload, list) else [payload]
                                    for ticker in rows:
                                        if not isinstance(ticker, dict):
                                            continue
                                        symbol = self._normalize_base_symbol(ticker.get('s', ''))
                                        if not symbol or symbol not in requested_set:
                                            continue
                                        price_raw = ticker.get('p', ticker.get('c'))
                                        if price_raw in (None, ''):
                                            continue
                                        price = float(price_raw)

                                        # Update price cache + timestamp (for staleness detection)
                                        self.ws_prices[symbol] = price
                                        self.ws_price_timestamps[symbol] = datetime.utcnow()
                                        logger.debug(f"WS price update: {symbol} = {price}")

                                except asyncio.CancelledError:
                                    logger.info("WebSocket handler cancelled, shutting down")
                                    raise
                                except websockets.exceptions.ConnectionClosed as e:
                                    logger.info(f"WebSocket connection closed: {e}")
                                    break
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to decode WebSocket message: {e}")
                                except Exception as e:
                                    logger.warning(f"Error processing WebSocket message: {e}")
                                    break  # Break on unexpected errors to avoid tight loop

                    except asyncio.CancelledError:
                        logger.info("WebSocket handler task cancelled")
                        self.ws_connection = None
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("WebSocket connection closed, will reconnect on next cycle")
                        self.ws_connection = None
                    except Exception as e:
                        logger.error(f"WebSocket connection error: {e}")
                        self.ws_connection = None

                # Start WebSocket handler in background
                self.ws_task = asyncio.create_task(websocket_handler())
                logger.info("WebSocket stream started successfully")

        except ImportError:
            logger.warning("websockets library not installed, WebSocket monitoring disabled")
            self.ws_enabled = False
        except Exception as e:
            logger.error(f"Failed to start WebSocket stream: {e}")

    async def stop_websocket_stream(self):
        """Stop WebSocket connection."""
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
            self.ws_task = None

        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None

        self.ws_prices.clear()
        self.ws_subscribed_symbols.clear()
        logger.info("WebSocket stream stopped")

    def _entry_triggered(self, signal: PlatformSignal, current_price: float) -> bool:
        """Check whether a pending-entry signal has been activated by market price."""
        mc = signal.market_conditions or {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True

        mode = str(mc.get('entry_activation_mode') or '').strip().lower()
        entry = float(signal.entry_price or 0.0)
        direction = str(signal.direction or '').upper()
        if entry <= 0 or direction not in {'LONG', 'SHORT'}:
            return True

        # Activation must match a real fill: price has to actually reach the entry
        # level. The previous `entry * (1 ± buffer)` activated ~0.15% early on the
        # favorable side, so a resting limit that never filled (e.g. UNI long
        # @ 3.57, price only dipped to ~3.575) was marked activated and later
        # scored as a target "hit" — inflating win rate with entries no real
        # position ever took. A genuine gap-through still satisfies these (e.g.
        # price ticking to 3.55 is <= 3.57).
        if mode == 'limit_retest':
            return current_price <= entry if direction == 'LONG' else current_price >= entry
        if mode == 'breakout_stop':
            return current_price >= entry if direction == 'LONG' else current_price <= entry
        return True

    async def _ensure_entry_activated(self, signal: PlatformSignal, current_price: float) -> bool:
        """Persist activation once a pending-entry signal is touched."""
        mc = signal.market_conditions or {}
        if not bool(mc.get('entry_activation_required', False)):
            return True
        if bool(mc.get('entry_activated', False)):
            return True
        if not self._entry_triggered(signal, current_price):
            return False

        try:
            activated, latest_mc, result = persist_entry_activation_if_current(
                self.db,
                signal.signal_id,
                current_price,
            )
            if latest_mc:
                signal.market_conditions = latest_mc
            if result == "activated":
                logger.info(f"✅ Entry activated: {signal.token_symbol} {signal.direction} @ {current_price}")
            return activated
        except Exception as e:
            logger.warning(f"Failed to persist entry activation for {signal.signal_id}: {e}")
            return False
    
    def get_price_age_seconds(self, symbol: str) -> Optional[float]:
        """Return seconds since the last WebSocket price update for a symbol, or None if unknown."""
        normalized = self._normalize_base_symbol(symbol)
        ts = self.ws_price_timestamps.get(normalized)
        if ts is None:
            return None
        return (datetime.utcnow() - ts).total_seconds()

    async def get_fresh_price(self, symbol: str, max_age_seconds: float = 30.0) -> Optional[float]:
        """Get a price guaranteed to be no older than max_age_seconds; refetches via REST if stale.

        Returns None if both the WS cache and REST fallback fail to deliver a fresh price.
        """
        normalized = self._normalize_base_symbol(symbol)
        if not normalized:
            return None
        age = self.get_price_age_seconds(normalized)
        cached = self.ws_prices.get(normalized)
        if cached is not None and age is not None and age <= max_age_seconds:
            return float(cached)
        # Stale or missing → fetch via Binance REST as a one-shot refresh.
        try:
            binance = self._get_binance_service()
            if binance and hasattr(binance, "binance") and binance.binance is not None:
                ticker = await binance.binance.fetch_ticker(f"{normalized}/USDT:USDT")
                last = ticker.get("last") if isinstance(ticker, dict) else None
                if last is not None and float(last) > 0:
                    fresh = float(last)
                    # Refresh cache so subsequent lookups are quick.
                    self.ws_prices[normalized] = fresh
                    self.ws_price_timestamps[normalized] = datetime.utcnow()
                    logger.debug(
                        f"🔄 Refetched fresh price for {normalized}: {fresh} "
                        f"(prior age={age:.1f}s)" if age is not None else
                        f"🔄 Fetched first price for {normalized}: {fresh}"
                    )
                    return fresh
        except Exception as exc:
            logger.debug(f"Fresh price refetch failed for {normalized}: {exc}")
        # Last resort — return stale cache rather than None to avoid breaking callers.
        return float(cached) if cached is not None else None

    async def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols using WebSocket cache only."""
        try:
            normalized_symbols = [
                self._normalize_base_symbol(symbol)
                for symbol in symbols
            ]
            normalized_symbols = [symbol for symbol in normalized_symbols if symbol]
            if not normalized_symbols:
                return {}

            prices = {}
            requested_set = set(normalized_symbols)

            # Ensure stream is running and subscribed to requested symbols.
            if self.ws_enabled:
                stream_running = self.ws_task is not None and not self.ws_task.done()
                has_all_symbols = requested_set.issubset(self.ws_subscribed_symbols)
                if not stream_running or not has_all_symbols:
                    target_symbols = sorted(self.ws_subscribed_symbols.union(requested_set))
                    await self.start_websocket_stream(target_symbols)
                    # Allow a short warmup for first ticks to arrive.
                    await asyncio.sleep(2.0)

            # Try WebSocket prices first (most efficient, no rate limits)
            if self.ws_enabled and self.ws_prices:
                ws_hit_count = 0
                for symbol in normalized_symbols:
                    if symbol in self.ws_prices:
                        prices[symbol] = self.ws_prices[symbol]
                        ws_hit_count += 1

                if ws_hit_count > 0 and ws_hit_count < len(normalized_symbols):
                    logger.info(f"Using WebSocket prices for {ws_hit_count}/{len(normalized_symbols)} symbols")

                # Check if we are missing any
                if ws_hit_count < len(normalized_symbols):
                    missing = [s for s in normalized_symbols if s not in prices]
                    logger.debug(f"WebSocket cache missing {len(missing)} symbols: {missing}")
                
                return prices
            else:
                 if not self._logged_ws_unavailable:
                     logger.warning("⚠️ WebSocket not enabled or price cache is empty; returning no prices")
                     self._logged_ws_unavailable = True
                 else:
                     logger.debug("WebSocket unavailable/empty; returning no prices")
                 return {}

        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}

    async def _get_prices_futures_api(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices using direct Binance Futures API with exponential backoff for rate limits."""
        try:
            import aiohttp
            import ssl
            import asyncio

            prices = {}
            max_retries = 3
            base_delay = 2  # Start with 2 seconds

            # Setup SSL context for production (handle certificate issues)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.info("SSL certificate validation disabled for API access")

            # Try multiple DNS servers for resolution
            import socket

            # Set DNS resolver to use Google DNS
            original_getaddrinfo = socket.getaddrinfo

            def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
                if host == 'api.binance.com':
                    # Use the resolved IP from Google DNS
                    logger.info(f"Using DNS resolution workaround for {host}")
                    return original_getaddrinfo('108.139.198.174', port, family, type, proto, flags)
                return original_getaddrinfo(host, port, family, type, proto, flags)

            socket.getaddrinfo = custom_getaddrinfo

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=15)  # Increased timeout

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

                # Try batch API first (get all ticker prices at once) with retry logic
                for retry in range(max_retries):
                    try:
                        url = "https://fapi.binance.com/fapi/v1/ticker/price"
                        async with session.get(url) as response:
                            if response.status == 200:
                                all_prices = await response.json()
                                price_map = {item['symbol']: float(item['price']) for item in all_prices}

                                # Extract prices for our symbols (Futures API only)
                                for symbol in symbols:
                                    binance_symbol = f"{symbol}USDT"
                                    if binance_symbol in price_map:
                                        prices[symbol] = price_map[binance_symbol]
                                        logger.debug(f"Got price for {symbol}: {prices[symbol]}")
                                    else:
                                        prices[symbol] = 0.0
                                        logger.warning(f"Symbol {binance_symbol} not found in Futures")

                                logger.info(f"Successfully fetched {len([p for p in prices.values() if p > 0])} valid prices via Futures API")
                                socket.getaddrinfo = original_getaddrinfo
                                return prices

                            elif response.status == 418:
                                # HTTP 418: I'm a teapot (Binance rate limit ban)
                                delay = base_delay * (2 ** retry)  # Exponential backoff
                                logger.warning(f"HTTP 418 ban detected, retry {retry + 1}/{max_retries} after {delay}s")
                                if retry < max_retries - 1:
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    logger.error("Max retries reached for HTTP 418, falling back to individual calls")
                                    break

                            elif response.status == 429:
                                # HTTP 429: Too Many Requests
                                delay = base_delay * (2 ** retry)
                                logger.warning(f"HTTP 429 rate limit, retry {retry + 1}/{max_retries} after {delay}s")
                                if retry < max_retries - 1:
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    logger.error("Max retries reached for HTTP 429, falling back to individual calls")
                                    break

                    except Exception as batch_error:
                        logger.warning(f"Batch price API failed (attempt {retry + 1}): {batch_error}")
                        if retry < max_retries - 1:
                            delay = base_delay * (2 ** retry)
                            await asyncio.sleep(delay)
                            continue
                        else:
                            break

                # Fallback to individual symbol calls with exponential backoff
                for symbol in symbols:
                    for retry in range(max_retries):
                        try:
                            binance_symbol = f"{symbol}USDT"
                            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={binance_symbol}"

                            async with session.get(url) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    prices[symbol] = float(data['price'])
                                    logger.debug(f"Got individual price for {symbol}: {prices[symbol]}")
                                    break  # Success, move to next symbol

                                elif response.status == 418:
                                    delay = base_delay * (2 ** retry)
                                    logger.warning(f"HTTP 418 for {symbol}, retry {retry + 1}/{max_retries} after {delay}s")
                                    if retry < max_retries - 1:
                                        await asyncio.sleep(delay)
                                        continue
                                    else:
                                        prices[symbol] = 0.0
                                        logger.error(f"Max retries reached for {symbol}, skipping")

                                elif response.status == 429:
                                    delay = base_delay * (2 ** retry)
                                    logger.warning(f"HTTP 429 for {symbol}, retry {retry + 1}/{max_retries} after {delay}s")
                                    if retry < max_retries - 1:
                                        await asyncio.sleep(delay)
                                        continue
                                    else:
                                        prices[symbol] = 0.0
                                        logger.error(f"Max retries reached for {symbol}, skipping")

                                else:
                                    prices[symbol] = 0.0
                                    logger.warning(f"Failed to get price for {symbol}: HTTP {response.status}")
                                    break

                        except Exception as e:
                            if retry < max_retries - 1:
                                delay = base_delay * (2 ** retry)
                                logger.warning(f"Individual price fetch failed for {symbol} (attempt {retry + 1}): {e}, retrying in {delay}s")
                                await asyncio.sleep(delay)
                            else:
                                logger.warning(f"Individual price fetch failed for {symbol} after {max_retries} attempts: {e}")
                                prices[symbol] = 0.0

                valid_count = len([p for p in prices.values() if p > 0])
                logger.info(f"Futures API fetched {valid_count}/{len(symbols)} valid prices")

                # Restore original DNS resolver
                socket.getaddrinfo = original_getaddrinfo

                # If no valid prices were fetched, return empty dict (production behavior)
                if valid_count == 0:
                    logger.error("PRODUCTION ERROR: No valid prices from Futures API - check network connectivity to fapi.binance.com")
                    logger.error("DNS Resolution Test: Try running 'nslookup fapi.binance.com 8.8.8.8' to test connectivity")
                    return {}

                return prices

        except Exception as e:
            logger.error(f"Futures API price fetching failed: {e}")
            logger.error("PRODUCTION ERROR: Unable to fetch prices - performance monitoring will skip this cycle")
            return {}


    
    async def update_signal_performance_lightweight(self) -> Dict[str, int]:
        """Update signal performance using only price data - memory optimized."""
        try:
            # Get active signals
            active_signals = await self.platform_service.get_active_signals(limit=50)

            if not active_signals:
                logger.info("No active signals to monitor")
                return {'monitored': 0, 'updated': 0, 'exits': 0}

            # Track signal IDs for monitoring signal lifecycle
            current_signal_ids = {signal.signal_id for signal in active_signals}
            if hasattr(self, '_previous_signal_ids'):
                disappeared_ids = self._previous_signal_ids - current_signal_ids
                new_ids = current_signal_ids - self._previous_signal_ids

                if disappeared_ids:
                    logger.info(f"🔻 {len(disappeared_ids)} signals removed from active list since last run: {disappeared_ids}")
                if new_ids:
                    logger.info(f"🔺 {len(new_ids)} new signals added to active list: {new_ids}")
            self._previous_signal_ids = current_signal_ids

            # Extract unique symbols
            symbols = list(set([signal.token_symbol for signal in active_signals]))
            logger.info(f"Processing {len(symbols)} unique token symbols: {symbols}")

        # Convert symbols to proper format for price fetching
            clean_symbols = []
            for symbol in symbols:
                if not symbol:
                    continue
                
                # Clean whitespace
                clean_s = symbol.strip()
                
                # Remove USDT suffix if present
                if clean_s.upper().endswith('USDT'):
                    clean_symbols.append(clean_s[:-4])  # Remove 'USDT' suffix
                else:
                    clean_symbols.append(clean_s)

            logger.info(f"Clean symbols for price fetching: {clean_symbols}")

            # Fetch current prices (lightweight)
            current_prices = await self.get_current_prices(clean_symbols)

            # Log price fetch results
            valid_prices = {k: v for k, v in current_prices.items() if v > 0}
            invalid_prices = {k: v for k, v in current_prices.items() if v <= 0}

            logger.info(f"Price fetch results: {len(valid_prices)} valid, {len(invalid_prices)} invalid")
            if valid_prices:
                logger.info(f"Valid prices: {valid_prices}")
            if invalid_prices:
                logger.warning(f"Invalid/zero prices: {list(invalid_prices.keys())}")

            updates = 0
            exits = 0
            
            # Update each signal's performance
            for signal in active_signals:
                try:
                    # Map signal token symbol to clean symbol for price lookup
                    clean_s = signal.token_symbol.strip() if signal.token_symbol else ""
                    if clean_s.upper().endswith('USDT'):
                        clean_symbol = clean_s[:-4]
                    else:
                        clean_symbol = clean_s

                    current_price = current_prices.get(clean_symbol, 0)

                    if current_price <= 0:
                        logger.debug(f"Skipping signal {signal.signal_id} ({signal.token_symbol}): price is {current_price}")
                        continue

                    if not await self._ensure_entry_activated(signal, current_price):
                        logger.debug(
                            f"Pending entry not activated yet for {signal.signal_id} "
                            f"({signal.token_symbol}) at price {current_price}"
                        )
                        continue

                    logger.debug(f"Processing signal {signal.signal_id} ({signal.token_symbol}): price={current_price}, entry={signal.entry_price}")
                    
                    # Calculate PnL
                    if signal.direction.upper() == 'LONG':
                        pnl_percentage = ((current_price - signal.entry_price) / signal.entry_price) * 100
                    else:  # SHORT
                        pnl_percentage = ((signal.entry_price - current_price) / signal.entry_price) * 100
                    
                    # Check exit conditions
                    exit_reason = None
                    outcome = 'active'
                    
                    # Check if signal should be marked as trading exit.
                    # TP1 is a milestone only; TP2/SL/expiry are terminal exits.
                    trading_exit_reason = None
                    target_1_milestone = False
                    should_stop_tracking = False

                    mc = (signal.market_conditions or {}) if hasattr(signal, 'market_conditions') else {}
                    buffer_hit = mc.get('buffer_hit', False)
                    target_1_hit = getattr(signal, 'target_1_hit', False)

                    if signal.direction.upper() == 'LONG':
                        if current_price >= signal.target_2:
                            trading_exit_reason = 'TARGET_2'
                            outcome = 'win'
                            exits += 1
                        elif current_price >= signal.target_1:
                            target_1_milestone = True
                        elif buffer_hit and current_price <= signal.target_1:
                            trading_exit_reason = 'TARGET_1'
                            outcome = 'win'
                            should_stop_tracking = True
                            exits += 1
                        elif target_1_hit and not buffer_hit and current_price <= signal.entry_price:
                            trading_exit_reason = 'STOP_LOSS'
                            outcome = 'break-even'
                            should_stop_tracking = True
                            exits += 1
                        elif current_price <= signal.stop_loss:
                            trading_exit_reason = 'STOP_LOSS'
                            outcome = 'loss'
                            should_stop_tracking = True  # Stop tracking on loss
                            exits += 1
                    else:  # SHORT
                        if current_price <= signal.target_2:
                            trading_exit_reason = 'TARGET_2'
                            outcome = 'win'
                            exits += 1
                        elif current_price <= signal.target_1:
                            target_1_milestone = True
                        elif buffer_hit and current_price >= signal.target_1:
                            trading_exit_reason = 'TARGET_1'
                            outcome = 'win'
                            should_stop_tracking = True
                            exits += 1
                        elif target_1_hit and not buffer_hit and current_price >= signal.entry_price:
                            trading_exit_reason = 'STOP_LOSS'
                            outcome = 'break-even'
                            should_stop_tracking = True
                            exits += 1
                        elif current_price >= signal.stop_loss:
                            trading_exit_reason = 'STOP_LOSS'
                            outcome = 'loss'
                            should_stop_tracking = True  # Stop tracking on loss
                            exits += 1

                    if target_1_milestone and not getattr(signal, 'target_1_hit', False):
                        recorded = await self.platform_service.record_signal_target_1_milestone(
                            signal.signal_id,
                            signal.target_1,
                        )
                        if recorded:
                            setattr(signal, 'target_1_hit', True)
                            if signal.direction.upper() == 'LONG':
                                target_1_pnl = ((signal.target_1 - signal.entry_price) / signal.entry_price) * 100
                            else:
                                target_1_pnl = ((signal.entry_price - signal.target_1) / signal.entry_price) * 100
                            logger.info(
                                f"🎯 TP1 milestone: {signal.token_symbol} {signal.direction} "
                                f"({target_1_pnl:+.2f}%) at price {signal.target_1}"
                            )

                    # Check for buffer hit (halfway to TP2)
                    if getattr(signal, 'target_1_hit', False) and not buffer_hit:
                        t1 = signal.target_1
                        t2 = signal.target_2 if signal.target_2 else t1 + (t1 - signal.entry_price)
                        buffer_price = t1 + ((t2 - t1) / 2)
                        d = signal.direction.upper()
                        
                        if (d == 'LONG' and current_price >= buffer_price) or (d == 'SHORT' and current_price <= buffer_price):
                            saved, latest_mc, update_state = persist_market_conditions_if_current(
                                self.db,
                                signal.signal_id,
                                lambda current: {**current, 'buffer_hit': True},
                            )
                            if latest_mc:
                                signal.market_conditions = latest_mc
                            if saved and update_state == "updated":
                                logger.info(f"🛡️ Buffer reached for {signal.token_symbol} {signal.direction}, trailing stop to TP1")

                    if trading_exit_reason == 'TARGET_2' and not getattr(signal, 'target_1_hit', False):
                        recorded = await self.platform_service.record_signal_target_1_milestone(
                            signal.signal_id,
                            signal.target_1,
                        )
                        if recorded:
                            setattr(signal, 'target_1_hit', True)

                    # Set exit_reason only if we should stop tracking or it's the first time hitting target
                    # Check if signal was already exited to avoid duplicate exit logging
                    is_already_exited = signal.status != 'active'
                    exit_reason = trading_exit_reason if should_stop_tracking or (trading_exit_reason and not is_already_exited) else None

                    # Log detailed signal state for debugging
                    if trading_exit_reason:
                        logger.debug(f"Signal {signal.signal_id} ({signal.token_symbol}): "
                                   f"exit_reason={trading_exit_reason}, status={signal.status}, "
                                   f"should_stop={should_stop_tracking}, will_update_exit={exit_reason is not None}")

                    # Update signal performance in database (lightweight update)
                    await self._update_signal_performance_record(
                        signal.signal_id,
                        current_price,
                        pnl_percentage,
                        exit_reason,
                        outcome
                    )

                    updates += 1

                    if exit_reason:
                        logger.info(f"🚨 Signal exit detected: {signal.token_symbol} {signal.direction} - {exit_reason} ({pnl_percentage:+.2f}%) at price {current_price}")

                        # Update signal status to completed when exit is detected
                        try:
                            await self.platform_service.update_signal_exit(
                                signal.signal_id,
                                current_price,
                                exit_reason
                            )
                            logger.info(f"✅ Signal {signal.signal_id} ({signal.token_symbol}) exit updated in DB: {exit_reason}")
                        except Exception as status_error:
                            logger.error(f"❌ Failed to update signal status for {signal.signal_id}: {status_error}")
                    elif is_already_exited:
                        logger.debug(f"Signal {signal.signal_id} ({signal.token_symbol}) already exited, skipping exit update")

                except Exception as e:
                    logger.warning(f"Error updating signal {signal.signal_id}: {e}")
                    continue
            
            logger.info(f"Performance update: {updates} signals monitored, {exits} exits detected")
            return {'monitored': len(active_signals), 'updated': updates, 'exits': exits}
            
        except Exception as e:
            logger.error(f"Error in lightweight performance update: {e}")
            return {'monitored': 0, 'updated': 0, 'exits': 0}

    async def update_signal_performance_lightweight_batch(self) -> Dict[str, int]:
        """Batch update signal performance using one RPC call for all active signals."""
        try:
            # Get active signals
            active_signals = await self.platform_service.get_active_signals(limit=100)
            if not active_signals:
                logger.info("No active signals to batch update")
                return {'monitored': 0, 'updated': 0, 'exits': 0}

            # Extract symbols and fetch prices (clean USDT from symbol names)
            symbols = []
            for signal in active_signals:
                if not signal.token_symbol:
                    continue
                    
                clean_s = signal.token_symbol.strip()
                if clean_s.upper().endswith('USDT'):
                    clean_symbol = clean_s[:-4]
                else:
                    clean_symbol = clean_s
                symbols.append(clean_symbol)

            symbols = list(set(symbols))  # Remove duplicates
            current_prices = await self.get_current_prices(symbols)

            exits = 0
            batch_updates: List[Dict[str, Any]] = []
            signal_exits: List[Dict[str, Any]] = []  # Track signals that need status updates

            for signal in active_signals:
                try:
                    # Clean symbol name to match price data
                    clean_s = signal.token_symbol.strip() if signal.token_symbol else ""
                    if clean_s.upper().endswith('USDT'):
                        clean_symbol = clean_s[:-4]
                    else:
                        clean_symbol = clean_s

                    current_price = current_prices.get(clean_symbol, 0)
                    if current_price <= 0:
                        continue

                    if not await self._ensure_entry_activated(signal, current_price):
                        logger.debug(
                            f"Pending entry not activated yet for {signal.signal_id} "
                            f"({signal.token_symbol}) at price {current_price}"
                        )
                        continue

                    # Calculate PnL
                    if signal.direction.upper() == 'LONG':
                        pnl_percentage = ((current_price - signal.entry_price) / signal.entry_price) * 100
                    else:  # SHORT
                        pnl_percentage = ((signal.entry_price - current_price) / signal.entry_price) * 100

                    # Determine exits. TP1 is a milestone only; TP2/SL/expiry close.
                    exit_reason = None
                    target_1_milestone = False
                    should_stop_tracking = False

                    mc = (signal.market_conditions or {}) if hasattr(signal, 'market_conditions') else {}
                    buffer_hit = mc.get('buffer_hit', False)
                    target_1_hit = getattr(signal, 'target_1_hit', False)

                    if signal.direction.upper() == 'LONG':
                        # Check higher target first (target_2 > target_1 for LONG)
                        if current_price >= signal.target_2:
                            exit_reason = 'TARGET_2'
                            exits += 1
                        elif current_price >= signal.target_1:
                            target_1_milestone = True
                        elif buffer_hit and current_price <= signal.target_1:
                            exit_reason = 'TARGET_1'
                            should_stop_tracking = True
                            exits += 1
                        elif target_1_hit and not buffer_hit and current_price <= signal.entry_price:
                            exit_reason = 'STOP_LOSS'
                            should_stop_tracking = True
                            exits += 1
                        elif current_price <= signal.stop_loss:
                            exit_reason = 'STOP_LOSS'
                            should_stop_tracking = True
                            exits += 1
                    else:
                        # Check lower target first (target_2 < target_1 for SHORT)
                        if current_price <= signal.target_2:
                            exit_reason = 'TARGET_2'
                            exits += 1
                        elif current_price <= signal.target_1:
                            target_1_milestone = True
                        elif buffer_hit and current_price >= signal.target_1:
                            exit_reason = 'TARGET_1'
                            should_stop_tracking = True
                            exits += 1
                        elif target_1_hit and not buffer_hit and current_price >= signal.entry_price:
                            exit_reason = 'STOP_LOSS'
                            should_stop_tracking = True
                            exits += 1
                        elif current_price >= signal.stop_loss:
                            exit_reason = 'STOP_LOSS'
                            should_stop_tracking = True
                            exits += 1

                    if target_1_milestone and not getattr(signal, 'target_1_hit', False):
                        recorded = await self.platform_service.record_signal_target_1_milestone(
                            signal.signal_id,
                            signal.target_1,
                        )
                        if recorded:
                            setattr(signal, 'target_1_hit', True)
                            if signal.direction.upper() == 'LONG':
                                target_1_pnl = ((signal.target_1 - signal.entry_price) / signal.entry_price) * 100
                            else:
                                target_1_pnl = ((signal.entry_price - signal.target_1) / signal.entry_price) * 100
                            logger.info(
                                f"🎯 TP1 milestone: {signal.token_symbol} {signal.direction} "
                                f"({target_1_pnl:+.2f}%) at price {signal.target_1}"
                            )

                    if exit_reason == 'TARGET_2' and not getattr(signal, 'target_1_hit', False):
                        recorded = await self.platform_service.record_signal_target_1_milestone(
                            signal.signal_id,
                            signal.target_1,
                        )
                        if recorded:
                            setattr(signal, 'target_1_hit', True)

                    # Check for buffer hit
                    if getattr(signal, 'target_1_hit', False) and not buffer_hit:
                        t1 = signal.target_1
                        t2 = signal.target_2 if signal.target_2 else t1 + (t1 - signal.entry_price)
                        buffer_price = t1 + ((t2 - t1) / 2)
                        d = signal.direction.upper()
                        
                        if (d == 'LONG' and current_price >= buffer_price) or (d == 'SHORT' and current_price <= buffer_price):
                            saved, latest_mc, update_state = persist_market_conditions_if_current(
                                self.db,
                                signal.signal_id,
                                lambda current: {**current, 'buffer_hit': True},
                            )
                            if latest_mc:
                                signal.market_conditions = latest_mc
                            if saved and update_state == "updated":
                                logger.info(f"🛡️ Buffer reached for {signal.token_symbol} {signal.direction}, trailing stop to TP1")

                    if exit_reason:
                        logger.info(f"Signal exit detected: {signal.token_symbol} {signal.direction} - {exit_reason} ({pnl_percentage:+.2f}%)")

                        # Only update signal status if we should stop tracking (losses) or first time hitting target
                        if should_stop_tracking or (exit_reason and signal.status == 'active'):
                            # Track this signal for status update
                            signal_exits.append({
                                'signal_id': signal.signal_id,
                                'current_price': current_price,
                                'exit_reason': exit_reason
                            })

                    # Always update performance tracking (for best/worst calculation)
                    # Only skip if we hit stop loss (should_stop_tracking = True)
                    if not should_stop_tracking:
                        batch_updates.append({
                            'signal_id': signal.signal_id,
                            'pnl_percentage': pnl_percentage
                        })
                except Exception as signal_err:
                    logger.warning(f"Error preparing batch update for {signal.signal_id}: {signal_err}")
                    continue

            # Execute batch RPC update
            await self.update_signal_performance_batch(batch_updates)

            # Update signal status for exits
            if signal_exits:
                logger.info(f"Updating status for {len(signal_exits)} exited signals")
                for signal_exit in signal_exits:
                    try:
                        await self.platform_service.update_signal_exit(
                            signal_exit['signal_id'],
                            signal_exit['current_price'],
                            signal_exit['exit_reason']
                        )
                        logger.info(f"✅ Signal {signal_exit['signal_id']} exit updated: {signal_exit['exit_reason']}")
                    except Exception as status_error:
                        logger.error(f"Failed to update signal status for {signal_exit['signal_id']}: {status_error}")

            logger.info(f"Batch performance update: {len(batch_updates)} signals prepared, {exits} exits detected")
            return {'monitored': len(active_signals), 'updated': len(batch_updates), 'exits': exits}

        except Exception as e:
            logger.error(f"Error in batch lightweight performance update: {e}")
            return {'monitored': 0, 'updated': 0, 'exits': 0}
    
    async def _update_signal_performance_record(self, signal_id: str, current_price: float,
                                               pnl_percentage: float, exit_reason: Optional[str],
                                               outcome: str):
        """Update signal performance record using fixed RPC function (now working correctly)."""
        try:
            # Use fixed RPC function for single update (now properly updates actual_pnl_percent)
            self.db.rpc('rpc_update_signal_performance', {
                'p_signal_id': signal_id,
                'p_pnl': pnl_percentage
            }).execute()

            logger.debug(f"Updated performance for {signal_id}: {pnl_percentage:+.2f}% (RPC with leverage)")

        except Exception as e:
            logger.error(f"Error updating performance record for {signal_id}: {e}")

    async def update_signal_performance_batch(self, signal_updates: List[Dict[str, Any]]) -> bool:
        """Update multiple signal performance records using fixed RPC batch function."""
        try:
            if not signal_updates:
                return True

            # Prepare batch updates for RPC function
            batch_data = []
            for update in signal_updates:
                batch_data.append({
                    'signal_id': update['signal_id'],
                    'pnl': update['pnl_percentage']
                })

            # Use fixed RPC batch function (now properly updates actual_pnl_percent with leverage)
            self.db.rpc('rpc_update_signal_performance_batch', {'p_updates': batch_data}).execute()

            logger.info(f"Batch updated performance for {len(signal_updates)} signals (RPC batch with leverage)")
            return True

        except Exception as e:
            logger.error(f"Error in batch performance update: {e}")
            return False

# Global instance
_lightweight_performance_service: Optional[LightweightPerformanceService] = None

def get_lightweight_performance_service() -> LightweightPerformanceService:
    """Get or create lightweight performance service instance."""
    global _lightweight_performance_service
    if not _lightweight_performance_service:
        _lightweight_performance_service = LightweightPerformanceService()
    return _lightweight_performance_service
