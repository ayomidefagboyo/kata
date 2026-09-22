"""
Hyperliquid WebSocket Data Ingestion Service

Real-time market data streaming for Hyperliquid futures trading:
- Live candlestick data for all perpetuals
- Real-time order book updates
- Open Interest and funding rate streams
- Position and balance updates
- Trade execution confirmations

Optimized for high-frequency trading with the Yuki agent.
"""

import asyncio
import json
import logging
import random
import re
import websockets
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

_TRANSIENT_HYPERLIQUID_STATUS_CODES = {
    408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524
}


def hyperliquid_error_status(error: Any) -> Optional[int]:
    """Extract an HTTP status from SDK, HTTP, and WebSocket exceptions."""
    response = getattr(error, "response", None)
    for candidate in (
        getattr(response, "status_code", None),
        getattr(response, "status", None),
        getattr(error, "status_code", None),
        getattr(error, "status", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            pass

    if getattr(error, "args", None):
        try:
            return int(error.args[0])
        except (TypeError, ValueError):
            pass

    match = re.search(r"\b(4\d{2}|5\d{2})\b", str(error or ""))
    return int(match.group(1)) if match else None


def is_transient_hyperliquid_error(error: Any) -> bool:
    """True for gateway/rate-limit/time-out failures expected to self-heal."""
    status = hyperliquid_error_status(error)
    if status in _TRANSIENT_HYPERLIQUID_STATUS_CODES:
        return True
    message = str(error or "").lower()
    return any(token in message for token in (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection closed",
        "server disconnected",
        "bad gateway",
        "gateway timeout",
    ))


def summarize_hyperliquid_error(error: Any) -> str:
    """Produce a concise log-safe failure reason without gateway HTML pages."""
    status = hyperliquid_error_status(error)
    if status is not None:
        return f"HTTP {status}"
    message = " ".join(str(error or "unknown error").split())
    if "<html" in message.lower() or "<!doctype" in message.lower():
        return "temporary gateway response"
    return message[:180]


class WSMessageType(Enum):
    """WebSocket message types for Hyperliquid."""
    CANDLE = "candle"
    L2_BOOK = "l2Book"
    TRADES = "trades"
    USER_EVENTS = "userEvents"
    USER_FILLS = "userFills"
    USER_FUNDING = "userFunding"
    ALL_MIDS = "allMids"
    WEB_DATA2 = "webData2"
    SPOT_STATE = "spotState"
    ALL_DEXS_CLEARINGHOUSE_STATE = "allDexsClearinghouseState"
    OPEN_ORDERS = "openOrders"


@dataclass
class CandleData:
    """Real-time candlestick data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float


@dataclass
class OrderBookLevel:
    """Order book level data."""
    price: float
    size: float


@dataclass
class OrderBookData:
    """Real-time order book data."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float


@dataclass
class TradeData:
    """Real-time trade data."""
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: str


@dataclass
class FundingData:
    """Real-time funding rate data."""
    symbol: str
    timestamp: datetime
    funding_rate: float
    next_funding_time: datetime
    premium: float


@dataclass
class OpenInterestData:
    """Open interest data."""
    symbol: str
    timestamp: datetime
    open_interest: float
    oi_change_24h: float


@dataclass
class UserEventData:
    """User-specific trading events."""
    event_type: str
    timestamp: datetime
    symbol: str
    data: Dict[str, Any]


class HyperliquidWebSocketClient:
    """
    High-performance WebSocket client for Hyperliquid real-time data.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Multiple subscription management
    - Data validation and error handling
    - Real-time data caching
    - Performance monitoring
    """
    
    def __init__(self, testnet: bool = False, user_address: Optional[str] = None):
        """Initialize WebSocket client."""
        self.testnet = testnet
        self.user_address = user_address
        
        # WebSocket URLs
        self.ws_url = (
            "wss://api.hyperliquid-testnet.xyz/ws" if testnet 
            else "wss://api.hyperliquid.xyz/ws"
        )
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_running = False
        self.reconnect_attempts = 0
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self._connect_failure_count = 0
        self._outage_started_at: Optional[datetime] = None
        self._last_outage_warning_at: Optional[datetime] = None
        
        # Subscriptions and callbacks
        self.subscriptions: Set[str] = set()
        self.callbacks: Dict[WSMessageType, List[Callable]] = defaultdict(list)
        
        # Data storage for real-time access
        self.latest_candles: Dict[str, CandleData] = {}
        self.latest_order_books: Dict[str, OrderBookData] = {}
        self.latest_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.latest_funding: Dict[str, FundingData] = {}
        self.latest_open_interest: Dict[str, OpenInterestData] = {}
        self.latest_mids: Dict[str, float] = {}

        # Pushed account state for this connection's single user_address. The
        # all-DEX clearinghouse and open-order feeds keep balances, positions,
        # and orders current without REST polling that Hyperliquid rate-limits.
        self.latest_account_state: Optional[Dict[str, Any]] = None
        self.latest_account_state_at: Optional[datetime] = None
        self.latest_spot_state: Optional[Dict[str, Any]] = None
        self.latest_spot_state_at: Optional[datetime] = None
        self.latest_open_orders: Optional[List[Dict[str, Any]]] = None
        self.latest_open_orders_at: Optional[datetime] = None
        self.latest_open_orders_by_dex: Dict[str, List[Dict[str, Any]]] = {}
        self.latest_all_dex_states: Optional[Dict[str, Dict[str, Any]]] = None
        self.latest_all_dex_states_at: Optional[datetime] = None
        
        # Performance tracking
        self.message_count = 0
        self.last_message_time = datetime.now()
        self.connection_start_time: Optional[datetime] = None
        
        # Keep a bounded hand-off between recv and parsing. Account-state
        # messages are small and infrequent, while market-data bursts can be
        # much faster; 1,000 entries provides ample backpressure without
        # allowing a stalled processor to retain tens of MB of raw JSON.
        self.message_queue = asyncio.Queue(maxsize=1000)
        self._processor_task: Optional[asyncio.Task] = None
        # Exactly one recv() loop may run per client. Two concurrent loops call
        # websocket.recv() at once ('cannot call recv while another coroutine is
        # already waiting for the next message'), which cascades into an
        # error->reconnect->resubscribe storm that leaks sockets/tasks and OOMs.
        self._listen_task: Optional[asyncio.Task] = None
        self._listening = False

        logger.info(f"HyperliquidWebSocketClient initialized (testnet: {testnet})")

    def _record_connect_success(self, strict_ssl: bool) -> None:
        now = datetime.now()
        if self._outage_started_at is not None:
            downtime = max(0.0, (now - self._outage_started_at).total_seconds())
            logger.info(
                "Hyperliquid WebSocket recovered after %d failed attempt(s) and %.0fs",
                self._connect_failure_count,
                downtime,
            )
        else:
            logger.info(
                "Successfully connected to Hyperliquid WebSocket%s",
                " with strict SSL" if strict_ssl else "",
            )
        self._connect_failure_count = 0
        self._outage_started_at = None
        self._last_outage_warning_at = None

    def _record_connect_failure(self, error: Exception) -> None:
        now = datetime.now()
        self._connect_failure_count += 1
        if self._outage_started_at is None:
            self._outage_started_at = now

        if not is_transient_hyperliquid_error(error):
            logger.error("Failed to connect to WebSocket: %s", summarize_hyperliquid_error(error))
            return

        should_warn = (
            self._last_outage_warning_at is None
            or (now - self._last_outage_warning_at).total_seconds() >= 300
        )
        if should_warn:
            logger.warning(
                "Hyperliquid WebSocket temporarily unavailable (%s); reconnecting automatically",
                summarize_hyperliquid_error(error),
            )
            self._last_outage_warning_at = now
        else:
            logger.debug(
                "Hyperliquid WebSocket reconnect attempt failed: %s",
                summarize_hyperliquid_error(error),
            )
    
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            logger.info(f"Connecting to Hyperliquid WebSocket: {self.ws_url}")

            # A server-closed protocol object can still retain receive buffers
            # until it is explicitly finalized. Reconnects replace the socket,
            # so close the old object first to release those buffers promptly.
            if self.websocket is not None:
                try:
                    await self.websocket.close()
                except Exception:
                    pass
                self.websocket = None
            
            # Handle SSL certificate issues (common on macOS)
            import ssl
            ssl_context = ssl.create_default_context()
            
            # For testnet or if SSL verification fails, disable strict verification
            if self.testnet:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            else:
                # For mainnet, try to use system certificates but fall back if needed
                try:
                    # Test with strict SSL first
                    self.websocket = await websockets.connect(
                        self.ws_url,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=10
                    )
                    self.is_connected = True
                    self.reconnect_attempts = 0
                    self.connection_start_time = datetime.now()

                    self._record_connect_success(strict_ssl=True)

                    self._ensure_processor_task()

                    return True
                    
                except ssl.SSLCertVerificationError:
                    logger.warning("SSL certificate verification failed, using relaxed SSL settings")
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
            
            # Use SSL context (either relaxed or testnet)
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
                ssl=ssl_context
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.connection_start_time = datetime.now()

            self._record_connect_success(strict_ssl=False)

            self._ensure_processor_task()

            return True

        except Exception as e:
            self._record_connect_failure(e)
            self.is_connected = False
            return False

    def _ensure_processor_task(self):
        """
        Start the queue-consumer task exactly once per client.

        is_running must be True before the task first runs: _message_processor
        loops on `while self.is_running`, and previously connect() spawned it
        while is_running was still False (start_listening set it later), so
        depending on scheduling the processor could exit immediately and leave
        the queue filling with no consumer. Spawning here (guarded) also stops
        each reconnect's connect() call from stacking a duplicate processor.
        """
        self.is_running = True
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._message_processor())
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        try:
            self.is_running = False
            self.is_connected = False
            
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            logger.info("Disconnected from Hyperliquid WebSocket")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def subscribe_to_candles(self, symbols: List[str], interval: str = "1m") -> bool:
        """
        Subscribe to real-time candlestick data.
        
        Args:
            symbols: List of symbols to subscribe to
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        """
        try:
            for symbol in symbols:
                subscription = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "candle",
                        "coin": symbol,
                        "interval": interval
                    }
                }
                
                await self._send_message(subscription)
                self.subscriptions.add(f"candle_{symbol}_{interval}")
            
            logger.info(f"Subscribed to candles for {symbols} ({interval})")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to candles: {e}")
            return False
    
    async def subscribe_to_orderbook(self, symbols: List[str]) -> bool:
        """Subscribe to real-time order book data."""
        try:
            for symbol in symbols:
                subscription = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "l2Book",
                        "coin": symbol
                    }
                }
                
                await self._send_message(subscription)
                self.subscriptions.add(f"l2Book_{symbol}")
            
            logger.info(f"Subscribed to order books for {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to order books: {e}")
            return False
    
    async def subscribe_to_trades(self, symbols: List[str]) -> bool:
        """Subscribe to real-time trade data."""
        try:
            for symbol in symbols:
                subscription = {
                    "method": "subscribe",
                    "subscription": {
                        "type": "trades",
                        "coin": symbol
                    }
                }
                
                await self._send_message(subscription)
                self.subscriptions.add(f"trades_{symbol}")
            
            logger.info(f"Subscribed to trades for {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            return False
    
    async def subscribe_to_user_events(self) -> bool:
        """Subscribe to user-specific events (requires authentication)."""
        try:
            if not self.user_address:
                raise ValueError("User address required for user events subscription")
            
            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "userEvents",
                    "user": self.user_address
                }
            }
            
            await self._send_message(subscription)
            self.subscriptions.add(f"userEvents_{self.user_address}")
            
            logger.info(f"Subscribed to user events for {self.user_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to user events: {e}")
            return False
    
    async def subscribe_to_web_data2(self) -> bool:
        """
        Subscribe to webData2 for this connection's user - pushes clearinghouseState
        (account value, withdrawable, margin summary) on every account-affecting
        update instead of requiring a REST clearinghouseState poll.
        """
        try:
            if not self.user_address:
                raise ValueError("User address required for webData2 subscription")

            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "webData2",
                    "user": self.user_address
                }
            }

            await self._send_message(subscription)
            self.subscriptions.add(f"webData2_{self.user_address}")

            logger.info(f"Subscribed to webData2 for {self.user_address}")
            return True

        except Exception as e:
            logger.error(f"Error subscribing to webData2: {e}")
            return False

    async def subscribe_to_all_dexs_clearinghouse_state(self) -> bool:
        """Subscribe to positions and margin state across the main and every HIP-3 DEX."""
        try:
            if not self.user_address:
                raise ValueError("User address required for all-DEX account state")

            await self._send_message({
                "method": "subscribe",
                "subscription": {
                    "type": "allDexsClearinghouseState",
                    "user": self.user_address,
                },
            })
            self.subscriptions.add(f"allDexsClearinghouseState_{self.user_address}")
            logger.info(f"Subscribed to all-DEX clearinghouse state for {self.user_address}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to all-DEX clearinghouse state: {e}")
            return False

    async def subscribe_to_spot_state(self, is_portfolio_margin: bool = False) -> bool:
        """Subscribe to spot balances, the collateral source for unified accounts."""
        try:
            if not self.user_address:
                raise ValueError("User address required for spot-state updates")

            await self._send_message({
                "method": "subscribe",
                "subscription": {
                    "type": "spotState",
                    "user": self.user_address,
                    "isPortfolioMargin": is_portfolio_margin,
                },
            })
            self.subscriptions.add(f"spotState|{int(is_portfolio_margin)}")
            logger.info(f"Subscribed to spot state for {self.user_address}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to spot state: {e}")
            return False

    async def subscribe_to_open_orders(self, dex: str = "") -> bool:
        """Subscribe to one perp DEX's open-order book for this wallet."""
        try:
            if not self.user_address:
                raise ValueError("User address required for open-order updates")
            await self._send_message({
                "method": "subscribe",
                "subscription": {
                    "type": "openOrders",
                    "user": self.user_address,
                    "dex": dex,
                },
            })
            self.subscriptions.add(f"openOrders|{dex}")
            logger.info(
                f"Subscribed to {dex or 'main'} DEX open orders for {self.user_address}"
            )
            return True
        except Exception as e:
            logger.error(f"Error subscribing to open orders for DEX {dex or 'main'}: {e}")
            return False

    async def subscribe_to_all_mids(self) -> bool:
        """Subscribe to all market mid prices."""
        try:
            subscription = {
                "method": "subscribe",
                "subscription": {
                    "type": "allMids"
                }
            }
            
            await self._send_message(subscription)
            self.subscriptions.add("allMids")
            
            logger.info("Subscribed to all mids")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to all mids: {e}")
            return False
    
    def add_callback(self, message_type: WSMessageType, callback: Callable):
        """Add callback for specific message type."""
        self.callbacks[message_type].append(callback)
        logger.debug(f"Added callback for {message_type.value}")
    
    def remove_callback(self, message_type: WSMessageType, callback: Callable):
        """Remove callback for specific message type."""
        if callback in self.callbacks[message_type]:
            self.callbacks[message_type].remove(callback)
            logger.debug(f"Removed callback for {message_type.value}")
    
    def has_active_listener(self) -> bool:
        """True when a recv() loop is already running (or mid-reconnect) for this
        client. A live listener self-heals its own drops via _attempt_reconnect,
        so callers must not spawn a second one or re-drive connect()."""
        return self._listening or (self._listen_task is not None and not self._listen_task.done())

    def ensure_listening(self) -> Optional[asyncio.Task]:
        """Start the recv loop exactly once. Idempotent across reconnects and
        concurrent callers — use this instead of asyncio.create_task(start_listening())
        so duplicate listeners can never race on recv()."""
        if self._listen_task is not None and not self._listen_task.done():
            return self._listen_task
        self._listen_task = asyncio.create_task(self.start_listening())
        return self._listen_task

    async def start_listening(self):
        """Listen for WebSocket messages, reconnecting and resuming on drops."""
        if not self.is_connected:
            raise ValueError("Not connected to WebSocket")

        # Refuse to run a second recv loop even if a duplicate task was spawned:
        # two coroutines awaiting websocket.recv() concurrently raise
        # 'cannot call recv while another coroutine is already waiting'.
        if self._listening:
            logger.debug("start_listening() ignored: a listener loop is already active")
            return
        self._listening = True

        self._ensure_processor_task()

        try:
            await self._listen_loop()
        finally:
            self._listening = False

    async def _listen_loop(self):
        while self.is_running:
            try:
                while self.is_running and self.is_connected:
                    try:
                        # Wait for incoming message
                        message = await asyncio.wait_for(
                            self.websocket.recv(),
                            timeout=30.0
                        )

                        # Add to processing queue
                        await self.message_queue.put(message)

                    except asyncio.TimeoutError:
                        # Hyperliquid requires an application-level heartbeat
                        # when a subscription has no server message for 60s.
                        # A WebSocket protocol ping alone does not reset that
                        # timer, which previously caused a reconnect every
                        # minute and retained socket/parser buffers over time.
                        await self._send_message({"method": "ping"})
                        continue

                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Hyperliquid WebSocket connection closed; reconnecting automatically")
                        self.is_connected = False
                        break

            except Exception as e:
                logger.error(f"Error in message listening loop: {e}")
                self.is_connected = False

            if not self.is_running:
                break

            # Reconnect and re-enter the recv loop. Previously this method
            # returned after one reconnect, so the client looked healthy
            # (is_connected True, subscriptions intact) but nothing ever
            # called recv() again and all live data silently stopped.
            await self._attempt_reconnect()
            if not self.is_connected:
                logger.info("Listening loop exiting because the WebSocket client stopped")
                break
    
    async def _message_processor(self):
        """Process incoming WebSocket messages."""
        while self.is_running:
            try:
                # Get message from queue
                raw_message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Parse and route message
                await self._process_message(raw_message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _process_message(self, raw_message: str):
        """Process and route a WebSocket message."""
        try:
            message = json.loads(raw_message)
            
            self.message_count += 1
            self.last_message_time = datetime.now()
            
            # Route message based on channel/type
            channel = message.get("channel")
            data = message.get("data")
            
            if not channel or not data:
                return
            
            # Process different message types
            if channel == "candle":
                await self._handle_candle_message(data)
            elif channel == "l2Book":
                await self._handle_orderbook_message(data)
            elif channel == "trades":
                await self._handle_trades_message(data)
            elif channel == "userEvents":
                await self._handle_user_events_message(data)
            elif channel == "allMids":
                await self._handle_all_mids_message(data)
            elif channel == "webData2":
                await self._handle_web_data2_message(data)
            elif channel == "spotState":
                await self._handle_spot_state_message(data)
            elif channel == "allDexsClearinghouseState":
                await self._handle_all_dexs_clearinghouse_state_message(data)
            elif channel == "openOrders":
                await self._handle_open_orders_message(data)
            else:
                logger.debug(f"Unknown message channel: {channel}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_candle_message(self, data: Dict[str, Any]):
        """Handle candlestick data message."""
        try:
            coin = data.get("coin")
            candle_data = data.get("data", {})
            
            if not coin or not candle_data:
                return
            
            # Create candle object
            candle = CandleData(
                symbol=coin,
                timestamp=datetime.fromtimestamp(candle_data.get("t", 0) / 1000),
                open=float(candle_data.get("o", 0)),
                high=float(candle_data.get("h", 0)),
                low=float(candle_data.get("l", 0)),
                close=float(candle_data.get("c", 0)),
                volume=float(candle_data.get("v", 0)),
                vwap=float(candle_data.get("vwap", 0))
            )
            
            # Store latest candle
            self.latest_candles[coin] = candle
            
            # Trigger callbacks
            for callback in self.callbacks[WSMessageType.CANDLE]:
                try:
                    await callback(candle)
                except Exception as e:
                    logger.error(f"Error in candle callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling candle message: {e}")
    
    async def _handle_orderbook_message(self, data: Dict[str, Any]):
        """Handle order book data message."""
        try:
            coin = data.get("coin")
            book_data = data.get("data", {})
            
            if not coin or not book_data:
                return
            
            # Parse bids and asks
            bids = [
                OrderBookLevel(price=float(level[0]), size=float(level[1]))
                for level in book_data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(level[0]), size=float(level[1]))
                for level in book_data.get("asks", [])
            ]
            
            # Calculate mid price
            mid_price = 0.0
            if bids and asks:
                mid_price = (bids[0].price + asks[0].price) / 2
            
            # Create order book object
            order_book = OrderBookData(
                symbol=coin,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                mid_price=mid_price
            )
            
            # Store latest order book
            self.latest_order_books[coin] = order_book
            
            # Trigger callbacks
            for callback in self.callbacks[WSMessageType.L2_BOOK]:
                try:
                    await callback(order_book)
                except Exception as e:
                    logger.error(f"Error in order book callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling order book message: {e}")
    
    async def _handle_trades_message(self, data: Any):
        """Handle trades data message.

        Hyperliquid's trades channel pushes a flat LIST of trade objects, each
        carrying its own 'coin' — not a {coin, data:[...]} envelope. Calling
        data.get() on that list raised 'list object has no attribute get' on
        every message, flooding the logs. Accept both shapes defensively.
        """
        try:
            if isinstance(data, list):
                # Flat list of trades, each with its own 'coin'.
                trades_by_coin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for trade_data in data:
                    if isinstance(trade_data, dict) and trade_data.get("coin"):
                        trades_by_coin[trade_data["coin"]].append(trade_data)
            elif isinstance(data, dict):
                # Envelope shape: {"coin": "BTC", "data": [ ... ]}
                coin = data.get("coin")
                trades_list = data.get("data", [])
                trades_by_coin = {coin: trades_list} if coin and isinstance(trades_list, list) else {}
            else:
                return

            if not trades_by_coin:
                return

            for coin, trades_list in trades_by_coin.items():
                for trade_data in trades_list:
                    try:
                        trade = TradeData(
                            symbol=coin,
                            timestamp=datetime.fromtimestamp(trade_data.get("time", 0) / 1000),
                            price=float(trade_data.get("px", 0)),
                            size=float(trade_data.get("sz", 0)),
                            side=trade_data.get("side", ""),
                            trade_id=str(trade_data.get("tid", ""))
                        )
                    except (TypeError, ValueError):
                        continue

                    # Store in rolling buffer
                    self.latest_trades[coin].append(trade)

                # Trigger callbacks per affected coin
                for callback in self.callbacks[WSMessageType.TRADES]:
                    try:
                        await callback(coin, list(self.latest_trades[coin]))
                    except Exception as e:
                        logger.error(f"Error in trades callback: {e}")

        except Exception as e:
            logger.error(f"Error handling trades message: {e}")
    
    async def _handle_user_events_message(self, data: Dict[str, Any]):
        """Handle user events message."""
        try:
            event = UserEventData(
                event_type=data.get("eventType", ""),
                timestamp=datetime.fromtimestamp(data.get("time", 0) / 1000),
                symbol=data.get("coin", ""),
                data=data
            )
            
            # Trigger callbacks
            for callback in self.callbacks[WSMessageType.USER_EVENTS]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in user events callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling user events message: {e}")
    
    async def _handle_all_mids_message(self, data: Dict[str, Any]):
        """Handle all mids message."""
        try:
            mids = data.get("mids", {})
            
            # Update latest mids
            for symbol, price in mids.items():
                self.latest_mids[symbol] = float(price)
            
            # Trigger callbacks
            for callback in self.callbacks[WSMessageType.ALL_MIDS]:
                try:
                    await callback(self.latest_mids.copy())
                except Exception as e:
                    logger.error(f"Error in all mids callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling all mids message: {e}")

    async def _handle_web_data2_message(self, data: Dict[str, Any]):
        """
        Handle webData2 push - cache the clearinghouseState so readiness/balance
        checks can read it instead of hitting the rate-limited REST endpoint.
        """
        try:
            clearinghouse_state = data.get("clearinghouseState")
            if not isinstance(clearinghouse_state, dict):
                # Unexpected payload shape: skip silently and let callers fall
                # back to REST rather than caching something unusable.
                return

            self.latest_account_state = clearinghouse_state
            self.latest_account_state_at = datetime.now()

            open_orders = data.get("openOrders")
            if isinstance(open_orders, list):
                self.latest_open_orders = open_orders

            for callback in self.callbacks[WSMessageType.WEB_DATA2]:
                try:
                    await callback(clearinghouse_state)
                except Exception as e:
                    logger.error(f"Error in webData2 callback: {e}")

        except Exception as e:
            logger.error(f"Error handling webData2 message: {e}")

    async def _handle_all_dexs_clearinghouse_state_message(self, data: Dict[str, Any]):
        """Cache the main + HIP-3 position states pushed for one wallet."""
        try:
            raw_states = data.get("clearinghouseStates")
            if isinstance(raw_states, dict):
                states = raw_states
            elif isinstance(raw_states, list):
                states = {
                    str(item[0]): item[1]
                    for item in raw_states
                    if isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[1], dict)
                }
            else:
                return
            self.latest_all_dex_states = states
            self.latest_all_dex_states_at = datetime.now()
            for callback in self.callbacks[WSMessageType.ALL_DEXS_CLEARINGHOUSE_STATE]:
                try:
                    await callback(states)
                except Exception as e:
                    logger.error(f"Error in all-DEX account-state callback: {e}")
        except Exception as e:
            logger.error(f"Error handling all-DEX clearinghouse state: {e}")

    async def _handle_spot_state_message(self, data: Dict[str, Any]):
        """Cache the pushed spot balance used by unified and portfolio accounts."""
        try:
            state = data.get("spotState")
            if not isinstance(state, dict):
                return
            self.latest_spot_state = state
            self.latest_spot_state_at = datetime.now()
            for callback in self.callbacks[WSMessageType.SPOT_STATE]:
                try:
                    await callback(state)
                except Exception as e:
                    logger.error(f"Error in spot-state callback: {e}")
        except Exception as e:
            logger.error(f"Error handling spot state: {e}")

    async def _handle_open_orders_message(self, data: Dict[str, Any]):
        """Cache a DEX-scoped open-order snapshot."""
        try:
            orders = data.get("orders")
            if not isinstance(orders, list):
                return
            dex = str(data.get("dex") or "")
            self.latest_open_orders_by_dex[dex] = orders
            self.latest_open_orders = [
                order
                for dex_orders in self.latest_open_orders_by_dex.values()
                for order in dex_orders
            ]
            self.latest_open_orders_at = datetime.now()
            for callback in self.callbacks[WSMessageType.OPEN_ORDERS]:
                try:
                    await callback(self.latest_open_orders)
                except Exception as e:
                    logger.error(f"Error in open-orders callback: {e}")
        except Exception as e:
            logger.error(f"Error handling open-orders message: {e}")

    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if not self.is_connected or not self.websocket:
            raise ValueError("Not connected to WebSocket")
        
        await self.websocket.send(json.dumps(message))
    
    async def _attempt_reconnect(self):
        """Reconnect indefinitely with jittered exponential backoff."""
        while self.is_running and not self.is_connected:
            self.reconnect_attempts += 1
            base_delay = min(
                self.reconnect_delay * (2 ** min(self.reconnect_attempts - 1, 6)),
                60,
            )
            # Multiple service replicas can share this upstream. Jitter prevents all
            # processes from retrying a recovering gateway at the same instant.
            delay = max(1.0, base_delay * random.uniform(0.8, 1.2))
            
            logger.debug(
                "Attempting Hyperliquid reconnect #%d in %.1fs",
                self.reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            
            if await self.connect():
                # Re-establish subscriptions
                await self._resubscribe()
                break
        
        if not self.is_connected and self.is_running:
            logger.warning("Hyperliquid WebSocket reconnect loop stopped before recovery")
    
    async def _resubscribe(self):
        """Re-establish all subscriptions after reconnect."""
        logger.info("Re-establishing subscriptions...")
        
        for subscription in list(self.subscriptions):
            try:
                if subscription.startswith("candle_"):
                    parts = subscription.split("_")
                    symbol, interval = parts[1], parts[2]
                    await self.subscribe_to_candles([symbol], interval)
                elif subscription.startswith("l2Book_"):
                    symbol = subscription.split("_")[1]
                    await self.subscribe_to_orderbook([symbol])
                elif subscription.startswith("trades_"):
                    symbol = subscription.split("_")[1]
                    await self.subscribe_to_trades([symbol])
                elif subscription.startswith("userEvents_"):
                    await self.subscribe_to_user_events()
                elif subscription.startswith("webData2_"):
                    await self.subscribe_to_web_data2()
                elif subscription.startswith("spotState|"):
                    await self.subscribe_to_spot_state(subscription.endswith("|1"))
                elif subscription.startswith("allDexsClearinghouseState_"):
                    await self.subscribe_to_all_dexs_clearinghouse_state()
                elif subscription.startswith("openOrders|"):
                    await self.subscribe_to_open_orders(subscription.split("|", 1)[1])
                elif subscription == "allMids":
                    await self.subscribe_to_all_mids()
                    
            except Exception as e:
                logger.error(f"Error re-establishing subscription {subscription}: {e}")
    
    def get_latest_candle(self, symbol: str) -> Optional[CandleData]:
        """Get latest candle data for symbol."""
        return self.latest_candles.get(symbol)
    
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Get latest order book for symbol."""
        return self.latest_order_books.get(symbol)
    
    def get_latest_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """Get latest trades for symbol."""
        trades = list(self.latest_trades.get(symbol, []))
        return trades[-limit:] if trades else []
    
    def get_latest_mid_price(self, symbol: str) -> Optional[float]:
        """Get latest mid price for symbol."""
        return self.latest_mids.get(symbol)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection and performance statistics."""
        uptime = None
        if self.connection_start_time:
            uptime = (datetime.now() - self.connection_start_time).total_seconds()
        
        return {
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "message_count": self.message_count,
            "last_message_time": self.last_message_time.isoformat(),
            "reconnect_attempts": self.reconnect_attempts,
            "subscriptions": list(self.subscriptions),
            "symbols_tracking": {
                "candles": list(self.latest_candles.keys()),
                "order_books": list(self.latest_order_books.keys()),
                "trades": list(self.latest_trades.keys()),
                "mid_prices": list(self.latest_mids.keys())
            }
        }


# Factory function
def create_hyperliquid_websocket_client(testnet: bool = False, user_address: Optional[str] = None) -> HyperliquidWebSocketClient:
    """Create and return HyperliquidWebSocketClient instance."""
    return HyperliquidWebSocketClient(testnet=testnet, user_address=user_address)
