"""
Hyperliquid Integration Service with Privy Wallet Support

This service integrates Hyperliquid trading with Privy wallet management following
the official Privy guide for Hyperliquid integration.

Features:
- Privy wallet authentication and management
- Hyperliquid trading with official Python SDK
- Real-time market data and position monitoring
- Comprehensive risk management
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from contextvars import ContextVar
import json
from collections import defaultdict, OrderedDict
import uuid
import hashlib

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
# from privy_eth_account import PrivyHTTPClient, PrivyRemoteAccount  # Temporarily disabled due to dependency conflicts

# Import our WebSocket client
from kata.services.hyperliquid_websocket import (
    HyperliquidWebSocketClient, WSMessageType, CandleData, OrderBookData, OrderBookLevel,
    TradeData, FundingData, OpenInterestData, create_hyperliquid_websocket_client,
    is_transient_hyperliquid_error, summarize_hyperliquid_error,
)

# Import bridge service for auto-bridging
from kata.services.bridge_service import get_bridge_service, BridgeStatus

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class MarketData:
    """Market data structure for perpetual futures."""
    symbol: str
    price: float
    mark_price: float
    funding_rate: float
    volume_24h: float
    timestamp: datetime
    open_interest: Optional[float] = None
    funding_premium: Optional[float] = None
    next_funding_time: Optional[datetime] = None


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    margin_used: float
    timestamp: datetime
    mark_price: float = 0.0
    position_value: float = 0.0
    liquidation_price: Optional[float] = None
    leverage: float = 1.0


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    filled_size: Optional[float] = None
    average_price: Optional[float] = None
    fee: Optional[float] = None
    status: Optional[str] = None


@dataclass 
class LiveMarketData:
    """Enhanced market data with real-time feeds."""
    symbol: str
    price: float
    bid: float
    ask: float
    mid_price: float
    volume_24h: float
    funding_rate: float
    open_interest: float
    mark_price: float
    last_trade_price: float
    last_trade_size: float
    last_trade_time: datetime
    timestamp: datetime
    vwap: float = 0.0
    price_change_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0


class HyperliquidService:
    """
    Hyperliquid integration service using Privy wallet management.
    
    This follows the official Privy guide for Hyperliquid integration.
    """
    
    def __init__(self, privy_app_id: Optional[str] = None, privy_app_secret: Optional[str] = None, testnet: Optional[bool] = None,
                 sentiment_service: Optional[Any] = None, sentiment_analyzer: Optional[Any] = None):
        """Initialize Hyperliquid service with Privy integration."""
        from kata.config.settings import settings
        self.privy_app_id = privy_app_id if privy_app_id is not None else getattr(settings, 'PRIVY_APP_ID', '')
        self.privy_app_secret = privy_app_secret if privy_app_secret is not None else getattr(settings, 'PRIVY_APP_SECRET', '')
        self.testnet = testnet if testnet is not None else getattr(settings, 'HYPERLIQUID_TESTNET', False)
        
        # Initialize Privy HTTP client (temporarily disabled due to dependency conflicts)
        self.privy_client = None
        # self.privy_client = PrivyHTTPClient(
        #     app_id=privy_app_id,
        #     app_secret=privy_app_secret
        # )
        
        # Sentiment services for enhanced trading signals
        self.sentiment_service = sentiment_service
        self.sentiment_analyzer = sentiment_analyzer
        
        # Current user's wallet and clients (set via authenticate_user)
        self.user_wallet_address: Optional[str] = None
        self.user_id: Optional[str] = None
        self.info_client: Optional[Info] = None
        self.exchange_client: Optional[Exchange] = None

        # Delegated workers serve many wallets concurrently. A single mutable
        # ``exchange_client`` lets one task replace another task's signer between
        # an authorization check and an order/cancel. Bind delegated clients to
        # the current asyncio context instead, while retaining exchange_client as
        # the legacy browser-authenticated fallback.
        self._active_exchange_client_ctx: ContextVar[Optional[Exchange]] = ContextVar(
            f"hyperliquid_exchange_client_{id(self)}",
            default=None,
        )
        self._active_wallet_address_ctx: ContextVar[Optional[str]] = ContextVar(
            f"hyperliquid_wallet_address_{id(self)}",
            default=None,
        )
        self._active_client_binding_ctx: ContextVar[bool] = ContextVar(
            f"hyperliquid_client_binding_{id(self)}",
            default=False,
        )
        self._delegated_exchange_clients: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._delegated_client_locks: Dict[str, asyncio.Lock] = {}
        try:
            self._delegated_client_cache_size = max(
                8,
                min(
                    4096,
                    int(os.getenv("HYPERLIQUID_DELEGATED_CLIENT_CACHE_SIZE", "512") or "512"),
                ),
            )
        except (TypeError, ValueError):
            self._delegated_client_cache_size = 512
        try:
            self._delegated_client_ttl_seconds = max(
                30,
                min(
                    3600,
                    int(os.getenv("HYPERLIQUID_DELEGATED_CLIENT_TTL_SECONDS", "300") or "300"),
                ),
            )
        except (TypeError, ValueError):
            self._delegated_client_ttl_seconds = 300
        self._exchange_dex_mappings_by_wallet: Dict[str, set] = {}
        self._exchange_mapping_locks_by_wallet: Dict[str, asyncio.Lock] = {}
        self._exchange_request_locks_by_wallet: Dict[str, asyncio.Lock] = {}
        self._legacy_exchange_request_lock: Optional[asyncio.Lock] = None
        
        # Initialize info client immediately (doesn't require authentication)
        try:
            from kata.services.hyperliquid_client_factory import create_info_client
            self.info_client = create_info_client(
                base_url=None if not self.testnet else "https://api.hyperliquid-testnet.xyz",
            )
            logger.info("Info client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize info client: {e}")
            self.info_client = None
        
        # Platform wallet and subaccount management
        self.user_platform_wallets: Dict[str, Dict[str, Any]] = {}  # user_id -> platform_wallet_info
        self.user_subaccounts: Dict[str, Dict[str, Any]] = {}  # user_id -> subaccount_info (for compatibility)
        self.user_allocations: Dict[str, Dict[str, Any]] = {}  # user_id -> agent_allocations
        self.privy_remote_account: Optional[Any] = None  # Privy remote account for signing
        
        # WebSocket client for real-time data
        self.ws_client: Optional[HyperliquidWebSocketClient] = None
        self.is_ws_connected = False
        
        # Real-time data storage
        self.live_market_data: Dict[str, LiveMarketData] = {}
        self.funding_rates: Dict[str, FundingData] = {}
        self.open_interest_data: Dict[str, OpenInterestData] = {}
        
        # Trading state tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order_data
        self.recent_trades: List[Dict[str, Any]] = []
        self.position_updates: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.trade_execution_times: List[float] = []
        self.websocket_latency: List[float] = []
        self.last_data_update = datetime.now()
        
        # Risk management parameters
        self.risk_params = {
            'max_position_size_usd': 50000.0,
            'margin_buffer_percent': 30.0,
            'liquidation_warning_threshold': 0.15,
            'max_correlation_exposure': 0.6,
            'funding_rate_threshold': 0.01,
            'max_daily_funding_cost': 1000.0
        }
        
        # Perp universe cache (symbol -> listing metadata) for execution-time checks
        self._perp_universe: Dict[str, Dict[str, Any]] = {}
        self._perp_universe_fetched_at: Optional[datetime] = None
        self._perp_dex_names: List[str] = [""]
        self._perp_dex_names_fetched_at: Optional[datetime] = None
        self._exchange_dex_mappings: set = {""}
        self._exchange_mapping_lock: Optional[asyncio.Lock] = None
        self._unified_account_wallets: set = set()
        self._user_abstraction_modes: Dict[str, Dict[str, Any]] = {}
        self._builder_fee_approved_wallets: set = set()

        # Per-wallet positions cache – avoids hammering Hyperliquid on every
        # frontend poll. Dashboard polls every 20 s; this 15 s TTL ensures we
        # always serve a recent result while absorbing bursts of back-to-back
        # requests that would otherwise trigger a 429.
        self._positions_cache: Dict[str, Dict[str, Any]] = {}  # wallet -> {positions, fetched_at}
        self._positions_cache_ttl: int = 15  # seconds
        self._position_outage_started_at: Optional[datetime] = None
        self._last_position_outage_warning_at: Optional[datetime] = None
        self._position_outage_failures: int = 0

        logger.info(f"HyperliquidService initialized with Privy integration (testnet: {testnet})")

    # ------------------------------------------------------------------
    # Perp universe (execution-time tradeability checks)
    # ------------------------------------------------------------------

    async def get_perp_dex_names(self, force_refresh: bool = False) -> List[str]:
        """Return the main perp DEX plus every currently registered HIP-3 DEX."""
        cache_age_limit = timedelta(minutes=10)
        if (
            not force_refresh
            and self._perp_dex_names_fetched_at
            and datetime.now() - self._perp_dex_names_fetched_at < cache_age_limit
        ):
            return list(self._perp_dex_names)

        if not self.info_client:
            return list(self._perp_dex_names)

        try:
            payload = await asyncio.get_event_loop().run_in_executor(
                None, self.info_client.perp_dexs
            )
            names = [""]
            for dex in payload or []:
                name = str((dex or {}).get("name") or "").strip() if isinstance(dex, dict) else ""
                if name and name not in names:
                    names.append(name)
            self._perp_dex_names = names
            self._perp_dex_names_fetched_at = datetime.now()
        except Exception as exc:
            logger.warning(f"Could not refresh Hyperliquid perp DEX list: {exc}")

        return list(self._perp_dex_names)

    async def get_enabled_perp_dex_names(self, force_refresh: bool = False) -> List[str]:
        """Return only DEXes Yuki is configured to inspect and trade."""
        registered = await self.get_perp_dex_names(force_refresh=force_refresh)
        from kata.config.settings import settings

        allowed_hip3 = {
            name.strip().lower()
            for name in str(settings.HYPERLIQUID_HIP3_DEX_ALLOWLIST or "xyz").split(",")
            if name.strip()
        }
        enabled = [
            name for name in registered
            if not name or str(name).lower() in allowed_hip3
        ]
        # A transient perpDexs 429 leaves only the main DEX in the registry
        # cache. Still try explicitly configured venues: their own meta/state
        # request validates availability and avoids making every fresh worker
        # process temporarily forget XYZ.
        for name in sorted(allowed_hip3):
            if name not in {str(value).lower() for value in enabled}:
                enabled.append(name)
        return enabled

    @staticmethod
    def _perp_dex_for_symbol(symbol: Optional[str]) -> str:
        """Return the HIP-3 DEX prefix for a canonical symbol, or main DEX ``""``."""
        raw = str(symbol or "").strip()
        if ":" not in raw:
            return ""
        return raw.split(":", 1)[0].lower()

    async def get_perp_universe(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Return the Hyperliquid perp universe keyed by symbol.

        Signals are generated venue-agnostically (Binance pairs shown in the UI);
        this cache lets the execution path skip symbols Hyperliquid cannot fill
        and clamp leverage to each asset's listed maximum.
        """
        cache_age_limit = timedelta(minutes=10)
        if (
            not force_refresh
            and self._perp_universe
            and self._perp_universe_fetched_at
            and datetime.now() - self._perp_universe_fetched_at < cache_age_limit
        ):
            return self._perp_universe

        if not self.info_client:
            return self._perp_universe

        try:
            registered_dex_names = await self.get_enabled_perp_dex_names(
                force_refresh=force_refresh
            )
            # perpDexs contains many abandoned/test deployments. Fetching meta
            # for every one caused a long startup burst and 429s. Yuki currently
            # supports the complete XYZ HIP-3 venue plus main Hyperliquid; more
            # production DEXes can be enabled without code via the allowlist.
            dex_names = registered_dex_names

            def fetch_all_meta():
                results = []
                for dex_name in dex_names:
                    try:
                        results.append((dex_name, self.info_client.meta(dex=dex_name)))
                    except Exception as exc:
                        logger.warning(
                            f"Could not fetch Hyperliquid perp metadata for DEX {dex_name or 'main'}: {exc}"
                        )
                return results

            all_meta = await asyncio.get_event_loop().run_in_executor(None, fetch_all_meta)
            universe = {}
            for dex_name, meta in all_meta:
                for asset in (meta or {}).get("universe", []):
                    canonical_name = str(asset.get("name", "")).strip()
                    lookup_name = canonical_name.upper()
                    if not lookup_name or asset.get("isDelisted"):
                        continue
                    universe[lookup_name] = {
                        "name": canonical_name,
                        "dex": dex_name,
                        "max_leverage": float(asset.get("maxLeverage") or 1),
                        "sz_decimals": asset.get("szDecimals"),
                        "only_isolated": bool(asset.get("onlyIsolated", False)),
                    }
            if universe:
                self._perp_universe = universe
                self._perp_universe_fetched_at = datetime.now()
                logger.info(
                    f"Refreshed Hyperliquid perp universe: {len(universe)} listed assets "
                    f"across {len(all_meta)} DEXs"
                )
        except Exception as e:
            logger.warning(f"Could not refresh Hyperliquid perp universe: {e}")

        return self._perp_universe

    async def is_perp_listed(self, symbol: str) -> bool:
        """Whether a symbol is currently tradeable as a Hyperliquid perp."""
        universe = await self.get_perp_universe()
        if not universe:
            # If the universe cannot be fetched, do not block trading on it.
            return True
        return symbol.upper() in universe

    async def get_perp_sz_decimals(self, symbol: str) -> Optional[int]:
        """Hyperliquid's size decimals for a listed perp, or None if unknown."""
        universe = await self.get_perp_universe()
        entry = universe.get(symbol.upper()) or {}
        sz_decimals = entry.get("sz_decimals")
        return int(sz_decimals) if sz_decimals is not None else None

    @staticmethod
    def _round_size_for_hl(size: float, sz_decimals: Optional[int]) -> float:
        """Floor an order size to the asset's szDecimals so the venue accepts it."""
        import math
        decimals = sz_decimals if sz_decimals is not None else 4
        factor = 10 ** decimals
        return math.floor(float(size) * factor) / factor

    @staticmethod
    def _round_price_for_hl(price: float, sz_decimals: Optional[int]) -> float:
        """
        Round a perp price to Hyperliquid's rules: at most 5 significant figures
        and at most (6 - szDecimals) decimal places.
        """
        import math
        price = float(price)
        if price <= 0:
            return price
        sig_decimals = 5 - int(math.floor(math.log10(abs(price)))) - 1
        max_decimals = max(0, 6 - (sz_decimals if sz_decimals is not None else 0))
        return round(price, min(max(sig_decimals, 0), max_decimals))

    async def get_perp_max_leverage(self, symbol: str) -> Optional[float]:
        """Hyperliquid's maximum leverage for a listed perp, or None if unknown."""
        universe = await self.get_perp_universe()
        entry = universe.get(symbol.upper())
        return entry.get("max_leverage") if entry else None

    @staticmethod
    def closest_supported_leverage(
        requested_leverage: Any,
        venue_max_leverage: Optional[Any] = None,
    ) -> int:
        """Return the closest integer leverage Hyperliquid can accept.

        Hyperliquid supports every integer from 1x through the market's advertised
        maximum. Preserve the generator's request whenever it is in that range,
        otherwise use the nearest venue-supported boundary.
        """
        try:
            requested = float(requested_leverage)
        except (TypeError, ValueError):
            requested = 1.0
        if requested != requested or requested in {float("inf"), float("-inf")}:
            requested = 1.0
        requested = max(1, int(requested + 0.5))

        if venue_max_leverage is None:
            return requested
        try:
            venue_max = float(venue_max_leverage)
        except (TypeError, ValueError):
            return requested
        if venue_max != venue_max or venue_max <= 0:
            return requested
        return min(requested, max(1, int(venue_max)))

    @staticmethod
    def _masked_wallet(wallet_address: Optional[str]) -> str:
        """Return a useful, non-identifying wallet label for routine logs."""
        address = str(wallet_address or "")
        if len(address) <= 12:
            return address or "unknown"
        return f"{address[:8]}…{address[-4:]}"

    def _active_exchange_client(self) -> Optional[Exchange]:
        """Exchange client bound to this task, or the legacy authenticated client."""
        binding_context = getattr(self, "_active_client_binding_ctx", None)
        context = getattr(self, "_active_exchange_client_ctx", None)
        if binding_context is not None and binding_context.get():
            return context.get() if context is not None else None
        return getattr(self, "exchange_client", None)

    def _active_wallet_address(self) -> Optional[str]:
        """Wallet bound to this task, or the legacy authenticated wallet."""
        binding_context = getattr(self, "_active_client_binding_ctx", None)
        context = getattr(self, "_active_wallet_address_ctx", None)
        if binding_context is not None and binding_context.get():
            return context.get() if context is not None else None
        return getattr(self, "user_wallet_address", None)

    def _bind_active_delegated_client(
        self,
        wallet_address: str,
        exchange_client: Exchange,
    ) -> None:
        """Bind a delegated signer without mutating another task's signer."""
        binding_context = getattr(self, "_active_client_binding_ctx", None)
        client_context = getattr(self, "_active_exchange_client_ctx", None)
        wallet_context = getattr(self, "_active_wallet_address_ctx", None)
        if binding_context is not None:
            binding_context.set(True)
        if client_context is not None:
            client_context.set(exchange_client)
        if wallet_context is not None:
            wallet_context.set(wallet_address)

    def _clear_active_delegated_client(self, wallet_address: Optional[str]) -> None:
        """Fail closed while a task is switching or revalidating its signer."""
        binding_context = getattr(self, "_active_client_binding_ctx", None)
        client_context = getattr(self, "_active_exchange_client_ctx", None)
        wallet_context = getattr(self, "_active_wallet_address_ctx", None)
        if binding_context is not None:
            binding_context.set(True)
        if client_context is not None:
            client_context.set(None)
        if wallet_context is not None:
            wallet_context.set(wallet_address)

    def get_active_exchange_client(
        self,
        wallet_address: Optional[str] = None,
    ) -> Optional[Exchange]:
        """Return this task's signer, optionally requiring a wallet match."""
        client = self._active_exchange_client()
        if not wallet_address:
            return client
        active_wallet = self._active_wallet_address()
        if active_wallet and active_wallet.lower() == wallet_address.lower():
            return client
        return None

    def _wallet_exchange_mappings(self) -> set:
        """HIP-3 signing maps loaded on the exchange client in this task."""
        wallet_context = getattr(self, "_active_wallet_address_ctx", None)
        wallet = wallet_context.get() if wallet_context is not None else None
        if not wallet:
            return self._exchange_dex_mappings
        mappings = getattr(self, "_exchange_dex_mappings_by_wallet", None)
        if mappings is None:
            mappings = self._exchange_dex_mappings_by_wallet = {}
        return mappings.setdefault(wallet.lower(), {""})

    def _active_exchange_request_lock(self) -> asyncio.Lock:
        """Serialize SDK writes for one wallet without blocking other wallets."""
        binding_context = getattr(self, "_active_client_binding_ctx", None)
        wallet_context = getattr(self, "_active_wallet_address_ctx", None)
        wallet = (
            wallet_context.get()
            if binding_context is not None and binding_context.get() and wallet_context is not None
            else None
        )
        if wallet:
            locks = getattr(self, "_exchange_request_locks_by_wallet", None)
            if locks is None:
                locks = self._exchange_request_locks_by_wallet = {}
            return locks.setdefault(wallet.lower(), asyncio.Lock())
        lock = getattr(self, "_legacy_exchange_request_lock", None)
        if lock is None:
            lock = self._legacy_exchange_request_lock = asyncio.Lock()
        return lock

    def _evict_delegated_client_cache_if_needed(self) -> None:
        cache = getattr(self, "_delegated_exchange_clients", None)
        if cache is None:
            return
        max_size = int(getattr(self, "_delegated_client_cache_size", 512) or 512)
        while len(cache) > max_size:
            wallet_key, _ = cache.popitem(last=False)
            getattr(self, "_delegated_client_locks", {}).pop(wallet_key, None)
            getattr(self, "_exchange_dex_mappings_by_wallet", {}).pop(wallet_key, None)
            getattr(self, "_exchange_mapping_locks_by_wallet", {}).pop(wallet_key, None)
            getattr(self, "_exchange_request_locks_by_wallet", {}).pop(wallet_key, None)

    async def get_user_abstraction_mode(self, wallet_address: Optional[str] = None) -> Optional[str]:
        """Return Hyperliquid's current account abstraction mode for a wallet."""
        address = wallet_address or self._active_wallet_address()
        if not self.info_client or not address:
            return None

        key = address.lower()
        mode_cache = getattr(self, "_user_abstraction_modes", None)
        if mode_cache is None:
            mode_cache = self._user_abstraction_modes = {}
        cached = mode_cache.get(key)
        if cached and datetime.now() - cached["fetched_at"] < timedelta(minutes=5):
            return cached["mode"]

        try:
            def fetch_mode():
                method = getattr(self.info_client, "query_user_abstraction_state", None)
                if callable(method):
                    return method(address)
                return self.info_client.post(
                    "/info", {"type": "userAbstraction", "user": address}
                )

            mode = await asyncio.get_event_loop().run_in_executor(None, fetch_mode)
            normalized_mode = str(mode) if mode is not None else None
            if normalized_mode:
                mode_cache[key] = {
                    "mode": normalized_mode,
                    "fetched_at": datetime.now(),
                }
            return normalized_mode
        except Exception as exc:
            if cached:
                logger.warning(
                    "Could not refresh Hyperliquid account mode for %s; using cached %s: %s",
                    self._masked_wallet(address),
                    cached["mode"],
                    exc,
                )
                return cached["mode"]
            logger.warning(
                "Could not query Hyperliquid account mode for %s: %s",
                self._masked_wallet(address),
                exc,
            )
            return None

    async def ensure_unified_account_for_hip3(self, wallet_address: Optional[str] = None) -> bool:
        """Enable unified-account collateral handling before trading an HIP-3 market."""
        address = wallet_address or self._active_wallet_address()
        if not address:
            logger.error("Cannot enable Hyperliquid unified account mode without a wallet address")
            return False

        wallet_key = address.lower()
        if wallet_key in self._unified_account_wallets:
            return True

        mode = await self.get_user_abstraction_mode(address)
        if mode in {"unifiedAccount", "portfolioMargin", "default", "disabled", ""}:
            self._unified_account_wallets.add(wallet_key)
            return True

        exchange_client = self._active_exchange_client()
        if not exchange_client:
            logger.error(
                "Cannot enable unified account mode for %s: no trading client",
                self._masked_wallet(address),
            )
            return False

        try:
            setter = getattr(exchange_client, "agent_set_abstraction", None)
            if not callable(setter):
                raise ValueError("Installed Hyperliquid SDK does not support agent_set_abstraction")
            async with self._active_exchange_request_lock():
                response = await asyncio.get_event_loop().run_in_executor(None, lambda: setter("u"))
            if not response or response.get("status") != "ok":
                logger.error(
                    "Hyperliquid rejected unified account mode for %s: %s",
                    self._masked_wallet(address),
                    response,
                )
                return False
            self._unified_account_wallets.add(wallet_key)
            self._user_abstraction_modes[wallet_key] = {
                "mode": "unifiedAccount",
                "fetched_at": datetime.now(),
            }
            logger.info(
                f"Enabled Hyperliquid unified account mode for {address} before HIP-3 execution"
            )
            return True
        except Exception as exc:
            logger.error(
                "Could not enable Hyperliquid unified account mode for %s: %s",
                self._masked_wallet(address),
                exc,
            )
            return False

    async def _ensure_exchange_symbol_mapping(self, symbol: str) -> bool:
        """Load only the HIP-3 DEX needed to sign ``symbol`` on the active client."""
        dex_name = self._perp_dex_for_symbol(symbol)
        if not dex_name:
            return True
        exchange_client = self._active_exchange_client()
        if not exchange_client or not self.info_client:
            return False
        exchange_mappings = self._wallet_exchange_mappings()
        if dex_name in exchange_mappings:
            return True

        wallet_context = getattr(self, "_active_wallet_address_ctx", None)
        wallet = wallet_context.get() if wallet_context is not None else None
        if wallet:
            mapping_locks = getattr(self, "_exchange_mapping_locks_by_wallet", None)
            if mapping_locks is None:
                mapping_locks = self._exchange_mapping_locks_by_wallet = {}
            mapping_lock = mapping_locks.setdefault(wallet.lower(), asyncio.Lock())
        else:
            if self._exchange_mapping_lock is None:
                self._exchange_mapping_lock = asyncio.Lock()
            mapping_lock = self._exchange_mapping_lock

        async with mapping_lock:
            if dex_name in exchange_mappings:
                return True

            dex_names = await self.get_perp_dex_names()
            if dex_name not in dex_names:
                logger.error(f"HIP-3 DEX {dex_name} is not registered on Hyperliquid")
                return False

            # The SDK reserves 10,000 asset ids per builder-deployed DEX, in
            # the order returned by perpDexs (the main DEX is index zero).
            offset = 110000 + (dex_names.index(dex_name) - 1) * 10000
            try:
                meta = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.info_client.meta(dex=dex_name)
                )
                async with self._active_exchange_request_lock():
                    exchange_client.info.set_perp_meta(meta, offset)
                exchange_mappings.add(dex_name)
                logger.info(
                    f"Loaded Hyperliquid HIP-3 signing map for {dex_name} before {symbol} execution"
                )
                return True
            except Exception as exc:
                logger.error(f"Could not load HIP-3 signing map for {symbol}: {exc}")
                return False

    # ------------------------------------------------------------------
    # Delegated trading client
    # ------------------------------------------------------------------

    async def ensure_delegated_trading_client(self, user_id: str, wallet_address: str) -> bool:
        """
        Bind a wallet-scoped Hyperliquid client from the user's delegation.

        Signs orders through Privy's wallet RPC with the platform's delegated
        session-signer key, so no access token or private key is needed. This
        also works from background workers where no browser session exists.

        Clients are cached per wallet and bound through ContextVar. Concurrent
        allocation tasks therefore cannot replace each other's signer. The
        delegation is periodically revalidated so a revoked delegation does not
        remain trusted indefinitely.
        """
        if not user_id or not wallet_address:
            return False

        wallet_key = wallet_address.lower()
        self._clear_active_delegated_client(wallet_address)
        try:
            cache = getattr(self, "_delegated_exchange_clients", None)
            if cache is None:
                cache = self._delegated_exchange_clients = OrderedDict()
            locks = getattr(self, "_delegated_client_locks", None)
            if locks is None:
                locks = self._delegated_client_locks = {}

            now = datetime.now()
            ttl_seconds = int(getattr(self, "_delegated_client_ttl_seconds", 300) or 300)
            cached = cache.get(wallet_key)
            if (
                cached
                and cached.get("user_id") == user_id
                and now - cached["verified_at"] < timedelta(seconds=ttl_seconds)
            ):
                cache.move_to_end(wallet_key)
                self._bind_active_delegated_client(wallet_address, cached["client"])
                return True

            build_lock = locks.setdefault(wallet_key, asyncio.Lock())
            async with build_lock:
                # Another task for the same wallet may have populated the cache
                # while this task waited for the build lock.
                now = datetime.now()
                cached = cache.get(wallet_key)
                if (
                    cached
                    and cached.get("user_id") == user_id
                    and now - cached["verified_at"] < timedelta(seconds=ttl_seconds)
                ):
                    cache.move_to_end(wallet_key)
                    self._bind_active_delegated_client(wallet_address, cached["client"])
                    return True

                from kata.services.delegation_service import get_delegation_service

                delegation = await get_delegation_service().get_user_delegation(user_id)
                if not delegation or not delegation.wallet_id:
                    cache.pop(wallet_key, None)
                    logger.warning(f"No active delegation with wallet id for user {user_id}")
                    return False
                if (
                    delegation.wallet_address
                    and delegation.wallet_address.lower() != wallet_key
                ):
                    cache.pop(wallet_key, None)
                    logger.warning(
                        "Delegated wallet %s does not match requested %s",
                        self._masked_wallet(delegation.wallet_address),
                        self._masked_wallet(wallet_address),
                    )
                    return False

                # Revalidation succeeded and the signer identity is unchanged;
                # retain the existing SDK client and simply refresh its TTL.
                if (
                    cached
                    and cached.get("user_id") == user_id
                    and cached.get("wallet_id") == delegation.wallet_id
                ):
                    cached["verified_at"] = now
                    cache.move_to_end(wallet_key)
                    self._bind_active_delegated_client(wallet_address, cached["client"])
                    return True

                from kata.services.hyperliquid_client_factory import create_exchange_client
                from kata.services.privy_delegated_account import PrivyDelegatedAccount

                account = PrivyDelegatedAccount(
                    wallet_id=delegation.wallet_id,
                    address=wallet_address,
                )
                meta = None
                spot_meta = None
                if self.info_client:
                    try:
                        meta_raw = getattr(self.info_client, "meta", None)
                        res_m = meta_raw() if callable(meta_raw) else meta_raw
                        meta = res_m if isinstance(res_m, dict) else None
                    except Exception:
                        meta = None
                    try:
                        spot_raw = getattr(self.info_client, "spot_meta", None)
                        res_s = spot_raw() if callable(spot_raw) else spot_raw
                        spot_meta = res_s if isinstance(res_s, dict) else None
                    except Exception:
                        spot_meta = None

                exchange_client = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: create_exchange_client(
                        wallet=account,
                        base_url=None if not self.testnet else "https://api.hyperliquid-testnet.xyz",
                        account_address=wallet_address,
                        meta=meta,
                        spot_meta=spot_meta,
                    ),
                )
                cache[wallet_key] = {
                    "client": exchange_client,
                    "user_id": user_id,
                    "wallet_id": delegation.wallet_id,
                    "verified_at": datetime.now(),
                }
                cache.move_to_end(wallet_key)
                mappings_by_wallet = getattr(self, "_exchange_dex_mappings_by_wallet", None)
                if mappings_by_wallet is None:
                    mappings_by_wallet = self._exchange_dex_mappings_by_wallet = {}
                mappings_by_wallet[wallet_key] = {""}
                self._evict_delegated_client_cache_if_needed()
                self._bind_active_delegated_client(wallet_address, exchange_client)
                logger.debug(
                    "Delegated Hyperliquid trading client cached for %s",
                    self._masked_wallet(wallet_address),
                )
                return True
        except Exception as e:
            logger.error(f"Could not build delegated trading client for {user_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Builder codes (platform revenue per order)
    # ------------------------------------------------------------------

    def _builder_info(self) -> Optional[Dict[str, Any]]:
        """Builder payload attached to orders, or None if not configured."""
        from kata.config.settings import settings

        address = settings.HYPERLIQUID_BUILDER_ADDRESS
        fee_tenth_bps = settings.HYPERLIQUID_BUILDER_FEE_TENTH_BPS
        if not address or fee_tenth_bps <= 0:
            return None
        return {"b": address.lower(), "f": int(fee_tenth_bps)}

    async def ensure_builder_fee_approved(self, wallet_address: str) -> bool:
        """
        Make sure the user's account has approved the platform builder fee.

        Approval is signed by the user's wallet (the authenticated exchange
        client), capped at HYPERLIQUID_BUILDER_MAX_FEE_RATE. Returns True when
        orders can carry the builder payload.
        """
        from kata.config.settings import settings

        builder = self._builder_info()
        if not builder or not wallet_address:
            return False

        wallet_key = wallet_address.lower()
        if wallet_key in self._builder_fee_approved_wallets:
            return True

        try:
            if self.info_client:
                current = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.info_client.post(
                        "/info",
                        {"type": "maxBuilderFee", "user": wallet_address, "builder": builder["b"]},
                    ),
                )
                if isinstance(current, (int, float)) and current >= builder["f"]:
                    self._builder_fee_approved_wallets.add(wallet_key)
                    return True
        except Exception as e:
            logger.warning(
                "Could not query builder fee approval for %s: %s",
                self._masked_wallet(wallet_address),
                e,
            )

        exchange_client = self.get_active_exchange_client(wallet_address)
        if not exchange_client:
            logger.info("Builder fee not approved and no authenticated exchange client to approve it")
            return False

        try:
            async with self._active_exchange_request_lock():
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: exchange_client.approve_builder_fee(
                        builder["b"], settings.HYPERLIQUID_BUILDER_MAX_FEE_RATE
                    ),
                )
            if isinstance(response, dict) and response.get("status") == "ok":
                self._builder_fee_approved_wallets.add(wallet_key)
                logger.info(
                    "Builder fee approved for %s (max %s)",
                    self._masked_wallet(wallet_address),
                    settings.HYPERLIQUID_BUILDER_MAX_FEE_RATE,
                )
                return True
            logger.warning(
                "Builder fee approval rejected for %s: %s",
                self._masked_wallet(wallet_address),
                response,
            )
            return False
        except Exception as e:
            logger.warning(
                "Builder fee approval failed for %s: %s",
                self._masked_wallet(wallet_address),
                e,
            )
            return False
        
        # Initialize WebSocket client for live data feeds
        asyncio.create_task(self._initialize_websocket())
    
    def _convert_user_id_to_uuid(self, user_id: str) -> str:
        """
        Convert Privy user ID (string) to a consistent UUID string format.
        
        This creates a deterministic UUID based on the Privy user ID,
        ensuring consistency across database operations.
        
        Args:
            user_id: Privy user ID (string)
            
        Returns:
            UUID string that can be stored in the database
        """
        try:
            # If already a UUID, return as-is
            if self._is_valid_uuid(user_id):
                return user_id
            
            # Create deterministic UUID from Privy user ID
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard namespace UUID
            user_uuid = uuid.uuid5(namespace, user_id)
            return str(user_uuid)
            
        except Exception as e:
            logger.error(f"Error converting user ID to UUID: {e}")
            # Fallback: use SHA-256 hash to create UUID-like string
            hash_obj = hashlib.sha256(user_id.encode())
            hash_hex = hash_obj.hexdigest()
            # Format as UUID
            formatted_uuid = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
            return formatted_uuid
    
    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID format."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False
    
    def _normalize_user_id(self, user_id: str) -> str:
        """
        Normalize user ID for consistent usage.
        
        This ensures that whether we receive a Privy user ID or a database UUID,
        we always work with a consistent format internally.
        
        Args:
            user_id: User ID in any format
            
        Returns:
            Normalized user ID string
        """
        try:
            # Convert to UUID format for database compatibility
            return self._convert_user_id_to_uuid(user_id)
        except Exception as e:
            logger.error(f"Error normalizing user ID {user_id}: {e}")
            return user_id  # Return original if conversion fails
    
    async def authenticate_user(self, access_token: str) -> Dict[str, Any]:
        """
        Authenticate user and setup platform wallet + Hyperliquid subaccount.
        
        Flow:
        1. User connects external wallet (MetaMask) for login
        2. Privy creates platform wallet (managed by user via Privy)
        3. Platform wallet address = Hyperliquid subaccount address
        4. User can fund platform wallet from external wallet or direct deposits
        5. User allocates funds from platform wallet to agents
        
        Args:
            access_token: Privy access token from frontend
            
        Returns:
            User authentication, platform wallet, and subaccount details
        """
        try:
            # Step 1: Verify Privy access token and get user info
            user_info = await self._verify_privy_token(access_token)
            if not user_info:
                raise ValueError("Invalid Privy access token")
            
            # Normalize user ID for database compatibility
            raw_user_id = user_info["user_id"]
            self.user_id = self._normalize_user_id(raw_user_id)
            self.user_wallet_address = user_info["wallet_address"]
            
            logger.info(f"User ID normalized: {raw_user_id} -> {self.user_id}")
            
            # Step 2: Verify info client is available
            if not self.info_client:
                raise ValueError("Info client not available - service initialization failed")
            
            # Step 3: Setup platform wallet and corresponding Hyperliquid subaccount
            platform_wallet_info = await self._setup_platform_wallet_and_subaccount(access_token, user_info)
            
            # Step 4: Initialize trading client with platform wallet/subaccount
            if platform_wallet_info["has_trading_access"]:
                from kata.services.hyperliquid_client_factory import create_exchange_client
                self.exchange_client = create_exchange_client(
                    wallet=self.privy_remote_account,
                    base_url=None if not self.testnet else "https://api.hyperliquid-testnet.xyz",
                    account_address=platform_wallet_info["platform_wallet_address"]
                )
                self._exchange_dex_mappings = {""}
                self._bind_active_delegated_client(
                    platform_wallet_info["platform_wallet_address"],
                    self.exchange_client,
                )
            
            # Step 5: Initialize WebSocket for real-time data
            if not self.is_ws_connected:
                await self._connect_websocket()
            
            logger.info(f"User authenticated with self-custodial subaccount: {self.user_wallet_address}")
            
            return {
                "status": "authenticated",
                "user_id": self.user_id,
                "external_wallet_address": self.user_wallet_address,  # User's connected wallet
                "platform_wallet_address": platform_wallet_info["platform_wallet_address"],  # Privy platform wallet
                "hyperliquid_subaccount_address": platform_wallet_info["platform_wallet_address"],  # Same as platform wallet
                "platform_balance_usdc": platform_wallet_info.get("platform_balance_usdc", 0.0),
                "allocated_to_agents": platform_wallet_info.get("total_allocated", 0.0),
                "available_for_allocation": platform_wallet_info.get("available_balance", 0.0),
                "has_trading_access": platform_wallet_info["has_trading_access"],
                "message": "Platform wallet and Hyperliquid subaccount ready"
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            raise ValueError(f"Authentication failed: {e}")
    
    async def _verify_privy_token(self, access_token: str) -> Dict[str, Any]:
        """Verify Privy access token and extract user information."""
        try:
            # Use Privy client to verify token and get user info (temporarily disabled)
            if self.privy_client is None:
                raise ValueError("Privy client not available - wallet authentication disabled")
            
            user_response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.privy_client.get_user_by_access_token(access_token)
            )
            
            if not user_response or "id" not in user_response:
                raise ValueError("Invalid user response from Privy")
            
            # Extract wallet information
            linked_accounts = user_response.get("linked_accounts", [])
            wallet_account = None
            
            for account in linked_accounts:
                if account.get("type") == "wallet":
                    wallet_account = account
                    break
            
            if not wallet_account:
                raise ValueError("No wallet linked to Privy account")
            
            return {
                "user_id": user_response["id"],
                "wallet_address": wallet_account["address"],
                "wallet_type": wallet_account.get("wallet_client_type", "embedded"),
                "verified": True
            }
            
        except Exception as e:
            logger.error(f"Error verifying Privy token: {e}")
            return None
    
    async def _setup_platform_wallet_and_subaccount(self, access_token: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup platform wallet via Privy and corresponding Hyperliquid subaccount.
        
        Flow:
        1. Privy creates platform wallet (different from external wallet)
        2. Platform wallet address = Hyperliquid subaccount address
        3. User can fund platform wallet from external wallet
        4. User allocates funds from platform wallet to trading agents
        """
        try:
            external_wallet_address = user_info["wallet_address"]  # User's external wallet for login
            user_id = user_info["user_id"]
            
            # Step 1: Check if user already has a platform wallet
            if user_id in self.user_platform_wallets:
                existing_wallet = self.user_platform_wallets[user_id]
                logger.info(f"Using existing platform wallet for user {user_id}")
                return existing_wallet
            
            # Step 2: Create Privy platform wallet (this would be done via Privy API)
            # Create Privy platform wallet using official SDK
            platform_wallet_address = await self._create_privy_platform_wallet(access_token, user_id)
            
            # Step 3: Create Privy remote account for signing (using platform wallet)
            self.privy_remote_account = PrivyRemoteAccount(
                access_token=access_token,
                address=platform_wallet_address
            )
            
            # Step 4: Check platform wallet balance on Hyperliquid
            try:
                account_info = self.info_client.user_state(platform_wallet_address)
                platform_balance_usdc = 0.0
                
                if account_info and "crossMarginSummary" in account_info:
                    platform_balance_usdc = float(account_info["crossMarginSummary"].get("accountValue", 0))
                
            except Exception as balance_error:
                logger.warning(f"Could not fetch platform wallet balance: {balance_error}")
                platform_balance_usdc = 0.0
            
            # Step 5: Initialize allocation tracking
            total_allocated = 0.0
            if user_id in self.user_allocations:
                total_allocated = sum(alloc["amount"] for alloc in self.user_allocations[user_id].values())
            
            available_balance = max(0, platform_balance_usdc - total_allocated)
            
            # Step 6: Store platform wallet information
            platform_wallet_info = {
                "platform_wallet_address": platform_wallet_address,
                "external_wallet_address": external_wallet_address,
                "user_id": user_id,
                "platform_balance_usdc": platform_balance_usdc,
                "total_allocated": total_allocated,
                "available_balance": available_balance,
                "has_trading_access": True,  # Privy provides signing capability
                "created_at": datetime.now(),
                "hyperliquid_subaccount_address": platform_wallet_address  # Same address
            }
            
            self.user_platform_wallets[user_id] = platform_wallet_info
            
            # Also populate user_subaccounts for compatibility with existing API endpoints
            subaccount_info = {
                "user_id": user_id,
                "master_address": external_wallet_address,
                "subaccount_address": platform_wallet_address,
                "self_custodial": True,
                "has_trading_access": True,
                "balance_usdc": platform_balance_usdc,
                "same_address_cross_chain": True,
                "created_at": datetime.now()
            }
            self.user_subaccounts[user_id] = subaccount_info
            
            logger.info(f"Platform wallet setup complete: {platform_wallet_address}")
            logger.info(f"External wallet: {external_wallet_address} → Platform wallet: {platform_wallet_address}")
            
            return platform_wallet_info
            
        except Exception as e:
            logger.error(f"Error setting up platform wallet and subaccount: {e}")
            raise
    
    async def _create_privy_platform_wallet(self, access_token: str, user_id: str) -> str:
        """
        Create a Privy platform wallet for the user using official SDK.
        """
        try:
            # Use Privy SDK to get user's existing wallet or create embedded wallet
            from .privy_auth_service import get_privy_auth_service

            privy_auth = get_privy_auth_service()

            # Verify the access token and get user info
            wallet_info = await privy_auth.verify_privy_token(access_token)

            if wallet_info and wallet_info.wallet_address:
                logger.info(f"Using existing Privy wallet for user {user_id}: {wallet_info.wallet_address}")
                return wallet_info.wallet_address
            else:
                # If no wallet exists, this would typically trigger wallet creation
                # in the frontend, but for now we'll return None
                logger.warning(f"No Privy wallet found for user {user_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error creating platform wallet: {e}")
            raise
    
    async def allocate_funds_to_agent(self, user_id: str, agent_type: str, allocation_amount: float, auto_bridge: bool = True) -> Dict[str, Any]:
        """
        Allocate funds from user's platform wallet to a specific trading agent with auto-bridging.
        
        Enhanced Flow:
        1. Check Base USDC balance in user's platform wallet
        2. If insufficient Hyperliquid balance, auto-bridge from Base to Hyperliquid
        3. Allocate funds to the specified agent for trading
        
        Args:
            user_id: User ID
            agent_type: Type of agent (e.g., 'yuki')
            allocation_amount: Amount in USDC to allocate
            auto_bridge: Whether to automatically bridge Base USDC to Hyperliquid if needed
            
        Returns:
            Allocation details, bridging status, and updated balances
        """
        try:
            # Normalize user ID for consistent lookup
            normalized_user_id = self._normalize_user_id(user_id)
            
            # Check if user has platform wallet
            if normalized_user_id not in self.user_platform_wallets:
                raise ValueError("User platform wallet not found - authenticate first")
            
            platform_wallet = self.user_platform_wallets[normalized_user_id]
            wallet_address = platform_wallet["platform_wallet_address"]
            
            # Get current allocations
            if normalized_user_id not in self.user_allocations:
                self.user_allocations[normalized_user_id] = {}
            
            current_allocations = self.user_allocations[normalized_user_id]
            total_allocated = sum(alloc["amount"] for alloc in current_allocations.values())
            
            # Check minimum allocation (disabled for testing)
            # if allocation_amount < 1.0:
            #     raise ValueError("Minimum allocation is $1 USDC")
            
            # Check total platform balance (Base + Hyperliquid)
            base_balance = platform_wallet.get("base_balance_usdc", 0.0)
            hyperliquid_balance = platform_wallet.get("hyperliquid_balance_usdc", 0.0)
            total_balance = base_balance + hyperliquid_balance
            available_balance = total_balance - total_allocated
            
            if allocation_amount > available_balance:
                raise ValueError(f"Insufficient total funds. Available: ${available_balance:.2f}, Requested: ${allocation_amount:.2f}")
            
            bridge_result = None
            
            # Check if we need to bridge funds for Hyperliquid trading
            if auto_bridge and hyperliquid_balance < allocation_amount:
                needed_amount = allocation_amount - hyperliquid_balance
                
                # Check if we have enough Base USDC to bridge
                if base_balance >= needed_amount:
                    logger.info(f"Auto-bridging ${needed_amount:.2f} from Base to Hyperliquid for allocation")
                    
                    # Get bridge service
                    bridge_service = get_bridge_service(testnet=self.testnet)
                    
                    # Start bridge operation
                    bridge_transaction = await bridge_service.bridge_base_to_hyperliquid(
                        user_id=user_id,
                        wallet_address=wallet_address,
                        amount=needed_amount
                    )
                    
                    bridge_result = {
                        "bridge_id": bridge_transaction.id,
                        "bridge_status": bridge_transaction.status.value,
                        "bridged_amount": needed_amount,
                        "estimated_time": bridge_transaction.estimated_time or 90,
                        "bridge_provider": bridge_transaction.bridge_provider
                    }
                    
                    # Update balances after successful bridge
                    if bridge_transaction.status == BridgeStatus.COMPLETED:
                        platform_wallet["base_balance_usdc"] -= needed_amount
                        platform_wallet["hyperliquid_balance_usdc"] += needed_amount
                        logger.info(f"Bridge completed: ${needed_amount} now available on Hyperliquid")
                    elif bridge_transaction.status == BridgeStatus.FAILED:
                        raise ValueError(f"Bridge failed: {bridge_transaction.error_message}")
                    else:
                        # Bridge is in progress - update balances optimistically
                        platform_wallet["base_balance_usdc"] -= needed_amount
                        platform_wallet["hyperliquid_balance_usdc"] += needed_amount
                        logger.info(f"Bridge initiated: ${needed_amount} being bridged to Hyperliquid")
                else:
                    raise ValueError(f"Insufficient Base USDC for bridging. Need ${needed_amount:.2f}, have ${base_balance:.2f}")
            
            # Create allocation record
            allocation_id = f"{agent_type}_{normalized_user_id}"
            allocation_info = {
                "agent_type": agent_type,
                "amount": allocation_amount,
                "allocated_at": datetime.now(),
                "status": "active",
                "bridge_transaction": bridge_result["bridge_id"] if bridge_result else None,
                "funding_source": "hyperliquid"
            }
            
            current_allocations[allocation_id] = allocation_info
            
            # Update platform wallet allocation tracking
            new_total_allocated = sum(alloc["amount"] for alloc in current_allocations.values())
            new_available_balance = (base_balance + hyperliquid_balance) - new_total_allocated
            
            platform_wallet["total_allocated"] = new_total_allocated
            platform_wallet["available_balance"] = new_available_balance
            
            logger.info(f"Allocated ${allocation_amount} to {agent_type} agent for user {normalized_user_id}")
            
            # Prepare response
            response = {
                "success": True,
                "allocation_id": allocation_id,
                "agent_type": agent_type,
                "allocated_amount": allocation_amount,
                "total_allocated": new_total_allocated,
                "available_balance": new_available_balance,
                "platform_balances": {
                    "base_usdc": platform_wallet.get("base_balance_usdc", 0.0),
                    "hyperliquid_usdc": platform_wallet.get("hyperliquid_balance_usdc", 0.0),
                    "total_usdc": platform_wallet.get("base_balance_usdc", 0.0) + platform_wallet.get("hyperliquid_balance_usdc", 0.0)
                },
                "message": f"Successfully allocated ${allocation_amount} to {agent_type} agent",
                "user_id": normalized_user_id,
                "bridge_info": bridge_result
            }
            
            if bridge_result:
                if bridge_result["bridge_status"] in ["completed"]:
                    response["message"] += f" (bridged ${bridge_result['bridged_amount']:.2f} from Base)"
                elif bridge_result["bridge_status"] in ["pending", "bridging", "depositing"]:
                    response["message"] += f" (bridging ${bridge_result['bridged_amount']:.2f} from Base - ETA: {bridge_result['estimated_time']}s)"
            
            return response
            
        except Exception as e:
            logger.error(f"Error allocating funds to agent: {e}")
            return {"success": False, "error": str(e)}
    
    async def deallocate_funds_from_agent(self, user_id: str, agent_type: str) -> Dict[str, Any]:
        """
        Deallocate funds from a trading agent back to platform wallet.
        
        Args:
            user_id: User ID
            agent_type: Type of agent to deallocate from
            
        Returns:
            Deallocation details and updated balances
        """
        try:
            # Normalize user ID for consistent lookup
            normalized_user_id = self._normalize_user_id(user_id)
            
            if normalized_user_id not in self.user_platform_wallets:
                raise ValueError("User platform wallet not found")
            
            if normalized_user_id not in self.user_allocations:
                raise ValueError("No allocations found for user")
            
            allocation_id = f"{agent_type}_{normalized_user_id}"
            current_allocations = self.user_allocations[normalized_user_id]
            
            if allocation_id not in current_allocations:
                raise ValueError(f"No allocation found for {agent_type} agent")
            
            # Get allocation amount
            deallocated_amount = current_allocations[allocation_id]["amount"]
            
            # Remove allocation
            del current_allocations[allocation_id]
            
            # Update platform wallet balances
            platform_wallet = self.user_platform_wallets[normalized_user_id]
            new_total_allocated = sum(alloc["amount"] for alloc in current_allocations.values())
            new_available_balance = platform_wallet["platform_balance_usdc"] - new_total_allocated
            
            platform_wallet["total_allocated"] = new_total_allocated
            platform_wallet["available_balance"] = new_available_balance
            
            logger.info(f"Deallocated ${deallocated_amount} from {agent_type} agent for user {normalized_user_id}")
            
            return {
                "success": True,
                "deallocated_amount": deallocated_amount,
                "total_allocated": new_total_allocated,
                "available_balance": new_available_balance,
                "message": f"Successfully deallocated ${deallocated_amount} from {agent_type} agent"
            }
            
        except Exception as e:
            logger.error(f"Error deallocating funds from agent: {e}")
            return {"success": False, "error": str(e)}
    
    def get_allocation_status(self, user_id: str) -> Dict[str, Any]:
        """Get current allocation status for user."""
        try:
            # Normalize user ID for consistent lookup
            normalized_user_id = self._normalize_user_id(user_id)
            
            if normalized_user_id not in self.user_platform_wallets:
                return {"error": "User platform wallet not found"}
            
            platform_wallet = self.user_platform_wallets[normalized_user_id]
            allocations = self.user_allocations.get(normalized_user_id, {})
            
            return {
                "platform_wallet_address": platform_wallet["platform_wallet_address"],
                "external_wallet_address": platform_wallet["external_wallet_address"],
                "total_balance": platform_wallet["platform_balance_usdc"],
                "total_allocated": platform_wallet["total_allocated"],
                "available_balance": platform_wallet["available_balance"],
                "allocations": {
                    agent_id: {
                        "agent_type": alloc["agent_type"],
                        "amount": alloc["amount"],
                        "allocated_at": alloc["allocated_at"].isoformat(),
                        "status": alloc["status"]
                    }
                    for agent_id, alloc in allocations.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting allocation status: {e}")
            return {"error": str(e)}
    
    async def get_cross_chain_balance_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get balance status across Base and Hyperliquid for same address.
        
        This shows the user's funds on both chains using the same wallet address.
        """
        try:
            # Normalize user ID for consistent lookup
            normalized_user_id = self._normalize_user_id(user_id)
            
            if normalized_user_id not in self.user_subaccounts:
                raise ValueError("User not authenticated or subaccount not setup")
            
            subaccount_info = self.user_subaccounts[normalized_user_id]
            wallet_address = subaccount_info["subaccount_address"]
            
            # Get Hyperliquid balance
            hyperliquid_balance = 0.0
            try:
                account_info = self.info_client.user_state(wallet_address)
                if account_info and "crossMarginSummary" in account_info:
                    hyperliquid_balance = float(account_info["crossMarginSummary"].get("accountValue", 0))
            except Exception as e:
                logger.warning(f"Could not fetch Hyperliquid balance: {e}")
            
            # Note: Base balance would need to be fetched via Base RPC
            # For now, indicating that it should be checked
            
            return {
                "wallet_address": wallet_address,
                "same_address_both_chains": True,
                "balances": {
                    "hyperliquid_usdc": hyperliquid_balance,
                    "base_usdc": "check_via_base_rpc",  # Would implement Base RPC call
                },
                "funding_instructions": {
                    "method": "direct_deposit",
                    "description": "Send USDC directly to your wallet address on Hyperliquid",
                    "address": wallet_address,
                    "network": "Hyperliquid",
                    "no_bridge_needed": True
                },
                "self_custodial": True,
                "ready_for_trading": hyperliquid_balance > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting cross-chain balance status: {e}")
            return {"error": str(e)}
    
    async def check_funding_readiness(self, user_id: str) -> Dict[str, Any]:
        """
        Check if user is ready to start trading (has funds on Hyperliquid).
        """
        try:
            balance_status = await self.get_cross_chain_balance_status(user_id)
            
            if "error" in balance_status:
                return balance_status
            
            hyperliquid_balance = balance_status["balances"]["hyperliquid_usdc"]
            min_trading_balance = 10.0  # Minimum $10 to start trading
            
            return {
                "ready_for_trading": hyperliquid_balance >= min_trading_balance,
                "current_balance": hyperliquid_balance,
                "minimum_required": min_trading_balance,
                "same_address_funding": True,
                "funding_method": "direct_deposit",
                "next_steps": (
                    "Ready to start trading!" if hyperliquid_balance >= min_trading_balance
                    else f"Deposit at least ${min_trading_balance} USDC to your Hyperliquid address: {balance_status['wallet_address']}"
                )
            }
            
        except Exception as e:
            logger.error(f"Error checking funding readiness: {e}")
            return {"error": str(e)}
    
    def get_funding_instructions(self, user_id: str) -> Dict[str, Any]:
        """
        Get funding instructions for user's self-custodial subaccount.
        
        With Privy-Hyperliquid partnership, funding is simple:
        1. Same address works on both Base and Hyperliquid
        2. User can deposit directly to their address on Hyperliquid
        3. No bridging required
        """
        try:
            # Normalize user ID for consistent lookup
            normalized_user_id = self._normalize_user_id(user_id)
            
            if normalized_user_id not in self.user_subaccounts:
                raise ValueError("User not authenticated")
            
            subaccount_info = self.user_subaccounts[normalized_user_id]
            wallet_address = subaccount_info["subaccount_address"]
            
            return {
                "method": "direct_deposit",
                "title": "Fund Your Self-Custodial Trading Account",
                "description": "Your Privy wallet works on both Base and Hyperliquid with the same address",
                "wallet_address": wallet_address,
                "steps": [
                    {
                        "step": 1,
                        "title": "Get Your Address",
                        "description": f"Your trading address: {wallet_address}",
                        "note": "Same address works on Base and Hyperliquid!"
                    },
                    {
                        "step": 2,
                        "title": "Deposit USDC",
                        "description": "Send USDC directly to your address on Hyperliquid",
                        "options": [
                            "Direct deposit from exchange (Coinbase, Binance, etc.)",
                            "Transfer from another Hyperliquid account",
                            "Use Hyperliquid's official bridge from other chains"
                        ]
                    },
                    {
                        "step": 3,
                        "title": "Start Trading",
                        "description": "Once funds arrive, your Yuki agent can start trading",
                        "minimum_amount": "$10 USDC"
                    }
                ],
                "benefits": [
                    "✅ Self-custodial - you control your keys via Privy",
                    "✅ Same address on Base and Hyperliquid",
                    "✅ No bridging required for most deposits",
                    "✅ Individual isolated subaccount",
                    "✅ Real-time balance monitoring"
                ],
                "self_custodial": True,
                "no_bridge_needed": True
            }
            
        except Exception as e:
            logger.error(f"Error getting funding instructions: {e}")
            return {"error": str(e)}
    
    async def _initialize_websocket(self):
        """Initialize WebSocket client for real-time data."""
        try:
            if not self.ws_client:
                self.ws_client = create_hyperliquid_websocket_client(
                    testnet=self.testnet,
                    user_address=self.user_wallet_address
                )
                
                # Setup callbacks
                self._setup_websocket_callbacks()
                
                logger.info("WebSocket client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket client: {e}")
    
    async def _connect_websocket(self, market_symbols: Optional[List[str]] = None):
        """Connect to WebSocket and subscribe to data feeds."""
        try:
            if not self.ws_client:
                await self._initialize_websocket()

            if self.ws_client and self.user_wallet_address:
                self.ws_client.user_address = self.user_wallet_address

            if self.ws_client and (not self.is_ws_connected or not self.ws_client.is_connected):
                # Connect to WebSocket
                connected = await self.ws_client.connect()
                
                if connected:
                    self.is_ws_connected = True
                    await self._subscribe_websocket_feeds(market_symbols)
                    
                    # Start listening for messages (one recv loop per client;
                    # ensure_listening() is idempotent and won't stack duplicates)
                    self.ws_client.ensure_listening()
                    logger.info("Connected to WebSocket and subscribed to essential feeds")
                else:
                    logger.error("Failed to connect to WebSocket")
            elif self.ws_client and self.is_ws_connected:
                await self._subscribe_websocket_feeds(market_symbols)
                    
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")

    async def _subscribe_websocket_feeds(self, market_symbols: Optional[List[str]] = None):
        """Subscribe to lean default feeds plus optional targeted detail streams."""
        if not self.ws_client or not self.ws_client.is_connected:
            return

        from kata.config.settings import settings

        subscriptions = self.ws_client.subscriptions
        if "allMids" not in subscriptions:
            await self.ws_client.subscribe_to_all_mids()

        if self.user_wallet_address and f"userEvents_{self.user_wallet_address}" not in subscriptions:
            await self.ws_client.subscribe_to_user_events()

        detail_streams_enabled = bool(
            getattr(settings, "HYPERLIQUID_WS_DETAIL_STREAMS_ENABLED", False)
        )
        detail_symbols = []
        for symbol in market_symbols or []:
            normalized_symbol = str(symbol or "").strip().upper()
            if normalized_symbol and normalized_symbol not in detail_symbols:
                detail_symbols.append(normalized_symbol)

        if not detail_streams_enabled:
            if detail_symbols:
                logger.debug(
                    "Hyperliquid detail streams disabled; using allMids/REST for "
                    f"{', '.join(detail_symbols)}"
                )
            return

        if not detail_symbols:
            logger.debug("Hyperliquid detail streams enabled but no target symbols were requested")
            return

        candle_symbols = [
            symbol for symbol in detail_symbols
            if f"candle_{symbol}_1m" not in subscriptions
        ]
        if candle_symbols:
            await self.ws_client.subscribe_to_candles(candle_symbols, "1m")

        orderbook_symbols = [
            symbol for symbol in detail_symbols
            if f"l2Book_{symbol}" not in subscriptions
        ]
        if orderbook_symbols:
            await self.ws_client.subscribe_to_orderbook(orderbook_symbols)

        trade_symbols = [
            symbol for symbol in detail_symbols
            if f"trades_{symbol}" not in subscriptions
        ]
        if trade_symbols:
            await self.ws_client.subscribe_to_trades(trade_symbols)
    
    def _setup_websocket_callbacks(self):
        """Setup WebSocket message callbacks."""
        if not self.ws_client:
            return
        
        # Candle data callback
        self.ws_client.add_callback(WSMessageType.CANDLE, self._on_candle_update)
        
        # Order book callback
        self.ws_client.add_callback(WSMessageType.L2_BOOK, self._on_orderbook_update)
        
        # Trades callback
        self.ws_client.add_callback(WSMessageType.TRADES, self._on_trades_update)
        
        # All mids callback
        self.ws_client.add_callback(WSMessageType.ALL_MIDS, self._on_all_mids_update)
        
        # User events callback
        self.ws_client.add_callback(WSMessageType.USER_EVENTS, self._on_user_event)
        
        logger.debug("WebSocket callbacks configured")
    
    async def _on_candle_update(self, candle: CandleData):
        """Handle real-time candle updates."""
        try:
            # Update live market data
            symbol = candle.symbol
            
            if symbol in self.live_market_data:
                market_data = self.live_market_data[symbol]
                market_data.price = candle.close
                market_data.vwap = candle.vwap
                market_data.volume_24h = candle.volume
                market_data.timestamp = candle.timestamp
            else:
                # Create new market data entry
                self.live_market_data[symbol] = LiveMarketData(
                    symbol=symbol,
                    price=candle.close,
                    bid=candle.close,  # Will be updated by order book
                    ask=candle.close,  # Will be updated by order book
                    mid_price=candle.close,
                    volume_24h=candle.volume,
                    funding_rate=0.0,  # Will be updated by funding data
                    open_interest=0.0,  # Will be updated by OI data
                    mark_price=candle.close,
                    last_trade_price=candle.close,
                    last_trade_size=0.0,
                    last_trade_time=candle.timestamp,
                    timestamp=candle.timestamp,
                    vwap=candle.vwap,
                    price_change_24h=0.0,
                    high_24h=candle.high,
                    low_24h=candle.low
                )
            
            self.last_data_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing candle update: {e}")
    
    async def _on_orderbook_update(self, order_book: OrderBookData):
        """Handle real-time order book updates."""
        try:
            symbol = order_book.symbol
            
            if symbol in self.live_market_data:
                market_data = self.live_market_data[symbol]
                
                # Update bid/ask from order book
                if order_book.bids:
                    market_data.bid = order_book.bids[0].price
                if order_book.asks:
                    market_data.ask = order_book.asks[0].price
                
                market_data.mid_price = order_book.mid_price
                market_data.timestamp = order_book.timestamp
            
            self.last_data_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing order book update: {e}")
    
    async def _on_trades_update(self, symbol: str, trades: List[TradeData]):
        """Handle real-time trades updates."""
        try:
            if trades and symbol in self.live_market_data:
                latest_trade = trades[-1]
                market_data = self.live_market_data[symbol]
                
                market_data.last_trade_price = latest_trade.price
                market_data.last_trade_size = latest_trade.size
                market_data.last_trade_time = latest_trade.timestamp
                market_data.timestamp = latest_trade.timestamp
            
            self.last_data_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing trades update: {e}")
    
    async def _on_all_mids_update(self, mids: Dict[str, float]):
        """Handle all mids price updates."""
        try:
            now = datetime.now()
            for symbol, price in mids.items():
                price = float(price)
                if symbol in self.live_market_data:
                    self.live_market_data[symbol].mid_price = price
                    self.live_market_data[symbol].price = price
                    self.live_market_data[symbol].timestamp = now
                else:
                    # allMids streams every listed perp - create entries so live
                    # data covers the full universe, not just the subscribed
                    # major symbols. Depth fields refine later if subscribed.
                    self.live_market_data[symbol] = LiveMarketData(
                        symbol=symbol,
                        price=price,
                        bid=price,
                        ask=price,
                        mid_price=price,
                        volume_24h=0.0,
                        funding_rate=0.0,
                        open_interest=0.0,
                        mark_price=price,
                        last_trade_price=price,
                        last_trade_size=0.0,
                        last_trade_time=now,
                        timestamp=now,
                    )

            self.last_data_update = datetime.now()

        except Exception as e:
            logger.error(f"Error processing mids update: {e}")
    
    async def _on_user_event(self, event):
        """Handle user-specific events."""
        try:
            # Log position updates, fills, etc.
            if event.event_type in ['fill', 'liquidation', 'funding']:
                self.position_updates.append({
                    'type': event.event_type,
                    'symbol': event.symbol,
                    'timestamp': event.timestamp,
                    'data': event.data
                })
                
                # Keep only recent updates
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.position_updates = [
                    update for update in self.position_updates 
                    if update['timestamp'] > cutoff_time
                ]
            
            logger.debug(f"User event: {event.event_type} for {event.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing user event: {e}")
    
    async def initialize_market_data_client(self) -> bool:
        """Initialize market data client without authentication."""
        try:
            from kata.services.hyperliquid_client_factory import create_info_client
            self.info_client = create_info_client(
                base_url=None if not self.testnet else "https://api.hyperliquid-testnet.xyz",
            )
            logger.info(f"Hyperliquid market data client initialized (testnet: {self.testnet})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize market data client: {e}")
            return False
    
    async def get_live_market_data(self, symbol: str) -> Optional[LiveMarketData]:
        """
        Get real-time market data, preferring WebSocket feeds with a REST fallback.

        Worker/API processes execute trades without a subscribed market-data
        WebSocket, so a cache miss falls back to the Hyperliquid info API. A REST
        mid price fetched at execution time is equally fresh for order placement.

        Args:
            symbol: Symbol to get data for

        Returns:
            LiveMarketData from the live feed or REST snapshot, None if unavailable
        """
        try:
            if symbol in self.live_market_data:
                return self.live_market_data[symbol]

            rest_data = self._fetch_rest_market_snapshot(symbol)
            if rest_data:
                logger.info(f"Using REST market snapshot for {symbol}: mid {rest_data.mid_price}")
                return rest_data

            logger.warning(f"No live market data available for {symbol} - WebSocket feed required")
            return None

        except Exception as e:
            logger.error(f"Error getting live market data for {symbol}: {e}")
            return None

    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookData]:
        """
        Return the latest L2 book for ``symbol`` from WebSocket cache or REST.

        The Yuki research / execution paths call this method directly. Some worker
        processes do not have a subscribed market-data websocket, so a REST
        snapshot keeps those code paths functional instead of raising an
        ``AttributeError`` or requiring live subscriptions.
        """
        try:
            normalized_symbol = str(symbol or "").strip()
            if not normalized_symbol:
                return None

            ws_client = getattr(self, "ws_client", None)
            if ws_client:
                cached = ws_client.get_latest_orderbook(normalized_symbol)
                if not cached:
                    cached = next(
                        (
                            book for coin, book in ws_client.latest_order_books.items()
                            if str(coin).upper() == normalized_symbol.upper()
                        ),
                        None,
                    )
                if cached:
                    return cached

            if not self.info_client:
                return None

            universe = await self.get_perp_universe()
            canonical_symbol = normalized_symbol
            if universe:
                entry = universe.get(normalized_symbol.upper())
                if entry and entry.get("name"):
                    canonical_symbol = str(entry["name"])

            def fetch_book():
                return self.info_client.post(
                    "/info",
                    {"type": "l2Book", "coin": canonical_symbol},
                )

            payload = await asyncio.get_event_loop().run_in_executor(None, fetch_book)
            levels = (payload or {}).get("levels") or []
            raw_bids = levels[0] if len(levels) > 0 else []
            raw_asks = levels[1] if len(levels) > 1 else []

            def _to_level(row: Any) -> Optional[OrderBookLevel]:
                try:
                    if isinstance(row, dict):
                        price = float(row.get("px") or 0.0)
                        size = float(row.get("sz") or 0.0)
                    else:
                        price = float(row[0])
                        size = float(row[1])
                    if price <= 0 or size <= 0:
                        return None
                    return OrderBookLevel(price=price, size=size)
                except (IndexError, TypeError, ValueError):
                    return None

            bids = [level for level in (_to_level(row) for row in raw_bids[:limit]) if level]
            asks = [level for level in (_to_level(row) for row in raw_asks[:limit]) if level]

            mid_price = 0.0
            if bids and asks:
                mid_price = (bids[0].price + asks[0].price) / 2
            elif bids:
                mid_price = bids[0].price
            elif asks:
                mid_price = asks[0].price

            if not bids and not asks:
                return None

            return OrderBookData(
                symbol=canonical_symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                mid_price=mid_price,
            )
        except Exception as e:
            logger.warning(f"Could not fetch Hyperliquid order book for {symbol}: {e}")
            return None

    def _fetch_rest_market_snapshot(self, symbol: str) -> Optional[LiveMarketData]:
        """Build a LiveMarketData snapshot from the Hyperliquid info API mid price."""
        try:
            if not self.info_client:
                return None

            dex_name = self._perp_dex_for_symbol(symbol)
            mids = self.info_client.all_mids(dex=dex_name)
            raw_mid = mids.get(symbol) if isinstance(mids, dict) else None
            if raw_mid is None and isinstance(mids, dict):
                raw_mid = next(
                    (value for coin, value in mids.items() if str(coin).upper() == str(symbol).upper()),
                    None,
                )
            mid_price = float(raw_mid) if raw_mid is not None else 0.0
            if mid_price <= 0:
                return None

            now = datetime.now()
            return LiveMarketData(
                symbol=symbol,
                price=mid_price,
                bid=mid_price,
                ask=mid_price,
                mid_price=mid_price,
                volume_24h=0.0,
                funding_rate=0.0,
                open_interest=0.0,
                mark_price=mid_price,
                last_trade_price=mid_price,
                last_trade_size=0.0,
                last_trade_time=now,
                timestamp=now,
            )
        except Exception as e:
            logger.warning(f"REST market snapshot failed for {symbol}: {e}")
            return None
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """
        Fetch real-time market data for a symbol.
        
        Args:
            symbol: Symbol to fetch data for (e.g., 'BTC')
            
        Returns:
            MarketData object with market information
        """
        try:
            if not self.info_client:
                raise ValueError("User not authenticated. Call authenticate_user() first.")
            
            # Get market data from Hyperliquid
            dex_name = self._perp_dex_for_symbol(symbol)
            all_mids = self.info_client.all_mids(dex=dex_name)
            meta = self.info_client.meta(dex=dex_name)
            
            # Find the symbol in the universe
            universe = meta['universe']
            canonical_symbol = symbol
            for asset in universe:
                if str(asset.get('name', '')).upper() == str(symbol).upper():
                    canonical_symbol = asset['name']
                    break

            if canonical_symbol not in all_mids:
                raise ValueError(f"Symbol {symbol} not found")
            
            # Get funding rate
            funding_rates = self.info_client.funding_history(canonical_symbol, startTime=0, endTime=None)
            current_funding = funding_rates[-1]['fundingRate'] if funding_rates else 0.0
            
            # Get 24h volume
            candles = self.info_client.candles_snapshot(canonical_symbol, "1d", 1)
            volume_24h = float(candles[0]['v']) if candles else 0.0
            
            return MarketData(
                symbol=canonical_symbol,
                price=float(all_mids[canonical_symbol]),
                mark_price=float(all_mids[canonical_symbol]),  # Simplified
                funding_rate=float(current_funding),
                volume_24h=volume_24h,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            raise
    
    async def cancel_order(self, symbol: str, order_id: Any) -> bool:
        """Cancel a resting order by exchange order id."""
        try:
            exchange_client = self._active_exchange_client()
            if not exchange_client:
                raise ValueError("No trading client available to cancel orders")
            if not await self._ensure_exchange_symbol_mapping(symbol):
                raise ValueError(f"No Hyperliquid signing map available for {symbol}")

            async with self._active_exchange_request_lock():
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: exchange_client.cancel(symbol, int(order_id))
                )
            ok = bool(response) and response.get('status') == 'ok'
            if ok:
                logger.info(f"Cancelled order {order_id} for {symbol}")
            else:
                logger.warning(f"Cancel order {order_id} for {symbol} returned: {response}")
            return ok
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            return False

    async def get_open_order_ids(self, wallet_address: Optional[str] = None) -> Optional[set]:
        """Open order ids for a wallet (default: active wallet), or None if they cannot be fetched."""
        open_orders = await self.get_open_orders(wallet_address)
        if open_orders is None:
            return None
        return {str(order.get('oid')) for order in open_orders if order.get('oid') is not None}

    async def get_open_orders(self, wallet_address: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Return the wallet's live Hyperliquid orders, preserving failure vs. an empty book."""
        try:
            if not self.info_client:
                raise ValueError("Hyperliquid info client is not initialized")
            address = wallet_address or self._active_wallet_address()
            if not address:
                raise ValueError("No wallet address available")
            dex_names = await self.get_enabled_perp_dex_names()

            def fetch_all_orders():
                orders = []
                seen = set()
                for dex_name in dex_names:
                    for order in self.info_client.open_orders(address, dex=dex_name) or []:
                        order_id = str(order.get("oid"))
                        if order_id in seen:
                            continue
                        seen.add(order_id)
                        orders.append(order)
                return orders

            return await asyncio.get_event_loop().run_in_executor(None, fetch_all_orders)
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")
            return None

    async def get_positions(self, wallet_address: Optional[str] = None) -> Dict[str, Position]:
        """
        Fetch current open positions.

        Args:
            wallet_address: Wallet to query; defaults to the authenticated user's wallet.

        Returns:
            Dictionary of positions keyed by symbol (empty on error)
        """
        try:
            positions = await self._fetch_positions(wallet_address or self._active_wallet_address())
            self._record_position_fetch_success()
            return positions
        except Exception as e:
            self._record_position_fetch_failure(e, wallet_address or self._active_wallet_address())
            return {}

    async def get_positions_for_wallet(self, wallet_address: str) -> Optional[Dict[str, Position]]:
        """
        Fetch open positions for a specific wallet, distinguishing failure from empty.

        Returns:
            Dictionary of positions keyed by symbol, or None when the fetch failed
            (so callers don't mistake an API error for "all positions closed").
        """
        try:
            positions = await self._fetch_positions(wallet_address)
            self._record_position_fetch_success()
            return positions
        except Exception as e:
            self._record_position_fetch_failure(e, wallet_address)
            return None

    def _record_position_fetch_failure(
        self,
        error: Exception,
        wallet_address: Optional[str],
    ) -> None:
        """Rate-limit expected outage logs and strip gateway HTML responses."""
        if not is_transient_hyperliquid_error(error):
            logger.error(
                "Error fetching Hyperliquid positions for %s: %s",
                str(wallet_address or "unknown")[:10],
                summarize_hyperliquid_error(error),
            )
            return

        now = datetime.now()
        started_at = getattr(self, "_position_outage_started_at", None)
        last_warning_at = getattr(self, "_last_position_outage_warning_at", None)
        self._position_outage_failures = int(
            getattr(self, "_position_outage_failures", 0) or 0
        ) + 1
        if started_at is None:
            self._position_outage_started_at = now

        if last_warning_at is None or (now - last_warning_at).total_seconds() >= 300:
            logger.warning(
                "Hyperliquid positions temporarily unavailable (%s); safety cycle will retry",
                summarize_hyperliquid_error(error),
            )
            self._last_position_outage_warning_at = now
        else:
            logger.debug(
                "Hyperliquid position retry deferred after %s",
                summarize_hyperliquid_error(error),
            )

    def _record_position_fetch_success(self) -> None:
        started_at = getattr(self, "_position_outage_started_at", None)
        if started_at is not None:
            downtime = max(0.0, (datetime.now() - started_at).total_seconds())
            logger.info(
                "Hyperliquid position API recovered after %d failure(s) and %.0fs",
                int(getattr(self, "_position_outage_failures", 0) or 0),
                downtime,
            )
        self._position_outage_started_at = None
        self._last_position_outage_warning_at = None
        self._position_outage_failures = 0

    async def _fetch_positions(self, wallet_address: Optional[str]) -> Dict[str, Position]:
        """Fetch and parse open positions from Hyperliquid user state. Raises on failure.

        Results are cached per wallet for ``_positions_cache_ttl`` seconds to
        prevent CloudFront 429s when the dashboard polls rapidly.
        """
        if not self.info_client:
            raise ValueError("Hyperliquid info client is not initialized")
        if not wallet_address:
            raise ValueError("No wallet address available. Call authenticate_user() or pass wallet_address.")

        # --- short-circuit cache ---
        cached = self._positions_cache.get(wallet_address)
        if cached:
            age = (datetime.now() - cached["fetched_at"]).total_seconds()
            if age < self._positions_cache_ttl:
                logger.debug(
                    f"_fetch_positions cache hit for {wallet_address[:10]}… ({age:.1f}s old)"
                )
                return dict(cached["positions"])

        dex_names = await self.get_enabled_perp_dex_names()

        def fetch_all_states():
            return [self.info_client.user_state(wallet_address, dex=dex_name) for dex_name in dex_names]

        user_states = await asyncio.get_event_loop().run_in_executor(None, fetch_all_states)
        positions: Dict[str, Position] = {}
        for user_state in user_states:
            positions.update(self.parse_positions_payload(user_state))

        # Store in cache
        self._positions_cache[wallet_address] = {"positions": dict(positions), "fetched_at": datetime.now()}
        return positions

    @staticmethod
    def parse_positions_payload(user_state: Optional[Dict[str, Any]]) -> Dict[str, "Position"]:
        """
        Parse a clearinghouseState-shaped payload into Position objects.

        Works for both the REST user_state response and the clearinghouseState
        pushed on the webData2 websocket channel (same schema).
        """
        if not user_state or 'assetPositions' not in user_state:
            return {}

        positions = {}

        for asset_pos in user_state['assetPositions']:
            position = asset_pos['position']
            # Hyperliquid nests the coin inside the position payload
            coin = position.get('coin') or asset_pos.get('coin')
            if not coin:
                continue

            # Only include open positions
            size = float(position.get('szi', 0))
            if abs(size) < 1e-8:  # Effectively zero
                continue

            side = 'long' if size > 0 else 'short'
            entry_price = float(position.get('entryPx', 0))
            unrealized_pnl = float(position.get('unrealizedPnl', 0))
            position_value = abs(float(position.get('positionValue', 0) or 0))
            # assetPositions has no markPx field; derive it from notional value
            mark_price = position_value / abs(size) if position_value > 0 else entry_price
            liq_raw = position.get('liquidationPx')
            leverage_raw = position.get('leverage')
            leverage = float(leverage_raw.get('value', 1)) if isinstance(leverage_raw, dict) else float(leverage_raw or 1)
            margin_used = float(position.get('marginUsed', 0) or 0) or (position_value / leverage if leverage > 0 else position_value)

            positions[coin] = Position(
                symbol=coin,
                side=side,
                size=abs(size),
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                margin_used=margin_used,
                timestamp=datetime.now(),
                mark_price=mark_price,
                position_value=position_value,
                liquidation_price=float(liq_raw) if liq_raw is not None else None,
                leverage=leverage,
            )

        return positions
    
    async def get_sentiment_enhanced_signals(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get sentiment-enhanced trading signals for specified symbols.
        
        Args:
            symbols: List of symbols to analyze (defaults to ['BTC', 'ETH', 'SOL'])
            
        Returns:
            Dictionary with sentiment-enhanced trading signals
        """
        try:
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL']
            
            signals = {}
            
            # Get unified sentiment if service is available
            unified_sentiment = None
            advanced_analysis = None
            
            if self.sentiment_service:
                unified_sentiment = await self.sentiment_service.get_unified_sentiment(24)
                
                if self.sentiment_analyzer:
                    advanced_analysis = self.sentiment_analyzer.analyze_sentiment(unified_sentiment)
            
            # Analyze each symbol
            for symbol in symbols:
                try:
                    # Get market data
                    market_data = await self.get_market_data(symbol)
                    
                    # Generate sentiment-enhanced signal
                    signal = await self._generate_sentiment_enhanced_signal(
                        symbol, market_data, unified_sentiment, advanced_analysis
                    )
                    
                    signals[symbol] = signal
                    
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol}: {e}")
                    signals[symbol] = {
                        'symbol': symbol,
                        'signal': 'NEUTRAL',
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            return {
                'signals': signals,
                'market_sentiment': unified_sentiment.sentiment_label if unified_sentiment else 'neutral',
                'market_regime': advanced_analysis.regime.regime if advanced_analysis else 'sideways',
                'risk_level': unified_sentiment.risk_level if unified_sentiment else 'medium',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment-enhanced signals: {e}")
            return {
                'signals': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_sentiment_enhanced_signal(self, symbol: str, market_data: MarketData, 
                                                unified_sentiment: Optional[Any], 
                                                advanced_analysis: Optional[Any]) -> Dict[str, Any]:
        """Generate sentiment-enhanced trading signal for a symbol."""
        try:
            # Base technical signal from price action
            price_change_24h = 0.0  # Would need historical data
            volume_trend = 'neutral'
            
            # Basic technical signal
            technical_signal = 'NEUTRAL'
            technical_confidence = 0.5
            
            # Check funding rate for bias
            if market_data.funding_rate > 0.01:  # 1% funding rate
                technical_signal = 'SELL'  # High funding suggests shorts may be squeezed
                technical_confidence = 0.6
            elif market_data.funding_rate < -0.01:
                technical_signal = 'BUY'  # Negative funding suggests longs may be favored
                technical_confidence = 0.6
            
            # Initialize sentiment components
            sentiment_signal = 'NEUTRAL'
            sentiment_confidence = 0.5
            sentiment_details = {}
            
            if unified_sentiment:
                # Convert sentiment to trading signal
                if unified_sentiment.overall_sentiment > 0.3:
                    sentiment_signal = 'BUY'
                    sentiment_confidence = unified_sentiment.confidence
                elif unified_sentiment.overall_sentiment < -0.3:
                    sentiment_signal = 'SELL'
                    sentiment_confidence = unified_sentiment.confidence
                
                sentiment_details = {
                    'overall_sentiment': unified_sentiment.overall_sentiment,
                    'sentiment_label': unified_sentiment.sentiment_label,
                    'fear_greed_index': unified_sentiment.fear_greed_index,
                    'market_mood': unified_sentiment.market_mood.primary_mood,
                    'trading_bias': unified_sentiment.trading_bias,
                    'market_stress': unified_sentiment.market_stress
                }
            
            # Advanced analysis components
            regime_signal = 'NEUTRAL'
            risk_adjustment = 1.0
            
            if advanced_analysis:
                regime = advanced_analysis.regime.regime
                risk_score = (
                    advanced_analysis.risk.systemic_risk * 0.3 +
                    advanced_analysis.risk.tail_risk * 0.3 +
                    advanced_analysis.risk.liquidity_risk * 0.4
                )
                
                # Regime-based signal adjustment
                if regime == 'bull_market':
                    regime_signal = 'BUY'
                elif regime == 'bear_market':
                    regime_signal = 'SELL'
                elif regime == 'crisis':
                    regime_signal = 'STRONG_SELL'
                elif regime == 'euphoria':
                    regime_signal = 'SELL'  # Contrarian in euphoria
                
                # Risk-based confidence adjustment
                risk_adjustment = max(0.3, 1 - risk_score)
                
                sentiment_details.update({
                    'regime': regime,
                    'risk_score': risk_score,
                    'volatility_regime': advanced_analysis.risk.volatility_regime,
                    'momentum': advanced_analysis.temporal.momentum
                })
            
            # Combine signals with weights
            signal_weights = {
                'technical': 0.4,
                'sentiment': 0.4,
                'regime': 0.2
            }
            
            # Convert signals to scores
            signal_scores = {
                'technical': self._signal_to_score(technical_signal) * technical_confidence,
                'sentiment': self._signal_to_score(sentiment_signal) * sentiment_confidence,
                'regime': self._signal_to_score(regime_signal)
            }
            
            # Calculate weighted score
            weighted_score = sum(
                signal_scores[component] * signal_weights[component] 
                for component in signal_weights
            )
            
            # Apply risk adjustment
            weighted_score *= risk_adjustment
            
            # Determine final signal
            if weighted_score > 0.3:
                if weighted_score > 0.6:
                    final_signal = 'STRONG_BUY'
                else:
                    final_signal = 'BUY'
            elif weighted_score < -0.3:
                if weighted_score < -0.6:
                    final_signal = 'STRONG_SELL'
                else:
                    final_signal = 'SELL'
            else:
                final_signal = 'NEUTRAL'
            
            # Calculate final confidence
            final_confidence = min(abs(weighted_score), 1.0)
            
            # Generate reasoning
            reasoning = self._generate_signal_reasoning(
                technical_signal, sentiment_signal, regime_signal, market_data, sentiment_details
            )
            
            # Calculate position sizing recommendation
            position_sizing = self._calculate_position_sizing(
                final_signal, final_confidence, sentiment_details
            )
            
            return {
                'symbol': symbol,
                'signal': final_signal,
                'confidence': final_confidence,
                'score': weighted_score,
                'components': {
                    'technical': {
                        'signal': technical_signal,
                        'confidence': technical_confidence,
                        'score': signal_scores['technical']
                    },
                    'sentiment': {
                        'signal': sentiment_signal,
                        'confidence': sentiment_confidence,
                        'score': signal_scores['sentiment']
                    },
                    'regime': {
                        'signal': regime_signal,
                        'score': signal_scores['regime']
                    }
                },
                'market_data': {
                    'price': market_data.price,
                    'funding_rate': market_data.funding_rate,
                    'volume_24h': market_data.volume_24h
                },
                'sentiment_details': sentiment_details,
                'reasoning': reasoning,
                'position_sizing': position_sizing,
                'risk_adjustment': risk_adjustment,
                'timestamp': market_data.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating sentiment-enhanced signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _signal_to_score(self, signal: str) -> float:
        """Convert signal string to numerical score."""
        signal_map = {
            'STRONG_BUY': 1.0,
            'BUY': 0.5,
            'NEUTRAL': 0.0,
            'SELL': -0.5,
            'STRONG_SELL': -1.0
        }
        return signal_map.get(signal, 0.0)
    
    def _generate_signal_reasoning(self, technical_signal: str, sentiment_signal: str, 
                                 regime_signal: str, market_data: MarketData, 
                                 sentiment_details: Dict[str, Any]) -> List[str]:
        """Generate reasoning for the trading signal."""
        reasoning = []
        
        # Technical reasoning
        if technical_signal != 'NEUTRAL':
            if market_data.funding_rate > 0.01:
                reasoning.append(f"High funding rate ({market_data.funding_rate:.3%}) suggests short squeeze potential")
            elif market_data.funding_rate < -0.01:
                reasoning.append(f"Negative funding rate ({market_data.funding_rate:.3%}) favors long positions")
        
        # Sentiment reasoning
        if sentiment_signal != 'NEUTRAL' and sentiment_details:
            sentiment_label = sentiment_details.get('sentiment_label', 'neutral')
            reasoning.append(f"Market sentiment is {sentiment_label}")
            
            fear_greed = sentiment_details.get('fear_greed_index')
            if fear_greed and fear_greed > 75:
                reasoning.append("Extreme greed detected - potential reversal risk")
            elif fear_greed and fear_greed < 25:
                reasoning.append("Extreme fear detected - potential buying opportunity")
        
        # Regime reasoning
        if regime_signal != 'NEUTRAL' and sentiment_details:
            regime = sentiment_details.get('regime')
            if regime:
                reasoning.append(f"Market regime: {regime}")
        
        # Risk reasoning
        market_stress = sentiment_details.get('market_stress', 0)
        if market_stress > 0.7:
            reasoning.append("High market stress detected - increased caution recommended")
        
        return reasoning
    
    def _calculate_position_sizing(self, signal: str, confidence: float, 
                                 sentiment_details: Dict[str, Any]) -> str:
        """Calculate position sizing recommendation."""
        if signal == 'NEUTRAL':
            return 'none'
        
        # Base sizing on signal and confidence
        if confidence > 0.8:
            base_size = 'large'
        elif confidence > 0.6:
            base_size = 'medium'
        elif confidence > 0.4:
            base_size = 'small'
        else:
            base_size = 'minimal'
        
        # Adjust for risk
        market_stress = sentiment_details.get('market_stress', 0)
        risk_score = sentiment_details.get('risk_score', 0.5)
        
        if market_stress > 0.8 or risk_score > 0.8:
            if base_size == 'large':
                return 'medium'
            elif base_size == 'medium':
                return 'small'
            else:
                return 'minimal'
        
        return base_size
    
    async def place_order_live(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "ioc",
        leverage: Optional[float] = None,
        trigger_price: Optional[float] = None,
        trigger_tpsl: Optional[str] = None,
        trigger_is_market: bool = True
    ) -> OrderResult:
        """
        Place a live trading order with enhanced execution and monitoring.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC')
            side: Order side (buy/sell)
            size: Order size
            order_type: Type of order (market/limit)
            price: Limit price (for limit orders)
            reduce_only: Whether this is a reduce-only order
            time_in_force: Time in force (ioc, gtc, alo)
            leverage: Leverage for the position
            trigger_price: Trigger price for stop-loss/take-profit orders
            trigger_tpsl: Trigger order type: "sl" or "tp"
            trigger_is_market: Whether the trigger executes as a market order
            
        Returns:
            Enhanced OrderResult with execution details
        """
        start_time = datetime.now()
        
        try:
            exchange_client = self._active_exchange_client()
            if not exchange_client:
                raise ValueError("User not authenticated for trading. Call authenticate_user() first.")

            if self._perp_dex_for_symbol(symbol):
                mapping_ready = await self._ensure_exchange_symbol_mapping(symbol)
                if not mapping_ready:
                    raise ValueError(f"Could not prepare HIP-3 execution mapping for {symbol}")
                unified_ready = await self.ensure_unified_account_for_hip3()
                if not unified_ready:
                    raise ValueError(
                        f"Unified Hyperliquid account mode is required before trading HIP-3 symbol {symbol}"
                    )

            if not self.is_ws_connected:
                # Best effort: real-time feeds improve execution pricing, but the
                # order itself is placed over REST. Worker processes trade via the
                # delegated client without authenticate_user(), so try to connect
                # once and otherwise proceed on REST market data.
                try:
                    await self._connect_websocket(market_symbols=[symbol])
                except Exception as ws_error:
                    logger.warning(f"WebSocket connect attempt before order failed: {ws_error}")
                if not self.is_ws_connected:
                    logger.warning("Placing order without WebSocket feed - using REST market data")

            # Get real-time market data for better execution
            live_data = await self.get_live_market_data(symbol)
            if not live_data:
                raise ValueError(f"No live market data available for {symbol} - WebSocket feed required")
            
            is_trigger_order = trigger_price is not None and trigger_tpsl is not None
            if is_trigger_order:
                trigger_tpsl = str(trigger_tpsl).lower()
                if trigger_tpsl not in {"tp", "sl"}:
                    raise ValueError("trigger_tpsl must be 'tp' or 'sl'")
                if trigger_is_market:
                    # A market-executing trigger is still filled against limit_px once
                    # it activates (Hyperliquid has no true "market" order type - see
                    # the SDK's own market_open/market_close, which pad by 5% for the
                    # same reason). Using the bare trigger price as limit_px means any
                    # price movement between trigger activation and matching - normal
                    # in a fast stop-out - leaves the order unable to cross the book,
                    # so the "stop-loss" silently fails to fill and the position stays
                    # open and unprotected. Pad the same direction the SDK does.
                    base = float(price or trigger_price)
                    execution_price = base * 1.05 if side == OrderSide.BUY else base * 0.95
                else:
                    execution_price = float(price or trigger_price)
            elif order_type == OrderType.MARKET or price is None:
                # Use real-time bid/ask for better execution
                if side == OrderSide.BUY:
                    execution_price = live_data.ask  # Buy at ask
                else:
                    execution_price = live_data.bid  # Sell at bid
                # IOC limit emulating a market order: pad the cap so it crosses
                # the spread; fills execute at the best available price, not the cap.
                execution_price *= 1.01 if side == OrderSide.BUY else 0.99
            else:
                execution_price = price

            # Conform size and prices to the venue's tick rules (szDecimals,
            # 5 significant figures) - raw floats are rejected by the exchange.
            sz_decimals = await self.get_perp_sz_decimals(symbol)
            size = self._round_size_for_hl(size, sz_decimals)
            if size <= 0:
                raise ValueError(f"Order size rounds to zero for {symbol} (szDecimals={sz_decimals})")
            execution_price = self._round_price_for_hl(execution_price, sz_decimals)
            if trigger_price is not None:
                trigger_price = self._round_price_for_hl(float(trigger_price), sz_decimals)

            # Preserve generator-selected leverage whenever the market supports it.
            # Only Hyperliquid's per-market boundary may alter the request.
            if leverage is not None and not reduce_only:
                requested_leverage = leverage
                venue_max_leverage = await self.get_perp_max_leverage(symbol)
                leverage = self.closest_supported_leverage(
                    requested_leverage,
                    venue_max_leverage,
                )
                if float(leverage) != float(requested_leverage):
                    logger.info(
                        "Adjusted %s leverage from %sx to closest Hyperliquid-supported %sx "
                        "(market max: %s)",
                        symbol,
                        requested_leverage,
                        leverage,
                        venue_max_leverage,
                    )

            # Validate order against non-leverage risk parameters
            await self._validate_order_risk(symbol, side, size, execution_price, leverage, reduce_only=reduce_only)
            
            # Set leverage if specified
            if leverage and leverage != 1.0 and not reduce_only:
                await self._set_leverage(symbol, leverage)

            if is_trigger_order:
                order_type_payload = {
                    'trigger': {
                        'triggerPx': float(trigger_price),
                        'isMarket': bool(trigger_is_market),
                        'tpsl': trigger_tpsl
                    }
                }
            else:
                tif_map = {
                    "ioc": "Ioc",
                    "gtc": "Gtc",
                    "alo": "Alo",
                }
                tif = tif_map.get(str(time_in_force or "ioc").lower(), "Ioc")
                order_type_payload = {'limit': {'tif': tif}}
            
            # Prepare enhanced order
            order = {
                'coin': symbol,
                'is_buy': side == OrderSide.BUY,
                'sz': size,
                'limit_px': execution_price,
                'order_type': order_type_payload,
                'reduce_only': reduce_only
            }
            
            # Place the order, attaching the platform builder fee when configured.
            builder = self._builder_info()
            async with self._active_exchange_request_lock():
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: exchange_client.bulk_orders([order], builder)
                )

                # Builder fees require a one-time user approval; never let a missing
                # approval block the trade itself.
                if builder and response.get('status') != 'ok':
                    builder_error = self._extract_error_message(response) or ""
                    if "builder" in builder_error.lower():
                        logger.warning(f"Order rejected for builder fee ({builder_error}); retrying without builder")
                        response = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: exchange_client.bulk_orders([order])
                        )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            self.trade_execution_times.append(execution_time)
            
            # Parse response
            if response.get('status') == 'ok':
                result = await self._parse_order_response(response, symbol, side, size, execution_price, start_time)
                
                # Track active order
                if result.order_id:
                    self.active_orders[result.order_id] = {
                        'symbol': symbol,
                        'side': side.value,
                        'size': size,
                        'price': execution_price,
                        'timestamp': start_time,
                        'status': 'pending',
                        'reduce_only': reduce_only,
                        'trigger_price': trigger_price,
                        'trigger_tpsl': trigger_tpsl
                    }
                
                order_label = f"{trigger_tpsl.upper()} trigger" if is_trigger_order else order_type.value
                logger.info(f"Order placed: {order_label} {side.value} {size} {symbol} @ ${execution_price} (execution: {execution_time:.1f}ms)")
                return result
            else:
                error_msg = self._extract_error_message(response)
                return OrderResult(
                    success=False, 
                    error=error_msg,
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=execution_price,
                    timestamp=start_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error placing order: {e} (execution: {execution_time:.1f}ms)")
            return OrderResult(
                success=False, 
                error=str(e),
                symbol=symbol,
                side=side.value if side else None,
                size=size,
                timestamp=start_time
            )
    
    async def _validate_order_risk(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        price: float,
        leverage: Optional[float],
        reduce_only: bool = False
    ):
        """Validate order against risk parameters."""
        try:
            # `size * price` is already the full leveraged notional. Applying
            # leverage again here rejects valid high-exposure orders twice over.
            position_value = size * price
            required_margin = position_value / max(float(leverage or 1), 1.0)
            
            # Reduce-only exits lower exposure, so balance and max-position checks are not applied.
            if reduce_only:
                return

            # Check maximum position size
            if position_value > self.risk_params['max_position_size_usd']:
                raise ValueError(f"Position value ${position_value:,.2f} exceeds max ${self.risk_params['max_position_size_usd']:,.2f}")
            
            # Check account balance and margin
            active_wallet = getattr(self, "_active_wallet_address", None)
            check_address = (
                active_wallet()
                if callable(active_wallet)
                else getattr(self, "user_wallet_address", None)
            )
            if check_address:
                account_summary = await self.get_wallet_balance_summary(check_address)
                balance = float(account_summary.get('balance', 0) or 0)
                if balance > 0 and balance < required_margin:
                    raise ValueError(f"Insufficient balance ${balance:.2f} for required margin ${required_margin:.2f}")
            
            # Check funding rate for high-cost positions
            live_data = await self.get_live_market_data(symbol)
            if live_data and abs(live_data.funding_rate) > self.risk_params['funding_rate_threshold']:
                logger.warning(f"High funding rate {live_data.funding_rate:.4%} for {symbol}")
            
        except Exception as e:
            logger.error(f"Risk validation failed: {e}")
            raise
    
    async def _set_leverage(self, symbol: str, leverage: float):
        """Set leverage for a symbol, automatically setting is_cross=False if the symbol is isolated-only."""
        try:
            exchange_client = self._active_exchange_client()
            if exchange_client:
                is_cross = True
                try:
                    universe = await self.get_perp_universe()
                    asset_data = universe.get(symbol.upper()) if universe else None
                    if asset_data and asset_data.get("only_isolated"):
                        is_cross = False
                except Exception as meta_err:
                    logger.warning(f"Could not check isolated status for {symbol}: {meta_err}")

                async with self._active_exchange_request_lock():
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: exchange_client.update_leverage(int(leverage), symbol, is_cross=is_cross)
                    )

                    if response.get('status') != 'ok' and is_cross:
                        err_msg = str(response)
                        if 'isolated' in err_msg.lower():
                            logger.info(f"Retrying isolated leverage for {symbol} after cross margin rejection")
                            response = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: exchange_client.update_leverage(int(leverage), symbol, is_cross=False)
                            )

                if response.get('status') != 'ok':
                    err_msg = str(response)
                    if 'does not exist' in err_msg.lower():
                        logger.warning(f"Hyperliquid update_leverage returned notice for {symbol}: {response}; proceeding with order placement")
                    else:
                        raise ValueError(f"Failed to set leverage: {response}")

                logger.info(f"Set leverage to {leverage}x (is_cross={is_cross}) for {symbol}")

        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            raise
    
    async def _parse_order_response(self, response: Dict[str, Any], symbol: str, side: OrderSide, 
                                  size: float, price: float, start_time: datetime) -> OrderResult:
        """Parse order response into OrderResult."""
        try:
            data = response.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            
            if not statuses:
                return OrderResult(
                    success=True,
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price,
                    timestamp=start_time,
                    status='unknown'
                )
            
            status = statuses[0]
            
            # Check for fill
            if 'filled' in status:
                filled = status['filled']
                return OrderResult(
                    success=True,
                    order_id=None,  # Filled immediately
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price,
                    filled_size=float(filled.get('sz', size)),
                    average_price=float(filled.get('avgPx', price)),
                    fee=float(filled.get('fee', 0)),
                    timestamp=start_time,
                    status='filled'
                )
            
            # Check for resting order
            elif 'resting' in status:
                resting = status['resting']
                return OrderResult(
                    success=True,
                    order_id=str(resting.get('oid')),
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price,
                    timestamp=start_time,
                    status='open'
                )
            
            # Check for error
            elif 'error' in status:
                return OrderResult(
                    success=False,
                    error=status['error'],
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price,
                    timestamp=start_time
                )
            
            else:
                return OrderResult(
                    success=True,
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=price,
                    timestamp=start_time,
                    status='unknown'
                )
                
        except Exception as e:
            logger.error(f"Error parsing order response: {e}")
            return OrderResult(
                success=False,
                error=f"Response parsing error: {e}",
                symbol=symbol,
                side=side.value,
                size=size,
                price=price,
                timestamp=start_time
            )
    
    def _extract_error_message(self, response: Dict[str, Any]) -> str:
        """Extract error message from response."""
        try:
            data = response.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            
            if statuses and 'error' in statuses[0]:
                return statuses[0]['error']
            
            return response.get('error', 'Unknown error')
            
        except Exception:
            return "Failed to parse error message"
    
    async def place_order(
        self, 
        symbol: str, 
        side: OrderSide, 
        size: float, 
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        reduce_only: bool = False
    ) -> OrderResult:
        """
        Place a trading order.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC')
            side: Order side (buy/sell)
            size: Order size
            order_type: Type of order
            price: Limit price (for limit orders)
            reduce_only: Whether this is a reduce-only order
            
        Returns:
            OrderResult with execution details
        """
        try:
            exchange_client = self._active_exchange_client()
            if not exchange_client:
                raise ValueError("User not authenticated. Call authenticate_user() first.")
            
            # Get current market price if needed
            if order_type == OrderType.MARKET or price is None:
                market_data = await self.get_market_data(symbol)
                limit_price = market_data.price
            else:
                limit_price = price
            
            # Validate size against risk parameters
            position_value = size * limit_price
            if position_value > self.risk_params['max_position_size_usd']:
                raise ValueError(f"Position value ${position_value:,.2f} exceeds max ${self.risk_params['max_position_size_usd']:,.2f}")
            
            # Prepare order
            order = {
                'coin': symbol,
                'is_buy': side == OrderSide.BUY,
                'sz': size,
                'limit_px': limit_price,
                'order_type': {
                    'limit': order_type == OrderType.LIMIT,
                    'ioc': order_type == OrderType.MARKET
                },
                'reduce_only': reduce_only
            }
            
            # Place the order using Hyperliquid exchange client
            async with self._active_exchange_request_lock():
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: exchange_client.order([order]),
                )
            
            if response.get('status') == 'ok':
                # Extract order details from response
                statuses = response.get('response', {}).get('data', {}).get('statuses', [])
                if statuses and 'resting' in statuses[0]:
                    order_id = statuses[0]['resting'].get('oid')
                else:
                    order_id = None
                
                result = OrderResult(
                    success=True,
                    order_id=order_id,
                    symbol=symbol,
                    side=side.value,
                    size=size,
                    price=limit_price,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Order placed successfully: {side.value} {size} {symbol} @ ${limit_price}")
                return result
            
            else:
                error_msg = response.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('error', 'Unknown error')
                return OrderResult(success=False, error=error_msg)
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> OrderResult:
        """
        Close a position (partial or full).
        
        Args:
            symbol: Symbol to close
            percentage: Percentage of position to close (default: 100%)
            
        Returns:
            OrderResult with closure details
        """
        try:
            positions = await self.get_positions()
            
            if symbol not in positions:
                return OrderResult(success=False, error=f"No open position for {symbol}")
            
            position = positions[symbol]
            
            # Calculate size to close
            size_to_close = position.size * (percentage / 100.0)
            
            # Determine order side (opposite of position)
            close_side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
            
            # Place market order to close
            result = await self.place_order(
                symbol=symbol,
                side=close_side,
                size=size_to_close,
                order_type=OrderType.MARKET,
                reduce_only=True
            )
            
            if result.success:
                logger.info(f"Position closed: {percentage}% of {symbol} position")
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def get_funding_rates(self, symbols: List[str] = None) -> Dict[str, FundingData]:
        """
        Get real-time funding rates for symbols.
        
        Args:
            symbols: List of symbols (defaults to major perpetuals)
            
        Returns:
            Dictionary of funding data by symbol
        """
        try:
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'AVAX']
            
            funding_data = {}
            
            for symbol in symbols:
                try:
                    # Get funding history from API
                    funding_history = self.info_client.funding_history(symbol, startTime=0, endTime=None)
                    
                    if funding_history:
                        latest_funding = funding_history[-1]
                        
                        funding_data[symbol] = FundingData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(latest_funding['time'] / 1000),
                            funding_rate=float(latest_funding['fundingRate']),
                            next_funding_time=datetime.fromtimestamp((latest_funding['time'] + 8 * 3600 * 1000) / 1000),  # Approx 8h
                            premium=float(latest_funding.get('premium', 0))
                        )
                        
                        # Store in cache
                        self.funding_rates[symbol] = funding_data[symbol]
                        
                except Exception as e:
                    logger.error(f"Error getting funding rate for {symbol}: {e}")
                    continue
            
            return funding_data
            
        except Exception as e:
            logger.error(f"Error getting funding rates: {e}")
            return {}
    
    async def get_open_interest_data(self, symbols: List[str] = None) -> Dict[str, OpenInterestData]:
        """
        Get open interest data for symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of open interest data by symbol
        """
        try:
            if not symbols:
                symbols = ['BTC', 'ETH', 'SOL', 'ARB', 'OP', 'AVAX']
            
            oi_data = {}
            
            # Get meta data for universe
            meta = self.info_client.meta()
            universe = meta.get('universe', [])
            
            for symbol in symbols:
                try:
                    # Find symbol in universe
                    symbol_info = None
                    for asset in universe:
                        if asset['name'] == symbol:
                            symbol_info = asset
                            break
                    
                    if not symbol_info:
                        continue
                    
                    # Get actual open interest data from Hyperliquid API
                    try:
                        # Get open interest from clearinghouse state
                        clearinghouse_state = self.info_client.open_orders(symbol)
                        if clearinghouse_state:
                            # Extract open interest from clearinghouse data
                            # This is a simplified extraction - in production you'd need to
                            # aggregate across all positions properly
                            total_oi = 0.0
                            # Implementation would depend on Hyperliquid's API response format
                            
                            oi_data[symbol] = OpenInterestData(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                open_interest=total_oi,
                                oi_change_24h=0.0  # Would calculate from historical data
                            )
                            
                            # Store in cache
                            self.open_interest_data[symbol] = oi_data[symbol]
                        else:
                            logger.warning(f"No open interest data available for {symbol}")
                    except Exception as oi_error:
                        logger.error(f"Failed to get OI data for {symbol}: {oi_error}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error getting open interest for {symbol}: {e}")
                    continue
            
            return oi_data
            
        except Exception as e:
            logger.error(f"Error getting open interest data: {e}")
            return {}
    
    async def monitor_position_health(self) -> Dict[str, Any]:
        """
        Monitor position health and margin levels in real-time.
        
        Returns:
            Position health metrics
        """
        try:
            if not self._active_wallet_address():
                return {"error": "No authenticated user"}
            
            # Get current positions
            positions = await self.get_positions()
            account_summary = await self.get_account_summary()
            
            position_health = {
                'total_positions': len(positions),
                'total_unrealized_pnl': 0.0,
                'margin_usage_percent': 0.0,
                'liquidation_risk': 'low',
                'positions_at_risk': [],
                'funding_cost_24h': 0.0,
                'recommendations': []
            }
            
            if not positions:
                return position_health
            
            total_pnl = 0.0
            total_margin_used = 0.0
            positions_at_risk = []
            
            for symbol, position in positions.items():
                # Calculate unrealized PnL
                live_data = await self.get_live_market_data(symbol)
                if live_data:
                    if position.side == 'long':
                        current_pnl = (live_data.price - position.entry_price) * position.size
                    else:
                        current_pnl = (position.entry_price - live_data.price) * position.size
                    
                    total_pnl += current_pnl
                    total_margin_used += position.margin_used
                    
                    # Check liquidation risk
                    liquidation_price = self._calculate_liquidation_price(position, live_data)
                    price_to_liquidation = abs(live_data.price - liquidation_price) / live_data.price
                    
                    if price_to_liquidation < 0.05:  # 5% to liquidation
                        positions_at_risk.append({
                            'symbol': symbol,
                            'side': position.side,
                            'size': position.size,
                            'current_price': live_data.price,
                            'liquidation_price': liquidation_price,
                            'risk_level': 'high'
                        })
                    elif price_to_liquidation < 0.15:  # 15% to liquidation
                        positions_at_risk.append({
                            'symbol': symbol,
                            'side': position.side,
                            'size': position.size,
                            'current_price': live_data.price,
                            'liquidation_price': liquidation_price,
                            'risk_level': 'medium'
                        })
            
            # Calculate metrics
            account_value = account_summary.get('balance', 0)
            if account_value > 0:
                position_health['margin_usage_percent'] = (total_margin_used / account_value) * 100
            
            position_health['total_unrealized_pnl'] = total_pnl
            position_health['positions_at_risk'] = positions_at_risk
            
            # Determine overall liquidation risk
            if len(positions_at_risk) > 0:
                high_risk_positions = [p for p in positions_at_risk if p['risk_level'] == 'high']
                if high_risk_positions:
                    position_health['liquidation_risk'] = 'high'
                else:
                    position_health['liquidation_risk'] = 'medium'
            
            # Generate recommendations
            if position_health['margin_usage_percent'] > 80:
                position_health['recommendations'].append("Consider reducing position sizes - high margin usage")
            
            if positions_at_risk:
                position_health['recommendations'].append("Monitor positions at risk closely - consider adding margin or reducing size")
            
            return position_health
            
        except Exception as e:
            logger.error(f"Error monitoring position health: {e}")
            return {"error": str(e)}
    
    def _calculate_liquidation_price(self, position: Position, live_data: LiveMarketData) -> float:
        """Calculate approximate liquidation price for a position."""
        try:
            # Simplified liquidation calculation
            # Actual calculation would need to consider maintenance margin, fees, etc.
            
            maintenance_margin_rate = 0.05  # 5% maintenance margin
            
            if position.side == 'long':
                # For long positions, liquidation occurs when price drops
                liquidation_price = position.entry_price * (1 - maintenance_margin_rate)
            else:
                # For short positions, liquidation occurs when price rises
                liquidation_price = position.entry_price * (1 + maintenance_margin_rate)
            
            return liquidation_price
            
        except Exception as e:
            logger.error(f"Error calculating liquidation price: {e}")
            return 0.0
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive real-time trading metrics.
        
        Returns:
            Real-time metrics for monitoring and decision making
        """
        try:
            metrics = {
                'websocket_status': {
                    'connected': self.is_ws_connected,
                    'last_update': self.last_data_update.isoformat() if self.last_data_update else None,
                    'symbols_tracked': list(self.live_market_data.keys())
                },
                'market_data': {},
                'funding_rates': {},
                'execution_performance': {
                    'avg_execution_time_ms': 0.0,
                    'recent_trades': len(self.recent_trades),
                    'active_orders': len(self.active_orders)
                },
                'account_status': await self.get_account_summary(),
                'position_health': await self.monitor_position_health()
            }
            
            # Add market data for tracked symbols
            for symbol, data in self.live_market_data.items():
                metrics['market_data'][symbol] = {
                    'price': data.price,
                    'bid': data.bid,
                    'ask': data.ask,
                    'spread': data.ask - data.bid,
                    'spread_bps': ((data.ask - data.bid) / data.mid_price) * 10000 if data.mid_price > 0 else 0,
                    'volume_24h': data.volume_24h,
                    'last_update': data.timestamp.isoformat()
                }
            
            # Add funding rates
            for symbol, funding in self.funding_rates.items():
                metrics['funding_rates'][symbol] = {
                    'rate': funding.funding_rate,
                    'annualized_rate': funding.funding_rate * 365 * 3,  # Assuming 8h funding
                    'next_funding': funding.next_funding_time.isoformat(),
                    'time_to_funding': (funding.next_funding_time - datetime.now()).total_seconds() / 3600
                }
            
            # Calculate execution performance
            if self.trade_execution_times:
                metrics['execution_performance']['avg_execution_time_ms'] = sum(self.trade_execution_times) / len(self.trade_execution_times)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {"error": str(e)}
    
    async def get_wallet_balance_summary(
        self,
        wallet_address: str,
        account_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read wallet equity from the correct state for its account abstraction mode."""
        if not self.info_client or not wallet_address:
            raise ValueError("Hyperliquid info client and wallet address are required")

        mode = account_mode
        if mode is None:
            mode = await self.get_user_abstraction_mode(wallet_address)
        if mode in {"unifiedAccount", "portfolioMargin"}:
            spot_state = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.info_client.spot_user_state(wallet_address)
            )
            usdc = next(
                (
                    balance for balance in (spot_state or {}).get("balances", [])
                    if str(balance.get("coin") or "").upper() == "USDC"
                    or balance.get("token") == 0
                ),
                {},
            )
            total = float(usdc.get("total") or 0)
            held = float(usdc.get("hold") or 0)
            return {
                "account_exists": bool(spot_state),
                "mode": mode,
                "balance": total,
                "withdrawable": max(0.0, total - held),
                "held": held,
            }

        user_state = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.info_client.user_state(wallet_address)
        )
        cross_margin_summary = (user_state or {}).get("crossMarginSummary", {}) or {}
        balance = float(
            cross_margin_summary.get("accountValue")
            or (user_state or {}).get("withdrawable")
            or 0
        )
        return {
            "account_exists": bool(user_state),
            "mode": mode or "default",
            "balance": balance,
            "withdrawable": float((user_state or {}).get("withdrawable") or 0),
            "held": max(0.0, balance - float((user_state or {}).get("withdrawable") or 0)),
        }

    async def get_account_summary(self, wallet_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account summary including balance and positions.
        
        Returns:
            Account summary data
        """
        try:
            address = wallet_address or self._active_wallet_address()
            if not self.info_client or not address:
                raise ValueError("User not authenticated. Call authenticate_user() first.")

            balance_summary = await self.get_wallet_balance_summary(address)
            if not balance_summary.get("account_exists"):
                return {
                    'account_exists': False,
                    'balance': 0.0,
                    'margin_used': 0.0,
                    'unrealized_pnl': 0.0,
                    'positions_count': 0
                }
            
            positions = await self.get_positions(address)
            margin_used = sum(float(position.margin_used or 0) for position in positions.values())
            unrealized_pnl = sum(float(position.unrealized_pnl or 0) for position in positions.values())
            
            return {
                'account_exists': True,
                'mode': balance_summary.get('mode'),
                'balance': float(balance_summary.get('balance') or 0),
                'margin_used': margin_used,
                'unrealized_pnl': unrealized_pnl,
                'positions_count': len(positions),
                'withdrawable': float(balance_summary.get('withdrawable') or 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
            return {
                'account_exists': False,
                'balance': 0.0,
                'margin_used': 0.0,
                'unrealized_pnl': 0.0,
                'positions_count': 0,
                'error': str(e)
            }


# Factory function for creating service instance
def create_hyperliquid_service(privy_app_id: str, privy_app_secret: str, testnet: bool = False) -> HyperliquidService:
    """Create and return HyperliquidService instance with Privy integration."""
    return HyperliquidService(privy_app_id, privy_app_secret, testnet)
