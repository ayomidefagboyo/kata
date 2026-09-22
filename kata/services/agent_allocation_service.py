"""
Agent Allocation Service - Manages fund allocations from platform wallets to trading agents.

This service connects platform wallet fund allocations with agent trading execution,
enabling users to delegate funds to agents like Yuki for automated trading.
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import uuid
import re
from decimal import Decimal

import httpx

from kata.services.privy_hyperliquid_service import (
    get_privy_hyperliquid_service,
    PrivyHyperliquidService
)
from kata.services.hyperliquid_service import (
    HyperliquidService,
    create_hyperliquid_service,
    OrderSide,
    OrderType,
    OrderResult
)
from kata.services.agent_learning_service import get_agent_learning_service
from kata.services.platform_wallet_service import get_platform_wallet_service
from kata.services.ryu_spot_trading_service import get_ryu_spot_trading_service
from kata.agents.yuki_agent import YukiAgent
from kata.services.agent_database_service import get_agent_db_service
from kata.services.platform_signal_service import (
    entry_execution_block_reason,
    entry_order_deadline,
    get_platform_signal_service,
)
from kata.services.signal_edge_policy import reliable_negative_edge_veto
from kata.config.database import get_service_client
from kata.config.settings import settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Available trading agents."""
    YUKI = "yuki"  # Futures trading specialist
    SAKURA = "sakura"  # Pendle yield specialist
    RYU = "ryu"  # Spot trading specialist


class AllocationStatus(Enum):
    """Status of fund allocation to agents."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    PENDING = "pending"


@dataclass
class TradingConfig:
    """User-configurable trading parameters."""
    max_position_size_percent: float = 15.0  # % of allocation per trade
    max_leverage: float = 5.0  # Maximum leverage
    stop_loss_percent: float = 5.0  # Stop loss percentage
    take_profit_percent: float = 20.0  # Take profit percentage
    max_daily_loss_percent: float = 5.0  # Daily loss limit
    min_confidence_threshold: float = 0.7  # Minimum signal confidence
    max_trades_per_day: int = 10  # Maximum trades per day
    trading_symbols: List[str] = None  # Symbols to trade (default: ["BTC", "ETH", "SOL"])
    signal_timeframe: str = "1h"  # Signal timeframe
    risk_tolerance: str = "medium"  # low, medium, high


@dataclass
class AgentAllocation:
    """Fund allocation to a trading agent."""
    allocation_id: str
    user_id: str
    agent_type: AgentType
    allocated_amount: float  # USDC amount allocated
    remaining_amount: float  # USDC amount remaining for trading
    status: AllocationStatus
    platform_wallet_address: str
    trading_config: TradingConfig  # User-configurable trading parameters
    created_at: datetime
    last_trade_at: Optional[datetime] = None
    total_trades: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    performance_metrics: Dict[str, Any] = None


@dataclass
class AgentTradeExecution:
    """Record of agent trade execution."""
    execution_id: str
    allocation_id: str
    user_id: str
    agent_type: AgentType
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    price: float
    amount_used: float  # USDC amount used for trade
    leverage: Optional[float] = None
    executed_at: datetime = None
    hyperliquid_order_id: Optional[str] = None
    status: str = "pending"  # pending, filled, failed
    error_message: Optional[str] = None


class AgentAllocationService:
    """Service for managing fund allocations to trading agents."""

    _YUKI_SIGNAL_ATTEMPT_STATUSES = ("pending", "filled", "closed", "cancelled")
    _HYPERLIQUID_SYMBOL_ALIASES = {
        # 1. Scaled Crypto Multipliers (Binance 1000x -> HL venue names)
        "1000SHIB": "KSHIB",
        "SHIB": "KSHIB",
        "1000PEPE": "KPEPE",
        "PEPE": "KPEPE",
        "1000NEIRO": "KNEIRO",
        "NEIRO": "KNEIRO",
        "1000BONK": "KBONK",
        "BONK": "KBONK",
        "1000LUNC": "KLUNC",
        "LUNC": "KLUNC",

        # 2. Commodities & Precious Metals
        "XAU": "XYZ:GOLD",
        "GOLD": "XYZ:GOLD",
        "GC": "XYZ:GOLD",
        "XAG": "XYZ:SILVER",
        "SILVER": "XYZ:SILVER",
        "SI": "XYZ:SILVER",
        "HG": "XYZ:COPPER",
        "COPPER": "XYZ:COPPER",
        "XPT": "XYZ:PLATINUM",
        "PLATINUM": "XYZ:PLATINUM",
        "PL": "XYZ:PLATINUM",
        "CL": "XYZ:CL",
        "CRUDE": "XYZ:CL",
        "CRUDEOIL": "XYZ:CL",
        "WTI": "XYZ:CL",
        "OIL": "XYZ:CL",
        "BZ": "XYZ:BRENTOIL",
        "BRENTOIL": "XYZ:BRENTOIL",
        "BRENT": "XYZ:BRENTOIL",
        "NG": "XYZ:NATGAS",
        "NATGAS": "XYZ:NATGAS",
        "NATURALGAS": "XYZ:NATGAS",

        # 3. FX Currencies & Indices
        "EUR": "XYZ:EUR",
        "EURUSD": "XYZ:EUR",
        "EURO": "XYZ:EUR",
        "JPY": "XYZ:JPY",
        "USDJPY": "XYZ:JPY",
        "YEN": "XYZ:JPY",
        "VIX": "XYZ:VIX",
        "VOLATILITY": "XYZ:VIX",
        "NDX": "XYZ:XYZ100",
        "QQQ": "XYZ:XYZ100",
        "SPX": "XYZ:XYZ100",
        "XYZ100": "XYZ:XYZ100",

        # 4. Equities / Stocks / ETFs
        "SAMSUNG": "XYZ:SMSN",
        "SMSN": "XYZ:SMSN",
        # SOXL itself is listed on the XYZ HIP-3 DEX.  CHIP is a separate
        # main-DEX semiconductor index, so routing SOXL to CHIP changes the
        # instrument rather than merely translating its venue symbol.
        "SOXL": "XYZ:SOXL",
        "SOX": "CHIP",
        "SMH": "CHIP",
        "CHIP": "CHIP",
        "AAPL": "XYZ:AAPL",
        "APPLE": "XYZ:AAPL",
        "AMD": "XYZ:AMD",
        "AMZN": "XYZ:AMZN",
        "AMAZON": "XYZ:AMZN",
        "BABA": "XYZ:BABA",
        "ALIBABA": "XYZ:BABA",
        "COIN": "XYZ:COIN",
        "COINBASE": "XYZ:COIN",
        "COST": "XYZ:COST",
        "COSTCO": "XYZ:COST",
        "CRCL": "XYZ:CRCL",
        "CRWV": "XYZ:CRWV",
        "GOOGL": "XYZ:GOOGL",
        "GOOG": "XYZ:GOOGL",
        "GOOGLE": "XYZ:GOOGL",
        "HOOD": "XYZ:HOOD",
        "ROBINHOOD": "XYZ:HOOD",
        "INTC": "XYZ:INTC",
        "INTEL": "XYZ:INTC",
        "LLY": "XYZ:LLY",
        "META": "XYZ:META",
        "FACEBOOK": "XYZ:META",
        "MSFT": "XYZ:MSFT",
        "MICROSOFT": "XYZ:MSFT",
        "MSTR": "XYZ:MSTR",
        "MICROSTRATEGY": "XYZ:MSTR",
        "MU": "XYZ:MU",
        "MICRON": "XYZ:MU",
        "NFLX": "XYZ:NFLX",
        "NETFLIX": "XYZ:NFLX",
        "NVDA": "XYZ:NVDA",
        "NVIDIA": "XYZ:NVDA",
        "ORCL": "XYZ:ORCL",
        "ORACLE": "XYZ:ORCL",
        "PLTR": "XYZ:PLTR",
        "PALANTIR": "XYZ:PLTR",
        "RIVN": "XYZ:RIVN",
        "RIVIAN": "XYZ:RIVN",
        "SKHX": "XYZ:SKHX",
        "SNDK": "XYZ:SNDK",
        "SANDISK": "XYZ:SNDK",
        "TSLA": "XYZ:TSLA",
        "TESLA": "XYZ:TSLA",
        "TSM": "XYZ:TSM",
        "TAIWANSEMI": "XYZ:TSM",
        "USAR": "XYZ:USAR",
    }

    def __init__(self):
        """Initialize agent allocation service."""
        self.privy_hyperliquid_service = get_privy_hyperliquid_service()
        self.platform_wallet_service = get_platform_wallet_service()
        self.supabase = get_service_client()
        self.db_service = get_agent_db_service()
        self.platform_signals_service = get_platform_signal_service()
        self.ryu_spot_trading_service = get_ryu_spot_trading_service()
        self.yuki_learning_service = None
        try:
            self.yuki_learning_service = get_agent_learning_service("yuki")
        except Exception as e:
            logger.warning(f"Yuki learning service initialization deferred: {e}")

        self.hyperliquid_service = None
        try:
            self.hyperliquid_service = create_hyperliquid_service(
                privy_app_id=settings.PRIVY_APP_ID,
                privy_app_secret=settings.PRIVY_APP_SECRET,
                testnet=settings.HYPERLIQUID_TESTNET
            )
        except Exception as e:
            logger.warning(f"Hyperliquid service initialization deferred: {e}")

        # Agent instances (initialized when needed)
        self.agent_instances: Dict[str, Dict[str, Any]] = {}  # user_id -> {agent_type: agent_instance}

        # Active allocations tracking
        self.active_allocations: Dict[str, AgentAllocation] = {}  # allocation_id -> allocation
        self.yuki_signal_cycle_lock = asyncio.Lock()
        self.yuki_position_monitor = None
        self.ryu_signal_cycle_lock = asyncio.Lock()

        # Last successfully-fetched Hyperliquid balance, keyed by wallet address.
        # Used to serve a real (if slightly stale) number during rate limits
        # instead of a fabricated placeholder.
        self._last_known_balances: Dict[str, Dict[str, Any]] = {}

        logger.info("AgentAllocationService initialized with official Privy-Hyperliquid integration")

    @staticmethod
    def _is_yuki_execution_owner() -> bool:
        """Allow exactly one Railway service to own autonomous Yuki actions."""
        railway_service = str(os.getenv("RAILWAY_SERVICE_NAME") or "").strip()
        configured_owner = str(settings.YUKI_EXECUTION_SERVICE_NAME or "").strip()
        # Local tests and one-process development have no Railway service name.
        return not railway_service or not configured_owner or railway_service == configured_owner

    @staticmethod
    def _hyperliquid_symbol_candidates(symbol: Optional[str]) -> List[str]:
        """
        Candidate Hyperliquid coin names for a venue-agnostic signal symbol.

        Platform signals are often stored as Binance-style pairs such as
        BTCUSDT, while Hyperliquid orders need the coin name, e.g. BTC.
        """
        raw_symbol = str(symbol or "").strip().upper()
        if not raw_symbol:
            return []

        raw_symbol = raw_symbol.replace("/", "-").replace("_", "-")
        compact_symbol = raw_symbol.replace("-", "")
        candidates: List[str] = []

        def add_candidate(candidate: str) -> None:
            candidate = candidate.strip().upper()
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add_candidate(raw_symbol)

        for quote_suffix in ("-USDT", "-USD"):
            if raw_symbol.endswith(quote_suffix) and len(raw_symbol) > len(quote_suffix):
                add_candidate(raw_symbol[: -len(quote_suffix)])

        add_candidate(compact_symbol)

        for quote_suffix in ("USDT", "USD"):
            if compact_symbol.endswith(quote_suffix) and len(compact_symbol) > len(quote_suffix):
                add_candidate(compact_symbol[: -len(quote_suffix)])

        for candidate in list(candidates):
            alias = AgentAllocationService._HYPERLIQUID_SYMBOL_ALIASES.get(candidate)
            if alias:
                add_candidate(alias)

        return candidates

    @staticmethod
    def _signal_entry_reanalysis_pending(market_conditions: Any) -> bool:
        """Treat paused or terminal signal entries as non-executable."""
        return bool(entry_execution_block_reason(market_conditions))

    @classmethod
    def _yuki_live_entry_guard(
        cls,
        signal: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Final profit-first gate before Yuki places a live entry.
        Filters out dangerous counter-trend signals where AI direction directly opposes technical rule direction.
        """
        edge_estimate = signal.get("edge_estimate")
        if isinstance(edge_estimate, dict):
            try:
                net_ev_pct = float(edge_estimate.get("net_expected_value_pct"))
            except (TypeError, ValueError):
                net_ev_pct = None
            min_net_ev_pct = float(settings.SIGNAL_MIN_NET_EXPECTED_VALUE_PCT or 0.0)
            if net_ev_pct is not None and reliable_negative_edge_veto(
                edge_estimate,
                threshold_pct=min_net_ev_pct,
            ):
                return (
                    False,
                    "negative_net_expected_value_live_guard",
                    (
                        f"Validated signal policy has net expected value ({net_ev_pct:+.3f}%) "
                        f"at/below the live threshold ({min_net_ev_pct:+.3f}%)"
                    ),
                )

            rule_direction = str(edge_estimate.get("rule_direction") or "").strip().upper()
            ai_rule_agreement = bool(edge_estimate.get("ai_rule_agreement"))
            setup_type = str(edge_estimate.get("setup_type") or "").strip().lower()
            direction = str(
                signal.get("direction") or signal.get("signal") or ""
            ).strip().upper()
            if direction in {"BUY", "STRONG_BUY", "BULLISH"}:
                direction = "LONG"
            elif direction in {"SELL", "STRONG_SELL", "BEARISH"}:
                direction = "SHORT"

            # Veto signals that try to go counter-trend against strong technical momentum (e.g. SHORT on breakout_long)
            if not ai_rule_agreement and rule_direction in {"LONG", "SHORT"}:
                if (rule_direction == "LONG" and direction == "SHORT") or (rule_direction == "SHORT" and direction == "LONG"):
                    if any(st in setup_type for st in ["breakout", "pullback", "reversal"]):
                        return (
                            False,
                            "counter_trend_rule_conflict_guard",
                            f"Signal direction {direction} strongly opposes technical trend rule {rule_direction} on {setup_type} setup",
                        )

        return True, None, None

    @staticmethod
    def _signal_attempt_consumed(
        attempts: List[Dict[str, Any]],
        entry_revision: Any,
        reentry_attempt: Any = 0,
    ) -> Optional[Dict[str, Any]]:
        """Return an attempt for this entry revision, allowing one retry after re-arm."""
        try:
            requested_revision = max(0, int(entry_revision or 0))
        except (TypeError, ValueError):
            requested_revision = 0
        try:
            requested_reentry_attempt = max(0, int(reentry_attempt or 0))
        except (TypeError, ValueError):
            requested_reentry_attempt = 0
        requested_key = (requested_revision, requested_reentry_attempt)
        for attempt in attempts or []:
            metadata = attempt.get("trade_metadata") or attempt.get("metadata") or {}
            attempt_key = AgentAllocationService._signal_attempt_key(metadata)
            if attempt_key >= requested_key:
                return attempt
        return None

    @staticmethod
    def _signal_attempt_key(metadata: Any) -> Tuple[int, int]:
        """Stable identity for one source-signal execution attempt."""
        if not isinstance(metadata, dict):
            return (0, 0)

        try:
            entry_revision = max(0, int(metadata.get("entry_revision") or 0))
        except (TypeError, ValueError):
            entry_revision = 0
        try:
            reentry_attempt = max(0, int(metadata.get("reentry_attempt") or 0))
        except (TypeError, ValueError):
            reentry_attempt = 0
        return (entry_revision, reentry_attempt)

    @classmethod
    def _next_signal_reentry_attempt(cls, attempts: List[Dict[str, Any]]) -> int:
        """Assign the next reviewed re-entry number for this signal/allocation."""
        next_attempt = 1
        for attempt in attempts or []:
            metadata = attempt.get("trade_metadata") or attempt.get("metadata") or {}
            next_attempt = max(next_attempt, cls._signal_attempt_key(metadata)[1] + 1)
        return next_attempt

    @classmethod
    def _signal_attempt_freshness(cls, attempt: Dict[str, Any]) -> datetime:
        for field in ("closed_at", "updated_at", "filled_at", "created_at"):
            parsed = cls._parse_row_datetime(attempt.get(field))
            if parsed is not None:
                return parsed
        return datetime.min.replace(tzinfo=timezone.utc)

    @classmethod
    def _latest_signal_attempt(
        cls,
        attempts: List[Dict[str, Any]],
        statuses: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the freshest attempt for this signal, optionally filtered by status."""
        allowed_statuses = {str(status).lower() for status in (statuses or [])}
        filtered = [
            attempt for attempt in (attempts or [])
            if not allowed_statuses
            or str(attempt.get("status") or "").lower() in allowed_statuses
        ]
        if not filtered:
            return None

        return max(filtered, key=cls._signal_attempt_freshness)

    @staticmethod
    def _reentry_retest_condition_met(
        side: OrderSide,
        live_price: Optional[float],
        retest_price: Optional[float],
    ) -> bool:
        if live_price is None or retest_price is None:
            return False
        if side == OrderSide.SELL:
            return float(live_price) >= float(retest_price)
        return float(live_price) <= float(retest_price)

    @classmethod
    def _signal_reentry_review_due(
        cls,
        signal: Dict[str, Any],
        latest_closed_attempt: Dict[str, Any],
        side: OrderSide,
        live_price: Optional[float],
    ) -> bool:
        """Only re-run a re-entry review after a material new event."""
        metadata = latest_closed_attempt.get("trade_metadata") or {}
        review = metadata.get("reentry_review") or {}
        if not isinstance(review, dict):
            return True

        action = str(review.get("action") or "").strip().upper()
        if not action:
            return True

        reviewed_at = cls._parse_row_datetime(review.get("reviewed_at"))
        signal_updated_at = cls._parse_row_datetime(signal.get("updated_at"))
        if reviewed_at and signal_updated_at and signal_updated_at > reviewed_at:
            return True

        if action == "WAIT_RETEST":
            retest_price = cls._positive_float(review.get("retest_price"))
            return cls._reentry_retest_condition_met(side, live_price, retest_price)

        if action == "SKIP":
            reviewed_price = cls._positive_float(review.get("price_at_review"))
            if reviewed_price is None or live_price is None:
                return True
            move_pct = abs(float(live_price) - reviewed_price) / reviewed_price * 100.0
            return move_pct >= max(0.25, float(settings.YUKI_REACT_PRICE_MOVE_TRIGGER_PCT or 3.0))

        return True

    @classmethod
    def _normalize_reentry_hash_value(
        cls,
        value: Any,
        *,
        depth: int = 0,
    ) -> Any:
        """Create a stable, compact payload for re-entry dedupe hashes."""
        if depth >= 4:
            return "<trimmed>"
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 6)
        if isinstance(value, str):
            return " ".join(value.split())[:160]
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for key in sorted(value.keys(), key=str)[:20]:
                compact[str(key)] = cls._normalize_reentry_hash_value(value.get(key), depth=depth + 1)
            if len(value) > 20:
                compact["_truncated_keys"] = len(value) - 20
            return compact
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            compact_items = [
                cls._normalize_reentry_hash_value(item, depth=depth + 1)
                for item in items[:10]
            ]
            if len(items) > 10:
                compact_items.append({"_truncated_items": len(items) - 10})
            return compact_items
        return cls._normalize_reentry_hash_value(str(value), depth=depth + 1)

    @classmethod
    def _signal_reentry_state_hash(
        cls,
        signal: Dict[str, Any],
        latest_closed_attempt: Dict[str, Any],
        side: OrderSide,
        live_price: Optional[float],
        attempt_context: List[Dict[str, Any]],
    ) -> str:
        """Hash the materially relevant signal re-entry review state."""
        payload = {
            "signal_id": signal.get("signal_id"),
            "symbol": signal.get("symbol"),
            "side": getattr(side, "value", side),
            "signal_status": signal.get("status"),
            "signal_updated_at": signal.get("updated_at"),
            "confidence": cls._safe_float(signal.get("confidence"), 0.0),
            "entry_price": signal.get("price"),
            "stop_loss": signal.get("stop_loss"),
            "target_1": signal.get("target_1"),
            "target_2": signal.get("target_2"),
            "time_horizon": signal.get("time_horizon"),
            "live_price": live_price,
            "latest_closed_attempt": latest_closed_attempt,
            "recent_attempts": attempt_context[:3],
        }
        serialized = json.dumps(
            cls._normalize_reentry_hash_value(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def _reentry_review_cache_is_usable(
        cls,
        review: Any,
        *,
        now: datetime,
    ) -> bool:
        """Allow a cached re-entry decision only when it is still fresh."""
        if not isinstance(review, dict):
            return False
        reviewed_at = cls._parse_row_datetime(review.get("reviewed_at"))
        if reviewed_at is None:
            return False
        ttl_minutes = max(
            1,
            int(getattr(settings, "YUKI_REACT_REVIEW_CACHE_MINUTES", 20) or 20),
        )
        if (now - reviewed_at).total_seconds() > ttl_minutes * 60.0:
            return False
        action = str(review.get("action") or "").strip().upper()
        return action in {"REENTER_NOW", "WAIT_RETEST", "SKIP"}

    @classmethod
    def _signal_attempt_review_context(cls, attempt: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(attempt.get("trade_metadata") or {})
        return {
            "trade_id": attempt.get("id"),
            "status": attempt.get("status"),
            "symbol": attempt.get("symbol"),
            "side": attempt.get("side"),
            "entry_price": attempt.get("entry_price"),
            "exit_price": attempt.get("exit_price"),
            "position_size": attempt.get("position_size"),
            "trade_amount": attempt.get("trade_amount"),
            "realized_pnl": attempt.get("realized_pnl"),
            "max_profit_reached": attempt.get("max_profit_reached"),
            "max_loss_reached": attempt.get("max_loss_reached"),
            "time_in_position_minutes": attempt.get("time_in_position_minutes"),
            "created_at": attempt.get("created_at"),
            "filled_at": attempt.get("filled_at"),
            "closed_at": attempt.get("closed_at"),
            "exit_reason": metadata.get("exit_reason"),
            "time_horizon": metadata.get("time_horizon"),
            "entry_revision": metadata.get("entry_revision", 0),
            "reentry_attempt": metadata.get("reentry_attempt", 0),
            "reentry_review": metadata.get("reentry_review"),
        }

    @classmethod
    def _quote_stripped_symbol(cls, symbol: Optional[str]) -> str:
        """Best-effort fallback symbol when the Hyperliquid universe is unavailable."""
        candidates = cls._hyperliquid_symbol_candidates(symbol)
        if not candidates:
            return ""

        raw_candidate = candidates[0]
        for candidate in candidates[1:]:
            if not candidate.endswith(("USDT", "USD")):
                return candidate
        return raw_candidate

    @staticmethod
    def _hyperliquid_order_position_side(order: Dict[str, Any]) -> Optional[str]:
        """Translate a plain Hyperliquid entry order into its resulting position side."""
        raw_side = str(order.get("side") or "").strip().upper()
        if raw_side in {"B", "BUY", "BID", "LONG"}:
            return "long"
        if raw_side in {"A", "SELL", "ASK", "SHORT"}:
            return "short"
        if isinstance(order.get("isBuy"), bool):
            return "long" if order["isBuy"] else "short"
        return None

    @classmethod
    def _normalize_hyperliquid_symbol(
        cls,
        symbol: Optional[str],
        universe: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Normalize a platform signal symbol into the Hyperliquid coin name."""
        candidates = cls._hyperliquid_symbol_candidates(symbol)
        if not candidates:
            return ""

        if universe:
            listings = {
                str(listed_symbol).upper(): listing
                for listed_symbol, listing in universe.items()
            }
            source_is_dex_qualified = ":" in str(symbol or "")
            for candidate in candidates:
                if candidate in listings:
                    if ":" in candidate and not source_is_dex_qualified:
                        base_symbol = candidate.rsplit(":", 1)[-1]
                        matching_venues = {
                            listed_symbol
                            for listed_symbol in listings
                            if ":" in listed_symbol
                            and listed_symbol.rsplit(":", 1)[-1] == base_symbol
                        }
                        if len(matching_venues) > 1:
                            # An alias is not permission to choose between two
                            # venues listing the same base ticker.
                            continue
                    return str((listings[candidate] or {}).get("name") or candidate)

            # HIP-3 coins are qualified as ``dex:COIN`` while platform signals
            # remain venue-agnostic (for example DRAMUSDT). Only infer the DEX
            # when the base symbol has exactly one HIP-3 listing; ambiguous
            # tickers must be configured explicitly rather than routed blindly.
            for candidate in candidates:
                matches = [
                    str((listing or {}).get("name") or listed_symbol)
                    for listed_symbol, listing in listings.items()
                    if ":" in listed_symbol and listed_symbol.rsplit(":", 1)[-1] == candidate
                ]
                unique_matches = list(dict.fromkeys(matches))
                if len(unique_matches) == 1:
                    return unique_matches[0]

            # We had a valid universe but could not resolve one unambiguous
            # listing. Return the venue-agnostic base so the tradeability filter
            # rejects it instead of silently choosing an alias venue.
            for candidate in candidates:
                if ":" not in candidate and not candidate.endswith(("USDT", "USD")):
                    return candidate

        return cls._quote_stripped_symbol(symbol)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Parse a float from user/config/signal data without leaking bad values."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _yuki_signal_priority(cls, candidate: Dict[str, Any]) -> tuple:
        """Rank Yuki entries by opportunity rank, freshness, then confidence.

        Opportunity rank remains the strategy's primary ordering.  Within a
        rank, a newly generated thesis must be able to supersede an older
        resting entry; confidence breaks ties between signals generated at the
        same instant.
        """
        raw_rank = candidate.get("opportunity_rank")
        try:
            rank = int(raw_rank)
            if rank <= 0:
                rank = 1_000_000
        except (TypeError, ValueError):
            rank = 1_000_000

        generated_at = cls._parse_row_datetime(
            candidate.get("timestamp")
            or candidate.get("signal_generated_at")
            or candidate.get("created_at")
        )
        if generated_at is not None:
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            freshness = generated_at.timestamp()
        else:
            freshness = 0.0

        confidence = cls._safe_float(
            candidate.get("confidence", candidate.get("signal_confidence"))
        )
        if confidence > 1:
            confidence = confidence / 100
        return (rank, -freshness, -confidence)

    @classmethod
    def _yuki_signal_edge_value(cls, candidate: Dict[str, Any]) -> Optional[float]:
        """Extract the stored edge value from either a live signal or trade metadata."""
        if not isinstance(candidate, dict):
            return None

        edge_estimate = candidate.get("edge_estimate")
        if not isinstance(edge_estimate, dict):
            edge_estimate = (candidate.get("trade_metadata") or {}).get("edge_estimate")
        if not isinstance(edge_estimate, dict):
            return None

        try:
            return float(edge_estimate.get("net_expected_value_pct"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _yuki_portfolio_utility_value(cls, candidate: Dict[str, Any]) -> Optional[float]:
        if not isinstance(candidate, dict):
            return None
        selection = candidate.get("portfolio_selection")
        if not isinstance(selection, dict):
            selection = (candidate.get("trade_metadata") or {}).get("portfolio_selection")
        if not isinstance(selection, dict):
            return None
        try:
            return float(selection.get("adjusted_profit_per_risk_dollar"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _yuki_risk_bucket(cls, symbol: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Group instruments conservatively without pretending to estimate precise correlation."""
        metadata = metadata or {}
        explicit = str(metadata.get("risk_bucket") or metadata.get("asset_class") or "").strip().lower()
        if explicit:
            return explicit

        normalized = cls._quote_stripped_symbol(symbol).upper()
        if ":" in normalized:
            normalized = normalized.rsplit(":", 1)[-1]
        energy = {"CL", "OIL", "WTI", "BRENTOIL", "BRENT", "NATGAS", "NG"}
        metals = {"GOLD", "XAU", "SILVER", "XAG", "COPPER", "PLATINUM"}
        fx = {"EUR", "EURUSD", "JPY", "USDJPY"}
        indices = {"VIX", "XYZ100", "NDX", "SPX", "QQQ", "CHIP"}
        if normalized in energy:
            return "energy"
        if normalized in metals:
            return "metals"
        if normalized in fx:
            return "fx"
        if normalized in indices:
            return "equity_index"
        if ":" in str(symbol or ""):
            return "equity"

        btc_correlation = str(metadata.get("btc_correlation") or "").upper()
        if btc_correlation == "LOW":
            return f"crypto_idiosyncratic:{normalized}"
        return "crypto_beta"

    @classmethod
    def _yuki_portfolio_side(cls, candidate: Dict[str, Any]) -> str:
        raw = str(candidate.get("signal") or candidate.get("side") or "").upper()
        if raw in {"BUY", "STRONG_BUY", "LONG"}:
            return "long"
        if raw in {"SELL", "STRONG_SELL", "SHORT"}:
            return "short"
        return "unknown"

    @classmethod
    def _yuki_portfolio_candidate_metrics(
        cls,
        allocation: Any,
        signal: Dict[str, Any],
        same_direction_exposures: int,
    ) -> Dict[str, Any]:
        entry = cls._positive_float(signal.get("price")) or 0.0
        stop = cls._positive_float(signal.get("stop_loss"))
        target = cls._positive_float(signal.get("target_1"))
        leverage = max(1.0, cls._safe_float(signal.get("leverage"), 1.0))
        position_fraction = cls._yuki_signal_position_fraction(
            allocation.trading_config.max_position_size_percent,
            signal.get("position_size_percent"),
        )
        required_margin = max(0.0, cls._safe_float(allocation.allocated_amount)) * position_fraction
        notional = required_margin * leverage

        configured_stop = max(
            0.001,
            cls._safe_float(allocation.trading_config.stop_loss_percent, 5.0) / 100.0,
        )
        risk_fraction = (
            abs(entry - stop) / entry
            if entry > 0 and stop is not None
            else configured_stop
        )
        risk_usd = max(notional * risk_fraction, 1e-9)

        net_ev_pct = cls._yuki_signal_edge_value(signal)
        edge_source = "stored_net_expected_value"
        if net_ev_pct is None:
            confidence = cls._safe_float(signal.get("confidence"), 0.5)
            if confidence > 1:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))
            reward_fraction = (
                abs(target - entry) / entry
                if entry > 0 and target is not None
                else risk_fraction
            )
            net_ev_pct = (
                confidence * reward_fraction - (1.0 - confidence) * risk_fraction
            ) * 100.0
            edge_source = "confidence_geometry_fallback"

        expected_profit_usd = notional * float(net_ev_pct) / 100.0
        profit_per_risk = expected_profit_usd / risk_usd
        correlation_penalty = max(
            0.0,
            cls._safe_float(
                getattr(settings, "YUKI_CORRELATED_EXPOSURE_PENALTY", 0.35),
                0.35,
            ),
        )
        penalty_multiplier = 1.0 + correlation_penalty * max(0, same_direction_exposures)
        adjusted_utility = (
            profit_per_risk / penalty_multiplier
            if profit_per_risk >= 0
            else profit_per_risk * penalty_multiplier
        )
        return {
            "risk_bucket": cls._yuki_risk_bucket(
                signal.get("symbol"),
                signal.get("portfolio_context") or {},
            ),
            "side": cls._yuki_portfolio_side(signal),
            "required_margin_usd": round(required_margin, 6),
            "notional_usd": round(notional, 6),
            "stop_risk_usd": round(risk_usd, 6),
            "net_expected_value_pct": round(float(net_ev_pct), 6),
            "incremental_expected_profit_usd": round(expected_profit_usd, 6),
            "profit_per_risk_dollar": round(profit_per_risk, 8),
            "adjusted_profit_per_risk_dollar": round(adjusted_utility, 8),
            "same_direction_correlated_exposures": max(0, same_direction_exposures),
            "correlation_penalty_multiplier": round(penalty_multiplier, 6),
            "edge_source": edge_source,
        }

    @classmethod
    def _rank_yuki_signals_for_portfolio(
        cls,
        allocation: Any,
        signals: List[Dict[str, Any]],
        exposure_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Greedily reserve user-specific capital for the best risk-adjusted setups."""
        if not signals or not getattr(settings, "YUKI_PORTFOLIO_SELECTION_ENABLED", True):
            return signals

        exposure_counts: Dict[Tuple[str, str], int] = {}
        releasable_pending_margin = 0.0
        for row in exposure_rows:
            metadata = dict(row.get("trade_metadata") or {})
            stored_selection = metadata.get("portfolio_selection") or {}
            bucket = str(
                stored_selection.get("risk_bucket")
                or cls._yuki_risk_bucket(row.get("symbol"), metadata)
            )
            side = cls._yuki_portfolio_side(row)
            if side != "unknown":
                key = (bucket, side)
                exposure_counts[key] = exposure_counts.get(key, 0) + 1
            if str(row.get("status") or "").lower() == "pending":
                releasable_pending_margin += max(0.0, cls._safe_float(row.get("trade_amount")))

        capital_capacity = max(0.0, cls._safe_float(allocation.remaining_amount)) + releasable_pending_margin
        remaining = [dict(signal) for signal in signals]
        ranked: List[Dict[str, Any]] = []
        selection_order = 0

        while remaining:
            scored = []
            for signal in remaining:
                bucket = cls._yuki_risk_bucket(
                    signal.get("symbol"),
                    signal.get("portfolio_context") or {},
                )
                side = cls._yuki_portfolio_side(signal)
                same_direction = exposure_counts.get((bucket, side), 0)
                metrics = cls._yuki_portfolio_candidate_metrics(
                    allocation,
                    signal,
                    same_direction,
                )
                required = metrics["required_margin_usd"]
                fundable = (
                    required >= float(settings.YUKI_MIN_TRADE_USDC)
                    and required <= capital_capacity + 1e-9
                )
                scored.append((
                    0 if fundable else 1,
                    -metrics["adjusted_profit_per_risk_dollar"],
                    cls._yuki_signal_priority(signal),
                    signal,
                    metrics,
                    fundable,
                ))

            _, _, _, selected, metrics, fundable = min(
                scored,
                key=lambda item: (item[0], item[1], item[2]),
            )
            remaining.remove(selected)
            selection_order += 1
            metrics.update({
                "selection_order": selection_order,
                "fundable_from_current_and_pending_capital": fundable,
                "capital_capacity_before_selection_usd": round(capital_capacity, 6),
                "method": "expected_profit_per_stop_risk_v1",
            })
            selected["portfolio_selection"] = metrics
            selected["portfolio_priority_score"] = metrics[
                "adjusted_profit_per_risk_dollar"
            ]
            ranked.append(selected)

            if fundable:
                capital_capacity = max(
                    0.0,
                    capital_capacity - metrics["required_margin_usd"],
                )
                key = (metrics["risk_bucket"], metrics["side"])
                exposure_counts[key] = exposure_counts.get(key, 0) + 1

        return ranked

    @classmethod
    def _yuki_pending_entry_fill_distance_pct(
        cls,
        side: Any,
        limit_price: Any,
        live_price: Optional[float],
    ) -> Optional[float]:
        """
        Estimate how far a resting entry still is from filling right now.

        `0` means price is already through or at the limit. Larger values mean
        the market still needs to travel further to trigger the entry.
        """
        limit_value = cls._positive_float(limit_price)
        live_value = cls._positive_float(live_price)
        if limit_value is None or live_value is None or limit_value <= 0:
            return None

        normalized_side = str(side or "").strip().lower()
        if normalized_side in {"buy", "long"}:
            return max(0.0, (live_value - limit_value) / limit_value * 100.0)
        if normalized_side in {"sell", "short"}:
            return max(0.0, (limit_value - live_value) / limit_value * 100.0)
        return None

    @classmethod
    def _yuki_pending_entry_should_yield(
        cls,
        pending_row: Dict[str, Any],
        pending_metadata: Dict[str, Any],
        incoming_signal: Optional[Dict[str, Any]],
        *,
        live_price: Optional[float] = None,
        as_of: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Decide whether a stale resting entry should release capital for a fresh one.

        The incoming signal already passed Yuki's live-entry guard before this
        rebalance path is reached. We still keep priority ordering as the first
        rule, but allow an older pending entry to yield when it is stale,
        drifting away from fill, or aging toward expiry.
        """
        if incoming_signal is None:
            return False, None, float("-inf")

        pending_candidate = {
            "opportunity_rank": pending_metadata.get("opportunity_rank"),
            "timestamp": pending_metadata.get("signal_generated_at") or pending_row.get("created_at"),
            "confidence": pending_row.get("signal_confidence"),
        }
        pending_priority = cls._yuki_signal_priority(pending_candidate)
        incoming_priority = cls._yuki_signal_priority(incoming_signal)

        if incoming_priority < pending_priority:
            generated_at = cls._parse_row_datetime(
                pending_metadata.get("signal_generated_at") or pending_row.get("created_at")
            )
            freshness_bonus = 0.0
            if generated_at is not None:
                if generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=timezone.utc)
                now = as_of or datetime.now(timezone.utc)
                freshness_bonus = max(
                    0.0,
                    (now - generated_at).total_seconds() / 3600.0,
                )
            return True, "fresh_signal_priority", 1_000.0 + freshness_bonus

        now = as_of or datetime.now(timezone.utc)
        generated_at = cls._parse_row_datetime(
            pending_metadata.get("signal_generated_at") or pending_row.get("created_at")
        )
        expires_at = cls._parse_row_datetime(
            pending_metadata.get("entry_order_expires_at")
            or pending_metadata.get("signal_expires_at")
        )
        if generated_at is not None and generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        age_hours = None
        ttl_window_hours = None
        ttl_remaining_hours = None
        if generated_at is not None:
            age_hours = max(0.0, (now - generated_at).total_seconds() / 3600.0)
        if generated_at is not None and expires_at is not None and expires_at > generated_at:
            ttl_window_hours = max(0.0, (expires_at - generated_at).total_seconds() / 3600.0)
        if expires_at is not None:
            ttl_remaining_hours = max(0.0, (expires_at - now).total_seconds() / 3600.0)

        stale_after_hours = 3.0
        if ttl_window_hours is not None and ttl_window_hours > 0:
            stale_after_hours = max(2.0, min(8.0, ttl_window_hours * 0.35))
        stale_pending = age_hours is not None and age_hours >= stale_after_hours

        aging_out = False
        if ttl_remaining_hours is not None:
            expiry_threshold = 2.0
            if ttl_window_hours is not None and ttl_window_hours > 0:
                expiry_threshold = max(1.0, min(4.0, ttl_window_hours * 0.2))
            aging_out = ttl_remaining_hours <= expiry_threshold

        fill_distance_pct = cls._yuki_pending_entry_fill_distance_pct(
            pending_row.get("side"),
            pending_row.get("entry_price")
            or pending_metadata.get("entry_limit_price")
            or pending_metadata.get("signal_entry_price"),
            live_price,
        )
        unlikely_fill_soon = fill_distance_pct is not None and fill_distance_pct >= 0.75

        pending_confidence = cls._safe_float(
            pending_row.get("signal_confidence"),
            default=None,
        )
        incoming_confidence = cls._safe_float(
            incoming_signal.get("confidence"),
            default=None,
        )
        if pending_confidence is not None and pending_confidence > 1:
            pending_confidence = pending_confidence / 100
        if incoming_confidence is not None and incoming_confidence > 1:
            incoming_confidence = incoming_confidence / 100

        pending_edge = cls._yuki_signal_edge_value(pending_metadata)
        incoming_edge = cls._yuki_signal_edge_value(incoming_signal)
        pending_edge_non_positive = pending_edge is not None and pending_edge <= 0.0
        incoming_not_materially_worse = (
            incoming_confidence is None
            or pending_confidence is None
            or incoming_confidence + 0.05 >= pending_confidence
        )
        incoming_edge_advantage = (
            incoming_edge is not None
            and pending_edge is not None
            and incoming_edge >= pending_edge + 0.15
        )
        pending_utility = cls._yuki_portfolio_utility_value(pending_metadata)
        incoming_utility = cls._yuki_portfolio_utility_value(incoming_signal)
        incoming_utility_advantage = (
            incoming_utility is not None
            and pending_utility is not None
            and incoming_utility >= pending_utility + 0.05
        )

        if pending_edge_non_positive and (stale_pending or unlikely_fill_soon or aging_out):
            score = (
                (age_hours or 0.0)
                + (fill_distance_pct or 0.0) * 1.5
                + (4.0 if aging_out else 0.0)
            )
            return True, "stale_pending_negative_edge", score

        if stale_pending and incoming_not_materially_worse and (
            unlikely_fill_soon
            or aging_out
            or incoming_edge_advantage
            or incoming_utility_advantage
        ):
            score = (
                (age_hours or 0.0)
                + (fill_distance_pct or 0.0) * 1.5
                + (3.0 if aging_out else 0.0)
                + (2.0 if incoming_edge_advantage else 0.0)
                + (2.0 if incoming_utility_advantage else 0.0)
            )
            return True, "stale_pending_rebalanced_for_fresher_signal", score

        return False, None, float("-inf")

    @classmethod
    def _yuki_signal_position_fraction(
        cls,
        max_position_size_percent: Any,
        signal_position_percent: Any,
    ) -> float:
        """Use the signal's collateral percentage, bounded by the user's limit."""
        config_fraction = max(0.0, cls._safe_float(max_position_size_percent)) / 100
        signal_percent = cls._positive_float(signal_position_percent)
        if signal_percent is None:
            return config_fraction

        return min(config_fraction, signal_percent / 100)

    @classmethod
    def _yuki_signal_margin(
        cls,
        allocated_amount: Any,
        realized_pnl: Any,
        available_amount: Any,
        max_position_size_percent: Any,
        signal_position_percent: Any,
    ) -> float:
        """Size every signal from compounded allocation equity.

        ``allocated_amount`` is capital-accounting v2 equity: principal plus
        cumulative realized P&L.  ``realized_pnl`` remains an attribution field
        for the dashboard and must not be added a second time.
        """
        equity = max(0.0, cls._safe_float(allocated_amount))
        available = max(0.0, cls._safe_float(available_amount))
        fraction = cls._yuki_signal_position_fraction(
            max_position_size_percent=max_position_size_percent,
            signal_position_percent=signal_position_percent,
        )
        requested = equity * fraction
        return requested if requested <= available + 1e-9 else 0.0

    @classmethod
    def _yuki_improved_entry_price(
        cls,
        entry_side: OrderSide,
        signal_entry_price: Any,
        live_price: Any,
        stop_loss: Any,
        target_1: Any,
        improvement_fraction: Any,
        target_2: Any = None,
    ) -> Optional[float]:
        """Move a resting entry partway toward market without crossing SL/TP bounds,
        and strictly reject signals where live market price has already hit target_1, target_2, or stop_loss."""
        ideal_entry = cls._positive_float(signal_entry_price)
        market_price = cls._positive_float(live_price)
        if ideal_entry is None:
            ideal_entry = market_price
        if ideal_entry is None:
            return None

        fraction = max(0.0, min(1.0, cls._safe_float(improvement_fraction, 0.0)))
        candidate = ideal_entry
        if market_price is not None:
            candidate = ideal_entry + (market_price - ideal_entry) * fraction

        stop = cls._positive_float(stop_loss)
        target1 = cls._positive_float(target_1)
        target2 = cls._positive_float(target_2)

        # 1. Strict live market price check: skip entry if target or stop HAS ALREADY BEEN HIT in real price action
        if market_price is not None:
            if entry_side == OrderSide.BUY:
                if target1 is not None and market_price >= target1:
                    logger.info(f"Skipping BUY signal: live market price ${market_price:.4f} has already reached/passed target_1 ${target1:.4f}")
                    return None
                if target2 is not None and market_price >= target2:
                    logger.info(f"Skipping BUY signal: live market price ${market_price:.4f} has already reached/passed target_2 ${target2:.4f}")
                    return None
                if stop is not None and market_price <= stop:
                    logger.info(f"Skipping BUY signal: live market price ${market_price:.4f} has already reached/passed stop_loss ${stop:.4f}")
                    return None
            else:
                # SELL / SHORT
                if target1 is not None and market_price <= target1:
                    logger.info(f"Skipping SELL signal: live market price ${market_price:.4f} has already reached/passed target_1 ${target1:.4f}")
                    return None
                if target2 is not None and market_price <= target2:
                    logger.info(f"Skipping SELL signal: live market price ${market_price:.4f} has already reached/passed target_2 ${target2:.4f}")
                    return None
                if stop is not None and market_price >= stop:
                    logger.info(f"Skipping SELL signal: live market price ${market_price:.4f} has already reached/passed stop_loss ${stop:.4f}")
                    return None

        # 2. Candidate limit price bounds check
        if entry_side == OrderSide.BUY:
            if stop is not None and candidate <= stop:
                return None
            if target1 is not None and candidate >= target1:
                return None
        else:
            if stop is not None and candidate >= stop:
                return None
            if target1 is not None and candidate <= target1:
                return None
        return candidate

    @classmethod
    def _yuki_live_geometry_terminal_event(
        cls,
        entry_side: OrderSide,
        live_price: Any,
        stop_loss: Any,
        target_1: Any,
        target_2: Any = None,
    ) -> Optional[str]:
        """Classify which stored boundary the live market has already crossed."""
        market_price = cls._positive_float(live_price)
        if market_price is None:
            return None
        stop = cls._positive_float(stop_loss)
        target1 = cls._positive_float(target_1)
        target2 = cls._positive_float(target_2)
        if entry_side == OrderSide.BUY:
            if target2 is not None and market_price >= target2:
                return "target_2_touched_before_entry"
            if target1 is not None and market_price >= target1:
                return "target_1_touched_before_entry"
            if stop is not None and market_price <= stop:
                return "stop_loss_touched_before_entry"
        else:
            if target2 is not None and market_price <= target2:
                return "target_2_touched_before_entry"
            if target1 is not None and market_price <= target1:
                return "target_1_touched_before_entry"
            if stop is not None and market_price >= stop:
                return "stop_loss_touched_before_entry"
        return None

    @staticmethod
    def _yuki_entry_order_expires_at(
        signal_started_at: Any,
        signal_expires_at: Any,
        ttl_hours: Any,
        now: Optional[datetime] = None,
    ) -> datetime:
        """Cap entry freshness at the configured TTL and validity fraction."""
        return entry_order_deadline(
            signal_started_at=signal_started_at,
            signal_expires_at=signal_expires_at,
            max_ttl_hours=AgentAllocationService._safe_float(ttl_hours, 12.0),
            validity_fraction=float(
                getattr(settings, "YUKI_ENTRY_TTL_VALIDITY_FRACTION", 0.50) or 0.50
            ),
            now=now,
        )

    async def _yuki_signal_path_terminal_event(
        self,
        signal: Dict[str, Any],
        symbol: str,
        side: OrderSide,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Check venue candles from signal creation to order submission.

        Current price alone misses a target/stop that was crossed and reversed.
        Any historical touch makes the old levels stale and forces a complete
        re-analysis before another entry can be submitted.
        """
        info_client = getattr(self.hyperliquid_service, "info_client", None)
        if not info_client:
            return None

        current_time = now or datetime.now(timezone.utc)
        started_at = self._parse_row_datetime(signal.get("timestamp"))
        if not started_at:
            return None
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if current_time <= started_at:
            return None

        stop = self._positive_float(signal.get("stop_loss"))
        target_1 = self._positive_float(signal.get("target_1"))
        target_2 = self._positive_float(signal.get("target_2"))
        if not any((stop, target_1, target_2)):
            return None

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": "1m",
                "startTime": int(started_at.timestamp() * 1000),
                "endTime": int(current_time.timestamp() * 1000),
            },
        }
        try:
            candles = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: info_client.post("/info", payload),
            )
        except Exception as exc:
            # A venue-history outage is not proof that the signal is invalid.
            logger.warning("Could not validate pre-entry path for %s: %s", symbol, exc)
            return None

        entry_price = self._positive_float(signal.get("price"))
        entry_touched = False

        for candle in sorted(candles or [], key=lambda item: int(item.get("t") or 0)):
            try:
                high = float(candle.get("h") or 0.0)
                low = float(candle.get("l") or 0.0)
            except (TypeError, ValueError):
                continue
            if side == OrderSide.BUY:
                if entry_price and low <= entry_price:
                    entry_touched = True
                if entry_touched:
                    if target_2 and high >= target_2:
                        return "target_2_touched_after_entry"
                    if target_1 and high >= target_1:
                        return "target_1_touched_after_entry"
                    if stop and low <= stop:
                        return "stop_loss_touched_after_entry"
            else:
                if entry_price and high >= entry_price:
                    entry_touched = True
                if entry_touched:
                    if target_2 and low <= target_2:
                        return "target_2_touched_after_entry"
                    if target_1 and low <= target_1:
                        return "target_1_touched_after_entry"
                    if stop and high >= stop:
                        return "stop_loss_touched_after_entry"
        return None

    async def _queue_yuki_signal_reanalysis(
        self,
        signal: Dict[str, Any],
        reason: str,
        observed_price: Optional[float] = None,
        *,
        expire_entry_order: bool = True,
        invalidate_pending_entry: bool = False,
    ) -> None:
        signal_id = signal.get("signal_id")
        if not signal_id:
            return
        await self.platform_signals_service.request_signal_reanalysis(
            str(signal_id),
            reason,
            observed_price=observed_price,
            expire_entry_order=expire_entry_order,
            invalidate_pending_entry=invalidate_pending_entry,
            source="yuki_order_submission",
        )

    def _persist_signal_reentry_review(
        self,
        attempt: Dict[str, Any],
        review_record: Dict[str, Any],
    ) -> None:
        """Store the latest Yuki re-entry decision on the closed trade attempt."""
        trade_id = attempt.get("id")
        if not trade_id:
            return

        metadata = dict(attempt.get("trade_metadata") or {})
        metadata["reentry_review"] = review_record
        metadata["reentry_review_state_hash"] = review_record.get("state_hash")
        try:
            self.supabase.table("agent_trades").update({
                "trade_metadata": metadata,
            }).eq("id", trade_id).execute()
        except Exception as exc:
            logger.warning("Could not persist Yuki re-entry review for trade %s: %s", trade_id, exc)

    async def _maybe_prepare_yuki_signal_reentry(
        self,
        user_id: str,
        allocation: AgentAllocation,
        yuki_agent: YukiAgent,
        signal: Dict[str, Any],
        side: OrderSide,
        signal_attempts: List[Dict[str, Any]],
        live_price: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Ask Yuki whether an active signal deserves a second entry attempt."""
        if self._safe_float(signal.get("reentry_attempt")) > 0:
            return signal

        closed_attempt = self._latest_signal_attempt(signal_attempts, statuses=["closed"])
        if not closed_attempt:
            return signal

        signal_id = str(signal.get("signal_id") or "")
        if not self._signal_reentry_review_due(signal, closed_attempt, side, live_price):
            review = (closed_attempt.get("trade_metadata") or {}).get("reentry_review") or {}
            logger.info(
                "Skipping signal %s for allocation %s: prior Yuki re-entry review remains active (%s)",
                signal_id or signal.get("symbol"),
                allocation.allocation_id,
                str(review.get("action") or "WAIT").upper(),
            )
            return None

        attempt_context = [
            self._signal_attempt_review_context(attempt)
            for attempt in sorted(
                signal_attempts,
                key=self._signal_attempt_freshness,
                reverse=True,
            )[:4]
        ]
        latest_closed_context = self._signal_attempt_review_context(closed_attempt)
        reentry_attempt = self._next_signal_reentry_attempt(signal_attempts)
        review_state_hash = self._signal_reentry_state_hash(
            signal,
            latest_closed_context,
            side,
            live_price,
            attempt_context,
        )
        prior_metadata = closed_attempt.get("trade_metadata") or {}
        prior_review = prior_metadata.get("reentry_review") or {}
        prior_state_hash = str(prior_metadata.get("reentry_review_state_hash") or "")
        now = datetime.now(timezone.utc)
        if (
            prior_state_hash
            and prior_state_hash == review_state_hash
            and self._reentry_review_cache_is_usable(prior_review, now=now)
        ):
            logger.info(
                "Skipping signal %s for allocation %s: Yuki re-entry state hash unchanged",
                signal_id or signal.get("symbol"),
                allocation.allocation_id,
            )
            return None

        review = await yuki_agent.execute_signal_reentry_react_cycle(
            signal_context={
                "allocation_id": allocation.allocation_id,
                "user_id": user_id,
                "symbol": signal.get("symbol"),
                "signal": signal,
                "live_price": live_price,
                "latest_closed_attempt": latest_closed_context,
                "signal_attempts": attempt_context,
            },
            trigger_context={
                "event": "active_signal_reentry_candidate",
                "signal_id": signal_id,
                "prior_trade_closed": True,
                "prior_trade_id": closed_attempt.get("id"),
            },
        )

        action = str((review or {}).get("action") or "SKIP").strip().upper()
        confidence = max(0.0, min(1.0, self._safe_float((review or {}).get("confidence"), 0.0)))
        retest_price = self._positive_float((review or {}).get("retest_price"))
        if action == "WAIT_RETEST" and retest_price is None:
            signal_entry = self._positive_float(signal.get("price"))
            if live_price is not None and signal_entry is not None:
                retest_price = (
                    max(signal_entry, live_price)
                    if side == OrderSide.SELL
                    else min(signal_entry, live_price)
                )
            else:
                retest_price = signal_entry or live_price

        review_record = {
            "action": action,
            "confidence": confidence,
            "retest_price": retest_price,
            "reviewed_at": now.isoformat(),
            "price_at_review": live_price,
            "market_regime": str((review or {}).get("market_regime") or "uncertain"),
            "reasoning": str((review or {}).get("reasoning") or ""),
            "evidence_used": list((review or {}).get("evidence_used") or []),
            "next_review_conditions": list((review or {}).get("next_review_conditions") or []),
            "approved_reentry_attempt": reentry_attempt,
            "source_trade_id": closed_attempt.get("id"),
            "llm_usage": (review or {}).get("llm_usage"),
            "llm_retry_used": bool((review or {}).get("llm_retry_used")),
            "state_hash": review_state_hash,
        }
        self._persist_signal_reentry_review(closed_attempt, review_record)

        if action != "REENTER_NOW":
            logger.info(
                "Yuki re-entry review skipped signal %s for allocation %s with action %s",
                signal_id or signal.get("symbol"),
                allocation.allocation_id,
                action,
            )
            return None

        reviewed_signal = dict(signal)
        reviewed_signal["reentry_attempt"] = reentry_attempt
        reviewed_signal["reentry_review"] = review_record
        return reviewed_signal

    @classmethod
    def _confidence_trade_scaling(cls, confidence: float) -> Dict[str, float]:
        """
        Risk curve for Yuki execution.

        Higher-confidence signals earn more margin and leverage, while weaker
        accepted signals are deliberately smaller. The leverage ceiling is a
        strategy cap; user config and Hyperliquid asset caps still apply later.
        """
        confidence = max(0.0, min(1.0, cls._safe_float(confidence)))

        if confidence >= 0.92:
            return {
                "position_multiplier": 1.60,
                "target_leverage": 20.0,
                "max_leverage": 40.0,
            }
        if confidence >= 0.85:
            return {
                "position_multiplier": 1.35,
                "target_leverage": 12.0,
                "max_leverage": 20.0,
            }
        if confidence >= 0.75:
            return {
                "position_multiplier": 1.10,
                "target_leverage": 6.0,
                "max_leverage": 10.0,
            }
        if confidence >= 0.65:
            return {
                "position_multiplier": 0.85,
                "target_leverage": 3.0,
                "max_leverage": 5.0,
            }

        return {
            "position_multiplier": 0.55,
            "target_leverage": 1.0,
            "max_leverage": 2.0,
        }

    @classmethod
    def _confidence_adjusted_leverage(
        cls,
        requested_leverage: float,
        max_config_leverage: float,
        confidence: float,
    ) -> float:
        """Resolve leverage from signal request, confidence tier, and user cap."""
        scaling = cls._confidence_trade_scaling(confidence)
        requested_leverage = max(1.0, cls._safe_float(requested_leverage, 1.0))
        max_config_leverage = max(1.0, cls._safe_float(max_config_leverage, 1.0))
        confidence_leverage = max(requested_leverage, scaling["target_leverage"])
        return min(confidence_leverage, scaling["max_leverage"], max_config_leverage)

    @classmethod
    def _confidence_adjusted_position_fraction(
        cls,
        max_position_size_percent: float,
        signal_position_percent: Optional[float],
        confidence: float,
    ) -> float:
        """
        Resolve margin fraction before the final confidence multiplier.

        Signal-provided sizing can be boosted for high confidence, but never
        exceeds the user's configured max position cap.
        """
        scaling = cls._confidence_trade_scaling(confidence)
        config_fraction = max(0.0, cls._safe_float(max_position_size_percent)) / 100

        if signal_position_percent is not None:
            signal_fraction = max(0.0, cls._safe_float(signal_position_percent)) / 100
            adjusted_signal_fraction = signal_fraction * scaling["position_multiplier"]
            return min(config_fraction, adjusted_signal_fraction)

        # Without a signal-level size, only reduce weak signals; do not boost
        # above the configured cap from confidence alone.
        return config_fraction * min(1.0, scaling["position_multiplier"])

    @staticmethod
    def _is_hyperliquid_rate_limit_error(error: Exception) -> bool:
        """Recognize both httpx and Hyperliquid SDK representations of HTTP 429."""
        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        if getattr(error, "status_code", None) == 429 or getattr(error, "status", None) == 429:
            return True
        if error.args and error.args[0] == 429:
            return True
        return "429" in str(error).lower()

    async def get_yuki_trading_readiness(self, user_id: str, wallet_address: str) -> Dict[str, Any]:
        """Check whether a user's Hyperliquid account is funded and delegated for Yuki."""
        blocking_reasons = []
        balance_usdc = 0.0
        withdrawable_usdc = 0.0
        hyperliquid_account_active = False
        balance_is_cached = False
        balance_available = False
        cache_key = (wallet_address or "").lower()

        try:
            if not wallet_address:
                blocking_reasons.append("No wallet address found for this user")
            else:
                streamed = None
                account_mode = None
                if self.hyperliquid_service:
                    account_mode = await self.hyperliquid_service.get_user_abstraction_mode(wallet_address)
                from kata.services.hyperliquid_account_stream_service import get_hyperliquid_account_stream_service

                stream_service = get_hyperliquid_account_stream_service()
                await stream_service.ensure_stream(wallet_address, account_mode=account_mode)
                if account_mode in {"unifiedAccount", "portfolioMargin"}:
                    streamed = stream_service.get_cached_spot_balance(wallet_address)
                    if streamed is None and self.hyperliquid_service:
                        unified = await self.hyperliquid_service.get_wallet_balance_summary(
                            wallet_address,
                            account_mode=account_mode,
                        )
                        streamed = {
                            "balance_usdc": float(unified.get("balance") or 0),
                            "withdrawable_usdc": float(unified.get("withdrawable") or 0),
                            "fetched_at": datetime.now(),
                        }
                elif account_mode is None:
                    # A rate-limited account-mode request must never make a unified
                    # wallet look empty. Prefer the last verified reading; otherwise
                    # show the larger live pushed state but keep readiness blocked
                    # until the mode itself is verified.
                    streamed = self._last_known_balances.get(cache_key)
                    if streamed:
                        balance_is_cached = True
                    else:
                        spot_streamed = stream_service.get_cached_spot_balance(wallet_address)
                        perp_streamed = stream_service.get_cached_state(wallet_address)
                        candidates = [value for value in (spot_streamed, perp_streamed) if value]
                        if candidates:
                            streamed = max(candidates, key=lambda value: value["balance_usdc"])
                    if streamed:
                        blocking_reasons.append(
                            "Hyperliquid account mode could not be verified; using the latest balance while Yuki retries"
                        )
                else:
                    streamed = stream_service.get_cached_state(wallet_address)

                if streamed:
                    hyperliquid_account_active = True
                    balance_available = True
                    balance_usdc = streamed["balance_usdc"]
                    withdrawable_usdc = streamed["withdrawable_usdc"]
                    self._last_known_balances[cache_key] = streamed
                else:
                    if account_mode is None:
                        raise RuntimeError("Hyperliquid account mode is temporarily unavailable")
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.post(
                            "https://api.hyperliquid.xyz/info",
                            json={"type": "clearinghouseState", "user": wallet_address},
                        )
                        response.raise_for_status()
                        user_state = response.json() or {}

                    if user_state:
                        hyperliquid_account_active = True
                        balance_available = True
                        cross_margin_summary = user_state.get("crossMarginSummary", {}) or {}
                        balance_usdc = float(
                            cross_margin_summary.get("accountValue")
                            or user_state.get("withdrawable")
                            or 0
                        )
                        withdrawable_usdc = float(user_state.get("withdrawable") or 0)
                        # Cache the real reading so a later rate limit can serve a true
                        # (if slightly stale) number instead of a fabricated one.
                        self._last_known_balances[cache_key] = {
                            "balance_usdc": balance_usdc,
                            "withdrawable_usdc": withdrawable_usdc,
                            "fetched_at": datetime.now(),
                        }
                    else:
                        blocking_reasons.append("No Hyperliquid account state found for wallet")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                cached = self._last_known_balances.get(cache_key)
                if cached:
                    logger.warning(
                        f"Hyperliquid rate limit hit for {user_id} readiness check; "
                        f"serving cached balance from {cached['fetched_at'].isoformat()}"
                    )
                    hyperliquid_account_active = True
                    balance_available = True
                    balance_usdc = cached["balance_usdc"]
                    withdrawable_usdc = cached["withdrawable_usdc"]
                    balance_is_cached = True
                else:
                    logger.warning(f"Hyperliquid rate limit hit for {user_id} readiness check; no cached balance available")
                    blocking_reasons.append("Hyperliquid is rate-limiting balance checks; try again shortly")
            else:
                logger.warning(f"Could not check Hyperliquid readiness for {user_id}: {e}")
                blocking_reasons.append(f"Hyperliquid readiness check failed: {e}")
        except Exception as e:
            cached = self._last_known_balances.get(cache_key)
            if self._is_hyperliquid_rate_limit_error(e):
                if cached:
                    logger.warning(
                        f"Hyperliquid SDK rate limit hit for {user_id}; serving cached "
                        f"balance from {cached['fetched_at'].isoformat()}"
                    )
                    hyperliquid_account_active = True
                    balance_available = True
                    balance_usdc = cached["balance_usdc"]
                    withdrawable_usdc = cached["withdrawable_usdc"]
                    balance_is_cached = True
                else:
                    logger.warning(f"Hyperliquid rate limit hit for {user_id}; no cached balance available")
                    blocking_reasons.append("Hyperliquid is rate-limiting balance checks; try again shortly")
            else:
                logger.warning(f"Could not check Hyperliquid readiness for {user_id}: {e}")
                blocking_reasons.append("Hyperliquid balance is temporarily unavailable; try again shortly")

        delegated = not settings.YUKI_REQUIRE_DELEGATION
        if settings.YUKI_REQUIRE_DELEGATION:
            try:
                from kata.services.delegation_service import get_delegation_service

                delegated = await get_delegation_service().has_delegation(user_id)
                if not delegated:
                    blocking_reasons.append("No active wallet delegation found for automated trading")
            except Exception as e:
                logger.warning(f"Could not check delegation readiness for {user_id}: {e}")
                blocking_reasons.append(f"Delegation readiness check failed: {e}")

        if balance_available and balance_usdc < settings.YUKI_MIN_ALLOCATION_USDC:
            blocking_reasons.append(
                f"Hyperliquid balance is below the minimum ${settings.YUKI_MIN_ALLOCATION_USDC:.2f} USDC allocation"
            )

        return {
            "success": True,
            "user_id": user_id,
            "wallet_address": wallet_address,
            "hyperliquid_account_active": hyperliquid_account_active,
            "balance_usdc": balance_usdc,
            "withdrawable_usdc": withdrawable_usdc,
            "balance_available": balance_available,
            "balance_is_cached": balance_is_cached,
            "funding_needed": max(0.0, settings.YUKI_MIN_ALLOCATION_USDC - balance_usdc),
            "live_trading_enabled": settings.YUKI_LIVE_TRADING_ENABLED,
            "delegation_required": settings.YUKI_REQUIRE_DELEGATION,
            "delegated": delegated,
            "testnet": settings.HYPERLIQUID_TESTNET,
            "ready_for_trading": len(blocking_reasons) == 0,
            "blocking_reasons": blocking_reasons,
        }

    async def _verify_ryu_funding_transfer(
        self,
        user_id: str,
        base_wallet_address: str,
        solana_wallet_address: str,
        funding_tx_hash: str,
        requested_received_usdc: float,
        input_usdc: float,
        quote_id: Optional[str],
    ) -> Dict[str, Any]:
        """Validate a single-use Base-USDC to Solana-USDC LI.FI transfer."""
        if input_usdc < requested_received_usdc:
            return {
                "success": False,
                "error": "Ryu funding output cannot exceed the Base USDC input",
            }
        try:
            existing = (
                self.supabase.table("ryu_funding_transfers")
                .select("id")
                .eq("source_tx_hash", funding_tx_hash)
                .limit(1)
                .execute()
            )
            if existing.data:
                return {
                    "success": False,
                    "error": "This Ryu funding transaction has already been allocated",
                }
        except Exception as e:
            logger.error(f"Ryu funding evidence table is unavailable: {e}")
            return {
                "success": False,
                "error": (
                    "Ryu funding storage is not ready. Apply "
                    "backend/migrations/add_ryu_multichain_funding.sql."
                ),
            }

        status = await self.ryu_spot_trading_service.lifi_service.get_transfer_status(
            tx_hash=funding_tx_hash,
            from_chain_id=8453,
            to_chain_id=1151111081099710,
        )
        if not status or str(status.get("status") or "").upper() != "DONE":
            return {
                "success": False,
                "error": (
                    "The Base-to-Solana funding route is still settling. "
                    "Wait for it to complete, then retry activation."
                ),
            }

        sending = status.get("sending") or {}
        receiving = status.get("receiving") or {}
        sending_token = sending.get("token") or {}
        receiving_token = receiving.get("token") or {}
        sending_chain = int(sending.get("chainId") or 0)
        receiving_chain = int(receiving.get("chainId") or 0)
        if sending_chain != 8453 or receiving_chain != 1151111081099710:
            return {
                "success": False,
                "error": "Funding evidence is not a Base-to-Solana transfer",
            }
        if (
            str(sending_token.get("address") or "").lower()
            != "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
            or str(receiving_token.get("address") or "")
            != "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        ):
            return {
                "success": False,
                "error": "Funding evidence does not settle Base USDC as Solana USDC",
            }

        receiving_address = str(
            receiving.get("toAddress")
            or receiving.get("to_address")
            or ""
        )
        if receiving_address and receiving_address != solana_wallet_address:
            return {
                "success": False,
                "error": "Funding settled to a different Solana wallet",
            }
        sending_address = str(
            sending.get("fromAddress")
            or sending.get("from_address")
            or ""
        )
        if (
            sending_address
            and sending_address.lower() != base_wallet_address.lower()
        ):
            return {
                "success": False,
                "error": "Funding originated from a different Base wallet",
            }

        decimals = int(receiving_token.get("decimals") or 6)
        received_usdc = int(receiving.get("amount") or 0) / (10 ** decimals)
        if received_usdc <= 0 or requested_received_usdc > received_usdc + 0.01:
            return {
                "success": False,
                "error": (
                    f"LI.FI confirms ${received_usdc:.2f} Solana USDC, below "
                    f"the requested ${requested_received_usdc:.2f} allocation"
                ),
            }

        readiness = await self.ryu_spot_trading_service.get_trading_readiness(
            user_id=user_id,
            ethereum_wallet_address=base_wallet_address,
            solana_wallet_address=solana_wallet_address,
        )
        if float(readiness.get("solana_usdc_balance") or 0) + 0.01 < requested_received_usdc:
            return {
                "success": False,
                "error": "The confirmed Solana USDC has not reached Ryu's wallet yet",
            }
        return {
            "success": True,
            "received_usdc": received_usdc,
            "quote_id": quote_id,
        }

    @staticmethod
    def _ryu_uses_base_home(allocation_row: Dict[str, Any]) -> bool:
        metrics = allocation_row.get("performance_metrics") or {}
        funding = metrics.get("funding") or {}
        capital = metrics.get("capital") or {}
        return (
            str(funding.get("home_chain") or "").lower() == "base"
            or int(capital.get("accounting_version") or 0) >= 3
        )

    def _base_reserved_usdc(self, user_id: str) -> float:
        """Return Base USDC reserved by active Sakura and Base-home Ryu rows."""
        rows = (
            self.supabase.table("agent_allocations")
            .select(
                "agent_type,allocated_amount,remaining_amount,"
                "performance_metrics,status"
            )
            .eq("user_id", user_id)
            .in_(
                "status",
                [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value],
            )
            .execute()
        ).data or []
        reserved = 0.0
        for row in rows:
            agent_type = str(row.get("agent_type") or "").lower()
            if agent_type == AgentType.SAKURA.value:
                reserved += float(row.get("allocated_amount") or 0)
            elif (
                agent_type == AgentType.RYU.value
                and self._ryu_uses_base_home(row)
            ):
                reserved += float(row.get("remaining_amount") or 0)
        return reserved

    async def allocate_funds_to_agent(
        self,
        user_id: str,
        agent_type: AgentType,
        amount: float,
        auto_trading: bool = True,
        trading_config: Optional[Dict[str, Any]] = None,
        access_token: Optional[str] = None,
        wallet_address: Optional[str] = None,
        solana_wallet_address: Optional[str] = None,
        funding_tx_hash: Optional[str] = None,
        funding_quote_id: Optional[str] = None,
        funding_input_usdc: Optional[float] = None,
        base_gas_tx_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Allocate funds from user's platform wallet to a trading agent.

        Args:
            user_id: User identifier
            agent_type: Type of agent to allocate to
            amount: USDC amount to allocate
            auto_trading: Whether to enable automatic trading

        Returns:
            Allocation details and status
        """
        try:
            minimum_allocation = (
                settings.RYU_MIN_ALLOCATION_USDC
                if agent_type == AgentType.RYU
                else settings.YUKI_MIN_ALLOCATION_USDC
            )
            if amount < minimum_allocation:
                return {
                    "success": False,
                    "error": f"Minimum allocation is ${minimum_allocation:.2f} USDC"
                }

            live_trading_ready = False
            readiness = {}
            live_trading_requested = auto_trading
            ryu_funding: Optional[Dict[str, Any]] = None

            if agent_type == AgentType.YUKI:
                readiness = await self.get_yuki_trading_readiness(
                    user_id=user_id,
                    wallet_address=wallet_address or ""
                )

                balance_usdc = float(readiness.get("balance_usdc", 0.0))
                existing_result = (
                    self.supabase.table("agent_allocations")
                    .select("allocated_amount")
                    .eq("user_id", user_id)
                    .eq("agent_type", AgentType.YUKI.value)
                    .in_("status", [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value])
                    .execute()
                )
                existing_allocated = sum(float(row.get("allocated_amount") or 0.0) for row in (existing_result.data or []))
                available_to_allocate = max(0.0, balance_usdc - existing_allocated)
                if available_to_allocate < amount:
                    # Quotes drift a little between funding preview and the actual
                    # Hyperliquid credit (slippage), so clamp small shortfalls to
                    # the real balance instead of failing the allocation.
                    if (
                        available_to_allocate >= settings.YUKI_MIN_ALLOCATION_USDC
                        and amount <= available_to_allocate * 1.05
                    ):
                        clamped = int(available_to_allocate * 100) / 100
                        logger.info(
                            f"Clamping requested allocation ${amount:.2f} to available "
                            f"unallocated Hyperliquid balance ${clamped:.2f} for user {user_id}"
                        )
                        amount = clamped
                    else:
                        return {
                            "success": False,
                            "error": (
                                f"Unallocated Hyperliquid USDC balance (${available_to_allocate:.2f}) "
                                f"is below requested allocation (${amount:.2f})."
                            ),
                            "readiness": readiness
                        }

                if live_trading_requested:
                    if not settings.YUKI_LIVE_TRADING_ENABLED:
                        logger.info("Yuki allocation will be saved paused because live trading is disabled")
                    elif not readiness.get("ready_for_trading"):
                        return {
                            "success": False,
                            "error": "; ".join(readiness.get("blocking_reasons", [])) or "Yuki trading is not ready",
                            "readiness": readiness
                        }
                    elif not self.hyperliquid_service:
                        return {"success": False, "error": "Hyperliquid service is not available for live trading"}
                    else:
                        # Trading access comes from the wallet delegation record,
                        # not the browser session - workers use the same path.
                        client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                            user_id=user_id,
                            wallet_address=wallet_address or readiness.get("wallet_address") or "",
                        )
                        if not client_ready:
                            return {
                                "success": False,
                                "error": (
                                    "Could not set up delegated Hyperliquid trading for this wallet. "
                                    "Re-run wallet delegation and try again."
                                ),
                                "readiness": readiness
                            }
                        live_trading_ready = True

                        # One-time builder fee approval so orders can carry the
                        # platform fee. Never blocks trading if it fails.
                        try:
                            approval_wallet = wallet_address or readiness.get("wallet_address") or ""
                            if approval_wallet:
                                await self.hyperliquid_service.ensure_builder_fee_approved(approval_wallet)
                        except Exception as e:
                            logger.warning(f"Builder fee approval skipped for user {user_id}: {e}")
                            
            elif agent_type == AgentType.RYU:
                if not wallet_address or not solana_wallet_address:
                    return {
                        "success": False,
                        "error": (
                            "Ryu's Base and Solana wallets must be ready before activation."
                        ),
                    }
                readiness = await self.ryu_spot_trading_service.get_trading_readiness(
                    user_id=user_id,
                    ethereum_wallet_address=wallet_address or "",
                    solana_wallet_address=solana_wallet_address or "",
                )
                base_balance_usdc = float(
                    readiness.get("base_usdc_balance") or 0
                )
                available_to_allocate = max(
                    0.0,
                    base_balance_usdc - self._base_reserved_usdc(user_id),
                )
                if available_to_allocate < amount:
                    if (
                        available_to_allocate >= settings.RYU_MIN_ALLOCATION_USDC
                        and amount <= available_to_allocate * 1.02
                    ):
                        amount = int(available_to_allocate * 100) / 100
                    else:
                        return {
                            "success": False,
                            "error": (
                                f"Only ${available_to_allocate:.2f} unallocated "
                                "Base USDC is available."
                            ),
                            "readiness": readiness,
                        }
                if live_trading_requested:
                    if not settings.RYU_LIVE_TRADING_ENABLED:
                        logger.info(
                            "Ryu allocation will be saved paused because live trading is disabled"
                        )
                    elif not readiness.get("ready_for_trading"):
                        return {
                            "success": False,
                            "error": (
                                "; ".join(readiness.get("blocking_reasons", []))
                                or "Ryu trading is not ready"
                            ),
                            "readiness": readiness,
                        }
                    else:
                        live_trading_ready = True
            else:
                # Sakura retains the standard platform-wallet allocation path.
                wallet_balance = await self.platform_wallet_service.get_user_balance(user_id, "USDC")
                available_to_allocate = (
                    max(
                        0.0,
                        float(wallet_balance.total_balance)
                        - self._base_reserved_usdc(user_id),
                    )
                    if wallet_balance
                    else 0.0
                )
                if available_to_allocate < amount:
                    return {
                        "success": False,
                        "error": (
                            f"Only ${available_to_allocate:.2f} unallocated platform wallet USDC is available. "
                            f"Requested ${amount:.2f}."
                        ),
                    }
                
                live_trading_ready = auto_trading

            # Step 2: Determine trading configuration approach
            if trading_config:
                # User provided custom config - "Configure Yuki"
                # Use user overrides with platform fallbacks for missing values
                logger.info("Using platform default config for Configure Yuki mode")
                default_config = {
                    "max_position_size_percent": 15.0,  # Conservative default for user configuration
                    "max_leverage": 5.0,  # Conservative default for user configuration
                    "stop_loss_percent": 5.0,
                    "take_profit_percent": 20.0,
                    "max_daily_loss_percent": 10.0,  # Conservative daily limit
                    "min_confidence_threshold": 0.7,  # Higher confidence for user configuration
                    "max_trades_per_day": 10,  # Conservative trade limit
                    "trading_symbols": ["BTC", "ETH", "SOL"],
                    "signal_timeframe": "1h",
                    "risk_tolerance": "medium"
                }

                final_config = {**default_config}
                final_config.update(trading_config)

                # Ensure trading_symbols is a list
                if final_config.get("trading_symbols") is None:
                    final_config["trading_symbols"] = ["BTC", "ETH", "SOL"]

                user_trading_config = TradingConfig(**final_config)
            else:
                # No user config provided - "Let Yuki Decide"
                # Use signal data directly with minimal filtering
                user_trading_config = TradingConfig(
                    max_position_size_percent=100.0,  # Signal sizing can use the full allocation
                    max_leverage=40.0,  # Allow aggressive confidence-scaled leverage
                    stop_loss_percent=50.0,  # Use signal's stop loss
                    take_profit_percent=1000.0,  # Use signal's targets
                    max_daily_loss_percent=20.0,  # Liberal daily limit
                    min_confidence_threshold=0.1,  # Take almost all signals (10%+ confidence)
                    max_trades_per_day=50,  # Liberal trade limit
                    trading_symbols=None,  # All symbols from signals
                    signal_timeframe="1h",  # Accept all timeframes
                    risk_tolerance="signal_driven"  # Let signals drive everything
                )

            # Step 4: Create agent allocation record
            allocation_id = str(uuid.uuid4())
            platform_wallet_address = (
                wallet_address
                or readiness.get("wallet_address")
                or ""
            )

            performance_metrics: Dict[str, Any] = {
                "capital": {
                    "accounting_version": (
                        3 if agent_type == AgentType.RYU else 2
                    ),
                    "principal_allocated_amount": amount,
                    "compounded_equity": amount,
                }
            }
            if agent_type == AgentType.RYU:
                performance_metrics.update({
                    "wallets": {
                        "base_address": platform_wallet_address,
                        "solana_address": solana_wallet_address,
                    },
                    "funding": {
                        "home_chain": "base",
                        "home_token": "USDC",
                        "execution_routing": "per_trade",
                        "capital_moved_on_activation": False,
                        "base_gas_tx_hash": base_gas_tx_hash,
                        "base_min_gas_reserve_eth": settings.RYU_BASE_MIN_GAS_RESERVE_ETH,
                        "destination_gas_reserve_usdc": settings.RYU_SOLANA_GAS_RESERVE_USDC,
                        "fee_payer": "user",
                    },
                })

            allocation = AgentAllocation(
                allocation_id=allocation_id,
                user_id=user_id,
                agent_type=agent_type,
                allocated_amount=amount,
                remaining_amount=amount,  # Initially all funds available
                status=AllocationStatus.ACTIVE if live_trading_ready else AllocationStatus.PAUSED,
                platform_wallet_address=platform_wallet_address,
                trading_config=user_trading_config,
                created_at=datetime.now(),
                performance_metrics=performance_metrics,
            )

            ryu_funding_claimed = False
            if agent_type == AgentType.RYU and ryu_funding:
                try:
                    self.supabase.table("ryu_funding_transfers").insert({
                        "user_id": user_id,
                        "allocation_id": allocation_id,
                        "source_tx_hash": funding_tx_hash,
                        "lifi_quote_id": funding_quote_id,
                        "base_wallet_address": platform_wallet_address,
                        "solana_wallet_address": solana_wallet_address,
                        "input_usdc": float(funding_input_usdc or 0),
                        "received_usdc": amount,
                        "sol_gas_reserve_usdc": settings.RYU_SOLANA_GAS_RESERVE_USDC,
                        "status": "allocated",
                    }).execute()
                    ryu_funding_claimed = True
                except Exception as e:
                    logger.error(
                        "Could not claim Ryu funding transaction %s: %s",
                        funding_tx_hash,
                        e,
                    )
                    return {
                        "success": False,
                        "error": (
                            "The Ryu funding transfer could not be recorded. "
                            "Contact support with the transaction hash."
                        ),
                    }

            # Step 3: Store allocation in database
            stored = await self._store_allocation(allocation)
            if not stored:
                if ryu_funding_claimed:
                    try:
                        self.supabase.table("ryu_funding_transfers").delete().eq(
                            "source_tx_hash",
                            funding_tx_hash,
                        ).eq("user_id", user_id).execute()
                    except Exception:
                        logger.exception("Could not release failed Ryu funding claim")
                return {
                    "success": False,
                    "error": (
                        f"Failed to store {agent_type.value.title()} allocation. "
                        "Apply backend/migrations/add_yuki_execution_tables.sql first."
                    )
                }

            # Step 4: Add to active allocations
            self.active_allocations[allocation_id] = allocation

            # Step 5: Initialize agent if live auto-trading is enabled and ready
            if live_trading_ready and agent_type == AgentType.YUKI:
                await self._initialize_yuki_agent(user_id, allocation)
                # Do not leave a fresh Yuki allocation idle until the next
                # periodic worker sweep; queue an immediate signal pass now.
                await self.trigger_yuki_signal_check_for_allocation(allocation_id)
            elif live_trading_ready and agent_type == AgentType.RYU:
                # Ryu otherwise waits for the shared platform schedule. If the
                # user funds just after that window, the allocation can sit idle
                # for hours even though it is already tradable.
                asyncio.create_task(
                    self.trigger_ryu_signal_check_for_user(
                        user_id=user_id,
                        allocation_id=allocation_id,
                        reason="allocation_created",
                    )
                )

            logger.info(f"Allocated ${amount} to {agent_type.value} agent for user {user_id}")

            message = f"Allocated ${amount:.2f} to {agent_type.value} agent"
            if live_trading_requested and not live_trading_ready:
                message = (
                    f"Saved ${amount:.2f} {agent_type.value.title()} allocation. "
                    "Automated trading is paused until live trading is enabled "
                    "and readiness checks pass."
                )

            return {
                "success": True,
                "allocation_id": allocation_id,
                "agent_type": agent_type.value,
                "allocated_amount": amount,
                "status": allocation.status.value,
                "auto_trading_enabled": live_trading_ready,
                "platform_wallet_address": platform_wallet_address,
                "requires_live_trading_enablement": live_trading_requested and not live_trading_ready,
                "readiness": readiness,
                "message": message
            }

        except Exception as e:
            logger.error(f"Error allocating funds to agent: {e}")
            return {"success": False, "error": str(e)}

    async def add_funds_to_allocation(
        self,
        user_id: str,
        allocation_id: str,
        amount: float,
        wallet_address: Optional[str] = None,
        solana_wallet_address: Optional[str] = None,
        funding_tx_hash: Optional[str] = None,
        funding_quote_id: Optional[str] = None,
        funding_input_usdc: Optional[float] = None,
        base_gas_tx_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move unallocated wallet USDC into an existing agent allocation."""
        try:
            result = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .execute()
            )
            rows = result.data or []
            if not rows:
                return {"success": False, "error": "Agent allocation not found"}

            allocation_row = rows[0]
            agent_type = str(allocation_row.get("agent_type") or "").lower()
            minimum_top_up = (
                settings.RYU_MIN_ALLOCATION_USDC
                if agent_type == AgentType.RYU.value
                else settings.YUKI_MIN_ALLOCATION_USDC
            )
            if amount < minimum_top_up:
                return {
                    "success": False,
                    "error": f"Minimum top-up is ${minimum_top_up:.2f} USDC",
                }
            status = str(allocation_row.get("status") or "").lower()
            if status not in {AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value}:
                return {
                    "success": False,
                    "error": "Funds can only be added to an active or paused agent"
                }

            allocation_wallet = wallet_address or allocation_row.get("platform_wallet_address") or ""
            readiness: Dict[str, Any] = {}
            if agent_type == AgentType.YUKI.value:
                readiness = await self.get_yuki_trading_readiness(
                    user_id=user_id,
                    wallet_address=allocation_wallet
                )
                balance_usdc = float(readiness.get("balance_usdc", 0.0))
                shared_agent_types = [AgentType.YUKI.value]
            elif agent_type == AgentType.RYU.value:
                performance_metrics = dict(
                    allocation_row.get("performance_metrics") or {}
                )
                if not self._ryu_uses_base_home(allocation_row):
                    return {
                        "success": False,
                        "error": (
                            "Return the legacy Solana allocation, then reactivate "
                            "Ryu with Base as its home balance."
                        ),
                    }
                wallets = performance_metrics.get("wallets") or {}
                expected_solana_wallet = str(
                    wallets.get("solana_address") or ""
                )
                if not wallet_address or not solana_wallet_address:
                    return {
                        "success": False,
                        "error": (
                            "Ryu's Base and Solana wallets must be ready before adding capital"
                        ),
                    }
                if solana_wallet_address != expected_solana_wallet:
                    return {
                        "success": False,
                        "error": "Ryu top-up targeted a different Solana wallet",
                    }
                readiness = await self.ryu_spot_trading_service.get_trading_readiness(
                    user_id=user_id,
                    ethereum_wallet_address=wallet_address or "",
                    solana_wallet_address=solana_wallet_address or "",
                )
                balance_usdc = float(readiness.get("base_usdc_balance") or 0)
                shared_agent_types = [AgentType.SAKURA.value, AgentType.RYU.value]
            else:
                wallet_balance = await self.platform_wallet_service.get_user_balance(user_id, "USDC")
                balance_usdc = float(wallet_balance.total_balance) if wallet_balance else 0.0
                shared_agent_types = [AgentType.SAKURA.value]

            if agent_type == AgentType.RYU.value:
                available_to_allocate = max(
                    0.0,
                    balance_usdc - self._base_reserved_usdc(user_id),
                )
            elif agent_type == AgentType.SAKURA.value:
                available_to_allocate = max(
                    0.0,
                    balance_usdc - self._base_reserved_usdc(user_id),
                )
            else:
                active_result = (
                    self.supabase.table("agent_allocations")
                    .select("allocation_id,allocated_amount")
                    .eq("user_id", user_id)
                    .in_("agent_type", shared_agent_types)
                    .in_("status", [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value])
                    .execute()
                )
                total_allocated = sum(float(row.get("allocated_amount") or 0.0) for row in (active_result.data or []))
                available_to_allocate = max(0.0, balance_usdc - total_allocated)

            if amount > available_to_allocate:
                if (
                    available_to_allocate >= settings.YUKI_MIN_ALLOCATION_USDC
                    and amount <= available_to_allocate * 1.05
                ):
                    clamped = int(available_to_allocate * 100) / 100
                    logger.info(
                        f"Clamping {agent_type} top-up ${amount:.2f} to unallocated "
                        f"wallet balance ${clamped:.2f} for user {user_id}"
                    )
                    amount = clamped
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Only ${available_to_allocate:.2f} unallocated USDC is available. "
                            "Fund the shared wallet first, then add funds to this agent."
                        ),
                        "readiness": readiness
                    }

            current_allocated = float(allocation_row.get("allocated_amount") or 0.0)
            current_remaining = float(allocation_row.get("remaining_amount") or 0.0)
            new_allocated = current_allocated + amount
            new_remaining = current_remaining + amount

            performance_metrics = dict(allocation_row.get("performance_metrics") or {})
            capital_metrics = dict(performance_metrics.get("capital") or {})
            accounting_version = int(capital_metrics.get("accounting_version") or 1)
            if accounting_version >= 2:
                principal = float(
                    capital_metrics.get("principal_allocated_amount")
                    if capital_metrics.get("principal_allocated_amount") is not None
                    else current_allocated
                )
            else:
                principal = current_allocated
            capital_metrics.update({
                "accounting_version": (
                    3 if agent_type == AgentType.RYU.value else 2
                ),
                "principal_allocated_amount": principal + amount,
                "compounded_equity": new_allocated,
                "last_capital_flow_at": datetime.now().isoformat(),
            })
            performance_metrics["capital"] = capital_metrics
            if agent_type == AgentType.RYU.value:
                funding_history = list(
                    performance_metrics.get("funding_history") or []
                )
                funding_history.append({
                    "home_chain": "base",
                    "home_token": "USDC",
                    "capital_added_usdc": amount,
                    "capital_moved_on_top_up": False,
                    "base_gas_tx_hash": base_gas_tx_hash,
                    "created_at": datetime.now().isoformat(),
                    "fee_payer": "user",
                })
                performance_metrics["funding_history"] = funding_history[-50:]

            next_status = status
            if next_status == AllocationStatus.PAUSED.value:
                if agent_type == AgentType.YUKI.value and settings.YUKI_LIVE_TRADING_ENABLED:
                    next_status = AllocationStatus.ACTIVE.value
                elif agent_type == AgentType.RYU.value and settings.RYU_LIVE_TRADING_ENABLED:
                    next_status = AllocationStatus.ACTIVE.value

            update_data = {
                "status": next_status,
                "allocated_amount": new_allocated,
                "remaining_amount": new_remaining,
                "performance_metrics": performance_metrics,
                "updated_at": datetime.now().isoformat(),
            }
            ryu_funding_claimed = False
            if (
                agent_type == AgentType.RYU.value
                and funding_tx_hash
                and not self._ryu_uses_base_home(allocation_row)
            ):
                try:
                    self.supabase.table("ryu_funding_transfers").insert({
                        "user_id": user_id,
                        "allocation_id": allocation_id,
                        "source_tx_hash": funding_tx_hash,
                        "lifi_quote_id": funding_quote_id,
                        "base_wallet_address": allocation_wallet,
                        "solana_wallet_address": solana_wallet_address,
                        "input_usdc": float(funding_input_usdc or 0),
                        "received_usdc": amount,
                        "sol_gas_reserve_usdc": settings.RYU_SOLANA_GAS_RESERVE_USDC,
                        "status": "allocated",
                    }).execute()
                    ryu_funding_claimed = True
                except Exception as e:
                    logger.error(
                        "Could not claim Ryu top-up transaction %s: %s",
                        funding_tx_hash,
                        e,
                    )
                    return {
                        "success": False,
                        "error": (
                            "Ryu received the funds, but the top-up evidence "
                            "could not be recorded. Contact support with the transaction hash."
                        ),
                    }
            update_result = (
                self.supabase.table("agent_allocations")
                .update(update_data)
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .execute()
            )
            if not update_result.data:
                if ryu_funding_claimed:
                    try:
                        self.supabase.table("ryu_funding_transfers").delete().eq(
                            "source_tx_hash",
                            funding_tx_hash,
                        ).eq("user_id", user_id).execute()
                    except Exception:
                        logger.exception("Could not release failed Ryu top-up claim")
                return {"success": False, "error": "Failed to update agent allocation"}

            cached = self.active_allocations.get(allocation_id)
            if cached and cached.user_id == user_id:
                cached.allocated_amount = new_allocated
                cached.remaining_amount = new_remaining
                cached.performance_metrics = performance_metrics
                cached.status = AllocationStatus(next_status)

            logger.info(
                f"Added ${amount:.2f} to {agent_type} allocation {allocation_id} for user {user_id}; "
                f"new allocation ${new_allocated:.2f}"
            )
            if (
                agent_type == AgentType.RYU.value
                and next_status == AllocationStatus.ACTIVE.value
            ):
                asyncio.create_task(
                    self.trigger_ryu_signal_check_for_user(
                        user_id=user_id,
                        allocation_id=allocation_id,
                        reason="allocation_topped_up",
                    )
                )
            return {
                "success": True,
                "allocation_id": allocation_id,
                "agent_type": agent_type,
                "allocated_amount": new_allocated,
                "added_amount": amount,
                "status": next_status,
                "auto_trading_enabled": next_status == AllocationStatus.ACTIVE.value,
                "platform_wallet_address": allocation_wallet,
                "requires_live_trading_enablement": next_status != AllocationStatus.ACTIVE.value,
                "readiness": readiness,
                "message": f"Added ${amount:.2f} USDC to {agent_type} allocation"
            }

        except Exception as e:
            logger.error(f"Error adding funds to agent allocation {allocation_id}: {e}")
            return {"success": False, "error": str(e)}

    async def withdraw_funds_from_allocation(
        self,
        user_id: str,
        allocation_id: str,
        amount: float,
    ) -> Dict[str, Any]:
        """Return unused allocation capital to the agent's shared wallet pool."""
        try:
            if amount <= 0:
                return {"success": False, "error": "Withdrawal amount must be greater than zero"}

            result = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            if not rows:
                return {"success": False, "error": "Agent allocation not found"}

            row = rows[0]
            status = str(row.get("status") or "").lower()
            if status not in {AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value}:
                return {"success": False, "error": "Funds can only be withdrawn from an active or paused agent"}

            current_allocated = float(row.get("allocated_amount") or 0.0)
            current_remaining = float(row.get("remaining_amount") or 0.0)
            if amount > current_remaining:
                return {
                    "success": False,
                    "error": f"Only ${current_remaining:.2f} is currently available to withdraw",
                }

            ryu_return: Optional[Dict[str, Any]] = None
            is_ryu = (
                str(row.get("agent_type") or "").lower()
                == AgentType.RYU.value
            )
            if is_ryu and not self._ryu_uses_base_home(row):
                allocation = self._allocation_from_row(row)
                if not allocation:
                    return {
                        "success": False,
                        "error": "Could not restore Ryu's wallet configuration",
                    }
                ryu_return = (
                    await self.ryu_spot_trading_service.withdraw_available_usdc_to_base(
                        allocation,
                        amount,
                    )
                )
                if not ryu_return.get("success"):
                    return ryu_return

            new_allocated = max(0.0, current_allocated - amount)
            new_remaining = max(0.0, current_remaining - amount)
            performance_metrics = dict(row.get("performance_metrics") or {})
            capital_metrics = dict(performance_metrics.get("capital") or {})
            principal = float(
                capital_metrics.get("principal_allocated_amount")
                if capital_metrics.get("principal_allocated_amount") is not None
                else current_allocated
            )
            capital_metrics.update({
                "accounting_version": (
                    3 if is_ryu and self._ryu_uses_base_home(row) else 2
                ),
                "principal_allocated_amount": max(0.0, principal - amount),
                "compounded_equity": new_allocated,
                "last_capital_flow_at": datetime.now().isoformat(),
            })
            performance_metrics["capital"] = capital_metrics
            if ryu_return:
                withdrawals = list(
                    performance_metrics.get("withdrawal_history") or []
                )
                withdrawals.append({
                    "source_chain": "solana",
                    "destination_chain": "base",
                    "input_usdc": amount,
                    "estimated_base_usdc": ryu_return.get("estimated_base_usdc"),
                    "minimum_base_usdc": ryu_return.get("minimum_base_usdc"),
                    "transaction_hash": ryu_return.get("transaction_hash"),
                    "provider": ryu_return.get("provider"),
                    "fee_payer": "user",
                    "created_at": datetime.now().isoformat(),
                })
                performance_metrics["withdrawal_history"] = withdrawals[-50:]
            next_status = status if new_allocated > 0 else AllocationStatus.STOPPED.value

            update = {
                "allocated_amount": new_allocated,
                "remaining_amount": new_remaining,
                "performance_metrics": performance_metrics,
                "status": next_status,
                "updated_at": datetime.now().isoformat(),
            }
            update_result = (
                self.supabase.table("agent_allocations")
                .update(update)
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .execute()
            )
            if not update_result.data:
                return {"success": False, "error": "Failed to update agent allocation"}

            cached = self.active_allocations.get(allocation_id)
            if cached and cached.user_id == user_id:
                cached.allocated_amount = new_allocated
                cached.remaining_amount = new_remaining
                cached.performance_metrics = performance_metrics
                cached.status = AllocationStatus(next_status)
                if next_status == AllocationStatus.STOPPED.value:
                    self.active_allocations.pop(allocation_id, None)

            return {
                "success": True,
                "allocation_id": allocation_id,
                "agent_type": row.get("agent_type"),
                "withdrawn_amount": amount,
                "allocated_amount": new_allocated,
                "remaining_amount": new_remaining,
                "status": next_status,
                "message": (
                    f"Returning ${amount:.2f} from Solana to Kata Balance on Base"
                    if ryu_return
                    else f"Released ${amount:.2f} Base USDC to your available balance"
                ),
                **(ryu_return or {}),
            }
        except Exception as e:
            logger.error(f"Error withdrawing funds from allocation {allocation_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _store_allocation(self, allocation: AgentAllocation) -> bool:
        """Store allocation in database."""
        try:
            allocation_data = {
                "allocation_id": allocation.allocation_id,
                "user_id": allocation.user_id,
                "agent_type": allocation.agent_type.value,
                "allocated_amount": allocation.allocated_amount,
                "remaining_amount": allocation.remaining_amount,
                "status": allocation.status.value,
                "platform_wallet_address": allocation.platform_wallet_address,
                "created_at": allocation.created_at.isoformat(),
                "last_trade_at": allocation.last_trade_at.isoformat() if allocation.last_trade_at else None,
                "total_trades": allocation.total_trades,
                "realized_pnl": allocation.realized_pnl,
                "unrealized_pnl": allocation.unrealized_pnl,
                "performance_metrics": allocation.performance_metrics or {},
                "trading_config": allocation.trading_config.__dict__ if allocation.trading_config else {}
            }

            result = self.supabase.table("agent_allocations").insert(allocation_data).execute()
            if result.data:
                logger.info(f"Stored allocation {allocation.allocation_id} in database")
                return True

            logger.error(f"Failed to store allocation {allocation.allocation_id}: {result}")
            return False

        except Exception as e:
            logger.error(f"Error storing allocation: {e}")
            return False

    async def _initialize_yuki_agent(self, user_id: str, allocation: AgentAllocation):
        """Initialize Yuki agent for automated trading."""
        try:
            # Agent configuration based on user's trading config
            trading_config = allocation.trading_config
            yuki_config = {
                "user_id": user_id,
                "allocation_id": allocation.allocation_id,
                "allocated_amount": allocation.allocated_amount,
                "risk_params": {
                    "max_position_size_percent": trading_config.max_position_size_percent,
                    "max_leverage": trading_config.max_leverage,
                    "stop_loss_percent": trading_config.stop_loss_percent,
                    "take_profit_percent": trading_config.take_profit_percent,
                    "max_daily_loss_percent": trading_config.max_daily_loss_percent,
                    "min_confidence_threshold": trading_config.min_confidence_threshold,
                    "max_trades_per_day": trading_config.max_trades_per_day
                },
                "trading_symbols": trading_config.trading_symbols,
                "signal_timeframe": trading_config.signal_timeframe,
                "risk_tolerance": trading_config.risk_tolerance
            }

            # Create Yuki agent instance
            yuki_agent = YukiAgent(
                user_id=user_id,
                config=yuki_config,
                hyperliquid_service=self.hyperliquid_service
            )

            # Store agent instance
            if user_id not in self.agent_instances:
                self.agent_instances[user_id] = {}
            self.agent_instances[user_id][AgentType.YUKI.value] = {
                "agent": yuki_agent,
                "allocation": allocation,
                "last_signal_check": datetime.now(),
                "trading_enabled": True
            }

            await self._start_yuki_position_monitor(user_id, allocation.allocation_id)

            logger.info(
                f"Initialized Yuki agent for user {user_id} with ${allocation.allocated_amount}; "
                "waiting for platform signal generation cycles"
            )

        except Exception as e:
            logger.error(f"Error initializing Yuki agent: {e}")

    async def _start_yuki_position_monitor(self, user_id: str, allocation_id: str) -> bool:
        """Start 30-second position monitoring for an active Yuki allocation."""
        try:
            if not self._is_yuki_execution_owner():
                logger.info(
                    "Skipping Yuki position monitor in Railway service %s; owner is %s",
                    os.getenv("RAILWAY_SERVICE_NAME"),
                    settings.YUKI_EXECUTION_SERVICE_NAME,
                )
                return False
            if not self.hyperliquid_service:
                logger.warning("Cannot start Yuki position monitor: Hyperliquid service is unavailable")
                return False

            if not self.yuki_position_monitor:
                from kata.services.yuki_position_monitor import create_yuki_position_monitor

                self.yuki_position_monitor = create_yuki_position_monitor(
                    self.hyperliquid_service,
                    react_agent_resolver=self._resolve_yuki_react_agent,
                )
            else:
                # Monitors can also be created by reconciliation/manual-close
                # paths. Attach the live agent resolver before lifecycle reviews.
                self.yuki_position_monitor.react_agent_resolver = self._resolve_yuki_react_agent

            return await self.yuki_position_monitor.start_monitoring(user_id, allocation_id)
        except Exception as e:
            logger.error(f"Error starting Yuki position monitor for allocation {allocation_id}: {e}")
            return False

    def _resolve_yuki_react_agent(self, user_id: str) -> Optional[YukiAgent]:
        """Return the user's existing Yuki instance so reviews share its memory and tools."""
        instance = (self.agent_instances.get(user_id) or {}).get(AgentType.YUKI.value) or {}
        agent = instance.get("agent")
        return agent if isinstance(agent, YukiAgent) else None

    @staticmethod
    def _parse_row_datetime(value: Any) -> Optional[datetime]:
        """Parse a Supabase timestamp string into a datetime; None if unparseable."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            raw = str(value).replace("Z", "+00:00")
            fractional = re.match(r"^(.*\.)(\d+)([+-]\d{2}:\d{2})$", raw)
            if fractional:
                raw = (
                    fractional.group(1)
                    + fractional.group(2)[:6].ljust(6, "0")
                    + fractional.group(3)
                )
            return datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            return None

    def _allocation_from_row(self, row: Dict[str, Any]) -> Optional[AgentAllocation]:
        """Rebuild an AgentAllocation dataclass from an agent_allocations DB row."""
        try:
            raw_config = row.get("trading_config") or {}
            config_fields = TradingConfig.__dataclass_fields__.keys()
            trading_config = TradingConfig(**{
                key: value for key, value in raw_config.items() if key in config_fields
            })

            return AgentAllocation(
                allocation_id=row["allocation_id"],
                user_id=row["user_id"],
                agent_type=AgentType(row["agent_type"]),
                allocated_amount=float(row.get("allocated_amount") or 0.0),
                remaining_amount=float(row.get("remaining_amount") or 0.0),
                status=AllocationStatus(row["status"]),
                platform_wallet_address=row.get("platform_wallet_address") or "",
                trading_config=trading_config,
                created_at=self._parse_row_datetime(row.get("created_at")) or datetime.now(),
                last_trade_at=self._parse_row_datetime(row.get("last_trade_at")),
                total_trades=int(row.get("total_trades") or 0),
                realized_pnl=float(row.get("realized_pnl") or 0.0),
                unrealized_pnl=float(row.get("unrealized_pnl") or 0.0),
                performance_metrics=row.get("performance_metrics") or {},
            )
        except Exception as e:
            logger.error(f"Failed to rebuild allocation from row {row.get('allocation_id')}: {e}")
            return None

    async def _rehydrate_active_yuki_allocations(self) -> int:
        """
        Synchronize live Yuki allocations from the database into in-memory state.

        active_allocations / agent_instances are per-process caches populated at
        allocation time. The signal-generation worker is a different process (and any
        process restart wipes them), so the database must remain the source of truth.
        Paused allocations are loaded as well because their existing positions still
        need monitoring. Their status prevents them from receiving new signals.
        """
        restored = 0
        try:
            result = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("agent_type", AgentType.YUKI.value)
                .in_("status", [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value])
                .execute()
            )
            for row in result.data or []:
                allocation_id = row.get("allocation_id")
                if not allocation_id:
                    continue

                allocation = self._allocation_from_row(row)
                if not allocation:
                    continue

                was_cached = allocation_id in self.active_allocations
                self.active_allocations[allocation_id] = allocation
                if not was_cached:
                    restored += 1

                user_id = allocation.user_id
                has_agent = (
                    user_id in self.agent_instances
                    and AgentType.YUKI.value in self.agent_instances[user_id]
                )
                if not has_agent:
                    await self._initialize_yuki_agent(user_id, allocation)
                else:
                    # Replace the stale allocation object held by the agent too.
                    # In particular, a pause performed by the web service must be
                    # observed by the separate signal worker before its next cycle.
                    self.agent_instances[user_id][AgentType.YUKI.value]["allocation"] = allocation
                    await self._start_yuki_position_monitor(user_id, allocation_id)

            if restored:
                logger.info(f"♻️ Rehydrated {restored} live Yuki allocation(s) from database")
        except Exception as e:
            logger.error(f"Failed to rehydrate Yuki allocations from database: {e}")
        return restored

    async def run_yuki_signal_cycle_after_platform_generation(
        self,
        analysis_run_id: Optional[str] = None,
        signals_generated: int = 0
    ) -> Dict[str, Any]:
        """Run Yuki once after a platform signal generation cycle finishes."""
        if not settings.YUKI_LIVE_TRADING_ENABLED:
            return {
                "success": True,
                "skipped": True,
                "reason": "Yuki live trading disabled",
                "analysis_run_id": analysis_run_id,
            }
        if not self._is_yuki_execution_owner():
            return {
                "success": True,
                "skipped": True,
                "reason": "Yuki execution is owned by another Railway service",
                "analysis_run_id": analysis_run_id,
            }

        async with self.yuki_signal_cycle_lock:
            # Reload allocations from the database: this may run in a worker process
            # that never saw the allocation API call, or after a restart.
            await self._rehydrate_active_yuki_allocations()

            active_allocations = [
                allocation for allocation in self.active_allocations.values()
                if allocation.agent_type == AgentType.YUKI and allocation.status == AllocationStatus.ACTIVE
            ]

            summary = {
                "success": True,
                "analysis_run_id": analysis_run_id,
                "signals_generated": signals_generated,
                "allocations_checked": len(active_allocations),
                "trades_attempted": 0,
                "trades_executed": 0,
                "errors": []
            }

            if not active_allocations:
                logger.info("Yuki post-generation cycle skipped: no active live allocations")
                return summary

            for allocation in active_allocations:
                try:
                    allocation_summary = await self._run_yuki_signal_cycle_for_allocation(allocation)
                    summary["trades_attempted"] += allocation_summary.get("trades_attempted", 0)
                    summary["trades_executed"] += allocation_summary.get("trades_executed", 0)
                except Exception as e:
                    logger.error(f"Yuki post-generation cycle failed for allocation {allocation.allocation_id}: {e}")
                    summary["errors"].append({
                        "allocation_id": allocation.allocation_id,
                        "error": str(e)
                    })

            return summary

    async def run_ryu_signal_cycle_after_platform_generation(
        self,
        analysis_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Ryu once after a platform signal generation cycle finishes.

        Unlike Yuki, Ryu keeps no persistent per-user in-memory agent instance --
        RyuSpotTradingService reads active allocations straight from the database
        each cycle (agent_allocations is already the source of truth; there is
        nothing else to rehydrate). The kill switch is checked here, at the top,
        before any discovery or LLM analysis runs, so a disabled feature costs
        nothing rather than quietly running (and paying for) LLM calls it will
        never act on.
        """
        if not settings.RYU_LIVE_TRADING_ENABLED:
            return {
                "success": True,
                "skipped": True,
                "reason": "Ryu live trading disabled",
                "analysis_run_id": analysis_run_id,
            }

        async with self.ryu_signal_cycle_lock:
            try:
                result = (
                    self.supabase.table("agent_allocations")
                    .select("*")
                    .eq("agent_type", AgentType.RYU.value)
                    .eq("status", AllocationStatus.ACTIVE.value)
                    .execute()
                )
            except Exception as e:
                logger.error(f"Failed to load active Ryu allocations: {e}")
                return {"success": False, "error": str(e), "analysis_run_id": analysis_run_id}

            active_allocations = [
                allocation for allocation in (
                    self._allocation_from_row(row) for row in (result.data or [])
                )
                if allocation is not None
            ]

            summary = {
                "success": True,
                "analysis_run_id": analysis_run_id,
                "allocations_checked": len(active_allocations),
                "candidates_analyzed": 0,
                "buys_executed": 0,
                "exits_executed": 0,
                "errors": 0,
            }

            if not active_allocations:
                logger.info("Ryu post-generation cycle skipped: no active live allocations")
                return summary

            try:
                cycle_result = await self.ryu_spot_trading_service.run_signal_cycle(active_allocations)
                summary.update(cycle_result)
            except Exception as e:
                logger.error(f"Ryu post-generation signal cycle failed: {e}")
                summary["success"] = False
                summary["errors"] = summary.get("errors", 0) + 1

            return summary

    async def run_ryu_position_risk_cycle(self) -> Dict[str, Any]:
        """Monitor existing Ryu spot holdings without discovery or LLM calls."""
        if not settings.RYU_LIVE_TRADING_ENABLED:
            return {
                "success": True,
                "skipped": True,
                "reason": "Ryu live trading disabled",
            }
        if not settings.RYU_POSITION_MONITOR_ENABLED:
            return {
                "success": True,
                "skipped": True,
                "reason": "Ryu position monitor disabled",
            }

        async with self.ryu_signal_cycle_lock:
            try:
                result = (
                    self.supabase.table("agent_allocations")
                    .select("*")
                    .eq("agent_type", AgentType.RYU.value)
                    .in_(
                        "status",
                        [
                            AllocationStatus.ACTIVE.value,
                            AllocationStatus.PAUSED.value,
                        ],
                    )
                    .execute()
                )
            except Exception as e:
                logger.error(f"Failed to load monitored Ryu allocations for risk monitor: {e}")
                return {"success": False, "error": str(e)}

            monitored_allocations = [
                allocation for allocation in (
                    self._allocation_from_row(row) for row in (result.data or [])
                )
                if allocation is not None
            ]
            if not monitored_allocations:
                return {
                    "success": True,
                    "allocations_checked": 0,
                    "positions_checked": 0,
                    "exits_executed": 0,
                    "partial_exits": 0,
                    "errors": 0,
                }

            try:
                summary = await self.ryu_spot_trading_service.monitor_active_positions(
                    monitored_allocations
                )
                return {
                    "success": True,
                    "allocations_checked": len(monitored_allocations),
                    **summary,
                }
            except Exception as e:
                logger.error(f"Ryu position risk cycle failed: {e}")
                return {
                    "success": False,
                    "allocations_checked": len(monitored_allocations),
                    "error": str(e),
                }

    async def trigger_ryu_signal_check_for_user(
        self,
        user_id: str,
        allocation_id: Optional[str] = None,
        reason: str = "manual",
    ) -> None:
        """Queue an immediate Ryu discovery cycle once a user's allocation is truly ready.

        Ryu normally runs on the shared platform schedule. If a user funds or
        completes delegation just after that window, the allocation can sit idle
        until the next cycle even though it is already tradable. This helper
        safely re-checks readiness and runs only the ready active allocations for
        that user.
        """
        try:
            if not settings.RYU_LIVE_TRADING_ENABLED or not user_id:
                return

            query = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("user_id", user_id)
                .eq("agent_type", AgentType.RYU.value)
                .eq("status", AllocationStatus.ACTIVE.value)
            )
            if allocation_id:
                query = query.eq("allocation_id", allocation_id)

            allocation_rows = query.execute().data or []
            candidate_allocations = [
                allocation for allocation in (
                    self._allocation_from_row(row) for row in allocation_rows
                )
                if allocation is not None
            ]
            if not candidate_allocations:
                return

            ready_allocations: List[AgentAllocation] = []
            for allocation in candidate_allocations:
                wallets = dict(allocation.performance_metrics.get("wallets") or {}) if isinstance(allocation.performance_metrics, dict) else {}
                ethereum_wallet = str(
                    wallets.get("base_address")
                    or allocation.platform_wallet_address
                    or ""
                )
                solana_wallet = str(wallets.get("solana_address") or "")
                if not ethereum_wallet or not solana_wallet:
                    logger.info(
                        "Skipping immediate Ryu catch-up for allocation %s: wallet pair incomplete",
                        allocation.allocation_id,
                    )
                    continue

                readiness = await self.ryu_spot_trading_service.get_trading_readiness(
                    user_id=allocation.user_id,
                    ethereum_wallet_address=ethereum_wallet,
                    solana_wallet_address=solana_wallet,
                )
                if readiness.get("ready_for_trading"):
                    ready_allocations.append(allocation)
                else:
                    logger.info(
                        "Skipping immediate Ryu catch-up for allocation %s after %s: %s",
                        allocation.allocation_id,
                        reason,
                        "; ".join(readiness.get("blocking_reasons", [])) or "not ready",
                    )

            if not ready_allocations:
                return

            logger.info(
                "⚡ Triggering immediate Ryu cycle after %s for user %s (%s ready allocation%s)",
                reason,
                user_id,
                len(ready_allocations),
                "" if len(ready_allocations) == 1 else "s",
            )

            async with self.ryu_signal_cycle_lock:
                summary = await self.ryu_spot_trading_service.run_signal_cycle(
                    ready_allocations
                )

            logger.info(
                "⚡ Immediate Ryu cycle after %s complete: allocations=%s candidates=%s buys=%s exits=%s errors=%s",
                reason,
                len(ready_allocations),
                summary.get("candidates_analyzed", 0),
                summary.get("buys_executed", 0),
                summary.get("exits_executed", 0),
                summary.get("errors", 0),
            )
        except Exception as e:
            logger.warning(
                "Could not trigger immediate Ryu signal check for user %s after %s: %s",
                user_id,
                reason,
                e,
            )

    async def _run_yuki_signal_cycle_for_allocation(self, allocation: AgentAllocation) -> Dict[str, int]:
        """Check platform signals once for a single active Yuki allocation."""
        user_id = allocation.user_id
        allocation_id = allocation.allocation_id

        if allocation_id not in self.active_allocations:
            logger.info(f"Allocation {allocation_id} no longer active; skipping Yuki cycle")
            return {"trades_attempted": 0, "trades_executed": 0}

        if allocation.status != AllocationStatus.ACTIVE:
            logger.info(f"Allocation {allocation_id} status is {allocation.status.value}; skipping Yuki cycle")
            return {"trades_attempted": 0, "trades_executed": 0}

        if user_id not in self.agent_instances or AgentType.YUKI.value not in self.agent_instances[user_id]:
            logger.warning(f"Yuki agent not found for user {user_id}; skipping allocation {allocation_id}")
            return {"trades_attempted": 0, "trades_executed": 0}

        agent_data = self.agent_instances[user_id][AgentType.YUKI.value]
        if not agent_data.get("trading_enabled", True):
            logger.info(f"Yuki trading disabled for allocation {allocation_id}; skipping post-generation cycle")
            return {"trades_attempted": 0, "trades_executed": 0}

        # Workers have no browser session; trading access comes from the
        # delegation record instead.
        if self.hyperliquid_service and allocation.platform_wallet_address:
            client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                user_id=user_id,
                wallet_address=allocation.platform_wallet_address,
            )
            if not client_ready:
                logger.warning(
                    f"Could not build delegated trading client for allocation {allocation_id}; skipping cycle"
                )
                return {"trades_attempted": 0, "trades_executed": 0}

        # A terminal source signal or expired/missing exchange order must stop
        # reserving margin before this cycle considers newer opportunities.
        # The regular 30-second monitor performs the same reconciliation; the
        # monitor lock makes this immediate pre-sweep pass race-safe.
        if not self.yuki_position_monitor:
            await self._start_yuki_position_monitor(user_id, allocation_id)
        if self.yuki_position_monitor and allocation.platform_wallet_address:
            try:
                await self.yuki_position_monitor.reconcile_pending_entries_for_allocation(
                    user_id=user_id,
                    allocation_id=allocation_id,
                    wallet_address=allocation.platform_wallet_address,
                )
            except Exception as exc:
                logger.warning(
                    "Could not reconcile pending entries before Yuki signal sweep for %s: %s",
                    allocation_id,
                    exc,
                )

        # Rebuild available collateral only after stale reservations have been
        # finalized so newly freed margin is usable in this exact signal sweep.
        ledger = await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
        if ledger is not None:
            allocation.allocated_amount = ledger["allocated_amount"]
            allocation.remaining_amount = ledger["remaining_amount"]
            allocation.realized_pnl = ledger["realized_pnl"]

        trading_config = allocation.trading_config
        signals = await self._get_yuki_trading_signals(
            symbols=trading_config.trading_symbols,
            min_confidence=trading_config.min_confidence_threshold
        )
        if signals and getattr(settings, "YUKI_PORTFOLIO_SELECTION_ENABLED", True):
            exposure_rows = await self._get_yuki_portfolio_exposure_rows(allocation_id)
            signals = self._rank_yuki_signals_for_portfolio(
                allocation,
                signals,
                exposure_rows,
            )
            logger.info(
                "Yuki portfolio plan for allocation %s: %s",
                allocation_id,
                [
                    {
                        "signal_id": signal.get("signal_id"),
                        "symbol": signal.get("symbol"),
                        "utility": (signal.get("portfolio_selection") or {}).get(
                            "adjusted_profit_per_risk_dollar"
                        ),
                        "fundable": (signal.get("portfolio_selection") or {}).get(
                            "fundable_from_current_and_pending_capital"
                        ),
                    }
                    for signal in signals
                ],
            )

        trades_executed = 0
        for signal in signals:
            executed = await self._execute_yuki_trade(
                user_id=user_id,
                allocation=allocation,
                yuki_agent=agent_data["agent"],
                signal=signal
            )
            if executed:
                trades_executed += 1

        agent_data["last_signal_check"] = datetime.now()
        return {"trades_attempted": len(signals), "trades_executed": trades_executed}

    async def _get_yuki_portfolio_exposure_rows(
        self,
        allocation_id: str,
    ) -> List[Dict[str, Any]]:
        """Load the allocation's live and reserved risk once per signal sweep."""
        try:
            return list((
                self.supabase.table("agent_trades")
                .select(
                    "symbol,side,status,entry_price,position_size,leverage,trade_amount,trade_metadata"
                )
                .eq("allocation_id", allocation_id)
                .in_("status", ["pending", "filled", "partially_filled"])
                .execute()
            ).data or [])
        except Exception as exc:
            logger.warning(
                "Could not load portfolio exposure for allocation %s; using signal-only ranking: %s",
                allocation_id,
                exc,
            )
            return []

    async def _get_yuki_trading_signals(
        self,
        symbols: Optional[List[str]],
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Get trading signals for Yuki agent from the platform signal pool only."""
        try:
            min_confidence = max(
                float(min_confidence or 0.0),
                # A published directional signal has already passed the
                # calibrated platform quality floor. Keep that as Yuki's hard
                # minimum, while allowing a user-configured allocation to ask
                # for a stricter threshold. Confidence above this floor is a
                # ranking/sizing input. Only a sufficiently sampled, validated
                # negative-EV model may hard-veto a published signal.
                float(settings.SIGNAL_PUBLISH_CONFIDENCE_FLOOR or 0.0),
            )
            universe = None
            if self.hyperliquid_service:
                universe = await self.hyperliquid_service.get_perp_universe()

            allowed_symbols = None
            if symbols:
                allowed_symbols = set()
                for configured_symbol in symbols:
                    allowed_symbols.update(self._hyperliquid_symbol_candidates(configured_symbol))
                    normalized_configured_symbol = self._normalize_hyperliquid_symbol(configured_symbol, universe)
                    if normalized_configured_symbol:
                        allowed_symbols.add(normalized_configured_symbol)

            # Rank is a priority, not an eligibility gate. Load the complete
            # active pool so Standard can act on any valid setup. The final
            # list is sorted best-first below, which lets the allocation's
            # remaining capital decide how many signals can actually be used.
            platform_signals = await self.platform_signals_service.get_active_signals(limit=100)

            converted_signals = []
            for platform_signal in platform_signals:
                source_symbol = platform_signal.token_symbol
                symbol = self._normalize_hyperliquid_symbol(source_symbol, universe)
                symbol_candidates = set(self._hyperliquid_symbol_candidates(source_symbol))
                if allowed_symbols and symbol not in allowed_symbols and not symbol_candidates.intersection(allowed_symbols):
                    continue

                confidence = float(platform_signal.confidence or 0)
                if confidence > 1:
                    confidence = confidence / 100
                if confidence < min_confidence:
                    continue

                direction = (platform_signal.direction or "").upper()
                signal_direction = None
                is_strong = confidence >= 0.85 or (platform_signal.signal_strength or "").upper() in {"STRONG", "VERY_STRONG"}
                if direction in {"LONG", "BUY", "BULLISH"}:
                    signal_direction = "STRONG_BUY" if is_strong else "BUY"
                elif direction in {"SHORT", "SELL", "BEARISH"}:
                    signal_direction = "STRONG_SELL" if is_strong else "SELL"

                if not signal_direction:
                    continue

                market_conditions = (
                    dict(getattr(platform_signal, "market_conditions", {}) or {})
                    if isinstance(getattr(platform_signal, "market_conditions", {}), dict)
                    else {}
                )
                entry_block_reason = entry_execution_block_reason(market_conditions)
                if entry_block_reason:
                    logger.info(
                        "Skipping signal %s for Yuki: %s",
                        platform_signal.signal_id,
                        entry_block_reason,
                    )
                    continue
                confidence_breakdown = (
                    dict(getattr(platform_signal, "ai_confidence_breakdown", {}) or {})
                    if isinstance(getattr(platform_signal, "ai_confidence_breakdown", {}), dict)
                    else {}
                )

                converted_signals.append({
                    "symbol": symbol,
                    "source_symbol": source_symbol,
                    "signal": signal_direction,
                    "confidence": confidence,
                    "price": platform_signal.entry_price,
                    "reasoning": platform_signal.ai_reasoning or platform_signal.analysis_notes or "",
                    "stop_loss": platform_signal.stop_loss,
                    "target_1": platform_signal.target_1,
                    "target_2": platform_signal.target_2,
                    "leverage": platform_signal.leverage,
                    "position_size_percent": platform_signal.position_size,
                    "signal_id": platform_signal.signal_id,
                    "opportunity_rank": getattr(platform_signal, "opportunity_rank", None),
                    "expires_at": getattr(platform_signal, "expires_at", None),
                    "time_horizon": getattr(platform_signal, "time_horizon", None),
                    "logo_url": getattr(platform_signal, "logo_url", None),
                    "timestamp": getattr(platform_signal, "analysis_timestamp", None) or datetime.now(timezone.utc),
                    "updated_at": getattr(platform_signal, "updated_at", None),
                    "execution_policy": "signal_led_adaptive",
                    "edge_estimate": market_conditions.get("edge_estimate"),
                    "entry_activation": {
                        key: market_conditions.get(key)
                        for key in (
                            "entry_activation_required",
                            "entry_activated",
                            "entry_activated_at",
                            "entry_order_expires_at",
                        )
                    },
                    "entry_order_expires_at": market_conditions.get("entry_order_expires_at"),
                    "entry_revision": market_conditions.get("entry_revision", 0),
                    "market_structure_context": market_conditions.get("market_structure_context"),
                    "policy_versions": confidence_breakdown.get("policy_versions"),
                    "policy_bundle_version": confidence_breakdown.get("policy_bundle_version"),
                    "prompt_version": confidence_breakdown.get("prompt_version"),
                    "portfolio_context": {
                        "market_regime": confidence_breakdown.get("market_regime"),
                        "btc_correlation": confidence_breakdown.get("btc_correlation"),
                        "asset_class": market_conditions.get("asset_class"),
                    },
                })

            # Signals stay venue-agnostic for the UI (users can take them on Binance etc.);
            # Yuki only executes the subset that Hyperliquid actually lists.
            if converted_signals and self.hyperliquid_service:
                if universe:
                    tradeable_signals = []
                    for candidate in converted_signals:
                        if str(candidate["symbol"]).upper() in universe:
                            tradeable_signals.append(candidate)
                        else:
                            logger.info(
                                f"Skipping signal {candidate.get('source_symbol') or candidate['symbol']} for Yuki: "
                                f"normalized to {candidate['symbol']}, not listed on Hyperliquid "
                                "(signal remains visible in the UI for manual trading)"
                            )
                    converted_signals = tradeable_signals

            converted_signals.sort(key=self._yuki_signal_priority)

            if converted_signals:
                logger.debug(
                    f"Loaded {len(converted_signals)} platform signals meeting confidence threshold "
                    f"{min_confidence} for Yuki (signal-led adaptive policy, prioritized by "
                    "opportunity rank, recency, and confidence)"
                )
                return converted_signals

            logger.debug("No active platform signals met Yuki trading criteria; skipping this cycle")
            return []

        except Exception as e:
            logger.error(f"Error getting Yuki trading signals: {e}")
            return []

    async def _close_opposite_yuki_position(
        self,
        allocation: AgentAllocation,
        symbol: str,
        desired_side: OrderSide,
        signal_id: Optional[str],
    ) -> Optional[bool]:
        """Enforce one symbol exposure, closing the old side before a reversal."""
        if not self.hyperliquid_service or not allocation.platform_wallet_address:
            return None

        live_positions = await self.hyperliquid_service.get_positions_for_wallet(
            allocation.platform_wallet_address
        )
        if live_positions is None:
            logger.warning(f"Cannot verify live {symbol} exposure before signal {signal_id}; skipping entry")
            return False

        target_symbol = str(symbol or "").strip()
        target_symbol_key = self._quote_stripped_symbol(target_symbol)
        desired_position_side = "long" if desired_side == OrderSide.BUY else "short"
        live_position = next(
            (
                position for coin, position in live_positions.items()
                if self._quote_stripped_symbol(coin) == target_symbol_key
            ),
            None,
        )

        # Orders are exposure too. A fresh signal id must not bypass an existing
        # same-direction resting entry for the same normalized symbol.
        open_orders = await self.hyperliquid_service.get_open_orders(allocation.platform_wallet_address)
        if open_orders is None:
            logger.warning(f"Cannot inspect {symbol} orders before signal {signal_id}; skipping entry")
            return False

        matching_orders = [
            order for order in open_orders
            if self._quote_stripped_symbol(order.get("coin")) == target_symbol_key
        ]
        same_direction_entry = next(
            (
                order for order in matching_orders
                if not order.get("reduceOnly")
                and not order.get("isTrigger")
                and self._hyperliquid_order_position_side(order) == desired_position_side
            ),
            None,
        )
        if same_direction_entry is not None:
            logger.warning(
                f"Skipping signal {signal_id}: {target_symbol} already has a live "
                f"{desired_position_side} entry order {same_direction_entry.get('oid')}"
            )
            return False

        if live_position is not None and str(live_position.side).lower() == desired_position_side:
            logger.warning(
                f"Skipping signal {signal_id}: {target_symbol} already has a live "
                f"{desired_position_side} position"
            )
            return False

        if live_position is not None:
            # Prevent a stale plain entry from filling while Yuki reassesses.
            # Existing stop/TP protection remains intact unless the selected
            # action itself changes or closes the position.
            for order in matching_orders:
                if order.get("reduceOnly") or order.get("isTrigger"):
                    continue
                order_id = order.get("oid")
                if order_id is None:
                    continue
                order_symbol = str(order.get("coin") or target_symbol)
                if not await self.hyperliquid_service.cancel_order(order_symbol, order_id):
                    logger.error(
                        "Could not cancel stale entry order %s for %s; blocking signal %s",
                        order_id,
                        target_symbol,
                        signal_id,
                    )
                    return False
            # An opposite signal is evidence, not an automatic reversal. Run the
            # autonomous position-management cycle now and execute the action it
            # selects. The new entry may proceed only if Yuki actually chose and
            # completed FULL_EXIT; HOLD/ADD/PARTIAL_EXIT/ADJUST_STOP keep the
            # existing thesis and block the opposite entry.
            return await self._review_opposite_yuki_position(
                allocation=allocation,
                live_position=live_position,
                symbol=target_symbol,
                desired_position_side=desired_position_side,
                signal_id=signal_id,
            )

        # With no live position, cancel any opposite resting entry. Protective
        # reduce-only/trigger orders are never removed by this cleanup path.
        for order in matching_orders:
            if live_position is None and (order.get("reduceOnly") or order.get("isTrigger")):
                continue
            order_id = order.get("oid")
            if order_id is None:
                continue
            order_symbol = str(order.get("coin") or target_symbol)
            cancelled = await self.hyperliquid_service.cancel_order(order_symbol, order_id)
            if not cancelled:
                logger.error(
                    f"Could not cancel conflicting order {order_id} for {target_symbol}; "
                    f"blocking reversal signal {signal_id}"
                )
                return False

        return None

    async def _review_opposite_yuki_position(
        self,
        allocation: AgentAllocation,
        live_position: Any,
        symbol: str,
        desired_position_side: str,
        signal_id: Optional[str],
    ) -> bool:
        """Let Yuki decide and execute the response to a new opposite signal."""
        get_positions = getattr(self.db_service, "get_agent_positions", None)
        if not get_positions:
            logger.error("Cannot load Yuki trade memory for %s reversal; blocking new entry", symbol)
            return False
        active_positions = await get_positions(allocation.allocation_id)
        normalize_symbol = getattr(
            self.db_service,
            "_normalize_agent_symbol",
            lambda value: self._quote_stripped_symbol(value),
        )
        normalized_symbol = normalize_symbol(symbol)
        db_position = next(
            (
                dict(position)
                for position in active_positions
                if normalize_symbol(position.get("symbol")) == normalized_symbol
            ),
            None,
        )
        if not db_position:
            logger.error(
                "Cannot match live %s exposure to its Yuki trade record; blocking signal %s",
                symbol,
                signal_id,
            )
            return False

        if not getattr(self, "yuki_position_monitor", None):
            from kata.services.yuki_position_monitor import create_yuki_position_monitor

            self.yuki_position_monitor = create_yuki_position_monitor(
                self.hyperliquid_service,
                react_agent_resolver=self._resolve_yuki_react_agent,
            )
        else:
            self.yuki_position_monitor.react_agent_resolver = self._resolve_yuki_react_agent

        current_price = float(live_position.mark_price or live_position.entry_price or 0.0)
        size = abs(float(live_position.size or 0.0))
        unrealized_pnl = float(live_position.unrealized_pnl or 0.0)
        leverage = max(1.0, float(live_position.leverage or db_position.get("leverage") or 1.0))
        position_value = float(live_position.position_value or size * current_price)
        margin_used = float(live_position.margin_used or position_value / leverage)
        db_position.update({
            "symbol": symbol,
            "side": str(live_position.side).lower(),
            "size": size,
            "entry_price": float(live_position.entry_price or db_position.get("entry_price") or 0.0),
            "current_price": current_price,
            "leverage": leverage,
            "position_value": position_value,
            "margin_used": margin_used,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": (
                unrealized_pnl / margin_used * 100.0 if margin_used > 0 else 0.0
            ),
        })
        metadata = dict(db_position.get("position_metadata") or {})
        review_event = "opposite_signal_arrived"
        if metadata.get("react_last_opposite_signal_id") == signal_id:
            logger.info("Yuki already reviewed opposite signal %s for %s", signal_id, symbol)
            return False
        metadata["react_last_opposite_signal_id"] = signal_id
        db_position["position_metadata"] = metadata
        self.yuki_position_monitor._persist_position_metadata(db_position.get("id"), metadata)

        risk_metrics = await self.yuki_position_monitor._calculate_position_risk(db_position)
        if risk_metrics.should_close:
            # Genuine stop-loss/liquidation protection remains hard even when an
            # opposite signal happens to arrive in the same cycle.
            closed = await self.yuki_position_monitor._execute_risk_management_action(
                allocation.user_id,
                allocation.allocation_id,
                db_position,
                risk_metrics,
            )
            return bool(closed)
        result = await self.yuki_position_monitor._review_position_event(
            user_id=allocation.user_id,
            allocation_id=allocation.allocation_id,
            position=db_position,
            risk_metrics=risk_metrics,
            event=review_event,
            trigger_context={
                "incoming_signal_id": signal_id,
                "incoming_position_side": desired_position_side,
                "current_position_side": str(live_position.side).lower(),
            },
            # Signal generation already owns this lock. Avoid nesting it while
            # still executing the selected lifecycle action synchronously.
            acquire_cycle_lock=False,
        )
        execution_status = str(result.get("execution_status") or "")
        if execution_status == "full_exit_filled":
            logger.info(
                "Yuki chose FULL_EXIT for %s after reviewing opposite signal %s; reversal may proceed",
                symbol,
                signal_id,
            )
            return True
        logger.info(
            "Yuki chose %s for existing %s after opposite signal %s; new entry blocked",
            (result.get("decision") or {}).get("action") or "HOLD",
            symbol,
            signal_id,
        )
        return False

    async def _free_funds_by_cancelling_pending_orders(
        self,
        allocation: Any,
        required_amount: Optional[float] = None,
        incoming_signal: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Release lower-priority resting-entry margin for a better new signal.

        Only pending ledger rows that still own a plain exchange entry order are
        eligible.  Protective orders, partially-filled symbols with a live
        position, and entries that outrank the incoming signal are preserved.
        """
        if not self.hyperliquid_service or not allocation.platform_wallet_address:
            return False

        try:
            open_orders = await self.hyperliquid_service.get_open_orders(allocation.platform_wallet_address)
            if not open_orders:
                return False

            # Never release a row whose entry has partially or fully filled.  A
            # plain remainder can still be on the book while the symbol already
            # has live exposure.
            live_positions = await self.hyperliquid_service.get_positions_for_wallet(
                allocation.platform_wallet_address
            )
            if live_positions is None:
                logger.warning(
                    "Cannot verify live positions before pending-order fund rebalancing; preserving orders"
                )
                return False
            live_symbol_keys = {
                self._quote_stripped_symbol(symbol)
                for symbol, position in live_positions.items()
                if float(getattr(position, "size", 0) or 0) != 0
            }

            pending_rows = (
                self.supabase.table("agent_trades")
                .select(
                    "id,symbol,side,entry_price,status,trade_amount,signal_confidence,created_at,"
                    "hyperliquid_order_id,trade_metadata"
                )
                .eq("allocation_id", allocation.allocation_id)
                .eq("status", "pending")
                .not_.is_("hyperliquid_order_id", "null")
                .execute()
            ).data or []
            if not pending_rows:
                return False

            open_orders_by_id = {
                str(order.get("oid")): order
                for order in open_orders
                if order.get("oid") is not None
            }
            cancellable_rows = []
            live_price_cache: Dict[str, Optional[float]] = {}
            for row in pending_rows:
                order = open_orders_by_id.get(str(row.get("hyperliquid_order_id")))
                if not order or order.get("reduceOnly") or order.get("isTrigger"):
                    continue

                row_symbol_key = self._quote_stripped_symbol(row.get("symbol"))
                if self._quote_stripped_symbol(order.get("coin")) != row_symbol_key:
                    continue
                if row_symbol_key in live_symbol_keys:
                    continue

                metadata = dict(row.get("trade_metadata") or {})
                row_priority = self._yuki_signal_priority({
                    "opportunity_rank": metadata.get("opportunity_rank"),
                    "timestamp": metadata.get("signal_generated_at") or row.get("created_at"),
                    "confidence": row.get("signal_confidence"),
                })
                live_price = None
                order_symbol = str(order.get("coin") or "")
                live_price_symbol = order_symbol or str(row.get("symbol") or "")
                if self.hyperliquid_service and live_price_symbol not in live_price_cache:
                    try:
                        live_market = await self.hyperliquid_service.get_live_market_data(live_price_symbol)
                        live_price_cache[live_price_symbol] = (
                            float(getattr(live_market, "price", 0.0) or 0.0)
                            if live_market is not None
                            else None
                        )
                    except Exception:
                        live_price_cache[live_price_symbol] = None
                live_price = live_price_cache.get(live_price_symbol)

                should_yield, rebalance_basis, rebalance_score = (
                    self._yuki_pending_entry_should_yield(
                        pending_row=row,
                        pending_metadata=metadata,
                        incoming_signal=incoming_signal,
                        live_price=live_price,
                    )
                )
                if not should_yield:
                    continue
                cancellable_rows.append(
                    (rebalance_score, row_priority, row, order, metadata, rebalance_basis, live_price)
                )

            # Free the weakest/oldest reservations first and stop as soon as
            # the requested margin is available.
            cancellable_rows.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)

            freed_any = False
            for _, _, row, order, metadata, rebalance_basis, live_price in cancellable_rows:
                if required_amount is not None and allocation.remaining_amount + 1e-9 >= required_amount:
                    break
                order_id = order.get("oid")
                order_symbol = str(order.get("coin") or "")
                if not order_id or not order_symbol:
                    continue

                cancelled = await self.hyperliquid_service.cancel_order(order_symbol, order_id)
                if cancelled:
                    now = datetime.now(timezone.utc).isoformat()
                    metadata["entry_cancellation_reason"] = "insufficient_funds_priority_rebalance"
                    metadata["entry_rebalance_basis"] = rebalance_basis
                    metadata["entry_cancelled_at"] = now
                    if live_price is not None:
                        metadata["entry_rebalance_live_price"] = live_price
                    metadata["superseded_by_signal_id"] = (
                        incoming_signal or {}
                    ).get("signal_id")
                    updated = (
                        self.supabase.table("agent_trades")
                        .update({
                            "status": "cancelled",
                            "closed_at": now,
                            "trade_metadata": metadata,
                        })
                        .eq("id", row.get("id"))
                        .eq("status", "pending")
                        .execute()
                    ).data or []
                    if not updated:
                        logger.error(
                            "Cancelled exchange entry %s for %s but could not finalize pending trade %s",
                            order_id,
                            order_symbol,
                            row.get("id"),
                        )
                        continue
                    freed_any = True
                    allocation.remaining_amount += max(
                        0.0,
                        self._safe_float(row.get("trade_amount")),
                    )
                    logger.info(
                        "Cancelled lower-priority pending entry %s (%s) to free collateral "
                        "for signal %s in allocation %s",
                        order_id,
                        order_symbol,
                        (incoming_signal or {}).get("signal_id"),
                        allocation.allocation_id,
                    )

            if freed_any:
                # Reconcile allocation ledger to credit freed margin back to remaining_amount
                ledger = await self.db_service.reconcile_agent_allocation_ledger(allocation.allocation_id)
                if ledger is not None:
                    allocation.allocated_amount = ledger["allocated_amount"]
                    allocation.remaining_amount = ledger["remaining_amount"]
                    allocation.realized_pnl = ledger["realized_pnl"]
                return True

        except Exception as e:
            logger.warning(f"Failed to cancel pending orders for fund rebalancing: {e}")
        return False

    async def trigger_yuki_signal_check_for_allocation(self, allocation_id: str) -> None:
        """Trigger an immediate Yuki signal cycle for an allocation when funds are freed (e.g. after position exit)."""
        try:
            if not allocation_id or not self._is_yuki_execution_owner():
                return

            allocation = self.active_allocations.get(allocation_id)
            if allocation is None:
                allocation_rows = (
                    self.supabase.table("agent_allocations")
                    .select("*")
                    .eq("allocation_id", allocation_id)
                    .eq("status", "active")
                    .limit(1)
                    .execute()
                ).data or []
                if allocation_rows:
                    allocation = self._allocation_from_row(allocation_rows[0])

            if allocation and allocation.agent_type == AgentType.YUKI:
                logger.info(
                    f"⚡ Funds freed for allocation {allocation_id}. "
                    f"Triggering immediate Yuki signal check for active unentered signals..."
                )
                asyncio.create_task(self._run_yuki_signal_cycle_for_allocation(allocation))
        except Exception as e:
            logger.warning(f"Could not trigger immediate Yuki signal check for allocation {allocation_id}: {e}")

    async def manually_close_yuki_position(
        self,
        user_id: str,
        allocation_id: str,
        position_id: str,
        expected_side: str,
    ) -> Dict[str, Any]:
        """Close one user-owned Yuki position and reconcile its trade ledger."""
        requested_side = str(expected_side or "").strip().lower()
        if requested_side not in {"long", "short"}:
            return {"success": False, "status_code": 400, "error": "Position side must be long or short"}
        if not self.hyperliquid_service:
            return {"success": False, "status_code": 503, "error": "Yuki trading is temporarily unavailable"}

        try:
            # The signal cycle lock prevents Yuki from opening or reversing this
            # allocation while the user-requested close is being submitted.
            async with self.yuki_signal_cycle_lock:
                allocation_rows = (
                    self.supabase.table("agent_allocations")
                    .select("*")
                    .eq("allocation_id", allocation_id)
                    .eq("user_id", user_id)
                    .eq("agent_type", AgentType.YUKI.value)
                    .limit(1)
                    .execute()
                ).data or []
                if not allocation_rows:
                    return {"success": False, "status_code": 404, "error": "Yuki allocation not found"}

                allocation_row = allocation_rows[0]
                
                if position_id.startswith("live-"):
                    live_position_prefix = f"live-{allocation_id}-"
                    if not position_id.startswith(live_position_prefix):
                        return {"success": False, "status_code": 400, "error": "Invalid live position ID format"}
                    raw_symbol = position_id[len(live_position_prefix):].strip()
                    target_symbol = self._quote_stripped_symbol(raw_symbol)
                    if not target_symbol:
                        return {"success": False, "status_code": 400, "error": "Invalid live position ID format"}
                    
                    db_position = {
                        "id": position_id,
                        "symbol": target_symbol,
                        "side": requested_side,
                        "allocation_id": allocation_id,
                    }
                else:
                    position_rows = (
                        self.supabase.table("agent_positions")
                        .select("*")
                        .eq("id", position_id)
                        .eq("allocation_id", allocation_id)
                        .eq("user_id", user_id)
                        .eq("agent_type", AgentType.YUKI.value)
                        .eq("is_active", True)
                        .limit(1)
                        .execute()
                    ).data or []
                    if not position_rows:
                        await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
                        await self.trigger_yuki_signal_check_for_allocation(allocation_id)
                        return {
                            "success": True,
                            "allocation_id": allocation_id,
                            "position_id": position_id,
                            "already_closed": True,
                            "message": "This position is already closed.",
                        }

                    db_position = position_rows[0]
                    stored_side = str(db_position.get("side") or "").lower()
                    if stored_side != requested_side:
                        return {
                            "success": False,
                            "status_code": 409,
                            "error": "The position direction changed. Refresh before closing it.",
                        }
                    target_symbol = self._quote_stripped_symbol(db_position.get("symbol"))

                wallet_address = str(allocation_row.get("platform_wallet_address") or "")
                if not wallet_address:
                    return {"success": False, "status_code": 409, "error": "Yuki's trading wallet is unavailable"}

                live_positions = await self.hyperliquid_service.get_positions_for_wallet(wallet_address)
                if live_positions is None:
                    return {
                        "success": False,
                        "status_code": 503,
                        "error": "The live position could not be verified. Try again shortly.",
                    }
                target_sym_upper = str(target_symbol or "").strip().upper()
                raw_sym_upper = str(db_position.get("symbol") or "").strip().upper()

                live_position_match = next(
                    (
                        (str(coin), position)
                        for coin, position in live_positions.items()
                        if str(coin or "").strip().upper() in (target_sym_upper, raw_sym_upper)
                        or str(self._quote_stripped_symbol(coin)).strip().upper() in (target_sym_upper, raw_sym_upper)
                    ),
                    None,
                )
                if live_position_match is None:
                    if not position_id.startswith("live-"):
                        try:
                            self.supabase.table("agent_positions").update({
                                "is_active": False,
                                "updated_at": datetime.now(timezone.utc).isoformat()
                            }).eq("id", position_id).execute()
                        except Exception as update_err:
                            logger.warning(f"Could not update inactive status for {position_id}: {update_err}")
                    await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
                    await self.trigger_yuki_signal_check_for_allocation(allocation_id)
                    return {
                        "success": True,
                        "allocation_id": allocation_id,
                        "position_id": position_id,
                        "already_closed": True,
                        "message": "This position is already closed on Hyperliquid.",
                    }

                # HIP-3 market names are case-sensitive in the signing map. The
                # database normalizer intentionally uppercases symbols for
                # matching, so submit the exact coin name returned by the live
                # wallet state (for example ``xyz:BRENTOIL``) to the exchange.
                live_symbol, live_position = live_position_match

                live_side = str(live_position.side or "").lower()
                if live_side != requested_side:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "The live position direction changed. Refresh before closing it.",
                    }

                size = float(live_position.size or 0)
                if size <= 0:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "This position no longer has any size to close.",
                    }

                client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                    user_id=user_id,
                    wallet_address=wallet_address,
                )
                if not client_ready:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "Yuki's delegated trading permission is unavailable.",
                    }

                # Remove every remaining entry/protection order for this net
                # symbol before the full close so an SL/TP cannot race the
                # user's market exit or reopen exposure afterward.
                open_orders = await self.hyperliquid_service.get_open_orders(wallet_address)
                if open_orders is None:
                    return {
                        "success": False,
                        "status_code": 503,
                        "error": "Yuki could not verify the position's open orders. Try again shortly.",
                    }
                matching_orders = [
                    order
                    for order in open_orders
                    if self._quote_stripped_symbol(order.get("coin")) == target_symbol
                ]
                for order in matching_orders:
                    order_id = order.get("oid")
                    if order_id is None:
                        continue
                    order_symbol = str(order.get("coin") or target_symbol)
                    if not await self.hyperliquid_service.cancel_order(order_symbol, order_id):
                        return {
                            "success": False,
                            "status_code": 409,
                            "error": "A protective order could not be cancelled, so the position was left open.",
                        }

                # Re-point the shared delegated client immediately before the
                # financial action in case another wallet used this process.
                client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                    user_id=user_id,
                    wallet_address=wallet_address,
                )
                if not client_ready:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "Yuki's delegated trading permission is unavailable.",
                    }

                close_side = OrderSide.SELL if live_side == "long" else OrderSide.BUY
                close_result = await self.hyperliquid_service.place_order_live(
                    symbol=live_symbol,
                    side=close_side,
                    size=size,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                )
                if not close_result.success:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": close_result.error or "Hyperliquid rejected the close order.",
                    }

                if (
                    close_result.filled_size is not None
                    and float(close_result.filled_size) + max(1e-9, size * 1e-6) < size
                ):
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "The close only filled partially. Refresh to review the remaining position.",
                    }
                if str(close_result.status or "").lower() not in {"filled"}:
                    return {
                        "success": False,
                        "status_code": 409,
                        "error": "The close was submitted but not confirmed as filled. Refresh before trying again.",
                    }

                exit_price = float(
                    close_result.average_price
                    or close_result.price
                    or live_position.mark_price
                    or live_position.entry_price
                    or 0
                )
                entry_price = float(live_position.entry_price or 0)
                gross_pnl = (
                    (exit_price - entry_price) * size
                    if live_side == "long"
                    else (entry_price - exit_price) * size
                )
                fee = max(0.0, float(close_result.fee or 0))
                realized_pnl = gross_pnl - fee

                if not self.yuki_position_monitor:
                    from kata.services.yuki_position_monitor import create_yuki_position_monitor

                    self.yuki_position_monitor = create_yuki_position_monitor(self.hyperliquid_service)

                fill_summary = None
                for delay in (0.0, 0.5, 1.0):
                    if delay:
                        await asyncio.sleep(delay)
                    fill_summary = await self.yuki_position_monitor._fetch_closed_trade_fill_summary(
                        db_position
                    )
                    if fill_summary:
                        exit_price = float(fill_summary["average_exit_price"])
                        realized_pnl = float(fill_summary["net_realized_pnl"])
                        fee = float(fill_summary["fees"])
                        break

                manual_close = {
                    "source": "user",
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "expected_side": requested_side,
                }
                metadata_updates: Dict[str, Any] = {"manual_close": manual_close}
                if fill_summary:
                    metadata_updates["fill_reconciliation"] = fill_summary

                reconciled = await self.db_service.close_agent_positions_for_symbol(
                    allocation_id=allocation_id,
                    symbol=target_symbol,
                    exit_price=exit_price,
                    realized_pnl=realized_pnl,
                    fees=fee,
                    trade_metadata_updates=metadata_updates,
                )
                if not reconciled:
                    logger.error(
                        "Manual close filled for %s/%s but the trade ledger did not reconcile",
                        allocation_id,
                        target_symbol,
                    )

                logger.info(
                    "User manually closed Yuki %s %s for allocation %s at %s",
                    requested_side,
                    target_symbol,
                    allocation_id,
                    exit_price,
                )
                asyncio.create_task(self.trigger_yuki_signal_check_for_allocation(allocation_id))
                return {
                    "success": True,
                    "allocation_id": allocation_id,
                    "position_id": position_id,
                    "symbol": live_symbol,
                    "side": requested_side,
                    "closed_size": float(close_result.filled_size or size),
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "fees": fee,
                    "ledger_reconciled": bool(reconciled),
                    "message": f"{target_symbol} {requested_side.upper()} position closed",
                }
        except Exception as e:
            logger.error(
                "Error manually closing Yuki position %s for allocation %s: %s",
                position_id,
                allocation_id,
                e,
            )
            return {"success": False, "status_code": 500, "error": "Could not close this position"}

    async def _execute_yuki_trade(
        self,
        user_id: str,
        allocation: AgentAllocation,
        yuki_agent: YukiAgent,
        signal: Dict[str, Any]
    ) -> bool:
        """Execute a trade for Yuki agent using allocated funds."""
        try:
            symbol = signal["symbol"]
            signal_direction = signal["signal"]

            if signal_direction in ["BUY", "STRONG_BUY"]:
                side = OrderSide.BUY
                default_leverage = 3.0 if signal_direction == "STRONG_BUY" else 2.0
            elif signal_direction in ["SELL", "STRONG_SELL"]:
                side = OrderSide.SELL
                default_leverage = 3.0 if signal_direction == "STRONG_SELL" else 2.0
            else:
                return False

            signal_id = signal.get("signal_id")
            reversal_result = await self._close_opposite_yuki_position(
                allocation=allocation,
                symbol=symbol,
                desired_side=side,
                signal_id=str(signal_id) if signal_id else None,
            )
            if reversal_result is False:
                return False

            # Database-side exposure guard complements the live exchange check.
            # It closes the brief monitor race where an order disappears from the
            # book before its pending/filled row and position snapshot converge.
            try:
                exposure_rows = (
                    self.supabase.table("agent_trades")
                    .select("id,status,symbol,side,trade_metadata")
                    .eq("allocation_id", allocation.allocation_id)
                    .in_("status", ["pending", "filled"])
                    .execute()
                ).data or []
                target_symbol_key = self._quote_stripped_symbol(symbol)
                duplicate_exposure = next(
                    (
                        row for row in exposure_rows
                        if self._quote_stripped_symbol(row.get("symbol")) == target_symbol_key
                        and str(row.get("side") or "").lower() == side.value
                    ),
                    None,
                )
                if duplicate_exposure:
                    logger.warning(
                        f"Skipping signal {signal_id}: allocation {allocation.allocation_id} already has "
                        f"{duplicate_exposure.get('status')} {side.value} exposure for {symbol} "
                        f"in trade {duplicate_exposure.get('id')}"
                    )
                    return False
            except Exception as exposure_error:
                logger.warning(f"Symbol exposure database check failed; live guard remains active: {exposure_error}")

            confidence = self._safe_float(signal.get("confidence"))
            if confidence > 1:
                confidence = confidence / 100
            confidence = max(0.0, min(1.0, confidence))
            ideal_entry_price = self._positive_float(signal.get("price"))
            live_price = None
            if self.hyperliquid_service:
                live_data = await self.hyperliquid_service.get_live_market_data(symbol)
                live_price = self._positive_float(getattr(live_data, "price", None)) if live_data else None

            signal_attempts: List[Dict[str, Any]] = []
            if signal_id:
                try:
                    existing = (
                        self.supabase.table("agent_trades")
                        .select(
                            "id,status,symbol,side,entry_price,exit_price,position_size,"
                            "trade_amount,realized_pnl,max_profit_reached,max_loss_reached,"
                            "time_in_position_minutes,created_at,filled_at,closed_at,trade_metadata"
                        )
                        .eq("allocation_id", allocation.allocation_id)
                        .eq("trade_metadata->>signal_id", str(signal_id))
                        .in_("status", list(self._YUKI_SIGNAL_ATTEMPT_STATUSES))
                        .limit(20)
                        .execute()
                    )
                    signal_attempts = list(existing.data or [])
                except Exception as dedupe_error:
                    logger.warning(f"Signal attempt lookup failed (continuing): {dedupe_error}")

            if signal_id and signal_attempts:
                reviewed_signal = await self._maybe_prepare_yuki_signal_reentry(
                    user_id=user_id,
                    allocation=allocation,
                    yuki_agent=yuki_agent,
                    signal=signal,
                    side=side,
                    signal_attempts=signal_attempts,
                    live_price=live_price,
                )
                if reviewed_signal is None:
                    return False
                signal = reviewed_signal

            live_entry_allowed, guard_reason, guard_detail = self._yuki_live_entry_guard(signal)
            if not live_entry_allowed:
                logger.info(
                    "Skipping signal %s for allocation %s: %s",
                    signal_id or symbol,
                    allocation.allocation_id,
                    guard_detail or guard_reason or "live entry guard blocked the setup",
                )
                await self._queue_yuki_signal_reanalysis(
                    signal,
                    guard_reason or "yuki_live_entry_guard",
                    observed_price=live_price,
                    expire_entry_order=False,
                    invalidate_pending_entry=False,
                )
                return False

            # Dedupe: one entry attempt per signal per allocation. A completed
            # or cancelled trade consumes its entry-window revision. A reviewed
            # re-entry receives a higher attempt token without pretending the
            # source signal itself changed.
            if signal_id:
                try:
                    consumed_attempt = self._signal_attempt_consumed(
                        signal_attempts,
                        signal.get("entry_revision"),
                        signal.get("reentry_attempt"),
                    )
                    if consumed_attempt:
                        logger.info(
                            f"Skipping signal {signal_id} for allocation {allocation.allocation_id}: "
                            f"entry attempt {self._signal_attempt_key(consumed_attempt.get('trade_metadata') or {})} "
                            f"already has a {consumed_attempt.get('status')} trade"
                        )
                        return False
                except Exception as dedupe_error:
                    logger.warning(f"Signal dedupe check failed (continuing): {dedupe_error}")

            stop_loss = signal.get("stop_loss")
            path_terminal_event = await self._yuki_signal_path_terminal_event(
                signal=signal,
                symbol=symbol,
                side=side,
            )
            if path_terminal_event:
                logger.info(
                    "Skipping stale Yuki signal %s: %s",
                    signal_id or symbol,
                    path_terminal_event,
                )
                await self._queue_yuki_signal_reanalysis(
                    signal,
                    path_terminal_event,
                    observed_price=live_price,
                )
                return False

            # Move the entry a configured fraction toward the live market. This
            # avoids making every valid pullback depend on an exact-price print,
            # while the geometry guard below still refuses a stop/target cross.
            entry_improvement_fraction = max(
                0.0,
                min(
                    1.0,
                    self._safe_float(
                        getattr(settings, "YUKI_ENTRY_PRICE_IMPROVEMENT_FRACTION", 0.25),
                        0.25,
                    ),
                ),
            )
            current_price = self._yuki_improved_entry_price(
                entry_side=side,
                signal_entry_price=ideal_entry_price,
                live_price=live_price,
                stop_loss=stop_loss,
                target_1=signal.get("target_1"),
                improvement_fraction=entry_improvement_fraction,
                target_2=signal.get("target_2"),
            )
            if current_price is None:
                geometry_event = self._yuki_live_geometry_terminal_event(
                    entry_side=side,
                    live_price=live_price,
                    stop_loss=stop_loss,
                    target_1=signal.get("target_1"),
                    target_2=signal.get("target_2"),
                )
                logger.warning(
                    f"Skipping {symbol}: signal entry is unavailable or outside its stop/target range"
                )
                await self._queue_yuki_signal_reanalysis(
                    signal,
                    geometry_event or "current_price_crossed_entry_stop_or_target_geometry",
                    observed_price=live_price,
                )
                return False

            # Pre-entry gate disabled: AI signals from platform generator execute directly
            pre_entry_gate = None

            if not stop_loss and current_price > 0:
                if side == OrderSide.BUY:
                    stop_loss = current_price * (1 - allocation.trading_config.stop_loss_percent / 100)
                else:
                    stop_loss = current_price * (1 + allocation.trading_config.stop_loss_percent / 100)

            entry_order_expires_at = self._parse_row_datetime(
                signal.get("entry_order_expires_at")
            )
            if entry_order_expires_at and entry_order_expires_at.tzinfo is None:
                entry_order_expires_at = entry_order_expires_at.replace(tzinfo=timezone.utc)
            if entry_order_expires_at is None:
                entry_order_expires_at = self._yuki_entry_order_expires_at(
                    signal_started_at=signal.get("timestamp"),
                    signal_expires_at=signal.get("expires_at"),
                    ttl_hours=settings.YUKI_ENTRY_ORDER_TTL_HOURS,
                )
            if datetime.now(timezone.utc) >= entry_order_expires_at:
                logger.info(f"Skipping stale Yuki signal {signal_id or symbol}: entry window already expired")
                await self._queue_yuki_signal_reanalysis(
                    signal,
                    "entry_ttl_elapsed_before_submission",
                    observed_price=live_price,
                )
                return False

            signal_started_at = self._parse_row_datetime(signal.get("timestamp"))
            if signal_started_at and signal_started_at.tzinfo is None:
                signal_started_at = signal_started_at.replace(tzinfo=timezone.utc)
            entry_age_hours = (
                max(
                    0.0,
                    (datetime.now(timezone.utc) - signal_started_at).total_seconds() / 3600.0,
                )
                if signal_started_at
                else None
            )
            entry_distance_pct_at_order = (
                abs(float(live_price) - float(ideal_entry_price)) / float(ideal_entry_price) * 100.0
                if live_price is not None and ideal_entry_price
                else None
            )

            # Check if we have enough remaining funds
            min_trade_amount = settings.YUKI_MIN_TRADE_USDC
            if allocation.remaining_amount < min_trade_amount:
                await self._free_funds_by_cancelling_pending_orders(
                    allocation,
                    required_amount=min_trade_amount,
                    incoming_signal=signal,
                )
            if allocation.remaining_amount < min_trade_amount:
                logger.info(
                    f"Insufficient remaining funds for Yuki trade: ${allocation.remaining_amount:.2f}. "
                    "No lower-priority unfilled entry could be released."
                )
                return False

            requested_leverage = float(HyperliquidService.closest_supported_leverage(
                signal.get("leverage") or default_leverage,
            ))
            hl_max_leverage = None
            if self.hyperliquid_service:
                venue_limit = await self.hyperliquid_service.get_perp_max_leverage(symbol)
                if venue_limit:
                    hl_max_leverage = venue_limit
            leverage = float(HyperliquidService.closest_supported_leverage(
                requested_leverage,
                hl_max_leverage,
            ))
            if leverage != requested_leverage:
                logger.info(
                    "Adjusted %s platform-signal leverage from %sx to closest "
                    "Hyperliquid-supported %sx (market max: %s)",
                    symbol,
                    requested_leverage,
                    leverage,
                    hl_max_leverage,
                )

            signal_position_percent = self._positive_float(signal.get("position_size_percent"))
            allocation_equity = max(0.0, allocation.allocated_amount)
            # Preserve the platform signal's leverage and collateral percentage.
            # If the full signal size is unavailable, leave the signal pending;
            # silently shrinking the initial thesis changes its intended risk.
            position_fraction = self._yuki_signal_position_fraction(
                max_position_size_percent=allocation.trading_config.max_position_size_percent,
                signal_position_percent=signal_position_percent,
            )
            requested_trade_amount = allocation_equity * position_fraction
            if requested_trade_amount > allocation.remaining_amount + 1e-9:
                await self._free_funds_by_cancelling_pending_orders(
                    allocation,
                    required_amount=requested_trade_amount,
                    incoming_signal=signal,
                )
            if requested_trade_amount > allocation.remaining_amount + 1e-9:
                logger.info(
                    f"Skipping {symbol}: platform signal requests ${requested_trade_amount:.2f} "
                    f"margin but only ${allocation.remaining_amount:.2f} is available. "
                    "No lower-priority unfilled entry could be released, and the "
                    "initial platform-signal position will not be resized."
                )
                return False
            trade_amount = requested_trade_amount
            if trade_amount < min_trade_amount:
                logger.info(
                    f"Skipping {symbol}: platform signal requests ${trade_amount:.2f} margin, "
                    f"below the ${min_trade_amount:.2f} execution minimum."
                )
                return False

            # Calculate position size
            position_size = (trade_amount * leverage) / current_price if current_price > 0 else 0

            if position_size <= 0:
                logger.warning(f"Invalid position size calculated: {position_size}")
                return False

            if settings.YUKI_REQUIRE_DELEGATION:
                try:
                    from kata.services.delegation_service import get_delegation_service

                    policy_valid, policy_error = await get_delegation_service().validate_trade_against_policy(
                        user_id,
                        {
                            "volume_usd": trade_amount * leverage,
                            "position_size_percent": (trade_amount / allocation.allocated_amount) * 100,
                            "token_symbol": symbol,
                            "leverage": leverage,
                            "stop_loss": stop_loss,
                        }
                    )
                    if not policy_valid:
                        logger.warning(f"Yuki trade blocked by delegation policy: {policy_error}")
                        return False
                except Exception as e:
                    logger.error(f"Yuki delegation policy validation failed: {e}")
                    return False

            # Create trade execution record
            execution_id = str(uuid.uuid4())
            trade_execution = AgentTradeExecution(
                execution_id=execution_id,
                allocation_id=allocation.allocation_id,
                user_id=user_id,
                agent_type=AgentType.YUKI,
                symbol=symbol,
                side=side.value,
                size=position_size,
                price=current_price,
                amount_used=trade_amount,
                leverage=leverage,
                executed_at=datetime.now()
            )

            # Execute trade via Yuki agent and Hyperliquid service
            from kata.agents.base_agent import TradingSignal, SignalType

            # Convert signal to TradingSignal format
            signal_type = SignalType.BUY if side == OrderSide.BUY else SignalType.SELL
            if signal_direction in ["STRONG_BUY", "STRONG_SELL"]:
                signal_type = SignalType.STRONG_BUY if side == OrderSide.BUY else SignalType.STRONG_SELL

            reasoning = signal.get("reasoning", "")
            if isinstance(reasoning, list):
                reasoning = "; ".join(str(reason) for reason in reasoning)
            elif not isinstance(reasoning, str):
                reasoning = str(reasoning)

            trading_signal = TradingSignal(
                signal_type=signal_type,
                token_symbol=symbol,
                confidence=confidence,
                reasoning=reasoning or f"Yuki {signal_direction} platform signal",
                suggested_amount=Decimal(str(trade_amount)),
                suggested_price=Decimal(str(current_price)),
                metadata={
                    'analysis_type': 'futures_technical',
                    'execution_policy': 'signal_led_adaptive',
                    'leverage': leverage,
                    'futures_symbol': symbol,
                    'position_size': position_size,
                    'trade_amount': trade_amount,
                    'requested_leverage': requested_leverage,
                    'effective_position_fraction': position_fraction,
                    'signal_position_size_percent': signal_position_percent,
                    'position_size_multiplier': 1.0,
                    'allocation_equity_base': allocation_equity,
                    'requested_trade_amount': requested_trade_amount,
                    'allocation_id': allocation.allocation_id,
                    'signal_direction': signal_direction,
                    'signal_id': signal.get("signal_id"),
                    'entry_revision': signal.get("entry_revision", 0),
                    'reentry_attempt': signal.get("reentry_attempt", 0),
                    'opportunity_rank': signal.get("opportunity_rank"),
                    'stop_loss': stop_loss,
                    'target_1': signal.get("target_1"),
                    'target_2': signal.get("target_2"),
                    'time_horizon': signal.get("time_horizon"),
                    'signal_expires_at': (
                        signal.get("expires_at").isoformat()
                        if isinstance(signal.get("expires_at"), datetime)
                        else signal.get("expires_at")
                    ),
                    'signal_entry_price': ideal_entry_price,
                    'live_price_at_order': live_price,
                    'entry_improvement_fraction': entry_improvement_fraction,
                    'entry_order_expires_at': entry_order_expires_at.isoformat(),
                    'entry_limit_price': current_price,
                    'signal_generated_at': (
                        signal_started_at.isoformat() if signal_started_at else None
                    ),
                    'entry_age_hours_at_order': entry_age_hours,
                    'entry_distance_pct_at_order': entry_distance_pct_at_order,
                    'pre_entry_gate_decision': pre_entry_gate.get("decision") if pre_entry_gate else None,
                    'pre_entry_gate_reason': pre_entry_gate.get("reason") if pre_entry_gate else None,
                    'pre_entry_gate_confidence': pre_entry_gate.get("effective_confidence") if pre_entry_gate else None,
                    'pre_entry_gate_hard_veto': pre_entry_gate.get("hard_veto") if pre_entry_gate else None,
                    'edge_estimate': signal.get("edge_estimate"),
                    'market_structure_context': signal.get("market_structure_context"),
                    'policy_versions': signal.get("policy_versions"),
                    'policy_bundle_version': signal.get("policy_bundle_version"),
                    'prompt_version': signal.get("prompt_version"),
                    'reentry_review_action': (signal.get("reentry_review") or {}).get("action"),
                    'reentry_review_confidence': (signal.get("reentry_review") or {}).get("confidence"),
                    'reentry_review_reasoning': (signal.get("reentry_review") or {}).get("reasoning"),
                    'reentry_review_market_regime': (signal.get("reentry_review") or {}).get("market_regime"),
                    'reentry_review_conditions': (signal.get("reentry_review") or {}).get("next_review_conditions"),
                    'reentry_source_trade_id': (signal.get("reentry_review") or {}).get("source_trade_id"),
                }
            )

            # Execute via Yuki agent
            trade_results = await yuki_agent.execute_trades([trading_signal])
            order_result = trade_results[0] if trade_results else None

            entry_is_resting = bool(
                order_result
                and order_result.get('success')
                and str(order_result.get('status') or '').lower() == 'open'
            )

            if order_result and order_result.get('success'):
                order_id = order_result.get('order_id') or order_result.get('trade_id')
                execution_price = order_result.get('average_price') or order_result.get('execution_price') or current_price
                position_size, trade_amount = self._reconcile_order_amount(
                    order_result=order_result,
                    requested_size=position_size,
                    requested_trade_amount=trade_amount,
                    leverage=leverage,
                    execution_price=float(execution_price),
                )

                # Update trade execution record
                trade_execution.status = "pending" if entry_is_resting else "filled"
                trade_execution.hyperliquid_order_id = order_id
                trade_execution.price = execution_price
                trade_execution.size = position_size
                trade_execution.amount_used = trade_amount

                # Reserve the trade budget while the entry is working; the position
                # monitor credits it back if the resting order is cancelled/expires.
                allocation.remaining_amount -= trade_amount
                allocation.last_trade_at = datetime.now()

                # Update active allocation
                self.active_allocations[allocation.allocation_id] = allocation
                self.supabase.table("agent_allocations").update({
                    "remaining_amount": allocation.remaining_amount,
                    "last_trade_at": allocation.last_trade_at.isoformat(),
                }).eq("allocation_id", allocation.allocation_id).execute()

                if entry_is_resting:
                    logger.info(
                        f"Yuki resting entry: {side.value} {position_size} {symbol} "
                        f"limit @ ${order_result.get('entry_limit_price')} (${trade_amount} reserved, awaiting fill)"
                    )
                else:
                    logger.info(f"Yuki executed trade: {side.value} {position_size} {symbol} @ ${current_price} (${trade_amount} used)")

            else:
                # Trade failed
                trade_execution.status = "failed"
                trade_execution.error_message = order_result.get('error', 'Unknown error') if order_result else 'No result from agent'
                logger.error(f"Yuki trade failed: {trade_execution.error_message}")

            # Store trade execution in database
            await self._store_trade_execution(trade_execution)
            
            # Store comprehensive trade record
            if order_result and order_result.get('success'):
                order_id = order_result.get('order_id') or order_result.get('trade_id')
                execution_price = order_result.get('execution_price') or order_result.get('average_price') or current_price

                protective_orders = None
                if not entry_is_resting:
                    # Protective orders only make sense once there is a position;
                    # for resting entries the position monitor places them on fill.
                    protective_orders = await self._place_yuki_protective_orders(
                        symbol=symbol,
                        entry_side=side,
                        position_size=position_size,
                        entry_price=float(execution_price),
                        stop_loss=stop_loss,
                        target_1=signal.get("target_1"),
                        target_2=signal.get("target_2"),
                    )

                expires_at = signal.get("expires_at")
                if isinstance(expires_at, datetime):
                    expires_at = expires_at.isoformat()

                trade_record = {
                    "user_id": user_id,
                    "allocation_id": allocation.allocation_id,
                    "agent_type": "yuki",
                    "trade_type": "open_long" if side == OrderSide.BUY else "open_short",
                    "symbol": symbol,
                    "side": side.value,
                    "entry_price": order_result.get('entry_limit_price') if entry_is_resting else execution_price,
                    "position_size": position_size,
                    "leverage": leverage,
                    "trade_amount": trade_amount,
                    "fees": order_result.get("fee") or 0.0,
                    "status": "pending" if entry_is_resting else "filled",
                    "filled_at": None if entry_is_resting else datetime.now().isoformat(),
                    "hyperliquid_order_id": order_id,
                    "signal_confidence": confidence,
                    "signal_reasoning": reasoning,
                    "metadata": {
                        "execution_id": execution_id,
                        "execution_policy": "signal_led_adaptive",
                        "signal_direction": signal_direction,
                        "allocation_remaining": allocation.remaining_amount,
                        "requested_leverage": requested_leverage,
                        "effective_position_fraction": position_fraction,
                        "signal_position_size_percent": signal_position_percent,
                        "position_size_multiplier": 1.0,
                        "allocation_equity_base": allocation_equity,
                        "requested_trade_amount": requested_trade_amount,
                        "initial_size": position_size,
                        "initial_entry_price": float(execution_price),
                        "initial_stop_loss": stop_loss,
                        "initial_margin_used": trade_amount,
                        "stop_loss": stop_loss,
                        "target_1": signal.get("target_1"),
                        "target_2": signal.get("target_2"),
                        "signal_id": signal.get("signal_id"),
                        "opportunity_rank": signal.get("opportunity_rank"),
                        "entry_revision": signal.get("entry_revision", 0),
                        "reentry_attempt": signal.get("reentry_attempt", 0),
                        "signal_expires_at": expires_at,
                        "time_horizon": signal.get("time_horizon"),
                        "entry_order_expires_at": entry_order_expires_at.isoformat(),
                        "logo_url": signal.get("logo_url"),
                        "signal_entry_price": ideal_entry_price,
                        "live_price_at_order": live_price,
                        "entry_improvement_fraction": entry_improvement_fraction,
                        "entry_limit_price": order_result.get('entry_limit_price'),
                        "entry_order_id": order_id if entry_is_resting else None,
                        "signal_generated_at": (
                            signal_started_at.isoformat() if signal_started_at else None
                        ),
                        "entry_age_hours_at_order": entry_age_hours,
                        "entry_distance_pct_at_order": entry_distance_pct_at_order,
                        "pre_entry_gate_decision": pre_entry_gate.get("decision") if pre_entry_gate else None,
                        "pre_entry_gate_reason": pre_entry_gate.get("reason") if pre_entry_gate else None,
                        "pre_entry_gate_confidence": pre_entry_gate.get("effective_confidence") if pre_entry_gate else None,
                        "pre_entry_gate_hard_veto": pre_entry_gate.get("hard_veto") if pre_entry_gate else None,
                        "edge_estimate": signal.get("edge_estimate"),
                        "portfolio_selection": signal.get("portfolio_selection"),
                        "market_structure_context": signal.get("market_structure_context"),
                        "policy_versions": signal.get("policy_versions"),
                        "policy_bundle_version": signal.get("policy_bundle_version"),
                        "prompt_version": signal.get("prompt_version"),
                        "reentry_review": signal.get("reentry_review"),
                        "protective_orders": protective_orders
                    }
                }

                stored_trade = await self.db_service.store_agent_trade_record(trade_record)

                if entry_is_resting:
                    # Fill detection, protective orders, cancellation on expiry, and
                    # budget credit-back are handled by the Yuki position monitor.
                    return False

                stored_position = await self._store_yuki_position_record(
                    user_id=user_id,
                    allocation=allocation,
                    trade_id=(stored_trade or {}).get("id"),
                    symbol=symbol,
                    side=side,
                    position_size=position_size,
                    entry_price=float(execution_price),
                    leverage=leverage,
                    order_id=order_id,
                    stop_loss=stop_loss,
                    target_1=signal.get("target_1"),
                    target_2=signal.get("target_2"),
                    protective_orders=protective_orders,
                    signal_expires_at=expires_at,
                    time_horizon=signal.get("time_horizon"),
                    signal_id=signal.get("signal_id"),
                    reentry_attempt=signal.get("reentry_attempt", 0),
                    reentry_review=signal.get("reentry_review"),
                )
                await self.db_service.reconcile_agent_allocation_ledger(
                    allocation.allocation_id
                )
                await self._trigger_immediate_yuki_post_fill_review(
                    user_id=user_id,
                    allocation_id=allocation.allocation_id,
                    stored_position=stored_position,
                    stored_trade=stored_trade,
                    protective_orders=protective_orders,
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error executing Yuki trade: {e}")
            return False

    @staticmethod
    def _reconcile_order_amount(
        order_result: Dict[str, Any],
        requested_size: float,
        requested_trade_amount: float,
        leverage: float,
        execution_price: float,
    ) -> tuple:
        """Match Yuki bookkeeping to the venue-rounded or actually-filled size."""
        actual_size = (
            AgentAllocationService._positive_float(order_result.get("filled_size"))
            or AgentAllocationService._positive_float(order_result.get("size"))
            or requested_size
        )
        actual_margin = (
            actual_size * execution_price / max(float(leverage or 1), 1.0)
            if execution_price > 0
            else requested_trade_amount
        )
        actual_amount = min(float(requested_trade_amount), float(actual_margin))
        return float(actual_size), max(0.0, actual_amount)

    @staticmethod
    def _positive_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
            return parsed if parsed > 0 else None
        except (TypeError, ValueError):
            return None

    def _is_valid_exit_price(
        self,
        entry_side: OrderSide,
        entry_price: float,
        exit_price: Optional[float],
        exit_type: str
    ) -> bool:
        if not exit_price or entry_price <= 0:
            return False
        if entry_side == OrderSide.BUY:
            return exit_price < entry_price if exit_type == "sl" else exit_price > entry_price
        return exit_price > entry_price if exit_type == "sl" else exit_price < entry_price

    @staticmethod
    def _is_rate_limit_error(error: Any) -> bool:
        text = str(error or "").lower()
        return "429" in text or "rate limit" in text or "too many requests" in text

    async def _place_protective_trigger_with_retry(
        self,
        symbol: str,
        close_side: OrderSide,
        size: float,
        trigger_price: float,
        trigger_tpsl: str,
        max_attempts: int = 3,
    ) -> OrderResult:
        """
        Place a reduce-only SL/TP trigger, retrying rate-limit rejections in
        place. Entry, stop and both TP legs land within the same second, so a
        429 burst that admits the stop but rejects the TPs is the common
        failure - retrying with short exponential backoff here prevents the
        position from starting life without its exit plan.
        """
        last_result: Optional[OrderResult] = None
        for attempt in range(1, max_attempts + 1):
            try:
                last_result = await self.hyperliquid_service.place_order_live(
                    symbol=symbol,
                    side=close_side,
                    size=size,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                    trigger_price=trigger_price,
                    trigger_tpsl=trigger_tpsl,
                    trigger_is_market=True,
                )
                if last_result.success or not self._is_rate_limit_error(last_result.error):
                    return last_result
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    raise
                last_result = OrderResult(success=False, error=str(e))
            if attempt < max_attempts:
                delay = 2 ** attempt  # 2s, 4s
                logger.warning(
                    "Rate-limited placing %s %s trigger for %s (attempt %d/%d); retrying in %ds",
                    trigger_tpsl, trigger_price, symbol, attempt, max_attempts, delay,
                )
                await asyncio.sleep(delay)
        return last_result or OrderResult(success=False, error="rate limited")

    async def _place_yuki_protective_orders(
        self,
        symbol: str,
        entry_side: OrderSide,
        position_size: float,
        entry_price: float,
        stop_loss: Any,
        target_1: Any,
        target_2: Any,
    ) -> Dict[str, Any]:
        """Place reduce-only stop-loss and take-profit trigger orders after an entry fill."""
        result = {
            "stop_loss": None,
            "take_profit": [],
            "runner_enabled": False,
            "runner_size": 0.0,
            "execution_policy": "signal_led_adaptive",
            "enabled": bool(self.hyperliquid_service),
        }

        if not self.hyperliquid_service:
            result["error"] = "Hyperliquid service unavailable"
            return result

        close_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
        stop_loss_price = self._positive_float(stop_loss)
        target_prices = []
        seen_targets = set()
        for target in (self._positive_float(target_1), self._positive_float(target_2)):
            if (
                target
                and self._is_valid_exit_price(entry_side, entry_price, target, "tp")
                and target not in seen_targets
            ):
                target_prices.append(target)
                seen_targets.add(target)

        if stop_loss_price and self._is_valid_exit_price(entry_side, entry_price, stop_loss_price, "sl"):
            try:
                stop_result = await self._place_protective_trigger_with_retry(
                    symbol=symbol,
                    close_side=close_side,
                    size=position_size,
                    trigger_price=stop_loss_price,
                    trigger_tpsl="sl",
                )
                result["stop_loss"] = {
                    "success": stop_result.success,
                    "order_id": stop_result.order_id,
                    "price": stop_loss_price,
                    "size": position_size,
                    "error": stop_result.error,
                }
                if not stop_result.success:
                    # No exception, but the exchange rejected it - the position monitor's
                    # software stop-loss check is the only protection until this is retried.
                    logger.warning(
                        f"Yuki stop-loss REJECTED by exchange for {symbol} @ ${stop_loss_price}: "
                        f"{stop_result.error}. Position is unprotected on-exchange; "
                        "relying on software risk monitor fallback."
                    )
            except Exception as e:
                logger.error(f"Failed placing Yuki stop-loss order for {symbol}: {e}")
                result["stop_loss"] = {
                    "success": False,
                    "price": stop_loss_price,
                    "size": position_size,
                    "error": str(e),
                }
        elif stop_loss_price:
            result["stop_loss"] = {
                "success": False,
                "price": stop_loss_price,
                "error": "Invalid stop-loss side for entry price",
            }

        if target_prices:
            sz_decimals = await self.hyperliquid_service.get_perp_sz_decimals(symbol)
            planned_runner_size = 0.0
            if len(target_prices) == 2:
                # Bank at both platform targets and leave the balance under
                # Yuki's adaptive trailing-stop manager.
                first_fraction = max(0.0, min(1.0, settings.YUKI_TARGET_1_EXIT_FRACTION))
                second_fraction = max(0.0, min(1.0, settings.YUKI_TARGET_2_EXIT_FRACTION))
                first_size = self.hyperliquid_service._round_size_for_hl(
                    position_size * first_fraction, sz_decimals
                )
                second_size = self.hyperliquid_service._round_size_for_hl(
                    position_size * second_fraction, sz_decimals
                )
                # All three inputs are already venue-sized. Use decimal-place
                # rounding for the subtraction so binary float noise (for example
                # 0.04 - 0.01 - 0.01 = 0.019999...) cannot erase a size tick.
                size_decimals = sz_decimals if sz_decimals is not None else 4
                planned_runner_size = round(
                    position_size - first_size - second_size,
                    size_decimals,
                )
                if (
                    first_size > 0
                    and second_size > 0
                    and planned_runner_size > 0
                    and first_size + second_size < position_size
                ):
                    target_orders = [
                        (target_prices[0], first_size),
                        (target_prices[1], second_size),
                    ]
                else:
                    planned_runner_size = 0.0
                    target_orders = [(target_prices[-1], position_size)]
            else:
                target_orders = [(target_prices[0], position_size)]

            result["target_1_exit_fraction"] = settings.YUKI_TARGET_1_EXIT_FRACTION
            result["target_2_exit_fraction"] = settings.YUKI_TARGET_2_EXIT_FRACTION
            for target_price, target_size in target_orders:
                try:
                    tp_result = await self._place_protective_trigger_with_retry(
                        symbol=symbol,
                        close_side=close_side,
                        size=target_size,
                        trigger_price=target_price,
                        trigger_tpsl="tp",
                    )
                    result["take_profit"].append({
                        "success": tp_result.success,
                        "order_id": tp_result.order_id,
                        "price": target_price,
                        "size": target_size,
                        "error": tp_result.error,
                    })
                except Exception as e:
                    logger.error(f"Failed placing Yuki take-profit order for {symbol}: {e}")
                    result["take_profit"].append({
                        "success": False,
                        "price": target_price,
                        "size": target_size,
                        "error": str(e),
                    })

            result["runner_enabled"] = bool(
                planned_runner_size > 0
                and len(result["take_profit"]) == 2
                and all(order.get("success") for order in result["take_profit"])
            )
            result["runner_size"] = planned_runner_size if result["runner_enabled"] else 0.0

        return result

    async def _store_yuki_position_record(
        self,
        user_id: str,
        allocation: AgentAllocation,
        trade_id: Optional[str],
        symbol: str,
        side: OrderSide,
        position_size: float,
        entry_price: float,
        leverage: float,
        order_id: Optional[str],
        stop_loss: Any,
        target_1: Any,
        target_2: Any,
        protective_orders: Dict[str, Any],
        signal_expires_at: Any = None,
        time_horizon: Any = None,
        signal_id: Optional[str] = None,
        reentry_attempt: Any = 0,
        reentry_review: Optional[Dict[str, Any]] = None,
    ):
        """Create the active position row immediately after a successful Yuki entry."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            position_value = position_size * entry_price
            margin_used = position_value / leverage if leverage else position_value
            stop_record = (protective_orders or {}).get("stop_loss") or {}
            exchange_stop_live = bool(stop_record.get("success") and stop_record.get("order_id"))
            position_data = {
                "user_id": user_id,
                "allocation_id": allocation.allocation_id,
                "trade_id": trade_id,
                "agent_type": "yuki",
                "symbol": symbol,
                "side": "long" if side == OrderSide.BUY else "short",
                "size": position_size,
                "entry_price": entry_price,
                "current_price": entry_price,
                "leverage": leverage,
                "position_value": position_value,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_percent": 0.0,
                "margin_used": margin_used,
                "margin_ratio": (margin_used / position_value) if position_value > 0 else 1.0,
                "is_active": True,
                "hyperliquid_position_id": order_id or f"{symbol}_{int(datetime.now().timestamp())}",
                "metadata": {
                    "entry_order_id": order_id,
                    "signal_id": signal_id,
                    "reentry_attempt": self._signal_attempt_key({
                        "reentry_attempt": reentry_attempt,
                    })[1],
                    "reentry_review": dict(reentry_review or {}) or None,
                    "execution_policy": "signal_led_adaptive",
                    "stop_loss": stop_loss,
                    "active_stop": stop_loss if exchange_stop_live else None,
                    "policy_stop": stop_loss,
                    "stop_stage": "initial" if exchange_stop_live else "initial_repair_pending",
                    "target_1": target_1,
                    "target_2": target_2,
                    "protective_orders": protective_orders,
                    "runner_enabled": bool((protective_orders or {}).get("runner_enabled")),
                    "runner_size": (protective_orders or {}).get("runner_size"),
                    "original_size": position_size,
                    "initial_size": position_size,
                    "initial_entry_price": entry_price,
                    "initial_stop_loss": stop_loss,
                    "initial_margin_used": margin_used,
                    "signal_expires_at": signal_expires_at,
                    "time_horizon": time_horizon,
                    "created_from_trade_execution": True,
                    "react_review_events": {
                        "entry_filled": {
                            "status": "pending",
                            "requested_at": now,
                        }
                    },
                }
            }
            return await self.db_service.store_agent_position(position_data)
        except Exception as e:
            logger.error(f"Error storing Yuki position record for {symbol}: {e}")
            return None

    async def _trigger_immediate_yuki_post_fill_review(
        self,
        user_id: str,
        allocation_id: str,
        stored_position: Optional[Dict[str, Any]],
        stored_trade: Optional[Dict[str, Any]],
        protective_orders: Optional[Dict[str, Any]],
    ) -> bool:
        """Review an immediately filled entry while its signal-cycle lock is held."""
        if not stored_position or not stored_position.get("id"):
            logger.error(
                "Could not start immediate Yuki post-fill review for allocation %s: "
                "the live position row is unavailable",
                allocation_id,
            )
            return False

        try:
            if not self.yuki_position_monitor:
                from kata.services.yuki_position_monitor import create_yuki_position_monitor

                self.yuki_position_monitor = create_yuki_position_monitor(
                    self.hyperliquid_service,
                    react_agent_resolver=self._resolve_yuki_react_agent,
                )
            else:
                self.yuki_position_monitor.react_agent_resolver = self._resolve_yuki_react_agent

            # Immediate entries are finalized inside yuki_signal_cycle_lock.
            # ReAct actions must reuse that ownership instead of attempting to
            # acquire the non-reentrant lock a second time.
            await self.yuki_position_monitor._trigger_post_fill_review(
                user_id=user_id,
                allocation_id=allocation_id,
                position_row=stored_position,
                live_position=None,
                filled_trade=stored_trade,
                protective_orders=protective_orders,
                acquire_cycle_lock=False,
            )
            return True
        except Exception as exc:
            # The position was stored with a durable pending entry_filled event;
            # the normal monitor will retry it on its next event-driven cycle.
            logger.error(
                "Immediate Yuki post-fill review failed for allocation %s; "
                "the durable monitor will retry it: %s",
                allocation_id,
                exc,
            )
            return False

    async def _store_trade_execution(self, execution: AgentTradeExecution):
        """Store trade execution in database."""
        try:
            execution_data = {
                "execution_id": execution.execution_id,
                "allocation_id": execution.allocation_id,
                "user_id": execution.user_id,
                "agent_type": execution.agent_type.value,
                "symbol": execution.symbol,
                "side": execution.side,
                "size": execution.size,
                "price": execution.price,
                "amount_used": execution.amount_used,
                "leverage": execution.leverage,
                "executed_at": execution.executed_at.isoformat(),
                "hyperliquid_order_id": execution.hyperliquid_order_id,
                "status": execution.status,
                "error_message": execution.error_message
            }

            result = self.supabase.table("agent_trade_executions").insert(execution_data).execute()
            if result.data:
                logger.info(f"Stored trade execution {execution.execution_id} in database")

        except Exception as e:
            logger.error(f"Error storing trade execution: {e}")

    async def get_allocation_status(self, user_id: str, allocation_id: str = None) -> Dict[str, Any]:
        """Get durable, reconciled allocation status for the user.

        The database is authoritative. A Railway process cache may be behind
        another worker's fills, pause, or P&L reconciliation, so it is only a
        fallback when no persisted row exists.
        """
        try:
            if allocation_id:
                result = self.supabase.table("agent_allocations").select("*").eq(
                    "allocation_id", allocation_id
                ).eq("user_id", user_id).execute()

                if result.data:
                    row = dict(result.data[0])
                    if str(row.get("agent_type") or "").lower() == AgentType.YUKI.value:
                        totals = await self.db_service.reconcile_agent_allocation_ledger(
                            allocation_id
                        )
                        if totals:
                            row.update({
                                "allocated_amount": totals["allocated_amount"],
                                "remaining_amount": totals["remaining_amount"],
                                "realized_pnl": totals["realized_pnl"],
                                "unrealized_pnl": totals.get("unrealized_pnl", 0.0),
                                "total_trades": totals.get("total_trades", 0),
                            })
                    return self._format_allocation_row(row)

                allocation = self.active_allocations.get(allocation_id)
                if allocation and allocation.user_id == user_id:
                    return self._format_allocation_status(allocation)

                return {"error": "Allocation not found"}

            result = self.supabase.table("agent_allocations").select("*").eq(
                "user_id", user_id
            ).order("created_at", desc=True).execute()

            user_allocations: List[Dict[str, Any]] = []
            persisted_ids = set()
            for persisted_row in result.data or []:
                row = dict(persisted_row)
                row_id = str(row.get("allocation_id") or "")
                if row_id:
                    persisted_ids.add(row_id)
                if str(row.get("agent_type") or "").lower() == AgentType.YUKI.value:
                    totals = await self.db_service.reconcile_agent_allocation_ledger(row_id)
                    if totals:
                        row.update({
                            "allocated_amount": totals["allocated_amount"],
                            "remaining_amount": totals["remaining_amount"],
                            "realized_pnl": totals["realized_pnl"],
                            "unrealized_pnl": totals.get("unrealized_pnl", 0.0),
                            "total_trades": totals.get("total_trades", 0),
                        })
                user_allocations.append(self._format_allocation_row(row))

            for cached_id, allocation in self.active_allocations.items():
                if allocation.user_id == user_id and cached_id not in persisted_ids:
                    user_allocations.append(self._format_allocation_status(allocation))

            return {"allocations": user_allocations}

        except Exception as e:
            logger.error(f"Error getting allocation status: {e}")
            return {"error": str(e)}

    def _format_allocation_status(self, allocation: AgentAllocation) -> Dict[str, Any]:
        """Format allocation status for API response."""
        execution_policy = (
            "signal_led_adaptive" if allocation.agent_type == AgentType.YUKI else None
        )
        return {
            "allocation_id": allocation.allocation_id,
            "agent_type": allocation.agent_type.value,
            "allocated_amount": allocation.allocated_amount,
            "remaining_amount": allocation.remaining_amount,
            "used_amount": allocation.allocated_amount - allocation.remaining_amount,
            "status": allocation.status.value,
            "total_trades": allocation.total_trades,
            "realized_pnl": allocation.realized_pnl,
            "unrealized_pnl": allocation.unrealized_pnl,
            "created_at": allocation.created_at.isoformat(),
            "last_trade_at": allocation.last_trade_at.isoformat() if allocation.last_trade_at else None,
            "performance_metrics": allocation.performance_metrics or {},
            "execution_policy": execution_policy,
        }

    def _format_allocation_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Format a persisted allocation row for API response."""
        allocated_amount = float(row.get("allocated_amount") or 0)
        remaining_amount = float(row.get("remaining_amount") or 0)
        execution_policy = (
            "signal_led_adaptive"
            if str(row.get("agent_type") or "").lower() == AgentType.YUKI.value
            else None
        )
        return {
            "allocation_id": row.get("allocation_id"),
            "agent_type": row.get("agent_type"),
            "allocated_amount": allocated_amount,
            "remaining_amount": remaining_amount,
            "used_amount": allocated_amount - remaining_amount,
            "status": row.get("status"),
            "total_trades": int(row.get("total_trades") or 0),
            "realized_pnl": float(row.get("realized_pnl") or 0),
            "unrealized_pnl": float(row.get("unrealized_pnl") or 0),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "last_trade_at": row.get("last_trade_at"),
            "performance_metrics": row.get("performance_metrics") or {},
            "execution_policy": execution_policy,
        }

    async def withdraw_from_hyperliquid(
        self,
        user_id: str,
        amount: float,
        access_token: str,
        wallet_address: str,
    ) -> Dict[str, Any]:
        """
        Withdraw USDC from the user's Hyperliquid account back to their Kata
        wallet on Arbitrum.

        Uses Hyperliquid's native withdraw3 action signed by the delegated
        wallet. Hyperliquid deducts a flat $1 fee and sends Arbitrum USDC to
        the destination in a few minutes. Only the withdrawable portion
        (balance not locked as margin) can move; Hyperliquid enforces this on
        its side as well.
        """
        HL_WITHDRAWAL_FEE_USDC = 1.0
        MIN_WITHDRAWAL_USDC = 2.0

        try:
            if not wallet_address:
                return {"success": False, "error": "No Kata wallet address found for this user"}
            if amount < MIN_WITHDRAWAL_USDC:
                return {
                    "success": False,
                    "error": f"Minimum withdrawal is ${MIN_WITHDRAWAL_USDC:.2f} USDC (Hyperliquid charges a $1 fee).",
                }

            # Resolve the true withdrawable balance in an account-mode-aware way.
            # Unified / portfolio-margin accounts keep their collateral in the
            # spot wallet, so the perp `withdrawable` field understates what can
            # actually be moved (and is what the withdraw UI's "Max" reflects).
            account_mode: Optional[str] = None
            withdrawable = 0.0
            if self.hyperliquid_service:
                try:
                    account_mode = await self.hyperliquid_service.get_user_abstraction_mode(wallet_address)
                    summary = await self.hyperliquid_service.get_wallet_balance_summary(
                        wallet_address, account_mode=account_mode
                    )
                    withdrawable = float(summary.get("withdrawable") or 0)
                except Exception as summary_error:
                    logger.warning(
                        f"Could not read mode-aware balance for {wallet_address}; "
                        f"falling back to perp state: {summary_error}"
                    )

            if withdrawable <= 0:
                # Fallback: read the perp clearinghouse state directly.
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        "https://api.hyperliquid.xyz/info",
                        json={"type": "clearinghouseState", "user": wallet_address},
                    )
                    response.raise_for_status()
                    user_state = response.json() or {}
                withdrawable = float(user_state.get("withdrawable") or 0)

            if withdrawable < MIN_WITHDRAWAL_USDC:
                return {
                    "success": False,
                    "error": (
                        f"Withdrawable Hyperliquid balance is ${withdrawable:.2f}. Funds locked as margin "
                        "in open positions cannot be withdrawn until the positions close."
                    ),
                }

            # If the requested amount exceeds live withdrawable balance (due to UI
            # drift, fee reserve, or margin difference), automatically clamp the
            # withdrawal to the maximum currently withdrawable on Hyperliquid.
            if amount > withdrawable:
                clamped = float(int(withdrawable * 100) / 100)
                logger.info(
                    "Clamping requested withdrawal of $%.2f to max withdrawable balance $%.2f for wallet %s",
                    amount, clamped, wallet_address
                )
                amount = clamped

            if not self.hyperliquid_service:
                return {"success": False, "error": "Hyperliquid service is not available"}

            client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                user_id=user_id,
                wallet_address=wallet_address,
            )
            exchange_client = (
                self.hyperliquid_service.get_active_exchange_client(wallet_address)
                if client_ready else None
            )
            if not exchange_client:
                return {"success": False, "error": "Hyperliquid trading access is not available for this wallet"}

            # In unified / portfolio-margin mode the spendable USDC lives in the
            # spot wallet, but the bridge withdrawal (withdraw3) draws from the
            # perp balance. Move any shortfall spot -> perp before withdrawing so
            # the full mode-aware withdrawable can actually leave the account.
            if account_mode in {"unifiedAccount", "portfolioMargin"}:
                await self._ensure_perp_balance_for_withdrawal(
                    exchange_client, wallet_address, amount
                )

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: exchange_client.withdraw_from_bridge(amount, wallet_address)
            )

            if not isinstance(result, dict) or result.get("status") != "ok":
                return {"success": False, "error": f"Hyperliquid rejected the withdrawal: {result}"}

            await self._reduce_allocations_after_withdrawal(user_id, amount)

            logger.info(f"Withdrew ${amount:.2f} USDC from Hyperliquid to {wallet_address} for user {user_id}")
            return {
                "success": True,
                "amount_usdc": amount,
                "fee_usdc": HL_WITHDRAWAL_FEE_USDC,
                "net_usdc": max(0.0, amount - HL_WITHDRAWAL_FEE_USDC),
                "destination": wallet_address,
                "estimated_arrival_minutes": 5,
                "message": (
                    f"Withdrawing ${amount:.2f} USDC to your Kata wallet on Arbitrum. "
                    f"Hyperliquid deducts a $1 fee; funds arrive in about 5 minutes."
                ),
            }

        except Exception as e:
            logger.error(f"Error withdrawing from Hyperliquid for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _read_perp_withdrawable(self, wallet_address: str) -> float:
        """Read the USDC currently withdrawable from the perp (bridge) balance."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.hyperliquid.xyz/info",
                    json={"type": "clearinghouseState", "user": wallet_address},
                )
                response.raise_for_status()
                state = response.json() or {}
            return float(state.get("withdrawable") or 0)
        except Exception as e:
            logger.warning(f"Could not read perp withdrawable for {wallet_address}: {e}")
            return 0.0

    async def _ensure_perp_balance_for_withdrawal(
        self, exchange_client: Any, wallet_address: str, amount: float
    ) -> None:
        """
        Top up the perp balance from spot so a bridge withdrawal of ``amount``
        can settle on unified / portfolio-margin accounts.

        The bridge withdrawal (withdraw3) sources USDC from the perp balance,
        but in these account modes the collateral sits in the spot wallet. Move
        the shortfall spot -> perp via a USD class transfer; Hyperliquid settles
        it internally within a second or two.
        """
        try:
            perp_withdrawable = await self._read_perp_withdrawable(wallet_address)
            shortfall = amount - perp_withdrawable
            if shortfall <= 0.01:
                return

            # Add a cent of buffer to absorb rounding at Hyperliquid's side.
            transfer_amount = round(shortfall + 0.01, 2)
            logger.info(
                f"Moving ${transfer_amount:.2f} USDC spot -> perp for {wallet_address} "
                f"before withdrawal (perp withdrawable ${perp_withdrawable:.2f}, need ${amount:.2f})"
            )
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: exchange_client.usd_class_transfer(transfer_amount, True)
            )

            # Wait for the internal transfer to reflect in the perp balance.
            for _ in range(10):
                await asyncio.sleep(0.5)
                if await self._read_perp_withdrawable(wallet_address) >= amount - 0.01:
                    return
            logger.warning(
                f"Perp balance did not reach ${amount:.2f} after spot -> perp transfer "
                f"for {wallet_address}; attempting withdrawal anyway"
            )
        except Exception as e:
            logger.warning(
                f"Could not move spot collateral to perp before withdrawal for {wallet_address}: {e}"
            )

    async def _reduce_allocations_after_withdrawal(self, user_id: str, withdrawn_usdc: float):
        """
        Shrink allocation records so they never claim more than what remains
        on Hyperliquid. Newest allocations are reduced first; an allocation
        reduced to zero is stopped.
        """
        try:
            result = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("user_id", user_id)
                .eq("agent_type", "yuki")
                # A worker can observe the emptied Hyperliquid account and mark
                # the allocation stopped before this request reaches its
                # accounting update. Include stopped rows so that race cannot
                # leave withdrawn capital recorded as still allocated.
                .in_("status", ["active", "paused", "stopped"])
                .order("created_at", desc=True)
                .execute()
            )
            remaining_to_reduce = withdrawn_usdc
            for row in result.data or []:
                if remaining_to_reduce <= 0:
                    break
                allocated = float(row.get("allocated_amount") or 0)
                remaining = float(row.get("remaining_amount") or 0)
                reduction = min(allocated, remaining_to_reduce)
                new_allocated = max(0.0, allocated - reduction)
                new_remaining = max(0.0, min(remaining, new_allocated))

                performance_metrics = dict(row.get("performance_metrics") or {})
                capital_metrics = dict(performance_metrics.get("capital") or {})
                accounting_version = int(capital_metrics.get("accounting_version") or 1)
                if accounting_version >= 2:
                    principal = float(
                        capital_metrics.get("principal_allocated_amount")
                        if capital_metrics.get("principal_allocated_amount") is not None
                        else allocated
                    )
                else:
                    principal = allocated
                capital_metrics.update({
                    "accounting_version": 2,
                    "principal_allocated_amount": max(0.0, principal - reduction),
                    "compounded_equity": new_allocated,
                    "last_capital_flow_at": datetime.now().isoformat(),
                })
                performance_metrics["capital"] = capital_metrics

                update: Dict[str, Any] = {
                    "allocated_amount": new_allocated,
                    "remaining_amount": new_remaining,
                    "performance_metrics": performance_metrics,
                    "updated_at": datetime.now().isoformat(),
                }
                if new_allocated < settings.YUKI_MIN_ALLOCATION_USDC:
                    update["status"] = AllocationStatus.STOPPED.value
                self.supabase.table("agent_allocations").update(update).eq(
                    "allocation_id", row["allocation_id"]
                ).eq("user_id", user_id).execute()

                cached = self.active_allocations.get(row["allocation_id"])
                if cached:
                    cached.allocated_amount = new_allocated
                    cached.remaining_amount = new_remaining
                    cached.performance_metrics = performance_metrics
                    if update.get("status") == AllocationStatus.STOPPED.value:
                        cached.status = AllocationStatus.STOPPED
                remaining_to_reduce -= reduction
        except Exception as e:
            logger.warning(f"Could not adjust allocations after withdrawal for {user_id}: {e}")

    async def _cancel_all_pending_orders_for_allocation(
        self,
        allocation_id: str,
        user_id: str,
        platform_wallet_address: Optional[str] = None,
    ) -> None:
        """Cancel open exchange resting orders and mark pending trades in agent_trades as cancelled."""
        try:
            pending_rows = (
                self.supabase.table("agent_trades")
                .select("id,symbol,status,hyperliquid_order_id,trade_metadata")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .eq("agent_type", AgentType.YUKI.value)
                .in_("status", ["pending", "pending_entry"])
                .execute()
            ).data or []
            if not pending_rows:
                return

            wallet_address = platform_wallet_address
            if not wallet_address and allocation_id in self.active_allocations:
                wallet_address = getattr(self.active_allocations[allocation_id], "platform_wallet_address", None)

            claimed_order_ids = {
                str(row.get("hyperliquid_order_id"))
                for row in pending_rows
                if row.get("hyperliquid_order_id") is not None
            }
            failed_order_ids: set[str] = set()
            if claimed_order_ids:
                if not (self.hyperliquid_service and wallet_address):
                    logger.warning(
                        "Cannot verify exchange entries while stopping allocation %s; preserving pending rows",
                        allocation_id,
                    )
                    return

                client_ready = await self.hyperliquid_service.ensure_delegated_trading_client(
                    user_id=user_id,
                    wallet_address=wallet_address,
                )
                open_orders = (
                    await self.hyperliquid_service.get_open_orders(wallet_address)
                    if client_ready
                    else None
                )
                if open_orders is None:
                    logger.warning(
                        "Cannot inspect exchange entries while stopping allocation %s; preserving pending rows",
                        allocation_id,
                    )
                    return

                # Only cancel plain entries claimed by this allocation. Never
                # touch protective orders or another allocation sharing the wallet.
                for order in open_orders:
                    order_id = str(order.get("oid"))
                    if order_id not in claimed_order_ids:
                        continue
                    if order.get("reduceOnly") or order.get("isTrigger"):
                        failed_order_ids.add(order_id)
                        continue
                    symbol = str(order.get("coin") or "")
                    if not symbol or not await self.hyperliquid_service.cancel_order(
                        symbol,
                        order.get("oid"),
                    ):
                        failed_order_ids.add(order_id)

            now_iso = datetime.now(timezone.utc).isoformat()
            finalized = 0
            for row in pending_rows:
                trade_id = row.get("id")
                if not trade_id:
                    continue
                order_id = (
                    str(row.get("hyperliquid_order_id"))
                    if row.get("hyperliquid_order_id") is not None
                    else None
                )
                if order_id and order_id in failed_order_ids:
                    continue
                metadata = dict(row.get("trade_metadata") or {})
                metadata.update({
                    "entry_cancellation_reason": "allocation_stopped",
                    "entry_cancelled_at": now_iso,
                })
                updated = (
                    self.supabase.table("agent_trades")
                    .update({
                        "status": "cancelled",
                        "closed_at": now_iso,
                        "trade_metadata": metadata,
                    })
                    .eq("id", trade_id)
                    .in_("status", ["pending", "pending_entry"])
                    .execute()
                ).data or []
                if updated:
                    finalized += 1

            if finalized:
                await self.db_service.reconcile_agent_allocation_ledger(allocation_id)
                logger.info(
                    "Marked %d pending trade(s) as cancelled for allocation %s",
                    finalized,
                    allocation_id,
                )
        except Exception as e:
            logger.error("Error cancelling pending orders for allocation %s: %s", allocation_id, e)

    async def pause_agent_trading(self, user_id: str, allocation_id: str) -> Dict[str, Any]:
        """Pause automated trading for an agent allocation."""
        try:
            if allocation_id in self.active_allocations:
                allocation = self.active_allocations[allocation_id]
                if allocation.user_id == user_id:
                    if allocation.status != AllocationStatus.ACTIVE:
                        return {
                            "success": False,
                            "error": "Only an active agent allocation can be paused",
                        }
                    result = self.supabase.table("agent_allocations").update({
                        "status": AllocationStatus.PAUSED.value
                    }).eq("allocation_id", allocation_id).eq(
                        "user_id", user_id
                    ).eq(
                        "status", AllocationStatus.ACTIVE.value
                    ).execute()
                    if result.data:
                        allocation.status = AllocationStatus.PAUSED
                        logger.info(f"Paused agent trading for allocation {allocation_id}")
                        await self._cancel_all_pending_orders_for_allocation(
                            allocation_id, user_id, getattr(allocation, "platform_wallet_address", None)
                        )
                        return {"success": True, "status": "paused"}
                    return {
                        "success": False,
                        "error": "Only an active agent allocation can be paused",
                    }
                else:
                    return {"success": False, "error": "Unauthorized"}
            else:
                result = self.supabase.table("agent_allocations").update({
                    "status": AllocationStatus.PAUSED.value
                }).eq("allocation_id", allocation_id).eq(
                    "user_id", user_id
                ).eq(
                    "status", AllocationStatus.ACTIVE.value
                ).execute()
                if result.data:
                    logger.info(f"Paused persisted agent allocation {allocation_id}")
                    await self._cancel_all_pending_orders_for_allocation(allocation_id, user_id)
                    return {"success": True, "status": "paused"}
                return {
                    "success": False,
                    "error": "Only an active agent allocation can be paused",
                }

        except Exception as e:
            logger.error(f"Error pausing agent trading: {e}")
            return {"success": False, "error": str(e)}

    async def stop_agent_trading(self, user_id: str, allocation_id: str) -> Dict[str, Any]:
        """Stop an allocation and return its recorded capital to the shared pool."""
        try:
            existing = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .in_("status", [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value])
                .limit(1)
                .execute()
            )
            rows = existing.data or []
            if not rows:
                return {"success": False, "error": "Allocation not found"}

            existing_row = rows[0]
            if str(existing_row.get("agent_type") or "").lower() == AgentType.RYU.value:
                # Freeze new entries before closing holdings. A failed stop
                # remains paused so the worker cannot race the recovery path.
                self.supabase.table("agent_allocations").update({
                    "status": AllocationStatus.PAUSED.value,
                    "updated_at": datetime.now().isoformat(),
                }).eq("allocation_id", allocation_id).eq("user_id", user_id).execute()
                allocation = self._allocation_from_row({
                    **existing_row,
                    "status": AllocationStatus.PAUSED.value,
                })
                if not allocation:
                    return {"success": False, "error": "Could not restore Ryu allocation"}
                liquidation = await self.ryu_spot_trading_service.liquidate_all_positions(
                    allocation
                )
                if not liquidation.get("success"):
                    return liquidation

                refreshed = (
                    self.supabase.table("agent_allocations")
                    .select("*")
                    .eq("allocation_id", allocation_id)
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                refreshed_rows = refreshed.data or []
                if not refreshed_rows:
                    return {"success": False, "error": "Ryu allocation disappeared during stop"}
                available = float(refreshed_rows[0].get("remaining_amount") or 0)
                if available > 0:
                    returned = await self.withdraw_funds_from_allocation(
                        user_id=user_id,
                        allocation_id=allocation_id,
                        amount=available,
                    )
                    if not returned.get("success"):
                        return returned
                    self.supabase.table("agent_allocations").update({
                        "status": AllocationStatus.STOPPED.value,
                        "allocated_amount": 0,
                        "remaining_amount": 0,
                        "updated_at": datetime.now().isoformat(),
                    }).eq("allocation_id", allocation_id).eq("user_id", user_id).execute()
                    self.active_allocations.pop(allocation_id, None)
                    return {
                        "success": True,
                        "status": AllocationStatus.STOPPED.value,
                        "returned_amount": available,
                        "positions_closed": liquidation.get("closed", 0),
                        "transaction_hash": returned.get("transaction_hash"),
                        "estimated_base_usdc": returned.get("estimated_base_usdc"),
                        "message": "Ryu stopped and is returning USDC to Kata Balance on Base",
                    }

            result = (
                self.supabase.table("agent_allocations")
                .update({
                    "status": AllocationStatus.STOPPED.value,
                    "allocated_amount": 0.0,
                    "remaining_amount": 0.0,
                    "updated_at": datetime.now().isoformat(),
                })
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .in_("status", [AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value])
                .execute()
            )
            if not result.data:
                return {"success": False, "error": "Allocation not found"}

            self.active_allocations.pop(allocation_id, None)
            existing_yuki = self.agent_instances.get(user_id, {}).get(AgentType.YUKI.value)
            if existing_yuki and getattr(existing_yuki.get("allocation"), "allocation_id", None) == allocation_id:
                self.agent_instances[user_id].pop(AgentType.YUKI.value, None)

            wallet_address = result.data[0].get("platform_wallet_address") or existing_row.get("platform_wallet_address")
            await self._cancel_all_pending_orders_for_allocation(allocation_id, user_id, wallet_address)

            logger.info(f"Stopped agent allocation {allocation_id} for user {user_id}")
            return {
                "success": True,
                "status": AllocationStatus.STOPPED.value,
                "returned_amount": float(result.data[0].get("allocated_amount") or 0.0),
            }
        except Exception as e:
            logger.error(f"Error stopping agent allocation {allocation_id}: {e}")
            return {"success": False, "error": str(e)}

    async def resume_agent_trading(self, user_id: str, allocation_id: str) -> Dict[str, Any]:
        """Resume automated trading for an agent allocation."""
        try:
            result = (
                self.supabase.table("agent_allocations")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            if not rows:
                return {"success": False, "error": "Allocation not found"}

            row = rows[0]
            agent_type = str(row.get("agent_type") or "").lower()
            if agent_type == AgentType.YUKI.value and not settings.YUKI_LIVE_TRADING_ENABLED:
                return {
                    "success": False,
                    "error": "Yuki live trading is disabled. Set YUKI_LIVE_TRADING_ENABLED=true before resuming."
                }
            if agent_type == AgentType.RYU.value and not settings.RYU_LIVE_TRADING_ENABLED:
                return {
                    "success": False,
                    "error": "Ryu live trading is disabled. Set RYU_LIVE_TRADING_ENABLED=true before resuming.",
                }

            current_status = str(row.get("status") or "")
            if current_status not in {AllocationStatus.ACTIVE.value, AllocationStatus.PAUSED.value}:
                return {"success": False, "error": "Only a paused agent allocation can be resumed"}

            allocated_amount = float(row.get("allocated_amount") or 0.0)
            minimum_allocation = (
                settings.RYU_MIN_ALLOCATION_USDC
                if agent_type == AgentType.RYU.value
                else settings.YUKI_MIN_ALLOCATION_USDC
            )
            if allocated_amount < minimum_allocation:
                return {
                    "success": False,
                    "error": f"This agent needs at least ${minimum_allocation:.2f} USDC to resume",
                }
            if agent_type == AgentType.RYU.value:
                wallets = (row.get("performance_metrics") or {}).get("wallets") or {}
                ryu_readiness = await self.ryu_spot_trading_service.get_trading_readiness(
                    user_id=user_id,
                    ethereum_wallet_address=row.get("platform_wallet_address") or "",
                    solana_wallet_address=wallets.get("solana_address") or "",
                )
                if not ryu_readiness.get("ready_for_trading"):
                    return {
                        "success": False,
                        "error": (
                            "; ".join(ryu_readiness.get("blocking_reasons") or [])
                            or "Ryu trading is not ready"
                        ),
                    }

            if current_status != AllocationStatus.ACTIVE.value:
                update_result = (
                    self.supabase.table("agent_allocations")
                    .update({
                        "status": AllocationStatus.ACTIVE.value,
                        "updated_at": datetime.now().isoformat(),
                    })
                    .eq("allocation_id", allocation_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                if not update_result.data:
                    return {"success": False, "error": "Allocation not found"}
                row = {**row, **update_result.data[0], "status": AllocationStatus.ACTIVE.value}

            # Keep this process coherent for status reads. The execution worker is
            # separate from this web process and rehydrates this durable ACTIVE status before
            # every signal cycle, so resume must never depend on this web process
            # already having the allocation in its in-memory cache.
            allocation = self._allocation_from_row(row)
            if not allocation:
                return {"success": False, "error": "Could not restore the agent allocation"}
            self.active_allocations[allocation_id] = allocation

            if agent_type == AgentType.YUKI.value:
                existing_agent = self.agent_instances.get(user_id, {}).get(AgentType.YUKI.value)
                if existing_agent:
                    existing_agent["allocation"] = allocation
                    await self._start_yuki_position_monitor(user_id, allocation_id)

            logger.info(f"Resumed agent trading for persisted allocation {allocation_id}")
            return {"success": True, "status": AllocationStatus.ACTIVE.value}

        except Exception as e:
            logger.error(f"Error resuming agent trading: {e}")
            return {"success": False, "error": str(e)}


# Singleton instance
_agent_allocation_service = None

def get_agent_allocation_service() -> AgentAllocationService:
    """Get singleton instance of agent allocation service."""
    global _agent_allocation_service
    if _agent_allocation_service is None:
        _agent_allocation_service = AgentAllocationService()
    return _agent_allocation_service
