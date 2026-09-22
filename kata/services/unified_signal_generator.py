#!/usr/bin/env python3
"""
Unified Signal Generation System for Flow Platform

Single file containing:
- Market opportunity discovery & scoring
- Technical analysis calculations
- AI-driven trade decision making
- Rate-limited API management
- Database integration

This replaces all complex dependencies with one reliable system.
"""

import asyncio
import hashlib
import logging
import numpy as np
import pandas as pd
import uuid
import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
import os
import sys

# Add app path for database and Claude access
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Minimal imports - only what we absolutely need
import ccxt.async_support as ccxt
import anthropic
import openai
import httpx
import websockets
import asyncio
import ssl
from kata.config.database import get_service_client
from kata.config.policy_versions import (
    compose_prompt_version,
    policy_bundle_version,
    policy_stamp,
)
from kata.config.settings import settings
from kata.services.platform_signal_service import (
    ENTRY_REANALYSIS_INVALIDATION_REASONS,
    ENTRY_TERMINAL_STATES,
    THESIS_LIVE_STATUSES,
    PlatformSignal,
    SignalPool,
    entry_order_deadline,
    get_platform_signal_service,
    signal_thesis_is_live,
    terminalize_entry_conditions,
)
from kata.services.agent_learning_service import get_agent_learning_service, LearningAdjustment
from kata.services.multi_level_learning_service import get_multi_level_learning_service
from kata.services.prompt_optimization_service import get_prompt_optimization_service
from kata.services.market_structure_context_service import get_market_structure_context_service
from kata.services.execution_fill_calibration import ExecutionFillCalibrator
from kata.services.signal_edge_policy import reliable_negative_edge_veto
from kata.services.signal_policy_learning import (
    POLICY_FEATURE_VERSION,
    POLICY_TRAINING_JSON_KEY,
    SignalPolicyBundle,
    build_policy_feature_vector,
    default_policy_bundle_path,
    load_policy_bundle,
    load_policy_bundle_from_db,
    policy_training_snapshot_from_live,
    train_policy_from_supabase,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pending entries retain the accepted thesis through ordinary HOLD, confidence,
# policy, disagreement, and provider outcomes. Cancellation requires either the
# accepted opposite-direction path below or an explicit AI-confirmed structural
# invalidation with a reason. This prevents a neutral re-test from erasing a
# valid resting order.
ACTIVE_THESIS_ACTIONS = frozenset({"KEEP", "INVALIDATE", "REVERSE"})

def safe_format(value, format_str: str = "", default=0):
    """Safely format a value that might be None, converting to appropriate type."""
    if value is None:
        value = default
    try:
        if format_str:
            if isinstance(value, str):
                # Try to convert string to float first
                value = float(value) if value not in ['', 'nan', 'None'] else default
            return format(value, format_str)
        return str(value)
    except (ValueError, TypeError):
        return str(default)

def clean_price_string(price_str) -> float:
    """
    Clean price strings from AI responses that may contain currency symbols.

    Examples:
    - "$0.03120" -> 0.03120
    - "0.03120" -> 0.03120
    - "$1,234.56" -> 1234.56
    - "3.1: 1 (T1) / 5.5: 1 (T2)" -> 3.1
    """
    if not price_str:
        return 0.0

    if isinstance(price_str, (int, float)):
        return float(price_str)

    # Convert to string and clean
    cleaned = str(price_str).strip()

    # Remove dollar signs, commas, and other common currency symbols
    cleaned = cleaned.replace('$', '').replace(',', '').replace('€', '').replace('£', '').replace('%', '')

    # Handle ratio strings like "3.1: 1 (T1) / 5.5: 1 (T2)" - extract first number
    if ':' in cleaned or '(' in cleaned or '/' in cleaned:
        # Use regex to extract first decimal number
        match = re.search(r'([0-9]+\.?[0-9]*)', cleaned)
        if match:
            cleaned = match.group(1)
        else:
            logger.warning(f"Could not extract number from complex string: {price_str}")
            return 0.0

    # Handle percentage signs and other text
    cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')

    try:
        return float(cleaned) if cleaned else 0.0
    except (ValueError, TypeError):
        logger.warning(f"Could not parse price string: {price_str} -> cleaned: {cleaned}")
        return 0.0

@dataclass
class TechnicalAnalysis:
    """Complete technical analysis for a token."""
    # Price data
    current_price: float
    price_change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float

    # Technical indicators
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float  # 0-1 where price sits in bands

    # Volume analysis
    volume_sma_10: float

    def __post_init__(self):
        """Ensure all numeric fields are properly converted and not None."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if value is None or (isinstance(value, str) and value in ['', 'nan', 'None']):
                    # Set reasonable defaults based on field type
                    if 'price' in field_name.lower() or 'level' in field_name.lower():
                        setattr(self, field_name, 0.0)
                    elif 'rsi' in field_name.lower():
                        setattr(self, field_name, 50.0)  # Neutral RSI
                    elif 'position' in field_name.lower():
                        setattr(self, field_name, 0.5)  # Middle position
                    elif 'score' in field_name.lower():
                        setattr(self, field_name, 0.5)  # Neutral score
                    else:
                        setattr(self, field_name, 0.0)
                elif isinstance(value, str):
                    try:
                        setattr(self, field_name, float(value))
                    except ValueError:
                        setattr(self, field_name, 0.0)
    volume_ratio: float  # Current vs average

    # Volatility metrics
    atr_14: float
    volatility_24h: float

    # Support/Resistance
    support_level: float
    resistance_level: float

    # Trend analysis
    ema_20: float
    ema_50: float
    ema_200: float
    trend_direction: str  # 'bullish', 'bearish', 'sideways'

    # Momentum
    momentum_score: float  # 0-1
    strength_score: float  # 0-1

    # Advanced Oscillators
    stoch_k: float  # Stochastic %K
    stoch_d: float  # Stochastic %D
    williams_r: float  # Williams %R
    cci_14: float  # Commodity Channel Index
    roc_10: float  # Rate of Change

    # Volume Analysis
    obv: float  # On Balance Volume
    vwap: float  # Volume Weighted Average Price
    volume_profile_poc: float  # Point of Control
    money_flow_index: float  # Money Flow Index

    # Market Microstructure
    spread_estimate: float  # Bid-ask spread estimate
    tick_rule_momentum: float  # Tick rule momentum
    price_efficiency: float  # Price efficiency measure

    # Multi-timeframe
    rsi_1h: float  # 1-hour RSI
    macd_1h_line: float  # 1-hour MACD
    macd_1h_signal: float  # 1-hour MACD signal
    trend_alignment: float  # Multi-timeframe trend alignment score

    # Fibonacci and Advanced S/R
    fib_23_6: float  # Fibonacci 23.6% level
    fib_38_2: float  # Fibonacci 38.2% level
    fib_50_0: float  # Fibonacci 50% level
    fib_61_8: float  # Fibonacci 61.8% level
    dynamic_support: float  # Dynamic support level
    dynamic_resistance: float  # Dynamic resistance level
    structure_candles: Optional[List[Dict[str, Any]]] = None
    
    # Advanced AI signal metrics
    taker_buy_sell_ratio: Optional[float] = None  # >1.0 means more taker buys (momentum)
    liquidation_volume: Optional[float] = None    # Volume of liquidations (exhaustion)
    basis_premium: Optional[float] = None         # Spot vs Futures spread (sentiment)

@dataclass
class OpportunityScore:
    """Market opportunity scoring."""
    symbol: str
    overall_score: float
    volume_score: float
    volatility_score: float
    momentum_score: float
    trend_score: float
    institutional_score: float
    breakdown: Dict[str, float]

    def __post_init__(self):
        """Ensure all numeric fields are properly converted and not None."""
        for field_name, field_type in self.__annotations__.items():
            if field_type == float:
                value = getattr(self, field_name)
                if value is None or (isinstance(value, str) and value in ['', 'nan', 'None']):
                    setattr(self, field_name, 0.5)  # Neutral score for all opportunity metrics
                elif isinstance(value, str):
                    try:
                        setattr(self, field_name, float(value))
                    except ValueError:
                        setattr(self, field_name, 0.5)

        # Ensure breakdown dict has proper values
        if self.breakdown:
            for key, value in self.breakdown.items():
                if value is None or (isinstance(value, str) and value in ['', 'nan', 'None']):
                    self.breakdown[key] = 0.0
                elif isinstance(value, str):
                    try:
                        self.breakdown[key] = float(value)
                    except ValueError:
                        self.breakdown[key] = 0.0

@dataclass
class AIDecision:
    """AI trading decision."""
    recommendation: str  # LONG, SHORT, HOLD
    confidence: float  # 0.0-1.0
    reasoning: str
    key_factors: List[str]

    # Price levels
    entry_price: float
    target_1: float
    target_1_probability: float
    target_2: float
    target_2_probability: float
    stop_loss: float
    risk_reward_ratio: float

    # Position management
    position_size: float  # Percentage
    leverage: int
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    time_horizon: str

    # Risk assessment
    risk_factors: List[str]
    risk_assessment: str
    entry_strategy: Optional[str] = None
    timeframe: Optional[str] = None
    # Levels exactly as the LLM proposed them, before any floor/clamp/adaptation —
    # persisted to market_conditions for level-quality analysis.
    ai_proposed_levels: Optional[Dict[str, float]] = None
    raw_ai_confidence: Optional[float] = None
    raw_ai_position_size: Optional[float] = None
    raw_ai_leverage: Optional[int] = None
    raw_ai_time_horizon: Optional[str] = None
    prompt_version: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_estimated_cost_usd: Optional[float] = None
    policy_win_probability: Optional[float] = None
    policy_eval_direction: Optional[str] = None
    policy_feature_version: Optional[str] = None
    ensemble_rule_confidence: Optional[float] = None
    ensemble_rule_direction: Optional[str] = None
    ensemble_regime_type: Optional[str] = None
    challenger_applied: bool = False
    challenger_provider: Optional[str] = None
    challenger_model: Optional[str] = None
    challenger_verdict: Optional[str] = None
    challenger_reason: Optional[str] = None
    challenger_prompt_tokens: Optional[int] = None
    challenger_completion_tokens: Optional[int] = None
    challenger_estimated_cost_usd: Optional[float] = None
    market_structure_context: Optional[Dict[str, Any]] = None
    active_thesis_action: Optional[str] = None
    structural_invalidation: bool = False
    structural_invalidation_reason: Optional[str] = None
    edge_estimate: Optional[Dict[str, Any]] = None
    calibrated_direction_probability: Optional[float] = None
    fill_probability: Optional[float] = None
    target_before_stop_probability: Optional[float] = None
    net_expected_value_pct: Optional[float] = None
    edge_score: Optional[float] = None

class UnifiedSignalGenerator:
    """
    Complete signal generation system in a single class.

    Handles everything from opportunity discovery to AI-driven trade decisions.
    """

    def __init__(self):
        self.binance = None
        self.llm_client = None
        self.supabase = None
        self.platform_service = None
        self.learning_service = None
        self.multi_level_learning = None
        self.prompt_optimization_service = None
        self.challenger_client = None
        self._challenger_calls_this_run = 0
        self._challenger_month_spend_usd = 0.0
        self._challenger_spend_last_sync_at: Optional[datetime] = None

        # Rate limiting
        self.api_calls_this_minute = []
        self.max_calls_per_minute = 100  # Reasonable limit for Binance API (actual limit is ~1200/min)

        # OHLCV cache: {"SYMBOL:INTERVAL": [candle_data]} — cleared each run
        self._ohlcv_cache: Dict[str, List[List]] = {}

        # Services will be initialized when needed
        self._services_initialized = False
        self._opportunity_score_cache: Dict[str, OpportunityScore] = {}
        self._eligible_usdt_symbols: Set[str] = set()
        self._eligible_symbols_cached_at: Optional[datetime] = None
        self._last_rejection_context: Dict[str, Dict[str, Any]] = {}
        self._reject_quality_conf_adjustment: float = 0.0
        self._reject_quality_tuning_state: Dict[str, Any] = {}
        self._signal_quality_conf_adjustment: float = 0.0
        self._signal_quality_state: Dict[str, Any] = {}
        self._adaptive_confidence_target: float = float(
            getattr(settings, "SIGNAL_PUBLISH_CONFIDENCE_FLOOR", 0.65) or 0.65
        )
        self._last_reject_tuning_persist_at: Optional[datetime] = None
        self._shutdown_requested: bool = False
        self._signal_policy_bundle: Optional[SignalPolicyBundle] = None
        self._signal_policy_source_path: Optional[Path] = None
        self._signal_policy_mtime: float = 0.0
        self._policy_train_last_attempt_utc: Optional[datetime] = None
        self._fill_calibrator = ExecutionFillCalibrator(
            minimum_samples=int(
                getattr(settings, "YUKI_FILL_CALIBRATION_MIN_SAMPLES", 30) or 30
            )
        )
        self._fill_model_last_attempt_utc: Optional[datetime] = None

        # Execution cost model for Yuki's Hyperliquid venue. Forward tests use
        # the same 4.5 bps taker assumption; realized learning later replaces
        # this estimate with exact fill fees, funding and slippage.
        self._taker_fee_pct: float = 0.045
        self._slippage_pct: float = 0.05
        # Round-trip cost = entry friction + exit friction.
        self._round_trip_cost_pct: float = (self._taker_fee_pct + self._slippage_pct) * 2.0

        # Real-data caches with TTL — avoid hammering external APIs.
        self._funding_rate_cache: Dict[str, Tuple[float, datetime]] = {}
        self._funding_rate_ttl_seconds: int = 300
        self._btc_dominance_cache: Tuple[Optional[float], Optional[datetime]] = (None, None)
        self._btc_dominance_ttl_seconds: int = 600
        self._orderbook_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._orderbook_ttl_seconds: int = 30
        # Liquidation cluster cache: symbol → ({price_bucket: notional}, cached_at)
        self._liq_cluster_cache: Dict[str, Tuple[Dict[float, float], datetime]] = {}
        self._liq_cluster_ttl_seconds: int = 300
        # Positioning signal cache: symbol → (signal_dict, cached_at)
        self._positioning_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._positioning_ttl_seconds: int = 300
        self._open_interest_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._open_interest_ttl_seconds: int = 300
        self._hyperliquid_snapshot_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._hyperliquid_snapshot_ttl_seconds: int = 30
        self._token_event_cache: Tuple[List[Dict[str, Any]], Optional[datetime]] = ([], None)
        self._token_event_ttl_seconds: int = 900

        # Drawdown circuit breaker.
        self._equity_high_water_mark: Optional[float] = None
        self._drawdown_pause_threshold: float = 0.20
        self._drawdown_halt_threshold: float = 0.30
        self._drawdown_paused: bool = False

        # Sliced attribution: rolling stats by (regime, direction, leverage_bucket).
        self._attribution_stats: Dict[str, Dict[str, float]] = {}

        # Concept drift watchdog: rolling 50-signal win rates.
        self._recent_outcomes: List[Tuple[datetime, bool, float]] = []  # (resolved_at, win, pnl_pct)
        self._concept_drift_alerted: bool = False
        self._peak_rolling_win_rate: float = 0.5

        # Directional self-learning: recent resolved win-rate by direction, cached.
        self._direction_perf_cache: Dict[str, Dict[str, float]] = {}
        self._direction_perf_cache_at: Optional[datetime] = None
        self._direction_perf_ttl_seconds: float = 1800.0  # refresh at most every 30 min

        # Validity-window calibration: maps canonical time_horizon label → median actual hours.
        # Refreshed from Supabase once per day; falls back to hardcoded table when empty.
        self._calibrated_validity_windows: Dict[str, int] = {}
        self._validity_calibration_last_run: Optional[datetime] = None

        # Ensemble accuracy weights: {'rule': w_rule, 'llm': w_llm} summing to 1.0.
        # Refreshed from resolved-signal history every 24 h; default equal-weight until data accrues.
        self._ensemble_weights: Dict[str, float] = {'rule': 0.5, 'llm': 0.5}
        self._ensemble_weights_last_run: Optional[datetime] = None

    def request_shutdown(self) -> None:
        """Signal long-running generator flows to stop as soon as possible."""
        self._shutdown_requested = True

    def _normalize_symbol_to_market_id(self, symbol: str) -> str:
        """Normalize symbol formats (e.g., BTC/USDT:USDT) to market id (BTCUSDT)."""
        if not symbol:
            return ""

        normalized = symbol.upper().strip()
        if '/USDT:USDT' in normalized:
            return f"{normalized.split('/')[0]}USDT"
        if '/USDT' in normalized:
            return f"{normalized.split('/')[0]}USDT"

        # Fallback for already-normalized ids and minor suffix variants.
        return normalized.replace(':USDT', '')

    def _signal_symbol_key(self, symbol: str) -> str:
        """Match stored base symbols and exchange-style USDT market ids."""
        normalized = self._normalize_symbol_to_market_id(symbol)
        return normalized[:-4] if normalized.endswith("USDT") else normalized

    @staticmethod
    def _pending_rejection_confirms_invalidation(context: Optional[Dict[str, Any]]) -> bool:
        """Make a thesis terminal only after an explicit structural break."""
        context = context or {}
        action = str(context.get("active_thesis_action") or "").upper().strip()
        structural = context.get("structural_invalidation") is True
        reason = str(context.get("structural_invalidation_reason") or "").strip()
        return action == "INVALIDATE" and structural and bool(reason)

    @staticmethod
    def _active_thesis_prompt_block(thesis: Optional[Dict[str, Any]]) -> str:
        """Give the model the complete accepted thesis before it re-analyses."""
        if not thesis:
            return "- No live thesis exists for this token."

        conditions = thesis.get("market_conditions") or {}
        if not isinstance(conditions, dict):
            conditions = {}
        rationale = str(
            thesis.get("ai_reasoning")
            or thesis.get("analysis_notes")
            or "No stored rationale"
        ).strip().replace("\x00", " ")[:1800]
        latest_result = str(
            conditions.get("last_entry_revalidation_result") or "not yet re-tested"
        )
        latest_reason = str(conditions.get("last_entry_revalidation_reason") or "")[:500]
        latest_at = str(conditions.get("last_entry_revalidated_at") or "n/a")
        return (
            f"- Signal id: {thesis.get('signal_id') or 'n/a'}\n"
            f"- Accepted direction: {str(thesis.get('direction') or '').upper() or 'n/a'}\n"
            f"- Generated: {thesis.get('analysis_timestamp') or thesis.get('created_at') or 'n/a'}\n"
            f"- Valid until: {thesis.get('expires_at') or 'n/a'}\n"
            f"- Confidence: {thesis.get('confidence') if thesis.get('confidence') is not None else 'n/a'}\n"
            f"- Entry / stop / target 1 / target 2: "
            f"{thesis.get('entry_price') or 'n/a'} / {thesis.get('stop_loss') or 'n/a'} / "
            f"{thesis.get('target_1') or 'n/a'} / {thesis.get('target_2') or 'n/a'}\n"
            f"- Horizon: {thesis.get('time_horizon') or 'n/a'} | "
            f"Leverage: {thesis.get('leverage') or 'n/a'} | "
            f"Collateral size: {thesis.get('position_size') or 'n/a'}%\n"
            f"- Original rationale: {rationale}\n"
            f"- Latest re-test: {latest_result} at {latest_at}"
            + (f" — {latest_reason}" if latest_reason else "")
        )

    async def _load_pending_theses_for_revalidation(self) -> Dict[str, Dict[str, Any]]:
        """
        Load live platform theses whose entry has not filled/activated yet.

        Platform activation covers manual/paper consumers while pending Yuki
        trade rows cover real resting orders. The newest row owns each symbol,
        consistent with the lifecycle guard in PlatformSignalService.
        """
        if not self.supabase:
            return {}

        try:
            rows = (
                self.supabase.from_("platform_signals")
                .select(
                    "id,signal_id,token_symbol,direction,status,expires_at,analysis_timestamp,"
                    "analysis_notes,ai_reasoning,market_conditions,confidence,entry_price,"
                    "stop_loss,target_1,target_2,time_horizon,leverage,position_size,created_at"
                )
                .in_("status", list(THESIS_LIVE_STATUSES))
                .order("analysis_timestamp", desc=True)
                .execute()
            ).data or []

            pending_trade_ids: Set[str] = set()
            try:
                pending_trades = (
                    self.supabase.from_("agent_trades")
                    .select("trade_metadata")
                    .eq("agent_type", "yuki")
                    .eq("status", "pending")
                    .execute()
                ).data or []
                pending_trade_ids = {
                    str((trade.get("trade_metadata") or {}).get("signal_id") or "")
                    for trade in pending_trades
                }
                pending_trade_ids.discard("")
            except Exception as trade_error:
                logger.warning("Could not load Yuki pending entries for signal revalidation: %s", trade_error)

            pending: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                if not signal_thesis_is_live(row):
                    continue
                market_conditions = row.get("market_conditions") or {}
                entry_state = str(
                    market_conditions.get("entry_order_state") or ""
                ).strip().lower()
                if entry_state in ENTRY_TERMINAL_STATES or entry_state == "terminal":
                    # The strategy thesis remains visible for learning, but its
                    # execution path is closed and must not consume another
                    # scheduled analysis slot.
                    continue
                platform_entry_pending = bool(
                    market_conditions.get("entry_activation_required", False)
                    and not market_conditions.get("entry_activated", False)
                )
                awaiting_confirmation = (
                    str(market_conditions.get("entry_order_state") or "").lower()
                    == "awaiting_confirmation"
                )
                yuki_entry_pending = str(row.get("signal_id") or "") in pending_trade_ids
                if not platform_entry_pending and not awaiting_confirmation and not yuki_entry_pending:
                    continue
                key = self._signal_symbol_key(str(row.get("token_symbol") or ""))
                if key and key not in pending:
                    pending[key] = row
            return pending
        except Exception as exc:
            logger.warning("Pending platform-signal revalidation lookup failed: %s", exc)
            return {}

    def _prepend_pending_revalidation_symbols(
        self,
        opportunities: List[str],
        pending_theses: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Put every pending thesis ahead of new discovery candidates."""
        by_market_id = {
            self._signal_symbol_key(symbol): symbol
            for symbol in opportunities
        }
        revalidation_symbols: List[str] = []
        for market_id, thesis in pending_theses.items():
            symbol = by_market_id.get(market_id)
            if not symbol:
                exchange_market_id = f"{market_id}USDT"
                if exchange_market_id not in self._eligible_usdt_symbols:
                    # Do not send HIP-3/unsupported symbols through Binance's
                    # retrying OHLCV path. Their existing stop and expiry remain
                    # authoritative until a compatible analysis adapter exists.
                    logger.info(
                        "Skipping scheduled re-test for pending %s: no eligible generation market data",
                        thesis.get("token_symbol") or market_id,
                    )
                    continue
                token = str(thesis.get("token_symbol") or market_id).upper().strip()
                base = token[:-4] if token.endswith("USDT") else token
                symbol = f"{base}/USDT:USDT"
            revalidation_symbols.append(symbol)

        combined = revalidation_symbols + list(opportunities)
        seen: Set[str] = set()
        result: List[str] = []
        for symbol in combined:
            market_id = self._signal_symbol_key(symbol)
            if not market_id or market_id in seen:
                continue
            seen.add(market_id)
            result.append(symbol)
        return result

    @classmethod
    def _normalize_reanalysis_hash_value(
        cls,
        value: Any,
        *,
        depth: int = 0,
    ) -> Any:
        """Create a stable, compact payload for pending re-analysis dedupe."""
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
                compact[str(key)] = cls._normalize_reanalysis_hash_value(value.get(key), depth=depth + 1)
            if len(value) > 20:
                compact["_truncated_keys"] = len(value) - 20
            return compact
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            compact_items = [
                cls._normalize_reanalysis_hash_value(item, depth=depth + 1)
                for item in items[:10]
            ]
            if len(items) > 10:
                compact_items.append({"_truncated_items": len(items) - 10})
            return compact_items
        return cls._normalize_reanalysis_hash_value(str(value), depth=depth + 1)

    def _reanalysis_request_state_hash(self, request: Dict[str, Any]) -> str:
        """Hash the materially relevant pending re-analysis request state."""
        conditions = request.get("market_conditions") or {}
        payload = {
            "signal_id": request.get("signal_id"),
            "token_symbol": request.get("token_symbol"),
            "direction": request.get("direction"),
            "status": request.get("status"),
            "analysis_timestamp": request.get("analysis_timestamp"),
            "confidence": request.get("confidence"),
            "entry_price": request.get("entry_price"),
            "stop_loss": request.get("stop_loss"),
            "target_1": request.get("target_1"),
            "target_2": request.get("target_2"),
            "time_horizon": request.get("time_horizon"),
            "reanalysis_reason": conditions.get("reanalysis_reason"),
            "reanalysis_source": conditions.get("reanalysis_source"),
            "entry_order_state": conditions.get("entry_order_state"),
            "last_entry_revalidation_result": conditions.get("last_entry_revalidation_result"),
            "last_entry_revalidation_direction": conditions.get("last_entry_revalidation_direction"),
            "last_entry_revalidation_confidence": conditions.get("last_entry_revalidation_confidence"),
            "current_price": (
                conditions.get("current_price")
                or conditions.get("market_price_at_generation")
                or request.get("entry_price")
            ),
            "volume_24h": conditions.get("volume_24h"),
            "trend_direction": conditions.get("trend_direction"),
            "market_regime": conditions.get("market_regime"),
        }
        serialized = json.dumps(
            self._normalize_reanalysis_hash_value(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _should_coalesce_reanalysis_request(
        self,
        request: Dict[str, Any],
        state_hash: str,
        *,
        now: datetime,
    ) -> bool:
        """Skip repeated re-analysis while the queued request state is unchanged."""
        conditions = request.get("market_conditions") or {}
        last_state_hash = str(conditions.get("reanalysis_last_attempt_state_hash") or "")
        if not last_state_hash or last_state_hash != state_hash:
            return False
        last_attempted_at = self._parse_iso_datetime(
            conditions.get("reanalysis_last_attempted_at"),
            default=datetime.min,
        )
        if last_attempted_at == datetime.min:
            return False
        coalesce_minutes = max(
            1,
            int(getattr(settings, "PLATFORM_REANALYSIS_COALESCE_MINUTES", 30) or 30),
        )
        return (now - last_attempted_at).total_seconds() < coalesce_minutes * 60.0

    async def _record_pending_revalidation(
        self,
        thesis: Dict[str, Any],
        result: str,
        reason: str,
        decision: Optional[AIDecision] = None,
    ) -> None:
        """Persist the latest pending-entry re-test without changing its levels."""
        if not self.supabase or not thesis.get("signal_id"):
            return
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        market_conditions = dict(thesis.get("market_conditions") or {})
        market_conditions.update({
            "last_entry_revalidated_at": now_iso,
            "last_entry_revalidation_result": result,
            "last_entry_revalidation_reason": str(reason or "")[:500],
        })
        if decision is not None:
            market_conditions["last_entry_revalidation_direction"] = decision.recommendation
            market_conditions["last_entry_revalidation_confidence"] = float(decision.confidence or 0.0)
            market_conditions["last_entry_revalidation_action"] = (
                str(getattr(decision, "active_thesis_action", None) or "KEEP").upper()
            )
            market_conditions["last_entry_structural_invalidation"] = bool(
                getattr(decision, "structural_invalidation", False)
            )
            structural_reason = str(
                getattr(decision, "structural_invalidation_reason", None) or ""
            ).strip()
            if structural_reason:
                market_conditions["last_entry_structural_invalidation_reason"] = structural_reason[:500]
            if isinstance(decision.edge_estimate, dict):
                market_conditions["last_entry_revalidation_edge_estimate"] = decision.edge_estimate

        if result == "skipped_negative_ev":
            terminalize_entry_conditions(
                market_conditions,
                "skipped_negative_ev",
                reason or "fresh re-analysis found non-positive net expected value",
                now_iso,
            )

        request_pending = (
            bool(market_conditions.get("reanalysis_requested"))
            and str(market_conditions.get("reanalysis_request_status") or "pending").lower()
            == "pending"
        )
        if (
            result in {"confirmed", "retained"}
            and not request_pending
            and str(market_conditions.get("entry_order_state") or "").lower()
            == "awaiting_confirmation"
        ):
            try:
                current_revision = max(0, int(market_conditions.get("entry_revision") or 0))
            except (TypeError, ValueError):
                current_revision = 0
            max_revisions = max(
                0,
                int(getattr(settings, "YUKI_MAX_ENTRY_REVISIONS", 2) or 2),
            )
            deadline = entry_order_deadline(
                signal_started_at=thesis.get("analysis_timestamp") or thesis.get("created_at"),
                signal_expires_at=thesis.get("expires_at"),
                max_ttl_hours=float(settings.YUKI_ENTRY_ORDER_TTL_HOURS or 12.0),
                validity_fraction=float(settings.YUKI_ENTRY_TTL_VALIDITY_FRACTION or 0.50),
                now=now,
            )
            if deadline > now and current_revision < max_revisions:
                revision = current_revision + 1
                market_conditions.update({
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
                result = "missed_entry"
                market_conditions["last_entry_revalidation_result"] = result
                market_conditions["last_entry_revalidation_reason"] = terminal_reason
                terminalize_entry_conditions(
                    market_conditions,
                    "missed_entry",
                    terminal_reason,
                    now_iso,
                )

        payload: Dict[str, Any] = {
            "market_conditions": market_conditions,
            "updated_at": now_iso,
        }
        if result == "invalidated":
            market_conditions.update({
                "entry_order_state": "invalidated",
                "invalidation_kind": "structural",
                "invalidated_at": now_iso,
                "invalidation_reason": str(reason or "structural invalidation")[:500],
            })
            original_notes = str(thesis.get("analysis_notes") or "").strip()
            lifecycle_note = f"Pending entry invalidated on scheduled re-test: {reason}"
            payload.update({
                "status": "invalidated",
                "analysis_notes": f"{original_notes}\n{lifecycle_note}".strip(),
            })
        elif result in {"missed_entry", "skipped_negative_ev"}:
            original_notes = str(thesis.get("analysis_notes") or "").strip()
            lifecycle_note = (
                "Pending entry closed as missed"
                if result == "missed_entry"
                else "Pending entry skipped after fresh negative-EV analysis"
            )
            payload["analysis_notes"] = (
                f"{original_notes}\n{lifecycle_note}: {reason}".strip()
            )
        self.supabase.from_("platform_signals").update(payload).eq(
            "signal_id", thesis["signal_id"]
        ).execute()
        # ``run_event_reanalysis_requests`` classifies and closes this queue item
        # later in the same cycle from the original request object. Keep that
        # object synchronized with the persisted re-test; otherwise the fresh
        # timestamp/result remain invisible in memory and a retained thesis is
        # incorrectly marked ``fresh_stack_rejected`` until another full cycle.
        thesis["market_conditions"] = market_conditions
        if result == "invalidated":
            try:
                (
                    self.supabase.from_("platform_signal_performance_tracking")
                    .update({
                        "outcome": "invalidated",
                        "exit_reason": "INVALIDATED",
                        "exit_timestamp": now_iso,
                        "last_updated": now_iso,
                    })
                    .eq("signal_id", thesis["signal_id"])
                    .eq("outcome", "active")
                    .execute()
                )
            except Exception as perf_exc:
                logger.warning(
                    "Could not synchronize pending-thesis invalidation for %s: %s",
                    thesis["signal_id"],
                    perf_exc,
                )
        if result == "invalidated":
            thesis["status"] = "invalidated"
        logger.info(
            "Pending entry revalidation %s: %s %s (%s)",
            result,
            thesis.get("token_symbol"),
            thesis.get("direction"),
            reason,
        )

    async def _get_eligible_usdt_symbols(
        self,
        min_listing_days: int = 35,
        cache_seconds: int = 3600,
    ) -> Set[str]:
        """
        Return active USDT futures symbols old enough for reliable multi-timeframe analysis.

        This avoids repeatedly sending newly listed markets to TA/LLM where 1d/4h history
        is insufficient and only produces expensive HOLD/reject loops.
        """
        now = datetime.utcnow()
        if (
            self._eligible_usdt_symbols
            and self._eligible_symbols_cached_at
            and (now - self._eligible_symbols_cached_at).total_seconds() < cache_seconds
        ):
            return self._eligible_usdt_symbols

        try:
            if not self._services_initialized:
                await self._initialize_services()

            markets = await self.rate_limited_api_call(self.binance.fetch_markets)
            eligible: Set[str] = set()

            for market in markets or []:
                if not market or not market.get('active', True):
                    continue

                quote = str(market.get('quote') or '').upper()
                settle = str(market.get('settle') or '').upper()
                market_id = str(market.get('id') or '').upper()

                if quote != 'USDT' or (settle and settle != 'USDT'):
                    continue
                if not market_id.endswith('USDT'):
                    continue

                info = market.get('info') or {}
                onboard_raw = info.get('onboardDate')
                if onboard_raw:
                    try:
                        onboard_ms = int(float(onboard_raw))
                        listed_at = datetime.utcfromtimestamp(onboard_ms / 1000.0)
                        listed_days = (now - listed_at).days
                        if listed_days < min_listing_days:
                            continue
                    except (TypeError, ValueError, OSError):
                        # If date parsing fails, keep symbol rather than over-filtering.
                        pass

                eligible.add(market_id)

            self._eligible_usdt_symbols = eligible
            self._eligible_symbols_cached_at = now
            logger.info(
                f"✅ Eligible USDT symbols for TA/LLM: {len(eligible)} (min listing age: {min_listing_days}d)"
            )
            return eligible
        except Exception as e:
            logger.warning(f"⚠️ Failed to build eligible symbol set, using unfiltered universe: {e}")
            return set()


    async def _initialize_services(self):
        """Initialize all required services."""
        try:
            # Initialize Binance with fallback handling
            binance_api_key = os.getenv('BINANCE_API_KEY')
            binance_secret = os.getenv('BINANCE_API_SECRET')

            if not binance_api_key or not binance_secret:
                logger.warning("⚠️ Binance API credentials not found - using public API only")

            # Use Futures API - REQUIRED for futures trading signals.
            # CRITICAL: Binance Futures API (fapi.binance.com) is geo-blocked from Singapore.
            # If you hit Error 451, run this worker in a Binance-supported US/EU region.
            self.binance = ccxt.binance({
                'apiKey': binance_api_key,
                'secret': binance_secret,
                'sandbox': False,
                'enableRateLimit': True,
                'rateLimit': 1200,  # 1.2s between calls — Binance futures allows ~1200 req/min weight budget
                'timeout': 60000,   # 60 second timeout
                'urls': {
                    'api': {
                        'public': 'https://fapi.binance.com/fapi/v1',  # Direct futures API
                        'private': 'https://fapi.binance.com/fapi/v1'
                    }
                },
                'options': {
                    'defaultType': 'future',  # Use futures trading
                    'adjustForTimeDifference': True
                },
                'headers': {
                    'User-Agent': 'FlowTrading/1.0 (https://flow-trading.com)',
                    'Accept': 'application/json'
                }
            })

            # Initialize LLM client based on provider
            provider = settings.LLM_PROVIDER.lower()
            if provider == "deepseek":
                if not settings.DEEPSEEK_API_KEY:
                    raise ValueError("LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY")
                self.llm_client = openai.OpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com"
                )
                logger.info("✅ DeepSeek LLM client initialized")
            elif provider == "openai":
                if not settings.OPENAI_API_KEY:
                    raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
                self.llm_client = openai.OpenAI(
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info("✅ OpenAI LLM client initialized")
            else:  # Default to Claude
                if not settings.ANTHROPIC_API_KEY:
                    raise ValueError("LLM_PROVIDER=claude requires ANTHROPIC_API_KEY")
                self.llm_client = anthropic.Anthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    timeout=httpx.Timeout(180.0, read=180.0, write=60.0, connect=30.0)
                )
                logger.info("✅ Claude Sonnet client initialized")

            # Initialize optional challenger client (independent provider/model path).
            self.challenger_client = None
            if settings.CHALLENGER_ENABLED:
                challenger_provider = settings.CHALLENGER_PROVIDER.lower()
                if challenger_provider == provider:
                    self.challenger_client = self.llm_client
                    logger.info("✅ Challenger reusing primary LLM client")
                elif challenger_provider == "deepseek":
                    if settings.DEEPSEEK_API_KEY:
                        self.challenger_client = openai.OpenAI(
                            api_key=settings.DEEPSEEK_API_KEY,
                            base_url="https://api.deepseek.com"
                        )
                        logger.info("✅ Challenger DeepSeek client initialized")
                    else:
                        logger.warning("⚠️ Challenger provider=deepseek but DEEPSEEK_API_KEY is missing; challenger disabled")
                elif challenger_provider == "openai":
                    if settings.OPENAI_API_KEY:
                        self.challenger_client = openai.OpenAI(
                            api_key=settings.OPENAI_API_KEY
                        )
                        logger.info("✅ Challenger OpenAI client initialized")
                    else:
                        logger.warning("⚠️ Challenger provider=openai but OPENAI_API_KEY is missing; challenger disabled")
                elif challenger_provider == "claude":
                    if settings.ANTHROPIC_API_KEY:
                        self.challenger_client = anthropic.Anthropic(
                            api_key=settings.ANTHROPIC_API_KEY,
                            timeout=httpx.Timeout(
                                connect=20.0,
                                read=float(settings.CHALLENGER_TIMEOUT_SECONDS),
                                write=20.0,
                                pool=20.0,
                            )
                        )
                        logger.info("✅ Challenger Claude client initialized")
                    else:
                        logger.warning("⚠️ Challenger provider=claude but ANTHROPIC_API_KEY is missing; challenger disabled")
                else:
                    logger.warning(f"⚠️ Unsupported challenger provider: {challenger_provider}; challenger disabled")
                if not self.challenger_client:
                    logger.info("ℹ️ Challenger checks disabled for this run due to missing/invalid challenger configuration")

            # Initialize database services
            self.supabase = get_service_client()
            self.platform_service = get_platform_signal_service()

            # Initialize learning services (use 'yuki' as default platform agent)
            self.learning_service = get_agent_learning_service('yuki')
            self.multi_level_learning = get_multi_level_learning_service('yuki')
            self.prompt_optimization_service = get_prompt_optimization_service('yuki')

            await self._refresh_execution_fill_model(force=True)

            self._services_initialized = True
            logger.info("✅ All services initialized successfully")

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    async def rate_limited_api_call(self, api_func, *args, max_retries=3, **kwargs):
        """Execute API call with enhanced rate limiting and retry logic."""
        # Clean old calls (older than 1 minute)
        current_time = datetime.now()
        self.api_calls_this_minute = [
            call_time for call_time in self.api_calls_this_minute
            if (current_time - call_time).seconds < 60
        ]

        # Wait if approaching limit
        if len(self.api_calls_this_minute) >= self.max_calls_per_minute:
            sleep_time = 60 - (current_time - min(self.api_calls_this_minute)).seconds + 10  # Add 10s buffer
            logger.info(f"⏱️ Rate limiting: sleeping {sleep_time}s")
            await asyncio.sleep(sleep_time)

        # Execute call with retry logic
        for attempt in range(max_retries):
            try:
                # Create fresh coroutine for each attempt
                result = await api_func(*args, **kwargs)
                self.api_calls_this_minute.append(datetime.now())
                return result

            except Exception as e:
                error_msg = str(e).lower()

                # Permanent "not supported" errors — never retry, never log as ERROR
                if 'not supported' in error_msg or 'notsupported' in error_msg:
                    logger.debug(f"⚡ API call not supported (skipping retries): {e}")
                    raise

                # Handle rate limit errors with reasonable backoff
                if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                    wait_time = min((2 ** attempt) * 30, 120)  # 30s, 60s, 120s (max 2min)
                    logger.warning(f"🚫 Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle geo-restriction Error 451 (Singapore IP blocked from Binance Futures API)
                elif '451' in error_msg or 'restricted location' in error_msg or 'service unavailable' in error_msg:
                    logger.error("🚨 BINANCE GEO-RESTRICTION ERROR 451:")
                    logger.error("   Binance Futures API (fapi.binance.com) is blocked from Singapore IP addresses")
                    logger.error("   SOLUTION: Move this worker out of Singapore into a Binance-supported US/EU region")
                    logger.error("   Steps: update your hosting region for the worker/backend service, then redeploy")
                    logger.error("   This will assign an IP address that Binance Futures allows")
                    raise Exception("Binance Futures API geo-restricted from Singapore. Redeploy the worker in a US/EU region.")

                # Handle IP ban or 403 errors with moderate wait
                elif '403' in error_msg or 'forbidden' in error_msg or 'ip banned' in error_msg:
                    wait_time = min((2 ** attempt) * 60, 300)  # 60s, 120s, 300s (max 5min)
                    logger.warning(f"🚫 IP/Access restricted (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle other errors with shorter backoff
                elif attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 15  # 15s, 30s
                    logger.warning(f"⚠️ API error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"❌ API call failed after {max_retries} attempts: {e}")
                    raise

    async def get_btc_market_data(self) -> Dict[str, float]:
        """
        Fetch real BTC market data for accurate market regime detection.
        Returns 24h and 7d price changes, volume, volatility, and structural TA (EMA, RSI, BB).
        """
        try:
            # Fetch BTC/USDT ticker data from Binance
            btc_ticker = await self.rate_limited_api_call(
                self.binance.fetch_ticker, 'BTC/USDT'
            )

            # Fetch 30d historical data for structural regime analysis
            btc_30d_data = await self.rate_limited_api_call(
                self.binance.fetch_ohlcv, 'BTC/USDT', '1d', limit=30
            )

            # Calculate 7d price change
            if len(btc_30d_data) >= 7:
                price_7d_ago = btc_30d_data[-7][1]  # Open price 7 days ago
                current_price = btc_ticker['last']
                btc_7d_change = ((current_price - price_7d_ago) / price_7d_ago) * 100
            else:
                btc_7d_change = 0.0

            # Calculate Structural TA
            ema_20 = 0.0
            ema_50 = 0.0
            rsi_14 = 50.0
            bb_width = 0.0
            
            if len(btc_30d_data) >= 20:
                df = pd.DataFrame(btc_30d_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                prices = df['close']
                ema_20 = float(prices.ewm(span=20).mean().iloc[-1])
                ema_50 = float(prices.ewm(span=50).mean().iloc[-1]) if len(prices) >= 50 else float(prices.ewm(span=len(prices)).mean().iloc[-1])
                
                # RSI 14
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, 1e-10)
                rsi = 100 - (100 / (1 + rs))
                rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
                
                # BB Width
                middle = prices.rolling(window=20).mean()
                std = prices.rolling(window=20).std()
                upper = middle + (std * 2)
                lower = middle - (std * 2)
                if float(middle.iloc[-1]) > 0:
                    bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1])

            # Real BTC dominance from CoinGecko (cached 10min) — falls back to last good value.
            btc_dominance = await self._fetch_btc_dominance()

            return {
                'price_24h': btc_ticker['percentage'] or 0.0,
                'price_7d': btc_7d_change,
                'volume_24h': btc_ticker['quoteVolume'] or 0.0,
                'volatility': abs(btc_ticker['percentage'] or 0.0),
                'dominance': btc_dominance,
                'current_price': btc_ticker['last'] or 0.0,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'rsi_14': rsi_14,
                'bb_width': bb_width
            }

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch BTC market data: {e}")
            # Return conservative defaults
            return {
                'price_24h': 0.0,
                'price_7d': 0.0,
                'volume_24h': 0.0,
                'volatility': 5.0,  # Assume moderate volatility
                'dominance': 55.0,
                'current_price': 95000.0,  # Approximate current BTC price
                'ema_20': 0.0,
                'ema_50': 0.0,
                'rsi_14': 50.0,
                'bb_width': 0.0
            }

    async def _fetch_btc_dominance(self) -> float:
        """Fetch real BTC dominance from free APIs (CoinGecko → CoinPaprika fallback) with TTL cache."""
        cached_value, cached_at = self._btc_dominance_cache
        if cached_value is not None and cached_at is not None:
            if (datetime.utcnow() - cached_at).total_seconds() < self._btc_dominance_ttl_seconds:
                return float(cached_value)
        # Primary: CoinGecko global endpoint
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/global",
                    headers={"Accept": "application/json", "User-Agent": "floww-signals/1.0"},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    pct = payload.get("data", {}).get("market_cap_percentage", {}).get("btc")
                    if pct is not None:
                        value = float(pct)
                        self._btc_dominance_cache = (value, datetime.utcnow())
                        logger.debug(f"📊 BTC dominance (CoinGecko): {value:.2f}%")
                        return value
                else:
                    logger.debug(f"CoinGecko dominance returned {resp.status_code}")
        except Exception as exc:
            logger.debug(f"CoinGecko BTC dominance fetch failed: {exc}")
        # Fallback: CoinPaprika global endpoint (also free, no key required)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.coinpaprika.com/v1/global",
                    headers={"Accept": "application/json", "User-Agent": "floww-signals/1.0"},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    pct = payload.get("bitcoin_dominance_percentage")
                    if pct is not None:
                        value = float(pct)
                        self._btc_dominance_cache = (value, datetime.utcnow())
                        logger.debug(f"📊 BTC dominance (CoinPaprika fallback): {value:.2f}%")
                        return value
                else:
                    logger.debug(f"CoinPaprika dominance returned {resp.status_code}")
        except Exception as exc:
            logger.debug(f"CoinPaprika BTC dominance fetch failed: {exc}")
        # Last resort: stale cache or neutral default
        if cached_value is not None:
            logger.debug(f"BTC dominance using stale cache: {cached_value:.2f}%")
            return float(cached_value)
        logger.warning("⚠️ BTC dominance unavailable from all sources — using 50% neutral default")
        return 50.0

    async def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch real funding rate (decimal, e.g. 0.0001 = 0.01%) with per-symbol TTL cache."""
        cached = self._funding_rate_cache.get(symbol)
        if cached is not None:
            value, fetched_at = cached
            if (datetime.utcnow() - fetched_at).total_seconds() < self._funding_rate_ttl_seconds:
                return float(value)
        try:
            if not self.binance:
                return None
            data = await self.rate_limited_api_call(self.binance.fetch_funding_rate, symbol)
            if isinstance(data, dict):
                rate = data.get("fundingRate")
                if rate is None and isinstance(data.get("info"), dict):
                    rate = data["info"].get("lastFundingRate")
                if rate is not None:
                    value = float(rate)
                    self._funding_rate_cache[symbol] = (value, datetime.utcnow())
                    return value
        except Exception as exc:
            logger.debug(f"Funding rate fetch failed for {symbol}: {exc}")
        return None

    async def _fetch_orderbook_depth(self, symbol: str, levels: int = 20) -> Optional[Dict[str, float]]:
        """Fetch L2 orderbook and compute imbalance + depth metrics."""
        cached = self._orderbook_cache.get(symbol)
        if cached is not None:
            value, fetched_at = cached
            if (datetime.utcnow() - fetched_at).total_seconds() < self._orderbook_ttl_seconds:
                return value
        try:
            if not self.binance:
                return None
            ob = await self.rate_limited_api_call(self.binance.fetch_order_book, symbol, levels)
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if not bids or not asks:
                return None
            bid_depth_usdt = sum(float(p) * float(q) for p, q in bids[:levels])
            ask_depth_usdt = sum(float(p) * float(q) for p, q in asks[:levels])
            total_depth = bid_depth_usdt + ask_depth_usdt
            imbalance = (bid_depth_usdt - ask_depth_usdt) / total_depth if total_depth > 0 else 0.0
            top_bid = float(bids[0][0])
            top_ask = float(asks[0][0])
            spread_pct = ((top_ask - top_bid) / top_bid) * 100.0 if top_bid > 0 else 0.0
            result = {
                "bid_depth_usdt": bid_depth_usdt,
                "ask_depth_usdt": ask_depth_usdt,
                "imbalance": imbalance,
                "top_bid": top_bid,
                "top_ask": top_ask,
                "spread_pct": spread_pct,
            }
            self._orderbook_cache[symbol] = (result, datetime.utcnow())
            return result
        except Exception as exc:
            logger.debug(f"Orderbook fetch failed for {symbol}: {exc}")
            return None

    @staticmethod
    def _passes_discovery_depth_gate(
        bid_depth_usdt: float,
        ask_depth_usdt: float,
        spread_pct: float,
        min_side_usdt: float,
        max_spread_pct: float,
    ) -> bool:
        """Return whether a candidate meets the configured L2 liquidity guardrails."""
        return (
            bid_depth_usdt >= min_side_usdt
            and ask_depth_usdt >= min_side_usdt
            and spread_pct <= max_spread_pct
        )

    async def _fetch_open_interest_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current and 24h historical open interest from Binance Futures."""
        cached = self._open_interest_cache.get(symbol)
        if cached is not None:
            value, fetched_at = cached
            if (datetime.utcnow() - fetched_at).total_seconds() < self._open_interest_ttl_seconds:
                return value

        if not self.binance:
            return None

        binance_sym = symbol.replace('/USDT:USDT', 'USDT').replace('/USDT', 'USDT').replace('/', '')
        result: Dict[str, Any] = {}

        try:
            current = await self.rate_limited_api_call(self.binance.fetch_open_interest, symbol)
            if isinstance(current, dict):
                oi_value = (
                    current.get("openInterestAmount")
                    or current.get("openInterestValue")
                    or current.get("openInterest")
                )
                if oi_value is None and isinstance(current.get("info"), dict):
                    oi_value = current["info"].get("openInterest")
                if oi_value is not None:
                    result["open_interest"] = float(oi_value)
        except Exception as exc:
            logger.debug("Current open interest fetch failed for %s: %s", symbol, exc)

        try:
            raw_hist = await self.rate_limited_api_call(
                self.binance.fapiDataGetOpenInterestHist,
                {"symbol": binance_sym, "period": "1h", "limit": 24},
            )
            if isinstance(raw_hist, list) and len(raw_hist) >= 2:
                first = raw_hist[0]
                latest = raw_hist[-1]
                first_oi = float(first.get("sumOpenInterest") or 0.0)
                latest_oi = float(latest.get("sumOpenInterest") or 0.0)
                if first_oi > 0 and latest_oi > 0:
                    result["open_interest_history_latest"] = latest_oi
                    result["open_interest_change_24h"] = ((latest_oi - first_oi) / first_oi) * 100.0
                    if "open_interest" not in result:
                        result["open_interest"] = latest_oi
        except Exception as exc:
            logger.debug("Open interest history fetch failed for %s: %s", symbol, exc)

        if not result:
            return None

        result["data_available"] = True
        self._open_interest_cache[symbol] = (result, datetime.utcnow())
        return result

    async def _fetch_hyperliquid_venue_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the actual execution-venue snapshot used for Yuki decisions.

        Includes mark/oracle basis, funding, OI, 24h notional volume and L2
        depth/imbalance. It degrades to the existing Binance context when a
        symbol is not listed or Hyperliquid is temporarily unavailable.
        """
        coin = self._signal_symbol_key(symbol)
        if not coin or ":" in coin:
            return None
        cached = self._hyperliquid_snapshot_cache.get(coin)
        if cached:
            value, fetched_at = cached
            if (datetime.utcnow() - fetched_at).total_seconds() < self._hyperliquid_snapshot_ttl_seconds:
                return value

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                meta_response, book_response = await asyncio.gather(
                    client.post(
                        "https://api.hyperliquid.xyz/info",
                        json={"type": "metaAndAssetCtxs"},
                    ),
                    client.post(
                        "https://api.hyperliquid.xyz/info",
                        json={"type": "l2Book", "coin": coin},
                    ),
                )
                if meta_response.status_code != 200:
                    return None
                meta_payload = meta_response.json()
                if not isinstance(meta_payload, list) or len(meta_payload) < 2:
                    return None
                meta, contexts = meta_payload[0], meta_payload[1]
                universe = (meta or {}).get("universe") or []
                asset_index = next(
                    (
                        index for index, asset in enumerate(universe)
                        if str((asset or {}).get("name") or "").upper() == coin.upper()
                    ),
                    None,
                )
                if asset_index is None or asset_index >= len(contexts):
                    return None
                context = contexts[asset_index] or {}

                mark = float(context.get("markPx") or context.get("midPx") or 0.0)
                oracle = float(context.get("oraclePx") or 0.0)
                open_interest = float(context.get("openInterest") or 0.0)
                snapshot: Dict[str, Any] = {
                    "source": "hyperliquid",
                    "coin": coin,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "mark_price": mark or None,
                    "oracle_price": oracle or None,
                    "basis_premium_pct": (
                        ((mark - oracle) / oracle) * 100.0
                        if mark > 0 and oracle > 0 else None
                    ),
                    "funding_rate": (
                        float(context.get("funding"))
                        if context.get("funding") is not None else None
                    ),
                    "open_interest": open_interest or None,
                    "open_interest_notional_usd": (
                        open_interest * mark if open_interest > 0 and mark > 0 else None
                    ),
                    "day_notional_volume_usd": (
                        float(context.get("dayNtlVlm"))
                        if context.get("dayNtlVlm") is not None else None
                    ),
                    "premium": (
                        float(context.get("premium"))
                        if context.get("premium") is not None else None
                    ),
                }

                if book_response.status_code == 200:
                    book = book_response.json() or {}
                    levels = book.get("levels") or []
                    bids = levels[0] if len(levels) > 0 else []
                    asks = levels[1] if len(levels) > 1 else []

                    def _notional(rows: List[Dict[str, Any]]) -> float:
                        return sum(
                            float(row.get("px") or 0.0) * float(row.get("sz") or 0.0)
                            for row in rows[:20]
                        )

                    bid_depth = _notional(bids)
                    ask_depth = _notional(asks)
                    total_depth = bid_depth + ask_depth
                    top_bid = float((bids[0] if bids else {}).get("px") or 0.0)
                    top_ask = float((asks[0] if asks else {}).get("px") or 0.0)
                    snapshot.update({
                        "bid_depth_usd": bid_depth,
                        "ask_depth_usd": ask_depth,
                        "orderbook_imbalance": (
                            (bid_depth - ask_depth) / total_depth if total_depth > 0 else None
                        ),
                        "top_bid": top_bid or None,
                        "top_ask": top_ask or None,
                        "spread_pct": (
                            ((top_ask - top_bid) / top_bid) * 100.0
                            if top_bid > 0 and top_ask > 0 else None
                        ),
                    })

                self._hyperliquid_snapshot_cache[coin] = (snapshot, datetime.utcnow())
                return snapshot
        except Exception as exc:
            logger.debug("Hyperliquid venue snapshot unavailable for %s: %s", coin, exc)
            return None

    async def _fetch_token_event_context(self, symbol: str) -> Dict[str, Any]:
        """Return recent supply/listing events relevant to this token, when available."""
        token = self._signal_symbol_key(symbol)
        if not token or ":" in token:
            return {"available": False, "events": []}

        cached_events, cached_at = self._token_event_cache
        if (
            cached_at is None
            or (datetime.utcnow() - cached_at).total_seconds() >= self._token_event_ttl_seconds
        ):
            try:
                from kata.services.news_service import create_news_service

                news_service = create_news_service(os.getenv("NEWSAPI_KEY"))
                try:
                    detected = await news_service.detect_market_events(hours_back=48)
                finally:
                    await news_service.close_session()
                cached_events = [
                    {
                        "title": event.title,
                        "description": event.description,
                        "event_type": event.event_type,
                        "impact_level": event.impact_level,
                        "affected_assets": list(event.affected_assets or []),
                        "sentiment": event.sentiment,
                        "source": event.source,
                        "timestamp": event.timestamp.isoformat(),
                        "url": event.url,
                    }
                    for event in detected
                ]
                self._token_event_cache = (cached_events, datetime.utcnow())
            except Exception as exc:
                logger.debug("Token event feed unavailable: %s", exc)
                cached_events = []
                self._token_event_cache = (cached_events, datetime.utcnow())

        event_keywords = (
            "unlock",
            "vesting",
            "listing",
            "delisting",
            "supply",
            "airdrop",
            "burn",
            "emission",
        )
        token_pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", re.IGNORECASE)
        relevant: List[Dict[str, Any]] = []
        for event in cached_events:
            assets = {
                str(asset or "").upper().strip()
                for asset in event.get("affected_assets") or []
            }
            text_value = " ".join(
                str(event.get(key) or "")
                for key in ("title", "description", "event_type")
            )
            token_match = token in assets or (
                len(token) >= 3 and token_pattern.search(text_value) is not None
            )
            supply_event = any(keyword in text_value.lower() for keyword in event_keywords)
            if token_match and supply_event:
                relevant.append(event)

        return {
            "available": bool(cached_events),
            "events": relevant[:5],
            "lookback_hours": 48,
            "source": "news_event_feed",
        }

    def _apply_execution_costs(self, gross_pnl_pct: float, leverage: float = 1.0,
                                hold_hours: float = 8.0, funding_rate: float = 0.0001) -> float:
        """Adjust gross PnL % for round-trip taker fees + slippage + funding cost (pre-leverage scale)."""
        funding_cost_pct = abs(funding_rate) * 100.0 * (hold_hours / 8.0)
        cost = self._round_trip_cost_pct + funding_cost_pct
        leveraged_cost = cost * max(1.0, float(leverage))
        return float(gross_pnl_pct) - leveraged_cost

    def _net_rr_after_costs(self, entry: float, target: float, stop: float,
                              direction: str, leverage: float = 1.0) -> float:
        """Net RR after applying round-trip execution cost. Used to gate signals (reject if < 1.5)."""
        if entry <= 0 or target <= 0 or stop <= 0:
            return 0.0
        if direction == "LONG":
            gross_reward = (target - entry) / entry * 100.0
            gross_risk = (entry - stop) / entry * 100.0
        else:
            gross_reward = (entry - target) / entry * 100.0
            gross_risk = (stop - entry) / entry * 100.0
        leveraged_cost = self._round_trip_cost_pct * max(1.0, float(leverage))
        net_reward = gross_reward - leveraged_cost
        net_risk = gross_risk + leveraged_cost
        return float(net_reward / net_risk) if net_risk > 0 else 0.0

    def _sample_confidence(self, sample_size: int, full_at: int = 80) -> float:
        """Map a sample size to a 0..1 confidence weight without pretending small-N is certain."""
        try:
            n = max(0, int(sample_size or 0))
            target = max(1, int(full_at or 80))
            return float(max(0.0, min(1.0, n / target)))
        except Exception:
            return 0.0

    def _recent_edge_stats(self, min_samples: int = 10) -> Dict[str, float]:
        """Return recent outcome edge statistics from resolved, informative signals only."""
        try:
            outcomes = list(self._recent_outcomes or [])
            if len(outcomes) < min_samples:
                return {}

            pnls = [float(pnl) for _, _, pnl in outcomes if pnl is not None]
            if len(pnls) < min_samples:
                return {}

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            if not wins or not losses:
                return {
                    "sample_size": float(len(pnls)),
                    "win_rate": float(len(wins) / len(pnls)),
                    "avg_win_pct": float(np.mean(wins)) if wins else 0.0,
                    "avg_loss_pct": abs(float(np.mean(losses))) if losses else 0.0,
                    "avg_pnl_pct": float(np.mean(pnls)),
                }

            return {
                "sample_size": float(len(pnls)),
                "win_rate": float(len(wins) / len(pnls)),
                "avg_win_pct": float(np.mean(wins)),
                "avg_loss_pct": abs(float(np.mean(losses))),
                "avg_pnl_pct": float(np.mean(pnls)),
            }
        except Exception as exc:
            logger.debug(f"Recent edge stats skipped: {exc}")
            return {}

    def _get_directional_performance(self) -> Dict[str, Dict[str, float]]:
        """Recent resolved win-rate by direction (LONG/SHORT), cached.

        This is the self-learning bias detector: it answers "have shorts been working
        lately?" from real resolved outcomes over a short, regime-sensitive window. Cached
        for _direction_perf_ttl_seconds to avoid querying on every signal.
        """
        now = datetime.utcnow()
        if (
            self._direction_perf_cache
            and self._direction_perf_cache_at is not None
            and (now - self._direction_perf_cache_at).total_seconds() < self._direction_perf_ttl_seconds
        ):
            return self._direction_perf_cache

        result: Dict[str, Dict[str, float]] = {}
        try:
            if not getattr(settings, "DIRECTION_LEARNING_ENABLED", True) or not self.supabase:
                return {}
            lookback_days = int(getattr(settings, "DIRECTION_LEARNING_LOOKBACK_DAYS", 14))
            cutoff = (now - timedelta(days=lookback_days)).isoformat()
            rows = (
                self.supabase.table("platform_signals")
                .select("direction,status")
                .gte("created_at", cutoff)
                .execute()
                .data
            ) or []
            win_st = {"hit_target_1", "hit_target_2"}
            loss_st = {"hit_stop_loss"}
            agg: Dict[str, List[int]] = {"LONG": [0, 0], "SHORT": [0, 0]}  # [wins, resolved]
            for r in rows:
                d = r.get("direction")
                if d not in agg:
                    continue
                st = str(r.get("status") or "").lower()
                if st in win_st:
                    agg[d][0] += 1
                    agg[d][1] += 1
                elif st in loss_st:
                    agg[d][1] += 1
            for d, (wins, resolved) in agg.items():
                if resolved > 0:
                    result[d] = {"win_rate": wins / resolved, "resolved": float(resolved)}
            self._direction_perf_cache = result
            self._direction_perf_cache_at = now
        except Exception as exc:
            logger.debug(f"Directional performance fetch skipped: {exc}")
            return self._direction_perf_cache or {}
        return result

    def _directional_confidence_adjustment(self, direction: str) -> float:
        """Confidence delta from recent per-direction performance.

        Negative when the direction has been losing recently (e.g. shorts in an uptrend),
        positive (smaller) when it has been winning. Scaled by sample size so thin evidence
        moves confidence only a little. Auto-reverses as the regime turns — no static rule.
        """
        if direction not in ("LONG", "SHORT"):
            return 0.0
        perf = self._get_directional_performance().get(direction)
        if not perf:
            return 0.0
        resolved = float(perf.get("resolved", 0.0))
        win_rate = float(perf.get("win_rate", 0.5))
        min_samples = max(1, int(getattr(settings, "DIRECTION_LEARNING_MIN_SAMPLES", 8)))
        max_adj = float(getattr(settings, "DIRECTION_LEARNING_MAX_CONF_ADJ", 0.10))
        # Sample confidence: full strength at min_samples, scaled below.
        sample_conf = min(1.0, resolved / float(min_samples))
        # Deviation from coin-flip drives the sign and magnitude.
        deviation = win_rate - 0.5  # range [-0.5, 0.5]
        delta = deviation * 2.0 * max_adj * sample_conf  # at wr=0, full -max_adj*sample_conf
        return max(-max_adj, min(max_adj, delta))

    def _price_move_pct(self, entry: float, level: float) -> float:
        """Absolute price move percentage between entry and a level."""
        try:
            ep = abs(float(entry or 0.0))
            lv = float(level or 0.0)
            if ep <= 0 or lv <= 0:
                return 0.0
            return abs(lv - ep) / ep * 100.0
        except Exception:
            return 0.0

    def _apply_learning_risk_reduction(
        self,
        decision: Any,
        adjustment: Optional[LearningAdjustment] = None,
        technical_analysis: Optional[TechnicalAnalysis] = None,
        opportunity_score: Optional[OpportunityScore] = None,
        policy_win_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply evidence-backed downside-only learning adjustments to model-selected risk.

        The LLM chooses the initial size and leverage. Learning may reduce either value
        when resolved outcomes or matched patterns are weak, but never increases them
        or replaces them with a fixed sizing preset.
        """
        context: Dict[str, Any] = {}
        try:
            direction = str(getattr(decision, "recommendation", "") or "").upper().strip()
            if direction not in {"LONG", "SHORT"}:
                return context

            original_pos = float(getattr(decision, "position_size", 0.0) or 0.0)
            original_lev = int(round(float(getattr(decision, "leverage", 1) or 1)))
            if original_pos <= 0:
                return context

            signal_state = dict(self._signal_quality_state or {})
            resolved_n = int(signal_state.get("resolved_signals") or 0)
            informative_n = int(signal_state.get("informative_signals") or 0)
            expired_n = int(signal_state.get("expired_signals") or 0)
            expiry_rate = (expired_n / resolved_n) if resolved_n > 0 else 0.0
            historical_wr = None
            if informative_n >= 10 and signal_state.get("win_rate") is not None:
                historical_wr = max(0.0, min(1.0, float(signal_state.get("win_rate") or 0.0)))

            conf = max(0.0, min(1.0, float(getattr(decision, "confidence", 0.0) or 0.0)))
            model_prob = max(0.42, min(0.70, 0.40 + conf * 0.30))
            opp_prob = None
            if opportunity_score is not None:
                try:
                    opp_prob = max(0.42, min(0.68, 0.43 + float(opportunity_score.overall_score or 0.0) * 0.24))
                except Exception:
                    opp_prob = None

            policy_prob = policy_win_probability
            if policy_prob is None:
                policy_prob = getattr(decision, "policy_win_probability", None)
            if policy_prob is not None:
                try:
                    policy_prob = max(0.05, min(0.95, float(policy_prob)))
                except (TypeError, ValueError):
                    policy_prob = None

            pattern_prob = None
            pattern_level_scale = 0.0
            pattern_matches = 0
            if adjustment is not None:
                confidence_level = str(adjustment.confidence_level or "medium").lower()
                pattern_level_scale = {"high": 1.0, "medium": 0.65, "low": 0.30}.get(confidence_level, 0.50)
                pattern_matches = len(adjustment.pattern_matches or [])
                try:
                    # DB pattern confidence_boost ~= (success_rate - 0.5) * 0.4.
                    pattern_prob = 0.50 + (float(adjustment.confidence_adjustment or 0.0) / 0.40)
                    pattern_prob = max(0.25, min(0.75, pattern_prob))
                except (TypeError, ValueError):
                    pattern_prob = None

            components: List[Tuple[float, float, str]] = [(model_prob, 0.20, "model_confidence")]
            if opp_prob is not None:
                components.append((opp_prob, 0.10, "opportunity_score"))
            if policy_prob is not None:
                components.append((policy_prob, 0.35, "policy"))
            if historical_wr is not None:
                hist_weight = 0.35 * self._sample_confidence(informative_n, full_at=80)
                if hist_weight > 0:
                    components.append((historical_wr, hist_weight, "recent_history"))
            if pattern_prob is not None and pattern_matches > 0:
                components.append((pattern_prob, 0.25 * pattern_level_scale, "pattern_learning"))

            total_weight = sum(weight for _, weight, _ in components)
            blended_prob = (
                sum(prob * weight for prob, weight, _ in components) / total_weight
                if total_weight > 0
                else model_prob
            )
            blended_prob = max(0.05, min(0.95, blended_prob))

            recent_stats = self._recent_edge_stats(min_samples=10)
            reward_pct = self._price_move_pct(getattr(decision, "entry_price", 0.0), getattr(decision, "target_1", 0.0))
            risk_pct = self._price_move_pct(getattr(decision, "entry_price", 0.0), getattr(decision, "stop_loss", 0.0))
            if reward_pct <= 0 and getattr(decision, "risk_reward_ratio", None):
                reward_pct = max(0.5, risk_pct * float(getattr(decision, "risk_reward_ratio") or 1.5))
            if risk_pct <= 0:
                risk_pct = max(0.75, reward_pct / max(float(getattr(decision, "risk_reward_ratio", 1.5) or 1.5), 1.0))

            avg_win = float(recent_stats.get("avg_win_pct") or 0.0)
            avg_loss = float(recent_stats.get("avg_loss_pct") or 0.0)
            if avg_win <= 0:
                avg_win = max(0.5, reward_pct)
            if avg_loss <= 0:
                avg_loss = max(0.5, risk_pct)

            raw_kelly = 0.0
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            if payoff_ratio > 0:
                raw_kelly = blended_prob - (1.0 - blended_prob) / payoff_ratio
            raw_kelly = max(0.0, raw_kelly)

            evidence_confidence = max(
                self._sample_confidence(informative_n, full_at=80),
                pattern_level_scale if pattern_matches else 0.0,
                0.65 if policy_prob is not None else 0.0,
            )
            if informative_n < 10 and not pattern_matches and policy_prob is None:
                evidence_confidence = 0.20
            evidence_confidence = max(0.15, min(1.0, evidence_confidence * (1.0 - min(expiry_rate, 0.75) * 0.30)))
            kelly_position_pct = raw_kelly * 100.0 * 0.25 * evidence_confidence

            sharpe = self._recent_sharpe_ratio(min_samples=20)
            vol = 0.0
            if technical_analysis is not None:
                try:
                    vol = max(0.0, float(getattr(technical_analysis, "volatility_24h", 0.0) or 0.0))
                except (TypeError, ValueError):
                    vol = 0.0

            negative_recent_edge = (
                (sharpe is not None and sharpe < 0.0)
                or (historical_wr is not None and historical_wr < 0.42)
                or float(signal_state.get("avg_leveraged_pnl_pct") or 0.0) < -0.25
            )
            strong_edge = (
                blended_prob >= 0.62
                and evidence_confidence >= 0.45
                and not negative_recent_edge
                and expiry_rate < 0.55
            )
            exceptional_edge = (
                blended_prob >= 0.68
                and evidence_confidence >= 0.65
                and not negative_recent_edge
                and expiry_rate < 0.35
                and vol < 4.0
            )
            moderate_edge = (
                blended_prob >= 0.56
                and evidence_confidence >= 0.30
                and not negative_recent_edge
                and expiry_rate < 0.65
            )
            edge_bucket = (
                "exceptional"
                if exceptional_edge
                else "strong"
                if strong_edge
                else "moderate"
                if moderate_edge
                else "defensive"
            )

            position_reduction = 1.0
            leverage_reduction = 1.0
            reduction_reasons: List[str] = []

            if adjustment is not None and pattern_matches > 0:
                scale = {"high": 1.0, "medium": 0.60, "low": 0.25}.get(
                    str(adjustment.confidence_level or "medium").lower(),
                    0.50,
                )
                try:
                    suggested_pos = max(0.35, min(1.0, float(adjustment.position_size_multiplier or 1.0)))
                    suggested_lev = max(0.35, min(1.0, float(adjustment.leverage_multiplier or 1.0)))
                    learned_pos_factor = 1.0 - ((1.0 - suggested_pos) * scale)
                    learned_lev_factor = 1.0 - ((1.0 - suggested_lev) * scale)
                    if learned_pos_factor < 0.999 or learned_lev_factor < 0.999:
                        position_reduction = min(position_reduction, learned_pos_factor)
                        leverage_reduction = min(leverage_reduction, learned_lev_factor)
                        reduction_reasons.append("matched_pattern")
                except (TypeError, ValueError):
                    pass

            if historical_wr is not None and informative_n >= 10 and historical_wr < 0.50:
                weakness = min(1.0, (0.50 - historical_wr) / 0.50)
                sample_scale = self._sample_confidence(informative_n, full_at=20)
                history_pos_factor = 1.0 - min(0.35, weakness * 0.50 * sample_scale)
                history_lev_factor = 1.0 - min(0.25, weakness * 0.35 * sample_scale)
                position_reduction = min(position_reduction, history_pos_factor)
                leverage_reduction = min(leverage_reduction, history_lev_factor)
                reduction_reasons.append("weak_recent_win_rate")

            if resolved_n >= 10 and expiry_rate > 0.50:
                expiry_overhang = min(1.0, (expiry_rate - 0.50) / 0.50)
                sample_scale = self._sample_confidence(resolved_n, full_at=30)
                expiry_pos_factor = 1.0 - min(0.20, expiry_overhang * 0.35 * sample_scale)
                expiry_lev_factor = 1.0 - min(0.12, expiry_overhang * 0.20 * sample_scale)
                position_reduction = min(position_reduction, expiry_pos_factor)
                leverage_reduction = min(leverage_reduction, expiry_lev_factor)
                reduction_reasons.append("high_expiry_rate")

            adjusted_pos = original_pos
            if position_reduction < 0.999:
                size_floor = min(original_pos, 0.01)
                adjusted_pos = min(
                    original_pos,
                    max(size_floor, round(float(original_pos * position_reduction), 2)),
                )

            adjusted_lev = original_lev
            if leverage_reduction < 0.999:
                adjusted_lev = int(max(1, np.floor((original_lev * leverage_reduction) + 0.5)))
                adjusted_lev = min(original_lev, adjusted_lev)

            post_processing_applied = adjusted_pos < original_pos or adjusted_lev < original_lev

            decision.position_size = adjusted_pos
            decision.leverage = adjusted_lev

            context = {
                "blended_win_probability": round(float(blended_prob), 4),
                "policy_win_probability": round(float(policy_prob), 4) if policy_prob is not None else None,
                "historical_win_rate": round(float(historical_wr), 4) if historical_wr is not None else None,
                "pattern_probability": round(float(pattern_prob), 4) if pattern_prob is not None else None,
                "evidence_confidence": round(float(evidence_confidence), 4),
                "resolved_signals": resolved_n,
                "informative_signals": informative_n,
                "expired_signals": expired_n,
                "expiry_rate": round(float(expiry_rate), 4),
                "recent_sharpe": round(float(sharpe), 4) if sharpe is not None else None,
                "avg_win_pct": round(float(avg_win), 4),
                "avg_loss_pct": round(float(avg_loss), 4),
                "reward_pct": round(float(reward_pct), 4),
                "risk_pct": round(float(risk_pct), 4),
                "raw_kelly_fraction": round(float(raw_kelly), 5),
                "fractional_kelly_position_pct": round(float(kelly_position_pct), 4),
                "volatility_24h": round(float(vol), 4),
                "position_size_before": round(float(original_pos), 4),
                "position_size_after": round(float(adjusted_pos), 4),
                "leverage_before": int(original_lev),
                "leverage_after": int(adjusted_lev),
                "position_reduction_factor": round(float(position_reduction), 4),
                "leverage_reduction_factor": round(float(leverage_reduction), 4),
                "reduction_reasons": reduction_reasons,
                "edge_bucket": edge_bucket,
                "post_processing_applied": post_processing_applied,
                "components": [name for _, _, name in components],
            }
            setattr(decision, "learning_sizing_context", context)

            logger.info(
                "📊 Downside-only learning sizing: p=%.2f edge=%s expiry=%.0f%% vol=%.1f%% "
                "pos %.2f%%->%.2f%% lev %dx->%dx reasons=%s",
                blended_prob,
                edge_bucket,
                expiry_rate * 100.0,
                vol,
                original_pos,
                adjusted_pos,
                original_lev,
                adjusted_lev,
                ",".join(reduction_reasons) or "none",
            )

            decision.reasoning = (
                f"{decision.reasoning}\n"
                f"Learning sizing context: edge={edge_bucket}, "
                f"p_win={blended_prob:.2f}, kelly={kelly_position_pct:.2f}%, "
                f"expiry={expiry_rate:.0%}, vol={vol:.1f}%, "
                f"position {original_pos:.2f}%->{adjusted_pos:.2f}%, "
                f"leverage {original_lev}x->{adjusted_lev}x (downside-only learning)."
            )
            return context
        except Exception as exc:
            logger.debug(f"Learning risk sizing skipped: {exc}")
            return context

    def _check_drawdown_state(self, current_equity: float) -> Tuple[bool, str]:
        """Update high-water mark and return (allowed, reason). Halts at 30%, pauses at 20%."""
        if current_equity <= 0:
            return True, "no_equity_data"
        if self._equity_high_water_mark is None or current_equity > self._equity_high_water_mark:
            self._equity_high_water_mark = float(current_equity)
            self._drawdown_paused = False
            return True, "high_water_mark_updated"
        drawdown = (self._equity_high_water_mark - current_equity) / self._equity_high_water_mark
        if drawdown >= self._drawdown_halt_threshold:
            self._drawdown_paused = True
            return False, f"halt_drawdown_{drawdown:.1%}"
        if drawdown >= self._drawdown_pause_threshold:
            self._drawdown_paused = True
            return False, f"pause_drawdown_{drawdown:.1%}"
        if self._drawdown_paused and drawdown < self._drawdown_pause_threshold * 0.7:
            self._drawdown_paused = False
        return (not self._drawdown_paused), f"drawdown_{drawdown:.1%}"

    def _attribution_key(self, regime: str, direction: str, leverage: float) -> str:
        """Bucket leverage into 1x/2-3x/4-5x/6-10x/10x+ for attribution slicing."""
        lev = max(1, int(round(leverage)))
        if lev <= 1:
            bucket = "1x"
        elif lev <= 3:
            bucket = "2-3x"
        elif lev <= 5:
            bucket = "4-5x"
        elif lev <= 10:
            bucket = "6-10x"
        else:
            bucket = "10x+"
        return f"{(regime or 'unknown').lower()}|{direction}|{bucket}"

    def _record_outcome_attribution(self, regime: str, direction: str, leverage: float,
                                     pnl_pct_leveraged: float, is_win: bool) -> None:
        """Record an outcome into the sliced attribution table for combo profitability tracking."""
        key = self._attribution_key(regime, direction, leverage)
        stats = self._attribution_stats.setdefault(key, {
            "n": 0, "wins": 0, "sum_pnl": 0.0, "sum_pnl_sq": 0.0,
            "max_loss": 0.0, "max_win": 0.0,
        })
        stats["n"] = stats["n"] + 1
        if is_win:
            stats["wins"] = stats["wins"] + 1
        stats["sum_pnl"] = stats["sum_pnl"] + float(pnl_pct_leveraged)
        stats["sum_pnl_sq"] = stats["sum_pnl_sq"] + float(pnl_pct_leveraged) ** 2
        if pnl_pct_leveraged < stats["max_loss"]:
            stats["max_loss"] = float(pnl_pct_leveraged)
        if pnl_pct_leveraged > stats["max_win"]:
            stats["max_win"] = float(pnl_pct_leveraged)

    def _attribution_combo_blocked(self, regime: str, direction: str, leverage: float,
                                     min_n: int = 20) -> Tuple[bool, str]:
        """Return (blocked, reason) for an attribution bucket. Blocks combos with WR<40% over 20+ samples."""
        key = self._attribution_key(regime, direction, leverage)
        stats = self._attribution_stats.get(key)
        if not stats or stats["n"] < min_n:
            return False, "insufficient_samples"
        wr = stats["wins"] / stats["n"]
        avg_pnl = stats["sum_pnl"] / stats["n"]
        if wr < 0.40 and avg_pnl < 0:
            return True, f"combo_blocked_wr={wr:.1%}_avgpnl={avg_pnl:.2f}%"
        return False, f"combo_ok_wr={wr:.1%}_avgpnl={avg_pnl:.2f}%"

    async def _htf_trend_allows(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Higher-timeframe (1d) trend gate. Reject directions counter to strong daily trend.

        Returns (allowed, reason). Allowed=True if direction aligns with or doesn't contradict
        a strong 1d trend; False if direction is counter to a strongly directional 1d trend.
        Uses 20-bar EMA slope and price-vs-EMA position as the trend definition.
        """
        try:
            if direction not in ("LONG", "SHORT"):
                return True, "non_directional"
            if not self.binance:
                return True, "no_exchange"
            ohlcv = await self.rate_limited_api_call(self.binance.fetch_ohlcv, symbol, "1d", limit=30)
            if not ohlcv or len(ohlcv) < 20:
                return True, "insufficient_htf_history"
            closes = np.array([float(c[4]) for c in ohlcv if c and c[4] is not None])
            if len(closes) < 20:
                return True, "insufficient_htf_closes"
            ema20 = float(np.mean(closes[-20:]))
            ema5 = float(np.mean(closes[-5:]))
            current = float(closes[-1])
            # Trend strength: % distance of price from EMA20, plus EMA5 vs EMA20 slope direction.
            distance_pct = (current - ema20) / ema20 * 100.0
            ema_slope = (ema5 - ema20) / ema20 * 100.0
            # Extreme-only safety net (configurable). This is NOT a short-blocker — it only
            # vetoes signals fighting a VERY strong daily trend (both distance AND slope
            # extreme). Direction selectivity in normal conditions is handled adaptively by
            # the directional-learning confidence haircut (_directional_confidence_adjustment),
            # which self-corrects from recent resolved outcomes instead of a static rule.
            dist_thresh = float(getattr(settings, "HTF_TREND_GATE_DISTANCE_PCT", 3.0) or 3.0)
            slope_thresh = float(getattr(settings, "HTF_TREND_GATE_SLOPE_PCT", 1.0) or 1.0)
            strong_up = distance_pct > dist_thresh and ema_slope > slope_thresh
            strong_down = distance_pct < -dist_thresh and ema_slope < -slope_thresh
            if direction == "SHORT" and strong_up:
                return False, f"htf_strong_uptrend_dist={distance_pct:+.1f}%_slope={ema_slope:+.1f}%"
            if direction == "LONG" and strong_down:
                return False, f"htf_strong_downtrend_dist={distance_pct:+.1f}%_slope={ema_slope:+.1f}%"
            return True, f"htf_aligned_or_neutral_dist={distance_pct:+.1f}%_slope={ema_slope:+.1f}%"
        except Exception as exc:
            logger.debug(f"HTF trend gate skipped for {symbol}: {exc}")
            return True, "htf_check_failed"

    def _recent_sharpe_ratio(self, min_samples: int = 20) -> Optional[float]:
        """Compute Sharpe-like ratio over recent outcomes (mean PnL / stdev). None if too few samples."""
        if len(self._recent_outcomes) < min_samples:
            return None
        pnls = [pnl for _, _, pnl in self._recent_outcomes]
        if not pnls:
            return None
        mean_p = float(np.mean(pnls))
        std_p = float(np.std(pnls))
        if std_p <= 0:
            return None
        return mean_p / std_p

    def _record_recent_outcome(self, is_win: bool, pnl_pct: float) -> None:
        """Track last 100 outcomes for concept-drift watchdog."""
        self._recent_outcomes.append((datetime.utcnow(), bool(is_win), float(pnl_pct)))
        if len(self._recent_outcomes) > 100:
            self._recent_outcomes = self._recent_outcomes[-100:]
        if len(self._recent_outcomes) >= 30:
            wins = sum(1 for _, w, _ in self._recent_outcomes if w)
            wr = wins / len(self._recent_outcomes)
            self._peak_rolling_win_rate = max(self._peak_rolling_win_rate, wr)
            if (self._peak_rolling_win_rate - wr) >= 0.10 and not self._concept_drift_alerted:
                logger.warning(
                    f"⚠️ Concept drift suspected: rolling WR dropped from peak "
                    f"{self._peak_rolling_win_rate:.1%} to {wr:.1%}. Review pipeline."
                )
                self._concept_drift_alerted = True
            elif (self._peak_rolling_win_rate - wr) < 0.05:
                self._concept_drift_alerted = False

    def calculate_correlation(self, token_change_24h: float, btc_change_24h: float) -> str:
        """
        Calculate correlation between token and BTC performance.
        Returns HIGH, MEDIUM, or LOW correlation.
        """
        # Simple correlation based on directional similarity
        correlation_score = abs(token_change_24h - btc_change_24h)

        if correlation_score < 2.0:
            return "HIGH"
        elif correlation_score < 5.0:
            return "MEDIUM"
        else:
            return "LOW"

    def determine_market_regime(self, btc_data: Dict[str, float]) -> str:
        """
        Determine overall market regime based on BTC structure, volatility, and momentum.
        Returns: BULL_TRENDING, BEAR_TRENDING, BULL_RANGING, BEAR_RANGING, or SIDEWAYS
        """
        current_price = btc_data.get('current_price', 0.0)
        ema_20 = btc_data.get('ema_20', 0.0)
        ema_50 = btc_data.get('ema_50', 0.0)
        rsi = btc_data.get('rsi_14', 50.0)
        bb_width = btc_data.get('bb_width', 0.0)
        
        # Fallback to legacy if TA data is missing
        if ema_20 == 0.0 or ema_50 == 0.0:
            btc_7d = btc_data.get('price_7d', 0.0)
            if btc_7d > 7.0: return "BULL_TRENDING"
            elif btc_7d < -7.0: return "BEAR_TRENDING"
            elif btc_7d > 3.0: return "BULL_RANGING"
            elif btc_7d < -3.0: return "BEAR_RANGING"
            else: return "SIDEWAYS"

        # Structural analysis
        is_uptrend = ema_20 > ema_50 and current_price > ema_50
        is_downtrend = ema_20 < ema_50 and current_price < ema_50
        
        # Volatility expansion (trending) vs contraction (ranging)
        # Typical daily BB width for BTC ranges from 0.05 (tight) to 0.20+ (explosive)
        is_expanding = bb_width > 0.10
        
        if is_uptrend:
            if is_expanding and rsi > 55:
                return "BULL_TRENDING"
            else:
                return "BULL_RANGING"
        elif is_downtrend:
            if is_expanding and rsi < 45:
                return "BEAR_TRENDING"
            else:
                return "BEAR_RANGING"
        else:
            return "SIDEWAYS"

    @staticmethod
    def _parse_all_market_mark_price_payload(payload: Any) -> Dict[str, Dict[str, Any]]:
        """Normalize Binance's complete all-market mark-price array."""
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = payload["data"]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        snapshot: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            # The post-migration stream can contain UM and CM contracts.
            # st=1 is USDⓈ-M; older payloads omit the field.
            market_type = row.get("st")
            if market_type not in (None, 1, "1"):
                continue
            symbol = str(row.get("s") or "").upper().strip()
            if not symbol.endswith("USDT"):
                continue
            try:
                mark_price = float(row.get("p") or 0.0)
                if mark_price <= 0:
                    continue
                funding_rate = (
                    float(row["r"])
                    if row.get("r") not in (None, "")
                    else None
                )
                index_price = (
                    float(row["i"])
                    if row.get("i") not in (None, "")
                    else None
                )
            except (TypeError, ValueError):
                continue
            snapshot[symbol] = {
                "symbol": symbol,
                "mark_price": mark_price,
                "index_price": index_price,
                "funding_rate": funding_rate,
                "timestamp": row.get("E") or int(datetime.utcnow().timestamp() * 1000),
            }
        return snapshot

    async def fetch_all_market_mark_prices_websocket(
        self,
        timeout: int = 15,
        max_attempts: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch Binance's complete USDⓈ-M mark-price/funding snapshot.

        Unlike !miniTicker@arr and !ticker@arr, this stream documents an
        all-symbol array. It enriches—but never replaces—the REST 24h OHLCV
        snapshot required by opportunity scoring.
        """
        ws_url = "wss://fstream.binance.com/market/ws/!markPrice@arr@1s"
        ssl_context = ssl.create_default_context()
        logger.info("📡 Connecting to Binance all-market mark-price WebSocket...")

        for attempt in range(1, max(1, int(max_attempts or 1)) + 1):
            if self._shutdown_requested:
                return {}
            try:
                async with websockets.connect(
                    ws_url,
                    ssl=ssl_context,
                    open_timeout=15,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=8_000_000,
                ) as websocket:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    snapshot = self._parse_all_market_mark_price_payload(
                        json.loads(message)
                    )
                    if snapshot:
                        logger.info(
                            "✅ Received %d USDⓈ-M mark prices from WebSocket",
                            len(snapshot),
                        )
                        return snapshot
                    logger.warning(
                        "⚠️ Binance all-market mark-price payload had no usable USDⓈ-M rows"
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ All-market mark-price WebSocket timed out after %ss "
                    "(attempt %s/%s)",
                    timeout,
                    attempt,
                    max_attempts,
                )
            except Exception as exc:
                logger.warning(
                    "⚠️ All-market mark-price WebSocket failed (attempt %s/%s): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
            if attempt < max_attempts:
                await asyncio.sleep(min(8, attempt * 2))
        return {}

    def _merge_mark_price_snapshot(
        self,
        tickers: Dict[str, Any],
        mark_prices: Dict[str, Dict[str, Any]],
    ) -> int:
        """Overlay live mark price/funding on the complete REST ticker universe."""
        ticker_keys = {
            self._normalize_symbol_to_market_id(symbol): symbol
            for symbol in tickers
        }
        merged = 0
        fetched_at = datetime.utcnow()
        for market_id, mark_data in (mark_prices or {}).items():
            ticker_key = ticker_keys.get(self._normalize_symbol_to_market_id(market_id))
            if ticker_key is None:
                continue
            ticker = tickers.get(ticker_key)
            if not isinstance(ticker, dict):
                continue
            try:
                mark_price = float(mark_data.get("mark_price") or 0.0)
            except (TypeError, ValueError):
                continue
            if mark_price <= 0:
                continue

            ticker["last"] = mark_price
            ticker["close"] = mark_price
            ticker["markPrice"] = mark_price
            ticker["timestamp"] = mark_data.get("timestamp") or ticker.get("timestamp")

            try:
                open_price = float(ticker.get("open") or 0.0)
            except (TypeError, ValueError):
                open_price = 0.0
            if open_price > 0:
                ticker["change"] = mark_price - open_price
                ticker["percentage"] = (
                    (mark_price - open_price) / open_price
                ) * 100.0

            try:
                ticker["high"] = max(float(ticker.get("high") or mark_price), mark_price)
                ticker["low"] = min(float(ticker.get("low") or mark_price), mark_price)
            except (TypeError, ValueError):
                ticker["high"] = mark_price
                ticker["low"] = mark_price

            index_price = mark_data.get("index_price")
            funding_rate = mark_data.get("funding_rate")
            if index_price is not None:
                ticker["indexPrice"] = index_price
            if funding_rate is not None:
                ticker["fundingRate"] = funding_rate
                cache_value = (float(funding_rate), fetched_at)
                self._funding_rate_cache[ticker_key] = cache_value
                self._funding_rate_cache[
                    self._normalize_symbol_to_market_id(ticker_key)
                ] = cache_value
            merged += 1
        return merged

    async def discover_market_opportunities(self, max_opportunities: int = 25) -> List[str]:
        """
        Discover best trading opportunities from entire market.

        Stage 1: Get all tickers (1 API call) - with fallback to CoinGecko
        Stage 2: Calculate opportunity scores
        Stage 3: Return top opportunities ranked by score
        """
        logger.info("🔍 Discovering market opportunities...")

        try:
            if self._shutdown_requested:
                logger.info("🛑 Shutdown requested - skipping market discovery")
                return []

            # Reset cache every discovery run to avoid stale scores.
            self._opportunity_score_cache = {}

            # Ensure services are initialized
            if not self._services_initialized:
                await self._initialize_services()

            tickers = {}
            if self._shutdown_requested:
                logger.info("🛑 Shutdown requested - aborting ticker fetch")
                return []

            # Discovery always starts from a complete 24h OHLCV snapshot. Binance's
            # !miniTicker@arr/!ticker@arr WebSockets only contain symbols changed in
            # that interval and therefore cannot define the full candidate universe.
            # Fetch the complete all-market mark/funding array concurrently and use
            # it only to freshen the REST snapshot.
            mark_price_task = asyncio.create_task(
                self.fetch_all_market_mark_prices_websocket(
                    timeout=15,
                    max_attempts=1,
                )
            )
            api_public_url = self.binance.urls.get('api', {}).get('public', 'Unknown')
            logger.info(f"🔗 Fetching complete Binance Futures REST snapshot: {api_public_url}")

            try:
                tickers = await self.rate_limited_api_call(self.binance.fetch_tickers)
            except Exception as full_snapshot_error:
                logger.warning(
                    f"⚠️ Full REST snapshot failed: {full_snapshot_error}. Falling back to curated symbols."
                )

                preferred_symbols = [
                    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
                    'DOGEUSDT', 'POLUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT',
                    'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'FILUSDT',
                    'NEARUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FTMUSDT'
                ]

                # Validate symbols against currently listed futures markets to avoid stale hardcoded pairs.
                validated_symbols = preferred_symbols
                try:
                    markets = await self.rate_limited_api_call(self.binance.load_markets, True)
                    active_market_ids = {
                        str((m or {}).get("id") or "").upper()
                        for m in (markets or {}).values()
                        if isinstance(m, dict) and (m.get("active", True) is not False)
                    }
                    validated_symbols = [s for s in preferred_symbols if s in active_market_ids]
                    skipped = [s for s in preferred_symbols if s not in active_market_ids]
                    if skipped:
                        logger.info(f"ℹ️ Skipping unavailable futures symbols: {', '.join(skipped)}")
                except Exception as market_load_err:
                    logger.warning(f"⚠️ Could not validate fallback symbols via load_markets: {market_load_err}")

                for symbol in validated_symbols:
                    if self._shutdown_requested:
                        logger.info("🛑 Shutdown requested during REST fallback fetch - stopping")
                        break
                    try:
                        ticker = await self.rate_limited_api_call(
                            self.binance.fetch_ticker, symbol
                        )
                        if ticker:
                            tickers[symbol] = ticker
                        # Small delay between individual calls (within rate limit)
                        await asyncio.sleep(1.0)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to fetch {symbol}: {e}")
                        continue

            mark_prices: Dict[str, Dict[str, Any]] = {}
            try:
                mark_prices = await mark_price_task
            except Exception as ws_exc:
                logger.warning("⚠️ Live mark-price enrichment unavailable: %s", ws_exc)
            if tickers and mark_prices:
                merged_count = self._merge_mark_price_snapshot(tickers, mark_prices)
                logger.info(
                    "📡 Enriched %d/%d REST tickers with live mark price/funding",
                    merged_count,
                    len(tickers),
                )

            if not tickers:
                logger.error("❌ No tickers received from Binance API")
                return []

            logger.info(f"📊 Retrieved {len(tickers)} tickers from Binance")

            # Filter out very new listings and inactive symbols that usually fail TA history checks.
            eligible_symbols = await self._get_eligible_usdt_symbols(min_listing_days=35)
            if eligible_symbols:
                logger.info(
                    f"🎯 Applying symbol eligibility filter: {len(eligible_symbols)} markets with sufficient listing history"
                )

            # Calculate opportunity scores for all USDT pairs
            opportunities = []
            processed = 0

            # Log first few symbols to understand format
            if tickers:
                sample_symbols = list(tickers.keys())[:5]
                logger.debug(f"🔍 Sample symbols: {sample_symbols}")
            else:
                logger.warning("⚠️ No tickers available for sampling")

            for symbol, ticker in tickers.items():
                # Filter for USDT pairs only - handle both spot (BTC/USDT) and futures (BTC/USDT:USDT) formats
                if not (symbol.endswith('USDT') or '/USDT:USDT' in symbol):
                    continue

                normalized_symbol = self._normalize_symbol_to_market_id(symbol)
                if eligible_symbols and normalized_symbol not in eligible_symbols:
                    continue

                try:
                    score = await self._calculate_opportunity_score(symbol, ticker)
                    if score:
                        opportunities.append((symbol, score))
                        processed += 1

                except Exception as e:
                    continue  # Skip failed tokens

            # Sort by opportunity score (highest first)
            opportunities.sort(key=lambda x: x[1].overall_score, reverse=True)

            # Adaptive quality floor: avoid sending low-quality names into expensive analysis.
            floor_base = 0.45
            dynamic_floor = floor_base
            if len(opportunities) >= 10:
                score_values = [s.overall_score for _, s in opportunities]
                percentile_65 = float(np.percentile(score_values, 65))
                dynamic_floor = max(floor_base, min(0.75, percentile_65 * 0.95))
            filtered_opportunities = [(sym, sc) for sym, sc in opportunities if sc.overall_score >= dynamic_floor]
            if not filtered_opportunities:
                filtered_opportunities = opportunities[:max_opportunities]
            logger.info(
                f"🎚️ Opportunity quality floor applied: {dynamic_floor:.3f} "
                f"({len(filtered_opportunities)}/{len(opportunities)} passed)"
            )

            # L2 orderbook depth gate: only on top 2x max_opportunities candidates to limit API cost.
            # Require the configured minimum depth on both sides (slippage protection).
            depth_check_pool = filtered_opportunities[: max_opportunities * 2]
            depth_passed: List[Tuple[str, OpportunityScore]] = []
            depth_rejected = 0
            depth_min_side_usdt = max(float(getattr(settings, "DISCOVERY_DEPTH_MIN_SIDE_USDT", 100_000.0) or 0.0), 0.0)
            depth_max_spread_pct = max(float(getattr(settings, "DISCOVERY_DEPTH_MAX_SPREAD_PCT", 0.25) or 0.0), 0.0)
            for sym, sc in depth_check_pool:
                ob = await self._fetch_orderbook_depth(sym, levels=20)
                if ob is None:
                    # Depth fetch failed → keep candidate (don't penalize for transient API error)
                    depth_passed.append((sym, sc))
                    continue
                bid_depth = float(ob.get("bid_depth_usdt") or 0.0)
                ask_depth = float(ob.get("ask_depth_usdt") or 0.0)
                spread_pct = float(ob.get("spread_pct") or 0.0)
                # Configured L2 liquidity/spread guardrails.
                if not self._passes_discovery_depth_gate(
                    bid_depth,
                    ask_depth,
                    spread_pct,
                    depth_min_side_usdt,
                    depth_max_spread_pct,
                ):
                    depth_rejected += 1
                    logger.debug(
                        f"🚫 Depth gate rejected {sym}: "
                        f"bid=${bid_depth:,.0f} ask=${ask_depth:,.0f} spread={spread_pct:.3f}%"
                    )
                    continue
                depth_passed.append((sym, sc))
            if depth_rejected:
                logger.info(
                    f"📐 L2 depth gate rejected {depth_rejected}/{len(depth_check_pool)} top candidates "
                    f"(min_side=${depth_min_side_usdt:,.0f}, max_spread={depth_max_spread_pct:.3f}%)"
                )
            # Fallback: if every top candidate fails depth, keep original order to avoid empty discovery.
            if not depth_passed:
                depth_passed = filtered_opportunities[:max_opportunities]
            filtered_opportunities = depth_passed

            # Preserve descending opportunity-score order after execution-quality
            # checks. Repeated or similarly moving tokens remain eligible when
            # they are the strongest current opportunities.

            # Cache every scored market, not only the newly selected top set.
            # Pending-entry revalidation deliberately revisits an older signal
            # even when it has fallen out of today's top 25; using its real
            # current score avoids the old flat 0.7 fallback bias.
            self._opportunity_score_cache = {}
            for symbol, score in opportunities:
                self._opportunity_score_cache[symbol] = score
                self._opportunity_score_cache[self._normalize_symbol_to_market_id(symbol)] = score

            # Log top opportunities
            logger.info(f"✅ Processed {processed} tokens for opportunity scoring")
            logger.debug("🏆 Top opportunities:")
            for i, (symbol, score) in enumerate(filtered_opportunities[:5]):
                logger.debug(f"  {i+1}. {symbol}: {score.overall_score:.3f}")

            # Return top symbols
            return [symbol for symbol, score in filtered_opportunities[:max_opportunities]]

        except Exception as e:
            logger.error(f"❌ Market discovery failed: {e}")
            return []

    @staticmethod
    def _fresh_momentum_base(price_change_pct: float) -> float:
        """Score early expansion while decaying moves that are already extended."""
        move_pct = abs(float(price_change_pct or 0.0))
        if move_pct <= 6.0:
            return move_pct / 6.0
        if move_pct <= 10.0:
            return 1.0 - 0.50 * ((move_pct - 6.0) / 4.0)
        if move_pct <= 20.0:
            return 0.50 - 0.40 * ((move_pct - 10.0) / 10.0)
        return 0.05

    async def _calculate_opportunity_score(self, symbol: str, ticker: Dict) -> Optional[OpportunityScore]:
        """
        Calculate comprehensive opportunity score for a token.

        Based on proven trading_scanner logic but self-contained.
        """
        try:
            # Extract basic data with None checking
            quote_volume = ticker.get('quoteVolume')
            percentage = ticker.get('percentage')
            high_price = ticker.get('high')
            low_price = ticker.get('low')
            last_price = ticker.get('last')

            # Skip if any essential data is missing
            if any(x is None for x in [quote_volume, percentage, high_price, low_price, last_price]):
                return None

            volume_usdt = float(quote_volume)
            price_change_pct_raw = float(percentage)  # Keep sign for direction
            price_change_pct = abs(price_change_pct_raw)  # Magnitude for scoring
            high = float(high_price)
            low = float(low_price)
            current_price = float(last_price)
            min_volume_usdt = max(float(getattr(settings, "OPPORTUNITY_MIN_VOLUME_USDT", 20_000_000.0) or 0.0), 0.0)

            # Quality filters with configurable liquidity floor.
            if (volume_usdt < min_volume_usdt or
                current_price <= 0 or
                price_change_pct > 75.0):  # Allow high-volatility momentum up to 75% for Short fades
                return None

            # Calculate volatility
            if current_price > 0:
                volatility = (high - low) / current_price
                if volatility < 0.005:  # Skip stablecoins
                    return None
            else:
                return None

            # Rank by tradable move quality after liquidity eligibility/depth checks.
            # Volume remains a quality input, but should not let quiet mega-caps
            # outrank cleaner, larger directional opportunities.
            weights = {
                'volume': 0.10,
                'volatility': 0.35,
                'momentum': 0.30,
                'trend': 0.15,
                'institutional': 0.10,
            }

            range_pct = max(volatility * 100.0, 0.01)
            direction_sign = 1 if price_change_pct_raw >= 0 else -1

            position_in_range = 0.5
            if high != low:
                position_in_range = max(0.0, min(1.0, (current_price - low) / (high - low)))

            # 1) Volume score: log-normalized so very large caps do not dominate linearly.
            log_volume = np.log10(max(volume_usdt, 1.0))
            volume_score = float(max(0.0, min(1.0, (log_volume - 6.7) / 1.6)))  # ~5M..200M maps to 0..1

            # 2) Asset-relative volatility score using directional efficiency vs total range.
            directional_efficiency = min(1.2, price_change_pct / max(range_pct, 0.5))
            if range_pct < 2.0:
                volatility_score = 0.20  # Too quiet / likely noise
            elif range_pct <= 18.0:
                base_vol = 0.45 + 0.45 * min(1.0, (range_pct - 2.0) / 16.0)
                volatility_score = base_vol * (0.65 + 0.35 * min(1.0, directional_efficiency))
            else:
                volatility_score = max(0.12, 0.95 - (range_pct - 18.0) / 40.0)
            volatility_score = float(max(0.0, min(1.0, volatility_score)))

            # 3) Direction-aware fresh-momentum score.
            # Reward early expansion (+3% to +10%) for continuation, but ALSO
            # reward extended moves for mean-reversion setups (large pumps >=12% for
            # Short fades, large dumps <=-12% for Long capitulation bounces).
            is_extended_pump = price_change_pct_raw >= 12.0 or (price_change_pct_raw >= 8.0 and position_in_range >= 0.75)
            is_extended_dump = price_change_pct_raw <= -12.0 or (price_change_pct_raw <= -8.0 and position_in_range <= 0.25)

            if is_extended_pump or is_extended_dump:
                # Extended moves offer prime mean-reversion opportunity (Short fade or Long bounce)
                momentum_base = float(min(0.95, 0.65 + 0.30 * min(1.0, (price_change_pct - 12.0) / 38.0)))
            else:
                # Standard early/mid expansion
                momentum_base = self._fresh_momentum_base(price_change_pct)

            momentum_quality = min(1.0, directional_efficiency)
            low_liquidity_penalty = 0.55 if (price_change_pct > 18.0 and volume_usdt < 25_000_000) else 1.0
            momentum_score = float(max(0.0, min(1.0, momentum_base * (0.45 + 0.55 * momentum_quality) * low_liquidity_penalty)))

            # 4) Direction-NEUTRAL setup score — measures structural interest only.
            # Stage 1 is purely an opportunity filter: "Is this at an interesting location?"
            # We do NOT pre-assign direction. A token near its daily high could be a breakout
            # OR a distribution top. A token near its daily low could be a bounce OR a breakdown.
            # The AI (Stage 2) with full order flow, EMAs, and funding data makes that call.
            near_high = position_in_range >= 0.85
            near_low  = position_in_range <= 0.15
            if near_high or near_low:
                # Extreme of range: highest structural interest, could go either way
                setup_score = 0.82
            elif 0.70 <= position_in_range < 0.85 or 0.15 < position_in_range <= 0.30:
                # Near extreme: still very interesting setup zone
                setup_score = 0.72
            elif 0.40 <= position_in_range <= 0.60:
                # Mid-range: least interesting, fewer clear levels nearby
                setup_score = 0.50
            else:
                # Transitioning zones
                setup_score = 0.62
            # Boost by raw volume — more liquid setups are higher quality regardless of direction
            setup_score = float(max(0.0, min(1.0, setup_score + min(0.10, volume_usdt / 50_000_000))))

            # 5) Institutional/liquidity proxy with depth-per-volatility instead of price denominator.
            notional_per_vol = volume_usdt / max(range_pct, 1.0)
            depth_score = min(1.0, notional_per_vol / 8_000_000)
            stability_score = max(0.0, 1.0 - min(1.0, abs(price_change_pct_raw) / 25.0))
            institutional_score = float(max(0.0, min(1.0, 0.45 * volume_score + 0.40 * depth_score + 0.15 * stability_score)))

            # 6. Market Regime Awareness (adjust scores based on overall market conditions)
            market_regime_multiplier = 1.0

            # Detect market regime based on price action and volatility
            if price_change_pct_raw > 15.0 and volatility > 0.15 and position_in_range < 0.70:
                # High volatility rally in mid-range - reduce momentum weight, increase caution
                market_regime_multiplier = 0.85
                momentum_score *= 0.8  # Reduce momentum in overheated conditions
            elif price_change_pct_raw < -10.0 and volatility > 0.20:
                # High volatility selloff - favor quality setups over momentum
                market_regime_multiplier = 0.90
                setup_score *= 1.1  # Increase setup importance in volatile markets
                institutional_score *= 1.1  # Favor institutional backing in fear
            elif 2.0 <= price_change_pct_raw <= 8.0 and volatility <= 0.10:
                # Steady uptrend - optimal conditions
                market_regime_multiplier = 1.1
            elif -5.0 <= price_change_pct_raw <= 2.0 and volatility <= 0.08:
                # Consolidation/low volatility - favor breakout setups
                market_regime_multiplier = 1.05
                volume_score *= 1.1  # Volume more important in quiet markets

            # Calculate weighted overall score
            overall_score = (
                weights['volume'] * volume_score +
                weights['volatility'] * volatility_score +
                weights['momentum'] * momentum_score +
                weights['trend'] * setup_score +  # Renamed for clarity
                weights['institutional'] * institutional_score
            )

            # Apply market regime multiplier
            overall_score *= market_regime_multiplier

            # Ensure score is 0-1
            overall_score = max(0.0, min(1.0, overall_score))

            return OpportunityScore(
                symbol=symbol,
                overall_score=overall_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                trend_score=setup_score,  # Now represents setup quality
                institutional_score=institutional_score,
                breakdown={
                    'volume': volume_score,
                    'volatility': volatility_score,
                    'momentum': momentum_score,
                    'setup_quality': setup_score,  # Setup quality at key levels
                    'institutional': institutional_score,
                    'price_direction': 'bullish' if price_change_pct_raw > 0 else 'bearish',
                    'price_change_24h': price_change_pct_raw,
                    'position_in_range': position_in_range,
                    'setup_type': (
                        'short_setup' if position_in_range >= 0.8 and price_change_pct_raw < 0 else
                        'long_setup' if position_in_range <= 0.2 and price_change_pct_raw > 0 else
                        'fade_risk' if position_in_range >= 0.8 and price_change_pct_raw >= 0 else
                        'fade_risk' if position_in_range <= 0.2 and price_change_pct_raw <= 0 else
                        'neutral'
                    ),
                    'range_pct': range_pct,
                    'directional_efficiency': directional_efficiency,
                }
            )

        except Exception as e:
            logger.error(f"Error calculating opportunity score for {symbol}: {e}")
            return None

    async def prefetch_ohlcv_batch(self, symbols: List[str]) -> None:
        """
        Batch-prefetch OHLCV data for all candidate symbols before signal generation.

        Fetches 1d, 4h, 1h candles for every symbol with controlled pacing,
        populating self._ohlcv_cache so that perform_technical_analysis() can
        read from cache instead of hitting the REST API per-symbol.
        """
        self._ohlcv_cache.clear()
        timeframes = [
            ('1d', 90),   # 90 days of daily data
            ('4h', 200),  # ~33 days of 4h data
            ('1h', 168),  # 1 week of hourly data
        ]
        total = len(symbols) * len(timeframes)
        fetched = 0
        logger.info(f"📦 Prefetching OHLCV data for {len(symbols)} symbols ({total} API calls)...")

        for symbol in symbols:
            for interval, limit in timeframes:
                cache_key = f"{symbol}:{interval}"
                try:
                    ohlcv = await self.rate_limited_api_call(
                        self.binance.fetch_ohlcv, symbol, interval, limit=limit
                    )
                    self._ohlcv_cache[cache_key] = ohlcv if ohlcv else []
                    fetched += 1
                    if fetched % 15 == 0:
                        logger.debug(f"📦 Prefetch progress: {fetched}/{total} calls done")
                except Exception as e:
                    logger.warning(f"⚠️ Prefetch failed for {cache_key}: {e}")
                    self._ohlcv_cache[cache_key] = []

        hit = sum(1 for v in self._ohlcv_cache.values() if v)
        logger.info(f"✅ OHLCV prefetch complete: {hit}/{total} successful")

    async def fetch_ohlcv_websocket(self, symbol: str, interval: str, limit: int = 100) -> List[List]:
        """
        Fetch OHLCV data — checks in-memory cache first, then falls back to REST API.
        """
        cache_key = f"{symbol}:{interval}"

        # Return cached data if available (populated by prefetch_ohlcv_batch)
        if cache_key in self._ohlcv_cache:
            cached = self._ohlcv_cache[cache_key]
            logger.debug(f"📦 Cache hit for {cache_key}: {len(cached)} candles")
            return cached

        logger.debug(f"📡 Fetching {interval} OHLCV for {symbol} (cache miss)")

        try:
            ohlcv = await self.rate_limited_api_call(
                self.binance.fetch_ohlcv, symbol, interval, limit=limit
            )
            result = ohlcv if ohlcv else []
            self._ohlcv_cache[cache_key] = result
            logger.debug(f"✅ Retrieved {len(result)} {interval} candles for {symbol}")
            return result

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {interval} OHLCV for {symbol}: {e}")
            return []

    async def perform_technical_analysis(self, symbol: str) -> Optional[TechnicalAnalysis]:
        """
        Perform comprehensive technical analysis with WebSocket OHLCV data.

        Enhanced with:
        - WebSocket-based OHLCV fetching (reduced weight consumption)
        - Multi-timeframe analysis (1h, 4h)
        - Advanced oscillators (Stochastic, Williams %R, CCI)
        - Volume analysis (OBV, Volume Profile, VWAP)
        - Market microstructure indicators
        - Fibonacci retracement levels
        - Support/Resistance detection
        """
        logger.debug(f"📊 Performing technical analysis for {symbol}")

        try:
            # Ensure services are initialized
            if not self._services_initialized:
                await self._initialize_services()

            # Get MAXIMUM multi-timeframe data for highest quality analysis
            # Fetch all required timeframes efficiently (memory optimized)
            ohlcv_1d = await self.fetch_ohlcv_websocket(symbol, '1d', limit=120)   # 120 days of daily data
            
            # No extra sleep needed — data is either cached (prefetch) or paced by ccxt rate limiter
            
            ohlcv_4h = await self.fetch_ohlcv_websocket(symbol, '4h', limit=300)  # 300 periods of 4h data
            ohlcv_1h = await self.fetch_ohlcv_websocket(symbol, '1h', limit=240)   # 240 hours of hourly data

            # Convert to pandas for comprehensive multi-timeframe analysis
            df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Validate comprehensive data quality (more data = better analysis)
            if len(df_1d) < 30 or len(df_4h) < 50 or len(df_1h) < 24:
                logger.warning(f"⚠️ Insufficient data for {symbol}: 1d={len(df_1d)}, 4h={len(df_4h)}, 1h={len(df_1h)}")
                return None

            # Current values
            current_price = float(df_4h['close'].iloc[-1])
            current_volume = float(df_4h['volume'].iloc[-1])

            # 24h metrics - safe indexing
            hours_available = min(24, len(df_1h))
            price_24h_ago = float(df_1h['close'].iloc[-hours_available]) if hours_available > 0 else current_price
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100 if price_24h_ago > 0 else 0.0
            volume_24h = float(df_1h['volume'].tail(hours_available).sum())
            high_24h = float(df_1h['high'].tail(hours_available).max())
            low_24h = float(df_1h['low'].tail(hours_available).min())

            # Technical Indicators

            # RSI (14 period)
            rsi_14 = self._calculate_rsi(df_4h['close'], 14)

            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df_4h['close'])

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df_4h['close'])
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

            # Volume analysis
            volume_sma_10 = float(df_4h['volume'].tail(10).mean())
            volume_ratio = current_volume / volume_sma_10 if volume_sma_10 > 0 else 1.0

            # ATR (volatility)
            atr_14 = self._calculate_atr(df_4h, 14)
            volatility_24h = ((high_24h - low_24h) / current_price) * 100

            # Advanced Oscillators
            stoch_k, stoch_d = self._calculate_stochastic(df_4h)
            williams_r = self._calculate_williams_r(df_4h, 14)
            cci_14 = self._calculate_cci(df_4h, 14)
            roc_10 = self._calculate_roc(df_4h['close'], 10)

            # Volume Analysis Indicators
            obv = self._calculate_obv(df_4h)
            vwap = self._calculate_vwap(df_4h)
            volume_profile_poc = self._calculate_volume_profile_poc(df_4h)
            money_flow_index = self._calculate_mfi(df_4h, 14)

            # Market Microstructure
            spread_estimate = (high_24h - low_24h) / current_price * 100
            tick_rule_momentum = self._calculate_tick_rule_momentum(df_1h)
            price_efficiency = self._calculate_price_efficiency(df_4h)

            # Multi-timeframe Analysis
            rsi_1h = self._calculate_rsi(df_1h['close'], 14)
            macd_1h_line, macd_1h_signal, _ = self._calculate_macd(df_1h['close'])
            trend_alignment = self._analyze_trend_alignment(df_1h, df_4h)

            # Advanced Support/Resistance with Fibonacci
            fib_levels = self._calculate_fibonacci_levels(df_4h)
            dynamic_sr = self._calculate_dynamic_support_resistance(df_4h)
            support_level = dynamic_sr['support']
            resistance_level = dynamic_sr['resistance']

            # EMAs
            ema_20 = self._calculate_ema(df_4h['close'], 20)
            ema_50 = self._calculate_ema(df_4h['close'], 50)
            # Use 1h for longer EMA only if we have enough data, otherwise fallback to 4h
            if len(df_1h) >= 200:
                ema_200 = self._calculate_ema(df_1h['close'], 200)
            else:
                # Fallback: use 4h data with shorter period
                ema_200 = self._calculate_ema(df_4h['close'], min(50, len(df_4h) - 1))

            # Trend analysis
            if current_price > ema_20 > ema_50:
                trend_direction = 'bullish'
            elif current_price < ema_20 < ema_50:
                trend_direction = 'bearish'
            else:
                trend_direction = 'sideways'

            # Direction-neutral momentum quality. Direction is determined by the
            # signed trend/MACD setup; this score measures whether that move has
            # clean participation without rewarding one token price scale.
            if 35 <= rsi_14 <= 65:
                rsi_momentum = 1.0
            elif rsi_14 < 35:
                rsi_momentum = max(0.0, rsi_14 / 35.0)
            else:
                rsi_momentum = max(0.0, (100.0 - rsi_14) / 35.0)

            # Normalize MACD histogram by price before comparing assets.
            macd_histogram_pct = abs(macd_histogram) / max(current_price, 1e-12) * 100.0
            macd_momentum = float(np.tanh(macd_histogram_pct * 3.0))

            # Volume momentum (optimal at 2x average volume)
            volume_momentum = min(1.0, volume_ratio / 2.0)

            momentum_score = (rsi_momentum + macd_momentum + volume_momentum) / 3

            strength_score = self._calculate_strength_score(bb_position, trend_direction, price_change_24h)
            # Advanced AI Metrics
            from kata.services.binance_service import BinanceService
            formatted_symbol = symbol.replace('/', '').replace(':', '')
            advanced_metrics = await BinanceService.get_advanced_ai_metrics(formatted_symbol)
            taker_ratio = advanced_metrics.get('taker_buy_sell_ratio')
            liq_vol = advanced_metrics.get('liquidation_volume')
            basis = advanced_metrics.get('basis_premium')

            structure_candles = [
                {
                    "timestamp": int(row.timestamp),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
                for row in df_4h.tail(160).itertuples(index=False)
            ]

            return TechnicalAnalysis(
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                rsi_14=rsi_14,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_position=bb_position,
                volume_sma_10=volume_sma_10,
                volume_ratio=volume_ratio,
                atr_14=atr_14,
                volatility_24h=volatility_24h,
                support_level=support_level,
                resistance_level=resistance_level,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_200=ema_200,
                trend_direction=trend_direction,
                momentum_score=momentum_score,
                strength_score=strength_score,
                # Advanced Oscillators
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                williams_r=williams_r,
                cci_14=cci_14,
                roc_10=roc_10,
                # Volume Analysis
                obv=obv,
                vwap=vwap,
                volume_profile_poc=volume_profile_poc,
                money_flow_index=money_flow_index,
                # Market Microstructure
                spread_estimate=spread_estimate,
                tick_rule_momentum=tick_rule_momentum,
                price_efficiency=price_efficiency,
                # Multi-timeframe
                rsi_1h=rsi_1h,
                macd_1h_line=macd_1h_line,
                macd_1h_signal=macd_1h_signal,
                trend_alignment=trend_alignment,
                # Fibonacci levels
                fib_23_6=fib_levels['23_6'],
                fib_38_2=fib_levels['38_2'],
                fib_50_0=fib_levels['50_0'],
                fib_61_8=fib_levels['61_8'],
                # Dynamic S/R
                dynamic_support=dynamic_sr['support'],
                dynamic_resistance=dynamic_sr['resistance'],
                structure_candles=structure_candles,
                # Advanced AI Metrics
                taker_buy_sell_ratio=taker_ratio,
                liquidation_volume=liq_vol,
                basis_premium=basis,
            )

        except Exception as e:
            logger.error(f"❌ Technical analysis failed for {symbol}: {e}")
            return None

    # Technical Indicator Calculations

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator with safe bounds checking."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal

        return (
            float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0
        )

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return (
            float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else 0.0,
            float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else 0.0,
            float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else 0.0
        )

    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average."""
        ema = prices.ewm(span=period).mean()
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else 0.0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0


    def _calculate_strength_score(self, bb_position: float, trend: str, price_change: float) -> float:
        """Calculate strength score (0-1)."""
        # Bollinger band position
        bb_score = 1.0 - abs(bb_position - 0.5) * 2  # Best in middle

        # Trend component
        trend_score = 0.8 if trend in ['bullish', 'bearish'] else 0.5

        # Early expansion is stronger than a trailing move that is already extended.
        momentum_score = self._fresh_momentum_base(price_change)

        return (bb_score + trend_score + momentum_score) / 3

    # Advanced Technical Indicator Calculations

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        try:
            high_k = df['high'].rolling(window=k_period).max()
            low_k = df['low'].rolling(window=k_period).min()
            k_percent = 100 * ((df['close'] - low_k) / (high_k - low_k))
            k_percent = k_percent.fillna(50)
            d_percent = k_percent.rolling(window=d_period).mean()
            return float(k_percent.iloc[-1]), float(d_percent.iloc[-1])
        except Exception as e:
            logger.debug(f"Stochastic fallback used: {e}")
            return 50.0, 50.0

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R."""
        try:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
            return float(williams_r.fillna(-50).iloc[-1])
        except Exception as e:
            logger.debug(f"Williams %R fallback used: {e}")
            return -50.0

    def _calculate_cci(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Commodity Channel Index."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            ma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (typical_price - ma) / (0.015 * mad)
            return float(cci.fillna(0).iloc[-1])
        except Exception as e:
            logger.debug(f"CCI fallback used: {e}")
            return 0.0

    def _calculate_roc(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate Rate of Change."""
        try:
            roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
            return float(roc.fillna(0).iloc[-1])
        except Exception as e:
            logger.debug(f"ROC fallback used: {e}")
            return 0.0

    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """Calculate On Balance Volume."""
        try:
            obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()
            return float(obv[-1]) if len(obv) > 0 else 0.0
        except Exception as e:
            logger.debug(f"OBV fallback used: {e}")
            return 0.0

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return float(vwap.iloc[-1])
        except Exception as e:
            logger.debug(f"VWAP fallback used: {e}")
            return float(df['close'].iloc[-1]) if len(df) > 0 else 0.0

    def _calculate_volume_profile_poc(self, df: pd.DataFrame) -> float:
        """Calculate Volume Profile Point of Control."""
        try:
            # Simplified POC - price level with highest volume
            price_bins = pd.cut(df['close'], bins=20, retbins=True)[1]
            volume_by_price = df.groupby(pd.cut(df['close'], bins=20), observed=False)['volume'].sum()
            poc_bin = volume_by_price.idxmax()
            return float((poc_bin.left + poc_bin.right) / 2)
        except Exception as e:
            logger.debug(f"Volume profile POC fallback used: {e}")
            return float(df['close'].iloc[-1]) if len(df) > 0 else 0.0

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']

            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

            mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1e-10)))
            return float(mfi.fillna(50).iloc[-1])
        except Exception as e:
            logger.debug(f"MFI fallback used: {e}")
            return 50.0

    def _calculate_tick_rule_momentum(self, df: pd.DataFrame) -> float:
        """Calculate tick rule momentum (uptick/downtick ratio)."""
        try:
            price_changes = df['close'].diff()
            upticks = (price_changes > 0).sum()
            downticks = (price_changes < 0).sum()
            if downticks == 0:
                return 1.0
            return float(upticks / downticks)
        except Exception as e:
            logger.debug(f"Tick rule momentum fallback used: {e}")
            return 0.5

    def _calculate_price_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate price efficiency (net price change / total price movement)."""
        try:
            net_change = abs(df['close'].iloc[-1] - df['close'].iloc[0])
            total_movement = df['close'].diff().abs().sum()
            if total_movement == 0:
                return 1.0
            return float(net_change / total_movement)
        except Exception as e:
            logger.debug(f"Price efficiency fallback used: {e}")
            return 0.5

    def _analyze_trend_alignment(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> float:
        """Analyze multi-timeframe trend alignment."""
        try:
            # 1h trend
            ema_1h_20 = self._calculate_ema(df_1h['close'], 20)
            ema_1h_50 = self._calculate_ema(df_1h['close'], 50)
            trend_1h = 1 if df_1h['close'].iloc[-1] > ema_1h_20 > ema_1h_50 else -1 if df_1h['close'].iloc[-1] < ema_1h_20 < ema_1h_50 else 0

            # 4h trend
            ema_4h_20 = self._calculate_ema(df_4h['close'], 20)
            ema_4h_50 = self._calculate_ema(df_4h['close'], 50)
            trend_4h = 1 if df_4h['close'].iloc[-1] > ema_4h_20 > ema_4h_50 else -1 if df_4h['close'].iloc[-1] < ema_4h_20 < ema_4h_50 else 0

            # Alignment score (-1 to 1)
            alignment = (trend_1h + trend_4h) / 2
            return float(alignment)
        except:
            return 0.0

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        try:
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low

            return {
                '23_6': float(high - diff * 0.236),
                '38_2': float(high - diff * 0.382),
                '50_0': float(high - diff * 0.5),
                '61_8': float(high - diff * 0.618)
            }
        except:
            price = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
            return {'23_6': price, '38_2': price, '50_0': price, '61_8': price}

    def _calculate_dynamic_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels using pivot points and EMA."""
        try:
            # Recent pivot highs and lows
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()

            # Filter actual pivot points
            pivot_highs = df[df['high'] == highs]['high'].tail(3)
            pivot_lows = df[df['low'] == lows]['low'].tail(3)

            # Dynamic resistance (average of recent pivot highs)
            resistance = float(pivot_highs.mean()) if not pivot_highs.empty else float(df['high'].tail(20).max())

            # Dynamic support (average of recent pivot lows)
            support = float(pivot_lows.mean()) if not pivot_lows.empty else float(df['low'].tail(20).min())

            return {'support': support, 'resistance': resistance}
        except:
            price = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
            return {'support': price * 0.95, 'resistance': price * 1.05}

    def _validate_ai_schema(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate AI JSON against expected schema and ranges."""
        required = ['recommendation', 'confidence', 'reasoning']
        for f in required:
            if f not in data:
                return False, f"missing_field:{f}"
        # Types and ranges
        try:
            rec = str(data['recommendation']).upper()
            if rec not in ['LONG', 'SHORT', 'HOLD']:
                return False, 'invalid_recommendation'
            conf = float(data['confidence'])
            if not (0.0 <= conf <= 1.0):
                return False, 'invalid_confidence_range'

            # For LONG/SHORT, require valid numeric trade fields.
            if rec in ['LONG', 'SHORT']:
                if 'entry_price' not in data or 'stop_loss' not in data:
                    return False, 'missing_required_prices'
                ep = clean_price_string(data['entry_price'])
                sl = clean_price_string(data['stop_loss'])
                if ep <= 0 or sl <= 0:
                    return False, 'non_positive_prices'

                # Optional with sane defaults (directional only).
                if 'target_1' in data and data['target_1'] is not None:
                    clean_price_string(data['target_1'])
                if 'target_2' in data and data['target_2'] is not None:
                    clean_price_string(data['target_2'])
                if 'risk_reward_ratio' in data and data['risk_reward_ratio'] is not None:
                    rr = clean_price_string(data['risk_reward_ratio'])
                    if rr <= 0 or rr > 10:
                        return False, 'invalid_rr_range'
            else:
                # HOLD can include placeholders like N/A/0 for numeric fields.
                pass
        except Exception as e:
            logger.warning(f"Schema validation type error details: {e}")
            logger.warning(f"Data causing error: {data}")
            return False, f'type_error:{str(e)}'
        return True, 'ok'

    def _target_distance_constraints(
        self,
        time_horizon: str,
        atr_pct: float,
        volatility_24h: float,
        confidence: float,
    ) -> Dict[str, float]:
        """Bound target/stop distances to what the horizon and volatility can support."""
        horizon_hours = self._extract_range_upper_hours(time_horizon)
        if horizon_hours is None:
            horizon_hours = self._calculate_validity_window_hours(time_horizon or "default")

        if horizon_hours <= 12:
            base_min, base_max, atr_mult = 0.70, 3.00, 1.20
        elif horizon_hours <= 24:
            base_min, base_max, atr_mult = 0.90, 4.20, 1.50
        elif horizon_hours <= 72:
            base_min, base_max, atr_mult = 1.10, 6.50, 2.00
        elif horizon_hours <= 168:
            base_min, base_max, atr_mult = 1.30, 8.50, 2.35
        else:
            base_min, base_max, atr_mult = 1.50, 10.00, 2.70

        atr_pct = max(0.0, min(25.0, float(atr_pct or 0.0)))
        volatility_pct = max(0.0, min(60.0, float(volatility_24h or 0.0)))
        conviction = max(0.0, min(1.0, float(confidence or 0.0)))

        vol_allowance = min(1.20, volatility_pct * 0.05)
        confidence_allowance = min(0.50, max(0.0, (conviction - 0.72) * 1.25))
        atr_cap = (atr_pct * atr_mult + vol_allowance) if atr_pct > 0.05 else base_max * 0.75
        max_t1_move_pct = max(base_min + 0.20, min(base_max, atr_cap + confidence_allowance))
        min_t1_move_pct = min(
            max_t1_move_pct,
            max(base_min, min(base_max, atr_pct * 0.45 if atr_pct > 0 else base_min)),
        )

        min_sl_move_pct = max(
            0.55,
            min(
                max_t1_move_pct / 1.25,
                max(0.75, atr_pct * 0.45, volatility_pct * 0.025),
            ),
        )
        max_sl_move_pct = max(min_sl_move_pct, max_t1_move_pct / 1.18)

        return {
            "horizon_hours": float(horizon_hours),
            "min_t1_move_pct": float(min_t1_move_pct),
            "max_t1_move_pct": float(max_t1_move_pct),
            "min_sl_move_pct": float(min_sl_move_pct),
            "max_sl_move_pct": float(max_sl_move_pct),
        }

    # Hard risk rails. Targets may be proposed by the thesis model, but the
    # invalidation level is calculated deterministically from structure + ATR.
    _T1_MAX_SANITY_PCT = 25.0   # a target further than this is degenerate output
    _SL_MIN_SANITY_PCT = 0.30   # a stop tighter than this is spread/noise, not invalidation
    _SL_MAX_SANITY_PCT = 15.0   # matches the entry-adaptation rejection ceiling

    def _adapt_risk_targets(self, decision: 'AIDecision', ta: TechnicalAnalysis) -> None:
        """
        Convert the narrative thesis into auditable execution geometry:
        - replaces missing/non-directional levels with ATR-derived fallbacks,
        - calculates the stop deterministically from nearby structure + ATR,
        - enforces wide hard sanity rails (degenerate output protection),
        - applies a confidence penalty + risk flag when target_1 sits beyond the
          learned horizon/ATR band (expiry risk) — WITHOUT moving the level,
        - recomputes risk/reward from the final levels so the RR gate filters
          on the final geometry.

        The thesis model can critique the stop in its response, but it cannot
        numerically choose risk. This keeps stop policy stable between prompts.
        """
        try:
            ep = float(decision.entry_price or 0.0)
            direction = str(decision.recommendation or "").upper().strip()
            if ep <= 0 or direction not in {"LONG", "SHORT"}:
                return

            base_conviction = (
                getattr(decision, "raw_ai_confidence", None)
                if getattr(decision, "raw_ai_confidence", None) is not None
                else decision.confidence
            )
            c = max(0.0, min(1.0, float(base_conviction or 0.0)))
            original_horizon = str(decision.time_horizon or "4h-12h").strip() or "4h-12h"
            atr = max(float(ta.atr_14 or 0.0), 1e-8)
            atr_pct = (atr / ep) * 100.0
            distance_constraints = self._target_distance_constraints(
                original_horizon,
                atr_pct,
                float(ta.volatility_24h or 0.0),
                c,
            )
            horizon_hours = float(distance_constraints["horizon_hours"])
            min_t1_move_pct = float(distance_constraints["min_t1_move_pct"])
            max_t1_move_pct = float(distance_constraints["max_t1_move_pct"])

            def level_is_directional(price: float, role: str) -> bool:
                if price <= 0:
                    return False
                if role == "stop":
                    return price < ep if direction == "LONG" else price > ep
                return price > ep if direction == "LONG" else price < ep

            def distance_pct(price: float) -> float:
                return (abs(float(price or 0.0) - ep) / ep) * 100.0 if ep > 0 and price else 0.0

            ai_t1_pct = distance_pct(decision.target_1) if level_is_directional(float(decision.target_1 or 0.0), "target") else 0.0
            ai_t2_pct = distance_pct(decision.target_2) if level_is_directional(float(decision.target_2 or 0.0), "target") else 0.0
            ai_sl_pct = distance_pct(decision.stop_loss) if level_is_directional(float(decision.stop_loss or 0.0), "stop") else 0.0

            # --- Target 1: AI's level, ATR fallback only when missing/invalid ---
            fallback_t1_pct = max(
                min_t1_move_pct,
                min(max_t1_move_pct, max(0.90, atr_pct * 1.10, float(ta.volatility_24h or 0.0) * 0.10)),
            )
            target_t1_move_pct = ai_t1_pct or fallback_t1_pct
            target_t1_move_pct = min(target_t1_move_pct, self._T1_MAX_SANITY_PCT)

            # --- Target 2: AI's level if beyond T1, else extend T1 distance ---
            target_t2_move_pct = ai_t2_pct if ai_t2_pct > target_t1_move_pct else target_t1_move_pct * 1.45
            target_t2_move_pct = min(target_t2_move_pct, self._T1_MAX_SANITY_PCT * 1.5)

            # --- Stop: deterministic invalidation, decoupled from target/RR ---
            fallback_sl_pct = max(
                self._SL_MIN_SANITY_PCT,
                min(max(0.75, atr_pct * 0.90), self._SL_MAX_SANITY_PCT),
            )
            structure_levels = (
                [ta.support_level, ta.dynamic_support, ta.vwap, ta.volume_profile_poc]
                if direction == "LONG"
                else [ta.resistance_level, ta.dynamic_resistance, ta.vwap, ta.volume_profile_poc]
            )
            if direction == "LONG":
                valid_structure = [
                    float(level) for level in structure_levels
                    if level is not None and 0 < float(level) < ep
                ]
                anchor = max(valid_structure) if valid_structure else None
                structure_stop = (anchor - atr * 0.25) if anchor is not None else None
                structure_sl_pct = (
                    ((ep - structure_stop) / ep) * 100.0
                    if structure_stop is not None and structure_stop < ep
                    else 0.0
                )
            else:
                valid_structure = [
                    float(level) for level in structure_levels
                    if level is not None and float(level) > ep
                ]
                anchor = min(valid_structure) if valid_structure else None
                structure_stop = (anchor + atr * 0.25) if anchor is not None else None
                structure_sl_pct = (
                    ((structure_stop - ep) / ep) * 100.0
                    if structure_stop is not None and structure_stop > ep
                    else 0.0
                )

            noise_floor_pct = max(0.50, atr_pct * 0.75)
            sl_move_pct = max(
                fallback_sl_pct,
                noise_floor_pct,
                structure_sl_pct,
            )
            sl_move_pct = max(self._SL_MIN_SANITY_PCT, min(sl_move_pct, self._SL_MAX_SANITY_PCT))
            if abs(sl_move_pct - ai_sl_pct) >= 0.10:
                risk_flags = list(decision.risk_factors or [])
                risk_flags.append("stop_replaced_by_deterministic_structure_policy")
                decision.risk_factors = list(dict.fromkeys(risk_flags))
            structure_context = dict(decision.market_structure_context or {})
            structure_context["deterministic_risk_geometry"] = {
                "policy": "nearest_structure_plus_0_25_atr_with_0_75_atr_floor",
                "structure_anchor": float(anchor) if anchor is not None else None,
                "atr": float(atr),
                "ai_proposed_stop_distance_pct": float(ai_sl_pct),
                "final_stop_distance_pct": float(sl_move_pct),
            }
            decision.market_structure_context = structure_context

            # Expiry risk: penalize confidence when T1 outruns what the horizon
            # historically supports — but leave the AI's level in place.
            beyond_band = target_t1_move_pct > max_t1_move_pct * 1.001
            if beyond_band:
                expiry_pressure = target_t1_move_pct / max(max_t1_move_pct, 0.01)
                penalty = min(0.08, max(0.0, (expiry_pressure - 1.0) * 0.04))
                if horizon_hours <= 24:
                    penalty = min(0.08, penalty + 0.01)
                if penalty >= 0.005:
                    decision.confidence = max(0.0, float(decision.confidence or 0.0) - penalty)
                    risk_flags = list(decision.risk_factors or [])
                    risk_flags.append("target_beyond_horizon_band_expiry_risk")
                    decision.risk_factors = list(dict.fromkeys(risk_flags))
                    logger.info(
                        f"⚠️ T1 {target_t1_move_pct:.2f}% beyond {original_horizon} band "
                        f"max {max_t1_move_pct:.2f}% — level kept, confidence -{penalty:.3f}"
                    )

            if direction == 'LONG':
                tp1 = ep * (1 + target_t1_move_pct / 100.0)
                tp2 = ep * (1 + target_t2_move_pct / 100.0)
                sl = ep * (1 - sl_move_pct / 100.0)
            else:
                tp1 = ep * (1 - target_t1_move_pct / 100.0)
                tp2 = ep * (1 - target_t2_move_pct / 100.0)
                sl = ep * (1 + sl_move_pct / 100.0)

            risk = abs(ep - sl)
            reward = abs(tp1 - ep)
            rr = (reward / risk) if risk > 0 else decision.risk_reward_ratio

            decision.target_1 = float(tp1)
            decision.target_2 = float(tp2)
            decision.stop_loss = float(sl)
            decision.risk_reward_ratio = float(round(rr, 2))
            decision.time_horizon = original_horizon
            final_c = max(0.0, min(1.0, float(decision.confidence or 0.0)))
            decision.timeframe = self._derive_signal_timeframe(original_horizon, final_c)
        except Exception as e:
            logger.debug(f"Adaptive risk/targets skipped: {e}")

    def _derive_signal_timeframe(self, time_horizon: str, conviction: float) -> str:
        """
        Derive execution timeframe from horizon and conviction.
        Shorter horizons use faster execution frames; broader horizons use slower frames.
        """
        canonical = self._normalise_horizon_key(str(time_horizon or ""))
        c = max(0.0, min(1.0, float(conviction or 0.0)))

        if canonical == "4h-12h":
            return "1h" if c >= 0.70 else "4h"
        if canonical == "4-24h":
            return "4h"
        if canonical in {"1-3d", "3-7d"}:
            return "1d"
        if canonical in {"1-2w", "2-4w", "1-3m"}:
            return "1d"
        return "4h"

    def _extract_token_usage(self, response: Any, provider: str) -> Tuple[int, int]:
        """Extract prompt/completion token usage across providers."""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            usage = getattr(response, 'usage', None)
            if usage is None and isinstance(response, dict):
                usage = response.get('usage')

            if provider in {'openai', 'deepseek'}:
                if usage is not None:
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get('prompt_tokens') or 0)
                        completion_tokens = int(usage.get('completion_tokens') or 0)
                    else:
                        prompt_tokens = int(getattr(usage, 'prompt_tokens', 0) or 0)
                        completion_tokens = int(getattr(usage, 'completion_tokens', 0) or 0)
            else:  # claude/anthropic
                if usage is not None:
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get('input_tokens') or 0)
                        completion_tokens = int(usage.get('output_tokens') or 0)
                    else:
                        prompt_tokens = int(getattr(usage, 'input_tokens', 0) or 0)
                        completion_tokens = int(getattr(usage, 'output_tokens', 0) or 0)
        except Exception:
            return 0, 0

        return prompt_tokens, completion_tokens

    def _estimate_llm_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
    ) -> float:
        """Estimate LLM cost in USD from token counts and configured per-1M rates."""
        try:
            in_cost = (max(prompt_tokens, 0) / 1_000_000.0) * max(input_cost_per_1m, 0.0)
            out_cost = (max(completion_tokens, 0) / 1_000_000.0) * max(output_cost_per_1m, 0.0)
            return round(in_cost + out_cost, 8)
        except Exception:
            return 0.0

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse a JSON object from mixed text."""
        if not text:
            return None

        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None

        raw = text[start:end + 1].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Lightweight cleanup for common LLM formatting artifacts.
            cleaned = raw.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _extract_breakdown_dict(self, raw: Any) -> Dict[str, Any]:
        """Parse ai_confidence_breakdown fields that may be dict or JSON string."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    async def _sync_challenger_month_spend(self, force: bool = False) -> float:
        """Sync current-month challenger spend from persisted signal metadata."""
        if settings.CHALLENGER_MONTHLY_BUDGET_USD <= 0:
            self._challenger_month_spend_usd = 0.0
            self._challenger_spend_last_sync_at = datetime.now()
            return 0.0

        now = datetime.now()
        if (
            not force
            and self._challenger_spend_last_sync_at
            and (now - self._challenger_spend_last_sync_at).total_seconds() < 300
        ):
            return self._challenger_month_spend_usd

        try:
            month_start = datetime(now.year, now.month, 1).isoformat()
            response = (
                self.supabase
                .from_('platform_signals')
                .select('ai_confidence_breakdown')
                .gte('created_at', month_start)
                .order('created_at', desc=True)
                .limit(5000)
                .execute()
            )
            rows = response.data or []

            total = 0.0
            for row in rows:
                breakdown = self._extract_breakdown_dict(row.get('ai_confidence_breakdown'))
                try:
                    total += max(0.0, float(breakdown.get('challenger_estimated_cost_usd') or 0.0))
                except (TypeError, ValueError):
                    continue

            self._challenger_month_spend_usd = round(total, 8)
            self._challenger_spend_last_sync_at = now
            return self._challenger_month_spend_usd
        except Exception as budget_error:
            logger.warning(f"⚠️ Failed syncing challenger monthly spend: {budget_error}")
            return self._challenger_month_spend_usd

    def _should_verify_with_challenger(
        self,
        decision: 'AIDecision',
        opportunity_score: OpportunityScore,
    ) -> bool:
        """Gate challenger invocation to keep quality gains while controlling spend."""
        if not settings.CHALLENGER_ENABLED:
            return False
        if not self.challenger_client:
            return False
        if decision.recommendation not in {'LONG', 'SHORT'}:
            return False
        if self._challenger_calls_this_run >= max(settings.CHALLENGER_MAX_SIGNALS_PER_RUN, 0):
            return False
        monthly_budget = max(float(settings.CHALLENGER_MONTHLY_BUDGET_USD or 0.0), 0.0)
        if monthly_budget > 0 and self._challenger_month_spend_usd >= monthly_budget:
            return False

        high_confidence = decision.confidence >= settings.CHALLENGER_CONFIDENCE_THRESHOLD
        strong_opportunity = opportunity_score.overall_score >= settings.CHALLENGER_MIN_OPPORTUNITY_SCORE
        exposure_score = (decision.leverage or 0) * max((decision.position_size or 0) / 100.0, 0.0)
        high_exposure = exposure_score >= settings.CHALLENGER_MIN_EXPOSURE_SCORE
        high_risk_label = str(decision.risk_level or '').upper() in {'HIGH', 'EXTREME'}

        return bool(high_confidence and strong_opportunity and (high_exposure or high_risk_label))

    async def _apply_challenger_verification(
        self,
        symbol: str,
        decision: 'AIDecision',
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
    ) -> Optional['AIDecision']:
        """Run optional challenger verification without modifying model-selected sizing."""
        provider = settings.CHALLENGER_PROVIDER.lower()
        model = settings.CHALLENGER_MODEL

        prompt = f"""
You are a strict trading risk verifier.
Evaluate the proposed trade and return ONLY valid JSON:
{{
  "verdict": "APPROVE|REJECT",
  "reason": "short explanation",
  "risk_flags": ["flag1", "flag2"]
}}

Symbol: {symbol}
Recommendation: {decision.recommendation}
Confidence: {decision.confidence:.4f}
Entry: {decision.entry_price:.8f}
Target1: {decision.target_1:.8f}
Target2: {decision.target_2:.8f}
StopLoss: {decision.stop_loss:.8f}
RiskReward: {decision.risk_reward_ratio:.4f}
Leverage: {decision.leverage}
PositionSizePct: {decision.position_size:.4f}
RiskLevel: {decision.risk_level}
OpportunityScore: {opportunity_score.overall_score:.4f}
RSI: {technical_analysis.rsi_14:.2f}
24hVolatilityPct: {technical_analysis.volatility_24h:.2f}
Trend: {technical_analysis.trend_direction}

Rules:
- REJECT for unsafe, unrealistic, or improperly sized setups.
- APPROVE only when the model-selected risk profile is acceptable as submitted.
- Do not suggest revised position size or leverage; this is approval review only.
"""

        try:
            if provider in {'deepseek', 'openai'}:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.challenger_client.chat.completions.create,
                        model=model,
                        max_tokens=settings.CHALLENGER_MAX_TOKENS,
                        temperature=settings.CHALLENGER_TEMPERATURE,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=float(settings.CHALLENGER_TIMEOUT_SECONDS),
                )
                challenger_text = response.choices[0].message.content
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.challenger_client.messages.create,
                        model=model,
                        max_tokens=settings.CHALLENGER_MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=float(settings.CHALLENGER_TIMEOUT_SECONDS),
                )
                challenger_text = response.content[0].text
        except Exception as challenger_error:
            logger.warning(f"⚠️ Challenger call failed for {symbol}: {challenger_error}")
            return decision

        self._challenger_calls_this_run += 1
        decision.challenger_applied = True
        decision.challenger_provider = provider
        decision.challenger_model = model
        prompt_tokens, completion_tokens = self._extract_token_usage(response, provider)
        decision.challenger_prompt_tokens = prompt_tokens
        decision.challenger_completion_tokens = completion_tokens
        decision.challenger_estimated_cost_usd = self._estimate_llm_cost_usd(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost_per_1m=settings.CHALLENGER_LLM_INPUT_COST_PER_1M,
            output_cost_per_1m=settings.CHALLENGER_LLM_OUTPUT_COST_PER_1M,
        )
        self._challenger_month_spend_usd = round(
            self._challenger_month_spend_usd + max(0.0, float(decision.challenger_estimated_cost_usd or 0.0)),
            8
        )

        parsed = self._extract_json_object(challenger_text or "")
        if not parsed:
            logger.warning(f"⚠️ Challenger response parse failed for {symbol}; skipping challenger adjustments")
            decision.challenger_verdict = 'PARSE_FAILED'
            return decision

        verdict = str(parsed.get('verdict') or 'APPROVE').upper().strip()
        reason = str(parsed.get('reason') or '').strip()
        risk_flags_raw = parsed.get('risk_flags') or []
        if not isinstance(risk_flags_raw, list):
            risk_flags_raw = []
        risk_flags = [str(flag).strip() for flag in risk_flags_raw if str(flag).strip()]

        decision.challenger_verdict = verdict
        decision.challenger_reason = reason
        if risk_flags:
            decision.risk_factors = list(dict.fromkeys((decision.risk_factors or []) + risk_flags))

        if verdict in {'HARD_REJECT', 'REJECT', 'SOFT_REJECT', 'DOWNSIZE', 'ADJUST'}:
            logger.info(f"🛑 Challenger hard-rejected {symbol}; skipping signal")
            return None

        return decision

    def _build_learning_context_signal(
        self,
        symbol: str,
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
        direction: str,
        confidence: float,
        time_horizon: str = "4h-24h",
    ) -> PlatformSignal:
        """Build a temporary signal so scoped learning can evaluate a candidate before LLM generation."""
        timeframe = self._derive_signal_timeframe(time_horizon, confidence)
        base_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
        setup_type = str((opportunity_score.breakdown or {}).get('setup_type') or '').strip()

        # Direction-aware placeholder levels: LONG targets resistance and stops at
        # support; SHORT is the mirror. Keeps scoped-learning matching on
        # structurally correct geometry.
        direction_u = str(direction or '').upper()
        current = float(technical_analysis.current_price or 0.0)
        resistance = float(technical_analysis.resistance_level or current or 0.0)
        support = float(technical_analysis.support_level or current or 0.0)
        if direction_u == 'SHORT':
            placeholder_target = support
            placeholder_stop = resistance
        else:
            placeholder_target = resistance
            placeholder_stop = support

        return PlatformSignal(
            signal_id=uuid.uuid4().hex[:12],
            token_symbol=base_symbol,
            direction=direction_u,
            timeframe=timeframe,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            overall_score=max(0.0, min(1.0, float(opportunity_score.overall_score or 0.0))),
            signal_strength=str(direction or 'HOLD').upper(),
            time_horizon=time_horizon,
            entry_price=current,
            target_1=placeholder_target,
            target_1_probability=0.5,
            target_2=placeholder_target,
            target_2_probability=0.3,
            stop_loss=placeholder_stop,
            risk_reward_ratio=1.5,
            market_conditions={
                'current_price': technical_analysis.current_price,
                'price_change_24h': technical_analysis.price_change_24h,
                'volume_24h': technical_analysis.volume_24h,
                'volatility_24h': technical_analysis.volatility_24h,
                'atr_14': technical_analysis.atr_14,
                'atr_pct_at_generation': (
                    float(technical_analysis.atr_14 or 0.0)
                    / max(float(technical_analysis.current_price or 0.0), 1e-12)
                    * 100.0
                ),
                'volume_ratio': technical_analysis.volume_ratio,
                'trend_direction': technical_analysis.trend_direction,
                'setup': setup_type,
            },
            technical_indicators={
                'rsi_14': technical_analysis.rsi_14,
                'rsi': technical_analysis.rsi_14,
                'macd_histogram': technical_analysis.macd_histogram,
                'macd': technical_analysis.macd_line,
                'macd_signal': technical_analysis.macd_signal,
                'bb_position': technical_analysis.bb_position,
                'strength_score': technical_analysis.strength_score,
                'momentum_score': technical_analysis.momentum_score,
            },
            sentiment_data={},
            risk_factors=[],
            signal_pool=SignalPool.FULL_MODE,
            opportunity_rank=1,
            analysis_timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=12),
            validity_window_hours=12,
            status='active',
            leverage=3,
            position_size=2.0,
            risk_level='MEDIUM',
            pattern_classification=setup_type or None,
        )

    async def _get_pre_generation_learning_context(
        self,
        symbol: str,
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
        direction: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """Evaluate scoped learning before spending an LLM call."""
        try:
            if not self.learning_service or direction not in {"LONG", "SHORT"}:
                return {
                    "has_evidence": False,
                    "hard_veto": False,
                    "prompt_text": "- Scoped learning not applicable for this pre-generation direction.",
                    "reason": "not_applicable",
                    "evidence": [],
                }
            probe_signal = self._build_learning_context_signal(
                symbol,
                technical_analysis,
                opportunity_score,
                direction,
                confidence,
            )
            return await self.learning_service.get_pre_generation_context(probe_signal)
        except Exception as exc:
            logger.debug(f"Pre-generation scoped learning skipped for {symbol}: {exc}")
            return {
                "has_evidence": False,
                "hard_veto": False,
                "prompt_text": "- Scoped learning context unavailable.",
                "reason": str(exc),
                "evidence": [],
            }

    async def ai_analyze_opportunity(
        self,
        symbol: str,
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
        *,
        policy_win_probability: Optional[float] = None,
        policy_eval_direction: Optional[str] = None,
        rule_direction: Optional[str] = None,
        rule_confidence: Optional[float] = None,
        regime_type: Optional[str] = None,
        scoped_learning_context: Optional[Dict[str, Any]] = None,
        active_thesis_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AIDecision]:
        """
        Get AI analysis and trading decision (synthesis + guardrails).

        Predictive P(win) from the offline policy is passed when available so the LLM can calibrate narrative.
        """
        logger.debug(f"🧠 Getting AI analysis for {symbol}")

        try:
            # Ensure services are initialized
            if not self._services_initialized:
                await self._initialize_services()

            # Fetch recent learning/performance context (non-blocking safe default)
            recent_performance = {}
            try:
                recent_performance = await self.learning_service.get_learning_statistics()
            except Exception as _:
                recent_performance = {}

            perf_summary_lines = []
            if recent_performance:
                wr = recent_performance.get('win_rate')
                avg = recent_performance.get('average_pnl')
                patterns = recent_performance.get('total_patterns')
                boosts = recent_performance.get('patterns_with_boost')
                cuts = recent_performance.get('patterns_with_reduction')
                if wr is not None and avg is not None:
                    perf_summary_lines.append(f"- 30d Win Rate: {wr:.1%}")
                    perf_summary_lines.append(f"- 30d Avg PnL: {avg:+.2f}%")
                if patterns is not None:
                    perf_summary_lines.append(f"- Patterns: total={patterns}, boost={boosts or 0}, reduce={cuts or 0}")
            perf_summary_text = "\n".join(perf_summary_lines) if perf_summary_lines else "- No recent learning data available"

            # Prepare comprehensive data for AI
            analysis_data = {
                'symbol': symbol.replace('/USDT', ''),
                'timeframe': '4h',
                'current_price': technical_analysis.current_price,
                'opportunity_score': opportunity_score.overall_score,

                # Technical Analysis
                'technical_indicators': {
                    # Basic indicators
                    'rsi_14': technical_analysis.rsi_14,
                    'macd_line': technical_analysis.macd_line,
                    'macd_signal': technical_analysis.macd_signal,
                    'macd_histogram': technical_analysis.macd_histogram,
                    'bb_position': technical_analysis.bb_position,
                    'ema_20': technical_analysis.ema_20,
                    'ema_50': technical_analysis.ema_50,
                    'ema_200': technical_analysis.ema_200,
                    'trend_direction': technical_analysis.trend_direction,
                    'momentum_score': technical_analysis.momentum_score,
                    'strength_score': technical_analysis.strength_score,

                    # Advanced oscillators
                    'stochastic_k': technical_analysis.stoch_k,
                    'stochastic_d': technical_analysis.stoch_d,
                    'williams_r': technical_analysis.williams_r,
                    'cci_14': technical_analysis.cci_14,
                    'roc_10': technical_analysis.roc_10,

                    # Volume analysis
                    'money_flow_index': technical_analysis.money_flow_index,
                    'on_balance_volume': technical_analysis.obv,
                    'vwap': technical_analysis.vwap,
                    'volume_profile_poc': technical_analysis.volume_profile_poc,

                    # Multi-timeframe
                    'rsi_1h': technical_analysis.rsi_1h,
                    'macd_1h_line': technical_analysis.macd_1h_line,
                    'macd_1h_signal': technical_analysis.macd_1h_signal,
                    'trend_alignment': technical_analysis.trend_alignment,

                    # Market microstructure
                    'spread_estimate': technical_analysis.spread_estimate,
                    'tick_rule_momentum': technical_analysis.tick_rule_momentum,
                    'price_efficiency': technical_analysis.price_efficiency
                },

                # Market Data
                'market_data': {
                    'price_change_24h': technical_analysis.price_change_24h,
                    'volume_24h': technical_analysis.volume_24h,
                    'volume_ratio': technical_analysis.volume_ratio,
                    'volatility_24h': technical_analysis.volatility_24h,
                    'atr_14': technical_analysis.atr_14
                },

                # Support/Resistance & Fibonacci
                'levels': {
                    'support': technical_analysis.support_level,
                    'resistance': technical_analysis.resistance_level,
                    'dynamic_support': technical_analysis.dynamic_support,
                    'dynamic_resistance': technical_analysis.dynamic_resistance,
                    'bb_upper': technical_analysis.bb_upper,
                    'bb_lower': technical_analysis.bb_lower,
                    'fibonacci_23_6': technical_analysis.fib_23_6,
                    'fibonacci_38_2': technical_analysis.fib_38_2,
                    'fibonacci_50_0': technical_analysis.fib_50_0,
                    'fibonacci_61_8': technical_analysis.fib_61_8
                },

                # Opportunity Breakdown
                'opportunity_breakdown': opportunity_score.breakdown
            }

            # Pre-format conditional values to avoid f-string format issues
            rsi_1h_str = f"{technical_analysis.rsi_1h:.1f}" if technical_analysis.rsi_1h is not None else 'N/A'
            macd_1h_str = f"{(technical_analysis.macd_1h_line - technical_analysis.macd_1h_signal):.6f}" if technical_analysis.macd_1h_line is not None and technical_analysis.macd_1h_signal is not None else 'N/A'
            trend_alignment_str = f"{technical_analysis.trend_alignment:.2f}" if technical_analysis.trend_alignment is not None else 'N/A'
            stoch_k_str = f"{technical_analysis.stoch_k:.1f}" if technical_analysis.stoch_k is not None else 'N/A'
            stoch_d_str = f"{technical_analysis.stoch_d:.1f}" if technical_analysis.stoch_d is not None else 'N/A'
            williams_r_str = f"{technical_analysis.williams_r:.1f}" if technical_analysis.williams_r is not None else 'N/A'
            cci_14_str = f"{technical_analysis.cci_14:.1f}" if technical_analysis.cci_14 is not None else 'N/A'
            roc_10_str = f"{technical_analysis.roc_10:.2f}" if technical_analysis.roc_10 is not None else 'N/A'
            mfi_str = f"{technical_analysis.money_flow_index:.1f}" if technical_analysis.money_flow_index is not None else 'N/A'
            vwap_str = f"${technical_analysis.vwap:.6f}" if technical_analysis.vwap is not None else 'N/A'
            vwap_distance_str = f"{abs(technical_analysis.current_price - technical_analysis.vwap)/technical_analysis.current_price*100:.2f}" if technical_analysis.vwap is not None else 'N/A'
            vol_poc_str = f"${technical_analysis.volume_profile_poc:.6f}" if technical_analysis.volume_profile_poc is not None else 'N/A'
            obv_str = f"{technical_analysis.obv:,.0f}" if technical_analysis.obv is not None else 'N/A'
            spread_str = f"{technical_analysis.spread_estimate:.2f}" if technical_analysis.spread_estimate is not None else 'N/A'
            tick_momentum_str = f"{technical_analysis.tick_rule_momentum:.2f}" if technical_analysis.tick_rule_momentum is not None else 'N/A'
            price_efficiency_str = f"{technical_analysis.price_efficiency:.2f}" if technical_analysis.price_efficiency is not None else 'N/A'
            
            # Advanced AI Metrics
            taker_ratio_str = f"{technical_analysis.taker_buy_sell_ratio:.2f}" if technical_analysis.taker_buy_sell_ratio is not None else 'N/A'
            liq_vol_str = f"${technical_analysis.liquidation_volume:,.0f}" if technical_analysis.liquidation_volume is not None else 'N/A'
            basis_str = f"{technical_analysis.basis_premium:+.4f}%" if technical_analysis.basis_premium is not None else 'N/A'

            # Support/Resistance and Fibonacci levels
            dynamic_support_str = f"${technical_analysis.dynamic_support:.6f}" if technical_analysis.dynamic_support is not None else 'N/A'
            dynamic_resistance_str = f"${technical_analysis.dynamic_resistance:.6f}" if technical_analysis.dynamic_resistance is not None else 'N/A'
            fib_23_6_str = f"${technical_analysis.fib_23_6:.6f}" if technical_analysis.fib_23_6 is not None else 'N/A'
            fib_38_2_str = f"${technical_analysis.fib_38_2:.6f}" if technical_analysis.fib_38_2 is not None else 'N/A'
            fib_50_0_str = f"${technical_analysis.fib_50_0:.6f}" if technical_analysis.fib_50_0 is not None else 'N/A'
            fib_61_8_str = f"${technical_analysis.fib_61_8:.6f}" if technical_analysis.fib_61_8 is not None else 'N/A'

            # Get pattern performance context for AI
            pattern_classification = await self._classify_market_pattern(technical_analysis)

            # Validate pattern classification
            if not pattern_classification or pattern_classification.strip() == "":
                pattern_classification = 'unknown_pattern'

            pattern_performance = await self._get_pattern_performance(pattern_classification)

            pattern_performance_text = "- No historical pattern data available"
            if pattern_performance:
                # Robust None handling and validation
                success_rate = pattern_performance.get('success_rate') or 0
                sample_size = pattern_performance.get('sample_size') or 0
                avg_pnl = pattern_performance.get('avg_pnl') or 0

                # Clamp success rate to valid range
                success_rate = min(max(float(success_rate), 0), 1)

                # Only show pattern data if sample size is meaningful
                if sample_size >= 5:
                    pattern_performance_text = f"""- Pattern: '{pattern_classification}' (Historical: {success_rate:.1%} success rate, {sample_size} trades, {avg_pnl:+.1f}% avg PnL)
- CONTEXT: Consider if current market conditions/regime might make this pattern more or less viable than historical average
- NOTE: Pattern history is guidance, not absolute - market conditions evolve and you should adapt accordingly"""
                else:
                    pattern_performance_text = f"""- Pattern: '{pattern_classification}' (Insufficient historical data: only {sample_size} trades)
- CONTEXT: No meaningful historical performance data - rely on current technical and market analysis
- NOTE: This pattern needs more historical data for reliable performance statistics"""

            # Fetch real BTC market data for accurate regime detection
            btc_data = await self.get_btc_market_data()
            market_regime = self.determine_market_regime(btc_data)
            correlation = self.calculate_correlation(technical_analysis.price_change_24h, btc_data['price_24h'])

            # Real funding rate + orderbook depth (replaces synthetic placeholders).
            (
                funding_rate_decimal,
                orderbook,
                positioning_signal,
                open_interest_context,
                venue_snapshot,
                token_event_context,
            ) = await asyncio.gather(
                self._fetch_funding_rate(symbol),
                self._fetch_orderbook_depth(symbol, levels=20),
                self._fetch_positioning_signal(symbol),
                self._fetch_open_interest_context(symbol),
                self._fetch_hyperliquid_venue_snapshot(symbol),
                self._fetch_token_event_context(symbol),
                return_exceptions=True,
            )
            if isinstance(funding_rate_decimal, Exception):
                funding_rate_decimal = None
            if isinstance(orderbook, Exception):
                orderbook = None
            if isinstance(positioning_signal, Exception):
                logger.debug("Positioning context unavailable for %s: %s", symbol, positioning_signal)
                positioning_signal = None
            if isinstance(open_interest_context, Exception):
                logger.debug("Open interest context unavailable for %s: %s", symbol, open_interest_context)
                open_interest_context = None
            if isinstance(venue_snapshot, Exception):
                logger.debug("Hyperliquid snapshot unavailable for %s: %s", symbol, venue_snapshot)
                venue_snapshot = None
            if isinstance(token_event_context, Exception):
                logger.debug("Token event context unavailable for %s: %s", symbol, token_event_context)
                token_event_context = {"available": False, "events": []}

            # Yuki executes on Hyperliquid, so venue-native observations take
            # precedence while Binance remains the discovery/history fallback.
            if isinstance(venue_snapshot, dict):
                venue_funding = venue_snapshot.get("funding_rate")
                if venue_funding is not None:
                    funding_rate_decimal = float(venue_funding)
                if venue_snapshot.get("bid_depth_usd") is not None:
                    orderbook = {
                        "bid_depth_usdt": float(venue_snapshot.get("bid_depth_usd") or 0.0),
                        "ask_depth_usdt": float(venue_snapshot.get("ask_depth_usd") or 0.0),
                        "imbalance": float(venue_snapshot.get("orderbook_imbalance") or 0.0),
                        "top_bid": float(venue_snapshot.get("top_bid") or 0.0),
                        "top_ask": float(venue_snapshot.get("top_ask") or 0.0),
                        "spread_pct": float(venue_snapshot.get("spread_pct") or 0.0),
                        "source": "hyperliquid",
                    }
                if venue_snapshot.get("open_interest") is not None:
                    existing_oi = (
                        dict(open_interest_context)
                        if isinstance(open_interest_context, dict)
                        else {}
                    )
                    existing_oi.update({
                        "open_interest": float(venue_snapshot["open_interest"]),
                        "open_interest_notional_usd": venue_snapshot.get("open_interest_notional_usd"),
                        "source": "hyperliquid",
                    })
                    open_interest_context = existing_oi

            market_structure_context = get_market_structure_context_service().build_context(
                symbol=symbol.replace('/USDT:USDT', '').replace('/USDT', ''),
                current_price=technical_analysis.current_price,
                candles=getattr(technical_analysis, "structure_candles", None),
                timeframe="4h",
                orderbook=orderbook,
                funding_rate=funding_rate_decimal,
                long_short_ratio=(
                    positioning_signal.get("ls_ratio")
                    if isinstance(positioning_signal, dict) and positioning_signal.get("ls_available")
                    else None
                ),
                open_interest=(
                    open_interest_context.get("open_interest")
                    if isinstance(open_interest_context, dict)
                    else None
                ),
                open_interest_change_24h=(
                    open_interest_context.get("open_interest_change_24h")
                    if isinstance(open_interest_context, dict)
                    else None
                ),
                taker_flow=positioning_signal if isinstance(positioning_signal, dict) else None,
                technical_snapshot={
                    "price_change_24h": technical_analysis.price_change_24h,
                    "trend_direction": technical_analysis.trend_direction,
                    "ema_20": technical_analysis.ema_20,
                    "ema_50": technical_analysis.ema_50,
                    "ema_200": technical_analysis.ema_200,
                },
            )
            market_structure_context["venue_snapshot"] = venue_snapshot
            market_structure_context["basis_premium_pct"] = (
                venue_snapshot.get("basis_premium_pct")
                if isinstance(venue_snapshot, dict)
                else technical_analysis.basis_premium
            )
            market_structure_context["data_sources"] = {
                "execution_venue": "hyperliquid" if venue_snapshot else None,
                "discovery_and_history": "binance",
                "venue_snapshot_available": bool(venue_snapshot),
            }
            market_structure_context["token_event_context"] = token_event_context
            analysis_data["market_structure_context"] = market_structure_context
            market_structure_prompt = (
                market_structure_context.get("prompt_block")
                or "- Real market-structure/order-flow context unavailable; do not infer missing zones, OI, taker flow, or depth."
            )
            token_events = (
                token_event_context.get("events")
                if isinstance(token_event_context, dict)
                else []
            ) or []
            if token_events:
                event_lines = [
                    (
                        f"- {event.get('event_type') or 'token event'} "
                        f"({event.get('impact_level') or 'unknown'} impact): "
                        f"{event.get('title') or event.get('description') or 'details unavailable'}"
                    )
                    for event in token_events[:3]
                ]
                market_structure_prompt += (
                    "\n\nTOKEN SUPPLY / VENUE EVENTS:\n" + "\n".join(event_lines)
                )

            funding_rate_pct_str = (
                f"{funding_rate_decimal * 100:+.4f}%" if funding_rate_decimal is not None else "N/A"
            )
            funding_pressure_str = (
                ("longs paying shorts" if funding_rate_decimal > 0 else "shorts paying longs")
                if funding_rate_decimal is not None else "unknown"
            )
            if orderbook:
                ob_imbalance_str = f"{orderbook['imbalance']:+.2f}"
                ob_bid_depth_str = f"${orderbook['bid_depth_usdt']:,.0f}"
                ob_ask_depth_str = f"${orderbook['ask_depth_usdt']:,.0f}"
                ob_spread_str = f"{orderbook['spread_pct']:.3f}%"
            else:
                ob_imbalance_str = "N/A"
                ob_bid_depth_str = "N/A"
                ob_ask_depth_str = "N/A"
                ob_spread_str = "N/A"

            # Feed learned expiry/volatility constraints into the model before it chooses levels.
            current_price_for_guidance = max(float(technical_analysis.current_price or 0.0), 1e-8)
            atr_pct_guidance = (float(technical_analysis.atr_14 or 0.0) / current_price_for_guidance) * 100.0
            target_guidance_lines = []
            for horizon_label in ("4h-12h", "4h-24h", "1-3 days", "3-7 days", "1-2 weeks"):
                constraints = self._target_distance_constraints(
                    horizon_label,
                    atr_pct_guidance,
                    float(technical_analysis.volatility_24h or 0.0),
                    0.75,
                )
                target_guidance_lines.append(
                    "- {horizon}: target_1 {t1_min:.2f}-{t1_max:.2f}% from entry; "
                    "stop {sl_min:.2f}-{sl_max:.2f}% from entry".format(
                        horizon=horizon_label,
                        t1_min=constraints["min_t1_move_pct"],
                        t1_max=constraints["max_t1_move_pct"],
                        sl_min=constraints["min_sl_move_pct"],
                        sl_max=constraints["max_sl_move_pct"],
                    )
                )
            target_guidance_text = "\n".join(target_guidance_lines)

            signal_quality_state = dict(self._signal_quality_state or {})
            resolved_for_sizing = int(signal_quality_state.get("resolved_signals") or 0)
            informative_for_sizing = int(signal_quality_state.get("informative_signals") or 0)
            expired_for_sizing = int(signal_quality_state.get("expired_signals") or 0)
            expiry_rate_for_sizing = (
                expired_for_sizing / resolved_for_sizing
                if resolved_for_sizing > 0
                else None
            )
            sizing_learning_context = "- Recent outcome history is not yet sufficient for sizing guidance."
            if informative_for_sizing >= 10 and signal_quality_state.get("win_rate") is not None:
                sizing_learning_context = (
                    f"- Recent informative outcomes: {informative_for_sizing} trades, "
                    f"{float(signal_quality_state.get('win_rate') or 0.0):.1%} target-hit win rate; "
                    f"expired share: {expiry_rate_for_sizing:.1%} of {resolved_for_sizing} terminal signals."
                )

            rd = rule_direction or ""
            rc = rule_confidence
            if policy_win_probability is not None:
                ped = policy_eval_direction or rd or "n/a"
                rc_note = f"- Rule engine snapshot: {rd or 'n/a'} @ {rc:.3f}\n" if rc is not None else ""
                policy_context_block = (
                    f"- Estimated P(hit target before stop | historical resolved signals, direction={ped}): "
                    f"{policy_win_probability:.2f}\n"
                    "- Statistical prior only — not a guarantee. Apply independent judgment and risk management.\n"
                    f"{rc_note}"
                )
            else:
                policy_context_block = (
                    "- Policy model inactive or not loaded. Rely on technicals, regime, and the adaptive threshold above.\n"
                )

            scoped_learning_text = "- Scoped learning context was not available for this candidate."
            if scoped_learning_context:
                scoped_learning_text = str(
                    scoped_learning_context.get("prompt_text")
                    or scoped_learning_context.get("reason")
                    or scoped_learning_text
                )

            active_thesis_prompt = self._active_thesis_prompt_block(active_thesis_context)
            active_thesis_output_fields = ""
            active_thesis_policy = "- No live thesis exists; create a new signal only if the current setup qualifies."
            if active_thesis_context:
                accepted_direction = str(active_thesis_context.get("direction") or "").upper().strip()
                active_thesis_output_fields = (
                    ", active_thesis_action (KEEP/INVALIDATE/REVERSE), "
                    "structural_invalidation (boolean), structural_invalidation_reason"
                )
                active_thesis_policy = f"""- This token already has an accepted {accepted_direction} thesis. Evaluate it as a lifecycle decision, not as an unrelated new candidate.
- KEEP when the original thesis remains plausible, including a neutral/HOLD result or reduced confidence. KEEP never creates a second same-direction signal and preserves the original order and levels.
- INVALIDATE only when current price structure has decisively broken the original thesis before entry. Set structural_invalidation=true and state the specific broken level/structure in structural_invalidation_reason. A generic HOLD, lower confidence, policy veto, disagreement, crowding, or missing data is not structural invalidation.
- REVERSE only with a confirmed opposite {('SHORT' if accepted_direction == 'LONG' else 'LONG')} recommendation. The normal opposite-signal workflow will cancel/close the old exposure before considering the new entry.
- If recommendation is HOLD, active_thesis_action must be KEEP unless a specific structural break justifies INVALIDATE.
- Never output INVALIDATE without both structural_invalidation=true and a concrete reason."""

            # Enhanced AI Analysis Prompt with real market regime awareness
            prompt_context = {}
            prompt_version = "yuki_v0"
            prompt = f"""
**INSTITUTIONAL CRYPTO ANALYSIS: {symbol.replace('/USDT', '')}** (4h Futures Signal)

**JSON OUTPUT REQUIRED** - End response with complete JSON containing: recommendation, confidence, reasoning, key_factors, entry_price, entry_strategy (MARKET/LIMIT_SUPPORT/LIMIT_RESISTANCE/PULLBACK/VOLUME_BASED), target_1/2, target_1/2_probability, stop_loss, risk_reward_ratio, position_size, leverage, risk_level (must be "LOW", "MEDIUM", "HIGH", or "EXTREME"), time_horizon, risk_factors, risk_assessment{active_thesis_output_fields}.
Return ONLY valid JSON (no markdown, no code fences, no prose before/after JSON).
For recommendation="HOLD": use numeric placeholders (entry_price=0, target_1=0, target_2=0, stop_loss=0, risk_reward_ratio=0, position_size=0, leverage=1) and NEVER output "N/A".
For recommendation="LONG" or "SHORT": position_size must be one numeric percentage and leverage must be one integer multiplier; never return labels or ranges for either value.

**CONFIDENCE CALCULATION**: Calculate confidence (0.0-1.0) based on the actual strength of technical confluence, setup quality, and market conditions. Strong setups with multiple confirmations should have confidence >0.80. Moderate setups 0.60-0.80. Weak/mixed setups <0.60.

**MARKET REGIME & CONTEXT:**
- BTC 24h: {btc_data['price_24h']:+.1f}% | 7d: {btc_data['price_7d']:+.1f}% | Regime: {market_regime}
- {symbol} vs BTC Correlation: {correlation} ({technical_analysis.price_change_24h:+.1f}% vs {btc_data['price_24h']:+.1f}%)
- BTC Dominance: {btc_data['dominance']:.1f}% | BTC Vol: {btc_data['volatility']:.1f}%
- Sector: {"DeFi" if any(x in symbol.lower() for x in ["uni", "aave", "comp", "mkr", "crv"]) else "Layer1" if any(x in symbol.lower() for x in ["eth", "sol", "ada", "dot", "avax"]) else "ALT"}
- Macro: {"Risk-On" if btc_data['price_24h'] > 1 else "Risk-Off" if btc_data['price_24h'] < -1 else "Neutral"}

**PERFORMANCE CONTEXT (30d):** {perf_summary_text}

	**PATTERN PERFORMANCE HISTORY:**
	{pattern_performance_text}

**SCOPED LEARNING PRIOR (PRE-GENERATION):**
	{scoped_learning_text}

**EXISTING ACCEPTED THESIS (AUTHORITATIVE LIFECYCLE CONTEXT):**
{active_thesis_prompt}

**EXISTING THESIS DECISION RULES:**
{active_thesis_policy}

	**LIVE MARKET DATA:**
	Price: ${technical_analysis.current_price:.6f} | 24h: {technical_analysis.price_change_24h:+.2f}% | Vol: ${technical_analysis.volume_24h:,.0f} | Volatility: {technical_analysis.volatility_24h:.1f}%

**TECHNICAL SNAPSHOT:**
| Timeframe | RSI | MACD | BB Pos | Trend |
|-----------|-----|------|--------|-------|
| 4h | {technical_analysis.rsi_14:.0f} | {"↗" if technical_analysis.macd_histogram > 0 else "↘"} | {technical_analysis.bb_position:.2f} | {technical_analysis.trend_direction} |
| 1h | {rsi_1h_str} | {macd_1h_str} | - | {trend_alignment_str} |

**OSCILLATORS:** Stoch: {stoch_k_str}/{stoch_d_str} | Williams: {williams_r_str} | CCI: {cci_14_str} | ROC: {roc_10_str}%

**VOLUME & FLOW:**
- MFI: {mfi_str} | VWAP: {vwap_distance_str}% | OBV: {obv_str}
- Volume POC: {vol_poc_str} | Spread: {spread_str}% | Tick Momentum: {tick_momentum_str}

**ORDER FLOW INDICATORS (REAL):**
- Funding Rate (8h): {funding_rate_pct_str} ({funding_pressure_str}) — high absolute rates pressure crowded side
- Order Book Imbalance (top 20 levels): {ob_imbalance_str} (positive = more bids/buy pressure)
- Order Book Depth: bids {ob_bid_depth_str} | asks {ob_ask_depth_str} | spread {ob_spread_str}
- Taker Buy/Sell Ratio: {taker_ratio_str} (>1.0 indicates aggressive buying momentum)
- Basis Premium (Spot vs Futures): {basis_str} (Contango=bullish/greedy, Backwardation=bearish/fear)
- Liquidation Volume: {liq_vol_str} (High volume indicates exhaustion/capitulation)
- Market Flow: {"Altcoin rotation" if correlation == "LOW" and btc_data['price_24h'] > 2 else "BTC dominance" if correlation == "HIGH" else "Mixed flows"}

**SHARED MARKET STRUCTURE CONTEXT (REAL DATA ONLY):**
{market_structure_prompt}

**KEY LEVELS:**
- Support: ${technical_analysis.support_level:.6f} | Resistance: ${technical_analysis.resistance_level:.6f}
- Fibs: 38.2%({fib_38_2_str}) | 50%({fib_50_0_str}) | 61.8%({fib_61_8_str})

**SIGNAL FRAMEWORK:**
- Overall Score: {opportunity_score.overall_score:.2f}/1.0 | Position in Range: {opportunity_score.breakdown.get('position_in_range', 0):.2f}
- Setup: {opportunity_score.breakdown.get('setup_type', 'neutral')} | Momentum: {technical_analysis.momentum_score:.2f} | Strength: {technical_analysis.strength_score:.2f}

**DECISION POLICY & DIRECTIONAL CONFLUENCE RULES (CRITICAL):**
- **Trend-following by default**: When trend, momentum, and volume align (e.g. EMA stack bullish + MACD positive + increasing OBV), trade WITH the trend. This is the highest-probability approach when there is no counter-trend data evidence.
- **Counter-trend trades ARE allowed** — but ONLY when supported by specific order flow data. You MUST cite at least 2 of the following in your reasoning to go against the trend:
  1. **Funding rate extreme**: Absolute funding > 0.08% (extreme long/short crowding creating flush risk)
  2. **Order book wall**: Large bid/ask imbalance > 0.3 in the opposing direction indicating institutional resistance/support
  3. **Liquidation cluster**: High liquidation volume indicating exhaustion/capitulation
  4. **Structural breakdown**: Price has broken below a key EMA or support/resistance level on 4h, not just 1h oscillator extremes
  5. **Basis extreme**: Significant contango/backwardation premium signaling directional divergence
- **Do NOT call counter-trend based solely on**: 1h RSI overbought/oversold, minor price extension, or "it's gone up a lot". These are weak signals that regularly fail against momentum.
- **Require Confluence**: Only recommend LONG/SHORT when at least 2 key indicators agree. If signals conflict and no strong order flow evidence exists, choose HOLD.
- Report your genuine confidence (0.0-1.0) based on actual setup strength — the system will apply its own quality filter

**OFFLINE PREDICTIVE POLICY (P(win | resolved history)):**
{policy_context_block}

**LEARNED TARGET/STOP CALIBRATION:**
- Current ATR: {atr_pct_guidance:.2f}% of price | 24h volatility: {technical_analysis.volatility_24h:.2f}%
{target_guidance_text}
- Choose the final levels yourself, but wider short-horizon target_1 distances have recently created expiry-heavy signals.
- If the setup needs a target beyond the matching band, either select a longer time_horizon, lower confidence, or HOLD.

**LEARNED POSITION/LEVERAGE POLICY:**
- Choose position_size and leverage yourself from current edge, volatility, expiry risk, policy/history support, and liquidation risk; resolved negative learning may reduce risk after generation, but will never increase it.
- Your full allowed range is position_size 1-100% and leverage 1-40x (venue per-asset caps still apply). There is no house bias toward small sizing: size to the edge, not to generic risk convention. Strong confluence with tight invalidation and supportive learned history justifies large size and higher leverage; weak or unproven edge justifies small size or HOLD.
{sizing_learning_context}
- Do not default repeatedly to the same position size, leverage, or time_horizon across different setups; make each output reflect that market's risk.
- For LONG/SHORT futures signals, do not use 1x as a substitute for insufficient conviction; choose HOLD when even 2x risk is not justified.
- If policy probability is weak or the chosen target is near the wide end of the band, independently choose smaller risk or HOLD.

**OPTIMAL ENTRY STRATEGIES:**
🎯 **Limit Orders at Key Levels:**
- LONG: Set entry_price near support levels (${technical_analysis.support_level:.6f}) for better risk/reward
- SHORT: Set entry_price near resistance levels (${technical_analysis.resistance_level:.6f}) for optimal entries
- Consider Fibonacci retracements: 38.2%({fib_38_2_str}), 50%({fib_50_0_str}), 61.8%({fib_61_8_str})

🎯 **Pullback Entries:**
- Wait for 0.5-2% pullback from breakout levels before entering
- Entry on retest of broken resistance (now support) or vice versa
- Use current price (${technical_analysis.current_price:.6f}) as reference for pullback calculation

🎯 **Volume-Based Entries:**
- Enter on volume exhaustion (low MFI: {mfi_str}) with oversold conditions
- Enter on volume confirmation (high volume + momentum alignment)
- Consider VWAP distance ({vwap_distance_str}%) for institutional entry zones

🎯 **Order Book Analysis Considerations:**
- Account for bid/ask spread impact: {spread_str}%
- Consider liquidity depth at key levels
- Factor in liquidation clusters for better entry timing
- Assess funding rate pressure for entry timing

**ENTRY PRICE DETERMINATION:**
- Market Order: Use current_price (${technical_analysis.current_price:.6f}) only if immediate execution required
- Limit Order: Set strategic entry_price based on support/resistance and pullback analysis
- Factor in volatility ({technical_analysis.volatility_24h:.1f}%) for entry buffer
- Consider regime impact: {market_regime} conditions favor {"aggressive" if market_regime in ["BULLISH", "BEARISH"] else "patient"} entries

**ANALYSIS REQUIREMENTS:**
1. Consider market regime impact on altcoin behavior
2. Factor in BTC correlation and sector rotation dynamics
3. Assess order flow implications (funding, liquidations)
4. Weight confluence of technical levels with regime context
5. **DETERMINE OPTIMAL ENTRY PRICE using above entry strategies**
6. **Calculate genuine confidence (0.0-1.0) based on setup strength, confluence, and market conditions**
7. **Recommend LONG/SHORT based on technical confluence and setup quality. Report your genuine confidence (0.0-1.0) — do not self-censor directional signals based on a confidence threshold.**
8. **Recommend dynamic post-TP1 trailing stop buffer percentage (post_tp1_stop_buffer_pct: 0.008 to 0.035) based on token volatility, ATR, spread, and market sentiment to allow runners to breathe without being choked out by 1-minute noise.**

**Your analysis should be unique and insightful, focusing on what makes THIS setup special in the current market context.**
"""

            # Inject prompt optimization directives from learning history.
            if self.prompt_optimization_service:
                try:
                    prompt_context = await self.prompt_optimization_service.get_signal_generation_context(prompt)
                    prompt_version = prompt_context.get('prompt_version') or prompt_version
                    guidance_block = prompt_context.get('guidance_text', '').strip()
                    if guidance_block:
                        prompt = f"{prompt}\n\n{guidance_block}"
                except Exception as prompt_opt_error:
                    logger.warning(f"⚠️ Prompt optimization context unavailable for {symbol}: {prompt_opt_error}")

            # Inject Yuki ReAct Agent Memory (Active Goals & Postmortem Lessons Learned)
            try:
                from kata.services.agent_memory_service import get_agent_memory_service
                memory_service = get_agent_memory_service("yuki")
                goals = await memory_service.get_active_goals()
                lessons = await memory_service.get_active_lessons(symbol=symbol, limit=5)
                
                agent_mem_lines = []
                if goals:
                    agent_mem_lines.append(f"🤖 YUKI AGENT ACTIVE STRATEGIC GOAL:\n{goals[0].goal_statement}")
                if lessons:
                    agent_mem_lines.append("💡 YUKI AGENT PAST POSTMORTEM LESSONS (DO NOT REPEAT PAST MISTAKES):")
                    for l in lessons:
                        agent_mem_lines.append(f"- {l}")
                
                if agent_mem_lines:
                    agent_mem_str = "\n".join(agent_mem_lines)
                    prompt = f"{prompt}\n\n{agent_mem_str}"
            except Exception as mem_err:
                logger.warning(f"⚠️ Yuki agent memory injection bypassed for {symbol}: {mem_err}")

            primary_prompt_tokens = 0
            primary_completion_tokens = 0
            primary_estimated_cost = 0.0

            # Call LLM API based on provider
            try:
                provider = settings.LLM_PROVIDER.lower()
                if provider in {"deepseek", "openai"}:
                    # OpenAI-compatible API call (DeepSeek and OpenAI).
                    response = await asyncio.to_thread(
                        self.llm_client.chat.completions.create,
                        model=settings.LLM_MODEL,
                        max_tokens=settings.LLM_MAX_TOKENS,
                        temperature=settings.LLM_TEMPERATURE,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_text = response.choices[0].message.content
                    logger.debug(f"{provider.title()} response for {symbol}: {len(ai_text) if ai_text else 0} chars")
                else:
                    # Claude API call - Sonnet can handle higher token limits
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.llm_client.messages.create,
                            model=settings.LLM_MODEL,  # claude-sonnet-4-5-20250929
                            max_tokens=settings.LLM_MAX_TOKENS,
                            messages=[{"role": "user", "content": prompt}]
                        ),
                        timeout=180.0  # 3 minute timeout for Sonnet
                    )
                    ai_text = response.content[0].text
                    logger.debug(f"Claude response for {symbol}: {len(ai_text) if ai_text else 0} chars")
            except asyncio.TimeoutError:
                logger.error(f"⏰ {provider.title()} API timeout for {symbol} (>3 minutes)")
                return None
            except Exception as e:
                logger.error(f"❌ {provider.title()} API error for {symbol}: {e}")
                if provider == "deepseek":
                    api_key_present = bool(settings.DEEPSEEK_API_KEY)
                elif provider == "openai":
                    api_key_present = bool(settings.OPENAI_API_KEY)
                else:
                    api_key_present = bool(settings.ANTHROPIC_API_KEY)
                logger.error(f"API Key present: {api_key_present}")

                # Check if it's a specific Claude error
                if "Streaming is required" in str(e):
                    logger.error(f"💡 Suggestion: Prompt may be too long or complex for {symbol}")

                return None

            # Debug logging for empty responses
            if not ai_text or ai_text.strip() == "":
                provider_name = settings.LLM_PROVIDER.title()
                logger.error(f"❌ {provider_name} returned empty response for {symbol}")
                logger.error(f"Full response object: {response}")
                return None

            primary_prompt_tokens, primary_completion_tokens = self._extract_token_usage(response, provider)
            primary_estimated_cost = self._estimate_llm_cost_usd(
                prompt_tokens=primary_prompt_tokens,
                completion_tokens=primary_completion_tokens,
                input_cost_per_1m=settings.PRIMARY_LLM_INPUT_COST_PER_1M,
                output_cost_per_1m=settings.PRIMARY_LLM_OUTPUT_COST_PER_1M,
            )

            # Check if response appears to be truncated (incomplete JSON)
            is_truncated = (
                len(ai_text) >= 7800 or  # Near new 8K max tokens limit
                ai_text.endswith('Maybe') or ai_text.endswith('I\'ll') or
                ('{' in ai_text and ai_text.count('{') > ai_text.count('}'))
            )

            if is_truncated:
                logger.warning(f"⚠️ Response appears truncated for {symbol} (length: {len(ai_text)})")

            effective_regime = str(regime_type or "unknown")
            try:
                base_rule_conf = float(rule_confidence) if rule_confidence is not None else None
            except (TypeError, ValueError):
                base_rule_conf = None

            # Extract JSON from response
            try:
                # Find JSON in response - look for the last complete JSON object
                start_idx = ai_text.find('{')
                if start_idx == -1:
                    provider_name = settings.LLM_PROVIDER.title()
                    logger.error(f"❌ No JSON found in {provider_name} response for {symbol}")
                    logger.error(f"Full response text (first 1000 chars): {ai_text[:1000]}")
                    logger.error(f"Full response text (last 1000 chars): {ai_text[-1000:]}")
                    return None
                
                # Find the matching closing brace by counting braces
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(ai_text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                logger.debug(f"JSON extraction for {symbol}: {start_idx}:{end_idx}, braces={brace_count}")
                
                if brace_count != 0:
                    logger.warning(f"⚠️ Unmatched braces in JSON for {symbol}, trying fallback extraction")
                    # Try to find JSON at the end of the response
                    last_brace = ai_text.rfind('}')
                    if last_brace != -1:
                        # Look backwards for the matching opening brace
                        brace_count = 0
                        for i in range(last_brace, -1, -1):
                            if ai_text[i] == '}':
                                brace_count += 1
                            elif ai_text[i] == '{':
                                brace_count -= 1
                                if brace_count == 0:
                                    start_idx = i
                                    end_idx = last_brace + 1
                                    json_str = ai_text[start_idx:end_idx]
                                    logger.debug(f"Fallback JSON extraction successful: {start_idx}:{end_idx}")
                                    break
                        else:
                            logger.error(f"❌ Fallback JSON extraction failed for {symbol}")
                            logger.error(f"Full response text (first 1000 chars): {ai_text[:1000]}")
                            logger.error(f"Full response text (last 1000 chars): {ai_text[-1000:]}")
                            return None
                    else:
                        logger.error(f"❌ No closing brace found for {symbol}")
                        logger.error(f"Full response text (first 1000 chars): {ai_text[:1000]}")
                        logger.error(f"Full response text (last 1000 chars): {ai_text[-1000:]}")
                        return None
                
                json_str = ai_text[start_idx:end_idx]

                # Clean JSON string for parsing - ultra-robust handling
                import re
                
                # Remove BOM
                json_str = json_str.lstrip('\ufeff')
                json_str = json_str.strip()

                # Step 1: Fix Python-style booleans and None
                json_str = re.sub(r'\bTrue\b', 'true', json_str)
                json_str = re.sub(r'\bFalse\b', 'false', json_str)
                json_str = re.sub(r'\bNone\b', 'null', json_str)

                # Step 2: CAREFULLY handle control characters inside strings vs structure
                # Instead of blindly removing, preserve JSON structure by only cleaning string content
                
                # First, temporarily replace escaped quotes to protect them
                json_str = json_str.replace('\\"', '<<<ESCAPED_QUOTE>>>')
                
                # Find all string values and clean them individually
                def clean_string_content(match):
                    content = match.group(0)
                    # Remove control characters only inside this string
                    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
                    # Clean up multiple spaces
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    return cleaned
                
                # Apply cleaning only to string values (between quotes)
                json_str = re.sub(r'"[^"]*"', clean_string_content, json_str)
                
                # Restore escaped quotes
                json_str = json_str.replace('<<<ESCAPED_QUOTE>>>', '\\"')
                
                # Step 3: Clean up spacing around JSON structural elements
                json_str = re.sub(r'\s*:\s*', ': ', json_str)
                json_str = re.sub(r'\s*,\s*', ', ', json_str)
                
                # Step 4: Fix common AI mistakes - missing commas between array/object elements
                # Add comma between }" and "{ (missing comma between objects)
                json_str = re.sub(r'"\s*"', '", "', json_str)  # Between string elements
                json_str = re.sub(r'"\s*\{', '", {', json_str)  # Between string and object
                json_str = re.sub(r'\}\s*"', '}, "', json_str)  # Between object and string
                json_str = re.sub(r'\}\s*\{', '}, {', json_str)  # Between objects
                json_str = re.sub(r'\]\s*"', '], "', json_str)  # Between array and string
                json_str = re.sub(r'"\s*\[', '", [', json_str)  # Between string and array

                ai_decision_data = json.loads(json_str)

                # Schema validation first
                ok, reason = self._validate_ai_schema(ai_decision_data)
                if not ok:
                    logger.warning(f"AI schema validation failed: {reason}")
                    logger.warning(f"AI response JSON: {json_str}")
                    logger.warning(f"Parsed data: {ai_decision_data}")
                    return None

                if active_thesis_context:
                    action = str(ai_decision_data.get("active_thesis_action") or "").upper().strip()
                    structural_raw = ai_decision_data.get("structural_invalidation", False)
                    structural = structural_raw is True or str(structural_raw).lower().strip() == "true"
                    structural_reason = str(
                        ai_decision_data.get("structural_invalidation_reason") or ""
                    ).strip()
                    accepted_direction = str(active_thesis_context.get("direction") or "").upper().strip()
                    recommendation = str(ai_decision_data.get("recommendation") or "").upper().strip()
                    opposite_direction = "SHORT" if accepted_direction == "LONG" else "LONG"
                    valid_lifecycle = action in ACTIVE_THESIS_ACTIONS
                    valid_lifecycle = valid_lifecycle and not (
                        action == "INVALIDATE"
                        and (not structural or not structural_reason or recommendation != "HOLD")
                    )
                    valid_lifecycle = valid_lifecycle and not (
                        action == "REVERSE" and recommendation != opposite_direction
                    )
                    valid_lifecycle = valid_lifecycle and not (
                        action == "KEEP" and recommendation == opposite_direction
                    )
                    if not valid_lifecycle:
                        logger.warning(
                            "AI lifecycle response invalid for %s: action=%s recommendation=%s "
                            "structural=%s reason=%s",
                            symbol,
                            action or "missing",
                            recommendation,
                            structural,
                            structural_reason or "missing",
                        )
                        return None
                    ai_decision_data["active_thesis_action"] = action
                    ai_decision_data["structural_invalidation"] = structural
                    ai_decision_data["structural_invalidation_reason"] = structural_reason

                # Create AI decision object - handle HOLD recommendations with null prices
                rec = ai_decision_data['recommendation'].upper()
                ai_conf_raw = float(ai_decision_data.get('confidence') or 0.5)
                effective_rule_conf = base_rule_conf if base_rule_conf is not None else ai_conf_raw
                effective_min_rr = self._dynamic_min_net_rr_threshold(
                    ai_confidence=ai_conf_raw,
                    opp_score=float(opportunity_score.overall_score or 0.0),
                    rule_confidence=effective_rule_conf,
                    regime_type=effective_regime,
                    volatility_24h=float(technical_analysis.volatility_24h or 0.0),
                )
                unfavorable_min_rr = max(0.9, effective_min_rr - 0.2)
                if rec == 'HOLD':
                    # For HOLD, use current price as defaults since we won't trade
                    current_price = technical_analysis.current_price
                    entry_price = current_price
                    stop_loss = current_price
                    target_1 = current_price
                    target_2 = current_price
                    risk_reward_ratio = 1.0
                    ai_proposed_levels = None
                else:
                    # For LONG/SHORT, use AI-provided values with safe price parsing
                    entry_price = clean_price_string(ai_decision_data['entry_price'])
                    stop_loss = clean_price_string(ai_decision_data['stop_loss'])
                    target_1 = clean_price_string(ai_decision_data.get('target_1')) or entry_price * 1.05
                    target_2 = clean_price_string(ai_decision_data.get('target_2')) or entry_price * 1.10
                    risk_reward_ratio = clean_price_string(ai_decision_data.get('risk_reward_ratio')) or 2.0

                    # Snapshot the LLM's levels before any sanitization/floors/adaptation
                    ai_proposed_levels = {
                        'entry_price': float(entry_price or 0.0),
                        'target_1': float(target_1 or 0.0),
                        'target_2': float(target_2 or 0.0),
                        'stop_loss': float(stop_loss or 0.0),
                    }

                    # Sanitise targets: AI sometimes returns target == entry (0% gain).
                    # Apply minimum 3% / 6% defaults rather than letting the signal fail feasibility.
                    if rec == 'LONG':
                        if target_1 <= entry_price:
                            logger.warning(f"⚠️ {symbol}: AI target_1 {target_1} ≤ entry {entry_price} — defaulting to +3%")
                            target_1 = entry_price * 1.03
                        if target_2 <= target_1:
                            logger.warning(f"⚠️ {symbol}: AI target_2 {target_2} ≤ target_1 {target_1} — defaulting to +6%")
                            target_2 = entry_price * 1.06
                    elif rec == 'SHORT':
                        if target_1 >= entry_price:
                            logger.warning(f"⚠️ {symbol}: AI target_1 {target_1} ≥ entry {entry_price} — defaulting to -3%")
                            target_1 = entry_price * 0.97
                        if target_2 >= target_1:
                            logger.warning(f"⚠️ {symbol}: AI target_2 {target_2} ≥ target_1 {target_1} — defaulting to -6%")
                            target_2 = entry_price * 0.94

                    # Validate AI's prices without modification - skip signal if not feasible
                    current_market_price = technical_analysis.current_price

                    # Adaptive entry price logic - handle favorable vs unfavorable price movements
                    price_deviation = abs(current_market_price - entry_price) / entry_price

                    if price_deviation > 0.10:  # 10% threshold for adaptation
                        # Calculate if price movement is favorable or unfavorable for the trade direction
                        price_moved_up = current_market_price > entry_price

                        if rec == 'SHORT' and price_moved_up:
                            # FAVORABLE: Price moved UP for SHORT - better entry point
                            logger.info(f"📈 {symbol} SHORT: Price moved favorably from {entry_price} to {current_market_price} (+{price_deviation:.1%}) - adapting entry")

                            # Update entry to current price and recalculate targets/stops
                            original_entry = entry_price
                            entry_price = current_market_price

                            # Recalculate targets and stops based on original percentages
                            entry_to_target1_pct = abs(target_1 - original_entry) / original_entry
                            entry_to_target2_pct = abs(target_2 - original_entry) / original_entry
                            entry_to_stop_pct = abs(stop_loss - original_entry) / original_entry

                            # Apply same percentages to new entry
                            target_1 = entry_price * (1 - entry_to_target1_pct)  # SHORT: targets are below entry
                            target_2 = entry_price * (1 - entry_to_target2_pct)
                            stop_loss = entry_price * (1 + entry_to_stop_pct)    # SHORT: stop is above entry

                            logger.info(f"🔄 Adapted {symbol} SHORT: Entry {original_entry} → {entry_price}, T1: {target_1:.4f}, T2: {target_2:.4f}, SL: {stop_loss:.4f}")

                            # Validate risk/reward ratio after adaptation
                            risk = abs(entry_price - stop_loss) / entry_price
                            reward = abs(target_1 - entry_price) / entry_price
                            adapted_rr_ratio = reward / risk if risk > 0 else 0

                            if adapted_rr_ratio < effective_min_rr:
                                logger.warning(
                                    f"❌ Skipping {symbol} SHORT - poor risk/reward after adaptation: "
                                    f"{adapted_rr_ratio:.2f}:1 < {effective_min_rr:.2f}:1 "
                                    f"(risk: {risk:.1%}, reward: {reward:.1%})"
                                )
                                return None

                            # Ensure stop loss isn't too tight or too loose after adaptation
                            if risk < 0.01:  # Less than 1% risk is too tight
                                logger.warning(f"❌ Skipping {symbol} SHORT - stop loss too tight after adaptation: {risk:.1%}")
                                return None
                            elif risk > 0.15:  # More than 15% risk is too loose
                                logger.warning(f"❌ Skipping {symbol} SHORT - stop loss too loose after adaptation: {risk:.1%}")
                                return None

                            logger.info(f"✅ {symbol} SHORT: Adapted risk/reward ratio: {adapted_rr_ratio:.2f}:1")

                        elif rec == 'LONG' and not price_moved_up:
                            # FAVORABLE: Price moved DOWN for LONG - better entry point
                            logger.info(f"📉 {symbol} LONG: Price moved favorably from {entry_price} to {current_market_price} (-{price_deviation:.1%}) - adapting entry")

                            # Update entry to current price and recalculate targets/stops
                            original_entry = entry_price
                            entry_price = current_market_price

                            # Recalculate targets and stops based on original percentages
                            entry_to_target1_pct = abs(target_1 - original_entry) / original_entry
                            entry_to_target2_pct = abs(target_2 - original_entry) / original_entry
                            entry_to_stop_pct = abs(stop_loss - original_entry) / original_entry

                            # Apply same percentages to new entry
                            target_1 = entry_price * (1 + entry_to_target1_pct)  # LONG: targets are above entry
                            target_2 = entry_price * (1 + entry_to_target2_pct)
                            stop_loss = entry_price * (1 - entry_to_stop_pct)    # LONG: stop is below entry

                            logger.info(f"🔄 Adapted {symbol} LONG: Entry {original_entry} → {entry_price}, T1: {target_1:.4f}, T2: {target_2:.4f}, SL: {stop_loss:.4f}")

                            # Validate risk/reward ratio after adaptation
                            risk = abs(entry_price - stop_loss) / entry_price
                            reward = abs(target_1 - entry_price) / entry_price
                            adapted_rr_ratio = reward / risk if risk > 0 else 0

                            if adapted_rr_ratio < effective_min_rr:
                                logger.warning(
                                    f"❌ Skipping {symbol} LONG - poor risk/reward after adaptation: "
                                    f"{adapted_rr_ratio:.2f}:1 < {effective_min_rr:.2f}:1 "
                                    f"(risk: {risk:.1%}, reward: {reward:.1%})"
                                )
                                return None

                            # Ensure stop loss isn't too tight or too loose after adaptation
                            if risk < 0.01:  # Less than 1% risk is too tight
                                logger.warning(f"❌ Skipping {symbol} LONG - stop loss too tight after adaptation: {risk:.1%}")
                                return None
                            elif risk > 0.15:  # More than 15% risk is too loose
                                logger.warning(f"❌ Skipping {symbol} LONG - stop loss too loose after adaptation: {risk:.1%}")
                                return None

                            logger.info(f"✅ {symbol} LONG: Adapted risk/reward ratio: {adapted_rr_ratio:.2f}:1")

                        else:
                            # UNFAVORABLE: Price moved against us - check if still viable
                            if price_deviation > 0.15:  # 15% threshold for rejection
                                logger.warning(f"❌ Skipping {symbol} {rec} signal - unfavorable price movement from {entry_price} to {current_market_price} ({price_deviation:.1%})")
                                return None
                            else:
                                # Still acceptable - moderate unfavorable movement, update entry
                                logger.warning(f"⚠️ {symbol} {rec}: Unfavorable price movement ({price_deviation:.1%}) but within tolerance - adapting entry")

                                original_entry = entry_price
                                entry_price = current_market_price

                                # Recalculate with original percentages but verify risk/reward still acceptable
                                entry_to_target1_pct = abs(target_1 - original_entry) / original_entry
                                entry_to_target2_pct = abs(target_2 - original_entry) / original_entry
                                entry_to_stop_pct = abs(stop_loss - original_entry) / original_entry

                                if rec == 'LONG':
                                    target_1 = entry_price * (1 + entry_to_target1_pct)
                                    target_2 = entry_price * (1 + entry_to_target2_pct)
                                    stop_loss = entry_price * (1 - entry_to_stop_pct)
                                else:  # SHORT
                                    target_1 = entry_price * (1 - entry_to_target1_pct)
                                    target_2 = entry_price * (1 - entry_to_target2_pct)
                                    stop_loss = entry_price * (1 + entry_to_stop_pct)

                                logger.info(f"🔄 Adapted {symbol} {rec}: Entry {original_entry} → {entry_price}, T1: {target_1:.4f}, T2: {target_2:.4f}, SL: {stop_loss:.4f}")

                                # Validate risk/reward ratio after unfavorable adaptation
                                risk = abs(entry_price - stop_loss) / entry_price
                                reward = abs(target_1 - entry_price) / entry_price
                                adapted_rr_ratio = reward / risk if risk > 0 else 0

                                if adapted_rr_ratio < unfavorable_min_rr:
                                    logger.warning(
                                        f"❌ Skipping {symbol} {rec} - poor risk/reward after unfavorable adaptation: "
                                        f"{adapted_rr_ratio:.2f}:1 < {unfavorable_min_rr:.2f}:1 "
                                        f"(risk: {risk:.1%}, reward: {reward:.1%})"
                                    )
                                    return None

                                # Ensure stop loss isn't too tight or too loose after unfavorable adaptation
                                if risk < 0.01:  # Less than 1% risk is too tight
                                    logger.warning(f"❌ Skipping {symbol} {rec} - stop loss too tight after unfavorable adaptation: {risk:.1%}")
                                    return None
                                elif risk > 0.20:  # More lenient for unfavorable moves (20%)
                                    logger.warning(f"❌ Skipping {symbol} {rec} - stop loss too loose after unfavorable adaptation: {risk:.1%}")
                                    return None

                                logger.info(f"⚠️ {symbol} {rec}: Unfavorable adapted risk/reward ratio: {adapted_rr_ratio:.2f}:1 (acceptable)")
                    else:
                        logger.debug(f"✅ {symbol} {rec}: Entry price {entry_price} vs current {current_market_price} deviation {price_deviation:.1%} acceptable")

                    # Skip signal if AI's targets are not feasible
                    if not self.targets_are_feasible(
                        rec,
                        current_market_price,
                        target_1,
                        target_2,
                        entry_price=entry_price,
                        entry_strategy=ai_decision_data.get('entry_strategy'),
                    ):
                        logger.warning(f"Skipping {symbol} {rec} signal - AI targets not feasible given current price {current_market_price}")
                        return None

                try:
                    # Safe numeric conversions with detailed error logging and percentage parsing
                    position_size_raw = ai_decision_data.get('position_size')
                    leverage_raw = ai_decision_data.get('leverage')

                    # Executable risk values must be explicit scalar model choices.
                    def parse_position_size(value):
                        if value is None:
                            raise ValueError("position_size is required")
                        if isinstance(value, str):
                            value_clean = value.replace('%', '').strip().lower()
                            if value_clean in {'n/a', 'na', 'none', ''}:
                                raise ValueError("position_size must be numeric")
                            if re.search(r'([0-9]*\.?[0-9]+)\s*(?:-|to|–)\s*([0-9]*\.?[0-9]+)', value_clean):
                                raise ValueError("position_size must be a single value")
                            try:
                                parsed = float(value_clean)
                            except ValueError:
                                raise ValueError(f"position_size must be numeric: {value!r}")
                        else:
                            parsed = float(value)
                        if parsed < 0:
                            raise ValueError("position_size cannot be negative")
                        return parsed

                    # Leverage is an integer multiplier selected by the model, not a translated label.
                    def parse_leverage(value):
                        if value is None:
                            raise ValueError("leverage is required")
                        if isinstance(value, str):
                            value_clean = value.strip().lower().replace('×', 'x')
                            if value_clean in {'n/a', 'na', 'none', ''}:
                                raise ValueError("leverage must be numeric")
                            if re.search(r'([0-9]*\.?[0-9]+)\s*x?\s*(?:-|to|–)\s*([0-9]*\.?[0-9]+)\s*x?', value_clean):
                                raise ValueError("leverage must be a single value")
                            value_clean = value_clean.replace('x', '').strip()
                            try:
                                parsed_float = float(value_clean)
                            except ValueError:
                                raise ValueError(f"leverage must be numeric: {value!r}")
                        else:
                            parsed_float = float(value)
                        if not parsed_float.is_integer():
                            raise ValueError("leverage must be an integer multiplier")
                        parsed = int(parsed_float)
                        if parsed < 1:
                            raise ValueError("leverage must be at least 1x")
                        return parsed

                    # Apply parsing with AI's actual decisions (no hardcoded limits)
                    parsed_position_size = parse_position_size(position_size_raw)
                    parsed_leverage = parse_leverage(leverage_raw)
                    if rec in {'LONG', 'SHORT'} and parsed_position_size <= 0:
                        raise ValueError("directional signals require a positive position_size")

                    # Dynamic SL distance floor:
                    # higher-conviction setups can run tighter stops; noisier regimes need wider stops.
                    min_sl_distance_pct = self._dynamic_min_sl_distance_pct(
                        ai_confidence=float(ai_decision_data.get('confidence') or 0.5),
                        opp_score=float(opportunity_score.overall_score or 0.0),
                        rule_confidence=effective_rule_conf,
                        regime_type=effective_regime,
                        volatility_24h=float(technical_analysis.volatility_24h or 0.0),
                    )
                    if entry_price > 0 and stop_loss > 0:
                        sl_distance_pct = abs(entry_price - stop_loss) / entry_price
                        if sl_distance_pct < min_sl_distance_pct:
                            widened_sl = (
                                entry_price * (1 - min_sl_distance_pct)
                                if rec == 'LONG'
                                else entry_price * (1 + min_sl_distance_pct)
                            )
                            logger.info(
                                f"🛡️ {symbol}: Widened SL from {sl_distance_pct:.2%} → "
                                f"{min_sl_distance_pct:.1%} floor "
                                f"(SL {stop_loss:.6f} → {widened_sl:.6f})"
                            )
                            stop_loss = widened_sl
                            # Recompute risk_reward_ratio with widened stop.
                            try:
                                risk = abs(entry_price - stop_loss) / entry_price
                                reward = abs(target_1 - entry_price) / entry_price
                                if risk > 0:
                                    risk_reward_ratio = reward / risk
                            except Exception:
                                pass

                    decision = AIDecision(
                        recommendation=rec,
                        confidence=float(ai_decision_data['confidence']),
                        reasoning=ai_decision_data['reasoning'],
                        key_factors=ai_decision_data.get('key_factors', []),
                        ai_proposed_levels=ai_proposed_levels,

                        entry_price=entry_price,
                        target_1=target_1,
                        target_1_probability=float(ai_decision_data.get('target_1_probability') or 0.7),
                        target_2=target_2,
                        target_2_probability=float(ai_decision_data.get('target_2_probability') or 0.4),
                        stop_loss=stop_loss,
                        risk_reward_ratio=risk_reward_ratio,

                        position_size=parsed_position_size,
                        leverage=parsed_leverage,
                        risk_level=ai_decision_data.get('risk_level') or 'MEDIUM',
                        time_horizon=ai_decision_data.get('time_horizon', '4h-12h'),
                        entry_strategy=str(ai_decision_data.get('entry_strategy') or '').strip().upper() or 'MARKET',
                        timeframe=self._derive_signal_timeframe(
                            ai_decision_data.get('time_horizon', '4h-12h'),
                            ai_conf_raw,
                        ),
                        raw_ai_confidence=ai_conf_raw,
                        raw_ai_position_size=parsed_position_size,
                        raw_ai_leverage=parsed_leverage,
                        raw_ai_time_horizon=str(ai_decision_data.get('time_horizon', '4h-12h')),

                        risk_factors=ai_decision_data.get('risk_factors', []),
                        risk_assessment=ai_decision_data.get('risk_assessment', 'Standard crypto trading risks'),
                        prompt_version=prompt_version,
                        llm_provider=settings.LLM_PROVIDER.lower(),
                        llm_model=settings.LLM_MODEL,
                        llm_prompt_tokens=primary_prompt_tokens,
                        llm_completion_tokens=primary_completion_tokens,
                        llm_estimated_cost_usd=primary_estimated_cost,
                        market_structure_context=market_structure_context,
                        active_thesis_action=(
                            str(ai_decision_data.get("active_thesis_action") or "").upper().strip()
                            or None
                        ),
                        structural_invalidation=bool(ai_decision_data.get("structural_invalidation", False)),
                        structural_invalidation_reason=(
                            str(ai_decision_data.get("structural_invalidation_reason") or "").strip()
                            or None
                        ),
                    )
                    decision.policy_win_probability = policy_win_probability
                    decision.policy_eval_direction = policy_eval_direction or rule_direction
                    if policy_win_probability is not None:
                        decision.policy_feature_version = POLICY_FEATURE_VERSION

                    # HOLD is non-executable; keep neutral sizing defaults and skip noisy fallbacks.
                    if rec == 'HOLD':
                        decision.leverage = 1
                        decision.position_size = 0.0
                    
                    # Log AI's original position sizing decisions (before learning adjustments)
                    logger.debug(f"🤖 AI Decision for {symbol}: {rec} @ {decision.confidence:.3f} | "
                               f"Leverage={decision.leverage}x, Position={decision.position_size:.1f}%, Risk={decision.risk_level}")
                except (ValueError, TypeError) as e:
                    logger.error(f"❌ Error converting AI numeric values for {symbol}: {e}")
                    logger.error(f"❌ Problematic data: position_size={position_size_raw}, leverage={leverage_raw}")
                    return None

                # HOLD is informational; skip leverage/size/confidence mutation from learning layer.
                if decision.recommendation == 'HOLD':
                    return decision

                learning_adjustment_for_sizing: Optional[LearningAdjustment] = None

                # Apply learning-based adjustments
                try:
                    decision_timeframe = (
                        str(getattr(decision, "timeframe", "") or "").strip()
                        or self._derive_signal_timeframe(
                            decision.time_horizon,
                            float(getattr(decision, "raw_ai_confidence", decision.confidence) or decision.confidence),
                        )
                    )
                    decision.timeframe = decision_timeframe
                    temp_signal = PlatformSignal(
                        signal_id=uuid.uuid4().hex[:12],
                        token_symbol=symbol.replace('/USDT:USDT', '').replace('/USDT', ''),
                        direction=decision.recommendation,
                        timeframe=decision_timeframe,
                        confidence=decision.confidence,
                        overall_score=decision.confidence,
                        signal_strength=decision.recommendation,
                        time_horizon=decision.time_horizon,
                        entry_price=decision.entry_price,
                        target_1=decision.target_1,
                        target_1_probability=decision.target_1_probability,
                        target_2=decision.target_2,
                        target_2_probability=decision.target_2_probability,
                        stop_loss=decision.stop_loss,
                        risk_reward_ratio=decision.risk_reward_ratio,
                        market_conditions={
                            'current_price': technical_analysis.current_price,
                            'price_change_24h': technical_analysis.price_change_24h,
                            'volume_24h': technical_analysis.volume_24h,
                            'volatility_24h': technical_analysis.volatility_24h,
                            'volume_ratio': technical_analysis.volume_ratio
                        },
                        technical_indicators={
                            'rsi': technical_analysis.rsi_14,
                            'macd': technical_analysis.macd_line,
                            'macd_signal': technical_analysis.macd_signal,
                            'bb_position': technical_analysis.bb_position
                        },
                        sentiment_data={'ai_confidence': decision.confidence},
                        risk_factors=decision.risk_factors,
                        signal_pool=SignalPool.FULL_MODE,
                        opportunity_rank=1,
                        analysis_timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=12),
                        validity_window_hours=12,
                        status='active',
                        ai_reasoning=decision.reasoning,
                        ai_key_factors=decision.key_factors,
                        ai_risk_assessment=decision.risk_assessment,
                        ai_confidence_breakdown=None,
                        leverage=decision.leverage,
                        position_size=decision.position_size,
                        risk_level=decision.risk_level,
                        run_id=None,
                        logo_url=None
                    )

                    adjustment: LearningAdjustment = await self.learning_service.get_learning_adjustment(temp_signal)
                    learning_adjustment_for_sizing = adjustment

                    original_conf = decision.confidence
                    original_lev = decision.leverage
                    original_pos = decision.position_size

                    # Scale learning adjustments by learning confidence to avoid
                    # over-penalizing good live setups on weak historical evidence.
                    confidence_level = (adjustment.confidence_level or 'medium').lower()
                    level_scale_map = {'high': 1.0, 'medium': 0.6, 'low': 0.25}
                    level_scale = level_scale_map.get(confidence_level, 0.5)

                    # Confidence learning remains bounded. Downside-only sizing reduction
                    # is applied once below with the complete learned context.
                    if confidence_level == 'high':
                        conf_cap = 0.10
                    elif confidence_level == 'medium':
                        conf_cap = 0.07
                    else:
                        conf_cap = 0.04

                    # Sharpe gate — when realized Sharpe is negative, skip positive adjustments
                    # (don't reward a strategy that's losing money on average).
                    sharpe = self._recent_sharpe_ratio(min_samples=20)
                    sharpe_scale = 1.0
                    if sharpe is not None:
                        if sharpe < 0:
                            sharpe_scale = 0.25  # heavy discount when losing
                        elif sharpe < 0.5:
                            sharpe_scale = 0.6
                        elif sharpe > 1.5:
                            sharpe_scale = 1.2
                        sharpe_scale = max(0.2, min(1.3, sharpe_scale))

                    scaled_conf_delta = adjustment.confidence_adjustment * level_scale * sharpe_scale
                    # Block positive boosts when Sharpe < 0 (mean recent PnL is negative).
                    if sharpe is not None and sharpe < 0 and scaled_conf_delta > 0:
                        scaled_conf_delta = 0.0
                    scaled_conf_delta = max(-conf_cap, min(conf_cap, scaled_conf_delta))

                    adjusted_conf = max(0.0, min(1.0, decision.confidence + scaled_conf_delta))

                    logger.info(
                        f"📈 Learning adjustment for {symbol}: "
                        f"conf {original_conf:.3f}→{adjusted_conf:.3f} (Δ={scaled_conf_delta:+.3f}, level={confidence_level}), "
                        f"LLM proposed sizing {original_lev}x / {original_pos:.2f}%"
                    )

                    decision.confidence = adjusted_conf

                    # Directional self-learning: haircut confidence for a direction that
                    # has been failing recently (e.g. shorts in an uptrend), reward one that
                    # has been working. Self-reverses as the regime turns — no static block.
                    dir_delta = self._directional_confidence_adjustment(decision.recommendation)
                    if dir_delta != 0.0:
                        before_dir = decision.confidence
                        decision.confidence = max(0.0, min(1.0, decision.confidence + dir_delta))
                        perf = self._direction_perf_cache.get(decision.recommendation, {})
                        logger.info(
                            f"🧭 Directional learning [{decision.recommendation}]: recent WR "
                            f"{perf.get('win_rate', 0.5):.0%} (n={int(perf.get('resolved', 0))}) → "
                            f"conf {before_dir:.3f}→{decision.confidence:.3f} (Δ={dir_delta:+.3f})"
                        )

                    if adjustment.reasoning:
                        decision.reasoning = f"{decision.reasoning}\nLearning adjustment: {adjustment.reasoning}"

                except Exception as le:
                    logger.debug(f"Learning adjustment skipped: {le}")

                # Adapt targets based on the LLM-selected horizon; horizon remains
                # model-selected while learned risk can reduce execution exposure.
                self._adapt_risk_targets(decision, technical_analysis)

                # Reduce exposure only when learned resolved outcomes support it.
                self._apply_learning_risk_reduction(
                    decision,
                    adjustment=learning_adjustment_for_sizing,
                    technical_analysis=technical_analysis,
                    opportunity_score=opportunity_score,
                    policy_win_probability=policy_win_probability,
                )

                # Optional cost-capped challenger verification for high-stakes signals.
                if self._should_verify_with_challenger(decision, opportunity_score):
                    challenged = await self._apply_challenger_verification(
                        symbol=symbol,
                        decision=decision,
                        technical_analysis=technical_analysis,
                        opportunity_score=opportunity_score,
                    )
                    if challenged is None:
                        return None
                    decision = challenged

                return decision

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"❌ Failed to parse AI response: {e}")
                logger.error(f"Original AI text (first 500 chars): {ai_text[:500]}")
                logger.error(f"Original AI text (last 500 chars): {ai_text[-500:]}")
                # Try to show cleaned JSON if it exists in scope
                try:
                    logger.error(f"Cleaned JSON (first 500 chars): {json_str[:500]}")
                    logger.error(f"Cleaned JSON (last 500 chars): {json_str[-500:]}")
                    logger.error(f"JSON string length: {len(json_str)}")
                    logger.error(f"JSON string starts with: {json_str[:50]}")
                    logger.error(f"JSON string ends with: {json_str[-50:]}")
                except:
                    pass
                return None

        except Exception as e:
            logger.error(f"❌ AI analysis failed for {symbol}: {e}")
            return None

    def _classify_volatility_environment(self, volatility_pct: float) -> str:
        """Classify volatility environment based on 24h volatility."""
        if volatility_pct >= 20:
            return 'extreme'
        elif volatility_pct >= 15:
            return 'high'
        elif volatility_pct >= 8:
            return 'elevated'
        elif volatility_pct >= 4:
            return 'normal'
        else:
            return 'low'

    def _generate_learning_data(
        self,
        symbol: str,
        ai_decision: AIDecision,
        technical_analysis: TechnicalAnalysis,
        btc_market_data: dict = None,
        learning_predictions: dict = None
    ) -> dict:
        """Generate learning-specific data for signal tracking."""

        # Market regime detection
        regime_data = self._detect_advanced_regime(technical_analysis)

        # BTC market context for regime confidence
        if not btc_market_data:
            # Fallback if BTC data not available
            btc_market_data = {
                'price_24h': 0, 'price_7d': 0, 'volatility': 5.0,
                'volume_24h': 0, 'dominance': 50.0
            }

        # Determine market regime
        market_regime = self.determine_market_regime(btc_market_data)

        # Calculate regime confidence based on technical and BTC alignment
        btc_correlation_str = self.calculate_correlation(technical_analysis.price_change_24h, btc_market_data['price_24h'])

        # Convert correlation string to numeric for confidence calculation
        correlation_numeric = {
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }.get(btc_correlation_str, 0.5)

        regime_confidence = min(1.0, correlation_numeric + regime_data.get('strength', 0.5))

        # Pattern classification from technical analysis
        pattern_classification = regime_data.get('type', 'unknown_pattern')
        pattern_confidence = regime_data.get('strength', 0.5)

        # Volatility classification
        volatility_environment = self._classify_volatility_environment(technical_analysis.volatility_24h)

        # Risk adjusted confidence based on multiple factors
        risk_adjusted_confidence = ai_decision.confidence * min(1.0,
            (technical_analysis.strength_score + pattern_confidence + regime_confidence) / 3.0
        )

        # Policy model win probability — populated when POLICY_LEARNING_ENABLED and bundle loaded
        predicted_success_probability = getattr(ai_decision, 'policy_win_probability', None)

        # Rough time-to-target estimate: assume signal resolves at ~60% of validity window
        predicted_time_to_target = None
        if hasattr(ai_decision, 'validity_window_hours') and ai_decision.validity_window_hours:
            predicted_time_to_target = float(ai_decision.validity_window_hours) * 0.6


        return {
            'market_regime': {
                'regime': market_regime,
                'regime_data': regime_data,
                'btc_correlation': btc_correlation_str,
                'btc_context': btc_market_data
            },
            'regime_confidence': regime_confidence,
            'volatility_environment': volatility_environment,
            'pattern_classification': pattern_classification,
            'pattern_confidence': pattern_confidence,
            'risk_adjusted_confidence': risk_adjusted_confidence,
            'learning_tracked': True,  # Enable for all AI-generated signals
            'learning_started_at': datetime.now(timezone.utc),
            'generated_by_agent': 'yuki',  # Default agent - could be parameterized
            'learning_version': 1,
            'prompt_version': compose_prompt_version(ai_decision.prompt_version),
            'llm_provider': ai_decision.llm_provider or settings.LLM_PROVIDER.lower(),
            'llm_model': ai_decision.llm_model or settings.LLM_MODEL,
            'llm_prompt_tokens': ai_decision.llm_prompt_tokens or 0,
            'llm_completion_tokens': ai_decision.llm_completion_tokens or 0,
            'llm_estimated_cost_usd': ai_decision.llm_estimated_cost_usd or 0.0,
            'challenger_applied': bool(ai_decision.challenger_applied),
            'challenger_provider': ai_decision.challenger_provider,
            'challenger_model': ai_decision.challenger_model,
            'challenger_verdict': ai_decision.challenger_verdict,
            'challenger_reason': ai_decision.challenger_reason,
            'challenger_prompt_tokens': ai_decision.challenger_prompt_tokens or 0,
            'challenger_completion_tokens': ai_decision.challenger_completion_tokens or 0,
            'challenger_estimated_cost_usd': ai_decision.challenger_estimated_cost_usd or 0.0,
        }

    async def generate_platform_signal(
        self,
        symbol: str,
        ai_decision: AIDecision,
        technical_analysis: TechnicalAnalysis,
        btc_market_data: dict = None,
        opportunity_score: Optional[OpportunityScore] = None,
    ) -> PlatformSignal:
        """Convert AI decision to platform signal with learning integration."""

        # Generate very short signal ID (max 12 chars for database safety)
        signal_id = f"{uuid.uuid4().hex[:12]}"  # 12 chars only

        # Logo will be fetched AFTER all signals are generated (in run_full_analysis_cycle)
        # This prevents slowing down signal generation and avoids rate limiting
        logo_url = None

        # Generate learning-specific data from existing analysis
        learning_data = self._generate_learning_data(symbol, ai_decision, technical_analysis, btc_market_data)

        policy_training_blob: Optional[Dict[str, Any]] = None
        if opportunity_score is not None and ai_decision.recommendation in ("LONG", "SHORT"):
            try:
                rule_c = float(
                    getattr(ai_decision, "ensemble_rule_confidence", None) or ai_decision.confidence
                )
                reg = str(getattr(ai_decision, "ensemble_regime_type", None) or "").strip()
                if not reg:
                    regime_data = (learning_data.get("market_regime") or {}).get("regime_data") or {}
                    reg = str(regime_data.get("type") or "unknown") if isinstance(regime_data, dict) else "unknown"
                policy_training_blob = policy_training_snapshot_from_live(
                    technical_analysis,
                    opportunity_score,
                    rule_c,
                    reg,
                    ai_decision.recommendation,
                    float(ai_decision.confidence),
                )
            except Exception as e:
                logger.debug("policy_training snapshot skipped: %s", e)

        breakdown_base: Dict[str, Any] = {
                'technical_confidence': technical_analysis.strength_score,
                'ai_confidence': ai_decision.confidence,
                'opportunity_confidence': technical_analysis.momentum_score,
                'llm_provider': learning_data['llm_provider'],
                'llm_model': learning_data['llm_model'],
                'prompt_version': learning_data['prompt_version'],
                # Full policy configuration that produced this signal, so
                # outcomes are attributed to the era that generated them.
                'policy_versions': policy_stamp(),
                'policy_bundle_version': policy_bundle_version(),
                'llm_prompt_tokens': learning_data['llm_prompt_tokens'],
                'llm_completion_tokens': learning_data['llm_completion_tokens'],
                'llm_estimated_cost_usd': learning_data['llm_estimated_cost_usd'],
                'challenger_applied': learning_data['challenger_applied'],
                'challenger_provider': learning_data['challenger_provider'],
                'challenger_model': learning_data['challenger_model'],
                'challenger_verdict': learning_data['challenger_verdict'],
                'challenger_reason': learning_data['challenger_reason'],
                'challenger_prompt_tokens': learning_data['challenger_prompt_tokens'],
                'challenger_completion_tokens': learning_data['challenger_completion_tokens'],
                'challenger_estimated_cost_usd': learning_data['challenger_estimated_cost_usd'],
                'total_llm_estimated_cost_usd': round(
                    float(learning_data['llm_estimated_cost_usd']) +
                    float(learning_data['challenger_estimated_cost_usd']),
                    8
                ),
                'market_regime': str((learning_data.get('market_regime') or {}).get('regime') or '').lower(),
                'btc_correlation': str(
                    (learning_data.get('market_regime') or {}).get('btc_correlation') or ''
                ).upper(),
                'policy_win_probability': getattr(ai_decision, 'policy_win_probability', None),
                'policy_eval_direction': getattr(ai_decision, 'policy_eval_direction', None),
                'policy_feature_version': getattr(ai_decision, 'policy_feature_version', None),
                # Stored so ensemble accuracy weighting can attribute wins/losses per source.
                'ensemble_rule_confidence': getattr(ai_decision, 'ensemble_rule_confidence', None),
                'ensemble_rule_direction': getattr(ai_decision, 'ensemble_rule_direction', None),
                'learning_risk_sizing': getattr(ai_decision, 'learning_sizing_context', None),
                'llm_position_size': getattr(ai_decision, 'raw_ai_position_size', None),
                'llm_leverage': getattr(ai_decision, 'raw_ai_leverage', None),
                'llm_time_horizon': getattr(ai_decision, 'raw_ai_time_horizon', None),
                'final_position_size': getattr(ai_decision, 'position_size', None),
                'final_leverage': getattr(ai_decision, 'leverage', None),
                'edge_estimate': getattr(ai_decision, 'edge_estimate', None),
                'calibrated_direction_probability': getattr(
                    ai_decision, 'calibrated_direction_probability', None
                ),
                'fill_probability': getattr(ai_decision, 'fill_probability', None),
                'target_before_stop_probability': getattr(
                    ai_decision, 'target_before_stop_probability', None
                ),
                'net_expected_value_pct': getattr(ai_decision, 'net_expected_value_pct', None),
                'edge_score': getattr(ai_decision, 'edge_score', None),
        }
        if policy_training_blob is not None:
            breakdown_base[POLICY_TRAINING_JSON_KEY] = policy_training_blob
        market_structure_context = getattr(ai_decision, "market_structure_context", None)
        if market_structure_context:
            breakdown_base["market_structure_context"] = market_structure_context

        signal_timeframe = (
            str(getattr(ai_decision, "timeframe", "") or "").strip()
            or self._derive_signal_timeframe(
                ai_decision.time_horizon,
                float(getattr(ai_decision, "raw_ai_confidence", ai_decision.confidence) or ai_decision.confidence),
            )
        )
        signal_validity_h = self._effective_published_validity_hours(ai_decision.time_horizon)
        analysis_timestamp = datetime.now(timezone.utc)
        signal_expires_at = analysis_timestamp + timedelta(hours=signal_validity_h)
        entry_activation_meta = self._build_entry_activation_metadata(
            direction=ai_decision.recommendation,
            entry_price=ai_decision.entry_price,
            current_price=technical_analysis.current_price,
            entry_strategy=getattr(ai_decision, "entry_strategy", None),
            signal_started_at=analysis_timestamp,
            signal_expires_at=signal_expires_at,
        )

        # --- Conviction tiering ---
        # Data-backed: the [0.70,0.80) confidence band shows the best realized
        # performance (75.5% WR, +5.54%/trade). Tag those as HIGH_CONVICTION for a
        # premium feed and scale size up (bounded) within existing risk limits.
        hc_threshold = float(getattr(settings, "SIGNAL_HIGH_CONVICTION_THRESHOLD", 0.70) or 0.70)
        hc_size_mult = max(1.0, float(getattr(settings, "SIGNAL_HIGH_CONVICTION_SIZE_MULT", 1.25) or 1.25))
        signal_confidence = float(ai_decision.confidence or 0.0)
        is_high_conviction = signal_confidence >= hc_threshold
        conviction_tier = "high_conviction" if is_high_conviction else "standard"
        final_position_size = float(getattr(ai_decision, "position_size", 0.0) or 0.0)
        if is_high_conviction:
            # position_size is a capital percentage; uplift but never exceed 100%.
            final_position_size = min(100.0, final_position_size * hc_size_mult)

        return PlatformSignal(
            signal_id=signal_id,
            token_symbol=symbol.replace('/USDT:USDT', '').replace('/USDT', ''),
            direction=ai_decision.recommendation,  # Already LONG/SHORT
            timeframe=signal_timeframe,

            confidence=ai_decision.confidence,
            overall_score=float(
                getattr(ai_decision, "edge_score", None)
                if getattr(ai_decision, "edge_score", None) is not None
                else float(ai_decision.confidence) * 100.0
            ),
            signal_strength=ai_decision.recommendation,
            time_horizon=ai_decision.time_horizon,

            entry_price=ai_decision.entry_price,
            target_1=ai_decision.target_1,
            target_1_probability=ai_decision.target_1_probability,
            target_2=ai_decision.target_2,
            target_2_probability=ai_decision.target_2_probability,
            stop_loss=ai_decision.stop_loss,
            risk_reward_ratio=ai_decision.risk_reward_ratio,

            market_conditions={
                'current_price': technical_analysis.current_price,
                'price_change_24h': technical_analysis.price_change_24h,
                'volume_24h': technical_analysis.volume_24h,
                'volatility_24h': technical_analysis.volatility_24h,
                'trend_direction': technical_analysis.trend_direction,
                'conviction_tier': conviction_tier,
                'high_conviction': is_high_conviction,
                'market_structure_context': market_structure_context,
                'ai_proposed_levels': getattr(ai_decision, 'ai_proposed_levels', None),
                'edge_estimate': getattr(ai_decision, 'edge_estimate', None),
                **entry_activation_meta,
            },

            technical_indicators={
                'rsi_14': technical_analysis.rsi_14,
                'macd_histogram': technical_analysis.macd_histogram,
                'bb_position': technical_analysis.bb_position,
                'momentum_score': technical_analysis.momentum_score,
                'strength_score': technical_analysis.strength_score,
                'atr_14': technical_analysis.atr_14,
            },

            sentiment_data={
                'ai_confidence': ai_decision.confidence,
                'opportunity_score': (
                    float(opportunity_score.overall_score)
                    if opportunity_score is not None
                    else technical_analysis.momentum_score
                ),
                'edge_score': getattr(ai_decision, 'edge_score', None),
                'net_expected_value_pct': getattr(ai_decision, 'net_expected_value_pct', None),
            },

            risk_factors=ai_decision.risk_factors,

            signal_pool=SignalPool.FULL_MODE,
            opportunity_rank=1,  # Will be set based on batch ranking
            analysis_timestamp=analysis_timestamp,
            expires_at=signal_expires_at,
            validity_window_hours=signal_validity_h,
            status='active',

            analysis_notes=f"AI-driven {signal_timeframe} analysis with {ai_decision.confidence:.1%} confidence",

            ai_reasoning=ai_decision.reasoning,
            ai_key_factors=ai_decision.key_factors,
            ai_risk_assessment=ai_decision.risk_assessment,  # Full AI risk assessment
            ai_confidence_breakdown=breakdown_base,

            leverage=ai_decision.leverage,
            position_size=final_position_size,
            risk_level=ai_decision.risk_level,

            run_id='',  # Will be set by calling function
            logo_url=logo_url,  # Token logo URL from CoinGecko

            # Learning System Integration - populated from existing analysis
            learning_tracked=learning_data['learning_tracked'],
            learning_started_at=learning_data['learning_started_at'],
            market_regime=learning_data['market_regime'],
            regime_confidence=learning_data['regime_confidence'],
            volatility_environment=learning_data['volatility_environment'],
            pattern_classification=learning_data['pattern_classification'],
            pattern_confidence=learning_data['pattern_confidence'],
            generated_by_agent=learning_data['generated_by_agent'],
            risk_adjusted_confidence=learning_data['risk_adjusted_confidence'],
            learning_version=learning_data['learning_version'],
            prompt_version=learning_data['prompt_version'],
        )

    def _detect_regime(self, ta: TechnicalAnalysis) -> str:
        """Detect market regime from technical context."""
        try:
            # Trend via EMA stack and price position
            if ta.current_price > ta.ema_20 > ta.ema_50:
                trend = 'trending_up'
            elif ta.current_price < ta.ema_20 < ta.ema_50:
                trend = 'trending_down'
            else:
                trend = 'ranging'

            # Volatility via ATR and 24h volatility
            vol = ta.volatility_24h
            if vol >= 15:
                vol_state = 'volatile'
            else:
                vol_state = 'normal'

            if trend in ['trending_up', 'trending_down'] and vol_state == 'normal':
                return trend
            if vol_state == 'volatile':
                return 'volatile'
            return 'ranging'
        except Exception:
            return 'ranging'

    def _classify_setup(self, ta: TechnicalAnalysis, opp: OpportunityScore) -> str:
        """Classify the current directional setup without assuming range extremes must fade."""
        try:
            pos = opp.breakdown.get('position_in_range', 0.5)
            macd_bullish = ta.macd_line > ta.macd_signal and ta.macd_histogram > 0
            macd_bearish = ta.macd_line < ta.macd_signal and ta.macd_histogram < 0
            fresh_momentum = self._fresh_momentum_base(ta.price_change_24h)

            logger.debug(
                "📊 Setup classification - pos: %.3f, trend: %s, fresh_momentum: %.3f",
                pos,
                ta.trend_direction,
                fresh_momentum,
            )

            # Continuation at the active edge of the range.
            if (
                pos >= 0.80
                and ta.trend_direction == 'bullish'
                and ta.price_change_24h > 0
                and macd_bullish
                and fresh_momentum >= 0.25
            ):
                return 'breakout_long'
            if (
                pos <= 0.20
                and ta.trend_direction == 'bearish'
                and ta.price_change_24h < 0
                and macd_bearish
                and fresh_momentum >= 0.25
            ):
                return 'breakout_short'

            # Reversal requires actual exhaustion plus a momentum turn.
            if pos >= 0.80 and ta.rsi_14 >= 68 and macd_bearish:
                return 'reversal_short'
            if pos <= 0.20 and ta.rsi_14 <= 32 and macd_bullish:
                return 'reversal_long'

            # Pullback requires proximity to the trend mean, not merely being on
            # the correct side of the slower EMA.
            atr_pct = abs(float(ta.atr_14 or 0.0)) / max(float(ta.current_price or 0.0), 1e-12)
            ema20_distance = abs(float(ta.current_price) - float(ta.ema_20)) / max(float(ta.current_price), 1e-12)
            pullback_buffer = max(0.01, min(0.04, atr_pct * 1.25))
            if (
                ema20_distance <= pullback_buffer
                and ta.trend_direction == 'bullish'
                and ta.current_price >= ta.ema_50
                and macd_bullish
            ):
                return 'pullback_long'
            if (
                ema20_distance <= pullback_buffer
                and ta.trend_direction == 'bearish'
                and ta.current_price <= ta.ema_50
                and macd_bearish
            ):
                return 'pullback_short'

            logger.debug(f"📊 Classified as neutral (no conditions met)")
            return 'neutral'
        except Exception as e:
            logger.debug(f"📊 Setup classification error: {e}, defaulting to neutral")
            return 'neutral'

    def _passes_confluence_gates(self, direction: str, ta: TechnicalAnalysis, opp: OpportunityScore) -> bool:
        """Require multiple independent signals to align."""
        try:
            signals = 0
            pos = opp.breakdown.get('position_in_range', 0.5)
            # BB confluence
            if direction == 'LONG' and ta.bb_position <= 0.35:
                signals += 1
            if direction == 'SHORT' and ta.bb_position >= 0.65:
                signals += 1
            # RSI confluence (oversold/overbought zones)
            if direction == 'LONG' and ta.rsi_14 <= 40:
                signals += 1
            if direction == 'SHORT' and ta.rsi_14 >= 60:
                signals += 1
            # Trend/level confluence
            if direction == 'LONG' and ta.trend_direction in ['bullish', 'sideways'] and pos <= 0.35:
                signals += 1
            if direction == 'SHORT' and ta.trend_direction in ['bearish', 'sideways'] and pos >= 0.65:
                signals += 1
            # Momentum
            if ta.momentum_score >= 0.55:
                signals += 1
            return signals >= 2  # require at least two
        except Exception:
            return False

    def _rule_engine_decision(self, ta: TechnicalAnalysis, opp: OpportunityScore, market_regime: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Advanced rule-based trading algorithm with sophisticated technical analysis.

        Uses multi-factor analysis including:
        - Market regime detection (trending/ranging/volatile)
        - Multi-timeframe confirmation
        - Volume profile analysis
        - Risk-adjusted momentum scoring
        - Advanced pattern recognition
        - Market microstructure signals
        """
        # Multi-factor analysis
        regime = self._detect_advanced_regime(ta)
        setup_strength = self._analyze_setup_strength(ta, opp, market_regime)
        momentum_score = self._calculate_momentum_score(ta)
        volume_confirmation = self._analyze_volume_profile(ta, opp)
        risk_metrics = self._calculate_risk_metrics(ta, opp)

        reasoning_parts = [
            f"regime={regime['type']}(strength={regime['strength']:.2f})",
            f"setup={setup_strength['pattern']}(score={setup_strength['score']:.2f})",
            f"momentum={momentum_score:.2f}",
            f"volume={volume_confirmation:.2f}",
            f"risk_adj={risk_metrics['risk_adjusted_score']:.2f}"
        ]

        # Base direction and confidence calculation
        direction, base_confidence = self._determine_base_signal(regime, setup_strength, momentum_score)

        # Advanced multi-factor confidence weighting using ALL indicators
        oscillator_confluence = self._calculate_oscillator_confluence(ta, direction)
        volume_confluence = self._calculate_advanced_volume_confluence(ta, direction)
        timeframe_confluence = self._calculate_timeframe_confluence(ta, direction)
        fibonacci_confluence = self._calculate_fibonacci_confluence(ta, direction)
        microstructure_quality = self._calculate_microstructure_quality(ta)

        # Comprehensive confidence adjustments with proper weighting
        confidence_adjustments = {
            'regime_strength': regime['strength'] * 0.18,
            'setup_quality': setup_strength['score'] * 0.20,
            'momentum_quality': momentum_score * 0.15,
            'oscillator_confluence': oscillator_confluence * 0.15,
            'volume_confluence': volume_confluence * 0.12,
            'timeframe_alignment': timeframe_confluence * 0.10,
            'fibonacci_levels': fibonacci_confluence * 0.05,
            'microstructure': microstructure_quality * 0.05
        }

        # Calculate final confidence with all advanced indicators
        final_confidence = base_confidence
        for factor, adjustment in confidence_adjustments.items():
            if direction in ['LONG', 'SHORT']:
                final_confidence += adjustment
                reasoning_parts.append(f"{factor}={adjustment:.3f}")

        # Risk-based confidence scaling
        final_confidence *= risk_metrics['confidence_multiplier']

        # Clamp confidence between 0.0 and 1.0
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Advanced confluence gates - multi-factor validation
        if direction in ['LONG', 'SHORT']:
            confluence_score = self._calculate_confluence_score(direction, ta, opp, regime, setup_strength)
            if confluence_score < 0.6:  # Strict confluence requirement
                reasoning_parts.append(f'confluence=fail({confluence_score:.2f})')
                logger.debug(f"📏 Advanced rule engine: {direction} rejected by confluence gates (score: {confluence_score:.2f})")
                return 'HOLD', 0.3, "; ".join(reasoning_parts)  # Low confidence HOLD
            else:
                reasoning_parts.append(f'confluence=pass({confluence_score:.2f})')

        # Additional confidence boost for high-quality setups
        if final_confidence > 0.8 and confluence_score > 0.8:
            final_confidence = min(1.0, final_confidence * 1.1)
            reasoning_parts.append('high_quality_boost')

        logger.debug(f"📏 Advanced rule engine decision: {direction} @ {final_confidence:.3f} ({'; '.join(reasoning_parts)})")
        return direction, final_confidence, "; ".join(reasoning_parts)

    def _detect_advanced_regime(self, ta: TechnicalAnalysis) -> Dict[str, Any]:
        """Advanced market regime detection with strength scoring."""
        bb_width = (ta.bb_upper - ta.bb_lower) / max(ta.bb_middle, 0.001)  # Prevent division by zero
        price_momentum = float(ta.price_change_24h or 0.0)
        volume_ratio = max(0.0, float(ta.volume_ratio or 0.0))

        # Multi-factor regime analysis
        trend_strength = abs(price_momentum) / 100
        volatility_score = min(bb_width * 5, 1.0)  # Normalize BB width
        volume_surge = min(volume_ratio / 2, 1.0)  # Volume vs average

        # Regime classification with strength
        if price_momentum > 5.0 and ta.bb_position > 0.7:
            return {'type': 'strong_uptrend', 'strength': trend_strength + 0.3}
        elif price_momentum < -5.0 and ta.bb_position < 0.3:
            return {'type': 'strong_downtrend', 'strength': trend_strength + 0.3}
        elif trend_strength > 0.02:
            regime_type = 'uptrend' if price_momentum > 0 else 'downtrend'
            return {'type': regime_type, 'strength': trend_strength + 0.1}
        elif volatility_score > 0.3:
            return {'type': 'volatile_ranging', 'strength': volatility_score}
        else:
            return {'type': 'quiet_ranging', 'strength': 0.2}

    def _analyze_setup_strength(self, ta: TechnicalAnalysis, opp: OpportunityScore, market_regime: Optional[str] = None) -> Dict[str, Any]:
        """Analyze setup patterns with strength scoring.

        `market_regime` is the BTC-macro regime (the same source token analysis consumes:
        BULL_TRENDING / BEAR_TRENDING / BULL_RANGING / BEAR_RANGING / SIDEWAYS), NOT a
        per-token 24h heuristic. We use it only to gently DISCOUNT counter-trend setups
        when the broad market is strongly TRENDING (e.g. don't reflexively short into a
        BTC bull trend). We do NOT boost with-trend setups — that would manufacture a long
        bias. In ranging/sideways regimes no weighting is applied, so mean reversion (a
        legitimate strategy) operates freely and the market itself decides direction.
        """
        rsi_oversold = ta.rsi_14 < 30
        rsi_overbought = ta.rsi_14 > 70
        rsi_neutral = 40 <= ta.rsi_14 <= 60

        macd_bullish = ta.macd_line > ta.macd_signal and ta.macd_histogram > 0
        macd_bearish = ta.macd_line < ta.macd_signal and ta.macd_histogram < 0

        bb_squeeze = (ta.bb_upper - ta.bb_lower) / ta.bb_middle < 0.1
        bb_expansion = (ta.bb_upper - ta.bb_lower) / ta.bb_middle > 0.2

        # Pattern scoring
        setup_scores = []

        # Oversold bounce setup
        if rsi_oversold and macd_bullish and ta.bb_position < 0.3:
            setup_scores.append(('oversold_bounce', 0.8))

        # Overbought reversal setup
        if rsi_overbought and macd_bearish and ta.bb_position > 0.7:
            setup_scores.append(('overbought_reversal', 0.8))

        fresh_momentum = self._fresh_momentum_base(ta.price_change_24h)

        # Breakout setup: require signed indicator agreement and early expansion.
        # A large trailing return by itself is evidence that the move happened,
        # not that a high-quality new entry remains.
        if bb_expansion and 1.0 <= abs(ta.price_change_24h) <= 10.0:
            if ta.price_change_24h > 0 and macd_bullish:
                setup_scores.append(('bullish_breakout', 0.55 + 0.20 * fresh_momentum))
            elif ta.price_change_24h < 0 and macd_bearish:
                setup_scores.append(('bearish_breakout', 0.55 + 0.20 * fresh_momentum))

        # Momentum continuation
        if fresh_momentum >= 0.25 and macd_bullish and ta.rsi_14 > 50 and ta.bb_position > 0.6:
            setup_scores.append(('bullish_momentum', 0.7))
        elif fresh_momentum >= 0.25 and macd_bearish and ta.rsi_14 < 50 and ta.bb_position < 0.4:
            setup_scores.append(('bearish_momentum', 0.7))

        # Mean reversion setup
        if rsi_neutral and bb_squeeze:
            setup_scores.append(('mean_reversion', 0.5))

        # Macro trend-alignment weighting. Only discount counter-trend setups when the
        # BTC market is STRONGLY trending; never boost (no manufactured long bias); leave
        # ranging/sideways untouched so mean reversion stays fully valid.
        if setup_scores and market_regime and getattr(settings, "RULE_TREND_ALIGN_ENABLED", True):
            mr = str(market_regime).upper()
            trend_bias = 1 if mr == 'BULL_TRENDING' else (-1 if mr == 'BEAR_TRENDING' else 0)
            if trend_bias != 0:
                boost = float(getattr(settings, "RULE_TREND_ALIGN_BOOST", 1.0) or 1.0)
                penalty = float(getattr(settings, "RULE_TREND_ALIGN_PENALTY", 0.80) or 0.80)
                bullish_patterns = {'oversold_bounce', 'bullish_breakout', 'bullish_momentum'}
                bearish_patterns = {'overbought_reversal', 'bearish_breakout', 'bearish_momentum'}
                weighted = []
                for pattern, score in setup_scores:
                    pat_dir = 1 if pattern in bullish_patterns else (-1 if pattern in bearish_patterns else 0)
                    if pat_dir == -trend_bias and pat_dir != 0:
                        score *= penalty  # counter-trend in a strong macro trend: deprioritise
                    elif pat_dir == trend_bias:
                        score *= boost     # with-trend: default 1.0 (no boost → no long bias)
                    weighted.append((pattern, score))
                setup_scores = weighted

        # Return best setup or neutral
        if setup_scores:
            best_pattern, best_score = max(setup_scores, key=lambda x: x[1])
            return {'pattern': best_pattern, 'score': min(1.0, best_score)}
        else:
            return {'pattern': 'no_clear_setup', 'score': 0.3}

    def _calculate_momentum_score(self, ta: TechnicalAnalysis) -> float:
        """Calculate fresh, direction-neutral momentum quality."""
        price_momentum = self._fresh_momentum_base(ta.price_change_24h)
        technical_momentum = max(0.0, min(1.0, float(ta.momentum_score or 0.0)))
        volume_momentum = max(0.0, min(1.0, float(ta.volume_ratio or 0.0) / 2.0))
        return min(
            1.0,
            price_momentum * 0.45
            + technical_momentum * 0.35
            + volume_momentum * 0.20,
        )

    def _analyze_volume_profile(self, ta: TechnicalAnalysis, opp: OpportunityScore) -> float:
        """Analyze volume confirmation and profile."""
        volume_ratio = max(0.0, float(ta.volume_ratio or 0.0))

        # Volume confirmation scoring
        if volume_ratio > 2.0:  # High volume surge
            return 0.9
        elif volume_ratio > 1.5:  # Above average volume
            return 0.7
        elif volume_ratio > 0.8:  # Normal volume
            return 0.5
        else:  # Low volume
            return 0.2

    def _calculate_risk_metrics(self, ta: TechnicalAnalysis, opp: OpportunityScore) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        # Volatility risk
        bb_width = (ta.bb_upper - ta.bb_lower) / ta.bb_middle
        volatility_risk = min(bb_width * 2, 1.0)

        # Price position risk (extreme positions are riskier)
        position_risk = abs(ta.bb_position - 0.5) * 2

        # Overall risk score (lower is better)
        total_risk = (volatility_risk + position_risk) / 2

        # Risk-adjusted scoring
        risk_adjusted_score = max(0.1, 1.0 - total_risk)

        # Confidence multiplier (reduce confidence for high risk)
        confidence_multiplier = max(0.5, 1.0 - total_risk * 0.3)

        return {
            'volatility_risk': volatility_risk,
            'position_risk': position_risk,
            'total_risk': total_risk,
            'risk_adjusted_score': risk_adjusted_score,
            'confidence_multiplier': confidence_multiplier
        }

    def _determine_base_signal(self, regime: Dict, setup_strength: Dict, momentum_score: float) -> Tuple[str, float]:
        """Determine base trading signal and confidence."""
        pattern = setup_strength['pattern']
        setup_score = setup_strength['score']

        # Bullish patterns
        if pattern in ['oversold_bounce', 'bullish_breakout', 'bullish_momentum']:
            base_confidence = 0.4 + (setup_score * 0.3) + (momentum_score * 0.2)
            return 'LONG', base_confidence

        # Bearish patterns
        elif pattern in ['overbought_reversal', 'bearish_breakout', 'bearish_momentum']:
            base_confidence = 0.4 + (setup_score * 0.3) + (momentum_score * 0.2)
            return 'SHORT', base_confidence

        # Neutral patterns
        else:
            return 'HOLD', 0.5

    def _calculate_confluence_score(self, direction: str, ta: TechnicalAnalysis, opp: OpportunityScore,
                                  regime: Dict, setup_strength: Dict) -> float:
        """Calculate confluence score for signal validation."""
        confluence_factors = []

        # RSI confluence
        if direction == 'LONG' and ta.rsi_14 < 70:
            confluence_factors.append(0.8)
        elif direction == 'SHORT' and ta.rsi_14 > 30:
            confluence_factors.append(0.8)
        else:
            confluence_factors.append(0.3)

        # MACD confluence
        if direction == 'LONG' and ta.macd_line > ta.macd_signal:
            confluence_factors.append(0.8)
        elif direction == 'SHORT' and ta.macd_line < ta.macd_signal:
            confluence_factors.append(0.8)
        else:
            confluence_factors.append(0.3)

        # Bollinger Band confluence
        if direction == 'LONG' and ta.bb_position < 0.8:
            confluence_factors.append(0.7)
        elif direction == 'SHORT' and ta.bb_position > 0.2:
            confluence_factors.append(0.7)
        else:
            confluence_factors.append(0.4)

        # Regime confluence
        regime_type = regime['type']
        if direction == 'LONG' and 'uptrend' in regime_type:
            confluence_factors.append(0.9)
        elif direction == 'SHORT' and 'downtrend' in regime_type:
            confluence_factors.append(0.9)
        else:
            confluence_factors.append(0.5)

        # Volume confluence
        volume_ratio = max(0.0, float(ta.volume_ratio or 0.0))
        if volume_ratio > 1.2:  # Above average volume
            confluence_factors.append(0.8)
        else:
            confluence_factors.append(0.4)

        # Return average confluence score
        return sum(confluence_factors) / len(confluence_factors)

    def _calculate_oscillator_confluence(self, ta: TechnicalAnalysis, direction: str) -> float:
        """Calculate confluence score from all oscillator indicators."""
        oscillator_scores = []

        # Stochastic %K/%D confluence
        if direction == 'LONG':
            if ta.stoch_k < 30 and ta.stoch_d < 30:  # Oversold
                oscillator_scores.append(0.9)
            elif ta.stoch_k < 50:
                oscillator_scores.append(0.7)
            else:
                oscillator_scores.append(0.3)
        elif direction == 'SHORT':
            if ta.stoch_k > 70 and ta.stoch_d > 70:  # Overbought
                oscillator_scores.append(0.9)
            elif ta.stoch_k > 50:
                oscillator_scores.append(0.7)
            else:
                oscillator_scores.append(0.3)

        # Williams %R confluence
        if direction == 'LONG' and ta.williams_r < -70:  # Oversold
            oscillator_scores.append(0.8)
        elif direction == 'SHORT' and ta.williams_r > -30:  # Overbought
            oscillator_scores.append(0.8)
        else:
            oscillator_scores.append(0.4)

        # CCI confluence
        if direction == 'LONG' and ta.cci_14 < -150:  # Strong oversold
            oscillator_scores.append(0.9)
        elif direction == 'LONG' and ta.cci_14 < -50:  # Mild oversold
            oscillator_scores.append(0.6)
        elif direction == 'SHORT' and ta.cci_14 > 150:  # Strong overbought
            oscillator_scores.append(0.9)
        elif direction == 'SHORT' and ta.cci_14 > 50:  # Mild overbought
            oscillator_scores.append(0.6)
        else:
            oscillator_scores.append(0.4)

        # Rate of Change momentum
        if direction == 'LONG' and ta.roc_10 > 2:  # Positive momentum
            oscillator_scores.append(0.7)
        elif direction == 'SHORT' and ta.roc_10 < -2:  # Negative momentum
            oscillator_scores.append(0.7)
        else:
            oscillator_scores.append(0.4)

        return sum(oscillator_scores) / len(oscillator_scores) if oscillator_scores else 0.5

    def _calculate_advanced_volume_confluence(self, ta: TechnicalAnalysis, direction: str) -> float:
        """Calculate confluence from advanced volume indicators."""
        volume_scores = []

        # Money Flow Index confluence
        if direction == 'LONG' and ta.money_flow_index < 30:  # Oversold money flow
            volume_scores.append(0.8)
        elif direction == 'SHORT' and ta.money_flow_index > 70:  # Overbought money flow
            volume_scores.append(0.8)
        else:
            volume_scores.append(0.5)

        # VWAP confluence (price vs institutional levels)
        vwap_distance = abs(ta.current_price - ta.vwap) / ta.current_price
        if vwap_distance < 0.02:  # Near VWAP
            volume_scores.append(0.8)
        elif vwap_distance < 0.05:  # Close to VWAP
            volume_scores.append(0.6)
        else:
            volume_scores.append(0.4)

        # Volume Profile POC confluence
        poc_distance = abs(ta.current_price - ta.volume_profile_poc) / ta.current_price
        if poc_distance < 0.03:  # Near Point of Control
            volume_scores.append(0.7)
        else:
            volume_scores.append(0.4)

        # On Balance Volume trend
        if direction == 'LONG' and ta.obv > 0:  # Positive OBV
            volume_scores.append(0.6)
        elif direction == 'SHORT' and ta.obv < 0:  # Negative OBV
            volume_scores.append(0.6)
        else:
            volume_scores.append(0.4)

        return sum(volume_scores) / len(volume_scores) if volume_scores else 0.5

    def _calculate_timeframe_confluence(self, ta: TechnicalAnalysis, direction: str) -> float:
        """Calculate multi-timeframe confluence."""
        timeframe_scores = []

        # Multi-timeframe RSI
        if direction == 'LONG':
            if ta.rsi_14 < 50 and ta.rsi_1h < 50:  # Both oversold
                timeframe_scores.append(0.8)
            elif ta.rsi_14 < 60 or ta.rsi_1h < 60:  # One oversold
                timeframe_scores.append(0.6)
            else:
                timeframe_scores.append(0.3)
        elif direction == 'SHORT':
            if ta.rsi_14 > 50 and ta.rsi_1h > 50:  # Both overbought
                timeframe_scores.append(0.8)
            elif ta.rsi_14 > 40 or ta.rsi_1h > 40:  # One overbought
                timeframe_scores.append(0.6)
            else:
                timeframe_scores.append(0.3)

        # Multi-timeframe MACD
        if direction == 'LONG':
            if ta.macd_line > ta.macd_signal and ta.macd_1h_line > ta.macd_1h_signal:
                timeframe_scores.append(0.9)
            elif ta.macd_line > ta.macd_signal or ta.macd_1h_line > ta.macd_1h_signal:
                timeframe_scores.append(0.6)
            else:
                timeframe_scores.append(0.3)
        elif direction == 'SHORT':
            if ta.macd_line < ta.macd_signal and ta.macd_1h_line < ta.macd_1h_signal:
                timeframe_scores.append(0.9)
            elif ta.macd_line < ta.macd_signal or ta.macd_1h_line < ta.macd_1h_signal:
                timeframe_scores.append(0.6)
            else:
                timeframe_scores.append(0.3)

        # Trend alignment score
        if ta.trend_alignment > 0.5:  # Strong bullish alignment
            timeframe_scores.append(0.8 if direction == 'LONG' else 0.2)
        elif ta.trend_alignment < -0.5:  # Strong bearish alignment
            timeframe_scores.append(0.8 if direction == 'SHORT' else 0.2)
        else:
            timeframe_scores.append(0.5)

        return sum(timeframe_scores) / len(timeframe_scores) if timeframe_scores else 0.5

    def _calculate_fibonacci_confluence(self, ta: TechnicalAnalysis, direction: str) -> float:
        """Calculate Fibonacci level confluence."""
        fib_scores = []
        price = ta.current_price

        # Check proximity to key Fibonacci levels
        fib_levels = [ta.fib_23_6, ta.fib_38_2, ta.fib_50_0, ta.fib_61_8]
        fib_weights = [0.6, 0.8, 1.0, 0.8]  # 50% level is most important

        for level, weight in zip(fib_levels, fib_weights):
            distance = abs(price - level) / price
            if distance < 0.01:  # Very close to Fib level
                if direction == 'LONG' and price < level:  # Below resistance, good for long
                    fib_scores.append(0.8 * weight)
                elif direction == 'SHORT' and price > level:  # Above support, good for short
                    fib_scores.append(0.8 * weight)
                else:
                    fib_scores.append(0.4 * weight)
            elif distance < 0.03:  # Near Fib level
                fib_scores.append(0.6 * weight)
            else:
                fib_scores.append(0.3 * weight)

        return sum(fib_scores) / sum(fib_weights) if fib_scores else 0.5

    def _calculate_microstructure_quality(self, ta: TechnicalAnalysis) -> float:
        """Calculate market microstructure quality score."""
        microstructure_scores = []

        # Spread quality (lower spread = better)
        if ta.spread_estimate < 1.0:  # Low spread
            microstructure_scores.append(0.8)
        elif ta.spread_estimate < 2.0:  # Medium spread
            microstructure_scores.append(0.6)
        else:  # High spread
            microstructure_scores.append(0.3)

        # Tick rule momentum (balanced is better)
        if 0.8 <= ta.tick_rule_momentum <= 1.2:  # Balanced
            microstructure_scores.append(0.8)
        elif 0.6 <= ta.tick_rule_momentum <= 1.4:  # Slightly imbalanced
            microstructure_scores.append(0.6)
        else:  # Highly imbalanced
            microstructure_scores.append(0.3)

        # Price efficiency (higher is better)
        if ta.price_efficiency > 0.7:  # High efficiency
            microstructure_scores.append(0.8)
        elif ta.price_efficiency > 0.5:  # Medium efficiency
            microstructure_scores.append(0.6)
        else:  # Low efficiency
            microstructure_scores.append(0.4)

        return sum(microstructure_scores) / len(microstructure_scores) if microstructure_scores else 0.5

    def _minimum_accept_confidence(self, opp: OpportunityScore, rule_conf: float) -> float:
        """
        Adaptive confidence threshold for directional acceptance.

        Uses a learning-derived base target and applies local context adjustments.
        """
        floor = float(getattr(settings, "SIGNAL_PUBLISH_CONFIDENCE_FLOOR", 0.65) or 0.65)
        threshold = float(self._adaptive_confidence_target or 0.65)
        if opp.overall_score >= 0.85 and rule_conf >= 0.65:
            threshold -= 0.04
        elif opp.overall_score >= 0.75 and rule_conf >= 0.55:
            threshold -= 0.02
        elif opp.overall_score < 0.55 or rule_conf < 0.45:
            threshold += 0.02
        # Hard publish floor: below this, historical resolved win rate is a coin flip.
        # Even the strongest-setup discount can't take us under the floor.
        return max(floor, min(0.80, threshold))

    def _conviction_score(self, ai_confidence: float, opp_score: float, rule_confidence: float) -> float:
        """Blend AI confidence, opportunity quality, and rule confidence into a bounded conviction score."""
        ai_c = max(0.0, min(1.0, float(ai_confidence or 0.0)))
        opp_c = max(0.0, min(1.0, float(opp_score or 0.0)))
        rule_c = max(0.0, min(1.0, float(rule_confidence or 0.0)))
        return max(0.0, min(1.0, 0.45 * ai_c + 0.35 * opp_c + 0.20 * rule_c))

    async def _refresh_execution_fill_model(self, force: bool = False) -> None:
        """Periodically retrain fill probability from terminal production orders."""
        if not getattr(self, "supabase", None):
            return
        now = datetime.now(timezone.utc)
        refresh_hours = max(
            1,
            int(getattr(settings, "YUKI_FILL_CALIBRATION_REFRESH_HOURS", 6) or 6),
        )
        if (
            not force
            and getattr(self, "_fill_model_last_attempt_utc", None)
            and now - self._fill_model_last_attempt_utc < timedelta(hours=refresh_hours)
        ):
            return
        self._fill_model_last_attempt_utc = now
        try:
            await asyncio.to_thread(
                self._fill_calibrator.train_from_supabase,
                self.supabase,
                int(getattr(settings, "YUKI_FILL_CALIBRATION_LOOKBACK_DAYS", 90) or 90),
            )
        except Exception as exc:
            logger.warning("Could not refresh Hyperliquid fill calibration: %s", exc)

    def _attach_signal_edge_estimate(
        self,
        ai: AIDecision,
        ta: TechnicalAnalysis,
        opp: OpportunityScore,
        rule_direction: str,
        rule_confidence: float,
        signal_age_hours: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Convert the numerical policy output and trade geometry into net EV.

        Raw LLM confidence remains an input/audit field; it is not the published
        score. When the trained logistic policy is unavailable, confidence is
        strongly shrunk toward 0.5 and cannot bypass the normal confidence gate.
        """
        entry = max(float(ai.entry_price or 0.0), 1e-12)
        reward_pct = abs(float(ai.target_1 or entry) - entry) / entry * 100.0
        risk_pct = abs(float(ai.stop_loss or entry) - entry) / entry * 100.0
        atr_pct = abs(float(ta.atr_14 or 0.0)) / entry * 100.0
        entry_distance_pct = abs(entry - float(ta.current_price or entry)) / entry * 100.0

        policy_probability = getattr(ai, "policy_win_probability", None)
        if policy_probability is not None:
            direction_probability = max(0.02, min(0.98, float(policy_probability)))
            probability_source = "calibrated_logistic_policy"
            evidence_samples = int(
                getattr(self._signal_policy_bundle, "n_samples", 0) or 0
            )
            policy_validation = dict(
                getattr(self._signal_policy_bundle, "validation_metrics", {}) or {}
            )
        else:
            raw_confidence = max(0.0, min(1.0, float(ai.confidence or 0.5)))
            direction_probability = 0.5 + (raw_confidence - 0.5) * 0.35
            probability_source = "cold_start_shrunk_llm_confidence"
            evidence_samples = 0
            policy_validation = {}

        aligned = (
            str(rule_direction or "").upper() == str(ai.recommendation or "").upper()
            and str(ai.recommendation or "").upper() in {"LONG", "SHORT"}
        )
        if aligned:
            direction_probability = min(
                0.98,
                direction_probability + 0.04 * max(0.0, min(1.0, float(rule_confidence or 0.0))),
            )
        elif str(rule_direction or "").upper() in {"LONG", "SHORT"}:
            direction_probability = max(0.02, direction_probability - 0.04)

        # Conditional TP-before-SL probability: distant targets are harder even
        # when direction is correct. Keep the adjustment bounded and auditable.
        geometry_ratio = reward_pct / max(risk_pct, 0.01)
        geometry_penalty = max(0.0, geometry_ratio - 1.25) * 0.025
        target_before_stop = max(
            0.02,
            min(0.98, direction_probability - min(0.12, geometry_penalty)),
        )

        structure = ai.market_structure_context or {}
        order_flow = structure.get("order_flow") or {}
        orderbook = order_flow.get("orderbook") or {}
        bid_depth = max(
            0.0,
            float(orderbook.get("bid_depth_usdt") or orderbook.get("bid_depth_usd") or 0.0),
        )
        ask_depth = max(
            0.0,
            float(orderbook.get("ask_depth_usdt") or orderbook.get("ask_depth_usd") or 0.0),
        )
        positive_depths = [depth for depth in (bid_depth, ask_depth) if depth > 0]
        depth_usd = min(positive_depths) if positive_depths else 25_000.0
        opportunity_breakdown = getattr(opp, "breakdown", None) or {}
        setup_type = str(opportunity_breakdown.get("setup_type") or "unknown")
        fill_features = {
            "entry_distance_pct": entry_distance_pct,
            "signal_age_hours": max(0.0, float(signal_age_hours or 0.0)),
            "spread_pct": float(
                orderbook.get("spread_pct") or orderbook.get("spread_percentage") or 0.08
            ),
            "depth_usd": depth_usd,
            "volatility_pct": atr_pct,
            "side": ai.recommendation,
            "setup_type": setup_type,
            "entry_strategy": ai.entry_strategy,
        }
        fill_calibrator = getattr(self, "_fill_calibrator", None)
        if fill_calibrator is None:
            fill_calibrator = ExecutionFillCalibrator(
                minimum_samples=int(
                    getattr(settings, "YUKI_FILL_CALIBRATION_MIN_SAMPLES", 30) or 30
                )
            )
            self._fill_calibrator = fill_calibrator
        fill_probability, fill_model_metadata = fill_calibrator.predict(fill_features)
        funding_rate = abs(float(order_flow.get("funding_rate") or 0.0))
        horizon_hours = float(
            self._extract_range_upper_hours(ai.time_horizon)
            or self._calculate_validity_window_hours(ai.time_horizon)
            or 8.0
        )
        funding_cost_pct = funding_rate * 100.0 * (horizon_hours / 8.0)
        execution_cost_pct = float(self._round_trip_cost_pct) + funding_cost_pct
        conditional_ev = (
            target_before_stop * reward_pct
            - (1.0 - target_before_stop) * risk_pct
            - execution_cost_pct
        )
        net_ev = fill_probability * conditional_ev
        edge_score = max(
            0.0,
            min(100.0, 50.0 + 45.0 * float(np.tanh(net_ev / max(risk_pct, 0.50)))),
        )

        estimate = {
            "version": "net_ev_v2_production_fill",
            "probability_source": probability_source,
            "evidence_samples": evidence_samples,
            "policy_validation_status": policy_validation.get("status"),
            "policy_validation_samples": policy_validation.get("validation_samples", 0),
            "policy_validation_brier_score": policy_validation.get("brier_score"),
            "policy_validation_baseline_brier_score": policy_validation.get(
                "baseline_brier_score"
            ),
            "direction_probability": round(direction_probability, 6),
            "fill_probability": round(fill_probability, 6),
            "target_before_stop_probability": round(target_before_stop, 6),
            "reward_pct": round(reward_pct, 6),
            "risk_pct": round(risk_pct, 6),
            "entry_distance_pct": round(entry_distance_pct, 6),
            "original_signal_age_hours": round(float(signal_age_hours or 0.0), 6),
            "atr_pct": round(atr_pct, 6),
            "spread_pct": round(float(fill_features["spread_pct"]), 6),
            "depth_usd": round(depth_usd, 2),
            "setup_type": setup_type,
            "entry_strategy": str(ai.entry_strategy or ""),
            "execution_cost_pct": round(execution_cost_pct, 6),
            "funding_cost_pct": round(funding_cost_pct, 6),
            "conditional_net_ev_pct": round(conditional_ev, 6),
            "net_expected_value_pct": round(net_ev, 6),
            "edge_score": round(edge_score, 4),
            "ai_rule_agreement": aligned,
            "rule_direction": str(rule_direction or "HOLD").upper(),
            "rule_confidence": round(float(rule_confidence or 0.0), 6),
            **fill_model_metadata,
        }
        ai.edge_estimate = estimate
        ai.calibrated_direction_probability = direction_probability
        ai.fill_probability = fill_probability
        ai.target_before_stop_probability = target_before_stop
        ai.net_expected_value_pct = net_ev
        ai.edge_score = edge_score
        return estimate

    @staticmethod
    def _calibrated_edge_allows_below_confidence(
        ai: Optional[AIDecision],
        rule_direction: str,
    ) -> bool:
        """Allow a low raw-confidence candidate only with trained positive EV and agreement."""
        if not ai or not ai.edge_estimate:
            return False
        estimate = ai.edge_estimate
        return (
            estimate.get("probability_source") == "calibrated_logistic_policy"
            and estimate.get("policy_validation_status") == "passed"
            and int(estimate.get("evidence_samples") or 0) >= 20
            and bool(estimate.get("ai_rule_agreement"))
            and str(rule_direction or "").upper() == str(ai.recommendation or "").upper()
            and float(estimate.get("net_expected_value_pct") or 0.0)
            > float(getattr(settings, "SIGNAL_MIN_NET_EXPECTED_VALUE_PCT", 0.0) or 0.0)
        )

    def _dynamic_min_net_rr_threshold(
        self,
        *,
        ai_confidence: float,
        opp_score: float,
        rule_confidence: float,
        regime_type: str,
        volatility_24h: float,
    ) -> float:
        """
        Dynamic RR floor driven by conviction + market conditions.
        Higher conviction/trend can tolerate lower RR; noisy/volatile regimes require higher RR.
        """
        base = max(float(getattr(settings, "SIGNAL_MIN_NET_RR_THRESHOLD", 1.15) or 1.15), 0.5)
        conviction = self._conviction_score(ai_confidence, opp_score, rule_confidence)
        threshold = base

        if conviction >= 0.78:
            threshold -= 0.18
        elif conviction >= 0.68:
            threshold -= 0.10
        elif conviction >= 0.58:
            threshold -= 0.05
        elif conviction < 0.45:
            threshold += 0.08

        regime = str(regime_type or "").lower()
        if "volatile" in regime:
            threshold += 0.05
        elif "ranging" in regime:
            threshold += 0.03
        elif any(tag in regime for tag in ("strong_uptrend", "strong_downtrend", "uptrend", "downtrend")):
            threshold -= 0.03

        vol = max(float(volatility_24h or 0.0), 0.0)
        if vol >= 10.0:
            threshold += 0.05
        elif vol <= 4.0:
            threshold -= 0.02

        return max(0.90, min(1.60, threshold))

    def _dynamic_min_sl_distance_pct(
        self,
        *,
        ai_confidence: float,
        opp_score: float,
        rule_confidence: float,
        regime_type: str,
        volatility_24h: float,
    ) -> float:
        """
        Dynamic SL distance floor.
        Strong conviction allows a tighter floor; volatile/noisy contexts widen it.
        """
        base = max(float(getattr(settings, "SIGNAL_MIN_SL_DISTANCE_PCT", 0.015) or 0.015), 0.005)
        conviction = self._conviction_score(ai_confidence, opp_score, rule_confidence)
        floor = base

        if conviction >= 0.78:
            floor -= 0.004
        elif conviction >= 0.68:
            floor -= 0.0025
        elif conviction >= 0.58:
            floor -= 0.001
        elif conviction < 0.45:
            floor += 0.003

        regime = str(regime_type or "").lower()
        if "volatile" in regime:
            floor += 0.0015
        elif "ranging" in regime:
            floor += 0.001
        elif any(tag in regime for tag in ("strong_uptrend", "strong_downtrend")):
            floor -= 0.001

        vol = max(float(volatility_24h or 0.0), 0.0)
        if vol >= 12.0:
            floor += 0.004
        elif vol >= 8.0:
            floor += 0.002
        elif vol <= 4.0:
            floor -= 0.001

        return max(0.008, min(0.03, floor))

    def _policy_bundle_path(self) -> Path:
        override = getattr(settings, "POLICY_MODEL_PATH", None)
        if override:
            return Path(str(override))
        return default_policy_bundle_path()

    def reload_signal_policy_model(self) -> None:
        """Load or hot-reload sklearn policy bundle from disk when POLICY_LEARNING_ENABLED."""
        if not settings.POLICY_LEARNING_ENABLED:
            self._signal_policy_bundle = None
            return
        path = self._policy_bundle_path()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            # File missing (e.g. after a cold restart) — try DB recovery first
            if self.supabase:
                recovered = load_policy_bundle_from_db(self.supabase, "yuki", path)
                if recovered:
                    self._signal_policy_bundle = recovered
                    self._signal_policy_source_path = path
                    try:
                        self._signal_policy_mtime = path.stat().st_mtime
                    except OSError:
                        self._signal_policy_mtime = 0.0
                    logger.info("Policy bundle recovered from DB for agent=yuki")
                    return
            self._signal_policy_bundle = None
            if settings.POLICY_AUTO_TRAIN_ENABLED:
                logger.warning(
                    "Signal policy bundle missing at %s — auto-train will create it when enough resolved signals exist",
                    path,
                )
            else:
                logger.warning(
                    "Signal policy bundle missing at %s — run python train_signal_policy.py or enable POLICY_AUTO_TRAIN_ENABLED",
                    path,
                )
            return
        if (
            self._signal_policy_bundle is not None
            and self._signal_policy_source_path == path
            and mtime <= self._signal_policy_mtime
        ):
            return
        bundle = load_policy_bundle(path)
        self._signal_policy_bundle = bundle
        self._signal_policy_source_path = path
        self._signal_policy_mtime = mtime
        if bundle:
            validation = bundle.validation_metrics or {}
            logger.info(
                "Signal policy loaded: %s (train_n=%s, hist_win_rate=%.3f, validation=%s)",
                path,
                bundle.n_samples,
                bundle.win_rate,
                validation.get("status", "legacy_unvalidated"),
            )

    def _maybe_auto_train_policy_sync(self) -> None:
        """
        Retrain policy bundle from Supabase on a schedule (no manual CLI).

        Runs synchronously; call via asyncio.to_thread from the analysis cycle.
        """
        if not settings.POLICY_AUTO_TRAIN_ENABLED or not self.supabase:
            return

        path = self._policy_bundle_path()
        now = datetime.now(timezone.utc)
        last = self._policy_train_last_attempt_utc
        interval_h = max(1, int(getattr(settings, "POLICY_AUTO_TRAIN_INTERVAL_HOURS", 24) or 24))
        retry_h = max(1, int(getattr(settings, "POLICY_AUTO_TRAIN_RETRY_HOURS_WHEN_MISSING", 6) or 6))
        min_between_sec = 3600  # avoid stacking sklearn fits if cycles run often

        if last and (now - last).total_seconds() < min_between_sec:
            return

        try:
            exists = path.is_file()
        except OSError:
            exists = False

        if exists:
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                mtime = None
            if mtime and (now - mtime).total_seconds() < interval_h * 3600:
                return
        else:
            if last and (now - last).total_seconds() < retry_h * 3600:
                return

        self._policy_train_last_attempt_utc = now
        lookback = max(int(getattr(settings, "POLICY_AUTO_TRAIN_LOOKBACK_DAYS", 90) or 90), 14)
        min_samples = max(int(getattr(settings, "POLICY_AUTO_TRAIN_MIN_SAMPLES", 60) or 60), 20)

        try:
            bundle = train_policy_from_supabase(
                self.supabase,
                lookback_days=lookback,
                min_samples=min_samples,
                save_path=path,
                agent_type="yuki",
            )
            if bundle:
                logger.info(
                    "Policy auto-train saved %s (n=%s hist_win_rate=%.3f)",
                    path,
                    bundle.n_samples,
                    bundle.win_rate,
                )
                self._signal_policy_mtime = 0.0
                self.reload_signal_policy_model()
            else:
                logger.debug(
                    "Policy auto-train: no new bundle (need >=%s resolved outcomes in last %sd)",
                    min_samples,
                    lookback,
                )
        except Exception as e:
            logger.warning("Policy auto-train failed: %s", e)

    def _eval_policy_win_prob(
        self,
        ta: TechnicalAnalysis,
        opp: OpportunityScore,
        rule_conf: float,
        regime_type: str,
        eval_direction: str,
        signal_confidence: Optional[float] = None,
    ) -> Optional[float]:
        if not settings.POLICY_LEARNING_ENABLED or not self._signal_policy_bundle:
            return None
        if (
            self._signal_policy_bundle.validation_metrics or {}
        ).get("status") != "passed":
            # Legacy and failed candidates remain loadable for audit, but they
            # cannot influence live entries until they pass unseen time folds.
            return None
        try:
            price = max(float(ta.current_price or 0.0), 1e-12)
            atr_pct = min(1.0, float(ta.atr_14 or 0.0) / price * 100.0)
            sc = float(signal_confidence) if signal_confidence is not None else float(rule_conf)
            feats = build_policy_feature_vector(
                rsi_14=float(ta.rsi_14),
                bb_position=float(ta.bb_position),
                macd_histogram=float(ta.macd_histogram),
                momentum_score=float(ta.momentum_score),
                strength_score=float(ta.strength_score),
                stoch_k=float(ta.stoch_k),
                trend_alignment=float(ta.trend_alignment),
                price_efficiency=float(ta.price_efficiency),
                volatility_24h=float(ta.volatility_24h),
                atr_pct=atr_pct,
                opp_overall=float(opp.overall_score),
                opp_volume=float(opp.volume_score),
                opp_momentum=float(opp.momentum_score),
                opp_trend=float(opp.trend_score),
                opp_institutional=float(opp.institutional_score),
                rule_confidence=float(rule_conf),
                eval_direction=eval_direction,
                regime_type=regime_type,
                signal_confidence=sc,
            )
            return self._signal_policy_bundle.predict_win_probability(feats)
        except Exception as e:
            logger.debug("Policy win-prob eval failed: %s", e)
            return None

    def _policy_tune_min_conf(self, base_min: float, p_win: Optional[float]) -> float:
        if p_win is None:
            return base_min
        scale = max(0.0, min(0.2, float(getattr(settings, "POLICY_MIN_CONF_ADJUST_SCALE", 0.08) or 0.08)))
        adj = scale * (float(p_win) - 0.5)
        return max(0.52, min(0.70, base_min - adj))

    def _final_policy_allows(self, symbol: str, ta: TechnicalAnalysis, opp: OpportunityScore, rule_conf: float, regime_type: str, ai: AIDecision) -> bool:
        if not settings.POLICY_LEARNING_ENABLED or not self._signal_policy_bundle:
            return True
        if ai.recommendation not in ("LONG", "SHORT"):
            return True
        p = self._eval_policy_win_prob(
            ta, opp, rule_conf, regime_type, ai.recommendation, signal_confidence=ai.confidence
        )
        if p is None:
            return True
        thresh = float(getattr(settings, "POLICY_MIN_WIN_PROB_FOR_ACCEPT", 0.42) or 0.42)
        if p < thresh:
            logger.info(
                "❌ %s: Policy guardrail rejected %s (p_win=%.3f < %.3f)",
                symbol,
                ai.recommendation,
                p,
                thresh,
            )
            return False
        ai.policy_win_probability = p
        ai.policy_eval_direction = ai.recommendation
        ai.policy_feature_version = POLICY_FEATURE_VERSION
        return True

    def _guardrail_policy_or_none(
        self,
        symbol: str,
        ta: TechnicalAnalysis,
        opp: OpportunityScore,
        rule_dir: str,
        rule_conf: float,
        regime_type: str,
        ai: AIDecision,
        min_conf: float,
        normalized_symbol: str,
    ) -> Optional[AIDecision]:
        if getattr(settings, "GATE_ENABLE_POLICY_POST_AI", False) and not self._final_policy_allows(symbol, ta, opp, rule_conf, regime_type, ai):
            reason = (
                f"Offline policy P(win) below minimum "
                f"{float(getattr(settings, 'POLICY_MIN_WIN_PROB_FOR_ACCEPT', 0.42) or 0.42):.2f}"
            )
            logger.info(f"🚫 {symbol}: guardrail BLOCKED — policy_low_win_probability: {reason}")
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="policy_low_win_probability",
                rejection_reason=reason,
                min_confidence=min_conf,
            )
            return None

        if ai.recommendation in {"LONG", "SHORT"} and not ai.edge_estimate:
            self._attach_signal_edge_estimate(
                ai,
                ta,
                opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
            )

        # Once the conventional policy has enough resolved examples, the final
        # decision is expected value—not the LLM's self-reported confidence.
        estimate = ai.edge_estimate or {}
        ev_threshold = float(
            getattr(settings, "SIGNAL_MIN_NET_EXPECTED_VALUE_PCT", 0.0) or 0.0
        )
        if reliable_negative_edge_veto(estimate, threshold_pct=ev_threshold):
            reason = (
                f"Calibrated net EV {float(estimate.get('net_expected_value_pct') or 0.0):+.3f}% "
                f"<= {ev_threshold:+.3f}% after fill probability, fees, funding and slippage"
            )
            logger.info("🚫 %s: calibrated EV guardrail blocked — %s", symbol, reason)
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="calibrated_net_ev_below_threshold",
                rejection_reason=reason,
                min_confidence=min_conf,
                regime_type=regime_type,
            )
            return None

        # Net RR floor is conviction-aware (bounded by safety clamps).
        min_net_rr = self._dynamic_min_net_rr_threshold(
            ai_confidence=float(ai.confidence or 0.0),
            opp_score=float(opp.overall_score or 0.0),
            rule_confidence=float(rule_conf or 0.0),
            regime_type=regime_type,
            volatility_24h=float(ta.volatility_24h or 0.0),
        )
        net_rr = self._net_rr_after_costs(
            entry=float(ai.entry_price or 0.0),
            target=float(ai.target_1 or 0.0),
            stop=float(ai.stop_loss or 0.0),
            direction=ai.recommendation,
            leverage=float(ai.leverage or 1.0),
        )
        if net_rr > 0.0 and net_rr < min_net_rr:
            logger.info(
                f"🟡 {symbol}: soft RR warning — net_rr_below_threshold: "
                f"RR={net_rr:.2f} < {min_net_rr:.2f} "
                f"(entry={ai.entry_price}, t1={ai.target_1}, sl={ai.stop_loss}, {ai.leverage}x)"
            )
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="net_rr_below_threshold",
                rejection_reason=f"Net RR after fees+slippage {net_rr:.2f} < {min_net_rr:.2f}",
                min_confidence=min_conf,
            )
            if ai.edge_estimate is not None:
                soft_flags = list(ai.edge_estimate.get("soft_flags") or [])
                soft_flags.append("net_rr_below_dynamic_reference")
                ai.edge_estimate["soft_flags"] = list(dict.fromkeys(soft_flags))

        # Attribution-based combo blocking: if (regime, direction, leverage_bucket) has 20+ samples
        # with WR<40% and avg PnL<0, suppress new signals in that combo.
        combo_blocked, combo_reason = self._attribution_combo_blocked(
            regime_type, ai.recommendation, float(ai.leverage or 1.0)
        )
        if getattr(settings, "GATE_ENABLE_ATTRIBUTION_COMBO", False) and combo_blocked:
            logger.info(f"🚫 {symbol}: guardrail BLOCKED — attribution_combo_underperforming: {combo_reason}")
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="attribution_combo_underperforming",
                rejection_reason=combo_reason,
                min_confidence=min_conf,
            )
            return None

        # Drawdown circuit breaker — if portfolio in deep drawdown, halt new signals.
        if self._drawdown_paused:
            logger.info(f"🚫 {symbol}: guardrail BLOCKED — drawdown_circuit_breaker: portfolio drawdown >= 20%")
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="drawdown_circuit_breaker",
                rejection_reason="Drawdown >= 20% — new signals paused",
                min_confidence=min_conf,
            )
            return None

        if normalized_symbol:
            self._last_rejection_context.pop(normalized_symbol, None)
        return ai

    async def _refresh_signal_outcome_confidence_context(self) -> None:
        """
        Build confidence adjustment from actual resolved platform signal outcomes.

        This complements rejected-candidate learning by reflecting live signal quality.
        """
        try:
            if not self.supabase:
                self._signal_quality_conf_adjustment = 0.0
                self._signal_quality_state = {
                    "adjustment": 0.0,
                    "applied": False,
                    "reason": "missing_supabase",
                    "updated_at": datetime.utcnow().isoformat(),
                }
                return

            lookback_days = max(int(settings.REJECT_TUNING_LOOKBACK_DAYS or 30), 14)
            lookback_start = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
            result = (
                self.supabase.from_("platform_signals")
                .select("signal_id, status, leverage, direction, market_conditions")
                .gte("created_at", lookback_start)
                .order("created_at", desc=True)
                .limit(3000)
                .execute()
            )
            rows = result.data or []

            terminal_statuses = {"hit_target_1", "hit_target_2", "hit_stop_loss", "expired"}
            informative_statuses = {"hit_target_1", "hit_target_2", "hit_stop_loss"}
            resolved_rows = [r for r in rows if str(r.get("status") or "").lower() in terminal_statuses]
            # Expired signals are ambiguous (not wins, not confirmed losses) — exclude them from
            # win-rate, avg-PnL, and attribution calculations so they don't drag metrics down.
            informative_rows = [r for r in resolved_rows if str(r.get("status") or "").lower() in informative_statuses]
            expired_count = len(resolved_rows) - len(informative_rows)

            min_resolved = max(int(settings.REJECT_TUNING_MIN_RESOLVED or 30), 10)
            if len(resolved_rows) < min_resolved:
                self._signal_quality_conf_adjustment = 0.0
                self._signal_quality_state = {
                    "resolved_signals": len(resolved_rows),
                    "informative_signals": len(informative_rows),
                    "expired_signals": expired_count,
                    "expired_rate": round(float(expired_count / len(resolved_rows)), 4) if resolved_rows else 0.0,
                    "min_resolved_required": min_resolved,
                    "adjustment": 0.0,
                    "applied": False,
                    "reason": "insufficient_signal_samples",
                    "lookback_days": lookback_days,
                    "updated_at": datetime.utcnow().isoformat(),
                }
                return

            # Warn if most signals are expiring — usually means validity window is too tight.
            if expired_count > 0 and len(resolved_rows) > 0:
                expired_rate = expired_count / len(resolved_rows)
                if expired_rate >= 0.40:
                    logger.warning(
                        "⚠️ High expiry rate: %d/%d resolved signals expired (%.0f%%) — "
                        "validity windows may be too tight",
                        expired_count, len(resolved_rows), expired_rate * 100,
                    )

            if len(informative_rows) < 10:
                self._signal_quality_conf_adjustment = 0.0
                self._signal_quality_state = {
                    "resolved_signals": len(resolved_rows),
                    "informative_signals": len(informative_rows),
                    "expired_signals": expired_count,
                    "expired_rate": round(float(expired_count / len(resolved_rows)), 4) if resolved_rows else 0.0,
                    "adjustment": 0.0,
                    "applied": False,
                    "reason": "insufficient_informative_signals",
                    "lookback_days": lookback_days,
                    "updated_at": datetime.utcnow().isoformat(),
                }
                return

            wins = 0
            peak_fallback_rows = 0
            leveraged_pnl_values: List[float] = []
            resolved_signal_ids = [str(r.get("signal_id") or "") for r in informative_rows if r.get("signal_id")]
            perf_by_signal: Dict[str, Dict[str, Any]] = {}
            if resolved_signal_ids and self.supabase:
                try:
                    perf_result = (
                        self.supabase.from_("platform_signal_performance_tracking")
                        .select("signal_id, max_profit_reached, max_loss_reached, leveraged_pnl_percent")
                        .in_("signal_id", resolved_signal_ids)
                        .execute()
                    )
                    perf_by_signal = {
                        str(p.get("signal_id")): p
                        for p in (perf_result.data or [])
                        if p.get("signal_id")
                    }
                except Exception as perf_err:
                    logger.debug(f"Signal-quality tuning perf fetch fallback: {perf_err}")

            # Reset attribution & drift tables before re-ingesting (avoid double-counting on each refresh).
            self._attribution_stats = {}
            self._recent_outcomes = []
            self._peak_rolling_win_rate = 0.0
            self._concept_drift_alerted = False

            for row in informative_rows:  # expired excluded — ambiguous outcomes skew metrics
                status = str(row.get("status") or "").lower()
                signal_id = str(row.get("signal_id") or "")
                is_win = status in {"hit_target_1", "hit_target_2"}
                if is_win:
                    wins += 1
                perf_row = perf_by_signal.get(signal_id, {})
                try:
                    max_profit = float(perf_row.get("max_profit_reached") or 0.0)
                except (TypeError, ValueError):
                    max_profit = 0.0
                try:
                    max_loss = float(perf_row.get("max_loss_reached") or 0.0)
                except (TypeError, ValueError):
                    max_loss = 0.0
                try:
                    realized = perf_row.get("leveraged_pnl_percent")
                    realized = float(realized) if realized is not None else None
                except (TypeError, ValueError):
                    realized = None

                # Learn from the realized exit, not the excursion. Feeding
                # max_profit_reached into learning credited every winner with
                # the best price it ever touched — a result no exit plan
                # actually banks — inflating avg PnL and every adjustment
                # derived from it. Peak/max-loss remain only as a fallback for
                # legacy rows that never recorded a realized exit, with the
                # sign check guarding against mislabeled rows.
                if status in {"hit_target_1", "hit_target_2"}:
                    if realized is not None and realized > 0:
                        pnl_value = realized
                    else:
                        pnl_value = max_profit if max_profit > 0 else 0.0
                        if pnl_value > 0:
                            peak_fallback_rows += 1
                elif status == "hit_stop_loss":
                    if realized is not None and realized < 0:
                        pnl_value = realized
                    else:
                        pnl_value = max_loss if max_loss < 0 else 0.0
                        if pnl_value < 0:
                            peak_fallback_rows += 1
                else:
                    pnl_value = 0.0
                leveraged_pnl_values.append(pnl_value)

                # Sliced attribution + concept drift recording — only for genuinely terminal trades.
                if status in {"hit_target_1", "hit_target_2", "hit_stop_loss"}:
                    try:
                        direction = str(row.get("direction") or "").upper()
                        leverage = float(row.get("leverage") or 1.0)
                        mc = row.get("market_conditions") or {}
                        if isinstance(mc, str):
                            try:
                                mc = json.loads(mc)
                            except Exception:
                                mc = {}
                        regime = str((mc or {}).get("trend_direction") or "unknown").lower()
                        self._record_outcome_attribution(
                            regime=regime,
                            direction=direction,
                            leverage=leverage,
                            pnl_pct_leveraged=float(pnl_value),
                            is_win=bool(is_win),
                        )
                        self._record_recent_outcome(is_win=bool(is_win), pnl_pct=float(pnl_value))
                    except Exception as attr_err:
                        logger.debug(f"Attribution record skipped for {signal_id}: {attr_err}")

            win_rate = wins / len(informative_rows)
            avg_leveraged_pnl = (
                sum(leveraged_pnl_values) / len(leveraged_pnl_values)
                if leveraged_pnl_values
                else 0.0
            )

            target_win_rate = 0.52
            target_avg_leveraged_pnl = 0.10
            win_error = target_win_rate - win_rate
            pnl_error = target_avg_leveraged_pnl - avg_leveraged_pnl
            raw_adjustment = (win_error * 0.10) + (pnl_error * 0.015)
            clipped_adjustment = max(-0.03, min(0.03, raw_adjustment))
            smoothing = max(0.0, min(1.0, float(settings.REJECT_TUNING_SMOOTHING or 0.35)))
            smoothed_adjustment = (
                (1.0 - smoothing) * float(self._signal_quality_conf_adjustment or 0.0)
                + smoothing * clipped_adjustment
            )
            self._signal_quality_conf_adjustment = max(-0.03, min(0.03, smoothed_adjustment))
            self._signal_quality_state = {
                "resolved_signals": len(resolved_rows),
                "informative_signals": len(informative_rows),
                "expired_signals": expired_count,
                "expired_rate": round(float(expired_count / len(resolved_rows)), 4) if resolved_rows else 0.0,
                "win_rate": round(float(win_rate), 4),
                "win_rate_sample_size": len(informative_rows),
                "target_win_rate": round(float(target_win_rate), 4),
                "avg_leveraged_pnl_pct": round(float(avg_leveraged_pnl), 4),
                "pnl_source": "realized_exit",
                "peak_fallback_rows": peak_fallback_rows,
                "target_avg_leveraged_pnl_pct": round(float(target_avg_leveraged_pnl), 4),
                "raw_adjustment": round(float(raw_adjustment), 5),
                "adjustment": round(float(self._signal_quality_conf_adjustment), 5),
                "applied": True,
                "lookback_days": lookback_days,
                "updated_at": datetime.utcnow().isoformat(),
            }
        except Exception as signal_quality_error:
            logger.warning(f"⚠️ Signal-quality confidence tuning failed: {signal_quality_error}")
            self._signal_quality_conf_adjustment = 0.0
            self._signal_quality_state = {
                "adjustment": 0.0,
                "applied": False,
                "reason": "signal_quality_tuning_error",
                "error": str(signal_quality_error),
                "updated_at": datetime.utcnow().isoformat(),
            }

    def _refresh_adaptive_confidence_target(self) -> None:
        """Blend learning-driven components into one base confidence target."""
        reject_adj = float(self._reject_quality_conf_adjustment or 0.0)
        signal_adj = float(self._signal_quality_conf_adjustment or 0.0)
        floor = float(getattr(settings, "SIGNAL_PUBLISH_CONFIDENCE_FLOOR", 0.65) or 0.65)
        # Centre the learning-driven target above the publish floor; never let auto-tuning
        # drift it back down into coin-flip territory.
        self._adaptive_confidence_target = max(floor, min(0.80, 0.60 + reject_adj + signal_adj))

    def _should_use_reduced_conservatism(
        self,
        regime_type: str,
        opp: OpportunityScore,
        rule_conf: float,
    ) -> bool:
        """
        Gate permissive ensemble mode using learning outcomes and market regime.
        """
        state = self._reject_quality_tuning_state if isinstance(self._reject_quality_tuning_state, dict) else {}
        min_resolved = max(int(settings.REJECT_TUNING_MIN_RESOLVED or 30), 5)
        informative_resolved = int(state.get("informative_resolved") or 0)

        target_bad_rate = max(0.05, min(0.95, float(settings.REJECT_TUNING_TARGET_BAD_RATE or 0.30)))
        target_reject_pnl_lev = float(settings.REJECT_TUNING_TARGET_LEVERAGED_PNL_PCT or -0.20)
        bad_rate_raw = state.get("bad_reject_rate")
        avg_reject_pnl_lev_raw = state.get("avg_reject_pnl_pct_leveraged")
        try:
            bad_rate = float(bad_rate_raw) if bad_rate_raw is not None else None
        except (TypeError, ValueError):
            bad_rate = None
        try:
            avg_reject_pnl_lev = float(avg_reject_pnl_lev_raw) if avg_reject_pnl_lev_raw is not None else None
        except (TypeError, ValueError):
            avg_reject_pnl_lev = None

        regime = str(regime_type or "").lower()
        regime_favorable = regime in {"bull_trending", "bear_trending", "bull_ranging", "bear_ranging"}
        quality_context = opp.overall_score >= 0.72 and rule_conf >= 0.52

        if informative_resolved < min_resolved:
            return regime_favorable and quality_context

        # If rejected setups are making money on average, we are likely under-trading.
        if avg_reject_pnl_lev is not None and avg_reject_pnl_lev >= target_reject_pnl_lev + 0.15:
            return True

        # If rejected setups are deeply negative, strict mode is usually preferable.
        if avg_reject_pnl_lev is not None and avg_reject_pnl_lev <= target_reject_pnl_lev - 0.25:
            return False

        if bad_rate is not None and bad_rate >= target_bad_rate + 0.02:
            return True

        if bad_rate is not None and bad_rate <= max(0.0, target_bad_rate - 0.05):
            return False

        return regime_favorable and quality_context

    async def _refresh_reject_quality_confidence_tuning(self) -> None:
        """
        Adapt confidence gate using resolved rejected-candidate outcomes.

        If bad rejects are high (missed winners), loosen threshold slightly.
        If bad rejects are low, tighten slightly to preserve precision.
        """
        try:
            if not settings.REJECT_TUNING_ENABLED or not self.supabase:
                self._reject_quality_conf_adjustment = 0.0
                self._reject_quality_tuning_state = {
                    "enabled": bool(settings.REJECT_TUNING_ENABLED),
                    "adjustment": 0.0,
                    "reason": "disabled_or_missing_supabase",
                }
                return

            lookback_days = max(int(settings.REJECT_TUNING_LOOKBACK_DAYS or 30), 7)
            min_resolved = max(int(settings.REJECT_TUNING_MIN_RESOLVED or 30), 5)
            target_bad_rate = max(0.05, min(0.95, float(settings.REJECT_TUNING_TARGET_BAD_RATE or 0.30)))
            gain = max(0.01, min(1.0, float(settings.REJECT_TUNING_GAIN or 0.18)))
            target_reject_pnl_lev = float(settings.REJECT_TUNING_TARGET_LEVERAGED_PNL_PCT or -0.20)
            pnl_gain = max(0.001, min(0.2, float(settings.REJECT_TUNING_PNL_GAIN or 0.012)))
            pnl_weight = max(0.0, min(1.0, float(settings.REJECT_TUNING_PNL_WEIGHT or 0.40)))
            max_abs_adj = max(0.005, min(0.12, float(settings.REJECT_TUNING_MAX_ABS_ADJUSTMENT or 0.04)))
            smoothing = max(0.0, min(1.0, float(settings.REJECT_TUNING_SMOOTHING or 0.35)))

            lookback_start = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
            result = (
                self.supabase.from_("agent_learning_insights")
                .select("id, insights, timestamp")
                .eq("agent_type", "yuki")
                .eq("learning_type", "signal_rejected_candidate")
                .gte("timestamp", lookback_start)
                .order("timestamp", desc=True)
                .limit(5000)
                .execute()
            )
            rows = result.data or []

            resolved = 0
            bad_rejects = 0
            good_rejects = 0
            ambiguous_rejects = 0
            unknown = 0
            leveraged_pnl_values: List[float] = []
            leveraged_pnl_bad: List[float] = []
            leveraged_pnl_good: List[float] = []
            non_ai_resolved_excluded = 0

            for row in rows:
                insights = row.get("insights")
                if not isinstance(insights, dict):
                    continue
                if str(insights.get("tracking_status") or "").lower() != "resolved":
                    continue
                candidate_source = str(insights.get("candidate_source") or "").lower()
                if candidate_source != "ai_directional":
                    non_ai_resolved_excluded += 1
                    continue

                resolved += 1
                reject_quality = self._normalize_reject_quality_from_insights(insights)
                if reject_quality == "bad_reject":
                    bad_rejects += 1
                elif reject_quality == "good_reject":
                    good_rejects += 1
                elif reject_quality == "ambiguous_reject":
                    ambiguous_rejects += 1
                else:
                    unknown += 1

                pnl_lev = insights.get("counterfactual_pnl_pct_leveraged")
                if pnl_lev is None:
                    pnl_lev = insights.get("counterfactual_pnl_pct")
                try:
                    pnl_lev_value = float(pnl_lev)
                    if reject_quality == "bad_reject":
                        leveraged_pnl_values.append(pnl_lev_value)
                        leveraged_pnl_bad.append(pnl_lev_value)
                    elif reject_quality == "good_reject":
                        leveraged_pnl_values.append(pnl_lev_value)
                        leveraged_pnl_good.append(pnl_lev_value)
                except (TypeError, ValueError):
                    pass

            informative_resolved = bad_rejects + good_rejects
            if informative_resolved < min_resolved:
                self._reject_quality_conf_adjustment = 0.0
                self._reject_quality_tuning_state = {
                    "enabled": True,
                    "resolved_candidates": resolved,
                    "informative_resolved": informative_resolved,
                    "good_rejects": good_rejects,
                    "bad_rejects": bad_rejects,
                    "ambiguous_rejects": ambiguous_rejects,
                    "unknown_resolutions": unknown,
                    "non_ai_resolved_excluded": non_ai_resolved_excluded,
                    "bad_reject_rate": None,
                    "target_bad_reject_rate": target_bad_rate,
                    "avg_reject_pnl_pct_leveraged": None,
                    "target_reject_pnl_pct_leveraged": target_reject_pnl_lev,
                    "adjustment": 0.0,
                    "applied": False,
                    "reason": "insufficient_resolved_samples",
                    "lookback_days": lookback_days,
                    "min_resolved_required": min_resolved,
                    "updated_at": datetime.utcnow().isoformat(),
                }
                return

            bad_rate = bad_rejects / informative_resolved
            avg_reject_pnl_lev = (sum(leveraged_pnl_values) / len(leveraged_pnl_values)) if leveraged_pnl_values else 0.0
            avg_bad_reject_pnl_lev = (sum(leveraged_pnl_bad) / len(leveraged_pnl_bad)) if leveraged_pnl_bad else 0.0
            avg_good_reject_pnl_lev = (sum(leveraged_pnl_good) / len(leveraged_pnl_good)) if leveraged_pnl_good else 0.0

            bad_rate_error = bad_rate - target_bad_rate
            pnl_error = avg_reject_pnl_lev - target_reject_pnl_lev

            raw_bad_rate_adjustment = -bad_rate_error * gain
            raw_pnl_adjustment = -pnl_error * pnl_gain
            raw_adjustment = (1.0 - pnl_weight) * raw_bad_rate_adjustment + pnl_weight * raw_pnl_adjustment
            clipped_adjustment = max(-max_abs_adj, min(max_abs_adj, raw_adjustment))
            smoothed_adjustment = (
                (1.0 - smoothing) * float(self._reject_quality_conf_adjustment or 0.0)
                + smoothing * clipped_adjustment
            )
            self._reject_quality_conf_adjustment = max(-max_abs_adj, min(max_abs_adj, smoothed_adjustment))

            self._reject_quality_tuning_state = {
                "enabled": True,
                "resolved_candidates": resolved,
                "informative_resolved": informative_resolved,
                "good_rejects": good_rejects,
                "bad_rejects": bad_rejects,
                "ambiguous_rejects": ambiguous_rejects,
                "unknown_resolutions": unknown,
                "non_ai_resolved_excluded": non_ai_resolved_excluded,
                "bad_reject_rate": round(float(bad_rate), 4),
                "target_bad_reject_rate": round(float(target_bad_rate), 4),
                "avg_reject_pnl_pct_leveraged": round(float(avg_reject_pnl_lev), 4),
                "avg_bad_reject_pnl_pct_leveraged": round(float(avg_bad_reject_pnl_lev), 4),
                "avg_good_reject_pnl_pct_leveraged": round(float(avg_good_reject_pnl_lev), 4),
                "target_reject_pnl_pct_leveraged": round(float(target_reject_pnl_lev), 4),
                "raw_bad_rate_adjustment": round(float(raw_bad_rate_adjustment), 5),
                "raw_pnl_adjustment": round(float(raw_pnl_adjustment), 5),
                "raw_adjustment": round(float(raw_adjustment), 5),
                "clipped_adjustment": round(float(clipped_adjustment), 5),
                "adjustment": round(float(self._reject_quality_conf_adjustment), 5),
                "applied": True,
                "lookback_days": lookback_days,
                "min_resolved_required": min_resolved,
                "updated_at": datetime.utcnow().isoformat(),
            }

            await self._persist_reject_tuning_snapshot()
            logger.info(
                "🧠 Reject-quality tuning: bad_rate=%.2f%% target=%.2f%% avg_lev_pnl=%+.3f%% target=%+.3f%% adjustment=%+.3f",
                bad_rate * 100.0,
                target_bad_rate * 100.0,
                avg_reject_pnl_lev,
                target_reject_pnl_lev,
                self._reject_quality_conf_adjustment,
            )
        except Exception as tuning_error:
            logger.warning(f"⚠️ Reject-quality confidence tuning failed: {tuning_error}")
            self._reject_quality_conf_adjustment = 0.0
            self._reject_quality_tuning_state = {
                "enabled": bool(settings.REJECT_TUNING_ENABLED),
                "adjustment": 0.0,
                "applied": False,
                "reason": "tuning_error",
                "error": str(tuning_error),
                "updated_at": datetime.utcnow().isoformat(),
            }

    @staticmethod
    def _counterfactual_min_bad_pnl_pct() -> float:
        return max(0.0, float(getattr(settings, "REJECT_COUNTERFACTUAL_MIN_BAD_PNL_PCT", 1.0) or 1.0))

    @staticmethod
    def _counterfactual_min_good_loss_pct() -> float:
        return max(0.0, float(getattr(settings, "REJECT_COUNTERFACTUAL_MIN_GOOD_LOSS_PCT", 0.5) or 0.5))

    @classmethod
    def _classify_reject_quality_from_counterfactual(cls, terminal_status: str, pnl_pct: float) -> str:
        """Only material target/stop outcomes are strong reject-quality training signals."""
        status = str(terminal_status or "").lower()
        pnl = float(pnl_pct or 0.0)
        min_bad_pnl = cls._counterfactual_min_bad_pnl_pct()
        min_good_loss = cls._counterfactual_min_good_loss_pct()
        if status == "hit_stop_loss":
            return "good_reject" if pnl <= -min_good_loss else "ambiguous_reject"
        if status in {"hit_target_1", "hit_target_2"}:
            return "bad_reject" if pnl >= min_bad_pnl else "ambiguous_reject"
        if status == "expired":
            return "good_reject" if pnl <= -min_good_loss else "ambiguous_reject"
        return "unknown"

    @staticmethod
    def _counterfactual_replay_hours(validity_window_hours: float) -> float:
        """Use a fair replay window without letting long-horizon rejects look too far ahead."""
        validity = max(1.0, float(validity_window_hours or 1.0))
        fraction = max(
            0.10,
            min(1.0, float(getattr(settings, "REJECT_COUNTERFACTUAL_REPLAY_FRACTION", 0.50) or 0.50)),
        )
        min_hours = max(
            1.0,
            float(getattr(settings, "REJECT_COUNTERFACTUAL_MIN_REPLAY_HOURS", 6.0) or 6.0),
        )
        max_hours = max(
            min_hours,
            float(getattr(settings, "REJECT_COUNTERFACTUAL_MAX_REPLAY_HOURS", 24.0) or 24.0),
        )
        return min(validity, max(min_hours, min(max_hours, validity * fraction)))

    @classmethod
    def _normalize_reject_quality_from_insights(cls, insights: Dict[str, Any]) -> str:
        """Normalize legacy rows so expired profit is visible but not treated as a missed target."""
        terminal_status = str(insights.get("counterfactual_status") or "").lower()
        if terminal_status in {"hit_stop_loss", "hit_target_1", "hit_target_2", "expired"}:
            try:
                pnl_pct = float(insights.get("counterfactual_pnl_pct") or 0.0)
            except (TypeError, ValueError):
                pnl_pct = 0.0
            return cls._classify_reject_quality_from_counterfactual(terminal_status, pnl_pct)

        reject_quality = str(insights.get("reject_quality") or "").lower()
        if reject_quality in {"good_reject", "bad_reject", "ambiguous_reject"}:
            return reject_quality
        return "unknown"

    async def _persist_reject_tuning_snapshot(self) -> None:
        """Persist tuning snapshots for dashboard visualization and auditability."""
        if not self.supabase or not self._reject_quality_tuning_state:
            return

        cooldown_hours = max(int(settings.REJECT_TUNING_UPDATE_COOLDOWN_HOURS or 6), 1)
        now = datetime.utcnow()
        if self._last_reject_tuning_persist_at:
            elapsed_hours = (now - self._last_reject_tuning_persist_at).total_seconds() / 3600.0
            if elapsed_hours < cooldown_hours:
                return

        insights_payload = dict(self._reject_quality_tuning_state)
        insights_payload["tracking_status"] = "resolved"

        payload = {
            "agent_type": "yuki",
            "signal_id": f"gate_tuning_{uuid.uuid4().hex[:12]}",
            "learning_level": "meta",
            "learning_type": "threshold_gate_adjustment",
            "confidence_score": min(max(abs(float(self._reject_quality_conf_adjustment or 0.0)) / 0.04, 0.0), 1.0),
            "timestamp": now.isoformat(),
            "insights": insights_payload,
            "performance_impact": {
                "confidence_adjustment": float(self._reject_quality_conf_adjustment or 0.0),
                "bad_reject_rate": insights_payload.get("bad_reject_rate"),
                "target_bad_reject_rate": insights_payload.get("target_bad_reject_rate"),
                "avg_reject_pnl_pct_leveraged": insights_payload.get("avg_reject_pnl_pct_leveraged"),
                "target_reject_pnl_pct_leveraged": insights_payload.get("target_reject_pnl_pct_leveraged"),
            },
        }

        self.supabase.from_("agent_learning_insights").insert(payload).execute()
        self._last_reject_tuning_persist_at = now

    def _parse_iso_datetime(self, value: Any, default: Optional[datetime] = None) -> datetime:
        """Parse datetime strings safely."""
        if default is None:
            default = datetime.utcnow()

        if isinstance(value, datetime):
            return value

        if isinstance(value, str) and value:
            cleaned = value.replace('Z', '+00:00')
            try:
                return datetime.fromisoformat(cleaned).replace(tzinfo=None)
            except ValueError:
                return default

        return default

    def _record_rejection_context(
        self,
        symbol: str,
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
        rule_direction: str,
        rule_confidence: float,
        ai_decision: Optional[AIDecision],
        rejection_code: str,
        rejection_reason: str,
        min_confidence: Optional[float] = None,
        regime_type: Optional[str] = None,
    ) -> None:
        """Cache structured rejection context for downstream counterfactual tracking."""
        normalized_symbol = self._normalize_symbol_to_market_id(symbol)
        if not normalized_symbol:
            normalized_symbol = str(symbol or "").upper().strip()

        ai_direction = None
        ai_confidence = None
        ai_entry = None
        ai_target_1 = None
        ai_target_2 = None
        ai_stop = None
        ai_horizon = None
        ai_leverage = None
        ai_position_size = None
        active_thesis_action = None
        structural_invalidation = False
        structural_invalidation_reason = None
        edge_estimate = None
        market_structure_context = None

        if ai_decision:
            ai_direction = str(ai_decision.recommendation or "").upper().strip() or None
            try:
                ai_confidence = float(ai_decision.confidence)
            except (TypeError, ValueError):
                ai_confidence = None
            try:
                ai_entry = float(ai_decision.entry_price)
                ai_target_1 = float(ai_decision.target_1)
                ai_target_2 = float(ai_decision.target_2)
                ai_stop = float(ai_decision.stop_loss)
                ai_leverage = int(ai_decision.leverage)
                ai_position_size = float(ai_decision.position_size)
            except (TypeError, ValueError):
                pass
            ai_horizon = str(ai_decision.time_horizon or "").strip() or None
            active_thesis_action = str(
                getattr(ai_decision, "active_thesis_action", None) or ""
            ).upper().strip() or None
            structural_invalidation = bool(
                getattr(ai_decision, "structural_invalidation", False)
            )
            structural_invalidation_reason = str(
                getattr(ai_decision, "structural_invalidation_reason", None) or ""
            ).strip() or None
            edge_estimate = getattr(ai_decision, "edge_estimate", None)
            market_structure_context = getattr(ai_decision, "market_structure_context", None)

        rejection_code_value = str(rejection_code or "unknown")
        payload = {
            "symbol": normalized_symbol,
            "recorded_at": datetime.utcnow().isoformat(),
            "rejection_code": rejection_code_value,
            "rejection_reason": str(rejection_reason or "Rejected by ensemble gates"),
            "rule_direction": str(rule_direction or "HOLD").upper(),
            "rule_confidence": float(rule_confidence or 0.0),
            "min_confidence": float(min_confidence) if min_confidence is not None else None,
            "opportunity_score": float(opportunity_score.overall_score or 0.0),
            "current_price": float(technical_analysis.current_price or 0.0),
            "volatility_24h": float(technical_analysis.volatility_24h or 0.0),
            "regime_type": str(regime_type or "").strip() or None,
            "ai_recommendation": ai_direction,
            "ai_confidence": ai_confidence,
            "ai_entry_price": ai_entry,
            "ai_target_1": ai_target_1,
            "ai_target_2": ai_target_2,
            "ai_stop_loss": ai_stop,
            "ai_time_horizon": ai_horizon,
            "ai_leverage": ai_leverage,
            "ai_position_size": ai_position_size,
            "active_thesis_action": active_thesis_action,
            "structural_invalidation": structural_invalidation,
            "structural_invalidation_reason": structural_invalidation_reason,
            "edge_estimate": edge_estimate,
            "market_structure_context": market_structure_context,
            "shadow_publish": rejection_code_value in {
                "ai_confidence_below_threshold",
                "net_rr_below_threshold",
            },
            "shadow_policy": (
                "confidence_rr_candidate_v1"
                if rejection_code_value in {
                    "ai_confidence_below_threshold",
                    "net_rr_below_threshold",
                }
                else None
            ),
        }

        self._last_rejection_context[normalized_symbol] = payload
        raw_symbol = str(symbol or "").upper().strip()
        if raw_symbol and raw_symbol != normalized_symbol:
            self._last_rejection_context[raw_symbol] = payload

    def _build_rule_counterfactual(
        self,
        direction: str,
        entry_price: float,
        technical_analysis: Optional[TechnicalAnalysis] = None,
        time_horizon: str = "4h-12h",
    ) -> Optional[Dict[str, float]]:
        """Create a synthetic directional setup from rule-engine direction for HOLD rejections."""
        if entry_price <= 0:
            return None

        normalized_direction = str(direction or "").upper().strip()
        atr = float(getattr(technical_analysis, "atr_14", 0.0) or 0.0)
        atr_pct = (atr / entry_price) * 100.0 if entry_price > 0 else 0.0
        volatility_24h = float(getattr(technical_analysis, "volatility_24h", 0.0) or 0.0)
        constraints = self._target_distance_constraints(time_horizon, atr_pct, volatility_24h, 0.62)
        target_1_pct = max(
            float(constraints["min_t1_move_pct"]),
            min(
                float(constraints["max_t1_move_pct"]),
                max(1.10, atr_pct * 1.10, volatility_24h * 0.12),
            ),
        )
        target_2_pct = max(
            target_1_pct * 1.25,
            min(target_1_pct * 1.60, float(constraints["max_t1_move_pct"]) * 1.60),
        )
        stop_pct = max(
            float(constraints["min_sl_move_pct"]),
            min(float(constraints["max_sl_move_pct"]), target_1_pct / 1.35),
        )

        if normalized_direction == "LONG":
            return {
                "entry_price": entry_price,
                "target_1": entry_price * (1 + target_1_pct / 100.0),
                "target_2": entry_price * (1 + target_2_pct / 100.0),
                "stop_loss": entry_price * (1 - stop_pct / 100.0),
            }
        if normalized_direction == "SHORT":
            return {
                "entry_price": entry_price,
                "target_1": entry_price * (1 - target_1_pct / 100.0),
                "target_2": entry_price * (1 - target_2_pct / 100.0),
                "stop_loss": entry_price * (1 + stop_pct / 100.0),
            }
        return None

    async def _log_rejected_signal_candidate(
        self,
        symbol: str,
        technical_analysis: TechnicalAnalysis,
        opportunity_score: OpportunityScore,
        run_id: str,
    ) -> None:
        """
        Persist rejected directional candidates for counterfactual outcome validation.

        This enables measurement of reject quality (good reject vs missed winner).
        """
        try:
            if not self.supabase:
                return

            normalized_symbol = self._normalize_symbol_to_market_id(symbol)
            context = (
                self._last_rejection_context.pop(normalized_symbol, None)
                or self._last_rejection_context.pop(str(symbol or "").upper().strip(), None)
                or {}
            )

            rule_direction = str(context.get("rule_direction") or "HOLD").upper().strip()
            rule_confidence = float(context.get("rule_confidence") or 0.0)
            ai_recommendation = str(context.get("ai_recommendation") or "").upper().strip()
            ai_confidence = context.get("ai_confidence")
            regime_type_ctx = str(context.get("regime_type") or "").strip() or None
            try:
                ai_confidence = float(ai_confidence) if ai_confidence is not None else None
            except (TypeError, ValueError):
                ai_confidence = None

            direction = None
            entry_price = 0.0
            target_1 = 0.0
            target_2 = 0.0
            stop_loss = 0.0
            leverage = 1
            position_size = 0.0
            time_horizon = "4h-12h"
            candidate_source = None

            if ai_recommendation in {"LONG", "SHORT"}:
                direction = ai_recommendation
                candidate_source = "ai_directional"
                entry_price = float(context.get("ai_entry_price") or 0.0)
                target_1 = float(context.get("ai_target_1") or 0.0)
                target_2 = float(context.get("ai_target_2") or 0.0)
                stop_loss = float(context.get("ai_stop_loss") or 0.0)
                leverage = max(1, int(context.get("ai_leverage") or 1))
                position_size = max(0.0, float(context.get("ai_position_size") or 0.0))
                time_horizon = str(context.get("ai_time_horizon") or "4h-12h").strip() or "4h-12h"
            elif rule_direction in {"LONG", "SHORT"} and rule_confidence >= 0.55:
                synthetic = self._build_rule_counterfactual(
                    rule_direction,
                    float(technical_analysis.current_price or 0.0),
                    technical_analysis=technical_analysis,
                    time_horizon=time_horizon,
                )
                if synthetic:
                    direction = rule_direction
                    candidate_source = "rule_counterfactual"
                    entry_price = float(synthetic["entry_price"])
                    target_1 = float(synthetic["target_1"])
                    target_2 = float(synthetic["target_2"])
                    stop_loss = float(synthetic["stop_loss"])
                    leverage = 1
                    position_size = 0.0
                    time_horizon = "4h-12h"

            if direction not in {"LONG", "SHORT"}:
                return
            if entry_price <= 0 or target_1 <= 0 or target_2 <= 0 or stop_loss <= 0:
                return

            validity_window_hours = self._calculate_validity_window_hours(time_horizon)
            replay_window_hours = self._counterfactual_replay_hours(validity_window_hours)
            analysis_timestamp = datetime.utcnow()
            expires_at = analysis_timestamp + timedelta(hours=max(validity_window_hours, 1))
            evaluation_due_at = analysis_timestamp + timedelta(hours=replay_window_hours)
            evaluation_due_at = min(evaluation_due_at, expires_at)
            candidate_id = f"rejected_{uuid.uuid4().hex[:12]}"

            confidence_basis = ai_confidence if ai_confidence is not None else rule_confidence
            confidence_basis = max(0.0, min(1.0, float(confidence_basis or 0.0)))

            payload = {
                "agent_type": "yuki",
                "signal_id": candidate_id,
                "learning_level": "pattern",
                "learning_type": "signal_rejected_candidate",
                "confidence_score": confidence_basis,
                "timestamp": analysis_timestamp.isoformat(),
                "insights": {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "symbol": normalized_symbol,
                    "direction": direction,
                    "candidate_source": candidate_source,
                    "tracking_status": "pending",
                    "analysis_timestamp": analysis_timestamp.isoformat(),
                    "evaluation_due_at": evaluation_due_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "validity_window_hours": validity_window_hours,
                    "replay_window_hours": replay_window_hours,
                    "entry_price": entry_price,
                    "target_1": target_1,
                    "target_2": target_2,
                    "stop_loss": stop_loss,
                    "leverage": leverage,
                    "position_size": position_size,
                    "time_horizon": time_horizon,
                    "rejection_code": context.get("rejection_code") or "ensemble_rejected",
                    "rejection_reason": context.get("rejection_reason") or "Rejected by decision gates",
                    "shadow_publish": bool(context.get("shadow_publish")),
                    "shadow_policy": context.get("shadow_policy"),
                    "rule_direction": rule_direction,
                    "rule_confidence": rule_confidence,
                    "ai_recommendation": ai_recommendation or None,
                    "ai_confidence": ai_confidence,
                    # regime_type enables AI-vs-rule accuracy breakdown by market regime
                    "regime_type": regime_type_ctx,
                    "opportunity_score": float(opportunity_score.overall_score or 0.0),
                    "current_price": float(technical_analysis.current_price or 0.0),
                    "edge_estimate": context.get("edge_estimate"),
                    "market_structure_context": context.get("market_structure_context"),
                },
            }

            self.supabase.from_("agent_learning_insights").insert(payload).execute()
            logger.info(
                f"🧪 Logged rejected candidate for counterfactual tracking: "
                f"{normalized_symbol} {direction} ({candidate_source})"
            )
        except Exception as rejection_log_error:
            logger.warning(f"⚠️ Failed to log rejected candidate for {symbol}: {rejection_log_error}")

    def _counterfactual_timeframe(self, validity_window_hours: int) -> str:
        """Pick an OHLCV timeframe balancing precision and API cost."""
        hours = max(int(validity_window_hours or 0), 1)
        if hours <= 24:
            return "5m"
        if hours <= 168:
            return "15m"
        if hours <= 720:
            return "1h"
        return "4h"

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
        }
        return mapping.get(timeframe, 3_600_000)

    async def _fetch_hyperliquid_replay_candles(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        until_ms: int,
    ) -> List[List[float]]:
        """Fetch counterfactual candles from Yuki's execution venue."""
        coin = self._signal_symbol_key(symbol)
        if not coin or ":" in coin:
            return []
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post(
                    "https://api.hyperliquid.xyz/info",
                    json={
                        "type": "candleSnapshot",
                        "req": {
                            "coin": coin,
                            "interval": timeframe,
                            "startTime": int(since_ms),
                            "endTime": int(until_ms),
                        },
                    },
                )
                response.raise_for_status()
                rows = response.json() or []
            return [
                [
                    int(row.get("t") or 0),
                    float(row.get("o") or 0.0),
                    float(row.get("h") or 0.0),
                    float(row.get("l") or 0.0),
                    float(row.get("c") or 0.0),
                    float(row.get("v") or 0.0),
                ]
                for row in rows
                if isinstance(row, dict) and row.get("t") is not None
            ]
        except Exception as exc:
            logger.debug("Hyperliquid replay candles unavailable for %s: %s", coin, exc)
            return []

    async def _evaluate_rejected_candidate_counterfactual(self, insights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replay rejected candidate levels on OHLCV to classify reject quality."""
        try:
            symbol = str(insights.get("symbol") or "").upper().strip()
            direction = str(insights.get("direction") or "").upper().strip()
            if direction not in {"LONG", "SHORT"} or not symbol:
                return {
                    "tracking_status": "resolved",
                    "resolved_at": datetime.utcnow().isoformat(),
                    "counterfactual_status": "invalid_candidate",
                    "reject_quality": "unknown",
                    "evaluation_note": "Invalid symbol or direction in candidate payload",
                }

            def _to_float(value: Any, default: float = 0.0) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            entry_price = _to_float(insights.get("entry_price"))
            target_1 = _to_float(insights.get("target_1"))
            target_2 = _to_float(insights.get("target_2"))
            stop_loss = _to_float(insights.get("stop_loss"))
            leverage = max(1.0, _to_float(insights.get("leverage"), 1.0))
            validity_window_hours = max(int(_to_float(insights.get("validity_window_hours"), 12)), 1)

            if entry_price <= 0 or target_1 <= 0 or target_2 <= 0 or stop_loss <= 0:
                return {
                    "tracking_status": "resolved",
                    "resolved_at": datetime.utcnow().isoformat(),
                    "counterfactual_status": "invalid_levels",
                    "reject_quality": "unknown",
                    "evaluation_note": "Candidate levels were incomplete or invalid",
                }

            analyzed_at = self._parse_iso_datetime(
                insights.get("analysis_timestamp"),
                default=datetime.utcnow() - timedelta(hours=validity_window_hours),
            )
            expires_at = self._parse_iso_datetime(
                insights.get("expires_at") or insights.get("evaluation_due_at"),
                default=analyzed_at + timedelta(hours=validity_window_hours),
            )
            replay_window_hours = self._counterfactual_replay_hours(validity_window_hours)
            # Replay a bounded early portion of the setup. The old 25% raw-window replay
            # resolved 12h candidates after only 3h, over-crediting noisy flat/expired rows.
            replay_window_end = analyzed_at + timedelta(hours=replay_window_hours)
            evaluation_end = min(datetime.utcnow(), expires_at, replay_window_end)
            if evaluation_end <= analyzed_at:
                evaluation_end = analyzed_at + timedelta(minutes=5)

            timeframe = self._counterfactual_timeframe(validity_window_hours)
            timeframe_ms = self._timeframe_to_ms(timeframe)
            since_ms = int(analyzed_at.timestamp() * 1000)
            until_ms = int(evaluation_end.timestamp() * 1000)
            estimated_candles = int(max(until_ms - since_ms, timeframe_ms) / timeframe_ms) + 5
            candle_limit = max(30, min(1000, estimated_candles))

            ohlcv = await self._fetch_hyperliquid_replay_candles(
                symbol,
                timeframe,
                since_ms,
                until_ms,
            )
            replay_venue = "hyperliquid" if ohlcv else "binance_fallback"
            fetch_errors: List[str] = []
            candidate_symbols = [symbol]
            if "/" not in symbol and symbol.endswith("USDT"):
                base = symbol[:-4]
                candidate_symbols.append(f"{base}/USDT:USDT")

            if not ohlcv and self.binance:
                for market_symbol in candidate_symbols:
                    try:
                        ohlcv = await self.rate_limited_api_call(
                            self.binance.fetch_ohlcv,
                            market_symbol,
                            timeframe,
                            since=since_ms,
                            limit=candle_limit,
                        )
                        if ohlcv:
                            break
                    except Exception as fetch_error:
                        fetch_errors.append(str(fetch_error))

            if not ohlcv:
                return {
                    "tracking_status": "resolved",
                    "resolved_at": datetime.utcnow().isoformat(),
                    "counterfactual_status": "data_unavailable",
                    "reject_quality": "unknown",
                    "evaluation_note": "No OHLCV data available for replay",
                    "evaluation_errors": fetch_errors[:2],
                }

            terminal_status = "expired"
            terminal_price = float(ohlcv[-1][4]) if ohlcv[-1][4] else entry_price
            terminal_ts_ms = until_ms

            for candle in ohlcv:
                if len(candle) < 5:
                    continue
                ts_ms = int(candle[0] or 0)
                open_price = _to_float(candle[1], entry_price)
                high_price = _to_float(candle[2], open_price)
                low_price = _to_float(candle[3], open_price)
                close_price = _to_float(candle[4], open_price)
                terminal_price = close_price

                # Same-candle race resolution: open price decides which side fired first.
                # If open is already past SL → SL hit at open; past target → target hit at open;
                # otherwise the closer level (in price terms) gets credit. This better matches
                # exchange execution than pure absolute distance.
                if direction == "LONG":
                    hit_stop = low_price <= stop_loss
                    hit_target_1 = high_price >= target_1
                    hit_target_2 = high_price >= target_2

                    if hit_stop or hit_target_1 or hit_target_2:
                        if hit_stop and (hit_target_1 or hit_target_2):
                            preferred_target = target_2 if hit_target_2 else target_1
                            if open_price <= stop_loss:
                                terminal_status = "hit_stop_loss"
                                terminal_price = stop_loss
                            elif open_price >= preferred_target:
                                terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                                terminal_price = preferred_target
                            elif (open_price - stop_loss) <= (preferred_target - open_price):
                                terminal_status = "hit_stop_loss"
                                terminal_price = stop_loss
                            else:
                                terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                                terminal_price = preferred_target
                        elif hit_stop:
                            terminal_status = "hit_stop_loss"
                            terminal_price = stop_loss
                        else:
                            terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                            terminal_price = target_2 if hit_target_2 else target_1
                        terminal_ts_ms = ts_ms
                        break
                else:
                    hit_stop = high_price >= stop_loss
                    hit_target_1 = low_price <= target_1
                    hit_target_2 = low_price <= target_2

                    if hit_stop or hit_target_1 or hit_target_2:
                        if hit_stop and (hit_target_1 or hit_target_2):
                            preferred_target = target_2 if hit_target_2 else target_1
                            if open_price >= stop_loss:
                                terminal_status = "hit_stop_loss"
                                terminal_price = stop_loss
                            elif open_price <= preferred_target:
                                terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                                terminal_price = preferred_target
                            elif (stop_loss - open_price) <= (open_price - preferred_target):
                                terminal_status = "hit_stop_loss"
                                terminal_price = stop_loss
                            else:
                                terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                                terminal_price = preferred_target
                        elif hit_stop:
                            terminal_status = "hit_stop_loss"
                            terminal_price = stop_loss
                        else:
                            terminal_status = "hit_target_2" if hit_target_2 else "hit_target_1"
                            terminal_price = target_2 if hit_target_2 else target_1
                        terminal_ts_ms = ts_ms
                        break

            if direction == "LONG":
                gross_pnl_pct = ((terminal_price - entry_price) / entry_price) * 100.0
            else:
                gross_pnl_pct = ((entry_price - terminal_price) / entry_price) * 100.0
            # Apply real execution costs (fees + slippage + funding) so feedback reflects realized PnL.
            hold_hours = max(0.5, (terminal_ts_ms - since_ms) / 1000.0 / 3600.0)
            funding_rate_decimal = await self._fetch_funding_rate(symbol) or 0.0001
            pnl_pct = self._apply_execution_costs(
                gross_pnl_pct, leverage=1.0, hold_hours=hold_hours, funding_rate=funding_rate_decimal
            )
            pnl_pct_leveraged = pnl_pct * leverage

            reject_quality = self._classify_reject_quality_from_counterfactual(terminal_status, pnl_pct)

            resolved_at = datetime.utcnow()
            exit_timestamp = datetime.utcfromtimestamp(terminal_ts_ms / 1000.0).isoformat() if terminal_ts_ms > 0 else resolved_at.isoformat()

            return {
                "tracking_status": "resolved",
                "resolved_at": resolved_at.isoformat(),
                "counterfactual_status": terminal_status,
                "counterfactual_exit_price": round(float(terminal_price), 8),
                "counterfactual_exit_timestamp": exit_timestamp,
                "replay_venue": replay_venue,
                "counterfactual_pnl_pct": round(float(pnl_pct), 4),
                "counterfactual_pnl_pct_leveraged": round(float(pnl_pct_leveraged), 4),
                "reject_quality": reject_quality,
                "replay_window_hours": round(float(replay_window_hours), 4),
                "evaluation_timeframe": timeframe,
                "evaluation_candle_count": len(ohlcv),
            }
        except Exception as evaluation_error:
            logger.warning(f"⚠️ Rejected candidate evaluation failed: {evaluation_error}")
            return {
                "tracking_status": "resolved",
                "resolved_at": datetime.utcnow().isoformat(),
                "counterfactual_status": "evaluation_error",
                "reject_quality": "unknown",
                "evaluation_note": str(evaluation_error),
            }

    async def _resolve_pending_rejected_candidates(self, max_to_process: int = 20) -> int:
        """Resolve pending rejected candidates whose evaluation window has elapsed."""
        try:
            if not self.supabase:
                return 0

            now = datetime.utcnow()
            lookback_start = (now - timedelta(days=60)).isoformat()
            query_limit = max(500, min(5000, int(max_to_process or 20) * 25))
            rows_result = (
                self.supabase.from_("agent_learning_insights")
                .select("id, insights, timestamp")
                .eq("agent_type", "yuki")
                .eq("learning_type", "signal_rejected_candidate")
                .gte("timestamp", lookback_start)
                .order("timestamp", desc=True)
                .limit(query_limit)
                .execute()
            )
            rows = rows_result.data or []
            if not rows:
                return 0

            processed = 0
            resolved = 0
            pending_seen = 0
            due_seen = 0

            for row in rows:
                if processed >= max_to_process:
                    break

                insights = row.get("insights")
                if not isinstance(insights, dict):
                    continue
                if str(insights.get("tracking_status") or "").lower() != "pending":
                    continue
                pending_seen += 1

                validity_window_hours = 12
                try:
                    validity_window_hours = max(1, int(float(insights.get("validity_window_hours") or 12)))
                except (TypeError, ValueError):
                    validity_window_hours = 12
                analyzed_at = self._parse_iso_datetime(
                    insights.get("analysis_timestamp") or row.get("timestamp"),
                    default=now - timedelta(hours=validity_window_hours),
                )
                replay_due_at = analyzed_at + timedelta(
                    hours=self._counterfactual_replay_hours(validity_window_hours)
                )
                expires_at = self._parse_iso_datetime(
                    insights.get("expires_at"),
                    default=analyzed_at + timedelta(hours=validity_window_hours),
                )
                stored_due_at = self._parse_iso_datetime(insights.get("evaluation_due_at"), default=replay_due_at)
                due_at = min(max(stored_due_at, replay_due_at), expires_at)
                if due_at > now:
                    continue
                due_seen += 1

                processed += 1
                evaluation = await self._evaluate_rejected_candidate_counterfactual(insights)
                if not evaluation:
                    continue

                updated_insights = dict(insights)
                updated_insights.update(evaluation)
                performance_impact = {
                    "counterfactual_status": updated_insights.get("counterfactual_status"),
                    "counterfactual_pnl_pct": updated_insights.get("counterfactual_pnl_pct"),
                    "reject_quality": updated_insights.get("reject_quality"),
                }

                self.supabase.from_("agent_learning_insights").update({
                    "insights": updated_insights,
                    "performance_impact": performance_impact,
                    "updated_at": datetime.utcnow().isoformat(),
                }).eq("id", row.get("id")).execute()
                resolved += 1

            if resolved > 0:
                logger.info(
                    "🧪 Resolved %d rejected signal counterfactual evaluations "
                    "(pending_seen=%d, due_seen=%d, scanned=%d)",
                    resolved,
                    pending_seen,
                    due_seen,
                    len(rows),
                )
            elif pending_seen > 0:
                logger.info(
                    "🧪 Reject counterfactual refresh found no resolvable rows "
                    "(pending_seen=%d, due_seen=%d, scanned=%d)",
                    pending_seen,
                    due_seen,
                    len(rows),
                )
            return resolved
        except Exception as resolve_error:
            logger.warning(f"⚠️ Failed to resolve rejected candidates: {resolve_error}")
            return 0

    async def refresh_reject_learning_backlog(self, max_to_process: int = 80) -> Dict[str, Any]:
        """Resolve due reject counterfactuals and refresh confidence tuning."""
        bounded_max = max(1, min(250, int(max_to_process or 80)))
        resolved = await self._resolve_pending_rejected_candidates(max_to_process=bounded_max)
        await self._refresh_reject_quality_confidence_tuning()
        return {
            "resolved": resolved,
            "max_to_process": bounded_max,
            "tuning_state": dict(self._reject_quality_tuning_state or {}),
        }

    async def _compute_conditional_expected_value(
        self, symbol: str, direction: str, regime_type: str
    ) -> Optional[Dict[str, float]]:
        """Compute conditional expected value from historical signals.

        Queries past completed signals matching (direction, regime) and computes:
          E[PnL] = P(win) × avg_win − P(loss) × avg_loss

        Returns None if insufficient data (< 20 trades).
        This is the core quant concept: "Expected value is your conviction."
        """
        MIN_TRADES = 20
        try:
            if not self.supabase:
                return None

            # Query terminal signals for this direction; then condition by regime when available.
            result = self.supabase.from_('platform_signals').select(
                'pnl_percentage, status, ai_confidence_breakdown'
            ).eq(
                'direction', direction
            ).in_(
                'status', ['hit_target_1', 'hit_target_2', 'hit_stop_loss', 'expired']
            ).gte(
                'created_at', (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
            ).limit(200).execute()

            records = result.data or []
            desired_regime = str(regime_type or "").lower().strip()
            if desired_regime:
                regime_filtered = []
                for row in records:
                    breakdown = self._extract_breakdown_dict(row.get("ai_confidence_breakdown"))
                    row_regime = str(breakdown.get("market_regime") or "").lower().strip()
                    if row_regime == desired_regime:
                        regime_filtered.append(row)
                if len(regime_filtered) >= MIN_TRADES:
                    records = regime_filtered
            if len(records) < MIN_TRADES:
                return None

            pnl_values = []
            for r in records:
                pnl = r.get('pnl_percentage')
                if pnl is not None:
                    try:
                        pnl_values.append(float(pnl))
                    except (TypeError, ValueError):
                        continue

            if len(pnl_values) < MIN_TRADES:
                return None

            wins = [p for p in pnl_values if p > 0]
            losses = [p for p in pnl_values if p <= 0]
            win_rate = len(wins) / len(pnl_values)
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
            expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss

            return {
                'expected_value': expected_value,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sample_size': len(pnl_values),
            }
        except Exception as e:
            logger.debug(f"EV computation skipped for {symbol}/{direction}: {e}")
            return None

    async def _fetch_liquidation_zones(self, symbol: str) -> Dict[float, float]:
        """
        Return a price-bucketed map of recent forced-liquidation notional (USD) using
        Binance's public force-orders endpoint (free, no auth).

        Buckets are 1% of the last known price wide.  Key = bucket mid-price,
        value = total notional liquidated in that bucket over the last ~200 events.

        Falls back to empty dict on any error so callers degrade gracefully.
        """
        now = datetime.utcnow()
        cached = self._liq_cluster_cache.get(symbol)
        if cached:
            clusters, cached_at = cached
            if (now - cached_at).total_seconds() < self._liq_cluster_ttl_seconds:
                return clusters

        try:
            # max_retries=1: "not supported" is permanent — no point waiting 45 s per symbol
            raw = await self.rate_limited_api_call(
                self.binance.fetch_liquidations,
                symbol,
                max_retries=1,
                limit=200,
            )
        except Exception as e:
            logger.debug("Liquidation fetch failed for %s: %s", symbol, e)
            return {}

        if not raw:
            return {}

        # Use the most recent liquidation price as reference for bucket sizing.
        try:
            ref_price = float(raw[-1].get("price") or raw[-1].get("info", {}).get("price") or 0)
        except Exception:
            ref_price = 0.0
        if ref_price <= 0:
            return {}

        bucket_pct = 0.01  # 1% buckets
        bucket_size = ref_price * bucket_pct
        clusters: Dict[float, float] = {}

        for entry in raw:
            try:
                price = float(entry.get("price") or entry.get("info", {}).get("price") or 0)
                cost = float(entry.get("cost") or 0)
                if price <= 0:
                    continue
                bucket = round(round(price / bucket_size) * bucket_size, 8)
                clusters[bucket] = clusters.get(bucket, 0.0) + cost
            except Exception:
                continue

        self._liq_cluster_cache[symbol] = (clusters, now)
        return clusters

    def _adjust_sl_for_liq_clusters(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        clusters: Dict[float, float],
    ) -> float:
        """
        Widen stop_loss past any dense liquidation cluster it currently sits inside.

        Logic: if stop_loss is within ±1 bucket of a cluster whose notional is in the
        top-25% of all clusters, assume a stop-hunt sweep is likely there.  Move SL
        to the next clean price (1 bucket beyond the cluster), capped at 5% from entry.
        """
        if not clusters or entry_price <= 0 or stop_loss <= 0:
            return stop_loss

        total_notional = sum(clusters.values())
        if total_notional <= 0:
            return stop_loss

        # Find the 75th-percentile notional threshold for "dense" classification.
        sorted_vals = sorted(clusters.values())
        p75_idx = max(0, int(len(sorted_vals) * 0.75) - 1)
        dense_threshold = sorted_vals[p75_idx]

        bucket_size = entry_price * 0.01  # 1% of entry

        dense_near_sl = [
            (p, n)
            for p, n in clusters.items()
            if abs(p - stop_loss) <= bucket_size * 1.5 and n >= dense_threshold
        ]

        if not dense_near_sl:
            return stop_loss

        # Find the outermost edge of the dense cluster band (farthest from entry).
        if direction == "LONG":
            # SL is below entry; push it further below the cluster.
            cluster_edge = min(p for p, _ in dense_near_sl) - bucket_size
            new_sl = min(stop_loss, cluster_edge)  # can only move SL further from entry
            max_sl = entry_price * (1 - 0.05)
            new_sl = max(new_sl, max_sl)
        else:
            # SHORT: SL is above entry; push it further above the cluster.
            cluster_edge = max(p for p, _ in dense_near_sl) + bucket_size
            new_sl = max(stop_loss, cluster_edge)
            max_sl = entry_price * (1 + 0.05)
            new_sl = min(new_sl, max_sl)

        if new_sl != stop_loss:
            logger.info(
                "🔥 %s: Widened SL past liq cluster — %s %.6f → %.6f "
                "(cluster notional $%.0f at ~%.6f)",
                symbol, direction, stop_loss, new_sl,
                sum(n for _, n in dense_near_sl),
                sum(p for p, _ in dense_near_sl) / len(dense_near_sl),
            )
        return new_sl

    async def _fetch_positioning_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Return a composite positioning signal from Binance's free fapi data endpoints:
        - Global long/short account ratio
        - Taker buy/sell volume ratio

        Returns dict with keys:
          ls_ratio      : float — long accounts / short accounts (>1 = more longs)
          taker_ratio   : float — taker buy vol / sell vol (>1 = buy-side dominant)
          bias          : str  — 'long_crowded' | 'short_crowded' | 'neutral'
          gate_direction: str  — direction to REJECT ('LONG'/'SHORT'/None)

        Crowded-long → bearish pressure → reject new LONG signals.
        Crowded-short → bullish pressure → reject new SHORT signals.
        Thresholds: ls_ratio > 1.25 or < 0.80, taker_ratio > 1.20 or < 0.80.
        Both must agree to gate (conservative; one signal not enough).

        Missing endpoint data is omitted from downstream context. The gate remains
        neutral unless both real L/S and taker-ratio data agree.
        """
        neutral = {
            'ls_ratio': None,
            'taker_ratio': None,
            'bias': 'neutral',
            'gate_direction': None,
            'data_available': False,
            'ls_available': False,
            'taker_available': False,
        }

        now = datetime.utcnow()
        cached = self._positioning_cache.get(symbol)
        if cached:
            sig, cached_at = cached
            if (now - cached_at).total_seconds() < self._positioning_ttl_seconds:
                return sig

        # Normalise symbol to Binance format: BTC/USDT:USDT → BTCUSDT
        binance_sym = symbol.replace('/USDT:USDT', 'USDT').replace('/USDT', 'USDT').replace('/', '')
        params = {'symbol': binance_sym, 'period': '1h', 'limit': 24}

        ls_ratio: Optional[float] = None
        taker_ratio: Optional[float] = None
        taker_buy_volume = 0.0
        taker_sell_volume = 0.0
        cvd = 0.0
        taker_periods = 0

        try:
            raw_ls = await self.rate_limited_api_call(
                self.binance.fapiDataGetGloballongshortaccountratio,
                params,
            )
            if raw_ls and isinstance(raw_ls, list) and raw_ls:
                latest = raw_ls[-1]
                ratio_value = latest.get('longShortRatio') or latest.get('buySellRatio')
                if ratio_value is not None:
                    ls_ratio = float(ratio_value)
        except Exception as e:
            logger.debug("Positioning L/S ratio fetch failed for %s: %s", symbol, e)

        try:
            raw_tk = await self.rate_limited_api_call(
                self.binance.fapiDataGetTakerlongshortratio,
                params,
            )
            if raw_tk and isinstance(raw_tk, list) and raw_tk:
                for row in raw_tk:
                    if not isinstance(row, dict):
                        continue
                    try:
                        buy = float(row.get('buyVol') or row.get('takerBuyVol') or row.get('buyVolume') or 0.0)
                        sell = float(row.get('sellVol') or row.get('takerSellVol') or row.get('sellVolume') or 0.0)
                        ratio_value = row.get('buySellRatio')
                        if ratio_value is not None:
                            taker_ratio = float(ratio_value)
                    except (TypeError, ValueError):
                        continue
                    taker_buy_volume += buy
                    taker_sell_volume += sell
                    cvd += buy - sell
                    taker_periods += 1
        except Exception as e:
            logger.debug("Taker ratio fetch failed for %s: %s", symbol, e)

        # Classify bias — require BOTH signals to agree before gating.
        long_crowded = (
            ls_ratio is not None
            and taker_ratio is not None
            and ls_ratio > 1.25
            and taker_ratio > 1.20
        )
        short_crowded = (
            ls_ratio is not None
            and taker_ratio is not None
            and ls_ratio < 0.80
            and taker_ratio < 0.80
        )

        if long_crowded:
            bias = 'long_crowded'
            gate_direction = 'LONG'   # reject new longs
        elif short_crowded:
            bias = 'short_crowded'
            gate_direction = 'SHORT'  # reject new shorts
        else:
            bias = 'neutral'
            gate_direction = None

        taker_total = taker_buy_volume + taker_sell_volume
        result = {
            'ls_ratio': round(ls_ratio, 4) if ls_ratio is not None else None,
            'taker_ratio': round(taker_ratio, 4) if taker_ratio is not None else None,
            'taker_buy_volume': round(taker_buy_volume, 4) if taker_periods > 0 else None,
            'taker_sell_volume': round(taker_sell_volume, 4) if taker_periods > 0 else None,
            'taker_delta': round(taker_buy_volume - taker_sell_volume, 4) if taker_periods > 0 else None,
            'taker_delta_pct': round((taker_buy_volume - taker_sell_volume) / taker_total, 4) if taker_total > 0 else None,
            'cvd': round(cvd, 4) if taker_periods > 0 else None,
            'periods': taker_periods if taker_periods > 0 else None,
            'bias': bias,
            'gate_direction': gate_direction,
            'data_available': bool(ls_ratio is not None or taker_periods > 0),
            'ls_available': ls_ratio is not None,
            'taker_available': taker_periods > 0,
        }
        self._positioning_cache[symbol] = (result, now)
        return result

    async def _ensemble_decision(
        self,
        symbol: str,
        ta: TechnicalAnalysis,
        opp: OpportunityScore,
        active_thesis_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AIDecision]:
        """Stack predictive policy (P(win)), rules, and LLM synthesis/guardrails; require ensemble agreement unless reduced-conservatism mode applies."""
        normalized_symbol = self._normalize_symbol_to_market_id(symbol)
        if normalized_symbol:
            self._last_rejection_context.pop(normalized_symbol, None)
        self._last_rejection_context.pop(str(symbol or "").upper().strip(), None)

        # BTC macro regime (same source token analysis uses) — informs counter-trend
        # discounting only; gracefully degrades to None if unavailable.
        market_regime: Optional[str] = None
        try:
            btc_data = await self.get_btc_market_data()
            market_regime = self.determine_market_regime(btc_data)
        except Exception as regime_exc:
            logger.debug(f"Macro regime unavailable for {symbol}: {regime_exc}")

        # Rule decision first
        rule_dir, rule_conf, rule_reason = self._rule_engine_decision(ta, opp, market_regime)
        regime = self._detect_advanced_regime(ta)
        regime_type = str(regime.get('type', 'unknown')).lower()
        reduced_conservatism = self._should_use_reduced_conservatism(regime_type, opp, rule_conf)

        # Early skip for very weak contexts to reduce expensive low-value LLM calls.
        if (
            not active_thesis_context
            and rule_dir == 'HOLD'
            and rule_conf <= 0.35
            and opp.overall_score < 0.55
        ):
            logger.info(
                f"⚠️ {symbol}: Skipping LLM analysis (weak pre-gate: rule={rule_conf:.2f}, opp={opp.overall_score:.2f})"
            )
            return None

        # Positioning gate: reject signals that fight a strongly crowded market.
        # Uses Binance's free taker L/S + global L/S ratio endpoints (5-min cache).
        if getattr(settings, "GATE_ENABLE_CROWDING", False) and rule_dir in ('LONG', 'SHORT'):
            pos_sig = await self._fetch_positioning_signal(symbol)
            if pos_sig.get('gate_direction') == rule_dir:
                bias = pos_sig.get('bias', 'unknown')
                ls = pos_sig.get('ls_ratio', 1.0)
                tk = pos_sig.get('taker_ratio', 1.0)
                logger.info(
                    "⚠️ %s: Positioning gate rejected %s — %s "
                    "(ls_ratio=%.2f, taker_ratio=%.2f)",
                    symbol, rule_dir, bias, ls, tk,
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=None,
                    rejection_code="crowded_positioning",
                    rejection_reason=(
                        f"Market too crowded in {rule_dir} direction: {bias} "
                        f"(ls={ls:.2f}, taker={tk:.2f})"
                    ),
                    min_confidence=None,
                )
                if not active_thesis_context:
                    return None
                logger.info(
                    "Continuing %s through contextual AI revalidation despite crowding gate",
                    symbol,
                )

        # Higher-timeframe trend gate: reject signals counter to a strong 1d trend.
        if rule_dir in ('LONG', 'SHORT'):
            htf_ok, htf_reason = await self._htf_trend_allows(symbol, rule_dir)
            if not htf_ok:
                logger.info(
                    f"⚠️ {symbol}: HTF (1d) trend gate rejected {rule_dir} — {htf_reason}"
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=None,
                    rejection_code="htf_trend_counter",
                    rejection_reason=f"Counter to 1d trend: {htf_reason}",
                    min_confidence=None,
                )
                if not active_thesis_context:
                    return None
                logger.info(
                    "Continuing %s through contextual AI revalidation despite HTF gate",
                    symbol,
                )

        # Expected Value gate: reject directions with historically negative E[PnL]
        if getattr(settings, "GATE_ENABLE_NEGATIVE_EV", False) and rule_dir in ('LONG', 'SHORT'):
            ev_data = await self._compute_conditional_expected_value(
                symbol, rule_dir, regime.get('type', 'unknown')
            )
            if ev_data and ev_data['expected_value'] < -0.3 and ev_data['sample_size'] >= 20:
                logger.info(
                    f"⚠️ {symbol}: Negative EV gate — {rule_dir} has "
                    f"E[PnL]={ev_data['expected_value']:+.2f}% "
                    f"(WR={ev_data['win_rate']:.1%}, n={ev_data['sample_size']}). "
                    f"Skipping LLM call."
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=None,
                    rejection_code="negative_expected_value",
                    rejection_reason=(
                        f"Historical E[PnL]={ev_data['expected_value']:+.2f}% for {rule_dir} "
                        f"(WR={ev_data['win_rate']:.1%}, avg_win={ev_data['avg_win']:+.1f}%, "
                        f"avg_loss={ev_data['avg_loss']:.1f}%, n={ev_data['sample_size']})"
                    ),
                    min_confidence=None,
                )
                if not active_thesis_context:
                    return None
                logger.info(
                    "Continuing %s through contextual AI revalidation despite negative-EV gate",
                    symbol,
                )

        scoped_learning_context: Dict[str, Any] = {
            "has_evidence": False,
            "hard_veto": False,
            "prompt_text": "- Scoped learning context not evaluated.",
            "reason": "not_evaluated",
            "evidence": [],
        }
        if rule_dir in ("LONG", "SHORT"):
            scoped_learning_context = await self._get_pre_generation_learning_context(
                symbol,
                ta,
                opp,
                rule_dir,
                max(float(rule_conf or 0.0), float(opp.overall_score or 0.0)),
            )
            if getattr(settings, "GATE_ENABLE_SCOPED_LEARNING_VETO", False) and scoped_learning_context.get("hard_veto"):
                veto_reason = str(scoped_learning_context.get("veto_reason") or scoped_learning_context.get("reason"))
                logger.info(
                    "⚠️ %s: Scoped learning pre-LLM veto — %s",
                    symbol,
                    veto_reason,
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=None,
                    rejection_code="scoped_learning_pre_llm_veto",
                    rejection_reason=veto_reason,
                    min_confidence=None,
                )
                if not active_thesis_context:
                    return None
                logger.info(
                    "Continuing %s through contextual AI revalidation despite scoped-learning veto",
                    symbol,
                )

        p_rule: Optional[float] = None
        if rule_dir in ("LONG", "SHORT"):
            p_rule = self._eval_policy_win_prob(ta, opp, rule_conf, regime_type, rule_dir)
            pre_thresh = float(getattr(settings, "POLICY_PRE_LLM_VETO_THRESHOLD", 0.36) or 0.36)
            if getattr(settings, "GATE_ENABLE_POLICY_PRE_LLM_VETO", False) and p_rule is not None and p_rule < pre_thresh:
                logger.info(
                    "⚠️ %s: Policy pre-LLM veto — %s p_win=%.3f < %.3f",
                    symbol,
                    rule_dir,
                    p_rule,
                    pre_thresh,
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=None,
                    rejection_code="policy_pre_llm_veto",
                    rejection_reason=f"Offline policy P(win)={p_rule:.3f} below {pre_thresh} for {rule_dir}",
                    min_confidence=None,
                )
                if not active_thesis_context:
                    return None
                logger.info(
                    "Continuing %s through contextual AI revalidation despite policy veto",
                    symbol,
                )

        min_conf = self._policy_tune_min_conf(self._minimum_accept_confidence(opp, rule_conf), p_rule)

        # Pre-fetch liquidation clusters (5-min cached) so the sync SL adjuster can use them.
        liq_clusters: Dict[float, float] = {}
        if rule_dir in ('LONG', 'SHORT'):
            try:
                liq_clusters = await self._fetch_liquidation_zones(symbol)
            except Exception:
                pass

        # LLM: synthesis, levels, and guardrails — predictive policy + rules set the prior.
        ai = await self.ai_analyze_opportunity(
            symbol,
            ta,
            opp,
            policy_win_probability=p_rule,
            policy_eval_direction=rule_dir if rule_dir in ("LONG", "SHORT") else None,
            rule_direction=rule_dir,
            rule_confidence=rule_conf,
            regime_type=regime_type,
            scoped_learning_context=scoped_learning_context,
            active_thesis_context=active_thesis_context,
        )
        if ai:
            ai.ensemble_rule_confidence = rule_conf
            ai.ensemble_rule_direction = rule_dir
            ai.ensemble_regime_type = regime_type
            if ai.recommendation in {"LONG", "SHORT"}:
                original_signal_age_hours = 0.0
                if active_thesis_context:
                    started_at = self._parse_iso_datetime(
                        active_thesis_context.get("analysis_timestamp")
                        or active_thesis_context.get("created_at"),
                        default=datetime.utcnow(),
                    )
                    original_signal_age_hours = max(
                        0.0,
                        (datetime.utcnow() - started_at).total_seconds() / 3600.0,
                    )
                self._attach_signal_edge_estimate(
                    ai,
                    ta,
                    opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    signal_age_hours=original_signal_age_hours,
                )

        calibrated_edge_override = self._calibrated_edge_allows_below_confidence(
            ai,
            rule_dir,
        )

        # Log detailed ensemble decision process
        ai_info = f"AI: {ai.recommendation if ai else 'None'} @ {ai.confidence:.3f}" if ai else "AI: None"
        logger.debug(f"🤖 Ensemble decision for {symbol}: Rule: {rule_dir} @ {rule_conf:.3f}, {ai_info}")

        if reduced_conservatism:
            # More permissive mode - accept AI if it has a directional recommendation AND meets enhanced minimum confidence
            if (
                ai
                and ai.recommendation in ['LONG', 'SHORT']
                and (ai.confidence >= min_conf or calibrated_edge_override)
            ):
                logger.info(
                    f"✅ {symbol}: Accepting AI decision ({ai.recommendation} @ {ai.confidence:.3f}, min={min_conf:.2f}) "
                    f"[adaptive reduced mode, regime={regime_type}]"
                )
                ai.reasoning = f"{ai.reasoning}\nEnsemble decision with reduced conservatism"
                ai.stop_loss = self._adjust_sl_for_liq_clusters(
                    symbol, ai.recommendation, ai.entry_price, ai.stop_loss, liq_clusters
                )
                return self._guardrail_policy_or_none(
                    symbol, ta, opp, rule_dir, rule_conf, regime_type, ai, min_conf, normalized_symbol
                )
            elif ai and ai.recommendation in ['LONG', 'SHORT'] and ai.confidence < min_conf:
                logger.info(
                    f"❌ {symbol}: AI confidence too low ({ai.confidence:.3f} < {min_conf:.2f}) "
                    f"[adaptive reduced mode, regime={regime_type}]"
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=ai,
                    rejection_code="ai_confidence_below_threshold",
                    rejection_reason=f"AI confidence {ai.confidence:.3f} below minimum {min_conf:.2f}",
                    min_confidence=min_conf,
                    regime_type=regime_type,
                )
            else:
                logger.info(
                    f"❌ {symbol}: AI did not provide directional signal ({ai_info}) "
                    f"[adaptive reduced mode, regime={regime_type}]"
                )
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=ai,
                    rejection_code="ai_non_directional",
                    rejection_reason="AI did not provide a directional LONG/SHORT decision",
                    min_confidence=min_conf,
                    regime_type=regime_type,
                )
            return None

        # ORIGINAL STRICT ENSEMBLE LOGIC BELOW
        if rule_dir == 'HOLD' or rule_conf < 0.50:
            # If rules say HOLD or have low confidence, allow AI to propose if confident
            if (
                ai
                and ai.recommendation in ['LONG', 'SHORT']
                and (ai.confidence >= min_conf or calibrated_edge_override)
            ):
                logger.info(f"✅ {symbol}: Accepting confident AI decision ({ai.recommendation} @ {ai.confidence:.3f}) over neutral/weak rules")
                ai.reasoning = f"{ai.reasoning}\nRule engine neutral/weak; accepting confident AI decision"
                ai.stop_loss = self._adjust_sl_for_liq_clusters(
                    symbol, ai.recommendation, ai.entry_price, ai.stop_loss, liq_clusters
                )
                return self._guardrail_policy_or_none(
                    symbol, ta, opp, rule_dir, rule_conf, regime_type, ai, min_conf, normalized_symbol
                )
            logger.debug(f"❌ {symbol}: Rules HOLD/weak, AI not confident enough ({ai_info})")
            self._record_rejection_context(
                symbol=symbol,
                technical_analysis=ta,
                opportunity_score=opp,
                rule_direction=rule_dir,
                rule_confidence=rule_conf,
                ai_decision=ai,
                rejection_code="rules_hold_or_weak",
                rejection_reason="Rule engine was HOLD/weak and AI confirmation was insufficient",
                min_confidence=min_conf,
                regime_type=regime_type,
            )
            return None

        # Check for agreement between rules and AI - both must meet enhanced minimum confidence
        if (
            ai
            and ai.recommendation in ['LONG', 'SHORT']
            and (ai.confidence >= min_conf or calibrated_edge_override)
        ):
            if (
                ai.recommendation == rule_dir
                and (ai.confidence >= min_conf or calibrated_edge_override)
                and rule_conf >= min(0.50, min_conf)
            ):
                logger.info(f"✅ {symbol}: AI and rules agree ({rule_dir}) with sufficient confidence")
                ai.reasoning = f"{ai.reasoning}\nRule engine agrees: {rule_reason}"
                # Blend confidences weighted by historical accuracy of each source.
                w_rule = self._ensemble_weights.get('rule', 0.5)
                w_llm = self._ensemble_weights.get('llm', 0.5)
                ai.confidence = min(1.0, w_rule * rule_conf + w_llm * ai.confidence + 0.02)
                logger.debug(
                    "⚖️  %s: blended confidence=%.3f (w_rule=%.2f, w_llm=%.2f)",
                    symbol, ai.confidence, w_rule, w_llm,
                )
                ai.stop_loss = self._adjust_sl_for_liq_clusters(
                    symbol, ai.recommendation, ai.entry_price, ai.stop_loss, liq_clusters
                )
                return self._guardrail_policy_or_none(
                    symbol, ta, opp, rule_dir, rule_conf, regime_type, ai, min_conf, normalized_symbol
                )

            # NEW: Allow high-confidence AI to override weak rule disagreement
            elif ai.confidence >= 0.70 and rule_conf < 0.7:
                logger.info(f"✅ {symbol}: High-confidence AI ({ai.recommendation} @ {ai.confidence:.3f}) overriding weak rules ({rule_dir} @ {rule_conf:.3f})")
                ai.reasoning = f"{ai.reasoning}\nHigh-confidence AI overriding weak rule signal ({rule_reason})"
                ai.stop_loss = self._adjust_sl_for_liq_clusters(
                    symbol, ai.recommendation, ai.entry_price, ai.stop_loss, liq_clusters
                )
                return self._guardrail_policy_or_none(
                    symbol, ta, opp, rule_dir, rule_conf, regime_type, ai, min_conf, normalized_symbol
                )
            elif not getattr(settings, "GATE_ENABLE_ENSEMBLE_DISAGREEMENT", False):
                # Disagreement gate removed: the AI is directional and already cleared the
                # confidence floor (outer if), so accept it rather than vetoing on a rule
                # mismatch. The confidence floor + directional learning govern quality.
                logger.info(
                    f"✅ {symbol}: Accepting confident AI ({ai.recommendation} @ {ai.confidence:.3f}) "
                    f"despite rule disagreement ({rule_dir} @ {rule_conf:.3f}) [disagreement gate off]"
                )
                ai.reasoning = f"{ai.reasoning}\nAI accepted over rule disagreement (disagreement gate disabled)"
                ai.stop_loss = self._adjust_sl_for_liq_clusters(
                    symbol, ai.recommendation, ai.entry_price, ai.stop_loss, liq_clusters
                )
                return self._guardrail_policy_or_none(
                    symbol, ta, opp, rule_dir, rule_conf, regime_type, ai, min_conf, normalized_symbol
                )
            else:
                logger.debug(f"❌ {symbol}: AI/rules disagree or insufficient confidence - AI: {ai.recommendation} @ {ai.confidence:.3f}, Rules: {rule_dir} @ {rule_conf:.3f}")
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=ai,
                    rejection_code="ensemble_disagreement",
                    rejection_reason="AI and rule-engine direction/confidence did not agree",
                    min_confidence=min_conf,
                    regime_type=regime_type,
                )

        else:
            if ai and ai.confidence < min_conf:
                logger.debug(f"❌ {symbol}: AI confidence below enhanced minimum threshold ({ai.confidence:.3f} < {min_conf:.2f})")
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=ai,
                    rejection_code="ai_confidence_below_threshold",
                    rejection_reason=f"AI confidence {ai.confidence:.3f} below minimum {min_conf:.2f}",
                    min_confidence=min_conf,
                    regime_type=regime_type,
                )
            else:
                logger.debug(f"❌ {symbol}: No valid AI decision received")
                self._record_rejection_context(
                    symbol=symbol,
                    technical_analysis=ta,
                    opportunity_score=opp,
                    rule_direction=rule_dir,
                    rule_confidence=rule_conf,
                    ai_decision=ai,
                    rejection_code="ai_missing_or_invalid",
                    rejection_reason="No valid AI decision received during strict ensemble",
                    min_confidence=min_conf,
                    regime_type=regime_type,
                )

        # Disagreement or no AI → HOLD
        logger.debug(f"❌ {symbol}: Ensemble rejected - disagreement or insufficient confidence")
        return None

    async def generate_signals_batch(
        self,
        opportunities: List[str],
        run_id: str,
        max_signals: int = 5,
        pending_theses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PlatformSignal]:
        """
        Generate high-quality signals from opportunity list.

        Process opportunities in batches with rate limiting.
        """
        logger.info(f"🎯 Generating signals for {len(opportunities)} opportunities (target: {max_signals})")

        # Batch-prefetch OHLCV data for all candidates to avoid per-symbol rate limiting
        await self.prefetch_ohlcv_batch(opportunities)

        signals = []
        processed = 0
        pending_theses = pending_theses or {}

        # Process opportunities in small batches
        for i in range(0, len(opportunities), 3):  # 3 tokens per batch
            batch = opportunities[i:i+3]
            logger.debug(f"📊 Processing batch {i//3 + 1}: {batch}")

            for symbol in batch:
                if len(signals) >= max_signals:
                    break

                try:
                    processed += 1
                    logger.debug(f"🔍 Analyzing {symbol} ({processed}/{len(opportunities)})")
                    pending_thesis = pending_theses.get(self._signal_symbol_key(symbol))

                    # 1. Technical analysis
                    technical_analysis = await self.perform_technical_analysis(symbol)
                    if not technical_analysis:
                        logger.debug(f"⚠️ Technical analysis failed for {symbol}")
                        continue

                    # 2. Calculate position in range for rule engine
                    if technical_analysis.high_24h != technical_analysis.low_24h:
                        position_in_range = (technical_analysis.current_price - technical_analysis.low_24h) / (technical_analysis.high_24h - technical_analysis.low_24h)
                    else:
                        position_in_range = 0.5

                    # Get opportunity score from discovery stage cache (avoid flat 0.7 baseline).
                    cached_score = (
                        self._opportunity_score_cache.get(symbol)
                        or self._opportunity_score_cache.get(self._normalize_symbol_to_market_id(symbol))
                    )
                    if cached_score:
                        base_overall = float(cached_score.overall_score)
                        base_volume = float(cached_score.volume_score)
                        base_volatility = float(cached_score.volatility_score)
                        base_institutional = float(cached_score.institutional_score)
                        base_breakdown = dict(cached_score.breakdown or {})
                    else:
                        # Fallback only if cache is unavailable for this symbol.
                        base_overall = 0.7
                        base_volume = 0.7
                        base_volatility = 0.7
                        base_institutional = 0.5
                        base_breakdown = {}

                    base_breakdown.update({
                        'position_in_range': position_in_range,
                        'price_direction': 'bullish' if technical_analysis.price_change_24h > 0 else 'bearish',
                        'price_change_24h': technical_analysis.price_change_24h
                    })

                    opportunity_score = OpportunityScore(
                        symbol=symbol,
                        overall_score=base_overall,
                        volume_score=base_volume,
                        volatility_score=base_volatility,
                        momentum_score=technical_analysis.momentum_score,
                        trend_score=technical_analysis.strength_score,
                        institutional_score=base_institutional,
                        breakdown=base_breakdown
                    )
                    opportunity_score.breakdown['setup_type'] = self._classify_setup(
                        technical_analysis,
                        opportunity_score,
                    )

                    # Load the accepted lifecycle before analysis so the model
                    # evaluates the original thesis rather than treating this as
                    # an unrelated new candidate for the same token.
                    token_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
                    token_candidates = [token_symbol]
                    if token_symbol.upper().endswith('USDT'):
                        token_candidates.append(token_symbol[:-4])
                    elif ':' not in token_symbol:
                        token_candidates.append(f'{token_symbol}USDT')
                    existing_check = self.supabase.from_('platform_signals').select(
                        'signal_id,direction,analysis_timestamp,created_at,expires_at,status,'
                        'confidence,id,entry_price,stop_loss,target_1,target_2,time_horizon,'
                        'leverage,position_size,analysis_notes,ai_reasoning,market_conditions'
                    ).in_('token_symbol', list(dict.fromkeys(token_candidates))).in_(
                        'status', list(THESIS_LIVE_STATUSES)
                    ).order('analysis_timestamp', desc=True).execute()

                    live_theses = [
                        row for row in (existing_check.data or [])
                        if signal_thesis_is_live(row)
                    ]
                    if (
                        pending_thesis
                        and signal_thesis_is_live(pending_thesis)
                        and not any(
                            str(row.get("signal_id")) == str(pending_thesis.get("signal_id"))
                            for row in live_theses
                        )
                    ):
                        # Stored symbols may be either XRP or XRPUSDT depending on
                        # the source adapter. The canonical pending lookup is
                        # authoritative when the exact-token query misses.
                        live_theses.insert(0, pending_thesis)
                    existing_signal = live_theses[0] if live_theses else None

                    # 3. Ensemble decision (rules + AI)
                    decision = await self._ensemble_decision(
                        symbol,
                        technical_analysis,
                        opportunity_score,
                        active_thesis_context=existing_signal,
                    )
                    if not decision:
                        logger.debug(f"⚠️ Ensemble rejected: {symbol}")
                        normalized_symbol = self._normalize_symbol_to_market_id(symbol)
                        rejection_context = (
                            self._last_rejection_context.get(normalized_symbol)
                            or self._last_rejection_context.get(str(symbol or "").upper().strip())
                            or {}
                        )
                        pending_conditions = (
                            pending_thesis.get("market_conditions") or {}
                            if pending_thesis
                            else {}
                        )
                        pending_reanalysis_reason = str(
                            pending_conditions.get("reanalysis_reason") or ""
                        ).lower()
                        fresh_negative_ev_review = (
                            pending_thesis
                            and pending_reanalysis_reason
                            == "negative_net_expected_value_live_guard"
                            and rejection_context.get("rejection_code") not in {
                                None,
                                "",
                                "ai_missing_or_invalid",
                            }
                        )
                        if fresh_negative_ev_review:
                            await self._record_pending_revalidation(
                                pending_thesis,
                                result="skipped_negative_ev",
                                reason=(
                                    "fresh full-stack review did not recover positive executable EV: "
                                    + str(
                                        rejection_context.get("rejection_reason")
                                        or rejection_context.get("rejection_code")
                                    )
                                ),
                            )
                        elif pending_thesis and self._pending_rejection_confirms_invalidation(rejection_context):
                            await self._record_pending_revalidation(
                                pending_thesis,
                                result="invalidated",
                                reason=str(
                                    rejection_context.get("rejection_reason")
                                    or rejection_context.get("rejection_code")
                                    or "setup no longer passes the full signal decision stack"
                                ),
                            )
                        elif pending_thesis and rejection_context.get("rejection_code") not in {
                            None,
                            "",
                            "ai_missing_or_invalid",
                        }:
                            rejection_code = str(rejection_context.get("rejection_code") or "")
                            revalidation_result = (
                                "paused"
                                if rejection_code in {"ensemble_disagreement", "rules_hold_or_weak"}
                                else "retained"
                            )
                            await self._record_pending_revalidation(
                                pending_thesis,
                                result=revalidation_result,
                                reason=(
                                    "scheduled re-test found no structural invalidation; "
                                    + (
                                        "entry remains paused until directional agreement returns"
                                        if revalidation_result == "paused"
                                        else "original thesis retained"
                                    )
                                ),
                            )
                        await self._log_rejected_signal_candidate(
                            symbol=symbol,
                            technical_analysis=technical_analysis,
                            opportunity_score=opportunity_score,
                            run_id=run_id,
                        )
                        await asyncio.sleep(1)  # Keep lightweight pacing after rejections
                        continue

                    # 4. Smart signal conflict resolution
                    reversed_theses: List[Dict[str, Any]] = []
                    if live_theses:
                        existing_direction = existing_signal['direction']
                        existing_confidence = existing_signal.get('confidence', 0.5)

                        # Check if this is a directional conflict (LONG vs SHORT)
                        if existing_direction != decision.recommendation:
                            reason = "accepted opposite-direction platform thesis"
                            logger.info(
                                f"🔄 Signal reversal for {symbol}: "
                                f"{existing_direction}→{decision.recommendation} "
                                f"(conf {existing_confidence:.3f}→{decision.confidence:.3f})"
                            )
                            await self._notify_signal_reversal(
                                symbol=token_symbol,
                                old_direction=existing_direction,
                                new_direction=decision.recommendation,
                                old_confidence=existing_confidence,
                                new_confidence=decision.confidence,
                                reason=reason
                            )

                            try:
                                invalidated_at = datetime.now(timezone.utc).isoformat()
                                for thesis in live_theses:
                                    thesis_mc = dict(thesis.get('market_conditions') or {})
                                    thesis_mc['reversed_at'] = invalidated_at
                                    thesis_mc['reversal_reason'] = reason
                                    thesis_mc['reversed_by_direction'] = decision.recommendation
                                    thesis_status = str(thesis.get('status') or '').lower()
                                    update_payload = {
                                        'analysis_notes': (
                                            f'Replaced by opposite {decision.recommendation} thesis '
                                            f'({reason})'
                                        ),
                                        'market_conditions': thesis_mc,
                                    }
                                    if thesis_status == 'active':
                                        thesis_mc['invalidated_at'] = invalidated_at
                                        thesis_mc['invalidation_reason'] = reason
                                        thesis_mc['invalidated_by_direction'] = decision.recommendation
                                        update_payload['status'] = 'invalidated'
                                    self.supabase.from_('platform_signals').update(
                                        update_payload
                                    ).eq('id', thesis['id']).execute()
                                    if thesis_status == 'active':
                                        try:
                                            (
                                                self.supabase.from_('platform_signal_performance_tracking')
                                                .update({
                                                    'outcome': 'invalidated',
                                                    'exit_reason': 'INVALIDATED',
                                                    'exit_timestamp': invalidated_at,
                                                    'last_updated': invalidated_at,
                                                })
                                                .eq('signal_id', thesis['signal_id'])
                                                .eq('outcome', 'active')
                                                .execute()
                                            )
                                        except Exception as perf_exc:
                                            logger.warning(
                                                "Could not synchronize reversal invalidation for %s: %s",
                                                thesis.get('signal_id'),
                                                perf_exc,
                                            )
                                    thesis['market_conditions'] = thesis_mc
                                reversed_theses = list(live_theses)
                                logger.info(f"✅ Invalidated previous {existing_direction} thesis for {symbol}")
                            except Exception as e:
                                logger.error(f"❌ Failed to invalidate existing thesis: {e}")
                                continue
                        else:
                            if pending_thesis:
                                fresh_rule_direction = str(
                                    getattr(decision, "ensemble_rule_direction", None) or "HOLD"
                                ).upper()
                                if fresh_rule_direction != decision.recommendation:
                                    await self._record_pending_revalidation(
                                        pending_thesis,
                                        result="paused",
                                        reason=(
                                            "fresh AI/rule agreement disappeared "
                                            f"(AI={decision.recommendation}, rule={fresh_rule_direction})"
                                        ),
                                        decision=decision,
                                    )
                                    continue
                                decision_edge = (
                                    decision.edge_estimate
                                    if isinstance(decision.edge_estimate, dict)
                                    else {}
                                )
                                try:
                                    fresh_net_ev = float(
                                        decision_edge.get("net_expected_value_pct")
                                    )
                                except (TypeError, ValueError):
                                    fresh_net_ev = None
                                min_net_ev = float(
                                    getattr(settings, "SIGNAL_MIN_NET_EXPECTED_VALUE_PCT", 0.0)
                                    or 0.0
                                )
                                if fresh_net_ev is not None and reliable_negative_edge_veto(
                                    decision_edge,
                                    threshold_pct=min_net_ev,
                                ):
                                    await self._record_pending_revalidation(
                                        pending_thesis,
                                        result="skipped_negative_ev",
                                        reason=(
                                            "fresh full-stack review estimated net EV "
                                            f"{fresh_net_ev:+.6f}% at/below {min_net_ev:+.6f}%"
                                        ),
                                        decision=decision,
                                    )
                                    continue
                                await self._record_pending_revalidation(
                                    pending_thesis,
                                    result="confirmed",
                                    reason="scheduled analysis still confirms the original direction",
                                    decision=decision,
                                )
                            logger.info(
                                f"⚠️ Skipping {symbol}: live {existing_direction} thesis "
                                f"{existing_signal.get('signal_id')} remains valid through "
                                f"{existing_signal.get('expires_at')}; re-analysis cannot create a second order"
                            )
                            continue

                    # 5. Accept decision and build signal with BTC market context
                    # Get BTC market data for learning integration
                    btc_data = await self.get_btc_market_data()
                    platform_signal = await self.generate_platform_signal(
                        symbol, decision, technical_analysis, btc_data, opportunity_score
                    )
                    platform_signal.run_id = run_id
                    platform_signal.opportunity_rank = len(signals) + 1

                    # Final execution-time sanity check with fresh market price.
                    fresh_price = await self._fetch_fresh_symbol_price(symbol)
                    if fresh_price and fresh_price > 0:
                        if not self.targets_are_feasible(
                            decision.recommendation,
                            float(fresh_price),
                            float(platform_signal.target_1),
                            float(platform_signal.target_2),
                            entry_price=float(platform_signal.entry_price),
                            entry_strategy=getattr(decision, 'entry_strategy', None),
                            stop_loss=float(platform_signal.stop_loss),
                        ):
                            logger.warning(
                                f"Skipping {symbol} {decision.recommendation} signal - "
                                f"targets no longer feasible at fresh price {fresh_price}"
                            )
                            continue
                        # Keep stored market context aligned with the latest known price.
                        if isinstance(platform_signal.market_conditions, dict):
                            platform_signal.market_conditions['current_price'] = float(fresh_price)
                            platform_signal.market_conditions.update(
                                self._build_entry_activation_metadata(
                                    direction=decision.recommendation,
                                    entry_price=decision.entry_price,
                                    current_price=float(fresh_price),
                                    entry_strategy=getattr(decision, 'entry_strategy', None),
                                    signal_started_at=platform_signal.analysis_timestamp,
                                    signal_expires_at=platform_signal.expires_at,
                                )
                            )

                    # Passively link the reversal pair (old thesis <-> replacement)
                    # so counterfactuals can be evaluated later. Recording only:
                    # nothing reads these links to alter production decisions.
                    if reversed_theses:
                        reversed_ids = [
                            str(thesis.get('signal_id'))
                            for thesis in reversed_theses
                            if thesis.get('signal_id')
                        ]
                        if isinstance(platform_signal.market_conditions, dict) and reversed_ids:
                            platform_signal.market_conditions['reversal_of_signal_ids'] = reversed_ids
                        try:
                            for thesis in reversed_theses:
                                thesis_mc = dict(thesis.get('market_conditions') or {})
                                thesis_mc['reversed_by_signal_id'] = platform_signal.signal_id
                                self.supabase.from_('platform_signals').update({
                                    'market_conditions': thesis_mc,
                                }).eq('id', thesis['id']).execute()
                        except Exception as link_error:
                            logger.warning(f"Could not link reversal pair for {symbol}: {link_error}")

                    signals.append(platform_signal)
                    logger.info(f"✅ Signal generated: {symbol} {decision.recommendation} (confidence: {decision.confidence:.2f})")

                    if len(signals) >= max_signals:
                        logger.info(f"🎯 Reached target of {max_signals} signals, stopping early")
                        break

                    # Rate limiting between tokens
                    await asyncio.sleep(1)  # Reasonable delay between tokens

                except Exception as e:
                    logger.warning(f"❌ Analysis failed for {symbol}: {e}")
                    continue

            # Break if we have enough signals
            if len(signals) >= max_signals:
                break

            # Rate limiting between batches
            if i + 3 < len(opportunities):
                await asyncio.sleep(2)  # Reasonable delay between batches

        # Rank by independent, cost-adjusted edge rather than generation order
        # or duplicated LLM confidence.
        signals.sort(
            key=lambda signal: float(
                ((signal.market_conditions or {}).get("edge_estimate") or {}).get("edge_score")
                or signal.overall_score
                or 0.0
            ),
            reverse=True,
        )
        for rank, signal in enumerate(signals, start=1):
            signal.opportunity_rank = rank

        logger.info(f"✅ Generated {len(signals)} high-quality signals from {processed} analyzed tokens")
        return signals

    async def run_event_reanalysis_requests(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Rebuild queued stale/materially-changed entries from fresh inputs.

        This is deliberately a new signal decision, not a resubmission of the
        old levels: direction, regime, entry, stop, targets and calibrated EV
        are regenerated before the replacement can reach Yuki.
        """
        if not requests:
            return {"processed": 0, "saved": 0, "rearmed": 0, "run_id": None}

        # The worker processes queue items independently so each completed
        # signal can commit before the next one starts. Reuse its long-lived
        # clients instead of replacing the Binance/LLM sessions per signal.
        if not self.binance or not self.llm_client or not self.platform_service:
            await self._initialize_services()
        await self._refresh_execution_fill_model()
        run_id = f"event_reanalysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._challenger_calls_this_run = 0
        symbols: List[str] = []
        request_by_symbol: Dict[str, Dict[str, Any]] = {}
        pending_theses: Dict[str, Dict[str, Any]] = {}
        now = datetime.utcnow()

        for request in requests:
            token = str(request.get("token_symbol") or "").upper().strip()
            if not token or ":" in token:
                # The Binance generation adapter cannot safely reconstruct HIP-3
                # markets; leave the request pending for a venue-compatible path.
                continue
            signal_id = str(request.get("signal_id") or "")
            request_state_hash = self._reanalysis_request_state_hash(request)
            if (
                signal_id
                and request_state_hash
                and self._should_coalesce_reanalysis_request(
                    request,
                    request_state_hash,
                    now=now,
                )
            ):
                logger.info(
                    "Skipping re-analysis for %s: pending request state hash unchanged",
                    signal_id or token,
                )
                await self.platform_service.mark_reanalysis_request_coalesced(
                    signal_id,
                    request_state_hash,
                )
                continue
            if token.endswith("USDT"):
                token = token[:-4]
            symbol = f"{token}/USDT:USDT"
            key = self._signal_symbol_key(symbol)
            if not key or key in request_by_symbol:
                continue
            if signal_id and request_state_hash:
                await self.platform_service.record_reanalysis_attempt(
                    signal_id,
                    request_state_hash,
                )
            request_by_symbol[key] = request
            symbols.append(symbol)
            if str(request.get("status") or "").lower() in THESIS_LIVE_STATUSES:
                pending_theses[key] = request

            # Rebuild the discovery score from a current ticker rather than the
            # legacy flat 0.7 fallback used by forced symbols.
            try:
                ticker = await self.rate_limited_api_call(self.binance.fetch_ticker, symbol)
                score = await self._calculate_opportunity_score(symbol, ticker or {})
                if score:
                    self._opportunity_score_cache[symbol] = score
                    self._opportunity_score_cache[self._normalize_symbol_to_market_id(symbol)] = score
            except Exception as exc:
                logger.warning("Fresh opportunity score unavailable for re-analysis %s: %s", symbol, exc)

        if not symbols:
            return {"processed": 0, "saved": 0, "rearmed": 0, "run_id": run_id}

        await self._refresh_signal_outcome_confidence_context()
        self._refresh_adaptive_confidence_target()
        self.reload_signal_policy_model()
        generated = await self.generate_signals_batch(
            symbols,
            run_id,
            max_signals=len(symbols),
            pending_theses=pending_theses,
        )

        saved_by_source: Dict[str, str] = {}
        for signal in generated:
            key = self._signal_symbol_key(signal.token_symbol)
            source = request_by_symbol.get(key)
            if source and isinstance(signal.market_conditions, dict):
                source_conditions = source.get("market_conditions") or {}
                old_price = float(
                    source_conditions.get("market_price_at_generation")
                    or source_conditions.get("current_price")
                    or source.get("entry_price")
                    or 0.0
                )
                new_price = float(signal.market_conditions.get("current_price") or 0.0)
                old_volume = float(source_conditions.get("volume_24h") or 0.0)
                new_volume = float(signal.market_conditions.get("volume_24h") or 0.0)
                price_drift = (
                    abs(new_price - old_price) / old_price if old_price > 0 and new_price > 0 else None
                )
                volume_drift = (
                    abs(new_volume - old_volume) / old_volume if old_volume > 0 and new_volume > 0 else None
                )
                old_trend = str(source_conditions.get("trend_direction") or "").lower() or None
                new_trend = str(signal.market_conditions.get("trend_direction") or "").lower() or None
                feature_drift = {
                    "price_drift_pct": (
                        round(price_drift * 100.0, 6) if price_drift is not None else None
                    ),
                    "volume_drift_pct": (
                        round(volume_drift * 100.0, 6) if volume_drift is not None else None
                    ),
                    "old_trend": old_trend,
                    "new_trend": new_trend,
                    "trend_changed": bool(old_trend and new_trend and old_trend != new_trend),
                }
                feature_drift["materially_changed"] = bool(
                    (price_drift or 0.0)
                    >= float(
                        getattr(settings, "YUKI_ENTRY_REVALIDATION_PRICE_MOVE_PCT", 0.0125)
                        or 0.0125
                    )
                    or (volume_drift or 0.0)
                    >= float(
                        getattr(settings, "YUKI_ENTRY_REVALIDATION_VOLUME_CHANGE_PCT", 0.10)
                        or 0.10
                    )
                    or feature_drift["trend_changed"]
                )
                signal.market_conditions.update({
                    "reanalysis_of_signal_id": source.get("signal_id"),
                    "reanalysis_trigger": source_conditions.get("reanalysis_reason"),
                    "reanalysis_generated_at": datetime.now(timezone.utc).isoformat(),
                    "feature_drift_from_original": feature_drift,
                })
                edge_estimate = signal.market_conditions.get("edge_estimate") or {}
                if not bool(edge_estimate.get("ai_rule_agreement")):
                    logger.info(
                        "Rejecting stale-entry replacement %s for %s: fresh AI/rule agreement disappeared",
                        signal.signal_id,
                        source.get("signal_id"),
                    )
                    continue
            signal.run_id = run_id
            if await self.platform_service.save_platform_signal(signal):
                if source:
                    saved_by_source[str(source.get("signal_id"))] = signal.signal_id
                if self.multi_level_learning:
                    await self.multi_level_learning.start_real_time_tracking(signal)

        processed = 0
        rearmed = 0
        for key, request in request_by_symbol.items():
            source_id = str(request.get("signal_id") or "")
            if not source_id:
                continue
            replacement_id = saved_by_source.get(source_id)
            current_status = str(request.get("status") or "").lower()
            request_conditions = request.get("market_conditions") or {}
            reanalysis_reason = str(
                request_conditions.get("reanalysis_reason") or ""
            ).lower()
            entry_geometry_is_terminal = (
                current_status in THESIS_LIVE_STATUSES
                and reanalysis_reason in ENTRY_REANALYSIS_INVALIDATION_REASONS
            )
            requested_at = self._parse_iso_datetime(
                request_conditions.get("reanalysis_requested_at"),
                default=datetime.min,
            )
            revalidated_at = self._parse_iso_datetime(
                request_conditions.get("last_entry_revalidated_at"),
                default=datetime.min,
            )
            latest_revalidation = str(
                request_conditions.get("last_entry_revalidation_result") or ""
            ).lower()
            fresh_revalidation = revalidated_at >= requested_at
            if replacement_id:
                result = "replaced"
            elif fresh_revalidation and latest_revalidation == "skipped_negative_ev":
                result = "skipped_negative_ev"
            elif entry_geometry_is_terminal:
                # A fresh HOLD/KEEP can preserve a broad directional thesis, but
                # it cannot make entry levels whose stop/target geometry was
                # already crossed executable again. Close that setup so a later
                # generation cycle can publish genuinely fresh levels.
                result = "fresh_stack_rejected"
            elif (
                current_status in THESIS_LIVE_STATUSES
                and fresh_revalidation
                and latest_revalidation in {"confirmed", "retained"}
            ):
                result = "revalidated_without_replacement"
            else:
                result = "fresh_stack_rejected"
            if await self.platform_service.mark_reanalysis_request_processed(
                source_id,
                result,
                replacement_signal_id=replacement_id,
            ):
                processed += 1
                if result == "revalidated_without_replacement":
                    rearmed += 1

        return {
            "processed": processed,
            "saved": len(saved_by_source),
            "rearmed": rearmed,
            "run_id": run_id,
            "replacement_signal_ids": saved_by_source,
        }

    async def run_full_analysis_cycle(self, max_signals: int = 5) -> str:
        """
        Run complete analysis cycle.

        Returns run_id of completed analysis.
        """
        run_id = f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"🚀 Starting unified analysis cycle: {run_id}")
        self._challenger_calls_this_run = 0

        try:
            # Ensure services are initialized
            if not self.binance or not self.llm_client or not self.platform_service:
                await self._initialize_services()
            await self._refresh_execution_fill_model()

            # Sync challenger monthly budget state once per run.
            current_month_spend = await self._sync_challenger_month_spend(force=True)
            monthly_budget = max(float(settings.CHALLENGER_MONTHLY_BUDGET_USD or 0.0), 0.0)
            if settings.CHALLENGER_ENABLED and monthly_budget > 0:
                usage_pct = (current_month_spend / monthly_budget * 100) if monthly_budget > 0 else 0.0
                logger.info(
                    f"💰 Challenger budget: ${current_month_spend:.4f}/${monthly_budget:.2f} used ({usage_pct:.1f}%)"
                )
                if current_month_spend >= monthly_budget:
                    logger.warning("🛑 Challenger budget cap reached; challenger disabled for this run")

            # Resolve pending rejected candidates so reject-quality metrics stay current.
            await self.refresh_reject_learning_backlog(max_to_process=80)
            # Add accepted-signal performance learning and blend into one confidence target.
            await self._refresh_signal_outcome_confidence_context()
            self._refresh_adaptive_confidence_target()
            # Calibrate validity windows and ensemble weights from resolved-signal data (24h TTL).
            await self._calibrate_validity_windows()
            await self._refresh_ensemble_accuracy_weights()
            logger.info(
                "🧠 Adaptive confidence target: %.3f (reject_adj=%+.3f, signal_adj=%+.3f)",
                self._adaptive_confidence_target,
                float(self._reject_quality_conf_adjustment or 0.0),
                float(self._signal_quality_conf_adjustment or 0.0),
            )
            self.reload_signal_policy_model()

            # 1. Discover market opportunities
            opportunities = await self.discover_market_opportunities(max_opportunities=25)  # Analyze all opportunities with proper batching
            pending_theses = await self._load_pending_theses_for_revalidation()
            if pending_theses:
                opportunities = self._prepend_pending_revalidation_symbols(opportunities, pending_theses)
                logger.info(
                    "🔁 Scheduled re-test queued for %d accepted pending signal(s)",
                    len(pending_theses),
                )
            scheduled_top_k = max(
                1,
                int(getattr(settings, "SIGNAL_GENERATION_EXPENSIVE_TOP_K", 10) or 10),
            )
            scheduled_candidate_limit = max(scheduled_top_k, len(pending_theses))
            if len(opportunities) > scheduled_candidate_limit:
                logger.info(
                    "✂️ Limiting scheduled expensive analysis from %d to %d top-ranked candidates",
                    len(opportunities),
                    scheduled_candidate_limit,
                )
                opportunities = opportunities[:scheduled_candidate_limit]
            if not opportunities:
                logger.warning("❌ No opportunities discovered")
                return run_id

            # 2. Generate signals
            signals = await self.generate_signals_batch(
                opportunities,
                run_id,
                max_signals,
                pending_theses=pending_theses,
            )

            # 3. Save signals to database and fetch logos
            saved_count = 0
            for signal in signals:
                try:
                    success = await self.platform_service.save_platform_signal(signal)
                    if success:
                        saved_count += 1

                        # Start real-time learning tracking for the new signal
                        if self.multi_level_learning:
                            await self.multi_level_learning.start_real_time_tracking(signal)

                        # Immediately fetch and update logo after successful save
                        try:
                            # Prepare symbol for logo fetching - handle symbols that already end with USDT
                            base_symbol = signal.token_symbol
                            if base_symbol.endswith('USDT'):
                                # Remove USDT suffix to get base token (e.g., TAOUSDT -> TAO)
                                base_symbol = base_symbol[:-4]
                            symbol_with_suffix = f"{base_symbol}/USDT"
                            logo_url = await self.fetch_token_logo_url(symbol_with_suffix)

                            if logo_url:
                                await self.platform_service.update_signal_logo(signal.signal_id, logo_url)
                                logger.info(f"🖼️ Updated logo for {signal.token_symbol}: {logo_url}")
                            else:
                                logger.warning(f"⚠️ No logo found for {signal.token_symbol}")

                            # Delay to avoid rate limiting logo API (CoinGecko free tier)
                            await asyncio.sleep(2.0)  # Increased from 1.0s to 2.0s

                        except Exception as logo_error:
                            logger.warning(f"⚠️ Logo fetch failed for {signal.token_symbol}: {logo_error}")

                except Exception as e:
                    logger.error(f"❌ Failed to save signal {signal.signal_id}: {e}")

            logger.info(f"✅ Analysis cycle complete: {run_id}")
            logger.info(f"📊 Results: {saved_count}/{len(signals)} signals saved with logos fetched")

            try:
                await asyncio.to_thread(self._maybe_auto_train_policy_sync)
            except Exception as auto_train_error:
                logger.warning("Policy auto-train thread failed: %s", auto_train_error)

            return run_id

        except Exception as e:
            logger.error(f"❌ Analysis cycle failed: {e}")
            raise
        finally:
            # Cleanup
            if self.binance:
                await self.binance.close()

    async def _calibrate_validity_windows(self) -> None:
        """
        Pull resolved signals from Supabase and compute the *median* actual resolution
        time (created_at → status change) per time_horizon bucket.  Results replace the
        hardcoded defaults in `_calculate_validity_window_hours` for any bucket that has
        ≥10 resolved samples.  Refreshed at most once per 24 h.
        """
        now = datetime.utcnow()
        if (
            self._validity_calibration_last_run is not None
            and (now - self._validity_calibration_last_run).total_seconds() < 86_400
        ):
            return
        if not self.supabase:
            return
        try:
            cutoff = (now - timedelta(days=90)).isoformat()
            result = (
                self.supabase
                .from_("platform_signals")
                .select("time_horizon,created_at,updated_at,status")
                .in_("status", ["hit_target_1", "hit_target_2", "hit_stop_loss", "expired"])
                .gte("created_at", cutoff)
                .limit(5000)
                .execute()
            )
            rows = result.data or []
            if not rows:
                self._validity_calibration_last_run = now
                return

            from collections import defaultdict
            import statistics

            bucket_hours: Dict[str, List[float]] = defaultdict(list)
            calibration_max_multiplier = max(
                float(getattr(settings, "VALIDITY_CALIBRATION_MAX_MULTIPLIER", 3.0) or 3.0),
                1.0,
            )
            for row in rows:
                horizon = str(row.get("time_horizon") or "").strip()
                if not horizon:
                    continue
                try:
                    created = datetime.fromisoformat(
                        str(row["created_at"]).replace("Z", "+00:00").replace("+00:00", "")
                    )
                    updated = datetime.fromisoformat(
                        str(row["updated_at"]).replace("Z", "+00:00").replace("+00:00", "")
                    )
                    elapsed_h = (updated - created).total_seconds() / 3600.0
                    if elapsed_h > 0:
                        # Normalise to canonical key (same logic as _calculate_validity_window_hours).
                        canonical = self._normalise_horizon_key(horizon)
                        baseline_h = float(self._default_validity_window_hours(canonical))
                        max_allowed_h = max(4.0, baseline_h * calibration_max_multiplier)
                        # Skip extreme stale-outcome outliers that distort short-horizon buckets.
                        if elapsed_h > max_allowed_h:
                            continue
                        bucket_hours[canonical].append(elapsed_h)
                except Exception:
                    continue

            calibrated: Dict[str, int] = {}
            for key, hours_list in bucket_hours.items():
                if len(hours_list) < 10:
                    continue
                median_h = statistics.median(hours_list)
                baseline_h = float(self._default_validity_window_hours(key))
                max_allowed_h = max(4.0, baseline_h * calibration_max_multiplier)
                # Cap to sane range and round to nearest hour.
                calibrated[key] = int(max(4, min(max_allowed_h, round(median_h))))

            if calibrated:
                self._calibrated_validity_windows = calibrated
                logger.info(
                    "📐 Validity-window calibration updated (%d buckets): %s",
                    len(calibrated),
                    {k: f"{v}h" for k, v in calibrated.items()},
                )
        except Exception as e:
            logger.warning("⚠️ Validity-window calibration failed: %s", e)
        finally:
            self._validity_calibration_last_run = now

    def _normalise_horizon_key(self, time_horizon: str) -> str:
        """Return a stable canonical key for a time_horizon string."""
        th = time_horizon.lower().strip()
        for token in ("4h-12h", "4-12h"):
            if token in th:
                return "4h-12h"
        for token in ("4h-24h", "4-24h", "4-24 hours"):
            if token in th:
                return "4-24h"
        for token in ("1-3 days", "1-3d"):
            if token in th:
                return "1-3d"
        for token in ("3-7 days", "3-7d"):
            if token in th:
                return "3-7d"
        for token in ("1-2 weeks", "1-2w"):
            if token in th:
                return "1-2w"
        for token in ("2-4 weeks", "2-4w"):
            if token in th:
                return "2-4w"
        for token in ("1-3 months", "1-3m"):
            if token in th:
                return "1-3m"
        for token in ("scalp",):
            if token in th:
                return "scalp"
        if "short" in th:
            return "short"
        if "medium" in th:
            return "medium"
        if "long" in th:
            return "long"
        if "swing" in th:
            return "swing"
        if "intraday" in th:
            return "intraday"
        return th  # fallback: use raw value as key

    def _extract_range_upper_hours(self, time_horizon: str) -> Optional[int]:
        """
        Parse explicit range/time horizons and return hours from the value.
        For ranges, returns upper bound; for single explicit times, returns that time.
        Examples:
        - "4h-12h" -> 12
        - "4-5h" -> 5
        - "1-3d" -> 72
        - "2-4 weeks" -> 672
        - "5h" -> 5
        - "2 days" -> 48
        """
        try:
            th = str(time_horizon or "").strip().lower()
            if not th:
                return None

            import math
            import re

            # Normalize dashes/spaces for robust matching.
            text = th.replace("—", "-").replace("–", "-")
            text = re.sub(r"\s+", " ", text)

            unit_specs = [
                (r"(?:h|hr|hrs|hour|hours)", 1.0),
                (r"(?:d|day|days)", 24.0),
                (r"(?:w|wk|wks|week|weeks)", 168.0),
                (r"(?:m|mo|mos|month|months)", 720.0),
            ]

            for unit_regex, mult in unit_specs:
                # Unit on first value, optional on second: "4h-12h", "4h-12"
                pattern_a = rf"(\d+(?:\.\d+)?)\s*{unit_regex}\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*{unit_regex}?"
                match_a = re.search(pattern_a, text)
                if match_a:
                    upper = max(float(match_a.group(1)), float(match_a.group(2)))
                    return max(1, int(math.ceil(upper * mult)))

                # Unit only on the end: "4-12h", "4 to 5 hours"
                pattern_b = rf"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*{unit_regex}"
                match_b = re.search(pattern_b, text)
                if match_b:
                    upper = max(float(match_b.group(1)), float(match_b.group(2)))
                    return max(1, int(math.ceil(upper * mult)))

                # Single explicit value: "5h", "5 hours", "2d", "2 days"
                pattern_single = rf"(\d+(?:\.\d+)?)\s*{unit_regex}\b"
                match_single = re.search(pattern_single, text)
                if match_single:
                    value = float(match_single.group(1))
                    return max(1, int(math.ceil(value * mult)))

            # Last resort: naked numeric range/value with no unit (assume hours).
            # This keeps LLM outputs like "4-5" or "6" usable.
            naked_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", text)
            if naked_range:
                upper = max(float(naked_range.group(1)), float(naked_range.group(2)))
                return max(1, int(math.ceil(upper)))
            naked_single = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
            if naked_single:
                return max(1, int(math.ceil(float(naked_single.group(1)))))
        except Exception:
            return None

        return None

    async def _refresh_ensemble_accuracy_weights(self) -> None:
        """
        Compute per-source (rule engine vs LLM) historical win rates and store them as
        blending weights in `self._ensemble_weights`.

        Method:
        - Pull last 90 days of resolved signals from platform_signals.
        - Extract `ensemble_rule_confidence` from ai_confidence_breakdown JSON.
        - Classify each signal as "rule-led" (rule_conf > ai_conf) or "llm-led".
        - Compute win rate per group; normalise to weights summing to 1.0.
        - Require ≥20 samples per group; otherwise keep 0.5/0.5 defaults.

        Refreshed at most once per 24 h.
        """
        now = datetime.utcnow()
        if (
            self._ensemble_weights_last_run is not None
            and (now - self._ensemble_weights_last_run).total_seconds() < 86_400
        ):
            return
        if not self.supabase:
            return
        try:
            cutoff = (now - timedelta(days=90)).isoformat()
            result = (
                self.supabase
                .from_("platform_signals")
                .select("status,confidence,ai_confidence_breakdown")
                .in_("status", ["hit_target_1", "hit_target_2", "hit_stop_loss"])
                .gte("created_at", cutoff)
                .limit(5000)
                .execute()
            )
            rows = result.data or []
            if not rows:
                self._ensemble_weights_last_run = now
                return

            rule_wins = rule_total = 0
            llm_wins = llm_total = 0

            for row in rows:
                status = str(row.get("status") or "")
                is_win = status in ("hit_target_1", "hit_target_2")
                ai_conf = float(row.get("confidence") or 0.0)

                breakdown = row.get("ai_confidence_breakdown") or {}
                if isinstance(breakdown, str):
                    try:
                        import json as _json
                        breakdown = _json.loads(breakdown)
                    except Exception:
                        breakdown = {}

                rule_conf = breakdown.get("ensemble_rule_confidence")
                if rule_conf is None:
                    # No attribution data yet; skip until enough tagged signals accrue.
                    continue
                try:
                    rule_conf = float(rule_conf)
                except (TypeError, ValueError):
                    continue

                if rule_conf >= ai_conf:
                    rule_total += 1
                    if is_win:
                        rule_wins += 1
                else:
                    llm_total += 1
                    if is_win:
                        llm_wins += 1

            MIN_SAMPLES = 20
            if rule_total >= MIN_SAMPLES and llm_total >= MIN_SAMPLES:
                wr_rule = rule_wins / rule_total
                wr_llm = llm_wins / llm_total
                total_wr = wr_rule + wr_llm
                if total_wr > 0:
                    w_rule = wr_rule / total_wr
                    w_llm = wr_llm / total_wr
                else:
                    w_rule = w_llm = 0.5
                self._ensemble_weights = {'rule': round(w_rule, 4), 'llm': round(w_llm, 4)}
                logger.info(
                    "⚖️  Ensemble accuracy weights updated — "
                    "rule=%.3f (WR=%.1f%%, n=%d)  llm=%.3f (WR=%.1f%%, n=%d)",
                    w_rule, wr_rule * 100, rule_total,
                    w_llm, wr_llm * 100, llm_total,
                )
            else:
                logger.debug(
                    "⚖️  Ensemble weights: insufficient attribution data "
                    "(rule_n=%d, llm_n=%d, need %d each) — keeping 0.5/0.5",
                    rule_total, llm_total, MIN_SAMPLES,
                )
        except Exception as e:
            logger.warning("⚠️ Ensemble accuracy weight refresh failed: %s", e)
        finally:
            self._ensemble_weights_last_run = now

    def _calculate_validity_window_hours(self, time_horizon: str) -> int:
        """Calculate validity window hours based on time_horizon.

        Checks the live-calibrated table first (populated from actual resolved-signal
        data by `_calibrate_validity_windows`); falls back to hardcoded heuristics.
        """
        if not time_horizon:
            return self._calibrated_validity_windows.get("default", 72)

        # Explicit ranges always use the maximum bound (e.g., 4-12h -> 12, 4-5h -> 5).
        range_upper_h = self._extract_range_upper_hours(time_horizon)
        if range_upper_h is not None:
            return int(range_upper_h)

        canonical = self._normalise_horizon_key(time_horizon)
        if canonical in self._calibrated_validity_windows:
            return self._calibrated_validity_windows[canonical]

        time_horizon_lower = time_horizon.lower().strip()
        
        # Handle specific time ranges
        if '4h-12h' in time_horizon_lower or '4-12h' in time_horizon_lower:
            return 12  # 12 hours for 4h-12h range
        elif '4h-24h' in time_horizon_lower or '4-24h' in time_horizon_lower or '4-24 hours' in time_horizon_lower:
            return 24  # 24 hours for 4-24h range
        elif '1-3 days' in time_horizon_lower or '1-3d' in time_horizon_lower:
            return 72  # 3 days for 1-3 days range
        elif '3-7 days' in time_horizon_lower or '3-7d' in time_horizon_lower:
            return 168  # 1 week for 3-7 days range
        elif '1-2 weeks' in time_horizon_lower or '1-2w' in time_horizon_lower:
            return 336  # 2 weeks for 1-2 weeks range
        elif '2-4 weeks' in time_horizon_lower or '2-4w' in time_horizon_lower:
            return 672  # 4 weeks for 2-4 weeks range
        elif '1-3 months' in time_horizon_lower or '1-3m' in time_horizon_lower:
            return 2160  # 3 months for 1-3 months range
        
        # Handle single values
        elif 'scalp' in time_horizon_lower:
            return 4  # 4 hours for scalping
        elif 'short' in time_horizon_lower:
            return 24  # 1 day for short-term
        elif 'medium' in time_horizon_lower:
            return 168  # 1 week for medium-term
        elif 'long' in time_horizon_lower:
            return 720  # 1 month for long-term
        elif 'swing' in time_horizon_lower:
            return 336  # 2 weeks for swing trading
        elif 'intraday' in time_horizon_lower:
            return 8  # 8 hours for intraday
        elif 'position' in time_horizon_lower:
            return 1440  # 2 months for position trading
        
        # Handle numeric patterns (e.g., "4h", "24h", "3d", "1w")
        import re
        
        # Hours pattern (e.g., "4h", "12h", "24h")
        hours_match = re.search(r'(\d+)h', time_horizon_lower)
        if hours_match:
            return int(hours_match.group(1))
        
        # Days pattern (e.g., "3d", "7d", "14d")
        days_match = re.search(r'(\d+)d', time_horizon_lower)
        if days_match:
            return int(days_match.group(1)) * 24
        
        # Weeks pattern (e.g., "1w", "2w", "4w")
        weeks_match = re.search(r'(\d+)w', time_horizon_lower)
        if weeks_match:
            return int(weeks_match.group(1)) * 168
        
        # Months pattern (e.g., "1m", "2m", "3m")
        months_match = re.search(r'(\d+)m', time_horizon_lower)
        if months_match:
            return int(months_match.group(1)) * 720
        
        # Default fallback
        return 72  # Default 3 days

    def _effective_published_validity_hours(self, time_horizon: str) -> int:
        """Validity window actually written to a published signal's expires_at.

        P0 expiry fix: the horizon-derived window (used for target-distance bounding)
        was being published verbatim, so short horizons (e.g. "4h-12h" -> 12h) gave
        price almost no time to reach target/stop. ~61% of historical signals expired
        unresolved as a result. We keep the raw window for target bounding but apply a
        floor + buffer (capped) for the published expiry so signals get a fair chance to
        resolve into a real win/loss the learning loop can use.
        """
        base = float(self._calculate_validity_window_hours(time_horizon))
        buffer_mult = max(1.0, float(getattr(settings, "SIGNAL_VALIDITY_BUFFER_MULT", 1.5) or 1.5))
        floor_h = max(1.0, float(getattr(settings, "SIGNAL_MIN_VALIDITY_HOURS", 24.0) or 24.0))
        cap_h = max(floor_h, float(getattr(settings, "SIGNAL_MAX_VALIDITY_HOURS", 336.0) or 336.0))
        effective = max(floor_h, base * buffer_mult)
        effective = min(effective, cap_h)
        return int(round(effective))

    def _default_validity_window_hours(self, canonical: str) -> int:
        """Canonical baseline validity windows used for calibration outlier limits."""
        defaults = {
            "4h-12h": 12,
            "4-24h": 24,
            "1-3d": 72,
            "3-7d": 168,
            "1-2w": 336,
            "2-4w": 672,
            "1-3m": 2160,
            "default": 72,
        }
        return int(defaults.get(str(canonical), defaults["default"]))

    async def _notify_signal_reversal(self, symbol: str, old_direction: str, new_direction: str,
                                       old_confidence: float, new_confidence: float, reason: str):
        """Notify users about signal reversals via WebSocket."""
        try:
            # Import WebSocket manager from main app
            import sys
            import os

            # Get the main module's WebSocket manager
            app_main = sys.modules.get('kata.main')
            if app_main and hasattr(app_main, 'manager'):
                websocket_manager = app_main.manager

                # Create reversal notification
                notification = {
                    "type": "signal_reversal",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "symbol": symbol,
                        "old_direction": old_direction,
                        "new_direction": new_direction,
                        "old_confidence": round(old_confidence, 3),
                        "new_confidence": round(new_confidence, 3),
                        "reason": reason,
                        "action_required": f"Consider closing {old_direction} and entering {new_direction}",
                        "urgency": "high" if abs(new_confidence - old_confidence) > 0.15 else "medium"
                    }
                }

                # Broadcast to all connected users
                await websocket_manager.broadcast(notification)
                logger.info(f"📡 Broadcasted signal reversal notification for {symbol}")

            else:
                logger.debug("WebSocket manager not available for reversal notification")

        except Exception as e:
            logger.error(f"❌ Failed to send reversal notification for {symbol}: {e}")


    async def fetch_token_logo_url(self, symbol: str) -> Optional[str]:
        """Resolve one market logo through the shared identity-aware service."""
        try:
            from kata.services.token_logo_service import get_token_logo_resolver

            resolved = await get_token_logo_resolver().resolve(symbol)
            if resolved:
                logger.debug(
                    "Resolved logo for %s from %s (%s)",
                    symbol,
                    resolved.source,
                    resolved.provider_id or resolved.identity_symbol,
                )
                return resolved.url
            logger.warning("No identity-safe logo found for %s", symbol)
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching logo for {symbol}: {e}")
            return None

    async def _fetch_fresh_symbol_price(self, symbol: str) -> Optional[float]:
        """Fetch a fresh last price for execution-time validation."""
        try:
            if not self.binance:
                return None
            ticker = await self.rate_limited_api_call(self.binance.fetch_ticker, symbol)
            if not ticker:
                return None
            for key in ("last", "close", "mark"):
                raw = ticker.get(key) if isinstance(ticker, dict) else None
                if raw is None:
                    continue
                price = float(raw)
                if price > 0:
                    return price
            return None
        except Exception as e:
            logger.debug(f"Fresh price fetch failed for {symbol}: {e}")
            return None

    def _build_entry_activation_metadata(
        self,
        direction: str,
        entry_price: float,
        current_price: float,
        entry_strategy: Optional[str],
        signal_started_at: Optional[datetime] = None,
        signal_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether a signal should be immediately active or wait for entry touch.
        """
        now = signal_started_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now_iso = now.isoformat()
        strategy = str(entry_strategy or "").strip().upper() or "UNKNOWN"
        direction_u = str(direction or "").strip().upper()
        entry = float(entry_price or 0.0)
        current = float(current_price or 0.0)
        buffer_pct = 0.0015  # 0.15% activation tolerance

        meta: Dict[str, Any] = {
            "entry_strategy": strategy,
            "entry_order_state": "active",
            "entry_revision": 0,
            "entry_activation_buffer_pct": buffer_pct,
            "entry_activation_required": False,
            "entry_activation_mode": "immediate_market",
            "entry_activated": True,
            "entry_activated_at": now_iso,
            "entry_price_at_generation": entry,
            "market_price_at_generation": current,
            "entry_order_expires_at": entry_order_deadline(
                signal_started_at=now,
                signal_expires_at=signal_expires_at,
                max_ttl_hours=float(
                    getattr(settings, "YUKI_ENTRY_ORDER_TTL_HOURS", 12.0) or 12.0
                ),
                validity_fraction=float(
                    getattr(settings, "YUKI_ENTRY_TTL_VALIDITY_FRACTION", 0.50) or 0.50
                ),
                now=now,
            ).isoformat(),
            "entry_revalidation_required_before_resubmit": False,
        }

        if entry <= 0 or current <= 0 or direction_u not in {"LONG", "SHORT"}:
            return meta

        # Explicit market entries are active immediately.
        if "MARKET" in strategy:
            return meta

        # For non-market entries, gate exits until entry has been touched.
        if direction_u == "LONG":
            if entry < current * (1 - buffer_pct):
                meta.update({
                    "entry_activation_required": True,
                    "entry_activation_mode": "limit_retest",
                    "entry_activated": False,
                    "entry_activated_at": None,
                })
            elif entry > current * (1 + buffer_pct):
                meta.update({
                    "entry_activation_required": True,
                    "entry_activation_mode": "breakout_stop",
                    "entry_activated": False,
                    "entry_activated_at": None,
                })
        else:  # SHORT
            if entry > current * (1 + buffer_pct):
                meta.update({
                    "entry_activation_required": True,
                    "entry_activation_mode": "limit_retest",
                    "entry_activated": False,
                    "entry_activated_at": None,
                })
            elif entry < current * (1 - buffer_pct):
                meta.update({
                    "entry_activation_required": True,
                    "entry_activation_mode": "breakout_stop",
                    "entry_activated": False,
                    "entry_activated_at": None,
                })

        return meta

    def targets_are_feasible(
        self,
        signal_type: str,
        current_price: float,
        target_1: float,
        target_2: float,
        entry_price: Optional[float] = None,
        entry_strategy: Optional[str] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        """
        Check that a newly published setup is still executable at live price.

        Pending retests still require valid levels relative to their intended entry,
        but a setup whose target or stop was crossed before publication is already
        stale for Yuki and should not occupy the shared active pool.
        """
        strategy = str(entry_strategy or '').strip().upper()
        entry = float(entry_price or 0.0)
        stop = float(stop_loss or 0.0)

        if signal_type == 'LONG':
            # Structural checks (always required)
            if target_2 <= target_1:
                logger.warning(f"LONG signal target_2 {target_2} should be higher than target_1 {target_1}")
                return False
            if entry > 0 and target_1 <= entry:
                logger.warning(f"LONG signal target_1 {target_1} should be higher than entry {entry}")
                return False

            if stop > 0 and current_price <= stop:
                logger.warning(
                    f"LONG signal stop_loss {stop} already crossed by current price {current_price} "
                    f"(strategy={strategy or 'UNKNOWN'})"
                )
                return False
            if current_price >= target_1:
                logger.warning(f"LONG signal target_1 {target_1} already exceeded by current price {current_price}")
                return False
            if current_price >= target_2:
                logger.warning(f"LONG signal target_2 {target_2} already exceeded by current price {current_price}")
                return False

        elif signal_type == 'SHORT':
            # Structural checks (always required)
            if target_2 >= target_1:
                logger.warning(f"SHORT signal target_2 {target_2} should be lower than target_1 {target_1}")
                return False
            if entry > 0 and target_1 >= entry:
                logger.warning(f"SHORT signal target_1 {target_1} should be lower than entry {entry}")
                return False

            if stop > 0 and current_price >= stop:
                logger.warning(
                    f"SHORT signal stop_loss {stop} already crossed by current price {current_price} "
                    f"(strategy={strategy or 'UNKNOWN'})"
                )
                return False
            if current_price <= target_1:
                logger.warning(f"SHORT signal target_1 {target_1} already exceeded by current price {current_price}")
                return False
            if current_price <= target_2:
                logger.warning(f"SHORT signal target_2 {target_2} already exceeded by current price {current_price}")
                return False

        return True

    async def _classify_market_pattern(self, technical_analysis: TechnicalAnalysis) -> str:
        """Classify market pattern based on technical analysis."""
        try:
            # Use existing regime detection method
            regime_data = self._detect_advanced_regime(technical_analysis)

            # Return pattern type from regime analysis
            pattern = regime_data.get('type', 'unknown_pattern')
            logger.debug(f"Pattern classified as: {pattern}")
            return pattern

        except Exception as e:
            logger.error(f"Error classifying pattern: {e}")
            return 'unknown_pattern'

    async def _get_pattern_performance(self, pattern_name: str) -> Optional[dict]:
        """Get historical performance data for a pattern."""
        try:
            # Validate input
            if not pattern_name or not isinstance(pattern_name, str):
                logger.debug(f"Invalid pattern_name: {pattern_name}")
                return None

            response = self.supabase.from_('pattern_performance_tracking').select(
                'success_rate, sample_size, avg_pnl, avg_time_to_target'
            ).eq('pattern_name', pattern_name).execute()

            if response.data:
                performance = response.data[0]
                logger.debug(f"Pattern {pattern_name}: {performance['success_rate']:.1%} success rate ({performance['sample_size']} trades)")
                return performance
            else:
                logger.debug(f"No performance data found for pattern: {pattern_name}")
                return None

        except Exception as e:
            logger.error(f"Error getting pattern performance for {pattern_name}: {e}")
            return None

# Main execution
async def main():
    """Test the unified signal generator."""
    generator = UnifiedSignalGenerator()

    try:
        await generator._initialize_services()
        run_id = await generator.run_full_analysis_cycle(max_signals=3)
        logger.info(f"🎉 Completed analysis run: {run_id}")

    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

# Singleton instance
_unified_signal_generator = None

def get_unified_signal_generator() -> UnifiedSignalGenerator:
    """Get singleton instance of unified signal generator."""
    global _unified_signal_generator
    if _unified_signal_generator is None:
        _unified_signal_generator = UnifiedSignalGenerator()
    return _unified_signal_generator
