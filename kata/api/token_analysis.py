"""
Token Analysis API Endpoints

Provides AI-powered token analysis endpoints including:
- Individual token analysis with AI agents
- Technical and fundamental analysis
- Risk assessment and recommendations
- Market sentiment integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict
from html import unescape
import re

from ..services.llm_analysis_service import LLMAnalysis, LLMAnalysisService, LLMProvider, MarketContext
from ..services.market_data_service import MarketDataService, create_market_data_service, ComprehensiveMarketData
from ..services.enhanced_market_data_service import EnhancedMarketDataService, create_enhanced_market_data_service
from ..services.sentiment_service import SentimentService, create_sentiment_service
from ..services.coingecko_service import CoinGeckoService
from ..services.advanced_metrics_service import AdvancedMetricsService, create_advanced_metrics_service, AdvancedMetrics
from ..services.credit_service import CreditService, get_credit_service
from ..services.market_structure_context_service import get_market_structure_context_service
from ..agents.yuki_analysis_agent import YukiAnalysisAgent
from ..agents.sakura_agent import SakuraAgent
from ..agents.ryu_agent import RyuAgent
from ..agents.base_agent import MarketData
from ..config.settings import settings
from ..config.database import get_service_client
from decimal import Decimal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/token-analysis", tags=["token-analysis"])

# Global service instances
llm_analysis_service: Optional[LLMAnalysisService] = None
market_data_service: Optional[MarketDataService] = None
enhanced_market_service: Optional[EnhancedMarketDataService] = None
sentiment_service: Optional[SentimentService] = None
coingecko_service: Optional[CoinGeckoService] = None
advanced_metrics_service: Optional[AdvancedMetricsService] = None
_PLATFORM_REGIME_MAX_AGE_HOURS = 96


def _platform_symbol_candidates(token_ticker: str) -> List[str]:
    """Return platform-signal aliases for spot tickers and Binance futures multipliers."""
    token = str(token_ticker or "").upper().replace("/", "").replace("-", "").strip()
    if not token:
        return []

    bases = [token[:-4] if token.endswith("USDT") else token]
    for base in list(bases):
        for prefix in ("1000000", "1000"):
            if base.startswith(prefix) and len(base) > len(prefix):
                bases.append(base[len(prefix):])

    candidates: List[str] = []
    for base in bases:
        for candidate in (base, f"{base}USDT", f"1000{base}USDT", f"1000000{base}USDT"):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _parse_platform_created_at(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_platform_regime(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    market_regime = row.get("market_regime") or {}
    if not isinstance(market_regime, dict):
        return None

    regime = market_regime.get("regime")
    if not regime and isinstance(market_regime.get("regime_data"), dict):
        regime = market_regime["regime_data"].get("type")
    if not regime:
        return None

    created_at = _parse_platform_created_at(row.get("created_at"))
    if created_at:
        now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.utcnow()
        if now - created_at > timedelta(hours=_PLATFORM_REGIME_MAX_AGE_HOURS):
            return None

    confidence = row.get("regime_confidence")
    if confidence is None:
        confidence = market_regime.get("confidence")

    return {
        "regime": str(regime).upper(),
        "confidence": _clamp_float(confidence, default=0.5) if confidence is not None else None,
        "token_symbol": row.get("token_symbol"),
        "created_at": row.get("created_at"),
        "btc_context": market_regime.get("btc_context"),
    }


def _get_latest_platform_market_regime(token_ticker: str) -> Optional[Dict[str, Any]]:
    """Use recent platform signal regime context as the token-analysis market-regime source."""
    fields = "signal_id,token_symbol,created_at,market_regime,regime_confidence"
    candidates = _platform_symbol_candidates(token_ticker)

    try:
        db = get_service_client()
        if candidates:
            result = (
                db.table("platform_signals")
                .select(fields)
                .in_("token_symbol", candidates)
                .order("created_at", desc=True)
                .limit(8)
                .execute()
            )
            for row in result.data or []:
                context = _extract_platform_regime(row)
                if context:
                    context["source"] = "token_platform_signal"
                    return context

        fallback = (
            db.table("platform_signals")
            .select(fields)
            .order("created_at", desc=True)
            .limit(12)
            .execute()
        )
        for row in fallback.data or []:
            context = _extract_platform_regime(row)
            if context:
                context["source"] = "latest_platform_signal"
                return context
    except Exception as exc:
        logger.debug("Platform market regime lookup unavailable for %s: %s", token_ticker, exc)

    return None


def _token_market_symbol_candidates(token_ticker: str) -> List[str]:
    """Return likely Binance futures market symbols for a user-facing token."""
    token = str(token_ticker or "").upper().replace("/", "").replace("-", "").strip()
    if token.endswith("USDT"):
        token = token[:-4]
    if not token:
        return []

    bases = [token]
    for prefix in ("1000000", "1000"):
        if token.startswith(prefix) and len(token) > len(prefix):
            bases.append(token[len(prefix):])

    candidates: List[str] = []
    for base in bases:
        for candidate in (
            f"{base}/USDT",
            f"{base}/USDT:USDT",
            f"1000{base}/USDT",
            f"1000{base}/USDT:USDT",
            f"1000000{base}/USDT",
            f"1000000{base}/USDT:USDT",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _token_fapi_symbol(token_ticker: str) -> str:
    token = str(token_ticker or "").upper().replace("/", "").replace("-", "").strip()
    return token if token.endswith("USDT") else f"{token}USDT"


async def _fetch_token_structure_candles(
    market_service: MarketDataService,
    token_ticker: str,
    *,
    interval: str = "4h",
    limit: int = 160,
) -> List[Dict[str, Any]]:
    """Fetch real Binance candles for token structure; return [] if unavailable."""
    binance = getattr(market_service, "binance", None)
    if not binance:
        return []

    candidates = _token_market_symbol_candidates(token_ticker)

    # When ccxt has already loaded the market list, it is authoritative: drop
    # unlisted variants up front instead of probing each one against the API
    # (6 variants x retries per unlisted token, all doomed).
    markets = getattr(getattr(binance, "exchange", None), "markets", None)
    if markets:
        listed = [symbol for symbol in candidates if symbol in markets]
        if not listed:
            logger.debug("%s has no Binance market in any variant; skipping candle fetch", token_ticker)
            return []
        candidates = listed

    for market_symbol in candidates:
        try:
            candles = await binance.get_historical_klines(market_symbol, interval=interval, limit=limit)
            valid = [
                candle for candle in candles or []
                if candle.get("open") and candle.get("high") and candle.get("low") and candle.get("close")
            ]
            if len(valid) >= 30:
                logger.info(
                    "Using %d real %s candles for %s market-structure context via %s",
                    len(valid),
                    interval,
                    token_ticker,
                    market_symbol,
                )
                return valid
        except Exception as exc:
            logger.debug("Historical candles unavailable for %s via %s: %s", token_ticker, market_symbol, exc)
    return []


async def _fetch_token_taker_flow(
    market_service: MarketDataService,
    token_ticker: str,
    *,
    period: str = "1h",
    limit: int = 24,
) -> Optional[Dict[str, Any]]:
    """Fetch real Binance Futures taker buy/sell flow; no fallback if unavailable."""
    binance = getattr(market_service, "binance", None)
    exchange = getattr(binance, "exchange", None)
    if not binance or not exchange:
        return None

    method = None
    for method_name in (
        "fapiDataGetTakerlongshortratio",
        "fapiDataGetTakerLongShortRatio",
        "fapiDataGetTakerlongshortRatio",
    ):
        method = getattr(exchange, method_name, None)
        if method:
            break
    if not method:
        return None

    try:
        payload = await binance._make_rate_limited_request(
            method,
            1,
            {"symbol": _token_fapi_symbol(token_ticker), "period": period, "limit": limit},
        )
    except Exception as exc:
        logger.debug("Taker flow fetch failed for %s: %s", token_ticker, exc)
        return None

    if not isinstance(payload, list) or not payload:
        return None

    buy_volume = 0.0
    sell_volume = 0.0
    cvd = 0.0
    latest_ratio = None
    periods = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            buy = float(row.get("buyVol") or row.get("takerBuyVol") or row.get("buyVolume") or 0.0)
            sell = float(row.get("sellVol") or row.get("takerSellVol") or row.get("sellVolume") or 0.0)
            ratio_value = row.get("buySellRatio")
            if ratio_value is not None:
                latest_ratio = float(ratio_value)
        except (TypeError, ValueError):
            continue
        buy_volume += buy
        sell_volume += sell
        cvd += buy - sell
        periods += 1

    total = buy_volume + sell_volume
    if periods <= 0 or total <= 0:
        return None

    return {
        "data_available": True,
        "period": period,
        "periods": periods,
        "taker_buy_volume": buy_volume,
        "taker_sell_volume": sell_volume,
        "taker_delta": buy_volume - sell_volume,
        "taker_delta_pct": (buy_volume - sell_volume) / total,
        "taker_ratio": latest_ratio if latest_ratio is not None else (buy_volume / sell_volume if sell_volume > 0 else None),
        "cvd": cvd,
    }


def _display_market_regime(preferred_regime: Any, llm_regime: Any = None) -> str:
    unknowns = {"", "UNCERTAIN", "UNKNOWN", "N/A", "NONE"}
    preferred = str(preferred_regime or "").upper()
    if preferred not in unknowns:
        return preferred
    llm_value = str(llm_regime or "").upper()
    if llm_value not in unknowns:
        return llm_value
    return "UNCERTAIN"


class TokenAnalysisRequest(BaseModel):
    """Request model for token analysis."""
    token_ticker: str
    agent_type: Optional[str] = "yuki"  # yuki, sakura, ryu
    include_sentiment: bool = True
    include_technical: bool = True
    include_fundamental: bool = True
    request_id: Optional[str] = None
    billing_user_id: Optional[str] = None
    coingecko_id: Optional[str] = None
    contract_address: Optional[str] = None
    chain_id: Optional[int] = None
    timeframe: Optional[str] = "4h"  # The timeframe for market structure (e.g., 15m, 1h, 4h, 1d)


class EntryStrategyResponse(BaseModel):
    """Entry strategy details for response."""
    optimal_entry: float
    entry_range_low: float
    entry_range_high: float
    market_order_ok: bool

class PriceTargetsResponse(BaseModel):
    """Price targets for response."""
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None

class RiskManagementResponse(BaseModel):
    """Risk management details for response."""
    stop_loss: Optional[float] = None
    position_size: str
    max_leverage: str

class DetailedInsights(BaseModel):
    """Detailed AI insights for modal display."""
    action_summary: str
    reasoning: str
    key_factors: List[str]
    time_horizon: str
    entry_strategy: Optional[EntryStrategyResponse] = None
    price_targets: Optional[PriceTargetsResponse] = None
    risk_management: Optional[RiskManagementResponse] = None
    execution_notes: str
    confidence: float
    market_regime: str
    trade_intent: str = "HOLD"

class TokenAnalysisResponse(BaseModel):
    """Enhanced response model for token analysis."""
    token_ticker: str
    token_name: str
    token_image: Optional[Dict[str, str]] = None  # Token logo images (thumb, small, large)
    analysis_timestamp: datetime
    agent_type: str
    overall_score: float
    risk_level: str
    recommendation: str
    trade_intent: str = "HOLD"
    request_id: Optional[str] = None
    credits_remaining: Optional[int] = None
    token_identity: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    sentiment_analysis: Optional[Dict[str, Any]]
    market_data: Dict[str, Any]
    agent_insights: str  # Legacy field for backward compatibility
    detailed_insights: Optional[DetailedInsights] = None  # New detailed insights for modal


def _clamp_float(value: Any, default: float = 0.0, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Convert to float and clamp into a stable range."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = default
    return max(min_value, min(max_value, numeric_value))


def _normalize_recommendation_label(recommendation: Any) -> str:
    """Normalize recommendation labels to BUY, SELL, or HOLD."""
    normalized = str(recommendation or "HOLD").upper().replace("-", "_").replace(" ", "_")
    if normalized in {"BUY", "STRONG_BUY", "LONG", "STRONG_LONG"}:
        return "BUY"
    if normalized in {"SELL", "STRONG_SELL", "SHORT", "STRONG_SHORT"}:
        return "SELL"
    if normalized in {"HOLD", "WAIT", "NO_TRADE", "WATCHLIST", "DON'T_BUY", "DONT_BUY"}:
        return "HOLD"
    return "HOLD"


def _recommendation_direction(recommendation: str) -> int:
    if recommendation == "BUY":
        return 1
    if recommendation == "SELL":
        return -1
    return 0


_PLACEHOLDER_TEXT_MARKERS = ("ai-determined", "llm unavailable", "determined based")


def _is_placeholder_text(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower()
    return any(marker in normalized for marker in _PLACEHOLDER_TEXT_MARKERS)


def _has_spot_short_language(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.lower()
    blocked_terms = (" short ", "shorting", "shorts", "short-side", "short entry", "short setup", "futures", "perp", "perpetual", "leverage", "leveraged")
    if normalized.startswith("short "):
        return True
    return any(term in normalized for term in blocked_terms)


def _leading_text_recommendation(value: Any) -> Optional[str]:
    """Read an explicit recommendation at the start of an LLM headline."""
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().upper().replace("’", "'")
    for prefix, recommendation in (
        ("STRONG SHORT", "SELL"),
        ("STRONG SELL", "SELL"),
        ("STRONG LONG", "BUY"),
        ("STRONG BUY", "BUY"),
        ("DON'T BUY", "HOLD"),
        ("DONT BUY", "HOLD"),
        ("NO TRADE", "HOLD"),
        ("WATCHLIST", "HOLD"),
        ("SHORT", "SELL"),
        ("SELL", "SELL"),
        ("LONG", "BUY"),
        ("BUY", "BUY"),
        ("HOLD", "HOLD"),
        ("WAIT", "HOLD"),
    ):
        if normalized.startswith(prefix):
            return recommendation
    return None


def _has_decisive_wait_language(value: Any) -> bool:
    """Detect conclusions that clearly say no entry is ready yet."""
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = " ".join(value.lower().replace("‑", "-").replace("–", "-").split())
    decisive_phrases = (
        "best to wait",
        "better to wait",
        "wait for a dip",
        "wait for the dip",
        "wait for confirmation",
        "wait for a confirmed",
        "wait for cleaner",
        "before committing",
        "before entering",
        "risk/reward for new longs is poor",
        "risk/reward for new shorts is poor",
        "risk-reward for new longs is poor",
        "risk-reward for new shorts is poor",
        "no clean trade",
        "no clear trade",
        "stand aside",
        "avoid fresh exposure",
        "avoid committing",
        "do not enter",
        "don't enter",
    )
    return any(phrase in normalized for phrase in decisive_phrases)


def _effective_llm_recommendation(llm_analysis: Any) -> tuple[str, str]:
    """Resolve inconsistent LLM fields conservatively before combining votes."""
    raw_recommendation = _normalize_recommendation_label(
        getattr(llm_analysis, "recommendation", "HOLD")
    )
    action_summary = getattr(llm_analysis, "action_summary", "") or ""
    reasoning = getattr(llm_analysis, "reasoning", "") or ""
    headline_recommendation = _leading_text_recommendation(action_summary)

    if _has_decisive_wait_language(action_summary) or _has_decisive_wait_language(reasoning):
        return "HOLD", "analysis says the entry is not ready"
    if headline_recommendation and headline_recommendation != raw_recommendation:
        return headline_recommendation, "explicit headline overrides inconsistent label"
    return raw_recommendation, "structured LLM recommendation accepted"


def _risk_guidance_for_display(
    *,
    recommendation: str,
    confidence: float,
    risk_level: str,
    volatility: float,
    trade_intent: Optional[str] = None,
) -> tuple[str, str]:
    recommendation = _normalize_recommendation_label(recommendation)
    confidence = _clamp_float(confidence, default=0.5)
    risk_level = str(risk_level or "MEDIUM").upper()
    volatility = max(0.0, float(volatility or 0.0))

    trade_intent = str(trade_intent or "").upper()
    if recommendation == "HOLD" or trade_intent == "HOLD":
        return "0% (wait for confirmation)", "1x only"
    if trade_intent == "SPOT_EXIT":
        return "Reduce/exit spot exposure", "Spot only"
    if trade_intent == "SPOT_BUY":
        return "1-3% spot allocation", "Spot only"
    if risk_level == "HIGH" or volatility >= 0.08 or confidence < 0.58:
        return "1% portfolio risk", "2x max"
    if risk_level == "LOW" and volatility <= 0.04 and confidence >= 0.75:
        return "2-3% portfolio risk", "5x max"
    return "1-2% portfolio risk", "3x max"


def _trade_intent_for_decision(recommendation: str, confidence: float, risk_level: str) -> str:
    """Separate leveraged entries from spot-only actions and no-trade outcomes."""
    recommendation = _normalize_recommendation_label(recommendation)
    confidence = _clamp_float(confidence, default=0.5)
    risk_level = str(risk_level or "MEDIUM").upper()
    leveraged_setup = confidence >= 0.58 and risk_level != "HIGH"
    if recommendation == "BUY":
        return "LONG_ENTRY" if leveraged_setup else "SPOT_BUY"
    if recommendation == "SELL":
        return "SHORT_ENTRY" if leveraged_setup else "SPOT_EXIT"
    return "HOLD"


def _is_directionally_valid_trade_plan(
    direction: str,
    entry: Optional[float],
    target: Optional[float],
    stop: Optional[float],
) -> bool:
    """Validate the price ordering required for a scoreable long or short plan."""
    entry = _first_positive_float(entry)
    target = _first_positive_float(target)
    stop = _first_positive_float(stop)
    if entry is None or target is None or stop is None:
        return False
    if direction == "long":
        return target > entry > stop
    if direction == "short":
        return target < entry < stop
    return False


def _is_generic_key_factor(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower()
    generic_prefixes = (
        "rule engine:",
        "llm vote:",
        "decision resolution:",
        "final recommendation:",
    )
    generic_values = {
        "technical analysis",
        "market conditions",
        "risk management",
        "volume",
        "rsi",
        "macd",
        "momentum",
    }
    return normalized in generic_values or any(normalized.startswith(prefix) for prefix in generic_prefixes)


def _estimate_rule_confidence(
    *,
    recommendation: str,
    overall_score: float,
    technical_score: float,
    fundamental_score: float,
    price_change_24h: float,
    market_regime: str
) -> float:
    """Estimate confidence for the non-LLM score engine vote."""
    recommendation = _normalize_recommendation_label(recommendation)
    overall_score = _clamp_float(overall_score, default=0.5)
    technical_score = _clamp_float(technical_score, default=0.5)
    fundamental_score = _clamp_float(fundamental_score, default=0.5)
    market_regime = (market_regime or "UNCERTAIN").upper()
    bullish_regime = "BULL" in market_regime or "UPTREND" in market_regime
    bearish_regime = "BEAR" in market_regime or "DOWNTREND" in market_regime

    if recommendation == "HOLD":
        return max(0.35, min(0.65, 0.62 - abs(overall_score - 0.5)))

    confidence = 0.42 + min(0.30, abs(overall_score - 0.5))
    if recommendation == "BUY":
        if overall_score >= 0.65:
            confidence += 0.12
        if technical_score >= 0.60:
            confidence += 0.06
        if fundamental_score >= 0.55:
            confidence += 0.04
        if price_change_24h > 2:
            confidence += 0.05
        if bullish_regime:
            confidence += 0.05
    elif recommendation == "SELL":
        if overall_score <= 0.35:
            confidence += 0.12
        if technical_score <= 0.45:
            confidence += 0.06
        if fundamental_score <= 0.45:
            confidence += 0.04
        if price_change_24h < -2:
            confidence += 0.05
        if bearish_regime:
            confidence += 0.05

    return max(0.35, min(0.90, confidence))


def _extreme_move_threshold_pct(realized_volatility: Optional[float]) -> float:
    """Scale the 'extreme move' 24h % to the asset's own volatility.

    A flat ±15% band is enormous for BTC (never fires) and trivial for a memecoin
    (always fires). Anchor on ~3x daily realized volatility, clamped to a sane band,
    and fall back to 15% when realized volatility is unavailable.
    """
    if realized_volatility and realized_volatility > 0:
        return max(8.0, min(40.0, float(realized_volatility) * 100.0 * 3.0))
    return 15.0


def _generate_rule_recommendation(
    *,
    overall_score: float,
    technical_score: float,
    price_change_24h: float,
    market_regime: str,
    is_uptrend: bool,
    is_downtrend: bool,
    is_overbought: bool,
    is_oversold: bool,
    realized_volatility: Optional[float] = None,
) -> str:
    """Generate a symmetric BUY/SELL/HOLD vote without chasing exhausted moves."""
    overall_score = _clamp_float(overall_score, default=0.5)
    technical_score = _clamp_float(technical_score, default=0.5)
    normalized_regime = str(market_regime or "UNCERTAIN").upper()
    bullish_regime = "BULL" in normalized_regime or "UPTREND" in normalized_regime
    bearish_regime = "BEAR" in normalized_regime or "DOWNTREND" in normalized_regime
    extreme_move_pct = _extreme_move_threshold_pct(realized_volatility)
    massive_pump = price_change_24h > extreme_move_pct
    massive_dump = price_change_24h < -extreme_move_pct

    # Do not chase an already-extended move. A pump can still become a SHORT when
    # overbought price action and weak scores align; a dump can still become a BUY
    # when oversold price action and stronger scores align. The logic is mirrored.
    if massive_pump:
        if is_overbought and technical_score <= 0.45 and overall_score <= 0.50:
            return "SELL"
        return "HOLD"
    if massive_dump:
        if is_oversold and technical_score >= 0.55 and overall_score >= 0.50:
            return "BUY"
        return "HOLD"

    if overall_score > 0.70:
        if is_oversold or is_uptrend or price_change_24h > 2:
            return "BUY"
        if abs(price_change_24h) < 3 and technical_score > 0.60:
            return "BUY"
        return "HOLD"

    if overall_score < 0.30:
        if is_overbought or is_downtrend or price_change_24h < -2:
            return "SELL"
        return "HOLD"

    # Moderate scores require directional structure and regime alignment. The BUY
    # and SELL paths deliberately use mirrored thresholds.
    if is_overbought and technical_score <= 0.45:
        return "SELL"
    if is_oversold and technical_score >= 0.55:
        return "BUY"
    if overall_score > 0.55 and bullish_regime and is_uptrend:
        return "BUY"
    if overall_score < 0.45 and bearish_regime and is_downtrend:
        return "SELL"
    return "HOLD"


def _build_rule_engine_advisory(
    *,
    rule_recommendation: str,
    rule_confidence: float,
    overall_score: float,
    technical_score: float,
    fundamental_score: float,
    price_change_24h: float,
    market_regime: str,
    realized_volatility: Optional[float] = None,
) -> str:
    """Render the deterministic scoring engine's read as advisory prompt context.

    Ryu (the LLM) is the analyst and makes the final call; this block hands it the
    rule engine's independent vote and the extreme-move context so it can agree,
    override, or refine. It is guidance, never a hard constraint.
    """
    rule_recommendation = _normalize_recommendation_label(rule_recommendation)
    rule_confidence = _clamp_float(rule_confidence, default=0.5)
    extreme_move_pct = _extreme_move_threshold_pct(realized_volatility)
    lines = [
        "DETERMINISTIC SCORING ENGINE (advisory — you are the analyst and make the final call):",
        f"- Rule-model vote: {rule_recommendation} (rule confidence {rule_confidence:.0%})",
        f"- Composite score: {_clamp_float(overall_score, default=0.5):.2f}/1.0 | "
        f"technical {_clamp_float(technical_score, default=0.5):.2f} | fundamental {_clamp_float(fundamental_score, default=0.5):.2f}",
        f"- Market regime: {str(market_regime or 'UNCERTAIN')}",
    ]
    if price_change_24h > extreme_move_pct:
        lines.append(
            f"- CAUTION: price is +{price_change_24h:.1f}% over 24h — an extended pump "
            f"(beyond the {extreme_move_pct:.0f}% extreme band). Chasing a fresh long here is high risk; "
            "weigh a pullback/retest or a contrarian short, but decide on the evidence."
        )
    elif price_change_24h < -extreme_move_pct:
        lines.append(
            f"- CAUTION: price is {price_change_24h:.1f}% over 24h — an extended dump "
            f"(beyond the {extreme_move_pct:.0f}% extreme band). Chasing a fresh short here is high risk; "
            "weigh an oversold reversal, but decide on the evidence."
        )
    lines.append(
        "Treat this as a second opinion: if you disagree with the rule vote, say why in your reasoning."
    )
    return "\n".join(lines)


def _normalize_leverage(value: Any) -> str:
    """Normalize an LLM-provided leverage string for display — no cap.

    The analyst sets leverage freely from its own conviction; we only tidy the
    format. 'spot' -> 'Spot only'; a bare 'Nx' -> 'Nx max'; anything else is left
    as the model wrote it.
    """
    import re

    text = str(value or "").strip()
    if not text:
        return "Spot only"
    lowered = text.lower()
    if "spot" in lowered:
        return "Spot only"
    match = re.search(r"(\d+(?:\.\d+)?)\s*x", lowered)
    if match:
        lev = float(match.group(1))
        lev_str = str(int(lev)) if lev.is_integer() else f"{lev:g}"
        return text if "max" in lowered else f"{lev_str}x max"
    return text  # descriptive guidance (e.g. "conservative") — leave as provided


def _finalize_llm_decision(
    *,
    llm_analysis: Any,
    rule_recommendation: str,
    rule_confidence: float,
) -> tuple[str, Dict[str, Any]]:
    """LLM-authoritative decision: the analyst's call stands.

    The rule engine's vote is fed to the LLM as advisory prompt context upstream,
    so here it is only recorded for transparency — it never overrides the LLM. The
    LLM recommendation is still made internally self-consistent first (an explicit
    'wait' conclusion, or a headline that contradicts the label, resolves to the
    honest call), then accepted as final. Displayed confidence is the LLM's own.
    """
    rule_recommendation = _normalize_recommendation_label(rule_recommendation)
    raw_llm_recommendation = _normalize_recommendation_label(
        getattr(llm_analysis, "recommendation", "HOLD")
    )
    final_recommendation, llm_consistency_resolution = _effective_llm_recommendation(llm_analysis)
    llm_confidence = _clamp_float(getattr(llm_analysis, "confidence", 0.5), default=0.5)
    rule_confidence = _clamp_float(rule_confidence, default=0.5)
    rule_agrees = _recommendation_direction(final_recommendation) == _recommendation_direction(rule_recommendation)

    decision_context = {
        "rule_recommendation": rule_recommendation,
        "rule_confidence": round(rule_confidence, 4),
        "llm_recommendation_raw": raw_llm_recommendation,
        "llm_recommendation": final_recommendation,
        "llm_confidence": round(llm_confidence, 4),
        "llm_consistency_resolution": llm_consistency_resolution,
        "rule_agrees_with_llm": rule_agrees,
        "final_confidence": round(llm_confidence, 4),
        "resolution": (
            "LLM analyst decision (rule engine agreed)"
            if rule_agrees
            else "LLM analyst decision (rule engine advisory, differed)"
        ),
    }
    return final_recommendation, decision_context


def _normalize_detailed_insights_for_recommendation(
    *,
    llm_analysis: Any,
    recommendation: str,
    token_ticker: str,
    current_price: float,
    technical_score: float,
    fundamental_score: float,
    overall_score: float,
    price_change_24h: float,
    market_regime: str,
    decision_context: Optional[Dict[str, Any]] = None,
    leverage_allowed: bool = True,
    resolved_risk_level: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> Optional[DetailedInsights]:
    """Make modal details agree with the final token-analysis recommendation."""
    if not llm_analysis:
        return None

    final_recommendation = _normalize_recommendation_label(recommendation)
    effective_llm_recommendation, _ = _effective_llm_recommendation(llm_analysis)
    llm_recommendation = _normalize_recommendation_label(
        (decision_context or {}).get("llm_recommendation")
        or effective_llm_recommendation
    )
    volatility = max(0.01, min(0.12, abs(price_change_24h) / 100 or 0.02))
    range_width = max(0.0025, min(0.02, volatility / 2))
    current_price = float(current_price or 0)
    confidence = _clamp_float(
        (decision_context or {}).get("final_confidence"),
        default=_clamp_float(getattr(llm_analysis, "confidence", 0.5), default=0.5),
    )
    risk_assessment = str(
        resolved_risk_level or getattr(llm_analysis, "risk_assessment", None) or "MEDIUM"
    ).upper()
    trade_intent = _trade_intent_for_decision(final_recommendation, confidence, risk_assessment)
    # Ryu is a spot-only agent. Its analyzer must emit the same intent vocabulary
    # its executor accepts; otherwise a confident BUY becomes LONG_ENTRY and is
    # silently ineligible for execution.
    if str(agent_type or "").strip().lower() == "ryu":
        if final_recommendation == "BUY":
            trade_intent = "SPOT_BUY"
        elif final_recommendation == "SELL":
            trade_intent = "SPOT_EXIT"
        else:
            trade_intent = "HOLD"
    elif not leverage_allowed:
        if trade_intent == "LONG_ENTRY":
            trade_intent = "SPOT_BUY"
        elif trade_intent == "SHORT_ENTRY":
            trade_intent = "SPOT_EXIT"
    strong_directional_setup = trade_intent in {"LONG_ENTRY", "SHORT_ENTRY"}
    entry_deviation_limit = max(0.015, min(0.18, volatility * 2.0))
    target_deviation_limit = max(0.06, min(0.45, volatility * 4.0))
    stop_deviation_limit = max(0.03, min(0.35, volatility * 3.0))

    def positive_price(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0 or numeric != numeric or abs(numeric) == float("inf"):
            return None
        return numeric

    def near_market(value: float, deviation_limit: float) -> bool:
        if current_price <= 0:
            return False
        return abs((value / current_price) - 1.0) <= deviation_limit

    def direction_summary() -> str:
        """Human-readable one-liner a trader would say to another trader."""
        if final_recommendation in {"SELL", "STRONG_SELL"}:
            if strong_directional_setup:
                return (
                    f"{final_recommendation} - {token_ticker} is showing a short setup; "
                    f"Ryu's scoring flags downside conviction with a defined trade plan — targets sit below entry."
                )
            return (
                f"{final_recommendation} - Ryu is not comfortable buying {token_ticker} here; "
                f"risk-adjusted conditions are weak. Better to reduce exposure and wait for the setup to clear."
            )
        if final_recommendation in {"BUY", "STRONG_BUY"}:
            if strong_directional_setup:
                return (
                    f"{final_recommendation} - {token_ticker} has a clean long setup; "
                    f"Ryu's scoring shows upside conviction with defined risk — respect the entry range."
                )
            return (
                f"{final_recommendation} - Ryu's final scoring supports a selective long on {token_ticker}; "
                f"conditions are constructive but not high-conviction — keep size small and the stop tight."
            )
        return (
            f"HOLD - {token_ticker} does not have a clean trade edge right now; "
            f"Ryu is waiting for stronger confirmation before committing capital."
        )

    def direction_reasoning() -> str:
        if final_recommendation in {"SELL", "STRONG_SELL"}:
            if strong_directional_setup:
                return (
                    f"{token_ticker} has a bearish trade bias, not just a weak spot view. "
                    f"The token is moving {price_change_24h:+.1f}% over 24h in a {market_regime} regime, "
                    f"with technical strength at {technical_score:.1%} versus fundamentals at {fundamental_score:.1%}. "
                    "Ryu treats this as a short setup only because conviction clears the trade gate; targets should sit below entry and invalidation belongs above resistance."
                )
            return (
                f"{token_ticker} is not clean enough for a fresh buy. "
                f"Price is moving {price_change_24h:+.1f}% over 24h in a {market_regime} regime, "
                f"while technical strength is {technical_score:.1%} and fundamentals are {fundamental_score:.1%}. "
                "Because conviction does not clear the short-trade gate, Ryu treats SELL as reduce exposure or avoid spot entry rather than forcing a short."
            )
        if final_recommendation in {"BUY", "STRONG_BUY"}:
            if strong_directional_setup:
                return (
                    f"{token_ticker} has enough upside conviction for a long setup, but it still needs defined invalidation. "
                    f"The 24h move is {price_change_24h:+.1f}% in a {market_regime} regime, "
                    f"with technical strength at {technical_score:.1%} and fundamentals at {fundamental_score:.1%}. "
                    "Ryu wants the entry range respected first, then upside targets can be managed from the trade plan."
                )
            return (
                f"{token_ticker} has a constructive buy bias, but it is not a high-conviction trade yet. "
                f"Price is moving {price_change_24h:+.1f}% over 24h in a {market_regime} regime, "
                f"with technical strength at {technical_score:.1%} and fundamentals at {fundamental_score:.1%}. "
                "Ryu would wait for confirmation inside the entry range and keep the stop tight because volatility can invalidate quickly."
            )
        return (
            f"{token_ticker} does not have a clean trade edge yet. "
            f"The token is moving {price_change_24h:+.1f}% over 24h in a {market_regime} regime, "
            f"with technical strength at {technical_score:.1%} and fundamentals at {fundamental_score:.1%}. "
            "Ryu is waiting because the setup does not justify fresh exposure at this point."
        )

    should_rewrite_text = llm_recommendation != final_recommendation
    action_summary = getattr(llm_analysis, "action_summary", "") or ""
    reasoning = getattr(llm_analysis, "reasoning", "") or ""

    # Always rewrite action_summary if it doesn't match the final recommendation
    should_rewrite_summary = (
        should_rewrite_text
        or not action_summary.upper().startswith(final_recommendation)
        or _is_placeholder_text(action_summary)
        or (_has_spot_short_language(action_summary) and not strong_directional_setup)
    )
    # A final rule/LLM direction change must rewrite both the headline and the
    # explanation. Keeping rich but directionally opposite reasoning is worse than a
    # concise explanation that agrees with the actual decision.
    should_rewrite_reasoning = (
        should_rewrite_text
        or _is_placeholder_text(reasoning)
        or (_has_spot_short_language(reasoning) and not strong_directional_setup)
    )

    if should_rewrite_summary:
        action_summary = direction_summary()
    if should_rewrite_reasoning:
        reasoning = direction_reasoning()

    llm_key_factors = [
        str(factor).strip()
        for factor in (getattr(llm_analysis, "key_factors", []) or [])
        if not _is_generic_key_factor(factor)
        and not (_has_spot_short_language(factor) and not strong_directional_setup)
    ]

    # Direction-aware fallback entry. A symmetric range around current price gave bad
    # entries (a BUY could "optimally" enter ABOVE market = chasing). Skew the range to the
    # favorable side: BUY enters at/below market (pullback), SELL at/above.
    if final_recommendation in {"BUY", "STRONG_BUY"}:
        fallback_entry_strategy = EntryStrategyResponse(
            optimal_entry=current_price * (1 - range_width * 0.5),
            entry_range_low=current_price * (1 - range_width * 1.5),
            entry_range_high=current_price,  # never chase above market on a buy
            market_order_ok=volatility < 0.08,
        )
    elif final_recommendation in {"SELL", "STRONG_SELL"}:
        fallback_entry_strategy = EntryStrategyResponse(
            optimal_entry=current_price * (1 + range_width * 0.5),
            entry_range_low=current_price,  # never sell below market
            entry_range_high=current_price * (1 + range_width * 1.5),
            market_order_ok=volatility < 0.08,
        )
    else:
        fallback_entry_strategy = EntryStrategyResponse(
            optimal_entry=current_price,
            entry_range_low=current_price * (1 - range_width),
            entry_range_high=current_price * (1 + range_width),
            market_order_ok=volatility < 0.08,
        )
    llm_entry_source = getattr(llm_analysis, "entry_strategy", None)
    llm_optimal_entry = positive_price(getattr(llm_entry_source, "optimal_entry", None))
    llm_entry_low = positive_price(getattr(llm_entry_source, "entry_range_low", None))
    llm_entry_high = positive_price(getattr(llm_entry_source, "entry_range_high", None))
    if (
        llm_optimal_entry
        and llm_entry_low
        and llm_entry_high
        and near_market(llm_optimal_entry, entry_deviation_limit)
        and near_market(llm_entry_low, entry_deviation_limit)
        and near_market(llm_entry_high, entry_deviation_limit)
    ):
        entry_low, entry_high = sorted((llm_entry_low, llm_entry_high))
        optimal_entry = min(max(llm_optimal_entry, entry_low), entry_high)
        entry_strategy = EntryStrategyResponse(
            optimal_entry=optimal_entry,
            entry_range_low=entry_low,
            entry_range_high=entry_high,
            market_order_ok=bool(getattr(llm_entry_source, "market_order_ok", fallback_entry_strategy.market_order_ok))
        )
    else:
        entry_strategy = fallback_entry_strategy

    price_targets = None
    support_or_stop = None
    if final_recommendation in {"BUY", "STRONG_BUY"}:
        target_step = max(0.03, min(0.08, volatility * 2))
        price_targets = PriceTargetsResponse(
            target_1=current_price * (1 + target_step),
            target_2=current_price * (1 + target_step * 2),
            target_3=current_price * (1 + target_step * 3)
        )
        support_or_stop = current_price * (1 - max(0.02, min(0.08, volatility * 1.5)))
    elif final_recommendation in {"SELL", "STRONG_SELL"}:
        if strong_directional_setup:
            target_step = max(0.03, min(0.08, volatility * 2))
            price_targets = PriceTargetsResponse(
                target_1=current_price * (1 - target_step),
                target_2=current_price * (1 - target_step * 2),
                target_3=current_price * (1 - target_step * 3)
            )
            support_or_stop = current_price * (1 + max(0.02, min(0.08, volatility * 1.5)))
        else:
            price_targets = None
            # SPOT_EXIT is not a short. Do not put a below-entry support reference in
            # the stop_loss field because the outcome resolver would treat it as an
            # immediately triggered short stop.
            support_or_stop = None
    else:
        support_or_stop = None

    llm_targets_source = getattr(llm_analysis, "price_targets", None)
    llm_targets = [
        positive_price(getattr(llm_targets_source, "target_1", None)),
        positive_price(getattr(llm_targets_source, "target_2", None)),
        positive_price(getattr(llm_targets_source, "target_3", None)),
    ]
    llm_targets = [
        target
        for target in llm_targets
        if target is not None and near_market(target, target_deviation_limit)
    ]
    if final_recommendation in {"BUY", "STRONG_BUY"}:
        aligned_targets = sorted(
            target for target in llm_targets if target > max(current_price, entry_strategy.optimal_entry)
        )
        if aligned_targets:
            price_targets = PriceTargetsResponse(
                target_1=aligned_targets[0] if len(aligned_targets) > 0 else None,
                target_2=aligned_targets[1] if len(aligned_targets) > 1 else None,
                target_3=aligned_targets[2] if len(aligned_targets) > 2 else None,
            )
    elif final_recommendation in {"SELL", "STRONG_SELL"} and strong_directional_setup:
        aligned_targets = sorted(
            (target for target in llm_targets if target < min(current_price, entry_strategy.optimal_entry)),
            reverse=True
        )
        if aligned_targets:
            price_targets = PriceTargetsResponse(
                target_1=aligned_targets[0] if len(aligned_targets) > 0 else None,
                target_2=aligned_targets[1] if len(aligned_targets) > 1 else None,
                target_3=aligned_targets[2] if len(aligned_targets) > 2 else None,
            )

    risk_source = getattr(llm_analysis, "risk_management", None)
    llm_stop_loss = positive_price(getattr(risk_source, "stop_loss", None))
    if llm_stop_loss and near_market(llm_stop_loss, stop_deviation_limit):
        if final_recommendation in {"BUY", "STRONG_BUY"} and llm_stop_loss < min(current_price, entry_strategy.optimal_entry):
            support_or_stop = llm_stop_loss
        elif (
            final_recommendation in {"SELL", "STRONG_SELL"}
            and strong_directional_setup
            and llm_stop_loss > max(current_price, entry_strategy.optimal_entry)
        ):
            support_or_stop = llm_stop_loss
        # SPOT_EXIT and HOLD deliberately carry no directional stop-loss.

    # Sizing is LLM-driven: the analyst sets position size and leverage. Deterministic
    # guidance is only a fallback (placeholder/empty LLM value) or an integrity override
    # (HOLD is always flat; spot intents never carry leverage; leverage is capped at 5x).
    default_position_size, default_max_leverage = _risk_guidance_for_display(
        recommendation=final_recommendation,
        confidence=confidence,
        risk_level=risk_assessment,
        volatility=volatility,
        trade_intent=trade_intent,
    )
    raw_position_size = getattr(risk_source, "position_size", None)
    raw_max_leverage = getattr(risk_source, "max_leverage", None)

    if trade_intent == "HOLD":
        position_size = default_position_size
        max_leverage = default_max_leverage
    else:
        position_size = (
            raw_position_size
            if not _is_placeholder_text(raw_position_size)
            else default_position_size
        )
        if trade_intent in {"LONG_ENTRY", "SHORT_ENTRY"}:
            max_leverage = (
                _normalize_leverage(raw_max_leverage)
                if not _is_placeholder_text(raw_max_leverage)
                else default_max_leverage
            )
        else:  # SPOT_BUY / SPOT_EXIT never carry leverage
            max_leverage = "Spot only"

    risk_management = RiskManagementResponse(
        stop_loss=support_or_stop,
        position_size=position_size,
        max_leverage=max_leverage,
    )

    # When the reasoning was rewritten to match a flipped final direction, the template
    # is generic (scores + regime, no levels). Append the concrete entry/target/stop the
    # trade plan actually resolved to, so the note stays specific and actionable instead
    # of reading like canned prose.
    if should_rewrite_reasoning and strong_directional_setup:
        level_bits = []
        if entry_strategy and entry_strategy.optimal_entry:
            level_bits.append(f"entry around {entry_strategy.optimal_entry:.6g}")
        if price_targets and price_targets.target_1:
            level_bits.append(f"first target near {price_targets.target_1:.6g}")
        if support_or_stop:
            level_bits.append(f"invalidation near {support_or_stop:.6g}")
        if level_bits:
            reasoning = reasoning.rstrip() + " Key levels: " + ", ".join(level_bits) + "."

    display_regime = _display_market_regime(market_regime, getattr(llm_analysis, "market_regime", None))
    if final_recommendation == "HOLD":
        decision_factor = f"No fresh trade: confidence {confidence:.0%}; position remains {risk_management.position_size}"
    elif final_recommendation == "SELL":
        if strong_directional_setup and support_or_stop is not None:
            decision_factor = f"Leveraged short setup: confidence {confidence:.0%}; stop reference {support_or_stop:.6g}"
        else:
            decision_factor = f"Exit/avoid spot buy: confidence {confidence:.0%}; no short trade opened"
    else:
        decision_factor = f"Leveraged long setup: confidence {confidence:.0%}; stop reference {support_or_stop:.6g}" if strong_directional_setup and support_or_stop is not None else f"Selective spot buy: confidence {confidence:.0%}"

    generated_factors = [
        decision_factor,
        f"Market regime {display_regime}; 24h move {price_change_24h:+.1f}% with estimated volatility {volatility:.1%}",
        f"Technical {technical_score:.0%} vs fundamentals {fundamental_score:.0%}",
    ]
    key_factors = []
    for factor in llm_key_factors + generated_factors:
        if factor and factor not in key_factors:
            key_factors.append(factor)
        if len(key_factors) >= 5:
            break

    execution_notes = getattr(llm_analysis, "execution_notes", "") or "Review price action before execution."
    if not strong_directional_setup and _has_spot_short_language(execution_notes):
        execution_notes = (
            "Reduce or exit existing spot exposure; do not open a leveraged position."
            if trade_intent == "SPOT_EXIT"
            else "Use spot only and wait for confirmation before sizing an entry."
        )

    return DetailedInsights(
        action_summary=action_summary,
        reasoning=reasoning or direction_reasoning(),
        key_factors=key_factors,
        time_horizon=getattr(llm_analysis, "time_horizon", "") or "4-24 hours",
        entry_strategy=entry_strategy,
        price_targets=price_targets,
        risk_management=risk_management,
        execution_notes=execution_notes,
        confidence=confidence,
        market_regime=display_regime,
        trade_intent=trade_intent,
    )


def _select_token_analysis_provider() -> LLMProvider:
    """Token Analysis intentionally uses the same DeepSeek path as platform signals."""
    provider_name = (settings.LLM_PROVIDER or "deepseek").lower()
    if provider_name != LLMProvider.DEEPSEEK.value:
        logger.warning("Token analysis is DeepSeek-only; ignoring LLM_PROVIDER=%s", provider_name)
    if not settings.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY missing for token analysis; dynamic technical fallback will be used")
    return LLMProvider.DEEPSEEK


def initialize_token_analysis_services():
    """Initialize token analysis services."""
    global llm_analysis_service, market_data_service, enhanced_market_service, sentiment_service, coingecko_service, advanced_metrics_service

    try:
        # Initialize LLM analysis service using settings configuration
        provider = _select_token_analysis_provider()
        llm_analysis_service = LLMAnalysisService(provider=provider)
        
        # Initialize market data service
        market_data_service = create_market_data_service()
        
        # Initialize enhanced market data service
        enhanced_market_service = create_enhanced_market_data_service()
        
        # Initialize sentiment service
        sentiment_service = create_sentiment_service()
        
        # Initialize CoinGecko service
        coingecko_service = CoinGeckoService()
        
        # Initialize advanced metrics service
        advanced_metrics_service = create_advanced_metrics_service()
        
        logger.info("Token analysis services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize token analysis services: {e}")
        raise


async def get_llm_analysis_service() -> LLMAnalysisService:
    """Get LLM analysis service dependency."""
    if not llm_analysis_service:
        initialize_token_analysis_services()
    return llm_analysis_service


async def get_market_data_service() -> MarketDataService:
    """Get market data service dependency."""
    if not market_data_service:
        initialize_token_analysis_services()
    return market_data_service


async def get_sentiment_service() -> SentimentService:
    """Get sentiment service dependency."""
    if not sentiment_service:
        initialize_token_analysis_services()
    return sentiment_service


async def get_enhanced_market_service() -> EnhancedMarketDataService:
    """Get enhanced market data service dependency."""
    global enhanced_market_service
    if not enhanced_market_service:
        initialize_token_analysis_services()
    return enhanced_market_service


async def get_coingecko_service() -> CoinGeckoService:
    """Get CoinGecko service dependency."""
    if not coingecko_service:
        initialize_token_analysis_services()
    return coingecko_service


async def get_advanced_metrics_service() -> AdvancedMetricsService:
    """Get advanced metrics service dependency."""
    if not advanced_metrics_service:
        initialize_token_analysis_services()
    return advanced_metrics_service


def _convert_to_market_data(
    comprehensive_data: ComprehensiveMarketData,
    token_ticker: str,
    sentiment_data: Any = None,
    token_data: Optional[Dict[str, Any]] = None,
) -> MarketData:
    """Convert ComprehensiveMarketData to MarketData format that agents expect."""
    try:
        # Calculate volatility from price changes (simplified)
        volatility = abs(comprehensive_data.price_change_24h) / 100.0 if comprehensive_data.price_change_24h else 0.05
        
        # Create MarketData structure that agents expect
        sentiment_value = getattr(sentiment_data, 'overall_sentiment', 0.0) if sentiment_data else 0.0
        overall_sentiment = 'bullish' if sentiment_value > 0.1 else 'bearish' if sentiment_value < -0.1 else 'neutral'
        return MarketData(
            timestamp=comprehensive_data.timestamp,
            token_prices={
                token_ticker: Decimal(str(comprehensive_data.current_price))
            },
            volume_24h={
                token_ticker: Decimal(str(comprehensive_data.volume_24h))
            },
            price_changes={
                token_ticker: Decimal(str(comprehensive_data.price_change_24h))
            },
            market_cap={
                token_ticker: Decimal(str(
                    _first_positive_float(
                        (token_data or {}).get('market_cap'),
                        comprehensive_data.market_cap,
                    ) or 0.0
                ))
            },
            volatility={
                token_ticker: volatility
            },
            trend_indicators={
                'overall_sentiment': overall_sentiment,
                'rsi': comprehensive_data.rsi,
                'ema_20': comprehensive_data.ema_20,
                'ema_50': comprehensive_data.ema_50,
                'macd': comprehensive_data.macd,
                'bollinger_bands': comprehensive_data.bollinger_bands,
                'funding_rate': comprehensive_data.funding_rate,
                'bid_ask_spread': comprehensive_data.bid_ask_spread
            }
        )
    except Exception as e:
        logger.error(f"Error converting market data: {e}")
        # Return minimal fallback data
        return MarketData(
            timestamp=datetime.now(),
            token_prices={token_ticker: Decimal('0')},
            volume_24h={token_ticker: Decimal('0')},
            price_changes={token_ticker: Decimal('0')},
            market_cap={token_ticker: Decimal('0')},
            volatility={token_ticker: 0.05},
            trend_indicators={'overall_sentiment': 'neutral'}
        )


def _convert_to_market_context(
    comprehensive_data: ComprehensiveMarketData,
    token_ticker: str,
    sentiment_data=None,
    token_data: Optional[Dict[str, Any]] = None,
    realized_volatility: Optional[float] = None,
) -> MarketContext:
    """Convert ComprehensiveMarketData to MarketContext for LLM analysis."""
    try:
        # Calculate bollinger band position (simplified)
        bb_position = 0.5  # Default to middle of bands
        if comprehensive_data.bollinger_bands and 'upper' in comprehensive_data.bollinger_bands and 'lower' in comprehensive_data.bollinger_bands:
            bb_upper = comprehensive_data.bollinger_bands['upper']
            bb_lower = comprehensive_data.bollinger_bands['lower']
            if bb_upper > bb_lower:
                bb_position = (comprehensive_data.current_price - bb_lower) / (bb_upper - bb_lower)
                bb_position = max(0, min(1, bb_position))  # Clamp to 0-1
        
        # Extract sentiment info
        fear_greed_index = 50  # Default neutral
        social_sentiment = 'neutral'
        news_sentiment = 'neutral'
        
        if sentiment_data:
            fear_greed_index = int(getattr(sentiment_data, 'fear_greed_index', 50))
            social_sentiment = getattr(sentiment_data, 'sentiment_label', 'neutral')
            news_sentiment = getattr(sentiment_data, 'sentiment_label', 'neutral')
        
        return MarketContext(
            timestamp=comprehensive_data.timestamp,
            symbol=token_ticker,
            current_price=comprehensive_data.current_price,
            price_change_24h=comprehensive_data.price_change_24h,
            volume_24h=comprehensive_data.volume_24h,
            # Prefer true realized volatility; the 24h-move proxy conflates a clean
            # directional trend with volatility and distorts risk/sizing gates.
            volatility=(
                float(realized_volatility)
                if realized_volatility and realized_volatility > 0
                else (abs(comprehensive_data.price_change_24h) / 100.0 if comprehensive_data.price_change_24h else 0.05)
            ),
            rsi=comprehensive_data.rsi,
            macd=comprehensive_data.macd.get('macd', 0) if isinstance(comprehensive_data.macd, dict) else 0,
            bb_position=bb_position,
            fear_greed_index=fear_greed_index,
            social_sentiment=social_sentiment,
            news_sentiment=news_sentiment,
            market_cap=_first_positive_float(
                (token_data or {}).get('market_cap'),
                comprehensive_data.market_cap,
            ) or 0.0,
            funding_rate=comprehensive_data.funding_rate,
            open_interest_change=comprehensive_data.open_interest_change_24h,
            # No hand-written narrative — an empty string is dropped from the prompt so
            # the model does not treat a boilerplate placeholder as real market context.
            market_narrative=""
        )
    except Exception as e:
        logger.error(f"Error converting to market context: {e}")
        # Return minimal fallback context
        return MarketContext(
            timestamp=datetime.now(),
            symbol=token_ticker,
            current_price=0,
            price_change_24h=0,
            volume_24h=0,
            volatility=0.05,
            rsi=50,
            macd=0,
            bb_position=0.5,
            fear_greed_index=50,
            social_sentiment='neutral',
            news_sentiment='neutral',
            market_narrative=""
        )


def _recommendation_snapshot_direction(recommendation: str, trade_intent: Optional[str] = None) -> str:
    """Map only scoreable entries to a directional performance bucket."""
    intent = str(trade_intent or "").upper()
    if intent in {"LONG_ENTRY", "SPOT_BUY"}:
        return "long"
    if intent == "SHORT_ENTRY":
        return "short"
    if intent in {"SPOT_EXIT", "HOLD"}:
        return "neutral"
    normalized = _normalize_recommendation_label(recommendation)
    if normalized == "BUY":
        return "long"
    if normalized == "SELL":
        return "short"
    return "neutral"


def _first_positive_float(*values: Any) -> Optional[float]:
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return None


def _persist_recommendation_snapshot(
    agent_type: str,
    token_ticker: str,
    token_name: Optional[str],
    recommendation: str,
    overall_score: Optional[float],
    confidence: Optional[float],
    risk_level: Optional[str],
    entry_price: Optional[float],
    detailed_insights: Optional["DetailedInsights"],
    llm_analysis: Optional[LLMAnalysis] = None,
    trade_intent: Optional[str] = None,
    token_identity: Optional[Dict[str, Any]] = None,
    data_quality: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist a token-analysis recommendation into agent_recommendation_tracking so the
    resolver worker can later score it against price (24h + 7d) and build a win rate.

    Best-effort and fully guarded: a persistence failure must never break the analysis
    response. HOLD recommendations are still stored (excluded from win rate downstream).
    """
    try:
        entry = _first_positive_float(entry_price)
        if entry is None:
            logger.info(f"⏭️  Skipping recommendation snapshot for {token_ticker}: no valid entry price")
            return

        price_targets = getattr(detailed_insights, "price_targets", None)
        risk_management = getattr(detailed_insights, "risk_management", None)
        target_1 = _first_positive_float(getattr(price_targets, "target_1", None))
        target_2 = _first_positive_float(getattr(price_targets, "target_2", None))
        stop_loss = _first_positive_float(getattr(risk_management, "stop_loss", None))

        now = datetime.now()
        direction = _recommendation_snapshot_direction(recommendation, trade_intent)
        valid_directional_plan = _is_directionally_valid_trade_plan(
            direction,
            entry,
            target_1,
            stop_loss,
        )
        exclusion_reason = None
        if direction in {"long", "short"} and not valid_directional_plan:
            exclusion_reason = "invalid_directional_trade_plan"
            direction = "neutral"
        elif direction == "neutral":
            exclusion_reason = "non_directional_intent"

        llm_provider = str(getattr(llm_analysis, "llm_provider", "") or "").strip() or None
        llm_model = str(getattr(llm_analysis, "llm_model", "") or "").strip() or None
        llm_prompt_tokens = max(int(getattr(llm_analysis, "llm_prompt_tokens", 0) or 0), 0)
        llm_completion_tokens = max(int(getattr(llm_analysis, "llm_completion_tokens", 0) or 0), 0)
        llm_estimated_cost_usd = round(max(float(getattr(llm_analysis, "llm_estimated_cost_usd", 0.0) or 0.0), 0.0), 8)
        llm_cache_hit = bool(getattr(llm_analysis, "llm_cache_hit", False))
        llm_analysis_profile = str(getattr(llm_analysis, "llm_analysis_profile", "") or "").strip() or None
        llm_breakdown = None
        if (
            llm_provider
            or llm_model
            or llm_prompt_tokens > 0
            or llm_completion_tokens > 0
            or llm_estimated_cost_usd > 0
            or llm_cache_hit
        ):
            llm_breakdown = {
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_prompt_tokens": llm_prompt_tokens,
                "llm_completion_tokens": llm_completion_tokens,
                "llm_total_tokens": llm_prompt_tokens + llm_completion_tokens,
                "llm_estimated_cost_usd": llm_estimated_cost_usd,
                "llm_cache_hit": llm_cache_hit,
                "analysis_profile": llm_analysis_profile or "trading",
                "telemetry_source": "token_analysis",
                "telemetry_version": 1,
            }

        record = {
            "agent_type": (agent_type or "").lower(),
            "token_symbol": (token_ticker or "").upper(),
            "token_name": token_name,
            "recommendation": _normalize_recommendation_label(recommendation),
            "direction": direction,
            "overall_score": round(float(overall_score), 4) if overall_score is not None else None,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "risk_level": str(risk_level).upper() if risk_level else None,
            "entry_price": entry,
            "target_1": target_1,
            "target_2": target_2,
            "stop_loss": stop_loss,
            "horizon_24h_due_at": (now + timedelta(hours=24)).isoformat(),
            "horizon_7d_due_at": (now + timedelta(days=7)).isoformat(),
            "created_at": now.isoformat(),
            "metadata": {
                "trade_intent": str(trade_intent or "HOLD").upper(),
                "valid_directional_plan": valid_directional_plan,
                "performance_exclusion_reason": exclusion_reason,
                "token_identity": token_identity or {},
                "data_quality": data_quality or {},
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "analysis_profile": llm_analysis_profile or "trading",
                "ai_confidence_breakdown": llm_breakdown,
            },
        }

        db = get_service_client()
        db.table("agent_recommendation_tracking").insert(record).execute()
        logger.info(
            f"🧾 Tracked {record['agent_type']} recommendation for {record['token_symbol']}: "
            f"{record['recommendation']} @ {entry}"
        )
    except Exception as exc:  # never break the analysis response
        logger.warning(f"Failed to persist recommendation snapshot for {token_ticker}: {exc}")


def _build_token_data_quality(
    token_data: Dict[str, Any],
    comprehensive_data: ComprehensiveMarketData,
    enhanced_data: Any,
) -> Dict[str, Any]:
    """Describe source quality and decide whether leveraged guidance is safe to emit."""
    identity = token_data.get('resolution') or {}
    flags: List[str] = []
    token_price = _first_positive_float(token_data.get('current_price'))
    market_price = _first_positive_float(getattr(comprehensive_data, 'current_price', None))
    enhanced_price = _first_positive_float(getattr(enhanced_data, 'price', None))
    ambiguity = int(identity.get('ambiguous_matches') or 0)
    requested_symbol = str(identity.get('requested_symbol') or token_data.get('symbol') or '').upper()
    resolved_symbol = str(identity.get('resolved_symbol') or token_data.get('symbol') or '').upper()

    if not identity.get('verified'):
        flags.append('token_identity_not_explicitly_verified')
    if ambiguity > 0:
        flags.append('ambiguous_ticker_symbol')
    if requested_symbol and resolved_symbol and requested_symbol != resolved_symbol:
        flags.append('resolved_symbol_mismatch')
    if identity.get('resolution_method') == 'contract_address' and not identity.get('verified'):
        flags.append('contract_identity_unverified')
    if token_price is None:
        flags.append('coingecko_price_unavailable')
    if market_price is None:
        flags.append('execution_market_price_unavailable')
    if enhanced_price is None:
        flags.append('enhanced_market_data_unavailable')
    ema_20 = _first_positive_float(getattr(comprehensive_data, 'ema_20', None))
    ema_50 = _first_positive_float(getattr(comprehensive_data, 'ema_50', None))
    if ema_20 is None or ema_50 is None:
        flags.append('technical_indicators_unavailable')
    if _first_positive_float(getattr(comprehensive_data, 'volume_24h', None)) is None:
        flags.append('liquidity_data_unavailable')
    sentiment_sources = getattr(
        getattr(enhanced_data, 'token_sentiment', None),
        'sentiment_sources',
        {},
    ) or {}
    if sentiment_sources.get('synthetic_social') or sentiment_sources.get('synthetic_news'):
        flags.append('sentiment_proxies_used')

    price_divergence = None
    if token_price and market_price:
        price_divergence = abs(token_price / market_price - 1.0)
        if price_divergence > 0.05:
            flags.append('cross_source_price_divergence')

    critical_flags = {
        'ambiguous_ticker_symbol',
        'resolved_symbol_mismatch',
        'contract_identity_unverified',
        'coingecko_price_unavailable',
        'execution_market_price_unavailable',
        'cross_source_price_divergence',
        'technical_indicators_unavailable',
        'liquidity_data_unavailable',
    }
    leverage_allowed = not any(flag in critical_flags for flag in flags)
    status = 'HIGH' if not flags else 'MEDIUM' if leverage_allowed else 'LOW'
    return {
        'status': status,
        'leverage_allowed': leverage_allowed,
        'flags': flags,
        'price_divergence_pct': round(price_divergence * 100, 4) if price_divergence is not None else None,
        'sources': {
            'identity': identity.get('resolution_method'),
            'fundamentals': 'coingecko',
            'execution_market': 'binance_or_coingecko',
            'enhanced_context': 'coingecko_and_market_regime' if enhanced_price else 'unavailable',
        },
    }


async def _analyze_token_impl(
    request: TokenAnalysisRequest,
    llm_service: LLMAnalysisService,
    market_service: MarketDataService,
    enhanced_service: EnhancedMarketDataService,
    sentiment_service: SentimentService,
    coingecko_service: CoinGeckoService,
    advanced_metrics_service: AdvancedMetricsService,
    *,
    analysis_profile: str = "trading",
    stream_queue: Optional[asyncio.Queue] = None,
) -> TokenAnalysisResponse:
    """
    Analyze a token using AI agents with comprehensive analysis.
    
    Args:
        request: Token analysis request with ticker and preferences
        
    Returns:
        Comprehensive token analysis with AI insights
    """
    try:
        token_ticker = request.token_ticker.upper().strip()
        if not token_ticker or len(token_ticker) > 40 or not token_ticker.replace('-', '').replace('_', '').isalnum():
            raise HTTPException(status_code=422, detail="Invalid token ticker")
        has_contract = bool(str(request.contract_address or '').strip())
        has_chain = request.chain_id is not None
        if has_contract != has_chain:
            raise HTTPException(
                status_code=422,
                detail="contract_address and chain_id must be provided together",
            )
        if has_contract and coingecko_service._asset_platform_for_chain(request.chain_id) is None:
            raise HTTPException(status_code=422, detail="Unsupported chain_id for contract resolution")

        token_task = asyncio.create_task(
            coingecko_service.get_token_data(
                token_ticker,
                coingecko_id=request.coingecko_id,
                contract_address=request.contract_address,
                chain_id=request.chain_id,
            )
        )
        comprehensive_task = asyncio.create_task(
            market_service.get_comprehensive_data(token_ticker, include_defi=False)
        )
        structure_candles_task = asyncio.create_task(
            _fetch_token_structure_candles(
                market_service, 
                token_ticker, 
                interval=getattr(request, "timeframe", "4h") or "4h", 
                limit=120
            )
        )
        micro_candles_task = asyncio.create_task(
            _fetch_token_structure_candles(
                market_service,
                token_ticker,
                interval="15m",
                limit=96
            )
        )
        taker_flow_task = asyncio.create_task(
            _fetch_token_taker_flow(market_service, token_ticker, period="1h", limit=24)
        )
        
        from kata.services.binance_service import BinanceService
        binance_metrics_task = asyncio.create_task(
            BinanceService.get_advanced_ai_metrics(f"{token_ticker}USDT")
        )

        try:
            token_data = await asyncio.wait_for(token_task, timeout=20)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Token identity lookup timed out") from exc
        if not token_data:
            for task in (comprehensive_task, structure_candles_task, micro_candles_task, taker_flow_task, binance_metrics_task):
                task.cancel()
            logger.warning(f"Token {token_ticker} not found in CoinGecko, aborting analysis")
            raise HTTPException(status_code=404, detail=f"Token {token_ticker} not found.")
        else:
            logger.info(f"Token data for {token_ticker}: name={token_data.get('name')}, has_image={bool(token_data.get('image'))}")
            if token_data.get('image'):
                logger.info(f"Image data for {token_ticker}: {token_data.get('image')}")
        
        identity = token_data.get('resolution') or {
            'coingecko_id': token_data.get('id'),
            'symbol': token_data.get('symbol'),
            'verified': False,
            'resolution_method': 'legacy',
        }
        enhanced_task = asyncio.create_task(
            enhanced_service.get_enhanced_token_analysis(
                token_ticker,
                coingecko_id=identity.get('coingecko_id') or token_data.get('id'),
                price_snapshot={
                    'price': token_data.get('current_price') or 0.0,
                    'price_change_24h': token_data.get('price_change_percentage_24h') or 0.0,
                    'volume_24h': token_data.get('total_volume') or 0.0,
                    'market_cap': token_data.get('market_cap') or 0.0,
                    'data_available': bool(token_data.get('current_price')),
                },
            )
        )
        try:
            enhanced_data, comprehensive_data, structure_candles_result, micro_candles_result, taker_flow_result, binance_metrics_result = await asyncio.wait_for(
                asyncio.gather(
                    enhanced_task,
                    comprehensive_task,
                    structure_candles_task,
                    micro_candles_task,
                    taker_flow_task,
                    binance_metrics_task,
                    return_exceptions=True,
                ),
                timeout=30,
            )
        except asyncio.TimeoutError as exc:
            for task in (enhanced_task, comprehensive_task, structure_candles_task, micro_candles_task, taker_flow_task, binance_metrics_task):
                task.cancel()
            raise HTTPException(status_code=504, detail="Market data analysis timed out") from exc

        if isinstance(comprehensive_data, Exception):
            raise comprehensive_data
        if isinstance(enhanced_data, Exception):
            logger.warning("Enhanced analysis unavailable for %s: %s", token_ticker, enhanced_data)
            enhanced_data = await enhanced_service._get_fallback_data(token_ticker)

        # Use enhanced sentiment data when requested; otherwise avoid representing it
        # as a measured source in the response.
        sentiment_data = enhanced_data.token_sentiment if enhanced_data.token_sentiment else None
        if not sentiment_data and request.include_sentiment:
            try:
                generic_sentiment = await sentiment_service.get_unified_sentiment(24)
                sentiment_data = generic_sentiment
            except Exception as e:
                logger.warning(f"Failed to get sentiment data: {e}")

        # Convert ComprehensiveMarketData to the agent format after sentiment is known.
        market_data = _convert_to_market_data(
            comprehensive_data,
            token_ticker,
            sentiment_data,
            token_data=token_data,
        )
        if isinstance(binance_metrics_result, dict):
            market_data.metadata['advanced_ai_metrics'] = binance_metrics_result
            
        data_quality = _build_token_data_quality(token_data, comprehensive_data, enhanced_data)
        
        # Initialize appropriate agent with required services
        agent = None
        if request.agent_type == "yuki":
            # Use YukiAnalysisAgent for token analysis (uses Binance + CoinGecko)
            from kata.agents.yuki_analysis_agent import YukiAnalysisAgent
            agent = YukiAnalysisAgent("analysis_user", {})
        elif request.agent_type == "sakura":
            agent = SakuraAgent("analysis_user", {})
        elif request.agent_type == "ryu":
            agent = RyuAgent("analysis_user", {})
        else:
            # Default to YukiAnalysisAgent for token analysis
            from kata.agents.yuki_analysis_agent import YukiAnalysisAgent
            agent = YukiAnalysisAgent("analysis_user", {})
        
        # Perform agent analysis with focus on the specific token
        logger.info(f"Starting agent analysis for {token_ticker} using {request.agent_type} agent")
        if request.agent_type == "yuki":
            # Yuki needs focus token for focused futures analysis
            agent_analysis = await agent.analyze_market(market_data, focus_token=token_ticker)
        else:
            # Other agents analyze the general market
            agent_analysis = await agent.analyze_market(market_data)
        logger.info(f"Agent analysis completed for {token_ticker}. Strategy: {agent_analysis.get('agent_strategy')}, Technical score: {agent_analysis.get('technical_score')}")

        structure_candles = (
            structure_candles_result
            if isinstance(structure_candles_result, list)
            else []
        )
        micro_candles = (
            micro_candles_result
            if isinstance(micro_candles_result, list)
            else []
        )
        taker_flow_context = (
            taker_flow_result
            if isinstance(taker_flow_result, dict)
            else None
        )
        if isinstance(structure_candles_result, Exception):
            logger.debug("Token structure candles unavailable for %s: %s", token_ticker, structure_candles_result)
        if isinstance(micro_candles_result, Exception):
            logger.debug("Token micro structure candles unavailable for %s: %s", token_ticker, micro_candles_result)
        if isinstance(taker_flow_result, Exception):
            logger.debug("Token taker flow unavailable for %s: %s", token_ticker, taker_flow_result)
        
        # Get historical price data for advanced metrics calculation
        logger.info(f"Fetching historical data for advanced metrics calculation for {token_ticker}")
        
        price_data = [
            float(candle["close"])
            for candle in structure_candles
            if candle.get("close") is not None and float(candle.get("close") or 0) > 0
        ]
        volume_data = [
            float(candle.get("volume") or 0.0) * float(candle.get("close") or 0.0)
            for candle in structure_candles
            if candle.get("close") is not None and float(candle.get("close") or 0) > 0
        ]
        advanced_metrics_available = len(price_data) >= 30 and len(volume_data) >= 30
        if not advanced_metrics_available:
            logger.info(
                "Advanced metrics for %s omitted from LLM context: real candle history unavailable",
                token_ticker,
            )
        
        # Calculate advanced metrics
        logger.info(f"Calculating advanced financial metrics for {token_ticker}")
        market_data_dict = {
            'bid_ask_spread': comprehensive_data.bid_ask_spread if comprehensive_data else 0.001,
            'market_cap': _first_positive_float(
                token_data.get('market_cap') if token_data else None,
                comprehensive_data.market_cap if comprehensive_data else None,
            ) or 0.0,
            'volume_24h': comprehensive_data.volume_24h if comprehensive_data else 0,
        }
        
        if advanced_metrics_available:
            advanced_metrics = await advanced_metrics_service.calculate_advanced_metrics(
                price_data=price_data,
                volume_data=volume_data,
                market_data=market_data_dict,
                timeframe=getattr(request, "timeframe", "4h") or "4h"
            )
        else:
            advanced_metrics = None
        
        # Get agent scores
        technical_score = agent_analysis.get('technical_score')
        fundamental_score = agent_analysis.get('fundamental_score')
        
        if technical_score is None or fundamental_score is None:
            raise HTTPException(
                status_code=500, 
                detail=f"Agent {request.agent_type} failed to provide required scores. Technical: {technical_score}, Fundamental: {fundamental_score}"
            )
        
        # Get agent-specific insights
        agent_strategy = agent_analysis.get('agent_strategy', 'unknown')
        agent_confidence = agent_analysis.get('agent_confidence', 0.5)
        
        # Use token-specific sentiment instead of generic market sentiment
        if hasattr(sentiment_data, 'overall_sentiment'):
            sentiment_score = (sentiment_data.overall_sentiment + 1) / 2  # Convert from -1,1 to 0,1
        elif sentiment_data and hasattr(sentiment_data, 'overall_sentiment'):
            sentiment_score = (sentiment_data.overall_sentiment + 1) / 2
        else:
            sentiment_score = 0.5
        
        # Calculate sophisticated overall score using real available metrics only.
        logger.info(f"Calculating sophisticated overall score for {token_ticker}")

        # Market regime factor. Prefer recent platform-signal regime because it is
        # generated from the same BTC-context engine used by platform signals.
        regime_name = enhanced_data.market_regime.regime if enhanced_data.market_regime else 'UNCERTAIN'
        regime_confidence = enhanced_data.market_regime.confidence if enhanced_data.market_regime else 0.5
        regime_strength = enhanced_data.market_regime.strength if enhanced_data.market_regime else 0.0
        platform_regime_context = await asyncio.to_thread(
            _get_latest_platform_market_regime,
            token_ticker,
        )
        if platform_regime_context:
            regime_name = platform_regime_context["regime"]
            regime_confidence = platform_regime_context.get("confidence") or regime_confidence
            logger.info(
                "Using platform market regime for %s: %s confidence=%.2f source=%s symbol=%s",
                token_ticker,
                regime_name,
                regime_confidence,
                platform_regime_context.get("source"),
                platform_regime_context.get("token_symbol"),
            )

        normalized_regime_name = str(regime_name or 'UNCERTAIN').upper()
        regime_score = 0.5  # Default neutral
        if 'BULL' in normalized_regime_name or 'UPTREND' in normalized_regime_name:
            regime_score = 0.7 + (regime_strength * 0.3)
        elif 'BEAR' in normalized_regime_name or 'DOWNTREND' in normalized_regime_name:
            regime_score = 0.3 - (regime_strength * 0.3)

        if advanced_metrics is not None:
            # Component scores from real advanced metrics (all 0-1 scale)
            technical_advanced = advanced_metrics.technical_score
            liquidity_score = advanced_metrics.liquidity_score
            risk_score = 1 - advanced_metrics.risk_score  # Invert risk (lower risk = higher score)
            efficiency_score = advanced_metrics.efficiency_score

            # Financial ratios scores
            sharpe_score = max(0, min(1, (advanced_metrics.sharpe_ratio + 1) / 3))
            sortino_score = max(0, min(1, (advanced_metrics.sortino_ratio + 1) / 4))

            score_components = {
                'technical_agent': technical_score * 0.15,
                'fundamental_agent': fundamental_score * 0.10,
                'technical_advanced': technical_advanced * 0.20,
                'liquidity': liquidity_score * 0.15,
                'risk_adjusted': risk_score * 0.15,
                'sharpe_ratio': sharpe_score * 0.10,
                'sortino_ratio': sortino_score * 0.05,
                'market_efficiency': efficiency_score * 0.05,
                'sentiment': sentiment_score * 0.03,
                'market_regime': regime_score * 0.02
            }
        else:
            liquidity_score = _clamp_float(getattr(enhanced_data, "liquidity_score", None), default=0.0)
            risk_score = 1 - _clamp_float(getattr(enhanced_data, "drawdown_risk", None), default=0.5)
            score_components = {
                'technical_agent': technical_score * 0.35,
                'fundamental_agent': fundamental_score * 0.20,
                'liquidity': liquidity_score * 0.20,
                'risk_adjusted': risk_score * 0.15,
                'sentiment': sentiment_score * 0.05,
                'market_regime': regime_score * 0.05
            }

        # Calculate final score as weighted sum (0-1 scale)
        overall_score_normalized = sum(score_components.values())
        
        # Ensure score is between 0 and 1
        overall_score_normalized = max(0.0, min(1.0, overall_score_normalized))
        
        logger.info(f"Score components for {token_ticker}: {score_components}")
        logger.info(f"Final score for {token_ticker}: {overall_score_normalized:.3f}")
        
        # Convert to enhanced MarketContext for LLM analysis with available real context
        market_context = _convert_to_market_context(
            comprehensive_data,
            token_ticker,
            sentiment_data,
            token_data=token_data,
            realized_volatility=getattr(advanced_metrics, 'realized_volatility', None),
        )

        market_structure_context = get_market_structure_context_service().build_context(
            symbol=token_ticker,
            current_price=comprehensive_data.current_price,
            candles=structure_candles,
            timeframe=getattr(request, "timeframe", "4h") or "4h",
            orderbook=comprehensive_data.orderbook_depth,
            funding_rate=comprehensive_data.funding_rate,
            long_short_ratio=comprehensive_data.long_short_ratio,
            top_trader_long_ratio=comprehensive_data.top_trader_long_ratio,
            open_interest=comprehensive_data.open_interest,
            open_interest_change_24h=comprehensive_data.open_interest_change_24h,
            taker_flow=taker_flow_context,
            technical_snapshot={
                "price_change_24h": comprehensive_data.price_change_24h,
                "ema_20": comprehensive_data.ema_20,
                "ema_50": comprehensive_data.ema_50,
                "trend_direction": str(regime_name or "").lower(),
            },
        )
        market_structure_prompt = market_structure_context.get("prompt_block") or ""

        # Micro structure (15m)
        if micro_candles:
            micro_structure_context = get_market_structure_context_service().build_context(
                symbol=token_ticker,
                current_price=comprehensive_data.current_price,
                candles=micro_candles,
                timeframe="15m"
            )
            micro_prompt = micro_structure_context.get("prompt_block") or ""
            if micro_prompt:
                market_structure_prompt += f"\n\n{micro_prompt}"

        # --- Deterministic scoring engine (advisory input to the LLM analyst) ---
        # Compute the rule-engine vote up front and hand it to the LLM in the prompt,
        # so its final call is informed by — but not bound to — the rule model.
        price_change = comprehensive_data.price_change_24h if comprehensive_data else 0.0
        volume = comprehensive_data.volume_24h if comprehensive_data else 0.0
        current_price = comprehensive_data.current_price if comprehensive_data else 0.0
        ema_20 = comprehensive_data.ema_20 if comprehensive_data else 0.0
        ema_50 = comprehensive_data.ema_50 if comprehensive_data else 0.0
        realized_volatility = getattr(advanced_metrics, 'realized_volatility', None)
        ext_band = (
            max(0.06, min(0.25, float(realized_volatility) * 3.0))
            if realized_volatility and realized_volatility > 0
            else 0.15
        )
        is_uptrend = ema_20 > ema_50 and ema_50 > 0
        is_downtrend = ema_20 < ema_50 and ema_50 > 0
        is_overbought = current_price > (ema_20 * (1 + ext_band)) if ema_20 > 0 else False
        is_oversold = current_price < (ema_20 * (1 - ext_band)) if ema_20 > 0 else False
        rule_recommendation = _generate_rule_recommendation(
            overall_score=overall_score_normalized,
            technical_score=technical_score,
            price_change_24h=price_change,
            market_regime=normalized_regime_name,
            is_uptrend=is_uptrend,
            is_downtrend=is_downtrend,
            is_overbought=is_overbought,
            is_oversold=is_oversold,
            realized_volatility=realized_volatility,
        )
        rule_confidence = _estimate_rule_confidence(
            recommendation=rule_recommendation,
            overall_score=overall_score_normalized,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            price_change_24h=price_change,
            market_regime=normalized_regime_name,
        )
        rule_engine_block = _build_rule_engine_advisory(
            rule_recommendation=rule_recommendation,
            rule_confidence=rule_confidence,
            overall_score=overall_score_normalized,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            price_change_24h=price_change,
            market_regime=regime_name,
            realized_volatility=realized_volatility,
        )

        if advanced_metrics is not None:
            trend_type = (
                'trending'
                if advanced_metrics.hurst_exponent > 0.55
                else 'mean-reverting'
                if advanced_metrics.hurst_exponent < 0.45
                else 'random'
            )
            advanced_metrics_block = f"""REAL HISTORICAL METRICS:
- Technical Score: {technical_score:.2f}/1.0 | Fundamental: {fundamental_score:.2f}/1.0 | Composite: {overall_score_normalized:.2f}/1.0
- Sharpe Ratio: {advanced_metrics.sharpe_ratio:.2f} | Sortino: {advanced_metrics.sortino_ratio:.2f}
- Liquidity Score: {advanced_metrics.liquidity_score:.2f}/1.0 | Risk Level: {advanced_metrics.risk_score:.2f}/1.0
- Volatility: {advanced_metrics.realized_volatility:.1%} | Efficiency: {advanced_metrics.efficiency_score:.2f}/1.0
- Trend Persistence: Hurst={advanced_metrics.hurst_exponent:.2f} ({trend_type})
- Spread Impact: {advanced_metrics.bid_ask_spread_ratio:.3f} | Market Impact: {advanced_metrics.market_impact_ratio:.3f}
- Volume/MCap Ratio: {advanced_metrics.turnover_ratio:.3f}"""
            scoring_breakdown = f"""SCORING BREAKDOWN (weighted):
- Technical: {score_components['technical_agent'] + score_components['technical_advanced']:.3f}
- Risk-Adjusted: {score_components['risk_adjusted'] + score_components['sharpe_ratio'] + score_components['sortino_ratio']:.3f}
- Liquidity: {score_components['liquidity']:.3f}"""
        else:
            advanced_metrics_block = """REAL HISTORICAL METRICS:
- Not included: real Binance candle history was unavailable or insufficient. Do not infer Sharpe, Hurst, advanced volatility, or market-impact metrics."""
            scoring_breakdown = f"""SCORING BREAKDOWN (weighted, available data only):
- Technical Agent: {score_components['technical_agent']:.3f}
- Fundamental Agent: {score_components['fundamental_agent']:.3f}
- Liquidity: {score_components['liquidity']:.3f}
- Risk: {score_components['risk_adjusted']:.3f}"""

        actual_price = token_data.get('current_price', comprehensive_data.current_price if comprehensive_data else 0) if token_data else (comprehensive_data.current_price if comprehensive_data else 0)
        
        ai_metrics_str = ""
        if isinstance(binance_metrics_result, dict):
            t_ratio = binance_metrics_result.get('taker_buy_sell_ratio')
            liq_vol = binance_metrics_result.get('liquidation_volume')
            basis = binance_metrics_result.get('basis_premium')
            
            t_ratio_str = f"{t_ratio:.2f}" if t_ratio is not None else "N/A"
            liq_vol_str = f"${liq_vol:,.0f}" if liq_vol is not None else "N/A"
            basis_str = f"{basis:+.4f}%" if basis is not None else "N/A"
            
            ai_metrics_str = f"""
BINANCE FUTURES DATA (REAL):
- Taker Buy/Sell Ratio: {t_ratio_str} (>1.0 indicates aggressive buying momentum)
- Basis Premium (Spot vs Futures): {basis_str} (Contango=bullish/greedy, Backwardation=bearish/fear)
- Liquidation Volume: {liq_vol_str} (High volume indicates exhaustion/capitulation)
"""

        token_fundamentals_block = ""
        if analysis_profile == "spot_investment":
            raw_description = str((token_data or {}).get("description") or "")
            plain_description = " ".join(
                unescape(re.sub(r"<[^>]+>", " ", raw_description)).split()
            )[:1200]
            categories = [
                str(category).strip()
                for category in ((token_data or {}).get("categories") or [])
                if str(category).strip()
            ][:6]
            fundamental_lines = [
                f"- Token / Project: {(token_data or {}).get('name') or token_ticker} ({token_ticker})",
                f"- Market Cap: ${float((token_data or {}).get('market_cap') or 0):,.0f}",
                f"- Market Cap Rank: #{int((token_data or {}).get('market_cap_rank') or 0)}",
                f"- 24h Spot Volume: ${float((token_data or {}).get('total_volume') or 0):,.0f}",
            ]
            if categories:
                fundamental_lines.append(f"- Verified Categories: {', '.join(categories)}")
            if plain_description:
                fundamental_lines.append(f"- Verified Project Summary: {plain_description}")
            else:
                fundamental_lines.append(
                    "- Verified Project Summary: Unavailable. Do not invent a utility, product, partnership, or adoption claim."
                )
            token_fundamentals_block = (
                "TOKEN FUNDAMENTALS FOR THE SPOT PURCHASE THESIS:\n"
                + "\n".join(fundamental_lines)
                + "\n"
            )

        enhanced_context = f"""Advanced Analysis for {token_ticker}:

{token_fundamentals_block}
{advanced_metrics_block}
{ai_metrics_str}
MARKET CONDITIONS:
- Current Price: ${actual_price:.6f}
- Market Regime: {regime_name} (confidence: {regime_confidence:.2f})
- Price Change 24h: {token_data.get('price_change_percentage_24h', 0) if token_data else comprehensive_data.price_change_24h:+.2f}% | Volume: {comprehensive_data.volume_24h:,.0f}

{market_structure_prompt}

{scoring_breakdown}

{rule_engine_block}

Focus: Analyze risk-adjusted performance, real market structure, and real order flow for optimal entry/exit strategy. You are the analyst — the scoring engine above is a second opinion, not a constraint. CRITICAL: Use the provided Current Price as the strict anchor for all price targets. Do not invent prices if market structure data is missing."""
        
        # Perform enhanced LLM analysis with all advanced metrics (full thinking enabled)
        logger.info(f"Starting enhanced LLM analysis for {token_ticker} with advanced metrics (thinking enabled)")
        try:
            if stream_queue is not None:
                await stream_queue.put({"type": "status", "message": f"Starting deep AI reasoning for {token_ticker}..."})
                llm_analysis = None
                async for stream_event in llm_service.analyze_trading_opportunity_stream(
                    market_context,
                    enhanced_context,
                    max_output_tokens=None,
                    strict=True,
                    analysis_profile=analysis_profile,
                    thinking_type="enabled",
                ):
                    if stream_event.get("type") in {"thinking", "content", "status"}:
                        await stream_queue.put(stream_event)
                    elif stream_event.get("type") == "complete":
                        llm_analysis = stream_event.get("analysis")
                if llm_analysis is None:
                    raise RuntimeError(f"Streaming LLM analysis for {token_ticker} did not yield a complete result")
            else:
                llm_analysis = await llm_service.analyze_trading_opportunity(
                    market_context,
                    enhanced_context,
                    max_output_tokens=None,
                    strict=True,
                    analysis_profile=analysis_profile,
                    thinking_type="enabled",
                )
        except Exception as exc:
            logger.error("Token-analysis LLM failed for %s: %s", token_ticker, exc)
            raise HTTPException(status_code=502, detail="AI analysis could not produce a complete result. Please try again.") from exc
        logger.info(f"Enhanced LLM analysis completed for {token_ticker}. Action summary: {llm_analysis.action_summary if llm_analysis else 'None'}")
        
        # Determine risk level based on multiple factors
        risk_factors = [
            enhanced_data.drawdown_risk,
            enhanced_data.volatility_percentile / 100,
            1 - enhanced_data.liquidity_score,
            1 - enhanced_data.market_regime.confidence if enhanced_data.market_regime else 0.5
        ]
        avg_risk = sum(risk_factors) / len(risk_factors)
        
        if avg_risk > 0.7:
            risk_level = "HIGH"
        elif avg_risk > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # LLM-authoritative decision: the analyst's call stands. The rule engine's vote
        # (rule_recommendation / rule_confidence, computed above) already informed the
        # LLM through the advisory block in the prompt; it does not override it here.
        recommendation, decision_context = _finalize_llm_decision(
            llm_analysis=llm_analysis,
            rule_recommendation=rule_recommendation,
            rule_confidence=rule_confidence,
        )
        logger.info(
            "Final token analysis decision for %s: rule=%s, llm=%s, final=%s, resolution=%s",
            token_ticker,
            decision_context.get("rule_recommendation"),
            decision_context.get("llm_recommendation"),
            recommendation,
            decision_context.get("resolution"),
        )
        
        detailed_insights = _normalize_detailed_insights_for_recommendation(
            llm_analysis=llm_analysis,
            recommendation=recommendation,
            token_ticker=token_ticker,
            current_price=token_data.get('current_price', comprehensive_data.current_price if comprehensive_data else 0) if token_data else (comprehensive_data.current_price if comprehensive_data else 0),
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            overall_score=overall_score_normalized,
            price_change_24h=price_change,
            market_regime=regime_name,
            decision_context=decision_context,
            leverage_allowed=bool(data_quality.get('leverage_allowed')),
            resolved_risk_level=risk_level,
            agent_type=request.agent_type,
        )

        trade_intent = detailed_insights.trade_intent if detailed_insights else "HOLD"

        # Persist this recommendation so the resolver can later score it into a win rate.
        # Best-effort: guarded internally and must never affect the analysis response.
        _snapshot_entry_price = _first_positive_float(
            getattr(getattr(detailed_insights, "entry_strategy", None), "optimal_entry", None),
            token_data.get('current_price') if token_data else None,
            comprehensive_data.current_price if comprehensive_data else None,
        )
        asyncio.create_task(asyncio.to_thread(
            _persist_recommendation_snapshot,
            agent_type=request.agent_type,
            token_ticker=token_ticker,
            token_name=token_data.get('name', token_ticker) if token_data else token_ticker,
            recommendation=recommendation,
            overall_score=overall_score_normalized,
            confidence=detailed_insights.confidence if detailed_insights else agent_confidence,
            risk_level=risk_level,
            entry_price=_snapshot_entry_price,
            detailed_insights=detailed_insights,
            llm_analysis=llm_analysis,
            trade_intent=trade_intent,
            token_identity=identity,
            data_quality=data_quality,
        ))

        return TokenAnalysisResponse(
            token_ticker=token_ticker,
            token_name=token_data.get('name', token_ticker) if token_data else token_ticker,
            token_image=token_data.get('image', None) if token_data else None,
            analysis_timestamp=datetime.now(),
            agent_type=request.agent_type,
            overall_score=overall_score_normalized,
            risk_level=risk_level,
            recommendation=recommendation,
            trade_intent=trade_intent,
            request_id=request.request_id,
            token_identity=identity,
            data_quality=data_quality,
            technical_analysis={
                'price': token_data.get('current_price', 0) if token_data else 0,
                'market_cap': token_data.get('market_cap', 0) if token_data else 0,
                'volume_24h': token_data.get('total_volume', 0) if token_data else 0,
                'price_change_24h': token_data.get('price_change_percentage_24h', comprehensive_data.price_change_24h if comprehensive_data else 0) if token_data else (comprehensive_data.price_change_24h if comprehensive_data else 0),
                'technical_score': technical_score,
                'indicators': {**agent_analysis.get('technical_indicators', {}), **(binance_metrics_result if isinstance(binance_metrics_result, dict) else {})},
                'decision_inputs': decision_context
            },
            fundamental_analysis={
                'market_cap_rank': token_data.get('market_cap_rank', 0) if token_data else 0,
                'circulating_supply': token_data.get('circulating_supply', 0) if token_data else 0,
                'total_supply': token_data.get('total_supply', 0) if token_data else 0,
                'fundamental_score': fundamental_score,
                'liquidity_score': agent_analysis.get('liquidity_score', 0.5)
            },
            sentiment_analysis=asdict(sentiment_data) if request.include_sentiment and sentiment_data else None,
            market_data={
                'price': token_data.get('current_price', comprehensive_data.current_price if comprehensive_data else 0) if token_data else (comprehensive_data.current_price if comprehensive_data else 0),
                'market_cap': token_data.get('market_cap', comprehensive_data.market_cap if comprehensive_data else 0) if token_data else (comprehensive_data.market_cap if comprehensive_data else 0),
                'volume_24h': token_data.get('total_volume', comprehensive_data.volume_24h if comprehensive_data else 0) if token_data else (comprehensive_data.volume_24h if comprehensive_data else 0),
                'price_change_24h': token_data.get('price_change_percentage_24h', comprehensive_data.price_change_24h if comprehensive_data else 0) if token_data else (comprehensive_data.price_change_24h if comprehensive_data else 0),
                'ath': token_data.get('ath', 0) if token_data else 0,
                'atl': token_data.get('atl', 0) if token_data else 0,
                'ath_change_percentage': token_data.get('ath_change_percentage', 0) if token_data else 0,
                'atl_change_percentage': token_data.get('atl_change_percentage', 0) if token_data else 0,
                'market_regime': regime_name,
                'market_regime_confidence': regime_confidence,
                'market_regime_source': platform_regime_context.get("source") if platform_regime_context else "token_analysis",
                'market_structure_context': market_structure_context
            },
            agent_insights=_generate_agent_specific_insights(
                request.agent_type,
                agent_strategy,
                technical_score,
                fundamental_score,
                agent_confidence,
                llm_analysis.reasoning,
                token_data={
                    'symbol': token_ticker,
                    'price_change_24h': comprehensive_data.price_change_24h,
                    'volume_24h': comprehensive_data.volume_24h,
                    'market_cap': comprehensive_data.market_cap
                },
                enhanced_data=enhanced_data.__dict__ if enhanced_data else None,
                sentiment_data=sentiment_data.__dict__ if sentiment_data else None,
                recommendation=recommendation,
                risk_level=risk_level,
                detailed_insights=detailed_insights,
                decision_context=decision_context
            ),
            detailed_insights=detailed_insights
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing token {request.token_ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze token: {str(e)}")


_analysis_inflight: Dict[str, asyncio.Task] = {}
_analysis_result_cache: Dict[str, tuple[datetime, TokenAnalysisResponse]] = {}
_analysis_request_lock = asyncio.Lock()
_ANALYSIS_IDEMPOTENCY_TTL = timedelta(minutes=5)
_ANALYSIS_END_TO_END_TIMEOUT_SECONDS = 110


async def _finalize_analysis_task(idempotency_key: str, task: asyncio.Task) -> None:
    """Cache successful work and always release the in-flight slot."""
    try:
        result = task.result()
    except (Exception, asyncio.CancelledError):
        result = None
    async with _analysis_request_lock:
        if result is not None:
            _analysis_result_cache[idempotency_key] = (datetime.utcnow(), result)
        if _analysis_inflight.get(idempotency_key) is task:
            _analysis_inflight.pop(idempotency_key, None)


async def _run_token_analysis_request(
    request: TokenAnalysisRequest,
    llm_service: LLMAnalysisService,
    market_service: MarketDataService,
    enhanced_service: EnhancedMarketDataService,
    sentiment_service: SentimentService,
    coingecko_service: CoinGeckoService,
    advanced_metrics_service: AdvancedMetricsService,
    credit_service: CreditService,
    stream_queue: Optional[asyncio.Queue] = None,
) -> TokenAnalysisResponse:
    """Run one bounded analysis, with server-side billing and automatic refund."""
    billing_user_id = str(request.billing_user_id or '').strip()
    credits_remaining: Optional[int] = None
    charged = False

    if billing_user_id:
        if not request.request_id:
            raise HTTPException(status_code=422, detail="request_id is required for billed analysis")
        deduction = await credit_service.deduct_credits(
            user_id=billing_user_id,
            amount=1,
            service_used="token_analysis",
            description=f"AI token analysis for {request.token_ticker.upper().strip()}",
        )
        if not deduction.get('success'):
            status_code = 402 if deduction.get('error') == 'insufficient_credits' else 503
            raise HTTPException(
                status_code=status_code,
                detail=deduction.get('message') or 'Unable to use analysis credit',
            )
        charged = True
        credits_remaining = int(deduction.get('credits_remaining') or 0)

    try:
        # For streaming requests, give ample time (up to 300s) since continuous SSE bytes prevent proxy drops
        effective_timeout = 300.0 if stream_queue is not None else _ANALYSIS_END_TO_END_TIMEOUT_SECONDS
        result = await asyncio.wait_for(
            _analyze_token_impl(
                request,
                llm_service,
                market_service,
                enhanced_service,
                sentiment_service,
                coingecko_service,
                advanced_metrics_service,
                analysis_profile="trading",
                stream_queue=stream_queue,
            ),
            timeout=effective_timeout,
        )
        result.credits_remaining = credits_remaining
        if stream_queue is not None:
            await stream_queue.put({"type": "complete", "result": result.dict()})
            await stream_queue.put({"type": "done"})
        return result
    except asyncio.TimeoutError as exc:
        if stream_queue is not None:
            await stream_queue.put({"type": "error", "message": "Token analysis timed out"})
            await stream_queue.put({"type": "done"})
        if charged:
            refund = await credit_service.add_credits(
                user_id=billing_user_id,
                amount=1,
                transaction_type="refund",
                description=f"Refund for timed-out token analysis: {request.token_ticker.upper().strip()}",
            )
            if not refund.get('success'):
                logger.error("Failed to refund timed-out token analysis for %s", billing_user_id)
        raise HTTPException(status_code=504, detail="Token analysis timed out") from exc
    except Exception as exc:
        if stream_queue is not None:
            await stream_queue.put({"type": "error", "message": str(exc)})
            await stream_queue.put({"type": "done"})
        if charged:
            refund = await credit_service.add_credits(
                user_id=billing_user_id,
                amount=1,
                transaction_type="refund",
                description=f"Refund for failed token analysis: {request.token_ticker.upper().strip()}",
            )
            if not refund.get('success'):
                logger.error("Failed to refund failed token analysis for %s", billing_user_id)
        raise


@router.post("/analyze-stream", summary="Stream token analysis with live AI reasoning")
async def analyze_token_stream(
    request: TokenAnalysisRequest,
    llm_service: LLMAnalysisService = Depends(get_llm_analysis_service),
    market_service: MarketDataService = Depends(get_market_data_service),
    enhanced_service: EnhancedMarketDataService = Depends(get_enhanced_market_service),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
    coingecko_service: CoinGeckoService = Depends(get_coingecko_service),
    advanced_metrics_service: AdvancedMetricsService = Depends(get_advanced_metrics_service),
    credit_service: CreditService = Depends(get_credit_service),
):
    """
    Stream token analysis via Server-Sent Events (SSE).
    Streams live 'thinking' reasoning tokens, 'status' updates, and emits 'complete' with the final analysis.
    """
    async def sse_event_generator():
        stream_queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_run_token_analysis_request(
            request,
            llm_service,
            market_service,
            enhanced_service,
            sentiment_service,
            coingecko_service,
            advanced_metrics_service,
            credit_service,
            stream_queue=stream_queue,
        ))

        # Emit initial event immediately to establish SSE connection and reset proxy idle timer
        yield f"data: {json.dumps({'type': 'status', 'message': f'Initializing analysis for {request.token_ticker.upper()}...'}, default=str)}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(stream_queue.get(), timeout=12.0)
                if event.get("type") == "done":
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.TimeoutError:
                # Send periodic heartbeat ping across the wire to keep proxy connection alive
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except Exception as e:
                logger.error(f"SSE generator error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

        try:
            await task
        except Exception as e:
            logger.error(f"Underlying analysis task finished with error: {e}")

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/analyze", summary="Analyze token with AI agent")
async def analyze_token(
    request: TokenAnalysisRequest,
    llm_service: LLMAnalysisService = Depends(get_llm_analysis_service),
    market_service: MarketDataService = Depends(get_market_data_service),
    enhanced_service: EnhancedMarketDataService = Depends(get_enhanced_market_service),
    sentiment_service: SentimentService = Depends(get_sentiment_service),
    coingecko_service: CoinGeckoService = Depends(get_coingecko_service),
    advanced_metrics_service: AdvancedMetricsService = Depends(get_advanced_metrics_service),
    credit_service: CreditService = Depends(get_credit_service),
) -> TokenAnalysisResponse:
    """Idempotent public wrapper around the full token-analysis pipeline."""
    request_id = str(request.request_id or '').strip()
    if not request_id:
        return await _run_token_analysis_request(
            request,
            llm_service,
            market_service,
            enhanced_service,
            sentiment_service,
            coingecko_service,
            advanced_metrics_service,
            credit_service,
        )
    if len(request_id) > 128 or not all(char.isalnum() or char in '-_:' for char in request_id):
        raise HTTPException(status_code=422, detail="Invalid request_id")

    billing_user_id = str(request.billing_user_id or '').strip()
    if len(billing_user_id) > 256:
        raise HTTPException(status_code=422, detail="Invalid billing_user_id")
    idempotency_key = f"{billing_user_id}:{request_id}"
    now = datetime.utcnow()
    async with _analysis_request_lock:
        expired = [
            key for key, (created_at, _) in _analysis_result_cache.items()
            if now - created_at > _ANALYSIS_IDEMPOTENCY_TTL
        ]
        for key in expired:
            _analysis_result_cache.pop(key, None)

        cached = _analysis_result_cache.get(idempotency_key)
        if cached:
            return cached[1]

        task = _analysis_inflight.get(idempotency_key)
        if task is None:
            task = asyncio.create_task(_run_token_analysis_request(
                request,
                llm_service,
                market_service,
                enhanced_service,
                sentiment_service,
                coingecko_service,
                advanced_metrics_service,
                credit_service,
            ))
            _analysis_inflight[idempotency_key] = task
            task.add_done_callback(
                lambda completed, key=idempotency_key: asyncio.create_task(
                    _finalize_analysis_task(key, completed)
                )
            )

    return await asyncio.shield(task)


@router.get("/quick-analysis/{token_ticker}", summary="Quick token analysis")
async def quick_token_analysis(
    token_ticker: str,
    agent_type: str = "yuki",
    llm_service: LLMAnalysisService = Depends(get_llm_analysis_service),
    coingecko_service: CoinGeckoService = Depends(get_coingecko_service)
) -> Dict[str, Any]:
    """
    Quick token analysis for dashboard display.
    
    Args:
        token_ticker: Token ticker symbol
        agent_type: Type of agent to use for analysis
        
    Returns:
        Quick analysis results
    """
    try:
        token_ticker = token_ticker.upper()
        
        # Get basic token data
        token_data = await coingecko_service.get_token_data(token_ticker)
        if not token_data:
            raise HTTPException(status_code=404, detail=f"Token {token_ticker} not found")
        
        # Simple scoring based on price change and market cap
        price_change = token_data.get('price_change_percentage_24h', 0) if token_data else 0
        market_cap = token_data.get('market_cap', 0) if token_data else 0
        
        # Calculate quick score
        momentum_score = min(100, max(0, 50 + (price_change * 2)))
        market_cap_score = min(100, max(0, 100 - (market_cap / 1000000000 * 10)))  # Prefer smaller caps
        
        overall_score = (momentum_score * 0.6 + market_cap_score * 0.4)
        
        return {
            "token_ticker": token_ticker,
            "token_name": token_data.get('name', token_ticker) if token_data else token_ticker,
            "price": token_data.get('current_price', 0) if token_data else 0,
            "market_cap": token_data.get('market_cap', 0) if token_data else 0,
            "price_change_24h": price_change,
            "volume_24h": token_data.get('total_volume', 0) if token_data else 0,
            "overall_score": overall_score,
            "momentum_score": momentum_score,
            "market_cap_score": market_cap_score,
            "agent_type": agent_type,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick analysis for {token_ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze token: {str(e)}")


def _generate_agent_specific_insights(
    agent_type: str,
    agent_strategy: str,
    technical_score: float,
    fundamental_score: float,
    agent_confidence: float,
    base_reasoning: str,
    token_data: dict = None,
    enhanced_data: dict = None,
    sentiment_data: dict = None,
    recommendation: str = "HOLD",
    risk_level: str = "MEDIUM",
    detailed_insights: Optional[DetailedInsights] = None,
    decision_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate agent-specific insights based on the agent's strategy and analysis.
    
    This makes the analysis more differentiated between agents.
    """
    try:
        # Base agent characteristics
        agent_characteristics = {
            'yuki': {
                'style': 'aggressive futures trader',
                'focus': 'momentum and volatility',
                'risk_tolerance': 'high',
                'timeframe': 'short-term'
            },
            'sakura': {
                'style': 'conservative investor',
                'focus': 'stability and risk management',
                'risk_tolerance': 'low',
                'timeframe': 'long-term'
            },
            'ryu': {
                'style': 'balanced trader',
                'focus': 'technical analysis and momentum',
                'risk_tolerance': 'medium',
                'timeframe': 'medium-term'
            }
        }
        
        agent_info = agent_characteristics.get(agent_type, agent_characteristics['yuki'])
        agent_name = agent_type.title() if agent_type in agent_characteristics else "AI"
        token_symbol = token_data.get('symbol', 'this token') if token_data else 'this token'
        price_change = token_data.get('price_change_24h', 0) if token_data else 0

        if detailed_insights:
            recommendation = _normalize_recommendation_label(recommendation)
            risk_management = detailed_insights.risk_management
            position_size = risk_management.position_size if risk_management else None
            max_leverage = risk_management.max_leverage if risk_management else None
            if _is_placeholder_text(position_size) or _is_placeholder_text(max_leverage):
                position_size, max_leverage = _risk_guidance_for_display(
                    recommendation=recommendation,
                    confidence=detailed_insights.confidence,
                    risk_level=risk_level,
                    volatility=abs(price_change) / 100
                )

            resolution = decision_context.get('resolution') if decision_context else None
            resolution_note = f" {str(resolution).capitalize()}." if resolution else ""
            if not _is_placeholder_text(detailed_insights.reasoning):
                return detailed_insights.reasoning
            if not _is_placeholder_text(detailed_insights.action_summary):
                return detailed_insights.action_summary
            return (
                f"{agent_name} final view: {recommendation} {token_symbol}. "
                f"Technical {technical_score:.0%}, fundamentals {fundamental_score:.0%}, "
                f"24h move {price_change:+.1f}%, confidence {detailed_insights.confidence:.0%}, "
                f"risk {risk_level}. Trade plan {position_size}; mode {max_leverage}.{resolution_note}"
            )
        
        # Generate agent-specific reasoning with real market data
        if agent_type == 'yuki':
            # Use real market data for token-specific insights
            volume = token_data.get('volume_24h', 0) if token_data else 0
            market_cap = token_data.get('market_cap', 0) if token_data else 0
            
            # Analyze actual market conditions
            if technical_score > 0.7:
                if price_change > 5:  # Strong uptrend
                    insight = f"Yuki's aggressive futures strategy identifies strong momentum patterns in {token_symbol}. "
                    insight += f"Technical score {technical_score:.1%} with {price_change:+.1f}% 24h gain suggests favorable entry conditions for futures trading. "
                    insight += f"High volume ({volume:,.0f}) confirms momentum. Consider leveraged long positions with tight risk management."
                elif price_change < -5:  # Strong downtrend
                    insight = f"Yuki's aggressive futures strategy detects strong bearish momentum in {token_symbol}. "
                    insight += f"Technical score {technical_score:.1%} despite {price_change:+.1f}% 24h decline suggests potential reversal setup. "
                    insight += f"Monitor for oversold conditions and funding rate arbitrage opportunities."
                else:  # Sideways with strong technicals
                    insight = f"Yuki's aggressive futures strategy identifies strong technical foundation in {token_symbol}. "
                    insight += f"Technical score {technical_score:.1%} with stable price action suggests accumulation phase. "
                    insight += f"Consider leveraged positions on breakout confirmation with tight risk management."
            elif technical_score > 0.5:
                if price_change > 2:  # Moderate uptrend
                    insight = f"Yuki detects moderate momentum with {technical_score:.1%} technical strength in {token_data.get('symbol', 'this token')}. "
                    insight += f"Price up {price_change:+.1f}% with {volume:,.0f} volume suggests developing trend. "
                    insight += f"Monitor for breakout opportunities and funding rate arbitrage."
                elif price_change < -2:  # Moderate downtrend
                    insight = f"Yuki detects moderate bearish pressure with {technical_score:.1%} technical strength in {token_data.get('symbol', 'this token')}. "
                    insight += f"Price down {price_change:+.1f}% suggests potential short opportunities or reversal setup. "
                    insight += f"Monitor volume for confirmation."
                else:  # Sideways
                    insight = f"Yuki detects moderate technical strength with {technical_score:.1%} in {token_data.get('symbol', 'this token')}. "
                    insight += f"Sideways price action with {volume:,.0f} volume suggests range-bound trading. "
                    insight += f"Monitor for breakout opportunities and funding rate arbitrage."
            else:
                if price_change < -10:  # Sharp decline
                    insight = f"Yuki's analysis shows weak technical signals ({technical_score:.1%}) in {token_data.get('symbol', 'this token')}. "
                    insight += f"Sharp {price_change:+.1f}% decline with weak volume suggests avoid aggressive futures strategies. "
                    insight += f"Wait for technical recovery or consider short positions on bounces."
                elif price_change > 10:  # Sharp rise
                    insight = f"Yuki's analysis shows weak technical foundation ({technical_score:.1%}) despite {price_change:+.1f}% surge in {token_data.get('symbol', 'this token')}. "
                    insight += f"High volume ({volume:,.0f}) suggests momentum but weak technicals indicate potential reversal. "
                    insight += f"Current conditions don't favor aggressive futures strategies."
                else:  # Weak technicals
                    insight = f"Yuki's analysis shows weak technical signals ({technical_score:.1%}) in {token_data.get('symbol', 'this token')}. "
                    insight += f"Price change {price_change:+.1f}% with {volume:,.0f} volume suggests lack of conviction. "
                    insight += f"Current conditions don't favor aggressive futures strategies."
                
        elif agent_type == 'sakura':
            if fundamental_score > 0.7:
                insight = f"Sakura's conservative approach finds stable fundamentals ({fundamental_score:.1%}). "
                insight += f"Token shows characteristics suitable for long-term investment with capital preservation focus."
            elif fundamental_score > 0.5:
                insight = f"Sakura identifies moderate fundamental strength ({fundamental_score:.1%}). "
                insight += f"Consider gradual position building with strict risk controls."
            else:
                insight = f"Sakura's analysis indicates weak fundamentals ({fundamental_score:.1%}). "
                insight += f"Current conditions don't meet conservative investment criteria."
                
        elif agent_type == 'ryu':
            if technical_score > 0.6 and fundamental_score > 0.6:
                insight = f"Ryu's balanced analysis shows strong convergence ({technical_score:.1%} technical, {fundamental_score:.1%} fundamental). "
                insight += f"Optimal conditions for balanced portfolio allocation with moderate risk."
            elif technical_score > 0.5 or fundamental_score > 0.5:
                insight = f"Ryu identifies mixed signals (technical: {technical_score:.1%}, fundamental: {fundamental_score:.1%}). "
                insight += f"Consider selective positions with balanced risk-reward profile."
            else:
                insight = f"Ryu's analysis shows weak signals across metrics. "
                insight += f"Current conditions suggest waiting for better opportunities."
        
        # Add confidence and strategy context
        insight += f"\n\nAgent Strategy: {agent_strategy.replace('_', ' ').title()}"
        insight += f"\nConfidence Level: {agent_confidence:.1%}"
        insight += f"\nAnalysis Style: {agent_info['style']} focusing on {agent_info['focus']}"
        
        return insight
        
    except Exception as e:
        logger.error(f"Error generating agent-specific insights: {e}")
        return base_reasoning
