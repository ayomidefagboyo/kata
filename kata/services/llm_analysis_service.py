"""
LLM-Enhanced Analysis Service

This service integrates LLMs (Claude, OpenAI, Llama) to provide advanced reasoning
for trading decisions, combining technical analysis with natural language understanding.
"""

import asyncio
import copy
import hashlib
import logging
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import os
import threading
import time
from decimal import Decimal

# LLM Client imports
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


_PLACEHOLDER_RISK_MARKERS = ("ai-determined", "determined based")


def _is_placeholder_risk_text(value: Any) -> bool:
    """Detect old placeholder guidance that should never be shown to users."""
    if not isinstance(value, str) or not value.strip():
        return True
    normalized = value.strip().lower()
    return any(marker in normalized for marker in _PLACEHOLDER_RISK_MARKERS)


def _risk_guidance(
    recommendation: str,
    confidence: float,
    risk_assessment: str,
    volatility: float
) -> tuple[str, str]:
    """Return conservative sizing guidance, with leverage only for strong trade setups."""
    recommendation = str(recommendation or "HOLD").upper()
    risk_assessment = str(risk_assessment or "MEDIUM").upper()
    confidence = max(0.0, min(1.0, float(confidence or 0.5)))
    volatility = max(0.0, float(volatility or 0.0))

    if recommendation == "HOLD":
        return "0% (wait for confirmation)", "1x only"

    high_risk = risk_assessment == "HIGH" or volatility >= 0.08 or confidence < 0.55
    low_risk = risk_assessment == "LOW" and volatility <= 0.04 and confidence >= 0.68
    strong_directional_setup = confidence >= 0.65 and risk_assessment != "HIGH"

    if recommendation in {"SELL", "STRONG_SELL"} and not strong_directional_setup:
        return "Reduce/exit spot exposure", "Spot only"

    if high_risk:
        return "1% portfolio risk", "2x max"
    if low_risk:
        return "2-3% portfolio risk", "5x max"
    return "1-2% portfolio risk", "3x max"


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"


@dataclass
class LLMUsageTelemetry:
    """Per-request token/cost telemetry for one LLM analysis call."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cache_hit: bool = False
    analysis_profile: str = "trading"


@dataclass
class MarketContext:
    """Market context for LLM analysis."""
    timestamp: datetime
    symbol: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    volatility: float
    
    # Technical indicators
    rsi: float
    macd: float
    bb_position: float  # Position within Bollinger Bands (0-1)
    
    # Market sentiment
    fear_greed_index: int
    social_sentiment: str
    news_sentiment: str
    
    # Market regime context
    market_regime: str = "sideways"  # bull_market, bear_market, or sideways
    
    # Fundamental data
    market_cap: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest_change: Optional[float] = None
    
    # Recent news/events
    recent_news: List[str] = field(default_factory=list)
    market_narrative: str = ""


@dataclass
class EntryStrategy:
    """Entry strategy details."""
    optimal_entry: float
    entry_range_low: float
    entry_range_high: float
    market_order_ok: bool = False

@dataclass
class PriceTargets:
    """Price target levels."""
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None

@dataclass
class RiskManagement:
    """Risk management parameters."""
    stop_loss: Optional[float] = None
    position_size: str = "2-3% spot allocation"
    max_leverage: str = "Spot only"

@dataclass
class LLMAnalysis:
    """Enhanced LLM analysis result with actionable trading details."""
    recommendation: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    action_summary: str  # Clear one-line action
    reasoning: str
    key_factors: List[str]
    risk_assessment: str
    time_horizon: str
    entry_strategy: Optional[EntryStrategy] = None
    price_targets: Optional[PriceTargets] = None
    risk_management: Optional[RiskManagement] = None
    market_regime: str = "UNCERTAIN"
    execution_notes: str = ""
    
    # Legacy fields for backward compatibility
    position_sizing: str = "MEDIUM"
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    contrarian_signals: List[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    llm_cache_hit: bool = False
    llm_analysis_profile: Optional[str] = None


class LLMAnalysisService:
    """
    Enhanced trading analysis using Large Language Models.
    
    Combines technical analysis, sentiment data, and market context
    to provide human-like reasoning for trading decisions.
    """
    
    def __init__(
        self,
        provider: LLMProvider = None,
        *,
        model_name_override: Optional[str] = None,
        thinking_type: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize LLM service."""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()

        # Auto-detect provider from environment if not specified
        if provider is None:
            provider_str = os.getenv('LLM_PROVIDER', 'deepseek').lower()
            try:
                provider = LLMProvider(provider_str)
                logger.info(f"Using LLM provider: {provider_str}")
            except ValueError:
                logger.warning(f"Unknown LLM_PROVIDER '{provider_str}', defaulting to DeepSeek")
                provider = LLMProvider.DEEPSEEK

        self.provider = provider
        self.client = None
        normalized_thinking = str(thinking_type or "").strip().lower()
        self.default_thinking_type = (
            normalized_thinking if normalized_thinking in {"enabled", "disabled"} else None
        )
        normalized_effort = str(reasoning_effort or "").strip().lower()
        self.default_reasoning_effort = (
            normalized_effort if normalized_effort in {"low", "medium", "high", "xhigh", "max"} else None
        )

        # Initialize client based on provider
        if provider == LLMProvider.CLAUDE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key and anthropic:
                try:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Claude client initialized successfully")
                except TypeError as e:
                    # Handle version compatibility issue
                    logger.warning(f"Claude client initialization failed: {e}")
                    logger.info("Attempting fallback initialization...")
                    try:
                        self.client = anthropic.Client(api_key=api_key)
                        logger.info("Claude client initialized with fallback method")
                    except Exception as fallback_error:
                        logger.error(f"Claude client fallback failed: {fallback_error}")
                        self.client = None
            else:
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY environment variable not found")
                elif not anthropic:
                    logger.warning("anthropic library not installed")
                else:
                    logger.warning("Unknown issue with Claude initialization")

        elif provider == LLMProvider.DEEPSEEK:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key and openai:
                try:
                    self.client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepseek.com"
                    )
                    logger.info("DeepSeek client initialized successfully")
                except Exception as e:
                    logger.error(f"DeepSeek client initialization failed: {e}")
                    self.client = None
            else:
                if not api_key:
                    logger.warning("DEEPSEEK_API_KEY environment variable not found")
                elif not openai:
                    logger.warning("openai library not installed")
                else:
                    logger.warning("Unknown issue with DeepSeek initialization")

        elif provider == LLMProvider.OPENAI:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and openai:
                try:
                    self.client = openai.OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"OpenAI client initialization failed: {e}")
                    self.client = None
            else:
                if not api_key:
                    logger.warning("OPENAI_API_KEY environment variable not found")
                elif not openai:
                    logger.warning("openai library not installed")
                else:
                    logger.warning("Unknown issue with OpenAI initialization")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Analysis configuration - use environment variables with provider-specific defaults
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '8000'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.analysis_cache_ttl = max(0, int(os.getenv('TOKEN_ANALYSIS_LLM_CACHE_SECONDS', '90') or '90'))
        self.analysis_cache: Dict[str, tuple[float, LLMAnalysis]] = {}

        # Model name from environment or provider-specific default
        if provider == LLMProvider.CLAUDE:
            self.max_context_length = 8000  # Claude context limit
            # claude-3-haiku-20240307 is deprecated (retires 2026-04-19) and weak for
            # nuanced trade judgment. Default to a current, cost-appropriate model that
            # still accepts the sampling params this service sends.
            self.model_name = os.getenv('LLM_MODEL', 'claude-haiku-4-5')
        elif provider == LLMProvider.DEEPSEEK:
            self.max_context_length = 32000  # DeepSeek context limit
            configured_model = os.getenv('LLM_MODEL', 'deepseek-v4-pro')
            if not configured_model.lower().startswith('deepseek'):
                logger.warning(
                    "Ignoring non-DeepSeek LLM_MODEL '%s' for DeepSeek provider; using deepseek-v4-pro",
                    configured_model
                )
                configured_model = 'deepseek-v4-pro'
            self.model_name = configured_model
        elif provider == LLMProvider.OPENAI:
            self.max_context_length = 128000  # GPT class models support large contexts
            self.model_name = os.getenv('LLM_MODEL', 'gpt-5.1')
        if model_name_override:
            override_name = str(model_name_override).strip()
            if provider == LLMProvider.DEEPSEEK and not override_name.lower().startswith("deepseek"):
                logger.warning(
                    "Ignoring non-DeepSeek override model '%s' for DeepSeek provider; keeping %s",
                    override_name,
                    self.model_name,
                )
            else:
                self.model_name = override_name
    
    def safe_float(self, value, fallback=0.0):
        """Safely convert a value to float, handling dicts and None values."""
        try:
            if isinstance(value, dict):
                return float(value.get('value', fallback))
            return float(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def _extract_token_usage(self, response: Any) -> tuple[int, int]:
        """Extract prompt/completion token counts from provider responses."""
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0

        prompt_tokens = 0
        completion_tokens = 0
        try:
            if isinstance(usage, dict):
                prompt_tokens = int(
                    usage.get("prompt_tokens")
                    or usage.get("input_tokens")
                    or 0
                )
                completion_tokens = int(
                    usage.get("completion_tokens")
                    or usage.get("output_tokens")
                    or 0
                )
            else:
                prompt_tokens = int(
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "input_tokens", 0)
                    or 0
                )
                completion_tokens = int(
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", 0)
                    or 0
                )
        except (TypeError, ValueError):
            return 0, 0

        return max(prompt_tokens, 0), max(completion_tokens, 0)

    def _estimate_llm_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate request cost using the primary-model pricing knobs."""
        try:
            input_cost_per_1m = float(os.getenv("PRIMARY_LLM_INPUT_COST_PER_1M", "0") or 0.0)
            output_cost_per_1m = float(os.getenv("PRIMARY_LLM_OUTPUT_COST_PER_1M", "0") or 0.0)
        except (TypeError, ValueError):
            return 0.0

        if input_cost_per_1m <= 0 and output_cost_per_1m <= 0:
            return 0.0

        estimate = (
            (max(int(prompt_tokens or 0), 0) / 1_000_000.0) * max(input_cost_per_1m, 0.0)
            + (max(int(completion_tokens or 0), 0) / 1_000_000.0) * max(output_cost_per_1m, 0.0)
        )
        return round(max(estimate, 0.0), 8)

    @staticmethod
    def _merge_usage(*telemetry_items: Optional[LLMUsageTelemetry]) -> LLMUsageTelemetry:
        """Accumulate usage across retries while preserving provider/model metadata."""
        merged = LLMUsageTelemetry(provider="", model="")
        for telemetry in telemetry_items:
            if not telemetry:
                continue
            if telemetry.provider:
                merged.provider = telemetry.provider
            if telemetry.model:
                merged.model = telemetry.model
            if telemetry.analysis_profile:
                merged.analysis_profile = telemetry.analysis_profile
            merged.prompt_tokens += max(int(telemetry.prompt_tokens or 0), 0)
            merged.completion_tokens += max(int(telemetry.completion_tokens or 0), 0)
            merged.estimated_cost_usd = round(
                merged.estimated_cost_usd + max(float(telemetry.estimated_cost_usd or 0.0), 0.0),
                8,
            )
            merged.cache_hit = merged.cache_hit or bool(telemetry.cache_hit)
        return merged

    def _apply_usage_telemetry(
        self,
        analysis: LLMAnalysis,
        telemetry: Optional[LLMUsageTelemetry],
        *,
        analysis_profile: str,
        cache_hit: bool,
    ) -> LLMAnalysis:
        """Attach normalized request telemetry to an analysis object."""
        provider = str((telemetry.provider if telemetry else self.provider.value) or self.provider.value)
        model = str((telemetry.model if telemetry else self.model_name) or self.model_name)
        analysis.llm_provider = provider
        analysis.llm_model = model
        analysis.llm_prompt_tokens = max(int(getattr(telemetry, "prompt_tokens", 0) or 0), 0) if not cache_hit else 0
        analysis.llm_completion_tokens = max(int(getattr(telemetry, "completion_tokens", 0) or 0), 0) if not cache_hit else 0
        analysis.llm_estimated_cost_usd = round(
            max(float(getattr(telemetry, "estimated_cost_usd", 0.0) or 0.0), 0.0) if not cache_hit else 0.0,
            8,
        )
        analysis.llm_cache_hit = cache_hit
        analysis.llm_analysis_profile = analysis_profile
        return analysis
        
    def _looks_truncated(self, response: str) -> bool:
        """Detect a response that was cut off mid-JSON (e.g. hit max_tokens)."""
        text = (response or "").strip()
        if not text:
            return True
        json_start = text.find('{')
        if json_start < 0:
            return False  # not JSON-shaped; freeform parsing handles this
        depth = 0
        in_string = False
        escape = False
        for ch in text[json_start:]:
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        return depth != 0

    async def analyze_trading_opportunity(
        self,
        context: MarketContext,
        additional_context: str = "",
        max_output_tokens: Optional[int] = None,
        strict: bool = False,
        analysis_profile: str = "trading",
        thinking_type: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMAnalysis:
        """
        Analyze a trading opportunity using LLM reasoning.

        Args:
            context: Market context and technical data
            additional_context: Additional context like news, events
            strict: When True, never degrade to a generic fallback/freeform
                analysis. A truncated JSON response is retried once; if it's
                still incomplete, or the LLM call otherwise fails, the
                exception propagates to the caller instead of being hidden
                behind a generic result.

        Returns:
            LLM analysis with recommendation and reasoning
        """
        try:
            # Check if LLM client is available
            if not self.client:
                if strict:
                    raise RuntimeError("LLM client not available")
                logger.warning("LLM client not available, using fallback analysis")
                return self._fallback_analysis(context)

            # Prepare prompt for LLM
            prompt = self._build_analysis_prompt(
                context,
                additional_context,
                analysis_profile=analysis_profile,
            )
            cache_material = (
                f"{self.provider.value}:{self.model_name}:{analysis_profile}:{context.symbol}:"
                f"{context.current_price:.10g}:{context.price_change_24h:.4f}:{context.rsi:.2f}:"
                f"{hashlib.sha256(additional_context.encode('utf-8')).hexdigest()}"
            )
            cache_key = hashlib.sha256(cache_material.encode('utf-8')).hexdigest()
            cached = self.analysis_cache.get(cache_key)
            if cached and time.monotonic() - cached[0] <= self.analysis_cache_ttl:
                logger.info("Using recent token-analysis LLM result for %s", context.symbol)
                return self._apply_usage_telemetry(
                    copy.deepcopy(cached[1]),
                    None,
                    analysis_profile=analysis_profile,
                    cache_hit=True,
                )
            logger.info("Performing LLM analysis for %s", context.symbol)

            # Get LLM response
            response, usage_telemetry = await self._query_llm(
                prompt,
                max_output_tokens=max_output_tokens,
                thinking_type=thinking_type,
                reasoning_effort=reasoning_effort,
            )

            if strict and self._looks_truncated(response):
                retry_tokens = min(self.max_tokens, (max_output_tokens or self.max_tokens) + 2000)
                logger.warning(
                    "LLM response for %s looked truncated (hit token limit?); retrying with %d tokens",
                    context.symbol,
                    retry_tokens,
                )
                retry_response, retry_usage = await self._query_llm(prompt, max_output_tokens=retry_tokens)
                response = retry_response
                usage_telemetry = self._merge_usage(usage_telemetry, retry_usage)
                if self._looks_truncated(response):
                    raise RuntimeError(
                        f"LLM response for {context.symbol} was truncated twice; refusing to return a partial analysis"
                    )

            # Parse LLM response into structured analysis
            analysis = self._parse_llm_response(response, context, strict=strict)
            analysis = self._apply_usage_telemetry(
                analysis,
                usage_telemetry,
                analysis_profile=analysis_profile,
                cache_hit=False,
            )
            if self.analysis_cache_ttl > 0:
                self.analysis_cache[cache_key] = (time.monotonic(), copy.deepcopy(analysis))
                if len(self.analysis_cache) > 256:
                    oldest_key = min(self.analysis_cache, key=lambda key: self.analysis_cache[key][0])
                    self.analysis_cache.pop(oldest_key, None)

            logger.debug(f"LLM analysis completed for {context.symbol}: {analysis.recommendation} "
                       f"(confidence: {analysis.confidence:.2f}) - Action: {analysis.action_summary}")

            return analysis

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            if strict:
                raise
            return self._fallback_analysis(context)

    async def analyze_trading_opportunity_stream(
        self,
        context: MarketContext,
        additional_context: str = "",
        max_output_tokens: Optional[int] = None,
        strict: bool = True,
        analysis_profile: str = "trading",
        thinking_type: Optional[str] = "enabled",
        reasoning_effort: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream LLM trading analysis with real-time reasoning and content chunks.
        Yields events:
          {"type": "status", "message": "..."}
          {"type": "thinking", "delta": "..."}
          {"type": "content", "delta": "..."}
          {"type": "complete", "analysis": LLMAnalysis}
        """
        if not self.client:
            if strict:
                raise RuntimeError("LLM client not available")
            yield {"type": "complete", "analysis": self._fallback_analysis(context)}
            return

        yield {"type": "status", "message": f"Building market structure analysis for {context.symbol}..."}

        prompt = self._build_analysis_prompt(
            context,
            additional_context,
            analysis_profile=analysis_profile,
        )

        output_tokens = self.max_tokens
        if max_output_tokens is not None:
            output_tokens = max(256, min(self.max_tokens, int(max_output_tokens)))
        selected_model = str(model_name or self.model_name or "").strip() or self.model_name

        effective_thinking = (
            str(thinking_type).strip().lower()
            if thinking_type is not None
            else self.default_thinking_type
        )
        effective_effort = (
            str(reasoning_effort).strip().lower()
            if reasoning_effort is not None
            else self.default_reasoning_effort
        )

        yield {"type": "status", "message": f"Engaging AI reasoning engine ({selected_model})..."}

        full_content_chunks: List[str] = []
        full_reasoning_chunks: List[str] = []
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        if self.provider in {LLMProvider.DEEPSEEK, LLMProvider.OPENAI}:
            request_kwargs: Dict[str, Any] = {
                "model": selected_model,
                "max_tokens": output_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            if self.provider == LLMProvider.DEEPSEEK:
                extra_body: Dict[str, Any] = {}
                if effective_thinking in {"enabled", "disabled"}:
                    extra_body["thinking"] = {"type": effective_thinking}
                if effective_effort in {"low", "medium", "high", "xhigh", "max"}:
                    extra_body["reasoning_effort"] = effective_effort
                if extra_body:
                    request_kwargs["extra_body"] = extra_body

            def stream_worker():
                try:
                    stream = self.client.chat.completions.create(**request_kwargs)
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        reasoning = getattr(delta, 'reasoning_content', None) or ""
                        content = delta.content or ""
                        if reasoning or content:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("data", (reasoning, content))), loop
                            )
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop)

            threading.Thread(target=stream_worker, daemon=True).start()

        elif self.provider == LLMProvider.CLAUDE:
            def claude_stream_worker():
                try:
                    claude_kwargs = {
                        "model": selected_model,
                        "max_tokens": output_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    model_lower = selected_model.lower()
                    rejects_sampling = any(
                        marker in model_lower
                        for marker in ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-5", "sonnet-4-6", "fable", "mythos")
                    )
                    if not rejects_sampling:
                        claude_kwargs["temperature"] = self.temperature

                    with self.client.messages.stream(**claude_kwargs) as stream:
                        for text in stream.text_stream:
                            if text:
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("data", ("", text))), loop
                                )
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop)

            threading.Thread(target=claude_stream_worker, daemon=True).start()

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # Consume stream queue and yield SSE events
        while True:
            msg_type, payload = await queue.get()
            if msg_type == "data":
                reasoning, content = payload
                if reasoning:
                    full_reasoning_chunks.append(reasoning)
                    yield {"type": "thinking", "delta": reasoning}
                if content:
                    full_content_chunks.append(content)
                    yield {"type": "content", "delta": content}
            elif msg_type == "done":
                break
            elif msg_type == "error":
                logger.error(f"Streaming error for {context.symbol}: {payload}")
                if strict:
                    raise payload
                yield {"type": "complete", "analysis": self._fallback_analysis(context)}
                return

        # Assemble and parse complete response
        raw_response = "".join(full_content_chunks)
        if not raw_response.strip() and full_reasoning_chunks:
            raw_response = "".join(full_reasoning_chunks)

        yield {"type": "status", "message": f"Structuring trade decision for {context.symbol}..."}

        analysis = self._parse_llm_response(raw_response, context, strict=strict)
        
        # Telemetry approximation for stream
        telemetry = LLMUsageTelemetry(
            provider=self.provider.value,
            model=selected_model,
            prompt_tokens=int(len(prompt) / 4),
            completion_tokens=int((len(raw_response) + sum(len(r) for r in full_reasoning_chunks)) / 4),
            estimated_cost_usd=0.001,
        )
        analysis = self._apply_usage_telemetry(
            analysis,
            telemetry,
            analysis_profile=analysis_profile,
            cache_hit=False,
        )

        yield {"type": "complete", "analysis": analysis}

    def _build_analysis_prompt(
        self,
        context: MarketContext,
        additional_context: str = "",
        analysis_profile: str = "trading",
    ) -> str:
        """Build comprehensive prompt for LLM analysis."""
        
        # Safe accessor functions for technical indicators
        def safe_numeric_value(value, default=0.0):
            """Safely extract numeric value from value that might be dict or numeric."""
            if isinstance(value, dict):
                return float(value.get('value', default))
            return float(value) if value is not None else default
        
        def safe_macd_value(macd_data):
            """Safely extract MACD value from dict or float."""
            try:
                if isinstance(macd_data, dict):
                    # Try multiple possible keys
                    for key in ['macd', 'value', 'signal', 'line']:
                        if key in macd_data:
                            return float(macd_data[key])
                    return 0.0  # Default if no valid key found
                elif macd_data is not None:
                    return float(macd_data)
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0
        
        def safe_fear_greed_value(fg_data):
            """Safely extract fear_greed_index value from dict or int."""
            try:
                if isinstance(fg_data, dict):
                    # Try multiple possible keys
                    for key in ['fear_greed_index', 'value', 'index', 'score']:
                        if key in fg_data:
                            return int(fg_data[key])
                    return 50  # Default if no valid key found
                elif fg_data is not None:
                    return int(fg_data)
                else:
                    return 50
            except (ValueError, TypeError):
                return 50

        def safe_percent_fraction(percent_data):
            """Normalize percent-point inputs into a fraction for percent formatting."""
            try:
                if percent_data is None:
                    return None
                numeric = float(percent_data)
                if not numeric:
                    return None
                if abs(numeric) > 1:
                    numeric = numeric / 100.0
                return numeric
            except (ValueError, TypeError):
                return None

        # Calculate technical signals with safety checks
        rsi_signal = "Oversold" if context.rsi < 30 else "Overbought" if context.rsi > 70 else "Neutral"
        macd_value = safe_macd_value(context.macd)
        macd_signal = "Bullish" if macd_value > 0 else "Bearish"
        bb_signal = "Oversold" if context.bb_position < 0.2 else "Overbought" if context.bb_position > 0.8 else "Neutral"
        
        # Safely extract fear_greed_index value
        fear_greed_value = safe_fear_greed_value(context.fear_greed_index)
        
        # Determine market sentiment
        if fear_greed_value < 25:
            fear_greed_sentiment = "Extreme Fear (potential buying opportunity)"
        elif fear_greed_value > 75:
            fear_greed_sentiment = "Extreme Greed (potential selling opportunity)"
        else:
            fear_greed_sentiment = f"Neutral sentiment (index: {fear_greed_value})"
        open_interest_change_fraction = safe_percent_fraction(context.open_interest_change)

        # Only surface sentiment/narrative that carries real, distinct information.
        # Empty/neutral placeholders (and social==news duplication) are dropped so the
        # model does not weigh — or hallucinate around — signals that were never provided.
        def _has_real_sentiment(value: Any) -> bool:
            return (
                isinstance(value, str)
                and value.strip().lower() not in {"", "neutral", "unavailable", "unknown", "none"}
            )

        sentiment_lines = []
        if fear_greed_value != 50:
            sentiment_lines.append(f"- Fear & Greed Index: {fear_greed_sentiment}")
        if _has_real_sentiment(context.social_sentiment):
            sentiment_lines.append(f"- Social Sentiment: {context.social_sentiment}")
        if _has_real_sentiment(context.news_sentiment) and context.news_sentiment != context.social_sentiment:
            sentiment_lines.append(f"- News Sentiment: {context.news_sentiment}")
        sentiment_block = (
            "SENTIMENT ANALYSIS:\n" + "\n".join(sentiment_lines) + "\n\n"
            if sentiment_lines
            else ""
        )
        narrative_line = (
            f"- Market Narrative: {context.market_narrative}\n"
            if _has_real_sentiment(context.market_narrative)
            else ""
        )

        spot_investment_profile = analysis_profile == "spot_investment"
        analyst_role = (
            f"You are Ryu, a spot-portfolio analyst evaluating {context.symbol} as an unleveraged token holding."
            if spot_investment_profile
            else f"You are Ryu, an expert cryptocurrency futures and spot trader analyzing {context.symbol} for potential trading opportunities."
        )
        direction_guidance = (
            "Recommend BUY to add a spot holding, SELL to reduce or exit spot exposure, or HOLD to wait. Never describe a short, futures, perpetual, leveraged, or long-position setup."
            if spot_investment_profile
            else "You are equally comfortable recommending LONG (BUY) or SHORT (SELL) setups. Your analysis is objective and data-first — you do not have a directional bias."
        )
        spot_investment_override = """
SPOT INVESTMENT EXPLANATION — THIS OVERRIDES THE GENERIC TRADER-NOTE INSTRUCTIONS ABOVE:
- The reasoning field is the user-facing answer to “Why Ryu bought this token?”
- Lead with what the token or project does, its category/use case, and the verified fundamental traits that made it eligible for Ryu’s spot portfolio.
- Explain why those token-specific traits, market size, liquidity, adoption or activity, and risk/reward justified allocating capital.
- Technical timing may appear in at most one supporting sentence. Do not lead with RSI, EMAs, Bollinger Bands, supply/demand zones, order flow, invalidation, targets, or reward-to-risk.
- Describe an unleveraged purchase and holding thesis. Never call it a long, short, futures position, technical trade, or leveraged setup.
- If verified project/fundamental context is sparse, say so plainly and explain the measurable selection factors Ryu did have instead of inventing a token utility or narrative.
- Keep the reasoning to 3 concise sentences suitable for a portfolio holding detail view.
""" if spot_investment_profile else ""
        closing_reminder = (
            "Remember: Users need a clear token-specific purchase thesis. Keep technical execution details in the structured entry, target, and risk fields rather than the reasoning."
            if spot_investment_profile
            else "Remember: Users need clear, actionable guidance. Be specific about entry, targets, and stops."
        )

        prompt = f"""
{analyst_role}
{direction_guidance}
Provide a comprehensive analysis combining technical, fundamental, sentiment, order-flow, and market-structure factors when those real data fields are provided.

MARKET DATA:
- Symbol: {context.symbol}
- Current Price: ${context.current_price:,.2f}
- 24h Change: {context.price_change_24h:+.2f}%
- 24h Volume: ${context.volume_24h:,.0f}
- Volatility: {context.volatility:.2%}

TECHNICAL INDICATORS:
- RSI (14): {context.rsi:.1f} ({rsi_signal})
- MACD: {macd_value:.4f} ({macd_signal})
- Bollinger Band Position: {context.bb_position:.2f} ({bb_signal})

{sentiment_block}MARKET CONTEXT:
- Market Regime: {context.market_regime.upper()} (Current macro market environment)
{f"- Market Cap: ${context.market_cap:,.0f}" if context.market_cap else ""}
{f"- Funding Rate: {context.funding_rate:.4%}" if context.funding_rate else ""}
{f"- Open Interest Change: {open_interest_change_fraction:+.2%}" if open_interest_change_fraction is not None else ""}
{narrative_line}

{"ADDITIONAL CONTEXT:" if additional_context else ""}
{additional_context if additional_context else ""}

Choose exactly one recommendation: BUY, SELL, or HOLD.
BUY means go long (spot accumulate or leveraged long) when evidence supports upside. SELL means go short or exit/reduce existing position when evidence supports downside. HOLD means the setup is unclear or risk/reward is not favorable — wait for confirmation.
Return ONLY valid JSON in exactly the shape below — same keys, same nesting, valid JSON syntax. The values shown are ILLUSTRATIVE placeholders describing what belongs in each field: never copy them verbatim, and derive every value from THIS token's data. Do not default to HOLD or to any particular confidence — choose the recommendation, confidence, and levels the evidence actually supports.
{{
    "recommendation": "BUY, SELL, or HOLD (choose what the evidence supports)",
    "confidence": 0.6,
    "action_summary": "One line that begins with your chosen recommendation",
    "reasoning": "A detailed 3-5 sentence desk note: name the decisive levels (support/resistance, EMAs), the RSI/MACD/structure read, order flow or liquidity, and the exact invalidation. Write it like a trader briefing a colleague — concrete and specific, never a generic recap.",
    "key_factors": ["Specific structural driver with a price level", "Specific risk or invalidation with a level", "Specific momentum, order-flow, or regime driver"],
    "risk_assessment": "LOW, MEDIUM, or HIGH",
    "time_horizon": "1-7 days",
    "entry_strategy": {{
        "optimal_entry": {context.current_price:.6g},
        "entry_range_low": {context.current_price * 0.98:.6g},
        "entry_range_high": {context.current_price * 1.02:.6g},
        "market_order_ok": true
    }},
    "price_targets": {{
        "target_1": null,
        "target_2": null,
        "target_3": null
    }},
    "risk_management": {{
        "stop_loss": null,
        "position_size": "1-2% portfolio risk",
        "max_leverage": "3x max"
    }},
    "market_regime": "BULL, BEAR, or UNCERTAIN"
}}

ANALYSIS GUIDELINES:
1. Use the Market Regime to guide your analysis:
   - BULL_MARKET: Favor LONG/BUY setups with higher confidence for clean upside breakouts and strong altcoin momentum
   - BEAR_MARKET: Favor SHORT/SELL positions, lower confidence for LONG trades, expect BTC dominance and altcoin weakness; call genuine SHORT setups when downside evidence is clear
   - SIDEWAYS: More conservative, require stronger technical confirmation, expect range-bound trading
2. Make recommendations actionable and specific
3. Provide exact price levels based on current price of ${context.current_price:,.2f}
4. Consider realistic profit targets (3-15% moves typically) within the current market regime
5. Set stop losses 2-8% from entry depending on volatility
6. Give practical execution advice for retail traders
7. Keep language simple and avoid jargon
8. Focus on risk-first approach
9. Keep direction consistent: BUY reasoning and targets should be bullish/upside; SELL reasoning should be bearish/downside or spot exit-risk and must not say BUY unless explaining a rejected alternative; HOLD should avoid entry/target claims.
10. Report your genuine directional conviction based on the data — do not self-censor a BUY or SELL merely because you lack 100% certainty. If two or more key indicators agree (trend + momentum, or structure + order flow), that is sufficient for a directional call. Use HOLD only when signals are genuinely contradictory or the setup offers poor risk/reward — not as a default safe answer.
11. For a SELL short setup, targets belong below current price and stop loss belongs above current price. For SELL exit/avoid analysis, use the entry_strategy fields as a reference exit/avoid range and set price_targets to null.
12. Always provide concrete position_size and max_leverage values. Never return "AI-determined" or vague placeholder wording. Size leverage to your own conviction: higher confidence and a cleaner risk/reward justify higher leverage; lower conviction means little or none. Use "Spot only" when you would not use leverage. Express leverage as "Nx max" (e.g. "3x max", "10x max").
13. The reasoning must sound like a human market note: mention the decisive data and levels, avoid score-only recaps, and do not describe internal rule/LLM blending unless it directly affects the trade.
14. The recommendation, action_summary, reasoning, targets, and stop must express one consistent decision. If the conclusion says to wait, that risk/reward is poor, or that neither long nor short is attractive, recommendation must be HOLD.
15. Stay directionally neutral: use BUY for a supported long, SELL for a supported short or spot exit, and HOLD when neither direction clears the evidence threshold. Never favor BUY merely because the broad regime is bullish or SELL merely because it is bearish.
16. Be thorough. The reasoning must be at least three full sentences that cite concrete numbers from the data above, and every key_factor must name a specific level, indicator, or flow — never generic labels like "technical analysis" or "market conditions". A HOLD still needs detailed reasoning explaining what is missing.

{spot_investment_override}
{closing_reminder}
"""
        
        return prompt
    
    async def analyze_trading_context(self, context: Dict[str, Any]) -> Optional['LLMAnalysis']:
        """
        Analyze trading context for opportunity enhancement (similar to technical process flow).
        
        Args:
            context: Trading context dictionary with symbol, indicators, etc.
            
        Returns:
            LLM analysis with confidence adjustments and risk factors
        """
        try:
            # Check if LLM client is available
            if not self.client:
                logger.warning("LLM client not available for trading context analysis")
                return None
            
            logger.info(f"🧠 Running AI context analysis for {context.get('symbol', 'unknown')}")
            
            # Convert context dict to MarketContext for consistency with existing flow
            market_context = MarketContext(
                timestamp=datetime.now(),
                symbol=context.get('symbol', 'UNKNOWN'),
                current_price=context.get('market_context', {}).get('price', 0.0),
                price_change_24h=context.get('market_context', {}).get('price_change_24h', 0.0),
                volume_24h=context.get('market_context', {}).get('volume_24h', 0.0),
                volatility=0.05,  # Default
                
                # Technical indicators
                rsi=context.get('technical_indicators', {}).get('rsi', 50.0),
                macd=context.get('technical_indicators', {}).get('macd', 0.0),
                bb_position=context.get('technical_indicators', {}).get('bollinger_position', 0.5),
                
                # Default sentiment values
                fear_greed_index=50,
                social_sentiment="neutral",
                news_sentiment="neutral",
                market_regime="sideways",
                market_narrative="Market analysis in progress"
            )
            
            # Build simplified prompt for context enhancement
            prompt = self._build_context_analysis_prompt(context, market_context)
            
            # Get LLM response
            response, _ = await self._query_llm(prompt)
            
            # Parse response into analysis
            analysis = self._parse_context_response(response, context)
            
            logger.info(f"🧠 AI context analysis completed for {context.get('symbol', 'unknown')}: confidence_score={analysis.confidence:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in trading context analysis: {e}")
            return None
    
    def _build_context_analysis_prompt(self, context: Dict[str, Any], market_context: MarketContext) -> str:
        """Build prompt for trading context analysis (similar to technical flow)."""
        
        def safe_macd_value(macd_data):
            """Safely extract MACD value from dict or float."""
            try:
                if isinstance(macd_data, dict):
                    # Try multiple possible keys
                    for key in ['macd', 'value', 'signal', 'line']:
                        if key in macd_data:
                            return float(macd_data[key])
                    return 0.0  # Default if no valid key found
                elif macd_data is not None:
                    return float(macd_data)
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0

        def safe_fear_greed_value(fg_data):
            """Safely extract fear_greed_index value from dict or int."""
            try:
                if isinstance(fg_data, dict):
                    # Try multiple possible keys
                    for key in ['fear_greed_index', 'value', 'index', 'score']:
                        if key in fg_data:
                            return int(fg_data[key])
                    return 50  # Default if no valid key found
                elif fg_data is not None:
                    return int(fg_data)
                else:
                    return 50
            except (ValueError, TypeError):
                return 50
        
        symbol = context.get('symbol', 'UNKNOWN')
        direction = context.get('direction', 'UNKNOWN')
        confidence = context.get('confidence', 0.5)
        
        prompt = f"""
You are an expert cryptocurrency trader providing quick confidence assessment and risk analysis for {symbol}.

CURRENT OPPORTUNITY:
- Symbol: {symbol}
- Direction: {direction}
- Mathematical Confidence: {confidence:.3f}
- Current Price: ${market_context.current_price:,.2f}
- 24h Change: {market_context.price_change_24h:+.2f}%

TECHNICAL INDICATORS:
- RSI: {market_context.rsi:.1f}
- MACD: {safe_macd_value(market_context.macd):.4f}
- Bollinger Position: {context.get('technical_indicators', {}).get('bollinger_position', 0.5):.2f}

Provide a quick assessment in this exact JSON format:
{{
    "confidence_score": 0.75,
    "risk_factors": ["Factor 1", "Factor 2"],
    "enhancement_reasoning": "Brief explanation of confidence adjustment"
}}

Guidelines:
1. confidence_score should be 0.3-0.9 (will be used to adjust mathematical confidence by ±5%)
2. risk_factors should be max 2 most important concerns
3. enhancement_reasoning should be 1-2 sentences explaining your assessment
4. Consider the mathematical confidence of {confidence:.3f} and whether technical setup supports it
5. Focus on immediate risk/reward for this specific setup
"""
        
        return prompt
    
    def _parse_context_response(self, response: str, context: Dict[str, Any]) -> 'LLMAnalysis':
        """Parse AI context response into analysis object."""
        try:
            # Try to parse JSON response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                # Clean JSON string
                json_str = self._clean_json_string(json_str)
                
                data = json.loads(json_str)
                
                # Validate that this isn't a generic/example response
                reasoning = data.get('enhancement_reasoning', 'AI analysis completed')
                generic_phrases = [
                    'technical indicators show oversold conditions',
                    'fundamentals remain strong',
                    'current price offers good risk-reward',
                    'Provide specific analysis based on actual',
                    'AI analysis completed',
                    'Technical analysis fallback'
                ]
                
                if any(phrase.lower() in reasoning.lower() for phrase in generic_phrases):
                    logger.warning(f"Detected generic AI response, filtering out: {reasoning}")
                    return None
                
                confidence_score = float(data.get('confidence_score', 0.5))
                risk_factors = data.get('risk_factors', [])
                
                # Also validate risk factors aren't generic
                if risk_factors and any('Technical analysis' in factor for factor in risk_factors):
                    logger.warning("Detected generic risk factors, filtering out")
                    return None
                
                # Create analysis object compatible with existing flow
                analysis = LLMAnalysis(
                    recommendation='HOLD',  # Not used in context analysis
                    confidence=confidence_score,
                    action_summary=reasoning,
                    reasoning=reasoning,
                    key_factors=risk_factors[:2],  # Limit to 2 factors
                    risk_assessment='MEDIUM',
                    time_horizon='medium'
                )
                
                # Store risk factors for enhancement
                analysis.risk_factors = risk_factors[:2]
                
                return analysis
                
        except Exception as e:
            logger.warning(f"Failed to parse AI context response: {e}")
        
        # No fallback analysis - return None if AI analysis fails
        return None
    
    async def _query_llm(
        self,
        prompt: str,
        max_output_tokens: Optional[int] = None,
        *,
        model_name: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        thinking_type: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> tuple[str, LLMUsageTelemetry]:
        """Query the configured LLM with the analysis prompt."""
        try:
            if not self.client:
                raise ValueError("LLM client not initialized")

            output_tokens = self.max_tokens
            if max_output_tokens is not None:
                output_tokens = max(256, min(self.max_tokens, int(max_output_tokens)))
            selected_model = str(model_name or self.model_name or "").strip() or self.model_name

            if self.provider == LLMProvider.CLAUDE:
                claude_kwargs = {
                    "model": selected_model,
                    "max_tokens": output_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # Sampling params were removed on Opus 4.6+/Sonnet 5/Fable and return a
                # 400. Only send temperature to models that still accept it.
                model_lower = self.model_name.lower()
                rejects_sampling = any(
                    marker in model_lower
                    for marker in ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-5", "sonnet-4-6", "fable", "mythos")
                )
                if not rejects_sampling:
                    claude_kwargs["temperature"] = self.temperature
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    **claude_kwargs,
                )
                prompt_tokens, completion_tokens = self._extract_token_usage(response)
                telemetry = LLMUsageTelemetry(
                    provider=self.provider.value,
                    model=selected_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=self._estimate_llm_cost_usd(prompt_tokens, completion_tokens),
                )
                if response.content and len(response.content) > 0:
                    return response.content[0].text or "", telemetry
                return "", telemetry

            elif self.provider in {LLMProvider.DEEPSEEK, LLMProvider.OPENAI}:
                request_kwargs: Dict[str, Any] = {
                    "model": selected_model,
                    "max_tokens": output_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if response_format:
                    request_kwargs["response_format"] = response_format
                effective_thinking = (
                    str(thinking_type).strip().lower()
                    if thinking_type is not None
                    else self.default_thinking_type
                )
                effective_effort = (
                    str(reasoning_effort).strip().lower()
                    if reasoning_effort is not None
                    else self.default_reasoning_effort
                )
                if self.provider == LLMProvider.DEEPSEEK:
                    extra_body: Dict[str, Any] = {}
                    if effective_thinking in {"enabled", "disabled"}:
                        extra_body["thinking"] = {"type": effective_thinking}
                    if effective_effort in {"low", "medium", "high", "xhigh", "max"}:
                        extra_body["reasoning_effort"] = effective_effort
                    if extra_body:
                        request_kwargs["extra_body"] = extra_body
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    **request_kwargs,
                )
                prompt_tokens, completion_tokens = self._extract_token_usage(response)
                telemetry = LLMUsageTelemetry(
                    provider=self.provider.value,
                    model=selected_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=self._estimate_llm_cost_usd(prompt_tokens, completion_tokens),
                )
                msg = response.choices[0].message
                content = msg.content or ""
                if not content and hasattr(msg, "reasoning_content") and getattr(msg, "reasoning_content", None):
                    content = str(getattr(msg, "reasoning_content") or "")
                return content, telemetry

            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

        except Exception as e:
            logger.error(f"Error querying {self.provider.value} LLM: {e}")
            raise
    
    def _parse_llm_response(self, response: str, context: MarketContext, strict: bool = False) -> LLMAnalysis:
        """Parse LLM response into structured analysis."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]

                # Clean common JSON issues
                json_str = self._clean_json_string(json_str)

                # Attempt JSON parsing with multiple fallback strategies
                analysis_data = self._parse_json_with_fallbacks(json_str)
            elif strict:
                raise ValueError("LLM response did not contain a JSON object")
            else:
                # Fallback: parse free-form response
                return self._parse_freeform_response(response, context)

            if strict and not analysis_data.get('recommendation') and not analysis_data.get('action_summary'):
                raise ValueError("LLM response JSON parsed but yielded no usable fields")

            # Parse nested structures
            entry_data = analysis_data.get('entry_strategy', {})
            targets_data = analysis_data.get('price_targets', {})
            risk_data = analysis_data.get('risk_management', {})
            
            # Create structured objects with null-safe float conversion
            
            entry_strategy = None
            if entry_data:
                entry_strategy = EntryStrategy(
                    optimal_entry=self.safe_float(entry_data.get('optimal_entry'), context.current_price),
                    entry_range_low=self.safe_float(entry_data.get('entry_range_low'), context.current_price * 0.98),
                    entry_range_high=self.safe_float(entry_data.get('entry_range_high'), context.current_price * 1.02),
                    market_order_ok=bool(entry_data.get('market_order_ok', False))
                )
            
            price_targets = None
            if targets_data:
                price_targets = PriceTargets(
                    target_1=self.safe_float(targets_data.get('target_1')) if targets_data.get('target_1') is not None else None,
                    target_2=self.safe_float(targets_data.get('target_2')) if targets_data.get('target_2') is not None else None,
                    target_3=self.safe_float(targets_data.get('target_3')) if targets_data.get('target_3') is not None else None
                )
            
            risk_management = None
            if risk_data:
                default_position_size, default_max_leverage = _risk_guidance(
                    analysis_data.get('recommendation', 'HOLD'),
                    self.safe_float(analysis_data.get('confidence'), 0.5),
                    analysis_data.get('risk_assessment', 'MEDIUM'),
                    context.volatility
                )
                position_size = risk_data.get('position_size') or default_position_size
                risk_management = RiskManagement(
                    stop_loss=self.safe_float(risk_data.get('stop_loss')) if risk_data.get('stop_loss') is not None else None,
                    position_size=default_position_size if _is_placeholder_risk_text(position_size) else position_size,
                    max_leverage=default_max_leverage
                )
            
            # Create enhanced analysis object
            return LLMAnalysis(
                recommendation=analysis_data.get('recommendation', 'HOLD').upper(),
                confidence=max(0.0, min(1.0, self.safe_float(analysis_data.get('confidence'), 0.5))),
                action_summary=analysis_data.get('action_summary', 'Hold position - market conditions unclear'),
                reasoning=analysis_data.get('reasoning', 'LLM analysis unavailable'),
                key_factors=analysis_data.get('key_factors', []),
                risk_assessment=analysis_data.get('risk_assessment', 'MEDIUM').upper(),
                time_horizon=analysis_data.get('time_horizon', '1-7 days'),
                entry_strategy=entry_strategy,
                price_targets=price_targets,
                risk_management=risk_management,
                market_regime=analysis_data.get('market_regime', 'UNCERTAIN').upper(),
                execution_notes=analysis_data.get('execution_notes', ''),
                # Legacy fields for backward compatibility
                position_sizing=analysis_data.get('position_sizing', 'MEDIUM').upper(),
                stop_loss_level=self.safe_float(risk_data.get('stop_loss')) if risk_data and risk_data.get('stop_loss') is not None else None,
                take_profit_level=self.safe_float(targets_data.get('target_1')) if targets_data and targets_data.get('target_1') is not None else None,
                contrarian_signals=analysis_data.get('contrarian_signals', [])
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            if 'json_str' in locals():
                logger.error(f"Failed JSON string (first 200 chars): {json_str[:200]}")
                logger.error(f"Raw LLM response (first 300 chars): {response[:300]}")
            else:
                logger.error(f"Raw LLM response (first 300 chars): {response[:300]}")
            if strict:
                raise
            logger.info(f"Falling back to freeform parsing for {context.symbol}")
            return self._parse_freeform_response(response, context)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON parsing issues from LLM responses."""
        import re
        
        # Remove any text before first { or after last }
        json_str = json_str.strip()
        
        # Handle completely malformed responses that start with invalid characters
        if not json_str.startswith('{') and not json_str.startswith('['):
            # Try to find JSON-like content
            match = re.search(r'(\{.*\})', json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # If no JSON structure found, return minimal valid JSON
                return '{}'
        
        # Fix common issues:
        # 1. Replace single quotes with double quotes (but not inside strings)
        # 2. Remove trailing commas before closing braces/brackets
        # 3. Fix unescaped newlines in strings
        # 4. Remove comments (// style)
        # 5. Quote unquoted property names
        # 6. Fix malformed strings
        
        # Remove comments
        json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Quote unquoted property names (handles the main error we're seeing)
        # This regex finds property names that aren't quoted but should be
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Replace single quotes with double quotes for property names and strings
        # But be careful not to replace single quotes inside double-quoted strings
        def replace_quotes(match):
            content = match.group(0)
            # Only replace single quotes that are likely property delimiters
            if content.count("'") == 2 and not content.startswith('"'):
                return content.replace("'", '"')
            return content
        
        # Find quoted strings and property names
        json_str = re.sub(r"'[^']*'", replace_quotes, json_str)
        
        # Replace unescaped newlines in string values
        json_str = re.sub(r'(?<!")(\n)(?!")', r'\\n', json_str)
        
        # Fix boolean values (case insensitive)
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        # Remove any remaining invalid control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix missing commas between array/object elements
        json_str = re.sub(r'(\}|\])\s*(\{|\[)', r'\1,\2', json_str)
        json_str = re.sub(r'("\s*)\s*(")', r'\1,\2', json_str)
        
        return json_str
    
    def _parse_json_with_fallbacks(self, json_str: str) -> dict:
        """Attempt to parse JSON with multiple fallback strategies."""
        import json
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Try fixing common remaining issues
        try:
            # Additional cleaning for stubborn cases
            fixed_str = json_str
            
            # Fix Python-style dictionaries that slipped through
            fixed_str = fixed_str.replace("'", '"')
            
            # Fix unquoted values that should be strings
            import re
            fixed_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*[a-zA-Z0-9_])\s*([,}])', r': "\1"\2', fixed_str)
            
            # Fix bare words that should be strings (but not true/false/null)
            fixed_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_\s]*)\s*([,}])', 
                              lambda m: f': "{m.group(1)}"{m.group(2)}' 
                              if m.group(1).lower() not in ['true', 'false', 'null'] else m.group(0), 
                              fixed_str)
            
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Second JSON parsing attempt failed: {e}")
        
        # Strategy 3: Try with ast.literal_eval for Python-like syntax
        try:
            import ast
            # Convert to Python dict first, then to JSON-compatible dict
            python_dict = ast.literal_eval(json_str)
            # Convert to JSON string and back to ensure compatibility
            json_compatible = json.loads(json.dumps(python_dict))
            return json_compatible
        except (ValueError, SyntaxError) as e:
            logger.debug(f"AST parsing failed: {e}")
        
        # Strategy 4: Manual key-value extraction as last resort
        try:
            import re
            
            # Extract key-value pairs manually
            result = {}
            
            # Find all key-value patterns
            patterns = [
                r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
                r'"([^"]+)"\s*:\s*([0-9.]+)',  # "key": number
                r'"([^"]+)"\s*:\s*(true|false|null)',  # "key": boolean/null
                r'(\w+)\s*:\s*"([^"]*)"',      # key: "value" (unquoted key)
                r'(\w+)\s*:\s*([0-9.]+)',      # key: number (unquoted key)
                r'(\w+)\s*:\s*(true|false|null)',  # key: boolean/null (unquoted key)
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, json_str, re.IGNORECASE)
                for match in matches:
                    key, value = match
                    # Convert value types
                    if value.lower() == 'true':
                        result[key] = True
                    elif value.lower() == 'false':
                        result[key] = False
                    elif value.lower() == 'null':
                        result[key] = None
                    elif re.match(r'^[0-9.]+$', str(value)):
                        try:
                            result[key] = float(value) if '.' in str(value) else int(value)
                        except ValueError:
                            result[key] = str(value)
                    else:
                        result[key] = str(value)
            
            if result:
                logger.info(f"Manual key-value extraction succeeded, found {len(result)} pairs")
                return result
                
        except Exception as e:
            logger.debug(f"Manual extraction failed: {e}")
        
        # Final fallback: return empty dict
        logger.warning("All JSON parsing strategies failed, returning empty dict")
        return {}
    
    def _parse_freeform_response(self, response: str, context: MarketContext) -> LLMAnalysis:
        """Parse free-form LLM response as fallback."""
        # Simple keyword-based parsing for fallback
        response_lower = response.lower()
        
        # Determine recommendation
        if 'buy' in response_lower and 'strong' in response_lower:
            recommendation = 'BUY'
            confidence = 0.8
        elif 'buy' in response_lower:
            recommendation = 'BUY'
            confidence = 0.6
        elif 'sell' in response_lower and 'strong' in response_lower:
            recommendation = 'SELL'
            confidence = 0.8
        elif 'sell' in response_lower:
            recommendation = 'SELL'
            confidence = 0.6
        else:
            recommendation = 'HOLD'
            confidence = 0.5
        
        # Extract risk level
        if 'high risk' in response_lower or 'extreme' in response_lower:
            risk_assessment = 'HIGH'
        elif 'low risk' in response_lower:
            risk_assessment = 'LOW'
        else:
            risk_assessment = 'MEDIUM'
        
        risk_position_size, risk_max_leverage = _risk_guidance(
            recommendation,
            confidence,
            risk_assessment,
            context.volatility
        )

        # Direction-aware targets/stop. A SELL must carry targets BELOW entry and a
        # stop ABOVE it; only BUY/HOLD frame upside. The old code hard-coded upside
        # targets and a downside stop for every recommendation, which produced an
        # internally contradictory SELL whenever JSON parsing failed.
        price = context.current_price
        if recommendation == 'SELL':
            price_targets = PriceTargets(
                target_1=price * 0.95,
                target_2=price * 0.90,
                target_3=price * 0.85,
            )
            fallback_stop = price * 1.05
        else:  # BUY or HOLD frame upside
            price_targets = PriceTargets(
                target_1=price * 1.05,
                target_2=price * 1.10,
                target_3=price * 1.15,
            )
            fallback_stop = price * 0.95

        return LLMAnalysis(
            recommendation=recommendation,
            confidence=confidence,
            action_summary=f"{recommendation} - {response[:100]}...",  # Short summary
            reasoning=response[:500],  # Truncate long responses
            key_factors=['LLM analysis', 'Technical indicators', 'Market context'],
            risk_assessment=risk_assessment,
            time_horizon='4-24 hours',
            entry_strategy=EntryStrategy(
                optimal_entry=context.current_price,
                entry_range_low=context.current_price * 0.98,
                entry_range_high=context.current_price * 1.02,
                market_order_ok=True
            ),
            price_targets=price_targets,
            risk_management=RiskManagement(
                stop_loss=fallback_stop,
                position_size=risk_position_size,
                max_leverage=risk_max_leverage
            ),
            execution_notes='LLM response was not in expected JSON format.',
            position_sizing='MEDIUM'
        )
    
    def _fallback_analysis(self, context: MarketContext) -> LLMAnalysis:
        """Provide fallback analysis when LLM is unavailable."""
        # Dynamic rule-based fallback using real market data
        recommendation = 'HOLD'
        reasoning = "Technical conditions are mixed, so Ryu is waiting for cleaner confirmation"
        
        # Calculate dynamic confidence based on multiple indicators
        confidence_factors = []
        
        # RSI confidence factor (0.3-0.8 based on RSI extremes)
        if context.rsi < 20:  # Extremely oversold
            rsi_confidence = 0.8
        elif context.rsi < 30:  # Oversold
            rsi_confidence = 0.7
        elif context.rsi > 80:  # Extremely overbought
            rsi_confidence = 0.8
        elif context.rsi > 70:  # Overbought
            rsi_confidence = 0.7
        elif 40 <= context.rsi <= 60:  # Neutral zone
            rsi_confidence = 0.4
        else:  # Moderate zones
            rsi_confidence = 0.5
        confidence_factors.append(rsi_confidence)
        
        # Volume confidence factor (higher volume = higher confidence)
        volume_confidence = min(0.8, max(0.3, context.volume_24h / 10000000))  # Scale based on volume
        confidence_factors.append(volume_confidence)
        
        # Volatility confidence factor (moderate volatility preferred)
        if 0.02 <= context.volatility <= 0.08:  # Optimal volatility range
            volatility_confidence = 0.7
        elif context.volatility > 0.15:  # Too volatile
            volatility_confidence = 0.3
        elif context.volatility < 0.01:  # Too stable
            volatility_confidence = 0.4
        else:  # Moderate
            volatility_confidence = 0.5
        confidence_factors.append(volatility_confidence)
        
        # MACD confirmation - handle dict or float safely
        def safe_macd_value(macd_data):
            if isinstance(macd_data, dict):
                return float(macd_data.get('macd', 0.0))
            return float(macd_data) if macd_data is not None else 0.0
        
        macd_value = safe_macd_value(context.macd)
        macd_confidence = 0.6 if abs(macd_value) > 0.001 else 0.4  # Strong signal vs weak
        confidence_factors.append(macd_confidence)
        
        # Calculate weighted average confidence
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Enhanced technical analysis with multiple confirmations
        rsi_signal = "oversold" if context.rsi < 30 else "overbought" if context.rsi > 70 else "neutral"
        macd_signal = "bullish" if macd_value > 0 else "bearish"
        momentum_signal = "positive" if context.price_change_24h > 2 else "negative" if context.price_change_24h < -2 else "neutral"
        
        # Dynamic recommendation logic
        if context.rsi < 30 and context.price_change_24h > -5 and context.volume_24h > 1000000:
            recommendation = 'BUY'
            confidence = min(0.8, confidence + 0.15)  # Boost confidence for strong signals
            reasoning = f"Oversold RSI ({context.rsi:.1f}) with limited downside and strong volume"
        elif context.rsi > 70 and context.price_change_24h > 5 and context.volatility < 0.1:
            recommendation = 'SELL'
            confidence = min(0.8, confidence + 0.15)
            reasoning = f"Overbought RSI ({context.rsi:.1f}) with strong upward movement and manageable volatility"
        elif context.rsi < 25:  # Extreme oversold
            recommendation = 'BUY'
            confidence = min(0.85, confidence + 0.2)
            reasoning = f"Extreme oversold conditions (RSI: {context.rsi:.1f}) suggest potential reversal"
        elif context.rsi > 75:  # Extreme overbought
            recommendation = 'SELL'
            confidence = min(0.85, confidence + 0.2)
            reasoning = f"Extreme overbought conditions (RSI: {context.rsi:.1f}) suggest potential correction"
        elif macd_value > 0.005 and context.price_change_24h > 0 and context.rsi < 60:
            recommendation = 'BUY'
            confidence = min(0.7, confidence + 0.1)
            reasoning = f"Bullish MACD divergence with positive momentum and room for growth"
        
        # Dynamic stop loss based on volatility and ATR estimate
        # Use volatility to calculate a more appropriate stop loss
        volatility_multiplier = max(0.02, min(0.12, context.volatility * 2))  # 2% to 12% based on volatility
        dynamic_stop_loss = context.current_price * (1 - volatility_multiplier) if recommendation == 'BUY' else context.current_price * (1 + volatility_multiplier)
        
        # Dynamic price targets based on volatility and momentum
        momentum_factor = abs(context.price_change_24h) / 100 + 1  # Adjust targets based on current momentum
        volatility_factor = min(2.0, max(0.5, context.volatility * 10))  # Scale targets by volatility
        
        if recommendation == 'BUY':
            target_1 = context.current_price * (1 + (0.03 * momentum_factor * volatility_factor))
            target_2 = context.current_price * (1 + (0.06 * momentum_factor * volatility_factor))
            target_3 = context.current_price * (1 + (0.10 * momentum_factor * volatility_factor))
        elif recommendation == 'SELL':
            # A short's targets sit BELOW entry; the stop (computed above) sits above.
            target_1 = context.current_price * (1 - (0.02 * volatility_factor))
            target_2 = context.current_price * (1 - (0.04 * volatility_factor))
            target_3 = context.current_price * (1 - (0.07 * volatility_factor))
        else:  # HOLD
            target_1 = context.current_price * (1 + (0.02 * volatility_factor))
            target_2 = context.current_price * (1 + (0.04 * volatility_factor))
            target_3 = context.current_price * (1 + (0.07 * volatility_factor))
        
        # Dynamic risk assessment
        risk_factors = [context.volatility, 1 - confidence, abs(context.price_change_24h) / 100]
        avg_risk = sum(risk_factors) / len(risk_factors)
        
        if avg_risk > 0.12:
            risk_assessment = 'HIGH'
        elif avg_risk > 0.06:
            risk_assessment = 'MEDIUM'
        else:
            risk_assessment = 'LOW'

        position_size, max_leverage = _risk_guidance(
            recommendation,
            confidence,
            risk_assessment,
            context.volatility
        )
        
        return LLMAnalysis(
            recommendation=recommendation,
            confidence=confidence,
            action_summary=f"{recommendation} - {reasoning}",
            reasoning=reasoning,
            key_factors=[f'RSI: {rsi_signal}', f'MACD: {macd_signal}', f'Momentum: {momentum_signal}', f'Volatility: {context.volatility:.1%}'],
            risk_assessment=risk_assessment,
            time_horizon='4-24 hours',
            entry_strategy=EntryStrategy(
                optimal_entry=context.current_price,
                entry_range_low=context.current_price * (1 - volatility_multiplier/2),
                entry_range_high=context.current_price * (1 + volatility_multiplier/2),
                market_order_ok=context.volatility < 0.08  # Only allow market orders in low volatility
            ),
            price_targets=PriceTargets(
                target_1=target_1,
                target_2=target_2,
                target_3=target_3
            ),
            risk_management=RiskManagement(
                stop_loss=dynamic_stop_loss,
                position_size=position_size,
                max_leverage=max_leverage
            ),
            execution_notes=f'Dynamic analysis based on volatility ({context.volatility:.1%}), RSI ({context.rsi:.1f}), and volume.',
            position_sizing='DYNAMIC' if context.volatility > 0.08 else 'SMALL'
        )
    
    async def analyze_portfolio_signals(
        self, 
        signals: List[Dict[str, Any]], 
        market_overview: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze multiple signals using LLM for portfolio-level insights.
        
        Args:
            signals: List of individual token signals
            market_overview: Overall market context
            
        Returns:
            Portfolio-level recommendations and insights
        """
        try:
            if not signals:
                return {'portfolio_action': 'HOLD', 'reasoning': 'No signals to analyze'}
            
            # Build portfolio analysis prompt
            signals_summary = []
            for signal in signals:
                signals_summary.append(
                    f"- {signal.get('symbol', 'Unknown')}: {signal.get('recommendation', 'HOLD')} "
                    f"(confidence: {signal.get('confidence', 0.5):.2f}) - {signal.get('reasoning', '')[:100]}"
                )
            
            prompt = f"""
Analyze this portfolio of trading signals and provide overall portfolio guidance:

INDIVIDUAL SIGNALS:
{chr(10).join(signals_summary)}

MARKET OVERVIEW:
{market_overview}

Provide portfolio-level recommendations considering:
1. Signal correlation and diversification
2. Overall market regime
3. Risk concentration
4. Optimal position sizing across signals
5. Market timing considerations

Response format:
{{
    "portfolio_action": "AGGRESSIVE_BUY|BUY|HOLD|SELL|AGGRESSIVE_SELL",
    "confidence": 0.0-1.0,
    "reasoning": "Portfolio-level analysis",
    "recommended_allocation": {{"symbol": percentage}},
    "risk_level": "LOW|MEDIUM|HIGH|EXTREME",
    "market_timing": "EXCELLENT|GOOD|FAIR|POOR",
    "diversification_score": 0.0-1.0
}}
"""
            
            response, _ = await self._query_llm(prompt)
            
            # Parse portfolio analysis
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(response[json_start:json_end])
            except:
                pass
            
            # Fallback portfolio analysis
            return {
                'portfolio_action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Portfolio analysis unavailable',
                'risk_level': 'MEDIUM'
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {'portfolio_action': 'HOLD', 'reasoning': f'Analysis error: {e}'}


# Factory function
def create_llm_analysis_service(
    provider: str = None,
    *,
    model_name_override: Optional[str] = None,
    thinking_type: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> LLMAnalysisService:
    """Create LLM analysis service with specified provider."""
    if provider:
        provider_enum = LLMProvider(provider.lower())
        return LLMAnalysisService(
            provider_enum,
            model_name_override=model_name_override,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
        )
    else:
        # Use environment variable or default
        return LLMAnalysisService(
            model_name_override=model_name_override,
            thinking_type=thinking_type,
            reasoning_effort=reasoning_effort,
        )


# Integration helper for Yuki agent
async def enhance_yuki_signals_with_llm(
    signals: List[Dict[str, Any]], 
    market_data: Dict[str, Any],
    llm_service: Optional[LLMAnalysisService] = None
) -> List[Dict[str, Any]]:
    """
    Enhance Yuki's algorithmic signals with LLM reasoning.
    
    This function takes Yuki's technical analysis signals and adds
    LLM-based market context and reasoning.
    """
    if not llm_service:
        llm_service = create_llm_analysis_service()
    
    enhanced_signals = []
    
    for signal in signals:
        try:
            # Create market context for LLM
            symbol = signal.get('symbol', '')
            context = MarketContext(
                timestamp=datetime.now(),
                symbol=symbol,
                current_price=signal.get('price', 0),
                price_change_24h=signal.get('price_change_24h', 0),
                volume_24h=signal.get('volume_24h', 0),
                volatility=signal.get('volatility', 0.1),
                rsi=signal.get('rsi', 50),
                macd=signal.get('macd', 0),
                bb_position=signal.get('bb_position', 0.5),
                fear_greed_index=market_data.get('fear_greed_index', 50),
                social_sentiment=market_data.get('social_sentiment', 'neutral'),
                news_sentiment=market_data.get('news_sentiment', 'neutral'),
                market_narrative=market_data.get('market_narrative', ''),
                funding_rate=signal.get('funding_rate')
            )
            
            # Get LLM analysis
            llm_analysis = await llm_service.analyze_trading_opportunity(context)
            
            # Combine algorithmic and LLM signals
            enhanced_signal = {
                **signal,  # Original Yuki signal
                'llm_recommendation': llm_analysis.recommendation,
                'llm_confidence': llm_analysis.confidence,
                'llm_reasoning': llm_analysis.reasoning,
                'llm_risk_assessment': llm_analysis.risk_assessment,
                'combined_confidence': (signal.get('confidence', 0.5) + llm_analysis.confidence) / 2,
                'stop_loss_level': llm_analysis.stop_loss_level,
                'take_profit_level': llm_analysis.take_profit_level,
                'market_regime': llm_analysis.market_regime,
                'enhanced_by_llm': True
            }
            
            enhanced_signals.append(enhanced_signal)
            
        except Exception as e:
            logger.error(f"Error enhancing signal for {signal.get('symbol', 'unknown')}: {e}")
            # Include original signal if LLM enhancement fails
            enhanced_signals.append({**signal, 'enhanced_by_llm': False})
    
    return enhanced_signals
