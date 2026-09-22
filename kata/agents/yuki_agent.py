"""
Yuki Agent - Aggressive Futures Trading Strategy

Yuki is the aggressive agent specialized for Hyperliquid futures trading:
- High-risk, high-reward perpetual futures
- Advanced technical analysis with 10+ indicators
- Dynamic leverage management (2x-10x)
- Funding rate arbitrage strategies
- Target: 30-100% annual returns
"""

import asyncio
import json
import logging
import statistics
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from kata.agents.base_agent import BaseAgent, MarketData, TradingSignal, SignalType, RiskLevel
from kata.services.llm_analysis_service import LLMAnalysisService, MarketContext, create_llm_analysis_service
from kata.services.hyperliquid_service import HyperliquidService, OrderSide, OrderType, LiveMarketData
from kata.services.agent_database_service import get_agent_db_service
from kata.config.settings import settings

logger = logging.getLogger(__name__)


class FuturesPosition(Enum):
    """Futures position types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators data structure."""
    # Moving Averages
    ema_20: float = 0.0
    ema_50: float = 0.0
    sma_200: float = 0.0
    
    # Momentum Indicators
    rsi: float = 50.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_rsi: float = 50.0
    
    # Trend Indicators
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 25.0
    
    # Volatility Indicators
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    atr: float = 0.0
    
    # Volume Indicators
    vwap: float = 0.0
    volume_ratio: float = 1.0
    
    # Support/Resistance
    pivot_point: float = 0.0
    support_1: float = 0.0
    resistance_1: float = 0.0


@dataclass
class FuturesMarketData:
    """Hyperliquid-specific market data."""
    symbol: str
    price: float
    funding_rate: float
    funding_rate_8h: float
    open_interest: float
    oi_change_24h: float
    volume_24h: float
    mark_price: float
    index_price: float
    last_funding_time: datetime
    next_funding_time: datetime
    liquidation_threshold: float
    max_leverage: float


@dataclass
class PositionMetrics:
    """Position risk and performance metrics."""
    unrealized_pnl: float
    unrealized_pnl_percent: float
    margin_ratio: float
    liquidation_price: float
    funding_cost_24h: float
    time_in_position_hours: float
    max_drawdown: float
    position_score: float  # Overall position health score


class YukiAgent(BaseAgent):
    """
    Aggressive futures trading agent specialized for Hyperliquid perpetuals.
    
    Yuki's trading philosophy:
    - Aggressive leveraged futures trading (2x-10x leverage)
    - Advanced technical analysis with 10+ indicators
    - Funding rate arbitrage strategies
    - Target: 30-100% annual returns
    - Focus on major perps: BTC, ETH, SOL
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any], hyperliquid_service: Optional[HyperliquidService] = None):
        """Initialize Yuki agent with aggressive futures trading defaults."""
        
        # Set aggressive futures risk parameters
        futures_risk_params = {
            'max_position_size_percent': 100.0,  # 100% per trade (aggressive)
            'max_leverage': 40.0,  # Maximum 40x leverage
            'min_leverage': 2.0,   # Minimum 2x leverage
            'stop_loss_percent': 8.0,  # Wider stops for futures volatility
            'take_profit_percent': 40.0,  # Higher profit targets
            'max_daily_loss_percent': 10.0,  # 10% daily loss limit
            'margin_buffer_percent': 30.0,  # Maintain 30% margin buffer
            'min_confidence_threshold': 0.65,  # Futures require higher confidence
            'max_trades_per_day': 15,  # Active futures trading
            'cooldown_minutes': 3,  # Short cooldown for futures
            'funding_rate_threshold': 0.01  # 1% funding rate threshold
        }
        
        # Initialize Hyperliquid service for live trading
        self.hyperliquid_service = hyperliquid_service
        
        # Database service for storing trades and decisions
        self.db_service = get_agent_db_service()
        
        # Separate base risk params from futures-specific params
        config_risk_params = config.get('risk_params', {})
        
        # Extract base risk parameters for parent class
        base_risk_params = {
            'max_position_size_percent': config_risk_params.get('max_position_size_percent', 15.0),
            'stop_loss_percent': config_risk_params.get('stop_loss_percent', 8.0),
            'take_profit_percent': config_risk_params.get('take_profit_percent', 40.0),
            'max_daily_loss_percent': config_risk_params.get('max_daily_loss_percent', 10.0),
            'min_confidence_threshold': config_risk_params.get('min_confidence_threshold', 0.65),
            'max_trades_per_day': config_risk_params.get('max_trades_per_day', 15),
            'cooldown_minutes': config_risk_params.get('cooldown_minutes', 3)
        }
        
        # Store futures-specific risk parameters separately
        self.futures_risk_params = {**futures_risk_params, **config_risk_params}
        
        # Update config with base risk params only
        config['risk_params'] = base_risk_params
        
        super().__init__(user_id, "yuki", config)
        
        # Hyperliquid futures-specific parameters
        self.supported_futures = {
            'BTC-USD': {'max_leverage': 50, 'tick_size': 1.0, 'min_size': 0.001},
            'ETH-USD': {'max_leverage': 50, 'tick_size': 0.1, 'min_size': 0.01},
            'SOL-USD': {'max_leverage': 20, 'tick_size': 0.01, 'min_size': 0.1},
            'AVAX-USD': {'max_leverage': 20, 'tick_size': 0.01, 'min_size': 0.1},
            'MATIC-USD': {'max_leverage': 20, 'tick_size': 0.0001, 'min_size': 1.0}
        }
        
        # Technical analysis parameters
        self.ta_config = {
            'ema_periods': [20, 50],
            'sma_periods': [200],
            'rsi_period': 14,
            'rsi_oversold': 25,  # More aggressive levels
            'rsi_overbought': 75,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'adx_period': 14,
            'adx_trend_threshold': 25,
            'atr_period': 14,
            'stoch_k': 14,
            'stoch_d': 3
        }
        
        # Funding rate strategy parameters
        self.funding_strategy = {
            'high_funding_threshold': 0.005,  # 0.5% per 8h
            'low_funding_threshold': -0.005,
            'funding_arbitrage_min_duration': 24,  # Hours
            'funding_position_max_size': 0.1  # 10% of portfolio for funding plays
        }
        
        # Market regime detection
        self.regime_thresholds = {
            'trending_adx': 25,
            'volatile_atr_multiplier': 1.5,
            'ranging_bb_width': 0.02,
            'breakout_volume_multiplier': 2.0
        }
        
        # Performance tracking
        self.futures_metrics = {
            'total_funding_earned': 0.0,
            'max_leverage_used': 1.0,
            'avg_holding_time_hours': 0.0,
            'liquidation_near_misses': 0
        }
        self.react_strict_json = bool(
            config.get("llm_strict_json", getattr(settings, "YUKI_REACT_STRICT_JSON", True))
        )
        self.react_max_output_tokens = max(
            256,
            int(
                config.get(
                    "llm_max_output_tokens",
                    getattr(settings, "YUKI_REACT_MAX_OUTPUT_TOKENS", 1200),
                )
                or 1200
            ),
        )
        self.react_goals_limit = max(
            1,
            int(config.get("react_goals_limit", getattr(settings, "YUKI_REACT_GOALS_LIMIT", 3)) or 3),
        )
        self.react_lessons_limit = max(
            1,
            int(
                config.get(
                    "react_lessons_limit",
                    getattr(settings, "YUKI_REACT_LESSONS_LIMIT", 4),
                )
                or 4
            ),
        )
        self.react_episodes_limit = max(
            1,
            int(
                config.get(
                    "react_episodes_limit",
                    getattr(settings, "YUKI_REACT_EPISODES_LIMIT", 3),
                )
                or 3
            ),
        )
        self.react_evidence_list_limit = max(
            2,
            int(
                config.get(
                    "react_evidence_list_limit",
                    getattr(settings, "YUKI_REACT_EVIDENCE_LIST_LIMIT", 6),
                )
                or 6
            ),
        )
        self.react_evidence_text_limit = max(
            120,
            int(
                config.get(
                    "react_evidence_text_limit",
                    getattr(settings, "YUKI_REACT_EVIDENCE_TEXT_LIMIT", 280),
                )
                or 280
            ),
        )
        
        # Initialize LLM analysis service for enhanced reasoning (defaulting to DeepSeek)
        try:
            llm_provider_choice = str(
                config.get("llm_provider")
                or getattr(settings, "YUKI_REACT_LLM_PROVIDER", "deepseek")
                or "deepseek"
            ).strip().lower()
            llm_model_choice = str(
                config.get("llm_model")
                or getattr(settings, "YUKI_REACT_LLM_MODEL", "")
                or ""
            ).strip() or None
            llm_thinking_type = str(
                config.get("llm_thinking_type")
                or getattr(settings, "YUKI_REACT_THINKING_TYPE", "")
                or ""
            ).strip().lower() or None
            llm_reasoning_effort = str(
                config.get("llm_reasoning_effort")
                or getattr(settings, "YUKI_REACT_REASONING_EFFORT", "")
                or ""
            ).strip().lower() or None
            self.llm_service = create_llm_analysis_service(
                provider=llm_provider_choice,
                model_name_override=llm_model_choice,
                thinking_type=llm_thinking_type,
                reasoning_effort=llm_reasoning_effort,
            )
            self.use_llm_enhancement = True
            logger.info(
                "LLM analysis service (%s:%s, thinking=%s, strict_json=%s) initialized for Yuki agent",
                llm_provider_choice,
                llm_model_choice or getattr(self.llm_service, "model_name", "default"),
                llm_thinking_type or "default",
                self.react_strict_json,
            )
        except Exception as e:
            logger.warning(f"LLM service initialization failed: {e}. Using algorithmic analysis only.")
            self.llm_service = None
            self.use_llm_enhancement = False

        # Initialize Agent Memory & Tools Services for ReAct Reasoning
        from kata.services.agent_memory_service import get_agent_memory_service
        from kata.services.yuki_agent_tools import YukiAgentTools
        
        self.memory_service = get_agent_memory_service(agent_id=f"yuki_{user_id}")
        self.tools = YukiAgentTools(
            hyperliquid_service=self.hyperliquid_service
        )
        
        logger.info(f"Yuki futures agent initialized for user {user_id} - targeting Hyperliquid perps with ReAct memory & tools")

    def _react_response_format(self) -> Optional[Dict[str, str]]:
        """Return strict JSON output mode for live reviews when enabled."""
        return {"type": "json_object"} if getattr(self, "react_strict_json", True) else None

    @staticmethod
    def _telemetry_to_dict(telemetry: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Normalize usage telemetry into a JSON-safe dict."""
        if telemetry is None:
            return None
        return {
            "provider": getattr(telemetry, "provider", None),
            "model": getattr(telemetry, "model", None),
            "prompt_tokens": int(getattr(telemetry, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(telemetry, "completion_tokens", 0) or 0),
            "estimated_cost_usd": float(getattr(telemetry, "estimated_cost_usd", 0.0) or 0.0),
        }

    def _compact_react_payload(
        self,
        value: Any,
        *,
        depth: int = 0,
    ) -> Any:
        """Trim repeated live-review context so refresh prompts stay small."""
        evidence_text_limit = max(120, int(getattr(self, "react_evidence_text_limit", 280) or 280))
        evidence_list_limit = max(2, int(getattr(self, "react_evidence_list_limit", 6) or 6))
        if depth >= 4:
            return "<trimmed>"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            normalized = " ".join(value.split())
            if len(normalized) <= evidence_text_limit:
                return normalized
            return normalized[: evidence_text_limit - 3].rstrip() + "..."
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            max_items = max(evidence_list_limit * 3, 12)
            compact: Dict[str, Any] = {}
            items = list(value.items())
            for key, nested_value in items[:max_items]:
                compact[str(key)] = self._compact_react_payload(nested_value, depth=depth + 1)
            if len(items) > max_items:
                compact["_truncated_keys"] = len(items) - max_items
            return compact
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            compact_items = [
                self._compact_react_payload(item, depth=depth + 1)
                for item in items[:evidence_list_limit]
            ]
            if len(items) > evidence_list_limit:
                compact_items.append({"_truncated_items": len(items) - evidence_list_limit})
            return compact_items
        return self._compact_react_payload(str(value), depth=depth + 1)

    def _build_react_memory_context(
        self,
        goals: List[Any],
        lessons: List[Any],
        recent_episodes: List[Any],
        *,
        episode_key: str,
    ) -> Dict[str, Any]:
        """Build a compact memory payload for live reviews."""
        goals_limit = max(1, int(getattr(self, "react_goals_limit", 3) or 3))
        lessons_limit = max(1, int(getattr(self, "react_lessons_limit", 4) or 4))
        episodes_limit = max(1, int(getattr(self, "react_episodes_limit", 3) or 3))
        goal_items = [
            self._compact_react_payload(getattr(goal, "goal_statement", str(goal)))
            for goal in goals[:goals_limit]
            if str(getattr(goal, "goal_statement", str(goal))).strip()
        ]
        lesson_items = [
            self._compact_react_payload(str(lesson))
            for lesson in lessons[:lessons_limit]
            if str(lesson).strip()
        ]
        episode_items = []
        for episode in recent_episodes[:episodes_limit]:
            episode_items.append(
                self._compact_react_payload({
                    "action": getattr(episode, "action_taken", None),
                    "confidence": getattr(episode, "confidence", None),
                    "observation": getattr(episode, "observation", None),
                    "trade_id": getattr(episode, "trade_id", None),
                    "created_at": getattr(episode, "created_at", None),
                })
            )
        return {
            "active_goals": goal_items,
            "lessons": lesson_items,
            episode_key: episode_items,
        }

    async def _query_react_json_review(
        self,
        prompt: str,
        *,
        symbol: str,
        review_name: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        """Run a live-review LLM call with strict JSON output and one bounded retry."""
        max_output_tokens = max(256, int(getattr(self, "react_max_output_tokens", 1200) or 1200))
        query_result = await self.llm_service._query_llm(
            prompt,
            max_output_tokens=max_output_tokens,
            response_format=self._react_response_format(),
        )
        if isinstance(query_result, tuple):
            raw_response, usage = query_result
        else:
            raw_response, usage = str(query_result or ""), None
        decision = self._parse_react_json_response(
            raw_response,
            symbol=symbol,
            review_name=review_name,
        )
        if decision:
            return raw_response, decision, self._telemetry_to_dict(usage), False

        retry_prompt = (
            prompt
            + "\n\nCRITICAL: Respond with json only. Return one valid JSON object and nothing else."
        )
        retry_result = await self.llm_service._query_llm(
            retry_prompt,
            max_output_tokens=max_output_tokens,
            response_format=self._react_response_format(),
        )
        if isinstance(retry_result, tuple):
            raw_response_retry, retry_usage = retry_result
        else:
            raw_response_retry, retry_usage = str(retry_result or ""), None
        decision = self._parse_react_json_response(
            raw_response_retry,
            symbol=symbol,
            review_name=f"{review_name} (retry)",
        )
        merge_usage = getattr(self.llm_service, "_merge_usage", None)
        usage_record = self._telemetry_to_dict(
            merge_usage(usage, retry_usage)
            if callable(merge_usage)
            else retry_usage or usage
        )
        return raw_response_retry, decision, usage_record, True
        
        # Initialize market data timestamp
        self._last_market_data_fetch = datetime.now()
    
    async def get_real_time_market_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive real-time market analysis using live Hyperliquid data.
        
        Returns:
            Enhanced market analysis with live data feeds
        """
        try:
            if not self.hyperliquid_service:
                return {"error": "No Hyperliquid service available - live trading service required"}
            
            if not self.hyperliquid_service.is_ws_connected:
                return {"error": "WebSocket connection required for real-time market analysis - no fallback data"}
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'live_data_available': True,
                'symbols_analysis': {},
                'funding_opportunities': [],
                'market_overview': {},
                'trading_recommendations': []
            }
            
            # Analyze each supported futures symbol
            for symbol in self.supported_futures.keys():
                try:
                    # Get live market data
                    live_data = await self.hyperliquid_service.get_live_market_data(symbol)
                    if not live_data:
                        continue
                    
                    # Get funding rate data
                    funding_data = await self.hyperliquid_service.get_funding_rates([symbol])
                    funding_info = funding_data.get(symbol)
                    
                    # Calculate spread and liquidity metrics
                    spread_bps = ((live_data.ask - live_data.bid) / live_data.mid_price) * 10000 if live_data.mid_price > 0 else 0
                    
                    # Analyze funding opportunities
                    funding_opportunity = None
                    if funding_info and abs(funding_info.funding_rate) > self.futures_risk_params['funding_rate_threshold']:
                        annualized_rate = funding_info.funding_rate * 365 * 3  # Assuming 8h funding
                        if abs(annualized_rate) > 0.1:  # 10% annualized
                            funding_opportunity = {
                                'symbol': symbol,
                                'funding_rate': funding_info.funding_rate,
                                'annualized_rate': annualized_rate,
                                'strategy': 'short' if funding_info.funding_rate > 0 else 'long',
                                'expected_return': abs(annualized_rate),
                                'risk_level': 'medium'
                            }
                            analysis['funding_opportunities'].append(funding_opportunity)
                    
                    # Technical analysis with live data
                    price_momentum = self._calculate_price_momentum(live_data)
                    volatility_score = self._calculate_volatility_score(live_data)
                    
                    analysis['symbols_analysis'][symbol] = {
                        'price': live_data.price,
                        'bid': live_data.bid,
                        'ask': live_data.ask,
                        'spread_bps': spread_bps,
                        'volume_24h': live_data.volume_24h,
                        'funding_rate': funding_info.funding_rate if funding_info else 0,
                        'price_momentum': price_momentum,
                        'volatility_score': volatility_score,
                        'funding_opportunity': funding_opportunity is not None,
                        'trading_signal': self._generate_live_trading_signal(live_data, funding_info, price_momentum, volatility_score),
                        'last_update': live_data.timestamp.isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Generate market overview
            analysis['market_overview'] = await self._generate_market_overview(analysis['symbols_analysis'])
            
            # Generate trading recommendations
            analysis['trading_recommendations'] = self._generate_trading_recommendations(
                analysis['symbols_analysis'], 
                analysis['funding_opportunities']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in real-time market analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_price_momentum(self, live_data: LiveMarketData) -> float:
        """Calculate price momentum from live data."""
        try:
            # Simple momentum calculation using price vs VWAP
            if live_data.vwap > 0:
                momentum = (live_data.price - live_data.vwap) / live_data.vwap
                return max(-1.0, min(1.0, momentum))  # Clamp between -1 and 1
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_volatility_score(self, live_data: LiveMarketData) -> float:
        """Calculate volatility score from live data."""
        try:
            # Simple volatility using high-low range
            if live_data.high_24h > 0 and live_data.low_24h > 0:
                volatility = (live_data.high_24h - live_data.low_24h) / live_data.price
                return min(1.0, volatility)  # Cap at 100%
            return 0.0
        except Exception:
            return 0.0
    
    def _generate_live_trading_signal(self, live_data: LiveMarketData, funding_info, 
                                     price_momentum: float, volatility_score: float) -> Dict[str, Any]:
        """Generate trading signal from live data."""
        try:
            signal_strength = 0.0
            signal_direction = 'neutral'
            reasons = []
            
            # Momentum-based signals
            if price_momentum > 0.02:  # 2% momentum
                signal_strength += 0.3
                signal_direction = 'bullish'
                reasons.append(f"Positive momentum: {price_momentum:.2%}")
            elif price_momentum < -0.02:
                signal_strength += 0.3
                signal_direction = 'bearish'
                reasons.append(f"Negative momentum: {price_momentum:.2%}")
            
            # Funding rate signals
            if funding_info and abs(funding_info.funding_rate) > 0.005:  # 0.5% funding
                signal_strength += 0.2
                if funding_info.funding_rate > 0:
                    signal_direction = 'bearish' if signal_direction != 'bullish' else 'neutral'
                    reasons.append(f"High funding rate: {funding_info.funding_rate:.3%}")
                else:
                    signal_direction = 'bullish' if signal_direction != 'bearish' else 'neutral'
                    reasons.append(f"Negative funding rate: {funding_info.funding_rate:.3%}")
            
            # Volatility adjustment
            if volatility_score > 0.15:  # High volatility
                signal_strength *= 0.8  # Reduce confidence in high volatility
                reasons.append(f"High volatility: {volatility_score:.1%}")
            
            # Spread quality
            spread_bps = ((live_data.ask - live_data.bid) / live_data.mid_price) * 10000 if live_data.mid_price > 0 else 100
            if spread_bps < 5:  # Tight spread
                signal_strength += 0.1
                reasons.append("Tight spread - good liquidity")
            elif spread_bps > 20:  # Wide spread
                signal_strength *= 0.7
                reasons.append("Wide spread - poor liquidity")
            
            return {
                'direction': signal_direction,
                'strength': min(1.0, signal_strength),
                'confidence': signal_strength,
                'reasons': reasons,
                'momentum': price_momentum,
                'volatility': volatility_score,
                'spread_bps': spread_bps
            }
            
        except Exception as e:
            logger.error(f"Error generating live trading signal: {e}")
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'reasons': ['Error in signal generation'],
                'error': str(e)
            }
    
    async def _generate_market_overview(self, symbols_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market overview."""
        try:
            if not symbols_analysis:
                return {}
            
            # Calculate aggregate metrics
            total_volume = sum(data.get('volume_24h', 0) for data in symbols_analysis.values())
            avg_spread = sum(data.get('spread_bps', 0) for data in symbols_analysis.values()) / len(symbols_analysis)
            
            # Count signals
            bullish_signals = sum(1 for data in symbols_analysis.values() 
                                if data.get('trading_signal', {}).get('direction') == 'bullish')
            bearish_signals = sum(1 for data in symbols_analysis.values() 
                                if data.get('trading_signal', {}).get('direction') == 'bearish')
            
            # Market sentiment
            if bullish_signals > bearish_signals:
                market_sentiment = 'bullish'
            elif bearish_signals > bullish_signals:
                market_sentiment = 'bearish'
            else:
                market_sentiment = 'neutral'
            
            return {
                'market_sentiment': market_sentiment,
                'total_volume_24h': total_volume,
                'avg_spread_bps': avg_spread,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'symbols_tracked': len(symbols_analysis),
                'liquidity_quality': 'good' if avg_spread < 10 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Error generating market overview: {e}")
            return {}
    
    def _generate_trading_recommendations(self, symbols_analysis: Dict[str, Any], 
                                        funding_opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable trading recommendations."""
        try:
            recommendations = []
            
            # High-confidence signals
            for symbol, data in symbols_analysis.items():
                signal = data.get('trading_signal', {})
                if signal.get('confidence', 0) > 0.7:
                    recommendations.append({
                        'type': 'directional_trade',
                        'symbol': symbol,
                        'direction': signal.get('direction'),
                        'confidence': signal.get('confidence'),
                        'reasoning': ', '.join(signal.get('reasons', [])),
                        'priority': 'high'
                    })
            
            # Funding arbitrage opportunities
            for opportunity in funding_opportunities:
                if opportunity.get('expected_return', 0) > 0.15:  # 15% annualized
                    recommendations.append({
                        'type': 'funding_arbitrage',
                        'symbol': opportunity['symbol'],
                        'strategy': opportunity['strategy'],
                        'expected_return': opportunity['expected_return'],
                        'reasoning': f"High funding rate: {opportunity['funding_rate']:.3%}",
                        'priority': 'medium'
                    })
            
            # Sort by priority and confidence
            recommendations.sort(key=lambda x: (
                x.get('priority') == 'high',
                x.get('confidence', 0)
            ), reverse=True)
            
            return recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return []
    
    async def analyze_market(self, market_data: MarketData, focus_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced technical analysis for Hyperliquid futures trading.
        
        Comprehensive analysis including:
        - 10+ technical indicators (EMA, MACD, RSI, Bollinger, VWAP, ADX, etc.)
        - Funding rate analysis for arbitrage opportunities
        - Open Interest and volume patterns
        - Market regime classification
        - Volatility-based leverage recommendations
        
        Args:
            market_data: General market data
            focus_token: If provided, focus analysis on this specific token (e.g., "ETH")
        """
        try:
            analysis = {
                'timestamp': market_data.timestamp.isoformat(),
                'market_regime': MarketRegime.RANGING,
                'overall_sentiment': 'neutral',
                'volatility_regime': 'normal',
                'futures_opportunities': [],
                'funding_arbitrage': [],
                'risk_level': RiskLevel.MEDIUM,
                'recommended_leverage': {},
                'technical_summary': {}
            }
            
            # Determine which symbols to analyze
            if focus_token:
                # Focus on the specific token being analyzed
                token_symbol = f"{focus_token}-USD"
                if token_symbol in self.supported_futures:
                    symbols_to_analyze = [token_symbol]
                else:
                    # If token not in supported futures, analyze ETH-USD as proxy
                    symbols_to_analyze = ['ETH-USD']
            else:
                # Analyze all supported futures markets (for general market analysis)
                symbols_to_analyze = list(self.supported_futures.keys())
            
            # Analyze the selected futures markets
            futures_analysis = {}
            for symbol in symbols_to_analyze:
                # Get futures-specific market data (simulated for now)
                futures_data = await self._fetch_futures_market_data(symbol, market_data)
                
                # Calculate technical indicators
                indicators = await self._calculate_technical_indicators(symbol, futures_data)
                
                # Classify market regime
                regime = self._classify_market_regime(indicators, futures_data)
                
                # Analyze funding rate opportunities
                funding_analysis = self._analyze_funding_opportunities(futures_data)
                
                # Calculate optimal leverage based on volatility
                optimal_leverage = self._calculate_optimal_leverage(indicators, futures_data)
                
                # Generate market structure analysis
                structure_analysis = self._analyze_market_structure(indicators, futures_data)
                
                futures_analysis[symbol] = {
                    'indicators': indicators,
                    'regime': regime,
                    'funding': funding_analysis,
                    'optimal_leverage': optimal_leverage,
                    'structure': structure_analysis,
                    'signal_strength': self._calculate_signal_strength(indicators, regime)
                }
                
                # Add to opportunities if strong signals
                if futures_analysis[symbol]['signal_strength'] > 0.7:
                    analysis['futures_opportunities'].append({
                        'symbol': symbol,
                        'direction': self._determine_trade_direction(indicators),
                        'strength': futures_analysis[symbol]['signal_strength'],
                        'leverage': optimal_leverage,
                        'reasoning': self._generate_trade_reasoning(indicators, regime)
                    })
                
                # Add funding arbitrage opportunities
                if funding_analysis['arbitrage_opportunity']:
                    analysis['funding_arbitrage'].append({
                        'symbol': symbol,
                        'funding_rate': futures_data.funding_rate,
                        'expected_duration': funding_analysis['duration_hours'],
                        'expected_return': funding_analysis['expected_return']
                    })
            
            analysis['technical_summary'] = futures_analysis
            
            # Determine overall market regime
            analysis['market_regime'] = self._determine_overall_regime(futures_analysis)
            
            # Calculate overall risk level
            analysis['risk_level'] = self._assess_futures_risk(futures_analysis)
            
            # Generate volatility regime assessment
            analysis['volatility_regime'] = self._assess_volatility_regime(futures_analysis)
            
            # Sort opportunities by signal strength
            analysis['futures_opportunities'].sort(
                key=lambda x: x['strength'], reverse=True
            )
            
            # Calculate agent-specific scores for token analysis
            technical_score = self._calculate_technical_score(futures_analysis)
            fundamental_score = self._calculate_fundamental_score(futures_analysis)
            
            # Add scores to analysis
            analysis['technical_score'] = technical_score
            analysis['fundamental_score'] = fundamental_score
            analysis['agent_strategy'] = 'aggressive_futures'
            analysis['agent_confidence'] = self._calculate_agent_confidence(futures_analysis)
            
            logger.info(f"Yuki futures analysis: {analysis['market_regime'].value} regime, "
                       f"{len(analysis['futures_opportunities'])} opportunities, "
                       f"{len(analysis['funding_arbitrage'])} funding plays, "
                       f"technical_score: {technical_score:.3f}, fundamental_score: {fundamental_score:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Yuki futures analysis: {e}")
            raise  # Let the error propagate instead of returning fallback data
    
    async def _fetch_futures_market_data(self, symbol: str, market_data: MarketData) -> FuturesMarketData:
        """
        Fetch real Hyperliquid futures market data.
        """
        try:
            if not self.hyperliquid_service:
                raise Exception("Hyperliquid service not available")
            
            # Get real futures data from Hyperliquid
            try:
                # Try live WebSocket data first (preferred)
                live_data = await self.hyperliquid_service.get_live_market_data(symbol)
                if live_data:
                    logger.info(f"Using live WebSocket data for {symbol}")
                else:
                    # Fallback to REST API data (still real data, just not live)
                    logger.info(f"Using REST API data for {symbol}")
                    live_data = await self.hyperliquid_service.get_market_data(symbol)
                    if not live_data:
                        raise Exception(f"No market data available for {symbol}")
            except Exception as e:
                raise Exception(f"Failed to get market data for {symbol}: {e}")
            
            # Get real funding rates
            funding_data = await self.hyperliquid_service.get_funding_rates([symbol])
            funding_info = funding_data.get(symbol, {})
            
            # Get real order book for spread analysis
            orderbook = await self.hyperliquid_service.get_orderbook(symbol)
            
            # Calculate real spread
            spread = 0.0
            if orderbook and orderbook.bids and orderbook.asks:
                best_bid = orderbook.bids[0].price if orderbook.bids else 0
                best_ask = orderbook.asks[0].price if orderbook.asks else 0
                if best_bid > 0 and best_ask > 0:
                    spread = (best_ask - best_bid) / best_bid
            
            return FuturesMarketData(
                symbol=symbol,
                price=live_data.price,
                funding_rate=funding_info.get('funding_rate', 0.0),
                funding_rate_8h=funding_info.get('funding_rate_8h', 0.0),
                open_interest=live_data.open_interest or 0.0,
                oi_change_24h=live_data.oi_change_24h or 0.0,
                volume_24h=live_data.volume_24h or 0.0,
                mark_price=live_data.mark_price or live_data.price,
                index_price=live_data.index_price or live_data.price,
                last_funding_time=live_data.last_funding_time or datetime.now(),
                next_funding_time=live_data.next_funding_time or datetime.now(),
                liquidation_threshold=0.05,  # Default liquidation threshold
                max_leverage=self.supported_futures[symbol]['max_leverage'],
                spread=spread
            )
            
        except Exception as e:
            logger.error(f"Error fetching real futures data for {symbol}: {e}")
            raise
    
    async def _calculate_technical_indicators(self, symbol: str, futures_data: FuturesMarketData) -> TechnicalIndicators:
        """
        Calculate comprehensive technical indicators using real historical data.
        """
        try:
            if not self.hyperliquid_service:
                raise Exception("Hyperliquid service not available")
            
            # Get real historical data from Hyperliquid
            historical_data = await self.hyperliquid_service.get_historical_data(
                symbol, 
                timeframe='1h', 
                limit=200  # Get 200 hours of data for indicators
            )
            
            if not historical_data or len(historical_data) < 50:
                raise Exception(f"Insufficient historical data for {symbol}: {len(historical_data) if historical_data else 0} data points")
            
            # Extract real price and volume data
            prices = [float(candle.close) for candle in historical_data]
            volumes = [float(candle.volume) for candle in historical_data]
            
            if not prices or len(prices) < 50:
                raise Exception(f"Invalid price data for {symbol}")
            
            indicators = TechnicalIndicators()
            
            # Moving Averages
            indicators.ema_20 = self._calculate_ema(prices, 20)
            indicators.ema_50 = self._calculate_ema(prices, 50)
            indicators.sma_200 = self._calculate_sma(prices, 200) if len(prices) >= 200 else prices[-1]
            
            # RSI and Stochastic
            indicators.rsi = self._calculate_rsi(prices, self.ta_config['rsi_period'])
            stoch_k, stoch_d = self._calculate_stochastic(prices, self.ta_config['stoch_k'], self.ta_config['stoch_d'])
            indicators.stoch_k = stoch_k
            indicators.stoch_d = stoch_d
            indicators.stoch_rsi = self._calculate_stoch_rsi(prices, 14)
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(
                prices, self.ta_config['macd_fast'], self.ta_config['macd_slow'], self.ta_config['macd_signal']
            )
            indicators.macd = macd_line
            indicators.macd_signal = signal_line
            indicators.macd_histogram = histogram
            
            # ADX for trend strength
            indicators.adx = self._calculate_adx(prices, self.ta_config['adx_period'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._calculate_bollinger_bands(
                prices, self.ta_config['bb_period'], self.ta_config['bb_std']
            )
            indicators.bb_upper = bb_upper
            indicators.bb_middle = bb_middle
            indicators.bb_lower = bb_lower
            indicators.bb_width = bb_width
            
            # ATR for volatility
            indicators.atr = self._calculate_atr(prices, self.ta_config['atr_period'])
            
            # VWAP
            indicators.vwap = self._calculate_vwap(prices, volumes)
            
            # Volume analysis using real data
            if volumes and len(volumes) > 0:
                avg_volume = sum(volumes) / len(volumes)
                indicators.volume_ratio = futures_data.volume_24h / avg_volume if avg_volume > 0 else 1.0
            else:
                indicators.volume_ratio = 1.0
            
            # Support/Resistance
            pivot, support1, resistance1 = self._calculate_pivot_points(prices)
            indicators.pivot_point = pivot
            indicators.support_1 = support1
            indicators.resistance_1 = resistance1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            raise  # Don't return default values, let the error propagate
    

    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        return sum(prices[-period:]) / period
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, prices: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(prices) < k_period:
            return 50.0, 50.0
        
        current_price = prices[-1]
        lowest_low = min(prices[-k_period:])
        highest_high = max(prices[-k_period:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified D% calculation - in production would use d_period for SMA of %K
        d_percent = k_percent  # TODO: Implement proper %D calculation using d_period
        
        return k_percent, d_percent
    
    def _calculate_stoch_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Stochastic RSI."""
        # Simplified implementation
        rsi = self._calculate_rsi(prices, period)
        # In practice, this would calculate stochastic of RSI values
        return min(100.0, max(0.0, rsi))
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation - in production would use signal_period for EMA
        signal_line = macd_line * 0.9  # TODO: Implement proper signal line using signal_period
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_adx(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average Directional Index."""
        if len(prices) < period + 1:
            return 25.0
        
        # Simplified ADX calculation
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_change = sum(price_changes[-period:]) / period
        
        # Normalize to 0-100 scale
        adx = min(100.0, (avg_change / prices[-1]) * 1000)
        
        return adx
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0.0
            return current_price, current_price, current_price, 0.0
        
        sma = self._calculate_sma(prices, period)
        recent_prices = prices[-period:]
        std = statistics.stdev(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        width = (upper_band - lower_band) / sma if sma > 0 else 0.0
        
        return upper_band, sma, lower_band, width
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            tr = abs(prices[i] - prices[i-1])
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        
        return sum(true_ranges[-period:]) / period
    
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate Volume Weighted Average Price."""
        if not prices or not volumes or len(prices) != len(volumes):
            return prices[-1] if prices else 0.0
        
        total_pv = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_pv / total_volume if total_volume > 0 else prices[-1]
    
    def _calculate_pivot_points(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate pivot points for support/resistance."""
        if len(prices) < 3:
            current_price = prices[-1] if prices else 0.0
            return current_price, current_price * 0.99, current_price * 1.01
        
        high = max(prices[-3:])
        low = min(prices[-3:])
        close = prices[-1]
        
        pivot = (high + low + close) / 3
        support1 = (2 * pivot) - high
        resistance1 = (2 * pivot) - low
        
        return pivot, support1, resistance1
    
    def _classify_market_regime(self, indicators: TechnicalIndicators, futures_data: FuturesMarketData) -> MarketRegime:
        """Classify current market regime for futures trading."""
        try:
            # Trend analysis using ADX and moving averages
            if indicators.adx > self.regime_thresholds['trending_adx']:
                if indicators.ema_20 > indicators.ema_50:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            
            # Volatility analysis using ATR and Bollinger Bands
            if indicators.atr > futures_data.price * self.regime_thresholds['volatile_atr_multiplier'] / 100:
                return MarketRegime.VOLATILE
            
            # Breakout detection using Bollinger Bands and volume
            if (indicators.bb_width > self.regime_thresholds['ranging_bb_width'] and 
                indicators.volume_ratio > self.regime_thresholds['breakout_volume_multiplier']):
                return MarketRegime.BREAKOUT
            
            # Default to ranging if no clear regime
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return MarketRegime.RANGING
    
    def _analyze_funding_opportunities(self, futures_data: FuturesMarketData) -> Dict[str, Any]:
        """Analyze funding rate arbitrage opportunities."""
        try:
            funding_rate = futures_data.funding_rate
            is_arbitrage = False
            expected_return = 0.0
            duration_hours = 8
            
            # High positive funding - consider shorting
            if funding_rate > self.funding_strategy['high_funding_threshold']:
                is_arbitrage = True
                expected_return = funding_rate * 3  # 3 funding periods (24h)
                duration_hours = self.funding_strategy['funding_arbitrage_min_duration']
            
            # High negative funding - consider longing
            elif funding_rate < self.funding_strategy['low_funding_threshold']:
                is_arbitrage = True
                expected_return = abs(funding_rate) * 3  # 3 funding periods (24h)
                duration_hours = self.funding_strategy['funding_arbitrage_min_duration']
            
            return {
                'arbitrage_opportunity': is_arbitrage,
                'funding_rate': funding_rate,
                'expected_return': expected_return,
                'duration_hours': duration_hours,
                'recommended_side': 'short' if funding_rate > 0 else 'long' if is_arbitrage else 'none'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing funding opportunities: {e}")
            return {'arbitrage_opportunity': False, 'funding_rate': 0.0}
    
    def _calculate_optimal_leverage(self, indicators: TechnicalIndicators, futures_data: FuturesMarketData) -> float:
        """Calculate optimal leverage based on volatility and market conditions."""
        try:
            # Base leverage calculation using volatility (ATR)
            if futures_data.price <= 0:
                return 2.0  # Conservative default if price is invalid
                
            volatility_ratio = indicators.atr / futures_data.price
            
            # Lower leverage for higher volatility
            if volatility_ratio > 0.05:  # High volatility
                base_leverage = self.futures_risk_params.get('min_leverage', 2.0)
            elif volatility_ratio > 0.03:  # Medium volatility
                base_leverage = 4.0
            else:  # Low volatility
                base_leverage = 6.0
            
            # Adjust based on trend strength (ADX)
            if indicators.adx > 30:  # Strong trend
                leverage_multiplier = 1.2
            elif indicators.adx < 20:  # Weak trend
                leverage_multiplier = 0.8
            else:
                leverage_multiplier = 1.0
            
            # Apply multiplier
            optimal_leverage = base_leverage * leverage_multiplier
            
            # Ensure within bounds
            min_leverage = self.futures_risk_params.get('min_leverage', 2.0)
            max_leverage = min(
                self.futures_risk_params.get('max_leverage', 10.0),
                futures_data.max_leverage
            )
            
            return max(min_leverage, min(max_leverage, optimal_leverage))
            
        except Exception as e:
            logger.error(f"Error calculating optimal leverage: {e}")
            return 2.0  # Conservative default
    
    def _analyze_market_structure(self, indicators: TechnicalIndicators, futures_data: FuturesMarketData) -> Dict[str, Any]:
        """Analyze market structure for entry/exit signals."""
        try:
            structure = {
                'trend_direction': 'neutral',
                'strength': 0.5,
                'support_level': indicators.support_1,
                'resistance_level': indicators.resistance_1,
                'key_levels': [],
                'momentum_alignment': False
            }
            
            # Determine trend direction
            if indicators.ema_20 > indicators.ema_50 > indicators.sma_200:
                structure['trend_direction'] = 'bullish'
                structure['strength'] = min(1.0, indicators.adx / 50.0)
            elif indicators.ema_20 < indicators.ema_50 < indicators.sma_200:
                structure['trend_direction'] = 'bearish'
                structure['strength'] = min(1.0, indicators.adx / 50.0)
            
            # Check momentum alignment
            price_above_vwap = futures_data.price > indicators.vwap
            rsi_bullish = 30 < indicators.rsi < 70  # Not overbought/oversold
            macd_bullish = indicators.macd > indicators.macd_signal
            
            if structure['trend_direction'] == 'bullish':
                structure['momentum_alignment'] = price_above_vwap and rsi_bullish and macd_bullish
            elif structure['trend_direction'] == 'bearish':
                structure['momentum_alignment'] = not price_above_vwap and rsi_bullish and not macd_bullish
            
            # Identify key levels
            structure['key_levels'] = [
                {'level': indicators.vwap, 'type': 'vwap'},
                {'level': indicators.pivot_point, 'type': 'pivot'},
                {'level': indicators.bb_upper, 'type': 'bb_upper'},
                {'level': indicators.bb_lower, 'type': 'bb_lower'}
            ]
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {'trend_direction': 'neutral', 'strength': 0.5}
    
    def _calculate_signal_strength(self, indicators: TechnicalIndicators, regime: MarketRegime) -> float:
        """Calculate overall signal strength for futures trading."""
        try:
            strength_components = []
            
            # RSI component
            if 40 <= indicators.rsi <= 60:
                rsi_score = 0.8  # Neutral RSI is good
            elif 30 <= indicators.rsi <= 70:
                rsi_score = 0.6
            else:
                rsi_score = 0.3  # Extreme RSI
            strength_components.append(rsi_score * 0.2)
            
            # MACD component
            if abs(indicators.macd_histogram) > 0.1:
                macd_score = 0.8  # Strong MACD signal
            else:
                macd_score = 0.5
            strength_components.append(macd_score * 0.2)
            
            # ADX component (trend strength)
            adx_score = min(1.0, indicators.adx / 50.0)
            strength_components.append(adx_score * 0.3)
            
            # Bollinger Bands component
            if indicators.bb_width > 0.04:  # Wide bands = high volatility
                bb_score = 0.7
            elif indicators.bb_width < 0.02:  # Narrow bands = low volatility
                bb_score = 0.4
            else:
                bb_score = 0.6
            strength_components.append(bb_score * 0.15)
            
            # Volume component
            volume_score = min(1.0, indicators.volume_ratio / 2.0)
            strength_components.append(volume_score * 0.15)
            
            # Regime bonus
            regime_bonus = {
                MarketRegime.TRENDING_UP: 0.1,
                MarketRegime.TRENDING_DOWN: 0.1,
                MarketRegime.BREAKOUT: 0.15,
                MarketRegime.VOLATILE: 0.05,
                MarketRegime.RANGING: 0.0
            }.get(regime, 0.0)
            
            total_strength = sum(strength_components) + regime_bonus
            return min(1.0, max(0.0, total_strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _determine_trade_direction(self, indicators: TechnicalIndicators) -> str:
        """Determine trade direction based on technical indicators."""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # EMA alignment
            if indicators.ema_20 > indicators.ema_50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI
            if indicators.rsi < 40:
                bullish_signals += 1
            elif indicators.rsi > 60:
                bearish_signals += 1
            
            # MACD
            if indicators.macd > indicators.macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Stochastic
            if indicators.stoch_k < 20:
                bullish_signals += 1
            elif indicators.stoch_k > 80:
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                return 'long'
            elif bearish_signals > bullish_signals:
                return 'short'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining trade direction: {e}")
            return 'neutral'
    
    def _generate_trade_reasoning(self, indicators: TechnicalIndicators, regime: MarketRegime) -> str:
        """Generate human-readable reasoning for trade signals."""
        try:
            reasons = []
            
            # Trend analysis
            if indicators.ema_20 > indicators.ema_50:
                reasons.append("bullish EMA alignment")
            elif indicators.ema_20 < indicators.ema_50:
                reasons.append("bearish EMA alignment")
            
            # Momentum
            if indicators.rsi < 30:
                reasons.append("oversold RSI")
            elif indicators.rsi > 70:
                reasons.append("overbought RSI")
            
            # MACD
            if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
                reasons.append("bullish MACD crossover")
            elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
                reasons.append("bearish MACD crossover")
            
            # Market regime
            regime_descriptions = {
                MarketRegime.TRENDING_UP: "uptrending market",
                MarketRegime.TRENDING_DOWN: "downtrending market",
                MarketRegime.BREAKOUT: "breakout conditions",
                MarketRegime.VOLATILE: "high volatility environment",
                MarketRegime.RANGING: "ranging market conditions"
            }
            reasons.append(regime_descriptions.get(regime, "neutral conditions"))
            
            return ", ".join(reasons) if reasons else "mixed technical signals"
            
        except Exception as e:
            logger.error(f"Error generating trade reasoning: {e}")
            return "technical analysis indicates opportunity"
    
    def _determine_overall_regime(self, futures_analysis: Dict[str, Any]) -> MarketRegime:
        """Determine overall market regime from individual symbol analysis."""
        try:
            regime_counts = {}
            
            for symbol_data in futures_analysis.values():
                regime = symbol_data.get('regime', MarketRegime.RANGING)
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Return most common regime
            if regime_counts:
                return max(regime_counts, key=regime_counts.get)
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.error(f"Error determining overall regime: {e}")
            return MarketRegime.RANGING
    
    def _assess_futures_risk(self, futures_analysis: Dict[str, Any]) -> RiskLevel:
        """Assess overall risk level for futures trading."""
        try:
            risk_factors = 0
            total_symbols = len(futures_analysis)
            
            if total_symbols == 0:
                return RiskLevel.HIGH
            
            # Count high volatility conditions
            volatile_count = sum(1 for data in futures_analysis.values() 
                               if data.get('regime') == MarketRegime.VOLATILE)
            
            if volatile_count / total_symbols > 0.5:
                risk_factors += 2
            
            # Check for conflicting signals
            signal_strengths = [data.get('signal_strength', 0.5) for data in futures_analysis.values()]
            avg_strength = sum(signal_strengths) / len(signal_strengths)
            
            if avg_strength < 0.4:
                risk_factors += 1
            
            # Assess based on risk factors
            if risk_factors >= 3:
                return RiskLevel.EXTREME
            elif risk_factors >= 2:
                return RiskLevel.HIGH
            elif risk_factors >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error assessing futures risk: {e}")
            return RiskLevel.HIGH
    
    def _assess_volatility_regime(self, futures_analysis: Dict[str, Any]) -> str:
        """Assess current volatility regime across futures markets."""
        try:
            volatile_symbols = sum(1 for data in futures_analysis.values() 
                                 if data.get('regime') == MarketRegime.VOLATILE)
            total_symbols = len(futures_analysis)
            
            if total_symbols == 0:
                return 'unknown'
            
            volatility_ratio = volatile_symbols / total_symbols
            
            if volatility_ratio > 0.6:
                return 'high_volatility'
            elif volatility_ratio > 0.3:
                return 'medium_volatility'
            else:
                return 'low_volatility'
                
        except Exception as e:
            logger.error(f"Error assessing volatility regime: {e}")
            return 'medium_volatility'
    
    async def generate_signals(self, market_data: MarketData) -> List[TradingSignal]:
        """
        Generate aggressive trading signals based on momentum and volatility.
        
        Yuki's signal generation focuses on:
        - Strong momentum entries
        - Volatility breakouts
        - Quick profit-taking
        - Dynamic position sizing
        """
        signals = []
        
        try:
            # Get market analysis
            analysis = await self.analyze_market(market_data)
            
            if analysis.get('error'):
                logger.error(f"Cannot generate signals due to analysis error: {analysis['error']}")
                return signals
            
            # Get current portfolio positions
            portfolio_positions = await self._get_portfolio_positions()
            current_holdings = {pos.token_symbol: pos for pos in portfolio_positions}
            
            # Generate momentum-based buy signals
            momentum_opportunities = analysis.get('momentum_opportunities', [])
            for opportunity in momentum_opportunities[:5]:  # Top 5 opportunities
                token_symbol = opportunity['token']
                
                # Check current position
                if token_symbol in current_holdings:
                    current_allocation = current_holdings[token_symbol].allocation_percent
                    if current_allocation >= 30:  # Max 30% per token for Yuki
                        continue
                
                # Generate aggressive buy signal
                signal = TradingSignal(
                    signal_type=SignalType.BUY,
                    token_symbol=token_symbol,
                    confidence=opportunity['confidence'],
                    reasoning=opportunity['reasoning'],
                    risk_level=self._determine_aggressive_risk_level(opportunity['score']),
                    metadata={
                        'momentum_score': opportunity['momentum_score'],
                        'volatility': opportunity['volatility'],
                        'volume_surge': opportunity.get('volume_surge', False),
                        'analysis_type': 'momentum_aggressive',
                        'market_sentiment': analysis['market_sentiment']
                    }
                )
                signals.append(signal)
            
            # Generate breakout signals
            breakout_candidates = analysis.get('breakout_candidates', [])
            for candidate in breakout_candidates[:3]:  # Top 3 breakouts
                if candidate['token'] not in current_holdings:
                    signal = TradingSignal(
                        signal_type=SignalType.BUY,
                        token_symbol=candidate['token'],
                        confidence=candidate['confidence'],
                        reasoning=f"Breakout signal: {candidate['pattern']} with {candidate['strength']:.2f} strength",
                        risk_level=RiskLevel.HIGH,
                        metadata={
                            'breakout_pattern': candidate['pattern'],
                            'breakout_strength': candidate['strength'],
                            'analysis_type': 'breakout'
                        }
                    )
                    signals.append(signal)
            
            # Existing-position lifecycle decisions are owned by the live Yuki
            # ReAct monitor. The legacy signal generator must not manufacture
            # timer/PnL/momentum exits outside that evidence-and-memory cycle.
            
            # Apply aggressive filtering
            filtered_signals = self._apply_aggressive_filters(signals, analysis)
            
            # Enhance signals with LLM reasoning if available
            if self.use_llm_enhancement and self.llm_service and filtered_signals:
                try:
                    enhanced_signals = await self._enhance_signals_with_llm(
                        filtered_signals, market_data, analysis
                    )
                    logger.info(f"Yuki generated {len(enhanced_signals)} LLM-enhanced aggressive signals")
                    return enhanced_signals
                except Exception as e:
                    logger.error(f"LLM enhancement failed: {e}. Using algorithmic signals only.")
            
            logger.info(f"Yuki generated {len(filtered_signals)} aggressive trading signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating Yuki signals: {e}")
            return []
    
    async def _enhance_signals_with_llm(
        self, 
        signals: List[TradingSignal], 
        market_data: MarketData, 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """
        Enhance algorithmic trading signals with LLM reasoning.
        
        This method combines Yuki's technical analysis with LLM-based market context,
        narrative understanding, and advanced reasoning capabilities.
        """
        try:
            enhanced_signals = []
            
            # Extract overall market context
            market_sentiment = analysis.get('market_sentiment', 'neutral')
            market_regime = analysis.get('market_regime', 'sideways')
            volatility_regime = analysis.get('volatility_regime', 'medium_volatility')
            
            # Create market narrative
            funding_opps = len(analysis.get('funding_arbitrage', []))
            futures_opps = len(analysis.get('futures_opportunities', []))
            
            market_narrative = (
                f"Market regime: {market_regime}, "
                f"volatility: {volatility_regime}, "
                f"sentiment: {market_sentiment}, "
                f"{futures_opps} futures opportunities, "
                f"{funding_opps} funding arbitrage plays detected"
            )
            
            # Process each signal with LLM enhancement
            for signal in signals:
                try:
                    # Build market context for LLM
                    price = float(market_data.token_prices.get(signal.token_symbol, 0))
                    price_change = float(market_data.price_changes.get(signal.token_symbol, 0))
                    volume = float(market_data.volume_24h.get(signal.token_symbol, 0))
                    volatility = market_data.volatility.get(signal.token_symbol, 0.1)
                    
                    # Get technical indicators from signal metadata
                    metadata = signal.metadata or {}
                    rsi = metadata.get('rsi', 50.0)
                    macd = metadata.get('macd', 0.0)
                    bb_position = metadata.get('bb_position', 0.5)
                    
                    # Build LLM context
                    context = MarketContext(
                        timestamp=datetime.now(),
                        symbol=signal.token_symbol,
                        current_price=price,
                        price_change_24h=price_change,
                        volume_24h=volume,
                        volatility=volatility,
                        rsi=rsi,
                        macd=macd,
                        bb_position=bb_position,
                        fear_greed_index=market_data.trend_indicators.get('fear_greed_index', 50),
                        social_sentiment=market_data.trend_indicators.get('social_sentiment', 'neutral'),
                        news_sentiment=market_data.trend_indicators.get('news_sentiment', 'neutral'),
                        market_narrative=market_narrative,
                        funding_rate=metadata.get('funding_rate'),
                        market_cap=float(market_data.market_cap.get(signal.token_symbol, 0))
                    )
                    
                    # Get LLM analysis
                    llm_analysis = await self.llm_service.analyze_trading_opportunity(
                        context, 
                        additional_context=f"Yuki algorithmic signal: {signal.signal_type.value} "
                                         f"with {signal.confidence:.2f} confidence. "
                                         f"Reasoning: {signal.reasoning}"
                    )
                    
                    # Create enhanced signal
                    enhanced_signal = self._combine_algorithmic_and_llm_signals(
                        signal, llm_analysis, context
                    )
                    
                    enhanced_signals.append(enhanced_signal)
                    
                except Exception as e:
                    logger.error(f"Error enhancing signal for {signal.token_symbol}: {e}")
                    # Fallback to original signal
                    enhanced_signals.append(signal)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error in LLM signal enhancement: {e}")
            return signals  # Return original signals on error
    
    def _combine_algorithmic_and_llm_signals(
        self, 
        algo_signal: TradingSignal, 
        llm_analysis, 
        context: MarketContext
    ) -> TradingSignal:
        """
        Combine algorithmic and LLM signals into an enhanced trading signal.
        
        This method weighs both algorithmic and LLM recommendations to create
        a more robust trading signal with enhanced reasoning.
        """
        try:
            # Determine signal agreement
            algo_direction = algo_signal.signal_type.value.upper()
            llm_direction = llm_analysis.recommendation.upper()
            
            # Calculate combined confidence
            if algo_direction == llm_direction:
                # Signals agree - boost confidence
                combined_confidence = min(0.95, (algo_signal.confidence + llm_analysis.confidence) * 0.6)
                signal_agreement = True
            else:
                # Signals disagree - reduce confidence and prefer more conservative approach
                combined_confidence = max(0.3, (algo_signal.confidence + llm_analysis.confidence) * 0.4)
                signal_agreement = False
                
                # In case of disagreement, prefer HOLD or the more conservative signal
                if algo_direction in ['STOP_LOSS', 'TAKE_PROFIT'] or llm_direction == 'HOLD':
                    final_direction = algo_direction  # Respect exit signals
                elif algo_direction == 'HOLD' or llm_direction in ['SELL', 'HOLD']:
                    final_direction = 'HOLD'  # Be conservative on disagreement
                else:
                    final_direction = llm_direction  # LLM has more context
            
            # Determine final signal type
            if signal_agreement:
                final_signal_type = algo_signal.signal_type
            else:
                # Map disagreement resolution
                if final_direction == 'BUY':
                    final_signal_type = SignalType.BUY
                elif final_direction == 'SELL':
                    final_signal_type = SignalType.SELL
                elif final_direction in ['STOP_LOSS']:
                    final_signal_type = SignalType.STOP_LOSS
                elif final_direction in ['TAKE_PROFIT']:
                    final_signal_type = SignalType.TAKE_PROFIT
                else:
                    final_signal_type = SignalType.SELL  # Default to exit on uncertainty
            
            # Enhanced reasoning
            agreement_note = "✓ Signals aligned" if signal_agreement else "⚠ Signals diverged"
            
            enhanced_reasoning = (
                f"LLM-Enhanced Analysis ({agreement_note}): "
                f"Algorithmic: {algo_signal.reasoning} | "
                f"LLM Context: {llm_analysis.reasoning[:200]}... | "
                f"Key factors: {', '.join(llm_analysis.key_factors[:3])}"
            )
            
            # Enhanced metadata
            enhanced_metadata = {
                **(algo_signal.metadata or {}),
                'llm_recommendation': llm_analysis.recommendation,
                'llm_confidence': llm_analysis.confidence,
                'llm_risk_assessment': llm_analysis.risk_assessment,
                'market_regime': llm_analysis.market_regime,
                'signal_agreement': signal_agreement,
                'algo_confidence': algo_signal.confidence,
                'combined_confidence': combined_confidence,
                'llm_key_factors': llm_analysis.key_factors,
                'stop_loss_level': llm_analysis.stop_loss_level,
                'take_profit_level': llm_analysis.take_profit_level,
                'llm_position_sizing': llm_analysis.position_sizing,
                'enhanced_by_llm': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Determine risk level (use higher of the two)
            risk_levels = {
                'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'EXTREME': 4
            }
            
            algo_risk_level = risk_levels.get(algo_signal.risk_level.value.upper(), 2)
            llm_risk_level = risk_levels.get(llm_analysis.risk_assessment.upper(), 2)
            
            final_risk_value = max(algo_risk_level, llm_risk_level)
            final_risk_level = [k for k, v in risk_levels.items() if v == final_risk_value][0]
            
            # Create enhanced signal
            enhanced_signal = TradingSignal(
                signal_type=final_signal_type,
                token_symbol=algo_signal.token_symbol,
                confidence=combined_confidence,
                reasoning=enhanced_reasoning,
                risk_level=RiskLevel(final_risk_level.lower()),
                metadata=enhanced_metadata
            )
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            # Return original signal with LLM metadata
            return TradingSignal(
                signal_type=algo_signal.signal_type,
                token_symbol=algo_signal.token_symbol,
                confidence=algo_signal.confidence,
                reasoning=f"{algo_signal.reasoning} (LLM enhancement failed)",
                risk_level=algo_signal.risk_level,
                metadata={
                    **(algo_signal.metadata or {}),
                    'llm_enhancement_error': str(e),
                    'enhanced_by_llm': False
                }
            )
    
    def _analyze_market_momentum(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze overall market momentum."""
        try:
            price_changes = [float(change) for change in market_data.price_changes.values()]
            avg_change = statistics.mean(price_changes)
            
            # Count strong movers
            strong_up = sum(1 for change in price_changes if change > self.momentum_thresholds['momentum_entry'])
            strong_down = sum(1 for change in price_changes if change < -self.momentum_thresholds['momentum_entry'])
            total_tokens = len(price_changes)
            
            # Calculate momentum strength
            if strong_up > strong_down:
                strength = (strong_up / total_tokens) * (1 + abs(avg_change) / 10)
                sentiment = 'bullish' if strength > 0.6 else 'cautiously_bullish'
            elif strong_down > strong_up:
                strength = (strong_down / total_tokens) * (1 + abs(avg_change) / 10)
                sentiment = 'bearish' if strength > 0.6 else 'cautiously_bearish'
            else:
                strength = 0.5
                sentiment = 'neutral'
            
            return {
                'strength': min(1.0, strength),
                'sentiment': sentiment,
                'avg_change': avg_change,
                'strong_movers_up': strong_up,
                'strong_movers_down': strong_down,
                'momentum_divergence': abs(strong_up - strong_down) / total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market momentum: {e}")
            return {'strength': 0.5, 'sentiment': 'neutral'}
    
    def _analyze_volatility_opportunities(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze volatility for trading opportunities."""
        try:
            volatilities = list(market_data.volatility.values())
            avg_volatility = statistics.mean(volatilities)
            
            opportunities = []
            total_score = 0
            
            for token, volatility in market_data.volatility.items():
                if (self.volatility_preferences['min_volatility'] <= volatility <= 
                    self.volatility_preferences['max_volatility']):
                    
                    # Score volatility opportunity
                    if (self.volatility_preferences['sweet_spot_min'] <= volatility <= 
                        self.volatility_preferences['sweet_spot_max']):
                        vol_score = 0.9  # Sweet spot
                    elif volatility > self.volatility_preferences['sweet_spot_max']:
                        vol_score = 0.7  # High but manageable
                    else:
                        vol_score = 0.6  # Lower but acceptable
                    
                    opportunities.append({
                        'token': token,
                        'volatility': volatility,
                        'score': vol_score,
                        'relative_volatility': volatility / avg_volatility
                    })
                    total_score += vol_score
            
            return {
                'score': total_score / len(opportunities) if opportunities else 0,
                'opportunities': sorted(opportunities, key=lambda x: x['score'], reverse=True),
                'avg_volatility': avg_volatility,
                'high_vol_count': len([v for v in volatilities if v > avg_volatility * 1.5])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility opportunities: {e}")
            return {'score': 0, 'opportunities': []}
    
    def _detect_volume_surges(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Detect tokens with volume surges."""
        try:
            volumes = [float(vol) for vol in market_data.volume_24h.values()]
            avg_volume = statistics.mean(volumes)
            
            surges = []
            for token, volume in market_data.volume_24h.items():
                volume_float = float(volume)
                if volume_float > avg_volume * self.momentum_thresholds['volume_surge']:
                    surge_ratio = volume_float / avg_volume
                    surges.append({
                        'token': token,
                        'volume': volume_float,
                        'surge_ratio': surge_ratio,
                        'surge_strength': min(1.0, surge_ratio / 3.0)  # Normalize
                    })
            
            return sorted(surges, key=lambda x: x['surge_ratio'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting volume surges: {e}")
            return []
    
    def _analyze_token_opportunity(self, token: str, market_data: MarketData) -> Dict[str, Any]:
        """Analyze individual token for aggressive trading opportunity."""
        try:
            price_change = float(market_data.price_changes.get(token, Decimal('0')))
            volatility = market_data.volatility.get(token, 0.1)
            volume = float(market_data.volume_24h.get(token, Decimal('0')))
            
            # Calculate component scores
            momentum_score = 0.5
            if price_change > self.momentum_thresholds['momentum_entry']:
                momentum_score = min(1.0, price_change / self.momentum_thresholds['strong_momentum'])
            elif price_change < -self.momentum_thresholds['momentum_entry']:
                momentum_score = 0.2  # Negative momentum
            
            # Volatility score (Yuki likes volatility)
            if (self.volatility_preferences['sweet_spot_min'] <= volatility <= 
                self.volatility_preferences['sweet_spot_max']):
                volatility_score = 0.9
            elif volatility > self.volatility_preferences['sweet_spot_max']:
                volatility_score = 0.7
            else:
                volatility_score = 0.4
            
            # Volume score
            avg_volume = statistics.mean([float(v) for v in market_data.volume_24h.values()])
            volume_score = min(1.0, volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # Market sentiment influence
            overall_sentiment = market_data.trend_indicators.get('overall_sentiment', 'neutral')
            sentiment_multiplier = {
                'bullish': 1.2,
                'neutral': 1.0,
                'bearish': 0.8
            }.get(overall_sentiment, 1.0)
            
            # Calculate overall opportunity score
            opportunity_score = (
                momentum_score * self.opportunity_weights['momentum'] +
                volatility_score * self.opportunity_weights['volatility'] +
                volume_score * self.opportunity_weights['volume'] +
                0.7 * self.opportunity_weights['market_sentiment']  # Base sentiment score
            ) * sentiment_multiplier
            
            # Determine confidence and reasoning
            confidence = min(0.95, opportunity_score)
            
            reasoning_parts = []
            if momentum_score > 0.7:
                reasoning_parts.append(f"strong momentum ({price_change:+.1f}%)")
            if volatility_score > 0.8:
                reasoning_parts.append(f"optimal volatility ({volatility:.3f})")
            if volume_score > 0.7:
                reasoning_parts.append("high volume")
            
            reasoning = f"Aggressive opportunity: {', '.join(reasoning_parts) if reasoning_parts else 'mixed signals'}"
            
            return {
                'token': token,
                'score': opportunity_score,
                'confidence': confidence,
                'momentum_score': momentum_score,
                'volatility': volatility,
                'volume_surge': volume_score > 0.8,
                'price_change': price_change,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity for {token}: {e}")
            return {
                'token': token,
                'score': 0.5,
                'confidence': 0.5,
                'reasoning': 'Analysis error'
            }
    
    def _identify_breakout_candidates(self, market_data: MarketData) -> List[Dict[str, Any]]:
        """Identify potential breakout opportunities."""
        try:
            candidates = []
            
            for token in self.high_potential_tokens:
                price_change = float(market_data.price_changes.get(token, Decimal('0')))
                volatility = market_data.volatility.get(token, 0.1)
                volume = float(market_data.volume_24h.get(token, Decimal('0')))
                
                # Look for breakout patterns
                if price_change > self.momentum_thresholds['strong_momentum']:
                    # Strong upward breakout
                    pattern = 'upward_breakout'
                    strength = min(1.0, price_change / (self.momentum_thresholds['strong_momentum'] * 2))
                    confidence = 0.8 * strength
                    
                elif (price_change > self.momentum_thresholds['momentum_entry'] and 
                      volatility > self.volatility_preferences['sweet_spot_min']):
                    # Moderate breakout with volatility
                    pattern = 'momentum_breakout'
                    strength = 0.7
                    confidence = 0.7
                    
                else:
                    continue
                
                # Volume confirmation
                avg_volume = statistics.mean([float(v) for v in market_data.volume_24h.values()])
                if volume > avg_volume * 1.5:
                    confidence += 0.1
                    strength += 0.1
                
                candidates.append({
                    'token': token,
                    'pattern': pattern,
                    'strength': min(1.0, strength),
                    'confidence': min(0.95, confidence),
                    'price_change': price_change,
                    'volatility': volatility,
                    'volume_ratio': volume / avg_volume if avg_volume > 0 else 1
                })
            
            return sorted(candidates, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error identifying breakout candidates: {e}")
            return []
    
    def _assess_aggressive_risk(
        self, 
        momentum_analysis: Dict[str, Any], 
        volatility_analysis: Dict[str, Any], 
        volume_surges: List[Dict[str, Any]]
    ) -> RiskLevel:
        """Assess risk level for aggressive trading strategy."""
        try:
            risk_score = 0
            
            # Momentum risk factors
            momentum_strength = momentum_analysis.get('strength', 0.5)
            if momentum_strength < 0.3:
                risk_score += 2  # Low momentum = higher risk for aggressive strategy
            elif momentum_strength > 0.8:
                risk_score += 1  # Very high momentum can be risky too
            
            # Volatility risk factors
            avg_volatility = volatility_analysis.get('avg_volatility', 0.1)
            if avg_volatility > 0.3:
                risk_score += 2  # Excessive volatility
            elif avg_volatility < 0.05:
                risk_score += 1  # Too low volatility for aggressive strategy
            
            # Volume surge risk
            if len(volume_surges) > 5:
                risk_score += 1  # Too many volume spikes can indicate instability
            elif len(volume_surges) == 0:
                risk_score += 1  # No volume activity
            
            # Market divergence
            divergence = momentum_analysis.get('momentum_divergence', 0)
            if divergence > 0.7:
                risk_score += 1  # High divergence = uncertain market
            
            # Determine risk level
            if risk_score >= 4:
                return RiskLevel.EXTREME
            elif risk_score >= 2:
                return RiskLevel.HIGH
            elif risk_score >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error assessing aggressive risk: {e}")
            return RiskLevel.HIGH
    
    def _determine_aggressive_risk_level(self, opportunity_score: float) -> RiskLevel:
        """Determine risk level for aggressive signals."""
        if opportunity_score >= 0.85:
            return RiskLevel.MEDIUM  # Even high-confidence aggressive trades are risky
        elif opportunity_score >= 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    async def _generate_aggressive_exit_signals(
        self, 
        position, 
        market_data: MarketData, 
        _: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Disabled: the live position ReAct cycle exclusively owns exits."""
        return []

        # Retained temporarily as historical policy context; unreachable by
        # design so no alternate agent loop can emit an automatic live exit.
        signals = []
        
        try:
            token_symbol = position.token_symbol
            price_change = float(market_data.price_changes.get(token_symbol, Decimal('0')))
            
            # Quick take-profit on momentum
            if (position.pnl_percent >= self.risk_params.take_profit_percent * 0.6 and  # 60% of target
                price_change > self.momentum_thresholds['momentum_entry']):
                
                signal = TradingSignal(
                    signal_type=SignalType.TAKE_PROFIT,
                    token_symbol=token_symbol,
                    confidence=0.9,
                    reasoning=f"Quick profit-taking at {position.pnl_percent:.1f}% with continued momentum",
                    risk_level=RiskLevel.LOW,
                    metadata={
                        'current_pnl': position.pnl_percent,
                        'momentum_continuation': True
                    }
                )
                signals.append(signal)
            
            # Aggressive stop-loss
            elif position.pnl_percent <= -self.risk_params.stop_loss_percent:
                signal = TradingSignal(
                    signal_type=SignalType.STOP_LOSS,
                    token_symbol=token_symbol,
                    confidence=1.0,
                    reasoning=f"Aggressive stop-loss at {position.pnl_percent:.1f}% loss",
                    risk_level=RiskLevel.LOW,
                    metadata={'current_pnl': position.pnl_percent}
                )
                signals.append(signal)
            
            # Exit on momentum reversal
            elif (price_change < self.momentum_thresholds['momentum_exit'] and 
                  position.pnl_percent > 5):  # Only if in profit
                
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    token_symbol=token_symbol,
                    confidence=0.8,
                    reasoning=f"Momentum reversal detected ({price_change:+.1f}%), securing profits",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={
                        'momentum_reversal': True,
                        'price_change': price_change,
                        'current_pnl': position.pnl_percent
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating aggressive exit signals for {position.token_symbol}: {e}")
            return []
    
    def _apply_aggressive_filters(
        self, 
        signals: List[TradingSignal], 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Apply aggressive trading filters to signals."""
        try:
            risk_assessment = analysis.get('risk_assessment', RiskLevel.MEDIUM)
            momentum_strength = analysis.get('momentum_strength', 0.5)
            
            filtered_signals = []
            buy_count = 0
            
            for signal in signals:
                should_include = True
                
                # In extreme risk conditions, only take exit signals
                if risk_assessment == RiskLevel.EXTREME:
                    if signal.signal_type == SignalType.BUY:
                        should_include = False
                
                # Limit buy signals in low momentum markets
                if (momentum_strength < 0.4 and 
                    signal.signal_type == SignalType.BUY and 
                    signal.confidence < 0.8):
                    should_include = False
                
                # Limit total number of buy signals
                if (signal.signal_type == SignalType.BUY):
                    if buy_count >= 4:  # Max 4 aggressive buy signals
                        should_include = False
                    else:
                        buy_count += 1
                
                # Boost confidence for high-momentum signals
                if (signal.signal_type == SignalType.BUY and 
                    momentum_strength > 0.7 and 
                    signal.confidence > 0.7):
                    signal.confidence = min(0.95, signal.confidence + 0.1)
                
                if should_include:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error applying aggressive filters: {e}")
            return signals
    
    def _determine_futures_risk_level(self, strength: float, leverage: float) -> RiskLevel:
        """Determine risk level for futures signals based on strength and leverage."""
        try:
            # Higher leverage = higher risk
            leverage_risk = leverage / 10.0  # Normalize to 0-1 scale
            
            # Lower strength = higher risk
            strength_risk = 1.0 - strength
            
            # Combined risk score
            total_risk = (leverage_risk + strength_risk) / 2.0
            
            if total_risk > 0.7:
                return RiskLevel.EXTREME
            elif total_risk > 0.5:
                return RiskLevel.HIGH
            elif total_risk > 0.3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining futures risk level: {e}")
            return RiskLevel.HIGH
    
    async def _generate_futures_exit_signals(
        self, 
        position, 
        _: MarketData, 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Disabled: the live position ReAct cycle exclusively owns exits."""
        return []

        # Retained temporarily as historical policy context; unreachable by
        # design so no alternate agent loop can emit an automatic live exit.
        signals = []
        
        try:
            token_symbol = position.token_symbol
            
            # Check if we have futures analysis for this token
            futures_analysis = analysis.get('technical_summary', {})
            token_analysis = None
            
            # Find matching futures symbol
            for symbol, data in futures_analysis.items():
                if symbol.startswith(token_symbol):
                    token_analysis = data
                    break
            
            if not token_analysis:
                return signals
            
            # Get technical indicators
            indicators = token_analysis.get('indicators', TechnicalIndicators())
            signal_strength = token_analysis.get('signal_strength', 0.5)
            
            # Take profit based on technical signals and leverage
            leverage_multiplier = 1.0  # Would get from position metadata in real implementation
            if (position.pnl_percent >= self.risk_params.take_profit_percent * 0.5 and  # 50% target for leveraged
                signal_strength < 0.4):  # Weak signals suggest exit
                
                signal = TradingSignal(
                    signal_type=SignalType.TAKE_PROFIT,
                    token_symbol=token_symbol,
                    confidence=0.8,
                    reasoning=f"Futures profit-taking: {position.pnl_percent:.1f}% gain with weakening signals",
                    risk_level=RiskLevel.LOW,
                    metadata={
                        'current_pnl': position.pnl_percent,
                        'signal_strength': signal_strength,
                        'leverage': leverage_multiplier
                    }
                )
                signals.append(signal)
            
            # Stop loss for futures (tighter due to leverage)
            elif position.pnl_percent <= -self.risk_params.stop_loss_percent * 0.8:  # 80% of normal stop
                signal = TradingSignal(
                    signal_type=SignalType.STOP_LOSS,
                    token_symbol=token_symbol,
                    confidence=1.0,
                    reasoning=f"Futures stop-loss: {position.pnl_percent:.1f}% loss (leveraged position)",
                    risk_level=RiskLevel.LOW,
                    metadata={
                        'current_pnl': position.pnl_percent,
                        'leverage': leverage_multiplier
                    }
                )
                signals.append(signal)
            
            # Technical exit on trend reversal
            elif (indicators.rsi > 80 and position.pnl_percent > 10):  # Overbought with profits
                signal = TradingSignal(
                    signal_type=SignalType.SELL,
                    token_symbol=token_symbol,
                    confidence=0.7,
                    reasoning=f"Technical exit: RSI overbought ({indicators.rsi:.1f}) with profits",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={
                        'rsi': indicators.rsi,
                        'current_pnl': position.pnl_percent
                    }
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating futures exit signals for {position.token_symbol}: {e}")
            return []
    
    def _apply_futures_filters(
        self, 
        signals: List[TradingSignal], 
        analysis: Dict[str, Any]
    ) -> List[TradingSignal]:
        """Apply futures-specific filters to trading signals."""
        try:
            risk_level = analysis.get('risk_level', RiskLevel.MEDIUM)
            market_regime = analysis.get('market_regime', MarketRegime.RANGING)
            
            filtered_signals = []
            futures_count = 0
            funding_count = 0
            
            for signal in signals:
                should_include = True
                analysis_type = signal.metadata.get('analysis_type', '')
                
                # In extreme risk, only allow exit signals
                if risk_level == RiskLevel.EXTREME:
                    if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                        should_include = False
                
                # Limit futures signals in ranging markets
                if (market_regime == MarketRegime.RANGING and 
                    analysis_type == 'futures_technical' and 
                    signal.confidence < 0.8):
                    should_include = False
                
                # Limit total futures signals
                if analysis_type == 'futures_technical':
                    if futures_count >= 3:  # Max 3 futures signals
                        should_include = False
                    else:
                        futures_count += 1
                
                # Limit funding arbitrage signals
                if analysis_type == 'funding_arbitrage':
                    if funding_count >= 2:  # Max 2 funding signals
                        should_include = False
                    else:
                        funding_count += 1
                
                # Boost confidence in trending markets
                if (market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] and
                    analysis_type == 'futures_technical' and 
                    signal.confidence > 0.7):
                    signal.confidence = min(0.95, signal.confidence + 0.1)
                
                if should_include:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error applying futures filters: {e}")
            return signals
    
    async def _execute_futures_trade(self, signal: TradingSignal, market_data: MarketData) -> Dict[str, Any]:
        """Execute a live futures trade with leverage via Hyperliquid."""
        try:
            if not self.hyperliquid_service:
                raise ValueError("Hyperliquid service is required for live trading - no fallback available")
            
            leverage = signal.metadata.get('leverage', 2.0)
            futures_symbol = signal.metadata.get('futures_symbol', signal.token_symbol)
            
            # Get real-time market data
            live_data = await self.hyperliquid_service.get_live_market_data(futures_symbol)
            if not live_data:
                raise ValueError(f"No live market data available for {futures_symbol}")
            
            # Prefer the allocation-scoped size calculated by AgentAllocationService.
            metadata_position_size = signal.metadata.get('position_size') if signal.metadata else None
            metadata_trade_amount = signal.metadata.get('trade_amount') if signal.metadata else None
            if metadata_position_size:
                position_size = float(metadata_position_size)
                position_value_usd = position_size * live_data.price
            else:
                base_position_size = self.risk_params.max_position_size_percent / 100.0
                account_summary = await self.hyperliquid_service.get_account_summary()
                account_value = account_summary.get('balance', 0)

                if account_value <= 0:
                    raise ValueError("Insufficient account balance")

                # Calculate position size in USD
                position_value_usd = account_value * base_position_size
                position_size = position_value_usd / live_data.price  # Convert to token size
            
            # Determine order side
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                order_side = OrderSide.BUY
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                order_side = OrderSide.SELL
            else:
                raise ValueError(f"Invalid signal type for futures trading: {signal.signal_type}")
            
            # Execute the trade. When the signal carries an entry price, rest a GTC
            # limit at that level instead of chasing the market. If the live price
            # is already at/better than entry, the limit crosses and fills at market.
            entry_limit_price = None
            if signal.metadata:
                try:
                    candidate = float(signal.metadata.get('entry_limit_price') or 0)
                    entry_limit_price = candidate if candidate > 0 else None
                except (TypeError, ValueError):
                    entry_limit_price = None

            start_time = datetime.now()
            venue_name = getattr(self, "venue_name", None) or getattr(settings, "DEFAULT_VENUE", "hyperliquid")
            venue_order_result = None
            try:
                from kata.venues import get_venue, OrderRequest, OrderSide as VenueOrderSide, OrderType as VenueOrderType, TimeInForce
                perp_venue = get_venue(venue_name)
                client_id = (signal.metadata.get('client_id') if signal.metadata else None) or f"kata_{futures_symbol}_{int(start_time.timestamp())}"
                v_side = VenueOrderSide.BUY if order_side == OrderSide.BUY else VenueOrderSide.SELL
                v_type = VenueOrderType.LIMIT if entry_limit_price else VenueOrderType.MARKET
                v_tif = TimeInForce.GTC if entry_limit_price else TimeInForce.IOC
                quantized_size = await perp_venue.quantize_size(futures_symbol, position_size)
                clamped_leverage = await perp_venue.clamp_leverage(futures_symbol, leverage)

                v_request = OrderRequest(
                    symbol=futures_symbol,
                    side=v_side,
                    size=quantized_size,
                    order_type=v_type,
                    price=entry_limit_price,
                    leverage=clamped_leverage,
                    time_in_force=v_tif,
                    client_id=client_id,
                )
                venue_res = await perp_venue.place_order(v_request)
                if venue_res:
                    from kata.services.hyperliquid_service import OrderResult as HLOrderResult
                    venue_order_result = HLOrderResult(
                        success=venue_res.success,
                        order_id=venue_res.order_id,
                        symbol=futures_symbol,
                        side=order_side.value,
                        size=venue_res.filled_size or venue_res.size or quantized_size,
                        price=venue_res.average_price or venue_res.price or live_data.price,
                        fee=venue_res.fee or 0.0,
                        status=venue_res.status or ("filled" if venue_res.success else "failed"),
                        error=venue_res.error
                    )
            except Exception as venue_err:
                logger.warning(f"PerpVenue routing via {venue_name} encountered error: {venue_err}; falling back to direct service", exc_info=True)
                venue_order_result = None

            if venue_order_result is not None:
                order_result = venue_order_result
            elif entry_limit_price:
                order_result = await self.hyperliquid_service.place_order_live(
                    symbol=futures_symbol,
                    side=order_side,
                    size=position_size,
                    order_type=OrderType.LIMIT,
                    price=entry_limit_price,
                    leverage=leverage,
                    time_in_force="gtc"
                )
            else:
                order_result = await self.hyperliquid_service.place_order_live(
                    symbol=futures_symbol,
                    side=order_side,
                    size=position_size,
                    order_type=OrderType.MARKET,
                    leverage=leverage,
                    time_in_force="ioc"
                )

            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update futures metrics
            self.futures_metrics['max_leverage_used'] = max(
                self.futures_metrics['max_leverage_used'],
                leverage
            )

            if order_result.success:
                is_resting = (order_result.status or '').lower() == 'open'
                executed_size = float(
                    order_result.filled_size
                    or order_result.size
                    or position_size
                )
                if is_resting:
                    logger.info(
                        f"Resting entry placed: {order_side.value} {executed_size:.6f} {futures_symbol} "
                        f"limit @ ${entry_limit_price} with {leverage:.1f}x leverage (awaiting fill)"
                    )
                else:
                    logger.info(f"Live futures trade executed: {order_side.value} {executed_size:.6f} {futures_symbol} "
                               f"@ ${order_result.price:.2f} with {leverage:.1f}x leverage "
                               f"(execution: {execution_time_ms:.1f}ms)")

                # Store filled trades in transaction history; resting orders are
                # tracked via the allocation trade record until they fill.
                if not is_resting:
                    trade_data = {
                        'order_id': order_result.order_id or f"futures_{futures_symbol}_{int(start_time.timestamp())}",
                        'symbol': futures_symbol,
                        'side': order_side.value,
                        'size': executed_size,
                        'price': order_result.price,
                        'value': float(metadata_trade_amount or (executed_size * order_result.price)),
                        'leverage': leverage,
                        'reasoning': signal.reasoning or f"Yuki futures {order_side.value} signal",
                        'confidence': signal.confidence
                    }
                    await self.db_service.store_agent_trade(
                        user_id=self.user_id,
                        agent_type=self.agent_type,
                        trade_data=trade_data
                    )

                return {
                    'success': True,
                    'trade_id': order_result.order_id or f"futures_{futures_symbol}_{int(start_time.timestamp())}",
                    'order_id': order_result.order_id,
                    'symbol': futures_symbol,
                    'side': order_side.value,
                    'size': executed_size,
                    'leverage': leverage,
                    'execution_price': order_result.price,
                    'filled_size': order_result.filled_size,
                    'average_price': order_result.average_price,
                    'fee': order_result.fee,
                    'execution_time_ms': execution_time_ms,
                    'timestamp': start_time.isoformat(),
                    'status': order_result.status,
                    'entry_limit_price': entry_limit_price,
                    'stored_in_db': not is_resting
                }
            else:
                logger.error(f"Live futures trade failed: {order_result.error}")
                return {
                    'success': False,
                    'error': order_result.error,
                    'symbol': futures_symbol,
                    'side': order_side.value,
                    'execution_time_ms': execution_time_ms
                }
            
        except Exception as e:
            logger.error(f"Error executing live futures trade: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_spot_trade(self, signal: TradingSignal, market_data: MarketData) -> Dict[str, Any]:
        """Execute a spot trade - not supported for Yuki (futures only)."""
        logger.warning(f"Spot trading not supported for Yuki agent - futures only")
        return {
            'success': False, 
            'error': 'Yuki agent only supports futures trading on Hyperliquid - spot trading not available'
        }

    # =====================================================
    # CRITICAL MISSING METHODS - Required by BaseAgent
    # =====================================================

    async def _fetch_market_data(self) -> Optional[MarketData]:
        """
        Fetch real-time market data for Hyperliquid futures trading.
        Overrides base agent's simulated data with live Hyperliquid data.
        """
        try:
            if not self.hyperliquid_service:
                logger.warning("No Hyperliquid service available, using base market data")
                return await super()._fetch_market_data()
            
            # Get live data for supported futures symbols
            live_data = {}
            token_prices = {}
            volume_24h = {}
            price_changes = {}
            market_cap = {}
            volatility = {}
            
            # Update market data timestamp
            self._last_market_data_fetch = datetime.now()
            
            for symbol in self.supported_futures.keys():
                try:
                    # Get live market data from Hyperliquid
                    symbol_data = await self.hyperliquid_service.get_live_market_data(symbol)
                    if symbol_data:
                        # Extract token symbol (remove -USD suffix)
                        token_symbol = symbol.split('-')[0]
                        
                        # Store live data
                        live_data[symbol] = symbol_data
                        token_prices[token_symbol] = Decimal(str(symbol_data.price))
                        volume_24h[token_symbol] = Decimal(str(symbol_data.volume_24h))
                        
                        # Calculate price change (simplified - would need historical data)
                        price_changes[token_symbol] = Decimal('0.0')  # Placeholder
                        
                        # Estimate market cap (simplified)
                        market_cap[token_symbol] = Decimal(str(symbol_data.volume_24h * 10))  # Rough estimate
                        
                        # Calculate volatility from high-low range
                        if symbol_data.high_24h > 0 and symbol_data.low_24h > 0:
                            vol = (symbol_data.high_24h - symbol_data.low_24h) / symbol_data.price
                            volatility[token_symbol] = min(1.0, vol)
                        else:
                            volatility[token_symbol] = 0.15  # Default volatility
                        
                        logger.debug(f"Fetched live data for {symbol}: ${symbol_data.price:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            # If no live data available, fall back to base implementation
            if not live_data:
                logger.warning("No live Hyperliquid data available, using base market data")
                return await super()._fetch_market_data()
            
            # Add USDC for stablecoin reference
            token_prices['USDC'] = Decimal('1.00')
            volume_24h['USDC'] = Decimal('1000000000')
            price_changes['USDC'] = Decimal('0.0')
            market_cap['USDC'] = Decimal('150000000000')
            volatility['USDC'] = 0.001
            
            # Determine market sentiment based on live data
            trend_indicators = self._analyze_market_sentiment(live_data)
            
            market_data = MarketData(
                timestamp=datetime.now(),
                token_prices=token_prices,
                volume_24h=volume_24h,
                price_changes=price_changes,
                market_cap=market_cap,
                volatility=volatility,
                trend_indicators=trend_indicators,
                metadata={
                    'source': 'hyperliquid_live',
                    'symbols_fetched': list(live_data.keys()),
                    'live_data_quality': 'high' if len(live_data) >= 3 else 'medium'
                }
            )
            
            logger.info(f"Fetched live market data for {len(live_data)} symbols from Hyperliquid")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching Hyperliquid market data: {e}")
            # Fall back to base implementation
            return await super()._fetch_market_data()

    def _analyze_market_sentiment(self, live_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment from live Hyperliquid data."""
        try:
            if not live_data:
                return {'overall_sentiment': 'neutral', 'fear_greed_index': 50}
            
            # Calculate sentiment based on price movements and volume
            bullish_count = 0
            bearish_count = 0
            total_volume = 0
            
            for symbol, data in live_data.items():
                if hasattr(data, 'price') and hasattr(data, 'vwap'):
                    if data.price > data.vwap * 1.001:  # 0.1% above VWAP
                        bullish_count += 1
                    elif data.price < data.vwap * 0.999:  # 0.1% below VWAP
                        bearish_count += 1
                
                if hasattr(data, 'volume_24h'):
                    total_volume += data.volume_24h
            
            # Determine sentiment
            if bullish_count > bearish_count:
                sentiment = 'bullish'
                fear_greed = min(100, 50 + (bullish_count - bearish_count) * 10)
            elif bearish_count > bullish_count:
                sentiment = 'bearish'
                fear_greed = max(0, 50 - (bearish_count - bullish_count) * 10)
            else:
                sentiment = 'neutral'
                fear_greed = 50
            
            return {
                'overall_sentiment': sentiment,
                'fear_greed_index': fear_greed,
                'bullish_symbols': bullish_count,
                'bearish_symbols': bearish_count,
                'total_volume_24h': total_volume,
                'market_trend': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return {'overall_sentiment': 'neutral', 'fear_greed_index': 50}

    async def execute_trades(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """
        Execute validated trading signals.
        
        For Yuki agent, this focuses on futures trading via Hyperliquid.
        """
        executed_trades = []
        
        try:
            logger.info(f"Yuki agent executing {len(signals)} trading signals")
            
            for signal in signals:
                try:
                    # Check if this is a futures signal
                    if signal.metadata and signal.metadata.get('analysis_type') == 'futures_technical':
                        # Execute futures trade
                        trade_result = await self._execute_futures_trade(signal, None)  # market_data not needed for futures
                    else:
                        # Execute spot trade (fallback)
                        trade_result = await self._execute_spot_trade(signal, None)
                    
                    if trade_result.get('success'):
                        executed_trades.append(trade_result)
                        logger.info(f"Successfully executed {signal.signal_type.value} trade for {signal.token_symbol}")
                        
                        # Update daily metrics
                        if trade_result.get('value'):
                            trade_value = float(trade_result['value'])
                            self.daily_pnl += trade_value * 0.01  # Assume 1% profit for now
                    else:
                        logger.warning(f"Trade execution failed for {signal.token_symbol}: {trade_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error executing trade for {signal.token_symbol}: {e}")
                    continue
            
            logger.info(f"Yuki agent completed execution of {len(executed_trades)} trades")
            return executed_trades
            
        except Exception as e:
            logger.error(f"Error in trade execution: {e}")
            return []

    async def validate_risk(self, signal: TradingSignal) -> bool:
        """
        Validate trading signal against risk parameters.
        
        Yuki agent has aggressive but controlled risk management.
        """
        try:
            # Basic confidence threshold
            if signal.confidence < self.risk_params.min_confidence_threshold:
                logger.debug(f"Signal rejected: confidence {signal.confidence:.2f} below threshold {self.risk_params.min_confidence_threshold}")
                return False
            
            # Check daily trade limits
            if self.daily_trades_count >= self.risk_params.max_trades_per_day:
                logger.debug(f"Signal rejected: daily trade limit reached ({self.daily_trades_count}/{self.risk_params.max_trades_per_day})")
                return False
            
            # Check cooldown period
            if self.last_execution_time:
                time_since_last = (datetime.now() - self.last_execution_time).total_seconds() / 60
                if time_since_last < self.risk_params.cooldown_minutes:
                    logger.debug(f"Signal rejected: cooldown period active ({time_since_last:.1f} min < {self.risk_params.cooldown_minutes} min)")
                    return False
            
            # Check daily loss limits
            if self.daily_pnl <= -self.risk_params.max_daily_loss_percent:
                logger.debug(f"Signal rejected: daily loss limit reached ({self.daily_pnl:.2f}%)")
                return False
            
            # Futures-specific risk validation
            if signal.metadata and signal.metadata.get('analysis_type') == 'futures_technical':
                leverage = signal.metadata.get('leverage', 2.0)
                if leverage > self.futures_risk_params['max_leverage']:
                    logger.debug(f"Futures signal rejected: leverage {leverage}x exceeds max {self.futures_risk_params['max_leverage']}x")
                    return False
            
            # All validations passed
            logger.debug(f"Signal validated successfully for {signal.token_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error in risk validation: {e}")
            return False  # Fail safe - reject on error

    async def _can_execute_cycle(self) -> bool:
        """
        Check if the agent can execute a trading cycle.
        
        Overrides base implementation with Yuki-specific checks.
        """
        try:
            # Check if agent is running
            if not self.is_running:
                return False
            
            # Check daily loss limits
            if self.daily_pnl <= -self.risk_params.max_daily_loss_percent:
                logger.info(f"Trading cycle blocked: daily loss limit reached ({self.daily_pnl:.2f}%)")
                return False
            
            # Check if Hyperliquid service is available (for futures trading)
            if not self.hyperliquid_service:
                logger.warning("Trading cycle blocked: Hyperliquid service not available")
                return False
            
            # Check if we have recent market data
            if not hasattr(self, '_last_market_data_fetch') or \
               (datetime.now() - self._last_market_data_fetch).total_seconds() > 300:  # 5 minutes
                logger.info("Trading cycle blocked: market data too old")
                return False
            
            # Update market data timestamp
            self._last_market_data_fetch = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking cycle execution: {e}")
            return False

    def _calculate_technical_score(self, futures_analysis: Dict[str, Any]) -> float:
        """
        Calculate technical analysis score based on Yuki's aggressive futures strategy.
        
        Yuki focuses on:
        - Trend strength (ADX, moving averages)
        - Momentum indicators (RSI, MACD, Stochastic)
        - Volatility patterns (Bollinger Bands, ATR)
        - Volume confirmation
        """
        try:
            if not futures_analysis:
                return 0.5
            
            total_score = 0.0
            valid_symbols = 0
            
            for symbol, analysis in futures_analysis.items():
                indicators = analysis.get('indicators', {})
                if not indicators:
                    continue
                
                # Trend strength (0-1 scale)
                adx = getattr(indicators, 'adx', 50)
                trend_strength = min(1.0, adx / 50.0)  # ADX > 25 is trending
                
                # Momentum (0-1 scale)
                rsi = getattr(indicators, 'rsi', 50)
                rsi_score = 1.0 - abs(rsi - 50) / 50.0  # Closer to 50 = better
                
                macd = getattr(indicators, 'macd', 0)
                macd_signal = getattr(indicators, 'macd_signal', 0)
                macd_score = 1.0 if (macd > macd_signal and macd > 0) else 0.5
                
                # Volatility (0-1 scale) - Yuki prefers moderate volatility
                atr = getattr(indicators, 'atr', 0)
                bb_width = getattr(indicators, 'bb_width', 0.02)
                volatility_score = min(1.0, (atr + bb_width * 100) / 0.1)  # Normalize
                
                # Volume confirmation
                volume_ratio = getattr(indicators, 'volume_ratio', 1.0)
                volume_score = min(1.0, volume_ratio / 2.0)  # Above 2x is good
                
                # Calculate symbol score (weighted average)
                symbol_score = (
                    trend_strength * 0.3 +
                    rsi_score * 0.2 +
                    macd_score * 0.2 +
                    volatility_score * 0.2 +
                    volume_score * 0.1
                )
                
                total_score += symbol_score
                valid_symbols += 1
            
            if valid_symbols == 0:
                return 0.5
            
            return total_score / valid_symbols
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    def _calculate_fundamental_score(self, futures_analysis: Dict[str, Any]) -> float:
        """
        Calculate fundamental score based on Yuki's futures strategy.
        
        Yuki considers:
        - Funding rate opportunities
        - Open interest trends
        - Market liquidity
        - Volatility regime
        """
        try:
            if not futures_analysis:
                return 0.5
            
            total_score = 0.0
            valid_symbols = 0
            
            for symbol, analysis in futures_analysis.items():
                # Funding rate analysis
                funding = analysis.get('funding', {})
                funding_score = 0.5
                if funding.get('arbitrage_opportunity'):
                    funding_score = 0.8  # Good funding opportunity
                
                # Market structure
                structure = analysis.get('structure', {})
                liquidity_score = 0.5
                if structure.get('liquidity', 'high') == 'high':
                    liquidity_score = 0.8
                elif structure.get('liquidity', 'medium') == 'medium':
                    liquidity_score = 0.6
                
                # Volatility regime
                volatility_score = 0.5
                if analysis.get('volatility_regime') == 'normal':
                    volatility_score = 0.7  # Yuki prefers normal volatility
                elif analysis.get('volatility_regime') == 'high':
                    volatility_score = 0.4  # High volatility is riskier
                
                # Calculate symbol score
                symbol_score = (
                    funding_score * 0.4 +
                    liquidity_score * 0.3 +
                    volatility_score * 0.3
                )
                
                total_score += symbol_score
                valid_symbols += 1
            
            if valid_symbols == 0:
                return 0.5
            
            return total_score / valid_symbols
            
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            raise  # Let the error propagate instead of returning fallback score

    def _calculate_agent_confidence(self, futures_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall agent confidence based on analysis quality and opportunities.
        """
        try:
            if not futures_analysis:
                return 0.5
            
            # Count strong signals
            strong_signals = sum(
                1 for analysis in futures_analysis.values()
                if analysis.get('signal_strength', 0) > 0.7
            )
            
            # Count funding opportunities
            funding_opportunities = sum(
                1 for analysis in futures_analysis.values()
                if analysis.get('funding', {}).get('arbitrage_opportunity', False)
            )
            
            # Calculate confidence based on opportunities
            total_opportunities = strong_signals + funding_opportunities
            if total_opportunities >= 3:
                return 0.9  # High confidence
            elif total_opportunities >= 1:
                return 0.7  # Medium confidence
            else:
                return 0.4  # Low confidence
                
        except Exception as e:
            logger.error(f"Error calculating agent confidence: {e}")
            return 0.5

    async def execute_react_cycle(self, symbol: str, market_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute an autonomous ReAct (Reason + Act) decision cycle for a trading asset.
        
        Steps:
        1. Fetch current active goals and historical lessons from AgentMemoryService.
        2. Format prompt for DeepSeek containing market data, goals, lessons, and available tools.
        3. Parse DeepSeek's step-by-step reasoning monologue ("Thought:", "Action:", "Decision:").
        4. Execute requested tool calls (e.g. orderbook depth, market anomalies, past trade outcomes).
        5. Formulate final action (BUY/SELL/HOLD, leverage, position size) and record episode in memory.
        """
        try:
            logger.info(f"🧠 [ReAct Engine] Starting autonomous ReAct cycle for {symbol}")
            
            # 1. Gather context from memory
            goals = await self.memory_service.get_active_goals()
            lessons = await self.memory_service.get_active_lessons(symbol=symbol, limit=5)
            recent_episodes = await self.memory_service.get_recent_episodes(symbol=symbol, limit=3)

            active_goal_str = goals[0].goal_statement if goals else "Maximize risk-adjusted returns on Hyperliquid perps."
            lessons_str = "\n".join([f"- {l}" for l in lessons]) if lessons else "No prior negative trade lessons recorded yet."

            # 2. Build ReAct Prompt
            react_prompt = f"""
You are Yuki, an aggressive perpetual futures trading agent on Hyperliquid.
You follow the ReAct (Reason + Act) paradigm: you reason step-by-step before acting, and you can invoke market tools.

YOUR ACTIVE GOAL:
{active_goal_str}

LESSONS LEARNED FROM PAST TRADES:
{lessons_str}

TARGET ASSET: {symbol}

AVAILABLE TOOLS YOU CAN INVOKE:
1. scan_market_anomalies: Scans all perps for abnormal volume or funding rate spikes.
2. inspect_orderbook_liquidity(symbol="{symbol}"): Checks orderbook depth and spread quality.
3. query_cross_agent_sentiments(symbol="{symbol}"): Checks peer agent directional bias (Sakura/Ryu).
4. fetch_recent_trade_outcomes(symbol="{symbol}"): Checks your past win/loss record for this asset.

INSTRUCTIONS:
First, write your internal monologue under "THOUGHT:".
Second, decide if you need a tool. If yes, output "ACTION: tool_name(param_name=value)".
Third, state your final decision under "DECISION:" with direction (BUY/SELL/HOLD), confidence (0.0 to 1.0), suggested leverage (2x-20x), and position size (%).

Format your output strictly like this:
THOUGHT: [Your step-by-step market reasoning]
ACTION: [Tool name or NONE]
DECISION: [BUY, SELL, or HOLD] | Confidence: [0.0-1.0] | Leverage: [2x-20x] | Size: [1-100%]
REASONING: [1-2 sentence executive explanation]
"""

            # 3. Query DeepSeek LLM for ReAct reasoning
            thought_chain = ""
            tool_calls_executed = []
            action_taken = "HOLD"
            confidence = 0.5
            leverage = "2x"
            position_size = "5%"
            executive_reasoning = "ReAct evaluation completed."

            if self.use_llm_enhancement and self.llm_service and self.llm_service.client:
                llm_raw_response, _ = await self.llm_service._query_llm(react_prompt)
                thought_chain = llm_raw_response
                
                # Parse action tool request if present
                if "ACTION:" in llm_raw_response:
                    action_line = [line for line in llm_raw_response.split('\n') if "ACTION:" in line][0]
                    tool_req = action_line.replace("ACTION:", "").strip()
                    if tool_req and tool_req.upper() != "NONE":
                        tool_result = await self._execute_tool_request(tool_req, symbol)
                        tool_calls_executed.append({'tool': tool_req, 'result': tool_result})
                
                # Parse DECISION line
                if "DECISION:" in llm_raw_response:
                    decision_line = [line for line in llm_raw_response.split('\n') if "DECISION:" in line][0]
                    parts = decision_line.replace("DECISION:", "").split("|")
                    if parts:
                        action_taken = parts[0].strip().upper()
                        if len(parts) > 1 and "Confidence:" in parts[1]:
                            try:
                                confidence = float(parts[1].replace("Confidence:", "").strip())
                            except ValueError:
                                confidence = 0.65
                        if len(parts) > 2 and "Leverage:" in parts[2]:
                            leverage = parts[2].replace("Leverage:", "").strip()
                        if len(parts) > 3 and "Size:" in parts[3]:
                            position_size = parts[3].replace("Size:", "").strip()

                if "REASONING:" in llm_raw_response:
                    reasoning_line = [line for line in llm_raw_response.split('\n') if "REASONING:" in line][0]
                    executive_reasoning = reasoning_line.replace("REASONING:", "").strip()
            else:
                thought_chain = "ReAct algorithmic mode: evaluated technical indicators and liquidity."
                action_taken = "HOLD"

            # 4. Save episode into working memory
            episode = await self.memory_service.append_episode(
                symbol=symbol,
                thought_chain=thought_chain,
                tool_calls=tool_calls_executed,
                action_taken=action_taken,
                confidence=confidence,
                observation=executive_reasoning
            )

            logger.info(f"✅ [ReAct Engine] {symbol} cycle completed: {action_taken} (Confidence: {confidence:.2f}, Leverage: {leverage})")

            return {
                'symbol': symbol,
                'action': action_taken,
                'confidence': confidence,
                'leverage': leverage,
                'position_size': position_size,
                'thought_chain': thought_chain,
                'tool_calls': tool_calls_executed,
                'reasoning': executive_reasoning,
                'episode_id': episode.id if episode else None
            }

        except Exception as e:
            logger.error(f"Error in ReAct cycle for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': f"ReAct execution fallback due to error: {e}"
            }

    async def execute_signal_reentry_react_cycle(
        self,
        signal_context: Dict[str, Any],
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decide whether an active signal deserves another entry attempt."""
        signal = dict(signal_context.get("signal") or {})
        symbol = str(signal_context.get("symbol") or signal.get("symbol") or "").strip()
        trigger_context = dict(trigger_context or {})
        signal_id = str(signal.get("signal_id") or trigger_context.get("signal_id") or "")
        latest_closed_attempt = dict(signal_context.get("latest_closed_attempt") or {})
        latest_trade_id = latest_closed_attempt.get("trade_id")

        try:
            logger.info("[ReAct Engine] Reviewing signal re-entry for %s", symbol)

            from kata.services.agent_memory_service import get_agent_memory_service

            global_memory = get_agent_memory_service("yuki")
            lessons_limit = max(1, int(getattr(self, "react_lessons_limit", 4) or 4))
            episodes_limit = max(1, int(getattr(self, "react_episodes_limit", 3) or 3))
            memory_results = await asyncio.gather(
                self.memory_service.get_active_goals(),
                self.memory_service.get_active_lessons(symbol=symbol, limit=lessons_limit),
                self.memory_service.get_recent_episodes(symbol=symbol, limit=episodes_limit),
                global_memory.get_active_goals(),
                global_memory.get_active_lessons(symbol=symbol, limit=lessons_limit),
                global_memory.get_recent_episodes(symbol=symbol, limit=episodes_limit),
                return_exceptions=True,
            )
            user_goals = [] if isinstance(memory_results[0], Exception) else memory_results[0]
            user_lessons = [] if isinstance(memory_results[1], Exception) else memory_results[1]
            user_episodes = [] if isinstance(memory_results[2], Exception) else memory_results[2]
            global_goals = [] if isinstance(memory_results[3], Exception) else memory_results[3]
            global_lessons = [] if isinstance(memory_results[4], Exception) else memory_results[4]
            global_episodes = [] if isinstance(memory_results[5], Exception) else memory_results[5]
            goals = [*user_goals, *global_goals]
            lessons = list(dict.fromkeys(str(value) for value in [*user_lessons, *global_lessons]))
            recent_episodes = [*user_episodes, *global_episodes]

            source_calls = [
                ("live_market", self.tools.inspect_live_market(symbol)),
                ("orderbook_liquidity", self.tools.inspect_orderbook_liquidity(symbol)),
                ("cross_agent_sentiment", self.tools.query_cross_agent_sentiments(symbol)),
                ("recent_trade_outcomes", self.tools.fetch_recent_trade_outcomes(symbol)),
                ("market_anomalies", self.tools.scan_market_anomalies()),
                (
                    "active_signal_context",
                    self.tools.fetch_active_trade_context(
                        trade_id=str(latest_trade_id) if latest_trade_id else None,
                        signal_id=signal_id or None,
                    ),
                ),
            ]
            source_results = await asyncio.gather(
                *(call for _, call in source_calls),
                return_exceptions=True,
            )
            evidence: Dict[str, Any] = {}
            tool_calls_executed: List[Dict[str, Any]] = []
            for (source_name, _), result in zip(source_calls, source_results):
                normalized = (
                    {"status": "unavailable", "error": str(result)}
                    if isinstance(result, Exception)
                    else result
                )
                evidence[source_name] = normalized
                tool_calls_executed.append({"tool": source_name, "result": normalized})

            memory_context = self._build_react_memory_context(
                goals,
                lessons,
                recent_episodes,
                episode_key="recent_decisions",
            )

            prompt_payload = {
                "review_trigger": self._compact_react_payload(trigger_context),
                "reentry_candidate": self._compact_react_payload(signal_context),
                "agent_memory": memory_context,
                "research_and_market_sources": self._compact_react_payload(evidence),
            }
            prompt_context = json.dumps(prompt_payload, default=str, separators=(",", ":"))
            react_prompt = f"""
You are Yuki, the autonomous ReAct manager for an ACTIVE platform signal that already had at least one completed trade attempt.

Your job is to decide whether this still-active signal deserves another entry attempt right now. The prior trade history is evidence, not a command. Make a fresh decision from the complete context below, including the original signal thesis, how the previous trade ended, live market state, orderbook/liquidity, peer views, recent outcomes, active goals, and learned lessons.

Choose exactly one re-entry action:
- REENTER_NOW: approve another entry attempt immediately using the existing signal-led execution path.
- WAIT_RETEST: do not enter yet; wait for price to retest a better level that could re-offer the thesis.
- SKIP: do not re-enter this signal unless a future material change creates a meaningfully different opportunity.

Do not choose REENTER_NOW just because the source signal is still active.
Do not choose WAIT_RETEST just because the current price differs from the old entry.
If the prior trade already captured the best available move, or the thesis is now stale, choose SKIP.
If the thesis still looks good but location is poor, choose WAIT_RETEST and provide a retest_price.
If evidence is mixed or unavailable, prefer WAIT_RETEST or SKIP over forcing a trade.

EVIDENCE:
{prompt_context}

Return ONLY one valid json object with this exact shape:
{{
  "action": "REENTER_NOW|WAIT_RETEST|SKIP",
  "confidence": 0.0,
  "retest_price": null,
  "next_review_conditions": ["up to three concrete market or thesis changes that would justify another review"],
  "market_regime": "short label",
  "reasoning": "concise evidence-based explanation",
  "evidence_used": ["source names that materially affected the decision"]
}}

For WAIT_RETEST, retest_price should be a positive price near the level that would make a new entry more attractive. For REENTER_NOW and SKIP, use null. Reviews are event-driven; next_review_conditions should describe evidence changes, not a time interval.
"""

            if not (self.use_llm_enhancement and self.llm_service and self.llm_service.client):
                decision = {
                    "action": "SKIP",
                    "confidence": 0.0,
                    "retest_price": None,
                    "next_review_conditions": ["ReAct model becomes available"],
                    "market_regime": "unavailable",
                    "reasoning": "Signal re-entry review deferred because the ReAct model is unavailable.",
                    "evidence_used": [],
                }
                raw_response = ""
                llm_usage = None
                llm_retry_used = False
            else:
                raw_response, decision, llm_usage, llm_retry_used = await self._query_react_json_review(
                    react_prompt,
                    symbol=symbol,
                    review_name="Signal re-entry ReAct review",
                )
                decision = decision or {
                    "action": "SKIP",
                    "confidence": 0.0,
                    "retest_price": None,
                    "next_review_conditions": ["Fresh supported review evidence appears"],
                    "market_regime": "uncertain",
                    "reasoning": (
                        "The model reply was not machine-readable, so Yuki skipped a forced "
                        "re-entry until a fresh supported review is available."
                    ),
                    "evidence_used": [],
                }

            action_aliases = {
                "REENTER": "REENTER_NOW",
                "RE-ENTER": "REENTER_NOW",
                "ENTER": "REENTER_NOW",
                "WAIT": "WAIT_RETEST",
                "WAIT_FOR_RETEST": "WAIT_RETEST",
                "WAIT_FOR_BETTER_ENTRY": "WAIT_RETEST",
                "NO_REENTRY": "SKIP",
                "PASS": "SKIP",
            }
            action = str(decision.get("action") or "SKIP").strip().upper()
            action = action_aliases.get(action, action)
            if action not in {"REENTER_NOW", "WAIT_RETEST", "SKIP"}:
                action = "SKIP"

            try:
                confidence = max(0.0, min(1.0, float(decision.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                retest_price = (
                    float(decision.get("retest_price"))
                    if decision.get("retest_price") is not None
                    else None
                )
                if retest_price is not None and retest_price <= 0:
                    retest_price = None
            except (TypeError, ValueError):
                retest_price = None
            raw_review_conditions = decision.get("next_review_conditions") or []
            if isinstance(raw_review_conditions, str):
                raw_review_conditions = [raw_review_conditions]
            next_review_conditions = [
                str(value).strip()
                for value in list(raw_review_conditions)[:3]
                if str(value).strip()
            ]

            reasoning = str(decision.get("reasoning") or "No supported re-entry was identified.").strip()
            normalized_decision = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "retest_price": retest_price,
                "next_review_conditions": next_review_conditions,
                "market_regime": str(decision.get("market_regime") or "uncertain"),
                "reasoning": reasoning,
                "evidence_used": list(decision.get("evidence_used") or []),
                "trigger": str(trigger_context.get("event") or "signal_reentry_review"),
                "llm_usage": llm_usage,
                "llm_retry_used": llm_retry_used,
            }

            episode = await self.memory_service.append_episode(
                symbol=symbol,
                thought_chain=reasoning,
                tool_calls=tool_calls_executed,
                action_taken=action,
                confidence=confidence,
                market_regime=normalized_decision["market_regime"],
                observation=(
                    f"Signal re-entry review selected {action}; future review requires "
                    "a new material event."
                ),
                trade_id=str(latest_trade_id) if latest_trade_id else None,
            )
            normalized_decision["episode_id"] = episode.id if episode else None
            normalized_decision["raw_response_recorded"] = bool(raw_response)
            logger.info(
                "[ReAct Engine] signal re-entry review for %s selected %s (confidence %.2f)",
                symbol,
                action,
                confidence,
            )
            return normalized_decision
        except Exception as exc:
            logger.error("Signal re-entry ReAct review failed for %s: %s", symbol, exc)
            return {
                "symbol": symbol,
                "action": "SKIP",
                "confidence": 0.0,
                "retest_price": None,
                "next_review_conditions": ["ReAct review infrastructure recovers"],
                "market_regime": "unavailable",
                "reasoning": (
                    "Signal re-entry review failed safely; Yuki will not force a new entry "
                    f"without a fresh supported review. Error: {exc}"
                ),
                "evidence_used": [],
                "trigger": str(trigger_context.get("event") or "signal_reentry_review"),
            }

    def _parse_react_json_response(
        self,
        raw_response: str,
        *,
        symbol: str,
        review_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction for ReAct model replies."""
        parser = getattr(self.llm_service, "_parse_json_with_fallbacks", None)
        normalized = str(raw_response or "").strip()

        # 1. Strip markdown code block fences if present
        cleaned = normalized
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                if lines[-1].strip() == "```":
                    cleaned = "\n".join(lines[1:-1]).strip()
                else:
                    cleaned = "\n".join(lines[1:]).strip()

        candidates: List[str] = []
        if cleaned:
            candidates.append(cleaned)
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = cleaned[json_start:json_end]
                if json_text != cleaned:
                    candidates.append(json_text)

        if normalized and normalized not in candidates:
            candidates.append(normalized)

        for candidate in candidates:
            try:
                parsed = parser(candidate) if parser else json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        preview = " ".join(normalized.split())[:240]
        logger.warning(
            "%s for %s returned a non-JSON response; defaulting safely. Preview: %s",
            review_name,
            symbol,
            preview or "<empty>",
        )
        return None

    async def execute_position_management_react_cycle(
        self,
        position_context: Dict[str, Any],
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Reassess an existing position when a lifecycle event needs a decision.

        A lifecycle event is context, not an exit order. This cycle gathers every
        currently available position-management source before asking the LLM for
        one unambiguous lifecycle action. A malformed or unavailable LLM response
        safely becomes HOLD so infrastructure failure cannot close a live trade.
        """
        symbol = str(position_context.get("symbol") or "").strip()
        trigger_context = dict(trigger_context or {})
        trigger_event = str(trigger_context.get("event") or "position_reassessment")
        trade_id = position_context.get("trade_id")
        metadata = dict(position_context.get("position_metadata") or {})
        signal_id = metadata.get("signal_id")

        try:
            logger.info("[ReAct Engine] Reviewing %s for %s", trigger_event, symbol)

            from kata.services.agent_memory_service import get_agent_memory_service

            global_memory = get_agent_memory_service("yuki")
            lessons_limit = max(1, int(getattr(self, "react_lessons_limit", 4) or 4))
            episodes_limit = max(1, int(getattr(self, "react_episodes_limit", 3) or 3))
            memory_results = await asyncio.gather(
                self.memory_service.get_active_goals(),
                self.memory_service.get_active_lessons(symbol=symbol, limit=lessons_limit),
                self.memory_service.get_recent_episodes(symbol=symbol, limit=episodes_limit),
                global_memory.get_active_goals(),
                global_memory.get_active_lessons(symbol=symbol, limit=lessons_limit),
                global_memory.get_recent_episodes(symbol=symbol, limit=episodes_limit),
                return_exceptions=True,
            )
            user_goals = [] if isinstance(memory_results[0], Exception) else memory_results[0]
            user_lessons = [] if isinstance(memory_results[1], Exception) else memory_results[1]
            user_episodes = [] if isinstance(memory_results[2], Exception) else memory_results[2]
            global_goals = [] if isinstance(memory_results[3], Exception) else memory_results[3]
            global_lessons = [] if isinstance(memory_results[4], Exception) else memory_results[4]
            global_episodes = [] if isinstance(memory_results[5], Exception) else memory_results[5]
            goals = [*user_goals, *global_goals]
            lessons = list(dict.fromkeys(str(value) for value in [*user_lessons, *global_lessons]))
            recent_episodes = [*user_episodes, *global_episodes]

            source_calls = [
                ("live_market", self.tools.inspect_live_market(symbol)),
                ("orderbook_liquidity", self.tools.inspect_orderbook_liquidity(symbol)),
                ("cross_agent_sentiment", self.tools.query_cross_agent_sentiments(symbol)),
                ("recent_trade_outcomes", self.tools.fetch_recent_trade_outcomes(symbol)),
                ("market_anomalies", self.tools.scan_market_anomalies()),
                (
                    "active_trade_context",
                    self.tools.fetch_active_trade_context(
                        trade_id=str(trade_id) if trade_id else None,
                        signal_id=str(signal_id) if signal_id else None,
                    ),
                ),
            ]
            source_results = await asyncio.gather(
                *(call for _, call in source_calls),
                return_exceptions=True,
            )
            evidence: Dict[str, Any] = {}
            tool_calls_executed: List[Dict[str, Any]] = []
            for (source_name, _), result in zip(source_calls, source_results):
                normalized = (
                    {"status": "unavailable", "error": str(result)}
                    if isinstance(result, Exception)
                    else result
                )
                evidence[source_name] = normalized
                tool_calls_executed.append({"tool": source_name, "result": normalized})

            memory_context = self._build_react_memory_context(
                goals,
                lessons,
                recent_episodes,
                episode_key="recent_position_decisions",
            )

            prompt_payload = {
                "review_trigger": self._compact_react_payload(trigger_context),
                "open_position": self._compact_react_payload(position_context),
                "agent_memory": memory_context,
                "research_and_market_sources": self._compact_react_payload(evidence),
            }
            prompt_context = json.dumps(prompt_payload, default=str, separators=(",", ":"))
            react_prompt = f"""
You are Yuki, the autonomous ReAct manager for an EXISTING Hyperliquid perpetual position.

A position-management event has occurred: {trigger_event}. This is an instruction to reassess the thesis, NOT an instruction to close. The event details are evidence, not a predetermined action. Make a fresh decision from the complete evidence below. Consider the original signal and trade history, current position and PnL, live price/funding/liquidity, peer views, recent outcomes, active goals, learned lessons, and your recent decisions. Explicitly weigh conflicts and stale or unavailable sources.

Choose exactly one position-lifecycle action:
- HOLD: keep the current size and protection.
- ADD: add to the same-side position only if the current evidence supports increasing exposure.
- PARTIAL_EXIT: reduce exposure but retain part of the thesis.
- ADJUST_STOP: retain size and propose a tighter protective stop. Never loosen an existing stop.
- FULL_EXIT: close the remaining position because the current thesis no longer justifies the risk.

Hard liquidation and stop-loss protections are enforced outside this review. Do not choose an action merely because a time horizon elapsed, a target was reached, a source signal changed, an opposite signal arrived, or the trade is profitable. Decide from the current thesis and complete evidence. If critical evidence is unavailable or the decision cannot be supported, choose HOLD and request a near-term review.

IMPORTANT — source_signal_invalidated guidance: When the trigger event is "source_signal_invalidated" and the position is currently in profit (unrealized_pnl_percent > 0), strongly prefer ADJUST_STOP to the fee-adjusted break-even price before considering any exit. Signal invalidation means a newer analysis pass superseded the original signal — it does NOT mean the trade thesis has collapsed. Only choose FULL_EXIT if live market evidence (price action, order book, regime change) provides an active reason to close, not just because the signal was superseded. Locking profit at break-even converts the trade to a risk-free hold so price can continue toward target.

If review_trigger includes a policy_proposal, treat it as a bounded suggestion from the execution layer, not a command. You may reject it, keep the current size, reduce risk, exit, or choose ADD with a stop_price that better fits the thesis. When proposing ADD, use stop_price for the protective stop you want on the combined position after adding; the execution layer will clamp it inside the hard-risk rails and refuse any unprotected add.

EVIDENCE:
{prompt_context}

Return ONLY one valid json object with this exact shape:
{{
  "action": "HOLD|ADD|PARTIAL_EXIT|ADJUST_STOP|FULL_EXIT",
  "confidence": 0.0,
  "size_fraction": 0.0,
  "stop_price": null,
  "next_review_conditions": ["up to three concrete market or thesis changes that would warrant another decision"],
  "market_regime": "short label",
  "reasoning": "concise evidence-based decision explanation",
  "evidence_used": ["source names that materially affected the decision"]
}}

For PARTIAL_EXIT, size_fraction is the fraction of the CURRENT position to close and must be between 0.10 and 0.90. For ADD, size_fraction is only a preference, and stop_price is optional but recommended when the policy proposal looks too tight or too loose; the execution layer independently enforces available collateral and the original risk budget. For other actions use 0.0. For ADJUST_STOP, stop_price must be a positive price on the protective side of the current price. Reviews are event-driven, never scheduled; next_review_conditions should describe evidence changes, not a time interval.
"""

            if not (self.use_llm_enhancement and self.llm_service and self.llm_service.client):
                decision = {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "size_fraction": 0.0,
                    "stop_price": None,
                    "next_review_conditions": ["ReAct model becomes available"],
                    "market_regime": "unavailable",
                    "reasoning": "Position review deferred because the ReAct model is unavailable; existing protection remains active.",
                    "evidence_used": [],
                }
                raw_response = ""
                llm_usage = None
                llm_retry_used = False
            else:
                raw_response, decision, llm_usage, llm_retry_used = await self._query_react_json_review(
                    react_prompt,
                    symbol=symbol,
                    review_name="Position-management ReAct review",
                )
                decision = decision or {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "size_fraction": 0.0,
                    "stop_price": None,
                    "next_review_conditions": ["Fresh supported review evidence appears"],
                    "market_regime": "uncertain",
                    "reasoning": (
                        "The model reply was not machine-readable, so Yuki kept the position "
                        "unchanged under its existing protection."
                    ),
                    "evidence_used": [],
                }

            action_aliases = {
                "BUY_MORE": "ADD",
                "SCALE_IN": "ADD",
                "REDUCE": "PARTIAL_EXIT",
                "PARTIAL_CLOSE": "PARTIAL_EXIT",
                "TRAIL_STOP": "ADJUST_STOP",
                "EXIT": "FULL_EXIT",
                "CLOSE": "FULL_EXIT",
                "SELL": "FULL_EXIT",
            }
            action = str(decision.get("action") or "HOLD").strip().upper()
            action = action_aliases.get(action, action)
            allowed_actions = {"HOLD", "ADD", "PARTIAL_EXIT", "ADJUST_STOP", "FULL_EXIT"}
            if action not in allowed_actions:
                action = "HOLD"

            try:
                confidence = max(0.0, min(1.0, float(decision.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                size_fraction = float(decision.get("size_fraction") or 0.0)
            except (TypeError, ValueError):
                size_fraction = 0.0
            size_fraction = max(0.10, min(0.90, size_fraction)) if action == "PARTIAL_EXIT" else 0.0
            try:
                stop_price = float(decision.get("stop_price")) if decision.get("stop_price") is not None else None
                if stop_price is not None and stop_price <= 0:
                    stop_price = None
            except (TypeError, ValueError):
                stop_price = None
            if action == "ADJUST_STOP" and stop_price is None:
                action = "HOLD"
            raw_review_conditions = decision.get("next_review_conditions") or []
            if isinstance(raw_review_conditions, str):
                raw_review_conditions = [raw_review_conditions]
            next_review_conditions = [
                str(value).strip()
                for value in list(raw_review_conditions)[:3]
                if str(value).strip()
            ]

            reasoning = str(decision.get("reasoning") or "No supported lifecycle change was identified.").strip()
            normalized_decision = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "size_fraction": size_fraction,
                "stop_price": stop_price,
                "next_review_conditions": next_review_conditions,
                "market_regime": str(decision.get("market_regime") or "uncertain"),
                "reasoning": reasoning,
                "evidence_used": list(decision.get("evidence_used") or []),
                "trigger": trigger_event,
                "llm_usage": llm_usage,
                "llm_retry_used": llm_retry_used,
            }

            episode = await self.memory_service.append_episode(
                symbol=symbol,
                thought_chain=reasoning,
                tool_calls=tool_calls_executed,
                action_taken=action,
                confidence=confidence,
                market_regime=normalized_decision["market_regime"],
                observation=(
                    f"{trigger_event} review selected {action}; future review requires "
                    "a new material event."
                ),
                trade_id=str(trade_id) if trade_id else None,
            )
            normalized_decision["episode_id"] = episode.id if episode else None
            normalized_decision["raw_response_recorded"] = bool(raw_response)
            logger.info(
                "[ReAct Engine] %s review for %s selected %s (confidence %.2f)",
                trigger_event,
                symbol,
                action,
                confidence,
            )
            return normalized_decision
        except Exception as exc:
            logger.error("Position-management ReAct review failed for %s: %s", symbol, exc)
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0.0,
                "size_fraction": 0.0,
                "stop_price": None,
                "next_review_conditions": ["ReAct review infrastructure recovers"],
                "market_regime": "unavailable",
                "reasoning": (
                    "Position review failed safely; the position remains open under its existing "
                    f"risk protection and will be reviewed again. Error: {exc}"
                ),
                "evidence_used": [],
                "trigger": trigger_event,
            }

    async def _execute_tool_request(self, tool_req: str, symbol: str) -> Dict[str, Any]:
        """Execute a tool request requested by DeepSeek during ReAct loop."""
        tool_req_lower = tool_req.lower()
        if "scan_market_anomalies" in tool_req_lower:
            return await self.tools.scan_market_anomalies()
        elif "inspect_orderbook_liquidity" in tool_req_lower:
            return await self.tools.inspect_orderbook_liquidity(symbol)
        elif "query_cross_agent_sentiments" in tool_req_lower:
            return await self.tools.query_cross_agent_sentiments(symbol)
        elif "fetch_recent_trade_outcomes" in tool_req_lower:
            return await self.tools.fetch_recent_trade_outcomes(symbol)
        else:
            return {'status': 'unknown_tool', 'tool': tool_req}


# Agent factory function
def create_yuki_agent(user_id: str, config: Dict[str, Any], hyperliquid_service: Optional[HyperliquidService] = None) -> YukiAgent:
    """Factory function to create a Yuki agent instance with optional Hyperliquid service."""
    return YukiAgent(user_id, config, hyperliquid_service)
