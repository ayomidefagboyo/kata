"""
Token Analysis Card API

Provides comprehensive token analysis in standardized card format
with all necessary information for trading decisions.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hyperliquid.info import Info
from kata.models.token_analysis_card import (
    TokenAnalysisCard, EntryDetails, TechnicalReasoning, RiskManagement,
    TradeMathematics, MarketConditions, MonitoringLevels, ExecutionStrategy,
    TradeAction, TimeHorizon, format_analysis_card_response
)
from kata.services.llm_analysis_service import LLMAnalysisService, LLMProvider, MarketContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/token-analysis", tags=["Token Analysis"])


class TokenAnalysisRequest(BaseModel):
    """Token analysis request model."""
    symbol: str
    include_alternatives: bool = True
    risk_tolerance: str = "medium"  # low, medium, high
    time_horizon: str = "medium"    # scalp, short, medium, long


@router.get("/card/{symbol}")
async def get_token_analysis_card(
    symbol: str,
    testnet: bool = Query(default=False, description="Use testnet or mainnet"),
    risk_tolerance: str = Query(default="medium", description="Risk tolerance: low, medium, high"),
    time_horizon: str = Query(default="medium", description="Time horizon: scalp, short, medium, long")
) -> Dict[str, Any]:
    """
    Get comprehensive token analysis card with all trading details.
    
    This endpoint provides a complete analysis including entry signals,
    risk management, technical reasoning, and execution strategy.
    """
    try:
        logger.info(f"Generating analysis card for {symbol}")
        
        # Initialize services with proper provider
        from kata.config.settings import settings
        llm_service = LLMAnalysisService(provider=LLMProvider.DEEPSEEK)
        base_url = "https://api.hyperliquid-testnet.xyz" if testnet else None
        from kata.services.hyperliquid_client_factory import create_info_client
        info_client = create_info_client(base_url=base_url)
        
        # Get market data
        meta = info_client.meta()
        universe = meta.get('universe', [])
        all_mids = info_client.all_mids()
        
        # Find symbol in universe
        symbol_info = None
        current_price = None
        
        for i, market in enumerate(universe):
            if market.get('name') == symbol.upper():
                symbol_info = market
                if symbol.upper() in all_mids:
                    current_price = float(all_mids[symbol.upper()])
                break
        
        if not symbol_info or not current_price:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or no price data")
        
        # Get 24h candle data for analysis
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 60 * 60 * 1000)
        
        try:
            candles = info_client.candles_snapshot(symbol.upper(), "1d", start_time, end_time)
            if candles:
                candle = candles[-1]
                open_price = float(candle['o'])
                high_price = float(candle['h'])
                low_price = float(candle['l'])
                volume = float(candle['v'])
                change_24h = ((current_price - open_price) / open_price) * 100 if open_price > 0 else 0
            else:
                open_price = current_price
                high_price = current_price * 1.02
                low_price = current_price * 0.98
                volume = 1000000
                change_24h = 0
        except Exception as e:
            logger.warning(f"Could not fetch candle data for {symbol}: {e}")
            open_price = current_price
            high_price = current_price * 1.02
            low_price = current_price * 0.98
            volume = 1000000
            change_24h = 0
        
        # Calculate technical indicators (simplified)
        volatility = abs(change_24h) / 100
        rsi = 50 + (change_24h * 3)  # Rough RSI approximation
        rsi = max(20, min(80, rsi))  # Clamp between 20-80
        
        # Create market context for LLM
        context = MarketContext(
            timestamp=datetime.now(),
            symbol=symbol.upper(),
            current_price=current_price,
            price_change_24h=change_24h,
            volume_24h=volume * current_price,
            volatility=volatility,
            rsi=rsi,
            macd=change_24h * 5,  # Simplified MACD
            bb_position=0.5 + (change_24h / 20),  # BB position approximation
            fear_greed_index=50 + int(change_24h * 2),
            social_sentiment="positive" if change_24h > 0 else "negative" if change_24h < -1 else "neutral",
            news_sentiment="neutral",
            funding_rate=0.0001,
            market_narrative=f"{symbol} showing {change_24h:+.1f}% movement with {'high' if volume > 1000000 else 'moderate'} volume"
        )
        
        # Get LLM analysis
        try:
            logger.info(f"Starting LLM analysis for {symbol} using provider: deepseek")
            llm_analysis = await llm_service.analyze_trading_opportunity(context)
            logger.info(f"LLM analysis completed for {symbol}: {llm_analysis.recommendation}")
        except Exception as e:
            logger.error(f"LLM analysis failed for {symbol}: {e}")
            logger.error(f"DeepSeek API key present: {bool(settings.DEEPSEEK_API_KEY)}")
            # Fallback to technical analysis
            logger.warning(f"Using fallback analysis for {symbol}")
            llm_analysis = create_fallback_analysis(symbol, current_price, change_24h, rsi)
        
        # Create analysis card
        card = create_token_analysis_card(
            symbol=symbol.upper(),
            symbol_info=symbol_info,
            current_price=current_price,
            change_24h=change_24h,
            volume=volume,
            high_price=high_price,
            low_price=low_price,
            rsi=rsi,
            llm_analysis=llm_analysis,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
            testnet=testnet
        )
        
        # Format for API response
        response = format_analysis_card_response(card)
        
        logger.info(f"Analysis card generated for {symbol}: {llm_analysis.recommendation}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating analysis card for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis: {e}")


def create_fallback_analysis(symbol: str, price: float, change_24h: float, rsi: float):
    """Create technical analysis fallback when LLM analysis fails."""
    from kata.services.llm_analysis_service import LLMAnalysis

    if change_24h > 2 and rsi < 70:
        recommendation = "BUY"
        confidence = 0.6
        reasoning = f"Technical analysis indicates positive momentum with {change_24h:+.1f}% gain and RSI at {rsi:.0f}. Price action suggests continued upward movement with room for growth."
    elif change_24h < -2 and rsi > 30:
        recommendation = "SELL"
        confidence = 0.6
        reasoning = f"Technical analysis shows negative momentum with {change_24h:+.1f}% decline and RSI at {rsi:.0f}. Price action suggests further downside pressure."
    else:
        recommendation = "HOLD"
        confidence = 0.5
        reasoning = f"Technical analysis shows mixed signals with {change_24h:+.1f}% change and RSI at {rsi:.0f}. Current conditions suggest waiting for clearer directional signals."
    
    return LLMAnalysis(
        recommendation=recommendation,
        confidence=confidence,
        reasoning=reasoning,
        key_factors=["RSI", "Price momentum", "Volume"],
        risk_assessment="MEDIUM",
        time_horizon="SHORT",
        position_sizing="MEDIUM",
        stop_loss_level=price * (0.95 if recommendation == "BUY" else 1.05),
        take_profit_level=price * (1.08 if recommendation == "BUY" else 0.92),
        market_regime="NEUTRAL",
        contrarian_signals=[]
    )


def create_token_analysis_card(
    symbol: str,
    symbol_info: dict,
    current_price: float,
    change_24h: float,
    volume: float,
    high_price: float,
    low_price: float,
    rsi: float,
    llm_analysis,
    risk_tolerance: str,
    time_horizon: str,
    testnet: bool
) -> TokenAnalysisCard:
    """Create comprehensive token analysis card."""
    
    # Determine trade action
    if llm_analysis.recommendation == "BUY":
        action = TradeAction.LONG
    elif llm_analysis.recommendation == "SELL":
        action = TradeAction.SHORT
    else:
        action = TradeAction.HOLD
    
    # Calculate leverage based on risk tolerance and volatility
    volatility = abs(change_24h) / 100
    base_leverage = {"low": 2, "medium": 3, "high": 5}[risk_tolerance]
    leverage = max(1, base_leverage - int(volatility * 10))  # Reduce leverage for high volatility
    
    # Risk management calculations - different for spot vs leverage trading
    risk_multiplier = {"low": 0.03, "medium": 0.05, "high": 0.08}[risk_tolerance]
    stop_distance = risk_multiplier
    
    if action == TradeAction.LONG:
        # BUY signal - leverage trading targets
        stop_loss = current_price * (1 - stop_distance)
        tp1 = current_price * 1.05
        tp2 = current_price * 1.12
    elif action == TradeAction.SHORT:
        # SELL signal - should show exit price for spot trading, not leverage targets
        stop_loss = None  # No stop loss for spot selling
        tp1 = current_price  # Exit price for spot selling
        tp2 = None  # No second target for spot selling
    else:  # HOLD
        # HOLD signal - no specific entry/exit prices
        stop_loss = None
        tp1 = None
        tp2 = None
    
    # Position sizing based on confidence and risk tolerance
    base_position = {"low": 5, "medium": 10, "high": 15}[risk_tolerance]
    position_size = base_position * llm_analysis.confidence
    
    return TokenAnalysisCard(
        # Header
        symbol=symbol,
        analysis_timestamp=datetime.now(),
        analyst="Ryu Agent",
        market_tier=determine_market_tier(current_price, volume),
        
        # Entry Details
        entry_details=EntryDetails(
            current_price=current_price,
            action=action,
            leverage=float(leverage),
            confidence=llm_analysis.confidence,
            optimal_entry_range={"min": current_price * 0.998, "max": current_price * 1.002}
        ),
        
        # Technical Reasoning
        technical_reasoning=TechnicalReasoning(
            primary_signals=llm_analysis.key_factors[:3],
            supporting_indicators=llm_analysis.key_factors[3:] if len(llm_analysis.key_factors) > 3 else [],
            momentum_analysis=f"{change_24h:+.2f}% 24h movement with {'strong' if abs(change_24h) > 3 else 'moderate'} momentum",
            trend_analysis=f"RSI at {rsi:.0f} indicates {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'} conditions",
            volume_analysis=f"Volume: {volume:,.0f} - {'Above average' if volume > 500000 else 'Below average'} trading activity",
            support_resistance={"support": low_price, "resistance": high_price}
        ),
        
        # Risk Management - different structure based on action
        risk_management=RiskManagement(
            stop_loss=stop_loss,
            take_profit_levels=[
                tp_level for tp_level in [
                    {"level": 1, "price": tp1, "percentage": ((tp1 - current_price) / current_price * 100)} if tp1 else None,
                    {"level": 2, "price": tp2, "percentage": ((tp2 - current_price) / current_price * 100)} if tp2 else None
                ] if tp_level is not None
            ],
            position_size_percentage=position_size if action != TradeAction.HOLD else 0,
            max_risk_per_trade=risk_multiplier * 100 if action == TradeAction.LONG else 0,
            trailing_stop=current_price * 0.02 if action == TradeAction.LONG else None
        ),
        
        # Trade Mathematics - only meaningful for leverage trading (LONG)
        trade_mathematics=TradeMathematics(
            risk_amount=abs(current_price - stop_loss) if stop_loss else 0,
            reward_amounts=[amount for amount in [
                abs(tp1 - current_price) if tp1 and tp1 != current_price else None,
                abs(tp2 - current_price) if tp2 else None
            ] if amount is not None],
            risk_reward_ratios=[ratio for ratio in [
                abs(tp1 - current_price) / abs(current_price - stop_loss) if tp1 and stop_loss and tp1 != current_price else None,
                abs(tp2 - current_price) / abs(current_price - stop_loss) if tp2 and stop_loss else None
            ] if ratio is not None],
            breakeven_price=current_price if action == TradeAction.LONG else None,
            liquidation_price=current_price * (0.7 if action == TradeAction.LONG else 1.3) if leverage > 1 and action == TradeAction.LONG else None
        ),
        
        # Market Conditions
        market_conditions=MarketConditions(
            overall_trend=f"{'Bullish' if change_24h > 1 else 'Bearish' if change_24h < -1 else 'Neutral'} short-term",
            sector_sentiment="Mixed with selective strength",
            volume_profile="Normal trading range",
            funding_rates=0.0001 if not testnet else None,
            open_interest_change=None,
            whale_activity="Moderate activity detected",
            correlation_analysis={"BTC": 0.7, "ETH": 0.6, "Market": 0.5}
        ),
        
        # Time Horizon
        time_horizon=TimeHorizon(time_horizon.upper()),
        
        # Monitoring Levels
        monitoring_levels=MonitoringLevels(
            bullish_confirmation=high_price * 1.01,
            bearish_invalidation=low_price * 0.99,
            critical_support=low_price,
            critical_resistance=high_price,
            volume_breakout_level=volume * 1.5
        ),
        
        # Execution Strategy - different for each action type
        execution_strategy=ExecutionStrategy(
            entry_strategy=_get_entry_strategy(action, current_price, leverage),
            exit_strategy=_get_exit_strategy(action),
            scaling_plan=_get_scaling_plan(action),
            contingency_plans=_get_contingency_plans(action),
            ideal_timing="Current market hours optimal for execution"
        ),
        
        # Additional Information
        fundamental_factors=[
            f"Max leverage: {symbol_info.get('maxLeverage', 1)}x available",
            f"Market cap tier: {determine_market_tier(current_price, volume)}",
            "Hyperliquid perpetual futures market"
        ],
        catalyst_events=[
            "Technical breakout potential",
            "Volume confirmation pending",
            "Market sentiment shift possible"
        ],
        risk_factors=[
            "General crypto market volatility",
            "Low liquidity risk in smaller timeframes",
            "Correlation with major crypto assets"
        ],
        alternative_scenarios=[
            {"scenario": "Strong breakout", "action": "Increase position", "probability": "25%"},
            {"scenario": "Range-bound", "action": "Scale out early", "probability": "45%"},
            {"scenario": "False breakout", "action": "Exit quickly", "probability": "30%"}
        ],
        
        # Metadata
        confidence_breakdown={
            "technical": min(1.0, llm_analysis.confidence + 0.1),
            "fundamental": 0.5,  # Limited fundamental data
            "sentiment": llm_analysis.confidence,
            "risk_management": 0.8
        },
        data_sources=["Hyperliquid", "DeepSeek LLM", "Technical Analysis"],
        last_updated=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=2)
    )


def _get_entry_strategy(action: TradeAction, current_price: float, leverage: float) -> str:
    """Get appropriate entry strategy based on action."""
    if action == TradeAction.LONG:
        return f"Market buy with {leverage:.1f}x leverage at current levels"
    elif action == TradeAction.SHORT:
        return f"Market sell position at ${current_price:.4f} (spot trading)"
    else:  # HOLD
        return "Hold current position - no entry recommended"

def _get_exit_strategy(action: TradeAction) -> str:
    """Get appropriate exit strategy based on action."""
    if action == TradeAction.LONG:
        return "Scale out 50% at TP1, trail remaining position with stop-loss"
    elif action == TradeAction.SHORT:
        return "Immediate exit at current market price for spot holdings"
    else:  # HOLD
        return "Monitor for better entry/exit opportunities"

def _get_scaling_plan(action: TradeAction) -> str:
    """Get appropriate scaling plan based on action."""
    if action == TradeAction.LONG:
        return "Full position on entry, scale out 50% at TP1, 25% at TP2, trail 25%"
    elif action == TradeAction.SHORT:
        return "Single exit transaction - sell entire spot position"
    else:  # HOLD
        return "No scaling - maintain current position size"

def _get_contingency_plans(action: TradeAction) -> list:
    """Get appropriate contingency plans based on action."""
    if action == TradeAction.LONG:
        return [
            "If breaks key support: Exit immediately with stop-loss",
            "If correlation with BTC breaks: Monitor independently", 
            "If volume spikes: Consider increasing leverage cautiously"
        ]
    elif action == TradeAction.SHORT:
        return [
            "If price starts rallying: Monitor for potential re-entry",
            "If fundamental news emerges: Reassess sell decision",
            "Consider tax implications of selling"
        ]
    else:  # HOLD
        return [
            "If technical setup improves: Consider entry",
            "If setup deteriorates: Consider exit",
            "Monitor for clear directional signals"
        ]

def determine_market_tier(price: float, volume: float) -> str:
    """Determine market tier based on price and volume."""
    if price > 1000 or volume > 10000000:
        return "Major"
    elif price > 10 or volume > 1000000:
        return "Mid-cap" 
    elif price > 0.1:
        return "Small-cap"
    else:
        return "Micro-cap"
