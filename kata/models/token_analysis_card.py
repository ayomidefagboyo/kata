"""
Token Analysis Card Model

Standardized format for comprehensive token analysis responses
that provides all necessary information for trading decisions.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TradeAction(Enum):
    """Trade action recommendations."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    AVOID = "AVOID"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TimeHorizon(Enum):
    """Trade time horizons."""
    SCALP = "SCALP"      # Minutes to hours
    SHORT = "SHORT"      # Hours to 1-2 days
    MEDIUM = "MEDIUM"    # 2-7 days
    LONG = "LONG"        # 1-4 weeks
    HOLD = "HOLD"        # Months+


@dataclass
class EntryDetails:
    """Trade entry information."""
    current_price: float
    action: TradeAction
    leverage: float
    confidence: float  # 0.0 - 1.0
    optimal_entry_range: Dict[str, float]  # {"min": price, "max": price}


@dataclass
class TechnicalReasoning:
    """Technical analysis reasoning."""
    primary_signals: List[str]
    supporting_indicators: List[str]
    momentum_analysis: str
    trend_analysis: str
    volume_analysis: str
    support_resistance: Dict[str, float]  # {"support": price, "resistance": price}


@dataclass
class RiskManagement:
    """Risk management parameters."""
    stop_loss: float
    take_profit_levels: List[Dict[str, Any]]  # [{"level": 1, "price": float, "percentage": float}]
    position_size_percentage: float
    max_risk_per_trade: float
    trailing_stop: Optional[float] = None


@dataclass
class TradeMathematics:
    """Trade calculation details."""
    risk_amount: float
    reward_amounts: List[float]
    risk_reward_ratios: List[float]
    breakeven_price: float
    liquidation_price: Optional[float] = None


@dataclass
class MarketConditions:
    """Current market conditions."""
    overall_trend: str
    sector_sentiment: str
    volume_profile: str
    funding_rates: Optional[float]
    open_interest_change: Optional[float]
    whale_activity: str
    correlation_analysis: Dict[str, float]  # Correlation with major assets


@dataclass
class MonitoringLevels:
    """Key levels to monitor."""
    bullish_confirmation: float
    bearish_invalidation: float
    critical_support: float
    critical_resistance: float
    volume_breakout_level: float


@dataclass
class ExecutionStrategy:
    """Detailed execution plan."""
    entry_strategy: str
    exit_strategy: str
    scaling_plan: str
    contingency_plans: List[str]
    ideal_timing: str


@dataclass
class TokenAnalysisCard:
    """Complete token analysis card."""
    # Header Information
    symbol: str
    analysis_timestamp: datetime
    analyst: str  # LLM provider or analyst name
    market_tier: str  # "Major", "Mid-cap", "Small-cap", "Micro"
    
    # Core Analysis
    entry_details: EntryDetails
    technical_reasoning: TechnicalReasoning
    risk_management: RiskManagement
    trade_mathematics: TradeMathematics
    
    # Market Context
    market_conditions: MarketConditions
    time_horizon: TimeHorizon
    monitoring_levels: MonitoringLevels
    execution_strategy: ExecutionStrategy
    
    # Additional Information
    fundamental_factors: List[str]
    catalyst_events: List[str]
    risk_factors: List[str]
    alternative_scenarios: List[Dict[str, str]]  # [{"scenario": str, "action": str, "probability": str}]
    
    # Metadata
    confidence_breakdown: Dict[str, float]  # Technical, Fundamental, Sentiment scores
    data_sources: List[str]
    last_updated: datetime
    expires_at: datetime


def create_eth_analysis_card() -> TokenAnalysisCard:
    """Example ETH analysis card."""
    return TokenAnalysisCard(
        # Header
        symbol="ETH",
        analysis_timestamp=datetime.now(),
        analyst="Claude-3-Haiku (Yuki Agent)",
        market_tier="Major",
        
        # Entry Details
        entry_details=EntryDetails(
            current_price=3487.75,
            action=TradeAction.LONG,
            leverage=3.0,
            confidence=0.70,
            optimal_entry_range={"min": 3480.00, "max": 3495.00}
        ),
        
        # Technical Reasoning
        technical_reasoning=TechnicalReasoning(
            primary_signals=[
                "Bullish MACD crossover with strong momentum",
                "RSI at 58 - room for upward movement",
                "Breaking above 21-day EMA with volume"
            ],
            supporting_indicators=[
                "Positive social sentiment trend",
                "Volume profile showing accumulation",
                "Bollinger Bands expansion to upside"
            ],
            momentum_analysis="Strong bullish momentum with +2.82% daily gain supported by increasing volume",
            trend_analysis="Short-term uptrend confirmed, medium-term neutral to bullish alignment expected",
            volume_analysis="Above-average volume (+15% vs 30-day avg) confirms buying interest",
            support_resistance={"support": 3420.00, "resistance": 3520.00}
        ),
        
        # Risk Management
        risk_management=RiskManagement(
            stop_loss=3300.00,
            take_profit_levels=[
                {"level": 1, "price": 3650.00, "percentage": 4.7},
                {"level": 2, "price": 3800.00, "percentage": 9.0}
            ],
            position_size_percentage=15.0,
            max_risk_per_trade=2.5,
            trailing_stop=50.0
        ),
        
        # Trade Mathematics
        trade_mathematics=TradeMathematics(
            risk_amount=187.75,
            reward_amounts=[162.25, 312.25],
            risk_reward_ratios=[0.86, 1.66],
            breakeven_price=3487.75,
            liquidation_price=3325.00
        ),
        
        # Market Conditions
        market_conditions=MarketConditions(
            overall_trend="Bullish short-term, neutral medium-term",
            sector_sentiment="Positive for large-cap altcoins",
            volume_profile="Accumulation pattern visible",
            funding_rates=0.0008,
            open_interest_change=0.12,
            whale_activity="Moderate accumulation detected",
            correlation_analysis={"BTC": 0.75, "SPY": 0.45, "DXY": -0.32}
        ),
        
        # Time Horizon
        time_horizon=TimeHorizon.MEDIUM,
        
        # Monitoring Levels
        monitoring_levels=MonitoringLevels(
            bullish_confirmation=3520.00,
            bearish_invalidation=3450.00,
            critical_support=3400.00,
            critical_resistance=3600.00,
            volume_breakout_level=450000000.0
        ),
        
        # Execution Strategy
        execution_strategy=ExecutionStrategy(
            entry_strategy="Market buy at current levels or limit buy at $3,480",
            exit_strategy="Scale out 50% at TP1, trail stop to breakeven, let 50% run to TP2",
            scaling_plan="Full position on entry, scale out on profits",
            contingency_plans=[
                "If breaks below $3,450: Reassess and potentially exit",
                "If gaps down overnight: Reduce position size by 50%",
                "If BTC correlation breaks: Monitor for independent movement"
            ],
            ideal_timing="Next 2-4 hours during US market hours"
        ),
        
        # Additional Information
        fundamental_factors=[
            "Ethereum 2.0 staking yield attractive",
            "DeFi TVL showing growth",
            "Layer 2 adoption increasing"
        ],
        catalyst_events=[
            "ETF approval rumors circulating",
            "Major DeFi protocol launches this week",
            "Ethereum Foundation developer call tomorrow"
        ],
        risk_factors=[
            "General crypto market volatility",
            "Regulatory uncertainty in key markets",
            "Potential BTC dominance increase"
        ],
        alternative_scenarios=[
            {"scenario": "BTC breaks $115k", "action": "Increase position size", "probability": "30%"},
            {"scenario": "Market correction", "action": "Exit immediately", "probability": "20%"},
            {"scenario": "Sideways consolidation", "action": "Take profits early", "probability": "35%"}
        ],
        
        # Metadata
        confidence_breakdown={
            "technical": 0.75,
            "fundamental": 0.65,
            "sentiment": 0.70,
            "risk_management": 0.80
        },
        data_sources=["Hyperliquid", "CoinGecko", "Binance", "Claude LLM"],
        last_updated=datetime.now(),
        expires_at=datetime.now().replace(hour=datetime.now().hour + 4)  # 4 hours
    )


def format_analysis_card_response(card: TokenAnalysisCard) -> Dict[str, Any]:
    """Format analysis card for API response."""
    def format_price(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"${value:,.2f}"

    def format_stop_loss() -> str:
        stop_loss = card.risk_management.stop_loss
        if stop_loss is None:
            if card.entry_details.action == TradeAction.HOLD:
                return "N/A (no active trade recommended)"
            return "N/A (spot exit guidance)"

        distance_pct = (
            (stop_loss - card.entry_details.current_price)
            / card.entry_details.current_price
            * 100
        )
        return f"{format_price(stop_loss)} ({distance_pct:+.1f}% from entry)"

    action_bias = {
        TradeAction.LONG: "bullish",
        TradeAction.SHORT: "bearish",
        TradeAction.HOLD: "neutral",
        TradeAction.AVOID: "risk-off",
    }.get(card.entry_details.action, "neutral")

    return {
        "header": {
            "symbol": card.symbol,
            "timestamp": card.analysis_timestamp.isoformat(),
            "analyst": card.analyst,
            "market_tier": card.market_tier,
            "expires_at": card.expires_at.isoformat()
        },
        "trade_setup": {
            "entry_details": {
                "current_price": format_price(card.entry_details.current_price),
                "action": f"{card.entry_details.action.value} ({card.entry_details.action.value.lower()}/{action_bias} stance)",
                "leverage": f"{card.entry_details.leverage}x ({'conservative' if card.entry_details.leverage <= 3 else 'moderate' if card.entry_details.leverage <= 5 else 'aggressive'})",
                "confidence": f"{card.entry_details.confidence:.0%} ({'high' if card.entry_details.confidence >= 0.7 else 'moderate' if card.entry_details.confidence >= 0.5 else 'low'})"
            },
            "technical_reasoning": {
                "primary_signals": card.technical_reasoning.primary_signals,
                "momentum_analysis": card.technical_reasoning.momentum_analysis,
                "trend_analysis": card.technical_reasoning.trend_analysis,
                "volume_analysis": card.technical_reasoning.volume_analysis
            },
            "risk_management": {
                "stop_loss": format_stop_loss(),
                "take_profit_levels": [
                    f"TP{tp['level']}: {format_price(tp.get('price'))} ({tp['percentage']:+.1f}% profit target)"
                    for tp in card.risk_management.take_profit_levels
                ],
                "position_size": f"Maximum {card.risk_management.position_size_percentage}% of portfolio"
            }
        },
        "trade_mathematics": {
            "leverage_impact": f"With {card.entry_details.leverage}x leverage:",
            "risk_calculation": f"Risk: ${card.trade_mathematics.risk_amount:.2f} per token",
            "reward_calculation": [
                f"Reward (TP{i+1}): ${reward:.2f} per token"
                for i, reward in enumerate(card.trade_mathematics.reward_amounts)
            ],
            "risk_reward_ratios": [
                f"R:R Ratio (TP{i+1}): 1:{ratio:.1f}"
                for i, ratio in enumerate(card.trade_mathematics.risk_reward_ratios)
            ]
        },
        "execution_strategy": {
            "time_horizon": f"{card.time_horizon.value.title()} ({card.execution_strategy.ideal_timing})",
            "entry_strategy": card.execution_strategy.entry_strategy,
            "exit_strategy": card.execution_strategy.exit_strategy,
            "monitoring_levels": {
                "bullish_confirmation": f"Break above ${card.monitoring_levels.bullish_confirmation:,.2f}",
                "invalidation": f"Break below ${card.monitoring_levels.bearish_invalidation:,.2f}",
                "critical_support": f"${card.monitoring_levels.critical_support:,.2f}",
                "critical_resistance": f"${card.monitoring_levels.critical_resistance:,.2f}"
            }
        },
        "market_context": {
            "market_conditions": card.market_conditions.overall_trend,
            "catalyst_events": card.catalyst_events,
            "risk_factors": card.risk_factors,
            "alternative_scenarios": card.alternative_scenarios
        },
        "confidence_breakdown": {
            f"{key.title()}": f"{value:.0%}"
            for key, value in card.confidence_breakdown.items()
        },
        "footer": {
            "data_sources": card.data_sources,
            "last_updated": card.last_updated.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "disclaimer": "This analysis is for educational purposes only. Not financial advice."
        }
    }
