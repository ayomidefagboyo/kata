"""
Sentiment Analysis API Endpoints for Yuki Agent

Provides comprehensive sentiment analysis endpoints including:
- Unified sentiment analysis from all sources
- Individual source sentiment data
- Advanced sentiment algorithms and trading signals
- Market mood and regime analysis
- Real-time sentiment monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..services.sentiment_service import SentimentService, create_sentiment_service
from ..services.sentiment_analyzer import SentimentAnalyzer, create_sentiment_analyzer
from ..services.news_service import NewsService, create_news_service
from ..services.reddit_service import RedditService, create_reddit_service
from ..services.twitter_service import TwitterService, create_twitter_service
from ..services.fred_service import FREDService, create_fred_service
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Global service instances (will be initialized in lifespan)
sentiment_service: Optional[SentimentService] = None
sentiment_analyzer: Optional[SentimentAnalyzer] = None
news_service: Optional[NewsService] = None
reddit_service: Optional[RedditService] = None
twitter_service: Optional[TwitterService] = None
fred_service: Optional[FREDService] = None


async def get_sentiment_service() -> SentimentService:
    """Get sentiment service dependency."""
    if sentiment_service is None:
        # Try to create a minimal fallback service
        try:
            fallback_service = create_sentiment_service()
            logger.warning("Using fallback sentiment service")
            return fallback_service
        except Exception as e:
            logger.error(f"Failed to create fallback sentiment service: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Sentiment service unavailable - API keys may not be configured"
            )
    return sentiment_service


async def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get sentiment analyzer dependency."""
    if sentiment_analyzer is None:
        # Try to create a fallback analyzer
        try:
            fallback_analyzer = create_sentiment_analyzer()
            logger.warning("Using fallback sentiment analyzer")
            return fallback_analyzer
        except Exception as e:
            logger.error(f"Failed to create fallback sentiment analyzer: {e}")
            raise HTTPException(
                status_code=503, 
                detail="Sentiment analyzer unavailable"
            )
    return sentiment_analyzer


def initialize_sentiment_services():
    """Initialize all sentiment-related services."""
    global sentiment_service, sentiment_analyzer, news_service, reddit_service, twitter_service, fred_service
    
    try:
        # Initialize individual services based on available API keys
        news_service = None
        if settings.NEWSAPI_API_KEY:
            news_service = create_news_service(settings.NEWSAPI_API_KEY)
            logger.info("NewsService initialized")
        
        reddit_service = None
        if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
            reddit_service = create_reddit_service(
                settings.REDDIT_CLIENT_ID,
                settings.REDDIT_CLIENT_SECRET,
                settings.REDDIT_USER_AGENT
            )
            logger.info("RedditService initialized")
        
        twitter_service = None
        if (settings.TWITTER_API_KEY and settings.TWITTER_API_SECRET and 
            settings.TWITTER_ACCESS_TOKEN and settings.TWITTER_ACCESS_TOKEN_SECRET and 
            settings.TWITTER_BEARER_TOKEN):
            twitter_service = create_twitter_service(
                settings.TWITTER_API_KEY,
                settings.TWITTER_API_SECRET,
                settings.TWITTER_ACCESS_TOKEN,
                settings.TWITTER_ACCESS_TOKEN_SECRET,
                settings.TWITTER_BEARER_TOKEN
            )
            logger.info("TwitterService initialized")
        
        fred_service = None
        if settings.FRED_API_KEY:
            fred_service = create_fred_service(settings.FRED_API_KEY)
            logger.info("FREDService initialized")
        
        # Initialize unified sentiment service
        sentiment_service = create_sentiment_service(
            news_service, reddit_service, twitter_service, fred_service
        )
        logger.info("SentimentService initialized")
        
        # Initialize sentiment analyzer
        sentiment_analyzer = create_sentiment_analyzer()
        logger.info("SentimentAnalyzer initialized")
        
        # Log which services are available
        available_services = []
        if news_service:
            available_services.append("News")
        if reddit_service:
            available_services.append("Reddit")
        if twitter_service:
            available_services.append("Twitter")
        if fred_service:
            available_services.append("FRED")
        
        logger.info(f"Sentiment services initialized: {', '.join(available_services) if available_services else 'None (running with defaults)'}")
        
    except Exception as e:
        logger.error(f"Error initializing sentiment services: {e}")
        # Initialize with minimal services for fallback
        try:
            sentiment_service = create_sentiment_service()
            sentiment_analyzer = create_sentiment_analyzer()
            logger.info("Fallback sentiment services initialized")
        except Exception as fallback_error:
            logger.critical(f"Failed to initialize fallback sentiment services: {fallback_error}")
            # Set to None - services will handle gracefully
            sentiment_service = None
            sentiment_analyzer = None


@router.get("/unified", summary="Get unified sentiment analysis")
async def get_unified_sentiment(
    hours_back: int = 24,
    service: SentimentService = Depends(get_sentiment_service)
) -> Dict[str, Any]:
    """
    Get comprehensive unified sentiment analysis from all available sources.
    
    Args:
        hours_back: Number of hours to look back for sentiment data (default: 24)
        
    Returns:
        Unified sentiment analysis including all sources, trends, and insights
    """
    try:
        unified_sentiment = await service.get_unified_sentiment(hours_back)
        
        return {
            "overall_sentiment": unified_sentiment.overall_sentiment,
            "sentiment_label": unified_sentiment.sentiment_label,
            "confidence": unified_sentiment.confidence,
            "sentiment_trend": unified_sentiment.sentiment_trend,
            "trend_strength": unified_sentiment.trend_strength,
            "trend_duration": unified_sentiment.trend_duration,
            "market_mood": {
                "primary_mood": unified_sentiment.market_mood.primary_mood,
                "secondary_mood": unified_sentiment.market_mood.secondary_mood,
                "mood_intensity": unified_sentiment.market_mood.mood_intensity,
                "mood_confidence": unified_sentiment.market_mood.mood_confidence,
                "dominant_emotion": unified_sentiment.market_mood.dominant_emotion,
                "market_phase": unified_sentiment.market_mood.market_phase
            },
            "sources": {
                "news": {
                    "sentiment": unified_sentiment.news_sentiment.sentiment,
                    "confidence": unified_sentiment.news_sentiment.confidence,
                    "sample_size": unified_sentiment.news_sentiment.sample_size,
                    "trend": unified_sentiment.news_sentiment.trend
                },
                "reddit": {
                    "sentiment": unified_sentiment.reddit_sentiment.sentiment,
                    "confidence": unified_sentiment.reddit_sentiment.confidence,
                    "sample_size": unified_sentiment.reddit_sentiment.sample_size,
                    "trend": unified_sentiment.reddit_sentiment.trend
                },
                "twitter": {
                    "sentiment": unified_sentiment.twitter_sentiment.sentiment,
                    "confidence": unified_sentiment.twitter_sentiment.confidence,
                    "sample_size": unified_sentiment.twitter_sentiment.sample_size,
                    "trend": unified_sentiment.twitter_sentiment.trend
                },
                "macro": {
                    "sentiment": unified_sentiment.macro_sentiment.sentiment,
                    "confidence": unified_sentiment.macro_sentiment.confidence,
                    "sample_size": unified_sentiment.macro_sentiment.sample_size,
                    "trend": unified_sentiment.macro_sentiment.trend
                },
                "social": {
                    "sentiment": unified_sentiment.social_sentiment.sentiment,
                    "confidence": unified_sentiment.social_sentiment.confidence,
                    "sample_size": unified_sentiment.social_sentiment.sample_size,
                    "trend": unified_sentiment.social_sentiment.trend
                }
            },
            "signals": {
                "bullish": unified_sentiment.bullish_signals,
                "bearish": unified_sentiment.bearish_signals,
                "neutral": unified_sentiment.neutral_signals
            },
            "risk_metrics": {
                "fear_greed_index": unified_sentiment.fear_greed_index,
                "volatility_expectation": unified_sentiment.volatility_expectation,
                "market_stress": unified_sentiment.market_stress,
                "source_agreement": unified_sentiment.source_agreement,
                "divergence_signals": unified_sentiment.divergence_signals
            },
            "trading_implications": {
                "trading_bias": unified_sentiment.trading_bias,
                "position_sizing": unified_sentiment.position_sizing,
                "risk_level": unified_sentiment.risk_level
            },
            "timestamp": unified_sentiment.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting unified sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment analysis: {str(e)}")


@router.get("/advanced", summary="Get advanced sentiment analysis")
async def get_advanced_sentiment(
    hours_back: int = 24,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
) -> Dict[str, Any]:
    """
    Get advanced sentiment analysis with sophisticated algorithms and trading signals.
    
    Args:
        hours_back: Number of hours to look back for sentiment data (default: 24)
        
    Returns:
        Advanced sentiment analysis with multi-dimensional scoring and trading signals
    """
    try:
        # Get unified sentiment first
        unified_sentiment = await sentiment_service.get_unified_sentiment(hours_back)
        
        # Perform advanced analysis
        advanced_analysis = analyzer.analyze_sentiment(unified_sentiment)
        
        return {
            "overall_score": advanced_analysis.overall_score,
            "dimensions": {
                "bullish_bearish": advanced_analysis.dimensions.bullish_bearish,
                "fear_greed": advanced_analysis.dimensions.fear_greed,
                "confidence_uncertainty": advanced_analysis.dimensions.confidence_uncertainty,
                "euphoria_depression": advanced_analysis.dimensions.euphoria_depression,
                "fomo_fud": advanced_analysis.dimensions.fomo_fud,
                "accumulation_distribution": advanced_analysis.dimensions.accumulation_distribution
            },
            "temporal": {
                "current_sentiment": advanced_analysis.temporal.current_sentiment,
                "momentum": advanced_analysis.temporal.momentum,
                "acceleration": advanced_analysis.temporal.acceleration,
                "volatility": advanced_analysis.temporal.volatility,
                "mean_reversion_signal": advanced_analysis.temporal.mean_reversion_signal,
                "breakout_signal": advanced_analysis.temporal.breakout_signal,
                "cycle_position": advanced_analysis.temporal.cycle_position,
                "time_decay_factor": advanced_analysis.temporal.time_decay_factor
            },
            "regime": {
                "regime": advanced_analysis.regime.regime,
                "confidence": advanced_analysis.regime.confidence,
                "regime_strength": advanced_analysis.regime.regime_strength,
                "regime_duration": advanced_analysis.regime.regime_duration,
                "transition_probability": advanced_analysis.regime.transition_probability,
                "next_likely_regime": advanced_analysis.regime.next_likely_regime,
                "characteristics": advanced_analysis.regime.regime_characteristics
            },
            "risk": {
                "risk_appetite": advanced_analysis.risk.risk_appetite,
                "systemic_risk": advanced_analysis.risk.systemic_risk,
                "tail_risk": advanced_analysis.risk.tail_risk,
                "correlation_risk": advanced_analysis.risk.correlation_risk,
                "liquidity_risk": advanced_analysis.risk.liquidity_risk,
                "volatility_regime": advanced_analysis.risk.volatility_regime,
                "stress_level": advanced_analysis.risk.stress_level
            },
            "trading_signals": {
                "primary_signal": advanced_analysis.signals.primary_signal,
                "signal_strength": advanced_analysis.signals.signal_strength,
                "time_horizon": advanced_analysis.signals.time_horizon,
                "confidence": advanced_analysis.signals.confidence,
                "risk_reward_ratio": advanced_analysis.signals.risk_reward_ratio,
                "position_size_recommendation": advanced_analysis.signals.position_size_recommendation,
                "entry_conditions": advanced_analysis.signals.entry_conditions,
                "exit_conditions": advanced_analysis.signals.exit_conditions,
                "stop_loss_level": advanced_analysis.signals.stop_loss_level,
                "take_profit_levels": advanced_analysis.signals.take_profit_levels
            },
            "anomalies": advanced_analysis.anomalies,
            "correlations": advanced_analysis.correlations,
            "contrarian_indicators": advanced_analysis.contrarian_indicators,
            "consensus_breakdown": advanced_analysis.consensus_breakdown,
            "timestamp": advanced_analysis.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting advanced sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get advanced sentiment analysis: {str(e)}")


@router.get("/news", summary="Get news sentiment analysis")
async def get_news_sentiment(
    hours_back: int = 24,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get sentiment analysis from financial news sources.
    
    Args:
        hours_back: Number of hours to look back (default: 24)
        limit: Maximum number of articles to analyze (default: 50)
        
    Returns:
        News sentiment analysis and recent articles
    """
    try:
        if not news_service:
            raise HTTPException(status_code=503, detail="News service not available - API key not configured")
        
        # Get news sentiment with error handling
        try:
            async with news_service:
                news_sentiment = await news_service.get_market_sentiment(hours_back)
                recent_articles = await news_service.get_crypto_news(limit, hours_back)
        except Exception as api_error:
            logger.error(f"Error fetching news data: {api_error}")
            # Return fallback data
            return {
                "sentiment": {
                    "overall_sentiment": 0.0,
                    "sentiment_label": "neutral",
                    "article_count": 0,
                    "confidence": 0.0,
                    "error": "News API temporarily unavailable"
                },
                "recent_articles": [],
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "sentiment": {
                "overall_sentiment": news_sentiment.overall_sentiment,
                "sentiment_label": news_sentiment.sentiment_label,
                "article_count": news_sentiment.article_count,
                "positive_articles": news_sentiment.positive_articles,
                "negative_articles": news_sentiment.negative_articles,
                "neutral_articles": news_sentiment.neutral_articles,
                "trending_keywords": news_sentiment.trending_keywords,
                "sentiment_trend": news_sentiment.sentiment_trend,
                "market_impact": news_sentiment.market_impact,
                "confidence": news_sentiment.confidence
            },
            "recent_articles": [
                {
                    "title": article.title,
                    "description": article.description,
                    "source": article.source,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "sentiment_score": article.sentiment_score,
                    "sentiment_label": article.sentiment_label,
                    "relevance_score": article.relevance_score,
                    "impact_score": article.impact_score,
                    "keywords": article.keywords,
                    "category": article.category
                }
                for article in recent_articles[:10]  # Return top 10 articles
            ],
            "timestamp": news_sentiment.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting news sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get news sentiment: {str(e)}")


@router.get("/social", summary="Get social media sentiment analysis")
async def get_social_sentiment(
    hours_back: int = 24,
    include_reddit: bool = True,
    include_twitter: bool = True
) -> Dict[str, Any]:
    """
    Get sentiment analysis from social media sources (Reddit and Twitter).
    
    Args:
        hours_back: Number of hours to look back (default: 24)
        include_reddit: Include Reddit sentiment analysis
        include_twitter: Include Twitter sentiment analysis
        
    Returns:
        Social media sentiment analysis
    """
    try:
        result = {
            "timestamp": datetime.now().isoformat()
        }
        
        # Reddit sentiment
        if include_reddit and reddit_service:
            try:
                reddit_sentiment = await reddit_service.get_crypto_sentiment(hours_back)
                result["reddit"] = {
                    "overall_sentiment": reddit_sentiment.overall_sentiment,
                    "sentiment_label": reddit_sentiment.sentiment_label,
                    "total_posts": reddit_sentiment.total_posts,
                    "total_comments": reddit_sentiment.total_comments,
                    "trending_coins": reddit_sentiment.trending_coins,
                    "market_mood": reddit_sentiment.market_mood,
                    "social_volume": reddit_sentiment.social_volume,
                    "sentiment_distribution": reddit_sentiment.sentiment_distribution,
                    "confidence": reddit_sentiment.confidence,
                    "subreddit_breakdown": {
                        name: {
                            "sentiment": data.overall_sentiment,
                            "post_count": data.post_count,
                            "activity_level": data.activity_level,
                            "trending_keywords": data.trending_keywords[:5]
                        }
                        for name, data in reddit_sentiment.subreddit_sentiments.items()
                    }
                }
            except Exception as e:
                logger.warning(f"Error getting Reddit sentiment: {e}")
                result["reddit"] = {"error": "Reddit sentiment unavailable"}
        elif include_reddit:
            result["reddit"] = {"error": "Reddit service not configured"}
        
        # Twitter sentiment
        if include_twitter and twitter_service:
            try:
                twitter_sentiment = await twitter_service.get_crypto_sentiment(hours_back)
                result["twitter"] = {
                    "overall_sentiment": twitter_sentiment.overall_sentiment,
                    "sentiment_label": twitter_sentiment.sentiment_label,
                    "tweet_count": twitter_sentiment.tweet_count,
                    "trending_hashtags": twitter_sentiment.trending_hashtags,
                    "trending_keywords": twitter_sentiment.trending_keywords,
                    "social_volume": twitter_sentiment.social_volume,
                    "engagement_rate": twitter_sentiment.engagement_rate,
                    "sentiment_trend": twitter_sentiment.sentiment_trend,
                    "market_mood": twitter_sentiment.market_mood,
                    "confidence": twitter_sentiment.confidence,
                    "top_influencers": [
                        {
                            "username": inf.username,
                            "followers": inf.followers_count,
                            "recent_sentiment": inf.recent_sentiment,
                            "influence_score": inf.influence_score
                        }
                        for inf in twitter_sentiment.top_influencers[:5]
                    ]
                }
            except Exception as e:
                logger.warning(f"Error getting Twitter sentiment: {e}")
                result["twitter"] = {"error": "Twitter sentiment unavailable"}
        elif include_twitter:
            result["twitter"] = {"error": "Twitter service not configured"}
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting social sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get social sentiment: {str(e)}")


@router.get("/macro", summary="Get macro economic sentiment")
async def get_macro_sentiment() -> Dict[str, Any]:
    """
    Get sentiment analysis from macro economic indicators.
    
    Returns:
        Macro economic sentiment and key indicators
    """
    try:
        if not fred_service:
            raise HTTPException(status_code=503, detail="FRED service not available - API key not configured")
        
        try:
            async with fred_service:
                macro_data = await fred_service.get_macro_data()
        except Exception as api_error:
            logger.error(f"Error fetching FRED data: {api_error}")
            return {
                "macro_score": 0.0,
                "risk_environment": "neutral",
                "market_regime": "unknown",
                "crypto_correlation": 0.0,
                "categories": {},
                "error": "FRED API temporarily unavailable",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "macro_score": macro_data.macro_score,
            "risk_environment": macro_data.risk_environment,
            "market_regime": macro_data.market_regime,
            "crypto_correlation": macro_data.crypto_correlation,
            "categories": {
                "monetary_policy": {
                    name: {
                        "value": indicator.value,
                        "change_percent": indicator.change_percent,
                        "trend": indicator.trend,
                        "market_impact": indicator.market_impact,
                        "significance": indicator.significance
                    }
                    for name, indicator in macro_data.monetary_policy.items()
                },
                "growth_indicators": {
                    name: {
                        "value": indicator.value,
                        "change_percent": indicator.change_percent,
                        "trend": indicator.trend,
                        "market_impact": indicator.market_impact
                    }
                    for name, indicator in macro_data.growth_indicators.items()
                },
                "market_indicators": {
                    name: {
                        "value": indicator.value,
                        "change_percent": indicator.change_percent,
                        "trend": indicator.trend,
                        "market_impact": indicator.market_impact
                    }
                    for name, indicator in macro_data.market_indicators.items()
                }
            },
            "timestamp": macro_data.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting macro sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get macro sentiment: {str(e)}")


@router.get("/summary", summary="Get sentiment summary dashboard")
async def get_sentiment_summary(
    hours_back: int = 24,
    service: SentimentService = Depends(get_sentiment_service)
) -> Dict[str, Any]:
    """
    Get a comprehensive sentiment summary for dashboard display.
    
    Args:
        hours_back: Number of hours to look back (default: 24)
        
    Returns:
        Sentiment summary with key metrics and insights
    """
    try:
        unified_sentiment = await service.get_unified_sentiment(hours_back)
        
        # Create summary with key insights
        summary = {
            "overall": {
                "sentiment_score": unified_sentiment.overall_sentiment,
                "sentiment_label": unified_sentiment.sentiment_label,
                "confidence": unified_sentiment.confidence,
                "fear_greed_index": unified_sentiment.fear_greed_index
            },
            "market_mood": {
                "primary": unified_sentiment.market_mood.primary_mood,
                "emotion": unified_sentiment.market_mood.dominant_emotion,
                "phase": unified_sentiment.market_mood.market_phase
            },
            "trend": {
                "direction": unified_sentiment.sentiment_trend,
                "strength": unified_sentiment.trend_strength,
                "duration_hours": unified_sentiment.trend_duration
            },
            "trading": {
                "bias": unified_sentiment.trading_bias,
                "position_sizing": unified_sentiment.position_sizing,
                "risk_level": unified_sentiment.risk_level
            },
            "sources": {
                "agreement": unified_sentiment.source_agreement,
                "active_sources": sum(1 for source in [
                    unified_sentiment.news_sentiment,
                    unified_sentiment.reddit_sentiment,
                    unified_sentiment.twitter_sentiment,
                    unified_sentiment.macro_sentiment
                ] if source.confidence > 0)
            },
            "key_signals": {
                "bullish_count": len(unified_sentiment.bullish_signals),
                "bearish_count": len(unified_sentiment.bearish_signals),
                "top_bullish": unified_sentiment.bullish_signals[:3],
                "top_bearish": unified_sentiment.bearish_signals[:3]
            },
            "risk_metrics": {
                "market_stress": unified_sentiment.market_stress,
                "volatility_expectation": unified_sentiment.volatility_expectation,
                "divergence_count": len(unified_sentiment.divergence_signals)
            },
            "timestamp": unified_sentiment.timestamp.isoformat()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment summary: {str(e)}")


@router.get("/health", summary="Check sentiment services health")
async def check_sentiment_health() -> Dict[str, Any]:
    """
    Check the health and availability of all sentiment services.
    
    Returns:
        Health status of each sentiment service
    """
    try:
        health_status = {
            "overall_status": "healthy",
            "services": {
                "news": {
                    "available": news_service is not None,
                    "configured": bool(settings.NEWSAPI_API_KEY)
                },
                "reddit": {
                    "available": reddit_service is not None,
                    "configured": bool(settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET)
                },
                "twitter": {
                    "available": twitter_service is not None,
                    "configured": bool(settings.TWITTER_API_KEY and settings.TWITTER_BEARER_TOKEN)
                },
                "fred": {
                    "available": fred_service is not None,
                    "configured": bool(settings.FRED_API_KEY)
                },
                "sentiment_service": {
                    "available": sentiment_service is not None,
                    "configured": True
                },
                "sentiment_analyzer": {
                    "available": sentiment_analyzer is not None,
                    "configured": True
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine overall status
        available_services = sum(1 for service in health_status["services"].values() if service["available"])
        if available_services == 0:
            health_status["overall_status"] = "critical"
        elif available_services < 3:
            health_status["overall_status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking sentiment health: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/status", summary="Get comprehensive sentiment system status")
async def get_sentiment_system_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the sentiment analysis system.
    
    Returns:
        Detailed status of all sentiment components and data sources
    """
    try:
        status = {
            "overall_status": "healthy",
            "core_services": {
                "sentiment_service": {"status": "unknown", "error": None},
                "sentiment_analyzer": {"status": "unknown", "error": None}
            },
            "data_sources": {
                "news": {"status": "unknown", "error": None, "last_check": None},
                "reddit": {"status": "unknown", "error": None, "last_check": None},
                "twitter": {"status": "unknown", "error": None, "last_check": None},
                "fred": {"status": "unknown", "error": None, "last_check": None}
            },
            "api_quotas": {
                "news": {"remaining": "unknown", "reset": None},
                "reddit": {"remaining": "unknown", "reset": None},
                "twitter": {"remaining": "unknown", "reset": None},
                "fred": {"remaining": "unknown", "reset": None}
            },
            "performance": {
                "avg_response_time": 0.0,
                "cache_hit_rate": 0.0,
                "error_rate": 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Check core services
        try:
            test_service = await get_sentiment_service()
            status["core_services"]["sentiment_service"]["status"] = "healthy"
        except Exception as e:
            status["core_services"]["sentiment_service"]["status"] = "error"
            status["core_services"]["sentiment_service"]["error"] = str(e)
        
        try:
            test_analyzer = await get_sentiment_analyzer()
            status["core_services"]["sentiment_analyzer"]["status"] = "healthy"
        except Exception as e:
            status["core_services"]["sentiment_analyzer"]["status"] = "error"
            status["core_services"]["sentiment_analyzer"]["error"] = str(e)
        
        # Check data sources
        if news_service:
            try:
                # Quick test of news service
                async with news_service:
                    await news_service.get_market_sentiment(1)  # 1 hour back
                status["data_sources"]["news"]["status"] = "healthy"
                status["data_sources"]["news"]["last_check"] = datetime.now().isoformat()
            except Exception as e:
                status["data_sources"]["news"]["status"] = "error"
                status["data_sources"]["news"]["error"] = str(e)
        else:
            status["data_sources"]["news"]["status"] = "not_configured"
        
        if reddit_service:
            try:
                # Quick test of Reddit service
                await reddit_service.get_crypto_sentiment(1)  # 1 hour back
                status["data_sources"]["reddit"]["status"] = "healthy"
                status["data_sources"]["reddit"]["last_check"] = datetime.now().isoformat()
            except Exception as e:
                status["data_sources"]["reddit"]["status"] = "error"
                status["data_sources"]["reddit"]["error"] = str(e)
        else:
            status["data_sources"]["reddit"]["status"] = "not_configured"
        
        if twitter_service:
            try:
                # Quick test of Twitter service
                await twitter_service.get_crypto_sentiment(1)  # 1 hour back
                status["data_sources"]["twitter"]["status"] = "healthy"
                status["data_sources"]["twitter"]["last_check"] = datetime.now().isoformat()
            except Exception as e:
                status["data_sources"]["twitter"]["status"] = "error"
                status["data_sources"]["twitter"]["error"] = str(e)
        else:
            status["data_sources"]["twitter"]["status"] = "not_configured"
        
        if fred_service:
            try:
                # Quick test of FRED service
                async with fred_service:
                    await fred_service.get_macro_data()
                status["data_sources"]["fred"]["status"] = "healthy"
                status["data_sources"]["fred"]["last_check"] = datetime.now().isoformat()
            except Exception as e:
                status["data_sources"]["fred"]["status"] = "error"
                status["data_sources"]["fred"]["error"] = str(e)
        else:
            status["data_sources"]["fred"]["status"] = "not_configured"
        
        # Determine overall status
        error_count = 0
        not_configured_count = 0
        
        for service_status in [status["core_services"]["sentiment_service"], status["core_services"]["sentiment_analyzer"]]:
            if service_status["status"] == "error":
                error_count += 1
        
        for source_status in status["data_sources"].values():
            if source_status["status"] == "error":
                error_count += 1
            elif source_status["status"] == "not_configured":
                not_configured_count += 1
        
        if error_count > 1:
            status["overall_status"] = "critical"
        elif error_count == 1:
            status["overall_status"] = "degraded"
        elif not_configured_count >= 3:
            status["overall_status"] = "limited"
        else:
            status["overall_status"] = "healthy"
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting sentiment system status: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }