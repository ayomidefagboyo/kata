"""
Unified Sentiment Analysis Service for Yuki Agent

Aggregates and analyzes sentiment from multiple sources including:
- News articles (NewsAPI and RSS feeds)
- Reddit discussions (crypto subreddits)
- Twitter/X social sentiment (crypto Twitter)
- Macro economic sentiment (FRED indicators)

Provides comprehensive sentiment scoring and market mood analysis
for informed trading decisions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
# from .news_service import NewsService, NewsSentiment  # Temporarily disabled due to newsapi dependency
# from .reddit_service import RedditService, CryptoSentiment  # Temporarily disabled due to praw dependency
# from .twitter_service import TwitterService, TwitterSentiment  # Temporarily disabled due to tweepy dependency
# from .fred_service import FREDService, MacroData  # Temporarily disabled

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Individual sentiment score from a source."""
    source: str
    sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    weight: float  # Source weight in overall calculation
    timestamp: datetime
    sample_size: int  # Number of data points analyzed
    trend: str  # 'improving', 'declining', 'stable'


@dataclass
class MarketMood:
    """Overall market mood assessment."""
    primary_mood: str  # 'bullish', 'bearish', 'neutral', 'fearful', 'euphoric'
    secondary_mood: str  # Additional mood descriptor
    mood_intensity: float  # 0 to 1
    mood_confidence: float  # 0 to 1
    dominant_emotion: str  # 'greed', 'fear', 'optimism', 'pessimism', 'uncertainty'
    market_phase: str  # 'accumulation', 'markup', 'distribution', 'markdown'


@dataclass
class UnifiedSentiment:
    """Comprehensive sentiment analysis from all sources."""
    overall_sentiment: float  # -1 to 1 (weighted average)
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0 to 1 (overall confidence in sentiment)
    
    # Individual source sentiments
    news_sentiment: SentimentScore
    social_sentiment: SentimentScore  # Combined Reddit + Twitter
    reddit_sentiment: SentimentScore
    twitter_sentiment: SentimentScore
    macro_sentiment: SentimentScore
    
    # Trend analysis
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0 to 1
    trend_duration: int  # Hours the trend has been active
    
    # Market mood
    market_mood: MarketMood
    
    # Key insights
    bullish_signals: List[str]
    bearish_signals: List[str]
    neutral_signals: List[str]
    
    # Risk assessment
    fear_greed_index: float  # 0 to 100 (0 = extreme fear, 100 = extreme greed)
    volatility_expectation: str  # 'high', 'medium', 'low'
    market_stress: float  # 0 to 1
    
    # Correlation and divergence
    source_agreement: float  # 0 to 1 (how much sources agree)
    divergence_signals: List[str]  # When sources disagree significantly
    
    # Trading implications
    trading_bias: str  # 'long', 'short', 'neutral', 'caution'
    position_sizing: str  # 'aggressive', 'moderate', 'conservative', 'minimal'
    risk_level: str  # 'high', 'medium', 'low'
    
    timestamp: datetime


class SentimentService:
    """
    Unified sentiment analysis service that combines multiple data sources
    to provide comprehensive market sentiment analysis.
    """
    
    def __init__(self,
                 news_service: Optional[Any] = None,  # NewsService temporarily disabled
                 reddit_service: Optional[Any] = None,  # RedditService temporarily disabled
                 twitter_service: Optional[Any] = None,  # TwitterService temporarily disabled
                 fred_service: Optional[Any] = None):  # FREDService temporarily disabled
        """Initialize unified sentiment service."""
        self.news_service = news_service
        self.reddit_service = reddit_service
        self.twitter_service = twitter_service
        self.fred_service = fred_service
        
        # Source weights for overall sentiment calculation
        self.source_weights = {
            'news': 0.25,      # 25% - News and media sentiment
            'reddit': 0.20,    # 20% - Reddit community sentiment
            'twitter': 0.25,   # 25% - Twitter social sentiment
            'macro': 0.30      # 30% - Macro economic environment (highest weight)
        }
        
        # Cache settings
        self.cache_duration = {
            'unified_sentiment': 300,  # 5 minutes
            'market_mood': 600,        # 10 minutes
            'trend_analysis': 900      # 15 minutes
        }
        self.cache = {}
        
        # Historical sentiment for trend analysis
        self.sentiment_history = []
        self.max_history = 48  # Keep 48 data points (24 hours if updated every 30 min)
        
        logger.info("SentimentService initialized")
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = (datetime.now() - cached_data['timestamp']).total_seconds()
        return cache_age < self.cache_duration.get(cache_type, 300)
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_unified_sentiment(self, hours_back: int = 24) -> UnifiedSentiment:
        """
        Get comprehensive unified sentiment analysis.
        
        Args:
            hours_back: How many hours back to analyze
            
        Returns:
            UnifiedSentiment object with complete analysis
        """
        try:
            cache_key = f"unified_sentiment_{hours_back}"
            
            if self._is_cache_valid(cache_key, 'unified_sentiment'):
                return self.cache[cache_key]['data']
            
            # Collect sentiment from all sources
            sentiment_scores = await self._collect_all_sentiments(hours_back)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_weighted_sentiment(sentiment_scores)
            
            # Determine sentiment label
            if overall_sentiment > 0.2:
                sentiment_label = "positive"
            elif overall_sentiment < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(sentiment_scores)
            
            # Analyze trends
            sentiment_trend, trend_strength, trend_duration = self._analyze_sentiment_trend(overall_sentiment)
            
            # Determine market mood
            market_mood = self._determine_market_mood(sentiment_scores, overall_sentiment)
            
            # Generate insights and signals
            bullish_signals, bearish_signals, neutral_signals = self._generate_signals(sentiment_scores)
            
            # Calculate fear & greed index
            fear_greed_index = self._calculate_fear_greed_index(sentiment_scores, overall_sentiment)
            
            # Assess volatility and stress
            volatility_expectation, market_stress = self._assess_market_stress(sentiment_scores)
            
            # Analyze source agreement and divergence
            source_agreement, divergence_signals = self._analyze_source_agreement(sentiment_scores)
            
            # Generate trading implications
            trading_bias, position_sizing, risk_level = self._generate_trading_implications(
                overall_sentiment, confidence, market_mood, source_agreement
            )
            
            unified_sentiment = UnifiedSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                confidence=confidence,
                news_sentiment=sentiment_scores.get('news', self._empty_sentiment_score('news')),
                social_sentiment=self._combine_social_sentiment(sentiment_scores),
                reddit_sentiment=sentiment_scores.get('reddit', self._empty_sentiment_score('reddit')),
                twitter_sentiment=sentiment_scores.get('twitter', self._empty_sentiment_score('twitter')),
                macro_sentiment=sentiment_scores.get('macro', self._empty_sentiment_score('macro')),
                sentiment_trend=sentiment_trend,
                trend_strength=trend_strength,
                trend_duration=trend_duration,
                market_mood=market_mood,
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
                neutral_signals=neutral_signals,
                fear_greed_index=fear_greed_index,
                volatility_expectation=volatility_expectation,
                market_stress=market_stress,
                source_agreement=source_agreement,
                divergence_signals=divergence_signals,
                trading_bias=trading_bias,
                position_sizing=position_sizing,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            # Store in history for trend analysis
            self._update_sentiment_history(overall_sentiment)
            
            self._cache_data(cache_key, unified_sentiment)
            return unified_sentiment
            
        except Exception as e:
            logger.error(f"Error getting unified sentiment: {e}")
            return self._empty_unified_sentiment()
    
    async def _collect_all_sentiments(self, hours_back: int) -> Dict[str, SentimentScore]:
        """Collect sentiment data from all available sources."""
        sentiment_scores = {}
        
        # Collect all sentiments concurrently
        tasks = []
        
        if self.news_service:
            tasks.append(self._get_news_sentiment(hours_back))
        
        if self.reddit_service:
            tasks.append(self._get_reddit_sentiment(hours_back))
        
        if self.twitter_service:
            tasks.append(self._get_twitter_sentiment(hours_back))
        
        if self.fred_service:
            tasks.append(self._get_macro_sentiment())
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            task_names = []
            if self.news_service:
                task_names.append('news')
            if self.reddit_service:
                task_names.append('reddit')
            if self.twitter_service:
                task_names.append('twitter')
            if self.fred_service:
                task_names.append('macro')
            
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and result:
                    sentiment_scores[task_names[i]] = result
                else:
                    logger.warning(f"Failed to get {task_names[i]} sentiment: {result}")
        
        return sentiment_scores
    
    async def _get_news_sentiment(self, hours_back: int) -> SentimentScore:
        """Get sentiment from news sources."""
        try:
            news_sentiment = await self.news_service.get_market_sentiment(hours_back)
            
            return SentimentScore(
                source='news',
                sentiment=news_sentiment.overall_sentiment,
                confidence=news_sentiment.confidence,
                weight=self.source_weights['news'],
                timestamp=news_sentiment.timestamp,
                sample_size=news_sentiment.article_count,
                trend=news_sentiment.sentiment_trend
            )
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return self._empty_sentiment_score('news')
    
    async def _get_reddit_sentiment(self, hours_back: int) -> SentimentScore:
        """Get sentiment from Reddit sources."""
        try:
            reddit_sentiment = await self.reddit_service.get_crypto_sentiment(hours_back)
            
            return SentimentScore(
                source='reddit',
                sentiment=reddit_sentiment.overall_sentiment,
                confidence=reddit_sentiment.confidence,
                weight=self.source_weights['reddit'],
                timestamp=reddit_sentiment.timestamp,
                sample_size=reddit_sentiment.total_posts + reddit_sentiment.total_comments,
                trend=self._infer_trend_from_reddit(reddit_sentiment)
            )
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return self._empty_sentiment_score('reddit')
    
    async def _get_twitter_sentiment(self, hours_back: int) -> SentimentScore:
        """Get sentiment from Twitter sources."""
        try:
            twitter_sentiment = await self.twitter_service.get_crypto_sentiment(hours_back)
            
            return SentimentScore(
                source='twitter',
                sentiment=twitter_sentiment.overall_sentiment,
                confidence=twitter_sentiment.confidence,
                weight=self.source_weights['twitter'],
                timestamp=twitter_sentiment.timestamp,
                sample_size=twitter_sentiment.tweet_count,
                trend=twitter_sentiment.sentiment_trend
            )
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return self._empty_sentiment_score('twitter')
    
    async def _get_macro_sentiment(self) -> SentimentScore:
        """Get sentiment from macro economic data."""
        try:
            macro_data = await self.fred_service.get_macro_data()
            
            # Convert macro score to sentiment trend
            if macro_data.macro_score > 0.1:
                trend = "improving"
            elif macro_data.macro_score < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            # Estimate sample size based on number of indicators
            sample_size = len(macro_data.monetary_policy) + len(macro_data.growth_indicators) + \
                         len(macro_data.inflation_metrics) + len(macro_data.employment_data) + \
                         len(macro_data.market_indicators) + len(macro_data.sentiment_indices) + \
                         len(macro_data.risk_indicators)
            
            return SentimentScore(
                source='macro',
                sentiment=macro_data.macro_score,
                confidence=0.8,  # Macro data is generally reliable
                weight=self.source_weights['macro'],
                timestamp=macro_data.timestamp,
                sample_size=sample_size,
                trend=trend
            )
        except Exception as e:
            logger.error(f"Error getting macro sentiment: {e}")
            return self._empty_sentiment_score('macro')
    
    def _calculate_weighted_sentiment(self, sentiment_scores: Dict[str, SentimentScore]) -> float:
        """Calculate weighted average sentiment from all sources."""
        if not sentiment_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score in sentiment_scores.values():
            # Adjust weight by confidence
            adjusted_weight = score.weight * score.confidence
            weighted_sum += score.sentiment * adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_overall_confidence(self, sentiment_scores: Dict[str, SentimentScore]) -> float:
        """Calculate overall confidence in sentiment analysis."""
        if not sentiment_scores:
            return 0.0
        
        # Weighted average of individual confidences
        confidence_sum = 0.0
        weight_sum = 0.0
        
        for score in sentiment_scores.values():
            confidence_sum += score.confidence * score.weight
            weight_sum += score.weight
        
        base_confidence = confidence_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Boost confidence when sources agree
        agreement = self._calculate_source_agreement(sentiment_scores)
        adjusted_confidence = base_confidence * (0.5 + 0.5 * agreement)
        
        # Factor in sample size
        total_samples = sum(score.sample_size for score in sentiment_scores.values())
        sample_factor = min(total_samples / 100, 1.0)  # Full confidence at 100+ samples
        
        return min(adjusted_confidence * (0.5 + 0.5 * sample_factor), 1.0)
    
    def _analyze_sentiment_trend(self, current_sentiment: float) -> tuple[str, float, int]:
        """Analyze sentiment trend based on historical data."""
        self.sentiment_history.append({
            'sentiment': current_sentiment,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.sentiment_history = [
            entry for entry in self.sentiment_history 
            if entry['timestamp'] > cutoff_time
        ][-self.max_history:]
        
        if len(self.sentiment_history) < 3:
            return "stable", 0.0, 0
        
        # Calculate trend over different time windows
        recent_points = self.sentiment_history[-6:]  # Last 6 points (3 hours if 30min intervals)
        older_points = self.sentiment_history[-12:-6] if len(self.sentiment_history) >= 12 else []
        
        recent_avg = sum(p['sentiment'] for p in recent_points) / len(recent_points)
        
        if older_points:
            older_avg = sum(p['sentiment'] for p in older_points) / len(older_points)
            trend_change = recent_avg - older_avg
            
            if trend_change > 0.1:
                trend = "improving"
                strength = min(abs(trend_change) / 0.5, 1.0)
            elif trend_change < -0.1:
                trend = "declining"
                strength = min(abs(trend_change) / 0.5, 1.0)
            else:
                trend = "stable"
                strength = 0.0
        else:
            # Not enough history for comparison
            trend = "stable"
            strength = 0.0
        
        # Calculate trend duration (simplified)
        duration = len(recent_points) * 30  # Assuming 30-minute intervals
        
        return trend, strength, duration
    
    def _determine_market_mood(self, sentiment_scores: Dict[str, SentimentScore], overall_sentiment: float) -> MarketMood:
        """Determine overall market mood and emotional state."""
        # Primary mood based on overall sentiment
        if overall_sentiment > 0.4:
            primary_mood = "euphoric"
            dominant_emotion = "greed"
        elif overall_sentiment > 0.1:
            primary_mood = "bullish"
            dominant_emotion = "optimism"
        elif overall_sentiment < -0.4:
            primary_mood = "fearful"
            dominant_emotion = "fear"
        elif overall_sentiment < -0.1:
            primary_mood = "bearish"
            dominant_emotion = "pessimism"
        else:
            primary_mood = "neutral"
            dominant_emotion = "uncertainty"
        
        # Secondary mood based on specific indicators
        secondary_mood = "cautious"
        
        # Check for extreme sentiment in individual sources
        for source, score in sentiment_scores.items():
            if source == 'macro' and score.sentiment < -0.3:
                secondary_mood = "defensive"
            elif source in ['reddit', 'twitter'] and score.sentiment > 0.5:
                secondary_mood = "exuberant"
            elif score.confidence < 0.3:
                secondary_mood = "uncertain"
        
        # Mood intensity based on absolute sentiment and confidence
        mood_intensity = min(abs(overall_sentiment) * 2, 1.0)
        
        # Mood confidence based on source agreement
        source_agreement = self._calculate_source_agreement(sentiment_scores)
        mood_confidence = source_agreement
        
        # Market phase based on sentiment and trend
        trend_info = self._get_latest_trend()
        if trend_info['trend'] == "improving" and overall_sentiment > 0:
            market_phase = "markup"
        elif trend_info['trend'] == "declining" and overall_sentiment < 0:
            market_phase = "markdown"
        elif overall_sentiment > 0.2:
            market_phase = "distribution"
        else:
            market_phase = "accumulation"
        
        return MarketMood(
            primary_mood=primary_mood,
            secondary_mood=secondary_mood,
            mood_intensity=mood_intensity,
            mood_confidence=mood_confidence,
            dominant_emotion=dominant_emotion,
            market_phase=market_phase
        )
    
    def _generate_signals(self, sentiment_scores: Dict[str, SentimentScore]) -> tuple[List[str], List[str], List[str]]:
        """Generate bullish, bearish, and neutral signals from sentiment data."""
        bullish_signals = []
        bearish_signals = []
        neutral_signals = []
        
        for source, score in sentiment_scores.items():
            if score.sentiment > 0.3 and score.confidence > 0.6:
                bullish_signals.append(f"{source.title()} shows strong positive sentiment ({score.sentiment:.2f})")
            elif score.sentiment < -0.3 and score.confidence > 0.6:
                bearish_signals.append(f"{source.title()} shows strong negative sentiment ({score.sentiment:.2f})")
            elif abs(score.sentiment) < 0.1:
                neutral_signals.append(f"{source.title()} sentiment is neutral ({score.sentiment:.2f})")
            
            if score.trend == "improving":
                bullish_signals.append(f"{source.title()} sentiment trend improving")
            elif score.trend == "declining":
                bearish_signals.append(f"{source.title()} sentiment trend declining")
        
        # Add cross-source signals
        source_agreement = self._calculate_source_agreement(sentiment_scores)
        if source_agreement > 0.8:
            if any(score.sentiment > 0.2 for score in sentiment_scores.values()):
                bullish_signals.append("High agreement across sources on positive sentiment")
            elif any(score.sentiment < -0.2 for score in sentiment_scores.values()):
                bearish_signals.append("High agreement across sources on negative sentiment")
        elif source_agreement < 0.3:
            neutral_signals.append("Low agreement between sentiment sources - mixed signals")
        
        return bullish_signals, bearish_signals, neutral_signals
    
    def _calculate_fear_greed_index(self, sentiment_scores: Dict[str, SentimentScore], overall_sentiment: float) -> float:
        """Calculate fear & greed index (0-100 scale)."""
        # Base score from overall sentiment
        base_score = (overall_sentiment + 1) * 50  # Convert from [-1,1] to [0,100]
        
        # Adjustments based on specific sources
        adjustments = 0
        
        # Social media extreme sentiment
        for source in ['reddit', 'twitter']:
            if source in sentiment_scores:
                score = sentiment_scores[source]
                if score.sentiment > 0.5:  # Extreme positive (greed)
                    adjustments += 10
                elif score.sentiment < -0.5:  # Extreme negative (fear)
                    adjustments -= 10
        
        # Macro environment
        if 'macro' in sentiment_scores:
            macro_score = sentiment_scores['macro']
            if macro_score.sentiment > 0.3:  # Strong macro environment
                adjustments += 5
            elif macro_score.sentiment < -0.3:  # Weak macro environment
                adjustments -= 5
        
        # News sentiment
        if 'news' in sentiment_scores:
            news_score = sentiment_scores['news']
            if news_score.sentiment > 0.4:  # Very positive news
                adjustments += 5
            elif news_score.sentiment < -0.4:  # Very negative news
                adjustments -= 5
        
        final_score = base_score + adjustments
        return max(0, min(100, final_score))
    
    def _assess_market_stress(self, sentiment_scores: Dict[str, SentimentScore]) -> tuple[str, float]:
        """Assess market volatility expectation and stress level."""
        stress_factors = []
        
        # Check for negative sentiment with high confidence
        for score in sentiment_scores.values():
            if score.sentiment < -0.3 and score.confidence > 0.7:
                stress_factors.append(0.3)
            elif score.sentiment < -0.5:
                stress_factors.append(0.5)
        
        # Check for divergent sentiment between sources
        source_agreement = self._calculate_source_agreement(sentiment_scores)
        if source_agreement < 0.4:
            stress_factors.append(0.2)
        
        # Calculate overall stress
        market_stress = min(sum(stress_factors), 1.0)
        
        # Determine volatility expectation
        if market_stress > 0.6:
            volatility_expectation = "high"
        elif market_stress > 0.3:
            volatility_expectation = "medium"
        else:
            volatility_expectation = "low"
        
        return volatility_expectation, market_stress
    
    def _analyze_source_agreement(self, sentiment_scores: Dict[str, SentimentScore]) -> tuple[float, List[str]]:
        """Analyze agreement between sentiment sources."""
        if len(sentiment_scores) < 2:
            return 1.0, []
        
        sentiments = [score.sentiment for score in sentiment_scores.values()]
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(sentiments)):
            for j in range(i + 1, len(sentiments)):
                # Agreement based on how close sentiments are
                diff = abs(sentiments[i] - sentiments[j])
                agreement = max(0, 1 - diff / 2)  # Max difference is 2 (from -1 to 1)
                agreements.append(agreement)
        
        overall_agreement = sum(agreements) / len(agreements) if agreements else 0
        
        # Find divergences
        divergence_signals = []
        source_names = list(sentiment_scores.keys())
        
        for i, source1 in enumerate(source_names):
            for j, source2 in enumerate(source_names[i + 1:], i + 1):
                score1 = sentiment_scores[source1]
                score2 = sentiment_scores[source2]
                
                diff = abs(score1.sentiment - score2.sentiment)
                if diff > 0.5 and score1.confidence > 0.5 and score2.confidence > 0.5:
                    divergence_signals.append(
                        f"{source1.title()} ({score1.sentiment:.2f}) vs {source2.title()} ({score2.sentiment:.2f})"
                    )
        
        return overall_agreement, divergence_signals
    
    def _generate_trading_implications(self, overall_sentiment: float, confidence: float, 
                                     market_mood: MarketMood, source_agreement: float) -> tuple[str, str, str]:
        """Generate trading bias, position sizing, and risk level recommendations."""
        
        # Trading bias
        if overall_sentiment > 0.2 and confidence > 0.6:
            trading_bias = "long"
        elif overall_sentiment < -0.2 and confidence > 0.6:
            trading_bias = "short"
        elif abs(overall_sentiment) < 0.1:
            trading_bias = "neutral"
        else:
            trading_bias = "caution"
        
        # Position sizing based on confidence and agreement
        if confidence > 0.8 and source_agreement > 0.7 and abs(overall_sentiment) > 0.3:
            position_sizing = "aggressive"
        elif confidence > 0.6 and source_agreement > 0.5 and abs(overall_sentiment) > 0.2:
            position_sizing = "moderate"
        elif confidence > 0.4 and abs(overall_sentiment) > 0.1:
            position_sizing = "conservative"
        else:
            position_sizing = "minimal"
        
        # Risk level
        if market_mood.primary_mood in ["euphoric", "fearful"] or source_agreement < 0.4:
            risk_level = "high"
        elif confidence < 0.5 or abs(overall_sentiment) > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return trading_bias, position_sizing, risk_level
    
    def _combine_social_sentiment(self, sentiment_scores: Dict[str, SentimentScore]) -> SentimentScore:
        """Combine Reddit and Twitter sentiment into social sentiment."""
        reddit_score = sentiment_scores.get('reddit')
        twitter_score = sentiment_scores.get('twitter')
        
        if not reddit_score and not twitter_score:
            return self._empty_sentiment_score('social')
        elif not reddit_score:
            return twitter_score
        elif not twitter_score:
            return reddit_score
        
        # Weighted average of Reddit and Twitter
        reddit_weight = 0.4
        twitter_weight = 0.6
        
        combined_sentiment = (reddit_score.sentiment * reddit_weight + 
                            twitter_score.sentiment * twitter_weight)
        combined_confidence = (reddit_score.confidence * reddit_weight + 
                             twitter_score.confidence * twitter_weight)
        combined_sample_size = reddit_score.sample_size + twitter_score.sample_size
        
        # Combine trends
        if reddit_score.trend == twitter_score.trend:
            combined_trend = reddit_score.trend
        else:
            combined_trend = "mixed"
        
        return SentimentScore(
            source='social',
            sentiment=combined_sentiment,
            confidence=combined_confidence,
            weight=self.source_weights['reddit'] + self.source_weights['twitter'],
            timestamp=max(reddit_score.timestamp, twitter_score.timestamp),
            sample_size=combined_sample_size,
            trend=combined_trend
        )
    
    def _calculate_source_agreement(self, sentiment_scores: Dict[str, SentimentScore]) -> float:
        """Calculate agreement between sentiment sources."""
        agreement, _ = self._analyze_source_agreement(sentiment_scores)
        return agreement
    
    def _infer_trend_from_reddit(self, reddit_sentiment: Any) -> str:  # CryptoSentiment temporarily disabled
        """Infer trend from Reddit sentiment data."""
        # Use subreddit sentiment trends if available
        trending_up = 0
        trending_down = 0
        
        for subreddit_sentiment in reddit_sentiment.subreddit_sentiments.values():
            if subreddit_sentiment.sentiment_trend == "improving":
                trending_up += 1
            elif subreddit_sentiment.sentiment_trend == "declining":
                trending_down += 1
        
        if trending_up > trending_down:
            return "improving"
        elif trending_down > trending_up:
            return "declining"
        else:
            return "stable"
    
    def _get_latest_trend(self) -> Dict[str, Any]:
        """Get latest trend information."""
        if len(self.sentiment_history) < 2:
            return {'trend': 'stable', 'strength': 0.0}
        
        recent = self.sentiment_history[-3:]
        if len(recent) < 3:
            return {'trend': 'stable', 'strength': 0.0}
        
        trend_change = recent[-1]['sentiment'] - recent[0]['sentiment']
        
        if trend_change > 0.1:
            return {'trend': 'improving', 'strength': min(trend_change / 0.5, 1.0)}
        elif trend_change < -0.1:
            return {'trend': 'declining', 'strength': min(abs(trend_change) / 0.5, 1.0)}
        else:
            return {'trend': 'stable', 'strength': 0.0}
    
    def _update_sentiment_history(self, sentiment: float):
        """Update sentiment history for trend analysis."""
        # This is already handled in _analyze_sentiment_trend
        pass
    
    def _empty_sentiment_score(self, source: str) -> SentimentScore:
        """Return empty sentiment score for a source."""
        return SentimentScore(
            source=source,
            sentiment=0.0,
            confidence=0.0,
            weight=self.source_weights.get(source, 0.0),
            timestamp=datetime.now(),
            sample_size=0,
            trend="stable"
        )
    
    def _empty_unified_sentiment(self) -> UnifiedSentiment:
        """Return empty unified sentiment object."""
        empty_mood = MarketMood(
            primary_mood="neutral",
            secondary_mood="uncertain",
            mood_intensity=0.0,
            mood_confidence=0.0,
            dominant_emotion="uncertainty",
            market_phase="accumulation"
        )
        
        return UnifiedSentiment(
            overall_sentiment=0.0,
            sentiment_label="neutral",
            confidence=0.0,
            news_sentiment=self._empty_sentiment_score('news'),
            social_sentiment=self._empty_sentiment_score('social'),
            reddit_sentiment=self._empty_sentiment_score('reddit'),
            twitter_sentiment=self._empty_sentiment_score('twitter'),
            macro_sentiment=self._empty_sentiment_score('macro'),
            sentiment_trend="stable",
            trend_strength=0.0,
            trend_duration=0,
            market_mood=empty_mood,
            bullish_signals=[],
            bearish_signals=[],
            neutral_signals=["No sentiment data available"],
            fear_greed_index=50.0,
            volatility_expectation="low",
            market_stress=0.0,
            source_agreement=0.0,
            divergence_signals=[],
            trading_bias="neutral",
            position_sizing="minimal",
            risk_level="high",
            timestamp=datetime.now()
        )


# Factory function
def create_sentiment_service(news_service: Optional[Any] = None,  # NewsService temporarily disabled
                           reddit_service: Optional[Any] = None,  # RedditService temporarily disabled
                           twitter_service: Optional[Any] = None,  # TwitterService temporarily disabled
                           fred_service: Optional[Any] = None) -> SentimentService:  # FREDService temporarily disabled
    """Create and return SentimentService instance."""
    return SentimentService(news_service, reddit_service, twitter_service, fred_service)