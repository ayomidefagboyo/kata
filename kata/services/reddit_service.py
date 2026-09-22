"""
Reddit API Service for Yuki Agent

Provides cryptocurrency sentiment analysis from Reddit including:
- r/CryptoCurrency, r/Bitcoin, r/ethereum and other crypto subreddits
- Real-time sentiment tracking from comments and posts
- Trending topics and community mood analysis
- Market sentiment correlation with Reddit activity
- Social sentiment scoring for trading decisions
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Reddit post data structure."""
    id: str
    title: str
    content: str
    author: str
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    relevance_score: float  # 0 to 1
    engagement_score: float  # Combined score/comments metric
    keywords: List[str]
    flair: str


@dataclass
class RedditComment:
    """Reddit comment data structure."""
    id: str
    body: str
    author: str
    score: int
    created_utc: datetime
    parent_id: str
    subreddit: str
    sentiment_score: float
    sentiment_label: str
    relevance_score: float


@dataclass
class SubredditSentiment:
    """Subreddit sentiment aggregation."""
    subreddit: str
    overall_sentiment: float  # -1 to 1
    sentiment_label: str
    post_count: int
    comment_count: int
    positive_posts: int
    negative_posts: int
    neutral_posts: int
    average_score: float
    trending_keywords: List[str]
    top_discussions: List[str]  # Top post titles
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    activity_level: str  # 'high', 'medium', 'low'
    confidence: float  # 0 to 1
    timestamp: datetime


@dataclass
class CryptoSentiment:
    """Overall crypto sentiment from Reddit."""
    overall_sentiment: float
    sentiment_label: str
    total_posts: int
    total_comments: int
    subreddit_sentiments: Dict[str, SubredditSentiment]
    trending_coins: List[str]
    market_mood: str  # 'bullish', 'bearish', 'neutral', 'fearful', 'greedy'
    social_volume: str  # 'high', 'medium', 'low'
    sentiment_distribution: Dict[str, float]  # positive/negative/neutral percentages
    confidence: float
    timestamp: datetime


class RedditService:
    """
    Comprehensive Reddit sentiment analysis service for cryptocurrency markets.
    
    Monitors major crypto subreddits and provides real-time sentiment analysis
    with market correlation scoring for trading decisions.
    """
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit service."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit: Optional[praw.Reddit] = None
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Cache settings
        self.cache_duration = {
            'posts': 300,  # 5 minutes for posts
            'sentiment': 600,  # 10 minutes for sentiment analysis
            'subreddit': 900  # 15 minutes for subreddit analysis
        }
        self.cache = {}
        
        # Crypto subreddits to monitor
        self.crypto_subreddits = [
            'CryptoCurrency',
            'Bitcoin', 
            'ethereum',
            'CryptoMarkets',
            'altcoin',
            'CryptoMoonShots',
            'SatoshiStreetBets',
            'defi',
            'NFT',
            'solana',
            'CardanoMarkets',
            'dogecoin',
            'Ripple'
        ]
        
        # Crypto-related keywords for relevance scoring
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'trading', 'hodl',
            'binance', 'coinbase', 'solana', 'cardano', 'polkadot', 'avalanche',
            'doge', 'shiba', 'matic', 'chainlink', 'uniswap', 'aave',
            'compound', 'maker', 'curve', 'sushi', 'pancake', 'bear market',
            'bull market', 'moon', 'lambo', 'diamond hands', 'paper hands',
            'buy the dip', 'ath', 'fud', 'fomo', 'whale', 'pump', 'dump',
            'rekt', 'to the moon', 'when lambo', 'gm', 'ngmi', 'wagmi'
        ]
        
        # Negative sentiment indicators
        self.negative_indicators = [
            'crash', 'dump', 'rekt', 'bear', 'bearish', 'down', 'falling',
            'loss', 'losing', 'sell', 'panic', 'fear', 'fud', 'dead', 'scam',
            'rugpull', 'worthless', 'terrible', 'awful', 'disaster'
        ]
        
        # Positive sentiment indicators  
        self.positive_indicators = [
            'moon', 'bull', 'bullish', 'up', 'rising', 'gain', 'profit',
            'buy', 'hodl', 'diamond hands', 'lambo', 'rocket', 'green',
            'pump', 'surge', 'rally', 'breakthrough', 'amazing', 'awesome'
        ]
        
        logger.info("RedditService initialized")
    
    def _initialize_reddit(self):
        """Initialize Reddit API client."""
        if not self.reddit:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                # Test connection
                _ = self.reddit.user.me()
                logger.info("Reddit API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.reddit = None
    
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
    
    async def get_crypto_sentiment(self, hours_back: int = 24, max_posts_per_sub: int = 50) -> CryptoSentiment:
        """
        Get overall cryptocurrency sentiment from Reddit.
        
        Args:
            hours_back: How many hours back to analyze
            max_posts_per_sub: Maximum posts to analyze per subreddit
            
        Returns:
            CryptoSentiment object with aggregated data
        """
        try:
            cache_key = f"crypto_sentiment_{hours_back}_{max_posts_per_sub}"
            
            if self._is_cache_valid(cache_key, 'sentiment'):
                return self.cache[cache_key]['data']
            
            self._initialize_reddit()
            if not self.reddit:
                return self._empty_crypto_sentiment()
            
            # Analyze each subreddit
            subreddit_sentiments = {}
            total_posts = 0
            total_comments = 0
            all_sentiments = []
            all_keywords = []
            trending_coins = []
            
            for subreddit_name in self.crypto_subreddits:
                try:
                    subreddit_sentiment = await self.get_subreddit_sentiment(
                        subreddit_name, hours_back, max_posts_per_sub
                    )
                    subreddit_sentiments[subreddit_name] = subreddit_sentiment
                    
                    total_posts += subreddit_sentiment.post_count
                    total_comments += subreddit_sentiment.comment_count
                    all_sentiments.append(subreddit_sentiment.overall_sentiment)
                    all_keywords.extend(subreddit_sentiment.trending_keywords)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing subreddit {subreddit_name}: {e}")
                    continue
            
            if not all_sentiments:
                return self._empty_crypto_sentiment()
            
            # Calculate overall sentiment
            overall_sentiment = sum(all_sentiments) / len(all_sentiments)
            
            # Determine sentiment label and market mood
            if overall_sentiment > 0.3:
                sentiment_label = "positive"
                market_mood = "bullish"
            elif overall_sentiment > 0.1:
                sentiment_label = "positive"
                market_mood = "neutral"
            elif overall_sentiment < -0.3:
                sentiment_label = "negative"
                market_mood = "bearish"
            elif overall_sentiment < -0.1:
                sentiment_label = "negative"
                market_mood = "fearful"
            else:
                sentiment_label = "neutral"
                market_mood = "neutral"
            
            # Extract trending coins from keywords
            coin_keywords = ['btc', 'eth', 'sol', 'ada', 'dot', 'avax', 'matic', 'link']
            trending_coins = [kw.upper() for kw in all_keywords if kw.lower() in coin_keywords]
            trending_coins = list(dict.fromkeys(trending_coins))[:10]  # Remove duplicates, top 10
            
            # Calculate social volume
            total_activity = total_posts + total_comments
            if total_activity > 1000:
                social_volume = "high"
            elif total_activity > 300:
                social_volume = "medium"
            else:
                social_volume = "low"
            
            # Calculate sentiment distribution
            positive_count = len([s for s in all_sentiments if s > 0.1])
            negative_count = len([s for s in all_sentiments if s < -0.1])
            neutral_count = len(all_sentiments) - positive_count - negative_count
            
            total_subs = len(all_sentiments)
            sentiment_distribution = {
                'positive': positive_count / total_subs if total_subs > 0 else 0,
                'negative': negative_count / total_subs if total_subs > 0 else 0,
                'neutral': neutral_count / total_subs if total_subs > 0 else 0
            }
            
            # Calculate confidence
            confidence = min(total_activity / 500, 1.0)  # More activity = higher confidence
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in all_sentiments) / len(all_sentiments)
            confidence *= max(0.1, 1.0 - sentiment_variance)  # Lower variance = higher confidence
            
            crypto_sentiment = CryptoSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                total_posts=total_posts,
                total_comments=total_comments,
                subreddit_sentiments=subreddit_sentiments,
                trending_coins=trending_coins,
                market_mood=market_mood,
                social_volume=social_volume,
                sentiment_distribution=sentiment_distribution,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, crypto_sentiment)
            return crypto_sentiment
            
        except Exception as e:
            logger.error(f"Error getting crypto sentiment: {e}")
            return self._empty_crypto_sentiment()
    
    async def get_subreddit_sentiment(self, subreddit_name: str, hours_back: int = 24, max_posts: int = 50) -> SubredditSentiment:
        """
        Analyze sentiment for a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            hours_back: How many hours back to analyze
            max_posts: Maximum posts to analyze
            
        Returns:
            SubredditSentiment object
        """
        try:
            cache_key = f"subreddit_sentiment_{subreddit_name}_{hours_back}_{max_posts}"
            
            if self._is_cache_valid(cache_key, 'subreddit'):
                return self.cache[cache_key]['data']
            
            self._initialize_reddit()
            if not self.reddit:
                return self._empty_subreddit_sentiment(subreddit_name)
            
            # Get recent posts
            posts = await self.get_subreddit_posts(subreddit_name, hours_back, max_posts)
            
            if not posts:
                return self._empty_subreddit_sentiment(subreddit_name)
            
            # Analyze sentiment
            sentiments = [post.sentiment_score for post in posts]
            overall_sentiment = sum(sentiments) / len(sentiments)
            
            # Count sentiment categories
            positive_posts = len([s for s in sentiments if s > 0.1])
            negative_posts = len([s for s in sentiments if s < -0.1])
            neutral_posts = len(posts) - positive_posts - negative_posts
            
            # Calculate average score (upvotes)
            average_score = sum(post.score for post in posts) / len(posts)
            
            # Extract keywords and topics
            all_keywords = [kw for post in posts for kw in post.keywords]
            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            trending_keywords = sorted(keyword_counts.keys(), key=lambda x: keyword_counts[x], reverse=True)[:10]
            
            # Get top discussions (high-scoring posts)
            top_posts = sorted(posts, key=lambda x: x.score, reverse=True)[:5]
            top_discussions = [post.title for post in top_posts]
            
            # Determine sentiment trend (simplified)
            recent_posts = [p for p in posts if (datetime.now() - p.created_utc).total_seconds() < 3600 * 6]  # Last 6 hours
            older_posts = [p for p in posts if p not in recent_posts]
            
            if recent_posts and older_posts:
                recent_sentiment = sum(p.sentiment_score for p in recent_posts) / len(recent_posts)
                older_sentiment = sum(p.sentiment_score for p in older_posts) / len(older_posts)
                
                if recent_sentiment > older_sentiment + 0.1:
                    sentiment_trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    sentiment_trend = "declining"
                else:
                    sentiment_trend = "stable"
            else:
                sentiment_trend = "stable"
            
            # Determine activity level
            total_comments = sum(post.num_comments for post in posts)
            if len(posts) > 30 and total_comments > 500:
                activity_level = "high"
            elif len(posts) > 15 and total_comments > 150:
                activity_level = "medium"
            else:
                activity_level = "low"
            
            # Determine sentiment label
            if overall_sentiment > 0.2:
                sentiment_label = "positive"
            elif overall_sentiment < -0.2:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Calculate confidence
            confidence = min(len(posts) / 30, 1.0)  # More posts = higher confidence
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
            confidence *= max(0.1, 1.0 - sentiment_variance)
            
            subreddit_sentiment = SubredditSentiment(
                subreddit=subreddit_name,
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                post_count=len(posts),
                comment_count=total_comments,
                positive_posts=positive_posts,
                negative_posts=negative_posts,
                neutral_posts=neutral_posts,
                average_score=average_score,
                trending_keywords=trending_keywords,
                top_discussions=top_discussions,
                sentiment_trend=sentiment_trend,
                activity_level=activity_level,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, subreddit_sentiment)
            return subreddit_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing subreddit {subreddit_name}: {e}")
            return self._empty_subreddit_sentiment(subreddit_name)
    
    async def get_subreddit_posts(self, subreddit_name: str, hours_back: int = 24, limit: int = 50) -> List[RedditPost]:
        """
        Get recent posts from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            hours_back: How many hours back to get posts
            limit: Maximum number of posts
            
        Returns:
            List of RedditPost objects
        """
        try:
            cache_key = f"subreddit_posts_{subreddit_name}_{hours_back}_{limit}"
            
            if self._is_cache_valid(cache_key, 'posts'):
                return self.cache[cache_key]['data']
            
            self._initialize_reddit()
            if not self.reddit:
                return []
            
            subreddit = self.reddit.subreddit(subreddit_name)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            posts = []
            
            # Get hot posts (most engagement)
            for submission in subreddit.hot(limit=limit):
                post_time = datetime.fromtimestamp(submission.created_utc)
                
                if post_time < cutoff_time:
                    continue
                
                try:
                    post = RedditPost(
                        id=submission.id,
                        title=submission.title,
                        content=submission.selftext or "",
                        author=str(submission.author) if submission.author else "[deleted]",
                        subreddit=subreddit_name,
                        score=submission.score,
                        upvote_ratio=submission.upvote_ratio,
                        num_comments=submission.num_comments,
                        created_utc=post_time,
                        url=f"https://reddit.com{submission.permalink}",
                        sentiment_score=0.0,  # Will be calculated
                        sentiment_label="neutral",
                        relevance_score=0.0,  # Will be calculated
                        engagement_score=0.0,  # Will be calculated
                        keywords=[],  # Will be extracted
                        flair=submission.link_flair_text or ""
                    )
                    
                    # Analyze post
                    self._analyze_post(post)
                    posts.append(post)
                    
                except Exception as e:
                    logger.warning(f"Error processing post {submission.id}: {e}")
                    continue
            
            self._cache_data(cache_key, posts)
            return posts
            
        except Exception as e:
            logger.error(f"Error getting posts from {subreddit_name}: {e}")
            return []
    
    def _analyze_post(self, post: RedditPost):
        """Perform sentiment and relevance analysis on post."""
        try:
            # Combine text for analysis
            full_text = f"{post.title} {post.content}"
            
            # Sentiment analysis using VADER
            vader_scores = self.vader_analyzer.polarity_scores(full_text)
            compound_score = vader_scores['compound']
            
            # Sentiment analysis using TextBlob as backup
            try:
                blob = TextBlob(full_text)
                textblob_score = blob.sentiment.polarity
                
                # Average the two scores
                post.sentiment_score = (compound_score + textblob_score) / 2
            except:
                post.sentiment_score = compound_score
            
            # Apply Reddit-specific sentiment adjustments
            post.sentiment_score = self._adjust_reddit_sentiment(post.sentiment_score, full_text)
            
            # Determine sentiment label
            if post.sentiment_score > 0.1:
                post.sentiment_label = "positive"
            elif post.sentiment_score < -0.1:
                post.sentiment_label = "negative"
            else:
                post.sentiment_label = "neutral"
            
            # Calculate relevance score
            post.relevance_score = self._calculate_relevance(full_text)
            
            # Calculate engagement score
            post.engagement_score = self._calculate_engagement(post)
            
            # Extract keywords
            post.keywords = self._extract_keywords(full_text)
            
        except Exception as e:
            logger.error(f"Error analyzing post: {e}")
    
    def _adjust_reddit_sentiment(self, sentiment_score: float, text: str) -> float:
        """Adjust sentiment based on Reddit-specific language patterns."""
        text_lower = text.lower()
        
        # Boost positive indicators
        positive_boost = sum(1 for indicator in self.positive_indicators if indicator in text_lower)
        sentiment_score += positive_boost * 0.1
        
        # Reduce for negative indicators
        negative_penalty = sum(1 for indicator in self.negative_indicators if indicator in text_lower)
        sentiment_score -= negative_penalty * 0.1
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score to cryptocurrency."""
        text_lower = text.lower()
        
        # Count crypto keywords
        crypto_matches = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)
        
        # Calculate score
        relevance = min(crypto_matches / 5, 1.0)  # Normalize to 0-1
        return relevance
    
    def _calculate_engagement(self, post: RedditPost) -> float:
        """Calculate engagement score based on votes and comments."""
        # Normalize score (upvotes can vary widely)
        score_component = min(post.score / 100, 1.0) if post.score > 0 else 0
        
        # Normalize comments
        comment_component = min(post.num_comments / 50, 1.0)
        
        # Consider upvote ratio
        ratio_component = post.upvote_ratio
        
        # Weighted combination
        engagement = (score_component * 0.4 + comment_component * 0.4 + ratio_component * 0.2)
        
        return min(engagement, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        
        # Find crypto keywords
        found_keywords = []
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract crypto tickers (3+ chars, all caps)
        tickers = re.findall(r'\b[A-Z]{3,6}\b', text)
        found_keywords.extend([ticker.lower() for ticker in tickers[:5]])
        
        return list(set(found_keywords))[:10]  # Unique keywords, max 10
    
    def _empty_crypto_sentiment(self) -> CryptoSentiment:
        """Return empty crypto sentiment data."""
        return CryptoSentiment(
            overall_sentiment=0.0,
            sentiment_label="neutral",
            total_posts=0,
            total_comments=0,
            subreddit_sentiments={},
            trending_coins=[],
            market_mood="neutral",
            social_volume="low",
            sentiment_distribution={'positive': 0, 'negative': 0, 'neutral': 1},
            confidence=0.0,
            timestamp=datetime.now()
        )
    
    def _empty_subreddit_sentiment(self, subreddit_name: str) -> SubredditSentiment:
        """Return empty subreddit sentiment data."""
        return SubredditSentiment(
            subreddit=subreddit_name,
            overall_sentiment=0.0,
            sentiment_label="neutral",
            post_count=0,
            comment_count=0,
            positive_posts=0,
            negative_posts=0,
            neutral_posts=0,
            average_score=0.0,
            trending_keywords=[],
            top_discussions=[],
            sentiment_trend="stable",
            activity_level="low",
            confidence=0.0,
            timestamp=datetime.now()
        )


# Factory function
def create_reddit_service(client_id: str, client_secret: str, user_agent: str) -> RedditService:
    """Create and return RedditService instance."""
    return RedditService(client_id, client_secret, user_agent)