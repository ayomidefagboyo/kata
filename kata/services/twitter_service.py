"""
Twitter/X API Service for Yuki Agent

Provides cryptocurrency sentiment analysis from Twitter/X including:
- Real-time crypto-related tweets and social sentiment
- Influencer sentiment tracking (crypto Twitter personalities)
- Trending hashtags and viral content analysis
- Social volume and engagement metrics
- Market sentiment correlation with Twitter activity
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import tweepy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Twitter tweet data structure."""
    id: str
    text: str
    author_username: str
    author_name: str
    author_followers: int
    author_verified: bool
    created_at: datetime
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    influence_score: float  # 0 to 1 (author influence)
    engagement_score: float  # Combined engagement metric
    relevance_score: float  # 0 to 1 (relevance to crypto)
    keywords: List[str]
    is_retweet: bool
    lang: str


@dataclass
class TwitterInfluencer:
    """Twitter influencer data structure."""
    username: str
    name: str
    followers_count: int
    verified: bool
    description: str
    recent_sentiment: float
    tweet_count: int
    avg_engagement: float
    influence_score: float
    category: str  # 'trader', 'analyst', 'developer', 'media', 'other'


@dataclass
class TwitterSentiment:
    """Twitter sentiment aggregation structure."""
    overall_sentiment: float  # -1 to 1
    sentiment_label: str
    tweet_count: int
    positive_tweets: int
    negative_tweets: int
    neutral_tweets: int
    trending_hashtags: List[str]
    trending_keywords: List[str]
    top_influencers: List[TwitterInfluencer]
    social_volume: str  # 'high', 'medium', 'low'
    engagement_rate: float
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    market_mood: str  # 'bullish', 'bearish', 'neutral', 'fearful', 'euphoric'
    viral_content: List[Tweet]  # High engagement tweets
    confidence: float  # 0 to 1
    timestamp: datetime


class TwitterService:
    """
    Comprehensive Twitter sentiment analysis service for cryptocurrency markets.
    
    Monitors crypto Twitter for real-time sentiment, influencer opinions,
    and social trends that may impact market movements.
    """
    
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str, bearer_token: str):
        """Initialize Twitter service."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.bearer_token = bearer_token
        
        # Initialize Twitter API clients
        self.api_v1: Optional[tweepy.API] = None
        self.api_v2: Optional[tweepy.Client] = None
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Cache settings
        self.cache_duration = {
            'tweets': 300,  # 5 minutes for tweets
            'sentiment': 600,  # 10 minutes for sentiment analysis
            'influencers': 1800  # 30 minutes for influencer data
        }
        self.cache = {}
        
        # Crypto-related search terms
        self.crypto_keywords = [
            'bitcoin', 'btc', '$btc', 'ethereum', 'eth', '$eth', 'crypto',
            'cryptocurrency', 'blockchain', 'defi', 'nft', 'altcoin',
            'solana', 'sol', '$sol', 'cardano', 'ada', '$ada', 'avalanche',
            'avax', '$avax', 'polygon', 'matic', '$matic', 'chainlink',
            'link', '$link', 'dogecoin', 'doge', '$doge', 'shiba'
        ]
        
        # High-influence crypto Twitter accounts (usernames without @)
        self.crypto_influencers = [
            'elonmusk', 'michael_saylor', 'VitalikButerin', 'cz_binance',
            'APompliano', 'DocumentingBTC', 'RaoulGMI', 'PlanB_BTC',
            'naval', 'balajis', 'laurashin', 'NeerajKA', 'Galaxy_Trading',
            'BitcoinMagazine', 'CoinDesk', 'Cointelegraph', 'zhusu',
            'novogratz', 'TimDraper', 'aantonop', 'erikvoorhees',
            'peterktodd', 'adam3us', 'jespow', 'brian_armstrong'
        ]
        
        # Sentiment boosters/reducers
        self.positive_indicators = [
            'moon', 'bull', 'bullish', 'pump', 'surge', 'rally', 'breakout',
            'ath', 'all time high', 'rocket', 'lambo', 'diamond hands',
            'hodl', 'buy the dip', 'gm', 'wagmi', 'lfg', 'based',
            'incredible', 'amazing', 'huge', 'massive', 'green', 'up'
        ]
        
        self.negative_indicators = [
            'bear', 'bearish', 'dump', 'crash', 'rekt', 'liquidated',
            'panic', 'fear', 'fud', 'down', 'red', 'selling', 'exit',
            'scam', 'rugpull', 'dead', 'worthless', 'disaster', 'ngmi'
        ]
        
        logger.info("TwitterService initialized")
    
    def _initialize_twitter_apis(self):
        """Initialize Twitter API clients."""
        try:
            # Initialize API v1.1 for additional features
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret,
                self.access_token, self.access_token_secret
            )
            self.api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
            
            # Initialize API v2 for modern features
            self.api_v2 = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True
            )
            
            # Test connection
            _ = self.api_v2.get_me()
            logger.info("Twitter API clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter clients: {e}")
            self.api_v1 = None
            self.api_v2 = None
    
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
    
    async def get_crypto_sentiment(self, hours_back: int = 24, max_tweets: int = 1000) -> TwitterSentiment:
        """
        Get overall cryptocurrency sentiment from Twitter.
        
        Args:
            hours_back: How many hours back to analyze
            max_tweets: Maximum tweets to analyze
            
        Returns:
            TwitterSentiment object with aggregated data
        """
        try:
            cache_key = f"twitter_sentiment_{hours_back}_{max_tweets}"
            
            if self._is_cache_valid(cache_key, 'sentiment'):
                return self.cache[cache_key]['data']
            
            self._initialize_twitter_apis()
            if not self.api_v2:
                return self._empty_twitter_sentiment()
            
            # Get crypto-related tweets
            tweets = await self.get_crypto_tweets(hours_back, max_tweets)
            
            if not tweets:
                return self._empty_twitter_sentiment()
            
            # Analyze sentiment
            sentiments = [tweet.sentiment_score for tweet in tweets]
            overall_sentiment = sum(sentiments) / len(sentiments)
            
            # Count sentiment categories
            positive_tweets = len([s for s in sentiments if s > 0.1])
            negative_tweets = len([s for s in sentiments if s < -0.1])
            neutral_tweets = len(tweets) - positive_tweets - negative_tweets
            
            # Determine sentiment label and market mood
            if overall_sentiment > 0.3:
                sentiment_label = "positive"
                market_mood = "bullish"
            elif overall_sentiment > 0.1:
                sentiment_label = "positive"
                market_mood = "euphoric"
            elif overall_sentiment < -0.3:
                sentiment_label = "negative"
                market_mood = "bearish"
            elif overall_sentiment < -0.1:
                sentiment_label = "negative"
                market_mood = "fearful"
            else:
                sentiment_label = "neutral"
                market_mood = "neutral"
            
            # Extract trending hashtags
            all_hashtags = [tag for tweet in tweets for tag in tweet.hashtags]
            hashtag_counts = {}
            for tag in all_hashtags:
                hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
            
            trending_hashtags = sorted(hashtag_counts.keys(), key=lambda x: hashtag_counts[x], reverse=True)[:10]
            
            # Extract trending keywords
            all_keywords = [kw for tweet in tweets for kw in tweet.keywords]
            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            trending_keywords = sorted(keyword_counts.keys(), key=lambda x: keyword_counts[x], reverse=True)[:15]
            
            # Get top influencers from tweets
            influencer_tweets = [t for t in tweets if t.author_username.lower() in [inf.lower() for inf in self.crypto_influencers]]
            top_influencers = await self._analyze_influencers(influencer_tweets)
            
            # Determine social volume
            tweet_rate = len(tweets) / hours_back if hours_back > 0 else 0
            if tweet_rate > 50:
                social_volume = "high"
            elif tweet_rate > 20:
                social_volume = "medium"
            else:
                social_volume = "low"
            
            # Calculate engagement rate
            total_engagement = sum(tweet.like_count + tweet.retweet_count for tweet in tweets)
            engagement_rate = total_engagement / len(tweets) if tweets else 0
            
            # Determine sentiment trend (simplified)
            recent_tweets = [t for t in tweets if (datetime.now() - t.created_at).total_seconds() < 3600 * 6]  # Last 6 hours
            older_tweets = [t for t in tweets if t not in recent_tweets]
            
            if recent_tweets and older_tweets:
                recent_sentiment = sum(t.sentiment_score for t in recent_tweets) / len(recent_tweets)
                older_sentiment = sum(t.sentiment_score for t in older_tweets) / len(older_tweets)
                
                if recent_sentiment > older_sentiment + 0.1:
                    sentiment_trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    sentiment_trend = "declining"
                else:
                    sentiment_trend = "stable"
            else:
                sentiment_trend = "stable"
            
            # Get viral content (high engagement tweets)
            viral_tweets = sorted(tweets, key=lambda x: x.engagement_score, reverse=True)[:10]
            
            # Calculate confidence
            confidence = min(len(tweets) / 100, 1.0)  # More tweets = higher confidence
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
            confidence *= max(0.1, 1.0 - sentiment_variance)  # Lower variance = higher confidence
            
            twitter_sentiment = TwitterSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                tweet_count=len(tweets),
                positive_tweets=positive_tweets,
                negative_tweets=negative_tweets,
                neutral_tweets=neutral_tweets,
                trending_hashtags=trending_hashtags,
                trending_keywords=trending_keywords,
                top_influencers=top_influencers,
                social_volume=social_volume,
                engagement_rate=engagement_rate,
                sentiment_trend=sentiment_trend,
                market_mood=market_mood,
                viral_content=viral_tweets,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, twitter_sentiment)
            return twitter_sentiment
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return self._empty_twitter_sentiment()
    
    async def get_crypto_tweets(self, hours_back: int = 24, max_tweets: int = 1000) -> List[Tweet]:
        """
        Get cryptocurrency-related tweets.
        
        Args:
            hours_back: How many hours back to search
            max_tweets: Maximum tweets to return
            
        Returns:
            List of Tweet objects
        """
        try:
            cache_key = f"crypto_tweets_{hours_back}_{max_tweets}"
            
            if self._is_cache_valid(cache_key, 'tweets'):
                return self.cache[cache_key]['data']
            
            self._initialize_twitter_apis()
            if not self.api_v2:
                return []
            
            # Build search query
            crypto_terms = ' OR '.join(self.crypto_keywords[:10])  # Use top 10 terms to avoid query length limits
            query = f"({crypto_terms}) -is:retweet lang:en"
            
            # Calculate start time
            start_time = datetime.now() - timedelta(hours=hours_back)
            
            tweets = []
            
            try:
                # Search for tweets
                response = tweepy.Paginator(
                    self.api_v2.search_recent_tweets,
                    query=query,
                    start_time=start_time,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics', 'author_id', 'context_annotations', 'entities', 'lang'],
                    user_fields=['username', 'name', 'public_metrics', 'verified'],
                    expansions=['author_id']
                ).flatten(limit=max_tweets)
                
                # Convert to our Tweet objects
                users_dict = {}
                if hasattr(response, 'includes') and response.includes and 'users' in response.includes:
                    users_dict = {user.id: user for user in response.includes['users']}
                
                for tweet_data in response:
                    try:
                        # Get author info
                        author = users_dict.get(tweet_data.author_id)
                        if not author:
                            continue
                        
                        # Extract entities
                        hashtags = []
                        mentions = []
                        urls = []
                        
                        if tweet_data.entities:
                            hashtags = [tag['tag'] for tag in tweet_data.entities.get('hashtags', [])]
                            mentions = [mention['username'] for mention in tweet_data.entities.get('mentions', [])]
                            urls = [url['expanded_url'] for url in tweet_data.entities.get('urls', [])]
                        
                        tweet = Tweet(
                            id=tweet_data.id,
                            text=tweet_data.text,
                            author_username=author.username,
                            author_name=author.name,
                            author_followers=author.public_metrics['followers_count'],
                            author_verified=author.verified or False,
                            created_at=tweet_data.created_at,
                            retweet_count=tweet_data.public_metrics['retweet_count'],
                            like_count=tweet_data.public_metrics['like_count'],
                            reply_count=tweet_data.public_metrics['reply_count'],
                            quote_count=tweet_data.public_metrics['quote_count'],
                            hashtags=hashtags,
                            mentions=mentions,
                            urls=urls,
                            sentiment_score=0.0,  # Will be calculated
                            sentiment_label="neutral",
                            influence_score=0.0,  # Will be calculated
                            engagement_score=0.0,  # Will be calculated
                            relevance_score=0.0,  # Will be calculated
                            keywords=[],  # Will be extracted
                            is_retweet=False,  # Filtered out in query
                            lang=tweet_data.lang or 'en'
                        )
                        
                        # Analyze tweet
                        self._analyze_tweet(tweet)
                        tweets.append(tweet)
                        
                    except Exception as e:
                        logger.warning(f"Error processing tweet: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error searching tweets: {e}")
            
            self._cache_data(cache_key, tweets)
            return tweets
            
        except Exception as e:
            logger.error(f"Error getting crypto tweets: {e}")
            return []
    
    def _analyze_tweet(self, tweet: Tweet):
        """Perform sentiment and relevance analysis on tweet."""
        try:
            text = tweet.text
            
            # Clean tweet text (remove URLs, mentions)
            clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            clean_text = re.sub(r'@\w+', '', clean_text)
            clean_text = re.sub(r'#(\w+)', r'\1', clean_text)  # Keep hashtag content
            
            # Sentiment analysis using VADER
            vader_scores = self.vader_analyzer.polarity_scores(clean_text)
            compound_score = vader_scores['compound']
            
            # Sentiment analysis using TextBlob as backup
            try:
                blob = TextBlob(clean_text)
                textblob_score = blob.sentiment.polarity
                
                # Average the two scores
                tweet.sentiment_score = (compound_score + textblob_score) / 2
            except:
                tweet.sentiment_score = compound_score
            
            # Apply Twitter-specific sentiment adjustments
            tweet.sentiment_score = self._adjust_twitter_sentiment(tweet.sentiment_score, clean_text)
            
            # Determine sentiment label
            if tweet.sentiment_score > 0.1:
                tweet.sentiment_label = "positive"
            elif tweet.sentiment_score < -0.1:
                tweet.sentiment_label = "negative"
            else:
                tweet.sentiment_label = "neutral"
            
            # Calculate relevance score
            tweet.relevance_score = self._calculate_relevance(clean_text)
            
            # Calculate influence score
            tweet.influence_score = self._calculate_influence(tweet)
            
            # Calculate engagement score
            tweet.engagement_score = self._calculate_engagement(tweet)
            
            # Extract keywords
            tweet.keywords = self._extract_keywords(clean_text)
            
        except Exception as e:
            logger.error(f"Error analyzing tweet: {e}")
    
    def _adjust_twitter_sentiment(self, sentiment_score: float, text: str) -> float:
        """Adjust sentiment based on Twitter-specific language patterns."""
        text_lower = text.lower()
        
        # Boost positive indicators
        positive_boost = sum(1 for indicator in self.positive_indicators if indicator in text_lower)
        sentiment_score += positive_boost * 0.1
        
        # Reduce for negative indicators
        negative_penalty = sum(1 for indicator in self.negative_indicators if indicator in text_lower)
        sentiment_score -= negative_penalty * 0.1
        
        # Emoji sentiment (simplified)
        if any(emoji in text for emoji in ['🚀', '🌙', '💎', '📈', '🔥', '💰', '🎉']):
            sentiment_score += 0.1
        if any(emoji in text for emoji in ['📉', '💸', '😰', '😭', '⚠️', '🔴', '💔']):
            sentiment_score -= 0.1
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment_score))
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score to cryptocurrency."""
        text_lower = text.lower()
        
        # Count crypto keywords
        crypto_matches = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)
        
        # Calculate score
        relevance = min(crypto_matches / 3, 1.0)  # Normalize to 0-1
        return relevance
    
    def _calculate_influence(self, tweet: Tweet) -> float:
        """Calculate influence score of tweet author."""
        # Normalize follower count (log scale)
        import math
        follower_score = min(math.log10(max(tweet.author_followers, 1)) / 7, 1.0)  # Max at 10M followers
        
        # Verification boost
        verification_score = 0.2 if tweet.author_verified else 0.0
        
        # Influencer boost
        influencer_score = 0.3 if tweet.author_username.lower() in [inf.lower() for inf in self.crypto_influencers] else 0.0
        
        return min(follower_score + verification_score + influencer_score, 1.0)
    
    def _calculate_engagement(self, tweet: Tweet) -> float:
        """Calculate engagement score based on interactions."""
        total_engagement = tweet.like_count + tweet.retweet_count + tweet.reply_count + tweet.quote_count
        
        # Normalize engagement (log scale)
        import math
        if total_engagement > 0:
            engagement_score = min(math.log10(total_engagement) / 4, 1.0)  # Max at 10k interactions
        else:
            engagement_score = 0.0
        
        return engagement_score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        
        # Find crypto keywords
        found_keywords = []
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract crypto tickers (with $ prefix or 3+ chars, all caps)
        tickers = re.findall(r'\$[A-Z]{2,6}\b|(?<!\w)[A-Z]{3,6}(?!\w)', text)
        found_keywords.extend([ticker.replace('$', '').lower() for ticker in tickers[:5]])
        
        return list(set(found_keywords))[:10]  # Unique keywords, max 10
    
    async def _analyze_influencers(self, tweets: List[Tweet]) -> List[TwitterInfluencer]:
        """Analyze influencer sentiment and engagement."""
        influencer_data = {}
        
        for tweet in tweets:
            username = tweet.author_username.lower()
            
            if username not in influencer_data:
                influencer_data[username] = {
                    'username': tweet.author_username,
                    'name': tweet.author_name,
                    'followers_count': tweet.author_followers,
                    'verified': tweet.author_verified,
                    'tweets': [],
                    'total_engagement': 0
                }
            
            influencer_data[username]['tweets'].append(tweet)
            influencer_data[username]['total_engagement'] += tweet.like_count + tweet.retweet_count
        
        influencers = []
        for data in influencer_data.values():
            if len(data['tweets']) > 0:
                recent_sentiment = sum(t.sentiment_score for t in data['tweets']) / len(data['tweets'])
                avg_engagement = data['total_engagement'] / len(data['tweets'])
                influence_score = data['tweets'][0].influence_score  # Should be same for all tweets
                
                # Determine category (simplified)
                category = "other"
                if any(word in data['name'].lower() for word in ['trader', 'trading']):
                    category = "trader"
                elif any(word in data['name'].lower() for word in ['analyst', 'research']):
                    category = "analyst"
                elif any(word in data['name'].lower() for word in ['dev', 'developer', 'founder']):
                    category = "developer"
                elif any(word in data['name'].lower() for word in ['news', 'media', 'magazine']):
                    category = "media"
                
                influencer = TwitterInfluencer(
                    username=data['username'],
                    name=data['name'],
                    followers_count=data['followers_count'],
                    verified=data['verified'],
                    description="",  # Could fetch from API if needed
                    recent_sentiment=recent_sentiment,
                    tweet_count=len(data['tweets']),
                    avg_engagement=avg_engagement,
                    influence_score=influence_score,
                    category=category
                )
                influencers.append(influencer)
        
        # Sort by influence score
        influencers.sort(key=lambda x: x.influence_score, reverse=True)
        return influencers[:10]  # Top 10 influencers
    
    def _empty_twitter_sentiment(self) -> TwitterSentiment:
        """Return empty Twitter sentiment data."""
        return TwitterSentiment(
            overall_sentiment=0.0,
            sentiment_label="neutral",
            tweet_count=0,
            positive_tweets=0,
            negative_tweets=0,
            neutral_tweets=0,
            trending_hashtags=[],
            trending_keywords=[],
            top_influencers=[],
            social_volume="low",
            engagement_rate=0.0,
            sentiment_trend="stable",
            market_mood="neutral",
            viral_content=[],
            confidence=0.0,
            timestamp=datetime.now()
        )


# Factory function
def create_twitter_service(api_key: str, api_secret: str, access_token: str, access_token_secret: str, bearer_token: str) -> TwitterService:
    """Create and return TwitterService instance."""
    return TwitterService(api_key, api_secret, access_token, access_token_secret, bearer_token)