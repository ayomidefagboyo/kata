"""
News API Service for Yuki Agent

Provides comprehensive financial news and market sentiment analysis including:
- Real-time financial news from multiple sources
- Crypto-specific news aggregation
- Sentiment analysis of news articles
- Market-moving events and announcements
- News impact scoring for trading decisions
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
from newsapi import NewsApiClient
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data structure."""
    title: str
    description: str
    content: str
    source: str
    author: str
    url: str
    published_at: datetime
    sentiment_score: float  # -1 to 1 (negative to positive)
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    relevance_score: float  # 0 to 1 (relevance to crypto/trading)
    impact_score: float  # 0 to 1 (potential market impact)
    keywords: List[str]
    category: str  # 'crypto', 'macro', 'tech', 'politics', etc.


@dataclass
class NewsSentiment:
    """News sentiment aggregation structure."""
    overall_sentiment: float  # -1 to 1
    sentiment_label: str
    article_count: int
    positive_articles: int
    negative_articles: int
    neutral_articles: int
    trending_keywords: List[str]
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    market_impact: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0 to 1
    timestamp: datetime


@dataclass
class MarketEvent:
    """Market-moving event structure."""
    title: str
    description: str
    event_type: str  # 'earnings', 'fed_meeting', 'regulation', 'partnership', etc.
    impact_level: str  # 'high', 'medium', 'low'
    affected_assets: List[str]
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    source: str
    timestamp: datetime
    url: str


class NewsService:
    """
    Comprehensive news and sentiment analysis service for financial markets.
    
    Aggregates news from multiple sources and provides sentiment analysis
    with market impact scoring for trading decisions.
    """
    
    def __init__(self, newsapi_key: Optional[str] = None):
        """Initialize News service."""
        self.newsapi_key = newsapi_key
        self.newsapi_client = NewsApiClient(api_key=newsapi_key) if newsapi_key else None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Cache settings
        self.cache_duration = {
            'news': 300,  # 5 minutes for news articles
            'sentiment': 600,  # 10 minutes for sentiment analysis
            'events': 900  # 15 minutes for market events
        }
        self.cache = {}
        
        # Crypto-related keywords for relevance scoring
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'trading', 'binance',
            'coinbase', 'solana', 'cardano', 'polkadot', 'avalanche',
            'doge', 'shiba', 'matic', 'chainlink', 'uniswap', 'aave',
            'compound', 'maker', 'curve', 'sushi', 'pancake', 'ftx',
            'kraken', 'huobi', 'kucoin', 'metamask', 'ledger', 'trezor',
            'mining', 'staking', 'yield', 'liquidity', 'swap', 'bridge'
        ]
        
        # Financial keywords for relevance
        self.financial_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
            'unemployment', 'cpi', 'ppi', 'fomc', 'treasury', 'bond',
            'stock market', 'nasdaq', 'dow jones', 's&p 500', 'volatility',
            'recession', 'economy', 'dollar', 'usd', 'euro', 'yen',
            'commodities', 'gold', 'oil', 'energy', 'powell', 'yellen'
        ]
        
        logger.info("NewsService initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()
    
    async def start_session(self):
        """Start HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("News HTTP session started")
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("News HTTP session closed")
    
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
    
    async def get_crypto_news(self, limit: int = 50, hours_back: int = 24) -> List[NewsArticle]:
        """
        Get cryptocurrency-related news articles.
        
        Args:
            limit: Maximum number of articles to return
            hours_back: How many hours back to search
            
        Returns:
            List of NewsArticle objects
        """
        try:
            cache_key = f"crypto_news_{limit}_{hours_back}"
            
            if self._is_cache_valid(cache_key, 'news'):
                return self.cache[cache_key]['data']
            
            articles = []
            
            # Get articles from NewsAPI if available
            if self.newsapi_client:
                articles.extend(await self._get_newsapi_articles('crypto', limit // 2, hours_back))
            
            # Get articles from free sources
            articles.extend(await self._get_free_crypto_news(limit // 2))
            
            # Remove duplicates and sort by relevance
            unique_articles = self._deduplicate_articles(articles)
            sorted_articles = sorted(unique_articles, key=lambda x: x.relevance_score, reverse=True)
            
            result = sorted_articles[:limit]
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
            return []
    
    async def get_market_sentiment(self, hours_back: int = 24) -> NewsSentiment:
        """
        Analyze overall market sentiment from news articles.
        
        Args:
            hours_back: How many hours back to analyze
            
        Returns:
            NewsSentiment object
        """
        try:
            cache_key = f"market_sentiment_{hours_back}"
            
            if self._is_cache_valid(cache_key, 'sentiment'):
                return self.cache[cache_key]['data']
            
            # Get recent articles
            articles = await self.get_crypto_news(limit=100, hours_back=hours_back)
            
            if not articles:
                return self._empty_sentiment()
            
            # Analyze sentiment
            sentiments = [article.sentiment_score for article in articles]
            overall_sentiment = sum(sentiments) / len(sentiments)
            
            # Count sentiment categories
            positive_articles = len([s for s in sentiments if s > 0.1])
            negative_articles = len([s for s in sentiments if s < -0.1])
            neutral_articles = len(articles) - positive_articles - negative_articles
            
            # Determine sentiment label
            if overall_sentiment > 0.2:
                sentiment_label = "positive"
                market_impact = "bullish"
            elif overall_sentiment < -0.2:
                sentiment_label = "negative"
                market_impact = "bearish"
            else:
                sentiment_label = "neutral"
                market_impact = "neutral"
            
            # Extract trending keywords
            all_keywords = [kw for article in articles for kw in article.keywords]
            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            trending_keywords = sorted(keyword_counts.keys(), key=lambda x: keyword_counts[x], reverse=True)[:10]
            
            # Calculate sentiment trend (simplified)
            recent_articles = [a for a in articles if (datetime.now() - a.published_at).total_seconds() < 3600 * 6]  # Last 6 hours
            older_articles = [a for a in articles if a not in recent_articles]
            
            if recent_articles and older_articles:
                recent_sentiment = sum(a.sentiment_score for a in recent_articles) / len(recent_articles)
                older_sentiment = sum(a.sentiment_score for a in older_articles) / len(older_articles)
                
                if recent_sentiment > older_sentiment + 0.1:
                    sentiment_trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    sentiment_trend = "declining"
                else:
                    sentiment_trend = "stable"
            else:
                sentiment_trend = "stable"
            
            # Calculate confidence based on article count and agreement
            confidence = min(len(articles) / 50, 1.0)  # More articles = higher confidence
            sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
            confidence *= max(0.1, 1.0 - sentiment_variance)  # Lower variance = higher confidence
            
            sentiment = NewsSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_label=sentiment_label,
                article_count=len(articles),
                positive_articles=positive_articles,
                negative_articles=negative_articles,
                neutral_articles=neutral_articles,
                trending_keywords=trending_keywords,
                sentiment_trend=sentiment_trend,
                market_impact=market_impact,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            self._cache_data(cache_key, sentiment)
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return self._empty_sentiment()
    
    async def detect_market_events(self, hours_back: int = 24) -> List[MarketEvent]:
        """
        Detect significant market-moving events from news.
        
        Args:
            hours_back: How many hours back to search
            
        Returns:
            List of MarketEvent objects
        """
        try:
            cache_key = f"market_events_{hours_back}"
            
            if self._is_cache_valid(cache_key, 'events'):
                return self.cache[cache_key]['data']
            
            # Get recent high-impact articles
            articles = await self.get_crypto_news(limit=100, hours_back=hours_back)
            high_impact_articles = [a for a in articles if a.impact_score > 0.7]
            
            events = []
            
            for article in high_impact_articles:
                event = self._article_to_event(article)
                if event:
                    events.append(event)
            
            # Sort by impact and remove duplicates
            events = self._deduplicate_events(events)
            events.sort(key=lambda x: self._event_impact_score(x.impact_level), reverse=True)
            
            result = events[:20]  # Top 20 events
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market events: {e}")
            return []
    
    async def _get_newsapi_articles(self, query: str, limit: int, hours_back: int) -> List[NewsArticle]:
        """Get articles from NewsAPI."""
        if not self.newsapi_client:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            # Search for crypto-related articles
            everything = self.newsapi_client.get_everything(
                q=f"{query} OR bitcoin OR ethereum OR crypto OR blockchain",
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=min(limit, 100)
            )
            
            articles = []
            for article_data in everything.get('articles', []):
                try:
                    article = await self._parse_newsapi_article(article_data)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing NewsAPI article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return []
    
    async def _get_free_crypto_news(self, limit: int) -> List[NewsArticle]:
        """Get crypto news from free sources."""
        articles = []
        
        try:
            # CoinDesk RSS feed
            coindesk_articles = await self._fetch_rss_feed(
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                limit // 3
            )
            articles.extend(coindesk_articles)
            
            # Cointelegraph RSS feed
            cointelegraph_articles = await self._fetch_rss_feed(
                "https://cointelegraph.com/rss",
                limit // 3
            )
            articles.extend(cointelegraph_articles)
            
            # CryptoNews RSS feed
            cryptonews_articles = await self._fetch_rss_feed(
                "https://cryptonews.com/news/feed",
                limit // 3
            )
            articles.extend(cryptonews_articles)
            
        except Exception as e:
            logger.error(f"Error fetching free crypto news: {e}")
        
        return articles
    
    async def _fetch_rss_feed(self, url: str, limit: int) -> List[NewsArticle]:
        """Fetch and parse RSS feed."""
        try:
            if not self.session:
                await self.start_session()
            
            async with self.session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'xml')
                
                articles = []
                items = soup.find_all('item')[:limit]
                
                for item in items:
                    try:
                        title = item.find('title').text if item.find('title') else ""
                        description = item.find('description').text if item.find('description') else ""
                        link = item.find('link').text if item.find('link') else ""
                        pub_date = item.find('pubDate').text if item.find('pubDate') else ""
                        
                        # Parse publication date
                        try:
                            from email.utils import parsedate_to_datetime
                            published_at = parsedate_to_datetime(pub_date)
                        except:
                            published_at = datetime.now()
                        
                        # Create article with sentiment analysis
                        article = NewsArticle(
                            title=title,
                            description=description,
                            content=description,  # RSS feeds typically don't have full content
                            source=url.split('/')[2],  # Extract domain
                            author="",
                            url=link,
                            published_at=published_at,
                            sentiment_score=0.0,  # Will be calculated
                            sentiment_label="neutral",
                            relevance_score=0.0,  # Will be calculated
                            impact_score=0.0,  # Will be calculated
                            keywords=[],  # Will be extracted
                            category="crypto"
                        )
                        
                        # Perform sentiment and relevance analysis
                        self._analyze_article(article)
                        articles.append(article)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing RSS item: {e}")
                        continue
                
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
            return []
    
    async def _parse_newsapi_article(self, article_data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Parse article from NewsAPI response."""
        try:
            # Parse publication date
            pub_date_str = article_data.get('publishedAt', '')
            try:
                published_at = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
            except:
                published_at = datetime.now()
            
            article = NewsArticle(
                title=article_data.get('title', ''),
                description=article_data.get('description', ''),
                content=article_data.get('content', ''),
                source=article_data.get('source', {}).get('name', ''),
                author=article_data.get('author', ''),
                url=article_data.get('url', ''),
                published_at=published_at,
                sentiment_score=0.0,
                sentiment_label="neutral",
                relevance_score=0.0,
                impact_score=0.0,
                keywords=[],
                category="crypto"
            )
            
            # Perform analysis
            self._analyze_article(article)
            return article
            
        except Exception as e:
            logger.error(f"Error parsing NewsAPI article: {e}")
            return None
    
    def _analyze_article(self, article: NewsArticle):
        """Perform sentiment and relevance analysis on article."""
        try:
            # Combine text for analysis
            full_text = f"{article.title} {article.description} {article.content}"
            
            # Sentiment analysis using VADER
            vader_scores = self.vader_analyzer.polarity_scores(full_text)
            compound_score = vader_scores['compound']
            
            # Sentiment analysis using TextBlob as backup
            try:
                blob = TextBlob(full_text)
                textblob_score = blob.sentiment.polarity
                
                # Average the two scores
                article.sentiment_score = (compound_score + textblob_score) / 2
            except:
                article.sentiment_score = compound_score
            
            # Determine sentiment label
            if article.sentiment_score > 0.1:
                article.sentiment_label = "positive"
            elif article.sentiment_score < -0.1:
                article.sentiment_label = "negative"
            else:
                article.sentiment_label = "neutral"
            
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance(full_text)
            
            # Calculate impact score
            article.impact_score = self._calculate_impact(article)
            
            # Extract keywords
            article.keywords = self._extract_keywords(full_text)
            
            # Categorize article
            article.category = self._categorize_article(full_text)
            
        except Exception as e:
            logger.error(f"Error analyzing article: {e}")
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score to crypto/financial markets."""
        text_lower = text.lower()
        
        # Count crypto keywords
        crypto_matches = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)
        financial_matches = sum(1 for keyword in self.financial_keywords if keyword in text_lower)
        
        # Calculate score
        total_matches = crypto_matches + financial_matches * 0.5  # Weight crypto higher
        max_possible = len(self.crypto_keywords) + len(self.financial_keywords) * 0.5
        
        relevance = min(total_matches / 10, 1.0)  # Normalize to 0-1
        return relevance
    
    def _calculate_impact(self, article: NewsArticle) -> float:
        """Calculate potential market impact score."""
        impact = 0.0
        
        # Source credibility
        credible_sources = ['reuters', 'bloomberg', 'coindesk', 'cointelegraph', 'wall street journal']
        if any(source in article.source.lower() for source in credible_sources):
            impact += 0.3
        
        # Title impact words
        impact_words = ['breaking', 'urgent', 'major', 'massive', 'huge', 'crash', 'surge', 'rally', 'dump']
        title_lower = article.title.lower()
        if any(word in title_lower for word in impact_words):
            impact += 0.2
        
        # Relevance bonus
        impact += article.relevance_score * 0.3
        
        # Sentiment extremity bonus
        impact += abs(article.sentiment_score) * 0.2
        
        return min(impact, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        
        # Find crypto and financial keywords
        found_keywords = []
        for keyword in self.crypto_keywords + self.financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract additional keywords using simple regex
        words = re.findall(r'\b[A-Z][A-Z0-9]{2,}\b', text)  # Crypto tickers (3+ chars)
        found_keywords.extend([w.lower() for w in words[:5]])  # Limit to 5
        
        return list(set(found_keywords))[:10]  # Unique keywords, max 10
    
    def _categorize_article(self, text: str) -> str:
        """Categorize article by topic."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['regulation', 'sec', 'cftc', 'legal', 'law']):
            return 'regulation'
        elif any(word in text_lower for word in ['fed', 'federal reserve', 'interest rate', 'inflation']):
            return 'macro'
        elif any(word in text_lower for word in ['partnership', 'acquisition', 'merger', 'funding']):
            return 'business'
        elif any(word in text_lower for word in ['hack', 'security', 'breach', 'exploit']):
            return 'security'
        elif any(word in text_lower for word in ['defi', 'nft', 'dao', 'yield', 'liquidity']):
            return 'defi'
        else:
            return 'crypto'
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication by title
            title_key = article.title.lower().strip()
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    def _article_to_event(self, article: NewsArticle) -> Optional[MarketEvent]:
        """Convert high-impact article to market event."""
        try:
            # Determine event type
            event_type = "announcement"
            title_lower = article.title.lower()
            
            if any(word in title_lower for word in ['earnings', 'quarterly', 'results']):
                event_type = "earnings"
            elif any(word in title_lower for word in ['fed', 'federal reserve', 'fomc']):
                event_type = "fed_meeting"
            elif any(word in title_lower for word in ['regulation', 'sec', 'cftc']):
                event_type = "regulation"
            elif any(word in title_lower for word in ['partnership', 'acquisition']):
                event_type = "partnership"
            elif any(word in title_lower for word in ['hack', 'breach', 'exploit']):
                event_type = "security"
            
            # Determine impact level
            if article.impact_score > 0.8:
                impact_level = "high"
            elif article.impact_score > 0.5:
                impact_level = "medium"
            else:
                impact_level = "low"
            
            # Determine affected assets (simplified)
            affected_assets = [kw.upper() for kw in article.keywords if kw.upper() in ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']]
            if not affected_assets:
                affected_assets = ['BTC', 'ETH']  # Default to major cryptos
            
            # Determine sentiment
            if article.sentiment_score > 0.1:
                sentiment = "bullish"
            elif article.sentiment_score < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return MarketEvent(
                title=article.title,
                description=article.description,
                event_type=event_type,
                impact_level=impact_level,
                affected_assets=affected_assets,
                sentiment=sentiment,
                source=article.source,
                timestamp=article.published_at,
                url=article.url
            )
            
        except Exception as e:
            logger.error(f"Error converting article to event: {e}")
            return None
    
    def _deduplicate_events(self, events: List[MarketEvent]) -> List[MarketEvent]:
        """Remove duplicate events."""
        if not events:
            return []
        
        unique_events = []
        seen_titles = set()
        
        for event in events:
            title_key = event.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_events.append(event)
        
        return unique_events
    
    def _event_impact_score(self, impact_level: str) -> int:
        """Convert impact level to numeric score for sorting."""
        return {"high": 3, "medium": 2, "low": 1}.get(impact_level, 0)
    
    def _empty_sentiment(self) -> NewsSentiment:
        """Return empty sentiment data."""
        return NewsSentiment(
            overall_sentiment=0.0,
            sentiment_label="neutral",
            article_count=0,
            positive_articles=0,
            negative_articles=0,
            neutral_articles=0,
            trending_keywords=[],
            sentiment_trend="stable",
            market_impact="neutral",
            confidence=0.0,
            timestamp=datetime.now()
        )


# Factory function
def create_news_service(newsapi_key: Optional[str] = None) -> NewsService:
    """Create and return NewsService instance."""
    return NewsService(newsapi_key)