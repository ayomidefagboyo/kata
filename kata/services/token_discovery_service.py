"""
Token Discovery Service for Flow AI Trading Platform.

Handles AI-powered token discovery, analysis, and real-time market data updates.
"""

import logging
import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4

from kata.models.token_discovery import DiscoveredToken, TokenAnalysis, TokenMarketData
from kata.services.token_discovery_database_service import TokenDiscoveryDatabaseService, get_token_discovery_db_service
from kata.services.coingecko_service import CoinGeckoService

logger = logging.getLogger(__name__)


class TokenDiscoveryService:
    """Service for managing token discovery with AI analysis and real-time data."""
    
    def __init__(self, db_service: TokenDiscoveryDatabaseService = None):
        self.db_service = db_service or get_token_discovery_db_service()
        self.coingecko_service = CoinGeckoService()
        logger.info("Token discovery service initialized")
    
    async def get_trending_tokens(
        self, 
        limit: int = 50, 
        blockchain_filter: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None
    ) -> List[DiscoveredToken]:
        """Get trending tokens with analysis and real-time market data."""
        try:
            # Get tokens from database with analysis
            tokens = await self.db_service.get_trending_tokens(
                limit=limit,
                blockchain_filter=blockchain_filter,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap
            )
            
            # Update real-time market data for returned tokens
            updated_tokens = []
            for token in tokens:
                try:
                    # Update market data if older than 5 minutes
                    if self._should_update_market_data(token.last_updated):
                        market_data = await self._fetch_real_time_data(token.address, token.blockchain)
                        if market_data:
                            token.current_price = market_data.get('price', token.current_price)
                            token.market_cap = market_data.get('market_cap', token.market_cap)
                            token.volume_24h = market_data.get('volume_24h', token.volume_24h)
                            token.price_change_24h = market_data.get('price_change_24h', token.price_change_24h)
                            token.last_updated = datetime.now()
                            
                            # Update in database
                            await self.db_service.update_token_market_data(token.id, market_data)
                    
                    updated_tokens.append(token)
                except Exception as e:
                    logger.warning(f"Failed to update market data for {token.symbol}: {e}")
                    updated_tokens.append(token)  # Include token even if market data update fails
            
            return updated_tokens
            
        except Exception as e:
            logger.error(f"Error getting trending tokens: {e}")
            # Return empty list instead of mock data
            return []
    
    async def search_tokens(self, query: str, limit: int = 20) -> List[DiscoveredToken]:
        """Search tokens by symbol or name."""
        try:
            return await self.db_service.search_tokens(query, limit)
        except Exception as e:
            logger.error(f"Error searching tokens: {e}")
            return []
    
    async def get_token_analysis(self, token_id: str) -> Optional[TokenAnalysis]:
        """Get detailed analysis for a specific token."""
        try:
            return await self.db_service.get_token_analysis(token_id)
        except Exception as e:
            logger.error(f"Error getting token analysis: {e}")
            return None
    
    async def analyze_token(self, token_id: str) -> Dict[str, Any]:
        """Trigger AI analysis for a specific token."""
        try:
            # Get token details
            token = await self.db_service.get_token_by_id(token_id)
            if not token:
                raise ValueError(f"Token {token_id} not found")
            
            # Run AI analysis
            analysis_result = await self._run_ai_analysis(token)
            
            # Store analysis results
            analysis_id = await self.db_service.store_token_analysis(token_id, analysis_result)
            
            return {
                "analysis_id": analysis_id,
                "status": "completed",
                "token_id": token_id
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_id}: {e}")
            return {
                "analysis_id": None,
                "status": "failed",
                "error": str(e)
            }
    
    async def run_daily_analysis(self) -> Dict[str, Any]:
        """Run daily analysis on all active tokens."""
        start_time = datetime.now()
        logger.info("Starting daily token analysis")
        
        if not getattr(settings, "RYU_BACKGROUND_DISCOVERY_ENABLED", True):
            logger.info("Ryu background discovery is disabled in settings. Skipping daily analysis.")
            return {
                "tokens_analyzed": 0,
                "new_discoveries": 0,
                "analysis_time": 0.0
            }
        
        try:
            # Get all active tokens
            active_tokens = await self.db_service.get_active_tokens()
            
            # Discover new tokens
            new_tokens = await self._discover_new_tokens()
            new_discoveries = len(new_tokens)
            
            # Analyze all tokens (existing + new)
            all_tokens = active_tokens + new_tokens
            tokens_analyzed = 0
            
            for token in all_tokens:
                try:
                    await self.analyze_token(token.id)
                    tokens_analyzed += 1
                except Exception as e:
                    logger.error(f"Failed to analyze token {token.symbol}: {e}")
            
            # Calculate analysis time
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Store daily analysis result
            await self.db_service.store_daily_analysis_result({
                "analysis_date": start_time,
                "tokens_analyzed": tokens_analyzed,
                "new_discoveries": new_discoveries,
                "analysis_duration": analysis_time,
                "status": "completed"
            })
            
            logger.info(f"Daily analysis completed: {tokens_analyzed} tokens analyzed, {new_discoveries} new discoveries")
            
            return {
                "tokens_analyzed": tokens_analyzed,
                "new_discoveries": new_discoveries,
                "analysis_time": analysis_time
            }
            
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")
            raise
    
    async def get_real_time_market_data(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time market data for a token."""
        try:
            token = await self.db_service.get_token_by_id(token_id)
            if not token:
                return None
            
            market_data = await self._fetch_real_time_data(token.address, token.blockchain)
            
            # Store market data in database
            if market_data:
                await self.db_service.store_market_data(token_id, market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting real-time market data: {e}")
            return None
    
    # Private helper methods
    
    def _should_update_market_data(self, last_updated: datetime) -> bool:
        """Check if market data should be updated (older than 5 minutes)."""
        return datetime.now() - last_updated > timedelta(minutes=5)
    
    async def _fetch_real_time_data(self, address: str, blockchain: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time market data from external APIs."""
        try:
            # Mock implementation - replace with actual API calls
            # This would call CoinGecko, DexScreener, etc.
            
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            # Return mock data for now
            import random
            base_price = random.uniform(0.0001, 100)
            
            return {
                "price": base_price,
                "market_cap": base_price * random.randint(1000000, 1000000000),
                "volume_24h": base_price * random.randint(10000, 10000000),
                "price_change_24h": random.uniform(-20, 20),
                "liquidity": base_price * random.randint(50000, 5000000),
                "last_updated": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {address}: {e}")
            return None
    
    async def discover_tokens_batch(
        self, 
        limit: int = 50,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume: Optional[float] = None
    ) -> List[DiscoveredToken]:
        """Discover new tokens from CoinGecko with batch processing."""
        try:
            logger.info(f"Discovering tokens with batch processing - limit={limit}")
            
            # Get trending tokens from CoinGecko
            trending_data = await self.coingecko_service.get_trending_data()
            trending_coins = [{'symbol': symbol, 'name': symbol} for symbol in trending_data.trending_coins]
            
            discovered_tokens = []
            for coin_data in trending_coins[:limit]:
                try:
                    # Convert CoinGecko data to DiscoveredToken
                    token = DiscoveredToken(
                        id=str(uuid4()),
                        symbol=coin_data.get('symbol', '').upper(),
                        name=coin_data.get('name', ''),
                        blockchain='ethereum',  # Default to Ethereum
                        chain_id=1,
                        address=coin_data.get('contract_address', ''),
                        logo_url=coin_data.get('large', ''),
                        current_price=coin_data.get('price', 0.0),
                        market_cap=coin_data.get('market_cap', 0.0),
                        volume_24h=coin_data.get('volume_24h', 0.0),
                        price_change_24h=coin_data.get('price_change_24h', 0.0),
                        last_updated=datetime.now(),
                        risk_score=5.0,  # Default neutral risk
                        potential_score=6.0,  # Default potential  
                        confidence_score=0.7,  # Default confidence
                        analysis_summary="Basic analysis from trending data",
                        trading_signals=["Trending token"],
                        analysis_date=datetime.now()
                    )
                    
                    # Apply filters if specified
                    if min_market_cap and token.market_cap < min_market_cap:
                        continue
                    if max_market_cap and token.market_cap > max_market_cap:
                        continue  
                    if min_volume and token.volume_24h < min_volume:
                        continue
                        
                    discovered_tokens.append(token)
                    
                except Exception as e:
                    logger.warning(f"Error processing coin data: {e}")
                    continue
            
            logger.info(f"Discovered {len(discovered_tokens)} tokens matching criteria")
            return discovered_tokens
            
        except Exception as e:
            logger.error(f"Error in batch token discovery: {e}")
            return []
    
    async def _run_ai_analysis(self, token: DiscoveredToken) -> Dict[str, Any]:
        """Run AI analysis on a token."""
        try:
            from kata.services.unified_signal_generator import get_unified_signal_generator
            # Initialize with default agent or the agent associated with discovery
            generator = get_unified_signal_generator(agent_id="yuki")
            
            # Fetch technical analysis (which now includes advanced AI metrics)
            technical_analysis = await generator.perform_technical_analysis(token.symbol)
            if not technical_analysis:
                raise ValueError(f"Insufficient technical data for {token.symbol}")
                
            # Calculate opportunity score
            opportunity_score = generator._calculate_opportunity_score(technical_analysis)
            
            # Get actual LLM decision
            ai_decision = await generator.ai_analyze_opportunity(
                symbol=token.symbol,
                technical_analysis=technical_analysis,
                opportunity_score=opportunity_score
            )
            
            if not ai_decision:
                raise ValueError(f"AI failed to generate a decision for {token.symbol}")
            
            # Map Risk Level to Score (1-10)
            risk_map = {"LOW": 3.0, "MEDIUM": 5.0, "HIGH": 8.0, "EXTREME": 10.0}
            risk_score = risk_map.get(ai_decision.risk_level.upper(), 5.0)
            
            # Calculate potential based on Risk-Reward Ratio and Confidence
            potential_score = min(10.0, ai_decision.risk_reward_ratio * ai_decision.confidence * 3)
            
            # Map sentiment
            sentiment_score = 0.5
            if ai_decision.recommendation == "LONG":
                sentiment_score = 0.5 + (0.5 * ai_decision.confidence)
            elif ai_decision.recommendation == "SHORT":
                sentiment_score = 0.5 - (0.5 * ai_decision.confidence)
                
            # Compile signals & opportunities
            signals = [ai_decision.recommendation] + ai_decision.key_factors
            opportunities = [f"Entry: {ai_decision.entry_price}", f"Target 1: {ai_decision.target_1}"]
            if ai_decision.target_2 > 0:
                opportunities.append(f"Target 2: {ai_decision.target_2}")

            # Prepare technical indicators including advanced metrics for Ryu UI
            tech_indicators = {
                "rsi": technical_analysis.rsi_14,
                "macd": technical_analysis.macd_histogram,
                "volume_trend": "increasing" if technical_analysis.volume_ratio > 1 else "decreasing",
                # Advanced AI Metrics injected for Ryu Dashboard
                "taker_buy_sell_ratio": technical_analysis.taker_buy_sell_ratio,
                "liquidation_volume": technical_analysis.liquidation_volume,
                "basis_premium": technical_analysis.basis_premium
            }

            return {
                "risk_score": round(risk_score, 2),
                "potential_score": round(potential_score, 2),
                "confidence_score": round(ai_decision.confidence, 2),
                "overall_score": round((potential_score + (10 - risk_score)) / 2, 2),
                "summary": ai_decision.reasoning,
                "signals": signals,
                "risk_factors": ai_decision.risk_factors,
                "opportunities": opportunities,
                "technical_indicators": tech_indicators,
                "sentiment_score": round(sentiment_score, 2),
                "analysis_version": "2.0-advanced"
            }
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {token.symbol}: {e}")
            raise
    
    async def _discover_new_tokens(self) -> List[DiscoveredToken]:
        """Discover new tokens from CoinGecko API with $1M-$500M market cap criteria."""
        try:
            logger.info("🔍 Discovering new tokens from CoinGecko API (pages 13-50, $1M-$500M market cap)...")
            
            # Don't clear existing tokens for now - just add new ones
            # await self._clear_existing_tokens()
            
            # Get tokens from CoinGecko API
            new_tokens = []
            
            # Fetch comprehensive pages for better token coverage
            # Process in batches to respect rate limits while getting full coverage
            for page in range(1, 101):  # Pages 1-100 for comprehensive coverage
                try:
                    page_tokens = await self._fetch_coingecko_page(page)
                    if page_tokens:
                        new_tokens.extend(page_tokens)
                        logger.info(f"✅ Successfully fetched {len(page_tokens)} tokens from page {page}")
                    else:
                        logger.warning(f"No tokens returned from page {page}")
                    
                    # Add delay to respect rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch page {page}: {e}")
                    # Continue with next page instead of stopping
                    continue
            
            # Filter tokens by market cap ($1M - $500M)
            filtered_tokens = []
            for token in new_tokens:
                if 1_000_000 <= token.market_cap <= 500_000_000:
                    filtered_tokens.append(token)
            
            logger.info(f"📊 Filtered {len(filtered_tokens)} tokens in $1M-$500M range from {len(new_tokens)} total")
            
            # Remove duplicates by symbol
            unique_tokens = {}
            for token in filtered_tokens:
                if token.symbol not in unique_tokens:
                    unique_tokens[token.symbol] = token
            
            final_tokens = list(unique_tokens.values())
            
            # Store new tokens in database
            stored_tokens = []
            for token in final_tokens:
                try:
                    stored_token = await self._store_discovered_token(token)
                    if stored_token:
                        stored_tokens.append(stored_token)
                except Exception as e:
                    logger.error(f"Failed to store token {token.symbol}: {e}")
            
            logger.info(f"✅ Discovered and stored {len(stored_tokens)} new quality tokens")
            return stored_tokens
            
        except Exception as e:
            logger.error(f"Error discovering new tokens: {e}")
            return []
    
    async def _fetch_coingecko_page(self, page: int) -> List[DiscoveredToken]:
        """Fetch a page of tokens from CoinGecko API."""
        try:
            # Use CoinGecko service to fetch market data
            await self.coingecko_service.start_session()
            
            # Construct the API endpoint URL manually since we need specific pagination
            import aiohttp
            import ssl
            
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': '100',  # Max per page
                'page': str(page),
                'sparkline': 'false',
                'price_change_percentage': '24h',
                'locale': 'en'
            }
            
            # Create SSL context that doesn't verify certificates for development
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        tokens = []
                        for coin_data in data:
                            try:
                                # Skip coins without proper market data
                                if not coin_data.get('market_cap') or not coin_data.get('current_price'):
                                    continue
                                
                                # Quality filtering
                                if not self._passes_quality_filter(coin_data):
                                    continue
                                
                                # Determine blockchain based on categories or default to ethereum
                                blockchain = self._determine_blockchain(coin_data)
                                
                                token = DiscoveredToken(
                                    id=str(uuid4()),
                                    symbol=coin_data['symbol'].upper(),
                                    name=coin_data['name'],
                                    blockchain=blockchain,
                                    chain_id=self._get_chain_id(blockchain),
                                    address=coin_data.get('contract_address', ''),  # May not be available
                                    logo_url=coin_data.get('image', ''),
                                    coingecko_id=coin_data['id'],
                                    discovery_date=datetime.now(),
                                    current_price=float(coin_data['current_price'] or 0),
                                    market_cap=float(coin_data['market_cap'] or 0),
                                    volume_24h=float(coin_data['total_volume'] or 0),
                                    price_change_24h=float(coin_data.get('price_change_24h', 0) or 0),
                                    liquidity=0,  # Not available from this endpoint
                                    last_updated=datetime.now(),
                                    is_trending=False,  # Will be determined by analysis
                                    is_active=True
                                )
                                tokens.append(token)
                                
                            except Exception as e:
                                logger.warning(f"Failed to process coin {coin_data.get('symbol', 'unknown')}: {e}")
                                continue
                        
                        logger.debug(f"📄 Fetched {len(tokens)} tokens from page {page}")
                        return tokens
                    else:
                        logger.warning(f"CoinGecko API returned status {response.status} for page {page}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching CoinGecko page {page}: {e}")
            return []
        finally:
            await self.coingecko_service.close_session()
    
    def _passes_quality_filter(self, coin_data: Dict[str, Any]) -> bool:
        """Apply quality filters to discovered tokens."""
        try:
            # Basic data requirements
            market_cap = coin_data.get('market_cap', 0)
            volume_24h = coin_data.get('total_volume', 0)
            price = coin_data.get('current_price', 0)
            name = coin_data.get('name', '').lower()
            symbol = coin_data.get('symbol', '').upper()

            # Filter 1: Check for stablecoins and wrapped tokens FIRST
            if self._is_stablecoin_or_wrapped_token(symbol, name):
                return False

            # Filter 2: Minimum volume requirement (at least $50k daily volume)
            if volume_24h < 50_000:
                return False

            # Filter 3: Price must be reasonable (not too cheap or expensive)
            if price <= 0 or price > 10_000:
                return False

            # Filter 4: Market cap rank exists (indicates legitimacy)
            if not coin_data.get('market_cap_rank'):
                return False

            # Filter 5: Minimum market cap (already filtered in main loop, but double-check)
            if market_cap < 500_000:  # At least $500k market cap
                return False

            # Filter 6: Volume to market cap ratio (avoid dead tokens)
            if market_cap > 0:
                volume_ratio = volume_24h / market_cap
                if volume_ratio < 0.002:  # Less than 0.2% of market cap traded daily
                    return False

            # Filter 7: Avoid obvious scam patterns and test tokens
            scam_keywords = [
                'scam', 'fake', 'test', 'ponzi', 'rugpull', 'honeypot',
                'airdrop', 'claim', 'free', 'double', 'x100', 'moon',
                'shib', 'doge', 'elon', 'musk', 'tesla', 'biden', 'trump'
            ]
            if any(keyword in name for keyword in scam_keywords):
                return False

            # Filter 8: Symbol length (reasonable length, but allow some flexibility)
            if len(symbol) > 20 or len(symbol) < 1:
                return False

            # Filter 9: Avoid tokens with suspicious symbols
            suspicious_patterns = ['xxx', '69', '420', '1000', '2.0', 'v2', 'old']
            if any(pattern in symbol.lower() for pattern in suspicious_patterns):
                return False

            # Filter 10: Avoid meme coin patterns in names
            meme_patterns = ['inu', 'shiba', 'doge', 'moon', 'safe', 'baby', 'mini', 'elon']
            meme_count = sum(1 for pattern in meme_patterns if pattern in name)
            if meme_count >= 2:  # Allow single meme keywords but not multiple
                return False

            return True

        except Exception as e:
            logger.warning(f"Error in quality filter: {e}")
            return False

    def _is_stablecoin_or_wrapped_token(self, symbol: str, name: str) -> bool:
        """Check if token is a stablecoin or wrapped token that should be filtered out."""
        symbol_upper = symbol.upper()
        name_lower = name.lower()

        # Comprehensive stablecoin symbols
        stablecoin_symbols = {
            # Major USD stablecoins
            'USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'TUSD', 'USDP', 'LUSD',
            'MIM', 'USDD', 'GUSD', 'SUSD', 'USDN', 'RSR', 'USTC', 'FDUSD',
            'PYUSD', 'CRVUSD', 'USDE', 'USDM', 'USDS', 'USDG', 'USDY', 'USDB',
            'DOLA', 'FEI', 'TRIBE', 'RAI', 'OUSD', 'USDK', 'MUSD', 'USDBC',
            'USDQ',  # Quantoz USDQ

            # Euro stablecoins
            'EURC', 'EURS', 'EURT', 'EURCV',

            # Other regional stablecoins
            'GYEN', 'XSGD', 'TRYB', 'BIDR', 'IDRT', 'VAI', 'DJED', 'DUSD', 'XUSD',

            # Synthetic and algorithmic stablecoins
            'FRAX', 'MIM', 'LUSD', 'RAI', 'DOLA'
        }

        # Comprehensive wrapped token symbols
        wrapped_token_symbols = {
            # Wrapped ETH variants
            'WETH', 'WETH9', 'ETHW',  # EthereumPoW

            # Major wrapped tokens
            'WBTC', 'WBNB', 'WMATIC', 'WAVAX', 'WFTM', 'WONE', 'WCRO',
            'WKLAY', 'WMOVR', 'WGLMR', 'WROSE', 'WXDAI', 'WMNT', 'WPOL',

            # Aave wrapped tokens
            'AWETH', 'AWBTC', 'AUSDC', 'AUSDT', 'ADAI',

            # Hyperliquid wrapped tokens
            'WHYPE',

            # Bridged variants
            'SBETH'  # Sui Bridged Ether
        }

        # Extended wrapped/bridged token name patterns
        wrapped_keywords = [
            'wrapped', 'bridged', 'pegged', 'synthetic', 'aave ethereum',
            'sui bridged', 'bridged usdt', 'bridged usdc', 'wormhole token',
            'polygon pos bridged', 'mantle bridged', 'vaultbridge bridged',
            'bridged usd coin', 'wrapped aave', 'lido wsteth', 'aave base'
        ]

        # Check stablecoin symbols
        if symbol_upper in stablecoin_symbols:
            return True

        # Check wrapped token symbols
        if symbol_upper in wrapped_token_symbols:
            return True

        # Check for complex wrapped tokens containing stablecoin names
        if any(stable in symbol_upper for stable in ['USDC', 'USDT', 'DAI', 'GHO']) and symbol_upper.startswith(('W', 'A', 'WA')) and len(symbol_upper) > 4:
            return True

        # Check wrapped prefixes with expanded exceptions
        if symbol_upper.startswith(('W', 'WW', 'AW')):
            # Legitimate tokens that start with W but are not wrapped
            exceptions = {
                'WLD', 'WOO', 'WNXM', 'WAXP', 'WIN', 'WAVES', 'WRX', 'WAN',
                'WOLF', 'WFI', 'WEN', 'WAI', 'WCT', 'WORTHLESS'
            }
            if symbol_upper not in exceptions:
                return True

        # Check name patterns for wrapped/bridged indicators
        if any(keyword in name_lower for keyword in wrapped_keywords):
            return True

        # Additional pattern checks for specific naming conventions
        # Check for "bridged" in parentheses (common pattern)
        if 'bridged' in name_lower and ('(' in name_lower and ')' in name_lower):
            return True

        # Check for "wrapped" in parentheses
        if 'wrapped' in name_lower and ('(' in name_lower and ')' in name_lower):
            return True

        # Check for chain-specific bridged tokens
        chain_bridge_patterns = ['polygon pos', 'base bridged', 'linea bridged', 'mantle bridged']
        if any(pattern in name_lower for pattern in chain_bridge_patterns):
            return True

        return False
    
    def _determine_blockchain(self, coin_data: Dict[str, Any]) -> str:
        """Determine blockchain from coin data."""
        # Simple heuristics based on coin name or symbol
        symbol = coin_data.get('symbol', '').upper()
        name = coin_data.get('name', '').lower()
        
        # Solana tokens
        if any(keyword in name for keyword in ['solana', 'sol', 'bonk', 'dogwifhat', 'wif']):
            return 'solana'
        
        # Base tokens
        if any(keyword in name for keyword in ['base', 'basechain']):
            return 'base'
        
        # Polygon tokens
        if any(keyword in name for keyword in ['polygon', 'matic']):
            return 'polygon'
        
        # BSC tokens
        if any(keyword in name for keyword in ['binance', 'bsc', 'bnb']):
            return 'bsc'
        
        # Default to Ethereum
        return 'ethereum'
    
    def _get_chain_id(self, blockchain: str) -> int:
        """Get chain ID for blockchain."""
        chain_ids = {
            'ethereum': 1,
            'base': 8453,
            'arbitrum': 42161,
            'polygon': 137,
            'bsc': 56,
            'solana': 101,
            'avalanche': 43114,
            'optimism': 10
        }
        return chain_ids.get(blockchain, 1)
    
    async def _clear_existing_tokens(self):
        """Clear existing tokens from database to replace with new ones."""
        try:
            # Delete all existing tokens using Supabase
            delete_tokens_response = self.db_service.db_manager.service_client.from_('discovered_tokens').delete().neq('id', '').execute()
            
            # Delete all existing analysis data
            delete_analysis_response = self.db_service.db_manager.service_client.from_('token_analysis').delete().neq('id', '').execute()
            
            logger.info("🗑️  Cleared existing tokens and analysis from database")
        except Exception as e:
            logger.error(f"Error clearing existing tokens: {e}")
    
    
    async def _store_discovered_token(self, token: DiscoveredToken) -> Optional[DiscoveredToken]:
        """Store a discovered token in the database."""
        try:
            # Insert token into database using Supabase
            token_data = {
                'id': token.id,
                'symbol': token.symbol,
                'name': token.name,
                'blockchain': token.blockchain,
                'chain_id': token.chain_id,
                'address': token.address,
                'logo_url': token.logo_url,
                'coingecko_id': token.coingecko_id,
                'discovery_date': token.discovery_date.isoformat(),
                'current_price': token.current_price,
                'market_cap': token.market_cap,
                'volume_24h': token.volume_24h,
                'price_change_24h': token.price_change_24h,
                'liquidity': token.liquidity,
                'last_updated': token.last_updated.isoformat(),
                'is_trending': token.is_trending,
                'is_active': token.is_active
            }
            
            response = self.db_service.db_manager.service_client.from_('discovered_tokens').insert(token_data).execute()
            
            if response.data:
                logger.debug(f"✅ Stored token {token.symbol}")
                return token
            else:
                logger.error(f"❌ Failed to store token {token.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error storing token {token.symbol}: {e}")
            return None
    
    async def cleanup_tokens_outside_criteria(
        self,
        min_market_cap: float = 1000000,
        max_market_cap: float = 500000000,
        min_volume: float = 1000000,  # Updated to $1M minimum daily volume
        min_quality_score: float = 50  # Minimum quality score of 50/100
    ) -> Dict[str, Any]:
        """Background cleanup task to deactivate/reactivate tokens based on criteria."""
        try:
            logger.info("🧹 Starting token cleanup based on market criteria")
            
            # Deactivate tokens outside criteria
            deactivated_count = await self.db_service.deactivate_tokens_outside_criteria(
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                min_volume=min_volume,
                min_quality_score=min_quality_score
            )
            
            # Reactivate tokens that now meet criteria
            reactivated_count = await self.db_service.reactivate_tokens_within_criteria(
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                min_volume=min_volume,
                min_quality_score=min_quality_score
            )
            
            result = {
                "status": "success",
                "deactivated_tokens": deactivated_count,
                "reactivated_tokens": reactivated_count,
                "criteria": {
                    "min_market_cap": min_market_cap,
                    "max_market_cap": max_market_cap,
                    "min_volume": min_volume
                },
                "cleanup_time": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Token cleanup complete: {deactivated_count} deactivated, {reactivated_count} reactivated")
            return result
            
        except Exception as e:
            logger.error(f"Error in token cleanup: {e}")
            return {"status": "error", "message": str(e)}


# Global service instance
_token_discovery_service: Optional[TokenDiscoveryService] = None


def get_token_discovery_service() -> TokenDiscoveryService:
    """Get or create token discovery service instance."""
    global _token_discovery_service
    
    if _token_discovery_service is None:
        _token_discovery_service = TokenDiscoveryService()
    
    return _token_discovery_service


def create_token_discovery_service() -> TokenDiscoveryService:
    """Create a new token discovery service instance."""
    return TokenDiscoveryService()