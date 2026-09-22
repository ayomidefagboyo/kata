"""
Token Discovery Database Service for Flow AI Trading Platform.

Handles database operations for token discovery, analysis storage, and market data.
"""

import logging
import json
from copy import deepcopy
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4

from kata.models.token_discovery import DiscoveredToken, TokenAnalysis, TokenMarketData
from kata.config.database import db_manager

logger = logging.getLogger(__name__)


class TokenDiscoveryDatabaseService:
    """Database service for token discovery operations."""
    
    def __init__(self):
        self.db_manager = db_manager
        # Initialize in-memory storage for mock functionality
        self.tokens_db = {}
        self.analysis_db = {}
        self.market_data_db = {}
        self.daily_analysis_db = {}
        self._preview_cache: Dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}
        self._preview_cache_ttl_seconds = 300
        logger.info("Token discovery database service initialized")
    
    async def get_trending_tokens(
        self,
        limit: int = 50,
        blockchain_filter: Optional[str] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume: Optional[float] = None,
        offset: int = 0,
        tier: str = 'quality'
    ) -> List[DiscoveredToken]:
        """Get trending tokens from database with optimized approach.

        tier='quality' returns curated CoinGecko-listed tokens, tier='moonshot'
        returns active early-stage DexScreener discoveries (no CoinGecko id),
        tier='past_moonshot' returns exited Moonshot history, tier='all'
        returns both active lanes.
        """
        try:
            logger.info(f"🔄 Getting trending tokens from database (limit: {limit}, offset: {offset}, tier: {tier})")
            is_past_moonshot = tier in {'past_moonshot', 'past-moonshot', 'past_moonshots'}

            if is_past_moonshot:
                return await self._get_past_moonshot_tokens(
                    limit=limit,
                    blockchain_filter=blockchain_filter,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    min_volume=min_volume,
                    offset=offset,
                )

            # Query tokens_with_analysis VIEW directly for best performance
            # This VIEW joins discovered_tokens with token_analysis automatically
            query_builder = self.db_manager.service_client.from_('tokens_with_analysis').select('''
                id, symbol, name, blockchain, chain_id, address, logo_url, coingecko_id,
                discovery_date, current_price, market_cap, volume_24h, price_change_24h,
                liquidity, bid_depth_2pct, ask_depth_2pct, last_updated, is_trending, is_active,
                analysis_id, analysis_type, risk_score, potential_score, confidence_score,
                overall_score, summary, signals, risk_factors, opportunities,
                technical_indicators, sentiment_score, analysis_version,
                analysis_created_at, analysis_updated_at
            ''')

            # Apply filters
            if blockchain_filter:
                query_builder = query_builder.eq('blockchain', blockchain_filter)
            if min_market_cap:
                query_builder = query_builder.gte('market_cap', int(min_market_cap))
            if max_market_cap:
                query_builder = query_builder.lte('market_cap', int(max_market_cap))
            if min_volume:
                query_builder = query_builder.gte('volume_24h', int(min_volume))

            # Active discovery lanes use active rows; past Moonshots intentionally
            # read exited inactive rows.
            query_builder = query_builder.eq('is_active', False if is_past_moonshot else True)

            # Tier separation: moonshots are DexScreener discoveries (no CoinGecko id).
            # They can come from any supported DexScreener chain; the API applies
            # the low-cap/volume bounds for the moonshot lane.
            if tier == 'quality':
                query_builder = query_builder.not_.is_('coingecko_id', 'null')
            elif tier == 'moonshot' or is_past_moonshot:
                query_builder = query_builder.is_('coingecko_id', 'null')

            # Order by opportunity indicators (best trading opportunities first).
            # Moonshots put trending/new DexScreener candidates first; quality
            # discovery remains volume/potential/mcap driven.
            if is_past_moonshot:
                query_builder = query_builder.order('last_updated', desc=True)
                query_builder = query_builder.order('discovery_date', desc=True)
            elif tier == 'moonshot':
                query_builder = query_builder.order('overall_score', desc=True)
                query_builder = query_builder.order('confidence_score', desc=True)
                query_builder = query_builder.order('is_trending', desc=True)
                query_builder = query_builder.order('volume_24h', desc=True)
                query_builder = query_builder.order('discovery_date', desc=True)
            else:
                query_builder = query_builder.order('volume_24h', desc=True)
                query_builder = query_builder.order('potential_score', desc=True)
                query_builder = query_builder.order('market_cap', desc=True)

            fetch_limit = limit
            if tier == 'moonshot':
                # Existing rows may predate mandatory AI review; overfetch so the
                # final page can still contain the best AI-approved moonshots only.
                fetch_limit = max(limit * 5, 50)
            elif is_past_moonshot:
                fetch_limit = max(limit * 10, 100)

            # Apply pagination
            response = query_builder.range(offset, offset + fetch_limit - 1).execute()

            if not response.data:
                logger.warning("No tokens found in tokens_with_analysis VIEW")
                return []

            tokens = []
            for row in response.data:
                if tier == 'moonshot' or is_past_moonshot:
                    indicators = row.get('technical_indicators') or {}
                    ai_reviewed = indicators.get('ai_reviewed') is True
                    ai_thesis = str(indicators.get('ai_thesis') or '').strip()
                    if not ai_reviewed or not ai_thesis:
                        continue
                    if tier == 'moonshot' and indicators.get('ai_include') is not True:
                        continue
                    lifecycle = indicators.get('moonshot_lifecycle')
                    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
                    if tier == 'moonshot' and lifecycle.get('status') == 'exited':
                        continue
                    if is_past_moonshot and lifecycle.get('status') != 'exited':
                        continue

                # Create analysis object if exists
                analysis = None
                if row.get('analysis_id'):
                    analysis = TokenAnalysis(
                        id=str(row['analysis_id']),
                        token_id=str(row['id']),
                        analysis_type=row.get('analysis_type', 'daily_discovery'),
                        risk_score=self._to_float(row.get('risk_score'), 5.0),
                        potential_score=self._to_float(row.get('potential_score'), 0.0),
                        confidence_score=self._to_float(row.get('confidence_score'), 0.0),
                        overall_score=self._to_float(row.get('overall_score'), 0.0),
                        summary=row.get('summary', ''),
                        signals=row.get('signals', []),
                        risk_factors=row.get('risk_factors', []),
                        opportunities=row.get('opportunities', []),
                        technical_indicators=row.get('technical_indicators', {}),
                        sentiment_score=self._to_float(row.get('sentiment_score'), 0.5),
                        analysis_version=row.get('analysis_version', '1.0'),
                        created_at=self._parse_datetime(row.get('analysis_created_at')),
                        updated_at=self._parse_datetime(row.get('analysis_updated_at'))
                    )

                # Create token object
                token = DiscoveredToken(
                    id=str(row['id']),
                    symbol=row['symbol'],
                    name=row['name'],
                    blockchain=row['blockchain'],
                    chain_id=row['chain_id'],
                    address=row['address'],
                    logo_url=row['logo_url'],
                    coingecko_id=row['coingecko_id'],
                    analysis=analysis,
                    discovery_date=self._parse_datetime(row['discovery_date']),
                    current_price=self._to_float(row.get('current_price')),
                    market_cap=self._to_float(row.get('market_cap')),
                    volume_24h=self._to_float(row.get('volume_24h')),
                    price_change_24h=self._to_float(row.get('price_change_24h')),
                    liquidity=self._to_float(row.get('liquidity')),
                    bid_depth_2pct=self._to_float(row.get('bid_depth_2pct'), 0.0),
                    ask_depth_2pct=self._to_float(row.get('ask_depth_2pct'), 0.0),
                    last_updated=self._parse_datetime(row['last_updated']),
                    is_trending=row['is_trending'],
                    is_active=row['is_active']
                )
                tokens.append(token)
                if (tier == 'moonshot' or is_past_moonshot) and len(tokens) >= limit:
                    break

            logger.info(f"✅ Retrieved {len(tokens)} trending tokens from tokens_with_analysis VIEW")
            return tokens

        except Exception as e:
            logger.error(f"Error getting trending tokens from database: {e}")
            return []

    @staticmethod
    def _past_moonshot_indicator_snapshots(indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return every completed episode stored for a Moonshot asset row."""
        snapshots: List[Dict[str, Any]] = []
        seen = set()

        archived = indicators.get('moonshot_history')
        candidates = archived if isinstance(archived, list) else []
        candidates = [snapshot for snapshot in candidates if isinstance(snapshot, dict)]

        current = indicators.get('moonshot_lifecycle')
        if isinstance(current, dict) and current.get('status') == 'exited':
            candidates = [*candidates, indicators]

        for candidate in candidates:
            lifecycle = candidate.get('moonshot_lifecycle')
            if not isinstance(lifecycle, dict) or lifecycle.get('status') != 'exited':
                continue
            episode_key = (lifecycle.get('entry_at'), lifecycle.get('exit_at'))
            if episode_key in seen:
                continue
            seen.add(episode_key)
            snapshots.append(deepcopy(candidate))

        return snapshots

    async def _get_past_moonshot_tokens(
        self,
        limit: int,
        blockchain_filter: Optional[str],
        min_market_cap: Optional[float],
        max_market_cap: Optional[float],
        min_volume: Optional[float],
        offset: int,
    ) -> List[DiscoveredToken]:
        """Read exited Moonshot episodes without relying on the active-only view.

        Re-entered contracts reuse their canonical discovered-token row because
        addresses are unique. Their earlier completed episodes live in the
        analysis JSON history and must remain visible while the row is active.
        """
        try:
            def build_token_query():
                query = self.db_manager.service_client.from_('discovered_tokens').select('''
                    id, symbol, name, blockchain, chain_id, address, logo_url, coingecko_id,
                    discovery_date, current_price, market_cap, volume_24h, price_change_24h,
                    liquidity, bid_depth_2pct, ask_depth_2pct, last_updated, is_trending, is_active
                ''')

                # Include active rows because a re-entered contract can have one
                # or more archived exits while its current episode remains active.
                query = query.is_('coingecko_id', 'null')
                if blockchain_filter:
                    query = query.eq('blockchain', blockchain_filter)
                if min_market_cap:
                    query = query.gte('market_cap', int(min_market_cap))
                if max_market_cap:
                    query = query.lte('market_cap', int(max_market_cap))
                if min_volume:
                    query = query.gte('volume_24h', int(min_volume))
                return query.order('last_updated', desc=True).order('discovery_date', desc=True)

            # Episode counts and asset-row counts are different after re-entry,
            # so page through every matching canonical row before sorting/slicing
            # completed episodes. This keeps offset pagination globally correct.
            token_rows: List[Dict[str, Any]] = []
            page_size = 200
            page_start = 0
            while True:
                response = (
                    build_token_query()
                    .range(page_start, page_start + page_size - 1)
                    .execute()
                )
                page = response.data or []
                token_rows.extend(page)
                if len(page) < page_size:
                    break
                page_start += page_size

            if not token_rows:
                return []

            token_ids = [str(row['id']) for row in token_rows if row.get('id')]
            analysis_by_token: Dict[str, Dict[str, Any]] = {}
            analysis_chunk_size = 50
            for chunk_start in range(0, len(token_ids), analysis_chunk_size):
                token_id_chunk = token_ids[chunk_start:chunk_start + analysis_chunk_size]
                analysis_response = (
                    self.db_manager.service_client.from_('token_analysis')
                    .select('''
                        id, token_id, analysis_type, risk_score, potential_score,
                        confidence_score, overall_score, summary, signals, risk_factors,
                        opportunities, technical_indicators, sentiment_score,
                        analysis_version, created_at, updated_at
                    ''')
                    .in_('token_id', token_id_chunk)
                    .order('created_at', desc=True)
                    .execute()
                )
                for analysis_row in analysis_response.data or []:
                    token_id = str(analysis_row.get('token_id'))
                    if token_id and token_id not in analysis_by_token:
                        analysis_by_token[token_id] = analysis_row

            episodes: List[Tuple[float, DiscoveredToken]] = []
            for row in token_rows:
                analysis_row = analysis_by_token.get(str(row['id']))
                if not analysis_row:
                    continue

                indicators = analysis_row.get('technical_indicators') or {}
                if isinstance(indicators, str):
                    try:
                        indicators = json.loads(indicators)
                    except json.JSONDecodeError:
                        indicators = {}
                if not isinstance(indicators, dict):
                    indicators = {}

                for episode_index, episode_indicators in enumerate(
                    self._past_moonshot_indicator_snapshots(indicators)
                ):
                    ai_reviewed = episode_indicators.get('ai_reviewed') is True
                    ai_thesis = str(episode_indicators.get('ai_thesis') or '').strip()
                    lifecycle = episode_indicators.get('moonshot_lifecycle') or {}
                    if not ai_reviewed or not ai_thesis:
                        continue

                    virtual_token_id = f"{row['id']}:past:{episode_index}"
                    virtual_analysis_id = f"{analysis_row['id']}:past:{episode_index}"
                    entry_at = self._parse_datetime(
                        lifecycle.get('entry_at') or row.get('discovery_date')
                    )
                    exit_at = self._parse_datetime(
                        lifecycle.get('exit_at') or row.get('last_updated')
                    )

                    analysis = TokenAnalysis(
                        id=virtual_analysis_id,
                        token_id=virtual_token_id,
                        analysis_type=analysis_row.get('analysis_type', 'daily_discovery'),
                        risk_score=float(analysis_row.get('risk_score', 5.0)),
                        potential_score=float(analysis_row.get('potential_score', 0.0)),
                        confidence_score=float(analysis_row.get('confidence_score', 0.0)),
                        overall_score=float(analysis_row.get('overall_score', 0.0)),
                        summary=analysis_row.get('summary', ''),
                        signals=analysis_row.get('signals', []),
                        risk_factors=analysis_row.get('risk_factors', []),
                        opportunities=analysis_row.get('opportunities', []),
                        technical_indicators=episode_indicators,
                        sentiment_score=float(analysis_row.get('sentiment_score', 0.5)),
                        analysis_version=analysis_row.get('analysis_version', '1.0'),
                        created_at=entry_at,
                        updated_at=exit_at,
                    )

                    token = DiscoveredToken(
                        id=virtual_token_id,
                        symbol=row['symbol'],
                        name=row['name'],
                        blockchain=row['blockchain'],
                        chain_id=row['chain_id'],
                        address=row['address'],
                        logo_url=row.get('logo_url'),
                        coingecko_id=row.get('coingecko_id'),
                        analysis=analysis,
                        discovery_date=entry_at,
                        current_price=self._to_float(
                            lifecycle.get('exit_price'),
                            self._to_float(row.get('current_price')),
                        ),
                        market_cap=self._to_float(row.get('market_cap')),
                        volume_24h=self._to_float(row.get('volume_24h')),
                        price_change_24h=self._to_float(row.get('price_change_24h')),
                        liquidity=self._to_float(row.get('liquidity')),
                        bid_depth_2pct=self._to_float(row.get('bid_depth_2pct')),
                        ask_depth_2pct=self._to_float(row.get('ask_depth_2pct')),
                        last_updated=exit_at,
                        is_trending=row.get('is_trending', False),
                        is_active=False,
                    )
                    episodes.append((exit_at.timestamp(), token))

            episodes.sort(key=lambda item: item[0], reverse=True)
            tokens = [token for _, token in episodes[offset:offset + limit]]
            logger.info(f"✅ Retrieved {len(tokens)} past Moonshot episodes from base tables")
            return tokens

        except Exception as e:
            logger.error(f"Error getting past Moonshot tokens from base tables: {e}")
            return []

    def _preview_cache_valid(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        cached = self._preview_cache.get(cache_key)
        if not cached:
            return None

        cached_at, tokens = cached
        if (datetime.now() - cached_at).total_seconds() > self._preview_cache_ttl_seconds:
            return None

        return [dict(token) for token in tokens]

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    async def get_dashboard_preview_tokens(
        self,
        limit: int = 12,
        min_market_cap: Optional[float] = 1000000,
        max_market_cap: Optional[float] = 500000000,
        min_volume: Optional[float] = 200000,
    ) -> List[Dict[str, Any]]:
        """Return a small, flattened token set for the dashboard discovery card."""
        safe_limit = max(1, min(int(limit or 12), 50))
        cache_key = f"{safe_limit}:{min_market_cap}:{max_market_cap}:{min_volume}"
        cached_tokens = self._preview_cache_valid(cache_key)
        if cached_tokens is not None:
            return cached_tokens

        try:
            query_builder = self.db_manager.service_client.from_('tokens_with_analysis').select('''
                id, symbol, name, blockchain, chain_id, address, logo_url, coingecko_id,
                current_price, market_cap, volume_24h, price_change_24h, liquidity,
                last_updated, is_trending, risk_score, potential_score, confidence_score,
                overall_score
            ''')

            query_builder = query_builder.eq('is_active', True)
            # Dashboard preview stays quality-tier only (moonshots have no CoinGecko id)
            query_builder = query_builder.not_.is_('coingecko_id', 'null')
            if min_market_cap:
                query_builder = query_builder.gte('market_cap', int(min_market_cap))
            if max_market_cap:
                query_builder = query_builder.lte('market_cap', int(max_market_cap))
            if min_volume:
                query_builder = query_builder.gte('volume_24h', int(min_volume))

            query_builder = query_builder.order('volume_24h', desc=True)
            query_builder = query_builder.order('potential_score', desc=True)
            query_builder = query_builder.order('market_cap', desc=True)

            response = query_builder.range(0, safe_limit - 1).execute()
            rows = response.data or []

            tokens: List[Dict[str, Any]] = []
            for row in rows:
                risk_score = self._to_float(row.get('risk_score'), 5.0)
                potential_score = self._to_float(row.get('potential_score'), 5.0)
                confidence_score = self._to_float(row.get('confidence_score'), 0.5)
                overall_score = self._to_float(row.get('overall_score'), potential_score)
                quality_score = overall_score * 10
                normalized_risk = risk_score * 10

                tokens.append({
                    'id': str(row.get('id')),
                    'symbol': row.get('symbol'),
                    'name': row.get('name'),
                    'blockchain': row.get('blockchain'),
                    'chainId': row.get('chain_id'),
                    'address': row.get('address'),
                    'price': self._to_float(row.get('current_price')),
                    'priceChange24h': self._to_float(row.get('price_change_24h')),
                    'marketCap': self._to_float(row.get('market_cap')),
                    'volume24h': self._to_float(row.get('volume_24h')),
                    'liquidity': self._to_float(row.get('liquidity')),
                    'logoUrl': row.get('logo_url') or '',
                    'coingeckoId': row.get('coingecko_id') or '',
                    'tradingScore': f"{quality_score:.1f}",
                    'securityScore': max(10, min(100, 100 - normalized_risk)),
                    'qualityScore': quality_score,
                    'riskScore': normalized_risk,
                    'potentialScore': potential_score * 10,
                    'confidenceScore': confidence_score * 10,
                    'trending': bool(row.get('is_trending')),
                    'lastUpdated': row.get('last_updated'),
                })

            self._preview_cache[cache_key] = (datetime.now(), [dict(token) for token in tokens])
            return tokens

        except Exception as e:
            logger.error(f"Error getting dashboard preview tokens: {e}")
            return []
    
    async def search_tokens(self, query: str, limit: int = 20, offset: int = 0) -> List[DiscoveredToken]:
        """Search tokens by symbol or name using tokens_with_analysis VIEW."""
        try:
            logger.info(f"🔍 Searching tokens for query: '{query}' (limit: {limit}, offset: {offset})")

            # Search using tokens_with_analysis VIEW directly
            search_pattern = f"%{query.lower()}%"

            query_builder = self.db_manager.service_client.from_('tokens_with_analysis').select('''
                id, symbol, name, blockchain, chain_id, address, logo_url, coingecko_id,
                discovery_date, current_price, market_cap, volume_24h, price_change_24h,
                liquidity, bid_depth_2pct, ask_depth_2pct, last_updated, is_trending, is_active,
                analysis_id, analysis_type, risk_score, potential_score, confidence_score,
                overall_score, summary, signals, risk_factors, opportunities,
                technical_indicators, sentiment_score, analysis_version,
                analysis_created_at, analysis_updated_at
            ''')

            # Add search filters
            query_builder = query_builder.eq('is_active', True)
            query_builder = query_builder.or_(f'symbol.ilike.{search_pattern},name.ilike.{search_pattern}')

            # Order by opportunity indicators (best trading opportunities first)
            query_builder = query_builder.order('volume_24h', desc=True)  # High volume activity
            query_builder = query_builder.order('potential_score', desc=True)
            query_builder = query_builder.order('market_cap', desc=True)

            # Apply pagination
            response = query_builder.range(offset, offset + limit - 1).execute()

            if not response.data:
                logger.info(f"No tokens found for search query: '{query}'")
                return []

            tokens = []
            for row in response.data:
                # Create analysis object if exists
                analysis = None
                if row.get('analysis_id'):
                    analysis = TokenAnalysis(
                        id=str(row['analysis_id']),
                        token_id=str(row['id']),
                        analysis_type=row.get('analysis_type', 'daily_discovery'),
                        risk_score=self._to_float(row.get('risk_score'), 5.0),
                        potential_score=self._to_float(row.get('potential_score'), 0.0),
                        confidence_score=self._to_float(row.get('confidence_score'), 0.0),
                        overall_score=self._to_float(row.get('overall_score'), 0.0),
                        summary=row.get('summary', ''),
                        signals=row.get('signals', []),
                        risk_factors=row.get('risk_factors', []),
                        opportunities=row.get('opportunities', []),
                        technical_indicators=row.get('technical_indicators', {}),
                        sentiment_score=self._to_float(row.get('sentiment_score'), 0.5),
                        analysis_version=row.get('analysis_version', '1.0'),
                        created_at=self._parse_datetime(row.get('analysis_created_at')),
                        updated_at=self._parse_datetime(row.get('analysis_updated_at'))
                    )

                # Create token object
                token = DiscoveredToken(
                    id=str(row['id']),
                    symbol=row['symbol'],
                    name=row['name'],
                    blockchain=row['blockchain'],
                    chain_id=row['chain_id'],
                    address=row['address'],
                    logo_url=row['logo_url'],
                    coingecko_id=row['coingecko_id'],
                    analysis=analysis,
                    discovery_date=self._parse_datetime(row['discovery_date']),
                    current_price=self._to_float(row.get('current_price')),
                    market_cap=self._to_float(row.get('market_cap')),
                    volume_24h=self._to_float(row.get('volume_24h')),
                    price_change_24h=self._to_float(row.get('price_change_24h')),
                    liquidity=self._to_float(row.get('liquidity')),
                    bid_depth_2pct=self._to_float(row.get('bid_depth_2pct'), 0.0),
                    ask_depth_2pct=self._to_float(row.get('ask_depth_2pct'), 0.0),
                    last_updated=self._parse_datetime(row['last_updated']),
                    is_trending=row['is_trending'],
                    is_active=row['is_active']
                )
                tokens.append(token)

            logger.info(f"✅ Found {len(tokens)} tokens using tokens_with_analysis VIEW search")
            return tokens

        except Exception as e:
            logger.error(f"Error searching tokens: {e}")
            return []

    
    async def get_token_by_id(self, token_id: str) -> Optional[DiscoveredToken]:
        """Get token by ID."""
        try:
            # Mock implementation
            if token_id in self.tokens_db:
                return self.tokens_db[token_id]
            
            # Return None if not found
            return None
            
        except Exception as e:
            logger.error(f"Error getting token by ID: {e}")
            return None
    
    async def get_token_analysis(self, token_id: str) -> Optional[TokenAnalysis]:
        """Get latest analysis for a token."""
        try:
            # Mock implementation
            if token_id in self.analysis_db:
                return self.analysis_db[token_id]
            
            # Return None if no analysis exists
            return None
            
        except Exception as e:
            logger.error(f"Error getting token analysis: {e}")
            return None
    
    async def store_token_analysis(self, token_id: str, analysis_data: Dict[str, Any]) -> str:
        """Store analysis results for a token."""
        try:
            analysis_id = str(uuid4())

            # Use Supabase client for insertion
            insert_data = {
                "id": analysis_id,
                "token_id": token_id,
                "analysis_type": analysis_data.get("analysis_type", "daily_discovery"),
                "risk_score": analysis_data["risk_score"],
                "potential_score": analysis_data["potential_score"],
                "confidence_score": analysis_data["confidence_score"],
                "overall_score": analysis_data["overall_score"],
                "summary": analysis_data["summary"],
                "signals": analysis_data.get("signals", []),
                "risk_factors": analysis_data.get("risk_factors", []),
                "opportunities": analysis_data.get("opportunities", []),
                "technical_indicators": analysis_data.get("technical_indicators", {}),
                "sentiment_score": analysis_data.get("sentiment_score", 0.5),
                "analysis_version": analysis_data.get("analysis_version", "1.0")
            }

            response = self.db_manager.service_client.from_('token_analysis').insert(insert_data).execute()

            if response.data:
                logger.info(f"Stored analysis {analysis_id} for token {token_id}")
                return analysis_id
            else:
                raise Exception("Failed to insert analysis data")

        except Exception as e:
            logger.error(f"Error storing token analysis: {e}")
            raise
    
    async def update_token_market_data(self, token_id: str, market_data: Dict[str, Any]) -> bool:
        """Update real-time market data for a token."""
        try:
            # Use Supabase client for update
            update_data = {
                "current_price": float(market_data.get("price", 0)),
                "market_cap": int(market_data.get("market_cap", 0)) if market_data.get("market_cap") else 0,
                "volume_24h": int(market_data.get("volume_24h", 0)) if market_data.get("volume_24h") else 0,
                "price_change_24h": float(market_data.get("price_change_24h", 0)),
                "liquidity": int(market_data.get("liquidity", 0)) if market_data.get("liquidity") else 0,
                "last_updated": "now()"
            }

            response = self.db_manager.service_client.from_('discovered_tokens').update(update_data).eq('id', token_id).execute()

            if response.data:
                logger.debug(f"Updated market data for token {token_id}")
                return True
            else:
                logger.warning(f"No token found with id {token_id} for market data update")
                return False

        except Exception as e:
            logger.error(f"Error updating token market data: {e}")
            return False
    
    async def deactivate_tokens_outside_criteria(
        self, 
        min_market_cap: float = 1000000, 
        max_market_cap: float = 500000000,
        min_volume: float = 1000000,  # Updated to $1M minimum daily volume
        min_quality_score: float = 50  # Minimum quality score of 50/100
    ) -> int:
        """Deactivate tokens that no longer meet our criteria."""
        try:
            # For market cap and volume criteria, use direct token table queries
            deactivated_count = 0
            
            # 1. Deactivate tokens with market cap outside range
            if min_market_cap or max_market_cap:
                market_cap_query = self.db_manager.service_client.from_('discovered_tokens').update({'is_active': False}).eq('is_active', True)
                
                if min_market_cap:
                    market_cap_query = market_cap_query.lt('market_cap', int(min_market_cap))
                if max_market_cap:
                    market_cap_query = market_cap_query.gt('market_cap', int(max_market_cap))
                
                market_cap_response = market_cap_query.execute()
                market_cap_deactivated = len(market_cap_response.data) if market_cap_response.data else 0
                deactivated_count += market_cap_deactivated
                
                if market_cap_deactivated > 0:
                    logger.info(f"🧹 Deactivated {market_cap_deactivated} tokens for market cap criteria")
            
            # 2. Deactivate tokens with volume below minimum
            if min_volume:
                volume_response = (
                    self.db_manager.service_client
                    .from_('discovered_tokens')
                    .update({'is_active': False})
                    .eq('is_active', True)
                    .lt('volume_24h', int(min_volume))
                    .execute()
                )
                volume_deactivated = len(volume_response.data) if volume_response.data else 0
                deactivated_count += volume_deactivated
                
                if volume_deactivated > 0:
                    logger.info(f"🧹 Deactivated {volume_deactivated} tokens for volume criteria")
            
            # 3. For quality score, use a simpler approach
            # Get active tokens and check their analysis scores in memory
            if min_quality_score:
                try:
                    # Get all active tokens first
                    active_tokens = await self.get_trending_tokens(limit=1000)
                    
                    tokens_to_deactivate = []
                    for token in active_tokens:
                        if token.analysis and hasattr(token.analysis, 'overall_score'):
                            if token.analysis.overall_score < min_quality_score:
                                tokens_to_deactivate.append(token.id)
                        elif not token.analysis:
                            # Tokens without analysis should also be deactivated for quality
                            tokens_to_deactivate.append(token.id)
                    
                    if tokens_to_deactivate:
                        # Deactivate tokens with low quality scores in batches
                        for i in range(0, len(tokens_to_deactivate), 50):  # Process 50 at a time
                            batch = tokens_to_deactivate[i:i+50]
                            quality_response = (
                                self.db_manager.service_client
                                .from_('discovered_tokens')
                                .update({'is_active': False})
                                .in_('id', batch)
                                .execute()
                            )
                            if quality_response.data:
                                quality_deactivated = len(quality_response.data)
                                deactivated_count += quality_deactivated
                        
                        if len(tokens_to_deactivate) > 0:
                            logger.info(f"🧹 Deactivated {len(tokens_to_deactivate)} tokens for quality score criteria")
                            
                except Exception as quality_error:
                    logger.warning(f"Error checking quality scores: {quality_error}")
            
            if deactivated_count > 0:
                logger.info(f"🧹 Deactivated {deactivated_count} tokens outside criteria")
            else:
                logger.debug("No tokens needed deactivation")
                
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error deactivating tokens outside criteria: {e}")
            return 0
    
    async def reactivate_tokens_within_criteria(
        self, 
        min_market_cap: float = 1000000, 
        max_market_cap: float = 500000000,
        min_volume: float = 1000000,  # Updated to $1M minimum daily volume
        min_quality_score: float = 50  # Minimum quality score of 50/100
    ) -> int:
        """Reactivate tokens that now meet our criteria."""
        try:
            # For basic criteria (market cap and volume), use direct queries
            reactivated_count = 0
            
            # Reactivate tokens that meet market cap and volume criteria
            response = (
                self.db_manager.service_client
                .from_('discovered_tokens')
                .update({'is_active': True})
                .eq('is_active', False)  # Only update currently inactive tokens
                .gte('market_cap', int(min_market_cap))
                .lte('market_cap', int(max_market_cap))
                .gte('volume_24h', int(min_volume))
                .execute()
            )
            
            basic_reactivated = len(response.data) if response.data else 0
            reactivated_count += basic_reactivated
            
            if basic_reactivated > 0:
                logger.info(f"♻️ Reactivated {basic_reactivated} tokens for basic criteria")
            
            # For quality score, we'll need to check individually
            # For now, we'll skip quality-based reactivation to keep it simple
            # Quality will be checked during the next analysis cycle
            
            if reactivated_count > 0:
                logger.info(f"♻️ Reactivated {reactivated_count} tokens now within criteria")
            else:
                logger.debug("No tokens needed reactivation")
                
            return reactivated_count
            
        except Exception as e:
            logger.error(f"Error reactivating tokens within criteria: {e}")
            return 0
    
    async def store_market_data(self, token_id: str, market_data: Dict[str, Any]) -> bool:
        """Store historical market data."""
        try:
            # Mock implementation
            if token_id not in self.market_data_db:
                self.market_data_db[token_id] = []
            
            market_data_obj = TokenMarketData(
                token_id=token_id,
                source=market_data.get("source", "api"),
                current_price=market_data["price"],
                price_change_24h=market_data.get("price_change_24h"),
                volume_24h=market_data["volume_24h"],
                market_cap=market_data["market_cap"],
                liquidity=market_data.get("liquidity", 0),
                trading_pairs_count=market_data.get("trading_pairs_count", 0),
                top_exchanges=market_data.get("top_exchanges", []),
                fetched_at=datetime.now()
            )
            
            self.market_data_db[token_id].append(market_data_obj)
            
            # Keep only last 100 records per token
            if len(self.market_data_db[token_id]) > 100:
                self.market_data_db[token_id] = self.market_data_db[token_id][-100:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    async def get_active_tokens(self) -> List[DiscoveredToken]:
        """Get all active tokens for analysis."""
        try:
            # Get real tokens from database
            response = self.db_manager.service_client.from_('discovered_tokens').select('*').eq('is_active', True).execute()
            
            if response.data is None:
                return []
                
            tokens = []
            for row in response.data:
                analysis = await self._get_latest_analysis(str(row['id']))
                token = DiscoveredToken(
                    id=str(row['id']),
                    symbol=row['symbol'],
                    name=row['name'],
                    blockchain=row['blockchain'],
                    chain_id=row['chain_id'],
                    address=row['address'],
                    logo_url=row['logo_url'],
                    coingecko_id=row['coingecko_id'],
                    analysis=analysis,
                    discovery_date=self._parse_datetime(row['discovery_date']),
                    current_price=self._to_float(row.get('current_price')),
                    market_cap=self._to_float(row.get('market_cap')),
                    volume_24h=self._to_float(row.get('volume_24h')),
                    price_change_24h=self._to_float(row.get('price_change_24h')),
                    liquidity=self._to_float(row.get('liquidity')),
                    bid_depth_2pct=self._to_float(row.get('bid_depth_2pct'), 0.0),
                    ask_depth_2pct=self._to_float(row.get('ask_depth_2pct'), 0.0),
                    last_updated=self._parse_datetime(row['last_updated']),
                    is_trending=row['is_trending'],
                    is_active=row['is_active']
                )
                tokens.append(token)
            
            return tokens

        except Exception as e:
            logger.error(f"Error getting active tokens: {e}")
            return []

    async def get_stale_tokens_batch(self, cutoff_time: datetime) -> List[DiscoveredToken]:
        """Get active tokens that need market data updates (batch optimized)."""
        try:
            # Single query to get stale tokens with their latest analysis
            response = self.db_manager.service_client.from_('tokens_with_analysis').select('''
                id, symbol, name, blockchain, chain_id, address, logo_url, coingecko_id,
                discovery_date, current_price, market_cap, volume_24h, price_change_24h,
                liquidity, bid_depth_2pct, ask_depth_2pct, last_updated, is_trending, is_active,
                analysis_id, analysis_type, risk_score, potential_score, confidence_score
            ''').eq('is_active', True).lt('last_updated', cutoff_time.isoformat()).execute()

            if not response.data:
                return []

            tokens = []
            for row in response.data:
                # Create analysis object from view data (no additional query needed)
                analysis = None
                if row.get('analysis_id'):
                    analysis = TokenAnalysis(
                        id=str(row['analysis_id']),
                        token_id=str(row['id']),
                        analysis_type=row.get('analysis_type', 'discovery'),
                        risk_score=self._to_float(row.get('risk_score'), 5.0),
                        potential_score=self._to_float(row.get('potential_score'), 5.0),
                        confidence_score=self._to_float(row.get('confidence_score'), 0.5),
                        overall_score=self._to_float(row.get('potential_score'), 5.0),
                        summary="Batch market data update",
                        signals=["market_update"],
                        risk_factors=["market_volatility"],
                        opportunities=["market_position"],
                        technical_indicators={},
                        social_metrics={},
                        sentiment_score=0.7,
                        analysis_version="1.0",
                        created_at=self._parse_datetime(row.get('analysis_created_at')) or datetime.now(),
                        updated_at=self._parse_datetime(row.get('analysis_updated_at')) or datetime.now()
                    )

                token = DiscoveredToken(
                    id=str(row['id']),
                    symbol=row['symbol'],
                    name=row['name'],
                    blockchain=row['blockchain'],
                    chain_id=row['chain_id'],
                    address=row['address'],
                    logo_url=row['logo_url'],
                    coingecko_id=row['coingecko_id'],
                    analysis=analysis,
                    discovery_date=self._parse_datetime(row['discovery_date']),
                    current_price=self._to_float(row.get('current_price')),
                    market_cap=self._to_float(row.get('market_cap')),
                    volume_24h=self._to_float(row.get('volume_24h')),
                    price_change_24h=self._to_float(row.get('price_change_24h')),
                    liquidity=self._to_float(row.get('liquidity')),
                    bid_depth_2pct=self._to_float(row.get('bid_depth_2pct'), 0.0),
                    ask_depth_2pct=self._to_float(row.get('ask_depth_2pct'), 0.0),
                    last_updated=self._parse_datetime(row['last_updated']),
                    is_trending=row['is_trending'],
                    is_active=row['is_active']
                )
                tokens.append(token)

            logger.info(f"🔍 Found {len(tokens)} stale tokens needing updates")
            return tokens

        except Exception as e:
            logger.error(f"Error getting stale tokens: {e}")
            return []

    async def update_tokens_market_data_batch(self, batch_updates: List[Dict]) -> int:
        """Update market data for multiple tokens in a single operation."""
        try:
            success_count = 0

            # Prepare batch update data
            update_records = []
            for update in batch_updates:
                token_id = update['token_id']
                market_data = update['market_data']

                update_record = {
                    'id': token_id,
                    'current_price': market_data.get('current_price', 0),
                    'market_cap': market_data.get('market_cap', 0),
                    'volume_24h': market_data.get('volume_24h', 0),
                    'price_change_24h': market_data.get('price_change_24h', 0),
                    'last_updated': datetime.now().isoformat()
                }
                update_records.append(update_record)

            # Execute updates per token ID to avoid violating symbol NOT NULL constraint on upsert
            if update_records:
                for rec in update_records:
                    rec_id = rec.pop('id')
                    try:
                        resp = self.db_manager.service_client.from_('discovered_tokens').update(rec).eq('id', rec_id).execute()
                        if resp.data:
                            success_count += len(resp.data)
                    except Exception as err:
                        logger.warning(f"Failed to update market data for token {rec_id}: {err}")

                if success_count > 0:
                    logger.info(f"📊 Batch updated {success_count} token market data records")

            return success_count

        except Exception as e:
            logger.error(f"Error in batch market data update: {e}")
            return 0
    
    async def store_daily_analysis_result(self, result_data: Dict[str, Any]) -> str:
        """Store daily analysis batch results."""
        try:
            result_id = str(uuid4())
            
            # Mock implementation
            self.daily_analysis_db[result_id] = {
                "id": result_id,
                "analysis_date": result_data["analysis_date"],
                "tokens_analyzed": result_data["tokens_analyzed"],
                "new_discoveries": result_data["new_discoveries"],
                "analysis_duration": result_data["analysis_duration"],
                "status": result_data["status"],
                "created_at": datetime.now()
            }
            
            logger.info(f"Stored daily analysis result {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Error storing daily analysis result: {e}")
            raise
    
    async def _get_latest_analysis(self, token_id: str) -> Optional[TokenAnalysis]:
        """Get latest analysis for a token."""
        try:
            response = self.db_manager.service_client.from_('token_analysis').select('*').eq('token_id', token_id).order('created_at', desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                row = response.data[0]
                return TokenAnalysis(
                    id=str(row['id']),
                    token_id=str(row['token_id']),
                    analysis_type=row['analysis_type'],
                    risk_score=self._to_float(row.get('risk_score')),
                    potential_score=self._to_float(row.get('potential_score')),
                    confidence_score=self._to_float(row.get('confidence_score')),
                    overall_score=self._to_float(row.get('overall_score')),
                    summary=row['summary'],
                    signals=row['signals'] or [],
                    risk_factors=row['risk_factors'] or [],
                    opportunities=row['opportunities'] or [],
                    technical_indicators=row['technical_indicators'] or {},
                    sentiment_score=self._to_float(row.get('sentiment_score'), 0.5),
                    analysis_version=row['analysis_version'] or "1.0",
                    created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')) if isinstance(row['created_at'], str) else row['created_at'],
                    updated_at=datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00')) if isinstance(row['updated_at'], str) else row['updated_at']
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"No analysis found for token {token_id}: {e}")
            return None
    

    # Private helper methods



    def _parse_datetime(self, dt_string):
        """Parse datetime string with proper handling of various formats."""
        try:
            if isinstance(dt_string, datetime):
                return dt_string
            if isinstance(dt_string, str):
                # Handle different datetime formats
                if dt_string.endswith('Z'):
                    dt_string = dt_string.replace('Z', '+00:00')
                if '.' in dt_string:
                    head, tail = dt_string.split('.', 1)
                    timezone_part = ''
                    fractional = tail
                    if '+' in tail:
                        fractional, timezone_part = tail.split('+', 1)
                        timezone_part = f'+{timezone_part}'
                    elif '-' in tail:
                        fractional, timezone_part = tail.split('-', 1)
                        timezone_part = f'-{timezone_part}'

                    if fractional.isdigit() and len(fractional) != 6:
                        fractional = fractional[:6].ljust(6, '0')
                        dt_string = f"{head}.{fractional}{timezone_part}"
                
                # Try parsing with fromisoformat
                try:
                    return datetime.fromisoformat(dt_string)
                except ValueError:
                    # If that fails, try parsing the microseconds manually
                    if '.' in dt_string and len(dt_string.split('.')[-1].split('+')[0].split('-')[0]) > 6:
                        # Truncate microseconds to 6 digits
                        parts = dt_string.split('.')
                        microseconds = parts[1].split('+')[0].split('-')[0][:6]
                        if '+' in parts[1]:
                            dt_string = f"{parts[0]}.{microseconds}+{parts[1].split('+')[1]}"
                        elif '-' in parts[1]:
                            dt_string = f"{parts[0]}.{microseconds}-{parts[1].split('-')[1]}"
                        else:
                            dt_string = f"{parts[0]}.{microseconds}"
                    
                    return datetime.fromisoformat(dt_string)
            
            return datetime.now()  # Fallback
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{dt_string}': {e}")
            return datetime.now()


# Global service instance
_token_discovery_db_service: Optional[TokenDiscoveryDatabaseService] = None


def get_token_discovery_db_service() -> TokenDiscoveryDatabaseService:
    """Get or create token discovery database service instance."""
    global _token_discovery_db_service
    
    if _token_discovery_db_service is None:
        _token_discovery_db_service = TokenDiscoveryDatabaseService()
    
    return _token_discovery_db_service


def create_token_discovery_db_service() -> TokenDiscoveryDatabaseService:
    """Create a new token discovery database service instance."""
    return TokenDiscoveryDatabaseService()
