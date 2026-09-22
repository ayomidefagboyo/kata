"""
Ryu Spot Trading Service

Executes Ryu's live spot-buy / spot-exit decisions on-chain. Ryu is a
spot-only specialist: unlike Yuki (leveraged Hyperliquid futures), live Ryu
never opens a leveraged or short position -- even though the token-analysis
card can surface LONG_ENTRY/SHORT_ENTRY as informational trade plans for
manual traders, this service only ever acts on SPOT_BUY (open) and
SPOT_EXIT (close) intents.

Pipeline per worker cycle, shared across every Ryu-funded user (LLM
analysis is expensive -- run once per cycle, not once per user):

  1. Source candidates from the platform's own `tokens_with_analysis` DB
     (the same table that powers the Token Discovery card).  No separate
     CoinGecko scrape -- the platform already does that daily.
  2. Universe filter: keep only tokens Ryu can safely buy and later sell via
     LI.FI (must have a Base or Solana contract address).
  3. Recency filter: skip tokens whose Ryu LLM analysis is less than
     RYU_ANALYSIS_CACHE_TTL seconds old (default 6 h).  Cached results
     are reused for execution decisions without an extra LLM call.
  4. LLM analysis (token_analysis card pipeline) on the remaining
     candidates, capped at RYU_MAX_CANDIDATES_PER_CYCLE (default 6).
  5. For each active Ryu allocation: check open positions for SPOT_EXIT,
     check candidates for SPOT_BUY, validate liquidity, execute via LI.FI.

Gated behind settings.RYU_LIVE_TRADING_ENABLED (default True).
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from web3 import Web3

from kata.config.database import get_service_client
from kata.config.settings import settings
from kata.services.enhanced_token_discovery_service import (
    DiscoveredToken as CoinGeckoToken,
    get_enhanced_token_discovery_service,
)
from kata.services.lifi_bridge_service import (
    LiFiSwapQuote,
    SOLANA_CHAIN_ID,
    SOLANA_USDC_ADDRESS,
    get_lifi_service,
)
from kata.services.privy_signing_service import get_privy_signing_service

logger = logging.getLogger(__name__)

# Floww deposits and idle Ryu capital remain Base USDC. A cross-chain buy moves
# only that position's amount to the destination token and can carve out a
# user-owned destination gas reserve through LI.Fuel.
BASE_CHAIN_ID = 8453
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
BASE_USDC_DECIMALS = 6
BASE_RPC_URL_BASE = "https://base-mainnet.g.alchemy.com/v2/"
SOLANA_ANALYSIS_CHAIN_ID = 101
SOLANA_USDC_DECIMALS = 6
SOLANA_NATIVE_ADDRESS = "11111111111111111111111111111111"
SUPPORTED_RYU_CHAINS = {
    "base": {
        "lifi_chain_id": BASE_CHAIN_ID,
        "analysis_chain_id": BASE_CHAIN_ID,
        "chain_type": "ethereum",
    },
    "solana": {
        "lifi_chain_id": SOLANA_CHAIN_ID,
        "analysis_chain_id": SOLANA_ANALYSIS_CHAIN_ID,
        "chain_type": "solana",
    },
}

ERC20_ALLOWANCE_APPROVE_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
]

_ENCODER = Web3()


@dataclass
class RyuCandidateAnalysis:
    """A platform-discovered token (plain dict from tokens_with_analysis DB) paired
    with its LLM-authoritative analysis for this Ryu cycle."""
    token: Any  # platform DB row dict, not a CoinGeckoToken object
    trade_intent: str
    confidence: float
    action_summary: str
    reasoning: str
    position_size_text: str
    chain_name: str = "solana"
    chain_id: int = SOLANA_CHAIN_ID
    token_address: str = ""
    opportunity_score: float = 0.0
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None

    @property
    def symbol(self) -> str:
        if isinstance(self.token, dict):
            return str(self.token.get("symbol") or "").upper()
        return str(getattr(self.token, "symbol", "") or "").upper()

    @property
    def current_price(self) -> float:
        if isinstance(self.token, dict):
            return float(self.token.get("current_price") or 0.0)
        return float(getattr(self.token, "current_price", 0.0) or 0.0)

    @property
    def selection_score(self) -> float:
        """Blend discovery quality and fresh LLM conviction onto a 0-1 scale."""
        opportunity = max(0.0, min(1.0, float(self.opportunity_score or 0.0)))
        confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        return opportunity * 0.60 + confidence * 0.40


_RANGE_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%")
_SINGLE_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _parse_position_size_fraction(text: str, default: float = 0.02) -> float:
    """Extract a spot-allocation fraction from the LLM's free-form position_size text.

    The card's position sizing is natural language (e.g. "1-3% spot allocation",
    "2% portfolio risk"). A range like "1-3%" must be matched as a range first --
    a naive single-number regex only captures the trailing "3%" and silently
    drops the "1", overstating the position. Falls back to `default` (a
    conservative 2%) if nothing parses.
    """
    if not text:
        return default

    range_match = _RANGE_PERCENT_RE.search(text)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        fraction = ((low + high) / 2.0) / 100.0
        return max(0.0, min(1.0, fraction))

    single_match = _SINGLE_PERCENT_RE.search(text)
    if single_match:
        fraction = float(single_match.group(1)) / 100.0
        return max(0.0, min(1.0, fraction))

    return default


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _erc20_encode_allowance_call(owner: str, spender: str) -> str:
    contract = _ENCODER.eth.contract(abi=ERC20_ALLOWANCE_APPROVE_ABI)
    return contract.encode_abi("allowance", args=[Web3.to_checksum_address(owner), Web3.to_checksum_address(spender)])


def _erc20_encode_approve_call(spender: str, amount_base_units: int) -> str:
    contract = _ENCODER.eth.contract(abi=ERC20_ALLOWANCE_APPROVE_ABI)
    return contract.encode_abi("approve", args=[Web3.to_checksum_address(spender), int(amount_base_units)])


def _erc20_encode_balance_call(owner: str) -> str:
    contract = _ENCODER.eth.contract(abi=ERC20_ALLOWANCE_APPROVE_ABI)
    return contract.encode_abi("balanceOf", args=[Web3.to_checksum_address(owner)])


# ------------------------------------------------------------------
# Ryu pipeline constants
# ------------------------------------------------------------------

# How many tokens to pull from the platform DB as the initial candidate pool.
# The pool is then re-ranked by Ryu's opportunity score (which favours smaller,
# higher-potential tokens). Only the top settings.RYU_CANDIDATE_COUNT candidates
# enter the expensive analysis stage; this larger pool controls ranking breadth.
RYU_CANDIDATE_POOL_SIZE: int = int(os.getenv("RYU_CANDIDATE_POOL_SIZE", "50"))
RYU_CANDIDATE_SOURCE_LIMIT: int = int(os.getenv("RYU_CANDIDATE_SOURCE_LIMIT", "1000"))

# How long (seconds) to reuse a Ryu LLM analysis before re-running it.
# Default 6 h — one full platform signal generation window.
RYU_ANALYSIS_CACHE_TTL: int = int(os.getenv("RYU_ANALYSIS_CACHE_TTL_SECONDS", str(6 * 3600)))

# Chains whose entry and recovery routes are implemented for Ryu.
RYU_TRADEABLE_CHAIN_IDS: set = {8453, 101}   # base, solana
RYU_TRADEABLE_BLOCKCHAINS: set = {"base", "solana"}


class RyuSpotTradingService:
    """Sources platform-discovered tokens, filters to tradeable universe,
    runs LLM analysis (with per-symbol result cache), and executes Ryu's
    live spot trades via LI.FI."""

    def __init__(self):
        self.supabase = get_service_client()
        # Still needed for Solana address hydration only — no discoverTokens() calls.
        self._discovery_service = get_enhanced_token_discovery_service()
        self.lifi_service = get_lifi_service(api_key=getattr(settings, "LIFI_API_KEY", None))
        self.signing_service = get_privy_signing_service(
            privy_app_id=settings.PRIVY_APP_ID,
            privy_app_secret=settings.PRIVY_APP_SECRET,
            testnet=settings.HYPERLIQUID_TESTNET,
        )
        self._rpc_client = httpx.AsyncClient(timeout=20.0)
        self._alchemy_api_key = os.getenv("ALCHEMY_API_KEY", "")
        self._base_rpc_url = (
            os.getenv("BASE_MAINNET_RPC_URL")
            or (
                f"{BASE_RPC_URL_BASE}{self._alchemy_api_key}"
                if self._alchemy_api_key
                else "https://mainnet.base.org"
            )
        )
        # Solana address cache (kept so we can map ethereum-listed tokens to Solana
        # route when available, without running a full CoinGecko discoverTokens).
        self._solana_platform_cache: Dict[str, str] = {}
        self._solana_platform_cache_updated_at: float = 0.0

        # Per-symbol Ryu LLM analysis cache: symbol -> (timestamp, RyuCandidateAnalysis)
        # Entries older than RYU_ANALYSIS_CACHE_TTL are re-analyzed; fresh ones are
        # reused so we don't pay LLM cost on every cycle for unchanged tokens.
        self._analysis_cache: Dict[str, tuple] = {}
        self._last_analysis_failures = 0

    # ------------------------------------------------------------------
    # Candidate sourcing from platform DB  (replaces CoinGecko discovery)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_base_address(address: Any) -> bool:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(address or "").strip()))

    @staticmethod
    def _is_valid_solana_address(address: Any) -> bool:
        return bool(
            re.fullmatch(
                r"[1-9A-HJ-NP-Za-km-z]{32,44}",
                str(address or "").strip(),
            )
        )

    @staticmethod
    def _contract_addresses(value: Any) -> Dict[str, str]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                return {}
        return {}

    @staticmethod
    def _select_token_route(token: Any) -> Optional[Dict[str, Any]]:
        """Resolve legacy DiscoveredToken objects to a supported safe route."""
        addresses = RyuSpotTradingService._contract_addresses(
            getattr(token, "contract_addresses", None)
        )
        base_address = str(addresses.get("base") or "").strip()
        if RyuSpotTradingService._is_valid_base_address(base_address):
            return {
                **SUPPORTED_RYU_CHAINS["base"],
                "chain_name": "base",
                "token_address": base_address,
            }
        solana_address = str(
            addresses.get("solana")
            or getattr(token, "primary_address", None)
            or ""
        ).strip()
        primary_chain_id = getattr(token, "primary_chain_id", None)
        if RyuSpotTradingService._is_valid_solana_address(solana_address) and (
            addresses.get("solana")
            or primary_chain_id in {SOLANA_ANALYSIS_CHAIN_ID, SOLANA_CHAIN_ID}
        ):
            return {
                **SUPPORTED_RYU_CHAINS["solana"],
                "chain_name": "solana",
                "token_address": solana_address,
            }
        return None

    @staticmethod
    def _select_token_route_from_db(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve the best LI.FI-tradeable route for a platform-discovered token.

        A token can be labelled Ethereum as its primary chain while also having
        a directly tradeable Base or Solana deployment. Prefer those verified
        secondary contracts before falling back to the primary route.
        """
        addresses = RyuSpotTradingService._contract_addresses(
            row.get("contract_addresses")
        )
        base_address = str(addresses.get("base") or "").strip()
        if RyuSpotTradingService._is_valid_base_address(base_address):
            return {
                **SUPPORTED_RYU_CHAINS["base"],
                "chain_name": "base",
                "token_address": base_address,
            }
        solana_address = str(addresses.get("solana") or "").strip()
        if RyuSpotTradingService._is_valid_solana_address(solana_address):
            return {
                **SUPPORTED_RYU_CHAINS["solana"],
                "chain_name": "solana",
                "token_address": solana_address,
            }

        blockchain = str(row.get("blockchain") or "").lower()
        chain_id = row.get("primary_chain_id") or row.get("chain_id")
        address = str(row.get("primary_address") or row.get("address") or "").strip()

        if (
            (blockchain == "base" or chain_id == BASE_CHAIN_ID)
            and RyuSpotTradingService._is_valid_base_address(address)
        ):
            return {
                **SUPPORTED_RYU_CHAINS["base"],
                "chain_name": "base",
                "token_address": address,
            }
        if (
            (
                blockchain == "solana"
                or chain_id in {SOLANA_ANALYSIS_CHAIN_ID, SOLANA_CHAIN_ID}
            )
            and RyuSpotTradingService._is_valid_solana_address(address)
        ):
            return {
                **SUPPORTED_RYU_CHAINS["solana"],
                "chain_name": "solana",
                "token_address": address,
            }
        return None

    @staticmethod
    def _opportunity_score(row: Dict[str, Any]) -> float:
        """Composite opportunity score for Ryu candidate ranking.

        Philosophy: smaller tokens offer better value-for-money (more room to
        grow, lower correlation to BTC macro) so the scoring deliberately
        rewards the small-to-mid-cap sweet spot over already-large tokens.

        Weights (all normalised 0–1 before weighting):
          40 %  Platform potential_score  — already encodes growth signals
          25 %  Market-cap size tier      — $1M-$50M best, $50M-$200M good,
                                            $200M-$500M neutral, >$500M penalised
          20 %  Volume/market-cap ratio   — higher ratio = more actively traded
          15 %  24h momentum              — slight positive bias, penalise
                                            extreme pumps (>30%) or heavy dumps
        """
        # 1. Platform potential score (0-10 scale -> normalise to 0-1)
        potential = float(row.get("potential_score") or row.get("overall_score") or 5.0)
        potential_norm = min(1.0, max(0.0, potential / 10.0))

        # 2. Market-cap size tier  (smaller = higher score for Ryu)
        mcap = float(row.get("market_cap") or 0)
        if 1_000_000 <= mcap < 10_000_000:        # $1M-$10M  — highest conviction
            mcap_score = 1.0
        elif 10_000_000 <= mcap < 50_000_000:     # $10M-$50M — strong
            mcap_score = 0.85
        elif 50_000_000 <= mcap < 200_000_000:    # $50M-$200M — moderate
            mcap_score = 0.60
        elif 200_000_000 <= mcap <= 500_000_000:  # $200M-$500M — low
            mcap_score = 0.30
        else:
            mcap_score = 0.0

        # 3. Volume / market-cap ratio (activity proxy)
        vol = float(row.get("volume_24h") or 0)
        vol_ratio = (vol / mcap) if mcap > 0 else 0.0
        vol_score = min(1.0, vol_ratio * 5.0)  # 20% vol/mcap = perfect score

        # 4. 24h momentum
        change = float(row.get("price_change_24h") or 0)
        if -5 <= change <= 20:     # healthy positive move
            momentum_score = 0.8 + (change / 20) * 0.2
        elif 20 < change <= 30:    # strong but watch for overextension
            momentum_score = 0.6
        elif change > 30:          # likely pump — penalise
            momentum_score = 0.2
        elif -15 <= change < -5:   # mild dip — still ok
            momentum_score = 0.5
        else:                      # heavy sell-off
            momentum_score = 0.15

        return (
            potential_norm * 0.40
            + mcap_score   * 0.25
            + vol_score    * 0.20
            + momentum_score * 0.15
        )

    def _hydrate_route_metadata(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge multi-chain fields omitted by older DB view definitions.

        PostgreSQL expands `dt.*` when a view is created, not whenever columns
        are later added to `discovered_tokens`. Production therefore has the
        route fields on the table but not on its older analysis view.
        """
        token_ids = [str(row.get("id")) for row in rows if row.get("id")]
        if not token_ids:
            return rows

        route_metadata: Dict[str, Dict[str, Any]] = {}
        try:
            for index in range(0, len(token_ids), 200):
                response = (
                    self.supabase
                    .table("discovered_tokens")
                    .select(
                        "id, contract_addresses, primary_chain_id, primary_address"
                    )
                    .in_("id", token_ids[index:index + 200])
                    .execute()
                )
                for metadata in response.data or []:
                    route_metadata[str(metadata.get("id"))] = metadata
        except Exception as exc:
            logger.warning(
                "Ryu could not hydrate secondary contract routes; using primary "
                "view fields only: %s",
                exc,
            )
            return rows

        return [
            {**row, **route_metadata.get(str(row.get("id")), {})}
            for row in rows
        ]

    @staticmethod
    def _candidate_from_position(position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reconstruct an analysis candidate from a persisted Ryu holding."""
        metadata = position.get("position_metadata") or {}
        symbol = str(position.get("symbol") or "").upper()
        token_address = str(metadata.get("contract_address") or "").strip()
        chain_id = int(metadata.get("chain_id") or 0)
        if not symbol or not token_address:
            return None

        if chain_id == BASE_CHAIN_ID:
            route = {
                **SUPPORTED_RYU_CHAINS["base"],
                "chain_name": "base",
                "token_address": token_address,
            }
            blockchain = "base"
            analysis_chain_id = BASE_CHAIN_ID
        elif chain_id == SOLANA_CHAIN_ID:
            route = {
                **SUPPORTED_RYU_CHAINS["solana"],
                "chain_name": "solana",
                "token_address": token_address,
            }
            blockchain = "solana"
            analysis_chain_id = SOLANA_ANALYSIS_CHAIN_ID
        else:
            logger.warning(
                "Ryu cannot independently re-analyze unsupported held route %s/%s",
                symbol,
                chain_id,
            )
            return None

        return {
            "symbol": symbol,
            "name": metadata.get("token_name") or symbol,
            "blockchain": blockchain,
            "chain_id": analysis_chain_id,
            "address": token_address,
            "coingecko_id": metadata.get("coingecko_id") or "",
            "current_price": position.get("current_price") or position.get("entry_price") or 0.0,
            "_ryu_opportunity_score": metadata.get("latest_opportunity_score")
            or metadata.get("entry_opportunity_score")
            or 0.0,
            "route": route,
        }

    async def get_ranked_tradeable_candidates(
        self,
        pool_size: int = RYU_CANDIDATE_POOL_SIZE,
    ) -> List[Dict[str, Any]]:
        """Pull candidates from the platform DB and rank by Ryu opportunity score.

        Steps:
          1. Fetch active, quality (CoinGecko-listed) tokens from
             `tokens_with_analysis`. Basic volume/mcap guards are applied in
             the DB query, but primary-chain labels are not used as a proxy for
             routability.
          2. Universe filter: resolve each token to a concrete LI.FI route and
             drop tokens whose chain has no supported route.
          3. Re-rank the pool using _opportunity_score(), which deliberately
             favours smaller-cap tokens (more upside potential per dollar).
          4. Return all ranked candidates.  analyse_candidates applies its own
             per-symbol LLM cache, so only new/stale tokens cost an LLM call.

        No CoinGecko API call is made here.
        """
        try:
            # Include enough of the quality universe to discover secondary
            # Base/Solana contracts before route filtering.
            fetch_limit = max(pool_size * 4, RYU_CANDIDATE_SOURCE_LIMIT)
            response = (
                self.supabase
                .table("tokens_with_analysis")
                .select(
                    "id, symbol, name, blockchain, chain_id, address, coingecko_id, "
                    "logo_url, current_price, market_cap, volume_24h, price_change_24h, "
                    "overall_score, potential_score, confidence_score, analysis_created_at"
                )
                .eq("is_active", True)
                .not_.is_("coingecko_id", "null")          # quality lane only
                .gte("volume_24h", 50_000)                 # loose volume floor
                .gte("market_cap", 1_000_000)              # absolute minimum $1M
                .lte("market_cap", 500_000_000)            # upper cap $500M
                .order("potential_score", desc=True)       # DB order: high potential first
                .order("market_cap", desc=False)           # small cap priority
                .limit(fetch_limit)
                .execute()
            )
        except Exception as e:
            logger.error("Ryu: failed to source candidates from platform DB: %s", e)
            return []

        rows = self._hydrate_route_metadata(response.data or [])

        # Resolve LI.FI route and drop tokens with no tradeable path
        pool: List[Dict[str, Any]] = []
        for row in rows:
            route = self._select_token_route_from_db(row)
            if route is None:
                continue
            pool.append({**row, "route": route})

        # Re-rank entire pool by Ryu's opportunity score (small-cap + potential + momentum)
        pool.sort(key=self._opportunity_score, reverse=True)

        # Slice to the top pool_size (50) best opportunity candidates
        selected_pool = pool[:pool_size]

        logger.info(
            "Ryu candidate pool: %d tokens fetched from DB, %d passed universe filter, "
            "top %d selected by opportunity score (top 5: %s)",
            len(rows),
            len(pool),
            len(selected_pool),
            ", ".join(c["symbol"] for c in selected_pool[:5]) if selected_pool else "none",
        )
        return selected_pool

    # kept for backward compat
    async def get_ranked_solana_candidates(self, limit: int) -> List[Dict[str, Any]]:
        return await self.get_ranked_tradeable_candidates(pool_size=limit)

    async def analyze_candidates(
        self,
        candidates: List[Dict[str, Any]],
        force_symbols: Optional[set] = None,
    ) -> Dict[str, "RyuCandidateAnalysis"]:
        """Run the token-analysis card LLM pipeline on candidates.

        Two layers of cost control:
          1. The candidate pool is ranked before this method is called.
          2. Per-symbol analysis cache: if Ryu already analyzed this symbol
             within RYU_ANALYSIS_CACHE_TTL seconds, the cached result is reused
             and no LLM call is made.

        Failed analyses are counted and surfaced in the cycle summary.
        """
        self._last_analysis_failures = 0
        if not candidates:
            return {}

        from kata.api.token_analysis import (
            TokenAnalysisRequest,
            _analyze_token_impl,
            get_advanced_metrics_service,
            get_coingecko_service,
            get_enhanced_market_service,
            get_llm_analysis_service,
            get_market_data_service,
            get_sentiment_service,
        )

        llm_service = await get_llm_analysis_service()
        market_service = await get_market_data_service()
        enhanced_service = await get_enhanced_market_service()
        sentiment_service = await get_sentiment_service()
        coingecko_service = await get_coingecko_service()
        advanced_metrics_service = await get_advanced_metrics_service()

        now = time.time()
        results: Dict[str, RyuCandidateAnalysis] = {}
        llm_calls = 0
        cache_hits = 0
        failures = 0
        forced = {str(symbol).upper() for symbol in (force_symbols or set())}

        for candidate in candidates:
            symbol = str(candidate.get("symbol") or "").upper()
            route = candidate.get("route")
            if not symbol or not route:
                failures += 1
                continue

            # ----------------------------------------------------------
            # Analysis cache check: reuse if fresh
            # ----------------------------------------------------------
            cached_entry = None if symbol in forced else self._analysis_cache.get(symbol)
            if cached_entry:
                cached_at, cached_analysis = cached_entry
                age = now - cached_at
                if age < RYU_ANALYSIS_CACHE_TTL:
                    results[symbol] = cached_analysis
                    cache_hits += 1
                    logger.debug(
                        "Ryu reusing cached analysis for %s (%.0f min old, TTL %.0f min)",
                        symbol,
                        age / 60,
                        RYU_ANALYSIS_CACHE_TTL / 60,
                    )
                    continue

            # ----------------------------------------------------------
            # LLM analysis
            # ----------------------------------------------------------
            try:
                request = TokenAnalysisRequest(
                    token_ticker=symbol,
                    agent_type="ryu",
                    coingecko_id=candidate.get("coingecko_id") or "",
                    contract_address=route["token_address"],
                    chain_id=route["analysis_chain_id"],
                )
                response = await _analyze_token_impl(
                    request,
                    llm_service,
                    market_service,
                    enhanced_service,
                    sentiment_service,
                    coingecko_service,
                    advanced_metrics_service,
                    analysis_profile="spot_investment",
                )
                insights = response.detailed_insights
                if not insights:
                    failures += 1
                    logger.warning(
                        "Ryu candidate analysis returned no detailed insights for %s",
                        symbol,
                    )
                    continue
                risk_management = insights.risk_management
                price_targets = insights.price_targets
                opportunity_score = _positive_float(
                    candidate.get("_ryu_opportunity_score")
                )
                if opportunity_score is None:
                    opportunity_score = self._opportunity_score(candidate)
                analysis_token = {
                    **candidate,
                    "current_price": (
                        _positive_float((response.market_data or {}).get("price"))
                        or candidate.get("current_price")
                        or 0.0
                    ),
                }

                analysis = RyuCandidateAnalysis(
                    token=analysis_token,
                    trade_intent=response.trade_intent,
                    confidence=insights.confidence,
                    action_summary=insights.action_summary,
                    reasoning=insights.reasoning,
                    position_size_text=(
                        risk_management.position_size if risk_management else ""
                    ),
                    chain_name=route["chain_name"],
                    chain_id=route["lifi_chain_id"],
                    token_address=route["token_address"],
                    opportunity_score=opportunity_score,
                    stop_loss=_positive_float(
                        risk_management.stop_loss if risk_management else None
                    ),
                    target_1=_positive_float(
                        price_targets.target_1 if price_targets else None
                    ),
                    target_2=_positive_float(
                        price_targets.target_2 if price_targets else None
                    ),
                    target_3=_positive_float(
                        price_targets.target_3 if price_targets else None
                    ),
                )
                results[symbol] = analysis
                # Store in analysis cache
                self._analysis_cache[symbol] = (time.time(), analysis)
                llm_calls += 1
            except Exception as e:
                failures += 1
                logger.warning("Ryu candidate analysis failed for %s: %s", symbol, e)
                continue

        self._last_analysis_failures = failures
        logger.info(
            "Ryu analyze_candidates: %d candidates → %d LLM calls + %d cache hits "
            "→ %d results, %d failures",
            len(candidates),
            llm_calls,
            cache_hits,
            len(results),
            failures,
        )
        return results

    # ------------------------------------------------------------------
    # Top-level cycle
    # ------------------------------------------------------------------

    async def run_signal_cycle(self, active_allocations: List[Any]) -> Dict[str, int]:
        """Run one Ryu signal cycle across every active allocation.

        Held assets are analyzed first and outside the discovery ranking so an
        asset cannot lose its exit path merely because it falls out of the top
        candidate pool. Explicit exits are also processed before the slower
        breadth scan, protecting the portfolio if a worker restart interrupts
        candidate analysis.
        """
        summary = {"candidates_analyzed": 0, "buys_executed": 0, "exits_executed": 0, "errors": 0}
        if not active_allocations:
            return summary

        positions_by_allocation: Dict[str, List[Dict[str, Any]]] = {}
        held_candidates_by_symbol: Dict[str, Dict[str, Any]] = {}
        for allocation in active_allocations:
            positions = await self._get_open_positions(allocation.allocation_id)
            positions_by_allocation[allocation.allocation_id] = positions
            for position in positions:
                candidate = self._candidate_from_position(position)
                if candidate:
                    held_candidates_by_symbol.setdefault(candidate["symbol"], candidate)

        held_analyses = await self.analyze_candidates(
            list(held_candidates_by_symbol.values()),
            force_symbols=set(held_candidates_by_symbol),
        )
        summary["candidates_analyzed"] += len(held_analyses)
        summary["errors"] += self._last_analysis_failures

        # Refresh conviction/risk metadata and honor explicit sell decisions
        # before the broad candidate analysis can delay execution.
        for allocation in active_allocations:
            positions = positions_by_allocation.get(allocation.allocation_id, [])
            try:
                for position in positions:
                    analysis = held_analyses.get(str(position.get("symbol") or "").upper())
                    if analysis:
                        await self._refresh_position_analysis_metadata(position, analysis)
                exits = await self._process_signal_exits(
                    allocation,
                    held_analyses,
                    open_positions=positions,
                )
                summary["exits_executed"] += exits
            except Exception as e:
                logger.error(
                    "Ryu held-position cycle failed for allocation %s: %s",
                    getattr(allocation, "allocation_id", "?"),
                    e,
                )
                summary["errors"] += 1

        ranked_candidates = await self.get_ranked_tradeable_candidates()
        analysis_limit = max(1, int(settings.RYU_CANDIDATE_COUNT or 8))
        candidates = ranked_candidates[:analysis_limit]
        logger.info(
            "Ryu expensive-analysis shortlist: %d of %d ranked candidates "
            "(configured limit=%d)",
            len(candidates),
            len(ranked_candidates),
            analysis_limit,
        )
        candidate_analyses = await self.analyze_candidates(candidates)
        summary["candidates_analyzed"] += len(candidate_analyses)
        summary["errors"] += self._last_analysis_failures
        analyses = {**candidate_analyses, **held_analyses}

        for allocation in active_allocations:
            try:
                result = await self._process_entries_and_rebalance(allocation, analyses)
                summary["buys_executed"] += result.get("buys", 0)
                summary["exits_executed"] += result.get("exits", 0)
            except Exception as e:
                logger.error(f"Ryu signal cycle failed for allocation {getattr(allocation, 'allocation_id', '?')}: {e}")
                summary["errors"] += 1

        return summary

    async def _process_allocation(self, allocation: Any, analyses: Dict[str, RyuCandidateAnalysis]) -> Dict[str, int]:
        """Compatibility wrapper used by focused tests and one-off callers."""
        result = {"buys": 0, "exits": 0}
        open_positions = await self._get_open_positions(allocation.allocation_id)
        result["exits"] = await self._process_signal_exits(
            allocation,
            analyses,
            open_positions=open_positions,
        )
        entry_result = await self._process_entries_and_rebalance(allocation, analyses)
        result["buys"] += entry_result.get("buys", 0)
        result["exits"] += entry_result.get("exits", 0)
        return result

    async def _process_signal_exits(
        self,
        allocation: Any,
        analyses: Dict[str, RyuCandidateAnalysis],
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        open_positions = (
            open_positions
            if open_positions is not None
            else await self._get_open_positions(allocation.allocation_id)
        )
        exits = 0
        # Exits first: only ever closes a position Ryu itself opened and recorded.
        for position in open_positions:
            symbol = str(position.get("symbol") or "").upper()
            analysis = analyses.get(symbol)
            if not analysis or analysis.trade_intent != "SPOT_EXIT":
                continue
            if analysis.confidence < settings.RYU_MIN_CONFIDENCE_TO_TRADE:
                continue
            if await self._execute_spot_exit(allocation, position, analysis):
                exits += 1
        return exits

    @staticmethod
    def _position_score(position: Dict[str, Any]) -> Optional[float]:
        metadata = position.get("position_metadata") or {}
        value = metadata.get("latest_selection_score")
        if value is None:
            value = metadata.get("entry_selection_score")
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _position_age_minutes(position: Dict[str, Any]) -> Optional[float]:
        raw = position.get("created_at")
        if isinstance(raw, datetime):
            created_at = raw
        else:
            try:
                created_at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 60.0)

    async def _process_entries_and_rebalance(
        self,
        allocation: Any,
        analyses: Dict[str, RyuCandidateAnalysis],
    ) -> Dict[str, int]:
        result = {"buys": 0, "exits": 0}
        open_positions = await self._get_open_positions(allocation.allocation_id)
        held_symbols = {
            str(position.get("symbol") or "").upper()
            for position in open_positions
        }
        eligible = sorted(
            (
                analysis
                for symbol, analysis in analyses.items()
                if str(symbol).upper() not in held_symbols
                and analysis.trade_intent == "SPOT_BUY"
                and analysis.confidence >= settings.RYU_MIN_CONFIDENCE_TO_TRADE
            ),
            key=lambda analysis: analysis.selection_score,
            reverse=True,
        )

        initial_open_count = len(open_positions)
        available_slots = max(0, settings.RYU_MAX_CONCURRENT_POSITIONS - initial_open_count)
        while eligible and available_slots > 0:
            analysis = eligible.pop(0)
            if await self._execute_spot_buy(allocation, analysis):
                result["buys"] += 1
                held_symbols.add(analysis.symbol)
                available_slots -= 1

        if not eligible:
            return result

        # Do not churn positions in the same cycle that still had room to add.
        # Rotation is for a full portfolio or one with too little idle cash to
        # fund another minimum-sized entry.
        needs_rotation = (
            initial_open_count >= settings.RYU_MAX_CONCURRENT_POSITIONS
            or float(allocation.remaining_amount or 0.0) < settings.RYU_SPOT_MIN_TRADE_USDC
        )
        if (
            not needs_rotation
            or not settings.RYU_REBALANCE_ENABLED
        ):
            return result

        rotations = 0
        max_rotations = max(0, int(settings.RYU_MAX_ROTATIONS_PER_CYCLE))
        while eligible and rotations < max_rotations:
            rotation_candidates = [
                position
                for position in open_positions
                if self._position_score(position) is not None
                and (
                    self._position_age_minutes(position) is not None
                    and self._position_age_minutes(position)
                    >= settings.RYU_REBALANCE_MIN_HOLD_MINUTES
                )
            ]
            if not rotation_candidates:
                break

            weakest = min(rotation_candidates, key=self._position_score)
            weakest_score = self._position_score(weakest)
            strongest = eligible[0]
            if (
                weakest_score is None
                or strongest.selection_score
                < weakest_score + settings.RYU_REBALANCE_MIN_SCORE_DELTA
            ):
                break

            weakest_symbol = str(weakest.get("symbol") or "").upper()
            weakest_analysis = analyses.get(weakest_symbol)
            exited = await self._execute_position_exit(
                allocation,
                weakest,
                weakest_analysis,
                exit_reason=f"rebalance_to_{strongest.symbol}",
                require_live_enabled=True,
            )
            if not exited:
                break

            result["exits"] += 1
            open_positions.remove(weakest)
            held_symbols.discard(weakest_symbol)
            eligible.pop(0)
            rotations += 1
            if await self._execute_spot_buy(allocation, strongest):
                result["buys"] += 1
                held_symbols.add(strongest.symbol)
            else:
                logger.warning(
                    "Ryu rotated out of %s but could not complete replacement buy %s; "
                    "proceeds remain in Base USDC",
                    weakest_symbol,
                    strongest.symbol,
                )

        return result

    # ------------------------------------------------------------------
    # Position bookkeeping
    # ------------------------------------------------------------------

    async def _get_open_positions(self, allocation_id: str) -> List[Dict[str, Any]]:
        # agent_positions has no spot-specific "status" column -- it was built around
        # Yuki's margin positions. is_active is the real open/closed flag there;
        # `size`/`position_metadata` are the real column names (not position_size/
        # trade_metadata, which only exist on agent_trades).
        try:
            rows = (
                self.supabase.table("agent_positions")
                .select("*")
                .eq("allocation_id", allocation_id)
                .eq("agent_type", "ryu")
                .eq("is_active", True)
                .execute()
            ).data or []
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch open Ryu positions for {allocation_id}: {e}")
            return []

    @staticmethod
    def _build_entry_risk_plan(
        analysis: RyuCandidateAnalysis,
        entry_price: float,
        original_size: float,
    ) -> Optional[Dict[str, Any]]:
        """Normalize the AI plan against the actual spot execution price."""
        entry = _positive_float(entry_price)
        stop_loss = _positive_float(analysis.stop_loss)
        reference = _positive_float(analysis.current_price) or entry
        if (
            entry is None
            or reference is None
            or stop_loss is None
            or stop_loss >= reference
        ):
            return None

        source_targets = sorted({
            target
            for target in (
                _positive_float(analysis.target_1),
                _positive_float(analysis.target_2),
                _positive_float(analysis.target_3),
            )
            if target is not None and target > reference
        })
        if not source_targets:
            return None
        # Preserve the AI plan's percentage distances when the executable LI.FI
        # entry differs from the analysis snapshot.
        normalized_stop = entry * (stop_loss / reference)
        targets = [entry * (target / reference) for target in source_targets]

        return {
            "stop_loss": normalized_stop,
            "target_1": targets[0],
            "target_2": targets[1] if len(targets) > 1 else None,
            "target_3": targets[2] if len(targets) > 2 else None,
            "original_size": max(0.0, float(original_size or 0.0)),
            "target_1_hit": False,
            "target_2_hit": False,
            "target_3_hit": False,
            "trailing_active": False,
            "high_watermark_price": entry,
            "trailing_stop_price": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def _has_valid_entry_risk_plan(cls, analysis: RyuCandidateAnalysis) -> bool:
        reference_price = _positive_float(analysis.current_price)
        return bool(
            reference_price
            and cls._build_entry_risk_plan(analysis, reference_price, 1.0)
        )

    async def _persist_position_metadata(
        self,
        position: Dict[str, Any],
        metadata: Dict[str, Any],
        mark_price: Optional[float] = None,
        position_value: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        unrealized_pnl_percent: Optional[float] = None,
    ) -> None:
        payload: Dict[str, Any] = {"position_metadata": metadata}
        if mark_price is not None:
            payload["current_price"] = mark_price
        if position_value is not None:
            payload["position_value"] = position_value
        if unrealized_pnl is not None:
            payload["unrealized_pnl"] = unrealized_pnl
        if unrealized_pnl_percent is not None:
            payload["unrealized_pnl_percent"] = unrealized_pnl_percent
        self.supabase.table("agent_positions").update(payload).eq(
            "id",
            position.get("id"),
        ).execute()
        position["position_metadata"] = metadata
        position.update(payload)

    async def _refresh_position_analysis_metadata(
        self,
        position: Dict[str, Any],
        analysis: RyuCandidateAnalysis,
    ) -> None:
        """Persist fresh conviction for rebalancing and only tighten live risk."""
        metadata = dict(position.get("position_metadata") or {})
        metadata.update({
            "latest_selection_score": analysis.selection_score,
            "latest_opportunity_score": analysis.opportunity_score,
            "latest_confidence": analysis.confidence,
            "latest_trade_intent": analysis.trade_intent,
            "last_analyzed_at": datetime.now(timezone.utc).isoformat(),
        })
        if not str(metadata.get("purchase_thesis") or "").strip():
            metadata["purchase_thesis"] = analysis.reasoning

        risk_plan = dict(metadata.get("risk_plan") or {})
        reference_price = _positive_float(
            position.get("current_price") or position.get("entry_price")
        )
        if not risk_plan:
            risk_plan = self._build_entry_risk_plan(
                analysis,
                float(position.get("entry_price") or reference_price or 0.0),
                float(position.get("size") or 0.0),
            ) or {}
        refreshed_stop = _positive_float(analysis.stop_loss)
        existing_stop = _positive_float(risk_plan.get("stop_loss"))
        if (
            reference_price is not None
            and refreshed_stop is not None
            and refreshed_stop < reference_price
        ):
            risk_plan["stop_loss"] = max(
                value for value in (existing_stop, refreshed_stop) if value is not None
            )
        if risk_plan:
            metadata["risk_plan"] = risk_plan
        await self._persist_position_metadata(position, metadata)

    @staticmethod
    def _risk_decision(
        position: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, Any]:
        """Return the next deterministic lifecycle action for one marked holding."""
        metadata = dict(position.get("position_metadata") or {})
        risk_plan = dict(metadata.get("risk_plan") or {})
        price = float(current_price or 0.0)
        if price <= 0 or not risk_plan:
            return {"action": "none", "metadata": metadata}

        entry_price = float(position.get("entry_price") or 0.0)
        current_size = float(position.get("size") or 0.0)
        original_size = float(risk_plan.get("original_size") or current_size)
        high_watermark = max(
            price,
            float(risk_plan.get("high_watermark_price") or entry_price or price),
        )
        risk_plan["high_watermark_price"] = high_watermark

        trailing_active = bool(risk_plan.get("trailing_active"))
        trailing_distance = max(
            0.0,
            min(0.95, float(settings.RYU_TRAILING_STOP_PERCENT)),
        )
        if trailing_active:
            trailing_stop = max(
                entry_price,
                high_watermark * (1.0 - trailing_distance),
            )
            risk_plan["trailing_stop_price"] = trailing_stop
        else:
            trailing_stop = None

        metadata["risk_plan"] = risk_plan
        hard_stop = _positive_float(risk_plan.get("stop_loss"))
        effective_stop = max(
            value for value in (hard_stop, trailing_stop) if value is not None
        ) if any(value is not None for value in (hard_stop, trailing_stop)) else None
        if effective_stop is not None and price <= effective_stop:
            reason = "trailing_stop" if trailing_stop and trailing_stop >= (hard_stop or 0) else "stop_loss"
            return {
                "action": "exit",
                "reason": reason,
                "exit_fraction": 1.0,
                "metadata": metadata,
            }

        target_1 = _positive_float(risk_plan.get("target_1"))
        target_2 = _positive_float(risk_plan.get("target_2"))
        target_3 = _positive_float(risk_plan.get("target_3"))
        if (
            target_1 is not None
            and not risk_plan.get("target_1_hit")
            and price >= target_1
        ):
            risk_plan["target_1_hit"] = True
            if entry_price > 0:
                risk_plan["stop_loss"] = max(hard_stop or 0.0, entry_price)
            if target_2 is None:
                risk_plan["trailing_active"] = True
            metadata["risk_plan"] = risk_plan
            desired_size = original_size * max(
                0.0,
                min(1.0, float(settings.RYU_TARGET_1_SELL_FRACTION)),
            )
            return {
                "action": "exit",
                "reason": "target_1",
                "exit_fraction": min(1.0, desired_size / max(current_size, 1e-18)),
                "metadata": metadata,
            }

        if (
            target_2 is not None
            and risk_plan.get("target_1_hit")
            and not risk_plan.get("target_2_hit")
            and price >= target_2
        ):
            risk_plan["target_2_hit"] = True
            risk_plan["trailing_active"] = True
            metadata["risk_plan"] = risk_plan
            desired_size = original_size * max(
                0.0,
                min(1.0, float(settings.RYU_TARGET_2_SELL_FRACTION)),
            )
            return {
                "action": "exit",
                "reason": "target_2",
                "exit_fraction": min(1.0, desired_size / max(current_size, 1e-18)),
                "metadata": metadata,
            }

        if (
            target_3 is not None
            and risk_plan.get("target_2_hit")
            and not risk_plan.get("target_3_hit")
            and price >= target_3
        ):
            risk_plan["target_3_hit"] = True
            risk_plan["trailing_active"] = True
            tighter_distance = trailing_distance / 2.0
            risk_plan["trailing_stop_price"] = max(
                entry_price,
                high_watermark * (1.0 - tighter_distance),
            )
            metadata["risk_plan"] = risk_plan

        return {"action": "none", "metadata": metadata}

    # ------------------------------------------------------------------
    # On-chain execution
    # ------------------------------------------------------------------

    async def _get_erc20_allowance(self, chain_id: int, token_address: str, owner: str, spender: str) -> int:
        if chain_id != BASE_CHAIN_ID:
            raise ValueError(f"No RPC configured for chain {chain_id}")

        call_data = _erc20_encode_allowance_call(owner, spender)
        result = await self._base_rpc(
            "eth_call",
            [{"to": token_address, "data": call_data}, "latest"],
        )
        if not result or result == "0x":
            return 0
        return int(result, 16)

    async def _get_erc20_balance_base_units(
        self,
        chain_id: int,
        token_address: str,
        owner: str,
    ) -> int:
        """Read an exact token balance so float rounding cannot oversell it."""
        if chain_id != BASE_CHAIN_ID:
            raise ValueError(f"No RPC configured for chain {chain_id}")

        result = await self._base_rpc(
            "eth_call",
            [{
                "to": token_address,
                "data": _erc20_encode_balance_call(owner),
            }, "latest"],
        )
        if not result or result == "0x":
            return 0
        return int(result, 16)

    async def _submit_transaction(self, user_id: str, wallet_address: str, chain_id: int, to: str, data: str, value: str = "0") -> Optional[str]:
        signing_result = (
            await self.signing_service.sign_and_send_delegated_evm_transaction(
                user_id=user_id,
                chain_id=chain_id,
                transaction_data={"to": to, "data": data, "value": value},
                wallet_address=wallet_address,
                reference_id=f"ryu-{chain_id}-{int(time.time())}",
            )
        )
        if not signing_result.success:
            logger.error(f"Ryu transaction signing/broadcast failed for {user_id}: {signing_result.error}")
            return None
        return signing_result.transaction_hash

    async def _ensure_approval(
        self, user_id: str, wallet_address: str, chain_id: int, token_address: str, spender: str, amount_base_units: int
    ) -> bool:
        try:
            current_allowance = await self._get_erc20_allowance(chain_id, token_address, wallet_address, spender)
        except Exception as e:
            logger.error(f"Ryu allowance check failed for {token_address}: {e}")
            return False

        if current_allowance >= amount_base_units:
            return True

        approve_data = _erc20_encode_approve_call(spender, amount_base_units)
        tx_hash = await self._submit_transaction(user_id, wallet_address, chain_id, token_address, approve_data)
        if not tx_hash:
            return False
        if not await self._wait_for_evm_confirmation(chain_id, tx_hash):
            logger.error("Ryu approval %s did not confirm", tx_hash)
            return False

        logger.info(f"Ryu approval submitted for {user_id}: {tx_hash}")
        return True

    async def _base_rpc(self, method: str, params: List[Any]) -> Any:
        response = await self._rpc_client.post(
            self._base_rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(f"Base RPC error: {payload['error']}")
        return payload.get("result")

    async def get_base_balances(self, wallet_address: str) -> Dict[str, float]:
        """Read the Base USDC capital pool and the user's native gas reserve."""
        native_result, usdc_result = await asyncio.gather(
            self._base_rpc("eth_getBalance", [wallet_address, "latest"]),
            self._base_rpc(
                "eth_call",
                [{
                    "to": BASE_USDC_ADDRESS,
                    "data": _erc20_encode_balance_call(wallet_address),
                }, "latest"],
            ),
        )
        return {
            "eth": int(native_result or "0x0", 16) / 10**18,
            "usdc": int(usdc_result or "0x0", 16) / 10**BASE_USDC_DECIMALS,
        }

    async def _has_user_funded_base_gas(self, wallet_address: str) -> bool:
        try:
            balances = await self.get_base_balances(wallet_address)
            if balances["eth"] < settings.RYU_BASE_MIN_GAS_RESERVE_ETH:
                logger.warning(
                    "Ryu Base wallet %s has %.9f ETH; minimum reserve is %.9f ETH",
                    wallet_address,
                    balances["eth"],
                    settings.RYU_BASE_MIN_GAS_RESERVE_ETH,
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Could not verify Ryu Base gas reserve: {e}")
            return False

    async def _wait_for_evm_confirmation(
        self,
        chain_id: int,
        transaction_hash: str,
        timeout_seconds: int = 120,
    ) -> bool:
        if chain_id != BASE_CHAIN_ID:
            return False
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                receipt = await self._base_rpc(
                    "eth_getTransactionReceipt",
                    [transaction_hash],
                )
                if receipt:
                    raw_status = receipt.get("status")
                    status_value = (
                        int(raw_status, 16)
                        if isinstance(raw_status, str)
                        else int(raw_status or 0)
                    )
                    return status_value == 1
            except Exception as e:
                logger.warning(
                    "Could not poll Base transaction %s: %s",
                    transaction_hash,
                    e,
                )
            await asyncio.sleep(2)
        return False

    async def _wait_for_lifi_completion(
        self,
        transaction_hash: str,
        from_chain_id: int,
        to_chain_id: int,
        timeout_seconds: int = 20 * 60,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status = await self.lifi_service.get_transfer_status(
                tx_hash=transaction_hash,
                from_chain_id=from_chain_id,
                to_chain_id=to_chain_id,
            )
            state = str((status or {}).get("status") or "").upper()
            if state == "DONE":
                return status
            if state in {"FAILED", "INVALID"}:
                return None
            await asyncio.sleep(10)
        return None

    @staticmethod
    def _received_usdc_from_status(
        status: Optional[Dict[str, Any]],
        fallback: float,
    ) -> float:
        receiving = (status or {}).get("receiving") or {}
        token = receiving.get("token") or {}
        if str(token.get("address") or "").lower() != BASE_USDC_ADDRESS.lower():
            return fallback
        try:
            decimals = int(token.get("decimals") or BASE_USDC_DECIMALS)
            amount = int(receiving.get("amount") or 0) / (10**decimals)
            return amount if amount > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _solana_wallet_address(allocation: Any) -> str:
        metrics = getattr(allocation, "performance_metrics", None) or {}
        wallets = metrics.get("wallets") or {}
        return str(wallets.get("solana_address") or "")

    async def _solana_rpc(self, method: str, params: List[Any]) -> Any:
        response = await self._rpc_client.post(
            settings.SOLANA_RPC_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(f"Solana RPC error: {payload['error']}")
        return payload.get("result")

    async def get_solana_balances(
        self,
        wallet_address: str,
    ) -> Dict[str, float]:
        """Read the user's real SOL and Solana USDC balances."""
        native_result = await self._solana_rpc(
            "getBalance",
            [wallet_address, {"commitment": "confirmed"}],
        )
        token_result = await self._solana_rpc(
            "getTokenAccountsByOwner",
            [
                wallet_address,
                {"mint": SOLANA_USDC_ADDRESS},
                {"encoding": "jsonParsed", "commitment": "confirmed"},
            ],
        )
        usdc = 0.0
        for account in (token_result or {}).get("value", []):
            info = (
                ((account.get("account") or {}).get("data") or {})
                .get("parsed", {})
                .get("info", {})
            )
            amount = (
                (info.get("tokenAmount") or {}).get("uiAmountString")
                or (info.get("tokenAmount") or {}).get("uiAmount")
                or 0
            )
            usdc += float(amount)
        return {
            "sol": int((native_result or {}).get("value") or 0) / 1_000_000_000,
            "usdc": usdc,
        }

    async def _has_user_funded_solana_gas(self, wallet_address: str) -> bool:
        try:
            balances = await self.get_solana_balances(wallet_address)
            if balances["sol"] < settings.RYU_SOLANA_MIN_GAS_RESERVE_SOL:
                logger.warning(
                    "Ryu wallet %s has %.9f SOL; minimum reserve is %.9f SOL",
                    wallet_address,
                    balances["sol"],
                    settings.RYU_SOLANA_MIN_GAS_RESERVE_SOL,
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Could not verify Ryu SOL gas reserve: {e}")
            return False

    async def _ensure_solana_gas_from_base(
        self,
        allocation: Any,
    ) -> float:
        """Fund SOL from Base USDC only when a Solana trade needs it.

        Returns the Base USDC spent on the reserve, or ``0`` when the existing
        user-owned SOL balance is already sufficient. Raises on failure so the
        token buy never proceeds without a viable future exit fee.
        """
        solana_wallet = self._solana_wallet_address(allocation)
        base_wallet = str(allocation.platform_wallet_address or "")
        if not solana_wallet or not base_wallet:
            raise ValueError("Ryu's Base or Solana wallet is unavailable")
        if await self._has_user_funded_solana_gas(solana_wallet):
            return 0.0

        reserve_usdc = float(settings.RYU_SOLANA_GAS_RESERVE_USDC)
        if reserve_usdc <= 0:
            raise ValueError("Ryu's Solana fee reserve is not configured")
        amount_units = int(reserve_usdc * (10**BASE_USDC_DECIMALS))
        quote = await self.lifi_service.get_token_transfer_quote(
            from_chain_id=BASE_CHAIN_ID,
            to_chain_id=SOLANA_CHAIN_ID,
            from_token_address=BASE_USDC_ADDRESS,
            to_token_address=SOLANA_NATIVE_ADDRESS,
            from_amount_base_units=str(amount_units),
            from_address=base_wallet,
            to_address=solana_wallet,
            from_token_decimals=BASE_USDC_DECIMALS,
            to_token_decimals=9,
            slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
            max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
        )
        if not quote:
            raise ValueError("No safe user-paid SOL reserve route is available")
        if not await self._has_user_funded_base_gas(base_wallet):
            raise ValueError("Ryu's user-funded Base fee reserve is too low")
        if quote.approval_address and not await self._ensure_approval(
            allocation.user_id,
            base_wallet,
            BASE_CHAIN_ID,
            BASE_USDC_ADDRESS,
            quote.approval_address,
            amount_units,
        ):
            raise ValueError("Ryu could not approve the SOL reserve route")

        transaction = quote.transaction_request
        tx_hash = await self._submit_transaction(
            user_id=allocation.user_id,
            wallet_address=base_wallet,
            chain_id=BASE_CHAIN_ID,
            to=transaction.get("to"),
            data=transaction.get("data"),
            value=transaction.get("value") or "0",
        )
        if not tx_hash:
            raise ValueError("Ryu could not submit the SOL reserve route")
        if not await self._wait_for_lifi_completion(
            tx_hash,
            BASE_CHAIN_ID,
            SOLANA_CHAIN_ID,
        ):
            raise ValueError("Ryu's SOL reserve route did not settle")

        await self._adjust_remaining_amount(
            allocation,
            -reserve_usdc,
            count_trade=False,
        )
        return reserve_usdc

    async def _submit_solana_transaction(
        self,
        user_id: str,
        wallet_address: str,
        transaction_base64: str,
        reference_id: str,
    ) -> Optional[str]:
        signing_result = await self.signing_service.sign_solana_transaction(
            user_id=user_id,
            transaction_base64=transaction_base64,
            wallet_address=wallet_address,
            reference_id=reference_id,
        )
        if not signing_result.success:
            logger.error(
                "Ryu Solana transaction failed for %s: %s",
                user_id,
                signing_result.error,
            )
            return None
        return signing_result.transaction_hash

    async def _wait_for_solana_confirmation(
        self,
        transaction_hash: str,
        timeout_seconds: int = 90,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                result = await self._solana_rpc(
                    "getSignatureStatuses",
                    [[transaction_hash], {"searchTransactionHistory": True}],
                )
                status = ((result or {}).get("value") or [None])[0]
                if status:
                    if status.get("err"):
                        return False
                    if status.get("confirmationStatus") in {"confirmed", "finalized"}:
                        return True
            except Exception as e:
                logger.warning(
                    "Could not poll Solana transaction %s: %s",
                    transaction_hash,
                    e,
                )
            await asyncio.sleep(2)
        return False

    def _size_buy_amount_usdc(self, allocation: Any, analysis: RyuCandidateAnalysis) -> float:
        fraction = _parse_position_size_fraction(analysis.position_size_text)
        raw_amount = float(allocation.remaining_amount or 0.0) * fraction
        return max(settings.RYU_SPOT_MIN_TRADE_USDC, min(raw_amount, settings.RYU_SPOT_MAX_TRADE_USDC))

    async def _execute_spot_buy(self, allocation: Any, analysis: RyuCandidateAnalysis) -> bool:
        trade_amount_usdc = self._size_buy_amount_usdc(allocation, analysis)
        if trade_amount_usdc > float(allocation.remaining_amount or 0.0):
            logger.info(f"Skipping Ryu buy for {analysis.symbol}: exceeds remaining allocation")
            return False
        if (
            settings.RYU_REQUIRE_RISK_PLAN
            and not self._has_valid_entry_risk_plan(analysis)
        ):
            logger.warning(
                "Ryu rejected %s: AI entry has no valid stop-and-target plan",
                analysis.symbol,
            )
            return False

        route = (
            {
                "chain_name": analysis.chain_name,
                "lifi_chain_id": analysis.chain_id,
                "token_address": analysis.token_address,
            }
            if analysis.token_address
            else (
                self._select_token_route_from_db(analysis.token)
                if isinstance(analysis.token, dict)
                else self._select_token_route(analysis.token)
            )
        )
        if not route:
            return False
        target_chain_id = int(route["lifi_chain_id"])
        target_chain_name = str(route["chain_name"])
        token_address = str(route["token_address"])
        base_wallet = str(allocation.platform_wallet_address or "")
        solana_wallet = self._solana_wallet_address(allocation)
        destination_wallet = (
            base_wallet
            if target_chain_id == BASE_CHAIN_ID
            else solana_wallet
        )
        if not base_wallet or not destination_wallet:
            logger.warning(
                "Ryu allocation %s is missing its Base or destination wallet",
                allocation.allocation_id,
            )
            return False

        amount_base_units = str(int(trade_amount_usdc * (10 ** BASE_USDC_DECIMALS)))
        needs_solana_gas = (
            target_chain_id == SOLANA_CHAIN_ID
            and not await self._has_user_funded_solana_gas(solana_wallet)
        )
        if (
            needs_solana_gas
            and trade_amount_usdc + settings.RYU_SOLANA_GAS_RESERVE_USDC
            > float(allocation.remaining_amount or 0)
        ):
            logger.info(
                "Skipping Ryu buy for %s: allocation cannot cover the trade and SOL reserve",
                analysis.symbol,
            )
            return False

        quote = await self.lifi_service.get_token_transfer_quote(
            from_chain_id=BASE_CHAIN_ID,
            to_chain_id=target_chain_id,
            from_token_address=BASE_USDC_ADDRESS,
            to_token_address=token_address,
            from_amount_base_units=amount_base_units,
            from_address=base_wallet,
            to_address=destination_wallet,
            from_token_decimals=BASE_USDC_DECIMALS,
            slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
            max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
        )
        if not quote:
            logger.warning(f"No LiFi swap quote for Ryu buy {analysis.symbol}; skipping")
            return False
        if (
            quote.price_impact is not None
            and abs(quote.price_impact) > settings.RYU_MAX_PRICE_IMPACT
        ):
            logger.warning(
                "Ryu rejected %s entry: %.2f%% price impact",
                analysis.symbol,
                quote.price_impact * 100,
            )
            return False

        # Never enter a token unless LI.FI can also quote the full return path
        # from that token back to Base USDC at the quoted output size.
        exit_probe_amount = str(
            max(
                1,
                int(quote.min_to_amount_decimal * (10 ** quote.to_token_decimals)),
            )
        )
        exit_probe = await self.lifi_service.get_token_transfer_quote(
            from_chain_id=target_chain_id,
            to_chain_id=BASE_CHAIN_ID,
            from_token_address=token_address,
            to_token_address=BASE_USDC_ADDRESS,
            from_amount_base_units=exit_probe_amount,
            from_address=destination_wallet,
            to_address=base_wallet,
            from_token_decimals=quote.to_token_decimals,
            to_token_decimals=BASE_USDC_DECIMALS,
            slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
            max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
        )
        if not exit_probe:
            logger.warning(
                "Ryu rejected %s: no executable round-trip route to Base USDC",
                analysis.symbol,
            )
            return False

        if not settings.RYU_LIVE_TRADING_ENABLED:
            logger.info(
                f"[RYU DRY RUN] Would buy {trade_amount_usdc:.2f} USDC of {analysis.symbol} "
                f"(confidence {analysis.confidence:.0%}); live trading disabled"
            )
            return False

        try:
            from kata.services.delegation_service import get_delegation_service

            policy_ok, policy_error = (
                await get_delegation_service().validate_trade_against_policy(
                    allocation.user_id,
                    {
                        "chain_type": "ethereum",
                        "chain": "base",
                        "volume_usd": trade_amount_usdc,
                        "position_size_percent": (
                            trade_amount_usdc
                            / max(float(allocation.allocated_amount or 0), 0.01)
                            * 100
                        ),
                        "leverage": 1,
                        "stop_loss": True,
                    },
                )
            )
            if not policy_ok:
                logger.warning(
                    "Ryu rejected %s by delegation policy: %s",
                    analysis.symbol,
                    policy_error,
                )
                return False
        except Exception as e:
            logger.error(f"Ryu could not validate delegation policy: {e}")
            return False

        if not await self._has_user_funded_base_gas(base_wallet):
            return False
        if target_chain_id == SOLANA_CHAIN_ID:
            try:
                reserve_spent = await self._ensure_solana_gas_from_base(
                    allocation
                )
            except Exception as e:
                logger.error(
                    "Ryu could not prepare Solana fees for %s: %s",
                    analysis.symbol,
                    e,
                )
                return False
            if reserve_spent > 0:
                refreshed_quote = (
                    await self.lifi_service.get_token_transfer_quote(
                        from_chain_id=BASE_CHAIN_ID,
                        to_chain_id=target_chain_id,
                        from_token_address=BASE_USDC_ADDRESS,
                        to_token_address=token_address,
                        from_amount_base_units=amount_base_units,
                        from_address=base_wallet,
                        to_address=destination_wallet,
                        from_token_decimals=BASE_USDC_DECIMALS,
                        slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
                        max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
                    )
                )
                if not refreshed_quote:
                    logger.warning(
                        "Ryu could not refresh %s after funding Solana fees",
                        analysis.symbol,
                    )
                    return False
                if (
                    refreshed_quote.price_impact is not None
                    and abs(refreshed_quote.price_impact)
                    > settings.RYU_MAX_PRICE_IMPACT
                ):
                    logger.warning(
                        "Ryu rejected refreshed %s entry: %.2f%% price impact",
                        analysis.symbol,
                        refreshed_quote.price_impact * 100,
                    )
                    return False
                quote = refreshed_quote

        if quote.approval_address and not await self._ensure_approval(
            allocation.user_id,
            base_wallet,
            BASE_CHAIN_ID,
            BASE_USDC_ADDRESS,
            quote.approval_address,
            int(amount_base_units),
        ):
            return False

        transaction = quote.transaction_request
        tx_hash = await self._submit_transaction(
            user_id=allocation.user_id,
            wallet_address=base_wallet,
            chain_id=BASE_CHAIN_ID,
            to=transaction.get("to"),
            data=transaction.get("data"),
            value=transaction.get("value") or "0",
        )
        if not tx_hash:
            return False

        if target_chain_id == BASE_CHAIN_ID:
            settled = await self._wait_for_evm_confirmation(
                BASE_CHAIN_ID,
                tx_hash,
            )
        else:
            settled = bool(await self._wait_for_lifi_completion(
                tx_hash,
                BASE_CHAIN_ID,
                target_chain_id,
            ))
        if not settled:
            logger.error(
                "Ryu buy %s was submitted but did not confirm; accounting was not changed",
                tx_hash,
            )
            return False

        await self._record_buy(
            allocation,
            analysis,
            quote,
            trade_amount_usdc,
            tx_hash,
            target_chain_name,
            target_chain_id,
            token_address,
        )
        await self._adjust_remaining_amount(allocation, -trade_amount_usdc)
        logger.info(
            "Ryu spot buy executed for %s: %.2f Base USDC -> %s on %s (tx %s)",
            allocation.user_id,
            trade_amount_usdc,
            analysis.symbol,
            target_chain_name,
            tx_hash,
        )
        return True

    async def _execute_spot_exit(self, allocation: Any, position: Dict[str, Any], analysis: RyuCandidateAnalysis) -> bool:
        return await self._execute_position_exit(
            allocation,
            position,
            analysis,
            exit_reason="signal",
            require_live_enabled=True,
        )

    async def _get_position_exit_quote(
        self,
        allocation: Any,
        position: Dict[str, Any],
        token_amount: float,
        amount_base_units: Optional[int] = None,
    ) -> Optional[LiFiSwapQuote]:
        metadata = position.get("position_metadata") or {}
        token_address = metadata.get("contract_address")
        chain_id = int(metadata.get("chain_id", SOLANA_CHAIN_ID))
        token_decimals = int(metadata.get("token_decimals") or 18)
        base_wallet = str(allocation.platform_wallet_address or "")
        solana_wallet = self._solana_wallet_address(allocation)
        source_wallet = base_wallet if chain_id == BASE_CHAIN_ID else solana_wallet
        if (
            not token_address
            or token_amount <= 0
            or not base_wallet
            or not source_wallet
            or chain_id not in {BASE_CHAIN_ID, SOLANA_CHAIN_ID}
        ):
            return None

        return await self.lifi_service.get_token_transfer_quote(
            from_chain_id=chain_id,
            to_chain_id=BASE_CHAIN_ID,
            from_token_address=token_address,
            to_token_address=BASE_USDC_ADDRESS,
            from_amount_base_units=str(
                int(amount_base_units)
                if amount_base_units is not None
                else int(token_amount * (10 ** token_decimals))
            ),
            from_address=source_wallet,
            to_address=base_wallet,
            from_token_decimals=token_decimals,
            to_token_decimals=BASE_USDC_DECIMALS,
            slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
            max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
        )

    async def _execute_position_exit(
        self,
        allocation: Any,
        position: Dict[str, Any],
        analysis: Optional[RyuCandidateAnalysis],
        exit_reason: str,
        require_live_enabled: bool,
        exit_fraction: float = 1.0,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        metadata = position.get("position_metadata") or {}
        token_address = metadata.get("contract_address")
        chain_id = int(metadata.get("chain_id", SOLANA_CHAIN_ID))
        held_amount = float(position.get("size") or 0.0)
        if not token_address or held_amount <= 0:
            return False

        fraction = max(0.0, min(1.0, float(exit_fraction or 0.0)))
        if fraction <= 0:
            return False
        cost_basis_usdc = float(
            position.get("margin_used")
            or position.get("position_value")
            or (float(position.get("entry_price") or 0.0) * held_amount)
        )
        if fraction < 1.0:
            exiting_cost = cost_basis_usdc * fraction
            remaining_cost = cost_basis_usdc - exiting_cost
            if (
                exiting_cost < settings.RYU_SPOT_MIN_TRADE_USDC
                or remaining_cost < settings.RYU_SPOT_MIN_TRADE_USDC
            ):
                logger.info(
                    "Ryu %s partial exit is below the %.2f USDC economic floor; "
                    "closing the full position",
                    position.get("symbol"),
                    settings.RYU_SPOT_MIN_TRADE_USDC,
                )
                fraction = 1.0

        exit_amount = held_amount * fraction
        token_decimals = int(metadata.get("token_decimals") or 18)
        requested_base_units = int(exit_amount * (10 ** token_decimals))
        base_wallet = str(allocation.platform_wallet_address or "")
        solana_wallet = self._solana_wallet_address(allocation)
        source_wallet = (
            base_wallet
            if chain_id == BASE_CHAIN_ID
            else solana_wallet
        )
        if (
            not base_wallet
            or not source_wallet
            or chain_id not in {BASE_CHAIN_ID, SOLANA_CHAIN_ID}
        ):
            return False

        amount_base_units = requested_base_units
        if chain_id == BASE_CHAIN_ID:
            try:
                wallet_balance = await self._get_erc20_balance_base_units(
                    BASE_CHAIN_ID,
                    str(token_address),
                    base_wallet,
                )
            except Exception as e:
                logger.error(
                    "Ryu could not verify %s wallet balance before exit: %s",
                    position.get("symbol"),
                    e,
                )
                return False
            amount_base_units = min(requested_base_units, wallet_balance)
            if amount_base_units <= 0:
                logger.warning(
                    "Ryu cannot exit %s: the linked Base wallet has no token balance",
                    position.get("symbol"),
                )
                return False
            if amount_base_units < requested_base_units:
                logger.info(
                    "Ryu clamped %s exit to the exact wallet balance (%d base units)",
                    position.get("symbol"),
                    amount_base_units,
                )
            exit_amount = amount_base_units / (10 ** token_decimals)

        quote = await self._get_position_exit_quote(
            allocation,
            position,
            exit_amount,
            amount_base_units=amount_base_units,
        )
        if not quote:
            logger.warning(f"No LiFi swap quote for Ryu exit {position.get('symbol')}; skipping")
            return False

        if require_live_enabled and not settings.RYU_LIVE_TRADING_ENABLED:
            logger.info(f"[RYU DRY RUN] Would exit {position.get('symbol')}; live trading disabled")
            return False

        if chain_id == BASE_CHAIN_ID:
            if not await self._has_user_funded_base_gas(base_wallet):
                return False
            if quote.approval_address and not await self._ensure_approval(
                allocation.user_id,
                base_wallet,
                BASE_CHAIN_ID,
                str(token_address),
                quote.approval_address,
                amount_base_units,
            ):
                return False
            transaction = quote.transaction_request
            tx_hash = await self._submit_transaction(
                user_id=allocation.user_id,
                wallet_address=base_wallet,
                chain_id=BASE_CHAIN_ID,
                to=transaction.get("to"),
                data=transaction.get("data"),
                value=transaction.get("value") or "0",
            )
        else:
            if not await self._has_user_funded_solana_gas(solana_wallet):
                return False
            tx_hash = await self._submit_solana_transaction(
                user_id=allocation.user_id,
                wallet_address=solana_wallet,
                transaction_base64=quote.transaction_request.get("data"),
                reference_id=f"ryu-exit-{allocation.allocation_id[:16]}-{int(time.time())}",
            )
        if not tx_hash:
            return False

        status: Optional[Dict[str, Any]] = None
        if chain_id == BASE_CHAIN_ID:
            settled = await self._wait_for_evm_confirmation(
                BASE_CHAIN_ID,
                tx_hash,
            )
        else:
            status = await self._wait_for_lifi_completion(
                tx_hash,
                SOLANA_CHAIN_ID,
                BASE_CHAIN_ID,
            )
            settled = bool(status)
        if not settled:
            logger.error(
                "Ryu exit %s was submitted but did not confirm; accounting was not changed",
                tx_hash,
            )
            return False

        proceeds_usdc = self._received_usdc_from_status(
            status,
            quote.to_amount_decimal,
        )
        await self._record_exit(
            allocation,
            position,
            analysis,
            quote,
            tx_hash,
            exit_reason=exit_reason,
            proceeds_usdc=proceeds_usdc,
            exited_size=exit_amount,
            held_size_before=held_amount,
            metadata_updates=metadata_updates,
        )
        # Exits always return to Base USDC, so realized gains/losses become
        # idle capital in the home pool again.
        await self._adjust_remaining_amount(allocation, proceeds_usdc)
        logger.info(
            "Ryu spot exit executed for %s: %s %.2f%% (%s, tx %s)",
            allocation.user_id,
            position.get("symbol"),
            fraction * 100,
            exit_reason,
            tx_hash,
        )
        return True

    async def monitor_active_positions(
        self,
        active_allocations: List[Any],
    ) -> Dict[str, int]:
        """Mark Ryu holdings and enforce stops, staged targets, and runners."""
        summary = {
            "positions_checked": 0,
            "exits_executed": 0,
            "partial_exits": 0,
            "errors": 0,
        }
        for allocation in active_allocations:
            positions = await self._get_open_positions(allocation.allocation_id)
            for position in positions:
                summary["positions_checked"] += 1
                try:
                    held_size = float(position.get("size") or 0.0)
                    if held_size <= 0:
                        continue
                    quote = await self._get_position_exit_quote(
                        allocation,
                        position,
                        held_size,
                    )
                    if not quote or quote.to_amount_decimal <= 0:
                        logger.warning(
                            "Ryu could not mark %s; no executable Base USDC exit quote",
                            position.get("symbol"),
                        )
                        summary["errors"] += 1
                        continue

                    position_value = float(quote.to_amount_decimal)
                    current_price = position_value / held_size
                    cost_basis = float(
                        position.get("margin_used")
                        or position.get("position_value")
                        or 0.0
                    )
                    unrealized_pnl = position_value - cost_basis
                    unrealized_pnl_percent = (
                        unrealized_pnl / cost_basis * 100.0
                        if cost_basis > 0
                        else 0.0
                    )
                    decision = self._risk_decision(position, current_price)
                    if decision.get("action") == "exit":
                        requested_fraction = float(
                            decision.get("exit_fraction") or 1.0
                        )
                        executed = await self._execute_position_exit(
                            allocation,
                            position,
                            analysis=None,
                            exit_reason=str(decision.get("reason") or "risk"),
                            require_live_enabled=True,
                            exit_fraction=requested_fraction,
                            metadata_updates=decision.get("metadata"),
                        )
                        if executed:
                            summary["exits_executed"] += 1
                            if position.get("is_active", True):
                                summary["partial_exits"] += 1
                        else:
                            summary["errors"] += 1
                        continue

                    await self._persist_position_metadata(
                        position,
                        decision.get("metadata")
                        or dict(position.get("position_metadata") or {}),
                        mark_price=current_price,
                        position_value=position_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_percent=unrealized_pnl_percent,
                    )
                except Exception as e:
                    logger.error(
                        "Ryu risk monitor failed for %s/%s: %s",
                        allocation.allocation_id,
                        position.get("symbol"),
                        e,
                    )
                    summary["errors"] += 1
        return summary

    async def get_trading_readiness(
        self,
        user_id: str,
        ethereum_wallet_address: str,
        solana_wallet_address: str,
    ) -> Dict[str, Any]:
        """Return Base-home capital, wallet delegation, and gas readiness."""
        blocking_reasons: List[str] = []
        base_balances = {"eth": 0.0, "usdc": 0.0}
        solana_balances = {"sol": 0.0, "usdc": 0.0}
        if not ethereum_wallet_address:
            blocking_reasons.append("Floww Base wallet is unavailable")
        else:
            try:
                base_balances = await self.get_base_balances(
                    ethereum_wallet_address
                )
            except Exception as e:
                logger.warning(f"Could not read Ryu Base balances: {e}")
                blocking_reasons.append("Floww Balance is temporarily unavailable")
        if not solana_wallet_address:
            blocking_reasons.append("Ryu Solana wallet is unavailable")
        else:
            try:
                solana_balances = await self.get_solana_balances(
                    solana_wallet_address
                )
            except Exception as e:
                logger.warning(f"Could not read Ryu Solana balances: {e}")

        ethereum_delegated = False
        solana_delegated = False
        try:
            from kata.services.delegation_service import get_delegation_service

            delegation_service = get_delegation_service()
            ethereum_delegation, solana_delegation = await asyncio.gather(
                delegation_service.get_user_delegation(
                    user_id,
                    chain_type="ethereum",
                ),
                delegation_service.get_user_delegation(
                    user_id,
                    chain_type="solana",
                ),
            )
            ethereum_delegated = bool(
                ethereum_delegation
                and ethereum_delegation.is_active
                and ethereum_delegation.wallet_address.lower()
                == ethereum_wallet_address.lower()
            )
            solana_delegated = bool(
                solana_delegation
                and solana_delegation.is_active
                and solana_delegation.wallet_address == solana_wallet_address
            )
        except Exception as e:
            logger.warning(f"Could not check Ryu wallet delegation: {e}")
        if not ethereum_delegated:
            blocking_reasons.append("Ryu needs permission to trade from the Base wallet")
        if not solana_delegated:
            blocking_reasons.append("Ryu needs permission to trade from the Solana wallet")
        if (
            ethereum_wallet_address
            and base_balances["eth"] < settings.RYU_BASE_MIN_GAS_RESERVE_ETH
        ):
            blocking_reasons.append(
                "Ryu's user-funded Base network reserve is too low"
            )

        return {
            "success": True,
            "user_id": user_id,
            "ethereum_wallet_address": ethereum_wallet_address,
            "solana_wallet_address": solana_wallet_address,
            "home_chain": "base",
            "base_usdc_balance": base_balances["usdc"],
            "base_eth_balance": base_balances["eth"],
            "minimum_base_eth_reserve": settings.RYU_BASE_MIN_GAS_RESERVE_ETH,
            "solana_usdc_balance": solana_balances["usdc"],
            "sol_balance": solana_balances["sol"],
            "minimum_sol_reserve": settings.RYU_SOLANA_MIN_GAS_RESERVE_SOL,
            "ethereum_delegated": ethereum_delegated,
            "solana_delegated": solana_delegated,
            "delegated": ethereum_delegated and solana_delegated,
            "live_trading_enabled": settings.RYU_LIVE_TRADING_ENABLED,
            "ready_for_trading": len(blocking_reasons) == 0,
            "blocking_reasons": blocking_reasons,
            "fee_payer": "user",
        }

    async def withdraw_available_usdc_to_base(
        self,
        allocation: Any,
        amount_usdc: float,
    ) -> Dict[str, Any]:
        """Bridge unused Ryu Solana USDC back to the user's Base address."""
        solana_wallet = self._solana_wallet_address(allocation)
        base_wallet = str(allocation.platform_wallet_address or "")
        if not solana_wallet or not base_wallet:
            return {
                "success": False,
                "error": "Ryu's linked Base and Solana wallets are unavailable",
            }
        if not await self._has_user_funded_solana_gas(solana_wallet):
            return {
                "success": False,
                "error": (
                    "Ryu's user-funded SOL fee reserve is too low to return funds. "
                    "Add funds from Base to replenish it first."
                ),
            }
        try:
            balances = await self.get_solana_balances(solana_wallet)
        except Exception as e:
            return {"success": False, "error": f"Could not read Solana USDC balance: {e}"}
        if amount_usdc > balances["usdc"] + 0.000001:
            return {
                "success": False,
                "error": f"Only ${balances['usdc']:.2f} Solana USDC is currently liquid",
            }

        amount_units = str(
            int(amount_usdc * (10 ** SOLANA_USDC_DECIMALS))
        )
        quote = await self.lifi_service.get_token_transfer_quote(
            from_chain_id=SOLANA_CHAIN_ID,
            to_chain_id=BASE_CHAIN_ID,
            from_token_address=SOLANA_USDC_ADDRESS,
            to_token_address=BASE_USDC_ADDRESS,
            from_amount_base_units=amount_units,
            from_address=solana_wallet,
            to_address=base_wallet,
            from_token_decimals=SOLANA_USDC_DECIMALS,
            to_token_decimals=BASE_USDC_DECIMALS,
            slippage=settings.RYU_SPOT_SLIPPAGE_TOLERANCE,
            max_price_impact=settings.RYU_MAX_PRICE_IMPACT,
        )
        if not quote:
            return {
                "success": False,
                "error": "No safe Solana USDC to Base USDC return route is available",
            }
        tx_hash = await self._submit_solana_transaction(
            user_id=allocation.user_id,
            wallet_address=solana_wallet,
            transaction_base64=quote.transaction_request.get("data"),
            reference_id=f"ryu-withdraw-{allocation.allocation_id[:16]}-{int(time.time())}",
        )
        if not tx_hash:
            return {"success": False, "error": "Ryu return transaction failed"}
        return {
            "success": True,
            "transaction_hash": tx_hash,
            "provider": quote.tool_name,
            "estimated_base_usdc": quote.to_amount_decimal,
            "minimum_base_usdc": quote.min_to_amount_decimal,
            "network": "Base",
            "fee_payer": "user",
        }

    async def liquidate_all_positions(self, allocation: Any) -> Dict[str, Any]:
        """Close every Ryu holding back to Base USDC for a user stop."""
        positions = await self._get_open_positions(allocation.allocation_id)
        if not positions:
            return {"success": True, "closed": 0}

        closed = 0
        for position in positions:
            exited = await self._execute_position_exit(
                allocation,
                position,
                analysis=None,
                exit_reason="user_stop",
                require_live_enabled=False,
            )
            if not exited:
                return {
                    "success": False,
                    "error": (
                        f"Could not safely return {position.get('symbol')} to Base USDC. "
                        "Ryu remains paused; retry Stop after the route settles."
                    ),
                }
            closed += 1
        return {"success": True, "closed": closed}

    async def _adjust_remaining_amount(
        self,
        allocation: Any,
        delta_usdc: float,
        count_trade: bool = True,
    ) -> None:
        """Update remaining_amount in-memory and in the DB, mirroring Yuki's trade-budget pattern."""
        allocation.remaining_amount = float(allocation.remaining_amount or 0.0) + delta_usdc
        allocation.total_trades = (
            int(getattr(allocation, "total_trades", 0) or 0)
            + (1 if count_trade else 0)
        )
        try:
            self.supabase.table("agent_allocations").update({
                "remaining_amount": allocation.remaining_amount,
                "total_trades": allocation.total_trades,
                "last_trade_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            }).eq("allocation_id", allocation.allocation_id).execute()
        except Exception as e:
            logger.error(f"Failed to persist remaining_amount for allocation {allocation.allocation_id}: {e}")

    # ------------------------------------------------------------------
    # DB recording
    # ------------------------------------------------------------------

    async def _record_buy(
        self,
        allocation: Any,
        analysis: RyuCandidateAnalysis,
        quote: LiFiSwapQuote,
        trade_amount_usdc: float,
        tx_hash: str,
        chain_name: str,
        chain_id: int,
        token_address: str,
    ) -> None:
        try:
            execution_price = (
                trade_amount_usdc / quote.to_amount_decimal
                if quote.to_amount_decimal > 0
                else analysis.current_price
            )
            token_row = analysis.token if isinstance(analysis.token, dict) else {}
            risk_plan = self._build_entry_risk_plan(
                analysis,
                execution_price,
                quote.to_amount_decimal,
            )
            lifecycle_metadata = {
                "contract_address": token_address,
                "chain_id": chain_id,
                "chain": chain_name,
                "home_chain": "base",
                "home_token": "USDC",
                "token_decimals": quote.to_token_decimals,
                "token_name": token_row.get("name") or analysis.symbol,
                "coingecko_id": token_row.get("coingecko_id") or "",
                "lifi_tool": quote.tool_name,
                "lifi_quote_id": quote.quote_id,
                "action_summary": analysis.action_summary,
                "purchase_thesis": analysis.reasoning,
                "entry_selection_score": analysis.selection_score,
                "latest_selection_score": analysis.selection_score,
                "entry_opportunity_score": analysis.opportunity_score,
                "latest_opportunity_score": analysis.opportunity_score,
                "latest_confidence": analysis.confidence,
                "latest_trade_intent": analysis.trade_intent,
                "last_analyzed_at": datetime.now(timezone.utc).isoformat(),
                "risk_plan": risk_plan or {},
            }
            trade_row = {
                "user_id": allocation.user_id,
                "allocation_id": allocation.allocation_id,
                "agent_type": "ryu",
                "trade_type": "open_long",
                "symbol": analysis.symbol,
                "side": "buy",
                "entry_price": execution_price,
                "position_size": quote.to_amount_decimal,
                "leverage": 1,
                "trade_amount": trade_amount_usdc,
                "status": "filled",
                "tx_hash": tx_hash,
                "signal_confidence": analysis.confidence,
                "signal_reasoning": analysis.reasoning,
                "trade_metadata": lifecycle_metadata,
            }
            trade_result = self.supabase.table("agent_trades").insert(trade_row).execute()
            trade_id = (trade_result.data or [{}])[0].get("id")

            # agent_positions was built around Yuki's margin model -- size/position_value/
            # current_price/margin_used/leverage are all NOT NULL there. Spot holdings map
            # onto it as an unleveraged, fully-margined "long": margin_used == the USDC
            # spent and there is no liquidation price. The periodic Ryu risk
            # monitor refreshes current value and enforces the persisted plan.
            position_row = {
                "user_id": allocation.user_id,
                "allocation_id": allocation.allocation_id,
                "trade_id": trade_id,
                "agent_type": "ryu",
                "symbol": analysis.symbol,
                "side": "long",
                "size": quote.to_amount_decimal,
                "entry_price": execution_price,
                "current_price": execution_price,
                "leverage": 1,
                "position_value": trade_amount_usdc,
                "unrealized_pnl": 0,
                "unrealized_pnl_percent": 0,
                "margin_used": trade_amount_usdc,
                "is_active": True,
                "position_metadata": trade_row["trade_metadata"],
            }
            self.supabase.table("agent_positions").insert(position_row).execute()
        except Exception as e:
            logger.error(f"Failed to record Ryu buy for {allocation.user_id}: {e}")

    async def _record_exit(
        self,
        allocation: Any,
        position: Dict[str, Any],
        analysis: Optional[RyuCandidateAnalysis],
        quote: LiFiSwapQuote,
        tx_hash: str,
        exit_reason: str = "signal",
        proceeds_usdc: Optional[float] = None,
        exited_size: Optional[float] = None,
        held_size_before: Optional[float] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            entry_price = float(position.get("entry_price") or 0.0)
            exit_value_usdc = (
                float(proceeds_usdc)
                if proceeds_usdc is not None
                else quote.to_amount_decimal
            )
            held_size = float(
                held_size_before
                if held_size_before is not None
                else position.get("size") or 0.0
            )
            exit_size = min(
                held_size,
                float(exited_size if exited_size is not None else held_size),
            )
            exit_fraction = exit_size / held_size if held_size > 0 else 1.0
            total_cost_basis_usdc = float(
                position.get("margin_used")
                or position.get("position_value")
                or (entry_price * held_size)
            )
            exited_cost_basis_usdc = total_cost_basis_usdc * exit_fraction
            realized_pnl = (
                exit_value_usdc - exited_cost_basis_usdc
                if exited_cost_basis_usdc > 0
                else None
            )
            remaining_size = max(0.0, held_size - exit_size)
            remaining_cost_basis = max(
                0.0,
                total_cost_basis_usdc - exited_cost_basis_usdc,
            )
            is_full_exit = remaining_size <= max(1e-18, held_size * 1e-9)
            execution_price = (
                exit_value_usdc / exit_size
                if exit_size > 0
                else entry_price
            )

            self.supabase.table("agent_trades").insert({
                "user_id": allocation.user_id,
                "allocation_id": allocation.allocation_id,
                "agent_type": "ryu",
                "trade_type": "close_long",
                "symbol": position.get("symbol"),
                "side": "sell",
                "entry_price": entry_price,
                "exit_price": execution_price,
                "position_size": exit_size,
                "leverage": 1,
                "trade_amount": exit_value_usdc,
                "realized_pnl": realized_pnl,
                "status": "closed",
                "tx_hash": tx_hash,
                "signal_confidence": analysis.confidence if analysis else None,
                "signal_reasoning": (
                    analysis.reasoning
                    if analysis
                    else f"Ryu lifecycle exit: {exit_reason}"
                ),
                "trade_metadata": {
                    "lifi_tool": quote.tool_name,
                    "lifi_quote_id": quote.quote_id,
                    "exit_reason": exit_reason,
                    "partial_exit": not is_full_exit,
                    "remaining_size": 0.0 if is_full_exit else remaining_size,
                    "destination_chain": "base",
                    "destination_token": "USDC",
                },
            }).execute()

            metadata = dict(metadata_updates or position.get("position_metadata") or {})
            metadata["last_exit_reason"] = exit_reason
            metadata["last_exit_at"] = datetime.now(timezone.utc).isoformat()
            metadata["realized_pnl_usdc"] = (
                float(metadata.get("realized_pnl_usdc") or 0.0)
                + float(realized_pnl or 0.0)
            )
            if is_full_exit:
                metadata["closed_reason"] = exit_reason
                metadata["closed_at"] = datetime.now(timezone.utc).isoformat()

            remaining_value = remaining_size * execution_price
            remaining_unrealized = (
                remaining_value - remaining_cost_basis
                if not is_full_exit
                else 0.0
            )
            remaining_unrealized_percent = (
                remaining_unrealized / remaining_cost_basis * 100.0
                if remaining_cost_basis > 0 and not is_full_exit
                else 0.0
            )
            update_payload = {
                "is_active": not is_full_exit,
                "size": 0.0 if is_full_exit else remaining_size,
                "current_price": execution_price,
                "position_value": 0.0 if is_full_exit else remaining_value,
                "margin_used": 0.0 if is_full_exit else remaining_cost_basis,
                "unrealized_pnl": remaining_unrealized,
                "unrealized_pnl_percent": remaining_unrealized_percent,
                "position_metadata": metadata,
            }
            self.supabase.table("agent_positions").update(update_payload).eq(
                "id",
                position.get("id"),
            ).execute()
            position.update(update_payload)
        except Exception as e:
            logger.error(f"Failed to record Ryu exit for {allocation.user_id}: {e}")

    async def close(self):
        try:
            await self._rpc_client.aclose()
        except Exception:
            pass


_ryu_spot_trading_service: Optional[RyuSpotTradingService] = None


def get_ryu_spot_trading_service() -> RyuSpotTradingService:
    """Get or create the singleton Ryu spot trading service instance."""
    global _ryu_spot_trading_service
    if _ryu_spot_trading_service is None:
        _ryu_spot_trading_service = RyuSpotTradingService()
    return _ryu_spot_trading_service
