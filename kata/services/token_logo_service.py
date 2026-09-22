"""Resolve token logos from market identity instead of symbol-specific exceptions."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import httpx


_QUOTE_SUFFIXES = (
    "/USDT:USDT",
    "/USDT",
    "-USDT",
    "USDT",
    "/USD",
    "-USD",
    "-PERP",
    "PERP",
)

_IDENTITY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "asset",
    "class",
    "common",
    "corp",
    "corporation",
    "develops",
    "dex",
    "exchange",
    "fund",
    "in",
    "inc",
    "index",
    "is",
    "market",
    "of",
    "on",
    "one",
    "perp",
    "perpetual",
    "price",
    "quoted",
    "references",
    "share",
    "stock",
    "the",
    "token",
    "tokenized",
    "usd",
}


def normalize_market_symbol(symbol: str) -> str:
    """Return a stable market identity while preserving a HIP-3 dex prefix."""
    normalized = str(symbol or "").strip().upper()
    for suffix in _QUOTE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.strip()


def base_token_symbol(symbol: str) -> str:
    """Return the display ticker used by metadata providers."""
    normalized = normalize_market_symbol(symbol)
    base = normalized.rsplit(":", 1)[-1]
    if base.startswith("1000") and len(base) > 4:
        base = base[4:]
    elif base.startswith("K") and len(base) > 3:
        base = base[1:]
    return base


def _identity_terms(value: Any) -> set[str]:
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(term) > 1 and term not in _IDENTITY_STOP_WORDS
    }
    return terms


def _annotation_terms(annotation: Optional[Dict[str, Any]], ticker: str) -> set[str]:
    if not annotation:
        return set()
    terms = _identity_terms(annotation.get("displayName"))
    terms.update(_identity_terms(annotation.get("description")))
    for keyword in annotation.get("keywords") or []:
        terms.update(_identity_terms(keyword))
    terms.discard(ticker.lower())
    return terms


def _candidate_image(candidate: Dict[str, Any]) -> Optional[str]:
    image = candidate.get("large") or candidate.get("small") or candidate.get("thumb")
    if isinstance(image, str) and image.startswith(("https://", "http://")):
        return image
    return None


def select_logo_candidate(
    coins: Iterable[Dict[str, Any]],
    ticker: str,
    annotation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Choose an exact-ticker result and use venue metadata to break ambiguity.

    Search providers can return unrelated high-ranking assets for short or newly
    listed tickers. Requiring an exact ticker prevents quote assets such as USDT
    from being selected, while a HIP-3 annotation supplies the semantic identity
    needed to distinguish multiple exact-ticker results.
    """
    normalized_ticker = base_token_symbol(ticker)
    semantic_terms = _annotation_terms(annotation, normalized_ticker)
    ranked: list[tuple[int, int, str, Dict[str, Any]]] = []

    for candidate in coins:
        if str(candidate.get("symbol") or "").strip().upper() != normalized_ticker:
            continue
        if not _candidate_image(candidate):
            continue

        candidate_terms = _identity_terms(candidate.get("name"))
        candidate_terms.update(_identity_terms(str(candidate.get("id") or "").replace("-", " ")))
        semantic_overlap = len(candidate_terms.intersection(semantic_terms))
        if semantic_terms and semantic_overlap == 0:
            continue

        try:
            market_cap_rank = int(candidate.get("market_cap_rank") or 10**9)
        except (TypeError, ValueError):
            market_cap_rank = 10**9
        ranked.append((-semantic_overlap, market_cap_rank, str(candidate.get("id") or ""), candidate))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3]


@dataclass(frozen=True)
class ResolvedTokenLogo:
    url: str
    identity_symbol: str
    source: str
    provider_id: Optional[str] = None


class TokenLogoResolver:
    """Shared logo resolver used by generation, the scanner and agent history."""

    def __init__(self) -> None:
        self._cache: Dict[str, tuple[float, Optional[ResolvedTokenLogo]]] = {}
        self._positive_ttl_seconds = 7 * 24 * 60 * 60
        self._negative_ttl_seconds = 60 * 60

    @staticmethod
    def _usable_stored_logo(logo_url: Optional[str]) -> bool:
        return bool(
            isinstance(logo_url, str)
            and logo_url.strip().startswith(("https://", "http://"))
            and "token-image" not in logo_url
        )

    async def _fetch_hyperliquid_annotation(
        self,
        market_symbol: str,
        client: httpx.AsyncClient,
    ) -> Optional[Dict[str, Any]]:
        if ":" not in market_symbol:
            return None
        try:
            response = await client.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "perpAnnotation", "coin": market_symbol.lower().split(":", 1)[0] + ":" + market_symbol.split(":", 1)[1]},
                timeout=5.0,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    async def _search_coingecko(
        self,
        ticker: str,
        annotation: Optional[Dict[str, Any]],
        client: httpx.AsyncClient,
    ) -> Optional[ResolvedTokenLogo]:
        api_key = str(os.getenv("COINGECKO_API_KEY") or "").strip()
        endpoints = []
        if api_key:
            endpoints.append((
                "https://pro-api.coingecko.com/api/v3/search",
                {"x-cg-pro-api-key": api_key},
            ))
        endpoints.append(("https://api.coingecko.com/api/v3/search", {}))

        for endpoint, headers in endpoints:
            try:
                response = await client.get(
                    endpoint,
                    params={"query": ticker},
                    headers=headers,
                    timeout=10.0,
                )
                if response.status_code != 200:
                    continue
                payload = response.json()
                candidate = select_logo_candidate(payload.get("coins") or [], ticker, annotation)
                if not candidate:
                    continue
                image = _candidate_image(candidate)
                if not image:
                    continue
                return ResolvedTokenLogo(
                    url=image,
                    identity_symbol=ticker,
                    source="coingecko_search",
                    provider_id=str(candidate.get("id") or "") or None,
                )
            except Exception:
                continue
        return None

    async def _try_binance_cdn(
        self,
        market_symbol: str,
        ticker: str,
        client: httpx.AsyncClient,
    ) -> Optional[ResolvedTokenLogo]:
        # Builder-deployed perp tickers can overlap unrelated crypto assets.
        # Only use symbol-only CDN lookup for the default, unnamespaced market.
        if ":" in market_symbol:
            return None
        url = f"https://bin.bnbstatic.com/static/assets/logos/{ticker}.png"
        try:
            response = await client.head(url, timeout=3.0)
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                return ResolvedTokenLogo(url=url, identity_symbol=market_symbol, source="binance_cdn")
        except Exception:
            return None
        return None

    async def resolve(
        self,
        symbol: str,
        *,
        stored_logo_url: Optional[str] = None,
    ) -> Optional[ResolvedTokenLogo]:
        market_symbol = normalize_market_symbol(symbol)
        ticker = base_token_symbol(market_symbol)
        if not market_symbol or not ticker:
            return None

        now = time.monotonic()
        cached = self._cache.get(market_symbol)
        if cached:
            cached_at, value = cached
            ttl = (
                self._positive_ttl_seconds
                if value and value.source != "stored_signal"
                else self._negative_ttl_seconds
            )
            # Do not let a previous miss hide a logo that has since been stored
            # with a platform signal.
            stored_logo_became_available = value is None and self._usable_stored_logo(stored_logo_url)
            if now - cached_at < ttl and not stored_logo_became_available:
                return value

        async with httpx.AsyncClient(verify=False, follow_redirects=True) as client:
            annotation = await self._fetch_hyperliquid_annotation(market_symbol, client)
            # A namespaced HIP-3 market can share a ticker with an unrelated
            # crypto asset. If its venue metadata is temporarily unavailable,
            # prefer a stored logo or no logo over a symbol-only guess.
            has_identity_context = bool(_annotation_terms(annotation, ticker))
            resolved = None
            if ":" not in market_symbol or has_identity_context:
                resolved = await self._search_coingecko(ticker, annotation, client)
            if not resolved:
                resolved = await self._try_binance_cdn(market_symbol, ticker, client)

        if not resolved and self._usable_stored_logo(stored_logo_url):
            resolved = ResolvedTokenLogo(
                url=str(stored_logo_url).strip(),
                identity_symbol=market_symbol,
                source="stored_signal",
            )

        self._cache[market_symbol] = (now, resolved)
        return resolved

_token_logo_resolver: Optional[TokenLogoResolver] = None


def get_token_logo_resolver() -> TokenLogoResolver:
    global _token_logo_resolver
    if _token_logo_resolver is None:
        _token_logo_resolver = TokenLogoResolver()
    return _token_logo_resolver
