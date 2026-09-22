"""
Hyperliquid client factory with resilient construction.

The SDK's Info.__init__ (and Exchange, which builds an Info internally) parses
the live spot metadata at construction time. Hyperliquid has shipped spot pairs
whose token indices fall outside the returned token list, which makes the
constructor raise IndexError and leaves the platform without market data.

These helpers first try normal construction, then retry with a sanitized
spot_meta payload (malformed pairs dropped). Perp trading never depends on the
dropped spot pairs.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

logger = logging.getLogger(__name__)

MAINNET_API_URL = "https://api.hyperliquid.xyz"
_PERP_DEX_CACHE_TTL_SECONDS = 600.0
_perp_dex_cache: Dict[str, Tuple[float, List[str]]] = {}


_META_CACHE_TTL_SECONDS = 600.0
_meta_cache: Dict[str, Tuple[float, Any]] = {}
_spot_meta_cache: Dict[str, Tuple[float, Any]] = {}


def fetch_perp_dex_names(base_url: Optional[str], force_refresh: bool = False) -> List[str]:
    """Return every live perp DEX name, including ``""`` for the main DEX."""
    api_root = (base_url or MAINNET_API_URL).rstrip("/")
    cached = _perp_dex_cache.get(api_root)
    if cached and not force_refresh and time.monotonic() - cached[0] < _PERP_DEX_CACHE_TTL_SECONDS:
        return list(cached[1])

    response = httpx.post(f"{api_root}/info", json={"type": "perpDexs"}, timeout=15.0)
    response.raise_for_status()
    payload = response.json() or []
    names = [""]
    for dex in payload:
        name = str((dex or {}).get("name") or "").strip() if isinstance(dex, dict) else ""
        if name and name not in names:
            names.append(name)

    _perp_dex_cache[api_root] = (time.monotonic(), names)
    return list(names)


def _load_perp_dex_names(base_url: Optional[str]) -> Optional[List[str]]:
    """Best-effort DEX discovery; falling back to the main DEX keeps startup resilient."""
    try:
        return fetch_perp_dex_names(base_url)
    except Exception as exc:
        logger.warning(f"Could not discover Hyperliquid HIP-3 DEXs; using main DEX only: {exc}")
        return None


def fetch_sanitized_spot_meta(base_url: Optional[str]) -> Dict[str, Any]:
    """Fetch spot metadata and drop pairs referencing out-of-range tokens."""
    api_root = (base_url or MAINNET_API_URL).rstrip("/")
    cached = _spot_meta_cache.get(api_root)
    if cached and time.monotonic() - cached[0] < _META_CACHE_TTL_SECONDS:
        return cached[1]

    response = httpx.post(f"{api_root}/info", json={"type": "spotMeta"}, timeout=15.0)
    response.raise_for_status()
    spot_meta = response.json()

    tokens = spot_meta.get("tokens", [])
    token_count = len(tokens)
    safe_universe = []
    dropped = 0
    for pair in spot_meta.get("universe", []):
        refs = pair.get("tokens", [])
        if (
            len(refs) == 2
            and all(isinstance(ref, int) and 0 <= ref < token_count for ref in refs)
        ):
            safe_universe.append(pair)
        else:
            dropped += 1

    if dropped:
        logger.warning(f"Sanitized Hyperliquid spot metadata: dropped {dropped} malformed spot pairs")
    spot_meta["universe"] = safe_universe
    _spot_meta_cache[api_root] = (time.monotonic(), spot_meta)
    return spot_meta


def _extract_dict_meta(val: Any) -> Optional[Dict[str, Any]]:
    """safely resolve meta/spot_meta whether it is a method or property dict."""
    if val is None:
        return None
    try:
        res = val() if callable(val) else val
        return res if isinstance(res, dict) else None
    except Exception:
        return None


def create_info_client(
    base_url: Optional[str] = None,
    skip_ws: bool = True,
    include_hip3: bool = False,
) -> Info:
    """
    Create an Info client, tolerating malformed spot metadata.

    skip_ws defaults to True: nothing in the app uses the SDK's built-in
    websocket (we run our own HyperliquidWebSocketClient), and every
    Info(skip_ws=False) instance spawns a persistent websocket thread with
    buffers and a ping loop - measurable memory per client on a 512MB instance.
    """
    api_root = (base_url or MAINNET_API_URL).rstrip("/")
    perp_dexs = _load_perp_dex_names(base_url) if include_hip3 else None
    def build() -> Info:
        try:
            info = Info(base_url=base_url, skip_ws=skip_ws, perp_dexs=perp_dexs)
            m = _extract_dict_meta(getattr(info, "meta", None))
            if m:
                _meta_cache[api_root] = (time.monotonic(), m)
            sm = _extract_dict_meta(getattr(info, "spot_meta", None))
            if sm:
                _spot_meta_cache[api_root] = (time.monotonic(), sm)
            return info
        except (IndexError, KeyError) as e:
            logger.warning(f"Info client init failed on spot metadata ({e}); retrying with sanitized spot_meta")
            spot_meta = fetch_sanitized_spot_meta(base_url)
            info = Info(
                base_url=base_url,
                skip_ws=skip_ws,
                spot_meta=spot_meta,
                perp_dexs=perp_dexs,
            )
            m = _extract_dict_meta(getattr(info, "meta", None))
            if m:
                _meta_cache[api_root] = (time.monotonic(), m)
            return info

    for attempt in range(5):
        try:
            return build()
        except Exception as exc:
            if "429" not in str(exc) or attempt == 4:
                raise
            delay = float(2.0 ** attempt)
            logger.warning(
                f"Hyperliquid rate-limited Info client startup; retrying in {delay:.1f}s (attempt {attempt + 1}/5)"
            )
            time.sleep(delay)

    raise RuntimeError("Could not create Hyperliquid Info client")


def create_exchange_client(
    wallet: Any,
    base_url: Optional[str] = None,
    account_address: Optional[str] = None,
    perp_dexs: Optional[List[str]] = None,
    meta: Optional[Any] = None,
    spot_meta: Optional[Any] = None,
) -> Exchange:
    """Create a signer with only the requested perp DEX mappings.

    Reuses cached meta and spot_meta payloads when available to prevent redundant
    /info network requests that trigger HTTP 429 rate limits from Hyperliquid CloudFront.
    """
    api_root = (base_url or MAINNET_API_URL).rstrip("/")

    # Resolve cached meta / spot_meta if not provided by caller
    meta = _extract_dict_meta(meta)
    if meta is None:
        cached_meta = _meta_cache.get(api_root)
        if cached_meta and time.monotonic() - cached_meta[0] < _META_CACHE_TTL_SECONDS:
            meta = _extract_dict_meta(cached_meta[1])

    spot_meta = _extract_dict_meta(spot_meta)
    if spot_meta is None:
        cached_spot = _spot_meta_cache.get(api_root)
        if cached_spot and time.monotonic() - cached_spot[0] < _META_CACHE_TTL_SECONDS:
            spot_meta = _extract_dict_meta(cached_spot[1])

    def build() -> Exchange:
        try:
            return Exchange(
                wallet=wallet,
                base_url=base_url,
                account_address=account_address,
                perp_dexs=perp_dexs,
                meta=meta,
                spot_meta=spot_meta,
            )
        except (IndexError, KeyError) as e:
            logger.warning(f"Exchange client init failed on spot metadata ({e}); retrying with sanitized spot_meta")
            sanitized_spot = fetch_sanitized_spot_meta(base_url)
            return Exchange(
                wallet=wallet,
                base_url=base_url,
                account_address=account_address,
                meta=meta,
                spot_meta=sanitized_spot,
                perp_dexs=perp_dexs,
            )

    for attempt in range(5):
        try:
            return build()
        except Exception as exc:
            if "429" not in str(exc) or attempt == 4:
                raise
            delay = float(2.0 ** attempt)
            logger.warning(
                f"Hyperliquid rate-limited Exchange client startup; retrying in {delay:.1f}s (attempt {attempt + 1}/5)"
            )
            time.sleep(delay)

    raise RuntimeError("Could not create Hyperliquid Exchange client")
