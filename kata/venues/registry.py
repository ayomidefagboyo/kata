"""Venue registry.

Resolves a venue name to a live `PerpVenue`. Agents ask for a venue by name and
never import a concrete adapter.
"""

import logging
from typing import Callable, Dict, List

from kata.venues.base import PerpVenue

logger = logging.getLogger(__name__)

_BUILDERS: Dict[str, Callable[[], PerpVenue]] = {}
_INSTANCES: Dict[str, PerpVenue] = {}


def register(name: str, builder: Callable[[], PerpVenue]) -> None:
    _BUILDERS[name] = builder


def available() -> List[str]:
    return sorted(_BUILDERS)


def get_venue(name: str) -> PerpVenue:
    if name in _INSTANCES:
        return _INSTANCES[name]
    if name not in _BUILDERS:
        raise KeyError(f"Unknown venue '{name}'. Registered: {available()}")
    venue = _BUILDERS[name]()
    _INSTANCES[name] = venue
    return venue


async def connect_all() -> Dict[str, bool]:
    """Connect every registered venue, tolerating individual failures."""
    results: Dict[str, bool] = {}
    for name in available():
        try:
            results[name] = await get_venue(name).connect()
        except Exception:
            logger.exception("Venue %s failed to connect", name)
            results[name] = False
    return results


def _register_defaults() -> None:
    from kata.venues.drift import DriftVenue
    from kata.venues.hyperliquid import HyperliquidVenue

    register("hyperliquid", HyperliquidVenue)
    register("drift", DriftVenue)


_register_defaults()
