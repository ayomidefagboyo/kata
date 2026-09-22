"""
Shared market-structure and order-flow context for Ryu and platform signals.

This module is intentionally data-only: it does not fetch external data and it
does not invent fallback levels. Callers pass whatever real candles/order-flow
they have; missing components are omitted from the prompt and chart overlays.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _finite_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> Optional[float]:
    parsed = _finite_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _pct_distance(reference: float, price: float) -> Optional[float]:
    if reference <= 0 or price <= 0:
        return None
    return ((price - reference) / reference) * 100.0


def _fmt_price(value: Any) -> str:
    parsed = _positive_float(value)
    if parsed is None:
        return "n/a"
    if parsed >= 1000:
        return f"${parsed:,.2f}"
    if parsed >= 1:
        return f"${parsed:.4f}"
    if parsed >= 0.01:
        return f"${parsed:.6f}"
    return f"${parsed:.8f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    parsed = _finite_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:+.{digits}f}%"


def _normalise_candles(candles: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for candle in candles or []:
        if isinstance(candle, dict):
            timestamp = candle.get("timestamp") or candle.get("time") or candle.get("t")
            row = {
                "timestamp": timestamp,
                "open": _positive_float(candle.get("open") or candle.get("o")),
                "high": _positive_float(candle.get("high") or candle.get("h")),
                "low": _positive_float(candle.get("low") or candle.get("l")),
                "close": _positive_float(candle.get("close") or candle.get("c")),
                "volume": _finite_float(candle.get("volume") or candle.get("v")),
            }
        elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
            row = {
                "timestamp": candle[0],
                "open": _positive_float(candle[1]),
                "high": _positive_float(candle[2]),
                "low": _positive_float(candle[3]),
                "close": _positive_float(candle[4]),
                "volume": _finite_float(candle[5]),
            }
        else:
            continue

        if row["open"] and row["high"] and row["low"] and row["close"]:
            normalised.append(row)

    return normalised


def _average_true_range(candles: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
    if len(candles) < 2:
        return None
    ranges: List[float] = []
    for index in range(1, len(candles)):
        current = candles[index]
        previous = candles[index - 1]
        high = _positive_float(current.get("high"))
        low = _positive_float(current.get("low"))
        prev_close = _positive_float(previous.get("close"))
        if high is None or low is None or prev_close is None:
            continue
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if not ranges:
        return None
    return sum(ranges[-period:]) / min(period, len(ranges))


class MarketStructureContextService:
    """Build strict market-structure context from real inputs only."""

    def build_context(
        self,
        *,
        symbol: str,
        current_price: float,
        candles: Optional[Iterable[Any]] = None,
        timeframe: str = "4h",
        orderbook: Optional[Dict[str, Any]] = None,
        funding_rate: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
        top_trader_long_ratio: Optional[float] = None,
        open_interest: Optional[float] = None,
        open_interest_change_24h: Optional[float] = None,
        taker_flow: Optional[Dict[str, Any]] = None,
        technical_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        price = _positive_float(current_price)
        if price is None:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data_used": [],
                "prompt_block": "",
                "chart_overlays": [],
            }

        candle_rows = _normalise_candles(candles)
        zones = self.detect_supply_demand_zones(
            candle_rows,
            current_price=price,
            timeframe=timeframe,
        )

        order_flow, order_flow_used = self._build_order_flow(
            price=price,
            orderbook=orderbook,
            funding_rate=funding_rate,
            long_short_ratio=long_short_ratio,
            top_trader_long_ratio=top_trader_long_ratio,
            open_interest=open_interest,
            open_interest_change_24h=open_interest_change_24h,
            taker_flow=taker_flow,
            technical_snapshot=technical_snapshot,
        )
        structure = self._build_structure_snapshot(
            price=price,
            candles=candle_rows,
            zones=zones,
            technical_snapshot=technical_snapshot,
        )
        chart_overlays = self._build_chart_overlays(zones, structure)

        data_used = []
        if candle_rows:
            data_used.append(f"{len(candle_rows)}_{timeframe}_candles")
        data_used.extend(order_flow_used)

        context = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": price,
            "data_used": data_used,
            "supply_zones": zones["supply"],
            "demand_zones": zones["demand"],
            "market_structure": structure,
            "order_flow": order_flow,
            "chart_overlays": chart_overlays,
        }
        context["prompt_block"] = self.to_prompt_block(context)
        return context

    def detect_supply_demand_zones(
        self,
        candles: List[Dict[str, Any]],
        *,
        current_price: float,
        timeframe: str,
        pivot_window: int = 3,
        max_zones_per_side: int = 4,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if len(candles) < pivot_window * 2 + 10:
            return {"supply": [], "demand": []}

        recent = candles[-160:]
        atr = _average_true_range(recent)
        if atr is None or atr <= 0:
            return {"supply": [], "demand": []}

        volumes = [_finite_float(c.get("volume")) for c in recent]
        valid_volumes = [v for v in volumes if v is not None and v > 0]
        median_volume = median(valid_volumes) if valid_volumes else None
        zone_width = max(atr * 0.35, current_price * 0.001)

        supply_candidates: List[Dict[str, Any]] = []
        demand_candidates: List[Dict[str, Any]] = []

        for index in range(pivot_window, len(recent) - pivot_window):
            candle = recent[index]
            high = _positive_float(candle.get("high"))
            low = _positive_float(candle.get("low"))
            open_price = _positive_float(candle.get("open"))
            close = _positive_float(candle.get("close"))
            if not high or not low or not open_price or not close or high <= low:
                continue

            window = recent[index - pivot_window : index + pivot_window + 1]
            window_high = max(_positive_float(c.get("high")) or high for c in window)
            window_low = min(_positive_float(c.get("low")) or low for c in window)
            candle_range = high - low
            upper_rejection = (high - max(open_price, close)) / candle_range
            lower_rejection = (min(open_price, close) - low) / candle_range
            volume = _finite_float(candle.get("volume"))
            volume_ratio = (
                volume / median_volume
                if volume is not None and median_volume and median_volume > 0
                else None
            )
            recency = 1.0 - (len(recent) - index) / len(recent)

            if high >= window_high and upper_rejection >= 0.28:
                low_bound = max(high - zone_width, 0)
                high_bound = high
                if high_bound >= current_price * 0.995:
                    supply_candidates.append(
                        self._zone_candidate(
                            "supply",
                            low_bound,
                            high_bound,
                            high,
                            upper_rejection,
                            volume_ratio,
                            recency,
                            timeframe,
                            candle.get("timestamp"),
                            recent[index + 1 :],
                        )
                    )

            if low <= window_low and lower_rejection >= 0.28:
                low_bound = low
                high_bound = low + zone_width
                if low_bound <= current_price * 1.005:
                    demand_candidates.append(
                        self._zone_candidate(
                            "demand",
                            low_bound,
                            high_bound,
                            low,
                            lower_rejection,
                            volume_ratio,
                            recency,
                            timeframe,
                            candle.get("timestamp"),
                            recent[index + 1 :],
                        )
                    )

        return {
            "supply": self._merge_and_rank_zones(
                supply_candidates,
                current_price=current_price,
                side="supply",
                max_zones=max_zones_per_side,
            ),
            "demand": self._merge_and_rank_zones(
                demand_candidates,
                current_price=current_price,
                side="demand",
                max_zones=max_zones_per_side,
            ),
        }

    def _zone_candidate(
        self,
        zone_type: str,
        low: float,
        high: float,
        anchor: float,
        rejection: float,
        volume_ratio: Optional[float],
        recency: float,
        timeframe: str,
        timestamp: Any,
        future_candles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        touches = 0
        for candle in future_candles:
            candle_high = _positive_float(candle.get("high"))
            candle_low = _positive_float(candle.get("low"))
            if candle_high is None or candle_low is None:
                continue
            if candle_low <= high and candle_high >= low:
                touches += 1

        volume_component = 0.0
        if volume_ratio is not None:
            volume_component = min(max((volume_ratio - 1.0) / 2.0, 0.0), 1.0)

        strength = (
            min(rejection, 1.0) * 0.42
            + min(touches / 4.0, 1.0) * 0.26
            + volume_component * 0.22
            + min(max(recency, 0.0), 1.0) * 0.10
        )

        return {
            "type": zone_type,
            "low": round(float(low), 12),
            "high": round(float(high), 12),
            "anchor": round(float(anchor), 12),
            "mid": round(float((low + high) / 2.0), 12),
            "strength": round(float(max(0.0, min(1.0, strength))), 4),
            "touches": touches,
            "rejection_ratio": round(float(rejection), 4),
            "volume_ratio": round(float(volume_ratio), 4) if volume_ratio is not None else None,
            "timeframe": timeframe,
            "source": "ohlcv_pivot_rejection",
            "timestamp": timestamp,
        }

    def _merge_and_rank_zones(
        self,
        zones: List[Dict[str, Any]],
        *,
        current_price: float,
        side: str,
        max_zones: int,
    ) -> List[Dict[str, Any]]:
        if not zones:
            return []

        ranked = sorted(zones, key=lambda z: float(z.get("strength") or 0.0), reverse=True)
        merged: List[Dict[str, Any]] = []
        for zone in ranked:
            low = float(zone["low"])
            high = float(zone["high"])
            overlapping = None
            for existing in merged:
                existing_low = float(existing["low"])
                existing_high = float(existing["high"])
                overlap = low <= existing_high and high >= existing_low
                near = abs(float(zone["mid"]) - float(existing["mid"])) / current_price < 0.006
                if overlap or near:
                    overlapping = existing
                    break

            if overlapping:
                overlapping["low"] = round(min(float(overlapping["low"]), low), 12)
                overlapping["high"] = round(max(float(overlapping["high"]), high), 12)
                overlapping["mid"] = round((float(overlapping["low"]) + float(overlapping["high"])) / 2.0, 12)
                overlapping["strength"] = round(max(float(overlapping["strength"]), float(zone["strength"])), 4)
                overlapping["touches"] = int(overlapping.get("touches") or 0) + int(zone.get("touches") or 0)
                continue

            merged.append(dict(zone))

        def sort_key(zone: Dict[str, Any]) -> Tuple[float, float]:
            distance = abs(float(zone["mid"]) - current_price) / current_price
            strength = float(zone.get("strength") or 0.0)
            return (distance, -strength)

        if side == "supply":
            filtered = [z for z in merged if float(z["high"]) >= current_price * 0.995]
        else:
            filtered = [z for z in merged if float(z["low"]) <= current_price * 1.005]

        final = sorted(filtered, key=sort_key)[:max_zones]
        for zone in final:
            distance = _pct_distance(current_price, float(zone["mid"]))
            zone["distance_pct"] = round(float(distance), 3) if distance is not None else None
        return final

    def _build_order_flow(
        self,
        *,
        price: float,
        orderbook: Optional[Dict[str, Any]],
        funding_rate: Optional[float],
        long_short_ratio: Optional[float],
        top_trader_long_ratio: Optional[float],
        open_interest: Optional[float],
        open_interest_change_24h: Optional[float],
        taker_flow: Optional[Dict[str, Any]],
        technical_snapshot: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        flow: Dict[str, Any] = {}
        used: List[str] = []
        score_parts: List[float] = []
        regimes: List[str] = []

        ob = self._normalise_orderbook(orderbook)
        if ob:
            flow["orderbook"] = ob
            used.append("orderbook_depth")
            imbalance = _finite_float(ob.get("imbalance"))
            if imbalance is not None:
                score_parts.append(max(-1.0, min(1.0, imbalance)) * 0.30)

        funding = _finite_float(funding_rate)
        if funding is not None:
            flow["funding_rate"] = funding
            used.append("funding_rate")
            if funding > 0.0005:
                regimes.append("positive_funding_long_crowding")
                score_parts.append(-0.14)
            elif funding < -0.0005:
                regimes.append("negative_funding_short_crowding")
                score_parts.append(0.14)
            else:
                regimes.append("balanced_funding")

        ls_ratio = _finite_float(long_short_ratio)
        if ls_ratio is not None and ls_ratio > 0:
            flow["long_short_ratio"] = ls_ratio
            used.append("global_long_short_ratio")
            if ls_ratio >= 1.25:
                regimes.append("long_accounts_crowded")
                score_parts.append(-0.12)
            elif ls_ratio <= 0.80:
                regimes.append("short_accounts_crowded")
                score_parts.append(0.12)

        top_long = _finite_float(top_trader_long_ratio)
        if top_long is not None:
            flow["top_trader_long_ratio"] = top_long
            used.append("top_trader_positioning")
            if top_long >= 0.62:
                regimes.append("top_traders_long_heavy")
                score_parts.append(-0.08)
            elif top_long <= 0.38:
                regimes.append("top_traders_short_heavy")
                score_parts.append(0.08)

        oi = _finite_float(open_interest)
        oi_change = _finite_float(open_interest_change_24h)
        price_change = _finite_float(_get(technical_snapshot, "price_change_24h")) if technical_snapshot else None
        if oi is not None:
            flow["open_interest"] = oi
            used.append("open_interest")
        if oi_change is not None:
            flow["open_interest_change_24h"] = oi_change
            used.append("open_interest_change")
            if price_change is not None:
                if price_change > 1.5 and oi_change > 3:
                    regimes.append("price_up_oi_up_long_build")
                    score_parts.append(0.22)
                elif price_change < -1.5 and oi_change > 3:
                    regimes.append("price_down_oi_up_short_build")
                    score_parts.append(-0.22)
                elif price_change > 1.5 and oi_change < -3:
                    regimes.append("price_up_oi_down_short_covering")
                    score_parts.append(0.08)
                elif price_change < -1.5 and oi_change < -3:
                    regimes.append("price_down_oi_down_long_unwind")
                    score_parts.append(-0.08)

        taker = self._normalise_taker_flow(taker_flow)
        if taker:
            flow["taker_flow"] = taker
            used.append("taker_flow")
            delta_pct = _finite_float(taker.get("taker_delta_pct"))
            if delta_pct is not None:
                score_parts.append(max(-1.0, min(1.0, delta_pct)) * 0.34)
            cvd = _finite_float(taker.get("cvd"))
            if cvd is not None:
                flow["cvd_bias"] = "positive" if cvd > 0 else "negative" if cvd < 0 else "flat"

        composite = sum(score_parts)
        if score_parts:
            composite = max(-1.0, min(1.0, composite))
            flow["composite_score"] = round(float(composite), 4)
            if composite >= 0.18:
                flow["bias"] = "bullish"
            elif composite <= -0.18:
                flow["bias"] = "bearish"
            else:
                flow["bias"] = "balanced"
        if regimes:
            flow["regimes"] = regimes

        return flow, used

    def _normalise_orderbook(self, orderbook: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(orderbook, dict):
            return {}

        bid_depth = _finite_float(orderbook.get("bid_depth_usdt") or orderbook.get("depth_1pct_bids"))
        ask_depth = _finite_float(orderbook.get("ask_depth_usdt") or orderbook.get("depth_1pct_asks"))
        top_bid = _positive_float(orderbook.get("top_bid"))
        top_ask = _positive_float(orderbook.get("top_ask"))
        spread_pct = _finite_float(orderbook.get("spread_pct") or orderbook.get("spread_percentage"))
        ratio = _finite_float(orderbook.get("bid_ask_ratio"))
        imbalance = _finite_float(orderbook.get("imbalance"))

        if imbalance is None and bid_depth is not None and ask_depth is not None:
            total = bid_depth + ask_depth
            if total > 0:
                imbalance = (bid_depth - ask_depth) / total
        if ratio is None and bid_depth is not None and ask_depth and ask_depth > 0:
            ratio = bid_depth / ask_depth
        if spread_pct is None and top_bid and top_ask:
            spread_pct = ((top_ask - top_bid) / top_bid) * 100.0

        result: Dict[str, Any] = {}
        if bid_depth is not None:
            result["bid_depth_usdt"] = round(float(bid_depth), 4)
        if ask_depth is not None:
            result["ask_depth_usdt"] = round(float(ask_depth), 4)
        if imbalance is not None:
            result["imbalance"] = round(float(imbalance), 4)
        if ratio is not None:
            result["bid_ask_ratio"] = round(float(ratio), 4)
        if spread_pct is not None:
            result["spread_pct"] = round(float(spread_pct), 5)
        return result

    def _normalise_taker_flow(self, taker_flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(taker_flow, dict):
            return {}
        if taker_flow.get("data_available") is False:
            return {}

        buy_volume = _finite_float(taker_flow.get("taker_buy_volume"))
        sell_volume = _finite_float(taker_flow.get("taker_sell_volume"))
        taker_ratio = _finite_float(taker_flow.get("taker_ratio"))
        cvd = _finite_float(taker_flow.get("cvd"))
        delta = _finite_float(taker_flow.get("taker_delta"))
        delta_pct = _finite_float(taker_flow.get("taker_delta_pct"))
        periods = _finite_float(taker_flow.get("periods"))

        if delta is None and buy_volume is not None and sell_volume is not None:
            delta = buy_volume - sell_volume
        if delta_pct is None and buy_volume is not None and sell_volume is not None:
            total = buy_volume + sell_volume
            if total > 0:
                delta_pct = (buy_volume - sell_volume) / total
        if taker_ratio is None and buy_volume is not None and sell_volume and sell_volume > 0:
            taker_ratio = buy_volume / sell_volume

        result: Dict[str, Any] = {}
        if buy_volume is not None:
            result["taker_buy_volume"] = round(float(buy_volume), 4)
        if sell_volume is not None:
            result["taker_sell_volume"] = round(float(sell_volume), 4)
        if taker_ratio is not None:
            result["taker_ratio"] = round(float(taker_ratio), 4)
        if delta is not None:
            result["taker_delta"] = round(float(delta), 4)
        if delta_pct is not None:
            result["taker_delta_pct"] = round(float(delta_pct), 4)
        if cvd is not None:
            result["cvd"] = round(float(cvd), 4)
        if periods is not None:
            result["periods"] = int(periods)
        return result

    def _build_structure_snapshot(
        self,
        *,
        price: float,
        candles: List[Dict[str, Any]],
        zones: Dict[str, List[Dict[str, Any]]],
        technical_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        structure: Dict[str, Any] = {}
        trend_direction = str(_get(technical_snapshot, "trend_direction", "") or "").lower()
        ema20 = _positive_float(_get(technical_snapshot, "ema_20")) if technical_snapshot else None
        ema50 = _positive_float(_get(technical_snapshot, "ema_50")) if technical_snapshot else None
        ema200 = _positive_float(_get(technical_snapshot, "ema_200")) if technical_snapshot else None

        if trend_direction:
            structure["trend_direction"] = trend_direction
        if ema20 and ema50:
            if price > ema20 > ema50:
                structure["ema_structure"] = "bullish_stack"
            elif price < ema20 < ema50:
                structure["ema_structure"] = "bearish_stack"
            else:
                structure["ema_structure"] = "mixed"
        if ema200:
            structure["price_vs_ema200_pct"] = round(float(_pct_distance(ema200, price) or 0.0), 3)

        if len(candles) >= 20:
            swing_high = max(float(c["high"]) for c in candles[-20:] if c.get("high"))
            swing_low = min(float(c["low"]) for c in candles[-20:] if c.get("low"))
            structure["recent_swing_high"] = round(swing_high, 12)
            structure["recent_swing_low"] = round(swing_low, 12)
            if swing_high > swing_low:
                structure["position_in_20_candle_range"] = round((price - swing_low) / (swing_high - swing_low), 4)
                higher_highs, higher_lows = self._swing_progression(candles[-80:])
                if higher_highs and higher_lows:
                    structure["swing_structure"] = "higher_highs_higher_lows"
                elif not higher_highs and not higher_lows:
                    structure["swing_structure"] = "lower_highs_lower_lows"
                else:
                    structure["swing_structure"] = "mixed_swings"

        nearest_supply = self._nearest_zone(zones.get("supply") or [], price)
        nearest_demand = self._nearest_zone(zones.get("demand") or [], price)
        if nearest_supply:
            structure["nearest_supply"] = nearest_supply
        if nearest_demand:
            structure["nearest_demand"] = nearest_demand

        location = "mid_range"
        warnings: List[str] = []
        supply_distance = _finite_float(nearest_supply.get("distance_pct")) if nearest_supply else None
        demand_distance = _finite_float(nearest_demand.get("distance_pct")) if nearest_demand else None
        if supply_distance is not None and 0 <= supply_distance <= 1.0:
            location = "near_supply"
            warnings.append("longs_are_close_to_supply")
        if demand_distance is not None and -1.0 <= demand_distance <= 0:
            location = "near_demand"
            warnings.append("shorts_are_close_to_demand")
        structure["trade_location"] = location
        if warnings:
            structure["warnings"] = warnings
        return structure

    def _swing_progression(self, candles: List[Dict[str, Any]]) -> Tuple[bool, bool]:
        if len(candles) < 24:
            return False, False
        segment_size = max(6, len(candles) // 4)
        segments = [candles[i : i + segment_size] for i in range(0, len(candles), segment_size)]
        segments = [s for s in segments if len(s) >= 4][-4:]
        if len(segments) < 3:
            return False, False
        highs = [max(float(c["high"]) for c in segment if c.get("high")) for segment in segments]
        lows = [min(float(c["low"]) for c in segment if c.get("low")) for segment in segments]
        higher_highs = highs[-1] > highs[-2] > highs[-3]
        higher_lows = lows[-1] > lows[-2] > lows[-3]
        return higher_highs, higher_lows

    def _nearest_zone(self, zones: List[Dict[str, Any]], price: float) -> Optional[Dict[str, Any]]:
        if not zones:
            return None
        ranked = sorted(zones, key=lambda z: abs(float(z.get("mid") or 0.0) - price))
        nearest = dict(ranked[0])
        distance = _pct_distance(price, float(nearest["mid"]))
        nearest["distance_pct"] = round(float(distance), 3) if distance is not None else None
        return nearest

    def _build_chart_overlays(
        self,
        zones: Dict[str, List[Dict[str, Any]]],
        structure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        overlays: List[Dict[str, Any]] = []
        for zone_type in ("demand", "supply"):
            for zone in zones.get(zone_type) or []:
                overlays.append(
                    {
                        "type": "zone",
                        "side": zone_type,
                        "label": f"{zone_type.title()} zone",
                        "low": zone.get("low"),
                        "high": zone.get("high"),
                        "mid": zone.get("mid"),
                        "strength": zone.get("strength"),
                        "source": zone.get("source"),
                    }
                )
        for key, label in (
            ("recent_swing_high", "Recent swing high"),
            ("recent_swing_low", "Recent swing low"),
        ):
            value = _positive_float(structure.get(key))
            if value:
                overlays.append(
                    {
                        "type": "level",
                        "label": label,
                        "price": value,
                        "source": "ohlcv_swing",
                    }
                )
        return overlays

    def to_prompt_block(self, context: Dict[str, Any]) -> str:
        lines: List[str] = []
        data_used = context.get("data_used") or []
        if not data_used:
            return ""

        lines.append("MARKET STRUCTURE AND ORDER FLOW CONTEXT (real data only):")
        lines.append(f"- Data used: {', '.join(data_used)}")

        structure = context.get("market_structure") or {}
        if structure:
            structure_bits = []
            for key in ("trend_direction", "ema_structure", "swing_structure", "trade_location"):
                if structure.get(key):
                    structure_bits.append(f"{key}={structure[key]}")
            if structure_bits:
                lines.append(f"- Structure: {', '.join(structure_bits)}")
            nearest_demand = structure.get("nearest_demand")
            nearest_supply = structure.get("nearest_supply")
            if nearest_demand:
                lines.append(
                    "- Nearest demand: "
                    f"{_fmt_price(nearest_demand.get('low'))}-{_fmt_price(nearest_demand.get('high'))} "
                    f"({ _fmt_pct(nearest_demand.get('distance_pct')) } from price, "
                    f"strength {nearest_demand.get('strength')})"
                )
            if nearest_supply:
                lines.append(
                    "- Nearest supply: "
                    f"{_fmt_price(nearest_supply.get('low'))}-{_fmt_price(nearest_supply.get('high'))} "
                    f"({ _fmt_pct(nearest_supply.get('distance_pct')) } from price, "
                    f"strength {nearest_supply.get('strength')})"
                )
            warnings = structure.get("warnings") or []
            if warnings:
                lines.append(f"- Trade-location warnings: {', '.join(warnings)}")

        order_flow = context.get("order_flow") or {}
        if order_flow:
            bias = order_flow.get("bias")
            score = order_flow.get("composite_score")
            if bias:
                lines.append(f"- Order-flow bias: {bias} (score={score})")
            ob = order_flow.get("orderbook") or {}
            if ob:
                lines.append(
                    "- Depth: "
                    f"bid={_fmt_price(ob.get('bid_depth_usdt'))} ask={_fmt_price(ob.get('ask_depth_usdt'))} "
                    f"imbalance={ob.get('imbalance', 'n/a')} spread={ob.get('spread_pct', 'n/a')}%"
                )
            taker = order_flow.get("taker_flow") or {}
            if taker:
                delta_pct = _finite_float(taker.get("taker_delta_pct"))
                delta_pct_text = _fmt_pct(delta_pct * 100.0) if delta_pct is not None else "n/a"
                lines.append(
                    "- Taker flow/CVD: "
                    f"ratio={taker.get('taker_ratio', 'n/a')} "
                    f"delta_pct={delta_pct_text} "
                    f"cvd={taker.get('cvd', 'n/a')}"
                )
            if order_flow.get("funding_rate") is not None:
                lines.append(f"- Funding: {float(order_flow['funding_rate']) * 100:+.4f}%")
            if order_flow.get("open_interest_change_24h") is not None:
                lines.append(f"- Open interest 24h: {_fmt_pct(order_flow.get('open_interest_change_24h'))}")
            regimes = order_flow.get("regimes") or []
            if regimes:
                lines.append(f"- Flow regimes: {', '.join(regimes)}")

        lines.append(
            "- Decision rule: use this context only where data_used proves it exists; do not infer missing flow or zones."
        )
        return "\n".join(lines)


market_structure_context_service = MarketStructureContextService()


def get_market_structure_context_service() -> MarketStructureContextService:
    return market_structure_context_service
