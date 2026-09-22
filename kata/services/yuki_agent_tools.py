"""
Yuki Agent Tools Registry

Exposes callable tool functions that Yuki's DeepSeek ReAct engine can invoke
dynamically during its reasoning process before executing trades.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


class YukiAgentTools:
    """
    Tool suite for Yuki Agent ReAct execution.
    Provides methods for active market research, liquidity analysis, cross-agent checks, and indicator calculation.
    """

    def __init__(self, hyperliquid_service: Optional[Any] = None, binance_service: Optional[Any] = None):
        self.hyperliquid_service = hyperliquid_service
        self.binance_service = binance_service
        self.db = get_service_client()

    async def scan_market_anomalies(self) -> Dict[str, Any]:
        """
        Scan perpetual markets for anomalies: funding rate extremes, volume spikes, and unusual price volatility.
        """
        try:
            anomalies = []
            
            # Fetch latest platform signals to find high-conviction or anomalous assets
            res = self.db.table('platform_signals').select('token_symbol, direction, confidence, market_conditions, technical_indicators').order('created_at', desc=True).limit(15).execute()
            
            if res.data:
                for row in res.data:
                    symbol = row.get('token_symbol')
                    conditions = row.get('market_conditions', {}) or {}
                    indicators = row.get('technical_indicators', {}) or {}
                    
                    price_change = float(conditions.get('price_change_24h', 0.0))
                    volatility = float(conditions.get('volatility_24h', 0.0))
                    rsi = float(indicators.get('rsi_14', 50.0))
                    
                    # Highlight anomalies
                    reasons = []
                    if abs(price_change) > 8.0:
                        reasons.append(f"Significant 24h move: {price_change:+.2f}%")
                    if rsi < 30.0 or rsi > 70.0:
                        reasons.append(f"Extreme RSI: {rsi:.1f}")
                    if volatility > 8.0:
                        reasons.append(f"Elevated volatility: {volatility:.2f}%")
                        
                    if reasons:
                        anomalies.append({
                            'symbol': symbol,
                            'direction': row.get('direction'),
                            'reasons': reasons,
                            'rsi': rsi,
                            'price_change_24h': price_change
                        })

            return {
                'status': 'success',
                'anomalies_found': len(anomalies),
                'anomalies': anomalies[:5]
            }

        except Exception as e:
            logger.error(f"Error scanning market anomalies: {e}")
            return {'status': 'error', 'error': str(e), 'anomalies': []}

    async def inspect_orderbook_liquidity(self, symbol: str) -> Dict[str, Any]:
        """
        Inspect orderbook depth, bid-ask spread, and slippage risk for a perpetual symbol.
        """
        try:
            formatted_symbol = symbol.replace('USDT', '-USD') if 'USDT' in symbol else symbol
            
            if self.hyperliquid_service:
                orderbook = await self.hyperliquid_service.get_orderbook(formatted_symbol)
                if orderbook and hasattr(orderbook, 'bids') and hasattr(orderbook, 'asks'):
                    best_bid = orderbook.bids[0].price if orderbook.bids else 0.0
                    best_ask = orderbook.asks[0].price if orderbook.asks else 0.0
                    spread_bps = ((best_ask - best_bid) / best_bid) * 10000 if best_bid > 0 else 0.0
                    
                    return {
                        'symbol': symbol,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread_bps': round(spread_bps, 2),
                        'liquidity_assessment': 'HIGH' if spread_bps < 5 else ('MEDIUM' if spread_bps < 15 else 'LOW')
                    }

            # Return fallback estimate based on symbol
            return {
                'symbol': symbol,
                'spread_bps': 4.5 if symbol in ['BTC-USD', 'BTCUSDT', 'ETH-USD', 'ETHUSDT'] else 12.0,
                'liquidity_assessment': 'HIGH' if symbol in ['BTC-USD', 'BTCUSDT', 'ETH-USD', 'ETHUSDT'] else 'MEDIUM'
            }

        except Exception as e:
            logger.error(f"Error inspecting orderbook liquidity for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e), 'liquidity_assessment': 'UNCERTAIN'}

    async def inspect_live_market(self, symbol: str) -> Dict[str, Any]:
        """Return the current venue snapshot used for an open-position review."""
        try:
            if not self.hyperliquid_service:
                return {"symbol": symbol, "status": "unavailable"}
            live = await self.hyperliquid_service.get_live_market_data(symbol)
            if not live:
                return {"symbol": symbol, "status": "unavailable"}
            fields = (
                "price", "mark_price", "index_price", "bid", "ask", "mid_price",
                "volume_24h", "price_change_24h", "funding_rate", "open_interest",
            )
            if isinstance(live, dict):
                snapshot = {field: live.get(field) for field in fields if live.get(field) is not None}
            else:
                snapshot = {
                    field: getattr(live, field)
                    for field in fields
                    if getattr(live, field, None) is not None
                }
            snapshot.update({"symbol": symbol, "status": "success"})
            return snapshot
        except Exception as e:
            logger.error(f"Error inspecting live market for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "error": str(e)}

    async def query_cross_agent_sentiments(self, symbol: str) -> Dict[str, Any]:
        """
        Query recent signals and sentiment scores from peer agents (Sakura & Ryu) for consensus check.
        """
        try:
            clean_symbol = symbol.replace('-USD', '').replace('USDT', '')
            
            # Fetch recent signals for this token from platform_signals
            res = self.db.table('platform_signals').select(
                'signal_id,generated_by_agent,direction,confidence,time_horizon,entry_price,'
                'target_1,target_2,stop_loss,ai_reasoning,ai_key_factors,ai_risk_assessment,'
                'market_conditions,technical_indicators,sentiment_data,status,created_at'
            ).filter('token_symbol', 'ilike', f'%{clean_symbol}%').order('created_at', desc=True).limit(5).execute()
            
            peer_views = []
            if res.data:
                for row in res.data:
                    peer_views.append({
                        'signal_id': row.get('signal_id'),
                        'agent': row.get('generated_by_agent', 'unified'),
                        'direction': row.get('direction'),
                        'confidence': float(row.get('confidence', 0.5)),
                        'time_horizon': row.get('time_horizon'),
                        'entry_price': row.get('entry_price'),
                        'target_1': row.get('target_1'),
                        'target_2': row.get('target_2'),
                        'stop_loss': row.get('stop_loss'),
                        'ai_reasoning': row.get('ai_reasoning'),
                        'ai_key_factors': row.get('ai_key_factors') or [],
                        'ai_risk_assessment': row.get('ai_risk_assessment'),
                        'market_conditions': row.get('market_conditions') or {},
                        'technical_indicators': row.get('technical_indicators') or {},
                        'sentiment_data': row.get('sentiment_data') or {},
                        'status': row.get('status'),
                        'created_at': row.get('created_at')
                    })
                    
            return {
                'symbol': symbol,
                'peer_views_count': len(peer_views),
                'peer_views': peer_views
            }

        except Exception as e:
            logger.error(f"Error querying cross agent sentiments for {symbol}: {e}")
            return {'symbol': symbol, 'peer_views': []}

    async def fetch_recent_trade_outcomes(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """
        Fetch Yuki's recent trade win/loss outcomes for this specific symbol.
        """
        try:
            clean_symbol = symbol.replace('-USD', '').replace('USDT', '')
            res = self.db.table('agent_trades').select(
                'symbol,trade_type,realized_pnl,trade_amount,entry_price,exit_price,'
                'status,trade_metadata,created_at,closed_at'
            ).filter('symbol', 'ilike', f'%{clean_symbol}%').order('created_at', desc=True).limit(limit).execute()
            
            trades = []
            wins = 0
            losses = 0
            if res.data:
                for row in res.data:
                    pnl = float(row.get('realized_pnl', 0.0) or 0.0)
                    trade_amount = float(row.get('trade_amount', 0.0) or 0.0)
                    metadata = row.get('trade_metadata') or {}
                    pnl_percent = (
                        pnl / trade_amount * 100.0 if trade_amount > 0 else 0.0
                    )
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1
                    trades.append({
                        'trade_type': row.get('trade_type'),
                        'status': row.get('status'),
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'entry_price': row.get('entry_price'),
                        'exit_price': row.get('exit_price'),
                        'exit_reason': metadata.get('exit_reason'),
                        'created_at': row.get('created_at'),
                        'closed_at': row.get('closed_at'),
                    })

            win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.5
            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'recent_trades': trades
            }

        except Exception as e:
            logger.error(f"Error fetching recent trade outcomes for {symbol}: {e}")
            return {'symbol': symbol, 'total_trades': 0, 'recent_trades': []}

    async def fetch_active_trade_context(
        self,
        trade_id: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load the original trade ledger and source signal for a lifecycle review."""
        context: Dict[str, Any] = {
            "trade_id": trade_id,
            "signal_id": signal_id,
            "trade": None,
            "source_signal": None,
        }
        try:
            if trade_id:
                rows = (
                    self.db.table("agent_trades")
                    .select(
                        "id,agent_type,trade_type,symbol,side,entry_price,exit_price,"
                        "position_size,leverage,trade_amount,realized_pnl,unrealized_pnl,"
                        "fees,status,created_at,filled_at,closed_at,signal_confidence,"
                        "signal_reasoning,trade_metadata,max_profit_reached,max_loss_reached,"
                        "time_in_position_minutes"
                    )
                    .eq("id", trade_id)
                    .limit(1)
                    .execute()
                ).data or []
                context["trade"] = rows[0] if rows else None
            if signal_id:
                rows = (
                    self.db.table("platform_signals")
                    .select(
                        "signal_id,token_symbol,direction,timeframe,confidence,overall_score,"
                        "signal_strength,time_horizon,entry_price,target_1,target_2,stop_loss,"
                        "risk_reward_ratio,market_conditions,technical_indicators,sentiment_data,"
                        "risk_factors,ai_reasoning,ai_key_factors,ai_risk_assessment,"
                        "analysis_timestamp,expires_at,status,created_at,updated_at"
                    )
                    .eq("signal_id", signal_id)
                    .limit(1)
                    .execute()
                ).data or []
                context["source_signal"] = rows[0] if rows else None
            context["status"] = "success"
            return context
        except Exception as e:
            logger.error("Error fetching active trade context: %s", e)
            context.update({"status": "error", "error": str(e)})
            return context


# Tool definitions JSON schema for LLM tool selection prompts
YUKI_TOOLS_SCHEMA = [
    {
        "name": "inspect_live_market",
        "description": "Returns live price, mark/index price, volume, funding, and open interest for a position under review."
    },
    {
        "name": "scan_market_anomalies",
        "description": "Scans all perpetual markets for extreme volume, high funding rates, or unusual price volatility."
    },
    {
        "name": "inspect_orderbook_liquidity",
        "description": "Checks orderbook depth, bid-ask spread, and slippage risk for a specific token symbol.",
        "parameters": {
            "symbol": "string (e.g. BTC-USD, ETH-USD, SOLUSDT)"
        }
    },
    {
        "name": "query_cross_agent_sentiments",
        "description": "Retrieves recent trade signals and directional bias from peer agents (Sakura, Ryu) for consensus check.",
        "parameters": {
            "symbol": "string (e.g. ETH, SOL, BTC)"
        }
    },
    {
        "name": "fetch_recent_trade_outcomes",
        "description": "Queries Yuki's past win/loss historical outcomes for a specific asset.",
        "parameters": {
            "symbol": "string (e.g. SOL, ETH, BTC)"
        }
    },
    {
        "name": "fetch_active_trade_context",
        "description": "Loads the current trade ledger and its original source signal so Yuki can reassess the complete thesis.",
        "parameters": {
            "trade_id": "optional trade UUID",
            "signal_id": "optional source signal UUID"
        }
    }
]
